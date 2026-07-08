"""
network.py — DepthOnlyNet: depth-only two-view depth alignment.

Overview
--------
DepthOnlyNet takes a *pair* of monocular depth predictions and a relative
pose T_12 and produces an aligned depth map for each view at 640×480
(ScanNet GT depth resolution).

Pipeline
--------
  1. Median-normalise both input depth maps in-network (see note below).
  2. Encode both depth maps through a shared ConvNeXt DepthEncoder
     → multi-scale features {s4, s8, s16}.
  3. Flatten s16 features → token sequences.
     Project to token_dim via enc_to_dec if token_dim ≠ feature_dim.
  4. Run CrossAttentionDecoder:
       - Optionally prepend a pose-conditioned token (from T_12 / T_21).
       - N symmetric CrossBlocks (self-attn → cross-attn → FFN).
       - Strip pose token from output.
  5. Decode each view's attended tokens + encoder skip features
     → depth & confidence at 640×480 via DepthDecoder.
  6. Return outputs for both views.

Median normalisation
--------------------
Input MDE depths are normalised by their spatial median before encoding:

    d_norm = d / (median(d) + eps)

This is applied *inside* the forward pass so inference is consistent
regardless of whether the data loader pre-normalises or not.  When the
preprocessing pipeline (preprocessing.py) also median-normalises, the
in-network step is a no-op (median ≈ 1.0), so there is no double-normalisation
penalty — it is idempotent.

Pose input (current)
--------------------
T_12 (4×4, view-1 → view-2) is a required input.  It is encoded into a
pose token prepended to each view's token sequence.  The model does NOT
yet regress poses.  To disable pose conditioning entirely pass
`use_pose_token=False` to the constructor (and omit T_12 from forward).

Future: camera token & pose head
---------------------------------
To add pose regression later:
  - Replace PoseEncoder with a learnable camera token (nn.Parameter)
    initialised at zeros, one per view.
  - Add a CameraHead MLP that reads the camera token after the blocks.
  - The `use_pose_token` flag already provides the structural hook.

Architecture hyperparameters (arch.yaml → 'depth_only' key)
-------------------------------------------------------------
  feature_dim     : 256   ConvNeXt output channels
  token_dim       : 256   Cross-attention token dim (= feature_dim unless
                          an enc_to_dec projection is configured)
  num_blocks      : 4     CrossBlock stack depth
  num_heads       : 8     Attention heads (token_dim must be divisible)
  mlp_ratio       : 4.0   MLP expansion ratio
  decoder_hidden  : 128   Decoder internal channel width
  use_pose_token  : true
  freeze_encoder  : false
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.model_image_depth.geometry import se3_inv, transform_pts

from .encoder          import DepthEncoder
from .image_encoder    import MASt3RImageEncoder
from .cross_attention  import CrossAttentionDecoder, _mast3r_init, _vec9_to_se3, _se3_inv
from .decoder          import DepthDecoder
from .pose_refinement  import PoseRefinementModule

# Reuse k_inv + se3_inv from losses.py so the unprojection convention
# here and in point_map_loss() can never drift out of sync.
from training.losses import k_inv as _k_inv



# ---------------------------------------------------------------------------
# Pose head  (used when predict_pose=True)
# ---------------------------------------------------------------------------

class PoseHead(nn.Module):
    """
    Regress T_12 from encoder global features (not cross-attention camera tokens).

    Input: global-average-pool of feats['s16'] from each view independently.
    These are genuinely view-specific (computed before any cross-attention) so
    the pose head has a real signal to work with even when the two views are
    geometrically similar.

    T_12 = f(enc1_global, enc2_global)
    T_21 = se3_inv(T_12)   — exact, no independent prediction needed.

    Outputs T_12_pred, T_21_pred (B,4,4) and per-sample log-confidence
    scalars for heteroscedastic uncertainty weighting in camera_pose_loss.
    """

    def __init__(self, in_dim: int, hidden: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2 * in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 9),   # 6D rotation + 3D translation
        )
        # Learnable log-confidence scalars (Kendall & Gal NeurIPS 2017).
        # log_conf_K is a dummy (depth_only does not predict intrinsics);
        # included so camera_pose_loss receives the expected keys.
        self.log_conf_pose = nn.Parameter(torch.zeros(1))
        self.log_conf_K    = nn.Parameter(torch.zeros(1))

        _mast3r_init(self.mlp)
        # Zero last-layer weights so all bias-free paths start at identity.
        # IMPORTANT: do NOT zero the bias for the rotation columns — a zero
        # 6D rotation vector causes F.normalize([0,0,0]) → NaN at init.
        # Set bias = identity in 6D repr (cols 1,2 of the rotation) + a small
        # but non-trivial translation (norm >> EPS_NORM=0.01) to avoid the
        # ~100× gradient amplification from the normalization chain-rule.
        nn.init.zeros_(self.mlp[-1].weight)
        with torch.no_grad():
            self.mlp[-1].bias.copy_(
                torch.tensor([1., 0., 0., 0., 1., 0.,   # identity in 6D
                               0., 0., 0.1])              # 10 cm forward; norm >> EPS_NORM
            )

    def forward(
        self,
        enc_global1: torch.Tensor,   # (B, in_dim)  global-avg-pool of encoder s16, view-1
        enc_global2: torch.Tensor,   # (B, in_dim)  global-avg-pool of encoder s16, view-2
    ) -> dict:
        B = enc_global1.shape[0]
        # T_12: conditioned on per-view encoder globals (view-specific before cross-attn)
        T_12_pred = _vec9_to_se3(self.mlp(torch.cat([enc_global1, enc_global2], dim=-1)))
        # T_21 is the exact SE(3) inverse of T_12 — no independent prediction needed,
        # so the identity round-trip loss is automatically satisfied by construction.
        T_21_pred = _se3_inv(T_12_pred)
        # Previous approach: predict T_21 independently with swapped token order.
        # Requires an identity loss to enforce T_21 ≈ inv(T_12).
        # T_21_pred = _vec9_to_se3(self.mlp(torch.cat([cam_tok2, cam_tok1], dim=-1)))

        return {
            "T_12_pred":     T_12_pred,                        # (B, 4, 4)
            "T_21_pred":     T_21_pred,                        # (B, 4, 4)
            "log_conf_pose": self.log_conf_pose.expand(B),     # (B,)
            "log_conf_K":    self.log_conf_K.expand(B),        # (B,)  dummy
        }


class DepthOnlyNet(nn.Module):
    """
    Depth-only two-view depth alignment network.

    Parameters
    ----------
    feature_dim      : int   Encoder output channels (projected to feature_dim).
    token_dim        : int   Cross-attention dimension.  If token_dim ≠ feature_dim
                             an enc_to_dec linear projection is inserted.
    num_blocks       : int   Number of symmetric CrossBlocks.
    num_heads        : int   Attention heads per block.
    mlp_ratio        : float MLP hidden / token_dim ratio.
    decoder_hidden   : int   Decoder internal channel width.
    use_pose_token   : bool  Prepend pose-conditioned token before cross-attn.
    freeze_encoder   : bool  Freeze ConvNeXt backbone (first-conv stays trainable).
    use_image_encoder: bool  Add a frozen MASt3R ViT image encoder whose tokens
                             are summed onto the depth tokens before cross-attn.
                             When False (default) the network is identical to
                             the original depth-only model.
    image_encoder_ckpt: str|None  MASt3R checkpoint to load image encoder weights
                             from (same .pth used for mast3r_ckpt works).
    """

    def __init__(
        self,
        feature_dim:        int   = 256,
        token_dim:          int   = 768,
        num_blocks:         int   = 4,
        num_heads:          int   = 12,
        mlp_ratio:          float = 4.0,
        decoder_hidden:     int   = 128,
        use_pose_token:     bool  = True,
        freeze_encoder:     bool  = False,
        mast3r_ckpt:        str | None = None,
        freeze_cross_attn:  bool  = False,
        max_encode_hw:      tuple | None = (480, 640),
        predict_pose:       bool  = False,
        pose_head_hidden:   int   = 128,
        pose_refine_iters:  int   = 0,
        pose_refine_feat_dim: int = 128,
        pose_refine_hidden: int   = 128,
        use_image_encoder:  bool  = False,
        image_encoder_ckpt: str | None = None,
    ):
        super().__init__()

        self.token_dim          = token_dim
        self.feature_dim        = feature_dim
        self.predict_pose       = predict_pose
        self.max_encode_hw      = max_encode_hw
        self.pose_refine_iters  = pose_refine_iters
        self.use_image_encoder  = use_image_encoder

        # ── Shared depth encoder ─────────────────────────────────────────
        self.encoder = DepthEncoder(
            pretrained      = False,
            feature_dim     = feature_dim,
            freeze_backbone = freeze_encoder,
        )

        # ── Enc-to-dec projection (only if dims differ) ──────────────────
        # Mirrors MASt3R's enc_to_dec design (1024→768).
        if feature_dim != token_dim:
            self.enc_to_dec = nn.Linear(feature_dim, token_dim, bias=True)
            _mast3r_init(self.enc_to_dec)
        else:
            self.enc_to_dec = nn.Identity()

        # ── Cross-attention decoder ──────────────────────────────────────
        self.cross_attn = CrossAttentionDecoder(
            token_dim      = token_dim,
            num_blocks     = num_blocks,
            num_heads      = num_heads,
            mlp_ratio      = mlp_ratio,
            use_pose_token = use_pose_token,
            predict_pose   = predict_pose,
        )

        # ── Optional MASt3R weight init ──────────────────────────────────
        if mast3r_ckpt is not None:
            self.cross_attn.load_mast3r_weights(mast3r_ckpt)
        if freeze_cross_attn:
            self.cross_attn.freeze_blocks()

        # ── Pose head (only when predict_pose=True) ──────────────────────
        # Reads encoder global features (feats['s16'] global-avg-pool) per view.
        # Using encoder features (computed before cross-attention) rather than
        # camera tokens: the camera_token is shared across views at init so
        # cam_tok1 ≈ cam_tok2 early in training, giving the MLP no useful signal.
        # Encoder features are genuinely view-specific (independent processing).
        if predict_pose:
            self.pose_head = PoseHead(feature_dim, hidden=pose_head_hidden)
        # ── Optional SE(3) GRU pose refinement ────────────────────────────
        # When pose_refine_iters > 0, a ConvGRU refines the coarse PoseHead
        # estimate using geometric warping of cross-attention spatial tokens.
        # pose_refine_iters=0 (default) → no refiner instantiated; pipeline
        # is completely identical to single-shot PoseHead.
        # K must be passed to forward(); if K is None at runtime, refinement
        # is silently skipped and the coarse PoseHead output is used instead.
        if predict_pose and pose_refine_iters > 0:
            self.pose_refiner = PoseRefinementModule(
                token_dim  = token_dim,
                feat_dim   = pose_refine_feat_dim,
                hidden_dim = pose_refine_hidden,
                num_iters  = pose_refine_iters,
            )
        # ── Optional image encoder (frozen MASt3R ViT-Large) ─────────────
        # Produces patch tokens summed onto depth tokens before cross-attn.
        # The output projection (1024 → token_dim) is zero-initialised so
        # the image branch is a no-op at t=0 — identical to the depth-only
        # baseline at the start of training.
        if use_image_encoder:
            self.image_encoder = MASt3RImageEncoder(
                token_dim   = token_dim,
                max_hw      = max_encode_hw,
                mast3r_ckpt = image_encoder_ckpt,
                freeze      = True,   # backbone always frozen; only proj trains
            )

        # ── Depth decoders (one per view, weight-tied) ───────────────────
        # Both views share the same decoder weights.
        self.decoder = DepthDecoder(
            token_dim = token_dim,
            skip_dim  = feature_dim,
            hidden    = decoder_hidden,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _unproject(depth: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
        """
        Unproject depth map to a 3D point map in the camera frame.

        Uses the same k_inv() helper as training/losses.py so that the
        point-prior construction here and the GT point-map construction
        in point_map_loss() share an identical convention.

        depth : (B, 1, H, W)  depth in metres
        K     : (B, 3, 3)     camera intrinsics (at depth resolution)
        Returns (B, 3, H, W)  metric XYZ point map
        """
        B, _, H, W = depth.shape
        K_inv = _k_inv(K)  # (B, 3, 3)  analytical, same as losses.k_inv

        ys = torch.arange(H, device=depth.device, dtype=depth.dtype)
        xs = torch.arange(W, device=depth.device, dtype=depth.dtype)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")   # (H, W)
        N = H * W
        xy1 = torch.stack(
            [gx.flatten(), gy.flatten(),
             torch.ones(N, device=depth.device, dtype=depth.dtype)], dim=0
        )                                                  # (3, N)
        xy1 = xy1.unsqueeze(0).expand(B, -1, -1)          # (B, 3, N)

        d_flat = depth.reshape(B, 1, N)
        xyz = torch.bmm(K_inv, xy1) * d_flat              # (B, 3, N)
        return xyz.reshape(B, 3, H, W)

    def _cap_resolution(self, d: torch.Tensor) -> torch.Tensor:
        """Resize d to at most max_encode_hw (preserving aspect ratio) if set."""
        if self.max_encode_hw is None:
            return d
        max_h, max_w = self.max_encode_hw
        _, _, H, W = d.shape
        if H <= max_h and W <= max_w:
            return d
        # Scale by the tighter dimension to stay within the box.
        scale = min(max_h / H, max_w / W)
        new_h = int(H * scale)
        new_w = int(W * scale)
        return F.interpolate(d, size=(new_h, new_w), mode="bilinear", align_corners=False)

    def _flatten_s16(self, feats: dict) -> torch.Tensor:
        """
        Flatten s16 spatial feature map → token sequence.
        (B, feature_dim, H/16, W/16) → (B, H/16 * W/16, token_dim)
        """
        s16 = feats["s16"]                               # (B, C, h, w)
        B, C, h, w = s16.shape
        tokens = s16.flatten(2).transpose(1, 2)          # (B, h*w, C)
        return self.enc_to_dec(tokens)                   # (B, h*w, token_dim)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        depth1: torch.Tensor,                # (B, 1, H1, W1) MDE depth view 1
        depth2: torch.Tensor,                # (B, 1, H2, W2) MDE depth view 2
        T_12:   torch.Tensor | None = None,  # (B, 4, 4)  pose view-1 → view-2
        K:      torch.Tensor | None = None,  # (B, 3, 3)  intrinsics — required for pose refinement
        rgb1:   torch.Tensor | None = None,  # (B, 3, H1, W1) RGB view 1  (required when use_image_encoder=True)
        rgb2:   torch.Tensor | None = None,  # (B, 3, H2, W2) RGB view 2  (required when use_image_encoder=True)
        point_norm_scale: float | None = None,  # optional isotropic scale: priors /= scale, output in normalised units
    ) -> dict:
        """
        Parameters
        ----------
        depth1, depth2 : (B, 1, H, W)  Raw MDE depths (metres).  H, W are the
                         *colour image* resolution (variable; not necessarily
                         640×480).
        T_12           : (B, 4, 4)  Relative pose.  Required when
                         use_pose_token=True.
        K              : (B, 3, 3)  Intrinsics.  Required for point-prior
                         construction and for pose refinement.
        rgb1, rgb2     : (B, 3, H, W)  RGB images in [0, 1] range.  Required
                         when use_image_encoder=True; ignored otherwise.

        Returns
        -------
        {
          "point1"      : (B, 3, 480, 640),  # XYZ in view-1 camera frame
          "point2"      : (B, 3, 480, 640),  # XYZ in view-1 camera frame
          "confidence1" : (B, 1, 480, 640),
          "confidence2" : (B, 1, 480, 640),
        }
        """
        # ── 1. Cap input resolution to bound s16 token count ─────────────────
        # depth_input (passed to decoder for the residual prior) stays at
        # the *original* resolution; only the encoder input is capped.
        d1_enc = self._cap_resolution(depth1)
        d2_enc = self._cap_resolution(depth2)

        # ── 2. Shared encoder ────────────────────────────────────────────
        # Normalize capped depths to [0, 1] per sample before encoding so
        # the single-channel ConvNeXt input matches the scale expected by
        # ImageNet-pretrained weights.  The original metric tensors (depth1,
        # depth2) are passed unchanged to the decoder for scale+residual.
        d1_max = d1_enc.flatten(1).max(dim=1).values.view(-1, 1, 1, 1).clamp(min=1e-3)
        d2_max = d2_enc.flatten(1).max(dim=1).values.view(-1, 1, 1, 1).clamp(min=1e-3)
        feats1 = self.encoder(d1_enc / d1_max)   # {s4, s8, s16}
        feats2 = self.encoder(d2_enc / d2_max)   # {s4, s8, s16}

        # Remember spatial dims for reshaping tokens back to spatial maps.
        _, _, H1, W1 = depth1.shape
        _, _, H2, W2 = depth2.shape

        # ── 3. Flatten s16 → tokens + optional enc_to_dec projection ────
        t1 = self._flatten_s16(feats1)   # (B, N1, token_dim)
        t2 = self._flatten_s16(feats2)   # (B, N2, token_dim)

        # ── 3b. Image encoder fusion (optional) ─────────────────────────
        # The frozen MASt3R ViT encodes each RGB view into patch tokens
        # (B, N, token_dim) via a zero-initialised projection.  These are
        # *summed* onto the depth tokens so that:
        #   • At t=0 (zero-init proj) the image contribution is exactly 0
        #     → model is numerically identical to depth-only baseline.
        #   • As training proceeds the projection learns to incorporate
        #     image context without altering sequence length or cross-attn cost.
        # Both the depth s16 grid and the ViT patch grid produce h×w tokens
        # for the same capped resolution, so no resampling is needed.
        # If the grids differ (unusual non-multiple-of-16 inputs) we fall back
        # to bilinear resampling of the image tokens.
        if self.use_image_encoder:
            if rgb1 is None or rgb2 is None:
                raise ValueError(
                    "rgb1 and rgb2 must be provided when use_image_encoder=True"
                )
            img_t1 = self.image_encoder(rgb1)   # (B, N_img1, token_dim)
            img_t2 = self.image_encoder(rgb2)   # (B, N_img2, token_dim)

            # Align spatial dimensions in case depth-s16 and image patches differ.
            # Under normal operation (both capped to max_encode_hw) they match.
            _, _, h1_s16, w1_s16 = feats1["s16"].shape
            _, _, h2_s16, w2_s16 = feats2["s16"].shape

            def _align_tokens(img_tok, h_depth, w_depth):
                """Bilinearly resample img tokens to depth s16 grid if sizes differ."""
                N_img = img_tok.shape[1]
                if N_img == h_depth * w_depth:
                    return img_tok
                # Infer img patch grid: assume square-ish; use sqrt heuristic.
                h_img = int(N_img ** 0.5)
                w_img = N_img // h_img
                B_, _, D = img_tok.shape
                img_map = img_tok.transpose(1, 2).reshape(B_, D, h_img, w_img)
                img_map = F.interpolate(
                    img_map.float(), size=(h_depth, w_depth),
                    mode="bilinear", align_corners=False,
                ).to(img_tok.dtype)
                return img_map.flatten(2).transpose(1, 2)   # (B, h*w, D)

            img_t1 = _align_tokens(img_t1, h1_s16, w1_s16)
            img_t2 = _align_tokens(img_t2, h2_s16, w2_s16)

            # Additive fusion: depth tokens + image tokens
            t1 = t1 + img_t1
            t2 = t2 + img_t2

        # ── 4. Cross-attention ───────────────────────────────────────────
        t1_out, t2_out, cam_tok1, cam_tok2 = self.cross_attn(t1, t2, T_12=T_12)
        # t1_out, t2_out : (B, N, token_dim)

        # ── 5. Build point priors + decode both views ────────────────────────
        # K is required to build the geometric prior; if not provided, fall
        # back to a unit approximation (X=Y=0, Z=depth) which is also safe
        # since the residual head is zero-initialised.
        if K is not None:
            # Point prior for view 1: unproject depth1 in view-1 frame.
            point_prior1 = self._unproject(depth1, K)          # (B, 3, H1, W1)

            # Point prior for view 2: unproject depth2 in view-2 frame, then
            # warp into view-1 frame via T_21 = se3_inv(T_12).
            if T_12 is not None:
                T_21 = se3_inv(T_12)                           # cam2 → cam1
                pp2_cam2 = self._unproject(depth2, K)          # (B, 3, H2, W2)
                point_prior2 = transform_pts(pp2_cam2, T_21)  # (B, 3, H2, W2)
            else:
                # No pose: view-2 prior stays in view-2 frame (best we can do)
                point_prior2 = self._unproject(depth2, K)
        else:
            # Fallback: trivial prior [0, 0, depth] — residual head corrects.
            zeros = torch.zeros_like(depth1)
            point_prior1 = torch.cat([zeros, zeros, depth1], dim=1)
            zeros2 = torch.zeros_like(depth2)
            point_prior2 = torch.cat([zeros2, zeros2, depth2], dim=1)

        # Optional isotropic scale normalisation — applied identically to X, Y, Z
        # of both priors so the decoder's residual head operates in ~O(1) units.
        # Zero-initialised residual head is a no-op at t=0 regardless of scale.
        if point_norm_scale is not None:
            point_prior1 = point_prior1 / point_norm_scale
            point_prior2 = point_prior2 / point_norm_scale

        # depth_input passed to decoder is the *original* (un-capped,
        # un-normalised) point prior so the residual head corrects in
        # metric-metres space.
        out1 = self.decoder(t1_out, feats1, point_prior=point_prior1, input_hw=(H1, W1))
        out2 = self.decoder(t2_out, feats2, point_prior=point_prior2, input_hw=(H2, W2))

        # ── 6. Pose prediction + optional iterative GRU refinement ────────────
        # PoseHead uses encoder global features (view-specific, before cross-attn).
        # When pose_refine_iters > 0 and K is provided, the coarse estimate is
        # refined by a ConvGRU that warps cross-attn spatial tokens geometrically.
        # With pose_refine_iters=0 (default) or K=None this block is identical
        # to the original single-shot PoseHead path — no extra cost.
        pose_preds: dict = {}
        if self.predict_pose:
            enc1_g = feats1["s16"].mean(dim=[2, 3])   # (B, feature_dim)
            enc2_g = feats2["s16"].mean(dim=[2, 3])   # (B, feature_dim)
            pose_preds = self.pose_head(enc1_g, enc2_g)

            if hasattr(self, "pose_refiner") and K is not None:
                # Geometric GRU refinement active.
                # If K is None (e.g. inference without intrinsics) the refiner
                # is skipped and the coarse PoseHead result is kept.
                _, _, h14, w14 = feats1["s16"].shape
                T_12_iters, T_21_iters = self.pose_refiner(
                    T_0         = pose_preds["T_12_pred"],
                    spatial1    = t1_out,
                    spatial2    = t2_out,
                    depth_mono1 = depth1,
                    K           = K,
                    conf1       = out1["confidence"],
                    H=H1, W=W1, h14=h14, w14=w14,
                )
                # Last iterate becomes the final prediction;
                # earlier iterates are exposed for deep supervision in losses.py.
                pose_preds = {
                    **pose_preds,
                    "T_12_pred":  T_12_iters[-1],
                    "T_21_pred":  T_21_iters[-1],
                    "T_12_iters": T_12_iters[:-1],   # [T_1 … T_{N-1}] for pose_iters_loss
                }

        return {
            "point1":           out1["point"],
            "point2":           out2["point"],
            "confidence1":      out1["confidence"],
            "confidence2":      out2["confidence"],
            "point_norm_scale": point_norm_scale,
            **pose_preds,
        }


# ---------------------------------------------------------------------------
# Config-driven constructor
# ---------------------------------------------------------------------------

def build_depth_only_net(cfg: dict) -> DepthOnlyNet:
    """
    Instantiate DepthOnlyNet from an arch.yaml 'depth_only' config dict.

    Example arch.yaml section::

        depth_only:
          feature_dim:    256
          token_dim:      256
          num_blocks:     4
          num_heads:      8
          mlp_ratio:      4.0
          decoder_hidden: 128
          use_pose_token: true
          freeze_encoder: false
    """
    c = cfg.get("depth_only", cfg)
    return DepthOnlyNet(
        feature_dim      = c.get("feature_dim",      256),
        token_dim        = c.get("token_dim",         768),
        num_blocks       = c.get("num_blocks",          4),
        num_heads        = c.get("num_heads",          12),
        mlp_ratio        = c.get("mlp_ratio",         4.0),
        decoder_hidden   = c.get("decoder_hidden",   128),
        use_pose_token   = c.get("use_pose_token",   True),
        freeze_encoder   = c.get("freeze_encoder",  False),
        mast3r_ckpt      = c.get("mast3r_ckpt",      None),
        freeze_cross_attn = c.get("freeze_cross_attn", False),
        max_encode_hw    = tuple(c["max_encode_hw"]) if "max_encode_hw" in c else (480, 640),
        predict_pose         = c.get("predict_pose",          False),
        pose_head_hidden     = c.get("pose_head_hidden",        128),
        pose_refine_iters    = c.get("pose_refine_iters",         0),
        pose_refine_feat_dim = c.get("pose_refine_feat_dim",     128),
        pose_refine_hidden   = c.get("pose_refine_hidden",       128),
        use_image_encoder    = c.get("use_image_encoder",       False),
        image_encoder_ckpt   = c.get("image_encoder_ckpt",       None),
    )
