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

from models.model_image_depth.geometry import se3_inv

from .encoder        import DepthEncoder
from .cross_attention import CrossAttentionDecoder, _mast3r_init, _vec9_to_se3, _se3_inv
from .decoder        import DepthDecoder



# ---------------------------------------------------------------------------
# Pose head  (used when predict_pose=True)
# ---------------------------------------------------------------------------

class PoseHead(nn.Module):
    """
    Regress T_12 and T_21 from the evolved camera tokens produced by
    CrossAttentionDecoder when predict_pose=True.

    T_12 = f(cam_tok1, cam_tok2)
    T_21 = f(cam_tok2, cam_tok1)   ← same weights, swapped order

    Outputs T_12_pred, T_21_pred (B,4,4) and per-sample log-confidence
    scalars for heteroscedastic uncertainty weighting in camera_pose_loss.
    """

    def __init__(self, token_dim: int, hidden: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2 * token_dim, hidden),
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
        cam_tok1: torch.Tensor,   # (B, token_dim)  evolved camera token view-1
        cam_tok2: torch.Tensor,   # (B, token_dim)  evolved camera token view-2
    ) -> dict:
        B = cam_tok1.shape[0]
        # T_12: conditioned on (tok1, tok2)
        T_12_pred = _vec9_to_se3(self.mlp(torch.cat([cam_tok1, cam_tok2], dim=-1)))
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
    feature_dim    : int   Encoder output channels (projected to feature_dim).
    token_dim      : int   Cross-attention dimension.  If token_dim ≠ feature_dim
                           an enc_to_dec linear projection is inserted.
    num_blocks     : int   Number of symmetric CrossBlocks.
    num_heads      : int   Attention heads per block.
    mlp_ratio      : float MLP hidden / token_dim ratio.
    decoder_hidden : int   Decoder internal channel width.
    use_pose_token : bool  Prepend pose-conditioned token before cross-attn.
    freeze_encoder : bool  Freeze ConvNeXt backbone (first-conv stays trainable).
    """

    def __init__(
        self,
        feature_dim:      int   = 256,
        token_dim:        int   = 768,
        num_blocks:       int   = 4,
        num_heads:        int   = 12,
        mlp_ratio:        float = 4.0,
        decoder_hidden:   int   = 128,
        use_pose_token:   bool  = True,
        freeze_encoder:   bool  = False,
        mast3r_ckpt:      str | None = None,
        freeze_cross_attn: bool = False,
        max_encode_hw:    tuple | None = (480, 640),
        predict_pose:     bool  = False,
        pose_head_hidden: int   = 128,
    ):
        super().__init__()

        self.token_dim       = token_dim
        self.feature_dim     = feature_dim
        self.predict_pose    = predict_pose
        self.max_encode_hw   = max_encode_hw

        # ── Shared depth encoder ─────────────────────────────────────────
        self.encoder = DepthEncoder(
            pretrained      = True,
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
        # Reads evolved camera tokens → T_12_pred, T_21_pred.
        # When predict_pose=False this is absent and no pose loss fires.
        if predict_pose:
            self.pose_head = PoseHead(token_dim, hidden=pose_head_hidden)

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
        depth1: torch.Tensor,              # (B, 1, H1, W1) MDE depth view 1
        depth2: torch.Tensor,              # (B, 1, H2, W2) MDE depth view 2
        T_12:   torch.Tensor | None = None, # (B, 4, 4)  pose view-1 → view-2
    ) -> dict:
        """
        Parameters
        ----------
        depth1, depth2 : (B, 1, H, W)  Raw MDE depths (metres).  H, W are the
                         *colour image* resolution (variable; not necessarily
                         640×480).
        T_12           : (B, 4, 4)  Relative pose.  Required when
                         use_pose_token=True.

        Returns
        -------
        {
          "depth1"      : (B, 1, 480, 640),
          "depth2"      : (B, 1, 480, 640),
          "confidence1" : (B, 1, 480, 640),
          "confidence2" : (B, 1, 480, 640),
          "log_scale1"  : (B, 1, 480, 640),
          "log_scale2"  : (B, 1, 480, 640),
        }
        """
        # ── 1. Cap input resolution to bound s16 token count ─────────────────
        # depth_input (passed to decoder for the residual prior) stays at
        # the *original* resolution; only the encoder input is capped.
        d1_enc = self._cap_resolution(depth1)
        d2_enc = self._cap_resolution(depth2)

        # ── 2. Shared encoder ────────────────────────────────────────────
        feats1 = self.encoder(d1_enc)   # {s4, s8, s16}
        feats2 = self.encoder(d2_enc)   # {s4, s8, s16}

        # Remember spatial dims for reshaping tokens back to spatial maps.
        _, _, H1, W1 = depth1.shape
        _, _, H2, W2 = depth2.shape

        # ── 3. Flatten s16 → tokens + optional enc_to_dec projection ────
        t1 = self._flatten_s16(feats1)   # (B, N1, token_dim)
        t2 = self._flatten_s16(feats2)   # (B, N2, token_dim)

        # ── 4. Cross-attention ───────────────────────────────────────────
        t1_out, t2_out, cam_tok1, cam_tok2 = self.cross_attn(t1, t2, T_12=T_12)
        # t1_out, t2_out : (B, N, token_dim)

        # ── 5. Decode both views ─────────────────────────────────────────
        # depth_input is the *original* (un-capped, un-normalised) depth so
        # the residual head corrects in metric-metres space.
        out1 = self.decoder(t1_out, feats1, depth_input=depth1, input_hw=(H1, W1))
        out2 = self.decoder(t2_out, feats2, depth_input=depth2, input_hw=(H2, W2))

        return {
            "depth1":      out1["depth"],
            "depth2":      out2["depth"],
            "confidence1": out1["confidence"],
            "confidence2": out2["confidence"],
            "log_scale1":  out1["log_scale"],
            "log_scale2":  out2["log_scale"],
            # Pose predictions (only when predict_pose=True).
            # .detach() on cam_tok: pose-head gradients must not flow back
            # through CrossAttentionDecoder → encoder (change A).  The large
            # initial translation gradient (~10× amplified through the norm
            # chain-rule) would otherwise corrupt the depth cross-attention
            # features before either task has converged.
            **(self.pose_head(cam_tok1.detach(), cam_tok2.detach()) if self.predict_pose else {}),
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
        predict_pose     = c.get("predict_pose",    False),
        pose_head_hidden = c.get("pose_head_hidden", 128),
    )
