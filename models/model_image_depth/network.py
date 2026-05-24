"""
network.py — Top-level DepthAlignNet: shared encoder + local cross-attention
             + iterative refinement + lightweight decoder.

Input contract (per batch)
--------------------------
  rgb1        : (B, 3, H, W)
  rgb2        : (B, 3, H, W)
  depth_mono1 : (B, 1, H, W)   monocular depth prior, view 1  (metres)
  depth_mono2 : (B, 1, H, W)   monocular depth prior, view 2  (metres)
  T_12        : (B, 4, 4)      cam1 → cam2 rigid transform
  K           : (B, 3, 3)      camera intrinsics (original image resolution)

Output
------
  depth1      : (B, 1, H/2, W/2)  aligned depth, view 1
  depth2      : (B, 1, H/2, W/2)  aligned depth, view 2
  confidence1 : (B, 1, H/2, W/2)
  confidence2 : (B, 1, H/2, W/2)
  depth1_iters: list of (B, 1, H/16, W/16)  for deep supervision
  depth2_iters: list of (B, 1, H/16, W/16)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder    import SharedEncoder
from .refinement import IterativeRefinement
from .decoder    import DepthDecoder
from .geometry   import rot6d_to_matrix, svd_orthogonalize, se3_inv


# ---------------------------------------------------------------------------
# Camera parameter prediction head
# ---------------------------------------------------------------------------

class CameraHead(nn.Module):
    """
    Decode a single camera-token embedding into intrinsics and a per-prediction
    confidence score.  Weight-tied across both views (same physical camera).

    Intrinsics — normalised by image size so the head is resolution-agnostic:
      fx = (softplus(·) + 0.3) * W   →  always positive; default ≈ 0.99 W
      fy = (softplus(·) + 0.3) * H
      cx = sigmoid(·) * W            →  ∈ (0, W); default = 0.5 W
      cy = sigmoid(·) * H

    Confidence — log-scale scalar s_K for heteroscedastic uncertainty weighting
      (Kendall & Gal, NeurIPS 2017).  Initialised to 0 (σ = 1, neutral weighting).
    """

    def __init__(self, feat_dim: int, hidden: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.to_intrinsics = nn.Linear(hidden, 4)   # fx_n, fy_n, cx_n, cy_n
        self.to_log_conf_K = nn.Linear(hidden, 1)   # log-confidence for intrinsics

        # Zero init → softplus(0)+0.3≈0.99 focal, sigmoid(0)=0.5 pp.
        nn.init.zeros_(self.to_intrinsics.weight)
        nn.init.zeros_(self.to_intrinsics.bias)

        # Confidence: zero init → log_conf=0 → σ=1 (no scaling at start).
        nn.init.zeros_(self.to_log_conf_K.weight)
        nn.init.zeros_(self.to_log_conf_K.bias)

    def forward(self, cam_embed: torch.Tensor, H: int, W: int) -> dict:
        """
        cam_embed : (B, feat_dim)
        H, W      : original image height / width (pixels)
        Returns { "K": (B, 3, 3), "log_conf_K": (B,) }
        """
        h    = self.mlp(cam_embed)                           # (B, hidden)
        intr = self.to_intrinsics(h)                         # (B, 4)

        fx = (F.softplus(intr[:, 0]) + 0.3) * W
        fy = (F.softplus(intr[:, 1]) + 0.3) * H
        cx = torch.sigmoid(intr[:, 2]) * W
        cy = torch.sigmoid(intr[:, 3]) * H

        B = cam_embed.shape[0]
        K = torch.zeros(B, 3, 3, device=cam_embed.device, dtype=cam_embed.dtype)
        K[:, 0, 0] = fx;  K[:, 1, 1] = fy
        K[:, 0, 2] = cx;  K[:, 1, 2] = cy
        K[:, 2, 2] = 1.0

        log_conf_K = self.to_log_conf_K(h).squeeze(-1)       # (B,)
        return {"K": K, "log_conf_K": log_conf_K}


# ---------------------------------------------------------------------------
# Relative pose prediction head
# ---------------------------------------------------------------------------

class RelativePoseHead(nn.Module):
    """
    Predict the relative SE(3) transform T_12 (cam1 → cam2) *directly* from
    the concatenated camera embeddings of both views.

    Input  : cat([cam_embed_1, cam_embed_2], dim=-1)  (B, 2*feat_dim)
    Output : T_12  (B, 4, 4)  — rotation via SVD Procrustes projection onto SO(3),
                                 translation as a direct 3-D vector.
             log_conf_pose (B,) — heteroscedastic log-confidence (Kendall & Gal 2017).

    By concatenating both embeddings the head sees the relationship between
    views in a single forward pass, rather than predicting two independent
    absolute world-frame poses whose difference we then hope is meaningful.
    The ordering [embed_1, embed_2] encodes the 1→2 direction so the same
    head with swapped inputs gives T_21.

    Initialised so the prior pose is the identity transform and confidence = 1.
    """

    def __init__(self, in_dim: int, hidden: int = 256, dropout: float = 0.0):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.to_pose          = nn.Linear(hidden, 9)   # 6D rotation + 3D translation
        self.to_log_conf_pose = nn.Linear(hidden, 1)

        # Bias → identity rotation in 6-D repr + small non-zero translation.
        # Initializing translation strictly to 0 makes its norm 0, meaning
        # normalized_translation_loss produces a math domain error or exactly
        # 1.5708 (pi/2) because dot([0,0,0], GT) = 0. We shift it slightly.
        nn.init.zeros_(self.to_pose.weight)
        with torch.no_grad():
            self.to_pose.bias.copy_(
                torch.tensor([1., 0., 0., 0., 1., 0.,   # identity in 6D
                               0., 0., 1e-4])             # non-zero translation
            )
        nn.init.zeros_(self.to_log_conf_pose.weight)
        nn.init.zeros_(self.to_log_conf_pose.bias)

    def forward(self, rel_embed: torch.Tensor) -> dict:
        """
        rel_embed : (B, 2*feat_dim) — cat([cam_embed_1, cam_embed_2])
        Returns { "T_12": (B, 4, 4), "log_conf_pose": (B,) }
        """
        h        = self.mlp(rel_embed)                                           # (B, hidden)
        pose_raw = self.to_pose(h)                                               # (B, 9)
        # 6D rotation (Zhou et al. 2019) via Gram-Schmidt:
        # Perfectly stable and avoids the NaN/Instability issues of SVD
        # when initialized to exactly the identity matrix.
        R        = rot6d_to_matrix(pose_raw[:, :6])                              # (B, 3, 3) ∈ SO(3)
        t        = pose_raw[:, 6:]                                               # (B, 3)

        B = rel_embed.shape[0]
        T_12 = (torch.eye(4, device=rel_embed.device, dtype=rel_embed.dtype)
                    .unsqueeze(0).expand(B, -1, -1).clone())
        T_12[:, :3, :3] = R
        T_12[:, :3,  3] = t

        log_conf_pose = self.to_log_conf_pose(h).squeeze(-1)  # (B,)
        return {"T_12": T_12, "log_conf_pose": log_conf_pose}


class DepthAlignNet(nn.Module):
    """
    Lightweight explicit-geometry depth alignment network.

    Parameters
    ----------
    feat_dim     : int   encoder output channels (each scale)
    hidden_dim   : int   ConvGRU hidden state dimension
    num_iters    : int   refinement iterations
    num_heads    : int   attention heads
    window_size  : int   local attention window
    pretrained   : bool  load ImageNet ConvNeXt-Tiny weights
    """

    def __init__(
        self,
        feat_dim        : int  = 128,
        hidden_dim      : int  = 128,
        num_iters       : int  = 4,
        num_heads       : int  = 4,
        window_size     : int  = 7,
        pretrained      : bool = True,
        freeze_backbone : bool = False,
        use_refinement  : bool = True,
        decoder_hidden  : int  = 64,
        camera_head_hidden: int  = 64,
    ):
        super().__init__()
        self.use_refinement = use_refinement

        # Shared encoder (weight-tied across views)
        self.encoder = SharedEncoder(pretrained=pretrained, out_channels=feat_dim, freeze_backbone=freeze_backbone)

        # Single shared cross-attention for s16 (camera token prepended) and
        # s8 (spatial only).  Weight-sharing across scales halves attention params.
        self.attn_cross   = nn.MultiheadAttention(feat_dim, num_heads, batch_first=True)
        self.attn_norm_q  = nn.LayerNorm(feat_dim)
        self.attn_norm_kv = nn.LayerNorm(feat_dim)
        # Normalise raw encoder features before building attention sequences.
        # Keeps feature magnitudes bounded as the projection layers train.
        self.feat_norm    = nn.LayerNorm(feat_dim)

        # Iterative refinement operates at 1/16 resolution for speed (optional)
        if use_refinement:
            self.refine = IterativeRefinement(
                feat_dim=feat_dim, hidden_dim=hidden_dim, num_iters=num_iters
            )

        # Decoder
        self.decoder = DepthDecoder(feat_dim=feat_dim, hidden=decoder_hidden)

        # Single shared camera token (ViSTA-SLAM style).
        # Both views use the same initial token value; attending to different
        # context sequences produces different embeddings for each view.
        self.camera_token = nn.Parameter(torch.empty(1, 1, feat_dim))
        nn.init.trunc_normal_(self.camera_token, std=0.02)

        # Weight-tied intrinsics head (same physical camera for both views).
        self.camera_head = CameraHead(feat_dim, hidden=camera_head_hidden)

        # Relative pose head — predicts T_12 directly from both embeddings.
        # Takes cat([cam_embed_1, cam_embed_2]) so it directly encodes the
        # view relationship rather than two independent absolute poses.
        self.relative_pose_head = RelativePoseHead(2 * feat_dim, hidden=camera_head_hidden)

    # ------------------------------------------------------------------
    # Intrinsics scaling
    # ------------------------------------------------------------------
    @staticmethod
    def _scale_K(K: torch.Tensor, src_h: int, src_w: int, dst_h: int, dst_w: int):
        """Return K rescaled to match a downsampled feature map."""
        K = K.clone()
        K[:, 0, 0] *= dst_w / src_w   # fx
        K[:, 1, 1] *= dst_h / src_h   # fy
        K[:, 0, 2] *= dst_w / src_w   # cx
        K[:, 1, 2] *= dst_h / src_h   # cy
        return K

    # ------------------------------------------------------------------
    # Per-view cross-attention  (operates on pre-encoded features)
    # ------------------------------------------------------------------
    def _cross_attend(
        self,
        feats_q:   dict,           # pre-encoded query features {s4, s8, s16}
        feats_ctx: dict,           # pre-encoded context features
        cam_token: torch.Tensor,   # (B, 1, C) shared camera token
        depth_q:   torch.Tensor,   # (B, 1, H, W) mono depth of query view
        depth_ctx: torch.Tensor,   # (B, 1, H, W) mono depth of context view
        K:         torch.Tensor,   # (B, 3, 3) full-resolution intrinsics
        H: int, W: int,
    ) -> tuple:
        """
        s16 — global cross-attention over [camera_token | spatial_flat].
              Camera token extracted from position 0 → CameraHead.
              Spatial tokens (1:) reshaped back to (B, C, H16, W16).
        s8  — same shared attn_cross over flattened spatial tokens only.

        Returns (attended_feats_q, cam_embed, depth_q_s16, depth_ctx_s16, K_s16).
        """
        B, C, H16, W16 = feats_q["s16"].shape
        _, _,  H8,  W8 = feats_q["s8"].shape
        K_s16 = self._scale_K(K, H, W, H16, W16)

        depth_q_s16   = F.interpolate(depth_q,   size=(H16, W16), mode="nearest")
        depth_ctx_s16 = F.interpolate(depth_ctx, size=(H16, W16), mode="nearest")

        # ── s16: prepend camera token then cross-attend ───────────────────
        sp_q   = self.feat_norm(feats_q["s16"].permute(0, 2, 3, 1).reshape(B, H16 * W16, C))
        sp_ctx = self.feat_norm(feats_ctx["s16"].permute(0, 2, 3, 1).reshape(B, H16 * W16, C))
        gap = feats_q["s16"].mean(dim=[2, 3], keepdim=False).unsqueeze(1)  # (B, 1, C)
        cam_token_enriched = cam_token + gap #enrich camera token with global average pooled feature (global view)
        seq_q = torch.cat([cam_token_enriched, sp_q], dim=1)   # (B, 1+H16*W16, C)
        seq_ctx = torch.cat([cam_token, sp_ctx], dim=1)
        out, _ = self.attn_cross(self.attn_norm_q(seq_q), self.attn_norm_kv(seq_ctx), self.attn_norm_kv(seq_ctx))
        seq_q  = seq_q + out
        cam_embed = seq_q[:, 0, :]                                               # (B, C)
        f16 = seq_q[:, 1:, :].reshape(B, H16, W16, C).permute(0, 3, 1, 2)      # (B,C,H16,W16)

        # ── s8: shared cross-attention, spatial tokens only ───────────────
        s8_q   = self.feat_norm(feats_q["s8"].permute(0, 2, 3, 1).reshape(B, H8 * W8, C))
        s8_ctx = self.feat_norm(feats_ctx["s8"].permute(0, 2, 3, 1).reshape(B, H8 * W8, C))
        out8, _ = self.attn_cross(self.attn_norm_q(s8_q), self.attn_norm_kv(s8_ctx), self.attn_norm_kv(s8_ctx))
        f8 = (s8_q + out8).reshape(B, H8, W8, C).permute(0, 3, 1, 2)           # (B,C,H8,W8)

        attended = {"s4": feats_q["s4"], "s8": f8, "s16": f16}
        return attended, cam_embed, depth_q_s16, depth_ctx_s16, K_s16

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        rgb1:        torch.Tensor,
        rgb2:        torch.Tensor,
        depth_mono1: torch.Tensor,
        depth_mono2: torch.Tensor,
        T_12:        torch.Tensor,   # current relative-pose estimate (cam1→cam2)
        K:           torch.Tensor,   # current intrinsics estimate
    ) -> dict:
        """
        T_12 and K are treated as *current estimates* fed from the iterative
        loop in train.py (identity pose + focal-length prior on the first
        iteration; network predictions on subsequent iterations).
        The camera head predicts updated K_pred / T_12_pred which are
        returned in the output dict for the next iteration.
        """
        B, _, H, W = rgb1.shape
        #T_21 = torch.linalg.inv(T_12)
        T_21 = se3_inv(T_12)

        # ── Encode both views once (weight-tied encoder) ─────────────────
        x1 = torch.cat([rgb1, depth_mono1], dim=1)   # (B, 4, H, W)
        x2 = torch.cat([rgb2, depth_mono2], dim=1)
        feats1_raw = self.encoder(x1)                  # {"s4", "s8", "s16"}
        feats2_raw = self.encoder(x2)

        # ── Expand shared camera token to batch size ─────────────────────
        cam_tok = self.camera_token.expand(B, -1, -1)   # (B, 1, C)

        # ── Cross-attend at s16 (camera token prepended) + s8 (spatial) ──
        feats1, cam_embed_1, d1_s16, d2_s16, K_s16 = self._cross_attend(
            feats1_raw, feats2_raw, cam_tok, depth_mono1, depth_mono2, K, H, W,
        )
        feats2_attended, cam_embed_2, d2_s16_v, d1_s16_v, _ = self._cross_attend(
            feats2_raw, feats1_raw, cam_tok, depth_mono2, depth_mono1, K, H, W,
        )

        # ── Camera predictions from attended tokens ───────────────────────
        cam1 = self.camera_head(cam_embed_1, H, W)
        cam2 = self.camera_head(cam_embed_2, H, W)
        # Average intrinsics (same physical camera for both views)
        K_pred     = (cam1["K"] + cam2["K"]) * 0.5
        log_conf_K = (cam1["log_conf_K"] + cam2["log_conf_K"]) * 0.5
        # Predict relative pose directly from both embeddings — avoids the
        # underconstrained absolute-pose regression with no world reference.
        # Swapping the concatenation order gives T_21 from the same shared
        # weights, so the head learns a consistent notion of direction.
        rel_embed_12  = torch.cat([cam_embed_1, cam_embed_2], dim=-1)  # (B, 2C)
        rel_embed_21  = torch.cat([cam_embed_2, cam_embed_1], dim=-1)  # (B, 2C)
        rel_pose_12   = self.relative_pose_head(rel_embed_12)
        rel_pose_21   = self.relative_pose_head(rel_embed_21)
        T_12_pred     = rel_pose_12["T_12"]                            # (B, 4, 4)
        T_21_pred     = rel_pose_21["T_12"]                            # (B, 4, 4)
        log_conf_pose = rel_pose_12["log_conf_pose"]                   # (B,)

        # ── Iterative refinement (at s16 resolution) ─────────────────────
        if self.use_refinement:
            ref1 = self.refine(
                depth_mono=d1_s16,
                depth2_mono=d2_s16,
                feat_cross=feats1["s16"],
                T_12=T_12,
                K=K_s16,
            )
            ref2 = self.refine(
                depth_mono=d2_s16_v,
                depth2_mono=d1_s16_v,
                feat_cross=feats2_attended["s16"],
                T_12=T_21,
                K=K_s16,
            )
            depth1_refined = ref1["depth"]
            depth2_refined = ref2["depth"]
            depth1_iters   = ref1["depth_iters"]
            depth2_iters   = ref2["depth_iters"]
            scale1, bias1  = ref1["scale"], ref1["bias"]
            scale2, bias2  = ref2["scale"], ref2["bias"]
        else:
            depth1_refined = d1_s16
            depth2_refined = d2_s16_v
            depth1_iters   = []
            depth2_iters   = []
            scale1 = bias1 = scale2 = bias2 = None

        # ── Decoder (upsample to H/2, W/2) ───────────────────────────────
        dec1 = self.decoder(feats1,          depth1_refined, depth_mono1)
        dec2 = self.decoder(feats2_attended, depth2_refined, depth_mono2)

        return {
            "depth1":       dec1["depth"],        # (B, 1, H/2, W/2)
            "depth2":       dec2["depth"],
            "confidence1":  dec1["confidence"],
            "confidence2":  dec2["confidence"],
            "depth1_iters": depth1_iters,         # list[(B, 1, H/16, W/16)] or []
            "depth2_iters": depth2_iters,
            "scale1":       scale1,
            "bias1":        bias1,
            "scale2":       scale2,
            "bias2":        bias2,
            # Camera predictions — fed back as K / T_12 on the next iteration
            "K_pred":        K_pred,               # (B, 3, 3)
            "T_12_pred":     T_12_pred,             # (B, 4, 4)  cam1→cam2  (direct)
            "T_21_pred":     T_21_pred,             # (B, 4, 4)  cam2→cam1  (direct, swapped embeds)
            "log_conf_K":    log_conf_K,            # (B,)  log-confidence for intrinsics
            "log_conf_pose": log_conf_pose,         # (B,)  log-confidence for pose
        }


# ---------------------------------------------------------------------------
# Quick shape-check (no GPU required)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    model = DepthAlignNet(pretrained=True, num_iters=2, freeze_backbone=True, use_refinement=False)
    model.eval()

    B, H, W = 2, 480, 640
    rgb1  = torch.randn(B, 3, H, W)
    rgb2  = torch.randn(B, 3, H, W)
    dm1   = torch.rand(B, 1, H, W) + 0.1
    dm2   = torch.rand(B, 1, H, W) + 0.1
    # K and T_12 here are the *initial estimates* (as in training iteration 0)
    T_12  = torch.eye(4).unsqueeze(0).expand(B, -1, -1).contiguous()
    fx    = float(max(H, W)) * 0.9
    K     = torch.tensor([[fx, 0, W/2], [0, fx, H/2], [0, 0, 1]],
                          dtype=torch.float32).unsqueeze(0).expand(B, -1, -1).contiguous()

    with torch.no_grad():
        out = model(rgb1, rgb2, dm1, dm2, T_12, K)

    print("── Depth / feature outputs ──")
    for k, v in out.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: {tuple(v.shape)}")
        elif isinstance(v, list):
            if len(v) == 0:
                print(f"  {k}: empty list")
            else:
                print(f"  {k}: list of {len(v)} × {tuple(v[0].shape)}")

    print("\n── Predicted camera parameters (first sample in batch) ──")
    K_p = out["K_pred"][0]
    print(f"  K_pred  fx={K_p[0,0]:.1f}  fy={K_p[1,1]:.1f}"
          f"  cx={K_p[0,2]:.1f}  cy={K_p[1,2]:.1f}")
    print(f"  T_12_pred:\n{out['T_12_pred'][0].numpy()}")

    #Print model size
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainable_names = [n for n, p in model.named_parameters() if p.requires_grad]
    print(f"\nTotal params:     {total_params/1e6:.2f}M")
    print(f"Trainable params: {trainable_params/1e6:.2f}M")
    print("Trainable names sample:", trainable_names[:20])
