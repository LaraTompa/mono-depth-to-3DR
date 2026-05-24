"""
network.py — DepthAlignNetV2: ViSTA-SLAM inspired depth alignment model.

Input contract (identical to DepthAlignNet V1)
----------------------------------------------
  rgb1        : (B, 3, H, W)   — H, W must be multiples of 14 (e.g. 392×518)
  rgb2        : (B, 3, H, W)
  depth_mono1 : (B, 1, H, W)   monocular depth prior (metres)
  depth_mono2 : (B, 1, H, W)
  T_12        : (B, 4, 4)      current cam1→cam2 estimate
  K           : (B, 3, 3)      current intrinsics estimate

Output contract (identical to V1 — same loss / metric code)
------------------------------------------------------------
  depth1, depth2         : (B, 1, H/2, W/2)
  confidence1, confidence2: (B, 1, H/2, W/2)
  depth1_iters, depth2_iters: []   (no iterative refinement in this model)
  scale1/bias1/scale2/bias2  : None
  K_pred, T_12_pred, T_21_pred : (B,3,3), (B,4,4), (B,4,4)
  log_conf_K, log_conf_pose: (B,)

Architecture
------------

  rgb1 ──► DinoEncoder (frozen ViT-L/14) ──► tokens1 (B, N, 1024)
  rgb2 ──► DinoEncoder                   ──► tokens2 (B, N, 1024)
               │
               ▼  enc_to_dec  Linear(1024 → 768)  [trainable]
               │
  [cam_tok | tokens1] ──►  CrossAttentionDecoder  ◄── [cam_tok | tokens2]
                                    │  (N blocks, init from MASt3R)
                                    ▼
                          cam_embed1, spatial1,  cam_embed2, spatial2
                                    │
  rgb1+depth1 ──► DepthStream ──► {s4,s8,s16}₁  (ConvNeXt-Small, trainable)
  rgb2+depth2 ──► DepthStream ──► {s4,s8,s16}₂
                                    │
                          FusionDecoder (late fusion)
                                    │
                         depth1/conf1  depth2/conf2  @ H/2

  CameraHead(cam_embed1, cam_embed2) → K_pred, T_12_pred
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder         import DinoEncoder
from .cross_attention import CrossAttentionDecoder
from .depth_stream    import DepthStream
from .decoder         import FusionDecoder

# Re-use CameraHead from V1; pose is now handled by PoseRefinementModule.
from models.model_image_depth.network import CameraHead
from .pose_refinement import PoseRefinementModule


class DepthAlignNetV2(nn.Module):
    """
    ViSTA-SLAM inspired depth alignment model with late fusion.

    Parameters
    ----------
    dino_model         : str   torch.hub model name for DINOv2
    freeze_dino        : bool  freeze DINOv2 weights (almost always True)
    decoder_dim        : int   cross-attention token dimension (768 for MASt3R BaseDecoder)
    num_decoder_blocks : int   number of CrossAttentionDecoder blocks
    num_decoder_heads  : int   attention heads in each block (12 for dim=768)
    depth_out_channels : int   DepthStream output channels per scale
    decoder_hidden     : int   FusionDecoder internal width
    camera_head_hidden : int   CameraHead MLP hidden width
    mast3r_ckpt        : str | None  path to MASt3R checkpoint for cross-attn init
    """

    def __init__(
        self,
        dino_model         : str        = "dinov2_vitl14",
        freeze_dino        : bool       = True,
        depth_backbone     : str        = "convnext_tiny",
        decoder_dim        : int        = 768,
        num_decoder_blocks : int        = 4,
        num_decoder_heads  : int        = 12,
        depth_out_channels : int        = 128,
        decoder_hidden     : int        = 256,
        camera_head_hidden : int        = 256,
        num_pose_iters     : int        = 4,
        mast3r_ckpt        : str | None = None,
    ):
        super().__init__()

        # ── Frozen DINOv2 RGB encoder ────────────────────────────────────
        self.dino = DinoEncoder(model_name=dino_model, freeze=freeze_dino)
        dino_dim  = self.dino.embed_dim   # 1024 for ViT-L/14

        # ── Projection: DINOv2 dim → decoder dim ─────────────────────────
        # MASt3R BaseDecoder uses 768-dim while DINOv2 ViT-L outputs 1024-dim.
        # This trainable linear bridges the gap.  It is NOT loaded from the
        # MASt3R checkpoint (MASt3R has its own encoder, not DINOv2).
        self.enc_to_dec = nn.Linear(dino_dim, decoder_dim)
        nn.init.xavier_uniform_(self.enc_to_dec.weight)
        nn.init.zeros_(self.enc_to_dec.bias)

        # ── Shared learnable camera token ────────────────────────────────
        # One token is prepended to the patch sequence for each view.
        # After cross-attention, its output is fed to CameraHead.
        # (ViSTA-SLAM style: camera token attends to ALL patches globally.)
        self.camera_token = nn.Parameter(torch.empty(1, 1, decoder_dim))
        nn.init.trunc_normal_(self.camera_token, std=0.02)

        # ── Cross-attention decoder (MASt3R init) ────────────────────────
        self.cross_attn = CrossAttentionDecoder(
            num_blocks=num_decoder_blocks,
            dim=decoder_dim,
            num_heads=num_decoder_heads,
        )
        if mast3r_ckpt is not None:
            self.cross_attn.load_mast3r_weights(mast3r_ckpt)

        # ── Trainable depth stream (late fusion — independent of DINOv2) ─
        self.depth_stream = DepthStream(
            backbone=depth_backbone, pretrained=True, out_channels=depth_out_channels
        )

        # ── Late-fusion decoder ──────────────────────────────────────────
        self.decoder = FusionDecoder(
            token_dim=decoder_dim,
            depth_ch=depth_out_channels,
            hidden=decoder_hidden,
        )

        # ── Camera head (shared / weight-tied across both views) ─────────
        self.camera_head = CameraHead(feat_dim=decoder_dim, hidden=camera_head_hidden)

        # ── Iterative SE(3) pose refinement ─────────────────────────────
        # Bridges the depth and pose streams via feature warping.
        # Runs after FusionDecoder so confidence maps are available.
        self.pose_refiner = PoseRefinementModule(
            token_dim=decoder_dim,
            num_iters=num_pose_iters,
        )

        # Store freeze flag to avoid iterating 307M params every forward.
        self._dino_frozen = freeze_dino

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        rgb1:        torch.Tensor,   # (B, 3, H, W)
        rgb2:        torch.Tensor,
        depth_mono1: torch.Tensor,   # (B, 1, H, W)
        depth_mono2: torch.Tensor,
        T_12:        torch.Tensor,   # (B, 4, 4)  — current estimate, not used in V2
        K:           torch.Tensor,   # (B, 3, 3)  — current estimate, not used in V2
    ) -> dict:
        B, _, H, W = rgb1.shape

        # ── Auto-resize RGB to nearest multiple of 14 for DINOv2 ─────────
        # The depth stream and all downstream ops use the original resolution.
        # Only the ViT patch embedding requires exact divisibility by 14.
        H14 = round(H / 14) * 14
        W14 = round(W / 14) * 14
        if H14 != H or W14 != W:
            rgb1_dino = F.interpolate(rgb1, size=(H14, W14), mode="bilinear", align_corners=False)
            rgb2_dino = F.interpolate(rgb2, size=(H14, W14), mode="bilinear", align_corners=False)
        else:
            rgb1_dino, rgb2_dino = rgb1, rgb2
        h14 = H14 // 14
        w14 = W14 // 14

        # ── 1. DINOv2 encoding (frozen) ──────────────────────────────────
        ctx = torch.no_grad() if self._dino_frozen else torch.enable_grad()
        with ctx:
            patches1, _ = self.dino(rgb1_dino)   # (B, N, 1024)
            patches2, _ = self.dino(rgb2_dino)

        # ── 2. Project to decoder dimension ──────────────────────────────
        t1 = self.enc_to_dec(patches1)   # (B, N, 768)
        t2 = self.enc_to_dec(patches2)

        # ── 3. Prepend camera token ───────────────────────────────────────
        cam_tok = self.camera_token.expand(B, -1, -1)   # (B, 1, 768)
        t1 = torch.cat([cam_tok, t1], dim=1)            # (B, 1+N, 768)
        t2 = torch.cat([cam_tok, t2], dim=1)

        # ── 4. Symmetric cross-attention decoder ─────────────────────────
        t1, t2 = self.cross_attn(t1, t2)                # (B, 1+N, 768) each

        # ── 5. Split camera embedding from spatial tokens ─────────────────
        cam_embed1 = t1[:, 0, :]                         # (B, 768)
        cam_embed2 = t2[:, 0, :]
        spatial1   = t1[:, 1:, :]                        # (B, N, 768)
        spatial2   = t2[:, 1:, :]

        # ── 6. Camera predictions ─────────────────────────────────────────
        cam1 = self.camera_head(cam_embed1, H, W)
        cam2 = self.camera_head(cam_embed2, H, W)

        K_pred     = (cam1["K"] + cam2["K"]) * 0.5
        log_conf_K = (cam1["log_conf_K"] + cam2["log_conf_K"]) * 0.5

        # ── 7. Depth stream (independent of DINOv2) ──────────────────────
        depth_feats1 = self.depth_stream(rgb1, depth_mono1)   # {s4, s8, s16}
        depth_feats2 = self.depth_stream(rgb2, depth_mono2)

        # ── 8. Late-fusion decoder ────────────────────────────────────────
        grid_hw = (h14, w14)
        dec1 = self.decoder(spatial1, depth_feats1, depth_mono1, grid_hw)
        dec2 = self.decoder(spatial2, depth_feats2, depth_mono2, grid_hw)

        # ── 9. Iterative SE(3) pose refinement ───────────────────────────
        # Runs after FusionDecoder so conf1 is available to gate the
        # geometric residual.  depth_mono1 (monocular prior) is used as the
        # warp depth — not pred depth, which is unreliable early in training.
        T_12_iters, T_21_iters, log_conf_pose = self.pose_refiner(
            cam_embed1, cam_embed2,
            spatial1, spatial2,
            depth_mono1, K_pred,
            dec1["confidence"],
            H, W, h14, w14,
        )
        T_12_pred = T_12_iters[-1]
        T_21_pred = T_21_iters[-1]

        return {
            # Depth outputs (same contract as V1)
            "depth1":       dec1["depth"],
            "depth2":       dec2["depth"],
            "confidence1":  dec1["confidence"],
            "confidence2":  dec2["confidence"],
            # No iterative depth refinement in V2
            "depth1_iters": [],
            "depth2_iters": [],
            # Pose iterations for deep supervision
            "T_12_iters":   T_12_iters,
            "scale1":       None,
            "bias1":        None,
            "scale2":       None,
            "bias2":        None,
            # Camera predictions
            "K_pred":        K_pred,
            "T_12_pred":     T_12_pred,
            "T_21_pred":     T_21_pred,
            "log_conf_K":    log_conf_K,
            "log_conf_pose": log_conf_pose,
        }


# ---------------------------------------------------------------------------
# Quick shape-check (no GPU required)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

    model = DepthAlignNetV2(
        dino_model="dinov2_vitl14",
        freeze_dino=True,
        decoder_dim=768,
        num_decoder_blocks=2,    # 2 blocks for quick test
        mast3r_ckpt=None,
    )
    model.eval()

    # 392×518 — recommended resolution (divisible by 14)
    B, H, W = 1, 392, 518
    rgb1  = torch.randn(B, 3, H, W)
    rgb2  = torch.randn(B, 3, H, W)
    dm1   = torch.rand(B, 1, H, W) + 0.1
    dm2   = torch.rand(B, 1, H, W) + 0.1
    T_12  = torch.eye(4).unsqueeze(0).expand(B, -1, -1).contiguous()
    fx    = float(max(H, W)) * 0.9
    K     = torch.tensor(
        [[fx, 0, W/2], [0, fx, H/2], [0, 0, 1]], dtype=torch.float32
    ).unsqueeze(0).expand(B, -1, -1).contiguous()

    print(f"Input: {B}×3×{H}×{W}  → {H//14}×{W//14} = {H//14 * W//14} tokens per view")
    with torch.no_grad():
        out = model(rgb1, rgb2, dm1, dm2, T_12, K)

    for k, v in out.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: {tuple(v.shape)}")
        elif isinstance(v, list):
            print(f"  {k}: {v}")
        else:
            print(f"  {k}: {v}")

    total   = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal params:     {total/1e6:.1f}M")
    print(f"Trainable params: {trainable/1e6:.1f}M")
