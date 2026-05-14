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
from .attention  import LocalGeoCrossAttention
from .refinement import IterativeRefinement
from .decoder    import DepthDecoder


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
        feat_dim    : int  = 128,
        hidden_dim  : int  = 128,
        num_iters   : int  = 4,
        num_heads   : int  = 4,
        window_size : int  = 7,
        pretrained  : bool = True,
    ):
        super().__init__()

        # Shared encoder (weight-tied across views)
        self.encoder = SharedEncoder(pretrained=pretrained, out_channels=feat_dim)

        # Local geometry-aware cross-attention at each scale
        self.attn16 = LocalGeoCrossAttention(feat_dim, num_heads, window_size)
        self.attn8  = LocalGeoCrossAttention(feat_dim, num_heads, window_size)

        # Iterative refinement operates at 1/16 resolution for speed
        self.refine = IterativeRefinement(
            feat_dim=feat_dim, hidden_dim=hidden_dim, num_iters=num_iters
        )

        # Decoder
        self.decoder = DepthDecoder(feat_dim=feat_dim, hidden=feat_dim // 2)

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
    # Per-view processing (encoder + attention, symmetric)
    # ------------------------------------------------------------------
    def _encode_and_attend(
        self,
        rgb_q, depth_q,          # query view
        rgb_ctx, depth_ctx,      # context view
        T_q2ctx,                 # cam_q → cam_ctx
        K,                       # full-resolution intrinsics
    ):
        """
        Encode both views with the shared encoder, then apply local
        cross-attention from query → context at s16 and s8.

        Returns feats_q (attended), feats_ctx (raw), depth_q_s16, K_s16
        """
        B, _, H, W = rgb_q.shape

        # Concatenate RGB + mono depth along channel dim
        x_q   = torch.cat([rgb_q,   depth_q],   dim=1)   # (B, 4, H, W)
        x_ctx = torch.cat([rgb_ctx, depth_ctx], dim=1)

        feats_q   = self.encoder(x_q)     # {"s4", "s8", "s16"}
        feats_ctx = self.encoder(x_ctx)

        # Scale K to s16 resolution for geometry ops in attention/refinement
        _, _, H16, W16 = feats_q["s16"].shape
        K_s16 = self._scale_K(K, H, W, H16, W16)

        # Scale K to s8 resolution
        _, _, H8, W8 = feats_q["s8"].shape
        K_s8 = self._scale_K(K, H, W, H8, W8)

        # Resize mono depth to s16 for geometry
        depth_q_s16   = F.interpolate(depth_q,   size=(H16, W16), mode="nearest")
        depth_ctx_s16 = F.interpolate(depth_ctx, size=(H16, W16), mode="nearest")

        # Resize mono depth to s8
        depth_q_s8 = F.interpolate(depth_q, size=(H8, W8), mode="nearest")

        # Cross-attention at s16: query uses view-q, context is view-ctx
        f16 = self.attn16(
            feats_q["s16"], feats_ctx["s16"],
            depth_q_s16, T_q2ctx, K_s16,
        )

        # Cross-attention at s8
        f8 = self.attn8(
            feats_q["s8"], feats_ctx["s8"],
            depth_q_s8, T_q2ctx, K_s8,
        )

        # Return attended features for q, raw for ctx, and geometry helpers
        attended_feats_q = {
            "s4":  feats_q["s4"],
            "s8":  f8,
            "s16": f16,
        }
        return attended_feats_q, feats_ctx, depth_q_s16, depth_ctx_s16, K_s16

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        rgb1:        torch.Tensor,
        rgb2:        torch.Tensor,
        depth_mono1: torch.Tensor,
        depth_mono2: torch.Tensor,
        T_12:        torch.Tensor,
        K:           torch.Tensor,
    ) -> dict:
        B, _, H, W = rgb1.shape
        T_21 = torch.linalg.inv(T_12)

        # ── View 1: query, View 2: context ──────────────────────────────
        feats1, feats2, d1_s16, d2_s16, K_s16 = self._encode_and_attend(
            rgb1, depth_mono1, rgb2, depth_mono2, T_12, K
        )

        # ── View 2: query, View 1: context ──────────────────────────────
        feats2_attended, _, d2_s16_v, d1_s16_v, _ = self._encode_and_attend(
            rgb2, depth_mono2, rgb1, depth_mono1, T_21, K
        )

        # ── Iterative refinement (at s16 resolution) ────────────────────
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

        # ── Decoder (upsample to H/2, W/2) ──────────────────────────────
        dec1 = self.decoder(feats1,          ref1["depth"], depth_mono1)
        dec2 = self.decoder(feats2_attended, ref2["depth"], depth_mono2)

        return {
            "depth1":       dec1["depth"],        # (B, 1, H/2, W/2)
            "depth2":       dec2["depth"],
            "confidence1":  dec1["confidence"],
            "confidence2":  dec2["confidence"],
            "depth1_iters": ref1["depth_iters"],  # list[(B, 1, H/16, W/16)]
            "depth2_iters": ref2["depth_iters"],
            "scale1":       ref1["scale"],
            "bias1":        ref1["bias"],
            "scale2":       ref2["scale"],
            "bias2":        ref2["bias"],
        }