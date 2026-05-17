"""
decoder.py — Lightweight bilinear decoder with skip connections.

Takes multi-scale encoder features + the low-resolution aligned depth from
the refinement stage, and upsamples to full (1/4) resolution.

Architecture:
  s16 features  →  upsample ×2  →  fuse with s8  →  Conv
                →  upsample ×2  →  fuse with s4  →  Conv
                →  upsample ×2  →  output at 1/2 resolution
                →  final 1×1 conv → aligned depth residual + confidence
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBnGelu(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, padding=kernel // 2, bias=False),
            nn.GroupNorm(min(8, out_ch), out_ch),
            nn.GELU(),
        )

    def forward(self, x):
        return self.block(x)


class DepthDecoder(nn.Module):
    """
    Lightweight decoder that fuses multi-scale features and an
    initial aligned depth map into a refined full-resolution output.

    Parameters
    ----------
    feat_dim : int
        Number of channels in each encoder scale (s4, s8, s16).
        Must match SharedEncoder.out_channels.
    hidden   : int
        Internal channel width of the decoder convolutions.
    """

    def __init__(self, feat_dim: int = 128, hidden: int = 64):
        super().__init__()

        # Fuse s16 depth with s16 features → start of decode path
        # Input = feat_dim (s16) + 1 (aligned depth from refinement)
        self.fuse16 = ConvBnGelu(feat_dim + 1, hidden)

        # After ×2 upsample, fuse with s8 skip
        self.fuse8  = ConvBnGelu(hidden + feat_dim, hidden)

        # After ×2 upsample, fuse with s4 skip
        self.fuse4  = ConvBnGelu(hidden + feat_dim, hidden)

        # Final upsample ×2 to half-resolution, then lightweight conv
        self.final_conv = ConvBnGelu(hidden, hidden // 2)

        # Output head: depth residual (1 ch) + confidence (1 ch)
        self.out_depth = nn.Conv2d(hidden // 2, 1, 1)
        self.out_conf  = nn.Conv2d(hidden // 2, 1, 1)
        self.out_scale = nn.Conv2d(hidden // 2, 1, 1)

        nn.init.normal_(self.out_depth.weight, std=1e-4); nn.init.zeros_(self.out_depth.bias)
        nn.init.zeros_(self.out_conf.weight);  nn.init.zeros_(self.out_conf.bias)
        nn.init.zeros_(self.out_scale.weight); nn.init.zeros_(self.out_scale.bias)

    def forward(
        self,
        feats: dict,                 # {"s4": ..., "s8": ..., "s16": ...}
        depth_init: torch.Tensor,   # (B, 1, H/16, W/16) from refinement
        depth_mono: torch.Tensor,   # (B, 1, H, W) original mono prior (full res)
    ) -> dict:
        """
        Returns
        -------
        dict with:
          "depth"      : (B, 1, H/2, W/2)  final aligned depth
          "confidence" : (B, 1, H/2, W/2)  ∈ (0, 1]
          "scale"      : (B, 1, H/2, W/2)  scale factor
        """
        s4  = feats["s4"]    # (B, C, H/4,  W/4)
        s8  = feats["s8"]    # (B, C, H/8,  W/8)
        s16 = feats["s16"]   # (B, C, H/16, W/16)

        _, _, H16, W16 = s16.shape

        # Resize depth_init to s16 spatial size (refinement may run at s16)
        d = F.interpolate(depth_init, size=(H16, W16), mode="bilinear", align_corners=True)

        # --- s16 stage ---
        x = self.fuse16(torch.cat([s16, d], dim=1))                    # (B, hidden, H/16, W/16)
        x = F.interpolate(x, size=s8.shape[-2:], mode="bilinear", align_corners=True)  # H/8

        # --- s8 stage ---
        x = self.fuse8(torch.cat([x, s8], dim=1))                      # (B, hidden, H/8, W/8)
        x = F.interpolate(x, size=s4.shape[-2:], mode="bilinear", align_corners=True)  # H/4

        # --- s4 stage ---
        x = self.fuse4(torch.cat([x, s4], dim=1))                      # (B, hidden, H/4, W/4)
        x = F.interpolate(x, size=(s4.shape[-2] * 2, s4.shape[-1] * 2), mode="bilinear", align_corners=True)  # H/2

        # --- final ---
        x = self.final_conv(x)                                          # (B, hidden//2, H/2, W/2)

        # Upsample mono prior to match output size for residual addition
        H2, W2 = x.shape[-2:]
        depth_mono_up = F.interpolate(depth_mono, size=(H2, W2), mode="bilinear", align_corners=True)

        loq_scale = self.out_scale(x)                                           # (B, 1, H/2, W/2)
        scale = F.softplus(loq_scale) + 1e-3  # ensure positive scale with min value
        depth_residual = self.out_depth(x)                              # (B, 1, H/2, W/2)
        depth_out = (scale*depth_mono_up + depth_residual).clamp(min=1e-3)

        confidence = torch.sigmoid(self.out_conf(x))                   # (B, 1, H/2, W/2) ∈ (0,1)

        return {
            "depth":      depth_out,
            "confidence": confidence,
        }
