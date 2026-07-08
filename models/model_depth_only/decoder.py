"""
decoder.py — Multi-scale decoder: depth tokens + encoder skips → point map at 640×480.

Output resolution
-----------------
ScanNet GT depth maps are stored at 640×480 (the original sensor resolution).
This decoder *always* produces outputs at that exact resolution regardless of
the input MDE depth resolution, so predictions can be directly compared to GT.

Fusion strategy
---------------
After the cross-attention blocks the attended token sequence is reshaped back
to a spatial feature map at s16 resolution (H/16 × W/16) based on the
*input depth* spatial dimensions.  Skip connections from the shared ConvNeXt
encoder are fused at s8 and s4 via additive fusion convolutions, then the
feature map is bilinearly upsampled to the fixed 640×480 output.

Point-map output  (DUSt3R-style)
---------------------------------
The decoder predicts a 3-channel XYZ residual on top of a geometric point-map
prior (unprojected from the MDE depth and, for view 2, warped into view-1's
camera frame).  Zero-initialising the head weight + bias ensures the network
starts as a passthrough (prior only) at t=0:

    point_out = point_prior_480 + residual_xyz

Everything stays in metric (metres).  No per-axis or global scalar
normalisation is applied to the point maps.

Confidence
----------
A separate sigmoid-activated head produces a per-pixel confidence map at
the same 640×480 resolution.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Fixed output resolution: ScanNet depth size.
OUT_H, OUT_W = 480, 640


class ConvBnGelu(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, padding=kernel // 2, bias=False),
            nn.GroupNorm(min(8, out_ch), out_ch),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DepthDecoder(nn.Module):
    """
    Multi-scale decoder producing a 3-channel XYZ point map + confidence at 640×480.

    Parameters
    ----------
    token_dim   : int  Dimension of the incoming attended tokens (must match
                       the cross-attention decoder's token_dim and, if an
                       enc_to_dec projection is used in network.py, the
                       projected dimension).
    skip_dim    : int  Channel width of encoder skip features (feature_dim
                       from DepthEncoder).
    hidden      : int  Internal decoder channel width.
    """

    def __init__(
        self,
        token_dim: int = 256,
        skip_dim:  int = 256,
        hidden:    int = 128,
    ):
        super().__init__()

        # Project token_dim → hidden at s16
        self.token_proj = nn.Conv2d(token_dim, hidden, 1)

        # Fuse s16 tokens with s16 skip (after token proj & upsample to s8)
        self.fuse16 = ConvBnGelu(hidden + skip_dim, hidden)
        # Fuse s8 features
        self.fuse8  = ConvBnGelu(hidden + skip_dim, hidden)
        # Fuse s4 features
        self.fuse4  = ConvBnGelu(hidden + skip_dim, hidden)

        # Final conv before output heads
        self.final_conv = ConvBnGelu(hidden, hidden)

        # Output heads — zero-init residual head so the network starts as a
        # passthrough at t=0 (output = point_prior + 0 = point_prior).
        self.head_resid_xyz = nn.Conv2d(hidden, 3, 1)  # 3-ch XYZ residual (metres)
        self.head_conf      = nn.Conv2d(hidden, 1, 1)  # confidence logit

        for head in (self.head_resid_xyz, self.head_conf):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(
        self,
        tokens:      torch.Tensor,   # (B, N, token_dim)  cross-attended tokens
        skips:       dict,           # {"s4", "s8", "s16"} from DepthEncoder
        point_prior: torch.Tensor,   # (B, 3, H, W) geometric point-map prior (metric XYZ)
        input_hw:    tuple,          # (H, W) spatial size of the point prior
    ) -> dict:
        """
        Returns
        -------
        {
          "point":      (B, 3, 480, 640),  # XYZ in view-1 camera frame, metric
          "confidence": (B, 1, 480, 640),
        }
        """
        B = tokens.shape[0]

        # Derive h16/w16 from the actual skip map, NOT from H//16.
        # ConvNeXt uses floor-division in each 2× downsample, so for
        # non-multiples of 16 the actual output size may differ from H//16
        # (e.g. 968 → stride-4 → 242 → stride-2 → 121 → stride-2 → 60,
        #  but 968//16 = 60 ✓ — however 969//16=60 while actual = 60 too,
        #  so always trust the skip map rather than recomputing).
        h16, w16 = skips["s16"].shape[-2:]

        # ── Reshape tokens → spatial map ──────────────────────────────────
        # tokens: (B, h16*w16, token_dim) → (B, token_dim, h16, w16)
        x = tokens.reshape(B, h16, w16, -1).permute(0, 3, 1, 2).contiguous()
        x = self.token_proj(x)                          # (B, hidden, h16, w16)

        s4  = skips["s4"]    # (B, skip_dim, H/4,  W/4)
        s8  = skips["s8"]    # (B, skip_dim, H/8,  W/8)
        s16 = skips["s16"]   # (B, skip_dim, H/16, W/16)

        # ── Fuse at s16 ───────────────────────────────────────────────────
        x = F.interpolate(x, size=s16.shape[-2:], mode="bilinear", align_corners=False)
        x = self.fuse16(torch.cat([x, s16], dim=1))    # (B, hidden, H/16, W/16)

        # ── Fuse at s8 ────────────────────────────────────────────────────
        x = F.interpolate(x, size=s8.shape[-2:], mode="bilinear", align_corners=False)
        x = self.fuse8(torch.cat([x, s8], dim=1))      # (B, hidden, H/8,  W/8)

        # ── Fuse at s4 ────────────────────────────────────────────────────
        x = F.interpolate(x, size=s4.shape[-2:], mode="bilinear", align_corners=False)
        x = self.fuse4(torch.cat([x, s4], dim=1))      # (B, hidden, H/4,  W/4)

        # ── Final conv ────────────────────────────────────────────────────
        x = F.interpolate(x, size=(OUT_H, OUT_W), mode="bilinear", align_corners=False)
        x = self.final_conv(x)                          # (B, hidden, 480, 640)

        # ── Upsample point prior to output resolution ───────────────────────
        prior_480 = F.interpolate(
            point_prior, size=(OUT_H, OUT_W), mode="bilinear", align_corners=False
        )                                                # (B, 3, 480, 640)

        # ── Output heads ─────────────────────────────────────────────────
        residual_xyz = self.head_resid_xyz(x)            # (B, 3, 480, 640)
        conf_logit   = self.head_conf(x)                 # (B, 1, 480, 640)

        # point = prior + XYZ residual  (metric metres, no per-axis rescaling)
        point = prior_480 + residual_xyz                 # (B, 3, 480, 640)
        conf  = torch.sigmoid(conf_logit)                # (B, 1, 480, 640)  ∈ (0,1)

        return {
            "point":      point,
            "confidence": conf,
        }
