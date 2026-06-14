"""
decoder.py — Multi-scale decoder: depth tokens + encoder skips → depth at 640×480.

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

Depth output
------------
The decoder predicts a scale factor and an additive residual on top of the
input mono-depth prior (upsampled to 640×480).  This residual formulation
ensures the network can start from a reasonable depth estimate at
initialisation (when all heads are near-zero):

    depth_out = softplus(scale) × depth_mono_480 + residual

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
    Multi-scale decoder producing depth + confidence at 640×480.

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

        # Output heads — near-zero init so network starts as passthrough.
        self.head_scale = nn.Conv2d(hidden, 1, 1)  # log-scale
        self.head_resid = nn.Conv2d(hidden, 1, 1)  # additive residual
        self.head_conf  = nn.Conv2d(hidden, 1, 1)  # confidence logit

        for head in (self.head_scale, self.head_resid, self.head_conf):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(
        self,
        tokens:      torch.Tensor,   # (B, N, token_dim)  cross-attended tokens
        skips:       dict,           # {"s4", "s8", "s16"} from DepthEncoder
        depth_input: torch.Tensor,   # (B, 1, H, W) original input depth (before median norm)
        input_hw:    tuple,          # (H, W) spatial size of the input depth
    ) -> dict:
        """
        Returns
        -------
        {
          "depth":      (B, 1, 480, 640),
          "confidence": (B, 1, 480, 640),
          "log_scale":  (B, 1, 480, 640),   # for auxiliary loss / inspection
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

        # ── Upsample input mono prior to output resolution ─────────────────
        depth_480 = F.interpolate(
            depth_input, size=(OUT_H, OUT_W), mode="bilinear", align_corners=False
        )                                                # (B, 1, 480, 640)

        # ── Output heads ─────────────────────────────────────────────────
        log_scale = self.head_scale(x)                   # (B, 1, 480, 640)
        residual  = self.head_resid(x)                   # (B, 1, 480, 640)
        conf_logit = self.head_conf(x)                   # (B, 1, 480, 640)

        scale = F.softplus(log_scale) + 1e-4             # positive, near 1 at init
        depth = scale * depth_480 + residual             # (B, 1, 480, 640)
        conf  = torch.sigmoid(conf_logit)                # (B, 1, 480, 640)  ∈ (0,1)

        return {
            "depth":      depth,
            "confidence": conf,
            "log_scale":  log_scale,
        }
