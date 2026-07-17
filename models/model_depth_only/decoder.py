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

Depth-map output  (predict_depth_map=True)
-------------------------------------------
Instead of an XYZ residual, the decoder predicts a scalar log-space
multiplicative correction on top of the scalar MDE depth prior:

    depth_out = depth_prior_480 * exp(log_scale)

Zero-initialising the head weight + bias makes log_scale = 0 at t=0, so
exp(log_scale) = 1× and the network starts as a passthrough (prior only).
This differs from ``LegacyDepthDecoder``, which applied a raw
softplus(·)-activated scale plus an additive residual.

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
        token_dim:         int  = 256,
        skip_dim:          int  = 256,
        hidden:            int  = 128,
        predict_depth_map: bool = False,
    ):
        super().__init__()

        self.predict_depth_map = predict_depth_map

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
        # passthrough at t=0 (output = prior + 0 = prior).
        if predict_depth_map:
            # 1-channel log-space multiplicative correction: depth = prior * exp(log_scale).
            # Zero-init → exp(0) = 1 → passthrough (scale=1) at t=0.
            self.head_log_scale = nn.Conv2d(hidden, 1, 1)
            zero_heads = (self.head_log_scale,)
        else:
            # 3-channel XYZ residual (metres) — DUSt3R-style point-map output.
            self.head_resid_xyz = nn.Conv2d(hidden, 3, 1)
            zero_heads = (self.head_resid_xyz,)
        self.head_conf = nn.Conv2d(hidden, 1, 1)  # confidence logit
        zero_heads = zero_heads + (self.head_conf,)

        for head in zero_heads:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(
        self,
        tokens:      torch.Tensor,   # (B, N, token_dim)  cross-attended tokens
        skips:       dict,           # {"s4", "s8", "s16"} from DepthEncoder
        point_prior: torch.Tensor,   # (B, 3, H, W) point-map prior  OR  (B, 1, H, W) depth prior
        input_hw:    tuple,          # (H, W) spatial size of the prior
    ) -> dict:
        """
        Returns (point-map mode)
        ------------------------
        {
          "point":      (B, 3, 480, 640),  # XYZ in view-1 camera frame, metric
          "confidence": (B, 1, 480, 640),
        }

        Returns (depth-map mode, predict_depth_map=True)
        -------------------------------------------------
        {
          "depth":      (B, 1, 480, 640),  # scalar depth in metres (multiplicative,
                                            # log-space correction of the prior)
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

        # ── Upsample prior to output resolution ────────────────────────────
        prior_480 = F.interpolate(
            point_prior, size=(OUT_H, OUT_W), mode="bilinear", align_corners=False
        )                                                # (B, C, 480, 640)  C=3 or 1

        conf = torch.sigmoid(self.head_conf(x))          # (B, 1, 480, 640)  ∈ (0,1)

        if self.predict_depth_map:
            # ── Depth-map output (multiplicative correction in log-space) ────
            # depth = prior * exp(log_scale).  Unlike LegacyDepthDecoder's raw
            # softplus(head_scale(x)) scale, the correction is predicted directly
            # in log-space so it is symmetric around 1× (log_scale=0 → passthrough)
            # and exp(·) guarantees strict positivity without a softplus/epsilon.
            log_scale = self.head_log_scale(x).clamp(min=-3.0, max=3.0)  # (B, 1, 480, 640)         # (B, 1, 480, 640)
            depth = prior_480 * torch.exp(log_scale)      # (B, 1, 480, 640)
            return {"depth": depth, "confidence": conf, "log_scale": log_scale}
        else:
            # ── Point-map output (3-channel XYZ residual, DUSt3R-style) ─────
            residual_xyz = self.head_resid_xyz(x)        # (B, 3, 480, 640)
            point = prior_480 + residual_xyz             # (B, 3, 480, 640)  metric metres
            return {"point": point, "confidence": conf}


class LegacyDepthDecoder(nn.Module):
    """
    Backward-compatible decoder matching the architecture of checkpoints trained
    before the XYZ point-map output was introduced.

    Old output heads
    ----------------
    Instead of a 3-channel ``head_resid_xyz`` the old decoder had two 1-channel
    heads that predicted a multiplicative scale and an additive residual on top
    of the MDE depth prior:

        scale = softplus(head_scale(x)) + 1e-4   # always positive, ≈1 at init
        depth = scale * depth_prior + head_resid(x)

    This class uses the **same call signature** as the current ``DepthDecoder``
    (``point_prior`` / ``input_hw``) so it can be dropped into ``DepthOnlyNet``
    transparently.  When ``predict_depth_map=True`` in ``DepthOnlyNet``,
    ``point_prior`` is a 1-channel scalar depth map — exactly what this decoder
    expects as ``depth_input``.

    Returns ``{"depth": …, "confidence": …}`` — the same keys the network reads
    back in depth-map mode.
    """

    def __init__(
        self,
        token_dim: int = 256,
        skip_dim:  int = 256,
        hidden:    int = 128,
    ):
        super().__init__()

        self.token_proj = nn.Conv2d(token_dim, hidden, 1)
        self.fuse16     = ConvBnGelu(hidden + skip_dim, hidden)
        self.fuse8      = ConvBnGelu(hidden + skip_dim, hidden)
        self.fuse4      = ConvBnGelu(hidden + skip_dim, hidden)
        self.final_conv = ConvBnGelu(hidden, hidden)

        # Old-style heads: scale + residual on 1-channel depth, plus confidence.
        self.head_scale = nn.Conv2d(hidden, 1, 1)
        self.head_resid = nn.Conv2d(hidden, 1, 1)
        self.head_conf  = nn.Conv2d(hidden, 1, 1)

        for head in (self.head_scale, self.head_resid, self.head_conf):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(
        self,
        tokens:      torch.Tensor,   # (B, N, token_dim)
        skips:       dict,           # {"s4", "s8", "s16"}
        point_prior: torch.Tensor,   # (B, 1, H, W) depth prior (called point_prior for API compat)
        input_hw:    tuple,          # (H, W) — unused here, kept for API compat
    ) -> dict:
        B = tokens.shape[0]
        h16, w16 = skips["s16"].shape[-2:]

        x = tokens.reshape(B, h16, w16, -1).permute(0, 3, 1, 2).contiguous()
        x = self.token_proj(x)

        s4  = skips["s4"]
        s8  = skips["s8"]
        s16 = skips["s16"]

        x = F.interpolate(x, size=s16.shape[-2:], mode="bilinear", align_corners=False)
        x = self.fuse16(torch.cat([x, s16], dim=1))

        x = F.interpolate(x, size=s8.shape[-2:], mode="bilinear", align_corners=False)
        x = self.fuse8(torch.cat([x, s8], dim=1))

        x = F.interpolate(x, size=s4.shape[-2:], mode="bilinear", align_corners=False)
        x = self.fuse4(torch.cat([x, s4], dim=1))

        x = F.interpolate(x, size=(OUT_H, OUT_W), mode="bilinear", align_corners=False)
        x = self.final_conv(x)

        depth_480 = F.interpolate(
            point_prior, size=(OUT_H, OUT_W), mode="bilinear", align_corners=False
        )

        log_scale = self.head_scale(x)
        residual  = self.head_resid(x)
        scale     = F.softplus(log_scale) + 1e-4
        depth     = scale * depth_480 + residual
        conf      = torch.sigmoid(self.head_conf(x))

        return {"depth": depth, "confidence": conf}
