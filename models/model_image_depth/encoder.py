"""
encoder.py — Shared lightweight feature-pyramid encoder.

Takes RGB + monocular depth (4 channels) and outputs three feature scales:
  scale_4  (stride 4)   → 1/4 resolution
  scale_8  (stride 8)   → 1/8 resolution
  scale_16 (stride 16)  → 1/16 resolution

Backbone choice: torchvision's ConvNeXt-Tiny, first-conv replaced to accept
4 input channels and the classification head removed.

Normalisation: LayerNorm (built into ConvNeXt) — no BatchNorm.
The same encoder instance is shared (weight-tied) across both views.
"""

import torch
import torch.nn as nn
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights


class SharedEncoder(nn.Module):
    """
    Shared 4-channel encoder producing a 3-scale feature pyramid.

    Parameters
    ----------
    pretrained : bool
        Whether to initialise from ImageNet weights.
        The first conv is patched (3→4 ch); RGB weights are kept, depth
        channel is zero-initialised so the ImageNet prior is preserved.
    out_channels : int
        Number of output channels at every scale (projected via 1×1 conv).
        Defaults to 128 — balances capacity vs. speed.
    """

    def __init__(self, pretrained: bool = True, out_channels: int = 128):
        super().__init__()

        # ── Load backbone ────────────────────────────────────────────────
        weights = ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        backbone = convnext_tiny(weights=weights)

        # ConvNeXt-Tiny feature stage output channels:
        #   stage 0 → 96   ch  (stride 4)
        #   stage 1 → 192  ch  (stride 8)
        #   stage 2 → 384  ch  (stride 16)
        #   stage 3 → 768  ch  (stride 32) — not used
        stage_channels = [96, 192, 384]

        # ── Patch first conv: 3 → 4 input channels ──────────────────────
        # ConvNeXt uses features[0][0] as the patchify stem (4×4, stride 4)
        old_conv = backbone.features[0][0]
        new_conv = nn.Conv2d(
            4, old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=(old_conv.bias is not None),
        )
        if pretrained:
            with torch.no_grad():
                new_conv.weight[:, :3] = old_conv.weight
                nn.init.zeros_(new_conv.weight[:, 3:])
                if old_conv.bias is not None:
                    new_conv.bias.copy_(old_conv.bias)
        backbone.features[0][0] = new_conv

        # ── Extract the four ConvNeXt stages ────────────────────────────
        # features[0] = stem (Conv + LN), stride 4 → 96 ch
        # features[1] = stage 1 blocks   (no extra downsampling)
        # features[2] = downsampling     stride 2  → 8 total
        # features[3] = stage 2 blocks
        # features[4] = downsampling     stride 2  → 16 total
        # features[5] = stage 3 blocks
        # features[6,7] = stage 4 (stride 32) — discarded
        self.stem    = backbone.features[0]    # → (B, 96,  H/4,  W/4)
        self.stage1  = backbone.features[1]    # → (B, 96,  H/4,  W/4)
        self.down1   = backbone.features[2]    # → (B, 192, H/8,  W/8)
        self.stage2  = backbone.features[3]    # → (B, 192, H/8,  W/8)
        self.down2   = backbone.features[4]    # → (B, 384, H/16, W/16)
        self.stage3  = backbone.features[5]    # → (B, 384, H/16, W/16)

        # ── 1×1 projection to uniform out_channels ──────────────────────
        self.proj4  = nn.Conv2d(stage_channels[0], out_channels, 1)
        self.proj8  = nn.Conv2d(stage_channels[1], out_channels, 1)
        self.proj16 = nn.Conv2d(stage_channels[2], out_channels, 1)

        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        x : (B, 4, H, W)   RGB (3 ch) + monocular depth (1 ch)

        Returns
        -------
        dict with keys "s4", "s8", "s16"  — feature maps at each scale.
        """
        f = self.stem(x)          # (B, 96,  H/4,  W/4)
        f = self.stage1(f)        # (B, 96,  H/4,  W/4)
        s4 = f

        f = self.down1(f)         # (B, 192, H/8,  W/8)
        f = self.stage2(f)        # (B, 192, H/8,  W/8)
        s8 = f

        f = self.down2(f)         # (B, 384, H/16, W/16)
        f = self.stage3(f)        # (B, 384, H/16, W/16)
        s16 = f

        return {
            "s4":  self.proj4(s4),
            "s8":  self.proj8(s8),
            "s16": self.proj16(s16),
        }
