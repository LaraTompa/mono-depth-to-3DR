"""
encoder.py — Depth-only ConvNeXt encoder for model_depth_only.

Input  : (B, 1, H, W)  predicted depth map (metric metres).
Output : {'s4':  (B, feature_dim, H/4,  W/4),
           's8':  (B, feature_dim, H/8,  W/8),
           's16': (B, feature_dim, H/16, W/16)}

Backbone: torchvision ConvNeXt-Tiny (same as model_image_depth's SharedEncoder).
  - First conv patched 3 → 1 channel; the single weight is initialised as
    the average of the three RGB channel weights so the ImageNet spatial
    prior is preserved.
  - Classification head discarded; only stages 0-2 (strides 4/8/16) kept.

The shared encoder instance is used for both views in network.py.
"""

import torch
import torch.nn as nn
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights


class DepthEncoder(nn.Module):
    """
    Single-channel ConvNeXt-Tiny feature pyramid encoder.

    Parameters
    ----------
    pretrained      : bool  Load ImageNet weights (recommended: True).
    feature_dim     : int   Uniform output channels at every scale via 1×1 proj.
    freeze_backbone : bool  Freeze all backbone parameters except the patched stem conv.
    """

    def __init__(
        self,
        pretrained:      bool = True,
        feature_dim:     int  = 256,
        freeze_backbone: bool = False,
    ):
        super().__init__()

        weights  = ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        backbone = convnext_tiny(weights=weights)

        # ConvNeXt-Tiny channel widths per stage (before the stride-32 final stage):
        #   stem/stage0 → 96  ch  (stride 4)
        #   stage1      → 192 ch  (stride 8)
        #   stage2      → 384 ch  (stride 16)
        stage_channels = [96, 192, 384]

        # ── Patch first conv: 3 ch → 1 ch ────────────────────────────────
        # ConvNeXt-Tiny stem: backbone.features[0][0] = Conv2d(3, 96, 4, stride=4)
        old_conv = backbone.features[0][0]
        new_conv = nn.Conv2d(
            1, old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=(old_conv.bias is not None),
        )
        if pretrained:
            with torch.no_grad():
                # Average RGB channels → single depth channel; preserves spatial prior.
                new_conv.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))
                if old_conv.bias is not None:
                    new_conv.bias.copy_(old_conv.bias)
        backbone.features[0][0] = new_conv

        # ── Extract stages 0-2; discard stride-32 stage 3 ───────────────
        self.stem   = backbone.features[0]   # Conv + LN  → (B, 96,  H/4,  W/4)
        self.stage1 = backbone.features[1]   # 3 blocks   → (B, 96,  H/4,  W/4)
        self.down1  = backbone.features[2]   # downsampler → (B, 192, H/8,  W/8)
        self.stage2 = backbone.features[3]   # 3 blocks   → (B, 192, H/8,  W/8)
        self.down2  = backbone.features[4]   # downsampler → (B, 384, H/16, W/16)
        self.stage3 = backbone.features[5]   # 9 blocks   → (B, 384, H/16, W/16)

        if freeze_backbone:
            for m in (self.stem, self.stage1, self.down1,
                      self.stage2, self.down2, self.stage3):
                for p in m.parameters():
                    p.requires_grad_(False)
            # Keep the patched stem conv trainable so it adapts to depth statistics.
            new_conv.weight.requires_grad_(True)
            if new_conv.bias is not None:
                new_conv.bias.requires_grad_(True)

        # ── 1×1 projections → uniform feature_dim ───────────────────────
        self.proj_s4  = nn.Conv2d(stage_channels[0], feature_dim, 1)
        self.proj_s8  = nn.Conv2d(stage_channels[1], feature_dim, 1)
        self.proj_s16 = nn.Conv2d(stage_channels[2], feature_dim, 1)

        for proj in (self.proj_s4, self.proj_s8, self.proj_s16):
            nn.init.kaiming_normal_(proj.weight, mode="fan_out")
            nn.init.zeros_(proj.bias)

        self.feature_dim = feature_dim

    def forward(self, x: torch.Tensor) -> dict:
        """
        x : (B, 1, H, W) — predicted depth in metric metres
        """
        f   = self.stem(x)
        s4  = self.stage1(f)          # (B, 96,  H/4,  W/4)
        f   = self.down1(s4)
        s8  = self.stage2(f)          # (B, 192, H/8,  W/8)
        f   = self.down2(s8)
        s16 = self.stage3(f)          # (B, 384, H/16, W/16)

        return {
            "s4":  self.proj_s4(s4),
            "s8":  self.proj_s8(s8),
            "s16": self.proj_s16(s16),
        }
