"""
depth_stream.py — Trainable geometry-aware depth encoder stream.

Takes 4-channel input (RGB + mono depth) and produces a 3-scale feature
pyramid {s4, s8, s16} using ConvNeXt-Small as backbone.

This stream is the geometry branch in the late-fusion two-stream design:
  - DINOv2 handles semantics and cross-view correspondence.
  - This stream encodes depth-structure, indoor geometry cues, and the
    mono-depth prior.  It never sees the other view — fusion happens only
    in the decoder.

ConvNeXt-Small stage channel widths are identical to ConvNeXt-Tiny
([96, 192, 384, 768]) but stage-3 has 27 blocks instead of 9, giving
more capacity for geometry adaptation.

Weight initialisation: ImageNet pretrained weights for the RGB channels;
depth channel (4th) zero-initialised to preserve the pretrained prior.
"""

import torch
import torch.nn as nn
from torchvision.models import (
    convnext_tiny,  ConvNeXt_Tiny_Weights,
    convnext_small, ConvNeXt_Small_Weights,
)

_BACKBONES = {
    "convnext_tiny":  (convnext_tiny,  ConvNeXt_Tiny_Weights),
    "convnext_small": (convnext_small, ConvNeXt_Small_Weights),
}


class DepthStream(nn.Module):
    """
    Trainable 4-channel ConvNeXt depth encoder producing {s4, s8, s16}.

    Parameters
    ----------
    backbone     : str   "convnext_tiny" (~28M) or "convnext_small" (~50M).
                         Both share identical channel widths [96,192,384];
                         Small has more blocks in stage 3 (27 vs 9).
    pretrained   : bool  load ImageNet weights for the 3 RGB channels
    out_channels : int   projected output channels at each scale (default 128)
    """

    def __init__(
        self,
        backbone:     str  = "convnext_tiny",
        pretrained:   bool = True,
        out_channels: int  = 128,
    ):
        super().__init__()

        if backbone not in _BACKBONES:
            raise ValueError(f"depth_backbone must be one of {list(_BACKBONES)}; got '{backbone}'")

        # Both ConvNeXt-Tiny and Small share the same stage channel widths:
        #   stage 0 →  96 ch (stride  4)
        #   stage 1 → 192 ch (stride  8)
        #   stage 2 → 384 ch (stride 16)
        #   stage 3 → 768 ch (stride 32) — not used
        stage_channels = [96, 192, 384]

        fn, weights_cls = _BACKBONES[backbone]
        weights  = weights_cls.DEFAULT if pretrained else None
        backbone_model = fn(weights=weights)

        # ── Patch stem: 3-channel → 4-channel ───────────────────────────
        # ConvNeXt uses features[0][0] as the 4×4 stride-4 patchify conv.
        old_conv = backbone_model.features[0][0]
        new_conv = nn.Conv2d(
            4, old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=(old_conv.bias is not None),
        )
        if pretrained:
            with torch.no_grad():
                new_conv.weight[:, :3] = old_conv.weight   # keep RGB weights
                nn.init.zeros_(new_conv.weight[:, 3:])     # depth channel: zero init
                if old_conv.bias is not None:
                    new_conv.bias.copy_(old_conv.bias)
        backbone_model.features[0][0] = new_conv

        # ── Extract stages ───────────────────────────────────────────────
        self.stem   = backbone_model.features[0]   # (B,  96, H/4,  W/4)
        self.stage1 = backbone_model.features[1]   # (B,  96, H/4,  W/4)
        self.down1  = backbone_model.features[2]   # (B, 192, H/8,  W/8)
        self.stage2 = backbone_model.features[3]   # (B, 192, H/8,  W/8)
        self.down2  = backbone_model.features[4]   # (B, 384, H/16, W/16)
        self.stage3 = backbone_model.features[5]   # (B, 384, H/16, W/16)
        # features[6,7] = stage 4 (stride 32) — discarded

        # ── 1×1 projections to uniform out_channels ─────────────────────
        self.proj4  = nn.Conv2d(stage_channels[0], out_channels, 1)
        self.proj8  = nn.Conv2d(stage_channels[1], out_channels, 1)
        self.proj16 = nn.Conv2d(stage_channels[2], out_channels, 1)

        self.out_channels = out_channels

    def forward(self, rgb: torch.Tensor, depth: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        rgb   : (B, 3, H, W)  RGB image in [0, 1]
        depth : (B, 1, H, W)  monocular depth prior (metres)

        Returns
        -------
        {"s4": (B, C, H/4, W/4), "s8": (B, C, H/8, W/8), "s16": (B, C, H/16, W/16)}
        """
        # Normalize depth to [0, 1] per sample so the depth channel enters the
        # ConvNeXt backbone at the same scale as RGB (also ~[0, 1]).  Metric
        # values are only needed for the decoder's scale+residual formula, which
        # receives `depth_mono` as a separate argument and is NOT affected here.
        d_max = depth.flatten(1).max(dim=1).values.view(-1, 1, 1, 1).clamp(min=1e-3)
        depth_enc = depth / d_max
        x = torch.cat([rgb, depth_enc], dim=1)   # (B, 4, H, W)

        f = self.stem(x)
        f = self.stage1(f)
        s4 = f                               # (B, 96,  H/4,  W/4)

        f = self.down1(f)
        f = self.stage2(f)
        s8 = f                               # (B, 192, H/8,  W/8)

        f = self.down2(f)
        f = self.stage3(f)
        s16 = f                              # (B, 384, H/16, W/16)

        return {
            "s4":  self.proj4(s4),
            "s8":  self.proj8(s8),
            "s16": self.proj16(s16),
        }
