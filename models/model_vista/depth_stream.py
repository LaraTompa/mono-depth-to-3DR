"""
depth_stream.py — Trainable geometry-aware depth encoder stream.

Takes 4-channel input (RGB + mono depth) and produces a 3-scale feature
pyramid {s4, s8, s16}.

Two backbone families are available — choose via arch.yaml:
  "convnext_tiny"  / "convnext_small"  — full ConvNeXt (~28M / ~50M params)
  "conv_lite"                          — custom lightweight encoder (~165K params)

The lightweight option makes sense when monocular depth priors (ZoeDepth,
MiDaS …) are already close to metric quality; the network only needs to
learn local refinements rather than extract depth from scratch.

Weight initialisation:
  ConvNeXt: ImageNet pretrained for RGB; depth channel zero-initialised.
  conv_lite: all randomly initialised (no pretrained weights needed).
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
        x = torch.cat([rgb, depth], dim=1)   # (B, 4, H, W)

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


# ---------------------------------------------------------------------------
# Lightweight alternative: conv_lite
# ---------------------------------------------------------------------------

class _DWSepBlock(nn.Module):
    """Depthwise-separable residual block with BN+GELU.

    The residual shortcut keeps the network close to identity early in
    training — ideal when the input depth prior is already accurate and
    we only want to learn local refinements.

    Params per block:
      ch=32  →  ~1.3K   ch=64  →  ~5.1K   ch=128 →  ~18.7K
    """

    def __init__(self, ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, groups=ch, bias=False),  # depthwise
            nn.Conv2d(ch, ch, 1, bias=False),                          # pointwise
            nn.BatchNorm2d(ch),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class LiteDepthStream(nn.Module):
    """
    Ultra-lightweight depth encoder for well-initialised monocular priors.

    Architecture
    ------------
    Stem      : Conv(4 → C, 3×3 s2) → BN → GELU → Conv(C → C, 3×3 s2) → BN → GELU
                Two stride-2 convs give stride-4 with better gradient flow and
                receptive field than a single stride-4 patch embedding.

    Stage 1   : num_blocks × _DWSepBlock(C)         → s4  features (B, C,   H/4,  W/4)
    Down1     : Conv(C → 2C, 3×3 s2) → BN → GELU   → stride 8
    Stage 2   : num_blocks × _DWSepBlock(2C)         → s8  features (B, 2C,  H/8,  W/8)
    Down2     : Conv(2C → 4C, 3×3 s2) → BN → GELU  → stride 16
    Stage 3   : num_blocks × _DWSepBlock(4C)         → s16 features (B, 4C, H/16, W/16)

    Projections: 1×1 Conv at each scale → out_channels  (matches FusionDecoder input)

    Parameter count (out_channels=64):
      base_ch=32, num_blocks=2  →  ~165 K   (≈170× less than ConvNeXt-Tiny)
      base_ch=64, num_blocks=2  →  ~660 K
      base_ch=64, num_blocks=4  →  ~730 K

    Parameters
    ----------
    base_ch      : int   channels at s4; s8=2×base_ch, s16=4×base_ch
    num_blocks   : int   depthwise-sep residual blocks per stage
    out_channels : int   output channels per scale (projected, matches FusionDecoder)
    """

    def __init__(
        self,
        base_ch:      int = 32,
        num_blocks:   int = 2,
        out_channels: int = 128,
    ):
        super().__init__()
        C = base_ch

        # Stride-4 stem via two stride-2 convolutions
        self.stem = nn.Sequential(
            nn.Conv2d(4, C, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(C),
            nn.GELU(),
            nn.Conv2d(C, C, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(C),
            nn.GELU(),
        )

        self.stage1 = nn.Sequential(*[_DWSepBlock(C)     for _ in range(num_blocks)])

        self.down1  = nn.Sequential(
            nn.Conv2d(C,     2 * C, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(2 * C),
            nn.GELU(),
        )
        self.stage2 = nn.Sequential(*[_DWSepBlock(2 * C) for _ in range(num_blocks)])

        self.down2  = nn.Sequential(
            nn.Conv2d(2 * C, 4 * C, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(4 * C),
            nn.GELU(),
        )
        self.stage3 = nn.Sequential(*[_DWSepBlock(4 * C) for _ in range(num_blocks)])

        # Project each scale to the unified out_channels for FusionDecoder
        self.proj4  = nn.Conv2d(C,     out_channels, 1)
        self.proj8  = nn.Conv2d(2 * C, out_channels, 1)
        self.proj16 = nn.Conv2d(4 * C, out_channels, 1)

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
        x = torch.cat([rgb, depth], dim=1)   # (B, 4, H, W)

        f = self.stem(x)                      # (B, C,   H/4,  W/4)
        f = self.stage1(f)
        s4 = f

        f = self.down1(f)                     # (B, 2C,  H/8,  W/8)
        f = self.stage2(f)
        s8 = f

        f = self.down2(f)                     # (B, 4C, H/16, W/16)
        f = self.stage3(f)
        s16 = f

        return {
            "s4":  self.proj4(s4),
            "s8":  self.proj8(s8),
            "s16": self.proj16(s16),
        }


# ---------------------------------------------------------------------------
# Factory — single entry-point used by network.py
# ---------------------------------------------------------------------------

def build_depth_stream(
    backbone:     str  = "convnext_tiny",
    pretrained:   bool = True,
    out_channels: int  = 128,
    lite_base_ch: int  = 32,
    lite_num_blocks: int = 2,
) -> nn.Module:
    """
    Return a depth-stream encoder with the requested backbone.

    backbone choices
    ----------------
    "convnext_tiny"   ~28M params  (same channel widths as Small, fewer blocks)
    "convnext_small"  ~50M params
    "conv_lite"       ~165K params  configured via lite_base_ch / lite_num_blocks
    """
    if backbone == "conv_lite":
        return LiteDepthStream(
            base_ch=lite_base_ch,
            num_blocks=lite_num_blocks,
            out_channels=out_channels,
        )
    # ConvNeXt family — delegate to original DepthStream
    return DepthStream(backbone=backbone, pretrained=pretrained, out_channels=out_channels)
