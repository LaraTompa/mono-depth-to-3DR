"""
decoder.py — Late-fusion decoder: token grid + depth stream → H/2 depth.

Design (late fusion)
--------------------
The DINOv2 cross-attention pipeline and the ConvNeXt depth stream run
completely independently.  They meet only here, in the decoder.

Stage layout
------------
Given input resolution H × W (e.g. 392 × 518):

  DINOv2 tokens  → (B, N, 768)  where N = (H/14) × (W/14) = 28 × 37
  after token_proj (1024 → 768) applied upstream in network.py

  1. Project tokens:  768 → decoder_dim  (1×1, spatial)
     Reshape to (B, decoder_dim, H/14, W/14)

  2. Upsample to depth-s16 spatial size → cat with s16 → FuseConv
     (B, decoder_dim + depth_ch, H/16, W/16) → (B, decoder_dim, H/16, W/16)
     Note: H/14 ≈ H/16 — bilinear resample bridges the stride mismatch.

  3. Upsample to s8 → cat with s8 → FuseConv  → (B, decoder_dim, H/8, W/8)

  4. Upsample to s4 → cat with s4 → FuseConv  → (B, decoder_dim, H/4, W/4)

  5. Upsample ×2 → FinalConv                  → (B, decoder_dim//2, H/2, W/2)

  6. Two output heads:
       depth      = softplus(scale) * depth_mono_upsampled + residual
       confidence = sigmoid(·)

The mono depth prior is re-injected at the output as a learned
scale + residual so the network never has to learn depth from scratch.
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class FusionDecoder(nn.Module):
    """
    Fuse cross-attended DINOv2 token grid with ConvNeXt depth stream
    features at three scales, producing depth + confidence at H/2.

    Parameters
    ----------
    token_dim   : int   dimension of incoming tokens (768 after enc_to_dec proj)
    depth_ch    : int   channels in each depth-stream scale (DepthStream.out_channels)
    hidden      : int   internal decoder channel width
    """

    def __init__(self, token_dim: int = 768, depth_ch: int = 128, hidden: int = 256):
        super().__init__()

        # Project token dimension → decoder hidden
        self.token_proj = nn.Conv2d(token_dim, hidden, 1)

        # Fusion convolutions at each scale
        # Input = hidden (upsampled tokens) + depth_ch (skip from depth stream)
        self.fuse16 = ConvBnGelu(hidden + depth_ch, hidden)
        self.fuse8  = ConvBnGelu(hidden + depth_ch, hidden)
        self.fuse4  = ConvBnGelu(hidden + depth_ch, hidden)

        # Final upsampling conv to H/2
        self.final_conv = ConvBnGelu(hidden, hidden // 2)

        # Output heads
        self.out_scale = nn.Conv2d(hidden // 2, 1, 1)
        self.out_depth = nn.Conv2d(hidden // 2, 1, 1)
        self.out_conf  = nn.Conv2d(hidden // 2, 1, 1)

        # Near-zero init: start as passthrough of mono prior
        for head in (self.out_scale, self.out_depth, self.out_conf):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(
        self,
        tokens:     torch.Tensor,   # (B, N, 768)   attended patch tokens
        depth_feats: dict,           # {"s4", "s8", "s16"}  from DepthStream
        depth_mono: torch.Tensor,   # (B, 1, H, W)  full-resolution mono prior
        grid_hw:    tuple,           # (h_tokens, w_tokens) = (H/14, W/14)
    ) -> dict:
        """
        Returns {"depth": (B,1,H/2,W/2), "confidence": (B,1,H/2,W/2)}
        """
        h14, w14 = grid_hw
        B = tokens.shape[0]

        # ── Reshape token sequence → spatial feature map ─────────────────
        # tokens: (B, N, 768) → (B, 768, h14, w14) → (B, hidden, h14, w14)
        x = tokens.reshape(B, h14, w14, -1).permute(0, 3, 1, 2).contiguous()
        x = self.token_proj(x)                         # (B, hidden, h14, w14)

        s4  = depth_feats["s4"]    # (B, depth_ch, H/4,  W/4)
        s8  = depth_feats["s8"]    # (B, depth_ch, H/8,  W/8)
        s16 = depth_feats["s16"]   # (B, depth_ch, H/16, W/16)

        # ── Stage 1: token grid → s16 ────────────────────────────────────
        # H/14 ≠ H/16 in general; bilinear resampling bridges the mismatch.
        x = F.interpolate(x, size=s16.shape[-2:], mode="bilinear", align_corners=False)
        x = self.fuse16(torch.cat([x, s16], dim=1))   # (B, hidden, H/16, W/16)

        # ── Stage 2: s16 → s8 ────────────────────────────────────────────
        x = F.interpolate(x, size=s8.shape[-2:], mode="bilinear", align_corners=False)
        x = self.fuse8(torch.cat([x, s8], dim=1))     # (B, hidden, H/8,  W/8)

        # ── Stage 3: s8 → s4 ─────────────────────────────────────────────
        x = F.interpolate(x, size=s4.shape[-2:], mode="bilinear", align_corners=False)
        x = self.fuse4(torch.cat([x, s4], dim=1))     # (B, hidden, H/4,  W/4)

        # ── Stage 4: s4 → H/2 ────────────────────────────────────────────
        H2 = s4.shape[-2] * 2
        W2 = s4.shape[-1] * 2
        x = F.interpolate(x, size=(H2, W2), mode="bilinear", align_corners=False)
        x = self.final_conv(x)                         # (B, hidden//2, H/2, W/2)

        # ── Output: scale × prior + residual ─────────────────────────────
        depth_mono_up = F.interpolate(
            depth_mono, size=(H2, W2), mode="bilinear", align_corners=False
        )
        scale = F.softplus(self.out_scale(x)) + 1e-3  # positive scale
        # Residual clamped to ±20% of local mono depth — network corrects the
        # prior rather than re-predicting depth from scratch.
        depth_residual = depth_mono_up * 0.2 * torch.tanh(self.out_depth(x))
        depth = (scale * depth_mono_up + depth_residual).clamp(min=1e-3)
        conf  = torch.sigmoid(self.out_conf(x))        # ∈ (0, 1)

        return {"depth": depth, "confidence": conf}
