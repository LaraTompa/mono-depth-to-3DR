"""
image_encoder.py — Frozen MASt3R ViT-Large image encoder for model_depth_only.

Overview
--------
Encodes an RGB image using the Vision Transformer encoder from a MASt3R
checkpoint (ViT-Large, patch_size=16, embed_dim=1024, depth=24, num_heads=16).

The backbone is kept *frozen* after loading.  Only the output projection
(embed_dim → token_dim) is trainable.  That projection is **zero-initialised**
so that at t=0 the image stream contributes nothing and the model is
numerically identical to the depth-only baseline — guaranteeing a stable
warm-up regardless of checkpoint quality.

Spatial alignment with the depth encoder
-----------------------------------------
When both encoders see the same capped resolution (max_encode_hw = (480, 640)):

  • Depth encoder  → ConvNeXt stride-16 → s16 grid: 30 × 40 = 1200 tokens
  • Image encoder  → ViT patch-16        → patch grid: 30 × 40 = 1200 tokens

They are perfectly aligned, so the projected image tokens can be directly
summed onto the depth tokens before cross-attention with no resampling.

ImageNet normalisation
----------------------
The MASt3R ViT was pre-trained on images normalised with ImageNet mean/std.
Normalisation is applied **inside** ``forward`` so callers can pass raw
[0, 1]-range RGB tensors (the format already used by the training loop).

Positional embeddings — RoPE2D, not additive / not learned
------------------------------------------------------------
MASt3R / CroCo v2 do **not** use a learned additive positional embedding or
a CLS token.  Position is injected purely inside attention via axial 2D
Rotary Position Embedding (``pos_embed='RoPE100'`` in the original CroCo
config, i.e. rotary base frequency θ=100).  Each attention head's feature
dimension is split in half: one half is rotated according to the patch's
row index, the other half according to its column index (see CroCo v2,
§3.2, "Positional embeddings").  Because RoPE encodes *relative* position
directly in attention rather than adding an absolute positional tensor to
the input, it generalises to any input resolution with **no interpolation
needed** — unlike a learned/interpolated absolute pos_embed.

The ``_RoPE2D`` class below is a direct port of CroCo's own pure-PyTorch
fallback implementation (``croco/models/pos_embed.py``, used automatically
when the optional CUDA ``curope`` kernels aren't compiled), so that loaded
attention weights see the same positional treatment they were trained with.

Checkpoint key structure (MASt3R / CroCo)
------------------------------------------
The weight loader expects the *actual* CroCo/MASt3R checkpoint layout
(verified against a real MASt3R_ViTLarge_BaseDecoder checkpoint):
  patch_embed.proj.weight / bias
  enc_blocks.{i}.norm1.weight / bias
  enc_blocks.{i}.attn.qkv.weight / bias
  enc_blocks.{i}.attn.proj.weight / bias
  enc_blocks.{i}.norm2.weight / bias
  enc_blocks.{i}.mlp.fc1.weight / bias
  enc_blocks.{i}.mlp.fc2.weight / bias
  enc_norm.weight / bias          ← stored at the checkpoint root

Note: there is no ``encoder.`` prefix, no ``cls_token``, and no
``pos_embed`` tensor in these checkpoints — those were incorrect
assumptions in an earlier version of this loader based on generic ViT
checkpoint conventions rather than the actual CroCo/MASt3R format.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# ImageNet mean / std used by MASt3R / CroCo during pre-training.
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
_IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225])


# ---------------------------------------------------------------------------
# RoPE2D — axial 2D rotary position embedding.
#
# Direct port of CroCo's pure-PyTorch fallback (croco/models/pos_embed.py,
# class RoPE2D, the branch used when the optional CUDA curope kernels are
# not compiled). Kept numerically identical to that reference so weights
# trained with it behave the same way here.
# ---------------------------------------------------------------------------

class _RoPE2D(nn.Module):
    """Axial 2D RoPE, base frequency `freq` (100.0 for MASt3R's RoPE100)."""

    def __init__(self, freq: float = 100.0, F0: float = 1.0):
        super().__init__()
        self.base = freq
        self.F0 = F0
        self.cache = {}

    def get_cos_sin(self, D: int, seq_len: int, device, dtype):
        if (D, seq_len, device, dtype) not in self.cache:
            inv_freq = 1.0 / (self.base ** (torch.arange(0, D, 2, device=device).float() / D))
            t = torch.arange(seq_len, device=device, dtype=inv_freq.dtype)
            freqs = torch.einsum("i,j->ij", t, inv_freq).to(dtype)
            freqs = torch.cat((freqs, freqs), dim=-1)
            cos = freqs.cos()  # (Seq, D)
            sin = freqs.sin()
            self.cache[D, seq_len, device, dtype] = (cos, sin)
        return self.cache[D, seq_len, device, dtype]

    @staticmethod
    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)

    def apply_rope1d(self, tokens: torch.Tensor, pos1d: torch.Tensor, cos, sin) -> torch.Tensor:
        assert pos1d.ndim == 2
        cos = F.embedding(pos1d, cos)[:, None, :, :]
        sin = F.embedding(pos1d, sin)[:, None, :, :]
        return (tokens * cos) + (self.rotate_half(tokens) * sin)

    def forward(self, tokens: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """
        tokens    : (B, nheads, ntokens, head_dim)
        positions : (B, ntokens, 2)   — (row, col) grid position per token
        """
        assert tokens.size(3) % 2 == 0, "head_dim must be even for RoPE2D"
        D = tokens.size(3) // 2
        assert positions.ndim == 3 and positions.shape[-1] == 2
        cos, sin = self.get_cos_sin(D, int(positions.max()) + 1, tokens.device, tokens.dtype)
        # Split the head dim in half: rotate one half by row, the other by column.
        y, x = tokens.chunk(2, dim=-1)
        y = self.apply_rope1d(y, positions[:, :, 0], cos, sin)
        x = self.apply_rope1d(x, positions[:, :, 1], cos, sin)
        return torch.cat((y, x), dim=-1)


# ---------------------------------------------------------------------------
# ViT building blocks  (key names match MASt3R / CroCo checkpoint exactly)
# ---------------------------------------------------------------------------

class _ViTAttention(nn.Module):
    """Multi-head self-attention with RoPE2D applied to q/k.
    Checkpoint keys: attn.qkv.{weight,bias}, attn.proj.{weight,bias}"""

    def __init__(self, dim: int, num_heads: int, rope: _RoPE2D):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim  = dim // num_heads
        self.scale     = self.head_dim ** -0.5
        self.qkv  = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim,     bias=True)
        self.rope = rope

    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)               # each (B, num_heads, N, head_dim)
        q = self.rope(q, positions)
        k = self.rope(k, positions)
        out = F.scaled_dot_product_attention(q, k, v, scale=self.scale)
        return self.proj(out.transpose(1, 2).reshape(B, N, C))


class _ViTMlp(nn.Module):
    """Standard ViT MLP.
    Checkpoint keys: mlp.fc1.{weight,bias}, mlp.fc2.{weight,bias}"""

    def __init__(self, dim: int, mlp_ratio: float = 4.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden, bias=True)
        self.fc2 = nn.Linear(hidden, dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class _ViTBlock(nn.Module):
    """Pre-norm ViT block: RoPE self-attn + MLP with residuals.
    Checkpoint keys: norm1.*, attn.*, norm2.*, mlp.*"""

    def __init__(self, dim: int, num_heads: int, rope: _RoPE2D, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = _ViTAttention(dim, num_heads, rope)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp   = _ViTMlp(dim, mlp_ratio)

    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), positions)
        x = x + self.mlp(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Patch embedding  (matches MASt3R key: patch_embed.proj.*)
# ---------------------------------------------------------------------------

class _PatchEmbed(nn.Module):
    """Convolutional patch embedding with key name matching MASt3R."""

    def __init__(self, in_chans: int, embed_dim: int, patch_size: int):
        super().__init__()
        self.proj = nn.Conv2d(
            in_chans, embed_dim,
            kernel_size=patch_size, stride=patch_size, bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, C, H, W) → (B, h*w, embed_dim)
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)


# ---------------------------------------------------------------------------
# Main image encoder
# ---------------------------------------------------------------------------

class MASt3RImageEncoder(nn.Module):
    """
    Frozen ViT-Large image encoder matching the MASt3R / CroCo checkpoint.

    Parameters
    ----------
    token_dim    : int    Target projection dimension (matches cross-attn token_dim).
    patch_size   : int    ViT patch size (16 for MASt3R ViT-Large).
    embed_dim    : int    ViT hidden dimension (1024 for ViT-Large).
    depth        : int    Number of ViT blocks (24 for ViT-Large).
    num_heads    : int    Attention heads (16 for ViT-Large).
    mlp_ratio    : float  MLP width ratio.
    max_hw       : tuple  (H, W) resolution cap — must match depth encoder cap.
    rope_freq    : float  RoPE base frequency (100.0 for MASt3R's 'RoPE100').
    mast3r_ckpt  : str|None  Path to MASt3R .pth checkpoint (None → random init).
    freeze       : bool   Freeze backbone after loading (default True).
                          The output projection (proj) is always trainable.
    """

    def __init__(
        self,
        token_dim:   int        = 768,
        patch_size:  int        = 16,
        embed_dim:   int        = 1024,
        depth:       int        = 24,
        num_heads:   int        = 16,
        mlp_ratio:   float      = 4.0,
        max_hw:      tuple | None = (480, 640),
        rope_freq:   float      = 100.0,
        mast3r_ckpt: str | None = None,
        freeze:      bool       = True,
    ):
        super().__init__()

        self.patch_size = patch_size
        self.embed_dim  = embed_dim
        self.max_hw     = max_hw

        # ── Patch embedding ───────────────────────────────────────────────
        # Checkpoint key prefix: patch_embed.*  (no "encoder." prefix)
        self.patch_embed = _PatchEmbed(3, embed_dim, patch_size)

        # ── RoPE2D (no learnable parameters; shared across all blocks) ────
        self.rope = _RoPE2D(freq=rope_freq)

        # ── Transformer blocks ────────────────────────────────────────────
        # Checkpoint keys: enc_blocks.{i}.*  → remapped to blocks.{i}.*
        self.blocks = nn.ModuleList([
            _ViTBlock(embed_dim, num_heads, self.rope, mlp_ratio)
            for _ in range(depth)
        ])

        # ── Final layer-norm ──────────────────────────────────────────────
        # Checkpoint key: enc_norm.*  (stored at root, not under encoder.*)
        self.norm = nn.LayerNorm(embed_dim)

        # ── Output projection: embed_dim → token_dim ─────────────────────
        # Zero-init: image stream is a no-op at t=0; training enables it
        # gradually.  Only this layer stays trainable when freeze=True.
        self.proj = nn.Linear(embed_dim, token_dim, bias=True)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

        # ── Load pretrained weights ───────────────────────────────────────
        if mast3r_ckpt is not None:
            self.load_mast3r_weights(mast3r_ckpt)

        # ── Freeze backbone (proj stays trainable) ────────────────────────
        if freeze:
            for name, p in self.named_parameters():
                if not name.startswith("proj"):
                    p.requires_grad_(False)

        # Register ImageNet stats as non-trainable buffers so they move to
        # the correct device automatically with .to(device).
        self.register_buffer("_img_mean", _IMAGENET_MEAN.view(1, 3, 1, 1))
        self.register_buffer("_img_std",  _IMAGENET_STD.view(1, 3, 1, 1))

    # ------------------------------------------------------------------
    # Weight loading
    # ------------------------------------------------------------------

    def load_mast3r_weights(self, ckpt_path: str) -> None:
        """
        Load encoder weights from a MASt3R or CroCo checkpoint.

        Expected key layout in the checkpoint's 'model' dict (verified
        against an actual MASt3R_ViTLarge_BaseDecoder checkpoint):
          patch_embed.proj.*
          enc_blocks.{i}.*     → mapped to self.blocks.{i}.*
          enc_norm.*           → mapped to self.norm.*

        There is no cls_token / pos_embed in these checkpoints (position
        is handled by RoPE2D, which has no learnable parameters), and no
        "encoder." prefix on any key.

        The output projection (self.proj) is NOT present in the checkpoint;
        it stays at zero-init after loading.
        """
        print(f"[MASt3RImageEncoder] Loading weights from {ckpt_path}")
        ckpt  = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state = ckpt.get("model", ckpt)

        # Build a remapped state-dict matching our module's parameter names.
        remapped: dict = {}
        for k, v in state.items():
            if k.startswith("enc_blocks."):
                remapped["blocks." + k[len("enc_blocks."):]] = v
            elif k.startswith("enc_norm."):
                remapped["norm." + k[len("enc_norm."):]] = v
            elif k.startswith("patch_embed."):
                remapped[k] = v
            # Everything else (dec_blocks.*, decoder_embed.*, dec_norm.*,
            # downstream_head*.*, mask_token, ...) belongs to the decoder /
            # prediction heads and is intentionally not loaded here.

        if not remapped:
            print("  [warn] No matching encoder keys found — skipping.")
            return

        result = self.load_state_dict(remapped, strict=False)

        # 'proj.*' will always appear as missing (not in checkpoint) — expected.
        proj_missing  = {k for k in result.missing_keys if k.startswith("proj")}
        other_missing = set(result.missing_keys) - proj_missing
        if other_missing:
            print(f"  [warn] Missing keys (unexpected): {sorted(other_missing)}")
        if result.unexpected_keys:
            print(f"  [warn] Unexpected keys: {sorted(result.unexpected_keys)[:10]}")

        loaded = len(remapped) - len(other_missing) - len(result.unexpected_keys)
        print(f"[MASt3RImageEncoder] Loaded {loaded} parameter tensors.")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _cap_resolution(self, img: torch.Tensor) -> torch.Tensor:
        """Resize to at most max_hw (preserving aspect ratio) if needed."""
        if self.max_hw is None:
            return img
        max_h, max_w = self.max_hw
        _, _, H, W = img.shape
        if H <= max_h and W <= max_w:
            return img
        scale = min(max_h / H, max_w / W)
        nh = int(H * scale)
        nw = int(W * scale)
        return F.interpolate(img, size=(nh, nw), mode="bilinear", align_corners=False)

    def _grid_positions(self, h: int, w: int, device) -> torch.Tensor:
        """(row, col) grid position for each of the h*w patch tokens, in the
        same row-major order produced by _PatchEmbed (flatten of H,W)."""
        rows = torch.arange(h, device=device).repeat_interleave(w)
        cols = torch.arange(w, device=device).repeat(h)
        return torch.stack([rows, cols], dim=1)  # (h*w, 2)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        img : (B, 3, H, W)  RGB image in [0, 1] range.

        Returns
        -------
        tokens : (B, h*w, token_dim)
                 h = H_capped // patch_size,  w = W_capped // patch_size.
                 No CLS token (CroCo/MASt3R encoders don't have one).
                 Projected to token_dim via the zero-initialised self.proj.
        """
        img = self._cap_resolution(img)

        # ImageNet normalisation (in-place on the capped copy; no side-effects).
        img = (img - self._img_mean) / self._img_std

        _, _, H, W = img.shape
        h = H // self.patch_size
        w = W // self.patch_size

        # Patch embed — no CLS token, no additive positional embedding.
        x = self.patch_embed(img)                          # (B, h*w, embed_dim)
        B = x.shape[0]
        positions = self._grid_positions(h, w, x.device)   # (h*w, 2)
        positions = positions.unsqueeze(0).expand(B, -1, -1)  # (B, h*w, 2)

        # Transformer blocks (frozen backbone), position injected via RoPE2D.
        for block in self.blocks:
            x = block(x, positions)
        x = self.norm(x)

        # Project to token_dim (no CLS token to drop — all h*w tokens are real).
        return self.proj(x)   # (B, h*w, token_dim)