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

Positional embeddings
---------------------
MASt3R is trained on 512 × 512 images (32 × 32 = 1024 patches).  For other
resolutions the stored positional embeddings are bicubically interpolated,
following DINOv2 / MASt3R practice.

Checkpoint key structure (MASt3R / CroCo)
------------------------------------------
The weight loader expects:
  encoder.patch_embed.proj.weight / bias
  encoder.cls_token
  encoder.pos_embed
  encoder.blocks.{i}.norm1.weight / bias
  encoder.blocks.{i}.attn.qkv.weight / bias
  encoder.blocks.{i}.attn.proj.weight / bias
  encoder.blocks.{i}.norm2.weight / bias
  encoder.blocks.{i}.mlp.fc1.weight / bias
  encoder.blocks.{i}.mlp.fc2.weight / bias
  enc_norm.weight / bias          ← stored at the checkpoint root, not under encoder.*
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# ImageNet mean / std used by MASt3R / CroCo during pre-training.
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
_IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225])


# ---------------------------------------------------------------------------
# ViT building blocks  (key names match MASt3R / CroCo checkpoint exactly)
# ---------------------------------------------------------------------------

class _ViTAttention(nn.Module):
    """Standard multi-head self-attention.
    Checkpoint keys: attn.qkv.{weight,bias}, attn.proj.{weight,bias}"""

    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim  = dim // num_heads
        self.scale     = self.head_dim ** -0.5
        self.qkv  = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim,     bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
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
    """Standard pre-norm ViT block: self-attn + MLP with residuals.
    Checkpoint keys: norm1.*, attn.*, norm2.*, mlp.*"""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = _ViTAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp   = _ViTMlp(dim, mlp_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Patch embedding  (matches MASt3R key: encoder.patch_embed.proj.*)
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
    mast3r_ckpt  : str|None  Path to MASt3R .pth checkpoint (None → random init).
    freeze       : bool   Freeze backbone after loading (default True).
                          The output projection (proj) is always trainable.
    """

    # MASt3R default training resolution for stored positional embeddings.
    _TRAIN_H: int = 512
    _TRAIN_W: int = 512

    def __init__(
        self,
        token_dim:   int        = 768,
        patch_size:  int        = 16,
        embed_dim:   int        = 1024,
        depth:       int        = 24,
        num_heads:   int        = 16,
        mlp_ratio:   float      = 4.0,
        max_hw:      tuple | None = (480, 640),
        mast3r_ckpt: str | None = None,
        freeze:      bool       = True,
    ):
        super().__init__()

        self.patch_size = patch_size
        self.embed_dim  = embed_dim
        self.max_hw     = max_hw

        # ── Patch embedding ───────────────────────────────────────────────
        # Checkpoint key prefix: encoder.patch_embed.*
        self.patch_embed = _PatchEmbed(3, embed_dim, patch_size)

        # ── CLS token ─────────────────────────────────────────────────────
        # Checkpoint key: encoder.cls_token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # ── Positional embedding (train resolution: 512×512 = 32×32 patches) ──
        # Checkpoint key: encoder.pos_embed  shape (1, 1 + 1024, 1024)
        n_train = (self._TRAIN_H // patch_size) * (self._TRAIN_W // patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + n_train, embed_dim))

        # ── Transformer blocks ────────────────────────────────────────────
        # Checkpoint keys: encoder.blocks.{i}.*
        self.blocks = nn.ModuleList([
            _ViTBlock(embed_dim, num_heads, mlp_ratio)
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

        Expected key layout in the checkpoint's 'model' dict:
          encoder.patch_embed.proj.*
          encoder.cls_token
          encoder.pos_embed
          encoder.blocks.{i}.*
          enc_norm.*   ← mapped to self.norm.*

        The output projection (self.proj) is NOT present in the checkpoint;
        it stays at zero-init after loading.
        """
        print(f"[MASt3RImageEncoder] Loading weights from {ckpt_path}")
        ckpt  = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state = ckpt.get("model", ckpt)

        # Build a remapped state-dict matching our module's parameter names.
        remapped: dict = {}
        for k, v in state.items():
            if k.startswith("encoder."):
                remapped[k[len("encoder."):]] = v       # strip "encoder." prefix
            elif k.startswith("enc_norm."):
                remapped["norm." + k[len("enc_norm."):]] = v   # enc_norm → norm

        if not remapped:
            print("  [warn] No encoder.* / enc_norm.* keys found — skipping.")
            return

        result = self.load_state_dict(remapped, strict=False)

        # 'proj.*' will always appear as missing (not in checkpoint) — expected.
        proj_missing  = {k for k in result.missing_keys if k.startswith("proj")}
        other_missing = set(result.missing_keys) - proj_missing
        if other_missing:
            print(f"  [warn] Missing keys (unexpected): {sorted(other_missing)}")
        if result.unexpected_keys:
            print(f"  [warn] Unexpected keys: {sorted(result.unexpected_keys)[:10]}")

        loaded = len(remapped) - len(result.missing_keys) - len(result.unexpected_keys)
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

    def _interpolate_pos_embed(self, h: int, w: int) -> torch.Tensor:
        """
        Bicubically interpolate positional embeddings to patch grid (h, w).

        The stored pos_embed was created for the MASt3R training resolution
        (512×512 → 32×32 patches).  For other resolutions we follow the
        DINOv2 / MASt3R practice of bicubic interpolation on the spatial
        grid, keeping the CLS token position fixed.
        """
        pos      = self.pos_embed                        # (1, 1+N_train, D)
        cls_pos  = pos[:, :1]                            # (1, 1, D)
        grid_pos = pos[:, 1:]                            # (1, N_train, D)

        h_train = self._TRAIN_H // self.patch_size       # 32
        w_train = self._TRAIN_W // self.patch_size       # 32

        if h == h_train and w == w_train:
            return pos                                   # fast-path: no interpolation

        D = grid_pos.shape[-1]
        grid_pos = (
            grid_pos
            .reshape(1, h_train, w_train, D)
            .permute(0, 3, 1, 2)                         # (1, D, h_train, w_train)
            .float()
        )
        grid_pos = F.interpolate(
            grid_pos, size=(h, w), mode="bicubic", align_corners=False
        ).to(pos.dtype)
        grid_pos = grid_pos.permute(0, 2, 3, 1).reshape(1, h * w, D)

        return torch.cat([cls_pos, grid_pos], dim=1)     # (1, 1+h*w, D)

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
                 CLS token is discarded.  Projected to token_dim via the
                 zero-initialised self.proj layer.
        """
        img = self._cap_resolution(img)

        # ImageNet normalisation (in-place on the capped copy; no side-effects).
        img = (img - self._img_mean) / self._img_std

        _, _, H, W = img.shape
        h = H // self.patch_size
        w = W // self.patch_size

        # Patch embed + CLS prepend + positional encoding
        x   = self.patch_embed(img)                       # (B, h*w, embed_dim)
        B   = x.shape[0]
        cls = self.cls_token.expand(B, -1, -1)            # (B, 1, embed_dim)
        x   = torch.cat([cls, x], dim=1)                  # (B, 1+h*w, embed_dim)
        x   = x + self._interpolate_pos_embed(h, w)

        # Transformer blocks (frozen backbone)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)

        # Drop CLS token, project to token_dim
        x = x[:, 1:]          # (B, h*w, embed_dim)
        return self.proj(x)   # (B, h*w, token_dim)
