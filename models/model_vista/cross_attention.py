"""
cross_attention.py — ViSTA-SLAM / MASt3R-compatible cross-attention decoder.

Architecture explanation
------------------------
ViSTA-SLAM (and DUSt3R / MASt3R before it) process two views jointly
through a stack of symmetric cross-attention blocks.  Unlike the single
nn.MultiheadAttention call in model_image_depth, this is a deep
transformer decoder that refines both views simultaneously:

    for each block:
        t1 = SelfAttn(t1) → CrossAttn(Q=t1, KV=t2) → FFN(t1)
        t2 = SelfAttn(t2) → CrossAttn(Q=t2, KV=t1) → FFN(t2)

After N blocks, t1 contains view-1 features that have "seen" view-2
geometry through progressively deeper cross-view reasoning.  This is
fundamentally different from one-shot attention because:
  - Each block refines the correspondence using the updated tokens from
    the previous block (iterative refinement).
  - Both views update simultaneously, so the cost volume is implicit
    rather than explicit.
  - The depth prediction head only needs to decode final tokens —
    no separate refinement GRU is required.

Weight naming
-------------
All sub-module names mirror CroCo / MASt3R's decoder block naming
exactly so that weights from the public checkpoint

    MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth

can be loaded without any key renaming.  That checkpoint stores decoder
blocks under the key pattern:

    dec_blocks.{i}.norm1.weight
    dec_blocks.{i}.attn.qkv.weight        (self-attn, merged QKV)
    dec_blocks.{i}.attn.proj.weight
    dec_blocks.{i}.norm2.weight
    dec_blocks.{i}.cross_attn.projq.weight
    dec_blocks.{i}.cross_attn.projk.weight
    dec_blocks.{i}.cross_attn.projv.weight
    dec_blocks.{i}.cross_attn.proj.weight
    dec_blocks.{i}.norm3.weight
    dec_blocks.{i}.mlp.fc1.weight
    dec_blocks.{i}.mlp.fc2.weight

Dimension note
--------------
MASt3R uses ViT-Large (1024-dim) as encoder but the decoder ("BaseDecoder")
operates at 768-dim with 12 heads.  An enc_to_dec projection (1024 → 768)
is used in MASt3R and is replicated in DepthAlignNetV2 as a trainable
linear layer (not loaded from the checkpoint).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# CroCo-compatible sub-modules (naming must match checkpoint exactly)
# ---------------------------------------------------------------------------

class Attention(nn.Module):
    """Multi-head self-attention.  Naming matches CroCo's Attention class."""

    def __init__(self, dim: int, num_heads: int = 12):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim  = dim // num_heads
        self.scale     = self.head_dim ** -0.5

        self.qkv  = nn.Linear(dim, dim * 3, bias=True)   # merged QKV
        self.proj = nn.Linear(dim, dim,     bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)           # (3, B, heads, N, head_dim)
        q, k, v = qkv.unbind(0)
        out = F.scaled_dot_product_attention(q, k, v, scale=self.scale)
        out = out.transpose(1, 2).reshape(B, N, C)
        return self.proj(out)


class CrossAttention(nn.Module):
    """Multi-head cross-attention.  Naming matches CroCo's CrossAttention."""

    def __init__(self, dim: int, num_heads: int = 12):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim  = dim // num_heads
        self.scale     = self.head_dim ** -0.5

        # Separate projections (not merged) — matches CroCo naming
        self.projq = nn.Linear(dim, dim, bias=True)
        self.projk = nn.Linear(dim, dim, bias=True)
        self.projv = nn.Linear(dim, dim, bias=True)
        self.proj  = nn.Linear(dim, dim, bias=True)

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        """
        q  : (B, Nq, dim)  — query tokens (this view)
        kv : (B, Nk, dim)  — key/value tokens (other view)
        """
        B, Nq, C = q.shape
        Nk = kv.shape[1]

        def _reshape(t: torch.Tensor, N: int) -> torch.Tensor:
            return t.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        Q = _reshape(self.projq(q),  Nq)
        K = _reshape(self.projk(kv), Nk)
        V = _reshape(self.projv(kv), Nk)

        out = F.scaled_dot_product_attention(Q, K, V, scale=self.scale)
        out = out.transpose(1, 2).reshape(B, Nq, C)
        return self.proj(out)


class Mlp(nn.Module):
    """Two-layer MLP.  Naming matches CroCo's Mlp class (fc1 / fc2)."""

    def __init__(self, dim: int, mlp_ratio: float = 4.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden, bias=True)
        self.fc2 = nn.Linear(hidden, dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


# ---------------------------------------------------------------------------
# Single decoder block  (CroCo DecoderBlock-compatible)
# ---------------------------------------------------------------------------

class CrossBlock(nn.Module):
    """
    One CroCo-compatible decoder block: self-attn → cross-attn → MLP.

    Sub-module attribute names are intentionally identical to those in
    MASt3R's checkpoint so that load_state_dict works without renaming:
        norm1, attn, norm2, cross_attn, norm3, mlp

    Parameters
    ----------
    dim       : int   token dimension (768 for MASt3R BaseDecoder)
    num_heads : int   attention heads (12 for dim=768)
    mlp_ratio : float MLP hidden / dim ratio
    """

    def __init__(self, dim: int = 768, num_heads: int = 12, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1      = nn.LayerNorm(dim)
        self.attn       = Attention(dim, num_heads)
        self.norm2      = nn.LayerNorm(dim)
        self.cross_attn = CrossAttention(dim, num_heads)
        self.norm3      = nn.LayerNorm(dim)
        self.mlp        = Mlp(dim, mlp_ratio)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        x       : (B, N, dim)  — tokens for this view
        context : (B, M, dim)  — tokens from the other view (key / value)

        Self-attention lets each view gather internal context first; then
        cross-attention brings in information from the other view.
        """
        x = x + self.attn(self.norm1(x))
        x = x + self.cross_attn(self.norm2(x), context)
        x = x + self.mlp(self.norm3(x))
        return x


# ---------------------------------------------------------------------------
# Full symmetric cross-attention decoder
# ---------------------------------------------------------------------------

class CrossAttentionDecoder(nn.Module):
    """
    Stack of CrossBlocks applied symmetrically to both views.

    Each block updates both views using the *previous block's output*
    from the other view as context — this is the ViSTA-SLAM / DUSt3R
    "alternating cross-attention" pattern.  After N blocks both token
    sequences carry rich mutual cross-view information for depth decoding.

    Parameters
    ----------
    num_blocks : int   number of CrossBlock layers (start: 4 ≈ 5 min/epoch;
                       scale to 8–10 once training is stable)
    dim        : int   token dimension — must match the MASt3R checkpoint
                       decoder dimension (768 for BaseDecoder)
    num_heads  : int   attention heads (12 for dim=768)
    mlp_ratio  : float MLP expansion ratio (4.0 matches CroCo default)

    Weight initialisation
    ---------------------
    After construction, call .load_mast3r_weights(path) to initialise
    all available blocks from the MASt3R public checkpoint.  Blocks
    beyond those in the checkpoint (if num_blocks > checkpoint depth)
    keep their default PyTorch initialisation.  Any shape mismatches are
    skipped with a printed warning so loading never crashes.
    """

    def __init__(
        self,
        num_blocks: int   = 4,
        dim:        int   = 768,
        num_heads:  int   = 12,
        mlp_ratio:  float = 4.0,
    ):
        super().__init__()
        self.dec_blocks = nn.ModuleList([
            CrossBlock(dim, num_heads, mlp_ratio) for _ in range(num_blocks)
        ])
        # Final layer-norm applied to both views' tokens before decoding.
        self.norm = nn.LayerNorm(dim)

    def forward(
        self,
        tokens1: torch.Tensor,    # (B, N, dim)
        tokens2: torch.Tensor,    # (B, N, dim)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns updated (tokens1, tokens2), each (B, N, dim).

        Iteration order: each block sees the OTHER view's tokens from the
        PREVIOUS block.  Both views are processed with the SAME block
        weights (weight-tied streams).
        """
        for block in self.dec_blocks:
            # Compute updates for both views before applying either,
            # so that block i+1 sees block i's output from both views.
            new1 = block(tokens1, tokens2)
            new2 = block(tokens2, tokens1)
            tokens1, tokens2 = new1, new2

        return self.norm(tokens1), self.norm(tokens2)

    # ------------------------------------------------------------------
    # MASt3R / DUSt3R weight initialisation
    # ------------------------------------------------------------------

    def load_mast3r_weights(self, ckpt_path: str) -> None:
        """
        Initialise cross-attention blocks from a MASt3R or DUSt3R checkpoint.

        The checkpoint stores decoder blocks under:
            dec_blocks.{i}.<submodule>.<param>

        Note: MASt3R also contains a second decoder branch under dec_blocks2
        (used for view-2 in the original asymmetric model).  This loader uses
        only dec_blocks because our CrossAttentionDecoder is weight-tied across
        views (the same block instance processes both views).
        Missing or shape-mismatched keys are skipped with a warning; the
        method never raises an exception on partial matches.

        Parameters
        ----------
        ckpt_path : str
            Path to the .pth file (e.g. MASt3R_ViTLarge_BaseDecoder_512_…).
        """
        print(f"[CrossAttentionDecoder] Loading MASt3R weights from {ckpt_path}")
        ckpt  = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        # Checkpoints may nest weights under a 'model' key
        state = ckpt.get("model", ckpt)

        total_loaded = 0
        for i, block in enumerate(self.dec_blocks):
            prefix     = f"dec_blocks.{i}."
            block_sd   = {
                k[len(prefix):]: v
                for k, v in state.items()
                if k.startswith(prefix)
            }
            if not block_sd:
                print(f"  block {i}: no matching keys in checkpoint — keeping default init")
                continue

            result = block.load_state_dict(block_sd, strict=False)
            n_loaded = len(block_sd) - len(result.missing_keys) - len(result.unexpected_keys)
            total_loaded += n_loaded

            if result.missing_keys:
                print(f"  block {i}: missing  {result.missing_keys}")
            if result.unexpected_keys:
                print(f"  block {i}: unexpected {result.unexpected_keys}")

        print(f"[CrossAttentionDecoder] Loaded {total_loaded} parameter tensors total.")
