"""
cross_attention.py — Symmetric cross-attention decoder with pose conditioning.

Architecture
------------
Stack of N CrossBlocks applied symmetrically to both depth views:

    for each block:
        t1 = SelfAttn(t1) → CrossAttn(Q=t1, KV=t2) → FFN(t1)
        t2 = SelfAttn(t2) → CrossAttn(Q=t2, KV=t1) → FFN(t2)

Pose conditioning (current strategy: prepend pose as a token)
-------------------------------------------------------------
  1. The relative pose T_12 (and its inverse T_21) is encoded by PoseEncoder
     into a single (B, 1, token_dim) pose token per view.
  2. The pose token is prepended to each view's token sequence before the
     first block and participates in self- and cross-attention throughout.
  3. After all blocks the pose token is stripped; only depth tokens are
     passed to the decoder.

  This is enabled by `use_pose_token=True` (default).  Setting it to False
  skips pose conditioning entirely (useful for ablation or when no pose is
  available at inference time).

  Future extension: replace PoseEncoder with a learned camera token and add
  a CameraHead that regresses pose from the camera token — matching the
  ViSTA / MASt3R paradigm.  The interface is kept compatible for that change.

MASt3R-style initialisation
----------------------------
  • trunc_normal_(std=0.02)  for all weight matrices (ViT convention).
  • zeros_                   for all biases.
  • zeros_                   for the output projections of attention and MLP
    (output projection weights) so residual paths start near identity at
    initialisation — following DUSt3R / MASt3R practice.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# MASt3R-style init
# ---------------------------------------------------------------------------

def _trunc_normal_(tensor: torch.Tensor, std: float = 0.02) -> None:
    """Truncated normal initialisation (ViT / MASt3R convention)."""
    nn.init.trunc_normal_(tensor, std=std, a=-2 * std, b=2 * std)


def _mast3r_init(module: nn.Module) -> None:
    """
    Apply MASt3R / ViT-style init to all Linear layers inside *module*.

    Rules:
      - Linear weights → trunc_normal_(std=0.02)
      - Linear biases  → zeros_
      - Layers named '*.proj' or '*.fc2' (residual output projections)
        additionally receive zeros_ on the weight so that, at init,
        each block is a near-identity residual connection.
    """
    for name, m in module.named_modules():
        if isinstance(m, nn.Linear):
            _trunc_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    # Zero-init output projections for near-identity residuals at t=0.
    for name, m in module.named_modules():
        tail = name.split(".")[-1]
        if isinstance(m, nn.Linear) and tail in ("proj", "fc2"):
            nn.init.zeros_(m.weight)


# ---------------------------------------------------------------------------
# Pose encoder  (9-dim pose → token_dim vector)
# ---------------------------------------------------------------------------

def _rot_to_6d(R: torch.Tensor) -> torch.Tensor:
    """Convert (B, 3, 3) rotation matrix to (B, 6) 6D representation (Zhou et al.)."""
    return R[..., :2].transpose(-1, -2).reshape(R.shape[0], 6)


class PoseEncoder(nn.Module):
    """
    Encode a 4×4 SE(3) pose matrix into a (B, 1, token_dim) conditioning token.

    Representation: [rot_6d (6), translation (3)] → 9-dim → MLP → token_dim

    Parameters
    ----------
    token_dim : int   Dimension of the output token (must match cross-attention dim).
    hidden    : int   Width of the intermediate MLP layer.
    """

    def __init__(self, token_dim: int = 256, hidden: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(9, hidden),
            nn.GELU(),
            nn.Linear(hidden, token_dim),
        )
        _mast3r_init(self.mlp)

    def forward(self, T: torch.Tensor) -> torch.Tensor:
        """
        T : (B, 4, 4) — SE(3) pose matrix.
        Returns (B, 1, token_dim).
        """
        R = T[:, :3, :3]          # (B, 3, 3)
        t = T[:, :3,  3]          # (B, 3)
        r6d = _rot_to_6d(R)       # (B, 6)
        pose_vec = torch.cat([r6d, t], dim=-1)   # (B, 9)
        token = self.mlp(pose_vec)               # (B, token_dim)
        return token.unsqueeze(1)                # (B, 1, token_dim)


# ---------------------------------------------------------------------------
# Transformer sub-modules
# ---------------------------------------------------------------------------

class Attention(nn.Module):
    """Multi-head self-attention."""

    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim  = dim // num_heads
        self.scale     = self.head_dim ** -0.5

        self.qkv  = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim,     bias=True)
        _mast3r_init(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        out = F.scaled_dot_product_attention(q, k, v, scale=self.scale)
        return self.proj(out.transpose(1, 2).reshape(B, N, C))


class CrossAttention(nn.Module):
    """Multi-head cross-attention: query from one view, key/value from the other."""

    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim  = dim // num_heads
        self.scale     = self.head_dim ** -0.5

        self.projq = nn.Linear(dim, dim, bias=True)
        self.projk = nn.Linear(dim, dim, bias=True)
        self.projv = nn.Linear(dim, dim, bias=True)
        self.proj  = nn.Linear(dim, dim, bias=True)
        _mast3r_init(self)

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        """q: (B, Nq, dim), kv: (B, Nk, dim) → (B, Nq, dim)"""
        B, Nq, C = q.shape
        Nk = kv.shape[1]

        def _split(t, N):
            return t.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        Q = _split(self.projq(q),  Nq)
        K = _split(self.projk(kv), Nk)
        V = _split(self.projv(kv), Nk)
        out = F.scaled_dot_product_attention(Q, K, V, scale=self.scale)
        return self.proj(out.transpose(1, 2).reshape(B, Nq, C))


class Mlp(nn.Module):
    """Two-layer MLP with GELU. fc2 is zero-initialised (MASt3R residual style)."""

    def __init__(self, dim: int, mlp_ratio: float = 4.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden, bias=True)
        self.fc2 = nn.Linear(hidden, dim, bias=True)
        _mast3r_init(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


# ---------------------------------------------------------------------------
# Single cross-attention block
# ---------------------------------------------------------------------------

class CrossBlock(nn.Module):
    """
    Self-attn → Cross-attn → FFN  (applied to one view per forward call).

    Parameters
    ----------
    dim       : int   Token dimension.
    num_heads : int   Attention heads (must divide dim).
    mlp_ratio : float MLP hidden / dim ratio.
    """

    def __init__(self, dim: int = 256, num_heads: int = 8, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1      = nn.LayerNorm(dim)
        self.attn       = Attention(dim, num_heads)
        self.norm2      = nn.LayerNorm(dim)
        self.cross_attn = CrossAttention(dim, num_heads)
        self.norm_y     = nn.LayerNorm(dim)   # normalise context before cross-attn (matches MASt3R checkpoint key)
        self.norm3      = nn.LayerNorm(dim)
        self.mlp        = Mlp(dim, mlp_ratio)

    def forward(self, x: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        """
        x   : (B, N, dim) — tokens for this view
        ctx : (B, M, dim) — context tokens from the other view (KV)
        """
        x = x + self.attn(self.norm1(x))
        x = x + self.cross_attn(self.norm2(x), self.norm_y(ctx))
        x = x + self.mlp(self.norm3(x))
        return x


# ---------------------------------------------------------------------------
# Full symmetric cross-attention decoder
# ---------------------------------------------------------------------------

class CrossAttentionDecoder(nn.Module):
    """
    Stack of N CrossBlocks applied symmetrically to two depth-token sequences.

    Optionally prepends a pose-conditioned token to each view before the blocks
    and strips it out before returning (use_pose_token=True, the default).

    Parameters
    ----------
    token_dim      : int   Cross-attention token dimension.
    num_blocks     : int   Number of CrossBlock layers.
    num_heads      : int   Attention heads per block.
    mlp_ratio      : float MLP ratio.
    use_pose_token : bool  Prepend encoded pose as a conditioning token.
    pose_hidden    : int   PoseEncoder MLP hidden width.
    """

    def __init__(
        self,
        token_dim:        int   = 256,
        num_blocks:       int   = 4,
        num_heads:        int   = 8,
        mlp_ratio:        float = 4.0,
        use_pose_token:   bool  = True,
        pose_hidden:      int   = 128,
        predict_pose:     bool  = False,
    ):
        super().__init__()

        self.token_dim      = token_dim
        self.use_pose_token = use_pose_token
        self.predict_pose   = predict_pose

        # Weight-tied blocks: same weights applied to both views.
        # Named 'dec_blocks' to match MASt3R checkpoint key dec_blocks.{i}.*
        self.dec_blocks = nn.ModuleList([
            CrossBlock(token_dim, num_heads, mlp_ratio)
            for _ in range(num_blocks)
        ])
        self.norm1 = nn.LayerNorm(token_dim)  # final LN view 1
        self.norm2 = nn.LayerNorm(token_dim)  # final LN view 2

        if use_pose_token:
            if predict_pose:
                # Learnable camera token — replaces GT-encoded pose token.
                # Both views start from the same init; they diverge through
                # cross-attention and the pose head reads the result.
                self.camera_token = nn.Parameter(torch.zeros(1, 1, token_dim))
                nn.init.trunc_normal_(self.camera_token, std=0.02)
            else:
                self.pose_enc = PoseEncoder(token_dim, pose_hidden)

    def forward(
        self,
        t1:   torch.Tensor,            # (B, N, token_dim) — view-1 tokens
        t2:   torch.Tensor,            # (B, N, token_dim) — view-2 tokens
        T_12: torch.Tensor | None = None,  # (B, 4, 4)  relative pose 1→2
    ) -> tuple:
        """
        Returns
        -------
        (t1_out, t2_out, cam_tok1, cam_tok2)

        t1_out, t2_out : (B, N, token_dim)  — cross-attended depth tokens
                         (pose token is stripped if use_pose_token=True)
        cam_tok1, cam_tok2 : (B, token_dim) evolved camera tokens, or None
                              when predict_pose=False.
        """
        cam_tok1 = cam_tok2 = None

        if self.use_pose_token:
            if self.predict_pose:
                # Broadcast learnable camera token to batch — same for both
                # views at input; cross-attention makes them view-specific.
                B = t1.shape[0]
                cam = self.camera_token.expand(B, -1, -1)  # (B, 1, token_dim)
                t1 = torch.cat([cam, t1], dim=1)
                t2 = torch.cat([cam, t2], dim=1)           # independent clone via cat
            else:
                assert T_12 is not None, "T_12 required when use_pose_token=True and predict_pose=False"
                # Encode T_12 and its inverse for the two views.
                pose_tok_12 = self.pose_enc(T_12)              # (B, 1, token_dim)
                T_21 = _se3_inv(T_12)
                pose_tok_21 = self.pose_enc(T_21)              # (B, 1, token_dim)
                t1 = torch.cat([pose_tok_12, t1], dim=1)
                t2 = torch.cat([pose_tok_21, t2], dim=1)

        for block in self.dec_blocks:
            t1_new = block(t1, t2)
            t2_new = block(t2, t1)
            t1, t2 = t1_new, t2_new

        t1 = self.norm1(t1)
        t2 = self.norm2(t2)

        if self.use_pose_token:
            if self.predict_pose:
                # Capture evolved camera tokens before stripping.
                cam_tok1 = t1[:, 0]   # (B, token_dim)
                cam_tok2 = t2[:, 0]
            # Strip the pose/camera token (position 0) before returning.
            t1 = t1[:, 1:]
            t2 = t2[:, 1:]

        return t1, t2, cam_tok1, cam_tok2

    # ------------------------------------------------------------------
    # MASt3R / DUSt3R weight initialisation
    # ------------------------------------------------------------------

    def load_mast3r_weights(self, ckpt_path: str) -> None:
        """
        Initialise dec_blocks from a MASt3R or DUSt3R checkpoint.

        The checkpoint stores decoder blocks under:
            dec_blocks.{i}.<submodule>.<param>

        where submodule names are:
            norm1, attn (qkv/proj), norm2, cross_attn (projq/projk/projv/proj),
            norm_y, norm3, mlp (fc1/fc2)

        These match our CrossBlock attribute names exactly (after the
        norm_ctx → norm_y rename), so load_state_dict works without key
        remapping.

        Blocks beyond the checkpoint depth keep default init.  Shape
        mismatches are skipped with a warning; this method never raises.

        Parameters
        ----------
        ckpt_path : str  Path to .pth file (e.g.
            MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth).
        """
        print(f"[CrossAttentionDecoder] Loading MASt3R weights from {ckpt_path}")
        ckpt  = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state = ckpt.get("model", ckpt)

        total_loaded = 0
        for i, block in enumerate(self.dec_blocks):
            prefix   = f"dec_blocks.{i}."
            block_sd = {
                k[len(prefix):]: v
                for k, v in state.items()
                if k.startswith(prefix)
            }
            if not block_sd:
                print(f"  block {i}: no matching keys in checkpoint — keeping default init")
                continue

            result   = block.load_state_dict(block_sd, strict=False)
            n_loaded = len(block_sd) - len(result.missing_keys) - len(result.unexpected_keys)
            total_loaded += n_loaded

            if result.missing_keys:
                print(f"  block {i}: missing   {result.missing_keys}")
            if result.unexpected_keys:
                print(f"  block {i}: unexpected {result.unexpected_keys}")

        print(f"[CrossAttentionDecoder] Loaded {total_loaded} parameter tensors total.")

    def freeze_blocks(self) -> None:
        """Freeze all dec_blocks (cross-attention weights). PoseEncoder stays trainable."""
        for block in self.dec_blocks:
            for p in block.parameters():
                p.requires_grad_(False)
        print("[CrossAttentionDecoder] All dec_blocks frozen.")


# ---------------------------------------------------------------------------
# SE(3) inverse  (local utility to avoid circular imports)
# ---------------------------------------------------------------------------

def _se3_inv(T: torch.Tensor) -> torch.Tensor:
    """Numerically stable SE(3) inverse. T: (B, 4, 4) → T_inv: (B, 4, 4)."""
    R = T[:, :3, :3]
    t = T[:, :3,  3]
    R_inv = R.transpose(1, 2)
    t_inv = -(R_inv @ t.unsqueeze(-1)).squeeze(-1)
    T_inv = torch.zeros_like(T)
    T_inv[:, :3, :3] = R_inv
    T_inv[:, :3,  3] = t_inv
    T_inv[:,  3,  3] = 1.0
    return T_inv


# ---------------------------------------------------------------------------
# 6D rotation + SE(3) helpers  (used by PoseHead in network.py)
# ---------------------------------------------------------------------------

def _rot6d_to_matrix(r6d: torch.Tensor) -> torch.Tensor:
    """
    6D rotation representation → (B, 3, 3) rotation matrix.
    Zhou et al., "On the Continuity of Rotation Representations", CVPR 2019.
    Uses Gram-Schmidt orthogonalisation on the first two columns.
    """
    a1 = F.normalize(r6d[:, :3], dim=-1)
    a2 = r6d[:, 3:6]
    a2 = F.normalize(a2 - (a1 * a2).sum(-1, keepdim=True) * a1, dim=-1)
    a3 = torch.cross(a1, a2, dim=-1)
    return torch.stack([a1, a2, a3], dim=-1)   # (B, 3, 3)  column vectors


def _vec9_to_se3(v: torch.Tensor) -> torch.Tensor:
    """
    Map a 9-dim pose vector to a (B, 4, 4) SE(3) matrix.
    v[:, :6]  →  rotation (6D → 3×3 via Gram-Schmidt)
    v[:, 6:]  →  translation
    """
    B = v.shape[0]
    R = _rot6d_to_matrix(v[:, :6])          # (B, 3, 3)
    t = v[:, 6:]                             # (B, 3)
    T = torch.eye(4, device=v.device, dtype=v.dtype).unsqueeze(0).expand(B, -1, -1).contiguous()
    T[:, :3, :3] = R
    T[:, :3,  3] = t
    return T
