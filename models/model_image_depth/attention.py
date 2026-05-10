"""
attention.py — Local geometry-aware cross-attention with FiLM pose conditioning.

Design
------
For every spatial position (query) in view-1 features:
  1. Use the reprojected pixel location in view-2 to define a local window
     (window_size × window_size) centred at the projected correspondence.
  2. Attend only within that window (O(H·W·w²) instead of O(H²·W²)).
  3. Keys and Values are modulated by a pose embedding (FiLM).

The result is a cross-attended feature map for view-1 that carries view-2
information, weighted by geometric plausibility.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt

from .geometry import reprojection_coords, rot_to_6d


# ---------------------------------------------------------------------------
# Pose encoder  →  FiLM scale + shift vectors
# ---------------------------------------------------------------------------

class PoseEncoder(nn.Module):
    """
    Encode a 4×4 rigid transform into a pose embedding, then produce
    per-channel FiLM (gamma, beta) vectors for Keys and Values.

    Parameters
    ----------
    embed_dim : int   feature dimension of K and V
    hidden    : int   MLP hidden size
    """

    def __init__(self, embed_dim: int, hidden: int = 256):
        super().__init__()
        # rotation as 6D + translation 3D = 9D
        self.mlp = nn.Sequential(
            nn.Linear(9, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        # separate heads for K and V modulation
        self.to_gamma_k = nn.Linear(hidden, embed_dim)
        self.to_beta_k  = nn.Linear(hidden, embed_dim)
        self.to_gamma_v = nn.Linear(hidden, embed_dim)
        self.to_beta_v  = nn.Linear(hidden, embed_dim)

        # init gamma to 1, beta to 0 → identity at start
        nn.init.ones_(self.to_gamma_k.weight); nn.init.zeros_(self.to_gamma_k.bias)
        nn.init.zeros_(self.to_beta_k.weight);  nn.init.zeros_(self.to_beta_k.bias)
        nn.init.ones_(self.to_gamma_v.weight); nn.init.zeros_(self.to_gamma_v.bias)
        nn.init.zeros_(self.to_beta_v.weight);  nn.init.zeros_(self.to_beta_v.bias)

    def forward(self, T: torch.Tensor):
        """
        T : (B, 4, 4)
        Returns gamma_k, beta_k, gamma_v, beta_v — each (B, embed_dim)
        """
        R   = T[:, :3, :3]
        t   = T[:, :3,  3]
        r6d = rot_to_6d(R)                # (B, 6)
        pose_vec = torch.cat([r6d, t], dim=-1)   # (B, 9)
        h = self.mlp(pose_vec)
        return (
            self.to_gamma_k(h),
            self.to_beta_k(h),
            self.to_gamma_v(h),
            self.to_beta_v(h),
        )


# ---------------------------------------------------------------------------
# Local geometry-aware cross-attention
# ---------------------------------------------------------------------------

class LocalGeoCrossAttention(nn.Module):
    """
    Single-scale local geometry-aware cross-attention block.

    Query  : view-1 features
    Key/Val: view-2 features (FiLM-modulated by pose)
    Window : local neighbourhood around projected view-1 position in view-2

    Parameters
    ----------
    dim         : int   channel dimension (Q, K, V all same)
    num_heads   : int   multi-head attention heads
    window_size : int   local window side length (e.g. 7)
    dropout     : float attention dropout
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        window_size: int = 7,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads   = num_heads
        self.head_dim    = dim // num_heads
        self.window_size = window_size
        self.scale       = self.head_dim ** -0.5

        self.to_q = nn.Conv2d(dim, dim, 1, bias=False)
        self.to_k = nn.Conv2d(dim, dim, 1, bias=False)
        self.to_v = nn.Conv2d(dim, dim, 1, bias=False)
        self.out  = nn.Conv2d(dim, dim, 1)

        self.attn_drop = nn.Dropout(dropout)
        self.norm_q    = nn.GroupNorm(1, dim)   # LayerNorm-equivalent over channels
        self.norm_kv   = nn.GroupNorm(1, dim)
        self.norm_out  = nn.GroupNorm(1, dim)

        self.pose_enc  = PoseEncoder(embed_dim=dim)

    def _film_modulate(
        self,
        feat: torch.Tensor,
        gamma: torch.Tensor,
        beta: torch.Tensor,
    ) -> torch.Tensor:
        """Apply FiLM: gamma * feat + beta. gamma/beta are (B, C) → (B, C, 1, 1)."""
        g = gamma.view(-1, gamma.shape[1], 1, 1)
        b = beta.view(-1, beta.shape[1], 1, 1)
        return g * feat + b

    def forward(
        self,
        feat1: torch.Tensor,
        feat2: torch.Tensor,
        depth1: torch.Tensor,
        T_12: torch.Tensor,
        K: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        feat1  : (B, C, H, W)   view-1 features (query source)
        feat2  : (B, C, H, W)   view-2 features (key/value source)
        depth1 : (B, 1, Hd, Wd) monocular depth of view 1 (may be different res)
        T_12   : (B, 4, 4)      cam1→cam2 rigid transform
        K      : (B, 3, 3)      camera intrinsics (scaled to feat resolution)

        Returns
        -------
        out : (B, C, H, W)  cross-attended features for view 1
        """
        B, C, H, W = feat1.shape
        ws = self.window_size
        pad = ws // 2

        # ── Depth to feature resolution ─────────────────────────────────
        if depth1.shape[-2:] != feat1.shape[-2:]:
            depth1 = F.interpolate(depth1, size=(H, W), mode="nearest")

        # Intrinsics must be scaled to feature map resolution
        # (caller is responsible for passing correctly scaled K)

        # ── Pose FiLM ───────────────────────────────────────────────────
        gk, bk, gv, bv = self.pose_enc(T_12)

        # ── Projections ─────────────────────────────────────────────────
        Q = self.to_q(self.norm_q(feat1))               # (B, C, H, W)
        K_ = self.to_k(self.norm_kv(feat2))             # (B, C, H, W)
        V_ = self.to_v(self.norm_kv(feat2))             # (B, C, H, W)

        K_ = self._film_modulate(K_, gk, bk)
        V_ = self._film_modulate(V_, gv, bv)

        # ── Reprojection → centre of local window ───────────────────────
        coords, valid = reprojection_coords(depth1, T_12, K)  # (B,H,W,2), (B,1,H,W)
        # coords: pixel (x,y) in view-2 feature space

        # ── Pad K_ and V_ so we can extract windows near borders ────────
        K_pad = F.pad(K_, (pad, pad, pad, pad), mode="replicate")  # (B,C,H+ws-1,W+ws-1)
        V_pad = F.pad(V_, (pad, pad, pad, pad), mode="replicate")

        # ── Reshape Q, K_, V_ for multi-head attention ──────────────────
        def to_heads(t):
            # (B, C, H, W) → (B*nh, hd, H, W)
            t = t.reshape(B, self.num_heads, self.head_dim, H, W)
            return t.reshape(B * self.num_heads, self.head_dim, H, W)

        Q_h = to_heads(Q)    # (B*nh, hd, H, W)

        # ── Build local key/value windows ────────────────────────────────
        # For each query pixel (h, w), collect a (ws×ws) window from K_pad / V_pad
        # centred at the reprojected coordinate.
        # We use unfold to extract all windows, then select the right one.

        # Clamp centre coords to valid index range in padded map
        cx = coords[..., 0].clamp(0, W - 1).long()    # (B, H, W)
        cy = coords[..., 1].clamp(0, H - 1).long()    # (B, H, W)

        # Offset by pad (due to padding) so index into padded tensor
        cx_p = cx + pad    # (B, H, W)
        cy_p = cy + pad    # (B, H, W)

        # Use unfold to get all (ws×ws) windows over the padded feature maps
        # K_pad: (B, C, H+ws-1, W+ws-1)
        # after unfold(2, ws, 1) → (B, C, H, ws, W+ws-1) — unfold rows
        # after unfold(4, ws, 1) → (B, C, H, ws, W, ws)
        Kw = K_pad.unfold(2, ws, 1).unfold(3, ws, 1)  # (B, C, H, W, ws, ws)
        Vw = V_pad.unfold(2, ws, 1).unfold(3, ws, 1)  # (B, C, H, W, ws, ws)
        # Now Kw[:, :, h, w] is the ws×ws neighbourhood around (h, w) in K_pad
        # but we want the window centred at the *reprojected* position, not (h,w).

        # Gather the ws×ws window centred at (cy_p, cx_p) for each query pixel.
        # Strategy: extract a per-pixel ws×ws crop by shifting the index.
        # For memory efficiency: use unfold on padded tensor with the centre offset.

        # Alternative (simpler, slightly approximate for non-integer centres):
        # sample K_ at a regular ws×ws grid around the reprojected centre.

        # Build sampling grid: for each pixel (h,w), sample ws×ws positions
        # around (cx[h,w], cy[h,w]) in feat2.
        offsets = torch.arange(-(ws // 2), ws // 2 + 1, device=feat1.device)
        off_y, off_x = torch.meshgrid(offsets, offsets, indexing="ij")   # (ws, ws)
        off_x = off_x.reshape(1, 1, 1, ws * ws)   # (1, 1, 1, ws²)
        off_y = off_y.reshape(1, 1, 1, ws * ws)

        # sample_x: (B, H, W, ws²)
        sample_x = cx.unsqueeze(-1).float() + off_x
        sample_y = cy.unsqueeze(-1).float() + off_y

        # Normalise to [-1, 1] for grid_sample
        sample_x_n = 2.0 * sample_x / (W - 1) - 1.0
        sample_y_n = 2.0 * sample_y / (H - 1) - 1.0
        grid = torch.stack([sample_x_n, sample_y_n], dim=-1)  # (B, H, W, ws², 2)

        # grid_sample expects (B, C, H_out, W_out) input and (B, H_out, W_out, 2) grid
        # Reshape: treat each query pixel as a separate "image row", ws² columns
        grid_flat = grid.reshape(B, H * W, ws * ws, 2)         # (B, H*W, ws², 2)

        K_flat = K_.reshape(B, C, 1, H * W).expand(-1, -1, 1, -1)  # not efficient
        # Better: reshape to (B, C, H*W, ws²) via grid_sample on 2D feature
        # Reshape feat2 for group-wise sampling:
        # grid_sample input: (B, C, H, W), grid: (B, H*W, ws², 2)
        Ks = F.grid_sample(
            K_, grid_flat.reshape(B, H * W * ws * ws, 1, 2)
                          .reshape(B, H * W, ws * ws, 2),
            mode="bilinear", padding_mode="zeros", align_corners=True,
        )
        # Doesn't work directly — reshape correctly:
        Ks = F.grid_sample(
            K_,
            grid.reshape(B, H * W, ws * ws, 2),
            mode="bilinear", padding_mode="zeros", align_corners=True,
        )
        # grid_sample with (B, H*W, ws², 2) grid samples into (B, C, H, W) feat
        # → output: (B, C, H*W, ws²)
        Ks = Ks  # (B, C, H*W, ws²)  ← already correct from grid_sample above

        Vs = F.grid_sample(
            V_,
            grid.reshape(B, H * W, ws * ws, 2),
            mode="bilinear", padding_mode="zeros", align_corners=True,
        )  # (B, C, H*W, ws²)

        # ── Multi-head attention ─────────────────────────────────────────
        # Q: (B, C, H, W) → (B, H*W, nh, hd) → (B*nh, H*W, hd)
        Q_r = Q.permute(0, 2, 3, 1).reshape(B, H * W, self.num_heads, self.head_dim)
        Q_r = Q_r.permute(0, 2, 1, 3).reshape(B * self.num_heads, H * W, self.head_dim)

        # Ks: (B, C, H*W, ws²) → (B, H*W, ws², nh, hd) → (B*nh, H*W, ws², hd)
        Ks_r = Ks.permute(0, 2, 3, 1).reshape(B, H * W, ws * ws, self.num_heads, self.head_dim)
        Ks_r = Ks_r.permute(0, 3, 1, 2, 4).reshape(B * self.num_heads, H * W, ws * ws, self.head_dim)

        Vs_r = Vs.permute(0, 2, 3, 1).reshape(B, H * W, ws * ws, self.num_heads, self.head_dim)
        Vs_r = Vs_r.permute(0, 3, 1, 2, 4).reshape(B * self.num_heads, H * W, ws * ws, self.head_dim)

        # Attention: Q (B*nh, HW, hd) × Ks (B*nh, HW, ws², hd)ᵀ → (B*nh, HW, ws²)
        attn = torch.einsum("bnd,bnsd->bns", Q_r, Ks_r) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # Aggregate: (B*nh, HW, ws²) × Vs (B*nh, HW, ws², hd) → (B*nh, HW, hd)
        out = torch.einsum("bns,bnsd->bnd", attn, Vs_r)

        # Reshape back: (B*nh, HW, hd) → (B, C, H, W)
        out = out.reshape(B, self.num_heads, H * W, self.head_dim)
        out = out.permute(0, 2, 1, 3).reshape(B, H * W, C)
        out = out.reshape(B, H, W, C).permute(0, 3, 1, 2)   # (B, C, H, W)

        out = self.out(self.norm_out(out))
        return feat1 + out   # residual connection
