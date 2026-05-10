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

from .geometry import reprojection_coords, rot_to_6d, unproject, transform_pts


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
        depth2: torch.Tensor | None = None,
        occlusion_thresh: float = 0.1,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        feat1            : (B, C, H, W)   view-1 features  (query source)
        feat2            : (B, C, H, W)   view-2 features  (key/value source)
        depth1           : (B, 1, H, W)   monocular depth of view 1
        T_12             : (B, 4, 4)      cam1->cam2 rigid transform
        K                : (B, 3, 3)      intrinsics scaled to this feature resolution
        depth2           : (B, 1, H, W)   optional view-2 depth for occlusion test
        occlusion_thresh : float          relative depth tolerance (default 0.1 = 10%)

        Returns
        -------
        (B, C, H, W)  feat1 + cross-attended update
        """
        B, C, H, W = feat1.shape
        ws = self.window_size
        nh, hd = self.num_heads, self.head_dim

        # depth to feature resolution if needed
        if depth1.shape[-2:] != (H, W):
            depth1 = F.interpolate(depth1, size=(H, W), mode="nearest")

        # ----------------------------------------------------------------
        # Step 1: geometry projection  p_i -> p_i'
        # coords : (B, H, W, 2)  pixel (u2, v2) in view-2 feature space
        # valid  : (B, 1, H, W)  1 where projection lands inside view-2 image
        # ----------------------------------------------------------------
        coords, valid = reprojection_coords(depth1, T_12, K)

        # projected Z for occlusion test — from the transform itself
        # unproject view-1 points then transform; Z is the view-2 depth
        pts_in_2 = transform_pts(unproject(depth1, K), T_12)  # (B, 3, H, W)
        proj_z = pts_in_2[:, 2:3]                             # (B, 1, H, W)

        # ----------------------------------------------------------------
        # Pose FiLM + linear projections
        # ----------------------------------------------------------------
        gk, bk, gv, bv = self.pose_enc(T_12)

        Q  = self.to_q(self.norm_q(feat1))                             # (B, C, H, W)
        K_ = self._film_modulate(self.to_k(self.norm_kv(feat2)), gk, bk)
        V_ = self._film_modulate(self.to_v(self.norm_kv(feat2)), gv, bv)

        # ----------------------------------------------------------------
        # Step 2: sample local ws x ws patch around (u2, v2)
        # One grid_sample call per tensor.
        # ----------------------------------------------------------------
        offsets = torch.arange(-(ws // 2), ws // 2 + 1,
                               dtype=torch.float32, device=feat1.device)
        off_y, off_x = torch.meshgrid(offsets, offsets, indexing="ij")  # (ws, ws)
        off_x = off_x.reshape(1, 1, 1, ws * ws)   # broadcast -> (B, H, W, ws*ws)
        off_y = off_y.reshape(1, 1, 1, ws * ws)

        # pixel coords of every patch sample: (B, H, W, ws*ws)
        px = coords[..., 0].unsqueeze(-1) + off_x
        py = coords[..., 1].unsqueeze(-1) + off_y

        # normalise to [-1, 1] for grid_sample
        px_n = 2.0 * px / max(W - 1, 1) - 1.0
        py_n = 2.0 * py / max(H - 1, 1) - 1.0

        # grid: (B, H*W, ws*ws, 2)
        grid = torch.stack([px_n, py_n], dim=-1).reshape(B, H * W, ws * ws, 2)

        # K_patch, V_patch: (B, C, H*W, ws*ws)
        K_patch = F.grid_sample(K_, grid, mode="bilinear",
                                padding_mode="zeros", align_corners=True)
        V_patch = F.grid_sample(V_, grid, mode="bilinear",
                                padding_mode="zeros", align_corners=True)

        # ----------------------------------------------------------------
        # Occlusion mask  (B, H*W, ws*ws) — True = keep
        #   Level 1: projection lands outside view-2 image  (valid mask)
        #   Level 2: view-2 depth < projected Z by >thresh  (occluded)
        # ----------------------------------------------------------------
        # Level 1: broadcast valid (B,1,H,W) -> (B, H*W, 1) -> (B, H*W, ws*ws)
        occ_mask = valid.reshape(B, 1, H * W).permute(0, 2, 1)  # (B, H*W, 1)
        occ_mask = occ_mask.expand(-1, -1, ws * ws)              # (B, H*W, ws*ws)

        if depth2 is not None:
            if depth2.shape[-2:] != (H, W):
                depth2 = F.interpolate(depth2, size=(H, W), mode="nearest")
            # sample view-2 depth at every patch location: (B, 1, H*W, ws*ws)
            d2_patch = F.grid_sample(depth2, grid, mode="bilinear",
                                     padding_mode="zeros", align_corners=True)
            # projected Z for the centre pixel, broadcast over ws*ws neighbours
            pz = proj_z.reshape(B, 1, H * W, 1).expand(-1, -1, -1, ws * ws)
            # pixel is NOT occluded if view-2 depth >= projected Z - thresh
            not_occluded = (d2_patch >= pz * (1.0 - occlusion_thresh))  # (B,1,H*W,ws*ws)
            occ_mask = occ_mask & not_occluded.squeeze(1)                # (B, H*W, ws*ws)

        # ----------------------------------------------------------------
        # Step 3: multi-head dot-product  a_ij = Q_i . K_j
        # ----------------------------------------------------------------
        # Q:       (B, C, H, W)       -> (B*nh, H*W, hd)
        Q_r = (Q.permute(0, 2, 3, 1)
                 .reshape(B, H * W, nh, hd)
                 .permute(0, 2, 1, 3)
                 .reshape(B * nh, H * W, hd))

        # K_patch: (B, C, H*W, ws*ws) -> (B*nh, H*W, ws*ws, hd)
        K_r = (K_patch.permute(0, 2, 3, 1)
                       .reshape(B, H * W, ws * ws, nh, hd)
                       .permute(0, 3, 1, 2, 4)
                       .reshape(B * nh, H * W, ws * ws, hd))
        V_r = (V_patch.permute(0, 2, 3, 1)
                       .reshape(B, H * W, ws * ws, nh, hd)
                       .permute(0, 3, 1, 2, 4)
                       .reshape(B * nh, H * W, ws * ws, hd))

        # (B*nh, H*W, ws*ws)
        attn = torch.einsum("bnd,bnsd->bns", Q_r, K_r) * self.scale

        # apply occlusion mask: set logits to -inf where occluded
        # occ_mask: (B, H*W, ws*ws) -> (B*nh, H*W, ws*ws)
        mask = occ_mask.unsqueeze(1).expand(-1, nh, -1, -1).reshape(B * nh, H * W, ws * ws)
        attn = attn.masked_fill(~mask, float("-inf"))
        # if ALL patch positions are masked (no valid neighbours), softmax -> nan
        # fall back to uniform attention over the full row in that case
        all_masked = (~mask).all(dim=-1, keepdim=True)   # (B*nh, H*W, 1)
        attn = torch.where(all_masked.expand_as(attn), torch.zeros_like(attn), attn)
        attn = self.attn_drop(attn.softmax(dim=-1))
        # where all positions were masked the softmax gives 0 -> output is 0 (no update)
        attn = attn.masked_fill(all_masked.expand_as(attn), 0.0)

        # ----------------------------------------------------------------
        # Step 4: aggregate  sum_j a_ij * V_j
        # ----------------------------------------------------------------
        # (B*nh, H*W, hd) -> (B, C, H, W)
        out = torch.einsum("bns,bnsd->bnd", attn, V_r)
        out = (out.reshape(B, nh, H * W, hd)
                  .permute(0, 2, 1, 3)
                  .reshape(B, H * W, C)
                  .reshape(B, H, W, C)
                  .permute(0, 3, 1, 2))

        return feat1 + self.out(self.norm_out(out))
