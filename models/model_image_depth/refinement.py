"""
refinement.py — Iterative residual geometric refinement.

Design (inspired by RAFT / deep iterative stereo)
--------------------------------------------------
Each iteration:
  1. Warp view-2 depth into view-1 using the *current* depth estimate.
  2. Concatenate: current depth estimate, warp residual, correlation features,
     cross-attended features.
  3. A ConvGRU cell updates a hidden state.
  4. A small head predicts (delta_s, delta_b, delta_D, log_sigma):
       D_next = s * D_mono + b + delta_D    (residual correction on the prior)

The global scale *s* and bias *b* start at 1 and 0 respectively and
accumulate across iterations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .geometry import warp


# ---------------------------------------------------------------------------
# Correlation feature  (current_depth ↔ warped_depth)
# ---------------------------------------------------------------------------

def correlation_feature(
    depth_cur: torch.Tensor,
    depth_warped: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    """
    Compute a 2-channel correlation feature from two aligned depth maps.

    Returns (B, 2, H, W):
      ch0 = signed difference  (cur - warped), zeroed where invalid
      ch1 = binary validity mask (float)
    """
    diff  = (depth_cur - depth_warped) * valid.float()
    return torch.cat([diff, valid.float()], dim=1)   # (B, 2, H, W)


# ---------------------------------------------------------------------------
# Lightweight ConvGRU cell
# ---------------------------------------------------------------------------

class ConvGRU(nn.Module):
    """
    Spatial ConvGRU (Cho et al.) operating on 2-D feature maps.

    Parameters
    ----------
    hidden_dim : int   number of channels in the hidden state
    input_dim  : int   number of channels in the input
    kernel_size: int   convolution kernel size
    """

    def __init__(self, hidden_dim: int, input_dim: int, kernel_size: int = 3):
        super().__init__()
        pad = kernel_size // 2
        self.reset_gate  = nn.Conv2d(hidden_dim + input_dim, hidden_dim, kernel_size, padding=pad)
        self.update_gate = nn.Conv2d(hidden_dim + input_dim, hidden_dim, kernel_size, padding=pad)
        self.new_gate    = nn.Conv2d(hidden_dim + input_dim, hidden_dim, kernel_size, padding=pad)

    def forward(self, h: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        h : (B, hidden_dim, H, W)
        x : (B, input_dim,  H, W)
        Returns updated hidden state (B, hidden_dim, H, W).
        """
        hx = torch.cat([h, x], dim=1)
        r = torch.sigmoid(self.reset_gate(hx))
        z = torch.sigmoid(self.update_gate(hx))
        n = torch.tanh(self.new_gate(torch.cat([r * h, x], dim=1)))
        return (1 - z) * h + z * n


# ---------------------------------------------------------------------------
# Per-iteration prediction head
# ---------------------------------------------------------------------------

class RefinementHead(nn.Module):
    """
    Small head predicting per-pixel residual corrections.

    Outputs per pixel:
      delta_s   : global-like scale correction   (1 ch, scalar tendency)
      delta_b   : global-like bias  correction   (1 ch, scalar tendency)
      delta_D   : local residual depth            (1 ch)
      log_sigma : log confidence                  (1 ch)
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim // 2, 3, padding=1),
            nn.GELU(),
        )
        # Each output is 1-channel; final activation applied in forward
        self.head_s     = nn.Conv2d(hidden_dim // 2, 1, 1)
        self.head_b     = nn.Conv2d(hidden_dim // 2, 1, 1)
        self.head_dD    = nn.Conv2d(hidden_dim // 2, 1, 1)
        self.head_sigma = nn.Conv2d(hidden_dim // 2, 1, 1)

        # Init final layers near zero so early iterations are small updates
        for head in [self.head_s, self.head_b, self.head_dD, self.head_sigma]:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(self, h: torch.Tensor):
        """h : (B, hidden_dim, H, W)"""
        feat = self.conv(h)
        delta_s    = torch.tanh(self.head_s(feat))      # ∈ (-1, 1)
        delta_b    = self.head_b(feat)                  # unbounded
        delta_D    = self.head_dD(feat)                 # unbounded residual
        log_sigma  = self.head_sigma(feat)              # log confidence
        return delta_s, delta_b, delta_D, log_sigma


# ---------------------------------------------------------------------------
# Full iterative refinement module
# ---------------------------------------------------------------------------

class IterativeRefinement(nn.Module):
    """
    Iterative geometric refinement over N iterations.

    At each iteration it predicts residual corrections to the depth prior.
    The depth estimate is updated and used to re-warp view-2 features for
    the next iteration (geometric re-anchoring).

    Parameters
    ----------
    feat_dim    : int   cross-attention feature dimension
    hidden_dim  : int   ConvGRU hidden state channels
    num_iters   : int   number of refinement iterations
    """

    def __init__(
        self,
        feat_dim: int   = 128,
        hidden_dim: int = 128,
        num_iters: int  = 4,
    ):
        super().__init__()
        self.num_iters = num_iters

        # Input to ConvGRU:
        #   cross-attended feat (feat_dim) +
        #   correlation (2) +
        #   current depth estimate (1) +
        #   mono prior (1)
        input_dim = feat_dim + 2 + 1 + 1

        # Initial hidden state projection
        self.h_init = nn.Conv2d(feat_dim, hidden_dim, 1)

        self.gru    = ConvGRU(hidden_dim=hidden_dim, input_dim=input_dim)
        self.head   = RefinementHead(hidden_dim)

    def forward(
        self,
        depth_mono: torch.Tensor,
        depth2_mono: torch.Tensor,
        feat_cross: torch.Tensor,
        feat2: torch.Tensor,
        T_12: torch.Tensor,
        K: torch.Tensor,
    ) -> dict:
        """
        Parameters
        ----------
        depth_mono  : (B, 1, H, W)  monocular depth prior, view 1
        depth2_mono : (B, 1, H, W)  monocular depth prior, view 2 (for warping)
        feat_cross  : (B, C, H, W)  cross-attended feature (view1 query, view2 context)
        feat2       : (B, C, H, W)  raw view-2 feature (for per-iter warping)
        T_12        : (B, 4, 4)     cam1 → cam2
        K           : (B, 3, 3)     intrinsics (at refinement resolution)

        Returns
        -------
        dict with:
          "depth"       : (B, 1, H, W)  final aligned depth
          "confidence"  : (B, 1, H, W)  exp(-log_sigma)  ∈ (0, 1]
          "depth_iters" : list of (B,1,H,W)  one per iteration  (for loss supervision)
          "scale"       : (B,)  accumulated global scale
          "bias"        : (B,)  accumulated global bias
        """
        B, _, H, W = depth_mono.shape
        device = depth_mono.device

        # Initialise hidden state from cross-attended features
        h = torch.tanh(self.h_init(feat_cross))   # (B, hidden_dim, H, W)

        # Running scale and bias (global, per sample in batch)
        s = torch.ones(B, 1, 1, 1, device=device)   # start at s=1
        b = torch.zeros(B, 1, 1, 1, device=device)  # start at b=0

        depth_cur = s * depth_mono + b               # initialise = mono prior
        depth_iters = []

        for _ in range(self.num_iters):
            # Re-warp view-2 depth with current estimate
            depth2_warped, valid = warp(depth2_mono, depth_cur, T_12, K)

            # Correlation between current depth and warped view-2 depth
            corr = correlation_feature(depth_cur, depth2_warped, valid)  # (B, 2, H, W)

            # Assemble GRU input
            inp = torch.cat([feat_cross, corr, depth_cur, depth_mono], dim=1)

            # GRU step
            h = self.gru(h, inp)

            # Predict residuals
            delta_s, delta_b, delta_D, log_sigma = self.head(h)

            # Global corrections: average pooled scale/bias
            ds = delta_s.mean(dim=[2, 3], keepdim=True)   # (B, 1, 1, 1)
            db = delta_b.mean(dim=[2, 3], keepdim=True)

            s = s + ds
            b = b + db

            # Updated aligned depth
            depth_cur = s * depth_mono + b + delta_D
            depth_cur = depth_cur.clamp(min=1e-3)

            depth_iters.append(depth_cur)

        confidence = torch.exp(-log_sigma)

        return {
            "depth":       depth_cur,
            "confidence":  confidence,
            "depth_iters": depth_iters,
            "scale":       s.squeeze(-1).squeeze(-1),   # (B, 1)
            "bias":        b.squeeze(-1).squeeze(-1),   # (B, 1)
        }
