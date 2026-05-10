"""
losses.py — All training losses for DepthAlignNet.

Losses
------
1. si_log_loss         — Scale-invariant log loss (supervised depth)
2. photometric_loss    — SSIM + L1 with differentiable warping
3. depth_consistency   — Warp predicted depth across views, compare
4. smooth_loss         — Edge-aware depth smoothness
5. feature_reprojection_loss  — Warped encoder feature consistency (optional)
6. total_loss          — Weighted sum with breakdown dict for logging
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.geometry import warp


EPS = 1e-8


# ---------------------------------------------------------------------------
# 1. Scale-invariant log loss  (Eigen et al., 2014)
# ---------------------------------------------------------------------------

def si_log_loss(
    pred: torch.Tensor,
    gt:   torch.Tensor,
    min_depth: float = 1e-3,
    max_depth: float = 80.0,
    lam: float = 0.5,
) -> torch.Tensor:
    """
    Scale-invariant logarithmic depth loss.

    pred, gt : (B, 1, H, W)
    lam      : variance-minimisation weight (0.5 per Eigen et al.)
    """
    mask = (gt > min_depth) & (gt < max_depth) & torch.isfinite(gt) & (pred > EPS)
    if mask.sum() == 0:
        return pred.new_tensor(0.0, requires_grad=True)

    log_diff = torch.log(pred[mask] + EPS) - torch.log(gt[mask] + EPS)
    n = log_diff.numel()
    loss = (log_diff ** 2).mean() - lam * (log_diff.sum() ** 2) / (n ** 2)
    return loss


# ---------------------------------------------------------------------------
# 2. Photometric loss  (SSIM + L1 with differentiable warp)
# ---------------------------------------------------------------------------

class SSIMLoss(nn.Module):
    """Differentiable SSIM (local, window-based)."""

    def __init__(self, window_size: int = 11, sigma: float = 1.5):
        super().__init__()
        self.window_size = window_size
        self.register_buffer("kernel", self._gaussian_kernel(window_size, sigma))

    @staticmethod
    def _gaussian_kernel(size: int, sigma: float) -> torch.Tensor:
        coords = torch.arange(size).float() - size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g /= g.sum()
        kernel = g.unsqueeze(0) * g.unsqueeze(1)         # (size, size)
        return kernel.reshape(1, 1, size, size)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """x, y : (B, C, H, W) ∈ [0, 1]"""
        C = x.shape[1]
        k = self.kernel.expand(C, 1, -1, -1).to(x.device)
        pad = self.window_size // 2

        def conv(t):
            return F.conv2d(t, k, padding=pad, groups=C)

        mu_x  = conv(x);  mu_y  = conv(y)
        mu_xx = conv(x * x);  mu_yy = conv(y * y);  mu_xy = conv(x * y)
        sig_x  = mu_xx - mu_x * mu_x
        sig_y  = mu_yy - mu_y * mu_y
        sig_xy = mu_xy - mu_x * mu_y

        C1 = 0.01 ** 2;  C2 = 0.03 ** 2
        num   = (2 * mu_x * mu_y + C1) * (2 * sig_xy + C2)
        denom = (mu_x ** 2 + mu_y ** 2 + C1) * (sig_x + sig_y + C2)
        ssim_map = num / (denom + EPS)

        return 1.0 - ssim_map.mean()   # loss ≥ 0


_ssim = SSIMLoss()


def photometric_loss(
    img_src:    torch.Tensor,   # (B, 3, H, W) view 1 image
    img_tgt:    torch.Tensor,   # (B, 3, H, W) view 2 image
    depth_src:  torch.Tensor,   # (B, 1, H, W) aligned depth of view 1
    T_12:       torch.Tensor,   # (B, 4, 4) cam1→cam2
    K:          torch.Tensor,   # (B, 3, 3) intrinsics scaled to img resolution
    alpha:      float = 0.85,   # SSIM weight
) -> torch.Tensor:
    """
    Warp img_tgt into img_src frame using depth_src and compute SSIM+L1.
    """
    img_warped, valid = warp(img_tgt, depth_src, T_12, K)   # (B,3,H,W), (B,1,H,W)
    if valid.sum() < 100:
        return img_src.new_tensor(0.0, requires_grad=True)

    # Mask to valid pixels
    v = valid.expand_as(img_src)
    ssim_val = _ssim(img_src * v, img_warped * v)
    l1_val   = (img_src - img_warped).abs()[v].mean()

    return alpha * ssim_val + (1.0 - alpha) * l1_val


# ---------------------------------------------------------------------------
# 3. Depth consistency loss
# ---------------------------------------------------------------------------

def depth_consistency_loss(
    depth1:  torch.Tensor,   # (B, 1, H, W) aligned depth view 1
    depth2:  torch.Tensor,   # (B, 1, H, W) aligned depth view 2
    T_12:    torch.Tensor,   # (B, 4, 4)
    K:       torch.Tensor,   # (B, 3, 3) scaled to depth resolution
    min_depth: float = 1e-3,
) -> torch.Tensor:
    """
    Warp depth2 into view-1 frame and compare with depth1 at valid overlapping pixels.
    """
    depth2_warped, valid = warp(depth2, depth1, T_12, K)

    valid = valid & (depth1 > min_depth) & (depth2_warped > min_depth)
    if valid.sum() < 100:
        return depth1.new_tensor(0.0, requires_grad=True)

    # L1 on log-depth for scale robustness
    log_d1 = torch.log(depth1 + EPS)
    log_d2 = torch.log(depth2_warped + EPS)
    return (log_d1 - log_d2).abs()[valid].mean()


# ---------------------------------------------------------------------------
# 4. Edge-aware smoothness
# ---------------------------------------------------------------------------

def smooth_loss(
    depth: torch.Tensor,    # (B, 1, H, W)
    image: torch.Tensor,    # (B, 3, H, W)  used to suppress smoothing at edges
) -> torch.Tensor:
    """
    Penalise depth gradient magnitude, down-weighted at image edges.
    Normalise depth by mean to handle scale ambiguity.
    """
    mean_d = depth.mean(dim=[2, 3], keepdim=True).clamp(min=EPS)
    d_norm = depth / mean_d

    dx_d = (d_norm[:, :, :, 1:] - d_norm[:, :, :, :-1]).abs()
    dy_d = (d_norm[:, :, 1:, :] - d_norm[:, :, :-1, :]).abs()

    dx_i = image[:, :, :, 1:] - image[:, :, :, :-1]
    dy_i = image[:, :, 1:, :] - image[:, :, :-1, :]
    dx_w = torch.exp(-dx_i.abs().mean(dim=1, keepdim=True))
    dy_w = torch.exp(-dy_i.abs().mean(dim=1, keepdim=True))

    return (dx_w * dx_d).mean() + (dy_w * dy_d).mean()


# ---------------------------------------------------------------------------
# 5. Feature reprojection consistency  (optional, more stable than RGB)
# ---------------------------------------------------------------------------

def feature_reprojection_loss(
    feat1:     torch.Tensor,   # (B, C, H, W) encoder features view 1
    feat2:     torch.Tensor,   # (B, C, H, W) encoder features view 2
    depth1:    torch.Tensor,   # (B, 1, H, W) aligned depth view 1
    T_12:      torch.Tensor,   # (B, 4, 4)
    K:         torch.Tensor,   # (B, 3, 3) scaled to feat resolution
) -> torch.Tensor:
    feat2_warped, valid = warp(feat2, depth1, T_12, K)
    if valid.sum() < 100:
        return feat1.new_tensor(0.0, requires_grad=True)

    v = valid.expand_as(feat1)
    return F.l1_loss(feat1[v], feat2_warped[v])


# ---------------------------------------------------------------------------
# 6. Deep supervision over iterative depth estimates
# ---------------------------------------------------------------------------

def iter_supervision_loss(
    depth_iters: list,          # list of (B, 1, H, W)
    gt_depth:    torch.Tensor,  # (B, 1, H, W) ground truth
    weights:     list | None = None,
    min_depth:   float = 1e-3,
    max_depth:   float = 80.0,
) -> torch.Tensor:
    """
    Weighted si-log loss over each refinement iteration.
    Later iterations get higher weight.
    """
    n = len(depth_iters)
    if weights is None:
        # exponential increase: [1, 2, 4, 8] → normalised
        w = [2 ** i for i in range(n)]
        total = sum(w)
        weights = [wi / total for wi in w]

    loss = depth_iters[0].new_tensor(0.0)
    for pred_i, wi in zip(depth_iters, weights):
        # resize gt to match iter depth resolution
        gt_i = F.interpolate(gt_depth, size=pred_i.shape[-2:], mode="nearest")
        loss = loss + wi * si_log_loss(pred_i, gt_i, min_depth, max_depth)
    return loss


# ---------------------------------------------------------------------------
# 7. Total loss — called from train.py compute_loss
# ---------------------------------------------------------------------------

def total_loss(
    outputs:  dict,             # from DepthAlignNet.forward()
    batch:    dict,             # from DataLoader
    weights:  dict,             # from cfg["loss"]
    K:        torch.Tensor,     # (B, 3, 3) full-resolution intrinsics
) -> tuple[torch.Tensor, dict]:
    """
    Compute the weighted total training loss.

    Expects batch to contain:
      "images"  : (B, N, 3, H, W)
      "depths"  : (B, N, 1, H, W)   GT depth in metres
      "poses"   : (B, N, 4, 4)      cam-to-world poses
    and outputs from DepthAlignNet.

    Uses views 0 and 1 from each sequence for the two-view losses.
    Remaining views contribute additional photometric/consistency terms.

    Returns
    -------
    total : scalar tensor
    parts : dict of scalar floats for logging
    """
    imgs   = batch["images"]    # (B, N, 3, H, W)
    depths = batch["depths"]    # (B, N, 1, H, W)
    poses  = batch["poses"]     # (B, N, 4, 4)

    B, N, _, H, W = imgs.shape

    rgb1 = imgs[:, 0]            # (B, 3, H, W)
    rgb2 = imgs[:, 1]
    gt1  = depths[:, 0]         # (B, 1, H, W)
    gt2  = depths[:, 1]

    # T_12 from world poses: T_12 = inv(pose2) @ pose1
    # pose is cam-to-world, so world-to-cam is inv(pose)
    pose1_cw = torch.linalg.inv(poses[:, 0])   # world→cam1
    pose2_cw = torch.linalg.inv(poses[:, 1])   # world→cam2
    T_12 = pose2_cw @ poses[:, 0]              # cam1→cam2

    pred1 = outputs["depth1"]   # (B, 1, H/2, W/2)
    pred2 = outputs["depth2"]

    # Scale intrinsics to predicted depth resolution
    pH, pW = pred1.shape[-2:]
    K_pred = K.clone()
    K_pred[:, 0, 0] *= pW / W;  K_pred[:, 0, 2] *= pW / W
    K_pred[:, 1, 1] *= pH / H;  K_pred[:, 1, 2] *= pH / H

    # Scale GT to predicted resolution for supervised loss
    gt1_s = F.interpolate(gt1, size=(pH, pW), mode="nearest")
    gt2_s = F.interpolate(gt2, size=(pH, pW), mode="nearest")

    # Scale images to predicted resolution for photometric loss
    rgb1_s = F.interpolate(rgb1, size=(pH, pW), mode="bilinear", align_corners=True)
    rgb2_s = F.interpolate(rgb2, size=(pH, pW), mode="bilinear", align_corners=True)

    w = weights

    # --- Supervised depth ---
    l_depth = (
        si_log_loss(pred1, gt1_s) +
        si_log_loss(pred2, gt2_s)
    ) * 0.5

    # --- Photometric ---
    l_photo = (
        photometric_loss(rgb1_s, rgb2_s, pred1, T_12, K_pred) +
        photometric_loss(rgb2_s, rgb1_s, pred2, torch.linalg.inv(T_12), K_pred)
    ) * 0.5

    # --- Depth consistency ---
    l_consist = depth_consistency_loss(pred1, pred2, T_12, K_pred)

    # --- Smoothness ---
    l_smooth = (
        smooth_loss(pred1, rgb1_s) +
        smooth_loss(pred2, rgb2_s)
    ) * 0.5

    # --- Deep supervision over refinement iters ---
    l_iters = (
        iter_supervision_loss(outputs["depth1_iters"], gt1) +
        iter_supervision_loss(outputs["depth2_iters"], gt2)
    ) * 0.5

    total = (
        float(w.get("depth",       1.0)) * l_depth   +
        float(w.get("photometric", 0.1)) * l_photo   +
        float(w.get("consistency", 0.1)) * l_consist +
        float(w.get("smooth",      0.05)) * l_smooth +
        float(w.get("iter_supervision", 0.5)) * l_iters
    )

    parts = {
        "depth":       float(l_depth.detach()),
        "photo":       float(l_photo.detach()),
        "consistency": float(l_consist.detach()),
        "smooth":      float(l_smooth.detach()),
        "iters":       float(l_iters.detach()),
    }
    return total, parts
