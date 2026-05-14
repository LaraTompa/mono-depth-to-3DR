"""
losses.py — All training losses for DepthAlignNet.

Active
------
1. si_log_loss           — Scale-invariant log loss (supervised depth)
2. smooth_loss           — Edge-aware depth smoothness
3. iter_supervision      — Deep supervision over refinement iterations
4. compute_depth_metrics — abs_rel / rmse / delta1 (validation only)
5. total_loss            — Weighted sum with breakdown dict for logging

Commented out (re-enable once network produces reasonable depths)
-----------------------------------------------------------------
- photometric_loss    — SSIM + L1 with differentiable warping
- depth_consistency   — Warp predicted depth across views, compare
"""

import torch
import torch.nn.functional as F


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
# 2. Photometric loss  (SSIM + L1 with differentiable warp)  — commented out
# Re-enable once the network produces reasonable initial depths.
# ---------------------------------------------------------------------------

# class SSIMLoss(nn.Module): ...
# def photometric_loss(...): ...


# ---------------------------------------------------------------------------
# 3. Depth consistency loss  — commented out
# Re-enable alongside photometric_loss after initial convergence.
# ---------------------------------------------------------------------------

# def depth_consistency_loss(...): ...


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
# 5. Deep supervision over iterative depth estimates
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
    if n == 0:
        return torch.tensor(0.0, device=gt_depth.device, requires_grad=False)
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
# 6. Total loss — called from train.py compute_loss
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

    B, N, _, H, W = imgs.shape

    rgb1 = imgs[:, 0]            # (B, 3, H, W)
    rgb2 = imgs[:, 1]
    gt1  = depths[:, 0]         # (B, 1, H, W)
    gt2  = depths[:, 1]

    pred1 = outputs["depth1"]   # (B, 1, H/2, W/2)
    pred2 = outputs["depth2"]

    # Scale GT to predicted resolution for supervised loss
    pH, pW = pred1.shape[-2:]
    gt1_s = F.interpolate(gt1, size=(pH, pW), mode="nearest")
    gt2_s = F.interpolate(gt2, size=(pH, pW), mode="nearest")

    # Scale images to predicted resolution for smoothness
    rgb1_s = F.interpolate(rgb1, size=(pH, pW), mode="bilinear", align_corners=True)
    rgb2_s = F.interpolate(rgb2, size=(pH, pW), mode="bilinear", align_corners=True)

    w = weights

    # --- Supervised depth ---
    l_depth = (
        si_log_loss(pred1, gt1_s) +
        si_log_loss(pred2, gt2_s)
    ) * 0.5

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
        float(w.get("depth",            1.0)) * l_depth  +
        float(w.get("smooth",           0.05)) * l_smooth +
        float(w.get("iter_supervision", 0.5)) * l_iters
    )

    parts = {
        "depth":  float(l_depth.detach()),
        "smooth": float(l_smooth.detach()),
        "iters":  float(l_iters.detach()),
    }
    return total, parts


# ---------------------------------------------------------------------------
# 7. Depth accuracy metrics  (validation only, no gradients needed)
# ---------------------------------------------------------------------------

def compute_depth_metrics(
    pred: torch.Tensor,   # (B, 1, H, W)
    gt:   torch.Tensor,   # (B, 1, H, W)
    min_depth: float = 1e-3,
    max_depth: float = 10.0,
) -> dict:
    """
    Standard depth evaluation metrics.
    pred and gt must be at the same resolution (interpolate before calling).

    Returns dict with abs_rel, rmse, delta1 (threshold 1.25).
    """
    mask = (gt > min_depth) & (gt < max_depth) & torch.isfinite(gt) & (pred > EPS)
    if mask.sum() == 0:
        return {"abs_rel": float("nan"), "rmse": float("nan"), "delta1": float("nan")}

    p = pred[mask]
    g = gt[mask]

    abs_rel = ((p - g).abs() / g).mean().item()
    rmse    = ((p - g) ** 2).mean().sqrt().item()
    ratio   = torch.max(p / g, g / p)
    delta1  = (ratio < 1.25).float().mean().item()

    return {"abs_rel": abs_rel, "rmse": rmse, "delta1": delta1}
