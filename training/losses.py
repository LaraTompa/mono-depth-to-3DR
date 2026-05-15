"""
losses.py — All training losses for DepthAlignNet.

Active
------
1. si_log_loss              — Scale-invariant log loss (supervised depth)
2. smooth_loss              — Edge-aware depth smoothness
3. iter_supervision         — Deep supervision over refinement iterations
4. geodesic_rotation_loss   — Geodesic distance on SO(3) between R_pred and R_gt
5. normalized_trans_loss    — L2 between unit-normalised predicted/GT translations
6. pose_identity_loss       — ||T_12 @ T_21 − I||_F  (numerical round-trip check)
7. camera_pose_loss         — Confidence-weighted composite camera loss
8. compute_depth_metrics    — abs_rel / rmse / delta1 (validation only)
9. total_loss               — Weighted sum with full breakdown dict for logging

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
# 6. Geodesic rotation loss
# ---------------------------------------------------------------------------

def geodesic_rotation_loss(
    R_pred: torch.Tensor,   # (B, 3, 3) predicted rotation
    R_gt:   torch.Tensor,   # (B, 3, 3) ground-truth rotation
) -> torch.Tensor:
    """
    Per-sample geodesic (great-circle) distance on SO(3):
        d(R_pred, R_gt) = arccos( (trace(R_pred^T R_gt) − 1) / 2 )

    Returns (B,) tensor of angles in radians.
    The trace formula gives the rotation angle of the relative rotation
    R_pred^T R_gt; the geodesic is the shortest arc on the unit sphere.
    """
    R_rel   = R_pred.transpose(-1, -2) @ R_gt              # (B, 3, 3)
    cos_ang = (R_rel.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0) * 0.5
    cos_ang = cos_ang.clamp(-1.0 + EPS, 1.0 - EPS)        # numerical safety
    return torch.acos(cos_ang)                              # (B,)


# ---------------------------------------------------------------------------
# 7. Normalised translation loss
# ---------------------------------------------------------------------------

def normalized_translation_loss(
    t_pred: torch.Tensor,   # (B, 3) predicted translation
    t_gt:   torch.Tensor,   # (B, 3) ground-truth translation
) -> torch.Tensor:
    """
    L2 distance between unit-normalised translations.

    Normalisation removes scale ambiguity: we care about the direction of
    the baseline, not its magnitude.  Zero-norm vectors (pure-rotation
    pairs) are handled by F.normalize's eps.

    Returns (B,) tensor.
    """
    t_pred_n = F.normalize(t_pred, dim=-1)
    t_gt_n   = F.normalize(t_gt,   dim=-1)
    return (t_pred_n - t_gt_n).norm(dim=-1)                # (B,)


# ---------------------------------------------------------------------------
# 8. Pose identity (round-trip consistency) loss
# ---------------------------------------------------------------------------

def pose_identity_loss(
    T_12: torch.Tensor,   # (B, 4, 4)
    T_21: torch.Tensor,   # (B, 4, 4)
) -> torch.Tensor:
    """
    Frobenius norm of T_12 @ T_21 − I_4.

    Analytically this is zero when T_21 = T_12^{-1}, but gradients through
    SVD orthogonalisation and torch.linalg.inv accumulate floating-point
    drift.  This loss acts as a soft numerical regulariser that explicitly
    re-enforces the round-trip identity constraint.

    Returns a scalar.
    """
    I4 = torch.eye(4, device=T_12.device, dtype=T_12.dtype).unsqueeze(0)
    return (T_12 @ T_21 - I4).norm(dim=(-2, -1)).mean()


# ---------------------------------------------------------------------------
# 9. Composite camera-pose loss with confidence weighting
# ---------------------------------------------------------------------------

def camera_pose_loss(
    outputs:  dict,
    T_12_gt:  torch.Tensor,          # (B, 4, 4) GT relative pose (cam1→cam2)
    K_gt:     torch.Tensor | None,   # (B, 3, 3) GT intrinsics, or None
    weights:  dict,
) -> tuple[torch.Tensor, dict]:
    """
    Confidence-weighted camera parameter losses.

    Pose losses (geodesic rotation + normalised translation) are combined
    and weighted by a per-sample heteroscedastic confidence score following
    Kendall & Gal (NeurIPS 2017):

        L_pose = exp(−s_pose) · (w_rot·L_rot + w_trans·L_trans) + s_pose

    Intrinsics regression (relative L1 vs GT) is similarly weighted:

        L_K = exp(−s_K) · L_K_data + s_K

    The "+s" terms prevent the network from collapsing the loss to zero by
    inflating uncertainty.  The pose identity loss is added unweighted as a
    geometric regulariser.

    Parameters
    ----------
    outputs  : DepthAlignNet output dict (must contain T_12_pred, T_c2w_1/2,
               K_pred, log_conf_K, log_conf_pose)
    T_12_gt  : (B, 4, 4) ground-truth relative pose
    K_gt     : (B, 3, 3) ground-truth intrinsics, or None (skips K loss)
    weights  : loss weight dict (keys: rot, trans, K_reg, identity, camera)

    Returns
    -------
    total_camera_loss : scalar tensor
    parts             : dict of float scalars for logging
    """
    T_12_pred    = outputs["T_12_pred"]     # (B, 4, 4)
    T_c2w_1      = outputs["T_c2w_1"]       # (B, 4, 4)
    T_c2w_2      = outputs["T_c2w_2"]       # (B, 4, 4)
    K_pred       = outputs["K_pred"]        # (B, 3, 3)
    s_pose       = outputs["log_conf_pose"] # (B,)
    s_K          = outputs["log_conf_K"]    # (B,)

    R_pred = T_12_pred[:, :3, :3]          # (B, 3, 3)
    t_pred = T_12_pred[:, :3,  3]          # (B, 3)
    R_gt   = T_12_gt[:, :3, :3]
    t_gt   = T_12_gt[:, :3,  3]

    # ── Geodesic rotation loss ────────────────────────────────────────────
    l_rot   = geodesic_rotation_loss(R_pred, R_gt)    # (B,)

    # ── Normalised translation loss ───────────────────────────────────────
    l_trans = normalized_translation_loss(t_pred, t_gt)  # (B,)

    # ── Confidence-weighted pose loss ─────────────────────────────────────
    w_rot   = float(weights.get("rot",   1.0))
    w_trans = float(weights.get("trans", 1.0))
    l_pose_data = w_rot * l_rot + w_trans * l_trans      # (B,)
    l_pose      = (torch.exp(-s_pose) * l_pose_data + s_pose).mean()

    # ── Intrinsics regression loss ────────────────────────────────────────
    l_K = T_12_pred.new_tensor(0.0)
    if K_gt is not None:
        K_gt_vec   = torch.stack([K_gt[:,  0,0], K_gt[:,  1,1],
                                  K_gt[:,  0,2], K_gt[:,  1,2]], dim=-1)  # (B,4)
        K_pred_vec = torch.stack([K_pred[:,0,0], K_pred[:,1,1],
                                  K_pred[:,0,2], K_pred[:,1,2]], dim=-1)  # (B,4)
        # Relative L1 error per intrinsic parameter
        l_K_data = ((K_pred_vec - K_gt_vec) / K_gt_vec.clamp(min=EPS)).abs().mean(-1)  # (B,)
        l_K      = (torch.exp(-s_K) * l_K_data + s_K).mean()

    # ── Pose identity (round-trip) regulariser ────────────────────────────
    # T_21 = T_c2w_1^{−1} @ T_c2w_2  (cam2→cam1 from absolute poses)
    T_21_pred = torch.linalg.inv(T_c2w_1) @ T_c2w_2
    l_id      = pose_identity_loss(T_12_pred, T_21_pred)

    # ── Combine ───────────────────────────────────────────────────────────
    w_K_reg   = float(weights.get("K_reg",    0.5))
    w_id      = float(weights.get("identity", 0.1))
    total_cam = l_pose + w_K_reg * l_K + w_id * l_id

    parts = {
        "cam_rot":      float(l_rot.mean().detach()),
        "cam_trans":    float(l_trans.mean().detach()),
        "cam_pose":     float(l_pose.detach()),
        "cam_K":        float(l_K.detach()),
        "cam_identity": float(l_id.detach()),
    }
    return total_cam, parts


# ---------------------------------------------------------------------------
# 10. Total loss — called from train.py
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

    # --- Camera pose / intrinsics losses ---
    if "poses" in batch and "T_12_pred" in outputs:
        poses   = batch["poses"]                                 # (B, N, 4, 4)
        T_12_gt = torch.linalg.inv(poses[:, 1]) @ poses[:, 0]  # cam1→cam2 GT
        K_gt    = batch.get("intrinsics")                        # (B, 3, 3) or None
        l_cam, cam_parts = camera_pose_loss(outputs, T_12_gt, K_gt, w)
        total   = total + float(w.get("camera", 0.5)) * l_cam
        parts.update(cam_parts)

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
