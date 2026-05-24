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
8. pixel_consistency_loss   — Multiview reprojection: ||proj(pred) − proj(gt)||² in pixels
9. compute_depth_metrics    — abs_rel / rmse / delta1 (validation only)
10. total_loss              — Weighted sum with full breakdown dict for logging

Commented out (re-enable once network produces reasonable depths)
-----------------------------------------------------------------
- photometric_loss    — SSIM + L1 with differentiable warping
- depth_consistency   — Warp predicted depth across views, compare
"""

import torch
import torch.nn.functional as F

def se3_inv(T: torch.Tensor) -> torch.Tensor:
    """Numerically stable inverse for SE(3) matrices (B, 4, 4)."""
    R = T[:, :3, :3]          # (B, 3, 3)
    t = T[:, :3,  3]          # (B, 3)
    R_inv = R.transpose(-1, -2)
    t_inv = -torch.bmm(R_inv, t.unsqueeze(-1)).squeeze(-1)
    T_inv = torch.zeros_like(T)
    T_inv[:, :3, :3] = R_inv
    T_inv[:, :3,  3] = t_inv
    T_inv[:, 3,   3] = 1.0
    return T_inv

EPS = 1e-8
EPS_NORM = 0.1   # floor for translation-direction normalisation (metres).
                  # Using squared-norm-clamp-sqrt (not .norm().clamp) so the backward
                  # at t=0 is ZERO instead of NaN.  EPS_NORM=0.1 bounds gradient
                  # amplification at 10×; pairs with < 10 cm baseline are treated as
                  # pure-rotation (direction suppressed) which is intentional.
ACOS_EPS = 1e-2  # raised from 1e-4: limits max acos gradient to ~7 (was ~70)
LOG_CONF_MIN = -3.0   # tightened: prevents runaway uncertainty / confidence collapse
LOG_CONF_MAX =  3.0


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
    mean_d = depth.mean(dim=[2, 3], keepdim=True).clamp(min=0.1)  # 0.1m floor; EPS would let d_norm → 1e5+
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
# 5b. Deep supervision over iterative pose estimates
# ---------------------------------------------------------------------------

def pose_iter_supervision_loss(
    T_12_iters: list,           # list of (B, 4, 4)  — one per GRU iteration
    T_12_gt:    torch.Tensor,   # (B, 4, 4) ground-truth cam1→cam2
    weights:    list | None = None,
) -> torch.Tensor:
    """
    Weighted geodesic + translation loss over each pose refinement iteration.

    Later iterations receive higher weight (exponential schedule, same as
    iter_supervision_loss for depth).  This provides a dense gradient signal
    at every GRU step rather than only supervising the final output.

    T_12_iters[-1] is the "final" pose and carries the most weight.
    """
    n = len(T_12_iters)
    if n == 0:
        return T_12_gt.new_tensor(0.0)
    if weights is None:
        w_list  = [2 ** i for i in range(n)]
        total_w = sum(w_list)
        weights = [wi / total_w for wi in w_list]

    R_gt = T_12_gt[:, :3, :3]
    t_gt = T_12_gt[:, :3,  3]

    loss = T_12_gt.new_tensor(0.0)
    for T_i, wi in zip(T_12_iters, weights):
        R_i = T_i[:, :3, :3]
        t_i = T_i[:, :3,  3]
        l_r = geodesic_rotation_loss(R_i, R_gt).mean()
        l_t = normalized_translation_loss(t_i, t_gt).mean()
        loss = loss + wi * (l_r + l_t)
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
    # Ensure computations happen in a safe float dtype (FP32) to support
    # mixed-precision training where R_pred may be float16 while R_gt is
    # float32 (autocast). Cast both to float32 on the same device.
    dev = R_pred.device
    R_pred_f = R_pred.to(torch.float32)
    R_gt_f   = R_gt.to(torch.float32)

    R_rel = R_pred_f.transpose(-1, -2) @ R_gt_f              # (B, 3, 3)

    # Chordal distance: ‖R_rel − I‖_F  ∈ [0, 2√2]
    # Monotonically related to geodesic θ = 2·arcsin(chord / (2√2)).
    # Unlike acos, the gradient of ‖M‖_F is M/‖M‖_F — bounded and smooth
    # everywhere, including at θ=0 (returns 0) and θ=π (returns 2√2).
    I3    = torch.eye(3, device=R_rel.device, dtype=R_rel.dtype).unsqueeze(0)
    diff  = R_rel - I3                             # (B, 3, 3)
    chord = (diff * diff).sum(dim=[-2, -1]).sqrt() # (B,)  in [0, 2√2]
    return torch.nan_to_num(chord, nan=0.0).to(R_pred.device)


# ---------------------------------------------------------------------------
# 7. Normalised translation loss
# ---------------------------------------------------------------------------

def normalized_translation_loss(
    t_pred: torch.Tensor,   # (B, 3) predicted translation
    t_gt:   torch.Tensor,   # (B, 3) ground-truth translation
) -> torch.Tensor:
    """
    Angular error between predicted and GT translation directions.

    Explicit norm clamp guards against NaN when the network predicts a
    near-zero translation early in training (0/0 → NaN in F.normalize).
    Pure-rotation pairs (‖t_gt‖ ≈ 0) produce nan_to_num → 0.0.

    Returns (B,) tensor of angles in radians.
    """
    # Safe squared-norm then sqrt: grad = 0 at t=0 (no NaN), rises to t/||t|| normally.
    # Unlike .norm().clamp(), the backward through (t*t).sum().clamp().sqrt() never
    # computes t/||t|| — the clamp zero-gates the gradient when ||t||^2 < EPS_NORM^2.
    t_pred_norm = (t_pred * t_pred).sum(dim=-1, keepdim=True).clamp(min=EPS_NORM**2).sqrt()
    t_gt_norm   = (t_gt   * t_gt  ).sum(dim=-1, keepdim=True).clamp(min=EPS_NORM**2).sqrt()
    t_pred_n    = t_pred / t_pred_norm
    t_gt_n      = t_gt   / t_gt_norm
    # Cosine loss: 1 - cos(θ) ∈ [0,2].  Smooth, bounded, NO singularity anywhere.
    # atan2 and cross.norm() both produce NaN gradients at (0,0) in backward.
    # (1-dot) ≈ θ²/2 near zero and 2 at antiparallel — monotonically related to angle.
    dot        = (t_pred_n * t_gt_n).sum(dim=-1)           # (B,)
    loss_trans = (1.0 - dot).clamp(min=0.0)                # (B,) ∈ [0,2]
    return torch.nan_to_num(loss_trans, nan=0.0)             # pure-rotation pairs → 0


# ---------------------------------------------------------------------------
# 8. Pose identity (round-trip consistency) loss
# ---------------------------------------------------------------------------

def pose_identity_loss(
    T_12: torch.Tensor,   # (B, 4, 4)
    T_21: torch.Tensor,   # (B, 4, 4)
    rot_weight: float = 1.0,
    trans_weight: float = 1.0,
) -> torch.Tensor:
    """
    Round-trip consistency loss, separated into rotation and translation.

    Rotation term  : geodesic( R_12 @ R_21, I_3 )
    Translation term: angle between t_12 and −R_12 · t_21
                      (paper's formulation: t_12 should be anti-parallel to
                       the translation of T_21 rotated into frame 1)

    Pure-rotation pairs (‖t‖ ≈ 0) are handled by nan_to_num → 0.
    """
    R_12 = T_12[:, :3, :3]                                  # (B, 3, 3)
    t_12 = T_12[:, :3,  3]                                  # (B, 3)
    R_21 = T_21[:, :3, :3]
    t_21 = T_21[:, :3,  3]

    # ── Rotation: R_12 @ R_21 should equal I_3 ─────────────────────────
    B = R_12.shape[0]
    I3 = torch.eye(3, device=R_12.device, dtype=R_12.dtype).unsqueeze(0).expand(B, -1, -1)
    rot_err = geodesic_rotation_loss(R_12 @ R_21, I3)        # (B,) radians
    rot_err = torch.nan_to_num(rot_err, nan=0.0)
 
    # ── Translation: t_12 should be anti-parallel to R_12 · t_21 ───────
    # Paper's constraint: t_12 ≈ −R_12 · t_21
    t_21_in_frame1 = torch.bmm(R_12, t_21.unsqueeze(-1)).squeeze(-1)  # (B, 3)
    neg_t21 = -t_21_in_frame1
    # Safe squared-norm normalisation (no NaN at zero)
    t12_n   = t_12  / (t_12  * t_12 ).sum(dim=-1, keepdim=True).clamp(min=EPS_NORM**2).sqrt()
    nt21_n  = neg_t21 / (neg_t21 * neg_t21).sum(dim=-1, keepdim=True).clamp(min=EPS_NORM**2).sqrt()
    # atan2-based angle: stable at all angles unlike acos
    # Cosine loss: same bounded/smooth trick as normalized_translation_loss
    cos_id    = (t12_n * nt21_n).sum(dim=-1)                 # (B,)
    trans_err = (1.0 - cos_id).clamp(min=0.0)               # (B,) ∈ [0,2]
    trans_err = torch.nan_to_num(trans_err, nan=0.0)         # pure-rotation pairs → 0
 
    return float(rot_weight) * rot_err.mean() + float(trans_weight) * trans_err.mean()    


# ---------------------------------------------------------------------------
# 9. Composite camera-pose loss with confidence weighting
# ---------------------------------------------------------------------------

def camera_pose_loss(
    outputs:        dict,
    T_12_gt:        torch.Tensor,          # (B, 4, 4) GT relative pose (cam1→cam2)
    K_gt:           torch.Tensor | None,   # (B, 3, 3) GT intrinsics, or None
    weights:        dict,
    use_confidence: bool = True,
) -> tuple[torch.Tensor, dict]:
    """
    Composite camera-pose loss with optional heteroscedastic confidence weighting.

    When use_confidence=True (default), pose and intrinsics losses are weighted
    by a per-sample uncertainty following Kendall & Gal (NeurIPS 2017):

        L_pose = exp(−s_pose) · L_pose_data + s_pose
        L_K    = exp(−s_K)    · L_K_data    + s_K

    The "+s" regulariser prevents the network driving uncertainty to ∞ and the
    data term to 0.  exp(−s)·loss + s is strictly positive and has minimum at
    s* = −log(L_data), i.e. the network learns to match its uncertainty to the
    actual error magnitude.

    When use_confidence=False (warmup phase), confidence is fixed at 1 so the
    network learns pose geometry before also optimising uncertainty estimates.

    The pose identity round-trip loss is always added unweighted.

    Parameters
    ----------
    outputs        : DepthAlignNet output dict (must contain T_12_pred,
                     K_pred, log_conf_K, log_conf_pose)
    T_12_gt        : (B, 4, 4) ground-truth relative pose
    K_gt           : (B, 3, 3) ground-truth intrinsics, or None (skips K loss)
    weights        : loss weight dict (keys: rot, trans, K_reg, identity, camera)
    use_confidence : if False, uses fixed conf=1 — no uncertainty weighting

    Returns
    -------
    total_camera_loss : scalar tensor
    parts             : dict of float scalars for logging
    """
    T_12_pred    = outputs["T_12_pred"]     # (B, 4, 4)
    K_pred       = outputs["K_pred"]        # (B, 3, 3)
    s_pose_raw   = outputs["log_conf_pose"] # (B,)
    s_K_raw      = outputs["log_conf_K"]    # (B,)
    s_pose       = s_pose_raw.clamp(LOG_CONF_MIN, LOG_CONF_MAX)
    s_K          = s_K_raw.clamp(LOG_CONF_MIN, LOG_CONF_MAX)

    R_pred = T_12_pred[:, :3, :3]          # (B, 3, 3)
    t_pred = T_12_pred[:, :3,  3]          # (B, 3)
    R_gt   = T_12_gt[:, :3, :3]
    t_gt   = T_12_gt[:, :3,  3]

    # ── Geodesic rotation loss ────────────────────────────────────────────
    l_rot   = geodesic_rotation_loss(R_pred, R_gt)    # (B,)

    # ── Normalised translation loss ───────────────────────────────────────
    l_trans = normalized_translation_loss(t_pred, t_gt)  # (B,)

    # NOTE: translation scale loss removed — log(‖t_pred‖) backward is NaN
    # when ‖t_pred‖ ≈ 0 (i.e. at init).  Re-enable after directional loss converges.

    # ── Confidence-weighted pose loss ─────────────────────────────────────
    w_rot   = float(weights.get("rot",   1.0))
    w_trans = float(weights.get("trans", 1.0))
    l_pose_data = w_rot * l_rot + w_trans * l_trans      # (B,)
    if use_confidence:
        # Kendall & Gal (NeurIPS 2017): exp(-s)·loss + s
        # Always positive; minimum at s* = -log(L_data) so the network
        # learns to match uncertainty magnitude to actual error.
        conf_pose = torch.exp(-s_pose)                   # (B,)  ∈ (e⁻³, e³)
        l_pose = (conf_pose * l_pose_data + s_pose).mean()
    else:
        l_pose = l_pose_data.mean()                      # fixed conf = 1

    # ── Intrinsics regression loss ────────────────────────────────────────
    l_K = T_12_pred.new_tensor(0.0)
    if K_gt is not None:
        K_gt_vec   = torch.stack([K_gt[:,  0,0], K_gt[:,  1,1],
                                  K_gt[:,  0,2], K_gt[:,  1,2]], dim=-1)  # (B,4)
        K_pred_vec = torch.stack([K_pred[:,0,0], K_pred[:,1,1],
                                  K_pred[:,0,2], K_pred[:,1,2]], dim=-1)  # (B,4)
        # Relative L1 error per intrinsic parameter
        l_K_data = ((K_pred_vec - K_gt_vec) / K_gt_vec.clamp(min=EPS)).abs().mean(-1)  # (B,)
        if use_confidence:
            conf_K = torch.exp(-s_K)                     # (B,)  ∈ (e⁻³, e³)
            l_K    = (conf_K * l_K_data + s_K).mean()
        else:
            l_K    = l_K_data.mean()                     # fixed conf = 1

    # ── Pose identity (round-trip) regulariser ────────────────────────────
    # T_21 is predicted directly (swapped embed order), not derived by inversion.
    T_21_pred = outputs["T_21_pred"]
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


def k_inv(K: torch.Tensor) -> torch.Tensor:
    """
    Analytical inverse of an upper-triangular camera intrinsics matrix (B, 3, 3).

    For K = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]] the closed-form inverse is:
        K⁻¹ = [[1/fx,    0, -cx/fx],
               [   0, 1/fy, -cy/fy],
               [   0,    0,      1]]

    Avoids the generic LU decomposition of torch.linalg.inv, keeping gradients
    clean and computation cheap — same philosophy as se3_inv for SE(3).
    """
    fx = K[:, 0, 0]   # (B,)
    fy = K[:, 1, 1]
    cx = K[:, 0, 2]
    cy = K[:, 1, 2]
    B  = K.shape[0]
    Ki = torch.zeros_like(K)
    Ki[:, 0, 0] =  1.0 / fx.clamp(min=EPS)
    Ki[:, 1, 1] =  1.0 / fy.clamp(min=EPS)
    Ki[:, 0, 2] = -cx  / fx.clamp(min=EPS)
    Ki[:, 1, 2] = -cy  / fy.clamp(min=EPS)
    Ki[:, 2, 2] =  1.0
    return Ki


# ---------------------------------------------------------------------------
# 8. Pixel consistency loss (multiview reprojection)
# ---------------------------------------------------------------------------

def pixel_consistency_loss(
    pred_depth1: torch.Tensor,   # (B, 1, pH, pW) predicted depth, view 1
    pred_depth2: torch.Tensor,   # (B, 1, pH, pW) predicted depth, view 2
    gt_depth1:   torch.Tensor,   # (B, 1, pH, pW) GT depth, view 1
    gt_depth2:   torch.Tensor,   # (B, 1, pH, pW) GT depth, view 2
    T_12:        torch.Tensor,   # (B, 4, 4) cam1 → cam2 relative pose
    K:           torch.Tensor,   # (B, 3, 3) intrinsics at pH × pW resolution
    min_depth: float = 1e-3,
    max_depth: float = 80.0,
) -> torch.Tensor:
    """
    Differentiable multiview pixel consistency loss.

    Mirrors compute_pixel_consistency() from metrics/pixel_consistency.py but
    runs entirely in PyTorch so gradients flow back through pred_depth.

    For every source pixel with valid GT depth:
      1. Unproject with GT depth    → 3-D point → project into target → p_gt
      2. Unproject with pred depth  → 3-D point → project into target → p_pred
      3. Penalise  ||p_gt − p_pred||²  (squared pixel distance)

    Using the GT depth to anchor the reference projection means the loss is
    zero when pred_depth == gt_depth and increases smoothly as predictions
    diverge — giving a multiview geometric signal without requiring dense
    photometric matching.

    Symmetrised: 1→2 direction with pred_depth1  +  2→1 with pred_depth2.
    """
    B, _, pH, pW = pred_depth1.shape
    dev   = pred_depth1.device
    dtype = pred_depth1.dtype

    # Pixel coordinate grid  (B, 3, pH*pW)
    ys = torch.arange(pH, device=dev, dtype=dtype)
    xs = torch.arange(pW, device=dev, dtype=dtype)
    gy, gx = torch.meshgrid(ys, xs, indexing='ij')       # (pH, pW)
    N = pH * pW
    ones = torch.ones(N, device=dev, dtype=dtype)
    xy1  = torch.stack([gx.flatten(), gy.flatten(), ones], dim=0)  # (3, N)
    xy1  = xy1.unsqueeze(0).expand(B, -1, -1)                       # (B, 3, N)

    K_inv = k_inv(K)                                   # (B, 3, 3) analytical

    def _project(depth: torch.Tensor, T: torch.Tensor):
        """Return (x_proj, y_proj, z_tgt) each (B, N)."""
        d    = depth.reshape(B, 1, N)                     # (B, 1, N)
        xyz  = torch.bmm(K_inv, xy1) * d                  # (B, 3, N)  3-D in cam1
        R, t = T[:, :3, :3], T[:, :3, 3]                 # (B,3,3), (B,3)
        xyz_t = torch.bmm(R, xyz) + t.unsqueeze(-1)       # (B, 3, N)  in cam2
        proj  = torch.bmm(K, xyz_t)                       # (B, 3, N)
        z     = proj[:, 2, :]                              # (B, N)
        x_p   = proj[:, 0, :] / (z + EPS)
        y_p   = proj[:, 1, :] / (z + EPS)
        return x_p, y_p, z

    def _one_dir(pred_d: torch.Tensor, gt_d: torch.Tensor, T: torch.Tensor):
        x_gt,   y_gt,   z_gt   = _project(gt_d,   T)
        x_pred, y_pred, z_pred = _project(pred_d, T)

        d_flat = gt_d.reshape(B, N)
        # Validity: require GT depth in range AND GT reprojection in-bounds AND z_pred > 0.
        # We do NOT require x_pred/y_pred to be in-bounds: wrong depth predictions can
        # project outside the image, and that large pixel distance is exactly the signal
        # we want to train on.  Excluding those pixels kills the gradient for the most
        # erroneous predictions — the opposite of what we want during early training.
        valid  = (
            (d_flat > min_depth) & (d_flat < max_depth) &
            torch.isfinite(d_flat) &
            torch.isfinite(x_pred) & torch.isfinite(y_pred) &
            (z_gt > 0) & (z_pred > 0) &
            (x_gt >= 0) & (x_gt < pW - 1) &
            (y_gt >= 0) & (y_gt < pH - 1)
        )  # (B, N)   — pred bounds intentionally omitted

        n_valid = valid.float().sum()
        if n_valid < 1:
            return pred_d.new_tensor(0.0)

        x_gt_n   = x_gt   / (pW - 1)
        y_gt_n   = y_gt   / (pH - 1)
        x_pred_n = x_pred / (pW - 1)
        y_pred_n = y_pred / (pH - 1)
        dist_sq  = (x_gt_n - x_pred_n) ** 2 + (y_gt_n - y_pred_n) ** 2
        # Cap at 4.0 (= 2 diagonals²) to down-weight catastrophically wrong
        # predictions without zeroing their gradients via clamping.
        dist_sq  = dist_sq.clamp(max=4.0)
        return (dist_sq * valid.float()).sum() / n_valid.clamp(min=1)

    T_21 = se3_inv(T_12)
    l_12 = _one_dir(pred_depth1, gt_depth1, T_12)
    l_21 = _one_dir(pred_depth2, gt_depth2, T_21)
    return (l_12 + l_21) * 0.5


# ---------------------------------------------------------------------------
# 11. Geometric consistency loss  (ViSTA-SLAM  Lgc)
# ---------------------------------------------------------------------------

def geometric_consistency_loss(
    pred_depth1: torch.Tensor,   # (B, 1, H, W)  predicted depth, frame i
    pred_depth2: torch.Tensor,   # (B, 1, H, W)  predicted depth, frame j
    T_12:        torch.Tensor,   # (B, 4, 4)  cam_i → cam_j  (GT or predicted)
    K:           torch.Tensor,   # (B, 3, 3)  intrinsics (at H×W resolution)
    min_depth:   float = 0.1,
    max_depth:   float = 80.0,
) -> torch.Tensor:
    """
    ViSTA-SLAM geometric consistency loss (Lgc).

        Lgc = (1/n) Σ_{x ∈ I_i}  ‖ T_ij P_i(x)  −  P_j(C_ij(x)) ‖

    Both terms are 3-D points expressed in frame j's coordinate system:

      T_ij P_i(x)    — unproject pixel x with pred_depth_i, warp into frame j
      P_j(C_ij(x))   — follow the warp to find the correspondence C_ij(x) in
                        frame j, bilinearly sample pred_depth_j there, unproject

    The loss is zero when pred_depth1, pred_depth2, and T_12 are mutually
    consistent in 3-D metric space.  Gradients flow through both depth maps.

    Unlike pixel_consistency_loss (2-D reprojection error), this penalises
    3-D Euclidean distance — scale-aware and directly in metres.

    T_12 can be GT (pure depth supervision) or predicted (joint supervision).
    When switching from GT → predicted mid-training the caller controls which
    is passed; no logic change is needed here.

    Parameters
    ----------
    pred_depth1 / pred_depth2 : (B, 1, H, W)  predicted depth maps
    T_12  : (B, 4, 4)  relative pose cam_1 → cam_2
    K     : (B, 3, 3)  intrinsics at the depth-map resolution
    min_depth / max_depth : validity thresholds (metres)

    Returns
    -------
    Scalar loss (mean over both directions and valid pixels).
    """
    B, _, H, W = pred_depth1.shape
    dev   = pred_depth1.device
    dtype = pred_depth1.dtype

    # ── Pixel coordinate grid ────────────────────────────────────────────
    ys = torch.arange(H, device=dev, dtype=dtype)
    xs = torch.arange(W, device=dev, dtype=dtype)
    gy, gx = torch.meshgrid(ys, xs, indexing='ij')   # (H, W)
    N   = H * W
    xy1 = torch.stack([gx.flatten(), gy.flatten(),
                       torch.ones(N, device=dev, dtype=dtype)], dim=0)   # (3, N)
    xy1 = xy1.unsqueeze(0).expand(B, -1, -1)                             # (B, 3, N)

    K_inv = k_inv(K)   # (B, 3, 3)  analytical, gradient-friendly

    def _one_dir(
        depth_i: torch.Tensor,   # (B, 1, H, W)  source depth
        depth_j: torch.Tensor,   # (B, 1, H, W)  target depth
        T: torch.Tensor,         # (B, 4, 4)
    ) -> torch.Tensor:
        R = T[:, :3, :3]          # (B, 3, 3)
        t = T[:, :3,  3]          # (B, 3)

        # 1. Unproject frame i → 3-D in frame i coords
        d_i  = depth_i.reshape(B, 1, N).clamp(min=min_depth, max=max_depth)
        P_i  = torch.bmm(K_inv, xy1) * d_i          # (B, 3, N)

        # 2. Warp into frame j coords:  Q = R @ P_i + t
        Q    = torch.bmm(R, P_i) + t.unsqueeze(-1)  # (B, 3, N)
        Q_z  = Q[:, 2, :]                            # (B, N)

        # 3. Project Q into frame j (homogeneous pixel coords)
        proj  = torch.bmm(K, Q)                      # (B, 3, N)
        u_j   = proj[:, 0, :] / (Q_z + EPS)          # (B, N)
        v_j   = proj[:, 1, :] / (Q_z + EPS)

        # 4. Convert to grid_sample coords in [-1, 1]
        #    grid_sample expects (B, 1, N, 2) with (x, y) = (u_norm, v_norm)
        u_n  = (2.0 * u_j / (W - 1) - 1.0)           # (B, N)
        v_n  = (2.0 * v_j / (H - 1) - 1.0)
        grid = torch.stack([u_n, v_n], dim=-1).reshape(B, 1, N, 2)   # (B,1,N,2)

        # 5. Bilinearly sample pred depth_j at correspondence
        #    align_corners=True matches the u_n / v_n formula above
        d_j_sampled = F.grid_sample(
            depth_j, grid,
            mode='bilinear', padding_mode='zeros', align_corners=True,
        ).reshape(B, N)                               # (B, N)

        # 6. Unproject frame j at the sampled pixel (u_j, v_j)
        #    Px_j = [u_j, v_j, 1]ᵀ (already in pixel homogeneous coords)
        #    P̂_j  = d_j_sampled · K⁻¹ · [u_j, v_j, 1]ᵀ
        uv1_j = torch.stack([u_j, v_j,
                              torch.ones_like(u_j)], dim=1)   # (B, 3, N)
        P_j   = torch.bmm(K_inv, uv1_j) * d_j_sampled.unsqueeze(1)  # (B, 3, N)

        # 7. 3-D distance: ‖ Q − P̂_j ‖  per pixel
        dist = (Q - P_j).norm(dim=1)                 # (B, N)

        # 8. Validity mask ────────────────────────────────────────────────
        valid = (
            (Q_z > 0) &
            # Correspondence must fall within image bounds (in pixels)
            (u_j >= 0) & (u_j < W) &
            (v_j >= 0) & (v_j < H) &
            # Sampled depth must be sensible (0 = padding_mode zeros → skip)
            (d_j_sampled > min_depth) & (d_j_sampled < max_depth) &
            # Source depth in range  (already clamped above, but also filter NaN)
            torch.isfinite(dist)
        )   # (B, N)

        n_valid = valid.float().sum()
        if n_valid < 1:
            return depth_i.new_tensor(0.0)

        # 9. Huber-like soft cap: clip at 5 m to down-weight outliers without
        #    zeroing gradients (same philosophy as the dist_sq.clamp in pixel loss)
        dist_capped = dist.clamp(max=5.0)
        return (dist_capped * valid.float()).sum() / n_valid

    T_21 = se3_inv(T_12)
    l_12 = _one_dir(pred_depth1, pred_depth2, T_12)
    l_21 = _one_dir(pred_depth2, pred_depth1, T_21)
    return (l_12 + l_21) * 0.5


# ---------------------------------------------------------------------------
# 10. Total loss — called from train.py
# ---------------------------------------------------------------------------

def total_loss(
    outputs:        dict,             # from DepthAlignNet.forward()
    batch:          dict,             # from DataLoader
    weights:        dict,             # from cfg["loss"]
    K:              torch.Tensor,     # (B, 3, 3) full-resolution intrinsics
    use_confidence: bool = True,      # False during warmup — disables heteroscedastic weighting
    camera_weight:  float | None = None,  # override cfg camera weight (for ramp)
    epoch:          int = 0,          # current epoch — controls Lgc activation
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

    # --- Pixel consistency (multiview reprojection) ---
    l_pixel = pred1.new_tensor(0.0)
    if "poses" in batch:
        poses_pc = batch["poses"]                           # (B, N, 4, 4)
        T_12_pc  = se3_inv(poses_pc[:, 1]) @ poses_pc[:, 0]
        # Prefer GT intrinsics for accurate reprojection; fall back to K_iter.
        K_for_pix = batch["intrinsics"].to(pred1.device, dtype=pred1.dtype) \
                    if "intrinsics" in batch else K
        # Scale K to predicted-depth resolution
        scale_w, scale_h = pW / W, pH / H
        K_s = K_for_pix.clone()
        K_s[:, 0, 0] = K_for_pix[:, 0, 0] * scale_w
        K_s[:, 1, 1] = K_for_pix[:, 1, 1] * scale_h
        K_s[:, 0, 2] = K_for_pix[:, 0, 2] * scale_w
        K_s[:, 1, 2] = K_for_pix[:, 1, 2] * scale_h
        l_pixel = pixel_consistency_loss(
            pred1, pred2, gt1_s, gt2_s, T_12_pc, K_s
        )
        total = total + float(w.get("pixel_consistency", 0.05)) * l_pixel
        parts["pixel_consistency"] = float(l_pixel.detach())

    # --- Geometric consistency loss  Lgc  (ViSTA-SLAM) ---
    # Activated only after geometric_warmup_epochs (default 30) so that
    # depths and pose are already reasonable before we couple them in 3-D.
    # Phase 1 (epoch < switch_epoch): use GT pose  → pure depth supervision.
    # Phase 2 (epoch >= switch_epoch): use predicted pose → joint supervision.
    l_gc = pred1.new_tensor(0.0)
    geo_warmup  = int(w.get("geometric_warmup_epochs", 30))
    geo_switch  = int(w.get("geometric_switch_epochs",  50))  # switch GT→pred pose
    geo_w       = float(w.get("geometric", 0.1))
    if geo_w > 0 and epoch >= geo_warmup and "poses" in batch:
        poses_gc = batch["poses"]                              # (B, N, 4, 4)
        T_12_gc_gt = se3_inv(poses_gc[:, 1]) @ poses_gc[:, 0]
        # Choose pose source based on training phase
        if epoch >= geo_switch and "T_12_pred" in outputs:
            T_gc = outputs["T_12_pred"].detach()   # detach: decouple gc grad from pose head
            # Only switch if the predicted pose is finite (NaN guard)
            if not torch.isfinite(T_gc).all():
                T_gc = T_12_gc_gt
        else:
            T_gc = T_12_gc_gt
        # Scale K to predicted-depth resolution (pred1 is at H/2 × W/2)
        scale_w_gc = pW / W
        scale_h_gc = pH / H
        K_gc = K.clone()
        if "intrinsics" in batch:
            K_gc = batch["intrinsics"].to(pred1.device, dtype=pred1.dtype).clone()
        K_gc[:, 0, 0] = K_gc[:, 0, 0] * scale_w_gc
        K_gc[:, 1, 1] = K_gc[:, 1, 1] * scale_h_gc
        K_gc[:, 0, 2] = K_gc[:, 0, 2] * scale_w_gc
        K_gc[:, 1, 2] = K_gc[:, 1, 2] * scale_h_gc
        l_gc = geometric_consistency_loss(pred1, pred2, T_gc, K_gc)
        total = total + geo_w * l_gc
        parts["geometric"] = float(l_gc.detach())

    # --- Camera pose / intrinsics losses ---
    T_12_gt_pose = None
    if "poses" in batch and "T_12_pred" in outputs:
        poses        = batch["poses"]                                 # (B, N, 4, 4)
        T_12_gt_pose = se3_inv(poses[:, 1]) @ poses[:, 0]  # cam1→cam2 GT
        # Debug: check GT translation norms (should NOT be near zero)
        t_gt_norms = T_12_gt_pose[:, :3, 3].norm(dim=-1)
        if t_gt_norms.mean() < 0.01:
            print(f"[WARNING] GT translation norms very small: min={t_gt_norms.min():.6f} max={t_gt_norms.max():.6f} mean={t_gt_norms.mean():.6f}")
            print(f"[WARNING] This suggests zero or near-zero GT motion — check pose convention!")
        K_gt    = batch.get("intrinsics")                        # (B, 3, 3) or None
        l_cam, cam_parts = camera_pose_loss(outputs, T_12_gt_pose, K_gt, w,
                                            use_confidence=use_confidence)
        eff_camera_w = camera_weight if camera_weight is not None else float(w.get("camera", 0.5))
        total   = total + eff_camera_w * l_cam
        parts.update(cam_parts)

    # --- Pose iterative deep supervision ---
    # Supervises every GRU iteration, not just the final pose.
    # Only active when the network returns T_12_iters (PoseRefinementModule).
    # Uses the same GT that was computed for camera_pose_loss above.
    l_pose_iters = pred1.new_tensor(0.0)
    if "T_12_iters" in outputs and T_12_gt_pose is not None:
        iters = outputs["T_12_iters"]
        if len(iters) > 0:
            l_pose_iters = pose_iter_supervision_loss(iters, T_12_gt_pose)
            pose_iters_w = float(w.get("pose_iters", 1.0))
            total = total + pose_iters_w * l_pose_iters
    parts["pose_iters"] = float(l_pose_iters.detach())

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
