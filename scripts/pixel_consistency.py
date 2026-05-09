import os
import argparse
import numpy as np
import cv2


# ── Loaders ────────────────────────────────────────────────────────────────────

def load_intrinsics(path):
    K = np.loadtxt(path)
    if K.shape == (4, 4):
        K = K[:3, :3]
    assert K.shape == (3, 3)
    return K


def load_pose(path):
    pose = np.loadtxt(path)
    if pose.shape == (3, 4):
        pose = np.vstack([pose, [0, 0, 0, 1]])
    assert pose.shape == (4, 4)
    return pose


def load_depth(path, scale=1.0):
    """Load a depth map from .npz, .npy, or image file."""
    if path.endswith(".npz"):
        data = np.load(path)
        for key in ["depth", "pred", "prediction", "arr_0"]:
            if key in data:
                depth = data[key]
                break
        else:
            depth = data[list(data.keys())[0]]
        depth = np.asarray(depth).astype(np.float32)
        if depth.ndim == 3 and depth.shape[0] == 1:
            depth = depth[0]
        if depth.ndim == 3 and depth.shape[-1] in (1, 3, 4):
            depth = depth[..., 0]
        return depth
    elif path.endswith(".npy"):
        depth = np.load(path).astype(np.float32)
        if depth.ndim == 3 and depth.shape[0] == 1:
            depth = depth[0]
        if depth.ndim == 3 and depth.shape[-1] in (1, 3, 4):
            depth = depth[..., 0]
        return depth
    else:
        depth = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise FileNotFoundError(path)
        depth = depth.astype(np.float32)
        if depth.ndim == 3:
            depth = cv2.cvtColor(depth, cv2.COLOR_BGR2GRAY)
        return depth / scale


# ── Geometry ───────────────────────────────────────────────────────────────────

def project_with_depth(depth, K, pose_src, pose_tgt, cam_to_world=True):
    """
    Unproject every source pixel using `depth`, then project into the target frame.

    Returns
    -------
    x_proj, y_proj : float32 arrays of shape (H, W) – projected pixel coordinates
    valid          : bool array of shape (H, W)
                     True where depth > 0, z_tgt > 0, and projection is in-bounds
    """
    H, W = depth.shape
    y_idx, x_idx = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')

    K_inv = np.linalg.inv(K)
    xy1 = np.stack([x_idx, y_idx, np.ones_like(x_idx)], axis=-1)          # (H,W,3)
    xyz = (K_inv @ xy1[..., None])[..., 0] * depth[..., None]              # (H,W,3)

    # Relative transform: src → tgt
    if cam_to_world:
        T = np.linalg.inv(pose_tgt) @ pose_src
    else:
        T = pose_tgt @ np.linalg.inv(pose_src)

    xyz_tgt = (T[:3, :3] @ xyz[..., None])[..., 0] + T[:3, 3]             # (H,W,3)
    proj    = (K @ xyz_tgt[..., None])[..., 0]                             # (H,W,3)

    z       = proj[..., 2]
    x_proj  = proj[..., 0] / (z + 1e-6)
    y_proj  = proj[..., 1] / (z + 1e-6)

    valid = (
        (z > 0) &
        (depth > 0) &
        (x_proj >= 0) & (x_proj < W - 1) &
        (y_proj >= 0) & (y_proj < H - 1)
    )

    return x_proj.astype(np.float32), y_proj.astype(np.float32), valid


# ── Metric ─────────────────────────────────────────────────────────────────────

def compute_pixel_consistency(
    gt_depth_src, pred_depth_src,
    gt_depth_tgt,
    K, pose_src, pose_tgt,
    cam_to_world=True,
):
    """
    Pixel consistency via reprojection error.

    Algorithm
    ---------
    For every pixel p in frame src with valid GT depth:
      1. Project p → frame tgt using GT depth    → position p_gt
      2. Verify GT depth in tgt at p_gt is valid  (real geometric correspondence)
      3. Project p → frame tgt using pred depth  → position p_pred
      4. Compute Euclidean pixel distance |p_gt - p_pred|

    Returns
    -------
    mae        : mean reprojection error (pixels)
    rmse       : root-mean-square reprojection error (pixels)
    valid_ratio: fraction of source pixels that passed all validity checks
    """
    H, W = gt_depth_src.shape

    # ── Step 1: project with GT depth ────────────────────────────────────────
    x_gt, y_gt, valid_gt_proj = project_with_depth(
        gt_depth_src, K, pose_src, pose_tgt, cam_to_world
    )

    # ── Step 2: check GT depth validity at projected position in tgt ─────────
    x_gt_int = np.clip(np.round(x_gt).astype(np.int32), 0, W - 1)
    y_gt_int = np.clip(np.round(y_gt).astype(np.int32), 0, H - 1)
    valid_gt_full = valid_gt_proj & (gt_depth_tgt[y_gt_int, x_gt_int] > 0)

    # ── Step 3: project with predicted depth ─────────────────────────────────
    x_pred, y_pred, valid_pred_proj = project_with_depth(
        pred_depth_src, K, pose_src, pose_tgt, cam_to_world
    )

    # ── Step 4: combined validity + reprojection error ────────────────────────
    valid = valid_gt_full & valid_pred_proj

    if np.sum(valid) < 100:
        return float("inf"), float("inf"), 0.0

    dist = np.sqrt((x_gt - x_pred) ** 2 + (y_gt - y_pred) ** 2)  # (H, W), pixels
    d    = dist[valid]

    mae  = np.mean(d)
    rmse = np.sqrt(np.mean(d ** 2))

    return mae, rmse, np.mean(valid)


# ── Main ───────────────────────────────────────────────────────────────────────

def main(args):
    gt_depth1   = load_depth(args.gt_depth1,   args.depth_scale_gt)
    gt_depth2   = load_depth(args.gt_depth2,   args.depth_scale_gt)
    pred_depth1 = load_depth(args.pred_depth1, args.depth_scale_pred)
    pred_depth2 = load_depth(args.pred_depth2, args.depth_scale_pred)

    H, W = gt_depth1.shape

    def resize_if_needed(d, h, w):
        return cv2.resize(d, (w, h), interpolation=cv2.INTER_NEAREST) if d.shape != (h, w) else d

    gt_depth1   = resize_if_needed(gt_depth1,   H, W)
    gt_depth2   = resize_if_needed(gt_depth2,   H, W)
    pred_depth1 = resize_if_needed(pred_depth1, H, W)
    pred_depth2 = resize_if_needed(pred_depth2, H, W)

    K = load_intrinsics(args.intrinsics)

    pose1 = load_pose(args.pose1)
    pose2 = load_pose(args.pose2)

    print("Running forward pixel consistency (1 → 2)...")
    mae_12, rmse_12, vr_12 = compute_pixel_consistency(
        gt_depth1, pred_depth1, gt_depth2,
        K, pose1, pose2,
        cam_to_world=args.cam_to_world,
    )

    print("Running backward pixel consistency (2 → 1)...")
    mae_21, rmse_21, vr_21 = compute_pixel_consistency(
        gt_depth2, pred_depth2, gt_depth1,
        K, pose2, pose1,
        cam_to_world=args.cam_to_world,
    )

    print("\n=== RESULTS ===")
    print("\nPixel Consistency:")
    print(f"  MAE  (1→2): {mae_12:.4f}")
    print(f"  MAE  (2→1): {mae_21:.4f}")
    print(f"  MAE  avg:   {(mae_12 + mae_21) / 2:.4f}")
    print(f"  RMSE (1→2): {rmse_12:.4f}")
    print(f"  RMSE (2→1): {rmse_21:.4f}")
    print(f"  RMSE avg:   {(rmse_12 + rmse_21) / 2:.4f}")
    print("\nValid pixel ratios:")
    print(f"  1→2: {vr_12:.3f}")
    print(f"  2→1: {vr_21:.3f}")


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pixel consistency: compare where GT depth vs predicted depth project pixels."
    )

    parser.add_argument("--gt_depth1",   required=True, help="GT depth for frame 1")
    parser.add_argument("--gt_depth2",   required=True, help="GT depth for frame 2")
    parser.add_argument("--pred_depth1", required=True, help="Predicted depth for frame 1")
    parser.add_argument("--pred_depth2", required=True, help="Predicted depth for frame 2")

    parser.add_argument("--intrinsics",  required=True)
    parser.add_argument("--pose1",       required=True)
    parser.add_argument("--pose2",       required=True)

    parser.add_argument("--depth_scale_gt",   type=float, default=1000.0,
                        help="Divisor for GT depth (e.g. 1000 for ScanNet mm→m PNG)")
    parser.add_argument("--depth_scale_pred", type=float, default=1.0,
                        help="Divisor for predicted depth (usually 1.0 for .npz)")

    parser.add_argument("--cam_to_world", action="store_true", default=True,
                        help="Poses are camera-to-world (default)")
    parser.add_argument("--world_to_cam", dest="cam_to_world", action="store_false",
                        help="Poses are world-to-camera")

    args = parser.parse_args()
    main(args)
