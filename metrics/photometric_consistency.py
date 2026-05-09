import os
import argparse
import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt


# Functions

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


def load_image(path, as_gray=True):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)
    if as_gray:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #print(f"Loaded image {path} with shape {img.shape}")
    return img.astype(np.float32) / 255.0


def load_depth(path, scale=1000.0):
    if path.endswith(".npz"):
        data = np.load(path)
        depth = data[list(data.keys())[0]]
        #handle common cases of extra dimensions
        if depth.ndim == 3 and depth.shape[0] == 1:
            depth = depth[0]
        if depth.ndim == 3 and depth.shape[-1] in (1, 3, 4):
            depth = depth[..., 0]
    elif path.endswith(".npy"):
        depth = np.load(path)
        #handle common cases of extra dimensions
        if depth.ndim == 3 and depth.shape[0] == 1:
            depth = depth[0]
        if depth.ndim == 3 and depth.shape[-1] in (1, 3, 4):
            depth = depth[..., 0]
    else:
        depth = cv2.imread(path, cv2.IMREAD_UNCHANGED)

    if depth is None:
        raise FileNotFoundError(path)

    depth = depth.astype(np.float32)

    # if RGB depth → convert
    if depth.ndim == 3:
        depth = cv2.cvtColor(depth, cv2.COLOR_BGR2GRAY)
    
    #print(f"Loaded depth {path} with shape {depth.shape}, dtype={depth.dtype}, min={depth.min():.3f}, max={depth.max():.3f}")

    return depth / scale


def visualize_warping(img_src, img_tgt, warped, valid, out_path=None):
    """
    Display source (with valid mask overlay), target, and warped side-by-side.

    Works for both grayscale (H, W) and RGB (H, W, 3) images.
    """

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # --- Ensure RGB for visualization ---
    def to_rgb(img):
        if img.ndim == 2:  # grayscale
            return np.stack([img, img, img], axis=-1)
        return img  # already RGB

    src_rgb = to_rgb(img_src)
    tgt_rgb = to_rgb(img_tgt)
    warped_rgb = to_rgb(warped)

    # --- Source with valid mask overlay ---
    img_src_overlay = src_rgb.copy()
    img_src_overlay[valid, 1] = np.clip(
        img_src_overlay[valid, 1] + 0.3, 0, 1
    )  # boost green

    axes[0].imshow(img_src_overlay)
    axes[0].set_title(f'Source (valid={np.mean(valid):.2%})')
    axes[0].axis('off')

    # --- Target ---
    axes[1].imshow(tgt_rgb)
    axes[1].set_title('Target Image')
    axes[1].axis('off')

    # --- Warped ---
    axes[2].imshow(warped_rgb)
    axes[2].set_title('Warped Target→Source')
    axes[2].axis('off')

    plt.tight_layout()

    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {out_path}")

    plt.show()

# Geometry and projections

def project_points(depth, K, pose_src, pose_tgt, cam_to_world=True):
    H, W = depth.shape

    y, x = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')

    K_inv = np.linalg.inv(K)

    xy1 = np.stack([x, y, np.ones_like(x)], axis=-1)
    xyz = (K_inv @ xy1[..., None])[..., 0] * depth[..., None]

    if cam_to_world:
        T = np.linalg.inv(pose_tgt) @ pose_src
    else:
        T = pose_tgt @ np.linalg.inv(pose_src)

    R = T[:3, :3]
    t = T[:3, 3]

    xyz_tgt = (R @ xyz[..., None])[..., 0] + t

    proj = (K @ xyz_tgt[..., None])[..., 0]

    z = proj[..., 2]
    x_proj = proj[..., 0] / (z + 1e-6)
    y_proj = proj[..., 1] / (z + 1e-6)

    valid = (
        (z > 0) &
        (depth > 0) &
        (x_proj >= 0) & (x_proj < W - 1) &
        (y_proj >= 0) & (y_proj < H - 1)
    )

    return x_proj.astype(np.float32), y_proj.astype(np.float32), valid

# Metrics

def compute_photometric(img_src, img_tgt, depth, K, pose_src, pose_tgt, cam_to_world=True, visualize=False, vis_out=None):
    """
    Warp target → source and compare
    """
    x_proj, y_proj, valid = project_points(depth, K, pose_src, pose_tgt, cam_to_world)

    if np.sum(valid) < 100:
        return 0.0, float("inf"), 0.0

    warped = cv2.remap(
        img_tgt,
        x_proj,
        y_proj,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )

    if visualize:
        visualize_warping(img_src, img_tgt, warped, valid, out_path=vis_out)

    # L2 over valid pixels
    l2 = np.mean(((img_src - warped)[valid] ** 2))

    # SSIM handling -- compute on full image
    if img_src.ndim == 3:  # RGB
        #ssim_score = ssim(img_src, warped, data_range=1.0, channel_axis=-1)
        ssim_map = ssim(img_src, warped, data_range=1.0, channel_axis=-1, full=True)[1] #ssim only on valid pixels
        ssim_score = np.mean(ssim_map[valid])
    else:  # Grayscale
        #ssim_score = ssim(img_src, warped, data_range=1.0)
        ssim_map = ssim(img_src, warped, data_range=1.0, full=True)[1]   #ssim only on valid pixels
        ssim_score = np.mean(ssim_map[valid])
    
    valid_ratio = np.mean(valid)



    return ssim_score, l2, valid_ratio


# Main

def main(args):
    img1 = load_image(args.img1, as_gray=True)
    img2 = load_image(args.img2, as_gray=True)

    depth1 = load_depth(args.depth1, args.depth_scale)
    depth2 = load_depth(args.depth2, args.depth_scale)

    if depth1.shape != img1.shape:
        depth1 = cv2.resize(depth1, (img1.shape[1], img1.shape[0]), interpolation=cv2.INTER_NEAREST)

    if depth2.shape != img2.shape:
        depth2 = cv2.resize(depth2, (img2.shape[1], img2.shape[0]), interpolation=cv2.INTER_NEAREST)

    K = load_intrinsics(args.intrinsics)

    pose1 = load_pose(args.pose1)
    pose2 = load_pose(args.pose2)

    print("Running forward consistency (1 → 2)...")
    ssim_12, l2_12, vr_12 = compute_photometric(img1, img2, depth1, K, pose1, pose2, cam_to_world=args.cam_to_world, visualize=args.visualize, vis_out=args.vis_out_12 if args.visualize else None)

    print("Running backward consistency (2 → 1)...")
    ssim_21, l2_21, vr_21 = compute_photometric(img2, img1, depth2, K, pose2, pose1, cam_to_world=args.cam_to_world, visualize=args.visualize, vis_out=args.vis_out_21 if args.visualize else None)

    print("\n=== RESULTS ===")

    print("\nPhotometric Consistency:")
    print(f"  SSIM (1→2): {ssim_12:.4f}")
    print(f"  SSIM (2→1): {ssim_21:.4f}")
    print(f"  SSIM avg:   {(ssim_12 + ssim_21)/2:.4f}")

    print(f"  L2 (1→2):   {l2_12:.4f}")
    print(f"  L2 (2→1):   {l2_21:.4f}")
    print(f"  L2 avg:     {(l2_12 + l2_21)/2:.4f}")

    print("\nValid pixel ratios:")
    print(f"  1→2: {vr_12:.3f}")
    print(f"  2→1: {vr_21:.3f}")
    print(f"  Valid ratio avg: {(vr_12 + vr_21) / 2:.4f}")


# Entry point

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--img1", required=True)
    parser.add_argument("--img2", required=True)

    parser.add_argument("--depth1", required=True)
    parser.add_argument("--depth2", required=True)

    parser.add_argument("--intrinsics", required=True)

    parser.add_argument("--pose1", required=True)
    parser.add_argument("--pose2", required=True)

    parser.add_argument("--depth_scale", type=float, default=1.0)
    parser.add_argument("--cam_to_world", action="store_true", default=True,
                        help="Poses are camera-to-world (default: True, as in ScanNet/TUM/etc)")
    parser.add_argument("--world_to_cam", dest="cam_to_world", action="store_false",
                        help="Poses are world-to-camera")
    
    parser.add_argument("--visualize", action="store_true", help="Whether to visualize warping")
    parser.add_argument("--vis_out_12", type=str, default=None, help="Path to save forward warping visualization")
    parser.add_argument("--vis_out_21", type=str, default=None, help="Path to save backward warping visualization")

    args = parser.parse_args()
    main(args)