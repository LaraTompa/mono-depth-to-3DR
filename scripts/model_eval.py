"""
model_eval.py — Evaluation script for DepthAlignNet on image pairs.

Usage
-----
# Minimal: only images and predicted depths (no GT evaluation)
python evaluation/model_eval.py \
  --img1 path/to/img1.jpg \
  --img2 path/to/img2.jpg \
  --pred_depth1 path/to/pred1.npz \
  --pred_depth2 path/to/pred2.npz \
  --checkpoint checkpoints/best.pt \
  --output_dir results/

# Full evaluation: with GT depths, intrinsics, poses
python evaluation/model_eval.py \
  --img1 path/to/img1.jpg \
  --img2 path/to/img2.jpg \
  --pred_depth1 path/to/pred1.npz \
  --pred_depth2 path/to/pred2.npz \
  --gt_depth1 path/to/gt1.png \
  --gt_depth2 path/to/gt2.png \
  --intrinsics path/to/intrinsic.txt \
  --pose1 path/to/pose1.txt \
  --pose2 path/to/pose2.txt \
  --checkpoint checkpoints/best.pt \
  --output_dir results/ \
  --depth_scale_gt 1000.0
"""

import argparse
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml

from models.model_image_depth.network import DepthAlignNet
from models.model_image_depth.geometry import se3_inv


# ─── Loaders ────────────────────────────────────────────────────────────────

def load_image(path, as_rgb=True):
    """Load RGB image normalized to [0,1] float32."""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {path}")
    if as_rgb:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype(np.float32) / 255.0


def load_depth(path, scale=1.0):
    """
    Load depth from .npz, .npy, or image file.
    Returns float32 array in metres.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npz":
        data = np.load(path, allow_pickle=True)
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
    elif ext == ".npy":
        depth = np.load(path).astype(np.float32)
        if depth.ndim == 3 and depth.shape[0] == 1:
            depth = depth[0]
        if depth.ndim == 3 and depth.shape[-1] in (1, 3, 4):
            depth = depth[..., 0]
        return depth
    else:
        # PNG/TIFF/JPG
        depth = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise FileNotFoundError(f"Depth file not found: {path}")
        depth = depth.astype(np.float32)
        if depth.ndim == 3:
            depth = cv2.cvtColor(depth, cv2.COLOR_BGR2GRAY)
        return depth / scale


def load_intrinsics(path):
    """Load 3×3 or 4×4 intrinsics matrix."""
    K = np.loadtxt(path, dtype=np.float32)
    if K.shape == (4, 4):
        K = K[:3, :3]
    assert K.shape == (3, 3), f"Invalid intrinsics shape: {K.shape}"
    return K


def load_pose(path):
    """Load 4×4 camera-to-world pose."""
    pose = np.loadtxt(path, dtype=np.float32)
    if pose.shape == (3, 4):
        pose = np.vstack([pose, [0, 0, 0, 1]])
    assert pose.shape == (4, 4), f"Invalid pose shape: {pose.shape}"
    return pose


# ─── Depth metrics (same as losses.py) ─────────────────────────────────────

def compute_depth_metrics(pred, gt, min_depth=1e-3, max_depth=80.0):
    """
    pred, gt: (H, W) numpy arrays in metres.
    Returns dict with abs_rel, rmse, delta1.
    """
    mask = (gt > min_depth) & (gt < max_depth) & np.isfinite(gt) & (pred > 1e-8)
    if mask.sum() < 100:
        return {"abs_rel": float("nan"), "rmse": float("nan"), "delta1": float("nan")}

    p = pred[mask]
    g = gt[mask]

    abs_rel = np.mean(np.abs(p - g) / g)
    rmse = np.sqrt(np.mean((p - g) ** 2))
    ratio = np.maximum(p / g, g / p)
    delta1 = np.mean(ratio < 1.25)

    return {"abs_rel": abs_rel, "rmse": rmse, "delta1": delta1}


# ─── Geometric projection for photometric/pixel consistency ────────────────

def project_points(depth, K, pose_src, pose_tgt):
    """
    Unproject source depth → 3D → transform → project to target.
    Returns x_proj, y_proj (float32), valid (bool).
    """
    H, W = depth.shape
    y_idx, x_idx = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')

    K_inv = np.linalg.inv(K)
    xy1 = np.stack([x_idx, y_idx, np.ones_like(x_idx)], axis=-1)  # (H,W,3)
    xyz = (K_inv @ xy1[..., None])[..., 0] * depth[..., None]     # (H,W,3)

    # cam-to-world: T = inv(pose_tgt) @ pose_src
    T = np.linalg.inv(pose_tgt) @ pose_src
    xyz_tgt = (T[:3, :3] @ xyz[..., None])[..., 0] + T[:3, 3]

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


def compute_photometric_consistency(img_src, img_tgt, depth_src, K, pose_src, pose_tgt):
    """
    Warp target → source using depth_src, compute L2 error over valid pixels.
    Returns l2, valid_ratio.
    """
    x_proj, y_proj, valid = project_points(depth_src, K, pose_src, pose_tgt)

    if np.sum(valid) < 100:
        return float("inf"), 0.0

    warped = cv2.remap(
        img_tgt, x_proj, y_proj,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )

    l2 = np.mean(((img_src - warped)[valid] ** 2))
    valid_ratio = np.mean(valid)

    return l2, valid_ratio


def compute_pixel_consistency(gt_depth_src, pred_depth_src, gt_depth_tgt, K, pose_src, pose_tgt):
    """
    Project GT depth src→tgt (p_gt), then pred depth src→tgt (p_pred).
    Compute pixel distance |p_gt - p_pred|.
    Returns mae, rmse, valid_ratio.
    """
    H, W = gt_depth_src.shape

    x_gt, y_gt, valid_gt_proj = project_points(gt_depth_src, K, pose_src, pose_tgt)
    x_gt_int = np.clip(np.round(x_gt).astype(np.int32), 0, W - 1)
    y_gt_int = np.clip(np.round(y_gt).astype(np.int32), 0, H - 1)
    valid_gt_full = valid_gt_proj & (gt_depth_tgt[y_gt_int, x_gt_int] > 0)

    x_pred, y_pred, valid_pred_proj = project_points(pred_depth_src, K, pose_src, pose_tgt)

    valid = valid_gt_full & valid_pred_proj

    if np.sum(valid) < 100:
        return float("inf"), float("inf"), 0.0

    dist = np.sqrt((x_gt - x_pred) ** 2 + (y_gt - y_pred) ** 2)
    d = dist[valid]

    mae = np.mean(d)
    rmse = np.sqrt(np.mean(d ** 2))

    return mae, rmse, np.mean(valid)


# ─── Save outputs ───────────────────────────────────────────────────────────

def save_depth(depth, path):
    """Save depth as .npy (float32, metres)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, depth.astype(np.float32))
    print(f"  Saved: {path}")


def save_visualization(depth, path, vmin=0.0, vmax=10.0):
    """Save depth as colorized PNG."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.imsave(path, depth, cmap='plasma', vmin=vmin, vmax=vmax)
    print(f"  Saved: {path}")


# ─── Main ───────────────────────────────────────────────────────────────────

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[eval] Device: {device}")

    # ── Load checkpoint ──────────────────────────────────────────────────────
    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg = ckpt.get("cfg", {})
    print(f"[eval] Loaded checkpoint from epoch {ckpt.get('epoch', '?')}")

    # ── Build model ──────────────────────────────────────────────────────────
    # Extract config with fallbacks for new vs old checkpoint formats
    arch_cfg = cfg.get("arch", {})  # new format: arch saved in checkpoint
    if not arch_cfg:
        # old format: manually specify defaults
        arch_cfg = {
            "encoder": {"out_channels": 64},
            "attention": {"num_heads": 4, "window_size": 7},
            "refinement": {"enabled": False, "num_iters": 4, "hidden_dim": 64},
            "decoder": {"hidden_dim": 32},
            "camera_head": {"hidden_dim": 64},
        }
        print("[eval] [WARNING] 'arch' not in checkpoint; using defaults")
 
    enc_cfg = arch_cfg.get("encoder", {})
    att_cfg = arch_cfg.get("attention", {})
    ref_cfg = arch_cfg.get("refinement", {})
    dec_cfg = arch_cfg.get("decoder", {})
    cam_cfg = arch_cfg.get("camera_head", {})
 
    model = DepthAlignNet(
        feat_dim            = int(enc_cfg.get("out_channels", 64)),
        hidden_dim          = int(ref_cfg.get("hidden_dim", 64)),
        num_iters           = int(ref_cfg.get("num_iters", 4)),
        num_heads           = int(att_cfg.get("num_heads", 4)),
        window_size         = int(att_cfg.get("window_size", 7)),
        pretrained          = False,
        freeze_backbone     = False,
        use_refinement      = bool(ref_cfg.get("enabled", False)),
        decoder_hidden      = int(dec_cfg.get("hidden_dim", 32)),
        camera_head_hidden  = int(cam_cfg.get("hidden_dim", 64)),
    ).to(device)

    model.load_state_dict(ckpt["model"])
    model.eval()
    print("[eval] Model loaded and set to eval mode.")

    # ── Load inputs ──────────────────────────────────────────────────────────
    print("[eval] Loading inputs...")
    img1_np = load_image(args.img1, as_rgb=True)                         # (H,W,3) float32
    img2_np = load_image(args.img2, as_rgb=True)
    pred_depth1_np = load_depth(args.pred_depth1, args.depth_scale_pred)  # (H,W) float32
    pred_depth2_np = load_depth(args.pred_depth2, args.depth_scale_pred)

    H, W, _ = img1_np.shape

    # Resize predictions to match image resolution if needed
    if pred_depth1_np.shape != (H, W):
        pred_depth1_np = cv2.resize(pred_depth1_np, (W, H), interpolation=cv2.INTER_NEAREST)
    if pred_depth2_np.shape != (H, W):
        pred_depth2_np = cv2.resize(pred_depth2_np, (W, H), interpolation=cv2.INTER_NEAREST)

    # Convert to torch tensors (B=1)
    rgb1 = torch.from_numpy(img1_np).permute(2, 0, 1).unsqueeze(0).to(device)           # (1,3,H,W)
    rgb2 = torch.from_numpy(img2_np).permute(2, 0, 1).unsqueeze(0).to(device)
    depth_mono1 = torch.from_numpy(pred_depth1_np).unsqueeze(0).unsqueeze(0).to(device)  # (1,1,H,W)
    depth_mono2 = torch.from_numpy(pred_depth2_np).unsqueeze(0).unsqueeze(0).to(device)

    # ── Intrinsics and poses ─────────────────────────────────────────────────
    if args.intrinsics:
        K_np = load_intrinsics(args.intrinsics)
    else:
        # Fallback: assume fx=fy=focal_length, cx=W/2, cy=H/2
        fx = args.focal_length if args.focal_length else 0.8 * W  # heuristic
        K_np = np.array([
            [fx,  0.0, W / 2.0],
            [0.0, fx,  H / 2.0],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)
        print(f"[eval] Using fallback intrinsics: fx={fx:.1f}, cx={W/2:.1f}, cy={H/2:.1f}")

    K = torch.from_numpy(K_np).unsqueeze(0).to(device)  # (1,3,3)

    if args.pose1 and args.pose2:
        pose1_np = load_pose(args.pose1)
        pose2_np = load_pose(args.pose2)
        pose1 = torch.from_numpy(pose1_np).unsqueeze(0).to(device)  # (1,4,4)
        pose2 = torch.from_numpy(pose2_np).unsqueeze(0).to(device)
        # SE(3) closed-form inverse — no gradient instability
        T_12_init = se3_inv(pose2) @ pose1 
    else:
        # Identity transform (assume images already aligned)
        T_12_init = torch.eye(4, device=device, dtype=torch.float32).unsqueeze(0)
        print("[eval] No poses provided; using identity as initial T_12 estimate.")

    # ── Forward pass ─────────────────────────────────────────────────────────
    print("[eval] Running model forward pass...")
    H_img, W_img = rgb1.shape[-2:]
 
    # Build initial K estimate (same heuristic as train.py)
    if args.intrinsics:
        K_iter = K  # use loaded intrinsics as initial estimate
    else:
        f_init = float(max(H_img, W_img)) * 0.9
        K_iter = torch.tensor(
            [[f_init, 0.0,    W_img / 2.0],
             [0.0,    f_init, H_img / 2.0],
             [0.0,    0.0,    1.0]],
            dtype=torch.float32, device=device,
        ).unsqueeze(0)
 
    T_12_iter = T_12_init
    # Get num_pose_iters from checkpoint config (not CLI arg anymore)
    train_cfg = cfg.get("train", {})
    num_pose_iters = int(train_cfg.get("num_pose_iters", 1))
    print(f"[eval] num_pose_iters={num_pose_iters} (from checkpoint config)")
 
    with torch.no_grad():
        for pose_it in range(num_pose_iters):
            outputs = model(
                rgb1=rgb1,
                rgb2=rgb2,
                depth_mono1=depth_mono1,
                depth_mono2=depth_mono2,
                T_12=T_12_iter,
                K=K_iter,
            )
            if pose_it < num_pose_iters - 1:
                K_pred_it = outputs["K_pred"]
                T_pred_it = outputs["T_12_pred"]
                if torch.isfinite(K_pred_it).all() and torch.isfinite(T_pred_it).all():
                    K_iter    = K_pred_it
                    T_12_iter = T_pred_it
                else:
                    print(f"[eval] iter {pose_it}: non-finite K/T pred, keeping previous.")

    pred1_out = outputs["depth1"].squeeze(0).squeeze(0).cpu().numpy()  # (H/2, W/2)
    pred2_out = outputs["depth2"].squeeze(0).squeeze(0).cpu().numpy()
    conf1 = outputs["confidence1"].squeeze(0).squeeze(0).cpu().numpy()
    conf2 = outputs["confidence2"].squeeze(0).squeeze(0).cpu().numpy()

    # ── Print predicted camera parameters ────────────────────────────────────
    K_pred_np    = outputs["K_pred"][0].cpu().numpy()
    T_12_pred_np = outputs["T_12_pred"][0].cpu().numpy()
    print(f"\n[eval] Predicted intrinsics: fx={K_pred_np[0,0]:.1f}  fy={K_pred_np[1,1]:.1f}"
          f"  cx={K_pred_np[0,2]:.1f}  cy={K_pred_np[1,2]:.1f}")
    print(f"[eval] Predicted T_12 (cam1→cam2):\n{T_12_pred_np}")
 
    # Save predicted poses and intrinsics
    np.savetxt(os.path.join(args.output_dir, "K_pred.txt"),    K_pred_np,    fmt="%.6f")
    np.savetxt(os.path.join(args.output_dir, "T_12_pred.txt"), T_12_pred_np, fmt="%.6f")
    print(f"  Saved: {args.output_dir}/K_pred.txt")
    print(f"  Saved: {args.output_dir}/T_12_pred.txt")

    # Upsample to full resolution for saving/evaluation
    pred1_full = cv2.resize(pred1_out, (W, H), interpolation=cv2.INTER_LINEAR)
    # Upsample to full resolution for saving/evaluation
    pred1_full = cv2.resize(pred1_out, (W, H), interpolation=cv2.INTER_LINEAR)
    pred2_full = cv2.resize(pred2_out, (W, H), interpolation=cv2.INTER_LINEAR)

    # ── Save outputs ─────────────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"[eval] Saving outputs to {args.output_dir}")

    save_depth(pred1_full, os.path.join(args.output_dir, "depth1_aligned.npy"))
    save_depth(pred2_full, os.path.join(args.output_dir, "depth2_aligned.npy"))
    save_depth(conf1, os.path.join(args.output_dir, "confidence1.npy"))
    save_depth(conf2, os.path.join(args.output_dir, "confidence2.npy"))

    save_visualization(pred1_full, os.path.join(args.output_dir, "depth1_aligned_vis.png"))
    save_visualization(pred2_full, os.path.join(args.output_dir, "depth2_aligned_vis.png"))
    save_visualization(conf1, os.path.join(args.output_dir, "confidence1_vis.png"), vmin=0.0, vmax=1.0)
    save_visualization(conf2, os.path.join(args.output_dir, "confidence2_vis.png"), vmin=0.0, vmax=1.0)

    # ── Optional: Evaluation against GT ──────────────────────────────────────
    if args.gt_depth1 and args.gt_depth2:
        print("\n[eval] Loading GT depths for evaluation...")
        gt_depth1_np = load_depth(args.gt_depth1, args.depth_scale_gt)
        gt_depth2_np = load_depth(args.gt_depth2, args.depth_scale_gt)

        if gt_depth1_np.shape != (H, W):
            gt_depth1_np = cv2.resize(gt_depth1_np, (W, H), interpolation=cv2.INTER_NEAREST)
        if gt_depth2_np.shape != (H, W):
            gt_depth2_np = cv2.resize(gt_depth2_np, (W, H), interpolation=cv2.INTER_NEAREST)

        # ── Depth metrics ────────────────────────────────────────────────────
        print("\n=== Depth Metrics ===")
        m1 = compute_depth_metrics(pred1_full, gt_depth1_np)
        m2 = compute_depth_metrics(pred2_full, gt_depth2_np)
        print(f"View 1: abs_rel={m1['abs_rel']:.4f}  rmse={m1['rmse']:.4f}  delta1={m1['delta1']:.4f}")
        print(f"View 2: abs_rel={m2['abs_rel']:.4f}  rmse={m2['rmse']:.4f}  delta1={m2['delta1']:.4f}")
        print(f"Average: abs_rel={(m1['abs_rel']+m2['abs_rel'])/2:.4f}  "
              f"rmse={(m1['rmse']+m2['rmse'])/2:.4f}  "
              f"delta1={(m1['delta1']+m2['delta1'])/2:.4f}")

        # ── Photometric consistency ──────────────────────────────────────────
        if args.pose1 and args.pose2:
            # Prefer GT intrinsics for projection; fall back to predicted
            K_eval_np = K_np if args.intrinsics else K_pred_np
            print("\n=== Photometric Consistency ===")
            l2_12, vr_12 = compute_photometric_consistency(
                img1_np, img2_np, pred1_full, K_eval_np, pose1_np, pose2_np
            )
            l2_21, vr_21 = compute_photometric_consistency(
                img2_np, img1_np, pred2_full, K_eval_np, pose2_np, pose1_np
            )
            print(f"L2 (1→2): {l2_12:.4f}  valid_ratio: {vr_12:.3f}")
            print(f"L2 (2→1): {l2_21:.4f}  valid_ratio: {vr_21:.3f}")
            print(f"L2 avg:   {(l2_12 + l2_21)/2:.4f}")

            # ── Pixel consistency ────────────────────────────────────────────
            print("\n=== Pixel Consistency ===")
            mae_12, rmse_12, vr_pc_12 = compute_pixel_consistency(
                gt_depth1_np, pred1_full, gt_depth2_np, K_eval_np, pose1_np, pose2_np
            )
            mae_21, rmse_21, vr_pc_21 = compute_pixel_consistency(
                gt_depth2_np, pred2_full, gt_depth1_np, K_eval_np, pose2_np, pose1_np
            )
            print(f"MAE  (1→2): {mae_12:.4f}  RMSE: {rmse_12:.4f}  valid_ratio: {vr_pc_12:.3f}")
            print(f"MAE  (2→1): {mae_21:.4f}  RMSE: {rmse_21:.4f}  valid_ratio: {vr_pc_21:.3f}")
            print(f"MAE  avg:   {(mae_12 + mae_21)/2:.4f}  RMSE avg: {(rmse_12 + rmse_21)/2:.4f}")

    print(f"\n[eval] Done. Results saved to {args.output_dir}")


# ─── CLI ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate DepthAlignNet on an image pair.")

    # Required inputs
    parser.add_argument("--img1", required=True, help="Path to RGB image 1 (JPG/PNG)")
    parser.add_argument("--img2", required=True, help="Path to RGB image 2")
    parser.add_argument("--pred_depth1", required=True, help="Path to predicted depth 1 (.npy/.npz/PNG)")
    parser.add_argument("--pred_depth2", required=True, help="Path to predicted depth 2")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint (best.pt)")
    parser.add_argument("--output_dir", required=True, help="Directory to save outputs")

    # Optional: GT for evaluation
    parser.add_argument("--gt_depth1", default=None, help="Path to GT depth 1 (for metrics)")
    parser.add_argument("--gt_depth2", default=None, help="Path to GT depth 2")
    parser.add_argument("--intrinsics", default=None, help="Path to 3×3 or 4×4 intrinsics file")
    parser.add_argument("--pose1", default=None, help="Path to 4×4 camera-to-world pose 1")
    parser.add_argument("--pose2", default=None, help="Path to 4×4 camera-to-world pose 2")

    # Depth scales
    parser.add_argument("--depth_scale_pred", type=float, default=1.0,
                        help="Scale for predicted depths (default: 1.0 for metres)")
    parser.add_argument("--depth_scale_gt", type=float, default=1.0,
                        help="Scale for GT depths (e.g. 1000.0 for ScanNet mm→m)")

    # Fallback intrinsics
    parser.add_argument("--focal_length", type=float, default=None,
                        help="Fallback focal length if intrinsics not provided (default: 0.8*W)")

    args = parser.parse_args()
    main(args)