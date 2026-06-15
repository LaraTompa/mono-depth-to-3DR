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

from metrics.utils import load_intrinsics, load_pose, load_depth, load_image
from metrics.pixel_consistency import compute_pixel_consistency, project_with_depth
from metrics.depth_consistency import depth_metrics
from metrics.photometric_consistency import compute_photometric


# ─── Save outputs ───────────────────────────────────────────────────────────

def save_depth(depth, path):
    """Save depth as .npy (float32, metres)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, depth.astype(np.float32))
    print(f"  Saved: {path}")


def save_point_cloud_ply(depth, K, path, rgb=None):
    """Back-project a depth map to a 3-D point cloud and save as binary PLY.

    Parameters
    ----------
    depth : (H, W) float32 ndarray  – metric depth in metres.
    K     : (3, 3) float32 ndarray  – camera intrinsic matrix.
    path  : str                      – output .ply file path.
    rgb   : (H, W, 3) float32 ndarray in [0, 1], optional – per-pixel colour.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    H, W = depth.shape
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])

    # Build pixel grid
    u = np.arange(W, dtype=np.float32)
    v = np.arange(H, dtype=np.float32)
    uu, vv = np.meshgrid(u, v)          # (H, W)

    # Valid-depth mask
    valid = np.isfinite(depth) & (depth > 0.0)

    Z = depth[valid]
    X = (uu[valid] - cx) * Z / fx
    Y = (vv[valid] - cy) * Z / fy

    pts = np.stack([X, Y, Z], axis=1).astype(np.float32)  # (N, 3)

    has_color = rgb is not None
    if has_color:
        rgb_u8 = (np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8)
        colors = rgb_u8[valid]          # (N, 3)

    N = len(pts)

    # Write binary-little-endian PLY
    with open(path, "wb") as f:
        # Header
        header = (
            "ply\n"
            "format binary_little_endian 1.0\n"
            f"element vertex {N}\n"
            "property float x\n"
            "property float y\n"
            "property float z\n"
        )
        if has_color:
            header += (
                "property uchar red\n"
                "property uchar green\n"
                "property uchar blue\n"
            )
        header += "end_header\n"
        f.write(header.encode("ascii"))

        # Data – interleave xyz + optional rgb into a structured array
        if has_color:
            dt = np.dtype([
                ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                ("r", "u1"),  ("g", "u1"),  ("b", "u1"),
            ])
            data = np.empty(N, dtype=dt)
            data["x"], data["y"], data["z"] = pts[:, 0], pts[:, 1], pts[:, 2]
            data["r"], data["g"], data["b"] = colors[:, 0], colors[:, 1], colors[:, 2]
        else:
            dt = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4")])
            data = np.empty(N, dtype=dt)
            data["x"], data["y"], data["z"] = pts[:, 0], pts[:, 1], pts[:, 2]

        f.write(data.tobytes())

    print(f"  Saved: {path}  ({N:,} points)")


def save_combined_point_cloud_ply(depth1, depth2, K, T_2to1, path, rgb1=None, rgb2=None):
    """Back-project two depth maps and merge them in view-1 camera frame.

    Parameters
    ----------
    depth1/depth2 : (H, W) float32 ndarray  – metric depths in metres.
    K             : (3, 3) float32 ndarray  – shared camera intrinsics.
    T_2to1        : (4, 4) float32 ndarray  – rigid transform cam2 → cam1.
    path          : str                      – output .ply file path.
    rgb1/rgb2     : (H, W, 3) float32 in [0,1], optional – per-pixel colour.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])

    def _backproject(depth, rgb):
        H, W = depth.shape
        u = np.arange(W, dtype=np.float32)
        v = np.arange(H, dtype=np.float32)
        uu, vv = np.meshgrid(u, v)
        valid = np.isfinite(depth) & (depth > 0.0)
        Z = depth[valid]
        X = (uu[valid] - cx) * Z / fx
        Y = (vv[valid] - cy) * Z / fy
        pts = np.stack([X, Y, Z], axis=1).astype(np.float32)
        col = None
        if rgb is not None:
            rgb_u8 = (np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8)
            col = rgb_u8[valid]
        return pts, col

    pts1, col1 = _backproject(depth1, rgb1)
    pts2, col2 = _backproject(depth2, rgb2)

    # Transform pts2 into cam1 frame
    R = T_2to1[:3, :3].astype(np.float32)
    t = T_2to1[:3,  3].astype(np.float32)
    pts2_in_1 = (R @ pts2.T).T + t  # (N2, 3)

    pts_all = np.concatenate([pts1, pts2_in_1], axis=0)
    has_color = col1 is not None and col2 is not None
    if has_color:
        col_all = np.concatenate([col1, col2], axis=0)

    N = len(pts_all)

    with open(path, "wb") as f:
        header = (
            "ply\n"
            "format binary_little_endian 1.0\n"
            f"element vertex {N}\n"
            "property float x\n"
            "property float y\n"
            "property float z\n"
        )
        if has_color:
            header += (
                "property uchar red\n"
                "property uchar green\n"
                "property uchar blue\n"
            )
        header += "end_header\n"
        f.write(header.encode("ascii"))

        if has_color:
            dt = np.dtype([
                ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                ("r", "u1"),  ("g", "u1"),  ("b", "u1"),
            ])
            data = np.empty(N, dtype=dt)
            data["x"], data["y"], data["z"] = pts_all[:, 0], pts_all[:, 1], pts_all[:, 2]
            data["r"], data["g"], data["b"] = col_all[:, 0], col_all[:, 1], col_all[:, 2]
        else:
            dt = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4")])
            data = np.empty(N, dtype=dt)
            data["x"], data["y"], data["z"] = pts_all[:, 0], pts_all[:, 1], pts_all[:, 2]

        f.write(data.tobytes())

    print(f"  Saved: {path}  ({N:,} points, merged in cam1 frame)")


def save_visualization(depth, path, vmin=0.0, vmax=None):
    """Save depth as colorized PNG."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(path), exist_ok=True)
    # Use a full matplotlib figure with colorbar to match test_npz_read output
    fig = plt.figure(frameon=False)
    ax = fig.add_subplot(111)
    im = ax.imshow(depth, cmap='hot', interpolation='nearest', vmin=vmin, vmax=vmax)
    ax.axis('off')
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.savefig(path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


# ─── Main ───────────────────────────────────────────────────────────────────

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[eval] Device: {device}")

    # ── Load checkpoint ──────────────────────────────────────────────────────
    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    #ckpt = torch.load(args.checkpoint, map_location=device)
    # Load checkpoint onto CPU to avoid allocating all tensors on GPU immediately.
    # This prevents OOM when the saved state dict is large.
    ckpt = torch.load(args.checkpoint, map_location="cpu")  
    cfg = ckpt.get("cfg", {})
    print(f"[eval] Loaded checkpoint from epoch {ckpt.get('epoch', '?')}")

    # ── Build model ──────────────────────────────────────────────────────────
    # Extract config with fallbacks for new vs old checkpoint formats
    arch_cfg = ckpt.get("arch", {})  # new format: arch saved in checkpoint
    model_variant = arch_cfg.get("model", "v1")
    print(f"[eval] Model variant: {model_variant}")

    if model_variant == "depth_only":
        from models.model_depth_only.network import build_depth_only_net
        model = build_depth_only_net(arch_cfg)

    elif model_variant == "vista":
        from models.model_vista.network import DepthAlignNetV2
        v_cfg = arch_cfg.get("vista", {})
        model = DepthAlignNetV2(
            dino_model         = str(v_cfg.get("dino_model",          "dinov2_vitl14")),
            freeze_dino        = False,
            depth_backbone     = str(v_cfg.get("depth_backbone",       "convnext_tiny")),
            decoder_dim        = int(v_cfg.get("decoder_dim",          768)),
            num_decoder_blocks = int(v_cfg.get("num_decoder_blocks",     4)),
            num_decoder_heads  = int(v_cfg.get("num_decoder_heads",     12)),
            depth_out_channels = int(v_cfg.get("depth_out_channels",   128)),
            decoder_hidden     = int(v_cfg.get("decoder_hidden",        256)),
            camera_head_hidden = int(v_cfg.get("camera_head_hidden",   256)),
            pose_dropout       = 0.0,
            mast3r_ckpt        = None,
            freeze_cross_attn  = False,
        )

    else:  # "v1"
        from models.model_image_depth.network import DepthAlignNet
        if not arch_cfg:
            arch_cfg = {
                "encoder": {"out_channels": 64},
                "attention": {"num_heads": 4, "window_size": 7},
                "refinement": {"enabled": False, "num_iters": 4, "hidden_dim": 64},
                "decoder": {"hidden_dim": 32},
                "camera_head": {"hidden_dim": 64},
            }
            print("[eval] [WARNING] 'arch' not in checkpoint; using v1 defaults")
        enc_cfg = arch_cfg.get("encoder",    {})
        att_cfg = arch_cfg.get("attention",  {})
        ref_cfg = arch_cfg.get("refinement", {})
        dec_cfg = arch_cfg.get("decoder",    {})
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
        )

    # Extract model weights only; drop optimizer state (2× model size, not needed for eval)
    model_state = ckpt.pop("model")
    ckpt.pop("optimizer", None)
    ckpt.pop("scheduler", None)
    # cfg, arch, epoch remain in ckpt for inspection
 
    # Move model to device, then load weights
    model = model.to(device)
    model.load_state_dict(model_state)
    model.eval()
    print("[eval] Model loaded and set to eval mode.")

    # ── Load inputs ──────────────────────────────────────────────────────────
    print("[eval] Loading inputs...")
    pred_depth1_np = load_depth(args.pred_depth1, args.depth_scale_pred)  # (H,W) float32
    pred_depth2_np = load_depth(args.pred_depth2, args.depth_scale_pred)

    img1_np = img2_np = img1_gray = img2_gray = rgb1 = rgb2 = None
    if args.img1 and args.img2:
        img1_np = load_image(args.img1, as_gray=False)   # (H,W,3) float32
        img2_np = load_image(args.img2, as_gray=False)
        # Grayscale copies for photometric metrics (consistent with photometric_consistency.py main())
        img1_gray = cv2.cvtColor(img1_np, cv2.COLOR_RGB2GRAY)  # (H,W) float32
        img2_gray = cv2.cvtColor(img2_np, cv2.COLOR_RGB2GRAY)
        H, W, _ = img1_np.shape
        rgb1 = torch.from_numpy(img1_np).permute(2, 0, 1).unsqueeze(0).to(device)  # (1,3,H,W)
        rgb2 = torch.from_numpy(img2_np).permute(2, 0, 1).unsqueeze(0).to(device)
    else:
        if model_variant not in ("depth_only",):
            raise ValueError("--img1 and --img2 are required for model variant '{model_variant}'")
        # Derive H, W from the depth map (depth_only does not use RGB)
        H, W = pred_depth1_np.shape[:2]
        print(f"[eval] No images provided; using depth map dimensions H={H}, W={W}")

    # Resize predictions to match image resolution if needed
    if pred_depth1_np.shape != (H, W):
        pred_depth1_np = cv2.resize(pred_depth1_np, (W, H), interpolation=cv2.INTER_NEAREST)
    if pred_depth2_np.shape != (H, W):
        pred_depth2_np = cv2.resize(pred_depth2_np, (W, H), interpolation=cv2.INTER_NEAREST)

    # MDE sources (DepthPro / ZoeDepth) are metric estimators — pass raw metric
    # depths directly.  No normalisation needed; the decoder uses them as a
    # metric prior and geometric consistency operates in metric space.
    depth_mono1 = torch.from_numpy(pred_depth1_np).unsqueeze(0).unsqueeze(0).to(device)  # (1,1,H,W) metres
    depth_mono2 = torch.from_numpy(pred_depth2_np).unsqueeze(0).unsqueeze(0).to(device)

    # ── Intrinsics and poses ─────────────────────────────────────────────────
    if args.intrinsics:
        K_np = load_intrinsics(args.intrinsics)
    else:
        # Fallback: assume fx=fy=focal_length, cx=W/2, cy=H/2
        fx = args.focal_length if args.focal_length else 0.9 * max(H, W)  # heuristic
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
        T_12_init = torch.linalg.inv(pose2) @ pose1
    else:
        T_12_init = torch.eye(4, device=device, dtype=torch.float32).unsqueeze(0)
        print("[eval] No poses provided; using identity as initial T_12 estimate.")

    T_12_init_np = T_12_init.squeeze().cpu().numpy()  # (4,4) – used for combined PLY fallback

    # ── Forward pass ───────────────────────────────────────────────────────────────
    print("[eval] Running model forward pass...")
    H_img, W_img = (rgb1.shape[-2:] if rgb1 is not None else (H, W))

    K_iter    = K
    T_12_iter = T_12_init
    train_cfg = cfg.get("train", {})

    with torch.no_grad():
        if model_variant == "depth_only":
            outputs = model(
                depth1=depth_mono1,
                depth2=depth_mono2,
                T_12=T_12_iter,
                K=K,  # passed for optional GRU refinement; ignored when pose_refine_iters=0
            )
        else:
            num_pose_iters = int(train_cfg.get("num_pose_iters", 1))
            print(f"[eval] num_pose_iters={num_pose_iters} (from checkpoint config)")
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
                    K_pred_it = outputs.get("K_pred")
                    T_pred_it = outputs.get("T_12_pred")
                    if (K_pred_it is not None and T_pred_it is not None
                            and torch.isfinite(K_pred_it).all()
                            and torch.isfinite(T_pred_it).all()):
                        K_iter    = K_pred_it
                        T_12_iter = T_pred_it
                    else:
                        print(f"[eval] iter {pose_it}: non-finite K/T pred, keeping previous.")

    pred1_out = outputs["depth1"].squeeze().cpu().numpy()   # (pH, pW) metres
    pred2_out = outputs["depth2"].squeeze().cpu().numpy()
    print(f"[eval] Output depth range view1: "
          f"{pred1_out[pred1_out > 0].min():.3f} – {pred1_out.max():.3f} m")
    print(f"[eval] Output depth range view2: "
          f"{pred2_out[pred2_out > 0].min():.3f} – {pred2_out.max():.3f} m")

    conf1 = outputs["confidence1"].squeeze().cpu().numpy()
    conf2 = outputs["confidence2"].squeeze().cpu().numpy()

    # Create output dir early so T_12_pred.txt / K_pred.txt saves don't fail
    os.makedirs(args.output_dir, exist_ok=True)
    # Prepare a metrics log file to capture printed metrics
    metrics_log_path = os.path.join(args.output_dir, "metrics_all.txt")
    try:
        metrics_f = open(metrics_log_path, "w", encoding="utf-8")
    except Exception:
        metrics_f = None

    def log(s="", end="\n"):
        # Print to stdout and append to metrics file when available
        print(s, end=end)
        if metrics_f:
            try:
                metrics_f.write(str(s) + end)
                metrics_f.flush()
            except Exception:
                pass

    # ── Print predicted camera parameters ────────────────────────────────────
    if "T_12_pred" in outputs:
        T_12_pred_np = outputs["T_12_pred"][0].cpu().numpy()
        print(f"\n[eval] Predicted T_12 (cam1\u2192cam2):\n{T_12_pred_np}")
        np.savetxt(os.path.join(args.output_dir, "T_12_pred.txt"), T_12_pred_np, fmt="%.6f")
        print(f"  Saved: {args.output_dir}/T_12_pred.txt")
    if "K_pred" in outputs:
        K_pred_np = outputs["K_pred"][0].cpu().numpy()
        print(f"[eval] Predicted intrinsics: fx={K_pred_np[0,0]:.1f}  fy={K_pred_np[1,1]:.1f}"
              f"  cx={K_pred_np[0,2]:.1f}  cy={K_pred_np[1,2]:.1f}")
        np.savetxt(os.path.join(args.output_dir, "K_pred.txt"), K_pred_np, fmt="%.6f")
        print(f"  Saved: {args.output_dir}/K_pred.txt")

    # Upsample to full resolution for saving/evaluation
    pred1_full = cv2.resize(pred1_out, (W, H), interpolation=cv2.INTER_LINEAR)
    pred2_full = cv2.resize(pred2_out, (W, H), interpolation=cv2.INTER_LINEAR)

    # ── Save outputs ─────────────────────────────────────────────────────────
    print(f"[eval] Saving outputs to {args.output_dir}")

    save_depth(pred1_full, os.path.join(args.output_dir, "depth1_aligned.npy"))
    save_depth(pred2_full, os.path.join(args.output_dir, "depth2_aligned.npy"))
    save_depth(conf1, os.path.join(args.output_dir, "confidence1.npy"))
    save_depth(conf2, os.path.join(args.output_dir, "confidence2.npy"))

    save_visualization(pred1_full, os.path.join(args.output_dir, "depth1_aligned_vis.png"))
    save_visualization(pred2_full, os.path.join(args.output_dir, "depth2_aligned_vis.png"))
    save_visualization(conf1, os.path.join(args.output_dir, "confidence1_vis.png"), vmin=0.0, vmax=1.0)
    save_visualization(conf2, os.path.join(args.output_dir, "confidence2_vis.png"), vmin=0.0, vmax=1.0)

    # Save aligned depth point clouds
    save_point_cloud_ply(pred1_full, K_np, os.path.join(args.output_dir, "depth1_aligned.ply"),
                         rgb=img1_np)
    save_point_cloud_ply(pred2_full, K_np, os.path.join(args.output_dir, "depth2_aligned.ply"),
                         rgb=img2_np)

    # Combined aligned point cloud: both views merged in cam1 frame
    if args.pose1 and args.pose2:
        _T_2to1_aligned = np.linalg.inv(pose1_np) @ pose2_np
    elif "T_12_pred" in outputs:
        _T_2to1_aligned = np.linalg.inv(outputs["T_12_pred"][0].cpu().numpy())
    else:
        _T_2to1_aligned = np.linalg.inv(T_12_init_np)
    save_combined_point_cloud_ply(
        pred1_full, pred2_full, K_np, _T_2to1_aligned,
        os.path.join(args.output_dir, "combined_aligned.ply"),
        rgb1=img1_np, rgb2=img2_np,
    )

    # Save monocular (input) depth heatmaps/npy
    try:
        save_depth(pred_depth1_np, os.path.join(args.output_dir, "depth1_mono.npy"))
        save_depth(pred_depth2_np, os.path.join(args.output_dir, "depth2_mono.npy"))
        save_visualization(pred_depth1_np, os.path.join(args.output_dir, "depth1_mono_vis.png"))
        save_visualization(pred_depth2_np, os.path.join(args.output_dir, "depth2_mono_vis.png"))
        save_point_cloud_ply(pred_depth1_np, K_np, os.path.join(args.output_dir, "depth1_mono.ply"),
                             rgb=img1_np)
        save_point_cloud_ply(pred_depth2_np, K_np, os.path.join(args.output_dir, "depth2_mono.ply"),
                             rgb=img2_np)
    except Exception:
        print("[eval] Warning: failed to save monocular depth visualizations")

    # ── Optional: Evaluation against GT ──────────────────────────────────────
    if args.gt_depth1 and args.gt_depth2:
        print("\n[eval] Loading GT depths for evaluation...")
        gt_depth1_np = load_depth(args.gt_depth1, args.depth_scale_gt)
        gt_depth2_np = load_depth(args.gt_depth2, args.depth_scale_gt)

        if gt_depth1_np.shape != (H, W):
            gt_depth1_np = cv2.resize(gt_depth1_np, (W, H), interpolation=cv2.INTER_NEAREST)
        if gt_depth2_np.shape != (H, W):
            gt_depth2_np = cv2.resize(gt_depth2_np, (W, H), interpolation=cv2.INTER_NEAREST)

        # Save GT heatmaps
        try:
            save_depth(gt_depth1_np, os.path.join(args.output_dir, "depth1_gt.npy"))
            save_depth(gt_depth2_np, os.path.join(args.output_dir, "depth2_gt.npy"))
            save_visualization(gt_depth1_np, os.path.join(args.output_dir, "depth1_gt_vis.png"))
            save_visualization(gt_depth2_np, os.path.join(args.output_dir, "depth2_gt_vis.png"))
            save_point_cloud_ply(gt_depth1_np, K_np, os.path.join(args.output_dir, "depth1_gt.ply"),
                                 rgb=img1_np)
            save_point_cloud_ply(gt_depth2_np, K_np, os.path.join(args.output_dir, "depth2_gt.ply"),
                                 rgb=img2_np)
            # Combined GT point cloud in cam1 frame (requires poses)
            if args.pose1 and args.pose2:
                _T_2to1_gt = np.linalg.inv(pose1_np) @ pose2_np
                save_combined_point_cloud_ply(
                    gt_depth1_np, gt_depth2_np, K_np, _T_2to1_gt,
                    os.path.join(args.output_dir, "combined_gt.ply"),
                    rgb1=img1_np, rgb2=img2_np,
                )
        except Exception:
            print("[eval] Warning: failed to save GT depth visualizations")

        # ── Depth metrics ────────────────────────────────────────────────────
        log("\n=== Depth Metrics ===")
        # Match training: min_depth=1e-3, max_depth=80.0 (see losses.py si_log_loss defaults)
        mask1 = (gt_depth1_np > 1e-3) & (gt_depth1_np < 80.0) & np.isfinite(gt_depth1_np)
        mask2 = (gt_depth2_np > 1e-3) & (gt_depth2_np < 80.0) & np.isfinite(gt_depth2_np)
        m1 = depth_metrics(pred1_full, gt_depth1_np, mask1)
        m2 = depth_metrics(pred2_full, gt_depth2_np, mask2)
        log(f"View 1: abs_rel={m1['abs_rel']:.4f}  rmse={m1['rmse']:.4f}  delta1={m1['delta1']:.4f}")
        log(f"View 2: abs_rel={m2['abs_rel']:.4f}  rmse={m2['rmse']:.4f}  delta1={m2['delta1']:.4f}")
        # depth_metrics also returns mae, delta2, delta3
          log(f"Average: abs_rel={(m1['abs_rel']+m2['abs_rel'])/2:.4f}  "
              f"rmse={(m1['rmse']+m2['rmse'])/2:.4f}  "
              f"delta1={(m1['delta1']+m2['delta1'])/2:.4f}  "
              f"delta2={(m1['delta2']+m2['delta2'])/2:.4f}  "
              f"delta3={(m1['delta3']+m2['delta3'])/2:.4f}")

        # ── Monocular (pre-alignment) depth metrics ───────────────────────
        log("\n=== Monocular (pre-align) Depth Metrics ===")
        try:
            mm1 = depth_metrics(pred_depth1_np, gt_depth1_np, mask1)
            mm2 = depth_metrics(pred_depth2_np, gt_depth2_np, mask2)
            log(f"Mono View 1: abs_rel={mm1['abs_rel']:.4f}  rmse={mm1['rmse']:.4f}  delta1={mm1['delta1']:.4f}")
            log(f"Mono View 2: abs_rel={mm2['abs_rel']:.4f}  rmse={mm2['rmse']:.4f}  delta1={mm2['delta1']:.4f}")
            # Save monocular metric summary
            with open(os.path.join(args.output_dir, "metrics_monocular.txt"), "w") as f:
                f.write("Monocular pre-alignment metrics\n")
                f.write(f"View1: {mm1}\n")
                f.write(f"View2: {mm2}\n")
        except Exception:
            print("[eval] Warning: failed to compute/save monocular metrics")

        # Compute median scale ratios and save scaled monocular point clouds + combined.
        try:
            eps = 1e-6
            valid1 = mask1 if 'mask1' in locals() else ((gt_depth1_np > 1e-3) & np.isfinite(gt_depth1_np) & (pred_depth1_np > 0))
            valid2 = mask2 if 'mask2' in locals() else ((gt_depth2_np > 1e-3) & np.isfinite(gt_depth2_np) & (pred_depth2_np > 0))
            n1 = int(np.count_nonzero(valid1))
            n2 = int(np.count_nonzero(valid2))
            if n1 > 0 and n2 > 0:
                s1 = float(np.median(gt_depth1_np[valid1] / (pred_depth1_np[valid1] + eps)))
                s2 = float(np.median(gt_depth2_np[valid2] / (pred_depth2_np[valid2] + eps)))
                log(f"[eval] Monocular median scales: s1={s1:.4f}, s2={s2:.4f}")

                # Save individually scaled monocular point clouds (each scaled by its median ratio)
                pred1_scaled = pred_depth1_np * s1
                pred2_scaled = pred_depth2_np * s2
                try:
                    save_point_cloud_ply(pred1_scaled, K_np, os.path.join(args.output_dir, "depth1_mono_scaled.ply"), rgb=img1_np)
                    save_point_cloud_ply(pred2_scaled, K_np, os.path.join(args.output_dir, "depth2_mono_scaled.ply"), rgb=img2_np)
                except Exception:
                    print("[eval] Warning: failed to save individually scaled monocular PLYs")

                # For the combined two-view monocular point cloud, use the minimum ratio
                s_comb = min(s1, s2)
                pred1_comb = pred_depth1_np * s_comb
                pred2_comb = pred_depth2_np * s_comb

                # Determine transform cam2 -> cam1 (prefer GT poses, then predicted, then fallback)
                if args.pose1 and args.pose2:
                    T_2to1_mono = np.linalg.inv(pose1_np) @ pose2_np
                elif "T_12_pred" in outputs:
                    T_2to1_mono = np.linalg.inv(outputs["T_12_pred"][0].cpu().numpy())
                else:
                    T_2to1_mono = np.linalg.inv(T_12_init_np)

                try:
                    save_combined_point_cloud_ply(
                        pred1_comb, pred2_comb, K_np, T_2to1_mono,
                        os.path.join(args.output_dir, "combined_mono_before.ply"),
                        rgb1=img1_np, rgb2=img2_np,
                    )
                except Exception:
                    print("[eval] Warning: failed to save combined monocular before-alignment PLY")
                # Compute and save depth metrics for the median-scaled monocular predictions
                try:
                    mm1s = depth_metrics(pred1_scaled, gt_depth1_np, mask1)
                    mm2s = depth_metrics(pred2_scaled, gt_depth2_np, mask2)
                    log("\n=== Monocular Scaled (median) Depth Metrics ===")
                    log(f"Mono Scaled View 1: abs_rel={mm1s['abs_rel']:.4f}  rmse={mm1s['rmse']:.4f}  delta1={mm1s['delta1']:.4f}")
                    log(f"Mono Scaled View 2: abs_rel={mm2s['abs_rel']:.4f}  rmse={mm2s['rmse']:.4f}  delta1={mm2s['delta1']:.4f}")

                    # Also compute metrics for the combined (min-ratio) scaled monocular depths
                    mm1c = depth_metrics(pred1_comb, gt_depth1_np, mask1)
                    mm2c = depth_metrics(pred2_comb, gt_depth2_np, mask2)
                    log(f"Mono Combined (min ratio) View 1: abs_rel={mm1c['abs_rel']:.4f}  rmse={mm1c['rmse']:.4f}  delta1={mm1c['delta1']:.4f}")
                    log(f"Mono Combined (min ratio) View 2: abs_rel={mm2c['abs_rel']:.4f}  rmse={mm2c['rmse']:.4f}  delta1={mm2c['delta1']:.4f}")

                    # Save scaled metrics to file
                    with open(os.path.join(args.output_dir, "metrics_monocular_scaled.txt"), "w") as f:
                        f.write("Monocular scaled metrics (median scale)\n")
                        f.write(f"s1: {s1}\n")
                        f.write(f"s2: {s2}\n")
                        f.write(f"View1_scaled: {mm1s}\n")
                        f.write(f"View2_scaled: {mm2s}\n")
                        f.write("\nMonocular combined (min ratio) metrics\n")
                        f.write(f"s_comb: {s_comb}\n")
                        f.write(f"View1_combined: {mm1c}\n")
                        f.write(f"View2_combined: {mm2c}\n")
                    # If possible, compute photometric and pixel consistency for scaled monocular depths
                    if args.pose1 and args.pose2 and img1_gray is not None and img2_gray is not None:
                        # Prefer GT intrinsics for projection; fall back to predicted if available
                        K_eval_np = K_np if args.intrinsics else (K_pred_np if 'K_pred_np' in locals() else K_np)
                        try:
                            # Photometric: scaled individual views
                            ssim_s_12, l2_s_12, vr_s_12 = compute_photometric(
                                img1_gray, img2_gray, pred1_scaled, K_eval_np, pose1_np, pose2_np, cam_to_world=True
                            )
                            ssim_s_21, l2_s_21, vr_s_21 = compute_photometric(
                                img2_gray, img1_gray, pred2_scaled, K_eval_np, pose2_np, pose1_np, cam_to_world=True
                            )
                            log(f"Scaled SSIM (1→2): {ssim_s_12:.4f}  L2: {l2_s_12:.4f}  valid_ratio: {vr_s_12:.3f}")
                            log(f"Scaled SSIM (2→1): {ssim_s_21:.4f}  L2: {l2_s_21:.4f}  valid_ratio: {vr_s_21:.3f}")

                            # Photometric: combined (min-ratio) scaled
                            ssim_c_12, l2_c_12, vr_c_12 = compute_photometric(
                                img1_gray, img2_gray, pred1_comb, K_eval_np, pose1_np, pose2_np, cam_to_world=True
                            )
                            ssim_c_21, l2_c_21, vr_c_21 = compute_photometric(
                                img2_gray, img1_gray, pred2_comb, K_eval_np, pose2_np, pose1_np, cam_to_world=True
                            )
                            log(f"Scaled SSIM avg: {(ssim_s_12+ssim_s_21)/2:.4f}  L2 avg: {(l2_s_12+l2_s_21)/2:.4f}")

                            # Pixel consistency: individual scaled views
                            mae_s_12, rmse_s_12, vr_pc_s_12 = compute_pixel_consistency(
                                gt_depth1_np, pred1_scaled, gt_depth2_np, K_eval_np, pose1_np, pose2_np
                            )
                            mae_s_21, rmse_s_21, vr_pc_s_21 = compute_pixel_consistency(
                                gt_depth2_np, pred2_scaled, gt_depth1_np, K_eval_np, pose2_np, pose1_np
                            )
                            log(f"Scaled MAE  (1→2): {mae_s_12:.4f}  RMSE: {rmse_s_12:.4f}  valid_ratio: {vr_pc_s_12:.3f}")
                            log(f"Scaled MAE  (2→1): {mae_s_21:.4f}  RMSE: {rmse_s_21:.4f}  valid_ratio: {vr_pc_s_21:.3f}")

                            # Pixel consistency: combined (min-ratio) scaled
                            mae_c_12, rmse_c_12, vr_pc_c_12 = compute_pixel_consistency(
                                gt_depth1_np, pred1_comb, gt_depth2_np, K_eval_np, pose1_np, pose2_np
                            )
                            mae_c_21, rmse_c_21, vr_pc_c_21 = compute_pixel_consistency(
                                gt_depth2_np, pred2_comb, gt_depth1_np, K_eval_np, pose2_np, pose1_np
                            )

                            # Append photometric/pixel results to the scaled metrics file
                            with open(os.path.join(args.output_dir, "metrics_monocular_scaled.txt"), "a") as f:
                                f.write("\nPhotometric consistency (scaled individual views)\n")
                                f.write(f"SSIM_1to2: {ssim_s_12}, L2_1to2: {l2_s_12}, valid_ratio: {vr_s_12}\n")
                                f.write(f"SSIM_2to1: {ssim_s_21}, L2_2to1: {l2_s_21}, valid_ratio: {vr_s_21}\n")
                                f.write("\nPhotometric consistency (combined min-ratio)\n")
                                f.write(f"SSIM_1to2: {ssim_c_12}, L2_1to2: {l2_c_12}, valid_ratio: {vr_c_12}\n")
                                f.write(f"SSIM_2to1: {ssim_c_21}, L2_2to1: {l2_c_21}, valid_ratio: {vr_c_21}\n")
                                f.write("\nPixel consistency (scaled individual views)\n")
                                f.write(f"MAE_1to2: {mae_s_12}, RMSE_1to2: {rmse_s_12}, valid_ratio: {vr_pc_s_12}\n")
                                f.write(f"MAE_2to1: {mae_s_21}, RMSE_2to1: {rmse_s_21}, valid_ratio: {vr_pc_s_21}\n")
                                f.write("\nPixel consistency (combined min-ratio)\n")
                                f.write(f"MAE_1to2: {mae_c_12}, RMSE_1to2: {rmse_c_12}, valid_ratio: {vr_pc_c_12}\n")
                                f.write(f"MAE_2to1: {mae_c_21}, RMSE_2to1: {rmse_c_21}, valid_ratio: {vr_pc_c_21}\n")
                        except Exception:
                            print("[eval] Warning: failed to compute/save scaled photometric/pixel consistency")
                except Exception:
                    print("[eval] Warning: failed to compute/save scaled monocular metrics")
            else:
                print("[eval] Warning: insufficient valid pixels to compute monocular scaling or combined mono PLY")
        except Exception:
            print("[eval] Warning: failed to compute/save scaled monocular point clouds")

        # ── Photometric consistency ──────────────────────────────────────────
        if args.pose1 and args.pose2 and img1_gray is not None and img2_gray is not None:
            # Prefer GT intrinsics for projection; fall back to predicted
            K_eval_np = K_np if args.intrinsics else K_pred_np
            log("\n=== Photometric Consistency ===")
            ssim_12, l2_12, vr_12 = compute_photometric(
                img1_gray, img2_gray, pred1_full, K_eval_np, pose1_np, pose2_np, cam_to_world=True
            )
            ssim_21, l2_21, vr_21 = compute_photometric(
                img2_gray, img1_gray, pred2_full, K_eval_np, pose2_np, pose1_np, cam_to_world=True
            )
            log(f"SSIM (1→2): {ssim_12:.4f}  L2: {l2_12:.4f}  valid_ratio: {vr_12:.3f}")
            log(f"SSIM (2→1): {ssim_21:.4f}  L2: {l2_21:.4f}  valid_ratio: {vr_21:.3f}")
            log(f"SSIM avg:   {(ssim_12+ssim_21)/2:.4f}  L2 avg: {(l2_12+l2_21)/2:.4f}")

            # ── Pixel consistency ────────────────────────────────────────────
            log("\n=== Pixel Consistency ===")
            mae_12, rmse_12, vr_pc_12 = compute_pixel_consistency(
                gt_depth1_np, pred1_full, gt_depth2_np, K_eval_np, pose1_np, pose2_np
            )
            mae_21, rmse_21, vr_pc_21 = compute_pixel_consistency(
                gt_depth2_np, pred2_full, gt_depth1_np, K_eval_np, pose2_np, pose1_np
            )
            log(f"MAE  (1→2): {mae_12:.4f}  RMSE: {rmse_12:.4f}  valid_ratio: {vr_pc_12:.3f}")
            log(f"MAE  (2→1): {mae_21:.4f}  RMSE: {rmse_21:.4f}  valid_ratio: {vr_pc_21:.3f}")
            log(f"MAE  avg:   {(mae_12 + mae_21)/2:.4f}  RMSE avg: {(rmse_12 + rmse_21)/2:.4f}")

    # ── Save model scale / residual heatmaps if provided by network ─────
    # handle a few possible output keys across model variants
    def _save_map(tensor, name, out_h=H, out_w=W, vmin=None, vmax=None):
        if tensor is None:
            return
        try:
            a = tensor.squeeze().cpu().numpy()
            # Upsample to full resolution if needed
            if a.ndim == 2 and a.shape != (out_h, out_w):
                a_up = cv2.resize(a, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
            else:
                a_up = a
            save_depth(a_up, os.path.join(args.output_dir, f"{name}.npy"))
            save_visualization(a_up, os.path.join(args.output_dir, f"{name}_vis.png"), vmin=(vmin if vmin is not None else a_up.min()), vmax=(vmax if vmax is not None else a_up.max()))
        except Exception:
            print(f"[eval] Warning: failed to save map {name}")

    # scale / bias from iterative refinement (s16) or direct outputs
    # possible keys: 'scale1','bias1','scale2','bias2','log_scale1','log_scale2'
    if "scale1" in outputs:
        _save_map(outputs.get("scale1"), "scale1")
    if "scale2" in outputs:
        _save_map(outputs.get("scale2"), "scale2")
    if "bias1" in outputs:
        _save_map(outputs.get("bias1"), "bias1")
    if "bias2" in outputs:
        _save_map(outputs.get("bias2"), "bias2")
    # handle log_scale keys (compute softplus to get positive scale)
    if "log_scale1" in outputs:
        try:
            ls = outputs.get("log_scale1").squeeze().cpu().numpy()
            scale_np = np.log1p(np.exp(ls)) + 1e-4
            _save_map(torch.from_numpy(scale_np), "log_scale1_as_scale")
        except Exception:
            print("[eval] Warning: failed to process log_scale1")
    if "log_scale2" in outputs:
        try:
            ls = outputs.get("log_scale2").squeeze().cpu().numpy()
            scale_np = np.log1p(np.exp(ls)) + 1e-4
            _save_map(torch.from_numpy(scale_np), "log_scale2_as_scale")
        except Exception:
            print("[eval] Warning: failed to process log_scale2")

    # Close metrics file if open
    try:
        if metrics_f:
            metrics_f.close()
    except Exception:
        pass

    print(f"\n[eval] Done. Results saved to {args.output_dir}")


# ─── CLI ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate DepthAlignNet on an image pair.")

    # Required inputs
    parser.add_argument("--img1", default=None, help="Path to RGB image 1 (JPG/PNG) — optional for depth_only")
    parser.add_argument("--img2", default=None, help="Path to RGB image 2 — optional for depth_only")
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