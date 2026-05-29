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
    img1_np = load_image(args.img1, as_gray=False)                         # (H,W,1) float32
    img2_np = load_image(args.img2, as_gray=False)
    # Grayscale copies for photometric metrics (consistent with photometric_consistency.py main())
    img1_gray = cv2.cvtColor(img1_np, cv2.COLOR_RGB2GRAY)  # (H,W) float32
    img2_gray = cv2.cvtColor(img2_np, cv2.COLOR_RGB2GRAY)
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

    # ── Forward pass ───────────────────────────────────────────────────────────────
    print("[eval] Running model forward pass...")
    H_img, W_img = rgb1.shape[-2:]

    K_iter    = K
    T_12_iter = T_12_init
    train_cfg = cfg.get("train", {})

    with torch.no_grad():
        if model_variant == "depth_only":
            outputs = model(
                depth1=depth_mono1,
                depth2=depth_mono2,
                T_12=T_12_iter,
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
        # Match training: min_depth=1e-3, max_depth=80.0 (see losses.py si_log_loss defaults)
        mask1 = (gt_depth1_np > 1e-3) & (gt_depth1_np < 80.0) & np.isfinite(gt_depth1_np)
        mask2 = (gt_depth2_np > 1e-3) & (gt_depth2_np < 80.0) & np.isfinite(gt_depth2_np)
        m1 = depth_metrics(pred1_full, gt_depth1_np, mask1)
        m2 = depth_metrics(pred2_full, gt_depth2_np, mask2)
        print(f"View 1: abs_rel={m1['abs_rel']:.4f}  rmse={m1['rmse']:.4f}  delta1={m1['delta1']:.4f}")
        print(f"View 2: abs_rel={m2['abs_rel']:.4f}  rmse={m2['rmse']:.4f}  delta1={m2['delta1']:.4f}")
        # depth_metrics also returns mae, delta2, delta3
        print(f"Average: abs_rel={(m1['abs_rel']+m2['abs_rel'])/2:.4f}  "
              f"rmse={(m1['rmse']+m2['rmse'])/2:.4f}  "
              f"delta1={(m1['delta1']+m2['delta1'])/2:.4f}  "
              f"delta2={(m1['delta2']+m2['delta2'])/2:.4f}  "
              f"delta3={(m1['delta3']+m2['delta3'])/2:.4f}")

        # ── Photometric consistency ──────────────────────────────────────────
        if args.pose1 and args.pose2:
            # Prefer GT intrinsics for projection; fall back to predicted
            K_eval_np = K_np if args.intrinsics else K_pred_np
            print("\n=== Photometric Consistency ===")
            ssim_12, l2_12, vr_12 = compute_photometric(
                img1_gray, img2_gray, pred1_full, K_eval_np, pose1_np, pose2_np, cam_to_world=True
            )
            ssim_21, l2_21, vr_21 = compute_photometric(
                img2_gray, img1_gray, pred2_full, K_eval_np, pose2_np, pose1_np, cam_to_world=True
            )
            print(f"SSIM (1→2): {ssim_12:.4f}  L2: {l2_12:.4f}  valid_ratio: {vr_12:.3f}")
            print(f"SSIM (2→1): {ssim_21:.4f}  L2: {l2_21:.4f}  valid_ratio: {vr_21:.3f}")
            print(f"SSIM avg:   {(ssim_12+ssim_21)/2:.4f}  L2 avg: {(l2_12+l2_21)/2:.4f}")

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