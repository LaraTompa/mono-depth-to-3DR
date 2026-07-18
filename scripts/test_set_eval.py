"""
test_set_eval.py — Full test-set evaluation of DepthAlignNet.

Iterates over every scene in a test dataset directory, runs model inference
on consecutive frame pairs, computes depth / pixel-consistency /
geometric-consistency / photometric-consistency metrics (and the same for
median-scaled monocular depths), and writes:

  <output_dir>/
    metrics_per_pair.csv        – one row per evaluated frame pair
    metrics_summary.txt         – mean, median, std for each numeric column
    best_worst/                 – visualizations of best & worst pairs by
                                  pixel-consistency MAE (aligned model output)

Dataset layout expected
-----------------------
  <data_dir>/
    batch<N>/
      sample<M>/
        <scene_id>/
          color/           <fid>.png
          depth/           <fid>.npy        (GT depth, metres, shape (1,H,W))
          intrinsic/       intrinsic_color.txt
          pose/            <fid>.txt        (4×4 cam-to-world)
          <pred_dir>/      <fid>.png        (MDE prior, uint16 mm by default)

Usage
-----
python scripts/test_set_eval.py \\
  --checkpoint checkpoints/best.pt \\
  --output_dir eval_out/test_set/ \\
  --data_dir datasets_test/sampled_data \\
  [--max_pairs 200] [--window 1] [--depth_scale_pred 1000.0]
"""

import argparse
import csv
import glob
import heapq
import itertools
import os
import sys
from collections import defaultdict

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from metrics.utils import load_intrinsics, load_pose, load_depth, load_image
from metrics.depth_consistency import depth_metrics
from metrics.pixel_consistency import compute_pixel_consistency
from metrics.photometric_consistency import compute_photometric

EPS = 1e-6


def _nanmean2(a, b):
    vals = np.array([a, b], dtype=np.float64)
    finite = vals[np.isfinite(vals)]
    if finite.size == 0:
        return float("nan")
    return float(finite.mean())


def normalize_depth_map(depth: torch.Tensor, params: dict | None) -> torch.Tensor:
    """Apply affine depth normalization while preserving invalid zeros."""
    if not params:
        return depth
    scale = float(params.get("scale", 1.0))
    offset = float(params.get("offset", 0.0))
    if scale == 0.0:
        raise ValueError("depth_normalization scale must be non-zero")
    valid = depth > 0
    depth_n = (depth - offset) / scale
    return torch.where(valid, depth_n.clamp(min=0.0), depth)


def denormalize_depth_map(depth: torch.Tensor, params: dict | None) -> torch.Tensor:
    """Invert normalize_depth_map for positive predicted depths."""
    if not params:
        return depth
    scale = float(params.get("scale", 1.0))
    offset = float(params.get("offset", 0.0))
    return depth * scale + offset

# ─── Dataset helpers ─────────────────────────────────────────────────────────

def find_scenes(data_dir, max_batches=None):
    """Return list of (batch, sample, scene_id, scene_path) for every scene."""
    scenes = []
    batch_dirs = sorted(glob.glob(os.path.join(data_dir, "batch*")))
    if max_batches:
        batch_dirs = batch_dirs[:max_batches]
    for bd in batch_dirs:
        b_name = os.path.basename(bd)
        for sd in sorted(glob.glob(os.path.join(bd, "sample*"))):
            s_name = os.path.basename(sd)
            for sc in sorted(d for d in glob.glob(os.path.join(sd, "*"))
                              if os.path.isdir(d)):
                scenes.append((b_name, s_name, os.path.basename(sc), sc))
    return scenes


def load_pred_depth_zoe(path, scale):
    """Load a ZoeDepth uint16 PNG (millimetres) → float32 metres."""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    return img.astype(np.float32) / scale


def load_gt_depth(path):
    """Load GT depth .npy (may have a leading size-1 dim) → (H,W) float32."""
    d = np.load(path).astype(np.float32)
    while d.ndim > 2 and d.shape[0] == 1:
        d = d[0]
    return d


# ─── Model helpers ───────────────────────────────────────────────────────────

def build_model(ckpt, device):
    """Reconstruct the network from checkpoint metadata."""
    arch_cfg = ckpt.get("arch", {})
    model_variant = arch_cfg.get("model", "v1")

    if model_variant == "depth_only":
        from models.model_depth_only.network import build_depth_only_net
        model = build_depth_only_net(arch_cfg)

    elif model_variant == "vista":
        from models.model_vista.network import DepthAlignNetV2
        v = arch_cfg.get("vista", {})
        model = DepthAlignNetV2(
            dino_model         = str(v.get("dino_model",          "dinov2_vitl14")),
            freeze_dino        = False,
            depth_backbone     = str(v.get("depth_backbone",       "convnext_tiny")),
            decoder_dim        = int(v.get("decoder_dim",          768)),
            num_decoder_blocks = int(v.get("num_decoder_blocks",     4)),
            num_decoder_heads  = int(v.get("num_decoder_heads",     12)),
            depth_out_channels = int(v.get("depth_out_channels",   128)),
            decoder_hidden     = int(v.get("decoder_hidden",        256)),
            camera_head_hidden = int(v.get("camera_head_hidden",   256)),
            pose_dropout       = 0.0,
            mast3r_ckpt        = None,
            freeze_cross_attn  = False,
        )

    else:  # "v1"
        from models.model_image_depth.network import DepthAlignNet
        if not arch_cfg:
            arch_cfg = {
                "encoder":     {"out_channels": 64},
                "attention":   {"num_heads": 4, "window_size": 7},
                "refinement":  {"enabled": False, "num_iters": 4, "hidden_dim": 64},
                "decoder":     {"hidden_dim": 32},
                "camera_head": {"hidden_dim": 64},
            }
        enc = arch_cfg.get("encoder",    {})
        att = arch_cfg.get("attention",  {})
        ref = arch_cfg.get("refinement", {})
        dec = arch_cfg.get("decoder",    {})
        cam = arch_cfg.get("camera_head", {})
        model = DepthAlignNet(
            feat_dim           = int(enc.get("out_channels", 64)),
            hidden_dim         = int(ref.get("hidden_dim", 64)),
            num_iters          = int(ref.get("num_iters", 4)),
            num_heads          = int(att.get("num_heads", 4)),
            window_size        = int(att.get("window_size", 7)),
            pretrained         = False,
            freeze_backbone    = False,
            use_refinement     = bool(ref.get("enabled", False)),
            decoder_hidden     = int(dec.get("hidden_dim", 32)),
            camera_head_hidden = int(cam.get("hidden_dim", 64)),
        )

    model = model.to(device)

    # ── Backward-compatible state-dict loading ────────────────────────────
    # Old depth_only checkpoints used a different decoder output head:
    #   head_scale + head_resid  (scale * depth_prior + residual)
    # instead of the current:
    #   head_resid_xyz           (XYZ point-map residual)
    #
    # When these old keys are detected we swap in LegacyDepthDecoder — which
    # has the *exact* old architecture — and set predict_depth_map=True so the
    # network's forward returns {"depth1", "depth2"} rather than point maps.
    # All trained weights (encoder, cross-attention, decoder backbone) load
    # with strict=True and nothing is discarded.
    sd = ckpt["model"]
    if "decoder.head_scale.weight" in sd:
        from models.model_depth_only.decoder import LegacyDepthDecoder
        c = arch_cfg.get("depth_only", arch_cfg)
        legacy_dec = LegacyDepthDecoder(
            token_dim = int(c.get("token_dim",      768)),
            skip_dim  = int(c.get("feature_dim",    256)),
            hidden    = int(c.get("decoder_hidden", 128)),
        ).to(device)
        model.decoder = legacy_dec
        model.predict_depth_map = True
        print(
            "[build_model] Old-style depth decoder detected "
            "(head_scale / head_resid).  Using LegacyDepthDecoder — "
            "all trained weights loaded with strict=True."
        )

    model.load_state_dict(sd)

    model.eval()
    return model, model_variant, arch_cfg


@torch.no_grad()
def run_inference(model, model_variant, cfg, rgb1_t, rgb2_t,
                  depth_mono1_t, depth_mono2_t, K_t, T_12_init_t, device):
    """Run one forward pass. Returns the raw outputs dict."""
    train_cfg = cfg.get("train", {})
    if model_variant == "depth_only":
        _use_img_enc = bool(getattr(model, "use_image_encoder", False))
        _pt_norm_cfg = cfg.get("point_normalization", {})
        _pt_scale = float(_pt_norm_cfg.get("scale", 1.0)) if _pt_norm_cfg.get("enabled", False) else None
        return model(
            depth1=depth_mono1_t,
            depth2=depth_mono2_t,
            T_12=T_12_init_t,
            K=K_t,
            rgb1=rgb1_t if _use_img_enc else None,
            rgb2=rgb2_t if _use_img_enc else None,
            point_norm_scale=_pt_scale,
        )

    num_pose_iters = int(train_cfg.get("num_pose_iters", 1))
    K_iter    = K_t
    T_12_iter = T_12_init_t
    outputs   = None
    for it in range(num_pose_iters):
        outputs = model(
            rgb1=rgb1_t, rgb2=rgb2_t,
            depth_mono1=depth_mono1_t, depth_mono2=depth_mono2_t,
            T_12=T_12_iter, K=K_iter,
        )
        if it < num_pose_iters - 1:
            K_p  = outputs.get("K_pred")
            T_p  = outputs.get("T_12_pred")
            if (K_p is not None and T_p is not None
                    and torch.isfinite(K_p).all()
                    and torch.isfinite(T_p).all()):
                K_iter    = K_p
                T_12_iter = T_p
    return outputs


# ─── Metric helpers ──────────────────────────────────────────────────────────

_DEPTH_MIN = 1e-3
_DEPTH_MAX = 80.0


def _depth_mask(gt):
    return (gt > _DEPTH_MIN) & (gt < _DEPTH_MAX) & np.isfinite(gt)


def _safe_depth_metrics(pred, gt):
    mask = _depth_mask(gt) & (pred > 0) & np.isfinite(pred)
    if mask.sum() < 10:
        return {k: float("nan") for k in
                ("abs_rel", "rmse", "mae", "delta1", "delta2", "delta3")}
    return {k: float(v) for k, v in depth_metrics(pred, gt, mask).items()}


def _safe_pixel_consistency(gt_src, pred_src, gt_tgt, K, pose_src, pose_tgt):
    try:
        mae, rmse, vr = compute_pixel_consistency(
            gt_src, pred_src, gt_tgt, K, pose_src, pose_tgt, cam_to_world=True
        )
        return float(mae or float("nan")), float(rmse or float("nan")), float(vr or float("nan"))
    except Exception:
        return float("nan"), float("nan"), float("nan")


def _compute_geometric_error(gt_src, pred_tgt, K, pose_src, pose_tgt):
    """Mirror training geometric_consistency_loss as an eval metric in metres."""
    H, W = gt_src.shape
    y_idx, x_idx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")

    K_inv = np.linalg.inv(K)
    xy1 = np.stack([x_idx, y_idx, np.ones_like(x_idx)], axis=-1).astype(np.float32)
    pts_src = (K_inv @ xy1[..., None])[..., 0] * gt_src[..., None]

    T_12 = np.linalg.inv(pose_tgt) @ pose_src
    pts_tgt = (T_12[:3, :3] @ pts_src[..., None])[..., 0] + T_12[:3, 3]

    z_tgt = pts_tgt[..., 2]
    proj = (K @ pts_tgt[..., None])[..., 0]
    u_tgt = proj[..., 0] / (z_tgt + EPS)
    v_tgt = proj[..., 1] / (z_tgt + EPS)

    pred_tgt = pred_tgt.astype(np.float32)
    sampled_tgt = cv2.remap(
        pred_tgt,
        u_tgt.astype(np.float32),
        v_tgt.astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    uv1_tgt = np.stack([u_tgt, v_tgt, np.ones_like(u_tgt)], axis=-1).astype(np.float32)
    pts_tgt_from_pred = (K_inv @ uv1_tgt[..., None])[..., 0] * sampled_tgt[..., None]
    dist = np.sqrt(np.sum((pts_tgt - pts_tgt_from_pred) ** 2, axis=-1))

    valid = (
        _depth_mask(gt_src) &
        (z_tgt > 0) &
        (u_tgt >= 0) & (u_tgt < W) &
        (v_tgt >= 0) & (v_tgt < H) &
        (sampled_tgt > _DEPTH_MIN) & (sampled_tgt < _DEPTH_MAX) &
        np.isfinite(dist)
    )

    n_valid = int(valid.sum())
    if n_valid < 100:
        return float("nan"), 0.0

    return float(dist[valid].mean()), float(valid.mean())


def _safe_geometric_error(gt_src, pred_tgt, K, pose_src, pose_tgt):
    try:
        err, vr = _compute_geometric_error(gt_src, pred_tgt, K, pose_src, pose_tgt)
        return float(err), float(vr)
    except Exception:
        return float("nan"), float("nan")


def _safe_photometric(img_src, img_tgt, depth, K, pose_src, pose_tgt):
    try:
        ssim, l2, vr = compute_photometric(
            img_src, img_tgt, depth, K, pose_src, pose_tgt, cam_to_world=True
        )
        return float(ssim or float("nan")), float(l2 or float("nan")), float(vr or float("nan"))
    except Exception:
        return float("nan"), float("nan"), float("nan")


def _median_scale(pred, gt):
    mask = _depth_mask(gt) & (pred > 0) & np.isfinite(pred)
    if mask.sum() < 10:
        return 1.0
    return float(np.median(gt[mask] / (pred[mask] + EPS)))


# ─── Visualization ───────────────────────────────────────────────────────────

def save_pair_visualization(
    img1_np, img2_np,
    gt1, gt2,
    pred1_aligned, pred2_aligned,
    pred1_mono_scaled, pred2_mono_scaled,
    conf1, conf2,
    title, out_path,
):
    """Save a 4-row comparison figure for one frame pair."""
    nrows, ncols = 4, 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(10, 16))
    fig.suptitle(title, fontsize=9, wrap=True)

    def _show(ax, data, label, cmap="hot", vmin=None, vmax=None):
        if data is None:
            ax.axis("off")
            ax.set_title(label, fontsize=7)
            return
        if data.ndim == 3:  # RGB
            ax.imshow(np.clip(data, 0, 1))
        else:
            vmin_ = vmin if vmin is not None else np.nanpercentile(data[data > 0], 2) if (data > 0).any() else 0
            vmax_ = vmax if vmax is not None else np.nanpercentile(data[data > 0], 98) if (data > 0).any() else 1
            im = ax.imshow(data, cmap=cmap, vmin=vmin_, vmax=vmax_)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(label, fontsize=7)
        ax.axis("off")

    # Compute shared depth range for fair comparison
    all_depths = [d for d in [gt1, gt2, pred1_aligned, pred2_aligned] if d is not None]
    valid_vals = np.concatenate([d[d > 0].ravel() for d in all_depths if (d > 0).any()])
    d_vmin = float(np.percentile(valid_vals, 2))  if len(valid_vals) else 0.0
    d_vmax = float(np.percentile(valid_vals, 98)) if len(valid_vals) else 10.0

    _show(axes[0, 0], img1_np, "RGB Frame 1")
    _show(axes[0, 1], img2_np, "RGB Frame 2")

    _show(axes[1, 0], gt1,           "GT Depth 1",      vmin=d_vmin, vmax=d_vmax)
    _show(axes[1, 1], gt2,           "GT Depth 2",      vmin=d_vmin, vmax=d_vmax)

    _show(axes[2, 0], pred1_aligned, "Aligned Depth 1", vmin=d_vmin, vmax=d_vmax)
    _show(axes[2, 1], pred2_aligned, "Aligned Depth 2", vmin=d_vmin, vmax=d_vmax)

    _show(axes[3, 0], pred1_mono_scaled, "Mono Scaled Depth 1", vmin=d_vmin, vmax=d_vmax)
    _show(axes[3, 1], pred2_mono_scaled, "Mono Scaled Depth 2", vmin=d_vmin, vmax=d_vmax)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ─── CSV helpers ─────────────────────────────────────────────────────────────

# Ordered list of all CSV columns (after the identifier columns)
_METRIC_COLS = [
    # ── aligned model output ──────────────────────────────────────────────
    "abs_rel_v1",  "rmse_v1",  "mae_v1",  "delta1_v1",  "delta2_v1",  "delta3_v1",
    "abs_rel_v2",  "rmse_v2",  "mae_v2",  "delta1_v2",  "delta2_v2",  "delta3_v2",
    "pc_mae_12",   "pc_rmse_12",  "pc_vr_12",
    "pc_mae_21",   "pc_rmse_21",  "pc_vr_21",
    "pc_mae_avg",  "pc_rmse_avg", "pc_vr_avg",
    "geo_12", "geo_vr_12",
    "geo_21", "geo_vr_21",
    "geo_avg", "geo_vr_avg",
    "photo_ssim_12", "photo_l2_12", "photo_vr_12",
    "photo_ssim_21", "photo_l2_21", "photo_vr_21",
    "photo_ssim_avg", "photo_l2_avg", "photo_vr_avg",
    # ── monocular (pre-alignment, scaled) ─────────────────────────────────
    "mono_scale1", "mono_scale2",
    "mono_abs_rel_v1", "mono_rmse_v1", "mono_mae_v1",
    "mono_delta1_v1", "mono_delta2_v1", "mono_delta3_v1",
    "mono_abs_rel_v2", "mono_rmse_v2", "mono_mae_v2",
    "mono_delta1_v2", "mono_delta2_v2", "mono_delta3_v2",
    "mono_pc_mae_12",  "mono_pc_rmse_12",  "mono_pc_vr_12",
    "mono_pc_mae_21",  "mono_pc_rmse_21",  "mono_pc_vr_21",
    "mono_pc_mae_avg", "mono_pc_rmse_avg", "mono_pc_vr_avg",
    "mono_geo_12", "mono_geo_vr_12",
    "mono_geo_21", "mono_geo_vr_21",
    "mono_geo_avg", "mono_geo_vr_avg",
    "mono_photo_ssim_12", "mono_photo_l2_12", "mono_photo_vr_12",
    "mono_photo_ssim_21", "mono_photo_l2_21", "mono_photo_vr_21",
    "mono_photo_ssim_avg", "mono_photo_l2_avg", "mono_photo_vr_avg",
    # ── predicted camera params (if available) ────────────────────────────
    "pred_fx", "pred_fy", "pred_cx", "pred_cy",
    "gt_fx",   "gt_fy",   "gt_cx",   "gt_cy",
]

_ID_COLS = ["batch", "sample", "scene", "frame1", "frame2"]
_ALL_COLS = _ID_COLS + _METRIC_COLS


def _nan_row():
    return {c: float("nan") for c in _METRIC_COLS}


def _row_mean(row, *keys):
    vals = [row.get(k) for k in keys]
    finite = [float(v) for v in vals if isinstance(v, (int, float, np.floating)) and np.isfinite(v)]
    if not finite:
        return float("nan")
    return float(np.mean(finite))


def _save_correlation_plots(all_rows, output_dir):
    corr_dir = os.path.join(output_dir, "correlations")
    os.makedirs(corr_dir, exist_ok=True)

    specs = [
        {
            "filename": "pixel_mae_vs_valid_ratio_aligned.png",
            "title": "Aligned Pixel Consistency vs Valid Pixel Coverage",
            "x_key": "pc_vr_avg",
            "y_key": "pc_mae_avg",
            "x_label": "Average valid pixel ratio",
            "y_label": "Average pixel consistency MAE",
        },
        {
            "filename": "geometric_error_vs_valid_ratio_aligned.png",
            "title": "Aligned Geometric Error vs Valid Pixel Coverage",
            "x_key": "geo_vr_avg",
            "y_key": "geo_avg",
            "x_label": "Average valid pixel ratio",
            "y_label": "Average geometric error (m)",
        },
        {
            "filename": "pixel_mae_vs_geometric_error_aligned.png",
            "title": "Aligned Pixel Consistency vs Geometric Error",
            "x_key": "geo_avg",
            "y_key": "pc_mae_avg",
            "x_label": "Average geometric error (m)",
            "y_label": "Average pixel consistency MAE",
        },
        {
            "filename": "photometric_l2_vs_valid_ratio_aligned.png",
            "title": "Aligned Photometric L2 vs Valid Pixel Coverage",
            "x_key": "photo_vr_avg",
            "y_key": "photo_l2_avg",
            "x_label": "Average valid pixel ratio",
            "y_label": "Average photometric L2",
        },
        {
            "filename": "pixel_mae_vs_valid_ratio_mono_scaled.png",
            "title": "Mono Scaled Pixel Consistency vs Valid Pixel Coverage",
            "x_key": "mono_pc_vr_avg",
            "y_key": "mono_pc_mae_avg",
            "x_label": "Average valid pixel ratio",
            "y_label": "Average pixel consistency MAE",
        },
        {
            "filename": "geometric_error_vs_valid_ratio_mono_scaled.png",
            "title": "Mono Scaled Geometric Error vs Valid Pixel Coverage",
            "x_key": "mono_geo_vr_avg",
            "y_key": "mono_geo_avg",
            "x_label": "Average valid pixel ratio",
            "y_label": "Average geometric error (m)",
        },
    ]

    summary_lines = [
        "CORRELATION PLOT SUMMARY",
        "=" * 70,
    ]
    saved = 0

    for spec in specs:
        points = [
            (float(r[spec["x_key"]]), float(r[spec["y_key"]]))
            for r in all_rows
            if isinstance(r.get(spec["x_key"]), (int, float, np.floating))
            and isinstance(r.get(spec["y_key"]), (int, float, np.floating))
            and np.isfinite(r[spec["x_key"]])
            and np.isfinite(r[spec["y_key"]])
        ]
        if len(points) < 3:
            summary_lines.append(f"{spec['filename']}: skipped (need at least 3 finite points, found {len(points)})")
            continue

        xs = np.array([p[0] for p in points], dtype=np.float64)
        ys = np.array([p[1] for p in points], dtype=np.float64)
        corr = float(np.corrcoef(xs, ys)[0, 1]) if len(xs) >= 2 else float("nan")

        fig, ax = plt.subplots(figsize=(7.5, 5.5))
        ax.scatter(xs, ys, s=20, alpha=0.7, edgecolors="none")
        if len(xs) >= 2 and np.std(xs) > 0 and np.std(ys) > 0:
            slope, intercept = np.polyfit(xs, ys, 1)
            x_line = np.linspace(xs.min(), xs.max(), 100)
            ax.plot(x_line, slope * x_line + intercept, color="tab:red", linewidth=2)

        ax.set_title(spec["title"])
        ax.set_xlabel(spec["x_label"])
        ax.set_ylabel(spec["y_label"])
        ax.grid(True, alpha=0.25)
        ax.text(
            0.02,
            0.98,
            f"n={len(xs)}\npearson={corr:.3f}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
        )
        fig.tight_layout()
        fig.savefig(os.path.join(corr_dir, spec["filename"]), dpi=150, bbox_inches="tight")
        plt.close(fig)

        summary_lines.append(f"{spec['filename']}: n={len(xs)}, pearson={corr:.4f}")
        saved += 1

    summary_lines.append("=" * 70)

    summary_path = os.path.join(corr_dir, "correlation_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as sf:
        sf.write("\n".join(summary_lines) + "\n")

    return corr_dir, saved


# ─── Main ─────────────────────────────────────────────────────────────────────

def main(args):
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu"
                          else "cpu")
    print(f"[eval] Device: {device}")

    # ── Load checkpoint & build model ────────────────────────────────────────
    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    print(f"[eval] Loaded checkpoint (epoch {ckpt.get('epoch', '?')})")
    cfg = ckpt.get("cfg", {})
    depth_norm_cfg = cfg.get("depth_normalization", {})
    normalize_depths = bool(depth_norm_cfg.get("enabled", False))
    print(f"[eval] depth_normalization.enabled={normalize_depths}")
    if normalize_depths:
        mde_cfg = depth_norm_cfg.get("mde", {})
        gt_cfg = depth_norm_cfg.get("gt", {})
        print("[eval] depth normalization params: "
              f"mde(scale={float(mde_cfg.get('scale', 1.0)):.6f}, offset={float(mde_cfg.get('offset', 0.0)):.6f}), "
              f"gt(scale={float(gt_cfg.get('scale', 1.0)):.6f}, offset={float(gt_cfg.get('offset', 0.0)):.6f})")
    point_norm_cfg = cfg.get("point_normalization", {})
    normalize_points = bool(point_norm_cfg.get("enabled", False))
    point_norm_scale_eval = float(point_norm_cfg.get("scale", 1.0)) if normalize_points else None
    print(f"[eval] point_normalization.enabled={normalize_points}")
    if normalize_points:
        print(f"[eval] point normalization scale={point_norm_scale_eval:.6f}")
    model, model_variant, arch_cfg = build_model(ckpt, device)
    print(f"[eval] Model variant: {model_variant}")

    # ── Discover scenes ──────────────────────────────────────────────────────
    scenes = find_scenes(args.data_dir, max_batches=args.max_batches)
    print(f"[eval] Found {len(scenes)} scenes in {args.data_dir}")

    # ── Prepare output ───────────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    csv_path  = os.path.join(args.output_dir, "metrics_per_pair.csv")
    bw_dir    = os.path.join(args.output_dir, "best_worst")
    os.makedirs(bw_dir, exist_ok=True)

    # ── Resume support ────────────────────────────────────────────────────
    # If --resume is set and a metrics_per_pair.csv already exists in
    # output_dir, load the (batch, sample, scene, frame1, frame2) keys that
    # were already evaluated so we can skip them and append new rows instead
    # of overwriting the file.
    resume_mode = bool(args.resume) and os.path.isfile(csv_path)
    done_keys = set()
    done_count_per_scene = defaultdict(int)
    all_rows = []        # accumulate lightweight rows for summary stats

    if resume_mode:
        rewrite_resume_csv = False
        with open(csv_path, "r", newline="", encoding="utf-8") as rf:
            reader = csv.DictReader(rf)
            existing_cols = reader.fieldnames or []
            rewrite_resume_csv = existing_cols != _ALL_COLS
            for r in reader:
                key = (r.get("batch"), r.get("sample"), r.get("scene"),
                       r.get("frame1"), r.get("frame2"))
                done_keys.add(key)
                done_count_per_scene[key[:3]] += 1
                conv = dict(r)
                for c in _METRIC_COLS:
                    try:
                        conv[c] = float(conv[c])
                    except (TypeError, ValueError):
                        conv[c] = float("nan")
                all_rows.append(conv)
        if rewrite_resume_csv:
            with open(csv_path, "w", newline="", encoding="utf-8") as wf:
                rewrite_writer = csv.DictWriter(wf, fieldnames=_ALL_COLS, extrasaction="ignore")
                rewrite_writer.writeheader()
                for r in all_rows:
                    rewrite_writer.writerow({col: r.get(col, "") for col in _ALL_COLS})
            print(f"[eval] --resume: upgraded CSV schema in {csv_path} to include new averaged correlation fields.")
        print(f"[eval] --resume: found {len(done_keys)} already-evaluated pairs "
              f"in {csv_path}; they will be skipped.")
    elif args.resume:
        print(f"[eval] --resume set but no existing CSV found at {csv_path}; "
              f"starting a fresh run.")

    csv_file  = open(csv_path, "a" if resume_mode else "w", newline="", encoding="utf-8")
    writer    = csv.DictWriter(csv_file, fieldnames=_ALL_COLS, extrasaction="ignore")
    if not resume_mode:
        writer.writeheader()

    total_pairs = len(all_rows)

    # Bounded best/worst candidate heaps (by pc_mae_avg) so we never hold
    # every pair's images in memory at once — this is what previously caused
    # OOM crashes on large runs (e.g. --window 0, which evaluates every pair).
    best_heap = []   # smallest pc_mae_avg -> stored as (-val, seq, row) max-heap
    worst_heap = []  # largest pc_mae_avg  -> stored as (val, seq, row) min-heap
    _bw_seq = itertools.count()

    # ── Iterate scenes ───────────────────────────────────────────────────────
    for scene_idx, (b_name, s_name, sc_name, sc_path) in enumerate(scenes):
        color_dir = os.path.join(sc_path, "color")
        depth_dir = os.path.join(sc_path, "depth")
        pose_dir  = os.path.join(sc_path, "pose")
        pred_dir  = os.path.join(sc_path, args.pred_depth_dir)
        intr_path = os.path.join(sc_path, "intrinsic", "intrinsic_color.txt")

        if not os.path.isdir(color_dir):
            print(f"  [skip] missing color dir: {color_dir}")
            continue

        frame_ids = sorted(
            os.path.splitext(os.path.basename(f))[0]
            for f in glob.glob(os.path.join(color_dir, "*.png"))
        )
        if len(frame_ids) < 2:
            print(f"  [skip] < 2 frames in {sc_path}")
            continue

        # Load intrinsics (shared for the scene)
        if os.path.isfile(intr_path):
            K_np = load_intrinsics(intr_path)
        else:
            print(f"  [warn] no intrinsics file, using heuristic for {sc_path}")
            K_np = None   # will fill per-pair from image size

        # Evaluate up to args.window consecutive pairs per scene
        # (start count from any pairs already done for this scene when resuming)
        pairs_this_scene = done_count_per_scene.get((b_name, s_name, sc_name), 0)
        for pi in range(0, len(frame_ids) - 1, args.pair_stride):
            if args.window > 0 and pairs_this_scene >= args.window:
                break
            if args.max_pairs > 0 and total_pairs >= args.max_pairs:
                break

            fid1, fid2 = frame_ids[pi], frame_ids[pi + 1]

            if resume_mode and (b_name, s_name, sc_name, fid1, fid2) in done_keys:
                continue

            color1_path = os.path.join(color_dir, f"{fid1}.png")
            color2_path = os.path.join(color_dir, f"{fid2}.png")
            depth1_path = os.path.join(depth_dir, f"{fid1}.npy")
            depth2_path = os.path.join(depth_dir, f"{fid2}.npy")
            pred1_path  = os.path.join(pred_dir,  f"{fid1}.png")
            pred2_path  = os.path.join(pred_dir,  f"{fid2}.png")
            pose1_path  = os.path.join(pose_dir,  f"{fid1}.txt")
            pose2_path  = os.path.join(pose_dir,  f"{fid2}.txt")

            missing = [p for p in (color1_path, color2_path, depth1_path,
                                    depth2_path, pred1_path, pred2_path)
                       if not os.path.isfile(p)]
            if missing:
                print(f"  [skip] missing files: {missing}")
                continue

            print(f"[{scene_idx+1}/{len(scenes)}] {b_name}/{s_name}/{sc_name} "
                  f"pair ({fid1},{fid2})")

            row = {"batch": b_name, "sample": s_name, "scene": sc_name,
                   "frame1": fid1, "frame2": fid2}
            row.update(_nan_row())

            try:
                # ── Load images ──────────────────────────────────────────────
                img1_np = load_image(color1_path, as_gray=False)   # (H,W,3) float32
                img2_np = load_image(color2_path, as_gray=False)
                H, W = img1_np.shape[:2]

                img1_gray = cv2.cvtColor(img1_np, cv2.COLOR_RGB2GRAY)
                img2_gray = cv2.cvtColor(img2_np, cv2.COLOR_RGB2GRAY)

                # ── Load GT depths ───────────────────────────────────────────
                gt1_raw = load_gt_depth(depth1_path)
                gt2_raw = load_gt_depth(depth2_path)
                if gt1_raw.shape != (H, W):
                    gt1_raw = cv2.resize(gt1_raw, (W, H), interpolation=cv2.INTER_NEAREST)
                if gt2_raw.shape != (H, W):
                    gt2_raw = cv2.resize(gt2_raw, (W, H), interpolation=cv2.INTER_NEAREST)

                # ── Load mono predictions ────────────────────────────────────
                pred_mono1 = load_pred_depth_zoe(pred1_path, args.depth_scale_pred)
                pred_mono2 = load_pred_depth_zoe(pred2_path, args.depth_scale_pred)
                if pred_mono1.shape != (H, W):
                    pred_mono1 = cv2.resize(pred_mono1, (W, H), interpolation=cv2.INTER_NEAREST)
                if pred_mono2.shape != (H, W):
                    pred_mono2 = cv2.resize(pred_mono2, (W, H), interpolation=cv2.INTER_NEAREST)

                # ── Intrinsics ───────────────────────────────────────────────
                if K_np is None:
                    fx = 0.9 * max(H, W)
                    K_np = np.array([[fx, 0., W/2.], [0., fx, H/2.], [0., 0., 1.]],
                                    dtype=np.float32)

                # ── Poses ────────────────────────────────────────────────────
                has_poses = os.path.isfile(pose1_path) and os.path.isfile(pose2_path)
                if has_poses:
                    pose1_np = load_pose(pose1_path)
                    pose2_np = load_pose(pose2_path)
                    T_12_init_np = np.linalg.inv(pose2_np) @ pose1_np
                else:
                    pose1_np = pose2_np = None
                    T_12_init_np = np.eye(4, dtype=np.float32)

                # ── Prepare tensors ──────────────────────────────────────────
                rgb1_t = (torch.from_numpy(img1_np)
                          .permute(2, 0, 1).unsqueeze(0).to(device))
                rgb2_t = (torch.from_numpy(img2_np)
                          .permute(2, 0, 1).unsqueeze(0).to(device))
                depth1_t = (torch.from_numpy(pred_mono1)
                            .unsqueeze(0).unsqueeze(0).to(device))
                depth2_t = (torch.from_numpy(pred_mono2)
                            .unsqueeze(0).unsqueeze(0).to(device))
                if normalize_depths:
                    depth1_t = normalize_depth_map(depth1_t, depth_norm_cfg.get("mde"))
                    depth2_t = normalize_depth_map(depth2_t, depth_norm_cfg.get("mde"))
                K_t = torch.from_numpy(K_np.astype(np.float32)).unsqueeze(0).to(device)
                T_12_t = torch.from_numpy(T_12_init_np.astype(np.float32)).unsqueeze(0).to(device)

                # ── Inference ────────────────────────────────────────────────
                outputs = run_inference(
                    model, model_variant, cfg,
                    rgb1_t, rgb2_t, depth1_t, depth2_t,
                    K_t, T_12_t, device,
                )

                # depth_only outputs point1/point2 (XYZ maps); derive depth from Z-channel.
                if "point1" in outputs:
                    # depth1 = Z of point1 in cam1 frame (correct).
                    pred1_out_t = outputs["point1"][:, 2:3].clamp(min=0)  # (B,1,pH,pW)
                    if normalize_points and point_norm_scale_eval is not None:
                        pred1_out_t = pred1_out_t * point_norm_scale_eval

                    # point2 is expressed in cam1 frame; get the real depth in cam2 frame
                    # by transforming the 3-D points with T_12 (cam1 → cam2) and taking Z.
                    _p2 = outputs["point2"]                          # (B, 3, pH, pW) in cam1
                    if normalize_points and point_norm_scale_eval is not None:
                        _p2 = _p2 * point_norm_scale_eval            # → metric units
                    _B2, _, _pH2, _pW2 = _p2.shape
                    _R12 = T_12_t[:, :3, :3].to(dtype=_p2.dtype)
                    _t12 = T_12_t[:, :3,  3].to(dtype=_p2.dtype)
                    _p2_cam2 = (
                        torch.bmm(_R12, _p2.reshape(_B2, 3, -1)) + _t12.unsqueeze(-1)
                    ).reshape(_B2, 3, _pH2, _pW2)
                    pred2_out_t = _p2_cam2[:, 2:3].clamp(min=0)     # depth in cam2 frame
                else:
                    pred1_out_t = outputs["depth1"]
                    pred2_out_t = outputs["depth2"]
                    if normalize_depths:
                        pred1_out_t = denormalize_depth_map(pred1_out_t, depth_norm_cfg.get("gt"))
                        pred2_out_t = denormalize_depth_map(pred2_out_t, depth_norm_cfg.get("gt"))
                pred1_out = pred1_out_t.squeeze().cpu().numpy()
                pred2_out = pred2_out_t.squeeze().cpu().numpy()

                # Resize to image resolution
                pred1_full = cv2.resize(pred1_out, (W, H), interpolation=cv2.INTER_LINEAR)
                pred2_full = cv2.resize(pred2_out, (W, H), interpolation=cv2.INTER_LINEAR)

                # Predicted camera params
                if "K_pred" in outputs:
                    Kp = outputs["K_pred"][0].cpu().numpy()
                    row["pred_fx"] = float(Kp[0, 0])
                    row["pred_fy"] = float(Kp[1, 1])
                    row["pred_cx"] = float(Kp[0, 2])
                    row["pred_cy"] = float(Kp[1, 2])
                row["gt_fx"] = float(K_np[0, 0])
                row["gt_fy"] = float(K_np[1, 1])
                row["gt_cx"] = float(K_np[0, 2])
                row["gt_cy"] = float(K_np[1, 2])

                # ── Use GT or predicted K for metric projection ───────────────
                K_eval = K_np  # use GT intrinsics when available

                # ── Depth metrics (aligned) ──────────────────────────────────
                m1 = _safe_depth_metrics(pred1_full, gt1_raw)
                m2 = _safe_depth_metrics(pred2_full, gt2_raw)
                for k, v in m1.items():
                    row[f"{k}_v1"] = v
                for k, v in m2.items():
                    row[f"{k}_v2"] = v

                # ── Pixel consistency (aligned) ──────────────────────────────
                if has_poses:
                    pc12_mae, pc12_rmse, pc12_vr = _safe_pixel_consistency(
                        gt1_raw, pred1_full, gt2_raw, K_eval, pose1_np, pose2_np)
                    pc21_mae, pc21_rmse, pc21_vr = _safe_pixel_consistency(
                        gt2_raw, pred2_full, gt1_raw, K_eval, pose2_np, pose1_np)
                    row["pc_mae_12"]  = pc12_mae
                    row["pc_rmse_12"] = pc12_rmse
                    row["pc_vr_12"]   = pc12_vr
                    row["pc_mae_21"]  = pc21_mae
                    row["pc_rmse_21"] = pc21_rmse
                    row["pc_vr_21"]   = pc21_vr
                    row["pc_mae_avg"]  = _nanmean2(pc12_mae, pc21_mae)
                    row["pc_rmse_avg"] = _nanmean2(pc12_rmse, pc21_rmse)
                    row["pc_vr_avg"]   = _nanmean2(pc12_vr, pc21_vr)

                    geo12, geo12_vr = _safe_geometric_error(
                        gt1_raw, pred2_full, K_eval, pose1_np, pose2_np)
                    geo21, geo21_vr = _safe_geometric_error(
                        gt2_raw, pred1_full, K_eval, pose2_np, pose1_np)
                    row["geo_12"] = geo12
                    row["geo_vr_12"] = geo12_vr
                    row["geo_21"] = geo21
                    row["geo_vr_21"] = geo21_vr
                    row["geo_avg"] = _nanmean2(geo12, geo21)
                    row["geo_vr_avg"] = _nanmean2(geo12_vr, geo21_vr)

                    # ── Photometric consistency (aligned) ────────────────────
                    ph12_ssim, ph12_l2, ph12_vr = _safe_photometric(
                        img1_gray, img2_gray, pred1_full, K_eval, pose1_np, pose2_np)
                    ph21_ssim, ph21_l2, ph21_vr = _safe_photometric(
                        img2_gray, img1_gray, pred2_full, K_eval, pose2_np, pose1_np)
                    row["photo_ssim_12"] = ph12_ssim
                    row["photo_l2_12"]   = ph12_l2
                    row["photo_vr_12"]   = ph12_vr
                    row["photo_ssim_21"] = ph21_ssim
                    row["photo_l2_21"]   = ph21_l2
                    row["photo_vr_21"]   = ph21_vr
                    row["photo_ssim_avg"] = _nanmean2(ph12_ssim, ph21_ssim)
                    row["photo_l2_avg"]   = _nanmean2(ph12_l2, ph21_l2)
                    row["photo_vr_avg"]   = _nanmean2(ph12_vr, ph21_vr)

                # ── Mono scaled metrics ──────────────────────────────────────
                s1 = _median_scale(pred_mono1, gt1_raw)
                s2 = _median_scale(pred_mono2, gt2_raw)
                row["mono_scale1"] = s1
                row["mono_scale2"] = s2

                mono1_scaled = pred_mono1 * s1
                mono2_scaled = pred_mono2 * s2

                mm1 = _safe_depth_metrics(mono1_scaled, gt1_raw)
                mm2 = _safe_depth_metrics(mono2_scaled, gt2_raw)
                for k, v in mm1.items():
                    row[f"mono_{k}_v1"] = v
                for k, v in mm2.items():
                    row[f"mono_{k}_v2"] = v

                if has_poses:
                    mpc12_mae, mpc12_rmse, mpc12_vr = _safe_pixel_consistency(
                        gt1_raw, mono1_scaled, gt2_raw, K_eval, pose1_np, pose2_np)
                    mpc21_mae, mpc21_rmse, mpc21_vr = _safe_pixel_consistency(
                        gt2_raw, mono2_scaled, gt1_raw, K_eval, pose2_np, pose1_np)
                    row["mono_pc_mae_12"]  = mpc12_mae
                    row["mono_pc_rmse_12"] = mpc12_rmse
                    row["mono_pc_vr_12"]   = mpc12_vr
                    row["mono_pc_mae_21"]  = mpc21_mae
                    row["mono_pc_rmse_21"] = mpc21_rmse
                    row["mono_pc_vr_21"]   = mpc21_vr
                    row["mono_pc_mae_avg"]  = _nanmean2(mpc12_mae, mpc21_mae)
                    row["mono_pc_rmse_avg"] = _nanmean2(mpc12_rmse, mpc21_rmse)
                    row["mono_pc_vr_avg"]   = _nanmean2(mpc12_vr, mpc21_vr)

                    mgeo12, mgeo12_vr = _safe_geometric_error(
                        gt1_raw, mono2_scaled, K_eval, pose1_np, pose2_np)
                    mgeo21, mgeo21_vr = _safe_geometric_error(
                        gt2_raw, mono1_scaled, K_eval, pose2_np, pose1_np)
                    row["mono_geo_12"] = mgeo12
                    row["mono_geo_vr_12"] = mgeo12_vr
                    row["mono_geo_21"] = mgeo21
                    row["mono_geo_vr_21"] = mgeo21_vr
                    row["mono_geo_avg"] = _nanmean2(mgeo12, mgeo21)
                    row["mono_geo_vr_avg"] = _nanmean2(mgeo12_vr, mgeo21_vr)

                    mph12_ssim, mph12_l2, mph12_vr = _safe_photometric(
                        img1_gray, img2_gray, mono1_scaled, K_eval, pose1_np, pose2_np)
                    mph21_ssim, mph21_l2, mph21_vr = _safe_photometric(
                        img2_gray, img1_gray, mono2_scaled, K_eval, pose2_np, pose1_np)
                    row["mono_photo_ssim_12"] = mph12_ssim
                    row["mono_photo_l2_12"]   = mph12_l2
                    row["mono_photo_vr_12"]   = mph12_vr
                    row["mono_photo_ssim_21"] = mph21_ssim
                    row["mono_photo_l2_21"]   = mph21_l2
                    row["mono_photo_vr_21"]   = mph21_vr
                    row["mono_photo_ssim_avg"] = _nanmean2(mph12_ssim, mph21_ssim)
                    row["mono_photo_l2_avg"]   = _nanmean2(mph12_l2, mph21_l2)
                    row["mono_photo_vr_avg"]   = _nanmean2(mph12_vr, mph21_vr)

                # Store inputs for best/worst visualisation
                row["_img1"]       = img1_np
                row["_img2"]       = img2_np
                row["_gt1"]        = gt1_raw
                row["_gt2"]        = gt2_raw
                row["_pred1"]      = pred1_full
                row["_pred2"]      = pred2_full
                row["_mono1_sc"]   = mono1_scaled
                row["_mono2_sc"]   = mono2_scaled
                conf1_np = outputs.get("confidence1")
                conf2_np = outputs.get("confidence2")
                row["_conf1"] = conf1_np.squeeze().cpu().numpy() if conf1_np is not None else None
                row["_conf2"] = conf2_np.squeeze().cpu().numpy() if conf2_np is not None else None

            except Exception as exc:
                print(f"  [ERROR] {b_name}/{s_name}/{sc_name} ({fid1},{fid2}): {exc}")

            # Write CSV row (without private _ keys)
            public_row = {k: v for k, v in row.items() if not k.startswith("_")}
            writer.writerow(public_row)
            csv_file.flush()

            # Keep only lightweight metrics for the running summary; heavy
            # image arrays are tracked separately in bounded best/worst heaps.
            all_rows.append(public_row)

            pc_val = row.get("pc_mae_avg")
            if isinstance(pc_val, float) and np.isfinite(pc_val) and "_img1" in row:
                heapq.heappush(best_heap, (-pc_val, next(_bw_seq), dict(row)))
                if len(best_heap) > args.n_best_worst:
                    heapq.heappop(best_heap)
                heapq.heappush(worst_heap, (pc_val, next(_bw_seq), dict(row)))
                if len(worst_heap) > args.n_best_worst:
                    heapq.heappop(worst_heap)

            total_pairs      += 1
            pairs_this_scene += 1

        if args.max_pairs > 0 and total_pairs >= args.max_pairs:
            print(f"[eval] Reached max_pairs={args.max_pairs}, stopping.")
            break

    csv_file.close()
    print(f"\n[eval] Evaluated {total_pairs} pairs. CSV saved to {csv_path}")

    if not all_rows:
        print("[eval] No pairs evaluated; exiting.")
        return

    # ── Summary statistics ─────────────────────────────────────────────────
    summary_path = os.path.join(args.output_dir, "metrics_summary.txt")
    numeric_cols = [c for c in _METRIC_COLS
                    if c not in ("pred_fx","pred_fy","pred_cx","pred_cy",
                                  "gt_fx","gt_fy","gt_cx","gt_cy")]

    with open(summary_path, "w", encoding="utf-8") as sf:
        def _w(s=""):
            print(s)
            sf.write(s + "\n")

        _w("=" * 70)
        _w("TEST SET EVALUATION SUMMARY")
        _w(f"  Dataset : {args.data_dir}")
        _w(f"  Checkpoint: {args.checkpoint}")
        _w(f"  Pairs evaluated: {total_pairs}")
        _w("=" * 70)

        # Group metrics by section for readability
        sections = {
            "Aligned Depth (view 1)": [
                "abs_rel_v1","rmse_v1","mae_v1","delta1_v1","delta2_v1","delta3_v1"],
            "Aligned Depth (view 2)": [
                "abs_rel_v2","rmse_v2","mae_v2","delta1_v2","delta2_v2","delta3_v2"],
            "Pixel Consistency (aligned)": [
                "pc_mae_12","pc_rmse_12","pc_vr_12",
                "pc_mae_21","pc_rmse_21","pc_vr_21",
                "pc_mae_avg","pc_rmse_avg","pc_vr_avg"],
            "Geometric Consistency (aligned, m)": [
                "geo_12","geo_vr_12",
                "geo_21","geo_vr_21",
                "geo_avg","geo_vr_avg"],
            "Photometric Consistency (aligned)": [
                "photo_ssim_12","photo_l2_12","photo_vr_12",
                "photo_ssim_21","photo_l2_21","photo_vr_21",
                "photo_ssim_avg","photo_l2_avg","photo_vr_avg"],
            "Monocular Scale Factors": [
                "mono_scale1","mono_scale2"],
            "Mono Scaled Depth (view 1)": [
                "mono_abs_rel_v1","mono_rmse_v1","mono_mae_v1",
                "mono_delta1_v1","mono_delta2_v1","mono_delta3_v1"],
            "Mono Scaled Depth (view 2)": [
                "mono_abs_rel_v2","mono_rmse_v2","mono_mae_v2",
                "mono_delta1_v2","mono_delta2_v2","mono_delta3_v2"],
            "Pixel Consistency (mono scaled)": [
                "mono_pc_mae_12","mono_pc_rmse_12","mono_pc_vr_12",
                "mono_pc_mae_21","mono_pc_rmse_21","mono_pc_vr_21",
                "mono_pc_mae_avg","mono_pc_rmse_avg","mono_pc_vr_avg"],
            "Geometric Consistency (mono scaled, m)": [
                "mono_geo_12","mono_geo_vr_12",
                "mono_geo_21","mono_geo_vr_21",
                "mono_geo_avg","mono_geo_vr_avg"],
            "Photometric Consistency (mono scaled)": [
                "mono_photo_ssim_12","mono_photo_l2_12","mono_photo_vr_12",
                "mono_photo_ssim_21","mono_photo_l2_21","mono_photo_vr_21",
                "mono_photo_ssim_avg","mono_photo_l2_avg","mono_photo_vr_avg"],
        }

        col_w = max(len(c) for cols in sections.values() for c in cols) + 2

        for section, cols in sections.items():
            _w(f"\n── {section} ──")
            _w(f"  {'Metric':<{col_w}}  {'Mean':>10}  {'Median':>10}  {'Std':>10}  {'N':>6}")
            _w(f"  {'-'*col_w}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*6}")
            for col in cols:
                vals = np.array([r[col] for r in all_rows
                                 if isinstance(r.get(col), float)
                                 and np.isfinite(r[col])], dtype=np.float64)
                if len(vals) == 0:
                    _w(f"  {col:<{col_w}}  {'n/a':>10}  {'n/a':>10}  {'n/a':>10}  {'0':>6}")
                    continue
                _w(f"  {col:<{col_w}}  {vals.mean():>10.4f}  "
                   f"{np.median(vals):>10.4f}  {vals.std():>10.4f}  {len(vals):>6}")

        _w("\n" + "=" * 70)

    print(f"[eval] Summary saved to {summary_path}")

    corr_dir, n_corr = _save_correlation_plots(all_rows, args.output_dir)
    print(f"[eval] Saved {n_corr} correlation plots to {corr_dir}")

    # ── Best / worst by pixel consistency MAE (avg of 1→2 and 2→1) ─────────
    # Note: best_heap/worst_heap only contain pairs evaluated in *this* run
    # (skipped/resumed pairs from a previous run aren't re-visualized since
    # their images are no longer held in memory).
    if not best_heap and not worst_heap:
        print("[eval] No rows with valid pc_mae_avg — skipping best/worst visualizations.")
        return

    best_sorted  = [entry[2] for entry in sorted(best_heap,  key=lambda e: -e[0])]
    worst_sorted = [entry[2] for entry in sorted(worst_heap, key=lambda e: e[0], reverse=True)]

    def _label(r):
        return (f"{r['batch']}/{r['sample']}/{r['scene']} "
                f"({r['frame1']},{r['frame2']})  pc_mae_avg={r['pc_mae_avg']:.4f}")

    print(f"\n[eval] Saving top-{len(best_sorted)} best and top-{len(worst_sorted)} worst visualizations...")
    for rank, r in enumerate(best_sorted):
        fname = (f"best_{rank+1:02d}_{r['batch']}_{r['sample']}_"
                 f"{r['scene']}_{r['frame1']}-{r['frame2']}.png")
        title = f"BEST #{rank+1}  {_label(r)}"
        save_pair_visualization(
            r["_img1"], r["_img2"], r["_gt1"], r["_gt2"],
            r["_pred1"], r["_pred2"], r["_mono1_sc"], r["_mono2_sc"],
            r.get("_conf1"), r.get("_conf2"),
            title, os.path.join(bw_dir, fname),
        )

    for rank, r in enumerate(worst_sorted):
        fname = (f"worst_{rank+1:02d}_{r['batch']}_{r['sample']}_"
                 f"{r['scene']}_{r['frame1']}-{r['frame2']}.png")
        title = f"WORST #{rank+1}  {_label(r)}"
        save_pair_visualization(
            r["_img1"], r["_img2"], r["_gt1"], r["_gt2"],
            r["_pred1"], r["_pred2"], r["_mono1_sc"], r["_mono2_sc"],
            r.get("_conf1"), r.get("_conf2"),
            title, os.path.join(bw_dir, fname),
        )

    print(f"[eval] Best/worst visualizations saved to {bw_dir}")
    print(f"[eval] Done.")


# ─── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Batch evaluation of DepthAlignNet on a full test set.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required
    parser.add_argument("--checkpoint", required=True,
                        help="Path to model checkpoint (.pt)")
    parser.add_argument("--output_dir", required=True,
                        help="Directory to write all outputs")
    parser.add_argument("--resume", action="store_true",
                        help="Resume evaluation using an existing --output_dir: "
                             "pairs already present in metrics_per_pair.csv are "
                             "skipped and new results are appended.")

    # Dataset
    parser.add_argument("--data_dir", default="datasets_test/sampled_data",
                        help="Root of test dataset (contains batch* dirs)")
    parser.add_argument("--pred_depth_dir", default="zoe-depth_pred",
                        help="Sub-folder name inside each scene with MDE predictions")

    # Sampling
    parser.add_argument("--max_batches", type=int, default=0,
                        help="Limit number of batch dirs (0 = all)")
    parser.add_argument("--max_pairs", type=int, default=0,
                        help="Stop after this many pairs total (0 = all)")
    parser.add_argument("--window", type=int, default=0,
                        help="Max consecutive pairs per scene to evaluate (0 = all)")
    parser.add_argument("--pair_stride", type=int, default=1,
                        help="Step between evaluated pairs within a scene")

    # Scales
    parser.add_argument("--depth_scale_pred", type=float, default=1000.0,
                        help="Divisor for MDE PNG depths (1000 = mm→m for ZoeDepth)")
    parser.add_argument("--depth_scale_gt", type=float, default=1.0,
                        help="Divisor for GT depth files (1.0 if already in metres)")

    # Output
    parser.add_argument("--n_best_worst", type=int, default=5,
                        help="Number of best/worst pairs to visualize")

    # Device
    parser.add_argument("--device", default="cuda",
                        help="torch device: cuda or cpu")

    args = parser.parse_args()
    main(args)
