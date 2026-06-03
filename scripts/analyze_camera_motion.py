"""
Standalone camera-motion distribution analysis for dataset splits.

The script mirrors the training split and pair construction used by
training/train.py and data/preprocessing.py:
  - train/val scenes are shuffled with seed 42 and split 90/10
  - test scenes come from dataset.test_root_dir when available
  - pairs are consecutive valid poses: (0,1), (2,3), ...
  - relative pose is T_12 = inv(T_cw2) @ T_cw1
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.spatial.transform import Rotation
from scipy.stats import ks_2samp, wasserstein_distance


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "data"))

from data.scene import ScanNetScene, find_scene_paths, load_pose  # noqa: E402


SPLITS = ("train", "val", "test")
SPLIT_COLORS = {"train": "tab:blue", "val": "tab:orange", "test": "tab:green"}
EPS = 1e-8


def resolve_path(path: str | os.PathLike | None) -> Path | None:
    """Resolve a path relative to the repository root."""
    if path is None:
        return None
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def load_config(config_path: Path) -> dict:
    """Load YAML configuration."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_split_scene_paths(cfg: dict) -> tuple[dict[str, list[str]], dict[str, Path]]:
    """Reproduce train.py's deterministic train/val/test scene split."""
    dataset_cfg = cfg.get("dataset", {})
    root_dir = resolve_path(dataset_cfg["root_dir"])
    test_root_dir = resolve_path(dataset_cfg.get("test_root_dir"))

    all_scene_paths = find_scene_paths(str(root_dir))
    rng = random.Random(42)
    rng.shuffle(all_scene_paths)

    max_scenes = dataset_cfg.get("max_scenes")
    if max_scenes:
        all_scene_paths = all_scene_paths[: int(max_scenes)]

    n_scenes = len(all_scene_paths)
    split_idx = int(n_scenes * 0.9)
    train_paths = all_scene_paths[:split_idx]
    val_paths = all_scene_paths[split_idx:]
    if not val_paths and train_paths:
        val_paths = train_paths[-1:]

    roots = {"train": root_dir, "val": root_dir, "test": test_root_dir or root_dir}
    if test_root_dir is not None:
        test_paths = find_scene_paths(str(test_root_dir))
        if not test_paths:
            print(
                f"[warn] test_root_dir {test_root_dir} yielded no scenes; "
                "using the last validation scene as a test placeholder."
            )
            test_paths = val_paths[-1:]
            roots["test"] = root_dir
    else:
        print("[warn] No dataset.test_root_dir configured; using the last validation scene as test.")
        test_paths = val_paths[-1:]

    return {"train": train_paths, "val": val_paths, "test": test_paths}, roots


def safe_load_scene_poses(scene: ScanNetScene) -> tuple[dict[str, np.ndarray], int]:
    """Load finite 4x4 poses and count frames skipped because pose I/O failed or was invalid."""
    poses = {}
    skipped_frames = 0
    for fid in scene.frame_ids:
        try:
            pose = load_pose(scene.pose_path(fid)).astype(np.float64)
        except Exception:
            skipped_frames += 1
            continue
        if pose.shape != (4, 4) or not np.all(np.isfinite(pose)):
            skipped_frames += 1
            continue
        poses[fid] = pose
    return poses, skipped_frames


def analyze_transform(T_12: np.ndarray) -> dict[str, float] | None:
    """Extract rotation angle, axis, translation magnitude, and direction statistics."""
    if T_12.shape != (4, 4) or not np.all(np.isfinite(T_12)):
        return None

    R = T_12[:3, :3]
    t = T_12[:3, 3]
    if not np.all(np.isfinite(R)) or not np.all(np.isfinite(t)):
        return None

    # Robust trace formula for the relative rotation angle.
    cos_theta = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    theta_rad = float(np.arccos(cos_theta))
    theta_deg = float(np.degrees(theta_rad))

    # SO(3) logarithm map. SciPy returns the rotation vector omega = axis * angle.
    try:
        omega = Rotation.from_matrix(R).as_rotvec()
    except ValueError:
        return None
    omega_norm = float(np.linalg.norm(omega))
    axis = omega / (omega_norm + EPS)

    # Translation magnitude and unit direction.
    trans_mag = float(np.linalg.norm(t))
    t_unit = t / (trans_mag + EPS)
    azimuth = float(np.arctan2(t_unit[1], t_unit[0]))
    elevation = float(np.arcsin(np.clip(t_unit[2], -1.0, 1.0)))

    values = {
        "rotation_angle_deg": theta_deg,
        "axis_x": float(axis[0]),
        "axis_y": float(axis[1]),
        "axis_z": float(axis[2]),
        "translation_magnitude": trans_mag,
        "t_unit_x": float(t_unit[0]),
        "t_unit_y": float(t_unit[1]),
        "t_unit_z": float(t_unit[2]),
        "azimuth": azimuth,
        "elevation": elevation,
    }
    return values if np.all(np.isfinite(list(values.values()))) else None


def safe_focal_length_x(scene: ScanNetScene) -> float | None:
    """Extract fx = K[0, 0] from the scene intrinsics matrix."""
    try:
        K = scene.intrinsics.astype(np.float64)
    except Exception:
        return None
    if K.shape[0] < 1 or K.shape[1] < 1 or not np.isfinite(K[0, 0]):
        return None
    return float(K[0, 0])


def collect_split_statistics(scene_paths: list[str], root_dir: Path, split: str) -> tuple[pd.DataFrame, dict]:
    """Collect one row per valid pair in a split."""
    rows = []
    skipped_pose_frames = 0
    skipped_samples = 0
    skipped_intrinsic_scenes = 0

    for scene_path in scene_paths:
        scene = ScanNetScene(scene_path)
        if not scene.is_valid() or not scene.frame_ids:
            continue
        focal_length_x = safe_focal_length_x(scene)
        if focal_length_x is None:
            skipped_intrinsic_scenes += 1
            continue

        poses, skipped = safe_load_scene_poses(scene)
        skipped_pose_frames += skipped
        valid_fids = [fid for fid in scene.frame_ids if fid in poses]

        for i in range(0, len(valid_fids) - 1, 2):
            fid_a, fid_b = valid_fids[i], valid_fids[i + 1]
            try:
                T_12 = np.linalg.inv(poses[fid_b]) @ poses[fid_a]
            except np.linalg.LinAlgError:
                skipped_samples += 1
                continue

            stats = analyze_transform(T_12)
            if stats is None:
                skipped_samples += 1
                continue

            stats.update(
                {
                    "split": split,
                    "scene": os.path.relpath(scene_path, root_dir),
                    "frame_1": fid_a,
                    "frame_2": fid_b,
                    "focal_length_x": focal_length_x,
                }
            )
            rows.append(stats)

    diagnostics = {
        "scenes": len(scene_paths),
        "valid_samples": len(rows),
        "skipped_samples": skipped_samples,
        "skipped_pose_frames": skipped_pose_frames,
        "skipped_intrinsic_scenes": skipped_intrinsic_scenes,
    }
    return pd.DataFrame(rows), diagnostics


def summary_table(data_by_split: dict[str, pd.DataFrame], value_col: str) -> pd.DataFrame:
    """Build count, central tendency, tail, and range statistics for a scalar column."""
    rows = []
    for split, df in data_by_split.items():
        x = df[value_col].to_numpy(dtype=np.float64) if value_col in df else np.array([])
        if x.size == 0:
            rows.append({"split": split, "count": 0})
            continue
        rows.append(
            {
                "split": split,
                "count": int(x.size),
                "mean": float(np.mean(x)),
                "std": float(np.std(x, ddof=1)) if x.size > 1 else 0.0,
                "median": float(np.median(x)),
                "p90": float(np.percentile(x, 90)),
                "p95": float(np.percentile(x, 95)),
                "p99": float(np.percentile(x, 99)),
                "min": float(np.min(x)),
                "max": float(np.max(x)),
            }
        )
    return pd.DataFrame(rows)


def distribution_tests(data_by_split: dict[str, pd.DataFrame], value_col: str) -> pd.DataFrame:
    """Run pairwise KS tests and Wasserstein distances."""
    rows = []
    for split_a, split_b in combinations(SPLITS, 2):
        a = (
            data_by_split[split_a][value_col].to_numpy(dtype=np.float64)
            if value_col in data_by_split[split_a]
            else np.array([])
        )
        b = (
            data_by_split[split_b][value_col].to_numpy(dtype=np.float64)
            if value_col in data_by_split[split_b]
            else np.array([])
        )
        if a.size == 0 or b.size == 0:
            rows.append(
                {
                    "split_a": split_a,
                    "split_b": split_b,
                    "KS_stat": np.nan,
                    "KS_p": np.nan,
                    "Wasserstein": np.nan,
                }
            )
            continue
        ks = ks_2samp(a, b)
        rows.append(
            {
                "split_a": split_a,
                "split_b": split_b,
                "KS_stat": float(ks.statistic),
                "KS_p": float(ks.pvalue),
                "Wasserstein": float(wasserstein_distance(a, b)),
            }
        )
    return pd.DataFrame(rows)


def save_overlaid_hist(data_by_split, value_col, bins, xlabel, title, output_path, density=True, transform=None):
    """Save overlaid KDE curves + step histograms for train/val/test.

    The x-axis is clipped to the 1st–99th percentile of the combined data so
    that a handful of outliers (or a tiny test split) cannot dominate the scale.
    Each split shows:
      - a step-outline histogram (no filled overlap)
      - a smooth KDE curve (when n ≥ 10)
      - a dashed vertical line at the median
    The legend includes the sample count and median for quick reference.
    """
    from scipy.stats import gaussian_kde

    fig, ax = plt.subplots(figsize=(9, 6))

    # Collect and optionally transform each split's values.
    split_arrays: dict[str, np.ndarray] = {}
    for split in SPLITS:
        if value_col not in data_by_split[split] or len(data_by_split[split]) == 0:
            split_arrays[split] = np.array([])
            continue
        x = data_by_split[split][value_col].to_numpy(dtype=np.float64)
        if transform is not None:
            x = transform(x)
        split_arrays[split] = x[np.isfinite(x)]

    combined = np.concatenate([v for v in split_arrays.values() if v.size > 0])
    if combined.size == 0:
        plt.close(fig)
        return

    # Clip x range to 1st–99th percentile so small/large outliers don't
    # compress the interesting part of the distribution.
    x_lo = float(np.percentile(combined, 1))
    x_hi = float(np.percentile(combined, 99))
    if x_lo >= x_hi:
        x_lo, x_hi = float(combined.min()), float(combined.max())
    x_eval = np.linspace(x_lo, x_hi, 500)

    for split in SPLITS:
        x = split_arrays[split]
        color = SPLIT_COLORS[split]
        n = x.size
        if n == 0:
            continue
        med = float(np.median(x))
        label = f"{split}  (n={n:,}, median={med:.3g})"

        # Step-outline histogram — no filled overlap between splits.
        ax.hist(
            x, bins=bins, density=True, histtype="step",
            color=color, linewidth=1.4, range=(x_lo, x_hi),
        )
        # Smooth KDE curve on top (requires at least 10 distinct points).
        if n >= 10:
            try:
                kde = gaussian_kde(x)
                ax.plot(x_eval, kde(x_eval), color=color, linewidth=2.5, label=label)
            except Exception:
                ax.plot([], [], color=color, linewidth=2.5, label=label)
        else:
            ax.plot([], [], color=color, linewidth=2.5, label=label)

        # Dashed vertical line at the median.
        ax.axvline(med, color=color, linewidth=1.2, linestyle="--", alpha=0.75)

    ax.set_xlim(x_lo, x_hi)
    ax.set_xlabel(xlabel)
    # "Probability density" is technically correct (area under curve = 1),
    # but we add a note so it's clear this is not a count or percentage.
    ax.set_ylabel("Probability density  (area = 1)")
    ax.set_title(title)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_empirical_cdf(data_by_split, value_col, xlabel, title, output_path):
    """Save empirical CDF curves for a scalar value.

    The y-axis is fraction of samples ≤ x, so 0.5 = 50 % of pairs have a
    value at or below that point.  The x-axis is clipped to the combined
    1st–99th percentile so a tiny test split cannot compress the view.
    """
    fig, ax = plt.subplots(figsize=(9, 6))

    all_vals = []
    for split in SPLITS:
        if value_col in data_by_split[split]:
            v = data_by_split[split][value_col].to_numpy(dtype=np.float64)
            all_vals.append(v[np.isfinite(v)])
    combined = np.concatenate(all_vals) if all_vals else np.array([])
    if combined.size == 0:
        plt.close(fig)
        return
    x_lo = float(np.percentile(combined, 1))
    x_hi = float(np.percentile(combined, 99))
    if x_lo >= x_hi:
        x_lo, x_hi = float(combined.min()), float(combined.max())

    for split in SPLITS:
        if value_col not in data_by_split[split]:
            continue
        x = np.sort(data_by_split[split][value_col].to_numpy(dtype=np.float64))
        x = x[np.isfinite(x)]
        if x.size:
            y = np.arange(1, x.size + 1) / x.size
            label = f"{split}  (n={x.size:,})"
            ax.plot(x, y, label=label, color=SPLIT_COLORS[split], linewidth=2)
    ax.set_xlim(x_lo, x_hi)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Fraction of pairs ≤ x")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9)
    # Add horizontal reference lines at 25 %, 50 %, 75 %.
    for frac in (0.25, 0.50, 0.75):
        ax.axhline(frac, color="grey", linewidth=0.7, linestyle=":")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_axis_histograms(data_by_split, output_dir):
    """Save overlaid histograms for each rotation-axis component."""
    axis_cols = [
        ("axis_x", "Rotation Axis X  (−1 = pure −X, +1 = pure +X)"),
        ("axis_y", "Rotation Axis Y  (−1 = pure −Y, +1 = pure +Y)"),
        ("axis_z", "Rotation Axis Z  (−1 = pure −Z, +1 = pure +Z)"),
    ]
    for col, title in axis_cols:
        save_overlaid_hist(
            data_by_split,
            col,
            bins=50,
            xlabel=col,
            title=title,
            output_path=output_dir / f"rotation_{col}.png",
            density=True,
        )


def component_limits(data_by_split, cols):
    """Compute shared component limits with a small margin."""
    values = []
    for split in SPLITS:
        for col in cols:
            if col in data_by_split[split]:
                values.append(data_by_split[split][col].to_numpy(dtype=np.float64))
    if not values:
        return (-1.0, 1.0)
    all_values = np.concatenate(values)
    if all_values.size == 0:
        return (-1.0, 1.0)
    vmin, vmax = float(np.min(all_values)), float(np.max(all_values))
    margin = max(0.05, (vmax - vmin) * 0.05)
    return vmin - margin, vmax + margin


def save_axis_pair_density(data_by_split, output_dir):
    """Save pairwise axis density plots with one panel per split."""
    pairs = [
        ("axis_x", "axis_y", "rotation_axis_xy.png"),
        ("axis_x", "axis_z", "rotation_axis_xz.png"),
        ("axis_y", "axis_z", "rotation_axis_yz.png"),
    ]
    lim = component_limits(data_by_split, ["axis_x", "axis_y", "axis_z"])
    for x_col, y_col, filename in pairs:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharex=True, sharey=True)
        for ax, split in zip(axes, SPLITS):
            df = data_by_split[split]
            x = df[x_col].to_numpy(dtype=np.float64)
            y = df[y_col].to_numpy(dtype=np.float64)
            if x.size:
                hist = ax.hist2d(x, y, bins=50, range=[lim, lim], density=True, cmap="viridis")
                fig.colorbar(hist[3], ax=ax, fraction=0.046, pad=0.04)
            ax.set_title(split)
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
            ax.set_xlim(lim)
            ax.set_ylim(lim)
        fig.suptitle(f"{x_col} vs {y_col}")
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=200)
        plt.close(fig)


def _save_log_translation_hist(data_by_split, output_path):
    """Translation magnitude on a log x-axis.

    A log x-axis is far clearer than manually computing log(m + eps) because:
    - x-tick labels remain in original metres (e.g. 0.01, 0.1, 1.0)
    - there are no negative values that shift the bulk of data to the right
    - the 2 cm training filter threshold is visible as a clear cut-off
    """
    from scipy.stats import gaussian_kde

    fig, ax = plt.subplots(figsize=(9, 6))

    split_arrays: dict[str, np.ndarray] = {}
    for split in SPLITS:
        if "translation_magnitude" not in data_by_split[split]:
            split_arrays[split] = np.array([])
            continue
        x = data_by_split[split]["translation_magnitude"].to_numpy(dtype=np.float64)
        x = x[np.isfinite(x) & (x > 0)]
        split_arrays[split] = x

    combined = np.concatenate([v for v in split_arrays.values() if v.size > 0])
    if combined.size == 0:
        plt.close(fig)
        return

    x_lo = max(float(np.percentile(combined, 1)), 1e-4)
    x_hi = float(np.percentile(combined, 99))
    x_eval = np.logspace(np.log10(x_lo), np.log10(x_hi), 500)

    for split in SPLITS:
        x = split_arrays[split]
        color = SPLIT_COLORS[split]
        n = x.size
        if n == 0:
            continue
        med = float(np.median(x))
        label = f"{split}  (n={n:,}, median={med:.3g} m)"
        log_x = np.log10(x)
        log_lo, log_hi = np.log10(x_lo), np.log10(x_hi)
        ax.hist(
            x, bins=np.logspace(log_lo, log_hi, 60),
            density=True, histtype="step", color=color, linewidth=1.4,
        )
        if n >= 10:
            try:
                kde_log = gaussian_kde(log_x)
                # Convert KDE from log10 space back to linear density.
                ax.plot(
                    x_eval,
                    kde_log(np.log10(x_eval)) / (x_eval * np.log(10)),
                    color=color, linewidth=2.5, label=label,
                )
            except Exception:
                ax.plot([], [], color=color, linewidth=2.5, label=label)
        else:
            ax.plot([], [], color=color, linewidth=2.5, label=label)
        ax.axvline(med, color=color, linewidth=1.2, linestyle="--", alpha=0.75)

    ax.set_xscale("log")
    ax.set_xlim(x_lo, x_hi)
    ax.set_xlabel("Translation magnitude (metres, log scale)")
    ax.set_ylabel("Probability density  (area = 1)")
    ax.set_title("Log-Scale Translation Magnitude Distribution")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_translation_direction(data_by_split, output_dir):
    """Save azimuth/elevation 2D histograms with identical axis limits."""
    xlim = (-np.pi, np.pi)
    ylim = (-np.pi / 2.0, np.pi / 2.0)
    for split in SPLITS:
        df = data_by_split[split]
        plt.figure(figsize=(8, 6))
        if len(df):
            plt.hist2d(
                df["azimuth"].to_numpy(dtype=np.float64),
                df["elevation"].to_numpy(dtype=np.float64),
                bins=60,
                range=[xlim, ylim],
                cmap="viridis",
            )
            plt.colorbar(label="Count")
        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.xlabel("Azimuth (rad)")
        plt.ylabel("Elevation (rad)")
        plt.title(f"Translation Direction: {split}")
        plt.tight_layout()
        plt.savefig(output_dir / f"translation_direction_{split}.png", dpi=200)
        plt.close()


def print_axis_reports(data_by_split):
    """Print mean and covariance of rotation-axis vectors for each split."""
    print("\nRotation axis mean and covariance")
    for split in SPLITS:
        axes = data_by_split[split][["axis_x", "axis_y", "axis_z"]].to_numpy(dtype=np.float64)
        print(f"\n[{split}]")
        if axes.size == 0:
            print("No valid samples.")
            continue
        mean = np.mean(axes, axis=0)
        cov = np.cov(axes, rowvar=False) if axes.shape[0] > 1 else np.zeros((3, 3))
        print("mean:", np.array2string(mean, precision=4, suppress_small=True))
        print("covariance:\n", np.array2string(cov, precision=4, suppress_small=True))


def interpret_results(data_by_split, rotation_stats, translation_stats, focal_stats, rot_tests, trans_tests, focal_tests):
    """Print concise data-driven interpretations."""
    print("\nInterpretation")

    train_rot = rotation_stats.set_index("split").loc["train"]
    train_trans = translation_stats.set_index("split").loc["train"]
    train_fx = focal_stats.set_index("split").loc["train"]
    for split in ("val", "test"):
        rot = rotation_stats.set_index("split").loc[split]
        trans = translation_stats.set_index("split").loc[split]
        fx = focal_stats.set_index("split").loc[split]
        pair_name = f"train vs {split}"
        rot_test = rot_tests[(rot_tests["split_a"] == "train") & (rot_tests["split_b"] == split)].iloc[0]
        trans_test = trans_tests[(trans_tests["split_a"] == "train") & (trans_tests["split_b"] == split)].iloc[0]
        fx_test = focal_tests[(focal_tests["split_a"] == "train") & (focal_tests["split_b"] == split)].iloc[0]

        if rot["median"] > train_rot["median"] * 1.2 and rot_test["KS_p"] < 0.05:
            print(f"- {split} contains significantly larger rotations than training.")
        elif rot["p99"] > train_rot["p99"] * 1.2:
            print(f"- {split} has a heavier tail of large-rotation samples than training.")
        else:
            print(f"- Rotation angles look broadly matched for {pair_name}.")

        if trans["median"] > train_trans["median"] * 1.2 and trans_test["KS_p"] < 0.05:
            print(f"- {split} contains significantly larger translation magnitudes than training.")
        elif trans["p99"] > train_trans["p99"] * 1.2:
            print(f"- {split} contains a heavier tail of large-baseline motions.")
        else:
            print(f"- Translation magnitudes look broadly matched for {pair_name}.")

        if fx["median"] > train_fx["median"] * 1.05 and fx_test["KS_p"] < 0.05:
            print(f"- {split} has significantly larger x focal lengths than training.")
        elif fx["median"] < train_fx["median"] * 0.95 and fx_test["KS_p"] < 0.05:
            print(f"- {split} has significantly smaller x focal lengths than training.")
        else:
            print(f"- X focal lengths look broadly matched for {pair_name}.")

    for split in SPLITS:
        df = data_by_split[split]
        if len(df) == 0:
            continue
        axes = df[["axis_x", "axis_y", "axis_z"]].to_numpy(dtype=np.float64)
        mean_abs_axis = np.mean(np.abs(axes), axis=0)
        dominant_axis = int(np.argmax(mean_abs_axis))
        axis_name = ["x", "y", "z"][dominant_axis]
        if mean_abs_axis[dominant_axis] > 0.65:
            print(f"- {split} rotations are concentrated around the {axis_name}-axis.")

        t_dirs = df[["t_unit_x", "t_unit_y", "t_unit_z"]].to_numpy(dtype=np.float64)
        mean_dir = np.linalg.norm(np.mean(t_dirs, axis=0))
        if mean_dir > 0.35:
            print(f"- {split} translation directions are biased toward a preferred direction.")
        else:
            print(f"- {split} translation directions are comparatively diffuse.")


def main():
    parser = argparse.ArgumentParser(description="Analyze camera-motion distributions by dataset split.")
    parser.add_argument("--config", default="config/training.yaml", help="Path to training YAML config.")
    parser.add_argument("--output-dir", default="results/camera_motion_analysis", help="Directory for PNG/CSV outputs.")
    args = parser.parse_args()

    cfg = load_config(resolve_path(args.config))
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    split_paths, split_roots = build_split_scene_paths(cfg)
    print(
        "[split] scenes: "
        + " / ".join(f"{split}={len(split_paths[split])}" for split in SPLITS)
    )

    data_by_split = {}
    diagnostics = {}
    for split in SPLITS:
        df, diag = collect_split_statistics(split_paths[split], split_roots[split], split)
        data_by_split[split] = df
        diagnostics[split] = diag
        print(
            f"[{split}] samples={diag['valid_samples']} "
            f"skipped_samples={diag['skipped_samples']} "
            f"skipped_pose_frames={diag['skipped_pose_frames']} "
            f"skipped_intrinsic_scenes={diag['skipped_intrinsic_scenes']}"
        )

    rotation_stats = summary_table(data_by_split, "rotation_angle_deg")
    translation_stats = summary_table(data_by_split, "translation_magnitude")
    focal_stats = summary_table(data_by_split, "focal_length_x")
    rot_tests = distribution_tests(data_by_split, "rotation_angle_deg")
    trans_tests = distribution_tests(data_by_split, "translation_magnitude")
    focal_tests = distribution_tests(data_by_split, "focal_length_x")

    rotation_stats.to_csv(output_dir / "rotation_statistics.csv", index=False)
    translation_stats.to_csv(output_dir / "translation_statistics.csv", index=False)
    focal_stats.to_csv(output_dir / "focal_length_x_statistics.csv", index=False)
    rot_tests.to_csv(output_dir / "rotation_distribution_tests.csv", index=False)
    trans_tests.to_csv(output_dir / "translation_distribution_tests.csv", index=False)
    focal_tests.to_csv(output_dir / "focal_length_x_distribution_tests.csv", index=False)

    save_overlaid_hist(
        data_by_split,
        "rotation_angle_deg",
        bins=50,
        xlabel="Rotation angle (deg)",
        title="Rotation Angle Distribution",
        output_path=output_dir / "rotation_histogram.png",
        density=True,
    )
    save_empirical_cdf(
        data_by_split,
        "rotation_angle_deg",
        xlabel="Rotation angle (deg)",
        title="Rotation Angle Empirical CDF",
        output_path=output_dir / "rotation_cdf.png",
    )
    save_axis_histograms(data_by_split, output_dir)
    save_axis_pair_density(data_by_split, output_dir)
    save_overlaid_hist(
        data_by_split,
        "translation_magnitude",
        bins=100,
        xlabel="Translation magnitude",
        title="Translation Magnitude Distribution",
        output_path=output_dir / "translation_histogram.png",
        density=True,
    )
    # Log-scale translation: plot the same data but on a log x-axis so
    # the shape of the distribution is visible without a manual transform
    # that produces confusing negative values.
    _save_log_translation_hist(data_by_split, output_dir / "translation_log_histogram.png")
    save_translation_direction(data_by_split, output_dir)
    save_overlaid_hist(
        data_by_split,
        "focal_length_x",
        bins=50,
        xlabel="Focal length fx (pixels)",
        title="X Focal Length Distribution",
        output_path=output_dir / "focal_length_x_histogram.png",
        density=True,
    )
    save_empirical_cdf(
        data_by_split,
        "focal_length_x",
        xlabel="Focal length fx (pixels)",
        title="X Focal Length Empirical CDF",
        output_path=output_dir / "focal_length_x_cdf.png",
    )

    print("\nRotation angle statistics")
    print(rotation_stats.to_string(index=False))
    print("\nTranslation magnitude statistics")
    print(translation_stats.to_string(index=False))
    print("\nX focal length statistics")
    print(focal_stats.to_string(index=False))
    print("\nRotation distribution comparison")
    print(rot_tests.to_string(index=False))
    print("\nTranslation magnitude distribution comparison")
    print(trans_tests.to_string(index=False))
    print("\nX focal length distribution comparison")
    print(focal_tests.to_string(index=False))
    print_axis_reports(data_by_split)
    interpret_results(data_by_split, rotation_stats, translation_stats, focal_stats, rot_tests, trans_tests, focal_tests)
    print(f"\nSaved outputs to: {output_dir}")


if __name__ == "__main__":
    main()
