import os
import argparse
import subprocess
import glob
import csv
import numpy as np
import re
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import concurrent
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading



def find_scenes_in_batches(sampled_data_dir, max_batches=None):
    """
    Find all scene folders organized as:
    sampled_data/batch1/sample1/scene0000_00/
    Returns list of tuples: (batch_name, sample_name, scene_name, scene_path)
    """
    scenes = []
    batch_dirs = sorted(glob.glob(os.path.join(sampled_data_dir, "batch*")))

    if max_batches:
        #randomize order to get a representative sample if not evaluating all
        #np.random.shuffle(batch_dirs)
        batch_dirs = batch_dirs[:max_batches]

    for batch_dir in batch_dirs:
        batch_name = os.path.basename(batch_dir)
        sample_dirs = sorted(glob.glob(os.path.join(batch_dir, "sample*")))

        for sample_dir in sample_dirs:
            sample_name = os.path.basename(sample_dir)
            scene_dirs = [d for d in glob.glob(os.path.join(sample_dir, "*"))
                         if os.path.isdir(d) and os.path.basename(d).startswith("scene")]

            for scene_dir in scene_dirs:
                scene_name = os.path.basename(scene_dir)
                scenes.append((batch_name, sample_name, scene_name, scene_dir))

    return scenes


def parse_depth_consistency_output(output: str) -> dict:
    """
    Parse output from metric/depth_consistency.py and return a dict with keys:
      rmse, mae, abs_rel, sq_rel (optional), delta1, delta2, delta3
    """
    output = (output or "").replace("\xa0", " ")
    float_re = r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"

    # 1) Try key=value style first
    kv_regex = (
        r"RMSE\s*[:=]?\s*" + float_re + r".*?"
        r"MAE\s*[:=]?\s*" + float_re + r".*?"
        r"AbsRel\s*[:=]?\s*" + float_re + r".*?"
    )
    m = re.search(kv_regex, output, flags=re.IGNORECASE | re.DOTALL)
    if m:
        try:
            rmse = float(m.group(1))
            mae = float(m.group(2))
            abs_rel = float(m.group(3))
            delta1 = delta2 = delta3 = None
            dm = re.search(r"delta1\s*[:=]?\s*" + float_re, output, flags=re.IGNORECASE)
            if dm:
                delta1 = float(dm.group(1))
            dm = re.search(r"delta2\s*[:=]?\s*" + float_re, output, flags=re.IGNORECASE)
            if dm:
                delta2 = float(dm.group(1))
            dm = re.search(r"delta3\s*[:=]?\s*" + float_re, output, flags=re.IGNORECASE)
            if dm:
                delta3 = float(dm.group(1))
            sq_rel = None
            sqm = re.search(r"SqRel\s*[:=]?\s*" + float_re, output, flags=re.IGNORECASE)
            if sqm:
                sq_rel = float(sqm.group(1))
            return {
                "rmse": rmse, "mae": mae, "abs_rel": abs_rel, "sq_rel": sq_rel,
                "delta1": delta1, "delta2": delta2, "delta3": delta3,
            }
        except Exception:
            pass

    # 2) Table-style: find "Average" row
    lines = output.splitlines()
    for idx, line in enumerate(lines):
        if "average" in line.lower():
            nums = re.findall(float_re, line)
            if not nums:
                for k in range(1, 3):
                    if idx + k < len(lines):
                        nums = re.findall(float_re, lines[idx + k])
                        if nums:
                            break
            if nums:
                vals = [float(x) for x in nums]
                if len(vals) == 6:
                    abs_rel, rmse, mae, delta1, delta2, delta3 = vals
                    return {"rmse": rmse, "mae": mae, "abs_rel": abs_rel, "sq_rel": None,
                            "delta1": delta1, "delta2": delta2, "delta3": delta3}
                elif len(vals) == 7:
                    abs_rel, rmse, mae, sq_rel, delta1, delta2, delta3 = vals
                    return {"rmse": rmse, "mae": mae, "abs_rel": abs_rel, "sq_rel": sq_rel,
                            "delta1": delta1, "delta2": delta2, "delta3": delta3}
                elif len(vals) >= 3:
                    return {
                        "rmse": vals[1] if len(vals) > 1 else None,
                        "mae": vals[2] if len(vals) > 2 else None,
                        "abs_rel": vals[0],
                        "sq_rel": None,
                        "delta1": vals[3] if len(vals) > 3 else None,
                        "delta2": vals[4] if len(vals) > 4 else None,
                        "delta3": vals[5] if len(vals) > 5 else None,
                    }

    # 3) Fallback
    all_nums = re.findall(float_re, output)
    if len(all_nums) >= 3:
        vals = [float(x) for x in all_nums[:6]]
        return {
            "rmse": vals[1] if len(vals) > 1 else None,
            "mae": vals[2] if len(vals) > 2 else None,
            "abs_rel": vals[0],
            "sq_rel": None,
            "delta1": None, "delta2": None, "delta3": None,
        }

    return {}

def run_photometric_pair(photo_cmd, i, j, si, sj, args):
    """Run one photometric pair and return (i, j, ssim, l2)."""
    try:
        photo_result = subprocess.run(photo_cmd, capture_output=True, text=True, timeout=60)
        photo_output = photo_result.stdout + photo_result.stderr
        ssim, l2 = parse_photometric_output(photo_output, debug=args.debug)
        return (i, j, si, sj, ssim, l2, None)
    except subprocess.TimeoutExpired:
        return (i, j, si, sj, None, None, "timeout")
    except Exception as e:
        return (i, j, si, sj, None, None, str(e))

def run_pixel_consistency_pair(pixel_cmd, i, j, si, sj, args):
    """Run one pixel consistency pair and return (i, j, mae, rmse)."""
    try:
        result = subprocess.run(pixel_cmd, capture_output=True, text=True, timeout=60)
        output = result.stdout + result.stderr
        mae, rmse = parse_pixel_consistency_output(output, debug=args.debug)
        return (i, j, si, sj, mae, rmse, None)
    except subprocess.TimeoutExpired:
        return (i, j, si, sj, None, None, "timeout")
    except Exception as e:
        return (i, j, si, sj, None, None, str(e))

def parse_pixel_consistency_output(output, debug=False):
    """Parse pixel_consistency.py output for MAE and RMSE averages."""
    output = output.replace('\xa0', ' ')
    float_re = r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
    mae_avg, rmse_avg = None, None

    mae_match  = re.search(rf"MAE\s*(?:avg|AVG)?\s*[:=]\s*{float_re}",  output, flags=re.IGNORECASE)
    rmse_match = re.search(rf"RMSE\s*(?:avg|AVG)?\s*[:=]\s*{float_re}", output, flags=re.IGNORECASE)

    if mae_match:
        mae_avg  = float(mae_match.group(1))
    if rmse_match:
        rmse_avg = float(rmse_match.group(1))

    if debug and (mae_avg is None or rmse_avg is None):
        print("      [DEBUG] Pixel consistency parse failed — raw subprocess output:")
        print("      " + "\n      ".join(output.splitlines()))
        print("      [DEBUG] End raw output")

    return mae_avg, rmse_avg

def parse_photometric_output(output, debug=False):
    """Parse photometric consistency output for SSIM and L2 metrics."""
    output = output.replace('\xa0', ' ')
    float_re = r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
    ssim_avg, l2_avg = None, None

    ssim_match = re.search(rf"SSIM\s*(?:avg|AVG)?\s*[:=]\s*{float_re}", output, flags=re.IGNORECASE)
    l2_match   = re.search(rf"L2\s*(?:avg|AVG)?\s*[:=]\s*{float_re}", output, flags=re.IGNORECASE)

    if ssim_match:
        ssim_avg = float(ssim_match.group(1))
    if l2_match:
        l2_avg = float(l2_match.group(1))

    if debug and (ssim_avg is None or l2_avg is None):
        print("      [DEBUG] Photometric parse failed — raw subprocess output:")
        print("      " + "\n      ".join(output.splitlines()))
        print("      [DEBUG] End raw output")

    return ssim_avg, l2_avg


def run_scene_eval(scene_path, args):
    """
    Run evaluation for a single scene.
    Photometric consistency is computed pairwise (mirroring run-scene-eval.py)
    and aggregated at the end, not stored per-frame.
    Returns dict with all metrics.
    """
    rgb_dir       = os.path.join(scene_path, "color")
    pose_dir      = os.path.join(scene_path, "pose")
    gt_depth_dir  = os.path.join(scene_path, "depth")
    intrinsics    = os.path.join(scene_path, "intrinsic", "intrinsic_color.txt")

    pred_depth_dir = os.path.join(scene_path, f"{args.model}_aligned") if args.align else os.path.join(scene_path, f"{args.model}_pred")


    # Validate required paths
    for d in [rgb_dir, pose_dir, gt_depth_dir, pred_depth_dir]:
        if not os.path.exists(d):
            if args.debug:
                print(f" Warning:  Missing directory: {d}")
            return None
    if not os.path.exists(intrinsics):
        if args.debug:
            print(f" Warning:  Missing intrinsics: {intrinsics}")
        return None

    results = {
        'scene_path': scene_path,
        'depth_metrics': {},
        'photometric_pairs': [],        # list of (ssim, l2) per valid pair
        'pixel_consistency_pairs': [],  # list of (mae, rmse) per valid pair
    }

    # Depth consistency
    #print(f"    Running depth consistency...")

    depth_cmd = [
        "python3", "metrics/depth_consistency.py",
        "--rgb_dir", rgb_dir,
        "--gt_dir", gt_depth_dir,
        "--pred_dir", pred_depth_dir,
        "--depth_scale_pred", str(args.depth_scale),
    ]
    try:
        depth_result = subprocess.run(depth_cmd, capture_output=True, text=True, timeout=120)
        depth_output = depth_result.stdout + depth_result.stderr
        if args.debug:
            print("      [DEBUG] depth_consistency raw output:")
            print("      " + "\n      ".join(depth_output.splitlines()))
        results['depth_metrics'] = parse_depth_consistency_output(depth_output)
        if not results['depth_metrics'] and args.debug:
            print("   Warning: Depth metrics parsed empty — check script output above with --debug")
    except Exception as e:
        print(f"   Warning: Depth consistency failed: {e}")
        return None

    # Photometric consistency

    #print(f"    Running photometric consistency...")

    def sorted_files(folder, ext):
        return sorted(glob.glob(os.path.join(folder, f"*.{ext}")))

    rgb_files        = sorted_files(rgb_dir, args.rgb_ext)
    pose_files       = sorted_files(pose_dir, args.pose_ext)
    pred_depth_files = sorted_files(pred_depth_dir, args.depth_ext)
    # also collect all GT depth files (any extension) and map by stem
    gt_depth_files = sorted(glob.glob(os.path.join(gt_depth_dir, "*")))

    # Stem-based matching so mismatched counts don't silently misalign
    def stem(p):
        return os.path.splitext(os.path.basename(p))[0]

    rgb_map  = {stem(p): p for p in rgb_files}
    pose_map = {stem(p): p for p in pose_files}
    pred_map = {stem(p): p for p in pred_depth_files}
    gt_map   = {stem(p): p for p in gt_depth_files}

    common_stems = sorted(set(rgb_map) & set(pose_map) & set(pred_map) & set(gt_map))
    n = len(common_stems)

    if n < 2:
        if args.debug:
            print(f" Warning:  Not enough matched frames (n={n}, need ≥2) — skipping photometric")
        return results

    #print(f"      Matched {n} frames, window={args.window}")

    # Build list of all pairs to process
    pair_tasks = []
    pixel_tasks = []
    for i in range(n):
        for j in range(i + 1, min(i + 1 + args.window, n)):
            si, sj = common_stems[i], common_stems[j]

            photo_cmd = [
                "python3", "metrics/photometric_consistency.py",
                "--img1",      rgb_map[si],
                "--img2",      rgb_map[sj],
                "--depth1",    pred_map[si],
                "--depth2",    pred_map[sj],
                "--intrinsics", intrinsics,
                "--pose1",     pose_map[si],
                "--pose2",     pose_map[sj],
                "--depth_scale", str(args.depth_scale),
            ]
            if args.cam_to_world:
                photo_cmd.append("--cam_to_world")
            else:
                photo_cmd.append("--world_to_cam")
            pair_tasks.append((photo_cmd, i, j, si, sj))

            pixel_cmd = [
                "python3", "metrics/pixel_consistency.py",
                "--gt_depth1",   gt_map[si],
                "--gt_depth2",   gt_map[sj],
                "--pred_depth1", pred_map[si],
                "--pred_depth2", pred_map[sj],
                "--intrinsics",  intrinsics,
                "--pose1",       pose_map[si],
                "--pose2",       pose_map[sj],
                "--depth_scale_gt",   str(args.depth_scale_gt),
                "--depth_scale_pred", str(args.depth_scale),
            ]
            if args.cam_to_world:
                pixel_cmd.append("--cam_to_world")
            else:
                pixel_cmd.append("--world_to_cam")
            pixel_tasks.append((pixel_cmd, i, j, si, sj))

    # Process all pairs in parallel
    max_workers = 4
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        photo_futures = {
            executor.submit(run_photometric_pair, cmd, i, j, si, sj, args): 'photo'
            for cmd, i, j, si, sj in pair_tasks
        }
        pixel_futures = {
            executor.submit(run_pixel_consistency_pair, cmd, i, j, si, sj, args): 'pixel'
            for cmd, i, j, si, sj in pixel_tasks
        }
        all_futures = {**photo_futures, **pixel_futures}

        for future in as_completed(all_futures):
            kind = all_futures[future]
            i, j, si, sj, val1, val2, error = future.result()

            if val1 is not None and val2 is not None:
                if kind == 'photo':
                    results['photometric_pairs'].append((val1, val2))       # (ssim, l2)
                else:
                    results['pixel_consistency_pairs'].append((val1, val2)) # (mae, rmse)
            elif args.debug:
                label = "Photometric" if kind == 'photo' else "Pixel"
                if error == "timeout":
                    print(f" Warning:  {label} pair {i}→{j} timed out")
                elif error:
                    print(f" Warning:  {label} pair {i}→{j} failed: {error}")
                else:
                    print(f" Warning:  {label} pair {i}→{j} ({si}→{sj}): parse returned None")

    # Print concise one-line summary
    photo_valid = results['photometric_pairs']
    pixel_valid = results['pixel_consistency_pairs']
    depth = results['depth_metrics']

    summary_parts = []
    if depth.get('rmse'):
        summary_parts.append(f"RMSE={depth['rmse']:.3f}")
    if depth.get('mae'):
        summary_parts.append(f"MAE={depth['mae']:.3f}")
    if photo_valid:
        mean_ssim = np.mean([s for s, _ in photo_valid])
        mean_l2   = np.mean([l for _, l in photo_valid])
        summary_parts.append(f"Photo-SSIM={mean_ssim:.3f}")
        summary_parts.append(f"Photo-L2={mean_l2:.4f}")
    if pixel_valid:
        mean_mae  = np.mean([m for m, _ in pixel_valid])
        mean_rmse = np.mean([r for _, r in pixel_valid])
        summary_parts.append(f"Pixel-MAE={mean_mae:.4f}")
        summary_parts.append(f"Pixel-RMSE={mean_rmse:.4f}")
    summary_parts.append(f"pairs={len(photo_valid)}")

    if summary_parts:
        print(f"    ✓ {', '.join(summary_parts)}")

    return results


def save_results_to_csv(all_results, output_dir):
    """Save per-pair photometric/pixel results and scene-level summary to CSV."""

    # --- Per-pair photometric CSV ---
    pairs_path = os.path.join(output_dir, "photometric_pairs_detailed.csv")
    with open(pairs_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['batch', 'sample', 'scene', 'pair_index', 'ssim', 'l2'])
        for result in all_results:
            for idx, (ssim, l2) in enumerate(result['photometric_pairs']):
                writer.writerow([result['batch'], result['sample'], result['scene'],
                                 idx, ssim, l2])
    print(f"\n Per-pair photometric results saved to: {pairs_path}")

    # --- Per-pair pixel consistency CSV ---
    pixel_pairs_path = os.path.join(output_dir, "pixel_consistency_pairs_detailed.csv")
    with open(pixel_pairs_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['batch', 'sample', 'scene', 'pair_index', 'mae', 'rmse'])
        for result in all_results:
            for idx, (mae, rmse) in enumerate(result['pixel_consistency_pairs']):
                writer.writerow([result['batch'], result['sample'], result['scene'],
                                 idx, mae, rmse])
    print(f" Per-pair pixel consistency results saved to: {pixel_pairs_path}")

    # --- Scene-level summary CSV ---
    summary_path = os.path.join(output_dir, "scene_metrics_summary.csv")
    with open(summary_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'batch', 'sample', 'scene',
            'rmse', 'mae', 'abs_rel', 'sq_rel', 'delta1', 'delta2', 'delta3',
            'mean_photo_ssim', 'std_photo_ssim', 'mean_photo_l2', 'std_photo_l2', 'num_photo_pairs',
            'mean_pixel_mae', 'std_pixel_mae', 'mean_pixel_rmse', 'std_pixel_rmse', 'num_pixel_pairs',
        ])
        for result in all_results:
            depth = result['depth_metrics']

            photo_valid = result['photometric_pairs']
            if photo_valid:
                ssims = [s for s, _ in photo_valid]
                l2s   = [l for _, l in photo_valid]
                mean_photo_ssim, std_photo_ssim = np.mean(ssims), np.std(ssims)
                mean_photo_l2,   std_photo_l2   = np.mean(l2s),   np.std(l2s)
                num_photo_pairs = len(photo_valid)
            else:
                mean_photo_ssim = std_photo_ssim = mean_photo_l2 = std_photo_l2 = num_photo_pairs = None

            pixel_valid = result['pixel_consistency_pairs']
            if pixel_valid:
                maes  = [m for m, _ in pixel_valid]
                rmses = [r for _, r in pixel_valid]
                mean_pixel_mae,  std_pixel_mae  = np.mean(maes),  np.std(maes)
                mean_pixel_rmse, std_pixel_rmse = np.mean(rmses), np.std(rmses)
                num_pixel_pairs = len(pixel_valid)
            else:
                mean_pixel_mae = std_pixel_mae = mean_pixel_rmse = std_pixel_rmse = num_pixel_pairs = None

            writer.writerow([
                result['batch'], result['sample'], result['scene'],
                depth.get('rmse'), depth.get('mae'), depth.get('abs_rel'),
                depth.get('sq_rel'), depth.get('delta1'), depth.get('delta2'), depth.get('delta3'),
                mean_photo_ssim, std_photo_ssim, mean_photo_l2, std_photo_l2, num_photo_pairs,
                mean_pixel_mae, std_pixel_mae, mean_pixel_rmse, std_pixel_rmse, num_pixel_pairs,
            ])
    print(f" Scene summary saved to: {summary_path}")


def create_visualizations(all_results, output_dir):
    """Create box plots and overall statistics."""

    photo_ssim_all, photo_l2_all = [], []
    pixel_mae_all, pixel_rmse_all = [], []
    rmse_all, mae_all, abs_rel_all, delta1_all = [], [], [], []

    for result in all_results:
        for ssim, l2 in result['photometric_pairs']:
            photo_ssim_all.append(ssim)
            photo_l2_all.append(l2)
        for mae, rmse in result['pixel_consistency_pairs']:
            pixel_mae_all.append(mae)
            pixel_rmse_all.append(rmse)

        depth = result['depth_metrics']
        if depth.get('rmse'):    rmse_all.append(depth['rmse'])
        if depth.get('mae'):     mae_all.append(depth['mae'])
        if depth.get('abs_rel'): abs_rel_all.append(depth['abs_rel'])
        if depth.get('delta1'):  delta1_all.append(depth['delta1'])

    sns.set_style("whitegrid")

    if photo_ssim_all and photo_l2_all:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        axes[0].boxplot(photo_ssim_all, vert=True, patch_artist=True)
        axes[0].set_ylabel('SSIM')
        axes[0].set_title(f'Photometric SSIM (n={len(photo_ssim_all)} pairs)')
        axes[0].grid(True, alpha=0.3)
        axes[1].boxplot(photo_l2_all, vert=True, patch_artist=True)
        axes[1].set_ylabel('L2 Error')
        axes[1].set_title(f'Photometric L2 (n={len(photo_l2_all)} pairs)')
        axes[1].grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'photometric_boxplots.png'), dpi=150)
        plt.close()

    if pixel_mae_all and pixel_rmse_all:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        axes[0].boxplot(pixel_mae_all, vert=True, patch_artist=True)
        axes[0].set_ylabel('MAE')
        axes[0].set_title(f'Pixel Consistency MAE (n={len(pixel_mae_all)} pairs)')
        axes[0].grid(True, alpha=0.3)
        axes[1].boxplot(pixel_rmse_all, vert=True, patch_artist=True)
        axes[1].set_ylabel('RMSE')
        axes[1].set_title(f'Pixel Consistency RMSE (n={len(pixel_rmse_all)} pairs)')
        axes[1].grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'pixel_consistency_boxplots.png'), dpi=150)
        plt.close()

    if rmse_all and mae_all:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        for ax, data, label in zip(
            axes.flat,
            [rmse_all, mae_all, abs_rel_all, delta1_all],
            ['RMSE', 'MAE', 'Abs Rel', 'δ1 (accuracy)']
        ):
            ax.boxplot(data, vert=True, patch_artist=True)
            ax.set_ylabel(label)
            ax.set_title(f'{label} Distribution (n={len(data)} scenes)')
            ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'depth_consistency_boxplots.png'), dpi=150)
        plt.close()

    stats_path = os.path.join(output_dir, 'overall_statistics.txt')
    with open(stats_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("OVERALL STATISTICS SUMMARY\n")
        f.write("=" * 60 + "\n\n")

        f.write("Photometric Consistency (pair-level):\n")
        f.write("-" * 40 + "\n")
        if photo_ssim_all:
            f.write(f"  SSIM: mean={np.mean(photo_ssim_all):.4f}, std={np.std(photo_ssim_all):.4f}, "
                    f"median={np.median(photo_ssim_all):.4f}, min={np.min(photo_ssim_all):.4f}, max={np.max(photo_ssim_all):.4f}\n")
        if photo_l2_all:
            f.write(f"  L2:   mean={np.mean(photo_l2_all):.6f}, std={np.std(photo_l2_all):.6f}, "
                    f"median={np.median(photo_l2_all):.6f}, min={np.min(photo_l2_all):.6f}, max={np.max(photo_l2_all):.6f}\n")

        f.write("\nPixel Consistency (pair-level):\n")
        f.write("-" * 40 + "\n")
        if pixel_mae_all:
            f.write(f"  MAE:  mean={np.mean(pixel_mae_all):.6f}, std={np.std(pixel_mae_all):.6f}, "
                    f"median={np.median(pixel_mae_all):.6f}, min={np.min(pixel_mae_all):.6f}, max={np.max(pixel_mae_all):.6f}\n")
        if pixel_rmse_all:
            f.write(f"  RMSE: mean={np.mean(pixel_rmse_all):.6f}, std={np.std(pixel_rmse_all):.6f}, "
                    f"median={np.median(pixel_rmse_all):.6f}, min={np.min(pixel_rmse_all):.6f}, max={np.max(pixel_rmse_all):.6f}\n")

        f.write("\nDepth Consistency (scene-level):\n")
        f.write("-" * 40 + "\n")
        for data, label in [(rmse_all, "RMSE"), (mae_all, "MAE"),
                            (abs_rel_all, "AbsRel"), (delta1_all, "δ1")]:
            if data:
                f.write(f"  {label:8s}: mean={np.mean(data):.4f}, std={np.std(data):.4f}, "
                        f"median={np.median(data):.4f}\n")

        f.write("\n" + "=" * 60 + "\n")
        f.write(f"Total scenes evaluated:          {len(all_results)}\n")
        f.write(f"Total photometric pairs:         {len(photo_ssim_all)}\n")
        f.write(f"Total pixel consistency pairs:   {len(pixel_mae_all)}\n")

    print(f" Saved: overall_statistics.txt")


def main():
    parser = argparse.ArgumentParser(description="Batch evaluation of sampled scenes")

    parser.add_argument("--sampled_data_dir", default="datasets/sampled_data")
    parser.add_argument("--output_dir", default="results/batch_eval")
    parser.add_argument("--max_batches", type=int, default=None)
    parser.add_argument("--model", default="depth-pro", choices=["depth-pro", "zoe-depth"])

    parser.add_argument("--rgb_ext",   default="jpg")
    parser.add_argument("--depth_ext", default="npz")
    parser.add_argument("--pose_ext",  default="txt")

    parser.add_argument("--depth_scale",    type=float, default=1.0,
                        help="Divisor for predicted depth maps (usually 1.0 for .npz)")
    parser.add_argument("--depth_scale_gt", type=float, default=1000.0,
                        help="Divisor for GT depth maps (e.g. 1000 for ScanNet mm→m PNG)")
    parser.add_argument("--cam_to_world", action="store_true")
    parser.add_argument("--window", type=int, default=1)
    parser.add_argument("--align", action="store_true", help="Use aligned predictions for evaluation")

    # New: print raw subprocess output when parsing returns None
    parser.add_argument("--debug", action="store_true",
                        help="Print raw subprocess output when metric parsing fails")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\n🔍 Searching for scenes in: {args.sampled_data_dir}")
    scenes = find_scenes_in_batches(args.sampled_data_dir, args.max_batches)
    print(f" Found {len(scenes)} scenes\n")

    if not scenes:
        print("No scenes found! Check directory structure.")
        return

    all_results = []

    for idx, (batch, sample, scene, scene_path) in enumerate(scenes, 1):
        print(f"\n[{idx}/{len(scenes)}] {batch}/{sample}/{scene}")
        result = run_scene_eval(scene_path, args)
        if result:
            result['batch']  = batch
            result['sample'] = sample
            result['scene']  = scene
            all_results.append(result)
            #print(f"Completed")
        else:
            print(f"Failed or skipped")

    print("\n" + "=" * 60)
    print(f" Saving results for {len(all_results)} scenes...")

    if all_results:
        save_results_to_csv(all_results, args.output_dir)
        create_visualizations(all_results, args.output_dir)
        print(f"\n All results saved to: {args.output_dir}")
    else:
        print("\n No successful evaluations to save.")

    print("\n Batch evaluation complete!\n")


if __name__ == "__main__":
    main()