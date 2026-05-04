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


def find_scenes_in_batches(sampled_data_dir, max_batches=None):
    """
    Find all scene folders organized as:
    sampled_data/batch1/sample1/scene0000_00/
    Returns list of tuples: (batch_name, sample_name, scene_name, scene_path)
    """
    scenes = []
    batch_dirs = sorted(glob.glob(os.path.join(sampled_data_dir, "batch*")))
    
    if max_batches:
        batch_dirs = batch_dirs[:max_batches]
    
    for batch_dir in batch_dirs:
        batch_name = os.path.basename(batch_dir)
        sample_dirs = sorted(glob.glob(os.path.join(batch_dir, "sample*")))
        
        for sample_dir in sample_dirs:
            sample_name = os.path.basename(sample_dir)
            # Each sample should contain one or more scene folders
            scene_dirs = [d for d in glob.glob(os.path.join(sample_dir, "*")) 
                         if os.path.isdir(d) and os.path.basename(d).startswith("scene")]
            
            for scene_dir in scene_dirs:
                scene_name = os.path.basename(scene_dir)
                scenes.append((batch_name, sample_name, scene_name, scene_dir))
    
    return scenes


def parse_depth_consistency_output(output):
    """
    Parse depth consistency script output to extract metrics.
    Looks for lines like:
    Average:  RMSE=X.XXX  MAE=X.XXX  AbsRel=X.XXX  SqRel=X.XXX  δ1=X.XXX  δ2=X.XXX  δ3=X.XXX
    """
    metrics = {}
    avg_pattern = r"Average:\s+RMSE=([\d.]+)\s+MAE=([\d.]+)\s+AbsRel=([\d.]+)\s+SqRel=([\d.]+)\s+δ1=([\d.]+)\s+δ2=([\d.]+)\s+δ3=([\d.]+)"
    
    match = re.search(avg_pattern, output)
    if match:
        metrics['rmse'] = float(match.group(1))
        metrics['mae'] = float(match.group(2))
        metrics['abs_rel'] = float(match.group(3))
        metrics['sq_rel'] = float(match.group(4))
        metrics['delta1'] = float(match.group(5))
        metrics['delta2'] = float(match.group(6))
        metrics['delta3'] = float(match.group(7))
    
    return metrics


def parse_photometric_output(output):
    """Parse photometric consistency output for SSIM and L2 metrics."""
    output = output.replace('\xa0', ' ')
    float_re = r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
    ssim_avg, l2_avg = None, None

    ssim_match = re.search(rf"SSIM\s*(?:avg|AVG)?\s*[:=]\s*{float_re}", output, flags=re.IGNORECASE)
    l2_match = re.search(rf"L2\s*(?:avg|AVG)?\s*[:=]\s*{float_re}", output, flags=re.IGNORECASE)

    if ssim_match:
        ssim_avg = float(ssim_match.group(1))
    if l2_match:
        l2_avg = float(l2_match.group(1))

    return ssim_avg, l2_avg


def run_scene_eval(scene_path, args):
    """
    Run evaluation for a single scene using run-scene-eval.py logic.
    Returns dict with all metrics.
    """
    rgb_dir = os.path.join(scene_path, "color")
    pose_dir = os.path.join(scene_path, "pose")
    gt_depth_dir = os.path.join(scene_path, "depth")
    intrinsics = os.path.join(scene_path, "intrinsic", "intrinsic_color.txt")
    
    # Determine pred_depth_dir based on model type
    pred_depth_dir = os.path.join(scene_path, f"{args.model}_pred")
    
    # Check if required directories exist
    required_dirs = [rgb_dir, pose_dir, gt_depth_dir, pred_depth_dir]
    for d in required_dirs:
        if not os.path.exists(d):
            print(f"  ⚠️  Missing directory: {d}")
            return None
    
    if not os.path.exists(intrinsics):
        print(f"  ⚠️  Missing intrinsics: {intrinsics}")
        return None
    
    results = {
        'scene_path': scene_path,
        'depth_metrics': {},
        'photometric_metrics': []
    }
    
    # 1. Run depth consistency
    print(f"    Running depth consistency...")
    depth_cmd = [
        "python3", "scripts/depth_consistency.py",
        "--rgb_dir", rgb_dir,
        "--gt_dir", gt_depth_dir,
        "--pred_dir", pred_depth_dir,
    ]
    
    try:
        depth_result = subprocess.run(depth_cmd, capture_output=True, text=True, timeout=120)
        depth_output = depth_result.stdout + depth_result.stderr
        results['depth_metrics'] = parse_depth_consistency_output(depth_output)
    except Exception as e:
        print(f"    ❌ Depth consistency failed: {e}")
        return None
    
    # 2. Run photometric consistency
    print(f"    Running photometric consistency...")
    rgb_files = sorted(glob.glob(os.path.join(rgb_dir, f"*.{args.rgb_ext}")))
    pose_files = sorted(glob.glob(os.path.join(pose_dir, f"*.{args.pose_ext}")))
    pred_depth_files = sorted(glob.glob(os.path.join(pred_depth_dir, f"*.{args.depth_ext}")))
    
    n = min(len(rgb_files), len(pose_files), len(pred_depth_files))
    
    if n < 2:
        print(f"    ⚠️  Not enough frames for photometric consistency (found {n})")
        return results
    
    for i in range(n):
        for j in range(i + 1, min(i + 1 + args.window, n)):
            photo_cmd = [
                "python3", "scripts/photometric_consistency.py",
                "--img1", rgb_files[i],
                "--img2", rgb_files[j],
                "--depth1", pred_depth_files[i],
                "--depth2", pred_depth_files[j],
                "--intrinsics", intrinsics,
                "--pose1", pose_files[i],
                "--pose2", pose_files[j],
                "--depth_scale", str(args.depth_scale),
            ]
            
            if args.cam_to_world:
                photo_cmd.append("--cam_to_world")
            else:
                photo_cmd.append("--world_to_cam")
            
            try:
                photo_result = subprocess.run(photo_cmd, capture_output=True, text=True, timeout=60)
                photo_output = photo_result.stdout + photo_result.stderr
                ssim, l2 = parse_photometric_output(photo_output)
                
                if ssim is not None and l2 is not None:
                    results['photometric_metrics'].append({'ssim': ssim, 'l2': l2})
            except Exception as e:
                print(f"    ⚠️  Photometric pair {i}-{j} failed: {e}")
                continue
    
    return results


def save_results_to_csv(all_results, output_dir):
    """Save per-scene results to CSV files."""
    csv_path = os.path.join(output_dir, "scene_metrics_summary.csv")
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        # Header
        writer.writerow([
            'batch', 'sample', 'scene', 
            'rmse', 'mae', 'abs_rel', 'sq_rel', 'delta1', 'delta2', 'delta3',
            'mean_ssim', 'std_ssim', 'mean_l2', 'std_l2', 'num_photo_pairs'
        ])
        
        for result in all_results:
            batch = result['batch']
            sample = result['sample']
            scene = result['scene']
            depth = result['depth_metrics']
            photo = result['photometric_metrics']
            
            # Aggregate photometric metrics
            if photo:
                ssim_values = [p['ssim'] for p in photo]
                l2_values = [p['l2'] for p in photo]
                mean_ssim = np.mean(ssim_values)
                std_ssim = np.std(ssim_values)
                mean_l2 = np.mean(l2_values)
                std_l2 = np.std(l2_values)
                num_pairs = len(photo)
            else:
                mean_ssim = std_ssim = mean_l2 = std_l2 = num_pairs = None
            
            writer.writerow([
                batch, sample, scene,
                depth.get('rmse'), depth.get('mae'), depth.get('abs_rel'), 
                depth.get('sq_rel'), depth.get('delta1'), depth.get('delta2'), depth.get('delta3'),
                mean_ssim, std_ssim, mean_l2, std_l2, num_pairs
            ])
    
    print(f"\n✅ Results saved to: {csv_path}")


def create_visualizations(all_results, output_dir):
    """Create box plots and histograms for overall statistics."""
    
    # Extract all metrics
    ssim_all = []
    l2_all = []
    rmse_all = []
    mae_all = []
    abs_rel_all = []
    delta1_all = []
    
    for result in all_results:
        if result['photometric_metrics']:
            ssim_all.extend([p['ssim'] for p in result['photometric_metrics']])
            l2_all.extend([p['l2'] for p in result['photometric_metrics']])
        
        depth = result['depth_metrics']
        if depth.get('rmse'):
            rmse_all.append(depth['rmse'])
        if depth.get('mae'):
            mae_all.append(depth['mae'])
        if depth.get('abs_rel'):
            abs_rel_all.append(depth['abs_rel'])
        if depth.get('delta1'):
            delta1_all.append(depth['delta1'])
    
    # Set style
    sns.set_style("whitegrid")
    
    # 1. Photometric consistency box plots
    if ssim_all and l2_all:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        axes[0].boxplot(ssim_all, vert=True, patch_artist=True)
        axes[0].set_ylabel('SSIM')
        axes[0].set_title(f'SSIM Distribution (n={len(ssim_all)} pairs)')
        axes[0].grid(True, alpha=0.3)
        
        axes[1].boxplot(l2_all, vert=True, patch_artist=True)
        axes[1].set_ylabel('L2 Error')
        axes[1].set_title(f'L2 Distribution (n={len(l2_all)} pairs)')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'photometric_boxplots.png'), dpi=150)
        plt.close()
        print(f"✅ Saved: photometric_boxplots.png")
    
    # 2. Depth consistency metrics box plots
    if rmse_all and mae_all:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        axes[0, 0].boxplot(rmse_all, vert=True, patch_artist=True)
        axes[0, 0].set_ylabel('RMSE')
        axes[0, 0].set_title(f'RMSE Distribution (n={len(rmse_all)} scenes)')
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].boxplot(mae_all, vert=True, patch_artist=True)
        axes[0, 1].set_ylabel('MAE')
        axes[0, 1].set_title(f'MAE Distribution (n={len(mae_all)} scenes)')
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[1, 0].boxplot(abs_rel_all, vert=True, patch_artist=True)
        axes[1, 0].set_ylabel('Abs Rel')
        axes[1, 0].set_title(f'Absolute Relative Error Distribution')
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].boxplot(delta1_all, vert=True, patch_artist=True)
        axes[1, 1].set_ylabel('δ1 (accuracy)')
        axes[1, 1].set_title(f'δ1 Distribution (higher is better)')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'depth_consistency_boxplots.png'), dpi=150)
        plt.close()
        print(f"✅ Saved: depth_consistency_boxplots.png")
    
    # 3. Overall statistics summary
    stats_path = os.path.join(output_dir, 'overall_statistics.txt')
    with open(stats_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("OVERALL STATISTICS SUMMARY\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("Photometric Consistency:\n")
        f.write("-" * 40 + "\n")
        if ssim_all:
            f.write(f"  SSIM: mean={np.mean(ssim_all):.4f}, std={np.std(ssim_all):.4f}, "
                   f"median={np.median(ssim_all):.4f}, min={np.min(ssim_all):.4f}, max={np.max(ssim_all):.4f}\n")
        if l2_all:
            f.write(f"  L2:   mean={np.mean(l2_all):.6f}, std={np.std(l2_all):.6f}, "
                   f"median={np.median(l2_all):.6f}, min={np.min(l2_all):.6f}, max={np.max(l2_all):.6f}\n")
        
        f.write("\nDepth Consistency:\n")
        f.write("-" * 40 + "\n")
        if rmse_all:
            f.write(f"  RMSE:    mean={np.mean(rmse_all):.4f}, std={np.std(rmse_all):.4f}, median={np.median(rmse_all):.4f}\n")
        if mae_all:
            f.write(f"  MAE:     mean={np.mean(mae_all):.4f}, std={np.std(mae_all):.4f}, median={np.median(mae_all):.4f}\n")
        if abs_rel_all:
            f.write(f"  AbsRel:  mean={np.mean(abs_rel_all):.4f}, std={np.std(abs_rel_all):.4f}, median={np.median(abs_rel_all):.4f}\n")
        if delta1_all:
            f.write(f"  δ1:      mean={np.mean(delta1_all):.4f}, std={np.std(delta1_all):.4f}, median={np.median(delta1_all):.4f}\n")
        
        f.write("\n" + "=" * 60 + "\n")
        f.write(f"Total scenes evaluated: {len(all_results)}\n")
        f.write(f"Total photometric pairs: {len(ssim_all)}\n")
    
    print(f"✅ Saved: overall_statistics.txt")


def main():
    parser = argparse.ArgumentParser(description="Batch evaluation of sampled scenes")
    
    # Input/output
    parser.add_argument("--sampled_data_dir", default="datasets/sampled_data",
                       help="Root directory containing batch*/sample*/scene* structure")
    parser.add_argument("--output_dir", default="results/batch_eval",
                       help="Directory to save results (CSV + plots)")
    parser.add_argument("--max_batches", type=int, default=None,
                       help="Maximum number of batches to process (default: all)")
    
    # Model selection
    parser.add_argument("--model", default="depth-pro", choices=["depth-pro", "zoe-depth"],
                       help="Which depth model predictions to evaluate")
    
    # File formats
    parser.add_argument("--rgb_ext", default="jpg")
    parser.add_argument("--depth_ext", default="npz", help="Extension for predicted depth files")
    parser.add_argument("--pose_ext", default="txt")
    
    # Settings
    parser.add_argument("--depth_scale", type=float, default=1.0,
                       help="Depth scale factor (1.0 for npz/npy, 1000.0 for uint16 PNG in mm)")
    parser.add_argument("--cam_to_world", action="store_true",
                       help="Poses are camera-to-world (default: world-to-cam)")
    parser.add_argument("--window", type=int, default=1,
                       help="Temporal window for photometric pairs")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find all scenes
    print(f"\n🔍 Searching for scenes in: {args.sampled_data_dir}")
    scenes = find_scenes_in_batches(args.sampled_data_dir, args.max_batches)
    print(f"✅ Found {len(scenes)} scenes across batches\n")
    
    if not scenes:
        print("❌ No scenes found! Check directory structure.")
        return
    
    # Process each scene
    all_results = []
    
    for idx, (batch, sample, scene, scene_path) in enumerate(scenes, 1):
        print(f"\n[{idx}/{len(scenes)}] Processing {batch}/{sample}/{scene}")
        
        result = run_scene_eval(scene_path, args)
        
        if result:
            result['batch'] = batch
            result['sample'] = sample
            result['scene'] = scene
            all_results.append(result)
            print(f"  ✅ Completed")
        else:
            print(f"  ❌ Failed or skipped")
    
    # Save results
    print("\n" + "=" * 60)
    print(f"📊 Saving results for {len(all_results)} scenes...")
    
    if all_results:
        save_results_to_csv(all_results, args.output_dir)
        create_visualizations(all_results, args.output_dir)
        print(f"\n✅ All results saved to: {args.output_dir}")
    else:
        print("\n❌ No successful evaluations to save.")
    
    print("\n🎉 Batch evaluation complete!\n")


if __name__ == "__main__":
    main()
