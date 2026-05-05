import numpy as np
import os
import cv2

#Script to compute median scaling factor to align predicted depth to GT depth, using only valid pixels defined by the mask

def find_scenes_in_batches(sampled_data_dir, max_batches=None):
    """
    Find all scene folders organized as:
    sampled_data/batch#/sample#/scene####_##/
    Returns list of tuples: (batch_name, sample_name, scene_name, scene_path)
    """
    import glob
    scenes = []
    batch_dirs = sorted(glob.glob(os.path.join(sampled_data_dir, "batch*")))
    
    if max_batches:
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

def align_scale(pred, gt, mask):
    scale = np.median(gt[mask]) / (np.median(pred[mask]) + EPS)
    return pred * scale

def load_depth(path, scale):
    path = os.path.expanduser(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Depth file not found: {path}")

    ext = os.path.splitext(path)[1].lower()

    if ext == ".npy":
        depth = np.load(path)
        depth = np.asarray(depth).astype(np.float32)
        # squeeze leading singleton (1,H,W) -> (H,W)
        if depth.ndim == 3 and depth.shape[0] == 1:
            depth = depth[0]
        # handle trailing singleton (H,W,1) or RGB (H,W,3)
        if depth.ndim == 3 and depth.shape[-1] in (1, 3, 4):
            depth = depth[..., 0]
        return depth.astype(np.float32)

    if ext == ".npz":
        data = np.load(path, allow_pickle=True)
        for key in ["depth", "pred", "prediction", "arr_0", "data"]:
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

    # fallback for PNG/JPG/other image formats
    depth = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise ValueError(f"Failed to load depth image: {path}")
    depth = depth.astype(np.float32)

    # keep existing behaviour: divide by scale for image-based depths (e.g. uint16 mm -> m)
    return depth / scale

EPS = 1e-8

def main():
    import argparse
    import glob
    
    parser = argparse.ArgumentParser(
        description="Batch align predicted depths to GT depths (median scaling) for all scenes in sampled_data")
    parser.add_argument("--sampled_data_dir", type=str, default="datasets/sampled_data",
                        help="Root directory containing batch*/sample*/scene*/ folders")
    parser.add_argument("--models", type=str, default="depth-pro,zoe-depth",
                        help="Comma-separated model names (e.g., depth-pro,zoe-depth)")
    parser.add_argument("--max_batches", type=int, default=None,
                        help="Limit number of batches to process (default: all)")
    parser.add_argument("--pred_scale_depth_pro", type=float, default=1.0,
                        help="Scale for depth-pro predictions (default: 1.0, already in meters)")
    parser.add_argument("--pred_scale_zoe_depth", type=float, default=1000.0,
                        help="Scale for zoe-depth predictions (default: 1000.0, PNG in mm)")
    parser.add_argument("--gt_scale", type=float, default=1.0,
                        help="Scale for GT depth files (default: 1.0, .npy in meters)")
    args = parser.parse_args()

    sampled_data_dir = os.path.expanduser(args.sampled_data_dir)
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    
    print(f"\n🔍 Searching for scenes in: {sampled_data_dir}")
    scenes = find_scenes_in_batches(sampled_data_dir, args.max_batches)
    print(f"✓ Found {len(scenes)} scenes")
    print(f"✓ Models to align: {', '.join(models)}\n")

    if not scenes:
        print(" Warning: No scenes found! Check directory structure.")
        return

    total_aligned = 0
    total_failed = 0

    for idx, (batch, sample, scene, scene_path) in enumerate(scenes, 1):
        print(f"[{idx}/{len(scenes)}] {batch}/{sample}/{scene}")
        
        gt_dir = os.path.join(scene_path, "depth")
        if not os.path.exists(gt_dir):
            print(f" Warning:  Missing GT depth directory, skipping")
            continue

        # Get GT file stems
        gt_files = sorted(glob.glob(os.path.join(gt_dir, "*.npy")))
        if not gt_files:
            print(f"  Warning  No GT .npy files found, skipping")
            continue
        
        gt_by_stem = {os.path.splitext(os.path.basename(f))[0]: f for f in gt_files}

        for model in models:
            pred_dir = os.path.join(scene_path, f"{model}_pred")
            if not os.path.exists(pred_dir):
                print(f" Warning:  Missing {model}_pred directory, skipping model")
                continue

            out_dir = os.path.join(scene_path, f"{model}_aligned")
            os.makedirs(out_dir, exist_ok=True)

            # Determine pred_scale based on model
            if model == "depth-pro":
                pred_scale = args.pred_scale_depth_pro
                pred_exts = ["npz"]
            elif model == "zoe-depth":
                pred_scale = args.pred_scale_zoe_depth
                pred_exts = ["png"]
            else:
                pred_scale = 1.0
                pred_exts = ["npz", "npy", "png"]

            # Find prediction files
            pred_files = []
            for ext in pred_exts:
                pred_files.extend(glob.glob(os.path.join(pred_dir, f"*.{ext}")))
            pred_files = sorted(pred_files)

            if not pred_files:
                print(f"  Warning:  No {model} prediction files found")
                continue

            scene_success = 0
            scene_failed = 0

            for pred_path in pred_files:
                stem = os.path.splitext(os.path.basename(pred_path))[0]
                
                # Find matching GT file
                if stem not in gt_by_stem:
                    scene_failed += 1
                    continue
                
                gt_path = gt_by_stem[stem]

                try:
                    pred_depth = load_depth(pred_path, scale=pred_scale)
                    gt_depth = load_depth(gt_path, scale=args.gt_scale)
                except Exception as e:
                    if idx <= 3:  # Only print details for first few scenes
                        print(f"   Warning:  Failed to load {stem}: {e}")
                    scene_failed += 1
                    continue

                mask = gt_depth > 0
                if not np.any(mask):
                    scene_failed += 1
                    continue

                # Resize if needed
                if pred_depth.shape != gt_depth.shape:
                    pred_depth = cv2.resize(
                        pred_depth,
                        (gt_depth.shape[1], gt_depth.shape[0]),
                        interpolation=cv2.INTER_NEAREST
                    )

                # Align
                median_gt = np.median(gt_depth[mask])
                median_pred = np.median(pred_depth[mask])
                scale_factor = median_gt / (median_pred + EPS)
                aligned_pred = pred_depth * scale_factor

                # Save as .npy with same stem
                out_path = os.path.join(out_dir, f"{stem}.npy")
                np.save(out_path, aligned_pred.astype(np.float32))
                scene_success += 1

            print(f"  ✓ {model}: aligned {scene_success}/{len(pred_files)} frames")
            total_aligned += scene_success
            total_failed += scene_failed

    print("\n" + "=" * 60)
    print(f"Batch alignment complete!")
    print(f"   Total aligned: {total_aligned}")
    print(f"   Total failed:  {total_failed}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()