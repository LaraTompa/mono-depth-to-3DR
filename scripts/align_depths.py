import numpy as np
import os
import cv2

#Write a script to compute median scaling factor to align predicted depth to GT depth, using only valid pixels defined by the mask
def align_scale(pred, gt, mask):
    scale = np.median(gt[mask]) / (np.median(pred[mask]) + EPS)
    return pred * scale

def load_depth_gt(path, scale):
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
    parser = argparse.ArgumentParser(
        description="Align predicted depths to GT depths (median scaling) for all matching filenames in directories")
    parser.add_argument("--pred_dir", type=str, required=True, help="Directory of predicted depth files")
    parser.add_argument("--gt_dir",   type=str, required=True, help="Directory of GT depth files")
    parser.add_argument("--out_dir",  type=str, default=None, help="Directory to write aligned predictions (default: pred_dir/_aligned)")
    parser.add_argument("--pred_scale", type=float, default=1.0, help="Scale to apply when loading predicted depth files (image-based depths)")
    parser.add_argument("--gt_scale",   type=float, default=1.0, help="Scale to apply when loading GT depth files (image-based depths)")
    parser.add_argument("--exts", type=str, default="npy,npz,png,jpg,jpeg", help="Comma-separated extensions to consider (by priority)")
    args = parser.parse_args()

    pred_dir = os.path.expanduser(args.pred_dir)
    gt_dir = os.path.expanduser(args.gt_dir)
    out_dir = os.path.expanduser(args.out_dir) if args.out_dir else os.path.join(pred_dir, "_aligned")
    os.makedirs(out_dir, exist_ok=True)

    exts = [e.strip().lower() for e in args.exts.split(",") if e.strip()]
    pred_files = sorted([f for f in os.listdir(pred_dir) if os.path.splitext(f)[1].lower().lstrip('.') in exts])

    if not pred_files:
        print(f"No prediction files found in {pred_dir} matching extensions: {exts}")
        return

    # Build GT lookup by stem -> filename
    gt_files = [f for f in os.listdir(gt_dir) if os.path.splitext(f)[1].lower().lstrip('.') in exts]
    gt_by_stem = {}
    for g in gt_files:
        stem = os.path.splitext(g)[0]
        gt_by_stem.setdefault(stem, []).append(g)

    for pred_fname in pred_files:
        stem = os.path.splitext(pred_fname)[0]
        pred_path = os.path.join(pred_dir, pred_fname)

        # Find GT file with same stem (prefer exact same ext order)
        candidates = gt_by_stem.get(stem, [])
        if not candidates:
            print(f"GT file not found for {pred_fname}, skipping...")
            continue
        # pick first candidate
        gt_fname = candidates[0]
        gt_path = os.path.join(gt_dir, gt_fname)

        try:
            pred_depth = load_depth_gt(pred_path, scale=args.pred_scale)
            gt_depth   = load_depth_gt(gt_path,   scale=args.gt_scale)
        except Exception as e:
            print(f"Failed to load {pred_fname} or GT {gt_fname}: {e}")
            continue

        mask = gt_depth > 0
        if not np.any(mask):
            print(f"No valid GT pixels for {gt_fname}, skipping...")
            continue

        # Resize predicted depth to match GT if needed
        if pred_depth.shape != gt_depth.shape:
            print(f"Resizing predicted depth {pred_fname} {pred_depth.shape} -> GT {gt_fname} {gt_depth.shape}")
            pred_depth = cv2.resize(pred_depth, (gt_depth.shape[1], gt_depth.shape[0]), interpolation=cv2.INTER_NEAREST)

        # compute scale and align
        median_gt = np.median(gt_depth[mask])
        median_pred = np.median(pred_depth[mask])
        scale_factor = median_gt / (median_pred + EPS)
        aligned_pred = pred_depth * scale_factor

        out_name = f"{stem}.npy"
        out_path = os.path.join(out_dir, out_name)
        np.save(out_path, aligned_pred.astype(np.float32))

        print(f"Aligned {pred_fname} -> {out_name} (scale={scale_factor:.6f})")

if __name__ == "__main__":
    main()