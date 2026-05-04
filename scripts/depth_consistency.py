import os
import argparse
import numpy as np
import cv2
from glob import glob
#from skimage.metrics import structural_similarity as ssim

EPS = 1e-6


# Functions

def load_rgb(path):
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Failed to load image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype(np.float32) / 255.0


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


def load_pred_depth(path):
    if path.endswith(".npz"):
        data = np.load(path)
        for key in ["depth", "pred", "prediction", "arr_0"]:
            if key in data:
                depth = data[key]
                break
        else:
            depth = data[list(data.keys())[0]]
        depth = np.asarray(depth).astype(np.float32)
        #handle common layouts: (1,H,W) -> (H,W), (H,W,1) -> (H,W), (H,W,3) -> (H,W)
        if depth.ndim == 3 and depth.shape[0] == 1:
            depth = depth[0]
        if depth.ndim == 3 and depth.shape[-1] in (1, 3, 4):
            depth = depth[..., 0]

    elif path.endswith(".npy"):
        depth = np.load(path).astype(np.float32)
        # handle common layouts: (1,H,W) -> (H,W) and (H,W,1) or (H,W,3) -> (H,W)
        if depth.ndim == 3 and depth.shape[0] == 1:
            depth = depth[0]
        if depth.ndim == 3 and depth.shape[-1] in (1, 3, 4):
            depth = depth[..., 0]

    else:
        # fallback for PNG/JPG/other image formats
        depth = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise ValueError(f"Failed to load: {path}")
        depth = depth.astype(np.float32)
        if depth.ndim == 3:
            depth = cv2.cvtColor(depth, cv2.COLOR_BGR2GRAY)

    return depth


def resize_to_match(src, target_shape):
    return cv2.resize(src, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST)


# Metrics


def depth_metrics(pred, gt, mask):
    pred = pred[mask]
    gt = gt[mask]

    abs_rel = np.mean(np.abs(pred - gt) / (gt + EPS))
    rmse = np.sqrt(np.mean((pred - gt) ** 2))
    mae = np.mean(np.abs(pred - gt))

    delta = np.maximum(pred / (gt + EPS), gt / (pred + EPS))
    delta1 = np.mean(delta < 1.25)
    delta2 = np.mean(delta < 1.25 ** 2)
    delta3 = np.mean(delta < 1.25 ** 3)

    return {
        "abs_rel": abs_rel,
        "rmse": rmse,
        "mae": mae,
        "delta1": delta1,
        "delta2": delta2,
        "delta3": delta3,
    }


def align_scale(pred, gt, mask):
    scale = np.median(gt[mask]) / (np.median(pred[mask]) + EPS)
    return pred * scale


# Main

def main(args):
    #allow rgb files with different extensions (e.g. .png) as long as they match gt/pred names
    
    rgb_files = sorted(glob(os.path.join(args.rgb_dir, "*.jpg")))
    rgb_files += sorted(glob(os.path.join(args.rgb_dir, "*.png")))

    if len(rgb_files) == 0:
        raise ValueError("No RGB images found.")

    all_results = []

    # formatting for output
    header_fmt = "{:20s} {:>10s} {:>10s} {:>10s} {:>8s} {:>8s} {:>8s}"
    row_fmt =    "{:20s} {:10.4f} {:10.4f} {:10.4f} {:8.4f} {:8.4f} {:8.4f}"
    printed_header = False

    for rgb_path in rgb_files:
        name = os.path.splitext(os.path.basename(rgb_path))[0]


        #allow gt files with different extensions (e.g. .npy) as long as they match pred names

        gt_path = None
        for ext in [".png", ".jpg", ".npy", ".npz"]:
            candidate = os.path.join(args.gt_dir, name + ext)
            if os.path.exists(candidate):
                gt_path = candidate
                break

        # support multiple pred formats
        pred_path = None
        for ext in [".png", ".jpg", ".npy", ".npz"]:
            candidate = os.path.join(args.pred_dir, name + ext)
            if os.path.exists(candidate):
                pred_path = candidate
                break

        if pred_path is None or not os.path.exists(gt_path):
            print(f"Skipping {name} (missing files)")
            continue

        #rgb = load_rgb(rgb_path)
        gt_depth = load_depth_gt(gt_path, args.depth_scale)
        pred_depth = load_pred_depth(pred_path)

        if pred_depth.shape != gt_depth.shape:
            pred_depth = resize_to_match(pred_depth, gt_depth.shape)

        mask = gt_depth > 0

        if np.sum(mask) == 0:
            print(f"Skipping {name} (no valid depth)")
            continue

        #pred_depth = align_scale(pred_depth, gt_depth, mask) # optional scale, only needed for relative MDEs such as MoGe

        d_metrics = depth_metrics(pred_depth, gt_depth, mask)


        result = {
            "name": name,
            **d_metrics,
        }

        all_results.append(result)

        # print header once, then each row
        if not printed_header:
            print(header_fmt.format("name", "abs_rel", "rmse", "mae", "delta1", "delta2", "delta3"))
            print("-" * 78)
            printed_header = True

        print(row_fmt.format(
            result["name"],
            result["abs_rel"],
            result["rmse"],
            result["mae"],
            result["delta1"],
            result["delta2"],
            result["delta3"],
        ))

    if len(all_results) == 0:
        print("No valid samples processed.")
        return

    # average
    avg = {}
    for k in all_results[0].keys():
        if k == "name":
            continue
        avg[k] = np.mean([r[k] for r in all_results])

    print("\n=== Averages ===")
    print(row_fmt.format(
        "Average",
        avg["abs_rel"],
        avg["rmse"],
        avg["mae"],
        avg["delta1"],
        avg["delta2"],
        avg["delta3"],
    ))


# Args parsing
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Depth evaluation script")

    parser.add_argument("--rgb_dir", type=str, required=True, help="Path to RGB JPG images")
    parser.add_argument("--gt_dir", type=str, required=True, help="Path to GT depth PNGs or .npy/.npz files")
    parser.add_argument("--pred_dir", type=str, required=True, help="Path to predicted depths")
    parser.add_argument("--depth_scale", type=float, default=1000.0,
                        help="Scale for GT depth (default: 1000 for ScanNet mm→m)")

    args = parser.parse_args()
    main(args)