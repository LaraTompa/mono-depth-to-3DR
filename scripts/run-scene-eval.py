
import os
import glob
import argparse
import subprocess
import numpy as np
import re


def sorted_files(folder, ext):
    return sorted(glob.glob(os.path.join(folder, f"*.{ext}")))


def run_cmd(cmd):
    result = subprocess.run(cmd, capture_output=True, text=True)
    # return both stdout and stderr combined
    return (result.stdout or "") + (result.stderr or "")


def run_depth(args):
    cmd = [
        "python3", "scripts/depth_consistency.py",
        "--rgb_dir", args.rgb_dir,
        "--gt_dir", args.gt_depth_dir,
        "--pred_dir", args.pred_depth_dir,
    ]
    return run_cmd(cmd)

def parse_photometric_output(output):
    output = output.replace('\xa0', ' ')
    # debug: show output when parsing fails
    float_re = r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
    ssim_avg, l2_avg = None, None

    ssim_match = re.search(rf"SSIM\s*(?:avg|AVG)?\s*[:=]\s*{float_re}", output, flags=re.IGNORECASE)
    l2_match   = re.search(rf"L2\s*(?:avg|AVG)?\s*[:=]\s*{float_re}", output, flags=re.IGNORECASE)

    if ssim_match:
        ssim_avg = float(ssim_match.group(1))
    if l2_match:
        l2_avg = float(l2_match.group(1))

    if ssim_avg is None or l2_avg is None:
        # helpful debug print to stderr so you can see what's being parsed
        print("=== Photometric raw output (for debugging) ===")
        print(output)
        print("=== End raw output ===")

    return ssim_avg, l2_avg


def run_photometric(args, img1, img2, depth1, depth2, pose1, pose2):
    cmd = [
        "python3", "scripts/photometric_consistency.py",
        "--img1", img1,
        "--img2", img2,
        "--depth1", depth1,
        "--depth2", depth2,
        "--intrinsics", args.intrinsics,
        "--pose1", pose1,
        "--pose2", pose2,
        "--depth_scale", str(args.depth_scale),
    ]

    if args.cam_to_world:
        cmd.append("--cam_to_world")
    else:
        cmd.append("--world_to_cam")

    if args.visualize:
        cmd.append("--visualize")

    return run_cmd(cmd)


def main():

    parser = argparse.ArgumentParser()

    # inputs
    parser.add_argument("--rgb_dir", required=True)
    parser.add_argument("--pose_dir", required=True)
    parser.add_argument("--pred_depth_dir", required=True)
    parser.add_argument("--intrinsics", required=True)
    parser.add_argument("--gt_depth_dir", required=True)

    # formats
    parser.add_argument("--rgb_ext", default="png")
    parser.add_argument("--depth_ext", default="npz")
    parser.add_argument("--pose_ext", default="txt")
    parser.add_argument("--gt_depth_ext", default="npy")

    # settings
    parser.add_argument("--depth_scale", type=float, default=1.0)
    parser.add_argument("--cam_to_world", action="store_true")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--window", type=int, default=1)

    args = parser.parse_args()

    rgb_files = sorted_files(args.rgb_dir, args.rgb_ext)
    pose_files = sorted_files(args.pose_dir, args.pose_ext)
    pred_depth_files = sorted_files(args.pred_depth_dir, args.depth_ext)
    gt_depth_files = sorted_files(args.gt_depth_dir, args.gt_depth_ext)

    n = len(rgb_files)

    assert len(pose_files) == n
    assert len(pred_depth_files) == n
    assert len(gt_depth_files) == n

    print(f"Found {n} frames")

    # Depth consistency

    print("\n=== Depth Consistency ===")
    depth_output = run_depth(args)
    print(depth_output)

    # Photometric consistency (pairwise, with window)
    print("\n=== Photometric Consistency ===")

    photometric_results = []

    for i in range(n):
        for j in range(i + 1, min(i + 1 + args.window, n)):

            print(f"\nPair {i} → {j}")

            output = run_photometric(
                args,
                rgb_files[i],
                rgb_files[j],
                pred_depth_files[i],
                pred_depth_files[j],
                pose_files[i],
                pose_files[j],
            )

            ssim, l2 = parse_photometric_output(output)

            print(f"SSIM avg: {ssim}, L2 avg: {l2}")

            photometric_results.append((ssim, l2))

    # Averages

    valid = [(s, l) for s, l in photometric_results if s is not None and l is not None]

    if valid:
        mean_ssim = np.mean([s for s, _ in valid])
        mean_l2 = np.mean([l for _, l in valid])

        print("\n=== FINAL AVERAGES ===")
        print(f"Mean SSIM: {mean_ssim:.4f}")
        print(f"Mean L2:   {mean_l2:.6f}")

    print("\nDone.")


# Entry point

if __name__ == "__main__":

    main()