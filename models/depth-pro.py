"""
Run Depth Pro on ScanNet sampled data.

Expects the following structure under SAMPLED_DATA_DIR:
  batch<N>/
    sample<M>/
      <scene_id>/          ← e.g. scene0000_00
        color/             ← *.jpg / *.png frames
        depth/
        pose/
        intrinsic/

Predictions are written to:
  <scene_dir>/depth-pro_pred/<frame_stem>.npz

Usage (from inside the depth-pro repo with the venv active):
  python run_depth_pro.py
  python run_depth_pro.py --data-dir ~/mono-depth-to-3DR/datasets/sampled_data
  python run_depth_pro.py --workers 4 --dry-run
"""

import argparse
import os
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
import numpy as np
import depth_pro

# Default settings

SAMPLED_DATA_DIR = os.path.expanduser("~/mono-depth-to-3DR/datasets/sampled_data")
COLOR_EXTENSIONS = {".jpg", ".jpeg", ".png"}
OUTPUT_SUBDIR = "depth-pro_pred"


# Helper functions

def find_scene_dirs(root: Path) -> list[Path]:
    """
    Walk root and return every directory that contains a 'color/' sub-folder.
    That heuristic identifies scene-level directories regardless of how many
    batch / sample levels sit above them.
    """
    scenes = []
    for dirpath, dirnames, _ in os.walk(root):
        if "color" in dirnames:
            scenes.append(Path(dirpath))
            dirnames.clear()          # don't descend further inside a scene
    return sorted(scenes)


def color_frames(scene_dir: Path) -> list[Path]:
    """Return sorted list of color image paths inside <scene>/color/."""
    color_dir = scene_dir / "color"
    frames = [
        p for p in sorted(color_dir.iterdir())
        if p.suffix.lower() in COLOR_EXTENSIONS
    ]
    return frames


def run_depth_pro(frame: Path, out_dir: Path, model, transform, dry_run: bool = False) -> tuple[Path, bool, str]:
    """
    Run Depth Pro inference for a single frame using the Python API.
    Saves a .npz file with depth and focallength_px to out_dir.
    Returns (frame, success, message).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_out = out_dir / f"{frame.stem}.npz"

    # Skip already processed frames
    if npz_out.exists():
        return frame, True, "[SKIPPED] already exists"

    if dry_run:
        return frame, True, f"[DRY-RUN] would process: {frame}"

    try:
        # Load and preprocess image
        image, _, f_px = depth_pro.load_rgb(frame)
        image = transform(image)
        if torch.cuda.is_available():
            image = image.cuda()

        # Run inference
        with torch.no_grad():
            prediction = model.infer(image, f_px=f_px)

        depth = prediction["depth"].cpu().numpy()            # Depth in [m]
        focallength_px = prediction["focallength_px"].cpu().numpy()

        # Save as .npz (same as CLI output)
        np.savez(npz_out, depth=depth, focallength_px=focallength_px)

        return frame, True, f"Saved: {npz_out}"

    except Exception as exc:
        return frame, False, str(exc)


def process_scene(scene_dir: Path, model, transform, workers: int, dry_run: bool) -> dict:
    """Process all frames in one scene and return a summary dict."""
    frames = color_frames(scene_dir)
    out_dir = scene_dir / OUTPUT_SUBDIR

    if not frames:
        return {"scene": scene_dir.name, "frames": 0, "ok": 0, "fail": 0, "errors": []}

    ok = fail = 0
    errors = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(run_depth_pro, f, out_dir, model, transform, dry_run): f for f in frames}
        for future in as_completed(futures):
            frame, success, msg = future.result()
            if success:
                ok += 1
            else:
                fail += 1
                errors.append(f"{frame.name}: {msg}")

    return {"scene": scene_dir.name, "frames": len(frames), "ok": ok, "fail": fail, "errors": errors}


# CLI

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Depth Pro over all ScanNet sampled scenes."
    )
    parser.add_argument(
        "--data-dir",
        default=SAMPLED_DATA_DIR,
        help=f"Root of sampled_data (default: {SAMPLED_DATA_DIR})",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel threads per scene (default: 1; set >1 with caution on GPU)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.data_dir).expanduser().resolve()

    if not root.exists():
        sys.exit(f"ERROR: data directory not found: {root}")

    print(f"Scanning for scenes under: {root}")
    scenes = find_scene_dirs(root)

    if not scenes:
        sys.exit("ERROR: No scene directories found (looked for folders containing 'color/').")

    print(f"Found {len(scenes)} scene(s).\n")

    # Load model ONCE here, before processing any scenes
    print("Loading Depth Pro model...")
    model, transform = depth_pro.create_model_and_transforms()
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
        print("Running on GPU.\n")
    else:
        print("Running on CPU.\n")

    totals = {"frames": 0, "ok": 0, "fail": 0}

    for i, scene_dir in enumerate(scenes, 1):
        summary = process_scene(scene_dir, model, transform, args.workers, args.dry_run)
        totals["frames"] += summary["frames"]
        totals["ok"] += summary["ok"]
        totals["fail"] += summary["fail"]

        status = "✓" if summary["fail"] == 0 else "✗"
        print(f"  [{i:>3}/{len(scenes)}] {status} {scene_dir.relative_to(root)}"
              + (f"  ({summary['fail']} frame(s) failed)" if summary["fail"] else ""))
        for err in summary["errors"]:
            print(f"           {err}", file=sys.stderr)

    print("\n" + "=" * 50)
    print(
        f"Done.  Total: {totals['ok']}/{totals['frames']} frames succeeded"
        + (f", {totals['fail']} failed." if totals["fail"] else ".")
    )
    if args.dry_run:
        print("(dry-run — no commands were actually executed)")


if __name__ == "__main__":
    main()