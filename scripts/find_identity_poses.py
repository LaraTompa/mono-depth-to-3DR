"""
find_identity_poses.py — Scan pose*.txt files under one or more sampled_data
roots and report scenes containing more than a threshold number of (near)
identity pose matrices, which indicates a missing/failed pose load that
silently defaulted to identity instead of raising an error. A single stray
identity pose can legitimately occur (e.g. the very first frame of a
ScanNet trajectory), so only scenes with several identity poses are flagged.

Usage:
    python scripts/find_identity_poses.py datasets/sampled_data datasets_test/sampled_data

Exit code is non-zero if any scenes were flagged (useful for CI).
"""
import argparse
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def is_identity(pose: np.ndarray, atol: float = 1e-6) -> bool:
    return np.allclose(pose, np.eye(4), atol=atol)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="+", help="One or more sampled_data root directories to scan")
    ap.add_argument("--atol", type=float, default=1e-6, help="Absolute tolerance for identity check")
    ap.add_argument("--min_identity_count", type=int, default=3,
                     help="Minimum number of identity poses in a scene required to flag it "
                          "(scenes with this many or fewer are not flagged)")
    ap.add_argument("--summary_out", default="identity_poses_summary.txt",
                     help="Path to write the summary report as a text file")
    ap.add_argument("--delete", action="store_true",
                     help="Delete the sample directories containing flagged scenes "
                          "(the whole sample dir is removed, not just the scene, "
                          "to avoid leaving empty samples behind)")
    args = ap.parse_args()

    total = 0
    identity_files = []
    scene_bad_poses = defaultdict(list)
    for root in args.roots:
        root_path = Path(root)
        if not root_path.exists():
            print(f"[WARN] root does not exist, skipping: {root_path}", file=sys.stderr)
            continue
        for pose_file in sorted(root_path.rglob("pose/*.txt")):
            total += 1
            try:
                pose = np.loadtxt(pose_file)
                if pose.shape == (3, 4):
                    pose = np.vstack([pose, [0, 0, 0, 1]])
                if pose.shape != (4, 4):
                    print(f"[WARN] unexpected shape {pose.shape} in {pose_file}", file=sys.stderr)
                    continue
            except Exception as e:
                print(f"[WARN] failed to load {pose_file}: {e}", file=sys.stderr)
                continue
            if is_identity(pose, atol=args.atol):
                identity_files.append(str(pose_file))
                scene_bad_poses[str(pose_file.parent.parent)].append(str(pose_file))

    flagged_scenes = {
        scene: files for scene, files in scene_bad_poses.items()
        if len(files) > args.min_identity_count
    }

    # Sample dir = parent of the scene dir (root/batchN/sampleM/sceneXXXX_YY).
    # Deleting the whole sample avoids leaving an empty/orphaned sample behind.
    flagged_samples = sorted({str(Path(scene).parent) for scene in flagged_scenes})

    lines = []
    lines.append(f"Scanned {total} pose files across {len(args.roots)} root(s).")
    lines.append(f"Found {len(identity_files)} identity pose file(s) across {len(scene_bad_poses)} scene(s).")
    lines.append(f"Flagged {len(flagged_scenes)} scene(s) with more than {args.min_identity_count} identity poses.")

    if flagged_scenes:
        lines.append("\nFlagged scenes (identity pose count > threshold):")
        for scene in sorted(flagged_scenes):
            files = flagged_scenes[scene]
            lines.append(f"\n  {scene}  ({len(files)} identity poses)")
            for f in files:
                lines.append(f"    {f}")

        lines.append(f"\nSample dirs containing flagged scenes ({len(flagged_samples)}):")
        for sample_dir in flagged_samples:
            lines.append(f"  {sample_dir}")

    if args.delete and flagged_samples:
        lines.append("\nDeleted sample dirs:")
        for sample_dir in flagged_samples:
            sample_path = Path(sample_dir)
            if sample_path.exists():
                shutil.rmtree(sample_path)
                lines.append(f"  {sample_dir}")
                print(f"[DELETED] {sample_dir}")
            else:
                print(f"[WARN] sample dir already missing, skipping: {sample_dir}", file=sys.stderr)

    report = "\n".join(lines)
    print("\n" + report)

    summary_path = Path(args.summary_out)
    summary_path.write_text(report + "\n", encoding="utf-8")
    print(f"\nSummary written to {summary_path.resolve()}")

    sys.exit(1 if flagged_scenes else 0)


if __name__ == "__main__":
    main()
