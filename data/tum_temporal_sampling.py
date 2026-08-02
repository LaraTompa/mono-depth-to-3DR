"""
tum_temporal_sampling.py

Samples temporal sequences from the TUM RGB-D benchmark and saves them to
datasets/sampled_data in the same format produced by temporal_sampling_old.py.

Batch numbering is continued automatically from the highest existing batch
found in the sampled_data output directory.

Expected source layout::

    <root_dir>/
      rgbd_dataset_freiburg1_xyz/
        rgb/               *.png
        depth/             *.png  (uint16, 1/5000 m per unit)
        rgb.txt
        depth.txt
        groundtruth.txt    (timestamp tx ty tz qx qy qz qw)
        intrinsics/
          intrinsics_color.txt
"""

import os
import shutil
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import random


# ---------------------------------------------------------------------------
# TUM parsing helpers
# ---------------------------------------------------------------------------

def parse_tum_file(filepath):
    """
    Parse a TUM-format file (rgb.txt, depth.txt, or groundtruth.txt).
    Returns list of [float_timestamp, *remaining_string_fields].
    """
    entries = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            entries.append([float(parts[0])] + parts[1:])
    return entries


def associate_rgb_depth(rgb_entries, depth_entries, max_diff=0.02):
    """
    Associate each RGB frame with its nearest depth frame by timestamp.
    Drops pairs whose timestamp delta exceeds max_diff seconds.
    Returns list of (rgb_ts, rgb_rel_path, depth_ts, depth_rel_path).
    """
    depth_ts = np.array([e[0] for e in depth_entries])
    pairs = []
    for entry in rgb_entries:
        rgb_ts, rgb_path = entry[0], entry[1]
        idx = int(np.argmin(np.abs(depth_ts - rgb_ts)))
        if abs(depth_ts[idx] - rgb_ts) <= max_diff:
            pairs.append((rgb_ts, rgb_path, float(depth_ts[idx]), depth_entries[idx][1]))
    return pairs


def quat_to_mat44(tx, ty, tz, qx, qy, qz, qw):
    """Convert a TUM translation + quaternion to a 4×4 camera-to-world matrix."""
    norm = (qx**2 + qy**2 + qz**2 + qw**2) ** 0.5
    if norm < 1e-10:
        return np.eye(4, dtype=np.float32)
    qx, qy, qz, qw = qx / norm, qy / norm, qz / norm, qw / norm

    R = np.array([
        [1 - 2*(qy**2 + qz**2),   2*(qx*qy - qz*qw),   2*(qx*qz + qy*qw)],
        [  2*(qx*qy + qz*qw), 1 - 2*(qx**2 + qz**2),   2*(qy*qz - qx*qw)],
        [  2*(qx*qz - qy*qw),   2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qy**2)],
    ], dtype=np.float32)

    mat = np.eye(4, dtype=np.float32)
    mat[:3, :3] = R
    mat[:3, 3] = [tx, ty, tz]
    return mat


def nearest_pose(gt_entries, query_ts):
    """
    Return the 4×4 camera-to-world pose whose timestamp is nearest to query_ts.
    Falls back to identity if gt_entries is empty.
    """
    if not gt_entries:
        return np.eye(4, dtype=np.float32)
    gt_ts = np.array([e[0] for e in gt_entries])
    idx = int(np.argmin(np.abs(gt_ts - query_ts)))
    e = gt_entries[idx]
    tx, ty, tz, qx, qy, qz, qw = (float(v) for v in e[1:8])
    return quat_to_mat44(tx, ty, tz, qx, qy, qz, qw)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def collate_tum_batch(batch):
    return {
        "images":    [s["images"]    for s in batch],
        "depths":    [s["depths"]    for s in batch],
        "poses":     [s["poses"]     for s in batch],
        "scene":     [s["scene"]     for s in batch],
        "frame_ids": [s["frame_ids"] for s in batch],
    }


class TUMTemporalDataset(Dataset):
    """Loads TUM RGB-D sequences and samples fixed-length temporal windows."""

    DEPTH_SCALE = 1.0 / 5000.0  # uint16 units → metres

    def __init__(
        self,
        root_dir,
        num_frames=6,
        min_stride=2,
        max_stride=10,
        transform=None,
        max_scenes=None,
    ):
        self.root_dir = root_dir
        self.num_frames = num_frames
        self.min_stride = min_stride
        self.max_stride = max_stride
        self.transform = transform
        self.seq_len = num_frames

        raw_scenes = []
        for seq_name in sorted(os.listdir(root_dir)):
            seq_path = os.path.join(root_dir, seq_name)
            if not os.path.isdir(seq_path):
                continue

            rgb_txt   = os.path.join(seq_path, "rgb.txt")
            depth_txt = os.path.join(seq_path, "depth.txt")
            gt_txt    = os.path.join(seq_path, "groundtruth.txt")
            rgb_dir   = os.path.join(seq_path, "rgb")
            depth_dir = os.path.join(seq_path, "depth")

            if not all([
                os.path.isfile(rgb_txt),
                os.path.isfile(depth_txt),
                os.path.isfile(gt_txt),
                os.path.isdir(rgb_dir),
                os.path.isdir(depth_dir),
            ]):
                continue

            rgb_entries   = parse_tum_file(rgb_txt)
            depth_entries = parse_tum_file(depth_txt)
            gt_entries    = parse_tum_file(gt_txt)

            pairs = associate_rgb_depth(rgb_entries, depth_entries)
            if len(pairs) < num_frames:
                print(f"  skip {seq_name}: only {len(pairs)} associated pairs (need {num_frames})")
                continue

            raw_scenes.append({
                "scene":      seq_name,
                "seq_path":   seq_path,
                "gt_entries": gt_entries,
                "pairs":      pairs,   # list of (rgb_ts, rgb_rel_path, depth_ts, depth_rel_path)
            })

        n_raw = len(raw_scenes)
        if max_scenes is None:
            self.scene_data = raw_scenes
        elif max_scenes <= n_raw:
            self.scene_data = raw_scenes[:max_scenes]
        else:
            # cycle to reach max_scenes
            self.scene_data = [raw_scenes[i % n_raw] for i in range(max_scenes)]

    def __len__(self):
        return len(self.scene_data)

    def _sample_sequence(self, scene_info):
        pairs   = scene_info["pairs"]
        n       = len(pairs)
        seq_len = self.seq_len
        stride  = random.randint(self.min_stride, self.max_stride)

        half       = seq_len // 2
        max_offset = half * stride

        if n < seq_len:
            indices = np.linspace(0, n - 1, seq_len, dtype=int).tolist()
            return [pairs[i] for i in indices]

        while max_offset >= n - max_offset and stride > 1:
            stride -= 1
            max_offset = half * stride

        if max_offset >= n - max_offset:
            indices = np.linspace(0, n - 1, seq_len, dtype=int).tolist()
            return [pairs[i] for i in indices]

        center = n // 2
        center = max(max_offset, min(n - max_offset - 1, center))

        indices = [center + (i - half) * stride for i in range(seq_len)]
        indices = [max(0, min(n - 1, idx)) for idx in indices]
        return [pairs[i] for i in indices]

    def _load_image(self, path):
        img = Image.open(path).convert("RGB")
        if self.transform:
            return self.transform(img)
        # torch.tensor() copies into resizable storage (safe for DataLoader workers)
        return torch.tensor(np.array(img), dtype=torch.float32).permute(2, 0, 1) / 255.0

    def _load_depth(self, path):
        arr = np.array(Image.open(path)).astype(np.float32) * self.DEPTH_SCALE
        return torch.tensor(arr).unsqueeze(0)   # (1, H, W), metres

    def __getitem__(self, idx):
        scene_info = self.scene_data[idx]
        seq_pairs  = self._sample_sequence(scene_info)

        images, depths, poses, frame_ids = [], [], [], []
        for rgb_ts, rgb_rel, _depth_ts, depth_rel in seq_pairs:
            img_path = os.path.join(scene_info["seq_path"], rgb_rel)
            dep_path = os.path.join(scene_info["seq_path"], depth_rel)
            pose_np  = nearest_pose(scene_info["gt_entries"], rgb_ts)

            images.append(self._load_image(img_path))
            depths.append(self._load_depth(dep_path))
            poses.append(torch.tensor(pose_np))
            frame_ids.append(f"{rgb_ts:.6f}")

        return {
            "images":    torch.stack(images),   # (N, 3, H, W)
            "depths":    torch.stack(depths),   # (N, 1, H, W)
            "poses":     torch.stack(poses),    # (N, 4, 4)
            "scene":     scene_info["scene"],
            "frame_ids": frame_ids,
        }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import yaml
    import torchvision
    from tqdm import tqdm

    CONFIG_PATH = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "config", "dataset.yaml",
    )
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    out_cfg = cfg["output"]
    tum_cfg = cfg["tum_dataset"]

    SAMPLE_DIR = out_cfg["sampled_data_dir"]
    OUT_DIR    = out_cfg["sample_output_dir"]
    os.makedirs(SAMPLE_DIR, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)

    # Determine the next batch number from whatever already exists in SAMPLE_DIR
    existing = [
        int(d[5:]) for d in os.listdir(SAMPLE_DIR)
        if d.startswith("batch") and d[5:].isdigit()
    ] if os.path.isdir(SAMPLE_DIR) else []
    last_batch  = max(existing) if existing else 0
    start_batch = last_batch + 1
    print(f"Last existing batch : batch{last_batch if existing else '(none)'}")
    print(f"TUM batches start at: batch{start_batch}\n")

    dataset = TUMTemporalDataset(
        root_dir   = tum_cfg["root_dir"],
        num_frames = tum_cfg["num_frames"],
        min_stride = tum_cfg["min_stride"],
        max_stride = tum_cfg["max_stride"],
        max_scenes = tum_cfg.get("max_scenes"),
    )

    print(f"Loaded {len(dataset.scene_data)} TUM sequences")
    print(f"Seq len  : {dataset.seq_len}")
    print(f"Saving to: {SAMPLE_DIR}/\n")

    loader = DataLoader(
        dataset,
        batch_size  = out_cfg["batch_size"],
        num_workers = out_cfg["num_workers"],
        shuffle     = False,
        collate_fn  = collate_tum_batch,
    )

    total_batches = len(loader)
    print(f"Sampling {total_batches} batches...\n")

    for i, batch in enumerate(tqdm(loader, desc="Processing batches", unit="batch")):
        batch_num = start_batch + i

        images   = batch["images"]
        depths   = batch["depths"]
        poses    = batch["poses"]
        scenes   = batch["scene"]
        all_fids = batch["frame_ids"]

        print(f"\nBatch {batch_num} ({i + 1}/{total_batches}):")
        print(f"  scenes : {list(scenes)}")

        for b, scene_name in enumerate(tqdm(scenes, desc="  Saving samples", leave=False, unit="sample")):
            sample_images = images[b]
            sample_depths = depths[b]
            sample_poses  = poses[b]
            N    = sample_images.shape[0]
            fids = all_fids[b]

            seq_dir     = os.path.join(SAMPLE_DIR, f"batch{batch_num}", f"sample{b + 1}", scene_name)
            rgb_raw_dir = os.path.join(seq_dir, "color")
            dep_raw_dir = os.path.join(seq_dir, "depth")
            pose_dir    = os.path.join(seq_dir, "pose")
            intr_dir    = os.path.join(seq_dir, "intrinsics")
            os.makedirs(rgb_raw_dir, exist_ok=True)
            os.makedirs(dep_raw_dir, exist_ok=True)
            os.makedirs(pose_dir, exist_ok=True)
            os.makedirs(intr_dir, exist_ok=True)

            # Copy intrinsics files from the source sequence directory
            seq_src = os.path.join(tum_cfg["root_dir"], scene_name)
            for intr_fname in ("intrinsics_color.txt"):
                src = os.path.join(seq_src, "intrinsics", intr_fname)
                if os.path.isfile(src):
                    shutil.copy2(src, os.path.join(intr_dir, intr_fname))
                else:
                    print(f"  warning: {intr_fname} not found for {scene_name}")

            for f in tqdm(range(N), desc="    Saving frames", leave=False, unit="frame"):
                fid = fids[f]
                torchvision.utils.save_image(sample_images[f], os.path.join(rgb_raw_dir, f"{fid}.png"))
                np.save(os.path.join(dep_raw_dir, f"{fid}.npy"), sample_depths[f].numpy())
                np.savetxt(os.path.join(pose_dir, f"{fid}.txt"), sample_poses[f].numpy())

            # Visualisation grids
            rgb_grid = torchvision.utils.make_grid(sample_images, nrow=N, padding=4)
            torchvision.utils.save_image(
                rgb_grid,
                os.path.join(OUT_DIR, f"batch{batch_num}_sample{b + 1}_{scene_name}_rgb.png"),
            )

            dep_min, dep_max = sample_depths.min(), sample_depths.max()
            dep_norm = (sample_depths - dep_min) / (dep_max - dep_min + 1e-6)
            dep_grid = torchvision.utils.make_grid(dep_norm.repeat(1, 3, 1, 1), nrow=N, padding=4)
            torchvision.utils.save_image(
                dep_grid,
                os.path.join(OUT_DIR, f"batch{batch_num}_sample{b + 1}_{scene_name}_depth.png"),
            )

    print("\nDone.")
