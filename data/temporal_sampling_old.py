import os
import shutil
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import random

from graph_based_sampling import ScanNetGraphDataset



def load_pose(path):
    return np.loadtxt(path)


def collate_temporal_batch(batch):
    """Avoid PyTorch's default shared-memory tensor collation path."""
    return {
        "images": torch.stack([sample["images"] for sample in batch], dim=0),
        "depths": torch.stack([sample["depths"] for sample in batch], dim=0),
        "poses": torch.stack([sample["poses"] for sample in batch], dim=0),
        "scene": [sample["scene"] for sample in batch],
        "frame_ids": [sample["frame_ids"] for sample in batch],
    }


class ScanNetTemporalDataset(Dataset):
    def __init__(
        self,
        root_dir,
        num_frames=6,
        num_samples=700,
        min_stride=2,
        max_stride=10,
        transform=None,
        max_scenes=None
    ):
        self.root_dir = root_dir
        self.num_frames = num_frames
        self.min_stride = min_stride
        self.max_stride = max_stride
        self.transform = transform

        # fixed seq_len per epoch; call resample_seq_len() to change between epochs
        self.seq_len = num_frames

        # Load every available scene
        raw_scenes = []
        for scene in sorted(os.listdir(root_dir)):
            scene_path = os.path.join(root_dir, scene)

            color_dir = os.path.join(scene_path, "color")
            depth_dir = os.path.join(scene_path, "depth")
            pose_dir = os.path.join(scene_path, "pose")

            if not (os.path.isdir(color_dir) and os.path.isdir(depth_dir) and os.path.isdir(pose_dir)):
                continue

            frame_ids = sorted([
                f.split(".")[0] for f in os.listdir(color_dir)
            ], key=lambda x: int(x))

            raw_scenes.append({
                "scene":     scene,
                "color_dir": color_dir,
                "depth_dir": depth_dir,
                "pose_dir":  pose_dir,   # lazy: load per-frame on demand
                "frame_ids": frame_ids,
            })

        # Build scene_data: cycle raw_scenes if max_scenes exceeds available count,
        # otherwise truncate to max_scenes (or use all if max_scenes is None).
        n_raw = len(raw_scenes)
        if max_scenes is None:
            self.scene_data = raw_scenes
        elif max_scenes <= n_raw:
            self.scene_data = raw_scenes[:max_scenes]
        else:
            self.scene_data = [raw_scenes[i % n_raw] for i in range(max_scenes)]

    def resample_seq_len(self):
        """No-op when num_frames is fixed; kept for API compatibility."""
        pass

    def __len__(self):
        return len(self.scene_data)

    def _sample_sequence(self, scene_info):
        frame_ids = scene_info["frame_ids"]
        n_frames = len(frame_ids)

        seq_len = self.seq_len
        stride = random.randint(self.min_stride, self.max_stride)

        # choose center index safely
        half = seq_len // 2
        max_offset = half * stride

        if n_frames < seq_len:
            # not enough frames: return evenly spaced seq_len indices
            indices = np.linspace(0, n_frames - 1, seq_len, dtype=int).tolist()
            return [frame_ids[i] for i in indices]

        # reduce stride until the sequence fits within available frames
        while max_offset >= n_frames - max_offset and stride > 1:
            stride -= 1
            max_offset = half * stride

        if max_offset >= n_frames - max_offset:
            # sequence still doesn't fit even at stride=1; return evenly spaced fallback
            indices = np.linspace(0, n_frames - 1, seq_len, dtype=int).tolist()
            return [frame_ids[i] for i in indices]

        # anchor the sequence at the middle of the scene
        center_idx = n_frames // 2
        center_idx = max(max_offset, min(n_frames - max_offset - 1, center_idx))

        indices = [
            center_idx + (i - half) * stride
            for i in range(seq_len)
        ]

        # clamp (safety)
        indices = [max(0, min(n_frames - 1, idx)) for idx in indices]

        seq_ids = [frame_ids[i] for i in indices]

        return seq_ids

    def _load_image(self, path):
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        else:
            # torch.tensor() copies data into PyTorch-owned storage (resizable).
            # torch.from_numpy() is NOT safe here: all numpy-backed storages are
            # marked non-resizable by PyTorch, causing DataLoader collation to crash
            # with "Trying to resize storage that is not resizable" (num_workers > 0).
            img = torch.tensor(np.array(img), dtype=torch.float32).permute(2, 0, 1) / 255.0
        return img

    def _load_depth(self, path):
        # torch.tensor() copies into PyTorch-owned resizable storage (see _load_image).
        depth = np.array(Image.open(path)).astype(np.float32) / 1000.0
        return torch.tensor(depth).unsqueeze(0)

    def __getitem__(self, idx):
        scene_info = self.scene_data[idx]

        seq_ids = self._sample_sequence(scene_info)

        images = []
        depths = []
        poses = []

        for fid in seq_ids:
            img_path = os.path.join(scene_info["color_dir"], f"{fid}.jpg")
            depth_path = os.path.join(scene_info["depth_dir"], f"{fid}.png")

            pose_np = np.loadtxt(os.path.join(scene_info["pose_dir"], f"{fid}.txt")).astype(np.float32)
            if not np.all(np.isfinite(pose_np)):
                pose_np = np.eye(4, dtype=np.float32)  # ScanNet marks failed tracking with inf

            images.append(self._load_image(img_path))
            depths.append(self._load_depth(depth_path))
            poses.append(torch.tensor(pose_np))  # torch.tensor() → resizable storage

        return {
            "images": torch.stack(images),   # (N, 3, H, W)
            "depths": torch.stack(depths),   # (N, 1, H, W)
            "poses": torch.stack(poses),     # (N, 4, 4)
            "scene": scene_info["scene"],
            "frame_ids": seq_ids             # original frame names e.g. ["0", "15", "30"]
        }


if __name__ == "__main__":
    import yaml
    import torchvision
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config", "dataset.yaml")
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    ds_cfg  = cfg["dataset"]
    out_cfg = cfg["output"]
    graph_cfg = cfg.get("graph_sampling", {})

    ROOT_DIR   = ds_cfg["root_dir"]
    OUT_DIR    = out_cfg["sample_output_dir"]
    SAMPLE_DIR = out_cfg["sampled_data_dir"]
    sampler_type = ds_cfg.get("sampler_type", "temporal")
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(SAMPLE_DIR, exist_ok=True)

    if sampler_type == "temporal":
        dataset = ScanNetTemporalDataset(
            root_dir=ROOT_DIR,
            num_frames=ds_cfg["num_frames"],
            num_samples=ds_cfg["num_samples"],
            min_stride=ds_cfg["min_stride"],
            max_stride=ds_cfg["max_stride"],
            max_scenes=ds_cfg["max_scenes"],
        )
    elif sampler_type == "graph":
        dataset = ScanNetGraphDataset(
            root_dir=ROOT_DIR,
            num_frames=ds_cfg["num_frames"],
            num_samples=ds_cfg["num_samples"],
            graph_cache=graph_cfg.get("graph_cache"),
            min_overlap=graph_cfg["min_overlap"],
            max_overlap=graph_cfg["max_overlap"],
            overlap_sample_step=graph_cfg["overlap_sample_step"],
            depth_tolerance=graph_cfg["depth_tolerance"],
            max_scenes=ds_cfg["max_scenes"],
        )
    else:
        raise ValueError(
            f"Unsupported sampler_type '{sampler_type}'. Use 'temporal' or 'graph'."
        )

    print(f"Loaded {len(dataset.scene_data)} scenes")
    print(f"Dataset length (virtual): {len(dataset)}")
    print(f"Sampler type            : {sampler_type}")
    print(f"Fixed seq_len this run  : {dataset.seq_len}")
    print(f"Saving visualisations to: {OUT_DIR}/")
    print(f"Saving sampled frames to: {SAMPLE_DIR}/\n")

    # -- Resume: detect the highest already-completed batch ------------------
    start_batch = 0
    if os.path.isdir(SAMPLE_DIR):
        done = [
            int(d[5:]) for d in os.listdir(SAMPLE_DIR)
            if d.startswith("batch") and d[5:].isdigit()
        ]
        if done:
            start_batch = max(done)
            print(f"[resume] Found batch dirs up to batch {start_batch} — "
                  f"resuming from batch {start_batch + 1}.\n")

    loader = DataLoader(
        dataset,
        batch_size=out_cfg["batch_size"],
        num_workers=out_cfg["num_workers"],
        shuffle=False,
        collate_fn=collate_temporal_batch,
    )

    total_batches = len(loader)
    print(f"Sampling {total_batches} batches...\n")

    for i, batch in enumerate(tqdm(loader, desc="Processing batches", unit="batch")):

        if i < start_batch:
            continue  # skip already-saved batches (fast: no disk I/O)

        images   = batch["images"]    # (B, N, 3, H, W)
        depths   = batch["depths"]    # (B, N, 1, H, W)
        poses    = batch["poses"]     # (B, N, 4, 4)
        scenes   = batch["scene"]
        all_fids = batch["frame_ids"] # list of N-length lists, one per sample

        print(f"\nBatch {i + 1}/{total_batches}:")
        print(f"  scenes : {list(scenes)}")

        # save each sample in the batch
        for b, scene_name in enumerate(tqdm(scenes, desc=f"  Saving samples", leave=False, unit="sample")):
            N = images.shape[1]
            fids = all_fids[b]

            # --- sampled_data: individual frames per sequence ---
            seq_dir = os.path.join(SAMPLE_DIR, f"batch{i+1}", f"sample{b+1}", scene_name)
            rgb_raw_dir = os.path.join(seq_dir, "color")
            dep_raw_dir = os.path.join(seq_dir, "depth")
            pose_dir    = os.path.join(seq_dir, "pose")
            intr_dir    = os.path.join(seq_dir, "intrinsic")
            os.makedirs(rgb_raw_dir, exist_ok=True)
            os.makedirs(dep_raw_dir, exist_ok=True)
            os.makedirs(pose_dir, exist_ok=True)
            os.makedirs(intr_dir, exist_ok=True)

            # Save camera intrinsics — files sit in scene/intrinsic/ subfolder.
            for intr_filename, intr_dst in [
                ("intrinsic_color.txt", os.path.join(intr_dir, "intrinsic_color.txt")),
                ("intrinsic_depth.txt", os.path.join(intr_dir, "intrinsic_depth.txt")),
            ]:
                src = os.path.join(ROOT_DIR, scene_name, "intrinsic", intr_filename)
                if os.path.isfile(src):
                    shutil.copy2(src, intr_dst)
                else:
                    print(f"  warning: could not find {intr_filename} for scene {scene_name}")

            for f in tqdm(range(N), desc=f"    Saving frames", leave=False, unit="frame"):
                fid = fids[f]
                torchvision.utils.save_image(images[b, f], os.path.join(rgb_raw_dir, f"{fid}.png"))
                np.save(os.path.join(dep_raw_dir, f"{fid}.npy"), depths[b, f].numpy())
                np.savetxt(os.path.join(pose_dir, f"{fid}.txt"), poses[b, f].numpy())

            # --- sample_output: visualisation grids ---
            rgb_frames = images[b]          # (N, 3, H, W)
            rgb_grid = torchvision.utils.make_grid(rgb_frames, nrow=N, padding=4)
            rgb_path = os.path.join(OUT_DIR, f"batch{i+1}_sample{b+1}_{scene_name}_rgb.png")
            torchvision.utils.save_image(rgb_grid, rgb_path)

            dep_frames = depths[b]          # (N, 1, H, W)
            dep_min, dep_max = dep_frames.min(), dep_frames.max()
            dep_norm = (dep_frames - dep_min) / (dep_max - dep_min + 1e-6)
            dep_rgb  = dep_norm.repeat(1, 3, 1, 1)
            dep_grid = torchvision.utils.make_grid(dep_rgb, nrow=N, padding=4)
            dep_path = os.path.join(OUT_DIR, f"batch{i+1}_sample{b+1}_{scene_name}_depth.png")
            torchvision.utils.save_image(dep_grid, dep_path)

            print(f"    saved frames : {seq_dir}/")
            print(f"    saved rgb grid  : {rgb_path}")
            print(f"    saved depth grid: {dep_path}")

    print("\nDone.")
