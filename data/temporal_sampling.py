import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import random


def load_pose(path):
    return np.loadtxt(path)


class ScanNetTemporalDataset(Dataset):
    def __init__(
        self,
        root_dir,
        seq_len_range=(4, 10),
        max_stride=10,
        transform=None,
        max_scenes=None
    ):
        self.root_dir = root_dir
        self.seq_len_range = seq_len_range
        self.max_stride = max_stride
        self.transform = transform

        self.scenes = sorted(os.listdir(root_dir))
        if max_scenes:
            self.scenes = self.scenes[:max_scenes]

        # fixed seq_len per epoch; call resample_seq_len() to change between epochs
        self.seq_len = random.randint(*seq_len_range)

        self.scene_data = []

        for scene in self.scenes:
            scene_path = os.path.join(root_dir, scene)

            color_dir = os.path.join(scene_path, "color")
            depth_dir = os.path.join(scene_path, "depth")
            pose_dir = os.path.join(scene_path, "pose")

            frame_ids = sorted([
                f.split(".")[0] for f in os.listdir(color_dir)
            ])

            poses = {
                fid: load_pose(os.path.join(pose_dir, f"{fid}.txt"))
                for fid in frame_ids
            }

            self.scene_data.append({
                "scene": scene,
                "color_dir": color_dir,
                "depth_dir": depth_dir,
                "poses": poses,
                "frame_ids": frame_ids
            })

    def resample_seq_len(self):
        """Call at the start of each epoch to draw a new fixed sequence length."""
        self.seq_len = random.randint(*self.seq_len_range)

    def __len__(self):
        return 1000  # dynamic sampling

    def _sample_sequence(self, scene_info):
        frame_ids = scene_info["frame_ids"]
        n_frames = len(frame_ids)

        seq_len = self.seq_len
        stride = random.randint(2, self.max_stride)

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

        center_idx = random.randint(max_offset, n_frames - max_offset - 1)

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
            img = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
        return img

    def _load_depth(self, path):
        depth = np.array(Image.open(path)).astype(np.float32) / 1000.0
        return torch.from_numpy(depth).unsqueeze(0)

    def __getitem__(self, idx):
        scene_info = random.choice(self.scene_data)

        seq_ids = self._sample_sequence(scene_info)

        images = []
        depths = []
        poses = []

        for fid in seq_ids:
            img_path = os.path.join(scene_info["color_dir"], f"{fid}.jpg")
            depth_path = os.path.join(scene_info["depth_dir"], f"{fid}.png")

            images.append(self._load_image(img_path))
            depths.append(self._load_depth(depth_path))
            poses.append(torch.from_numpy(scene_info["poses"][fid]).float())

        return {
            "images": torch.stack(images),   # (N, 3, H, W)
            "depths": torch.stack(depths),   # (N, 1, H, W)
            "poses": torch.stack(poses),     # (N, 4, 4)
            "scene": scene_info["scene"]
        }


if __name__ == "__main__":
    import torchvision
    from torch.utils.data import DataLoader

    ROOT_DIR    = "/storage/group/dataset_mirrors/scannet/tasks/scannet_frames_test"
    OUT_DIR     = "data/sample_output"   # visualisation grids
    SAMPLE_DIR  = "data/sampled_data"    # individual saved frames
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(SAMPLE_DIR, exist_ok=True)

    dataset = ScanNetTemporalDataset(
        root_dir=ROOT_DIR,
        seq_len_range=(4, 10),
        max_stride=10,
        max_scenes=5  # limit to 5 scenes for a quick test; remove to use all
    )

    print(f"Loaded {len(dataset.scene_data)} scenes")
    print(f"Dataset length (virtual): {len(dataset)}")
    print(f"Fixed seq_len this run  : {dataset.seq_len}")
    print(f"Saving visualisations to: {OUT_DIR}/")
    print(f"Saving sampled frames to: {SAMPLE_DIR}/\n")

    answer = input("Show all sampled batches or just a few? [all / few]: ").strip().lower()
    if answer == "few":
        while True:
            try:
                max_batches = int(input("How many batches? ").strip())
                if max_batches > 0:
                    break
                print("Please enter a positive integer.")
            except ValueError:
                print("Please enter a valid integer.")
    else:
        max_batches = None  # no limit

    loader = DataLoader(dataset, batch_size=2, num_workers=2, shuffle=False)

    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break

        images = batch["images"]   # (B, N, 3, H, W)
        depths = batch["depths"]   # (B, N, 1, H, W)
        poses  = batch["poses"]    # (B, N, 4, 4)
        scenes = batch["scene"]

        print(f"\nBatch {i + 1}:")
        print(f"  scenes : {list(scenes)}")
        print(f"  images : {tuple(images.shape)}  dtype={images.dtype}")
        print(f"  depths : {tuple(depths.shape)}  dtype={depths.dtype}")
        print(f"  poses  : {tuple(poses.shape)}   dtype={poses.dtype}")

        # save each sample in the batch
        for b, scene_name in enumerate(scenes):
            N = images.shape[1]

            # --- sampled_data: individual frames per sequence ---
            seq_dir = os.path.join(SAMPLE_DIR, f"batch{i+1}_sample{b+1}_{scene_name}")
            rgb_raw_dir = os.path.join(seq_dir, "color")
            dep_raw_dir = os.path.join(seq_dir, "depth")
            pose_dir    = os.path.join(seq_dir, "pose")
            os.makedirs(rgb_raw_dir, exist_ok=True)
            os.makedirs(dep_raw_dir, exist_ok=True)
            os.makedirs(pose_dir,    exist_ok=True)

            for f in range(N):
                torchvision.utils.save_image(images[b, f], os.path.join(rgb_raw_dir, f"{f:04d}.png"))
                np.save(os.path.join(dep_raw_dir, f"{f:04d}.npy"), depths[b, f].numpy())
                np.savetxt(os.path.join(pose_dir, f"{f:04d}.txt"), poses[b, f].numpy())

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

            print(f"  saved frames : {seq_dir}/")
            print(f"  saved rgb grid  : {rgb_path}")
            print(f"  saved depth grid: {dep_path}")

    print("\nDone.")
