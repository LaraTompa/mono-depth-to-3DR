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
    from torch.utils.data import DataLoader

    ROOT_DIR = "/storage/group/dataset_mirrors/scannet/tasks/scannet_frames_test"

    dataset = ScanNetTemporalDataset(
        root_dir=ROOT_DIR,
        seq_len_range=(4, 10),
        max_stride=10,
        max_scenes=5  # limit to 5 scenes for a quick test; remove to use all
    )

    print(f"Loaded {len(dataset.scene_data)} scenes")
    print(f"Dataset length (virtual): {len(dataset)}")

    loader = DataLoader(dataset, batch_size=2, num_workers=2, shuffle=False)

    for i, batch in enumerate(loader):
        images = batch["images"]   # (B, N, 3, H, W)
        depths = batch["depths"]   # (B, N, 1, H, W)
        poses  = batch["poses"]    # (B, N, 4, 4)
        scenes = batch["scene"]

        print(f"\nBatch {i + 1}:")
        print(f"  scenes : {list(scenes)}")
        print(f"  images : {tuple(images.shape)}  dtype={images.dtype}")
        print(f"  depths : {tuple(depths.shape)}  dtype={depths.dtype}")
        print(f"  poses  : {tuple(poses.shape)}   dtype={poses.dtype}")

        if i >= 2:  # print 3 batches then stop
            break

    print("\nDone.")
