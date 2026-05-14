"""
preprocessing.py — Dataset that reads directly from the pre-sampled data directory.

Exposes PreSampledPairDataset, which is the only dataset class used during
training.  It reads consecutive frame pairs (0,1), (2,3), (4,5) from every
scene directory discovered under root_dir.

Expected on-disk layout
-----------------------
    <root_dir>/batch*/sample*/<scene>/
        color/          {fid}.png
        depth/          {fid}.npy          GT depth, float32, metres
        zoe-depth_pred/ {fid}.png          MDE prior, uint16 PNG, millimetres
        pose/           {fid}.txt
        intrinsic/      intrinsic_depth.txt
"""

import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from scene import find_scene_paths, ScanNetScene


class PreSampledPairDataset(Dataset):
    """
    Flat list of frame pairs built from every pre-sampled scene under root_dir.

    For a scene with 6 frames [f0, f1, f2, f3, f4, f5] this yields 3 pairs:
        (f0, f1), (f2, f3), (f4, f5)

    Parameters
    ----------
    root_dir    : str   Path to datasets/sampled_data.
    scene_paths : list  Optional pre-split list of scene dirs (train/val/test
                        split computed in train.py).  When supplied, root_dir
                        is not searched again.
    """

    def __init__(self, root_dir: str, scene_paths=None):
        if scene_paths is None:
            scene_paths = find_scene_paths(root_dir)

        self.pairs = []
        for scene_path in scene_paths:
            scene = ScanNetScene(scene_path)
            if not scene.is_valid() or not scene.frame_ids:
                continue
            poses, valid_fids = scene.load_all_poses()
            if not valid_fids:
                continue
            info = {
                "scene":         os.path.relpath(scene_path, root_dir),
                "color_dir":     scene.color_dir,
                "depth_dir":     scene.depth_dir,
                "mde_depth_dir": scene.mde_depth_dir,
                "intrinsics":    scene.intrinsics,   # (3, 3) float32
                "poses":         poses,              # dict[fid → 4×4 ndarray]
            }
            for i in range(0, len(valid_fids) - 1, 2):
                self.pairs.append((info, valid_fids[i], valid_fids[i + 1]))

        if not self.pairs:
            raise RuntimeError(f"No valid frame pairs found under {root_dir!r}")
        print(f"[Dataset] {len(self.pairs)} pairs from {len(scene_paths)} scenes")

    # ------------------------------------------------------------------
    # PyTorch interface
    # ------------------------------------------------------------------

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        info, fid_a, fid_b = self.pairs[idx]

        images, depths, mde_depths, poses = [], [], [], []
        for fid in (fid_a, fid_b):
            images.append(self._load_image(
                os.path.join(info["color_dir"],     f"{fid}{ScanNetScene.COLOR_EXT}")))
            depths.append(self._load_npy(
                os.path.join(info["depth_dir"],     f"{fid}{ScanNetScene.DEPTH_EXT}")))
            mde_depths.append(self._load_png_depth(
                os.path.join(info["mde_depth_dir"], f"{fid}{ScanNetScene.MDE_DEPTH_EXT}")))
            poses.append(torch.from_numpy(info["poses"][fid]).float())

        return {
            "images":     torch.stack(images),                         # (2, 3, H, W)
            "depths":     torch.stack(depths),                         # (2, 1, H, W)  GT
            "mde_depths": torch.stack(mde_depths),                     # (2, 1, H, W)  MDE prior
            "poses":      torch.stack(poses),                          # (2, 4, 4)
            "intrinsics": torch.from_numpy(info["intrinsics"]),        # (3, 3)
            "scene":      info["scene"],
            "frame_ids":  [fid_a, fid_b],
        }

    # ------------------------------------------------------------------
    # I/O helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_image(path: str) -> torch.Tensor:
        img = Image.open(path).convert("RGB")
        return torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0

    @staticmethod
    def _load_npy(path: str) -> torch.Tensor:
        t = torch.from_numpy(np.load(path).astype(np.float32))
        return t if t.ndim == 3 else t.unsqueeze(0)

    @staticmethod
    def _load_png_depth(path: str) -> torch.Tensor:
        """Load ZoeDepth prediction: uint16 PNG in mm → float32 metres."""
        t = torch.from_numpy(np.array(Image.open(path)).astype(np.float32) / 1000.0)
        return t if t.ndim == 3 else t.unsqueeze(0)
