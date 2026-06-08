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
        zoe-depth_pred/ {fid}.png          MDE prior, uint16 PNG, millimetres  (zoedepth)
        depth_pro_pred/ {fid}.npz          MDE prior, float32 metres            (depthpro)
        pose/           {fid}.txt
        intrinsic/      intrinsic_depth.txt
"""

import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

from scene import find_scene_paths, ScanNetScene
from augmentation import augment_pair


class PreSampledPairDataset(Dataset):
    """
    Flat list of frame pairs built from every pre-sampled scene under root_dir.

    For a scene with 6 frames [f0, f1, f2, f3, f4, f5] this yields 3 pairs:
        (f0, f1), (f2, f3), (f4, f5)

    Parameters
    ----------
    root_dir    : str   Path to datasets/sampled_data.
    mde_source  : str   "zoedepth" (default) | "depthpro"
    scene_paths : list  Optional pre-split list of scene dirs (train/val/test
                        split computed in train.py).  When supplied, root_dir
                        is not searched again.
    """

    def __init__(
        self,
        root_dir:   str,
        mde_source: str  = "zoedepth",
        scene_paths=None,
        aug_cfg:    dict | None = None,
        image_size: tuple | list | None = None,
    ):
        """
        aug_cfg    : dict mapping augmentation parameters (see data/augmentation.py).
                     Pass None (default) to disable all augmentation (val/test).
        image_size : (H, W) target size for color images and MDE-depth priors.
                     When set, all images/MDE depths are resized to this resolution
                     at load time, ensuring all batch items are collatable even when
                     scenes have different native resolutions.
        """
        if scene_paths is None:
            scene_paths = find_scene_paths(root_dir)

        self.mde_source = mde_source
        self.aug_cfg    = aug_cfg
        self.image_size = tuple(image_size) if image_size is not None else None
        self.pairs = []
        for scene_path in scene_paths:
            scene = ScanNetScene(scene_path, mde_source=mde_source)
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
                "mde_depth_ext": scene.mde_depth_ext,
                "intrinsics":    scene.intrinsics,   # (3, 3) float32
                "poses":         poses,              # dict[fid → 4×4 ndarray]
            }
            n_skipped = 0
            for i in range(0, len(valid_fids) - 1, 2):
                fid_a, fid_b = valid_fids[i], valid_fids[i + 1]
                # Filter pairs where the camera barely moved: compute the
                # relative translation analytically (SE(3) inverse avoids
                # np.linalg.inv instability for near-degenerate matrices).
                #   T_12 = inv(pose_b) @ pose_a
                #   t_12 = R_b^T (t_a − t_b)
                pa, pb = poses[fid_a].astype(np.float32), poses[fid_b].astype(np.float32)
                t_12 = pb[:3, :3].T @ (pa[:3, 3] - pb[:3, 3])
                if np.linalg.norm(t_12) < 2e-2:   # < 2 cm relative motion
                    n_skipped += 1
                    continue
                self.pairs.append((info, fid_a, fid_b))
            if n_skipped:
                print(f"[Dataset]   {scene.name}: skipped {n_skipped} near-stationary pair(s)")

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
            mde_depths.append(self._load_mde_depth(
                os.path.join(info["mde_depth_dir"], f"{fid}{info['mde_depth_ext']}"),
                self.mde_source,
            ))
            poses.append(torch.from_numpy(info["poses"][fid]).float())

        imgs_t  = torch.stack(images)                           # (2, 3, H, W)
        deps_t  = torch.stack(depths)                           # (2, 1, H, W)  GT
        mde_t   = torch.stack(mde_depths)                       # (2, 1, H, W)  MDE prior
        poses_t = torch.stack(poses)                            # (2, 4, 4)
        intr_t  = torch.from_numpy(info["intrinsics"]).clone()    # (3, 3)  clone: from_numpy shares storage (non-resizable); collate requires owned storage

        # Resize color images and MDE priors to a fixed spatial size so that
        # all samples in a batch are collatable (scenes may have different native
        # resolutions, e.g. 480×640 vs 968×1296).
        # GT depths are NOT resized — they stay at the depth-camera native resolution.
        if self.image_size is not None:
            if imgs_t.shape[-2:] != torch.Size(self.image_size):
                imgs_t = F.interpolate(
                    imgs_t, size=self.image_size, mode="bilinear", align_corners=False
                )
            if mde_t.shape[-2:] != torch.Size(self.image_size):
                mde_t = F.interpolate(
                    mde_t, size=self.image_size, mode="bilinear", align_corners=False
                )
            # Scale intrinsics to match the resized spatial resolution.
            # intrinsic_depth.txt is defined for the GT-depth native resolution
            # (deps_t has never been resized, so its shape is the reference).
            native_h, native_w = deps_t.shape[-2], deps_t.shape[-1]
            target_h, target_w = self.image_size
            if (target_h, target_w) != (native_h, native_w):
                scale_w = target_w / native_w
                scale_h = target_h / native_h
                intr_t = intr_t.clone()
                intr_t[0, 0] *= scale_w   # fx
                intr_t[1, 1] *= scale_h   # fy
                intr_t[0, 2] *= scale_w   # cx
                intr_t[1, 2] *= scale_h   # cy

        if self.aug_cfg is not None:
            imgs_t, deps_t, mde_t, poses_t, intr_t = augment_pair(
                imgs_t, deps_t, mde_t, poses_t, intr_t, self.aug_cfg
            )

        return {
            "images":     imgs_t,
            "depths":     deps_t,
            "mde_depths": mde_t,
            "poses":      poses_t,
            "intrinsics": intr_t,
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
    def _load_mde_depth(path: str, mde_source: str) -> torch.Tensor:
        if mde_source == "depthpro":
            data = np.load(path)
            for key in ["depth", "pred", "prediction", "arr_0"]:
                if key in data:
                    depth = data[key].astype(np.float32)
                    break
            else:
                depth = data[list(data.keys())[0]].astype(np.float32)
            if depth.ndim == 3 and depth.shape[0] == 1:
                depth = depth[0]
            t = torch.from_numpy(depth)
        else:
            # ZoeDepth: uint16 PNG in mm → float32 metres
            t = torch.from_numpy(np.array(Image.open(path)).astype(np.float32) / 1000.0)
        t = t if t.ndim == 3 else t.unsqueeze(0)   # ensure (1, H, W)
        return t
