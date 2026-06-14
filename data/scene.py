"""
scene.py — Per-scene abstraction for ScanNet.

Provides:
  - load_pose / load_intrinsics / load_depth_png  (shared I/O helpers)
  - ScanNetScene                                  (lazy, path-aware scene object)
"""

import os
import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Shared I/O helpers
# ---------------------------------------------------------------------------

def load_pose(path: str) -> np.ndarray:
    """Load a 4×4 camera-to-world pose matrix from a text file."""
    return np.loadtxt(path)


def load_intrinsics(path: str) -> np.ndarray:
    """Load camera intrinsics, returning a 3×3 float32 matrix."""
    K = np.loadtxt(path).astype(np.float32)
    if K.shape == (4, 4):
        K = K[:3, :3]
    return K


def load_depth_png(path: str) -> np.ndarray:
    """Load a ScanNet uint16 PNG depth map and convert to metres (float32)."""
    return np.array(Image.open(path)).astype(np.float32) / 1000.0


def load_depth(path: str) -> np.ndarray:
    """Load a .npy depth map (float32, metres)."""
    return np.load(path).astype(np.float32)


# ---------------------------------------------------------------------------
# Scene class
# ---------------------------------------------------------------------------

class ScanNetScene:
    """
    Lightweight wrapper around a single ScanNet scene directory.

    Directory layout (sampled data format)::

        <scene_path>/
            color/          {fid}.png
            depth/          {fid}.npy  (float32, shape (1,H,W), metres)
            pose/           {fid}.txt
            intrinsic/
                intrinsic_depth.txt
                intrinsic_color.txt
    """

    COLOR_EXT = ".png"
    DEPTH_EXT = ".npy"
    MDE_DEPTH_EXT = ".png"          # ZoeDepth predictions (uint16 PNG, mm → metres)
    MDE_DEPTH_DIR = "zoe-depth_pred"
    DEPTHPRO_EXT = ".npz"           # DepthPro predictions (float32 metres)
    DEPTHPRO_DIR = "depth-pro_pred"
    POSE_EXT = ".txt"
    INTRINSIC_FNAME = "intrinsic_depth.txt"

    def __init__(self, scene_path: str, mde_source: str = "zoedepth"):
        self.scene_path = scene_path
        self.name = os.path.basename(scene_path)
        self.mde_source = mde_source

        self.color_dir = os.path.join(scene_path, "color")
        self.depth_dir = os.path.join(scene_path, "depth")
        if mde_source == "depthpro":
            self.mde_depth_dir = os.path.join(scene_path, self.DEPTHPRO_DIR)
            self.mde_depth_ext = self.DEPTHPRO_EXT
        else:
            self.mde_depth_dir = os.path.join(scene_path, self.MDE_DEPTH_DIR)
            self.mde_depth_ext = self.MDE_DEPTH_EXT
        self.pose_dir = os.path.join(scene_path, "pose")
        self.intrinsic_dir = os.path.join(scene_path, "intrinsic")
        self.intrinsic_path = os.path.join(
            self.intrinsic_dir, self.INTRINSIC_FNAME
        )

        self._frame_ids: list | None = None
        self._intrinsics: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Validity check
    # ------------------------------------------------------------------

    def is_valid(self) -> bool:
        """Return True if the scene has all required subdirectories."""
        return (
            os.path.isdir(self.color_dir)
            and os.path.isdir(self.depth_dir)
            and os.path.isdir(self.pose_dir)
            and os.path.isdir(self.mde_depth_dir)
            and os.path.isfile(self.intrinsic_path)
        )

    # ------------------------------------------------------------------
    # Lazy properties
    # ------------------------------------------------------------------

    @property
    def frame_ids(self) -> list[str]:
        """Sorted list of frame IDs derived from .png files in the color directory."""
        if self._frame_ids is None:
            self._frame_ids = sorted(
                (
                    f[: -len(self.COLOR_EXT)]
                    for f in os.listdir(self.color_dir)
                    if f.endswith(self.COLOR_EXT)
                ),
                key=int,
            )
        return self._frame_ids

    @property
    def intrinsics(self) -> np.ndarray:
        """3×3 float32 depth-camera intrinsic matrix (cached after first load)."""
        if self._intrinsics is None:
            self._intrinsics = load_intrinsics(self.intrinsic_path)
        return self._intrinsics

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------

    def color_path(self, fid: str) -> str:
        return os.path.join(self.color_dir, f"{fid}{self.COLOR_EXT}")

    def depth_path(self, fid: str) -> str:
        return os.path.join(self.depth_dir, f"{fid}{self.DEPTH_EXT}")

    def mde_depth_path(self, fid: str) -> str:
        return os.path.join(self.mde_depth_dir, f"{fid}{self.MDE_DEPTH_EXT}")

    def pose_path(self, fid: str) -> str:
        return os.path.join(self.pose_dir, f"{fid}{self.POSE_EXT}")

    def intrinsic_color_path(self) -> str:
        return os.path.join(self.intrinsic_dir, "intrinsic_color.txt")

    # ------------------------------------------------------------------
    # Pose loading
    # ------------------------------------------------------------------

    def load_all_poses(self) -> tuple[dict, list]:
        """
        Load poses for every frame in the scene.

        Returns
        -------
        poses : dict[str, np.ndarray]
            Mapping from frame ID to 4×4 camera-to-world matrix.
        valid_fids : list[str]
            Frame IDs whose pose file exists, contains no non-finite values,
            and has a non-zero translation column.
            (ScanNet marks missing/failed poses with inf; some pre-sampled
            scenes may contain all-zero pose matrices for stationary frames —
            those are also excluded so they do not produce degenerate T_12.)
        """
        # One listdir call to find existing pose files (avoids per-frame stat).
        pose_files_exist = {
            f[: -len(self.POSE_EXT)]
            for f in os.listdir(self.pose_dir)
            if f.endswith(self.POSE_EXT)
        }
        loadable_fids = [fid for fid in self.frame_ids if fid in pose_files_exist]
        poses = {fid: load_pose(self.pose_path(fid)) for fid in loadable_fids}
        valid_fids = [
            fid for fid in loadable_fids
            if np.all(np.isfinite(poses[fid]))
            and np.linalg.norm(poses[fid][:3, 3]) > 1e-6  # skip zero-translation frames
        ]
        return poses, valid_fids

    def valid_fids_fast(self) -> list[str]:
        """
        Return frame IDs that have a corresponding pose file, without reading
        any pose file contents.  ~1000× faster than load_all_poses() at startup
        because it only calls os.listdir() once instead of np.loadtxt() per frame.

        Trade-off: does not filter out ScanNet's inf-pose frames.  Those are a
        small minority and are caught at load time in __getitem__.
        """
        pose_files = {
            f[: -len(self.POSE_EXT)]
            for f in os.listdir(self.pose_dir)
            if f.endswith(self.POSE_EXT)
        }
        return [fid for fid in self.frame_ids if fid in pose_files]

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"ScanNetScene(name={self.name!r}, path={self.scene_path!r})"


# ---------------------------------------------------------------------------
# Recursive scene discovery
# ---------------------------------------------------------------------------

def find_scene_paths(root_dir: str) -> list[str]:
    """
    Recursively search root_dir for valid ScanNetScene directories.

    A directory is considered a scene if it contains color/, depth/, pose/
    and intrinsic/intrinsic_depth.txt.  This handles both flat layouts
    (root_dir/<scene>/) and nested layouts (root_dir/batch/sample/<scene>/).

    Returns a list of absolute scene directory paths.
    """
    scenes = []
    for dirpath, dirnames, _ in os.walk(root_dir):
        scene = ScanNetScene(dirpath)
        if scene.is_valid() and scene.frame_ids:
            scenes.append(dirpath)
            dirnames.clear()   # don't recurse into a valid scene directory
    return sorted(scenes)
