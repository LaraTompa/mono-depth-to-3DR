"""
augmentation.py — Geometric and photometric augmentations for frame pairs.

Applied per-sample in PreSampledPairDataset.__getitem__ during training only.
All transforms operate on CPU tensors and preserve geometric consistency:
depths, mde_depths, poses, and intrinsics are updated alongside RGB images
for any spatial operation.

Supported augmentations (controlled via aug_cfg dict from training.yaml):
    horizontal_flip : both views flipped together; intrinsics + c2w poses updated.
    color_jitter    : brightness/contrast/saturation/hue, per-view independently;
                      only RGB images modified, no geometry update.

Design notes
------------
- Horizontal flip flips BOTH views with the same random decision so that the
  relative pose between them is consistent.
- Pose update derivation (c2w, camera-to-world):
    A horizontal flip reflects the camera's x-axis (u → W-1-u in pixel space,
    x → -x in normalised camera coords).  The same world point p_w must satisfy:
        p_w = T_cw_new @ p_c_flip  where  p_c_flip = F3 @ p_c_orig
    Solving:  T_cw_new = T_cw_orig @ F4
    where F4 = diag(-1, 1, 1, 1) in homogeneous coordinates.
    This negates the first column of R (only) and leaves t unchanged.
- Intrinsics update: cx ← W - 1 - cx  (K[0, 2]).  fx, fy, cy are unchanged.
- The relative pose T_12_gt = inv(T_cw2) @ T_cw1 is recomputed inside
  training/train.py from the (already updated) batch poses, so no separate
  relative-pose correction is needed here.
"""

import random

import torch
import torchvision.transforms.functional as TF


# ---------------------------------------------------------------------------
# Horizontal flip (geometric — affects images, depths, poses, intrinsics)
# ---------------------------------------------------------------------------

# Pre-built flip matrix F4 = diag(-1, 1, 1, 1); constructed once at import time.
_F4 = torch.tensor(
    [[-1., 0., 0., 0.],
     [ 0., 1., 0., 0.],
     [ 0., 0., 1., 0.],
     [ 0., 0., 0., 1.]],
    dtype=torch.float32,
)


def horizontal_flip_pair(
    images:     torch.Tensor,   # (2, 3, H, W)  float32 in [0, 1]
    depths:     torch.Tensor,   # (2, 1, H, W)  float32, metres
    mde_depths: torch.Tensor,   # (2, 1, H, W)  float32, metres
    poses:      torch.Tensor,   # (2, 4, 4)     camera-to-world float32
    intrinsics: torch.Tensor,   # (3, 3)        float32
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Flip both views of a frame pair horizontally.

    Spatial data (images, depths, mde_depths) are mirrored along the W axis.
    Intrinsics and poses are updated analytically so the 3-D geometry is
    consistent with the flipped pixel coordinates.

    Returns the same five tensors in the same order, all updated.
    """
    W = images.shape[-1]

    # --- Spatial flip (all views) -------------------------------------------
    images_f     = images.flip(-1)
    depths_f     = depths.flip(-1)
    mde_depths_f = mde_depths.flip(-1)

    # --- Intrinsics: shift principal point cx ← W - 1 - cx -----------------
    intr_f = intrinsics.clone()
    intr_f[0, 2] = float(W) - 1.0 - intr_f[0, 2]

    # --- Poses (c2w): T_cw_new = T_cw @ F4 ----------------------------------
    # Broadcasting: (2, 4, 4) @ (4, 4) → (2, 4, 4)
    F4 = _F4.to(poses.device, poses.dtype)
    poses_f = poses @ F4

    return images_f, depths_f, mde_depths_f, poses_f, intr_f


# ---------------------------------------------------------------------------
# Color jitter (photometric — RGB images only, per-view independently)
# ---------------------------------------------------------------------------

def color_jitter_pair(
    images:     torch.Tensor,  # (2, 3, H, W)  float32 in [0, 1]
    brightness: float = 0.2,
    contrast:   float = 0.2,
    saturation: float = 0.2,
    hue:        float = 0.05,
) -> torch.Tensor:
    """
    Apply independent random color jitter to each view.

    Each view draws its own random factor so the two views of the same pair
    have different photometric appearances, mimicking real exposure variation.
    Clamps output to [0, 1] to guard against floating-point overshoot.

    Returns modified images tensor; does NOT modify depths, poses, or
    intrinsics.
    """
    out = images.clone()
    for v in range(images.shape[0]):
        img_v = out[v]  # (3, H, W)

        if brightness > 0.0:
            b = random.uniform(max(0.0, 1.0 - brightness), 1.0 + brightness)
            img_v = TF.adjust_brightness(img_v, b)
        if contrast > 0.0:
            c = random.uniform(max(0.0, 1.0 - contrast), 1.0 + contrast)
            img_v = TF.adjust_contrast(img_v, c)
        if saturation > 0.0:
            s = random.uniform(max(0.0, 1.0 - saturation), 1.0 + saturation)
            img_v = TF.adjust_saturation(img_v, s)
        if hue > 0.0:
            h = random.uniform(-hue, hue)
            img_v = TF.adjust_hue(img_v, h)

        out[v] = img_v.clamp(0.0, 1.0)
    return out


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def augment_pair(
    images:     torch.Tensor,   # (2, 3, H, W)
    depths:     torch.Tensor,   # (2, 1, H, W)
    mde_depths: torch.Tensor,   # (2, 1, H, W)
    poses:      torch.Tensor,   # (2, 4, 4)
    intrinsics: torch.Tensor,   # (3, 3)
    aug_cfg:    dict,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Apply the augmentations specified in aug_cfg to a single frame pair.

    aug_cfg keys (all optional; omitting or setting to 0 disables that transform):

        p_hflip    (float, default 0.0) : probability of applying horizontal flip.
        brightness (float, default 0.0) : max brightness jitter magnitude.
        contrast   (float, default 0.0) : max contrast jitter magnitude.
        saturation (float, default 0.0) : max saturation jitter magnitude.
        hue        (float, default 0.0) : max hue shift as a fraction of 0.5
                                          (range [-hue*0.5, hue*0.5] degrees is
                                           internally used by TF.adjust_hue).

    Returns (images, depths, mde_depths, poses, intrinsics) — same shapes,
    possibly transformed.
    """
    # ---- Horizontal flip (geometric) ---------------------------------------
    p_hflip = float(aug_cfg.get("p_hflip", 0.0))
    if p_hflip > 0.0 and random.random() < p_hflip:
        images, depths, mde_depths, poses, intrinsics = horizontal_flip_pair(
            images, depths, mde_depths, poses, intrinsics
        )

    # ---- Color jitter (photometric, RGB only) ------------------------------
    brightness = float(aug_cfg.get("brightness", 0.0))
    contrast   = float(aug_cfg.get("contrast",   0.0))
    saturation = float(aug_cfg.get("saturation", 0.0))
    hue        = float(aug_cfg.get("hue",        0.0))
    if any(v > 0.0 for v in (brightness, contrast, saturation, hue)):
        images = color_jitter_pair(images, brightness, contrast, saturation, hue)

    return images, depths, mde_depths, poses, intrinsics
