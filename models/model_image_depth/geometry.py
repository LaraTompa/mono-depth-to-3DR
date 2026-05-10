"""
geometry.py — Differentiable projective geometry utilities.

All functions work on batched tensors (B, ...) and are fully differentiable.
Conventions:
  - K:     (B, 3, 3)  camera intrinsics
  - T_12:  (B, 4, 4)  rigid transform from view-1 cam frame to view-2 cam frame
           (i.e. p2 = T_12 @ p1)
  - depth: (B, 1, H, W)  in metres, zeros / negatives = invalid
"""

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Pixel grid
# ---------------------------------------------------------------------------

def make_pixel_grid(H: int, W: int, device: torch.device) -> torch.Tensor:
    """
    Return homogeneous pixel coordinates (3, H, W): [x, y, 1].
    x is the column index (0 … W-1), y is the row index (0 … H-1).
    """
    ys = torch.arange(H, dtype=torch.float32, device=device)
    xs = torch.arange(W, dtype=torch.float32, device=device)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")   # (H, W)
    ones = torch.ones_like(grid_x)
    return torch.stack([grid_x, grid_y, ones], dim=0)          # (3, H, W)


# ---------------------------------------------------------------------------
# Unproject: image + depth → 3-D points in camera space
# ---------------------------------------------------------------------------

def unproject(depth: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    """
    Lift every pixel to a 3-D point in the local camera frame.

    Parameters
    ----------
    depth : (B, 1, H, W)
    K     : (B, 3, 3)

    Returns
    -------
    pts3d : (B, 3, H, W)   X, Y, Z in metres
    """
    B, _, H, W = depth.shape
    grid = make_pixel_grid(H, W, depth.device)                 # (3, H, W)
    grid = grid.unsqueeze(0).expand(B, -1, -1, -1)             # (B, 3, H, W)

    K_inv = torch.linalg.inv(K)                                # (B, 3, 3)
    # flatten spatial dims for matmul
    grid_flat = grid.reshape(B, 3, H * W)                      # (B, 3, HW)
    rays = K_inv @ grid_flat                                    # (B, 3, HW)
    rays = rays.reshape(B, 3, H, W)

    pts3d = rays * depth                                        # (B, 3, H, W)
    return pts3d


# ---------------------------------------------------------------------------
# Project: 3-D points → pixel coordinates
# ---------------------------------------------------------------------------

def project(pts3d: torch.Tensor, K: torch.Tensor):
    """
    Project 3-D points (in camera frame) back to pixel coordinates.

    Parameters
    ----------
    pts3d : (B, 3, H, W)
    K     : (B, 3, 3)

    Returns
    -------
    coords : (B, H, W, 2)  (x, y) pixel coordinates — NOT normalised
    z      : (B, 1, H, W)  projected depth (Z component)
    """
    B, _, H, W = pts3d.shape
    pts_flat = pts3d.reshape(B, 3, H * W)                      # (B, 3, HW)
    proj = K @ pts_flat                                         # (B, 3, HW)
    proj = proj.reshape(B, 3, H, W)

    z = proj[:, 2:3, :, :]                                     # (B, 1, H, W)
    x = proj[:, 0:1, :, :] / (z + 1e-8)
    y = proj[:, 1:2, :, :] / (z + 1e-8)

    coords = torch.cat([x, y], dim=1).permute(0, 2, 3, 1)     # (B, H, W, 2)
    return coords, z


# ---------------------------------------------------------------------------
# Transform: apply rigid body transform T to a point cloud
# ---------------------------------------------------------------------------

def transform_pts(pts3d: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
    """
    Apply a rigid 4×4 transform T to a 3-D point field.

    Parameters
    ----------
    pts3d : (B, 3, H, W)
    T     : (B, 4, 4)

    Returns
    -------
    pts_out : (B, 3, H, W)
    """
    B, _, H, W = pts3d.shape
    R = T[:, :3, :3]    # (B, 3, 3)
    t = T[:, :3, 3:]    # (B, 3, 1)

    pts_flat = pts3d.reshape(B, 3, H * W)                      # (B, 3, HW)
    pts_out  = R @ pts_flat + t                                 # (B, 3, HW)
    return pts_out.reshape(B, 3, H, W)


# ---------------------------------------------------------------------------
# Normalise pixel coords → grid_sample format [-1, 1]
# ---------------------------------------------------------------------------

def normalise_coords(coords: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """
    Convert pixel coordinates (x, y) ∈ [0, W-1] × [0, H-1]
    to grid_sample coordinates ∈ [-1, 1].

    Parameters
    ----------
    coords : (B, H, W, 2)   (x, y)

    Returns
    -------
    coords_norm : (B, H, W, 2)  normalised (x, y)
    """
    x = coords[..., 0]
    y = coords[..., 1]
    x_n = 2.0 * x / (W - 1) - 1.0
    y_n = 2.0 * y / (H - 1) - 1.0
    return torch.stack([x_n, y_n], dim=-1)


# ---------------------------------------------------------------------------
# Warp feature map / image from view 1 → view 2
# ---------------------------------------------------------------------------

def warp(
    feat: torch.Tensor,
    depth_src: torch.Tensor,
    T_12: torch.Tensor,
    K: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Differentiably warp *feat* (in view-1 frame) into view-2 frame using
    depth_src and the relative pose T_12.

    Parameters
    ----------
    feat      : (B, C, H, W)   feature map / image to warp
    depth_src : (B, 1, H, W)   depth of view 1
    T_12      : (B, 4, 4)      transform  p2 = T_12 @ p1
    K         : (B, 3, 3)      shared intrinsics (both views same resolution)

    Returns
    -------
    warped : (B, C, H, W)   feat sampled at reprojected locations
    valid  : (B, 1, H, W)   bool mask: True where reprojection is in-bounds
                            and source depth > 0
    """
    B, C, H, W = feat.shape

    pts1 = unproject(depth_src, K)           # (B, 3, H, W)  — view-1 cam frame
    pts2 = transform_pts(pts1, T_12)         # (B, 3, H, W)  — view-2 cam frame
    coords, z2 = project(pts2, K)            # (B, H, W, 2), (B, 1, H, W)

    coords_n = normalise_coords(coords, H, W)   # (B, H, W, 2)

    warped = F.grid_sample(
        feat, coords_n,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )

    valid = (
        (z2 > 0) &
        (depth_src > 0) &
        (coords[..., 0:1].permute(0, 3, 1, 2) >= 0) &
        (coords[..., 0:1].permute(0, 3, 1, 2) <= W - 1) &
        (coords[..., 1:2].permute(0, 3, 1, 2) >= 0) &
        (coords[..., 1:2].permute(0, 3, 1, 2) <= H - 1)
    )

    return warped, valid


# ---------------------------------------------------------------------------
# Compute reprojection coordinates only (used by local attention)
# ---------------------------------------------------------------------------

def reprojection_coords(
    depth_src: torch.Tensor,
    T_12: torch.Tensor,
    K: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Return the projected pixel locations in view 2 for each pixel in view 1.

    Returns
    -------
    coords : (B, H, W, 2)   (x, y) in view-2 pixel space
    valid  : (B, 1, H, W)   bool mask
    """
    B, _, H, W = depth_src.shape

    pts1 = unproject(depth_src, K)
    pts2 = transform_pts(pts1, T_12)
    coords, z2 = project(pts2, K)

    valid = (
        (z2 > 0) &
        (depth_src > 0) &
        (coords[..., 0:1].permute(0, 3, 1, 2) >= 0) &
        (coords[..., 0:1].permute(0, 3, 1, 2) <= W - 1) &
        (coords[..., 1:2].permute(0, 3, 1, 2) >= 0) &
        (coords[..., 1:2].permute(0, 3, 1, 2) <= H - 1)
    )

    return coords, valid


# ---------------------------------------------------------------------------
# Rotation → 6D representation (Zhou et al., 2019)
# ---------------------------------------------------------------------------

def rot_to_6d(R: torch.Tensor) -> torch.Tensor:
    """
    R : (B, 3, 3)  →  6d : (B, 6)  (first two columns of R, row-major)
    """
    return R[:, :, :2].permute(0, 2, 1).reshape(-1, 6)   # (B, 6)
