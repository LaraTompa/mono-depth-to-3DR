import numpy as np
import cv2


def load_intrinsics(path):
    K = np.loadtxt(path)
    if K.shape == (4, 4):
        K = K[:3, :3]
    assert K.shape == (3, 3)
    return K


def load_pose(path):
    pose = np.loadtxt(path)
    if pose.shape == (3, 4):
        pose = np.vstack([pose, [0, 0, 0, 1]])
    assert pose.shape == (4, 4)
    return pose


def load_image(path, as_gray=True):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)
    if as_gray:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #print(f"Loaded image {path} with shape {img.shape}")
    return img.astype(np.float32) / 255.0


def load_depth(path, scale=1.0):
    if path.endswith(".npz"):
        data = np.load(path)
        depth = data[list(data.keys())[0]]
        #handle common cases of extra dimensions
        if depth.ndim == 3 and depth.shape[0] == 1:
            depth = depth[0]
        if depth.ndim == 3 and depth.shape[-1] in (1, 3, 4):
            depth = depth[..., 0]
    elif path.endswith(".npy"):
        depth = np.load(path)
        #handle common cases of extra dimensions
        if depth.ndim == 3 and depth.shape[0] == 1:
            depth = depth[0]
        if depth.ndim == 3 and depth.shape[-1] in (1, 3, 4):
            depth = depth[..., 0]
    else:
        depth = cv2.imread(path, cv2.IMREAD_UNCHANGED)

    if depth is None:
        raise FileNotFoundError(path)

    depth = depth.astype(np.float32)

    # if RGB depth → convert
    if depth.ndim == 3:
        depth = cv2.cvtColor(depth, cv2.COLOR_BGR2GRAY)
    
    #print(f"Loaded depth {path} with shape {depth.shape}, dtype={depth.dtype}, min={depth.min():.3f}, max={depth.max():.3f}")

    return depth / scale