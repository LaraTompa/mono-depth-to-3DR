import sys
import os
import numpy as np

# ensure project root is on sys.path for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
# also add the data directory so local relative imports like `from scene import ...` resolve
sys.path.insert(0, os.path.join(project_root, 'data'))
from graph_based_sampling import ScanNetGraphDataset

# construct instance without triggering __init__
ds = object.__new__(ScanNetGraphDataset)
# set required attributes
setattr(ds, 'max_frame_gap', 5)
setattr(ds, 'min_overlap', 0.0)
setattr(ds, 'max_overlap', 1.0)
setattr(ds, 'overlap_sample_step', 1)
setattr(ds, 'depth_tolerance', 0.05)

# synthetic scene data
valid_fids = ['0', '1', '2']
poses = {fid: np.eye(4) for fid in valid_fids}
world_to_camera = {fid: np.linalg.inv(poses[fid]) for fid in valid_fids}
intrinsics = np.array([[100.0, 0.0, 2.0], [0.0, 100.0, 2.0], [0.0, 0.0, 1.0]])

# simple depth arrays (small)
def get_depth(fid):
    # return a small 4x4 depth array with ones
    return np.ones((4, 4), dtype=np.float32)

print('Running synthetic _compute_all_pairs test...')
graph = ds._compute_all_pairs(valid_fids, poses, world_to_camera, get_depth, intrinsics)
print('Resulting graph:')
print(graph)
print('Test complete.')
