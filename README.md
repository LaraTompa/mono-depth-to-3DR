# From monocular depth to multiview reconstruction
The idea of this project is with the given predicted monocular depth predictions, design and train a lightweight network to align these monocular depths and predict camera parameters to ensure multiview consistency.

## Setup

Create and activate a conda environment named `mono-depth-3dr`:

```bash
conda create -n mono-depth-3dr python=3.11 -y
conda activate mono-depth-3dr
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

## Data

### ScanNet Temporal Dataset

`data/temporal_sampling.py` implements `ScanNetTemporalDataset`, a PyTorch `Dataset` for dynamic temporal sequence sampling from [ScanNet](http://www.scan-net.org/).

Each `__getitem__` call randomly selects a scene and samples a temporally ordered sequence of frames with a random stride.

**Expected dataset structure:**
```
<root_dir>/
  scene0000_00/
    color/          # *.jpg RGB frames
    depth/          # *.png depth frames (uint16, millimetres)
    pose/           # *.txt 4×4 camera-to-world matrices
    intrinsics_color.txt
    intrinsics_depth.txt
  scene0001_00/
  ...
```

**Usage:**
```python
from data.temporal_sampling import ScanNetTemporalDataset
from torch.utils.data import DataLoader

dataset = ScanNetTemporalDataset(
    root_dir="/storage/group/dataset_mirrors/scannet/tasks/scannet_frames_test",
    seq_len_range=(4, 10),  # random sequence length in this range
    max_stride=10,          # stride sampled uniformly from [2, max_stride]
    max_scenes=None,        # set an int to limit scenes loaded
)

# N is fixed per epoch — call this at the start of each epoch to resample it
dataset.resample_seq_len()

loader = DataLoader(dataset, batch_size=2, num_workers=4)
batch = next(iter(loader))
# batch["images"]  → (B, N, 3, H, W)  float32 in [0, 1]
# batch["depths"]  → (B, N, 1, H, W)  float32 in metres
# batch["poses"]   → (B, N, 4, 4)     float32 camera-to-world
# batch["scene"]   → list of scene name strings
```

**Quick test:**
```bash
python data/temporal_sampling.py
```

Prompts whether to run all batches or a fixed number, then writes two output folders:

| Folder | Contents |
|---|---|
| `data/sample_output/` | RGB and depth visualisation grids (PNG) |
| `data/sampled_data/` | Individual frames per sequence — `color/*.png`, `depth/*.npy`, `pose/*.txt` |
