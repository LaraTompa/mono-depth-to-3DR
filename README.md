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
  num_frames=6,           # exact number of frames per sample
  num_samples=700,        # number of samples per epoch
    max_stride=10,          # stride sampled uniformly from [2, max_stride]
    max_scenes=None,        # set an int to limit scenes loaded
)

loader = DataLoader(dataset, batch_size=2, num_workers=4)
batch = next(iter(loader))
# batch["scene"]   → list of scene name strings
```

**Quick test:**
```bash
python data/temporal_sampling.py
```

Prompts whether to run all batches or a fixed number, then writes two output folders:

| Folder | Contents |
|---|---|
| `datasets/sample_output/` | RGB and depth visualisation grids (PNG) |
| `datasets/sampled_data/` | Individual frames per sequence — `color/*.png`, `depth/*.npy`, `pose/*.txt`, plus `intrinsics_color.txt` and `intrinsics_depth.txt` |
