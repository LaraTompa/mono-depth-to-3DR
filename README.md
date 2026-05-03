# Monocular depth to multiview reconstruction
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
    depth/          # *.png depth frames (uint16 millimetres)
    pose/           # *.txt 4×4 camera-to-world matrices
    intrinsic/
      intrinsic_color.txt
      intrinsic_depth.txt
  scene0001_00/
  ...
```

**Change settings**
```
config/dataset.yaml

dataset:
  root_dir: "/storage/group/dataset_mirrors/scannet/scans"
  num_samples: 700       # number of samples per epoch (__len__)
  num_frames: 5          # exact number of frames per sample (seq_len)
  min_stride: 80          # minimum temporal stride between frames
  max_stride: 120         # maximum temporal stride between frames
  max_scenes: null       # limit scenes loaded; set to an integer to restrict (e.g. 5)

output:
  sample_output_dir: "datasets/sample_output"   # visualisation grids
  sampled_data_dir:  "datasets/sampled_data"    # individual saved frames
  batch_size: 2
  num_workers: 2
```

**Run:**
```bash
python data/temporal_sampling.py
```

Prompts whether to run all batches or a fixed number, then writes two output folders:

| Folder | Contents |
|---|---|
| `datasets/sample_output/` | RGB and depth visualisation grids (PNG) |
| `datasets/sampled_data/` | Individual frames per sequence — `color/*.png`, `depth/*.npy`, `pose/*.txt`, and `intrinsics/intrinsics_color.txt`, `intrinsics/intrinsics_depth.txt` |

## Monocular Depth Inference

### ZoeDepth

`model/zoe_depth.py` runs ZoeDepth inference on all sequences produced by the samplers.

**Prerequisites:** clone [ZoeDepth](https://github.com/isl-org/ZoeDepth) on the workstation and activate its environment.

**Run** (from the ZoeDepth root so `source="local"` resolves correctly):
```bash
cd ~/ZoeDepth
python ~/mono-depth-to-3DR/model/zoe_depth.py
```

Walks every sequence folder under `datasets/sampled_data/`, reads `color/*.png`, and writes predicted metric depths to `depth_pred/` alongside each sequence:

```
datasets/sampled_data/
  <seq_name>/
    color/          # input RGB frames
    depth/          # ScanNet GT depth (.npy)
    depth_pred/     # ZoeDepth predictions (16-bit PNG, millimetres)
    pose/
    intrinsic/
```

## References
1. ScanNet: Angela Dai, Angel X. Chang, Manolis Savva, Maciej Halber, Thomas Funkhouser, Matthias Nießner, "ScanNet: Richly-annotated 3D Reconstructions of Indoor Scenes", arXiv, 2017, url: https://arxiv.org/abs/1702.04405
2. DepthPro: Aleksei Bochkovskii, Amael Delaunoy, Hugo Germain, Marcel Santos, Yichao Zhou, Stephan R. Richter, Vladlen Koltun, "Depth Pro: Sharp Monocular Metric Depth in Less Than a Second", International Conference on Learning Representations, 2025, url: https://arxiv.org/abs/2410.02073
3. ZoeDepth: Bhat, Shariq Farooq and Birkl, Reiner and Wofk, Diana and Wonka, Peter and Müller, Matthias, "ZoeDepth: Zero-shot Transfer by Combining Relative and Metric Depth", arXiv, 2023, url: https://arxiv.org/abs/2302.12288
