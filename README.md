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

## Run scripts with metrics

Use python3 to run the utilities from the project root (expand ~ or use absolute paths).

1) Depth consistency check (per-RGB sample table)
- Script: scripts/depth_consistency.py
- Typical args: --rgb_dir, --gt_dir, --pred_dir

Example:

```bash
cd ~/Downloads/mono-depth-to-3DR
python3 scripts/depth_consistency.py \
  --rgb_dir /path/to/rgb/frame_00000.jpg \
  --gt_dir  /path/to/gt_depth/frame_00000.png \
  --pred_dir /path/to/pred_depth_/frame_00000.png
```

Notes:
- The script prints a per-image table and an average row.
- Make sure depth loader knows the PNG format (16-bit with scale 1000 → meters) if applicable.

2) Photometric consistency (SSIM + L2 using depth + poses + intrinsics)
- Script: scripts/photometric_consistency.py
- Typical args: --img1, --img2, --depth1, --depth2, --intrinsics_color, --pose1, --pose2, --depth_scale, --cam_to_world / --world_to_cam, --visualize.

Example (camera-to-world poses, depth in uint16 mm):

```bash
python3 scripts/photometric_consistency.py \
  --img1 /path/to/rgb/frame_00000.jpg \
  --img2 /path/to/rgb/frame_00001.jpg \
  --depth1 /path/to/depth/frame_00000.png \
  --depth2 /path/to/depth/frame_00001.png \
  --intrinsics_color /path/to/calib/color_intrinsics.txt \
  --pose1 /path/to/poses/pose_000000.txt \
  --pose2 /path/to/poses/pose_000001.txt \
  --depth_scale 1000.0 \
  --cam_to_world \
  --visualize
```
## Data

### ScanNet Temporal Dataset

`data/temporal_sampling.py` is the runnable sampling entry point. It can build batches either with `ScanNetTemporalDataset` or with the graph-based sampler from `data/graph_based_sampling.py`.

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
  sampler_type: "temporal"  # choose "temporal" or "graph"
  num_samples: 700       # number of samples per epoch (__len__)
  num_frames: 5          # exact number of frames per sample (seq_len)
  min_stride: 80          # minimum temporal stride between frames
  max_stride: 120         # maximum temporal stride between frames
  max_scenes: null       # limit scenes loaded; set to an integer to restrict (e.g. 5)

graph_sampling:
  graph_cache: null
  min_overlap: 0.2
  max_overlap: 0.9
  overlap_sample_step: 16
  depth_tolerance: 0.05

output:
  sample_output_dir: "datasets/sample_output"   # visualisation grids
  sampled_data_dir:  "datasets/sampled_data"    # individual saved frames
  batch_size: 2
  num_workers: 2
```

**Run data loader:**
```bash
python data/temporal_sampling.py
```
2. Graph-based sampling:
```bash
python data/graph_based_sampling.py
```

Prompts whether to run all batches or a fixed number, then writes two output folders:

| Folder | Contents |
|---|---|
| `datasets/sample_output/` | RGB and depth visualisation grids (PNG) |
| `datasets/sampled_data/` | Individual frames per sequence with structure: 
```
batch1/
    sample1/
      scene0000_00/
        color/
        depth/
        pose/
        intrinsic/
    sample2/
      scene0079_01/
        ...
  batch2/
    ...
```
|

## Monocular Depth Inference

### ZoeDepth

`model/zoe_depth.py` runs ZoeDepth inference on all sequences produced by the samplers.

**Prerequisites:** clone [ZoeDepth](https://github.com/isl-org/ZoeDepth) on the workstation and activate its environment.

**Run** (from the ZoeDepth root so `source="local"` resolves correctly):
```bash
cd ~/ZoeDepth
export PYTHONPATH=/usr/prakt/s0014/ZoeDepth:$PYTHONPATH
python /usr/prakt/s0014/mono-depth-to-3DR/models/zoe_depth.py
```
### Depth Pro 

`model/depth-pro.py` runs Depth Pro inference on all sequences produced by the samplers.

**Prerequisites:** clone [Depth Pro](https://github.com/apple/ml-depth-pro) on the workstation and activate its environment.

**Run** (from the Depth Pro root so `source="local"` resolves correctly):
```bash
cd ~/ml-depth-pro
python ~/mono-depth-to-3DR/models/depth-pro.py
```

```
datasets/sampled_data/
  <seq_name>/
    color/          # input RGB frames
    depth/          # ScanNet GT depth (.npy)
    zoe-depth_pred/     # ZoeDepth predictions (16-bit PNG, millimetres)
    depth-pro_pred/     # Depth Pro predictions (npz files, meters)
    pose/
    intrinsic/
```


## References
1. ScanNet: Angela Dai, Angel X. Chang, Manolis Savva, Maciej Halber, Thomas Funkhouser, Matthias Nießner, "ScanNet: Richly-annotated 3D Reconstructions of Indoor Scenes", arXiv, 2017, url: https://arxiv.org/abs/1702.04405
2. DepthPro: Aleksei Bochkovskii, Amael Delaunoy, Hugo Germain, Marcel Santos, Yichao Zhou, Stephan R. Richter, Vladlen Koltun, "Depth Pro: Sharp Monocular Metric Depth in Less Than a Second", International Conference on Learning Representations, 2025, url: https://arxiv.org/abs/2410.02073
3. ZoeDepth: Bhat, Shariq Farooq and Birkl, Reiner and Wofk, Diana and Wonka, Peter and Müller, Matthias, "ZoeDepth: Zero-shot Transfer by Combining Relative and Metric Depth", arXiv, 2023, url: https://arxiv.org/abs/2302.12288
