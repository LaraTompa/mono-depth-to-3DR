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

**Run data loader:**
```bash
python data/temporal_sampling.py
```

Prompts whether to run all batches or a fixed number, then writes two output folders:

| Folder | Contents |
|---|---|
| `datasets/sample_output/` | RGB and depth visualisation grids (PNG) |
| `datasets/sampled_data/` | Individual frames per sequence — `color/*.png`, `depth/*.npy`, `pose/*.txt`, and `intrinsics/intrinsics_color.txt`, `intrinsics/intrinsics_depth.txt` |

## References
1. ScanNet: Angela Dai, Angel X. Chang, Manolis Savva, Maciej Halber, Thomas Funkhouser, Matthias Nießner, "ScanNet: Richly-annotated 3D Reconstructions of Indoor Scenes", arXiv, 2017, url: https://arxiv.org/abs/1702.04405
