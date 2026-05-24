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
- Script: metrics/depth_consistency.py
- Typical args: --rgb_dir, --gt_dir, --pred_dir

Example:

```bash
cd ~/Downloads/mono-depth-to-3DR
python3 metrics/depth_consistency.py \
  --rgb_dir /path/to/rgb/frame_00000.jpg \
  --gt_dir  /path/to/gt_depth/frame_00000.png \
  --pred_dir /path/to/pred_depth_/frame_00000.png
```

Notes:
- The script prints a per-image table and an average row.
- Make sure depth loader knows the PNG format (16-bit with scale 1000 → meters) if applicable.

2) Photometric consistency (SSIM + L2 using depth + poses + intrinsics)
- Script: metrics/photometric_consistency.py
- Typical args: --img1, --img2, --depth1, --depth2, --intrinsics_color, --pose1, --pose2, --depth_scale, --cam_to_world / --world_to_cam, --visualize.

Example (camera-to-world poses, depth in uint16 mm):

```bash
python3 metrics/photometric_consistency.py \
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

3) Pixel-wise reprojection consistency
- Script: metrics/pixel_consistency.py
- Typical args: --gt_depth1, --gt_depth2, --pred_depth1, --pred_depth2, --intrinsics_color, --pose1, --pose2, --depth_scale_gt, --depth_scale_pred, --cam_to_world / --world_to_cam.
Example:

```bash
python3 metrics/pixel_consistency.py \
  --gt_depth1 /path/to/gt_depth/frame_00000.png \
  --gt_depth2 /path/to/gt_depth/frame_00001.png \
  --pred_depth1 /path/to/pred_depth/frame_00000.png \
  --pred_depth2 /path/to/pred_depth/frame_00001.png \
  --intrinsics_color /path/to/calib/color_intrinsics.txt \
  --pose1 /path/to/poses/pose_000000.txt \
  --pose2 /path/to/poses/pose_000001.txt \
  --depth_scale_gt 1.0 \
  --depth_scale_pred 1.0 \
  --cam_to_world
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

## Batch Evaluation

`scripts/batch_eval.py` evaluates depth consistency and photometric consistency across all sampled scenes in batch.

**Typical usage:**

```bash
python3 scripts/batch_eval.py \
  --sampled_data_dir datasets/sampled_data \
  --output_dir results/batch_eval \
  --model depth-pro \
  --rgb_ext png \
  --depth_ext npz \
  --pose_ext txt \
  --depth_scale 1.0 \
  --cam_to_world \
  --window 1 \
  --max_batches 10
```

**Arguments:**

| Argument | Description | Default |
|---|---|---|
| `--sampled_data_dir` | Root directory containing `batch*/sample*/scene*/` folders | `datasets/sampled_data` |
| `--output_dir` | Directory to save CSV results and visualizations | `results/batch_eval` |
| `--model` | Depth prediction model (`depth-pro` or `zoe-depth`) | `depth-pro` |
| `--rgb_ext` | RGB image extension | `jpg` |
| `--depth_ext` | Depth file extension (`npz`, `png`, `npy`) | `npz` |
| `--pose_ext` | Pose file extension | `txt` |
| `--depth_scale` | Scale factor for depth values (e.g., 1000 for mm→m) | `1.0` |
| `--cam_to_world` | Flag: poses are camera-to-world (default for ScanNet) | `False` |
| `--window` | Temporal window for photometric pairs (1 = adjacent frames only) | `1` |
| `--max_batches` | Limit number of batches to evaluate (None = all) | `None` |
| `--debug` | Print verbose subprocess output when parsing fails | `False` |

**Output:**

| File | Contents |
|---|---|
| `photometric_pairs_detailed.csv` | Per-pair SSIM and L2 metrics for all photometric consistency checks |
| `scene_metrics_summary.csv` | Per-scene aggregated metrics: depth consistency (RMSE, MAE, δ¹, etc.) and photometric statistics (mean SSIM, L2) |
| `photometric_boxplots.png` | Box plots of SSIM and L2 distributions across all pairs |
| `depth_consistency_boxplots.png` | Box plots of depth metrics (RMSE, MAE, AbsRel, δ¹) across all scenes |
| `overall_statistics.txt` | Summary statistics (mean, std, median, min, max) for all metrics |

**Example with ZoeDepth predictions (16-bit PNG in millimetres):**

```bash
python3 scripts/batch_eval.py \
  --sampled_data_dir datasets/sampled_data \
  --output_dir results/batch_eval_zoe \
  --model zoe-depth \
  --rgb_ext png \
  --depth_ext png \
  --pose_ext txt \
  --depth_scale 1000.0 \
  --cam_to_world \
  --window 1 \
  --max_batches 50 \
```
## DepthAlignNet

A lightweight network that takes monocular depth predictions and stereo RGB pairs and produces **multiview-consistent aligned depths** together with **predicted camera intrinsics and relative pose** — no GT camera parameters required at test time.

### Architecture

```
RGB + mono-depth (×2 views)
        │
        ▼
SharedEncoder  (ConvNeXt-Tiny, weight-tied across views)
  → s4  (H/4,  W/4,  C)
  → s8  (H/8,  W/8,  C)
  → s16 (H/16, W/16, C)
        │
        ▼  (called twice: v1→v2 and v2→v1)
Cross-Attention  (single shared MultiheadAttention)
  s16: [camera_token | s16_flat] attends to other view's [camera_token | s16_flat]
       ├─ token at pos 0  →  CameraHead  (K, T_c2w, log-confidence)
       └─ tokens at pos 1: →  s16 spatial features  (decoder path)
  s8:  s8_flat attends to other view's s8_flat  (same shared weights)
        │
        ▼
DepthDecoder  (FPN-style, s4 + s8 + s16 → H/2, W/2)
  → aligned depth  (B, 1, H/2, W/2)
  → confidence     (B, 1, H/2, W/2)
```

**Camera token** a single learnable token is prepended to each view's s16 feature sequence before cross-attention. After attending to the other view's full sequence, the token at position 0 is decoded by the shared `CameraHead` into intrinsics K and camera-to-world pose T_c2w. Both views share the token parameter and the head weights — differentiation comes entirely from the attended context.

**Iterative camera initialisation** (training): K is initialised from a focal-length prior and T_12 to identity. The network runs for `num_pose_iters` iterations, feeding its own predictions back as the next iteration's pose prior.

**Heteroscedastic confidence:** the camera head outputs log-confidence scalars for intrinsics and pose. Losses are weighted by `exp(-s)·L + s` (Kendall & Gal 2017), letting the network learn its own uncertainty.

### Configuration

`config/arch.yaml` controls all architecture hyper-parameters:

```yaml
encoder:
  backbone: "convnext_tiny"
  pretrained: true
  freeze_backbone: true
  out_channels: 64          # feat_dim C

attention:
  num_heads: 4

refinement:
  enabled: false            # set true to add IterativeRefinement at s16
  num_iters: 4
  hidden_dim: 64

decoder:
  hidden_dim: 32
```

### Training

```bash
python training/train.py
python training/train.py --config config/training.yaml
python training/train.py --config config/training.yaml --resume checkpoints/last.pt
```

Training hyper-parameters live in `config/training.yaml`. Checkpoints are saved to `checkpoints/` (`best.pt` and `last.pt`).

### Smoke test

Verifies all module shapes and that gradients flow end-to-end (CPU, no data required):

```bash
python scripts/smoke_test.py
```

## DepthAlignNet

A lightweight network that takes monocular depth predictions and stereo RGB pairs and produces **multiview-consistent aligned depths** together with **predicted camera intrinsics and relative pose** — no GT camera parameters required at test time.

### Architecture

```
RGB + mono-depth (×2 views)
        │
        ▼
SharedEncoder  (ConvNeXt-Tiny, weight-tied across views)
  → s4  (H/4,  W/4,  C)
  → s8  (H/8,  W/8,  C)
  → s16 (H/16, W/16, C)
        │
        ▼  (called twice: v1→v2 and v2→v1)
Cross-Attention  (single shared MultiheadAttention)
  s16: [camera_token | s16_flat] attends to other view's [camera_token | s16_flat]
       ├─ token at pos 0  →  CameraHead  (K, T_c2w, log-confidence)
       └─ tokens at pos 1: →  s16 spatial features  (decoder path)
  s8:  s8_flat attends to other view's s8_flat  (same shared weights)
        │
        ▼
DepthDecoder  (FPN-style, s4 + s8 + s16 → H/2, W/2)
  → aligned depth  (B, 1, H/2, W/2)
  → confidence     (B, 1, H/2, W/2)
```

**Camera token (ViSTA-SLAM style):** a single learnable token is prepended to each view's s16 feature sequence before cross-attention. After attending to the other view's full sequence, the token at position 0 is decoded by the shared `CameraHead` into intrinsics K and camera-to-world pose T_c2w. Both views share the token parameter and the head weights — differentiation comes entirely from the attended context.

**Iterative camera initialisation** (training): K is initialised from a focal-length prior and T_12 to identity. The network runs for `num_pose_iters` iterations, feeding its own predictions back as the next iteration's pose prior.

**Heteroscedastic confidence:** the camera head outputs log-confidence scalars for intrinsics and pose. Losses are weighted by `exp(-s)·L + s` (Kendall & Gal 2017), letting the network learn its own uncertainty.

### Configuration

`config/arch.yaml` controls all architecture hyper-parameters:

```yaml
encoder:
  backbone: "convnext_tiny"
  pretrained: true
  freeze_backbone: true
  out_channels: 64          # feat_dim C

attention:
  num_heads: 4

refinement:
  enabled: false            # set true to add IterativeRefinement at s16
  num_iters: 4
  hidden_dim: 64

decoder:
  hidden_dim: 32
```

### Training

```bash
python training/train.py
python training/train.py --config config/training.yaml
python training/train.py --config config/training.yaml --resume checkpoints/last.pt
```

Training hyper-parameters live in `config/training.yaml`. Checkpoints are saved to `checkpoints/` (`best.pt` and `last.pt`).

### Smoke test

Verifies all module shapes and that gradients flow end-to-end (CPU, no data required):

```bash
python scripts/smoke_test.py
```

## References

1. ScanNet: Angela Dai, Angel X. Chang, Manolis Savva, Maciej Halber, Thomas Funkhouser, Matthias Nießner, "ScanNet: Richly-annotated 3D Reconstructions of Indoor Scenes", arXiv, 2017, url: https://arxiv.org/abs/1702.04405
2. DepthPro: Aleksei Bochkovskii, Amael Delaunoy, Hugo Germain, Marcel Santos, Yichao Zhou, Stephan R. Richter, Vladlen Koltun, "Depth Pro: Sharp Monocular Metric Depth in Less Than a Second", International Conference on Learning Representations, 2025, url: https://arxiv.org/abs/2410.02073
3. ZoeDepth: Bhat, Shariq Farooq and Birkl, Reiner and Wofk, Diana and Wonka, Peter and Müller, Matthias, "ZoeDepth: Zero-shot Transfer by Combining Relative and Metric Depth", arXiv, 2023, url: https://arxiv.org/abs/2302.12288
