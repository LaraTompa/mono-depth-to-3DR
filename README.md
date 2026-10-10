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

### Training

```bash
python training/train.py
python training/train.py --config config/training.yaml --arch config/arch/depth_only.yaml
python training/train.py --config config/training.yaml --arch config/arch/depth_only.yaml --resume checkpoints/last.pt
```

Training hyper-parameters live in `config/training.yaml`. Checkpoints are saved to `checkpoints/` (`best.pt` and `last.pt`).


## Architectures

All variants take a *pair* of monocular depth predictions (e.g. Depth Pro / ZoeDepth) and output aligned depth for both views (plus relative pose where enabled), trained for multiview consistency.

- `models/model_image_depth` : **Mono3DR v1**, DepthAlignNet. Image + monocular depth inputs, ConvNeXt-Tiny FPN encoder, windowed cross-attention, camera head, optional iterative refinement. Outputs a depth map. Config: `config/arch/v1.yaml`.
- `models/model_depth_only`  : **Mono3DR v2–v4**, DepthOnlyNet. Median-normalised depth pairs go through a shared ConvNeXt-Tiny depth encoder, then MASt3R-initialised symmetric cross-attention blocks (768-d tokens), then a fusion decoder with skip connections. Config: `config/arch/depth_only.yaml`. Two switches select the variant:
  - `use_image_encoder`: adds a frozen MASt3R ViT-Large RGB encoder whose tokens are summed onto the depth tokens (zero-initialised projection, so training starts identical to the depth-only model).
  - `predict_depth_map`: `true` predicts a 1-channel depth map (SI-log + MSE losses); `false` predicts a 3-channel XYZ point map in the view-1 camera frame (DUSt3R-style, `point_map_loss`).
- `models/model_vista`       : ViSTA-style variant (DINOv2 ViT-L/14 RGB encoder + MASt3R cross-attention + ConvNeXt depth stream; experimental). Config: `config/arch/vista.yaml`.

| Version | Output | RGB input | Model | Key config |
|---|---|---|---|---|
| Mono3DR v1 | depth | yes | `model_image_depth` | `config/arch/v1.yaml` |
| Mono3DR v2 | pointmap | yes | `model_depth_only` | `use_image_encoder: true`, `predict_depth_map: false` |
| Mono3DR v3 | depth | no | `model_depth_only` | `use_image_encoder: false`, `predict_depth_map: true` |
| Mono3DR v4 | pointmap | no | `model_depth_only` | `use_image_encoder: false`, `predict_depth_map: false` |

Select variant via the saved checkpoint `arch` field or architecture configs in `config/arch/`.

## Results

Evaluated with `scripts/test_set_eval.py`. Metrics are averaged over 3315 evaluated frame pairs. *Monocular Baseline (Scaled)* applies a median scale to the raw monocular depth using the GT depth. Best Mono3DR variant: **v1 (depth + image)**.

| Method | RMSE ↓ | Abs Rel ↓ | PC RMSE (px) ↓ | Geo L2 (m) ↓ | SSIM ↑ |
|---|---|---|---|---|---|
| Monocular Baseline | 1.4235 | 0.7112 | 193.02 | 1.8764 | 0.4635 |
| Monocular Baseline (Scaled) | 0.1938 | 0.0708 | 31.88 | 0.2151 | 0.6391 |
| **Mono3DR v1 (depth + image)** | 0.2698 | 0.1352 | 44.95 | 0.2866 | 0.6249 |
| Mono3DR v2 (pointmap + image) | 0.2813 | 0.1299 | 58.83 | 0.3440 | 0.5894 |
| Mono3DR v3 (depth) | 0.3098 | 0.1575 | 56.17 | 0.3356 | 0.5894 |
| Mono3DR v4 (pointmap) | 0.3259 | 0.1616 | 61.15 | 0.3839 | 0.5925 |

## Losses (training/losses.py)

Key active losses used during training:

- `si_log_loss`: scale-invariant log depth supervision (supervised depth term).
- `smooth_loss`: edge-aware depth smoothness (uses RGB edges to weight smoothing).
- `iter_supervision_loss`: deep supervision over refinement iterations.
- `camera_pose_loss`: composite pose + intrinsics loss with optional heteroscedastic confidence (Kendall & Gal 2017).
- `pose_identity_loss`: round-trip consistency penalty for predicted poses.
- `pixel_consistency_loss`: multiview 2D reprojection (pixel) consistency.
- `geometric_consistency_loss`: 3-D geometric consistency (ViSTA-SLAM Lgc), enabled after warmup epochs.
- `total_loss`: weighted sum that composes the above (see `training/losses.py` for weight names and schedule).

Note: `photometric_loss` and `depth_consistency` implementations exist but are commented out (re-enable when network depths are stable).

## Evaluation: scripts/model_eval.py

Quick usage (from repository root). Minimal run (no GT):

```bash
python scripts/model_eval.py \
  --pred_depth1 /path/to/pred1.npz \
  --pred_depth2 /path/to/pred2.npz \
  --checkpoint checkpoints/best.pt \
  --output_dir results/eval_minimal
```

Full evaluation (with GT depths, intrinsics and poses):

```bash
python scripts/model_eval.py \
  --img1 /path/to/img1.jpg \
  --img2 /path/to/img2.jpg \
  --pred_depth1 /path/to/pred1.npz \
  --pred_depth2 /path/to/pred2.npz \
  --gt_depth1 /path/to/gt1.png \
  --gt_depth2 /path/to/gt2.png \
  --intrinsics /path/to/intrinsic.txt \
  --pose1 /path/to/pose1.txt \
  --pose2 /path/to/pose2.txt \
  --checkpoint checkpoints/best.pt \
  --output_dir results/eval_full \
  --depth_scale_gt 1000.0
```

Defaults: `--depth_scale_pred` is `1.0` (metres); use `--depth_scale_gt 1000.0` for ScanNet PNGs in millimetres.




## References

1. ScanNet: Angela Dai, Angel X. Chang, Manolis Savva, Maciej Halber, Thomas Funkhouser, Matthias Nießner, "ScanNet: Richly-annotated 3D Reconstructions of Indoor Scenes", arXiv, 2017, url: https://arxiv.org/abs/1702.04405
2. DepthPro: Aleksei Bochkovskii, Amael Delaunoy, Hugo Germain, Marcel Santos, Yichao Zhou, Stephan R. Richter, Vladlen Koltun, "Depth Pro: Sharp Monocular Metric Depth in Less Than a Second", International Conference on Learning Representations, 2025, url: https://arxiv.org/abs/2410.02073
3. ZoeDepth: Bhat, Shariq Farooq and Birkl, Reiner and Wofk, Diana and Wonka, Peter and Müller, Matthias, "ZoeDepth: Zero-shot Transfer by Combining Relative and Metric Depth", arXiv, 2023, url: https://arxiv.org/abs/2302.12288
4. MASt3R: Vincent Leroy, Yohann Cabon, Jérôme Revaud, "Grounding Image Matching in 3D with MASt3R", arXiv, 2024, url: https://arxiv.org/abs/2406.09756
5. ViSTA-SLAM: Ganlin Zhang, Shenhan Qian, Xi Wang, Daniel Cremers, "ViSTA-SLAM: Visual SLAM with Symmetric Two-view Association", 3DV, 2026, arXiv: https://arxiv.org/abs/2509.01584
6. ConvNeXt: Zhuang Liu, Hanzi Mao, Chao-Yuan Wu, Christoph Feichtenhofer, Trevor Darrell, Saining Xie, "A ConvNet for the 2020s", CVPR, 2022, arXiv: https://arxiv.org/abs/2201.03545
7. MoGe: Ruicheng Wang, Sicheng Xu, Cassie Dai, Jianfeng Xiang, Yu Deng, Xin Tong, Jiaolong Yang, "MoGe: Unlocking Accurate Monocular Geometry Estimation for Open-Domain Images with Optimal Training Supervision", CVPR, 2025, url: https://arxiv.org/abs/2410.19115
8. Metric3D: Wei Yin, Chi Zhang, Hao Chen, Zhipeng Cai, Gang Yu, Kaixuan Wang, Xiaozhi Chen, Chunhua Shen, "Metric3D: Towards Zero-shot Metric 3D Prediction from A Single Image", ICCV, 2023, url: https://arxiv.org/abs/2307.10984
9. Metric3D v2: Mu Hu, Wei Yin, Chi Zhang, Zhipeng Cai, Xiaoxiao Long, Hao Chen, Kaixuan Wang, Gang Yu, Chunhua Shen, Shaojie Shen, "Metric3D v2: A Versatile Monocular Geometric Foundation Model for Zero-shot Metric Depth and Surface Normal Estimation", IEEE TPAMI, 2024, url: https://arxiv.org/abs/2404.15506
10. MiDaS: René Ranftl, Katrin Lasinger, David Hafner, Konrad Schindler, Vladlen Koltun, "Towards Robust Monocular Depth Estimation: Mixing Datasets for Zero-Shot Cross-Dataset Transfer", IEEE TPAMI, 2022, url: https://arxiv.org/abs/1907.01341
11. VGGT: Jianyuan Wang, Minghao Chen, Nikita Karaev, Andrea Vedaldi, Christian Rupprecht, David Novotny, "VGGT: Visual Geometry Grounded Transformer", CVPR, 2025, url: https://arxiv.org/abs/2503.11651
12. MUSt3R: Yohann Cabon, Lucas Stoffl, Leonid Antsfeld, Gabriela Csurka, Boris Chidlovskii, Jerome Revaud, Vincent Leroy, "MUSt3R: Multi-view Network for Stereo 3D Reconstruction", arXiv, 2025, url: https://arxiv.org/abs/2503.01661
13. DUSt3R: Shuzhe Wang, Vincent Leroy, Yohann Cabon, Boris Chidlovskii, Jerome Revaud, "DUSt3R: Geometric 3D Vision Made Easy", CVPR, 2024, url: https://arxiv.org/abs/2312.14132
14. Depth Anything 3: Haotong Lin, Sili Chen, Jun Hao Liew, Donny Y. Chen, Zhenyu Li, Guang Shi, Jiashi Feng, Bingyi Kang, "Depth Anything 3: Recovering the Visual Space from Any Views", arXiv, 2025, url: https://arxiv.org/abs/2511.10647
15. LocalBins: Shariq Farooq Bhat, Ibraheem Alhashim, Peter Wonka, "LocalBins: Improving Depth Estimation by Learning Local Distributions", arXiv, 2022, url: https://arxiv.org/abs/2203.15132
16. NeWCRFs: Weihao Yuan, Xiaodong Gu, Zuozhuo Dai, Siyu Zhu, Ping Tan, "NeWCRFs: Neural Window Fully-connected CRFs for Monocular Depth Estimation", CVPR, 2022, url: https://arxiv.org/abs/2203.01502
