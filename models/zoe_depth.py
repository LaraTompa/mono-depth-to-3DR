import torch
import os
from PIL import Image
import numpy as np
import time
from zoedepth.utils.misc import save_raw_16bit

# ======================
# Output: depth_pred/{fid}.png (16-bit, millimetres) written next to each color/
# ======================
SAMPLED_DATA_DIR = os.path.expanduser("~/mono-depth-to-3DR/datasets/sampled_data")

# ======================
# DEVICE
# ======================
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# ======================
# LOAD MODEL
# ZoeD_NK works well for mixed indoor/outdoor; ScanNet is indoor so ZoeD_N
# is also a valid choice if you want indoor-only specialisation.
# ======================
model = torch.hub.load(".", "ZoeD_NK", source="local", pretrained=True)
model = model.to(device)
model.eval()


def format_seconds(seconds):
    if seconds is None or seconds < 0:
        return "n/a"
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"

# ======================
# WALK SAMPLED SEQUENCES AND RUN INFERENCE
# Layout: sampled_data/batch<N>/sample<M>/<scene_name>/color/
# ======================
seq_dirs = sorted([
    os.path.join(SAMPLED_DATA_DIR, batch, sample, scene)
    for batch in sorted(os.listdir(SAMPLED_DATA_DIR))
    if os.path.isdir(os.path.join(SAMPLED_DATA_DIR, batch))
    for sample in sorted(os.listdir(os.path.join(SAMPLED_DATA_DIR, batch)))
    if os.path.isdir(os.path.join(SAMPLED_DATA_DIR, batch, sample))
    for scene in sorted(os.listdir(os.path.join(SAMPLED_DATA_DIR, batch, sample)))
    if os.path.isdir(os.path.join(SAMPLED_DATA_DIR, batch, sample, scene))
])

if not os.path.isdir(SAMPLED_DATA_DIR):
    raise RuntimeError(f"SAMPLED_DATA_DIR does not exist: {SAMPLED_DATA_DIR}")

if not seq_dirs:
    raise RuntimeError(f"No sequence folders found under {SAMPLED_DATA_DIR}")

print(f"Found {len(seq_dirs)} sequence(s) under {SAMPLED_DATA_DIR}\n")

total_frames = 0
for seq_dir in seq_dirs:
    color_dir = os.path.join(seq_dir, "color")
    if not os.path.isdir(color_dir):
        continue
    total_frames += len([
        f for f in os.listdir(color_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ])

if total_frames == 0:
    raise RuntimeError("No image files found under any color/ folders.")

print(f"Total frames to process: {total_frames}")

global_start = time.time()
processed_frames = 0

for seq_idx, seq_dir in enumerate(seq_dirs, start=1):
    color_dir = os.path.join(seq_dir, "color")
    pred_dir  = os.path.join(seq_dir, "zoe-depth_pred")

    if not os.path.isdir(color_dir):
        print(f"  [skip] no color/ in {seq_dir}")
        continue

    os.makedirs(pred_dir, exist_ok=True)

    image_files = sorted([
        f for f in os.listdir(color_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ])

    if len(image_files) == 0:
        print(f"  [skip] no images in {color_dir}")
        continue

    print(
        f"\n[{seq_idx}/{len(seq_dirs)}] Sequence: {os.path.basename(seq_dir)} "
        f"({len(image_files)} frames)"
    )

    seq_start = time.time()

    for frame_idx, fname in enumerate(image_files, start=1):
        img_path = os.path.join(color_dir, fname)
        img = Image.open(img_path).convert("RGB")

        with torch.no_grad():
            depth = model.infer_pil(img)  # numpy float32, metres

        # Save as 16-bit PNG (millimetres) — consistent with ScanNet GT format
        out_path = os.path.join(pred_dir, os.path.splitext(fname)[0] + ".png")
        save_raw_16bit(depth, out_path)

        processed_frames += 1
        elapsed = time.time() - global_start
        avg_per_frame = elapsed / processed_frames if processed_frames > 0 else None
        remaining = (total_frames - processed_frames) * avg_per_frame if avg_per_frame else None

        print(
            f"    frame {frame_idx}/{len(image_files)} | "
            f"global {processed_frames}/{total_frames} | "
            f"elapsed {format_seconds(elapsed)} | "
            f"eta {format_seconds(remaining)}"
        )

    seq_elapsed = time.time() - seq_start
    print(f"    -> saved predicted depths to {pred_dir}/")
    print(f"    -> sequence time: {format_seconds(seq_elapsed)}")

total_elapsed = time.time() - global_start
print(f"\nDone. Processed {processed_frames}/{total_frames} frames in {format_seconds(total_elapsed)}.")