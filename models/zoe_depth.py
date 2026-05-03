import torch
import os
from PIL import Image
import numpy as np
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

if not seq_dirs:
    raise RuntimeError(f"No sequence folders found under {SAMPLED_DATA_DIR}")

print(f"Found {len(seq_dirs)} sequence(s) under {SAMPLED_DATA_DIR}\n")

for seq_dir in seq_dirs:
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

    print(f"  Sequence: {os.path.basename(seq_dir)}  ({len(image_files)} frames)")

    for fname in image_files:
        img_path = os.path.join(color_dir, fname)
        img = Image.open(img_path).convert("RGB")

        with torch.no_grad():
            depth = model.infer_pil(img)  # numpy float32, metres

        # Save as 16-bit PNG (millimetres) — consistent with ScanNet GT format
        out_path = os.path.join(pred_dir, os.path.splitext(fname)[0] + ".png")
        save_raw_16bit(depth, out_path)

    print(f"    → saved predicted depths to {pred_dir}/")

print("\nDone.")