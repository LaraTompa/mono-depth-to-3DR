"""
model_image_depth.py — End-to-end runnable demo of the DepthAlignNet pipeline.

Exercises every module and geometry utility individually, then runs the full
forward pass through DepthAlignNet.  No real data required — everything uses
synthetic random tensors.

Run:
    python model_image_depth.py           # CPU
    python model_image_depth.py --cuda    # GPU (if available)
"""

import argparse
import os
import sys

# Force UTF-8 stdout so box-drawing characters render correctly on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

import torch
import torch.nn.functional as F

# Make sure repo root is importable regardless of cwd
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

# ── geometry utilities ───────────────────────────────────────────────────────
from models.model_image_depth.geometry import (
    make_pixel_grid,
    unproject,
    project,
    transform_pts,
    normalise_coords,
    warp,
    reprojection_coords,
    rot_to_6d,
)

# ── modules ──────────────────────────────────────────────────────────────────
from models.model_image_depth.encoder    import SharedEncoder
from models.model_image_depth.attention  import PoseEncoder, LocalGeoCrossAttention
from models.model_image_depth.refinement import correlation_feature, ConvGRU, RefinementHead, IterativeRefinement
from models.model_image_depth.decoder    import DepthDecoder
from models.model_image_depth.network    import DepthAlignNet


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def section(title: str) -> None:
    bar = "─" * 60
    print(f"\n{bar}")
    print(f"  {title}")
    print(bar)


def shape(label: str, t) -> None:
    if isinstance(t, torch.Tensor):
        print(f"  {label:<35} {tuple(t.shape)}")
    elif isinstance(t, list):
        print(f"  {label:<35} list of {len(t)} × {tuple(t[0].shape)}")
    else:
        print(f"  {label:<35} {t}")


def stat(label: str, t: torch.Tensor, fmt: str = ".3f") -> None:
    """Print min/mean/max; silently skip if tensor has no elements."""
    if t.numel() == 0:
        print(f"  {label:<35} (empty — batch=0)")
        return
    lo  = t.min().item()
    hi  = t.max().item()
    avg = t.float().mean().item()
    print(f"  {label:<35} min={lo:{fmt}}  mean={avg:{fmt}}  max={hi:{fmt}}")





# ─────────────────────────────────────────────────────────────────────────────
# Synthetic inputs
# ─────────────────────────────────────────────────────────────────────────────

def make_inputs(B: int, H: int, W: int, device: torch.device):
    """Return a dict of every tensor the pipeline uses."""
    rgb1        = torch.rand(B, 3, H, W, device=device)
    rgb2        = torch.rand(B, 3, H, W, device=device)
    depth_mono1 = torch.rand(B, 1, H, W, device=device) * 4.0 + 0.5   # 0.5–4.5 m
    depth_mono2 = torch.rand(B, 1, H, W, device=device) * 4.0 + 0.5

    # Small random rotation + translation for a realistic relative pose
    angle = 0.05   # ~3 degrees
    R = torch.tensor([
        [1,      0,           0     ],
        [0,  torch.cos(torch.tensor(angle)), -torch.sin(torch.tensor(angle))],
        [0,  torch.sin(torch.tensor(angle)),  torch.cos(torch.tensor(angle))],
    ]).float()
    t = torch.tensor([0.1, 0.0, 0.0])
    T_12 = torch.eye(4)
    T_12[:3, :3] = R
    T_12[:3,  3] = t
    T_12 = T_12.unsqueeze(0).expand(B, -1, -1).to(device)

    fx = 577.0
    K = torch.tensor([
        [fx,  0.0, W / 2.0],
        [0.0, fx,  H / 2.0],
        [0.0, 0.0, 1.0    ],
    ]).float().unsqueeze(0).expand(B, -1, -1).to(device)

    return dict(
        rgb1=rgb1, rgb2=rgb2,
        depth_mono1=depth_mono1, depth_mono2=depth_mono2,
        T_12=T_12, K=K,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 1. Geometry utilities
# ─────────────────────────────────────────────────────────────────────────────

def demo_geometry(inp: dict) -> None:
    section("1. Geometry utilities  (models/geometry.py)")

    B  = inp["depth_mono1"].shape[0]
    H  = inp["depth_mono1"].shape[-2]
    W  = inp["depth_mono1"].shape[-1]
    d  = inp["depth_mono1"]
    K  = inp["K"]
    T  = inp["T_12"]
    dev = d.device

    # make_pixel_grid
    grid = make_pixel_grid(H, W, dev)
    shape("make_pixel_grid(H,W)", grid)

    # rot_to_6d
    R6d = rot_to_6d(T[:, :3, :3])
    shape("rot_to_6d(R)", R6d)

    # unproject
    pts3d = unproject(d, K)
    shape("unproject(depth, K)", pts3d)

    # transform_pts
    pts3d_2 = transform_pts(pts3d, T)
    shape("transform_pts(pts, T_12)", pts3d_2)

    # project
    coords, z = project(pts3d_2, K)
    shape("project → coords (B,H,W,2)", coords)
    shape("project → z     (B,1,H,W)", z)

    # normalise_coords
    coords_n = normalise_coords(coords, H, W)
    shape("normalise_coords", coords_n)

    # reprojection_coords
    rp_coords, rp_valid = reprojection_coords(d, T, K)
    shape("reprojection_coords → coords", rp_coords)
    shape("reprojection_coords → valid ", rp_valid)

    # warp
    feat_dummy = torch.rand(B, 3, H, W, device=dev)
    warped, valid = warp(feat_dummy, d, T, K)
    shape("warp → warped", warped)
    shape("warp → valid ", valid)
    stat("valid pixel ratio", valid.float())


# ─────────────────────────────────────────────────────────────────────────────
# 2. Shared encoder
# ─────────────────────────────────────────────────────────────────────────────

def demo_encoder(inp: dict) -> SharedEncoder:
    section("2. Shared encoder  (models/encoder.py)")

    encoder = SharedEncoder(pretrained=False, out_channels=128).to(inp["rgb1"].device)
    encoder.eval()

    x = torch.cat([inp["rgb1"], inp["depth_mono1"]], dim=1)   # (B, 4, H, W)
    shape("encoder input (RGB+depth)", x)

    with torch.no_grad():
        feats = encoder(x)

    for scale, feat in feats.items():
        shape(f"features['{scale}']", feat)

    total = sum(p.numel() for p in encoder.parameters())
    print(f"  {'encoder parameters':<35} {total / 1e6:.2f} M")

    return encoder


# ─────────────────────────────────────────────────────────────────────────────
# 3. Pose encoder
# ─────────────────────────────────────────────────────────────────────────────

def demo_pose_encoder(inp: dict) -> None:
    section("3. Pose encoder / FiLM  (models/attention.py :: PoseEncoder)")

    B   = inp["T_12"].shape[0]
    dev = inp["T_12"].device
    pe  = PoseEncoder(embed_dim=128).to(dev)
    pe.eval()

    with torch.no_grad():
        gk, bk, gv, bv = pe(inp["T_12"])

    shape("gamma_k  (B, embed_dim)", gk)
    shape("beta_k   (B, embed_dim)", bk)
    shape("gamma_v  (B, embed_dim)", gv)
    shape("beta_v   (B, embed_dim)", bv)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Local geometry-aware cross-attention
# ─────────────────────────────────────────────────────────────────────────────

def demo_attention(inp: dict, encoder: SharedEncoder) -> None:
    section("4. Local geometry-aware cross-attention  (models/attention.py)")

    dev = inp["rgb1"].device
    B, _, H, W = inp["rgb1"].shape

    attn = LocalGeoCrossAttention(dim=128, num_heads=4, window_size=7).to(dev)
    attn.eval()

    # Encode both views
    x1 = torch.cat([inp["rgb1"], inp["depth_mono1"]], dim=1)
    x2 = torch.cat([inp["rgb2"], inp["depth_mono2"]], dim=1)
    with torch.no_grad():
        f1 = encoder(x1)
        f2 = encoder(x2)

    # Scale K and depth to s16 resolution
    H16, W16 = f1["s16"].shape[-2:]
    K_s16 = DepthAlignNet._scale_K(inp["K"], H, W, H16, W16)
    d1_s16 = F.interpolate(inp["depth_mono1"], size=(H16, W16), mode="nearest")

    shape("query features  s16 (B,C,H/16,W/16)", f1["s16"])
    shape("context features s16", f2["s16"])
    shape("depth at s16", d1_s16)

    with torch.no_grad():
        out = attn(f1["s16"], f2["s16"], d1_s16, inp["T_12"], K_s16)

    shape("cross-attended output  (B,C,H/16,W/16)", out)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Correlation feature + ConvGRU + RefinementHead
# ─────────────────────────────────────────────────────────────────────────────

def demo_refinement_parts(inp: dict, encoder: SharedEncoder) -> None:
    section("5. Refinement building blocks  (models/refinement.py)")

    dev = inp["rgb1"].device
    B, _, H, W = inp["rgb1"].shape

    x1 = torch.cat([inp["rgb1"], inp["depth_mono1"]], dim=1)
    x2 = torch.cat([inp["rgb2"], inp["depth_mono2"]], dim=1)
    with torch.no_grad():
        f1 = encoder(x1)
        f2 = encoder(x2)

    H16, W16 = f1["s16"].shape[-2:]
    K_s16 = DepthAlignNet._scale_K(inp["K"], H, W, H16, W16)
    d1_s16 = F.interpolate(inp["depth_mono1"], size=(H16, W16), mode="nearest")
    d2_s16 = F.interpolate(inp["depth_mono2"], size=(H16, W16), mode="nearest")

    # correlation_feature
    d2_warped, valid = warp(d2_s16, d1_s16, inp["T_12"], K_s16)
    corr = correlation_feature(d1_s16, d2_warped, valid)
    shape("correlation_feature  (B, 2, H/16, W/16)", corr)

    # ConvGRU
    hidden_dim = 64
    input_dim  = 128 + 2 + 1 + 1   # feat + corr + depth_cur + depth_mono
    gru = ConvGRU(hidden_dim=hidden_dim, input_dim=input_dim).to(dev)
    gru.eval()

    h     = torch.zeros(B, hidden_dim, H16, W16, device=dev)
    dummy_input = torch.cat([f1["s16"], corr, d1_s16, d1_s16], dim=1)

    with torch.no_grad():
        h_new = gru(h, dummy_input)
    shape("ConvGRU hidden state  (B, hidden_dim, H/16, W/16)", h_new)

    # RefinementHead
    head = RefinementHead(hidden_dim=hidden_dim).to(dev)
    head.eval()
    with torch.no_grad():
        ds, db, dD, log_sigma = head(h_new)
    shape("RefinementHead → delta_s    (B,1,H/16,W/16)", ds)
    shape("RefinementHead → delta_b    (B,1,H/16,W/16)", db)
    shape("RefinementHead → delta_D    (B,1,H/16,W/16)", dD)
    shape("RefinementHead → log_sigma  (B,1,H/16,W/16)", log_sigma)


# ─────────────────────────────────────────────────────────────────────────────
# 6. Full iterative refinement
# ─────────────────────────────────────────────────────────────────────────────

def demo_iterative_refinement(inp: dict, encoder: SharedEncoder) -> None:
    section("6. Full iterative refinement  (models/refinement.py :: IterativeRefinement)")

    dev = inp["rgb1"].device
    B, _, H, W = inp["rgb1"].shape

    x1 = torch.cat([inp["rgb1"], inp["depth_mono1"]], dim=1)
    x2 = torch.cat([inp["rgb2"], inp["depth_mono2"]], dim=1)
    with torch.no_grad():
        f1 = encoder(x1)
        f2 = encoder(x2)

    H16, W16 = f1["s16"].shape[-2:]
    K_s16 = DepthAlignNet._scale_K(inp["K"], H, W, H16, W16)
    d1_s16 = F.interpolate(inp["depth_mono1"], size=(H16, W16), mode="nearest")
    d2_s16 = F.interpolate(inp["depth_mono2"], size=(H16, W16), mode="nearest")

    refine = IterativeRefinement(feat_dim=128, hidden_dim=128, num_iters=3).to(dev)
    refine.eval()

    with torch.no_grad():
        result = refine(
            depth_mono=d1_s16,
            depth2_mono=d2_s16,
            feat_cross=f1["s16"],
            T_12=inp["T_12"],
            K=K_s16,
        )

    shape("refined depth          (B,1,H/16,W/16)", result["depth"])
    shape("confidence             (B,1,H/16,W/16)", result["confidence"])
    shape("depth_iters", result["depth_iters"])
    shape("accumulated scale      (B,1)", result["scale"])
    shape("accumulated bias       (B,1)", result["bias"])
    stat("accumulated scale values", result['scale'], fmt=".4f")
    stat("accumulated bias  values", result['bias'],  fmt=".4f")


# ─────────────────────────────────────────────────────────────────────────────
# 7. Lightweight decoder
# ─────────────────────────────────────────────────────────────────────────────

def demo_decoder(inp: dict, encoder: SharedEncoder) -> None:
    section("7. Lightweight decoder  (models/decoder.py :: DepthDecoder)")

    dev = inp["rgb1"].device
    B, _, H, W = inp["rgb1"].shape

    x1 = torch.cat([inp["rgb1"], inp["depth_mono1"]], dim=1)
    with torch.no_grad():
        feats = encoder(x1)

    H16, W16 = feats["s16"].shape[-2:]
    depth_init = torch.rand(B, 1, H16, W16, device=dev) + 0.5

    decoder = DepthDecoder(feat_dim=128, hidden=64).to(dev)
    decoder.eval()

    with torch.no_grad():
        out = decoder(feats, depth_init, inp["depth_mono1"])

    shape("decoder → depth        (B,1,H/2,W/2)", out["depth"])
    shape("decoder → confidence   (B,1,H/2,W/2)", out["confidence"])
    stat("depth   range", out['depth'])
    stat("conf    range", out['confidence'])


# ─────────────────────────────────────────────────────────────────────────────
# 8. Full DepthAlignNet forward pass
# ─────────────────────────────────────────────────────────────────────────────

def demo_full_network(inp: dict) -> None:
    section("8. Full DepthAlignNet forward pass  (models/network.py)")

    dev = inp["rgb1"].device

    model = DepthAlignNet(
        feat_dim    = 128,
        hidden_dim  = 128,
        num_iters   = 3,
        num_heads   = 4,
        window_size = 7,
        pretrained  = False,
    ).to(dev)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  {'total parameters':<35} {total_params / 1e6:.2f} M")

    with torch.no_grad():
        out = model(
            rgb1        = inp["rgb1"],
            rgb2        = inp["rgb2"],
            depth_mono1 = inp["depth_mono1"],
            depth_mono2 = inp["depth_mono2"],
            T_12        = inp["T_12"],
            K           = inp["K"],
        )

    print()
    print("  Outputs:")
    shape("  depth1          (B,1,H/2,W/2)", out["depth1"])
    shape("  depth2          (B,1,H/2,W/2)", out["depth2"])
    shape("  confidence1     (B,1,H/2,W/2)", out["confidence1"])
    shape("  confidence2     (B,1,H/2,W/2)", out["confidence2"])
    shape("  depth1_iters", out["depth1_iters"])
    shape("  depth2_iters", out["depth2_iters"])
    shape("  scale1          (B,1)", out["scale1"])
    shape("  bias1           (B,1)", out["bias1"])
    shape("  scale2          (B,1)", out["scale2"])
    shape("  bias2           (B,1)", out["bias2"])

    print()
    stat("depth1  range", out['depth1'])
    stat("depth2  range", out['depth2'])
    stat("conf1   range", out['confidence1'])

    return model, out


# ─────────────────────────────────────────────────────────────────────────────
# 9. Backward pass smoke-test
# ─────────────────────────────────────────────────────────────────────────────

def demo_backward(inp: dict) -> None:
    section("9. Backward pass smoke-test (gradients flow through all stages)")

    if inp["rgb1"].shape[0] == 0:
        print("  (skipped — batch=0 produces empty tensors, no backward possible)")
        return

    dev = inp["rgb1"].device

    model = DepthAlignNet(
        feat_dim=128, hidden_dim=128, num_iters=2,
        num_heads=4, window_size=7, pretrained=False,
    ).to(dev)
    model.train()

    out = model(
        rgb1=inp["rgb1"], rgb2=inp["rgb2"],
        depth_mono1=inp["depth_mono1"], depth_mono2=inp["depth_mono2"],
        T_12=inp["T_12"], K=inp["K"],
    )

    # Simple surrogate loss: L1 on output depths
    loss = out["depth1"].mean() + out["depth2"].mean()
    loss.backward()

    # Check that gradients reached the encoder stem
    stem_grad = model.encoder.stem[0].weight.grad
    print(f"  loss value                          {loss.item():.6f}")
    print(f"  encoder stem grad norm              {stem_grad.norm().item():.6f}")
    print(f"  backward pass:                      OK ✓")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="DepthAlignNet end-to-end demo")
    parser.add_argument("--cuda",  action="store_true", help="Use CUDA if available")
    parser.add_argument("--batch", type=int, default=2,   help="Batch size")
    parser.add_argument("--height", type=int, default=240, help="Image height")
    parser.add_argument("--width",  type=int, default=320, help="Image width")
    args = parser.parse_args()

    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}  |  B={args.batch}  H={args.height}  W={args.width}")

    inp = make_inputs(args.batch, args.height, args.width, device)

    # Run each stage independently so failures are easy to locate
    demo_geometry(inp)

    encoder = demo_encoder(inp)

    demo_pose_encoder(inp)

    demo_attention(inp, encoder)

    demo_refinement_parts(inp, encoder)

    demo_iterative_refinement(inp, encoder)

    demo_decoder(inp, encoder)

    demo_full_network(inp)

    demo_backward(inp)

    section("All stages completed successfully")


if __name__ == "__main__":
    main()
