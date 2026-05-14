"""
train.py — Training loop for mono-depth-to-3DR.

All hyper-parameters are read from config/training.yaml.

Usage
-----
    python training/train.py
    python training/train.py --config config/training.yaml
    python training/train.py --config config/training.yaml --resume checkpoints/last.pt
"""

import argparse
import os
import random
import sys
import time

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.join(_REPO_ROOT, "data"))

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from data.scene import find_scene_paths
from models.model_image_depth.network import DepthAlignNet
from training.losses import total_loss as compute_total_loss, compute_depth_metrics
from training.utils import (
    seed_everything, build_dataset, build_loader,
    build_optimizer, build_scheduler,
    save_checkpoint, load_checkpoint, keep_top_k,
    optimizer_step, fallback_intrinsics, debug_depth_check,
)


# ---------------------------------------------------------------------------
# One epoch
# ---------------------------------------------------------------------------

def run_epoch(
    model,
    loader: DataLoader,
    optimizer,
    cfg: dict,
    device,
    train: bool,
    epoch: int,
) -> dict:
    model.train(train)
    train_cfg = cfg["train"]
    log_every = int(train_cfg.get("log_every", 50))
    grad_clip = train_cfg.get("grad_clip")

    totals    = {"loss": 0.0, "depth": 0.0, "smooth": 0.0, "iters": 0.0}
    metric_sums = {"abs_rel": 0.0, "rmse": 0.0, "delta1": 0.0}
    metric_count = 0
    n_batches = 0
    t0        = time.time()


    ctx = torch.no_grad() if not train else torch.enable_grad()
    with ctx:
        for step, batch in enumerate(loader):
            # move tensors to device
            batch = {
                k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            B = batch["images"].shape[0]

            # Per-batch intrinsics — use dataset-provided if available, else config fallback
            K = batch["intrinsics"] if "intrinsics" in batch else fallback_intrinsics(cfg, B, device)

            imgs   = batch["images"]       # (B, N, 3, H, W)
            depths = batch["depths"]       # (B, N, 1, H, W)  ground-truth depths
            poses  = batch["poses"]        # (B, N, 4, 4)

            # MDE predictions used as the monocular prior fed to the model.
            # These come from zoe-depth_pred/ — NOT the GT sensor depths.
            if "mde_depths" not in batch:
                raise KeyError(
                    "[BUG] 'mde_depths' missing from batch. "
                    "Dataset must supply ZoeDepth predictions separately from GT depths."
                )
            mde_depths = batch["mde_depths"]   # (B, N, 1, H, W)

            rgb1        = imgs[:, 0]
            rgb2        = imgs[:, 1]
            depth_mono1 = mde_depths[:, 0]   # MDE prior for view 1
            depth_mono2 = mde_depths[:, 1]   # MDE prior for view 2

            # --- debug prints (first step of first epoch only) ---
            if epoch == 1 and step == 0 and train:
                debug_depth_check(depths, mde_depths)

            # T_12 = inv(pose2) @ pose1  (cam1 → cam2)
            T_12 = torch.linalg.inv(poses[:, 1]) @ poses[:, 0]

            outputs = model(
                rgb1=rgb1,
                rgb2=rgb2,
                depth_mono1=depth_mono1,
                depth_mono2=depth_mono2,
                T_12=T_12,
                K=K,
            )
            loss, breakdown = compute_total_loss(outputs, batch, cfg.get("loss", {}), K)

            if train:
                loss.backward()
                optimizer_step(optimizer, model, grad_clip)

            totals["loss"]   += loss.item()
            totals["depth"]  += breakdown.get("depth", 0.0)
            totals["smooth"] += breakdown.get("smooth", 0.0)
            totals["iters"]  += breakdown.get("iters", 0.0)

            if not train:
                pred1 = outputs["depth1"]                         # (B,1,pH,pW)
                gt1   = depths[:, 0]                              # (B,1,H,W)
                gt1_s = F.interpolate(gt1, size=pred1.shape[-2:], mode="nearest")
                m = compute_depth_metrics(pred1.detach(), gt1_s)
                if not any(v != v for v in m.values()):           # skip NaN batches
                    for k, v in m.items():
                        metric_sums[k] += v
                    metric_count += 1
            n_batches += 1

            if train and (step + 1) % log_every == 0:
                elapsed = time.time() - t0
                avg_loss = totals["loss"] / n_batches
                lr = optimizer.param_groups[0]["lr"]
                print(
                    f"  epoch {epoch:03d}  step {step+1:04d}/{len(loader):04d}"
                    f"  loss={avg_loss:.4f}  lr={lr:.2e}  {elapsed:.1f}s"
                )

    out = {k: v / max(n_batches, 1) for k, v in totals.items()}
    if not train and metric_count > 0:
        for k, v in metric_sums.items():
            out[k] = v / metric_count
    return out


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(cfg: dict, arch_cfg: dict, resume: str | None = None) -> None:
    train_cfg = cfg["train"]
    ckpt_cfg  = cfg["checkpoint"]

    seed_everything(int(train_cfg.get("seed", 0)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] Device: {device}")

    # --- datasets ---
    root_dir = cfg["dataset"]["root_dir"]
    # resolve relative paths from the repo root
    if not os.path.isabs(root_dir):
        root_dir = os.path.join(_REPO_ROOT, root_dir)

    # 80/10/10 scene split (deterministic via fixed seed)
    all_scene_paths = find_scene_paths(root_dir)
    rng = random.Random(42)
    rng.shuffle(all_scene_paths)
    max_scenes = cfg["dataset"].get("max_scenes")
    if max_scenes:
        all_scene_paths = all_scene_paths[:int(max_scenes)]
    n = len(all_scene_paths)
    s1 = int(n * 0.8)
    s2 = int(n * 0.9)
    train_paths = all_scene_paths[:s1]
    val_paths   = all_scene_paths[s1:s2]
    test_paths  = all_scene_paths[s2:]
    print(f"[train] Scenes: {len(train_paths)} train / {len(val_paths)} val / {len(test_paths)} test")

    train_ds = build_dataset(cfg, root_dir=root_dir, scene_paths=train_paths)
    val_ds   = build_dataset(cfg, root_dir=root_dir, scene_paths=val_paths)
    test_ds  = build_dataset(cfg, root_dir=root_dir, scene_paths=test_paths)

    train_loader = build_loader(cfg, train_ds, shuffle=True)
    val_loader   = build_loader(cfg, val_ds,   shuffle=False)
    test_loader  = build_loader(cfg, test_ds,  shuffle=False)

    print(f"[train] Batches train={len(train_loader)}  val={len(val_loader)}  test={len(test_loader)}"
          f"  batch_size={cfg['loader']['batch_size']}")

    # --- model ---
    enc_cfg = arch_cfg.get("encoder",    {})
    att_cfg = arch_cfg.get("attention",  {})
    ref_cfg = arch_cfg.get("refinement", {})
    dec_cfg = arch_cfg.get("decoder",    {})
    model = DepthAlignNet(
        feat_dim        = int(enc_cfg.get("out_channels",      128)),
        hidden_dim      = int(ref_cfg.get("hidden_dim",        128)),
        num_iters       = int(ref_cfg.get("num_iters",           4)),
        num_heads       = int(att_cfg.get("num_heads",           4)),
        window_size     = int(att_cfg.get("window_size",         7)),
        pretrained      = bool(enc_cfg.get("pretrained",      True)),
        freeze_backbone = bool(enc_cfg.get("freeze_backbone", False)),
        use_refinement  = bool(ref_cfg.get("enabled",         True)),
        decoder_hidden  = int(dec_cfg.get("hidden_dim",         64)),
    ).to(device)

    optimizer  = build_optimizer(cfg, model)
    scheduler  = build_scheduler(cfg, optimizer, num_epochs=int(train_cfg["epochs"]))

    # --- resume ---
    start_epoch = 1
    best_metric = float("inf") if ckpt_cfg.get("mode", "min") == "min" else -float("inf")
    if resume and os.path.isfile(resume):
        start_epoch, best_metric = load_checkpoint(resume, model, optimizer, scheduler, device)

    save_dir  = ckpt_cfg.get("save_dir", "checkpoints/")
    top_k     = int(ckpt_cfg.get("save_top_k", 3))
    monitor   = ckpt_cfg.get("monitor", "val/loss")
    mode      = ckpt_cfg.get("mode", "min")
    val_every = int(train_cfg.get("val_every", 1))

    def is_better(new, best):
        return new < best if mode == "min" else new > best

    # --- epoch loop ---
    for epoch in range(start_epoch, int(train_cfg["epochs"]) + 1):
        t_epoch = time.time()

        train_metrics = run_epoch(
            model, train_loader, optimizer, cfg, device, train=True, epoch=epoch
        )

        val_metrics = {}
        if epoch % val_every == 0:
            val_metrics = run_epoch(
                model, val_loader, None, cfg, device, train=False, epoch=epoch
            )

        # --- scheduler step ---
        if scheduler is not None:
            monitor_val = val_metrics.get("loss", train_metrics["loss"])
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(monitor_val)
            else:
                scheduler.step()

        # --- logging ---
        lr  = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t_epoch
        val_str = ""
        if val_metrics:
            val_str = f"  val_loss={val_metrics.get('loss', float('nan')):.4f}"
            if "abs_rel" in val_metrics:
                val_str += (
                    f"  abs_rel={val_metrics['abs_rel']:.4f}"
                    f"  rmse={val_metrics['rmse']:.4f}"
                    f"  delta1={val_metrics['delta1']:.4f}"
                )
        print(
            f"[epoch {epoch:03d}/{train_cfg['epochs']}]"
            f"  train_loss={train_metrics['loss']:.4f}{val_str}"
            f"  lr={lr:.2e}  {elapsed:.1f}s"
        )

        # --- checkpoint ---
        monitor_key = monitor.split("/")[-1]   # e.g. "val/abs_rel" → "abs_rel"
        current = val_metrics.get(monitor_key, train_metrics.get(monitor_key, train_metrics["loss"]))

        state = {
            "epoch":       epoch,
            "model":       model.state_dict(),
            "optimizer":   optimizer.state_dict(),
            "scheduler":   scheduler.state_dict() if scheduler else None,
            "best_metric": best_metric,
            "cfg":         cfg,
        }

        # always save last
        save_checkpoint(state, os.path.join(save_dir, "last.pt"))

        if is_better(current, best_metric):
            best_metric = current
            state["best_metric"] = best_metric
            save_checkpoint(state, os.path.join(save_dir, "best.pt"))
            print(f"  [ckpt] New best {monitor}={best_metric:.4f}  →  {save_dir}/best.pt")

        epoch_path = os.path.join(save_dir, f"epoch_{epoch:04d}.pt")
        save_checkpoint(state, epoch_path)
        keep_top_k(save_dir, monitor, top_k)

    print(f"\n[train] Done. Best {monitor}={best_metric:.4f}")

    # --- Final test evaluation on best checkpoint ---
    print("\n[test] Loading best checkpoint for final evaluation...")
    best_ckpt = os.path.join(save_dir, "best.pt")
    if os.path.isfile(best_ckpt):
        ckpt = torch.load(best_ckpt, map_location=device)
        model.load_state_dict(ckpt["model"])
    test_metrics = run_epoch(model, test_loader, None, cfg, device, train=False, epoch=0)
    print(
        f"[test] loss={test_metrics['loss']:.4f}"
        + (f"  abs_rel={test_metrics['abs_rel']:.4f}"
           f"  rmse={test_metrics['rmse']:.4f}"
           f"  delta1={test_metrics['delta1']:.4f}"
           if "abs_rel" in test_metrics else "")
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train mono-depth scale-shift network.")
    parser.add_argument(
        "--config",
        default=os.path.join(_REPO_ROOT, "config", "training.yaml"),
        help="Path to training.yaml",
    )
    parser.add_argument(
        "--arch",
        default=os.path.join(_REPO_ROOT, "config", "arch.yaml"),
        help="Path to arch.yaml (architecture definition).",
    )
    parser.add_argument(
        "--resume",
        default=None,
        help="Path to a checkpoint to resume from (overrides training.yaml resume).",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    with open(args.arch) as f:
        arch_cfg = yaml.safe_load(f)

    resume = args.resume or cfg.get("checkpoint", {}).get("resume")
    train(cfg, arch_cfg, resume=resume)
