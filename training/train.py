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
import sys
import time

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.join(_REPO_ROOT, "data"))

import torch
import yaml
from torch.utils.data import DataLoader

from data.temporal_sampling import ScanNetTemporalDataset
from data.graph_based_sampling import ScanNetGraphDataset
from models.network import DepthAlignNet
from training.losses import total_loss as compute_total_loss


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def seed_everything(seed: int) -> None:
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Dataset factory
# ---------------------------------------------------------------------------

def build_dataset(cfg: dict, root_dir: str, num_samples: int):
    ds_cfg    = cfg["dataset"]
    graph_cfg = cfg.get("graph_sampling", {})
    sampler   = ds_cfg.get("sampler_type", "temporal")

    if sampler == "temporal":
        return ScanNetTemporalDataset(
            root_dir=root_dir,
            num_frames=ds_cfg["num_frames"],
            num_samples=num_samples,
            min_stride=ds_cfg["min_stride"],
            max_stride=ds_cfg["max_stride"],
        )
    elif sampler == "graph":
        return ScanNetGraphDataset(
            root_dir=root_dir,
            num_frames=ds_cfg["num_frames"],
            num_samples=num_samples,
            graph_cache=graph_cfg.get("graph_cache"),
            min_overlap=graph_cfg["min_overlap"],
            max_overlap=graph_cfg["max_overlap"],
            overlap_sample_step=graph_cfg["overlap_sample_step"],
            depth_tolerance=graph_cfg["depth_tolerance"],
            max_frame_gap=graph_cfg.get("max_frame_gap", 50),
        )
    else:
        raise ValueError(f"Unknown sampler_type: '{sampler}'")


def build_loader(cfg: dict, dataset, shuffle: bool) -> DataLoader:
    ldr = cfg["loader"]
    return DataLoader(
        dataset,
        batch_size=ldr["batch_size"],
        num_workers=ldr["num_workers"],
        pin_memory=ldr.get("pin_memory", True),
        drop_last=ldr.get("drop_last", True),
        shuffle=shuffle,
    )


# ---------------------------------------------------------------------------
# Optimiser / scheduler
# ---------------------------------------------------------------------------

def build_optimizer(cfg: dict, model: torch.nn.Module) -> torch.optim.Optimizer:
    opt_cfg = cfg["optimizer"]
    kind    = opt_cfg.get("type", "adamw").lower()
    lr      = float(opt_cfg["lr"])
    wd      = float(opt_cfg.get("weight_decay", 0.0))

    if kind == "adamw":
        betas = tuple(opt_cfg.get("betas", [0.9, 0.999]))
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd, betas=betas)
    elif kind == "adam":
        betas = tuple(opt_cfg.get("betas", [0.9, 0.999]))
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd, betas=betas)
    elif kind == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer type: '{kind}'")


def build_scheduler(cfg: dict, optimizer: torch.optim.Optimizer, num_epochs: int):
    sch_cfg = cfg["scheduler"]
    kind    = sch_cfg.get("type", "cosine").lower()
    warmup  = int(sch_cfg.get("warmup_epochs", 0))

    def make_base():
        if kind == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=int(sch_cfg.get("T_max", num_epochs)),
                eta_min=float(sch_cfg.get("eta_min", 1e-6)),
            )
        elif kind == "step":
            return torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=int(sch_cfg.get("step_size", 10)),
                gamma=float(sch_cfg.get("gamma", 0.5)),
            )
        elif kind == "plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                patience=int(sch_cfg.get("patience", 5)),
                factor=float(sch_cfg.get("gamma", 0.5)),
            )
        elif kind == "none":
            return None
        else:
            raise ValueError(f"Unknown scheduler type: '{kind}'")

    base = make_base()
    if warmup > 0 and base is not None:
        warmup_sched = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1e-3, end_factor=1.0, total_iters=warmup
        )
        return torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_sched, base], milestones=[warmup]
        )
    return base


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(state: dict, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path: str, model, optimizer, scheduler, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and ckpt.get("scheduler"):
        scheduler.load_state_dict(ckpt["scheduler"])
    start_epoch = ckpt.get("epoch", 0) + 1
    best_metric = ckpt.get("best_metric", float("inf"))
    print(f"[ckpt] Resumed from {path}  (epoch {ckpt.get('epoch', 0)}, best={best_metric:.4f})")
    return start_epoch, best_metric


def _keep_top_k(save_dir: str, monitor_tag: str, top_k: int) -> None:
    """Remove oldest checkpoints if more than top_k exist."""
    ckpts = sorted(
        [f for f in os.listdir(save_dir) if f.startswith("epoch_") and f.endswith(".pt")],
        key=lambda f: os.path.getmtime(os.path.join(save_dir, f)),
    )
    for old in ckpts[:-top_k]:
        os.remove(os.path.join(save_dir, old))


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def compute_loss(batch: dict, outputs: dict, cfg: dict, K: torch.Tensor):
    """
    Thin wrapper so train() stays clean.
    """
    return compute_total_loss(outputs, batch, cfg.get("loss", {}), K)


# ---------------------------------------------------------------------------
# One epoch
# ---------------------------------------------------------------------------

def run_epoch(
    model,
    loader: DataLoader,
    optimizer,
    scaler,
    cfg: dict,
    device,
    train: bool,
    epoch: int,
) -> dict:
    model.train(train)
    train_cfg = cfg["train"]
    log_every = int(train_cfg.get("log_every", 50))
    grad_clip = train_cfg.get("grad_clip")
    use_amp   = train_cfg.get("mixed_precision", False) and device.type == "cuda"
    accum     = int(train_cfg.get("accumulate_grad_batches", 1))

    totals    = {"loss": 0.0, "depth": 0.0, "photo": 0.0, "consistency": 0.0, "smooth": 0.0, "iters": 0.0}
    n_batches = 0
    t0        = time.time()

    # Build a fixed K from config (will be overridden per-batch if dataset provides it)
    model_cfg = cfg.get("model", {})
    fx = float(model_cfg.get("fx", 577.0))
    fy = float(model_cfg.get("fy", 577.0))
    cx = float(model_cfg.get("cx", 320.0))
    cy = float(model_cfg.get("cy", 240.0))

    ctx = torch.no_grad() if not train else torch.enable_grad()
    with ctx:
        for step, batch in enumerate(loader):
            # move tensors to device
            batch = {
                k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            B = batch["images"].shape[0]
            H = batch["images"].shape[-2]
            W = batch["images"].shape[-1]

            # Per-batch intrinsics — use dataset-provided if available, else config fallback
            if "intrinsics" in batch:
                K = batch["intrinsics"]    # (B, 3, 3)
            else:
                K = torch.tensor(
                    [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
                    dtype=torch.float32, device=device
                ).unsqueeze(0).expand(B, -1, -1)

            imgs   = batch["images"]       # (B, N, 3, H, W)
            depths = batch["depths"]       # (B, N, 1, H, W)
            poses  = batch["poses"]        # (B, N, 4, 4)

            rgb1        = imgs[:, 0]
            rgb2        = imgs[:, 1]
            depth_mono1 = depths[:, 0]
            depth_mono2 = depths[:, 1]

            # T_12 = inv(pose2) @ pose1  (cam1 → cam2)
            T_12 = torch.linalg.inv(poses[:, 1]) @ poses[:, 0]

            with torch.autocast(device_type=device.type, enabled=use_amp):
                outputs = model(
                    rgb1=rgb1,
                    rgb2=rgb2,
                    depth_mono1=depth_mono1,
                    depth_mono2=depth_mono2,
                    T_12=T_12,
                    K=K,
                )
                loss, breakdown = compute_loss(batch, outputs, cfg, K)
                if accum > 1:
                    loss = loss / accum

            if train:
                scaler.scale(loss).backward()
                if (step + 1) % accum == 0:
                    if grad_clip:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)

            totals["loss"]        += loss.item() * (accum if accum > 1 else 1)
            totals["depth"]       += breakdown.get("depth", 0.0)
            totals["photo"]       += breakdown.get("photo", 0.0)
            totals["consistency"] += breakdown.get("consistency", 0.0)
            totals["smooth"]      += breakdown.get("smooth", 0.0)
            totals["iters"]       += breakdown.get("iters", 0.0)
            n_batches += 1

            if train and (step + 1) % log_every == 0:
                elapsed = time.time() - t0
                avg_loss = totals["loss"] / n_batches
                lr = optimizer.param_groups[0]["lr"]
                print(
                    f"  epoch {epoch:03d}  step {step+1:04d}/{len(loader):04d}"
                    f"  loss={avg_loss:.4f}  lr={lr:.2e}  {elapsed:.1f}s"
                )

    return {k: v / max(n_batches, 1) for k, v in totals.items()}


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(cfg: dict, resume: str | None = None) -> None:
    train_cfg = cfg["train"]
    ckpt_cfg  = cfg["checkpoint"]

    seed_everything(int(train_cfg.get("seed", 0)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] Device: {device}")

    # --- datasets ---
    root_dir = cfg["dataset"]["root_dir"]
    n_train  = cfg["dataset"]["num_samples"]
    n_val    = max(1, n_train // 4)    # ~20 % of train virtual length

    train_ds = build_dataset(cfg, root_dir=os.path.join(root_dir, "train"), num_samples=n_train)
    val_ds   = build_dataset(cfg, root_dir=os.path.join(root_dir, "val"),   num_samples=n_val)

    train_loader = build_loader(cfg, train_ds, shuffle=True)
    val_loader   = build_loader(cfg, val_ds,   shuffle=False)

    print(f"[train] Batches train={len(train_loader)}  val={len(val_loader)}  "
          f"batch_size={cfg['loader']['batch_size']}")

    # --- model ---
    model_cfg = cfg.get("model", {})
    model = DepthAlignNet(
        feat_dim    = int(model_cfg.get("feat_dim",    128)),
        hidden_dim  = int(model_cfg.get("hidden_dim", 128)),
        num_iters   = int(model_cfg.get("num_iters",    4)),
        num_heads   = int(model_cfg.get("num_heads",    4)),
        window_size = int(model_cfg.get("window_size",  7)),
        pretrained  = bool(model_cfg.get("pretrained", True)),
    ).to(device)

    optimizer  = build_optimizer(cfg, model)
    scheduler  = build_scheduler(cfg, optimizer, num_epochs=int(train_cfg["epochs"]))
    scaler     = torch.cuda.amp.GradScaler(enabled=(
        train_cfg.get("mixed_precision", False) and device.type == "cuda"
    ))

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
            model, train_loader, optimizer, scaler, cfg, device, train=True, epoch=epoch
        )

        val_metrics = {}
        if epoch % val_every == 0:
            val_metrics = run_epoch(
                model, val_loader, None, scaler, cfg, device, train=False, epoch=epoch
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
        val_str = f"  val_loss={val_metrics.get('loss', float('nan')):.4f}" if val_metrics else ""
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
        _keep_top_k(save_dir, monitor, top_k)

    print(f"\n[train] Done. Best {monitor}={best_metric:.4f}")


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
        "--resume",
        default=None,
        help="Path to a checkpoint to resume from (overrides training.yaml resume).",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    resume = args.resume or cfg.get("checkpoint", {}).get("resume")
    train(cfg, resume=resume)
