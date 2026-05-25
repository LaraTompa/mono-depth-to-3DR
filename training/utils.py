"Training helper functions"

import torch
from torch.utils.data import DataLoader
from data.preprocessing import PreSampledPairDataset
import os

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

def build_dataset(cfg: dict, root_dir: str, scene_paths=None, augment: bool = False):
    mde_source = cfg.get("dataset", {}).get("mde_source", "zoedepth")
    aug_cfg    = cfg.get("augmentation") if augment else None
    return PreSampledPairDataset(
        root_dir=root_dir,
        mde_source=mde_source,
        scene_paths=scene_paths,
        aug_cfg=aug_cfg,
    )


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
    
def optimizer_step(optimizer, model, grad_clip, scaler=None) -> None:
    # When using AMP, unscale before clipping so norms are in fp32 scale.
    if scaler is not None:
        scaler.unscale_(optimizer)
    if grad_clip:
        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        if not torch.isfinite(total_norm):
            print(f"[NaN grad] total_norm={total_norm:.4f} — skipping update")
            # Also zero out any NaN/Inf values that snuck into model weights
            # (once weights are NaN the forward pass stays broken indefinitely).
            nan_params = 0
            for p in model.parameters():
                if not torch.isfinite(p.data).all():
                    p.data = torch.nan_to_num(p.data, nan=0.0, posinf=1.0, neginf=-1.0)
                    nan_params += 1
            if nan_params:
                print(f"[NaN grad] reset {nan_params} parameter tensors with non-finite values")
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.update()   # must call update() even when skipping
            return
    if scaler is not None:
        scaler.step(optimizer)
        scaler.update()
    else:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    optimizer.zero_grad(set_to_none=True)


def build_scheduler(cfg: dict, optimizer: torch.optim.Optimizer, num_epochs: int):
    sch_cfg = cfg["scheduler"]
    kind    = sch_cfg.get("type", "cosine").lower()

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


def keep_top_k(save_dir: str, monitor_tag: str, top_k: int) -> None:
    """Remove oldest checkpoints if more than top_k exist."""
    ckpts = sorted(
        [f for f in os.listdir(save_dir) if f.startswith("epoch_") and f.endswith(".pt")],
        key=lambda f: os.path.getmtime(os.path.join(save_dir, f)),
    )
    for old in ckpts[:-top_k]:
        os.remove(os.path.join(save_dir, old))

def debug_depth_check(depths: torch.Tensor, mde_depths: torch.Tensor) -> None:
    gt_d, mde_d = depths[:, 0], mde_depths[:, 0]
    valid_gt, valid_mde = gt_d[gt_d > 0], mde_d[mde_d > 0]
    print(f"\n[DEBUG] MDE prior  shape={tuple(mde_d.shape)}"
          f"  min={valid_mde.min():.3f}  max={valid_mde.max():.3f}  mean={valid_mde.mean():.3f} m")
    print(f"[DEBUG] GT depth   shape={tuple(gt_d.shape)}"
          f"  min={valid_gt.min():.3f}  max={valid_gt.max():.3f}  mean={valid_gt.mean():.3f} m")
    if torch.allclose(gt_d, mde_d, atol=1e-3):
        print("[DEBUG] WARNING: MDE prior and GT are identical — still feeding GT as input!")
    else:
        print("[DEBUG] OK: MDE prior differs from GT.")

def fallback_intrinsics(cfg: dict, B: int, device) -> torch.Tensor:
    m = cfg.get("model", {})
    fx, fy = float(m.get("fx", 577.0)), float(m.get("fy", 577.0))
    cx, cy = float(m.get("cx", 320.0)), float(m.get("cy", 240.0))
    K = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=torch.float32, device=device)
    return K.unsqueeze(0).expand(B, -1, -1)
