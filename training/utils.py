"Training helper functions"

import torch
from torch.utils.data import DataLoader
from data.preprocessing import PreSampledPairDataset
from collections import deque
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
    ds_cfg       = cfg.get("dataset", {})
    mde_source   = ds_cfg.get("mde_source", "zoedepth")
    aug_cfg      = cfg.get("augmentation") if augment else None
    image_size   = ds_cfg.get("image_size")     # e.g. [480, 640]
    pair_lag_max = int(ds_cfg.get("pair_lag_max", 1))
    return PreSampledPairDataset(
        root_dir=root_dir,
        mde_source=mde_source,
        scene_paths=scene_paths,
        aug_cfg=aug_cfg,
        image_size=image_size,
        pair_lag_max=pair_lag_max,
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

    # Per-module LR scaling: cross_attn block gets a lower rate so that
    # pretrained MASt3R weights are fine-tuned more conservatively.
    # Controlled by optimizer.cross_attn_lr_scale in the config (default 0.1).
    cross_attn_scale = float(opt_cfg.get("cross_attn_lr_scale", 0.1))
    cross_attn_mod   = getattr(model, "cross_attn", None)
    use_cross_attn_split = cross_attn_mod is not None and cross_attn_scale != 1.0
    cross_attn_ids = {id(p) for p in cross_attn_mod.parameters()} if use_cross_attn_split else set()

    # Weight-decay exclusion split: parameters with ndim > 1 (weight matrices /
    # conv kernels) get weight_decay=wd, everything else (biases, norm
    # weight+bias, standalone scalar params like log_conf_pose/log_conf_K)
    # gets weight_decay=0.0. Combined with the cross_attn lr split above this
    # yields up to 4 param groups: {main, cross_attn} x {decay, no_decay}.
    main_decay, main_no_decay = [], []
    cross_decay, cross_no_decay = [], []
    for p in model.parameters():
        is_cross = id(p) in cross_attn_ids
        is_decay = p.ndim > 1
        if is_cross:
            (cross_decay if is_decay else cross_no_decay).append(p)
        else:
            (main_decay if is_decay else main_no_decay).append(p)

    cross_lr = lr * cross_attn_scale
    param_groups = []
    if main_decay:
        param_groups.append({"params": main_decay, "lr": lr, "weight_decay": wd})
    if main_no_decay:
        param_groups.append({"params": main_no_decay, "lr": lr, "weight_decay": 0.0})
    if use_cross_attn_split:
        if cross_decay:
            param_groups.append({"params": cross_decay, "lr": cross_lr, "weight_decay": wd})
        if cross_no_decay:
            param_groups.append({"params": cross_no_decay, "lr": cross_lr, "weight_decay": 0.0})
        print(f"[optimizer] cross_attn lr scale={cross_attn_scale}  "
              f"(main lr={lr:.2e}, cross_attn lr={cross_lr:.2e})")
    else:
        # No separate cross_attn lr: fold any cross_attn params (there are
        # none here since cross_attn_ids is empty) into the main groups.
        pass

    n_main_decay      = sum(p.numel() for p in main_decay)
    n_main_no_decay   = sum(p.numel() for p in main_no_decay)
    n_cross_decay     = sum(p.numel() for p in cross_decay)
    n_cross_no_decay  = sum(p.numel() for p in cross_no_decay)
    print(f"[optimizer] main/decay: {n_main_decay} params, "
          f"main/no_decay: {n_main_no_decay} params, "
          f"cross_attn/decay: {n_cross_decay} params, "
          f"cross_attn/no_decay: {n_cross_no_decay} params")

    if kind == "adamw":
        betas = tuple(opt_cfg.get("betas", [0.9, 0.999]))
        return torch.optim.AdamW(param_groups, betas=betas)
    elif kind == "adam":
        betas = tuple(opt_cfg.get("betas", [0.9, 0.999]))
        return torch.optim.Adam(param_groups, betas=betas)
    elif kind == "sgd":
        return torch.optim.SGD(param_groups, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer type: '{kind}'")


def optimizer_step(optimizer, model, grad_clip, scaler=None, log_every: int | None = None, step: int | None = None) -> None:
    # When using AMP, unscale before clipping so norms are in fp32 scale.
    if scaler is not None:
        scaler.unscale_(optimizer)
    if grad_clip:
        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        pre_clip_norm = float(total_norm)

        # --- diagnostic only: track how often clipping actually engages ---
        # Reuses the value clip_grad_norm_ already computed; no extra pass
        # over gradients. History window mirrors the log_every cadence.
        if not hasattr(optimizer, "_grad_norm_history"):
            optimizer._grad_norm_history = deque(maxlen=max(int(log_every or 50), 1))
        hist = optimizer._grad_norm_history
        hist.append(pre_clip_norm)
        if log_every and step is not None and (step + 1) % log_every == 0:
            clip_frac = sum(1 for n in hist if n > grad_clip) / len(hist)
            print(f"[grad-norm] step={step + 1}  pre_clip_norm={pre_clip_norm:.4f}  "
                  f"grad_clip={grad_clip:.4f}  clip_frac(last {len(hist)})={clip_frac:.2%}")

        if not torch.isfinite(total_norm):
            print(f"[NaN grad] total_norm={total_norm:.4f} — skipping update")
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.update()   # must call update() even when skipping
            return
    if scaler is not None:
        scaler.step(optimizer)
        scaler.update()
    else:
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
    
    # Load model with strict=False to allow missing optional heads (pose, encoding)
    incompatible_keys = model.load_state_dict(ckpt["model"], strict=False)
    
    # Log information about missing/unexpected keys
    if incompatible_keys.missing_keys:
        # Check if missing keys are from optional heads (pose prediction, encoding)
        optional_keys = {'pose_head', 'to_pose', 'to_log_conf_pose', 'pose_encoder', 'encoding_head'}
        missing_optional = [k for k in incompatible_keys.missing_keys if any(opt in k for opt in optional_keys)]
        missing_critical = [k for k in incompatible_keys.missing_keys if not any(opt in k for opt in optional_keys)]
        
        if missing_optional:
            print(f"[ckpt] Missing optional heads (not in checkpoint): {missing_optional}")
        if missing_critical:
            print(f"[ckpt] WARNING: Missing critical layers: {missing_critical}")
    
    if incompatible_keys.unexpected_keys:
        print(f"[ckpt] WARNING: Unexpected keys in checkpoint (likely from old version): {incompatible_keys.unexpected_keys}")
    
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
    if gt_d.shape == mde_d.shape:
        if torch.allclose(gt_d, mde_d, atol=1e-3):
            print("[DEBUG] WARNING: MDE prior and GT are identical — still feeding GT as input!")
        else:
            print("[DEBUG] OK: MDE prior differs from GT.")
    else:
        # Different spatial resolutions (e.g. color 1296×968 vs depth 640×480) —
        # they cannot be identical, so no accidental GT-leak is possible.
        print("[DEBUG] OK: MDE prior and GT have different resolutions; no GT-leak.")

def fallback_intrinsics(cfg: dict, B: int, device) -> torch.Tensor:
    m = cfg.get("model", {})
    fx, fy = float(m.get("fx", 577.0)), float(m.get("fy", 577.0))
    cx, cy = float(m.get("cx", 320.0)), float(m.get("cy", 240.0))
    K = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=torch.float32, device=device)
    return K.unsqueeze(0).expand(B, -1, -1)