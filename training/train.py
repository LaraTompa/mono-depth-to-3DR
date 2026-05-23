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

try:
    from torch.utils.tensorboard import SummaryWriter as _SummaryWriter
except ImportError:   # tensorboard not installed — logging silently disabled
    _SummaryWriter = None

from data.scene import find_scene_paths
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
    writer=None,          # torch.utils.tensorboard.SummaryWriter or None
    use_confidence: bool = True,
    camera_weight: float | None = None,
    scaler=None,          # torch.amp.GradScaler or None (enables AMP when set)
) -> dict:
    model.train(train)
    train_cfg = cfg["train"]
    log_every        = int(train_cfg.get("log_every", 50))
    grad_clip        = train_cfg.get("grad_clip")
    grad_accum_steps = int(train_cfg.get("gradient_accumulation_steps", 1))

    totals    = {
        "loss": 0.0,
        "depth": 0.0,
        "smooth": 0.0,
        "iters": 0.0,
        "pixel_consistency": 0.0,
        "cam_pose": 0.0,
        "cam_rot": 0.0,
        "cam_trans": 0.0,
        "cam_K": 0.0,
        "cam_identity": 0.0,
    }
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

            # GT poses / intrinsics retained for future pose-supervision losses.
            K_gt    = batch["intrinsics"] if "intrinsics" in batch else fallback_intrinsics(cfg, B, device)
            T_12_gt = torch.linalg.inv(poses[:, 1]) @ poses[:, 0]

            # ── Iterative camera-pose initialisation ─────────────────────────
            # Iteration 0: identity relative pose + focal-length prior.
            # Justification: 0.9 × max(H, W) gives fx ≈ fy ≈ 576 px for
            # 640×480 (close to ScanNet's actual ~577) and is a safe starting
            # point for ~60° diagonal FoV without any calibration data.
            H_img, W_img = rgb1.shape[-2:]
            f_init = float(max(H_img, W_img)) * 0.9
            K_iter = torch.tensor(
                [[f_init, 0.0,    W_img / 2.0],
                 [0.0,    f_init, H_img / 2.0],
                 [0.0,    0.0,    1.0]],
                dtype=torch.float32, device=device,
            ).unsqueeze(0).expand(B, -1, -1).contiguous()
            T_12_iter = (torch.eye(4, device=device, dtype=torch.float32)
                         .unsqueeze(0).expand(B, -1, -1).contiguous())

            num_pose_iters = train_cfg.get("num_pose_iters", 2)
            with torch.amp.autocast("cuda", enabled=(scaler is not None)):
                for pose_it in range(num_pose_iters):
                    outputs = model(
                        rgb1=rgb1,
                        rgb2=rgb2,
                        depth_mono1=depth_mono1,
                        depth_mono2=depth_mono2,
                        T_12=T_12_iter,
                        K=K_iter,
                    )
                    if pose_it < num_pose_iters - 1:
                        # Feed predictions into next iteration.
                        # .detach() avoids backpropagating through the unrolled
                        # initialisation graph; remove to enable full unrolling.
                        K_pred_it    = outputs["K_pred"].detach()
                        T_pred_it    = outputs["T_12_pred"].detach()
                        # Guard: only accept if both tensors are fully finite;
                        # if NaN crept in, keep the previous-iteration values.
                        if torch.isfinite(K_pred_it).all() and torch.isfinite(T_pred_it).all():
                            K_iter    = K_pred_it
                            T_12_iter = T_pred_it
                        else:
                            print(f"[NaN guard] iter {pose_it}: non-finite K/T pred "
                                  f"at step {step}; keeping previous init values.")

                loss, breakdown = compute_total_loss(
                    outputs, batch, cfg.get("loss", {}), K_iter,
                    use_confidence=use_confidence,
                    camera_weight=camera_weight,
                )

            # ── Loss spike / NaN guard ────────────────────────────────────
            spike_thresh = float(train_cfg.get("loss_spike_threshold", 50.0))
            if train and (not torch.isfinite(loss) or loss.item() > spike_thresh):
                nan_keys = [k for k, v in outputs.items()
                            if isinstance(v, torch.Tensor) and not torch.isfinite(v).all()]
                print(f"[skip] step {step}  loss={float(loss):.4f}  "
                      f"non-finite outputs: {nan_keys}")
                optimizer.zero_grad(set_to_none=True)
                continue

            if train:
                # Divide by accumulation steps so the effective gradient
                # magnitude stays the same regardless of grad_accum_steps.
                accum_loss = loss / grad_accum_steps
                if scaler is not None:
                    scaler.scale(accum_loss).backward()
                else:
                    accum_loss.backward()
                # Only update weights at the end of each accumulation window
                # (or on the very last batch of the epoch).
                is_last_batch = (step + 1 == len(loader))
                if (step + 1) % grad_accum_steps == 0 or is_last_batch:
                    optimizer_step(optimizer, model, grad_clip, scaler=scaler)

            totals["loss"]       += loss.item()
            totals["depth"]      += breakdown.get("depth",     0.0)
            totals["smooth"]     += breakdown.get("smooth",    0.0)
            totals["iters"]      += breakdown.get("iters",     0.0)
            totals["cam_pose"]   += breakdown.get("cam_pose",  0.0)
            totals["cam_rot"]    += breakdown.get("cam_rot",   0.0)
            totals["cam_trans"]  += breakdown.get("cam_trans", 0.0)
            totals["cam_K"]      += breakdown.get("cam_K",     0.0)

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
                n = n_batches
                avg_loss  = totals["loss"] / n
                avg_depth = totals["depth"] / n
                avg_smooth= totals["smooth"]/ n
                avg_iters = totals["iters"] / n
                avg_pix   = totals.get("pixel_consistency", 0.0) / n
                avg_cam   = totals["cam_pose"] / n
                avg_rot   = totals["cam_rot"]  / n
                avg_trans = totals["cam_trans"]/ n
                avg_camK  = totals["cam_K"]    / n
                avg_cid   = totals.get("cam_identity", 0.0) / n
                lr = optimizer.param_groups[0]["lr"]
                print(
                    f"  epoch {epoch:03d}  step {step+1:04d}/{len(loader):04d}"
                    f"  loss={avg_loss:.4f}"
                    f"  depth={avg_depth:.4f} smooth={avg_smooth:.4f} iters={avg_iters:.4f}"
                    f"  pix={avg_pix:.6f}"
                    f"  cam={avg_cam:.4f} (rot={avg_rot:.3f} trans={avg_trans:.3f} K={avg_camK:.3f} id={avg_cid:.3f})"
                    f"  lr={lr:.2e}  {elapsed:.1f}s"
                )
                if writer is not None:
                    gs = (epoch - 1) * len(loader) + step
                    writer.add_scalar("step/loss",      loss.item(), gs)
                    writer.add_scalar("step/lr",        lr,          gs)
                    for k, v in breakdown.items():
                        writer.add_scalar(f"step/{k}", v, gs)

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

    # --- TensorBoard writer ---
    log_cfg = cfg.get("logging", {})
    writer = None
    if _SummaryWriter is not None and log_cfg.get("tensorboard", True):
        run_name = log_cfg.get("run_name") or time.strftime("%Y%m%d_%H%M%S")
        log_dir  = os.path.join(
            _REPO_ROOT,
            log_cfg.get("log_dir", "runs"),
            log_cfg.get("project", "mono-depth-3DR"),
            run_name,
        )
        writer = _SummaryWriter(log_dir=log_dir)
        print(f"[train] TensorBoard log dir: {log_dir}")
    elif _SummaryWriter is None:
        print("[train] TensorBoard not available (pip install tensorboard to enable).")

    # resolve relative paths from the repo root
    root_dir = cfg["dataset"]["root_dir"]
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
    # With very few scenes the 10 % slices can be empty; fall back to last train scene.
    if not val_paths:
        val_paths = train_paths[-1:]
    if not test_paths:
        test_paths = train_paths[-1:]
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
    model_variant = arch_cfg.get("model", "v1")

    if model_variant == "vista":
        from models.model_vista.network import DepthAlignNetV2
        v_cfg = arch_cfg.get("vista", {})
        model = DepthAlignNetV2(
            dino_model         = str(v_cfg.get("dino_model",          "dinov2_vitl14")),
            freeze_dino        = bool(v_cfg.get("freeze_dino",         True)),
            depth_backbone     = str(v_cfg.get("depth_backbone",       "convnext_tiny")),
            decoder_dim        = int(v_cfg.get("decoder_dim",          768)),
            num_decoder_blocks = int(v_cfg.get("num_decoder_blocks",     4)),
            num_decoder_heads  = int(v_cfg.get("num_decoder_heads",     12)),
            depth_out_channels = int(v_cfg.get("depth_out_channels",   128)),
            decoder_hidden     = int(v_cfg.get("decoder_hidden",        256)),
            camera_head_hidden = int(v_cfg.get("camera_head_hidden",   256)),
            mast3r_ckpt        = v_cfg.get("mast3r_ckpt") or None,
        ).to(device)
    else:   # "v1" — original ConvNeXt-Tiny FPN model
        from models.model_image_depth.network import DepthAlignNet
        enc_cfg = arch_cfg.get("encoder",    {})
        att_cfg = arch_cfg.get("attention",  {})
        ref_cfg = arch_cfg.get("refinement", {})
        dec_cfg = arch_cfg.get("decoder",    {})
        cam_cfg = arch_cfg.get("camera_head", {})
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
            camera_head_hidden = int(cam_cfg.get("hidden_dim", 64)),
        ).to(device)

    # Print number of parameters and trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[train] Model parameters: {total_params:,} total, {trainable_params:,} trainable")

    optimizer  = build_optimizer(cfg, model)
    scheduler  = build_scheduler(cfg, optimizer, num_epochs=int(train_cfg["epochs"]))
    # AMP GradScaler — only meaningful on CUDA; None on CPU (disables AMP transparently).
    scaler = torch.amp.GradScaler("cuda") if torch.cuda.is_available() else None

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

    patience = int(train_cfg.get("patience", 0)) # number of epochs with no improvement to wait before stopping (0 disables)
    epochs_no_improve = 0

    # Camera-loss schedule
    pose_warmup_epochs   = int(cfg.get("loss", {}).get("pose_warmup_epochs", 0))
    camera_warmup_epochs = int(cfg.get("loss", {}).get("camera_warmup_epochs", 0))
    final_camera_w       = float(cfg.get("loss", {}).get("camera", 0.5))

    for epoch in range(start_epoch, int(train_cfg["epochs"]) + 1):
        t_epoch = time.time()

        # Disable uncertainty weighting for the first pose_warmup_epochs
        use_conf = (epoch > pose_warmup_epochs)

        # Linearly ramp camera weight from 0 → final over camera_warmup_epochs
        if camera_warmup_epochs > 0:
            cam_w = final_camera_w * min(1.0, epoch / camera_warmup_epochs)
        else:
            cam_w = final_camera_w
        if epoch == start_epoch or epoch == pose_warmup_epochs + 1:
            print(f"[train] epoch {epoch}: use_confidence={use_conf}  "
                  f"camera_weight={cam_w:.4f}")

        train_metrics = run_epoch(
            model, train_loader, optimizer, cfg, device, train=True, epoch=epoch,
            writer=writer, use_confidence=use_conf, camera_weight=cam_w, scaler=scaler,
        )

        val_metrics = {}
        if epoch % val_every == 0:
            val_metrics = run_epoch(
                model, val_loader, None, cfg, device, train=False, epoch=epoch,
                writer=writer, use_confidence=use_conf, camera_weight=cam_w, scaler=scaler,
            )

        # --- scheduler step ---
        # Build validation breakdown string (mirrors train_break format)
        val_break = ""
        if val_metrics:
            val_break = (
                f"  val_loss={val_metrics.get('loss', float('nan')):.4f} "
                f"depth={val_metrics.get('depth', 0.0):.4f} "
                f"smooth={val_metrics.get('smooth', 0.0):.4f} "
                f"iters={val_metrics.get('iters', 0.0):.4f} "
                f"pix={val_metrics.get('pixel_consistency', 0.0):.6f} "
                f"cam={val_metrics.get('cam_pose', 0.0):.4f} "
                f"cam_rot={val_metrics.get('cam_rot', 0.0):.4f} "
                f"cam_trans={val_metrics.get('cam_trans', 0.0):.4f}"
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
#        if val_metrics:
#            val_str = f"  val_loss={val_metrics.get('loss', float('nan')):.4f}"
#            if "abs_rel" in val_metrics:
#                val_str += (
#                    f"  abs_rel={val_metrics['abs_rel']:.4f}"
#                    f"  rmse={val_metrics['rmse']:.4f}"
#                    f"  delta1={val_metrics['delta1']:.4f}"
#                )
        # Build a compact train breakdown string from available keys
        train_break = (
            f"train_loss={train_metrics.get('loss', float('nan')):.4f} "
            f"depth={train_metrics.get('depth', 0.0):.4f} "
            f"smooth={train_metrics.get('smooth', 0.0):.4f} "
            f"iters={train_metrics.get('iters', 0.0):.4f} "
            f"pix={train_metrics.get('pixel_consistency', 0.0):.4f} "
            f"cam={train_metrics.get('cam_pose', 0.0):.4f} "
            f"cam_rot={train_metrics.get('cam_rot', 0.0):.4f} "
            f"cam_trans={train_metrics.get('cam_trans', 0.0):.4f}"
        )
        print(
            f"[epoch {epoch:03d}/{train_cfg['epochs']}]  {train_break}"
            f"{val_break}  lr={lr:.2e}  {elapsed:.1f}s"
        )

        # --- TensorBoard per-epoch scalars ---
        if writer is not None:
            writer.add_scalar("epoch/lr", lr, epoch)
            for k, v in train_metrics.items():
                writer.add_scalar(f"epoch/train_{k}", v, epoch)
            for k, v in val_metrics.items():
                writer.add_scalar(f"epoch/val_{k}", v, epoch)

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
            "arch":       arch_cfg,
        }

        # always save last
        save_checkpoint(state, os.path.join(save_dir, "last.pt"))

        if is_better(current, best_metric):
            best_metric = current
            state["best_metric"] = best_metric
            save_checkpoint(state, os.path.join(save_dir, "best.pt"))
            print(f"  [ckpt] New best {monitor}={best_metric:.4f}  →  {save_dir}/best.pt")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if patience > 0 and epochs_no_improve >= patience:
                print(f"[train] Early stopping (no improvement for {epochs_no_improve} epochs), patience={patience}")
                save_checkpoint(state, os.path.join(save_dir, f"epoch_{epoch:04d}_early_stop.pt"))
                break

        epoch_path = os.path.join(save_dir, f"epoch_{epoch:04d}.pt")
        save_checkpoint(state, epoch_path)
        keep_top_k(save_dir, monitor, top_k)

    print(f"\n[train] Done. Best {monitor}={best_metric:.4f}")
    if writer is not None:
        writer.close()

    # --- Final test evaluation on best checkpoint ---
    print("\n[test] Loading best checkpoint for final evaluation...")
    best_ckpt = os.path.join(save_dir, "best.pt")
    if os.path.isfile(best_ckpt):
        ckpt = torch.load(best_ckpt, map_location=device)
        model.load_state_dict(ckpt["model"])
    test_metrics = run_epoch(model, test_loader, None, cfg, device, train=False, epoch=0)

    test_break = ""
    if test_metrics:
        test_break = (
            f"  test_loss={test_metrics.get('loss', float('nan')):.4f} "
            f"depth={test_metrics.get('depth', 0.0):.4f} "
            f"smooth={test_metrics.get('smooth', 0.0):.4f} "
            f"iters={test_metrics.get('iters', 0.0):.4f} "
            f"pix={test_metrics.get('pixel_consistency', 0.0):.6f} "
            f"cam={test_metrics.get('cam_pose', 0.0):.4f} "
            f"cam_rot={test_metrics.get('cam_rot', 0.0):.4f} "
            f"cam_trans={test_metrics.get('cam_trans', 0.0):.4f}"
        )

    print(
        f"[test]{test_break}"
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
