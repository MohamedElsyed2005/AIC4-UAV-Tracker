"""
train_hit.py  –  HiT Tracker Training Loop
===========================================
Trains the full HiTTracker pipeline end-to-end.

Usage:
    # From project root:
    python tools/train_hit.py

    # Custom config:
    python tools/train_hit.py \
        --manifest data/contest_release/metadata/contestant_manifest.json \
        --data_root data/contest_release \
        --output_dir output/hit_run1 \
        --epochs 50 \
        --batch_size 16 \
        --lr 1e-4

Training strategy:
  - AdamW optimizer with cosine LR schedule + linear warmup
  - Mixed-precision (AMP) when CUDA is available
  - Gradient clipping (max norm 0.1) for stable transformer training
  - Best checkpoint saved by validation loss
  - Full log written to output_dir/train.log
"""

import argparse
import logging
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# ── Project path setup ────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from data.dataset        import TrainingDataset
from models.hit.model    import build_hit_tracker, HiTConfig


# ─────────────────────────────────────────────────────────────────────────────
# Argument parser
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train HiT Tracker")

    # Data
    p.add_argument("--manifest",
        default=str(_ROOT / "data/contest_release/metadata/contestant_manifest.json"))
    p.add_argument("--data_root",
        default=str(_ROOT / "data/contest_release"))

    # Output
    p.add_argument("--output_dir",
        default=str(_ROOT / "output/hit_run1"))
    p.add_argument("--resume", default=None,
        help="Path to checkpoint .pth to resume from")

    # Training hyper-params
    p.add_argument("--epochs",          type=int,   default=50)
    p.add_argument("--batch_size",      type=int,   default=16)
    p.add_argument("--num_workers",     type=int,   default=4)
    p.add_argument("--samples_per_epoch", type=int, default=4000)
    p.add_argument("--val_samples",     type=int,   default=500)

    # Optimiser
    p.add_argument("--lr",              type=float, default=1e-4)
    p.add_argument("--weight_decay",    type=float, default=1e-4)
    p.add_argument("--grad_clip",       type=float, default=0.1)
    p.add_argument("--warmup_epochs",   type=int,   default=5)

    # Misc
    p.add_argument("--seed",            type=int,   default=42)
    p.add_argument("--log_interval",    type=int,   default=20,
        help="Print loss every N batches")
    p.add_argument("--val_interval",    type=int,   default=1,
        help="Run validation every N epochs")

    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Logger
# ─────────────────────────────────────────────────────────────────────────────

def setup_logger(output_dir: str) -> logging.Logger:
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "train.log")

    logger = logging.getLogger("HiT")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S")

    # Console
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File
    fh = logging.FileHandler(log_path)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


# ─────────────────────────────────────────────────────────────────────────────
# LR schedule: linear warmup → cosine decay
# ─────────────────────────────────────────────────────────────────────────────

def cosine_lr_lambda(epoch: int, warmup_epochs: int, total_epochs: int) -> float:
    """Returns LR multiplier for current epoch."""
    if epoch < warmup_epochs:
        return (epoch + 1) / max(warmup_epochs, 1)
    progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


# ─────────────────────────────────────────────────────────────────────────────
# One epoch of training
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, scaler, device,
                    epoch, args, logger):
    model.train()

    total_loss   = 0.0
    total_cls    = 0.0
    total_giou   = 0.0
    total_l1     = 0.0
    n_batches    = 0
    t_start      = time.time()

    for batch_idx, batch in enumerate(loader):
        template = batch["template"].to(device, non_blocking=True)  # (B,3,128,128)
        search   = batch["search"].to(device, non_blocking=True)    # (B,3,256,256)
        gt_boxes = batch["gt_box"].to(device, non_blocking=True)    # (B,4)

        optimizer.zero_grad()

        # Forward + loss under AMP
        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            output = model(template, search)
            losses = model.compute_loss(output, gt_boxes)

        loss = losses["loss"]

        # Backward
        scaler.scale(loss).backward()

        # Gradient clipping
        if args.grad_clip > 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        scaler.step(optimizer)
        scaler.update()

        # Accumulate metrics
        total_loss += loss.item()
        total_cls  += losses["loss_cls"].item()
        total_giou += losses["loss_giou"].item()
        total_l1   += losses["loss_l1"].item()
        n_batches  += 1

        # Periodic log
        if (batch_idx + 1) % args.log_interval == 0:
            avg = total_loss / n_batches
            logger.info(
                f"Epoch {epoch:3d} [{batch_idx+1:4d}/{len(loader)}]  "
                f"loss={avg:.4f}  cls={total_cls/n_batches:.4f}  "
                f"giou={total_giou/n_batches:.4f}  "
                f"l1={total_l1/n_batches:.4f}"
            )

    elapsed = time.time() - t_start
    return {
        "loss":      total_loss / max(n_batches, 1),
        "loss_cls":  total_cls  / max(n_batches, 1),
        "loss_giou": total_giou / max(n_batches, 1),
        "loss_l1":   total_l1   / max(n_batches, 1),
        "time_s":    elapsed,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def validate(model, loader, device):
    model.eval()

    total_loss  = 0.0
    total_cls   = 0.0
    total_giou  = 0.0
    total_l1    = 0.0
    n_batches   = 0

    for batch in loader:
        template = batch["template"].to(device, non_blocking=True)
        search   = batch["search"].to(device, non_blocking=True)
        gt_boxes = batch["gt_box"].to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            output = model(template, search)
            losses = model.compute_loss(output, gt_boxes)

        total_loss += losses["loss"].item()
        total_cls  += losses["loss_cls"].item()
        total_giou += losses["loss_giou"].item()
        total_l1   += losses["loss_l1"].item()
        n_batches  += 1

    return {
        "loss":      total_loss / max(n_batches, 1),
        "loss_cls":  total_cls  / max(n_batches, 1),
        "loss_giou": total_giou / max(n_batches, 1),
        "loss_l1":   total_l1   / max(n_batches, 1),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ─────────────────────────────────────────────────────────────────────────────

def save_checkpoint(state: dict, path: str):
    torch.save(state, path)


def load_checkpoint(path: str, model, optimizer, scheduler, scaler, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    scheduler.load_state_dict(ckpt["scheduler_state"])
    if "scaler_state" in ckpt and scaler is not None:
        scaler.load_state_dict(ckpt["scaler_state"])
    start_epoch = ckpt.get("epoch", 0) + 1
    best_val    = ckpt.get("best_val_loss", float("inf"))
    return start_epoch, best_val


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    logger = setup_logger(args.output_dir)

    # ── Reproducibility ────────────────────────────────────────────────────
    torch.manual_seed(args.seed)

    # ── Device ─────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    if device.type == "cuda":
        logger.info(f"  GPU: {torch.cuda.get_device_name(0)}")

    # ── Datasets ───────────────────────────────────────────────────────────
    logger.info("Building datasets …")
    train_ds = TrainingDataset(
        manifest_path     = args.manifest,
        data_root         = args.data_root,
        split             = "train",
        samples_per_epoch = args.samples_per_epoch,
        augment           = True,
    )
    val_ds = TrainingDataset(
        manifest_path     = args.manifest,
        data_root         = args.data_root,
        split             = "public_lb",       # use leaderboard split for val
        samples_per_epoch = args.val_samples,
        augment           = False,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size  = args.batch_size,
        shuffle     = True,
        num_workers = args.num_workers,
        pin_memory  = (device.type == "cuda"),
        drop_last   = True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size  = args.batch_size,
        shuffle     = False,
        num_workers = args.num_workers,
        pin_memory  = (device.type == "cuda"),
    )

    logger.info(f"  Train: {len(train_ds)} samples/epoch  "
                f"({len(train_loader)} batches)")
    logger.info(f"  Val:   {len(val_ds)} samples/epoch  "
                f"({len(val_loader)} batches)")

    # ── Model ──────────────────────────────────────────────────────────────
    logger.info("Building model …")
    model = build_hit_tracker().to(device)
    params = model.param_count()
    logger.info(f"  Total params: {params['total_M']}M  "
                f"(backbone {params['backbone']/1e6:.2f}M  "
                f"transformer {params['transformer']/1e6:.2f}M  "
                f"head {params['head']/1e6:.2f}M)")

    # ── Optimiser ──────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr           = args.lr,
        weight_decay = args.weight_decay,
    )

    # Cosine schedule with linear warmup
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda ep: cosine_lr_lambda(
            ep, args.warmup_epochs, args.epochs)
    )

    # AMP scaler (no-op on CPU)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    # ── Resume ─────────────────────────────────────────────────────────────
    start_epoch   = 0
    best_val_loss = float("inf")

    if args.resume:
        logger.info(f"Resuming from {args.resume}")
        start_epoch, best_val_loss = load_checkpoint(
            args.resume, model, optimizer, scheduler, scaler, device)
        logger.info(f"  Resumed at epoch {start_epoch}, "
                    f"best_val_loss={best_val_loss:.4f}")

    # ── Training loop ──────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info(f"Starting training  "
                f"epochs={args.epochs}  bs={args.batch_size}  "
                f"lr={args.lr}  device={device}")
    logger.info("=" * 60)

    for epoch in range(start_epoch, args.epochs):
        current_lr = optimizer.param_groups[0]["lr"]
        logger.info(f"\n── Epoch {epoch+1}/{args.epochs}  "
                    f"lr={current_lr:.2e} ──")

        # Train
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, scaler,
            device, epoch + 1, args, logger)

        scheduler.step()

        logger.info(
            f"  [Train]  loss={train_metrics['loss']:.4f}  "
            f"cls={train_metrics['loss_cls']:.4f}  "
            f"giou={train_metrics['loss_giou']:.4f}  "
            f"l1={train_metrics['loss_l1']:.4f}  "
            f"({train_metrics['time_s']:.0f}s)"
        )

        # Validate
        if (epoch + 1) % args.val_interval == 0:
            val_metrics = validate(model, val_loader, device)
            logger.info(
                f"  [Val]    loss={val_metrics['loss']:.4f}  "
                f"cls={val_metrics['loss_cls']:.4f}  "
                f"giou={val_metrics['loss_giou']:.4f}  "
                f"l1={val_metrics['loss_l1']:.4f}"
            )

            # Save best checkpoint
            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                best_path = os.path.join(args.output_dir, "best.pth")
                save_checkpoint({
                    "epoch":          epoch,
                    "model_state":    model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "scaler_state":   scaler.state_dict(),
                    "best_val_loss":  best_val_loss,
                    "train_metrics":  train_metrics,
                    "val_metrics":    val_metrics,
                    "args":           vars(args),
                }, best_path)
                logger.info(f"  ✓ New best saved → {best_path}  "
                            f"(val_loss={best_val_loss:.4f})")

        # Save latest checkpoint every epoch (for resuming)
        latest_path = os.path.join(args.output_dir, "latest.pth")
        save_checkpoint({
            "epoch":           epoch,
            "model_state":     model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "scaler_state":    scaler.state_dict(),
            "best_val_loss":   best_val_loss,
            "args":            vars(args),
        }, latest_path)

    # ── Done ───────────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("Training complete.")
    logger.info(f"  Best val loss:  {best_val_loss:.4f}")
    logger.info(f"  Best model:     {args.output_dir}/best.pth")
    logger.info(f"  Log:            {args.output_dir}/train.log")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()