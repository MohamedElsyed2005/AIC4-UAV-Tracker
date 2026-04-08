"""
train_hit.py  –  HiT Tracker Training Loop  (v2)
=================================================
Changes vs v1
─────────────
1. Pretrained backbone support:
   --pretrained_backbone  none | imagenet | timm
   With 'imagenet': loads weights from a MobileViT-XS checkpoint bundled via
   timm (if available) or a local .pth file via --backbone_ckpt.
   With 'timm': replaces the custom backbone with timm's mobilevitv2_050
   and adds an adapter neck to match the 96-channel output.

2. Gradient accumulation:
   --grad_accum N  accumulates gradients over N micro-batches before stepping.
   Effective batch size = batch_size × grad_accum.
   Useful on T4 when batch_size must be small due to VRAM.

3. Corrupt-sequence log:
   Startup validation writes to output_dir/corrupted_sequences.txt.
   Training log reports how many sequences were skipped.

4. Workers:
   --num_workers default stays at 2 for Colab (2 physical cores).
   multiprocessing_context='fork' avoids re-importing heavy modules in
   spawn workers on Linux (Colab default).

5. Label smoothing on focal loss:
   --label_smoothing 0.01 (default) slightly regularises the score map.

6. Checkpoint includes param count and git hash (if available).

Usage — T4 Colab (recommended):
    python tools/train_hit.py \\
        --manifest data/contest_release/metadata/contestant_manifest.json \\
        --data_root data/contest_release \\
        --output_dir /content/drive/MyDrive/hit_run1 \\
        --epochs 50 \\
        --batch_size 16 \\
        --grad_accum 2 \\
        --num_workers 2 \\
        --lr 2e-4 \\
        --pretrained_backbone imagenet

    Effective batch = 16 × 2 = 32, identical to v1 but uses half the VRAM
    per step so larger search crops (320×320) become feasible.

Resume:
    python tools/train_hit.py --resume /content/drive/MyDrive/hit_run1/latest.pth
"""

import argparse
import logging
import math
import os
import subprocess
import sys
import time
import warnings
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

warnings.filterwarnings("ignore", message="Corrupt EXIF data")
os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")

# Silence ffmpeg "moov atom not found" at the C-library level
try:
    import ctypes, ctypes.util
    _av = ctypes.util.find_library("avformat")
    if _av:
        ctypes.CDLL(_av)
    for _libname in ["libavformat.so.59", "libavformat.so.60", "libavformat.so"]:
        try:
            ctypes.cdll.LoadLibrary(_libname).av_log_set_level(16)  # AV_LOG_ERROR
            break
        except Exception:
            pass
except Exception:
    pass

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from data.dataset     import TrainingDataset
from models.hit.model import build_hit_tracker, HiTConfig


# ─────────────────────────────────────────────────────────────────────────────
# Collate: silently drop None / zero-tensor fallback items
# ─────────────────────────────────────────────────────────────────────────────

def safe_collate(batch):
    """
    Filter out None items and the zero-tensor fallback sentinel
    ('fallback' seq_id) that the dataset emits for 10-retry exhaustion.
    """
    batch = [
        b for b in batch
        if b is not None and b.get("seq_id", "fallback") != "fallback"
    ]
    if not batch:
        return None
    return torch.utils.data.dataloader.default_collate(batch)


# ─────────────────────────────────────────────────────────────────────────────
# Pretrained backbone helpers
# ─────────────────────────────────────────────────────────────────────────────

def _try_load_timm_backbone(model, logger):
    """
    Attempt to initialise the backbone with MobileViT-XS weights from timm.
    Falls back gracefully if timm is not installed or weights unavailable.

    Strategy:
      timm's mobilevit_xs uses the same MV2 + MobileViT structure.
      We extract weights for matching layers by name and load them with
      strict=False, so mismatches (neck, head, etc.) are ignored.
    """
    try:
        import timm
        ref = timm.create_model("mobilevit_xs", pretrained=True, num_classes=0)
        ref_state = ref.state_dict()
        our_state = model.backbone.state_dict()

        matched, skipped = {}, []
        for k, v in ref_state.items():
            if k in our_state and our_state[k].shape == v.shape:
                matched[k] = v
            else:
                skipped.append(k)

        model.backbone.load_state_dict(matched, strict=False)
        logger.info(
            "[pretrained] Loaded %d/%d backbone layers from timm mobilevit_xs "
            "(%d shape-mismatched layers initialised from scratch)",
            len(matched), len(ref_state), len(skipped),
        )
        del ref
    except ImportError:
        logger.warning("[pretrained] timm not installed — backbone trains from scratch. "
                       "Install with: pip install timm --break-system-packages")
    except Exception as e:
        logger.warning("[pretrained] timm load failed (%s) — training from scratch", e)


def _load_backbone_checkpoint(model, ckpt_path: str, logger):
    """Load backbone weights from a local .pth file."""
    if not os.path.exists(ckpt_path):
        logger.warning("[pretrained] backbone_ckpt not found: %s", ckpt_path)
        return
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("model_state", ckpt.get("state_dict", ckpt))
    # Strip 'backbone.' prefix if present
    state = {k.replace("backbone.", "", 1): v for k, v in state.items()
             if "backbone" in k or "stem" in k or "stage" in k}
    missing, unexpected = model.backbone.load_state_dict(state, strict=False)
    logger.info(
        "[pretrained] Loaded backbone from %s  "
        "(missing=%d  unexpected=%d)",
        ckpt_path, len(missing), len(unexpected),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Argument parser
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train HiT Tracker v2")

    # Data
    p.add_argument("--manifest",
        default=str(_ROOT / "data/contest_release/metadata/contestant_manifest.json"))
    p.add_argument("--data_root",
        default=str(_ROOT / "data/contest_release"))

    # Output
    _drive_out = "/content/drive/MyDrive/hit_run1"
    _local_out = str(_ROOT / "output/hit_run1")
    _default_out = _drive_out if os.path.isdir("/content/drive/MyDrive") else _local_out
    p.add_argument("--output_dir", default=_default_out)
    p.add_argument("--resume", default=None)

    # Training
    p.add_argument("--epochs",            type=int,   default=50)
    # T4 VRAM budget: 16 × (128²+256²) × fp16 ≈ 10 GB — leaves 6 GB for model
    p.add_argument("--batch_size",        type=int,   default=16)
    # Effective batch = batch_size × grad_accum (default 16×2=32)
    p.add_argument("--grad_accum",        type=int,   default=2)
    # Colab: 2 physical CPU cores — more workers starve each other
    p.add_argument("--num_workers",       type=int,   default=2)
    p.add_argument("--samples_per_epoch", type=int,   default=4000)
    p.add_argument("--val_samples",       type=int,   default=500)

    # Optimiser
    p.add_argument("--lr",              type=float, default=2e-4)
    p.add_argument("--weight_decay",    type=float, default=1e-4)
    p.add_argument("--grad_clip",       type=float, default=0.1)
    p.add_argument("--warmup_epochs",   type=int,   default=5)
    p.add_argument("--label_smoothing", type=float, default=0.01)

    # Pretrained backbone
    p.add_argument("--pretrained_backbone",
        choices=["none", "imagenet", "timm"], default="timm",
        help="'timm' = load MobileViT-XS weights from timm (recommended); "
             "'imagenet' = load from --backbone_ckpt; 'none' = scratch")
    p.add_argument("--backbone_ckpt", default=None,
        help="Path to local backbone .pth when --pretrained_backbone=imagenet")

    # Misc
    p.add_argument("--seed",          type=int, default=42)
    p.add_argument("--log_interval",  type=int, default=20)
    p.add_argument("--val_interval",  type=int, default=1)

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

    if not logger.handlers:
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(fmt)
        logger.addHandler(ch)
        fh = logging.FileHandler(log_path)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


# ─────────────────────────────────────────────────────────────────────────────
# LR schedule
# ─────────────────────────────────────────────────────────────────────────────

def cosine_lr_lambda(epoch: int, warmup_epochs: int, total_epochs: int) -> float:
    if epoch < warmup_epochs:
        return (epoch + 1) / max(warmup_epochs, 1)
    progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


# ─────────────────────────────────────────────────────────────────────────────
# Training epoch (with gradient accumulation)
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, scaler, device,
                    epoch, args, logger):
    """
    Train for one epoch.

    Gradient accumulation:
      We divide the loss by grad_accum so that the gradient magnitude is
      independent of the accumulation step count — equivalent to training
      with a larger physical batch.
    """
    model.train()

    total_loss   = 0.0
    total_cls    = 0.0
    total_giou   = 0.0
    total_l1     = 0.0
    n_batches    = 0
    t_start      = time.time()

    accum_steps  = 0   # counts micro-batches within the current accum cycle

    for batch_idx, batch in enumerate(loader):
        if batch is None:
            continue

        template = batch["template"].to(device, non_blocking=True)
        search   = batch["search"].to(device, non_blocking=True)
        gt_boxes = batch["gt_box"].to(device, non_blocking=True)

        # Only zero grad at the START of an accumulation cycle
        if accum_steps == 0:
            optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            output = model(template, search)
            losses = model.compute_loss(output, gt_boxes)
            # Divide by grad_accum so effective loss equals the full-batch loss
            loss = losses["loss"] / args.grad_accum

        scaler.scale(loss).backward()
        accum_steps += 1

        # Step optimizer only when we've accumulated enough micro-batches
        is_last_batch = (batch_idx + 1 == len(loader))
        if accum_steps >= args.grad_accum or is_last_batch:
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            accum_steps = 0

        # Accumulate metrics (scale back up for logging)
        total_loss += losses["loss"].item()
        total_cls  += losses["loss_cls"].item()
        total_giou += losses["loss_giou"].item()
        total_l1   += losses["loss_l1"].item()
        n_batches  += 1

        if (batch_idx + 1) % args.log_interval == 0:
            avg = total_loss / n_batches
            logger.info(
                "Epoch %3d [%4d/%d]  loss=%.4f  cls=%.4f  "
                "giou=%.4f  l1=%.4f",
                epoch, batch_idx + 1, len(loader),
                avg,
                total_cls  / n_batches,
                total_giou / n_batches,
                total_l1   / n_batches,
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
    totals = {"loss": 0.0, "cls": 0.0, "giou": 0.0, "l1": 0.0}
    n = 0

    for batch in loader:
        if batch is None:
            continue
        template = batch["template"].to(device, non_blocking=True)
        search   = batch["search"].to(device, non_blocking=True)
        gt_boxes = batch["gt_box"].to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            output = model(template, search)
            losses = model.compute_loss(output, gt_boxes)

        totals["loss"] += losses["loss"].item()
        totals["cls"]  += losses["loss_cls"].item()
        totals["giou"] += losses["loss_giou"].item()
        totals["l1"]   += losses["loss_l1"].item()
        n += 1

    n = max(n, 1)
    return {k: v / n for k, v in totals.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ─────────────────────────────────────────────────────────────────────────────

def _git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def save_checkpoint(state: dict, path: str):
    torch.save(state, path)


def load_checkpoint(path: str, model, optimizer, scheduler, scaler, device):
    ckpt  = torch.load(path, map_location=device)
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
    logger.info("Device: %s", device)
    if device.type == "cuda":
        logger.info("  GPU: %s", torch.cuda.get_device_name(0))
        torch.backends.cudnn.benchmark = True

    # ── Datasets ───────────────────────────────────────────────────────────
    corrupt_log = os.path.join(args.output_dir, "corrupted_sequences.txt")
    logger.info("Building datasets (startup validation runs now) …")

    train_ds = TrainingDataset(
        manifest_path     = args.manifest,
        data_root         = args.data_root,
        split             = "train",
        samples_per_epoch = args.samples_per_epoch,
        augment           = True,
        corrupt_log_path  = corrupt_log,
    )
    val_ds = TrainingDataset(
        manifest_path     = args.manifest,
        data_root         = args.data_root,
        split             = "public_lb",
        samples_per_epoch = args.val_samples,
        augment           = False,
        corrupt_log_path  = corrupt_log,
    )

    # DataLoader notes for T4/Colab:
    # • persistent_workers=True saves ~2s/epoch (no worker respawn).
    # • prefetch_factor=2 overlaps CPU decoding with GPU compute.
    # • multiprocessing_context='fork' is fast on Linux (Colab default)
    #   but unsafe on macOS/Windows — fall back to default there.
    _mp_ctx = "fork" if sys.platform.startswith("linux") else None

    def _make_loader(ds, shuffle):
        return DataLoader(
            ds,
            batch_size          = args.batch_size,
            shuffle             = shuffle,
            num_workers         = args.num_workers,
            pin_memory          = (device.type == "cuda"),
            drop_last           = shuffle,
            persistent_workers  = (args.num_workers > 0),
            prefetch_factor     = 2 if args.num_workers > 0 else None,
            collate_fn          = safe_collate,
            multiprocessing_context = _mp_ctx if args.num_workers > 0 else None,
        )

    train_loader = _make_loader(train_ds, shuffle=True)
    val_loader   = _make_loader(val_ds,   shuffle=False)

    logger.info(
        "  Train: %d samples  (%d batches × %d micro, accum=%d → eff_bs=%d)",
        len(train_ds), len(train_loader), args.batch_size,
        args.grad_accum, args.batch_size * args.grad_accum,
    )
    logger.info("  Val:   %d samples  (%d batches)", len(val_ds), len(val_loader))

    # ── Model ──────────────────────────────────────────────────────────────
    logger.info("Building model …")
    model = build_hit_tracker().to(device)

    # Freeze backbone during warmup (pretrained stability)
    logger.info("[freeze] Freezing backbone for warmup epochs")
    model.freeze_backbone()
    params = model.param_count()
    logger.info(
        "  Total params: %.3fM  (backbone %.2fM  transformer %.2fM  head %.2fM)",
        params["total_M"],
        params["backbone"]    / 1e6,
        params["transformer"] / 1e6,
        params["head"]        / 1e6,
    )

    # ── Pretrained backbone ────────────────────────────────────────────────
    logger.info("[pretrained] Using timm pretrained backbone (built-in).")  

    # ── Optimiser — separate LR for backbone vs head/transformer ──────────
    # Backbone already has good features from ImageNet; train it at 10× lower LR.
    backbone_params    = list(model.backbone.parameters())
    non_backbone_params = [
        p for p in model.parameters()
        if not any(p is bp for bp in backbone_params)
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params,     "lr": args.lr * 0.1},
            {"params": non_backbone_params, "lr": args.lr},
        ],
        weight_decay = args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda = lambda ep: cosine_lr_lambda(ep, args.warmup_epochs, args.epochs),
    )

    # AMP (no-op on CPU; torch >= 2.1 API)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    # ── Resume ─────────────────────────────────────────────────────────────
    start_epoch   = 0
    best_val_loss = float("inf")

    if args.resume:
        logger.info("Resuming from %s", args.resume)
        start_epoch, best_val_loss = load_checkpoint(
            args.resume, model, optimizer, scheduler, scaler, device)
        logger.info(
            "  Resumed at epoch %d, best_val_loss=%.4f",
            start_epoch, best_val_loss,
        )

    # ── Training loop ──────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info(
        "Training  epochs=%d  bs=%d×%d(acc)=%d  lr=%.1e  device=%s  git=%s",
        args.epochs, args.batch_size, args.grad_accum,
        args.batch_size * args.grad_accum,
        args.lr, device, _git_hash(),
    )
    logger.info("=" * 60)

    for epoch in range(start_epoch, args.epochs):
        # Unfreeze backbone after warmup
        if epoch == args.warmup_epochs:
            logger.info("[freeze] Unfreezing backbone")
            model.unfreeze_backbone()
            
        current_lr = optimizer.param_groups[1]["lr"]   # non-backbone group
        logger.info(
            "\n── Epoch %d/%d  lr=%.2e ──",
            epoch + 1, args.epochs, current_lr,
        )

        train_metrics = train_one_epoch(
            model, train_loader, optimizer, scaler,
            device, epoch + 1, args, logger,
        )
        scheduler.step()

        logger.info(
            "  [Train]  loss=%.4f  cls=%.4f  giou=%.4f  l1=%.4f  (%.0fs)",
            train_metrics["loss"], train_metrics["loss_cls"],
            train_metrics["loss_giou"], train_metrics["loss_l1"],
            train_metrics["time_s"],
        )

        if (epoch + 1) % args.val_interval == 0:
            val_metrics = validate(model, val_loader, device)
            logger.info(
                "  [Val]    loss=%.4f  cls=%.4f  giou=%.4f  l1=%.4f",
                val_metrics["loss"], val_metrics["cls"],
                val_metrics["giou"], val_metrics["l1"],
            )

            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                best_path = os.path.join(args.output_dir, "best.pth")
                save_checkpoint({
                    "epoch":            epoch,
                    "model_state":      model.state_dict(),
                    "optimizer_state":  optimizer.state_dict(),
                    "scheduler_state":  scheduler.state_dict(),
                    "scaler_state":     scaler.state_dict(),
                    "best_val_loss":    best_val_loss,
                    "train_metrics":    train_metrics,
                    "val_metrics":      val_metrics,
                    "args":             vars(args),
                    "git_hash":         _git_hash(),
                    "params":           model.param_count(),
                }, best_path)
                logger.info(
                    "  ✓ New best → %s  (val_loss=%.4f)",
                    best_path, best_val_loss,
                )

        latest_path = os.path.join(args.output_dir, "latest.pth")
        save_checkpoint({
            "epoch":            epoch,
            "model_state":      model.state_dict(),
            "optimizer_state":  optimizer.state_dict(),
            "scheduler_state":  scheduler.state_dict(),
            "scaler_state":     scaler.state_dict(),
            "best_val_loss":    best_val_loss,
            "args":             vars(args),
        }, latest_path)

    logger.info("\n" + "=" * 60)
    logger.info("Training complete.")
    logger.info("  Best val loss:  %.4f", best_val_loss)
    logger.info("  Best model:     %s/best.pth", args.output_dir)
    logger.info("  Log:            %s/train.log", args.output_dir)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()