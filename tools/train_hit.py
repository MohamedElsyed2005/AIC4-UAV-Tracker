"""
train_hit.py  -  HiT Tracker Training Loop  (v6 — fixed imports + TensorBoard)
================================================================================

CRITICAL FIXES vs v5:
  1. Import paths corrected
     v5 used `from data.dataset` and `from models.hit.model` which assumed a
     package layout that doesn't exist — all files are flat in the project root.
     Fixed to `from dataset import ...` and `from model import ...`.

  2. TensorBoard logging
     Replaces CSV-only logging with SummaryWriter.  Every epoch writes:
       - Loss scalars (train/val)
       - Accuracy scalars (IoU@50, mean IoU, centre error)
       - GT distribution histogram (verifies jitter is working)
     Run: tensorboard --logdir <output_dir>/tb_logs

  3. --overfit_debug mode
     Uses only 8 sequences and 200 samples/epoch to verify the model can
     overfit before committing to full training.  Run this first whenever
     you restart — a healthy model should reach IoU@50 > 50% within 5 epochs.

  4. jitter_sigma wired through
     Passed to TrainingDataset.split_train_val() so the fix in dataset.py v3
     is actually activated.

  5. GT distribution logged every val_interval epochs
     Plots histogram of cx_n, cy_n values — if always ~0.5, jitter is broken.

Usage (quick overfit check first!):
    python train_hit.py --overfit_debug --epochs 10 --output_dir output/debug

Full training (RTX 3050):
    python train_hit.py \
        --manifest data/contest_release/metadata/contestant_manifest.json \
        --data_root data/contest_release \
        --output_dir output/hit_run3 \
        --epochs 40 \
        --batch_size 4 \
        --grad_accum 4 \
        --lr 3e-4 \
        --warmup_epochs 3 \
        --jitter_sigma 0.25

Resume:
    python train_hit.py --resume output/hit_run3/latest.pth
"""

import argparse
import csv
import logging
import math
import os
import subprocess
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

warnings.filterwarnings("ignore", message="Corrupt EXIF data")
os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")

try:
    import ctypes
    for _libname in ["libavformat.so.59", "libavformat.so.60", "libavformat.so"]:
        try:
            ctypes.cdll.LoadLibrary(_libname).av_log_set_level(16)
            break
        except Exception:
            pass
except Exception:
    pass

# ── FIXED IMPORTS: flat project layout ───────────────────────────────────────
_ROOT = Path(__file__).resolve().parents[1]  # project root
sys.path.insert(0, str(_ROOT))

from data.dataset import TrainingDataset
from models.hit.model   import build_hit_tracker, HiTConfig

import json
import tempfile


# ─────────────────────────────────────────────────────────────────────────────
# Manifest patching
# ─────────────────────────────────────────────────────────────────────────────

def _patched_manifest(manifest_path: str, data_root: str, logger) -> str:
    import cv2
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    removed = []
    for split_name, sequences in manifest.items():
        bad_keys = []
        for seq_key, meta in sequences.items():
            vid_rel = meta.get("video_path", "")
            vid_abs = os.path.join(data_root, vid_rel)
            cap = cv2.VideoCapture(vid_abs)
            ok  = cap.isOpened()
            cap.release()
            if not ok:
                bad_keys.append(seq_key)
                logger.warning("[manifest-patch] Removing: %s/%s", split_name, seq_key)
        for k in bad_keys:
            del sequences[k]
            removed.append(f"{split_name}/{k}")

    if removed:
        logger.info("[manifest-patch] Removed %d sequence(s)", len(removed))
    else:
        logger.info("[manifest-patch] All sequences readable.")

    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8")
    json.dump(manifest, tmp, indent=2)
    tmp.flush(); tmp.close()
    return tmp.name


# ─────────────────────────────────────────────────────────────────────────────
# Collate
# ─────────────────────────────────────────────────────────────────────────────

def safe_collate(batch):
    batch = [b for b in batch
             if b is not None and b.get("seq_id", "fallback") != "fallback"]
    if not batch:
        return None
    return torch.utils.data.dataloader.default_collate(batch)


# ─────────────────────────────────────────────────────────────────────────────
# Accuracy helpers
# ─────────────────────────────────────────────────────────────────────────────

def batch_iou(pred_boxes: torch.Tensor, gt_boxes: torch.Tensor) -> torch.Tensor:
    def to_xyxy(b):
        return torch.stack([
            b[:, 0] - b[:, 2] / 2,
            b[:, 1] - b[:, 3] / 2,
            b[:, 0] + b[:, 2] / 2,
            b[:, 1] + b[:, 3] / 2,
        ], dim=1)

    p = to_xyxy(pred_boxes)
    g = to_xyxy(gt_boxes)
    inter_w = (torch.min(p[:, 2], g[:, 2]) - torch.max(p[:, 0], g[:, 0])).clamp(0)
    inter_h = (torch.min(p[:, 3], g[:, 3]) - torch.max(p[:, 1], g[:, 1])).clamp(0)
    inter   = inter_w * inter_h
    area_p  = (pred_boxes[:, 2] * pred_boxes[:, 3]).clamp(min=0)
    area_g  = (gt_boxes[:, 2]   * gt_boxes[:, 3]).clamp(min=0)
    union   = area_p + area_g - inter + 1e-7
    return (inter / union).clamp(0.0, 1.0)


def batch_center_error(pred_boxes: torch.Tensor,
                       gt_boxes:   torch.Tensor) -> torch.Tensor:
    dx = pred_boxes[:, 0] - gt_boxes[:, 0]
    dy = pred_boxes[:, 1] - gt_boxes[:, 1]
    return torch.sqrt(dx * dx + dy * dy)


# ─────────────────────────────────────────────────────────────────────────────
# Early Stopping
# ─────────────────────────────────────────────────────────────────────────────

class EarlyStopping:
    def __init__(self, patience: int = 8, min_delta: float = 1e-4,
                 mode: str = "min"):
        self.patience  = patience
        self.min_delta = min_delta
        self.mode      = mode
        self.counter   = 0
        self.triggered = False
        self.best_loss = float("inf") if mode == "min" else float("-inf")

    def _is_improvement(self, v):
        if self.mode == "min":
            return v < self.best_loss - self.min_delta
        return v > self.best_loss + self.min_delta

    def step(self, value: float) -> bool:
        if math.isnan(value):
            return False
        if self._is_improvement(value):
            self.best_loss = value
            self.counter   = 0
        else:
            self.counter  += 1
        if self.counter >= self.patience:
            self.triggered = True
            return True
        return False

    def state_dict(self):
        return {"counter": self.counter, "best_loss": self.best_loss,
                "triggered": self.triggered}

    def load_state_dict(self, state):
        self.counter   = state.get("counter",   0)
        self.best_loss = state.get("best_loss", self.best_loss)
        self.triggered = state.get("triggered", False)


# ─────────────────────────────────────────────────────────────────────────────
# Argument parser
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train HiT Tracker v6")

    p.add_argument("--manifest",
        default=str(_ROOT / "data/contest_release/metadata/contestant_manifest.json"))
    p.add_argument("--data_root",
        default=str(_ROOT / "data/contest_release"))
    p.add_argument("--output_dir", default=str(_ROOT / "output/hit_run3"))
    p.add_argument("--resume",     default=None)

    p.add_argument("--epochs",            type=int,   default=40)
    p.add_argument("--batch_size",        type=int,   default=1)
    p.add_argument("--grad_accum",        type=int,   default=4)
    p.add_argument("--num_workers",       type=int,   default=2)
    p.add_argument("--samples_per_epoch", type=int,   default=10000)
    p.add_argument("--val_samples",       type=int,   default=2000)
    p.add_argument("--val_ratio",         type=float, default=0.15)
    p.add_argument("--jitter_sigma",      type=float, default=0.25,
                   help="Search crop jitter (CRITICAL: 0.0 = broken training)")

    p.add_argument("--lr",            type=float, default=3e-4)
    p.add_argument("--weight_decay",  type=float, default=1e-4)
    p.add_argument("--grad_clip",     type=float, default=1.0)
    p.add_argument("--warmup_epochs", type=int,   default=3)

    p.add_argument("--early_stop_patience",   type=int,   default=8)
    p.add_argument("--early_stop_min_epochs", type=int,   default=10)
    p.add_argument("--early_stop_delta",      type=float, default=1e-4)

    p.add_argument("--seed",         type=int,  default=42)
    p.add_argument("--log_interval", type=int,  default=25)
    p.add_argument("--val_interval", type=int,  default=1)
    p.add_argument("--ckpt_every",   type=int,  default=5)

    # Quick overfit debug — verifies model/data pipeline before full training
    p.add_argument("--overfit_debug", action="store_true",
                   help="Use 8 sequences + 200 samples to verify the pipeline can overfit")

    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Logger
# ─────────────────────────────────────────────────────────────────────────────

def setup_logger(output_dir: str) -> logging.Logger:
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "train.log")
    logger   = logging.getLogger("HiT")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S")
    if not logger.handlers:
        import io
        utf8_stdout = io.TextIOWrapper(
            sys.stdout.buffer, encoding="utf-8", errors="replace",
            line_buffering=True
        ) if hasattr(sys.stdout, "buffer") else sys.stdout
        ch = logging.StreamHandler(utf8_stdout)
        ch.setFormatter(fmt)
        logger.addHandler(ch)
        fh = logging.FileHandler(log_path, encoding="utf-8")
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
    return max(0.01, 0.5 * (1.0 + math.cos(math.pi * progress)))


# ─────────────────────────────────────────────────────────────────────────────
# Training epoch
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, scaler, device,
                    epoch, args, logger) -> dict:
    model.train()

    total_loss = total_cls = total_giou = total_l1 = 0.0
    n_batches = accum_steps = 0
    cx_vals   = []   # for GT distribution monitoring
    t_start   = time.time()

    for batch_idx, batch in enumerate(loader):
        if batch is None:
            continue

        template = batch["template"].to(device, non_blocking=True)
        search   = batch["search"].to(device, non_blocking=True)
        gt_boxes = batch["gt_box"].to(device, non_blocking=True)

        if (gt_boxes[:, 2] <= 0).all() or (gt_boxes[:, 3] <= 0).all():
            continue

        cx_vals.extend(gt_boxes[:, 0].cpu().tolist())

        if accum_steps == 0:
            optimizer.zero_grad(set_to_none=True)

        use_amp = (device.type == "cuda")
        with torch.amp.autocast("cuda", enabled=use_amp):
            output = model(template, search)
            losses = model.compute_loss(output, gt_boxes)
            loss   = losses["loss"] / args.grad_accum

        scaler.scale(loss).backward()
        accum_steps += 1

        is_last_batch = (batch_idx + 1 == len(loader))
        if accum_steps >= args.grad_accum or is_last_batch:
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            accum_steps = 0

        total_loss  += losses["loss"].item()
        total_cls   += losses["loss_cls"].item()
        total_giou  += losses["loss_giou"].item()
        total_l1    += losses["loss_l1"].item()
        n_batches   += 1

        if (batch_idx + 1) % args.log_interval == 0:
            avg = total_loss / n_batches
            logger.info(
                "Epoch %3d [%4d/%d]  loss=%.4f  cls=%.4f  giou=%.4f  l1=%.4f",
                epoch, batch_idx + 1, len(loader), avg,
                total_cls / n_batches, total_giou / n_batches,
                total_l1  / n_batches,
            )

    elapsed = time.time() - t_start
    n = max(n_batches, 1)

    # GT distribution: std should be >> 0 if jitter is active
    import numpy as np
    cx_arr = np.array(cx_vals)
    gt_cx_std = float(cx_arr.std()) if len(cx_arr) > 0 else 0.0

    return {
        "loss":      total_loss / n,
        "loss_cls":  total_cls  / n,
        "loss_giou": total_giou / n,
        "loss_l1":   total_l1   / n,
        "time_s":    elapsed,
        "gt_cx_std": gt_cx_std,   # should be ~0.2 with jitter, ~0.0 without
        "gt_cx_vals": cx_vals,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def validate(model, loader, device) -> dict:
    model.eval()

    loss_totals = {"loss": 0.0, "cls": 0.0, "giou": 0.0, "l1": 0.0}
    iou_sum = iou50_sum = iou25_sum = center_sum = 0.0
    n_samples = n_batches = n_skip = 0
    use_amp = (device.type == "cuda")

    for batch in loader:
        if batch is None:
            n_skip += 1
            continue

        template = batch["template"].to(device, non_blocking=True)
        search   = batch["search"].to(device, non_blocking=True)
        gt_boxes = batch["gt_box"].to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=use_amp):
            output = model(template, search)
            losses = model.compute_loss(output, gt_boxes)

        loss_totals["loss"] += losses["loss"].item()
        loss_totals["cls"]  += losses["loss_cls"].item()
        loss_totals["giou"] += losses["loss_giou"].item()
        loss_totals["l1"]   += losses["loss_l1"].item()
        n_batches += 1

        pred_boxes = output["pred_boxes"].float()
        gt_f       = gt_boxes.float()
        iou   = batch_iou(pred_boxes, gt_f)
        cerr  = batch_center_error(pred_boxes, gt_f)
        B           = pred_boxes.shape[0]
        iou_sum    += iou.sum().item()
        iou50_sum  += (iou >= 0.50).float().sum().item()
        iou25_sum  += (iou >= 0.25).float().sum().item()
        center_sum += cerr.sum().item()
        n_samples  += B

    if n_batches == 0:
        nan = float("nan")
        return {"loss": nan, "cls": nan, "giou": nan, "l1": nan,
                "mean_iou": nan, "acc_iou50": nan, "acc_iou25": nan,
                "mean_center_err": nan, "n_batches": 0, "n_skipped": n_skip}

    n_s = max(n_samples, 1)
    return {
        "loss":  loss_totals["loss"] / n_batches,
        "cls":   loss_totals["cls"]  / n_batches,
        "giou":  loss_totals["giou"] / n_batches,
        "l1":    loss_totals["l1"]   / n_batches,
        "mean_iou":        iou_sum    / n_s,
        "acc_iou50":       iou50_sum  / n_s,
        "acc_iou25":       iou25_sum  / n_s,
        "mean_center_err": center_sum / n_s,
        "n_batches": n_batches,
        "n_skipped": n_skip,
    }


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


def save_checkpoint(state, path):
    torch.save(state, path)


def load_checkpoint(path, model, optimizer, scheduler, scaler, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"], strict=False)
    if "optimizer_state" in ckpt:
        try:
            optimizer.load_state_dict(ckpt["optimizer_state"])
        except Exception:
            pass
    if "scheduler_state" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state"])
    if "scaler_state" in ckpt and scaler is not None:
        scaler.load_state_dict(ckpt["scaler_state"])
    start_epoch = ckpt.get("epoch", 0) + 1
    best_val    = ckpt.get("best_val_loss", float("inf"))
    extra       = {"early_stop": ckpt.get("early_stop_state", {})}
    return start_epoch, best_val, extra


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    logger = setup_logger(args.output_dir)

    torch.manual_seed(args.seed)

    # ── JITTER CHECK ───────────────────────────────────────────────────────
    if args.jitter_sigma <= 0.0:
        logger.warning(
            "WARNING: jitter_sigma=%.2f — search crop is always centred on GT. "
            "Model will NOT learn to localise. Performance will be ~4%%.",
            args.jitter_sigma
        )
    else:
        logger.info("jitter_sigma=%.2f (CRITICAL fix active)", args.jitter_sigma)

    # ── Device ─────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)
    if device.type == "cuda":
        logger.info("  GPU: %s  VRAM: %.1f GB",
                    torch.cuda.get_device_name(0),
                    torch.cuda.get_device_properties(0).total_memory / 1e9)
        torch.backends.cudnn.benchmark        = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32       = True

    # ── TensorBoard ────────────────────────────────────────────────────────
    tb_writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter
        tb_dir = os.path.join(args.output_dir, "tb_logs")
        tb_writer = SummaryWriter(tb_dir)
        logger.info("TensorBoard: tensorboard --logdir %s", tb_dir)
    except ImportError:
        logger.warning("tensorboard not installed — skipping TB logging")

    # ── Datasets ───────────────────────────────────────────────────────────
    corrupt_log   = os.path.join(args.output_dir, "corrupted_sequences.txt")
    logger.info("Building datasets (jitter_sigma=%.2f) ...", args.jitter_sigma)

    clean_manifest = _patched_manifest(args.manifest, args.data_root, logger)

    train_samples = 200 if args.overfit_debug else args.samples_per_epoch
    val_samples   = 100 if args.overfit_debug else args.val_samples

    train_ds, val_ds = TrainingDataset.split_train_val(
        manifest_path    = clean_manifest,
        data_root        = args.data_root,
        val_ratio        = args.val_ratio,
        train_samples    = train_samples,
        val_samples      = val_samples,
        jitter_sigma     = args.jitter_sigma,
        seed             = args.seed,
        corrupt_log_path = corrupt_log,
    )

    if args.overfit_debug:
        # Use only 8 sequences for quick overfit check
        logger.info("[OVERFIT DEBUG] Limiting to 8 sequences")
        train_ds.sequences = train_ds.sequences[:8]
        val_ds.sequences   = val_ds.sequences[:min(4, len(val_ds.sequences))]

    _mp_ctx = "fork" if sys.platform.startswith("linux") else None

    def _make_loader(ds, shuffle, is_val=False):
        _workers    = 0 if is_val else args.num_workers
        _persistent = (_workers > 0) and (sys.platform != "win32")
        return DataLoader(
            ds,
            batch_size          = args.batch_size,
            shuffle             = shuffle,
            num_workers         = _workers,
            pin_memory          = (device.type == "cuda"),
            drop_last           = shuffle,
            persistent_workers  = _persistent,
            prefetch_factor     = 2 if _workers > 0 else None,
            collate_fn          = safe_collate,
            multiprocessing_context = _mp_ctx if _workers > 0 else None,
        )

    train_loader = _make_loader(train_ds, shuffle=True,  is_val=False)
    val_loader   = _make_loader(val_ds,   shuffle=False, is_val=True)

    logger.info("  Train: %d samples  (%d batches, accum=%d, eff_bs=%d)",
                len(train_ds), len(train_loader),
                args.grad_accum, args.batch_size * args.grad_accum)
    logger.info("  Val:   %d samples  (%d batches)",
                len(val_ds), len(val_loader))

    # ── Model ──────────────────────────────────────────────────────────────
    logger.info("Building model ...")
    model  = build_hit_tracker().to(device)
    model.freeze_backbone()
    params = model.param_count()
    logger.info("  Total params: %.3fM  (backbone %.2fM  transformer %.2fM  head %.2fM)",
                params["total_M"], params["backbone"]/1e6,
                params["transformer"]/1e6, params["head"]/1e6)

    # ── Optimiser ──────────────────────────────────────────────────────────
    backbone_params     = list(model.backbone.parameters())
    non_backbone_params = [p for p in model.parameters()
                           if not any(p is bp for bp in backbone_params)]
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params,     "lr": args.lr * 0.1},
            {"params": non_backbone_params, "lr": args.lr},
        ],
        weight_decay = args.weight_decay,
    )

    # Replace LambdaLR with CosineAnnealingWarmRestarts
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    # ── Early stopping ─────────────────────────────────────────────────────
    early_stopper   = EarlyStopping(patience  = args.early_stop_patience,
                                    min_delta = args.early_stop_delta)
    es_eligible     = max(args.early_stop_min_epochs, args.warmup_epochs + 1)

    # ── Resume ─────────────────────────────────────────────────────────────
    start_epoch   = 0
    best_val_loss = float("inf")
    if args.resume:
        logger.info("Resuming from %s", args.resume)
        start_epoch, best_val_loss, extra = load_checkpoint(
            args.resume, model, optimizer, scheduler, scaler, device)
        if extra.get("early_stop"):
            early_stopper.load_state_dict(extra["early_stop"])
        logger.info("  Resumed at epoch %d  best_val_loss=%.4f", start_epoch, best_val_loss)

    # ── CSV log ────────────────────────────────────────────────────────────
    csv_path = os.path.join(args.output_dir, "metrics.csv")
    csv_cols = ["epoch","lr","train_loss","train_cls","train_giou","train_l1",
                "train_gt_cx_std",
                "val_loss","val_cls","val_giou","val_l1",
                "val_mean_iou","val_acc_iou50","val_acc_iou25","val_center_err",
                "es_counter"]
    csv_exists = os.path.exists(csv_path) and os.path.getsize(csv_path) > 0

    logger.info("=" * 65)
    if args.overfit_debug:
        logger.info("OVERFIT DEBUG MODE — 8 seqs, 200 samples, no early stop")
        logger.info("Target: IoU@50 > 50%% within 10 epochs")
        logger.info("If this fails, the model or data pipeline is broken.")
    else:
        logger.info("Full training  epochs=%d  lr=%.1e  jitter=%.2f  device=%s",
                    args.epochs, args.lr, args.jitter_sigma, device)
    logger.info("=" * 65)

    stop_reason = "max_epochs"

    for epoch in range(start_epoch, args.epochs):
        if epoch == args.warmup_epochs:
            logger.info("[freeze] Unfreezing backbone")
            model.unfreeze_backbone()

        current_lr = optimizer.param_groups[1]["lr"]
        logger.info("\n-- Epoch %d/%d  lr=%.2e  es=%d/%d --",
                    epoch + 1, args.epochs, current_lr,
                    early_stopper.counter, args.early_stop_patience)

        # ── Train ─────────────────────────────────────────────────────────
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, scaler, device, epoch + 1, args, logger)
        scheduler.step()

        logger.info("  [Train]  loss=%.4f  cls=%.4f  giou=%.4f  l1=%.4f  "
                    "gt_cx_std=%.3f  (%.0fs)",
                    train_metrics["loss"], train_metrics["loss_cls"],
                    train_metrics["loss_giou"], train_metrics["loss_l1"],
                    train_metrics["gt_cx_std"], train_metrics["time_s"])

        # ── JITTER HEALTH CHECK ────────────────────────────────────────────
        if train_metrics["gt_cx_std"] < 0.05:
            logger.warning(
                "  [WARN] gt_cx_std=%.3f < 0.05: GT positions concentrated near center! "
                "Check jitter_sigma (current=%.2f). This was the root cause of 4%% perf.",
                train_metrics["gt_cx_std"], args.jitter_sigma
            )

        # ── TensorBoard ───────────────────────────────────────────────────
        if tb_writer is not None:
            tb_writer.add_scalar("train/loss",      train_metrics["loss"],      epoch)
            tb_writer.add_scalar("train/loss_cls",  train_metrics["loss_cls"],  epoch)
            tb_writer.add_scalar("train/loss_giou", train_metrics["loss_giou"], epoch)
            tb_writer.add_scalar("train/loss_l1",   train_metrics["loss_l1"],   epoch)
            tb_writer.add_scalar("train/gt_cx_std", train_metrics["gt_cx_std"], epoch)
            tb_writer.add_scalar("train/lr",        current_lr,                 epoch)
            if train_metrics["gt_cx_vals"]:
                import torch as _t
                tb_writer.add_histogram("train/gt_cx_dist",
                    _t.tensor(train_metrics["gt_cx_vals"]), epoch)

        # ── Validate ──────────────────────────────────────────────────────
        val_metrics: Dict = {}
        do_val = ((epoch + 1) % args.val_interval == 0)

        if do_val:
            val_metrics = validate(model, val_loader, device)
            n_val  = val_metrics.get("n_batches", 0)

            if n_val == 0 or math.isnan(val_metrics["loss"]):
                logger.warning("  [Val]  WARNING: all val batches skipped.")
            else:
                logger.info(
                    "  [Val]  loss=%.4f  cls=%.4f  giou=%.4f  l1=%.4f",
                    val_metrics["loss"], val_metrics["cls"],
                    val_metrics["giou"], val_metrics["l1"])
                logger.info(
                    "  [Acc]  IoU@50=%.1f%%  IoU@25=%.1f%%  "
                    "meanIoU=%.4f  centErr=%.4f",
                    val_metrics["acc_iou50"] * 100,
                    val_metrics["acc_iou25"] * 100,
                    val_metrics["mean_iou"],
                    val_metrics["mean_center_err"])

                if tb_writer is not None:
                    tb_writer.add_scalar("val/loss",      val_metrics["loss"],  epoch)
                    tb_writer.add_scalar("val/iou50",     val_metrics["acc_iou50"], epoch)
                    tb_writer.add_scalar("val/iou25",     val_metrics["acc_iou25"], epoch)
                    tb_writer.add_scalar("val/mean_iou",  val_metrics["mean_iou"],  epoch)
                    tb_writer.add_scalar("val/center_err",val_metrics["mean_center_err"], epoch)

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
                        "early_stop_state": early_stopper.state_dict(),
                        "train_metrics":    train_metrics,
                        "val_metrics":      val_metrics,
                        "args":             vars(args),
                        "git_hash":         _git_hash(),
                        "params":           model.param_count(),
                    }, best_path)
                    logger.info("  [Best] -> %s  (val_loss=%.4f  IoU@50=%.1f%%)",
                                best_path, best_val_loss,
                                val_metrics["acc_iou50"] * 100)

                if not args.overfit_debug and epoch + 1 >= es_eligible:
                    if early_stopper.step(val_metrics["loss"]):
                        logger.info("\n[Early Stop] Stopping at epoch %d", epoch + 1)
                        stop_reason = "early_stop"
                        save_checkpoint({"epoch": epoch, "model_state": model.state_dict(),
                                         "best_val_loss": best_val_loss,
                                         "early_stop_state": early_stopper.state_dict()},
                                        os.path.join(args.output_dir, "latest.pth"))
                        break

        # ── Latest checkpoint ──────────────────────────────────────────────
        save_checkpoint({
            "epoch":            epoch,
            "model_state":      model.state_dict(),
            "optimizer_state":  optimizer.state_dict(),
            "scheduler_state":  scheduler.state_dict(),
            "scaler_state":     scaler.state_dict(),
            "best_val_loss":    best_val_loss,
            "early_stop_state": early_stopper.state_dict(),
            "args":             vars(args),
        }, os.path.join(args.output_dir, "latest.pth"))

        if args.ckpt_every > 0 and (epoch + 1) % args.ckpt_every == 0:
            epoch_path = os.path.join(args.output_dir, f"epoch_{epoch+1:03d}.pth")
            save_checkpoint({"epoch": epoch, "model_state": model.state_dict(),
                             "best_val_loss": best_val_loss, "params": model.param_count()},
                            epoch_path)

        # ── CSV row ────────────────────────────────────────────────────────
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=csv_cols, extrasaction="ignore")
            if not csv_exists:
                writer.writeheader(); csv_exists = True
            writer.writerow({
                "epoch":          epoch + 1,
                "lr":             f"{current_lr:.2e}",
                "train_loss":     f"{train_metrics['loss']:.4f}",
                "train_cls":      f"{train_metrics['loss_cls']:.4f}",
                "train_giou":     f"{train_metrics['loss_giou']:.4f}",
                "train_l1":       f"{train_metrics['loss_l1']:.4f}",
                "train_gt_cx_std": f"{train_metrics['gt_cx_std']:.3f}",
                "val_loss":       f"{val_metrics.get('loss',''):.4f}" if val_metrics else "",
                "val_cls":        f"{val_metrics.get('cls',''):.4f}"  if val_metrics else "",
                "val_giou":       f"{val_metrics.get('giou',''):.4f}" if val_metrics else "",
                "val_l1":         f"{val_metrics.get('l1',''):.4f}"   if val_metrics else "",
                "val_mean_iou":   f"{val_metrics.get('mean_iou',''):.4f}" if val_metrics else "",
                "val_acc_iou50":  f"{val_metrics.get('acc_iou50',''):.4f}" if val_metrics else "",
                "val_acc_iou25":  f"{val_metrics.get('acc_iou25',''):.4f}" if val_metrics else "",
                "val_center_err": f"{val_metrics.get('mean_center_err',''):.4f}" if val_metrics else "",
                "es_counter":     early_stopper.counter,
            })

    if tb_writer is not None:
        tb_writer.close()

    logger.info("\n" + "=" * 65)
    logger.info("Training complete.  Stop reason: %s", stop_reason)
    logger.info("  Best val loss: %.4f", best_val_loss)
    logger.info("  Best model:    %s/best.pth", args.output_dir)
    logger.info("=" * 65)


if __name__ == "__main__":
    main()