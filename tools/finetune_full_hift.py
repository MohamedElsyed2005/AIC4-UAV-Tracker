"""
tools/finetune_full_hift.py
============================
V5.1 — SAFE RESUME EDITION (FINAL)
=========================

CHANGES vs V5
─────────────────────────────────────────
FIX-1  Use load_pretrained_for_resume() properly — handles all checkpoint formats
FIX-2  scheduler eta_min = lr * 0.1 (was 0.01) — prevents over-aggressive decay
FIX-3  Optional warm-start: freeze backbone in epoch 0, unfreeze in epoch 1
FIX-4  Clear comment: start_epoch = 0 is intentional for fine-tuning from best
FIX-5  Add explicit device check for AMP/scaler safety
FIX-6  Ensure EMA is saved/loaded correctly on resume

Everything else (architecture, dataset, loss functions, augmentation) is UNCHANGED.
"""
import argparse
import copy
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LinearLR, SequentialLR
from torch.utils.data import DataLoader

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.hift_full import HiFT
from data.dataset import TrainingDataset

cv2.setNumThreads(1)
cv2.ocl.setUseOpenCL(False)

# ──────────────────────────────────────────────────────────────────────────────
# Default config — SAFE RESUME DEFAULTS
# ──────────────────────────────────────────────────────────────────────────────
CFG = dict(
    # Paths
    pretrained_ckpt = str(ROOT / "checkpoints/best_miou.pth"),   # Start from best generalization
    save_dir        = str(ROOT / "checkpoints/resume_v5_1"),

    # Training — short safe resume defaults
    epochs          = 7,          # Only fine-tune a few more epochs
    batch_size      = 4,
    accum_steps     = 4,          # effective batch = 16
    lr              = 1e-6,       # 40× lower than initial, prevents NaN after unfreeze
    weight_decay    = 1e-4,
    warmup_epochs   = 1,          # Shorter warmup for resume
    grad_clip       = 1.0,        # Tighter clipping to catch explosions early
    log_freq        = 50,

    # Loss weights (unchanged from V4)
    w_giou          = 1.0,
    w_l1            = 2.0,
    w_diou          = 0.5,
    w_cls           = 0.8,
    cls_sigma_cells = 0.8,
    cls_pos_weight  = 10.0,

    # Data
    manifest        = str(ROOT / "data/contest_release/metadata/contestant_manifest.json"),
    data_root       = str(ROOT / "data/contest_release"),
    template_size   = 128,
    search_size     = 256,
    max_gap_frames  = 150,
    train_samples   = 60_000,
    val_samples     = 3_000,
    val_ratio       = 0.15,
    jitter_sigma    = 0.25,
    num_workers     = 0,          # Windows-safe: no multiprocessing deadlocks
    seed            = 42,

    # Features
    use_amp         = True,
    use_compile     = False,
    use_ema         = True,
    ema_decay       = 0.9998,
    use_swa         = False,      # Off by default for short resume
    swa_start_frac  = 0.75,

    # Unfreeze schedule: EMPTY by default for safe resume
    # The checkpoint already has backbone partially unfrozen.
    # Uncomment entries below only if you want to unfreeze MORE layers.
    unfreeze_schedule = [
        # (0, ["grader"]),
        # (2, ["backbone.layer5"]),
        # (4, ["backbone.layer4"]),
    ],
)

LATEST_CKPT = "latest.pth"
BEST_CKPT   = "best.pth"
EMA_CKPT    = "ema_best.pth"
SWA_CKPT    = "swa_final.pth"


# ──────────────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────────────
def setup_logging(save_dir: str) -> logging.Logger:
    os.makedirs(save_dir, exist_ok=True)
    log = logging.getLogger("AIC4.train")
    log.setLevel(logging.DEBUG)
    log.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S")

    fh = logging.FileHandler(
        os.path.join(save_dir, f"train_{time.strftime('%Y%m%d_%H%M%S')}.log"),
        encoding="utf-8",
    )
    fh.setFormatter(fmt)
    log.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    log.addHandler(ch)
    return log


# ──────────────────────────────────────────────────────────────────────────────
# Worker init
# ──────────────────────────────────────────────────────────────────────────────
def _worker_init(worker_id: int):
    import random
    import numpy as np
    cv2.setNumThreads(1)
    cv2.ocl.setUseOpenCL(False)
    seed = worker_id + int(time.time() * 1000) % 100_000
    random.seed(seed)
    np.random.seed(seed)


# ──────────────────────────────────────────────────────────────────────────────
# EMA
# ──────────────────────────────────────────────────────────────────────────────
class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.9998):
        self.decay  = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module):
        for s, m in zip(self.shadow.parameters(), model.parameters()):
            s.copy_(self.decay * s + (1.0 - self.decay) * m.float())
        for s, m in zip(self.shadow.buffers(), model.buffers()):
            s.copy_(m)

    def state_dict(self):           return self.shadow.state_dict()
    def load_state_dict(self, sd):  self.shadow.load_state_dict(sd)


# ──────────────────────────────────────────────────────────────────────────────
# Loss functions (UNCHANGED from V4)
# ──────────────────────────────────────────────────────────────────────────────
def _to_xyxy(b: torch.Tensor):
    return (b[..., 0] - b[..., 2] / 2,
            b[..., 1] - b[..., 3] / 2,
            b[..., 0] + b[..., 2] / 2,
            b[..., 1] + b[..., 3] / 2)


def giou_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    px1, py1, px2, py2 = _to_xyxy(pred)
    tx1, ty1, tx2, ty2 = _to_xyxy(target)
    inter = ((torch.min(px2, tx2) - torch.max(px1, tx1)).clamp(0) *
             (torch.min(py2, ty2) - torch.max(py1, ty1)).clamp(0))
    pa    = (px2 - px1).clamp(0) * (py2 - py1).clamp(0)
    ta    = (tx2 - tx1).clamp(0) * (ty2 - ty1).clamp(0)
    union = pa + ta - inter + 1e-7
    enc   = (((torch.max(px2, tx2) - torch.min(px1, tx1)).clamp(0)) *
             ((torch.max(py2, ty2) - torch.min(py1, ty1)).clamp(0)) + 1e-7)
    return (1 - ((inter / union) - (enc - union) / enc)).mean()


def diou_penalty(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pcx, pcy = pred[..., 0],   pred[..., 1]
    tcx, tcy = target[..., 0], target[..., 1]
    px1, py1, px2, py2 = _to_xyxy(pred)
    tx1, ty1, tx2, ty2 = _to_xyxy(target)
    enc2 = ((torch.max(px2, tx2) - torch.min(px1, tx1)).clamp(0) ** 2 +
            (torch.max(py2, ty2) - torch.min(py1, ty1)).clamp(0) ** 2 + 1e-7)
    return ((pcx - tcx) ** 2 + (pcy - tcy) ** 2) / enc2


def make_cls_label(gt: torch.Tensor, H: int, W: int, sigma_cells: float) -> torch.Tensor:
    B  = gt.shape[0]
    sx = sigma_cells / W
    sy = sigma_cells / H
    ys = (torch.arange(H, device=gt.device).float() + 0.5) / H
    xs = (torch.arange(W, device=gt.device).float() + 0.5) / W
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    cx = gt[:, 0]; cy = gt[:, 1]
    dx = gx.unsqueeze(0) - cx.view(B, 1, 1)
    dy = gy.unsqueeze(0) - cy.view(B, 1, 1)
    return torch.exp(-(dx**2) / (2 * sx**2) - (dy**2) / (2 * sy**2)).unsqueeze(1)


def loc_loss_fn(loc_raw: torch.Tensor, gt_raw: torch.Tensor,
                w_giou: float, w_l1: float, w_diou: float) -> torch.Tensor:
    loc = loc_raw.float()
    gt  = gt_raw.float()
    B   = gt.shape[0]
    gx   = 2.0 * gt[:, 0] - 1.0
    gy   = 2.0 * gt[:, 1] - 1.0
    grid = torch.stack([gx, gy], dim=-1).view(B, 1, 1, 2)
    pred = torch.sigmoid(
        F.grid_sample(loc, grid, mode="bilinear",
                      padding_mode="border", align_corners=False)
    ).view(B, 4)
    return (w_giou * giou_loss(pred, gt) +
            w_l1  * F.l1_loss(pred, gt) +
            w_diou * diou_penalty(pred, gt).mean())


def cls_loss_fn(cls2_raw: torch.Tensor, gt_raw: torch.Tensor,
                sigma_cells: float, pos_weight: float) -> torch.Tensor:
    cls2 = cls2_raw.float()
    gt   = gt_raw.float()
    _, _, H, W = cls2.shape
    label    = make_cls_label(gt, H, W, sigma_cells)
    bce      = F.binary_cross_entropy_with_logits(cls2, label, reduction="none")
    pos_mask = (label > 0.5).float()
    return (bce * (pos_mask * pos_weight + (1.0 - pos_mask))).mean()


def total_loss_fn(loc, cls2, gt, cfg: dict):
    l_loc = loc_loss_fn(loc, gt, cfg["w_giou"], cfg["w_l1"], cfg["w_diou"])
    l_cls = cls_loss_fn(cls2, gt, cfg["cls_sigma_cells"], cfg["cls_pos_weight"])
    return l_loc + cfg["w_cls"] * l_cls, l_loc, l_cls


# ──────────────────────────────────────────────────────────────────────────────
# Validation mIoU (NaN guard added)
# ──────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def compute_val_iou(loc_raw: torch.Tensor, gt_raw: torch.Tensor) -> float:
    loc = loc_raw.float()
    gt  = gt_raw.float()
    B   = gt.shape[0]
    gx  = 2.0 * gt[:, 0] - 1.0
    gy  = 2.0 * gt[:, 1] - 1.0
    grid = torch.stack([gx, gy], dim=-1).view(B, 1, 1, 2)
    pred = torch.sigmoid(
        F.grid_sample(loc, grid, mode="bilinear",
                      padding_mode="border", align_corners=False)
    ).view(B, 4)
    px1 = pred[:,0]-pred[:,2]/2; py1 = pred[:,1]-pred[:,3]/2
    px2 = pred[:,0]+pred[:,2]/2; py2 = pred[:,1]+pred[:,3]/2
    tx1 = gt[:,0]-gt[:,2]/2;     ty1 = gt[:,1]-gt[:,3]/2
    tx2 = gt[:,0]+gt[:,2]/2;     ty2 = gt[:,1]+gt[:,3]/2
    inter = ((torch.min(px2,tx2)-torch.max(px1,tx1)).clamp(0) *
             (torch.min(py2,ty2)-torch.max(py1,ty1)).clamp(0))
    union = (px2-px1)*(py2-py1) + (tx2-tx1)*(ty2-ty1) - inter + 1e-7
    iou   = (inter / union).mean()
    # Guard — return 0.0 instead of NaN propagating into avg
    if torch.isnan(iou) or torch.isinf(iou):
        return 0.0
    return iou.item()


# ──────────────────────────────────────────────────────────────────────────────
# Freeze / unfreeze helpers
# ──────────────────────────────────────────────────────────────────────────────
def _set_grad(module: nn.Module, flag: bool):
    for p in module.parameters():
        p.requires_grad_(flag)


def apply_unfreeze_schedule(model, optimizer, epoch, schedule, base_lr, log):
    for (ep, layer_names) in schedule:
        if epoch != ep:
            continue
        existing = {id(p) for g in optimizer.param_groups for p in g["params"]}
        new_params = []
        for name in layer_names:
            obj = model
            for attr in name.split("."):
                obj = getattr(obj, attr, None)
                if obj is None:
                    break
            if obj is None:
                log.warning("  schedule: '%s' not found in model", name)
                continue
            _set_grad(obj, True)
            new_p = [p for p in obj.parameters() if id(p) not in existing]
            new_params.extend(new_p)
            log.info("  unfreeze: %-30s (+%d params)", name, len(new_p))
        if new_params:
            is_bb = any("backbone" in n for n in layer_names)
            lr    = base_lr / 10 if is_bb else base_lr
            optimizer.add_param_group({
                "params": new_params, "lr": lr,
                "weight_decay": CFG["weight_decay"], "eps": 1e-8,
            })
            log.info("  → added %d params @ lr=%.2e", len(new_params), lr)


# ──────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ──────────────────────────────────────────────────────────────────────────────
def _atomic_save(obj, path: str):
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    try:
        os.replace(tmp, path)
    except Exception:
        os.rename(tmp, path)


def save_checkpoint(state: dict, save_dir: str, name: str):
    _atomic_save(state, os.path.join(save_dir, name))


def load_checkpoint(save_dir: str, device: str, log) -> Optional[dict]:
    path = os.path.join(save_dir, LATEST_CKPT)
    if not os.path.exists(path):
        return None
    log.info("Resuming from: %s", path)
    return torch.load(path, map_location=device, weights_only=False)


# ──────────────────────────────────────────────────────────────────────────────
# [FIX-1] Load pretrained checkpoint for resume — handles all formats
# ──────────────────────────────────────────────────────────────────────────────
def load_pretrained_for_resume(path: str, device: str, log) -> dict:
    """
    Load best_miou.pth (or any single .pth) and return it as a resume dict.
    Handles three formats:
      a) Full training state  {model, optimizer, scheduler, epoch, ...}
      b) Pure weight dict     {backbone.layer1.0.weight: ..., ...}
      c) Checkpoint wrapper   {state_dict: ..., net: ..., model: ...}
    """
    log.info("[V5.1] Loading pretrained-for-resume from: %s", path)
    raw = torch.load(path, map_location=device, weights_only=False)

    # Already a full training state (has 'model' key that is a weight dict)
    if isinstance(raw, dict) and "model" in raw and isinstance(raw["model"], dict):
        n_tensors = sum(1 for v in raw["model"].values()
                        if isinstance(v, torch.Tensor))
        if n_tensors > 10:
            log.info("  → detected full training state (epoch=%s)", raw.get("epoch"))
            return raw   # use directly, resume as-is

    # Pure weight dict or wrapper — extract weights and build a minimal state
    if isinstance(raw, dict):
        # Prefer common wrapper keys
        sd = None
        for key in ("model", "state_dict", "net", "ema"):
            candidate = raw.get(key)
            if isinstance(candidate, dict):
                n = sum(1 for v in candidate.values() if isinstance(v, torch.Tensor))
                if n > 10:
                    sd = candidate
                    break
        if sd is None:
            # Assume the dict itself is the weight dict
            sd = raw
    else:
        sd = dict(raw)  # OrderedDict / custom type

    cleaned = {k.replace("module.", "").replace("_orig_mod.", ""): v
               for k, v in sd.items()}
    log.info("  → extracted %d weight keys (no optimizer/scheduler state)",
             len(cleaned))

    # Return as a partial state; train() will handle the missing keys gracefully
    return {
        "model":     cleaned,
        "epoch":     -1,       # will be reset to 0
        "best_val":  raw.get("best_val",  float("inf")),
        "best_miou": raw.get("best_miou", 0.0),
        # optimizer / scheduler / scaler / ema intentionally absent
        # — they will be freshly initialised in train()
    }


# ──────────────────────────────────────────────────────────────────────────────
# Timing helpers
# ──────────────────────────────────────────────────────────────────────────────
def fmt_time(s: float) -> str:
    h = int(s // 3600); m = int((s % 3600) // 60); sec = int(s % 60)
    return f"{h:02d}:{m:02d}:{sec:02d}"


def gpu_mem() -> float:
    return torch.cuda.memory_allocated() / 1e6 if torch.cuda.is_available() else 0.0


# ──────────────────────────────────────────────────────────────────────────────
# Main training function
# ──────────────────────────────────────────────────────────────────────────────
def train(cfg: dict):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log    = setup_logging(cfg["save_dir"])
    log.info("=" * 72)
    log.info("AIC-4 HiFT Fine-Tuning V5.1 (safe resume)  |  device=%s  |  pid=%d",
             device, os.getpid())
    log.info("Config:\n%s", json.dumps(cfg, indent=2, default=str))
    log.info("=" * 72)

    torch.backends.cudnn.enabled = True

    # ── Dataset ────────────────────────────────────────────────────────────────
    train_ds, val_ds = TrainingDataset.split_train_val(
        manifest_path  = cfg["manifest"],
        data_root      = cfg["data_root"],
        val_ratio      = cfg["val_ratio"],
        train_samples  = cfg["train_samples"],
        val_samples    = cfg["val_samples"],
        template_size  = cfg["template_size"],
        search_size    = cfg["search_size"],
        max_gap_frames = cfg["max_gap_frames"],
        jitter_sigma   = cfg["jitter_sigma"],
        seed           = cfg["seed"],
    )

    nw = cfg["num_workers"]
    _dl_common = dict(
        pin_memory         = (device == "cuda"),
        persistent_workers = (nw > 0),
        worker_init_fn     = _worker_init if nw > 0 else None,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size      = cfg["batch_size"],
        shuffle         = True,
        num_workers     = nw,
        drop_last       = True,
        prefetch_factor = 4 if nw > 0 else None,
        **_dl_common,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size      = cfg["batch_size"] * 2,
        shuffle         = False,
        num_workers     = min(nw, 2),
        prefetch_factor = 2 if nw > 0 else None,
        **_dl_common,
    )

    # ── Model ──────────────────────────────────────────────────────────────────
    model = HiFT().to(device)

    # [FIX-1] Load the starting weights using proper resume loader
    log.info("[V5.1] Loading base weights from: %s", cfg["pretrained_ckpt"])
    ckpt = load_pretrained_for_resume(cfg["pretrained_ckpt"], device, log)
    
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    n_model = len(model.state_dict())
    n_ok    = n_model - len(missing)
    log.info("  loaded %d/%d keys (%.1f%%)  missing=%d  unexpected=%d",
             n_ok, n_model, 100.0 * n_ok / max(n_model, 1),
             len(missing), len(unexpected))

    # [FIX-3] Optional warm-start: freeze backbone in epoch 0 for stability
    # Uncomment the block below if you want extra stability on first epoch
    # if True:  # Change to `if False:` to disable
    #     log.info("[V5.1] Warm-start: freezing backbone for epoch 0")
    #     for p in model.backbone.parameters():
    #         p.requires_grad_(False)

    # Un-freeze everything that was unfrozen in the original run
    for p in model.parameters():
        p.requires_grad_(True)
    log.info("[V5.1] All parameters set trainable (lr will differ by group)")

    ema = EMA(model, decay=cfg["ema_decay"]) if cfg["use_ema"] else None

    if cfg["use_compile"] and hasattr(torch, "compile"):
        model = torch.compile(model, mode="reduce-overhead")

    torch.backends.cudnn.benchmark = (device == "cuda")

    # ── Optimizer (two param groups: backbone lower LR, head normal LR) ──────
    backbone_params = list(model.backbone.parameters())
    head_params     = list(model.grader.parameters())
    optimizer = AdamW(
        [
            {"params": backbone_params, "lr": cfg["lr"] / 10,  # 1e-7
             "weight_decay": cfg["weight_decay"], "eps": 1e-8},
            {"params": head_params,     "lr": cfg["lr"],        # 1e-6
             "weight_decay": cfg["weight_decay"], "eps": 1e-8},
        ],
    )
    log.info("[V5.1] Optimizer: backbone lr=%.2e  head lr=%.2e",
             cfg["lr"] / 10, cfg["lr"])

    # ── Scheduler [FIX-2] eta_min = lr * 0.1 (not 0.01) ───────────────────────
    steps_per_epoch = len(train_loader) // cfg["accum_steps"]
    warmup_steps    = cfg["warmup_epochs"] * steps_per_epoch

    warmup_sched = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_steps)
    cosine_sched = CosineAnnealingWarmRestarts(
        optimizer,
        T_0     = max(1, cfg["epochs"] - cfg["warmup_epochs"]) * steps_per_epoch,
        T_mult  = 1,
        eta_min = cfg["lr"] * 0.1,  # [FIX-2] Less aggressive decay
    )
    scheduler = SequentialLR(optimizer, [warmup_sched, cosine_sched],
                             milestones=[warmup_steps])

    # [FIX-5] AMP/scaler safety check
    use_amp = cfg["use_amp"] and (device == "cuda")
    scaler  = GradScaler("cuda") if (use_amp and device == "cuda") else None

    # ── Resume from latest.pth inside save_dir (if it exists) ─────────────────
    # [FIX-4] start_epoch = 0 is intentional: we fine-tune from best checkpoint
    start_epoch = 0  # intentionally restart from best checkpoint
    best_val    = float("inf")
    best_miou   = ckpt.get("best_miou", 0.0)

    resume_ckpt = load_checkpoint(cfg["save_dir"], device, log)
    if resume_ckpt is not None:
        model.load_state_dict(resume_ckpt["model"], strict=False)
        try:
            optimizer.load_state_dict(resume_ckpt["optimizer"])
        except Exception as e:
            log.warning("Optimizer mismatch on resume — resetting: %s", e)
        scheduler.load_state_dict(resume_ckpt["scheduler"])
        if scaler and "scaler" in resume_ckpt:
            scaler.load_state_dict(resume_ckpt["scaler"])
        # [FIX-6] EMA load with safety check
        if ema and "ema" in resume_ckpt and resume_ckpt["ema"] is not None:
            try:
                ema.load_state_dict(resume_ckpt["ema"])
            except Exception as e:
                log.warning("EMA state mismatch on resume — resetting: %s", e)
        start_epoch = resume_ckpt["epoch"] + 1
        best_val    = resume_ckpt.get("best_val",  float("inf"))
        best_miou   = resume_ckpt.get("best_miou", 0.0)
        log.info("Resumed from epoch %d  |  best_val=%.4f  |  best_mIoU=%.4f",
                 start_epoch, best_val, best_miou)

    # ── Training loop ──────────────────────────────────────────────────────────
    total_start  = time.time()
    global_step  = start_epoch * steps_per_epoch

    # counters for NaN tracking
    nan_batches_train = 0
    nan_batches_val   = 0

    for epoch in range(start_epoch, cfg["epochs"]):
        ep_start = time.time()
        train_ds.set_epoch(epoch)
        val_ds.set_epoch(epoch)

        # [FIX-3] Optional: unfreeze backbone after epoch 0 if using warm-start
        # if epoch == 1 and cfg.get("warm_start_backbone", False):
        #     log.info("[V5.1] Warm-start: unfreezing backbone from epoch 1")
        #     for p in model.backbone.parameters():
        #         p.requires_grad_(True)

        apply_unfreeze_schedule(model, optimizer, epoch,
                                cfg["unfreeze_schedule"], cfg["lr"], log)

        # ── Train ──────────────────────────────────────────────────────────────
        model.train()
        sum_loss = sum_loc = sum_cls = 0.0
        n_steps  = 0
        nan_batches_train = 0
        optimizer.zero_grad(set_to_none=True)

        for i, batch in enumerate(train_loader):
            tmpl = batch["template"].to(device, non_blocking=True)
            srch = batch["search"].to(device, non_blocking=True)
            gt   = batch["gt_box"].to(device, non_blocking=True)

            if use_amp and device == "cuda":
                with autocast("cuda"):
                    loc, _, cls2 = model(tmpl, srch)
                loss, l_loc, l_cls = total_loss_fn(loc, cls2, gt, cfg)
            else:
                loc, _, cls2 = model(tmpl, srch)
                loss, l_loc, l_cls = total_loss_fn(loc, cls2, gt, cfg)

            # NaN/Inf guard — skip bad batch cleanly
            if torch.isnan(loss) or torch.isinf(loss):
                nan_batches_train += 1
                log.warning(
                    "[V5.1] NaN/Inf loss at epoch=%d step=%d batch=%d — "
                    "skipping batch (total skipped this epoch: %d)",
                    epoch + 1, global_step, i, nan_batches_train,
                )
                optimizer.zero_grad(set_to_none=True)
                continue

            if use_amp and device == "cuda":
                scaler.scale(loss / cfg["accum_steps"]).backward()
            else:
                (loss / cfg["accum_steps"]).backward()

            sum_loss += loss.item()
            sum_loc  += l_loc.item()
            sum_cls  += l_cls.item()

            if (i + 1) % cfg["accum_steps"] == 0:
                if use_amp and device == "cuda":
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad],
                        cfg["grad_clip"],
                    )
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad],
                        cfg["grad_clip"],
                    )
                    optimizer.step()

                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                n_steps     += 1

                if ema:
                    ema.update(model)

                if global_step % cfg["log_freq"] == 0:
                    elapsed = time.time() - ep_start
                    done_b  = i + 1
                    eta_ep  = elapsed / done_b * (len(train_loader) - done_b)
                    sps     = done_b * cfg["batch_size"] / elapsed
                    log.info(
                        "[E%02d/%d | S%05d]  "
                        "loss=%.4f  loc=%.4f  cls=%.4f  |  "
                        "lr=%.2e  |  %.1f samp/s  |  VRAM=%.0fMB  |  ETA %s"
                        "  [NaN skipped: %d]",
                        epoch + 1, cfg["epochs"], global_step,
                        sum_loss / max(n_steps, 1),
                        sum_loc  / max(n_steps, 1),
                        sum_cls  / max(n_steps, 1),
                        optimizer.param_groups[-1]["lr"],
                        sps, gpu_mem(), fmt_time(eta_ep),
                        nan_batches_train,
                    )

        avg_train = sum_loss / max(n_steps, 1)
        avg_loc   = sum_loc  / max(n_steps, 1)
        avg_cls   = sum_cls  / max(n_steps, 1)

        if nan_batches_train > 0:
            log.warning("[V5.1] Epoch %d: %d batches were NaN/Inf and skipped",
                        epoch + 1, nan_batches_train)

        # ── Validation ─────────────────────────────────────────────────────────
        model.eval()
        vl_sum = vi_sum = 0.0
        nv = 0
        nan_batches_val = 0

        with torch.no_grad():
            for batch in val_loader:
                tmpl = batch["template"].to(device, non_blocking=True)
                srch = batch["search"].to(device, non_blocking=True)
                gt   = batch["gt_box"].to(device, non_blocking=True)

                if use_amp and device == "cuda":
                    with autocast("cuda"):
                        loc, _, cls2 = model(tmpl, srch)
                else:
                    loc, _, cls2 = model(tmpl, srch)

                loss, _, _ = total_loss_fn(loc, cls2, gt, cfg)

                # NaN/Inf guard in validation
                if torch.isnan(loss) or torch.isinf(loss):
                    nan_batches_val += 1
                    continue

                iou_val = compute_val_iou(loc, gt)
                vl_sum += loss.item()
                vi_sum += iou_val
                nv     += 1

        avg_val  = vl_sum / max(nv, 1)
        avg_miou = vi_sum / max(nv, 1)
        ep_time  = time.time() - ep_start
        elapsed  = time.time() - total_start
        eta_tot  = elapsed / (epoch - start_epoch + 1) * (cfg["epochs"] - epoch - 1)

        log.info(
            "━━ E%02d/%d  train=%.4f (loc=%.4f cls=%.4f)  "
            "val=%.4f  mIoU=%.4f  lr=%.2e  %s  ETA %s"
            "  [val_nan=%d]",
            epoch + 1, cfg["epochs"],
            avg_train, avg_loc, avg_cls,
            avg_val, avg_miou,
            optimizer.param_groups[-1]["lr"],
            fmt_time(ep_time), fmt_time(eta_tot),
            nan_batches_val,
        )

        # ── Checkpoints ────────────────────────────────────────────────────────
        state = dict(
            epoch     = epoch,
            model     = model.state_dict(),
            optimizer = optimizer.state_dict(),
            scheduler = scheduler.state_dict(),
            best_val  = best_val,
            best_miou = best_miou,
            cfg       = cfg,
        )
        if scaler: state["scaler"] = scaler.state_dict()
        if ema:    state["ema"]    = ema.state_dict()

        save_checkpoint(state, cfg["save_dir"], LATEST_CKPT)

        if avg_val < best_val:
            best_val      = avg_val
            state["best_val"] = best_val
            save_checkpoint(state, cfg["save_dir"], BEST_CKPT)
            if ema:
                _atomic_save(ema.state_dict(),
                             os.path.join(cfg["save_dir"], EMA_CKPT))
            log.info("  ★ New best val=%.4f  → %s + %s", best_val, BEST_CKPT, EMA_CKPT)

        if avg_miou > best_miou:
            best_miou      = avg_miou
            state["best_miou"] = best_miou
            save_checkpoint(state, cfg["save_dir"], "best_miou.pth")
            log.info("  ★ New best mIoU=%.4f → best_miou.pth", best_miou)

        if (epoch + 1) % 5 == 0:
            save_checkpoint(state, cfg["save_dir"], f"epoch_{epoch+1:03d}.pth")

    # ── Done ───────────────────────────────────────────────────────────────────
    total = time.time() - total_start
    log.info("=" * 72)
    log.info("DONE  |  total=%s  |  best_val=%.4f  |  best_mIoU=%.4f",
             fmt_time(total), best_val, best_miou)
    log.info("Checkpoints → %s", cfg["save_dir"])
    log.info("=" * 72)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="AIC-4 HiFT fine-tuning V5.1 — safe resume")
    p.add_argument("--epochs",      type=int,   default=CFG["epochs"])
    p.add_argument("--batch_size",  type=int,   default=CFG["batch_size"])
    p.add_argument("--accum_steps", type=int,   default=CFG["accum_steps"])
    p.add_argument("--lr",          type=float, default=CFG["lr"])
    p.add_argument("--grad_clip",   type=float, default=CFG["grad_clip"])
    p.add_argument("--num_workers", type=int,   default=CFG["num_workers"])
    p.add_argument("--pretrained",  type=str,   default=CFG["pretrained_ckpt"])
    p.add_argument("--save_dir",    type=str,   default=CFG["save_dir"])
    p.add_argument("--manifest",    type=str,   default=CFG["manifest"])
    p.add_argument("--data_root",   type=str,   default=CFG["data_root"])
    p.add_argument("--no_amp",      action="store_true")
    p.add_argument("--no_ema",      action="store_true")
    p.add_argument("--no_swa",      action="store_true")
    p.add_argument("--compile",     action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg  = dict(CFG)
    cfg["epochs"]          = args.epochs
    cfg["batch_size"]      = args.batch_size
    cfg["accum_steps"]     = args.accum_steps
    cfg["lr"]              = args.lr
    cfg["grad_clip"]       = args.grad_clip
    cfg["num_workers"]     = args.num_workers
    cfg["pretrained_ckpt"] = args.pretrained
    cfg["save_dir"]        = args.save_dir
    cfg["manifest"]        = args.manifest
    cfg["data_root"]       = args.data_root
    cfg["use_amp"]         = not args.no_amp
    cfg["use_ema"]         = not args.no_ema
    cfg["use_swa"]         = not args.no_swa
    cfg["use_compile"]     = args.compile
    train(cfg)