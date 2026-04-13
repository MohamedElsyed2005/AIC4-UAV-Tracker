"""
tools/finetune_full_hift.py
Fixed Version v2 + Timing:
  - Added classification loss (BCE on cls2 head) with Gaussian label mask.
  - train_samples raised to 60_000 for more diversity.
  - EMA_CENTRE tuned comment added for submit.py reference.
  - ⏱️ Added comprehensive timing: total, per-epoch, per-batch, ETA.
"""
import os
import sys
import logging
import math
import time  # ← Added for timing
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from models.hift_full import HiFT
from data.dataset import TrainingDataset

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)
cv2.setNumThreads(2)
cv2.ocl.setUseOpenCL(False)

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
CKPT_PATH     = os.path.join(ROOT, "checkpoints/first.pth")
SAVE_DIR      = os.path.join(ROOT, "checkpoints")
EPOCHS        = 20
BATCH_SIZE    = 4
LR            = 5e-5
NUM_WORKERS   = 0
WARMUP_EPOCHS = 2

# Weight of cls loss relative to loc loss.  Start small — loc is the priority.
CLS_LOSS_WEIGHT   = 0.5
# Gaussian radius for positive cls label mask (in normalised [0,1] coords).
CLS_SIGMA         = 0.05


# ─────────────────────────────────────────────────────────────────────────────
# Timing helper
# ─────────────────────────────────────────────────────────────────────────────
def format_time(seconds: float) -> str:
    """Convert seconds to HH:MM:SS format."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


# ─────────────────────────────────────────────────────────────────────────────
# Classification label helper
# ─────────────────────────────────────────────────────────────────────────────
def make_cls_label(gt: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """
    Build a soft [B, 1, H, W] label map for cls2 training.

    The GT centre (cx, cy) in [0,1] is projected onto the H×W score map.
    A Gaussian with sigma=CLS_SIGMA (in normalised units) is placed at that
    location so that nearby cells also receive a positive signal.

    Args:
        gt : [B, 4] tensor  [cx, cy, w, h] in [0,1] normalised coords
        H, W: spatial size of cls2 output map
    Returns:
        label: [B, 1, H, W] float tensor in [0, 1]
    """
    B = gt.shape[0]
    device = gt.device

    # Grid of normalised cell-centre coordinates
    ys = (torch.arange(H, device=device).float() + 0.5) / H  # [H]
    xs = (torch.arange(W, device=device).float() + 0.5) / W  # [W]
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")   # [H, W]

    cx = gt[:, 0]  # [B]
    cy = gt[:, 1]  # [B]

    # Squared distances from each grid cell to the GT centre
    dx = grid_x.unsqueeze(0) - cx.view(B, 1, 1)  # [B, H, W]
    dy = grid_y.unsqueeze(0) - cy.view(B, 1, 1)  # [B, H, W]
    dist2 = dx ** 2 + dy ** 2

    sigma2 = CLS_SIGMA ** 2
    label = torch.exp(-dist2 / (2 * sigma2))  # [B, H, W]
    return label.unsqueeze(1)                 # [B, 1, H, W]


# ─────────────────────────────────────────────────────────────────────────────
# Loss: GIoU + L1  (loc head)
# ─────────────────────────────────────────────────────────────────────────────
def giou_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Generalised IoU loss for [cx, cy, w, h] boxes in [0,1] normalised coords."""
    def to_xyxy(b):
        return (b[..., 0] - b[..., 2] / 2,
                b[..., 1] - b[..., 3] / 2,
                b[..., 0] + b[..., 2] / 2,
                b[..., 1] + b[..., 3] / 2)

    px1, py1, px2, py2 = to_xyxy(pred)
    tx1, ty1, tx2, ty2 = to_xyxy(target)

    inter = ((torch.min(px2, tx2) - torch.max(px1, tx1)).clamp(0) *
             (torch.min(py2, ty2) - torch.max(py1, ty1)).clamp(0))
    pa    = (px2 - px1).clamp(0) * (py2 - py1).clamp(0)
    ta    = (tx2 - tx1).clamp(0) * (ty2 - ty1).clamp(0)
    union = pa + ta - inter + 1e-7
    iou   = inter / union

    enc = (((torch.max(px2, tx2) - torch.min(px1, tx1)).clamp(0)) *
           ((torch.max(py2, ty2) - torch.min(py1, ty1)).clamp(0)) + 1e-7)

    return (1 - (iou - (enc - union) / enc)).mean()


def loc_loss(loc: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """
    Spatially-aware regression loss using grid_sample.

    loc : [B, 4, H, W]  regression map (logits)
    gt  : [B, 4]         [cx, cy, w, h] in [0,1] normalised coords
    """
    B, C, H, W = loc.shape

    # Convert GT [0,1] → grid_sample [-1,1]
    cx = gt[:, 0]
    cy = gt[:, 1]
    grid_x = (2 * cx - 1).view(B, 1, 1, 1)
    grid_y = (2 * cy - 1).view(B, 1, 1, 1)
    grid = torch.cat([grid_x, grid_y], dim=-1)                    # [B,1,1,2]

    # Sample at GT position: [B,4,H,W] → [B,4,1,1] → [B,4]
    pred = F.grid_sample(loc, grid, mode='bilinear', padding_mode='border',
                         align_corners=False)
    pred = pred.view(B, 4)
    pred = torch.sigmoid(pred)

    return giou_loss(pred, gt) + F.l1_loss(pred, gt)


def cls_loss(cls2: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """
    Binary cross-entropy loss on the cls2 confidence map.

    cls2 : [B, 1, H, W]  raw logits
    gt   : [B, 4]         [cx, cy, w, h] in [0,1]

    A Gaussian label centred at the GT position is used so that cells close
    to the target also contribute as positives — this gives a smoother gradient
    and avoids extremely sparse supervision on the H×W map.
    """
    _, _, H, W = cls2.shape
    label = make_cls_label(gt, H, W).to(cls2.device)   # [B,1,H,W] in [0,1]

    # Focal-style weighting: down-weight easy negatives
    # (plain BCE works too; focal just trains faster on imbalanced maps)
    bce = F.binary_cross_entropy_with_logits(cls2, label, reduction='none')

    # Weight positives more strongly (label > 0.5 → positive region)
    pos_mask = (label > 0.5).float()
    neg_mask = 1.0 - pos_mask
    pos_weight = 5.0   # increase positive contribution
    weighted = bce * (pos_mask * pos_weight + neg_mask)
    return weighted.mean()


def task_loss(loc: torch.Tensor, cls2: torch.Tensor,
              gt: torch.Tensor) -> tuple:
    """Combined loc + cls loss. Returns (total, loc, cls)."""
    l_loc = loc_loss(loc, gt)
    l_cls = cls_loss(cls2, gt)
    total_loss = l_loc + CLS_LOSS_WEIGHT * l_cls
    return total_loss, l_loc, l_cls


# ─────────────────────────────────────────────────────────────────────────────
# GPU memory helper
# ─────────────────────────────────────────────────────────────────────────────
def gpu_mem():
    return torch.cuda.memory_allocated() / 1024 ** 2 if torch.cuda.is_available() else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Freezing / optimizer / scheduler helpers
# ─────────────────────────────────────────────────────────────────────────────
def _set_grad(module: nn.Module, flag: bool):
    for p in module.parameters():
        p.requires_grad_(flag)


PHASES = [
    (0,  lambda m: (_set_grad(m.backbone, False),
                    log.info("  freeze: backbone (head-only warmup)"))),
    (5,  lambda m: (_set_grad(m.backbone.layer5, True),
                    log.info("  unfreeze: layer5"))),
    (8,  lambda m: (_set_grad(m.backbone.layer4, True),
                    log.info("  unfreeze: layer4"))),
    (11, lambda m: (_set_grad(m.backbone.layer3, True),
                    log.info("  unfreeze: layer3"))),
    (14, lambda m: (_set_grad(m.backbone, True),
                    log.info("  unfreeze: all backbone"))),
]


def build_optimizer(model: nn.Module, base_lr: float) -> AdamW:
    backbone_params = [p for n, p in model.named_parameters()
                       if "backbone" in n and p.requires_grad]
    head_params     = [p for n, p in model.named_parameters()
                       if "backbone" not in n and p.requires_grad]

    groups = []
    if backbone_params:
        groups.append({"params": backbone_params, "lr": base_lr / 10})
    if head_params:
        groups.append({"params": head_params, "lr": base_lr})

    return AdamW(groups, weight_decay=1e-4, eps=1e-8)


def build_scheduler(opt, total_epochs, warmup, base_lr):
    return SequentialLR(
        opt,
        schedulers=[
            LinearLR(opt, start_factor=0.1, total_iters=warmup),
            CosineAnnealingLR(opt, T_max=max(1, total_epochs - warmup),
                              eta_min=base_lr * 0.01),
        ],
        milestones=[warmup],
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main training loop
# ─────────────────────────────────────────────────────────────────────────────
def train():
    log.info("Fine-tuning HiFT | epochs=%d | BS=%d | device=%s | lr=%g",
             EPOCHS, BATCH_SIZE, DEVICE, LR)

    # ── Dataset ───────────────────────────────────────────────────────────
    train_ds, val_ds = TrainingDataset.split_train_val(
        manifest_path=os.path.join(ROOT, "data/contest_release/metadata/contestant_manifest.json"),
        data_root=os.path.join(ROOT, "data/contest_release"),
        val_ratio=0.15,
        train_samples=60_000,
        val_samples=2_000,
        template_size=128,
        search_size=256,
        seed=42,
        jitter_sigma=0.25,
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True,
                              persistent_workers=(NUM_WORKERS > 0))
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True,
                              persistent_workers=(NUM_WORKERS > 0))

    # ── Model ─────────────────────────────────────────────────────────────
    model = HiFT().to(DEVICE)
    model.load_pretrained(CKPT_PATH, device=DEVICE)

    _set_grad(model.backbone, False)
    log.info("  freeze: backbone (head-only warmup)")

    optimizer = build_optimizer(model, LR)
    scheduler = build_scheduler(optimizer, EPOCHS, WARMUP_EPOCHS, LR)
    use_amp   = "cuda" in DEVICE
    scaler    = GradScaler("cuda") if use_amp else None

    os.makedirs(SAVE_DIR, exist_ok=True)
    best_val = float("inf")

    # ── TIMING: Start total timer ─────────────────────────────────────────
    total_start = time.time()
    log.info("⏱️  Training started at %s", time.strftime("%H:%M:%S"))

    # ── Training loop ─────────────────────────────────────────────────────
    for ep in range(EPOCHS):
        epoch_start = time.time()
        
        # Phase transitions (staged unfreezing)
        for phase_ep, phase_fn in PHASES:
            if ep == phase_ep:
                log.info("=== Epoch %d: phase transition ===", ep)
                phase_fn(model)
                optimizer = build_optimizer(model, LR)
                scheduler = build_scheduler(optimizer, EPOCHS - ep,
                                            WARMUP_EPOCHS, LR)
                if use_amp:
                    scaler = GradScaler("cuda")

        # ── Train ─────────────────────────────────────────────────────────
        model.train()
        total_loss, loc_loss_sum, cls_loss_sum, n = 0.0, 0.0, 0.0, 0
        batch_times = []  # Track batch times for ETA

        for i, batch in enumerate(train_loader):
            batch_start = time.time()
            
            tmpl = batch["template"].to(DEVICE, non_blocking=True)
            srch = batch["search"].to(DEVICE, non_blocking=True)
            gt   = batch["gt_box"].to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            if use_amp:
                with autocast("cuda"):
                    loc, _, cls2 = model(tmpl, srch)
                    loss, l_loc, l_cls = task_loss(loc, cls2, gt)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loc, _, cls2 = model(tmpl, srch)
                loss, l_loc, l_cls = task_loss(loc, cls2, gt)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()

            # ── TIMING: Record batch time ────────────────────────────────
            batch_time = time.time() - batch_start
            batch_times.append(batch_time)

            total_loss   += loss.item()
            loc_loss_sum += l_loc.item()
            cls_loss_sum += l_cls.item()
            n += 1

            # Log every 50 steps with timing info
            if i % 50 == 0:
                avg_batch_time = sum(batch_times[-10:]) / min(10, len(batch_times))
                remaining_batches = (len(train_loader) - i - 1)
                eta_seconds = avg_batch_time * remaining_batches
                
                log.info(
                    "[Epoch %d/%d] step %d/%d | loss=%.4f (loc=%.3f cls=%.3f) | "
                    "avg=%.4f | lr=%.2e | VRAM=%.0f MB | "
                    "batch=%.2fs | ETA %s",
                    ep + 1, EPOCHS, i, len(train_loader),
                    loss.item(), l_loc.item(), l_cls.item(),
                    total_loss / n,
                    optimizer.param_groups[-1]["lr"], gpu_mem(),
                    batch_time, format_time(eta_seconds)
                )

        # ── Epoch timing summary ─────────────────────────────────────────
        epoch_time = time.time() - epoch_start
        elapsed_total = time.time() - total_start
        avg_epoch_time = elapsed_total / (ep + 1)
        remaining_epochs = EPOCHS - ep - 1
        eta_total = avg_epoch_time * remaining_epochs
        
        scheduler.step()

        # ── Validate ──────────────────────────────────────────────────────
        val_start = time.time()
        model.eval()
        val_loss, nv = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                loc, _, cls2 = model(
                    batch["template"].to(DEVICE, non_blocking=True),
                    batch["search"].to(DEVICE, non_blocking=True),
                )
                loss, _, _ = task_loss(
                    loc, cls2, batch["gt_box"].to(DEVICE, non_blocking=True)
                )
                val_loss += loss.item()
                nv += 1
        val_time = time.time() - val_start

        avg_train = total_loss / max(n, 1)
        avg_val   = val_loss   / max(nv, 1)
        
        log.info(
            "Epoch %3d/%d | train=%.4f (loc=%.4f cls=%.4f) | val=%.4f | "
            "lr=%.2e | epoch_time=%s | val_time=%s | ETA %s",
            ep + 1, EPOCHS, avg_train,
            loc_loss_sum / max(n, 1), cls_loss_sum / max(n, 1),
            avg_val, optimizer.param_groups[-1]["lr"],
            format_time(epoch_time), format_time(val_time),
            format_time(eta_total)
        )

        if avg_val < best_val:
            best_val = avg_val
            save_path = os.path.join(SAVE_DIR, "hift_finetuned_best_v2.pth")
            torch.save(model.state_dict(), save_path)
            log.info("  ✓ Saved best -> %s (val=%.4f)", save_path, best_val)

    # ── Final timing summary ─────────────────────────────────────────────
    total_time = time.time() - total_start
    log.info("=" * 70)
    log.info("⏱️  TRAINING COMPLETE")
    log.info("   Total time      : %s", format_time(total_time))
    log.info("   Avg time/epoch  : %s", format_time(total_time / EPOCHS))
    log.info("   Best val loss   : %.4f", best_val)
    log.info("   Model saved to  : %s", SAVE_DIR)
    log.info("=" * 70)


if __name__ == "__main__":
    train()