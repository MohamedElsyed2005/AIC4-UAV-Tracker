"""
tools/finetune_full_hift.py
Fixed Version v2 + Timing 
Added classification loss (BCE on cls2 head) with Gaussian label mask.
train_samples raised to 60_000 for more diversity.
EMA_CENTRE tuned comment added for submit.py reference.
Added comprehensive timing: total, per-epoch, per-batch, ETA.
Fixed: num_workers support + file logging with UTF-8 encoding.
"""
import os
import sys
import logging
import math
import time
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

# [FIX] Disable cudnn benchmark to prevent stream mismatch errors with AMP
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = True

# Set root directory and add to Python path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from models.hift_full import HiFT
from data.dataset import TrainingDataset

# ─────────────────────────────────────────────────────────────────────────────
# Logging Setup: Console + File (UTF-8 safe for Windows)
# ─────────────────────────────────────────────────────────────────────────────
SAVE_DIR_BASE = os.path.join(ROOT, "checkpoints")
os.makedirs(SAVE_DIR_BASE, exist_ok=True)
log_file = os.path.join(SAVE_DIR_BASE, f"train_log_{time.strftime('%Y%m%d_%H%M%S')}.txt")

# Configure logging to output to both console and file
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler(log_file, mode='a', encoding='utf-8'),  # File output with UTF-8
        logging.StreamHandler(sys.stdout)                            # Console output
    ],
    force=True  # Reset any previous logging configuration
)
log = logging.getLogger(__name__)
log.info(f"Logging initialized. Output saved to: {log_file}")

# OpenCV settings: Reduce thread contention when using num_workers > 0
# Important: Set to 1 per worker to avoid OpenCV deadlock issues
cv2.setNumThreads(1)
cv2.ocl.setUseOpenCL(False)

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CKPT_PATH = os.path.join(ROOT, "checkpoints/first.pth")
SAVE_DIR = SAVE_DIR_BASE
EPOCHS = 12
BATCH_SIZE = 4
LR = 5e-5
NUM_WORKERS = 2 # Increased from  to 2 for faster data loading
WARMUP_EPOCHS = 1

# Weight of classification loss relative to localization loss.
# Start small because localization is the primary objective.
CLS_LOSS_WEIGHT = 0.5

# [CHANGE] CLS_SIGMA removed — now computed adaptively in make_cls_label
# Gaussian radius for positive classification label mask (in normalized [0,1] coordinates).
# Controls how spread out the positive signal is around the ground-truth center.
# CLS_SIGMA = 0.05  <-- [FIX] DELETED: replaced with adaptive sigma

# [CHANGE] Add flag to easily toggle AMP for debugging cuDNN issues
USE_AMP = True  # Set to False to disable autocast/GradScaler temporarily

# ─────────────────────────────────────────────────────────────────────────────
# Timing Helper Function
# ─────────────────────────────────────────────────────────────────────────────
def format_time(seconds: float) -> str:
    """
    Convert seconds to HH:MM:SS format for readable time display.
    
    Args:
        seconds: Time duration in seconds.
    
    Returns:
        Formatted string in HH:MM:SS format.
    """
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

# ─────────────────────────────────────────────────────────────────────────────
# Classification Label Helper: Gaussian Mask Generation
# ─────────────────────────────────────────────────────────────────────────────
def make_cls_label(gt: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """
    Build a soft [B, 1, H, W] label map for cls2 training.
    The ground-truth center (cx, cy) in normalized [0,1] coordinates is projected
    onto the H×W score map. A Gaussian with sigma=CLS_SIGMA is placed at that
    location so that nearby cells also receive a positive signal, providing
    smoother gradients during training.
    
    [FIX] Sigma is now adaptive to output map size instead of fixed 0.05.
    Old behavior: sigma=0.05 in normalized coords → 0.05×4 = 0.2 pixel on a 4×4 map (too small!)
    New behavior: sigma=1.5 cells on the actual output map → proper Gaussian coverage.
    Rule of thumb: sigma should cover 1–2 cells regardless of map resolution.

    Args:
        gt: [B, 4] tensor containing [cx, cy, w, h] in normalized [0,1] coords.
        H: Height of the cls2 output map.
        W: Width of the cls2 output map.

    Returns:
        label: [B, 1, H, W] float tensor with values in [0, 1].
    """
    B = gt.shape[0]
    device = gt.device

    # [FIX] Adaptive sigma: 1.5 cells regardless of map size
    # Expressed in normalized [0,1] coords = 1.5 / map_dimension
    sigma_cells = 1.5
    sigma_x = sigma_cells / W   # normalized sigma in x direction
    sigma_y = sigma_cells / H   # normalized sigma in y direction

    # Create grid of normalized cell-center coordinates
    ys = (torch.arange(H, device=device).float() + 0.5) / H  # [H]
    xs = (torch.arange(W, device=device).float() + 0.5) / W  # [W]
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")   # [H, W]

    cx = gt[:, 0]  # [B] - x-coordinates of ground-truth centers
    cy = gt[:, 1]  # [B] - y-coordinates of ground-truth centers

    # Compute squared distances from each grid cell to the GT center
    dx = grid_x.unsqueeze(0) - cx.view(B, 1, 1)  # [B, H, W]
    dy = grid_y.unsqueeze(0) - cy.view(B, 1, 1)  # [B, H, W]

    # [FIX] Anisotropic Gaussian with adaptive sigma per axis
    label = torch.exp(
        -(dx**2) / (2 * sigma_x**2) - (dy**2) / (2 * sigma_y**2)
    )
    return label.unsqueeze(1)  # [B, 1, H, W]

# ─────────────────────────────────────────────────────────────────────────────
# Loss Functions: GIoU + L1 for Localization Head
# ─────────────────────────────────────────────────────────────────────────────
def giou_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Generalized IoU loss for [cx, cy, w, h] boxes in normalized [0,1] coords.
    
    Args:
        pred: Predicted boxes [B, 4].
        target: Ground-truth boxes [B, 4].
    
    Returns:
        GIoU loss value (scalar tensor).
    """
    def to_xyxy(b):
        """Convert [cx, cy, w, h] to [x1, y1, x2, y2] format."""
        return (b[..., 0] - b[..., 2] / 2,
                b[..., 1] - b[..., 3] / 2,
                b[..., 0] + b[..., 2] / 2,
                b[..., 1] + b[..., 3] / 2)

    px1, py1, px2, py2 = to_xyxy(pred)
    tx1, ty1, tx2, ty2 = to_xyxy(target)

    # Compute intersection area
    inter = ((torch.min(px2, tx2) - torch.max(px1, tx1)).clamp(0) *
             (torch.min(py2, ty2) - torch.max(py1, ty1)).clamp(0))

    # Compute areas
    pa = (px2 - px1).clamp(0) * (py2 - py1).clamp(0)  # Predicted area
    ta = (tx2 - tx1).clamp(0) * (ty2 - ty1).clamp(0)  # Target area
    union = pa + ta - inter + 1e-7
    iou = inter / union

    # Compute enclosing box area for GIoU
    enc = (((torch.max(px2, tx2) - torch.min(px1, tx1)).clamp(0)) *
           ((torch.max(py2, ty2) - torch.min(py1, ty1)).clamp(0)) + 1e-7)

    return (1 - (iou - (enc - union) / enc)).mean()


def loc_loss(loc: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """
    Spatially-aware regression loss using grid_sample.
    Samples the regression map at the ground-truth position and computes
    GIoU + L1 loss between predicted and target boxes.
    
    [FIX] Grid shape is now correctly [B, 1, 1, 2] using stack+reshape.
    Previous implementation used view(B,1,1,1) + cat which could silently
    produce [B,1,1,1,2] in some PyTorch versions, causing subtle bugs.
    grid_sample expects: input [B,C,H,W], grid [B,H_out,W_out,2]
    where grid[..., 0] = x (col direction) and grid[..., 1] = y (row direction)

    Args:
        loc: [B, 4, H, W] regression map (logits).
        gt: [B, 4] ground-truth boxes [cx, cy, w, h] in normalized coords.

    Returns:
        Combined GIoU + L1 loss value.
    """
    B, C, H, W = loc.shape

    # Convert GT [0,1] normalized coords to grid_sample [-1,1] range
    cx = gt[:, 0]
    cy = gt[:, 1]
    
    # [FIX] stack then reshape — produces exactly [B, 1, 1, 2]
    gx = (2.0 * cx - 1.0)
    gy = (2.0 * cy - 1.0)
    grid = torch.stack([gx, gy], dim=-1).view(B, 1, 1, 2)

    # Sample predictions at GT position: [B,4,H,W] -> [B,4,1,1] -> [B,4]
    pred = F.grid_sample(loc, grid, mode='bilinear', padding_mode='border',
                         align_corners=False)
    pred = pred.view(B, 4)
    pred = torch.sigmoid(pred)  # Apply sigmoid to get [0,1] predictions

    return giou_loss(pred, gt) + F.l1_loss(pred, gt)


def cls_loss(cls2: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """
    Binary cross-entropy loss on the cls2 confidence map with focal-style weighting.
    Uses a Gaussian label centered at the GT position so that cells close to
    the target also contribute as positives, providing smoother gradients.

    Args:
        cls2: [B, 1, H, W] raw logits from cls2 head.
        gt: [B, 4] ground-truth boxes [cx, cy, w, h] in normalized coords.

    Returns:
        Weighted BCE loss value.
    """
    _, _, H, W = cls2.shape
    label = make_cls_label(gt, H, W).to(cls2.device)  # [B,1,H,W] in [0,1]

    # Compute binary cross-entropy with logits
    bce = F.binary_cross_entropy_with_logits(cls2, label, reduction='none')

    # Focal-style weighting: emphasize positives, down-weight easy negatives
    pos_mask = (label > 0.5).float()  # Positive region
    neg_mask = 1.0 - pos_mask          # Negative region
    pos_weight = 5.0  # Increase contribution of positive samples

    weighted = bce * (pos_mask * pos_weight + neg_mask)
    return weighted.mean()


def task_loss(loc: torch.Tensor, cls2: torch.Tensor,
              gt: torch.Tensor) -> tuple:
    """
    Combined localization + classification loss.
    
    Args:
        loc: Localization head output.
        cls2: Classification head output.
        gt: Ground-truth boxes.
    
    Returns:
        Tuple of (total_loss, loc_loss, cls_loss).
    """
    l_loc = loc_loss(loc, gt)
    l_cls = cls_loss(cls2, gt)
    total_loss = l_loc + CLS_LOSS_WEIGHT * l_cls
    return total_loss, l_loc, l_cls

# ─────────────────────────────────────────────────────────────────────────────
# GPU Memory Utility
# ─────────────────────────────────────────────────────────────────────────────
def gpu_mem():
    """
    Get current GPU memory usage in MB.
    
    Returns:
        Allocated memory in MB, or 0.0 if CUDA not available.
    """
    return torch.cuda.memory_allocated() / 1024 ** 2 if torch.cuda.is_available() else 0.0

# ─────────────────────────────────────────────────────────────────────────────
# Model Freezing / Optimizer / Scheduler Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _set_grad(module: nn.Module, flag: bool):
    """
    Toggle requires_grad for all parameters in a module.
    
    Args:
        module: PyTorch module.
        flag: True to enable gradients, False to freeze.
    """
    for p in module.parameters():
        p.requires_grad_(flag)


# Training phases for staged backbone unfreezing
# Format: (epoch_number, function_to_execute)
# [FIX] PHASES lambdas now accept optimizer as second argument for add_param_group

# [FIX] New helper to add params to existing optimizer instead of rebuilding it
def unfreeze_and_add_params(
    optimizer: torch.optim.AdamW,
    new_params,
    base_lr: float,
    is_backbone: bool = True,
) -> None:
    """
    [FIX] Add newly unfrozen parameters to existing optimizer
    instead of rebuilding it (which loses accumulated momentum).
    
    AdamW maintains first/second moment estimates (m, v) for each parameter.
    Rebuilding the optimizer resets these to zero, causing training instability
    immediately after each unfreeze phase.
    
    This function preserves existing moments and smoothly introduces new
    parameters at zero momentum (the correct behavior for newly trainable params).
    
    Args:
        optimizer  : The existing AdamW optimizer (modified in-place).
        new_params : List of newly unfrozen nn.Parameter objects.
        base_lr    : Base learning rate for head parameters.
        is_backbone: If True uses base_lr/10, else uses base_lr.
    """
    if not new_params:
        return

    # Check if these params are already in the optimizer
    existing_ids = {id(p) for group in optimizer.param_groups
                    for p in group["params"]}
    truly_new = [p for p in new_params if id(p) not in existing_ids]

    if not truly_new:
        return

    lr = base_lr / 10 if is_backbone else base_lr
    optimizer.add_param_group({
        "params": truly_new,
        "lr": lr,
        "weight_decay": 1e-4,
        "eps": 1e-8,
    })
    log.info("  Added %d new params to optimizer at lr=%.2e", len(truly_new), lr)


PHASES = [
    (0,  lambda m, opt: (
        _set_grad(m.backbone, False),
        log.info("  freeze: backbone (head-only warmup)")
    )),
    (4,  lambda m, opt: (
        _set_grad(m.backbone.layer5, True),
        unfreeze_and_add_params(opt, list(m.backbone.layer5.parameters()), LR, is_backbone=True),
        log.info("  unfreeze: layer5")
    )),
    (7,  lambda m, opt: (
        _set_grad(m.backbone.layer4, True),
        unfreeze_and_add_params(opt, list(m.backbone.layer4.parameters()), LR, is_backbone=True),
        log.info("  unfreeze: layer4")
    )),
    (10, lambda m, opt: (
        _set_grad(m.backbone, True),
        unfreeze_and_add_params(opt, [p for p in m.backbone.parameters() if not p.requires_grad], LR, is_backbone=True),
        log.info("  unfreeze: all backbone")
    )),
]

def build_optimizer(model: nn.Module, base_lr: float) -> AdamW:
    """
    Create AdamW optimizer with differential learning rates.
    Backbone parameters use lr/10 for stable fine-tuning,
    while head parameters use the full base_lr for faster adaptation.

    Args:
        model: PyTorch model.
        base_lr: Base learning rate for head parameters.

    Returns:
        Configured AdamW optimizer.
    """
    backbone_params = [p for n, p in model.named_parameters()
                       if "backbone" in n and p.requires_grad]
    head_params = [p for n, p in model.named_parameters()
                   if "backbone" not in n and p.requires_grad]

    groups = []
    if backbone_params:
        groups.append({"params": backbone_params, "lr": base_lr / 10})
    if head_params:
        groups.append({"params": head_params, "lr": base_lr})

    return AdamW(groups, weight_decay=1e-4, eps=1e-8)


def build_scheduler(opt, total_epochs, warmup, base_lr):
    """
    Create learning rate scheduler: linear warmup + cosine annealing.
    
    Args:
        opt: Optimizer instance.
        total_epochs: Total number of training epochs.
        warmup: Number of warmup epochs.
        base_lr: Base learning rate.
    
    Returns:
        SequentialLR scheduler instance.
    """
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
# Main Training Loop
# ─────────────────────────────────────────────────────────────────────────────
def train():
    """Main training function for HiFT fine-tuning."""
    log.info("Fine-tuning HiFT | epochs=%d | BS=%d | device=%s | lr=%g",
             EPOCHS, BATCH_SIZE, DEVICE, LR)

    # ── Dataset Preparation ────────────────────────────────────────────────
    train_ds, val_ds = TrainingDataset.split_train_val(
        manifest_path=os.path.join(ROOT, "data/contest_release/metadata/contestant_manifest.json"),
        data_root=os.path.join(ROOT, "data/contest_release"),
        val_ratio=0.15,
        train_samples=45_000,  # Number of training samples per epoch
        val_samples=2_000,     # Number of validation samples
        template_size=128,     # Template image size
        search_size=256,       # Search image size
        seed=42,               # Random seed for reproducibility
        jitter_sigma=0.25,     # Augmentation jitter strength
    )

    # Create data loaders with num_workers support
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True,
                              persistent_workers=(NUM_WORKERS > 0),
                              prefetch_factor=2)  # Preload 2 batches for smoother training
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True,
                            persistent_workers=(NUM_WORKERS > 0),
                            prefetch_factor=2)

    # ── Model Initialization ───────────────────────────────────────────────
    model = HiFT().to(DEVICE)
    model.load_pretrained(CKPT_PATH, device=DEVICE)

    # Freeze backbone initially for head-only warmup
    _set_grad(model.backbone, False)
    log.info("  freeze: backbone (head-only warmup)")

    # Setup optimizer and scheduler
    optimizer = build_optimizer(model, LR)
    scheduler = build_scheduler(optimizer, EPOCHS, WARMUP_EPOCHS, LR)

    # Mixed precision training setup
    # [CHANGE] Use USE_AMP flag instead of hardcoded check
    use_amp = USE_AMP and "cuda" in DEVICE
    scaler = GradScaler("cuda") if use_amp else None

    os.makedirs(SAVE_DIR, exist_ok=True)
    best_val = float("inf")

    # ── Timing: Start total training timer ─────────────────────────────────
    total_start = time.time()
    log.info("Training started at %s", time.strftime("%H:%M:%S"))

    # ── Training Loop ──────────────────────────────────────────────────────
    for ep in range(EPOCHS):
        epoch_start = time.time()
        
        # Handle staged backbone unfreezing at specified epochs
        for phase_ep, phase_fn in PHASES:
            if ep == phase_ep:
                log.info("=== Epoch %d: phase transition ===", ep)
                # [FIX] Pass optimizer to phase function instead of rebuilding
                phase_fn(model, optimizer)
                # [CHANGE] Removed optimizer/scheduler rebuild — handled by add_param_group

        # ── Training Phase ─────────────────────────────────────────────────
        model.train()
        total_loss, loc_loss_sum, cls_loss_sum, n = 0.0, 0.0, 0.0, 0
        batch_times = []  # Track batch processing times for ETA calculation

        for i, batch in enumerate(train_loader):
            batch_start = time.time()
            
            # Move batch data to device
            tmpl = batch["template"].to(DEVICE, non_blocking=True)
            srch = batch["search"].to(DEVICE, non_blocking=True)
            gt = batch["gt_box"].to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            # Forward + backward pass with optional mixed precision
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

            # Record batch timing for ETA estimation
            batch_time = time.time() - batch_start
            batch_times.append(batch_time)

            # Accumulate loss metrics
            total_loss += loss.item()
            loc_loss_sum += l_loc.item()
            cls_loss_sum += l_cls.item()
            n += 1

            # Log progress every 50 steps with timing information
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

        # ── Epoch Summary ──────────────────────────────────────────────────
        epoch_time = time.time() - epoch_start
        elapsed_total = time.time() - total_start
        avg_epoch_time = elapsed_total / (ep + 1)
        remaining_epochs = EPOCHS - ep - 1
        eta_total = avg_epoch_time * remaining_epochs
        
        scheduler.step()

        # ── Validation Phase ───────────────────────────────────────────────
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
        avg_val = val_loss / max(nv, 1)
        
        log.info(
            "Epoch %3d/%d | train=%.4f (loc=%.4f cls=%.4f) | val=%.4f | "
            "lr=%.2e | epoch_time=%s | val_time=%s | ETA %s",
            ep + 1, EPOCHS, avg_train,
            loc_loss_sum / max(n, 1), cls_loss_sum / max(n, 1),
            avg_val, optimizer.param_groups[-1]["lr"],
            format_time(epoch_time), format_time(val_time),
            format_time(eta_total)
        )

        # Save best model based on validation loss
        if avg_val < best_val:
            best_val = avg_val
            save_path = os.path.join(SAVE_DIR, "hift_finetuned_best_v2.pth")
            torch.save(model.state_dict(), save_path)
            log.info("  Saved best -> %s (val=%.4f)", save_path, best_val)

    # ── Final Training Summary ─────────────────────────────────────────────
    total_time = time.time() - total_start
    log.info("=" * 70)
    log.info("TRAINING COMPLETE")
    log.info("   Total time      : %s", format_time(total_time))
    log.info("   Avg time/epoch  : %s", format_time(total_time / EPOCHS))
    log.info("   Best val loss   : %.4f", best_val)
    log.info("   Model saved to  : %s", SAVE_DIR)
    log.info("=" * 70)


if __name__ == "__main__":
    train()