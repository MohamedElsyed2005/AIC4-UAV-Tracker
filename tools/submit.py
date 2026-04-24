"""
submit.py  —  AIC-4 HiFT UAV Tracker  (Inference + Visualization)
==================================================================
Architecture-aware inference pipeline for HiFT with AlexNet backbone.

Usage:
    python submit.py --video_path <video> --checkpoint <ckpt.pth>
    python submit.py --video_path <video> --checkpoint <ckpt.pth> --save_video out.mp4
    python submit.py --video_path <video> --checkpoint <ckpt.pth> --init_box x,y,w,h

Design decisions (based on codebase analysis):
  1. Template/Search sizes exactly match training: 128 / 256
  2. Context factors from TrainingDataset: context_tmpl=2.0, context_srch=4.0
  3. Decoding: grid_sample at cls2 peak → sigmoid(loc) → normalised [cx,cy,w,h]
  4. Checkpoint: tries EMA weights first (more stable), falls back to model/raw
  5. Search crop: centered on current box, side = sqrt(w*h) * context_srch
  6. Template: fixed at first frame (standard SOT protocol)
  7. Stability:
       - EMA smoothing on predicted bbox (alpha tunable)
       - Adaptive search area expansion on low-confidence frames
       - Velocity-based search centre prediction for fast motion
       - Re-detection (global search) after N consecutive low-conf frames
  8. Peak selection: weighted centroid around cls2 peak (sub-pixel precision)
  9. No future-frame access (online-only, as required by competition rules)
"""

import argparse
import math
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

# ── Path setup: allow running from project root or alongside model files ──────
_HERE = Path(__file__).resolve().parent
for candidate in [_HERE, _HERE.parent]:
    model_dir = candidate / "models"
    if model_dir.exists():
        sys.path.insert(0, str(candidate))
        break
else:
    sys.path.insert(0, str(_HERE))

# ── Import model and preprocessing helpers ────────────────────────────────────
try:
    from models.hift_full import HiFT
except ImportError:
    # Fallback: model file in the same directory
    try:
        from models.hift_full import HiFT
    except ImportError as e:
        raise ImportError(
            "Cannot import HiFT. Make sure hift_full.py is on the Python path.\n"
            f"Tried: {_HERE} and {_HERE.parent}/models/\n"
            f"Original error: {e}"
        )

# ──────────────────────────────────────────────────────────────────────────────
# Constants — must match training pipeline exactly
# ──────────────────────────────────────────────────────────────────────────────
TEMPLATE_SIZE   = 128       # dataset.py: template_size=128
SEARCH_SIZE     = 256       # dataset.py: search_size=256
CONTEXT_TMPL    = 2.0       # dataset.py: context_tmpl=2.0
CONTEXT_SRCH    = 4.0       # dataset.py: context_srch=4.0

# ImageNet normalisation (dataset.py: _MEAN, _STD)
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Tracking hyper-parameters
EMA_ALPHA           = 0.35   # bbox EMA smoothing (lower = smoother but laggier)
CONF_THRESHOLD      = 0.15   # cls2 peak below this → low confidence
REDETECT_PATIENCE   = 8      # consecutive low-conf frames before global re-detect
VELOCITY_DAMPING    = 0.75   # velocity EMA decay (prevents over-shooting)
SEARCH_EXPAND_RATIO = 1.5    # expand search area on low-confidence frames
MAX_SEARCH_EXPAND   = 3.0    # cap on search area expansion multiplier
PEAK_WINDOW         = 3      # weighted centroid radius (sub-pixel peak)


# ──────────────────────────────────────────────────────────────────────────────
# Preprocessing  (identical to dataset.py)
# ──────────────────────────────────────────────────────────────────────────────

def crop_square(
    frame: np.ndarray,
    cx: float, cy: float,
    side: float,
    out_size: int,
) -> Tuple[np.ndarray, float, float, float]:
    """
    Crop a square of `side` pixels centred at (cx, cy), resize to out_size.
    Returns: (crop, scale, ox1, oy1)
      - scale = out_size / side
      - ox1, oy1 = top-left corner in ORIGINAL frame coords (may be negative)

    Identical to dataset.py::crop_square — must not diverge from training.
    """
    H, W = frame.shape[:2]
    side = max(side, 1.0)

    x1 = int(round(cx - side / 2))
    y1 = int(round(cy - side / 2))
    x2 = int(round(cx + side / 2))
    y2 = int(round(cy + side / 2))
    ox1, oy1 = float(x1), float(y1)

    pt = max(0, -y1); pl = max(0, -x1)
    pb = max(0, y2 - H); pr = max(0, x2 - W)

    if pt or pl or pb or pr:
        frame = cv2.copyMakeBorder(frame, pt, pb, pl, pr,
                                   cv2.BORDER_CONSTANT, value=(114, 114, 114))
        x1 += pl; x2 += pl; y1 += pt; y2 += pt

    patch = frame[y1:y2, x1:x2]
    actual_side = max(patch.shape[0], patch.shape[1], 1)
    crop = cv2.resize(patch, (out_size, out_size), interpolation=cv2.INTER_LINEAR)
    scale = out_size / actual_side
    return crop, scale, ox1, oy1


def to_tensor(img: np.ndarray) -> torch.Tensor:
    """
    uint8 HWC BGR → float32 CHW RGB, ImageNet-normalised.
    Identical to dataset.py::to_tensor.
    """
    img = img[:, :, ::-1].copy().astype(np.float32) / 255.0
    img = (img - _MEAN) / _STD
    return torch.from_numpy(img.transpose(2, 0, 1))


# ──────────────────────────────────────────────────────────────────────────────
# Model loading
# ──────────────────────────────────────────────────────────────────────────────

def load_model(checkpoint_path: str, device: torch.device) -> torch.nn.Module:
    """
    Load HiFT from checkpoint, preferring EMA weights (best generalisation).
    Handles all checkpoint formats used by finetune_full_hift.py:
      - Full training state:  {model: ..., ema: ..., epoch: ...}
      - Pure weight dict
      - load_pretrained() wrappers with state_dict / net / model keys
    """
    print(f"\n[submit.py] Loading checkpoint: {checkpoint_path}")
    raw = torch.load(checkpoint_path, map_location=device, weights_only=False)

    def _extract_sd(obj) -> Optional[dict]:
        if isinstance(obj, dict):
            n = sum(1 for v in obj.values() if isinstance(v, torch.Tensor))
            if n > 10:
                return obj
        return None

    sd = None

    if isinstance(raw, dict):
        # Prefer EMA weights — trained with ema_decay=0.9998, more stable
        for key in ("ema", "model", "state_dict", "net"):
            candidate = raw.get(key)
            if candidate is not None:
                extracted = _extract_sd(candidate)
                if extracted is not None:
                    sd = extracted
                    print(f"  → Using weights from key: '{key}'")
                    break
        if sd is None:
            sd = _extract_sd(raw)
            if sd is not None:
                print("  → Treating entire dict as weight dict")
    else:
        sd = dict(raw)
        print("  → Non-dict checkpoint (OrderedDict?)")

    if sd is None:
        raise RuntimeError("Could not extract weights from checkpoint.")

    # Strip common prefixes from compiled / DDP checkpoints
    cleaned = {
        k.replace("module.", "").replace("_orig_mod.", ""): v
        for k, v in sd.items()
    }

    model = HiFT()
    model_sd = model.state_dict()
    loaded, skipped = [], []
    new_sd = {}
    for k, v in cleaned.items():
        if k in model_sd and model_sd[k].shape == v.shape:
            new_sd[k] = v
            loaded.append(k)
        else:
            skipped.append(k)

    model_sd.update(new_sd)
    model.load_state_dict(model_sd, strict=False)
    model = model.to(device)
    model.eval()

    missing = [k for k in model.state_dict() if k not in loaded]
    print(f"  Loaded  : {len(loaded)}/{len(model_sd)} keys")
    if missing:
        print(f"  Missing : {len(missing)} (random init) — first 5: {missing[:5]}")
    if skipped:
        print(f"  Skipped : {len(skipped)} (shape mismatch) — first 5: {skipped[:5]}")
    print()
    return model


# ──────────────────────────────────────────────────────────────────────────────
# Decoding  (mirrors finetune_full_hift.py::compute_val_iou + loc_loss_fn)
# ──────────────────────────────────────────────────────────────────────────────

def decode_at_point(
    loc: torch.Tensor,
    cx_n: float,
    cy_n: float,
) -> np.ndarray:
    """
    Decode the loc map at normalised position (cx_n, cy_n).

    Training uses:
        gx = 2*gt[:,0] - 1      (normalised [-1,1] for grid_sample)
        gy = 2*gt[:,1] - 1
        pred = sigmoid(grid_sample(loc, grid, align_corners=False))

    Returns float32 array [cx_n, cy_n, w_n, h_n] in normalised search coords.
    """
    gx = 2.0 * cx_n - 1.0
    gy = 2.0 * cy_n - 1.0
    grid = torch.tensor([[[[gx, gy]]]], dtype=torch.float32, device=loc.device)
    pred = torch.sigmoid(
        F.grid_sample(loc, grid, mode="bilinear",
                      padding_mode="border", align_corners=False)
    )  # shape [1,4,1,1]
    return pred.view(4).cpu().float().numpy()


def find_peak_cls2(
    cls2: torch.Tensor,
    window: int = PEAK_WINDOW,
) -> Tuple[float, float, float]:
    """
    Find the peak location in cls2 (1-channel heatmap) with sub-pixel precision
    via weighted centroid in a local window around the argmax.

    Returns: (cx_n, cy_n, peak_score)
      where cx_n, cy_n are in [0,1] (normalised search-region coordinates)
    """
    # cls2 shape: [1, 1, H, W]
    hmap = torch.sigmoid(cls2[0, 0]).cpu().float().numpy()   # H x W
    H, W = hmap.shape

    flat_idx = np.argmax(hmap)
    py, px   = divmod(int(flat_idx), W)
    peak_val = float(hmap[py, px])

    # Weighted centroid around peak for sub-pixel accuracy
    y0 = max(0, py - window); y1 = min(H, py + window + 1)
    x0 = max(0, px - window); x1 = min(W, px + window + 1)
    patch = hmap[y0:y1, x0:x1]
    patch = np.maximum(patch - patch.min(), 0)
    total = patch.sum() + 1e-8

    ys = np.arange(y0, y1).astype(np.float32)
    xs = np.arange(x0, x1).astype(np.float32)
    YY, XX = np.meshgrid(ys, xs, indexing="ij")
    cy_map = float((YY * patch).sum() / total)
    cx_map = float((XX * patch).sum() / total)

    # Convert map coordinates → normalised [0,1]
    cx_n = (cx_map + 0.5) / W
    cy_n = (cy_map + 0.5) / H
    cx_n = float(np.clip(cx_n, 0.0, 1.0))
    cy_n = float(np.clip(cy_n, 0.0, 1.0))

    return cx_n, cy_n, peak_val


# ──────────────────────────────────────────────────────────────────────────────
# Coordinate conversion helpers
# ──────────────────────────────────────────────────────────────────────────────

def norm_to_frame(
    cx_n: float, cy_n: float, w_n: float, h_n: float,
    search_cx: float, search_cy: float,
    search_side: float,
    frame_h: int, frame_w: int,
) -> List[float]:
    """
    Convert normalised search-crop bbox → absolute frame bbox [x,y,w,h].

    The search crop has:
      - origin (top-left) at (search_cx - search_side/2, search_cy - search_side/2)
      - scale = SEARCH_SIZE / search_side  (pixels per original pixel)

    So: cx_frame = (cx_n * SEARCH_SIZE) / scale + ox1
    """
    scale = SEARCH_SIZE / search_side
    ox1   = search_cx - search_side / 2
    oy1   = search_cy - search_side / 2

    cx_px = cx_n * SEARCH_SIZE / scale + ox1
    cy_px = cy_n * SEARCH_SIZE / scale + oy1
    w_px  = w_n  * SEARCH_SIZE / scale
    h_px  = h_n  * SEARCH_SIZE / scale

    x = cx_px - w_px / 2
    y = cy_px - h_px / 2

    # Clip to frame
    x = float(np.clip(x, 0, frame_w - 1))
    y = float(np.clip(y, 0, frame_h - 1))
    w = float(np.clip(w_px, 1.0, frame_w - x))
    h = float(np.clip(h_px, 1.0, frame_h - y))
    return [x, y, w, h]


def box_to_norm_search(
    box: List[float],
    search_cx: float, search_cy: float,
    search_side: float,
) -> Tuple[float, float]:
    """
    Convert frame-space bbox centre → normalised search-crop coords.
    Used when re-positioning the search crop centre after update.
    """
    cx_f = box[0] + box[2] / 2
    cy_f = box[1] + box[3] / 2
    scale = SEARCH_SIZE / search_side
    ox1   = search_cx - search_side / 2
    oy1   = search_cy - search_side / 2
    cx_n  = (cx_f - ox1) * scale / SEARCH_SIZE
    cy_n  = (cy_f - oy1) * scale / SEARCH_SIZE
    return float(np.clip(cx_n, 0.0, 1.0)), float(np.clip(cy_n, 0.0, 1.0))


# ──────────────────────────────────────────────────────────────────────────────
# HiFT Tracker  (online, no future-frame access)
# ──────────────────────────────────────────────────────────────────────────────

class HiFTTracker:
    """
    Online single-object tracker built around the HiFT architecture.

    Tracking state maintained:
      - template tensor (fixed after init, from first frame)
      - current bbox  [x, y, w, h]  in frame coordinates
      - EMA-smoothed bbox  for stable visualisation
      - velocity estimate [dvx, dvy]  for search-centre prediction
      - confidence history  for re-detection logic
    """

    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        ema_alpha: float = EMA_ALPHA,
        conf_threshold: float = CONF_THRESHOLD,
        redetect_patience: int = REDETECT_PATIENCE,
        velocity_damping: float = VELOCITY_DAMPING,
    ):
        self.model           = model
        self.device          = device
        self.ema_alpha       = ema_alpha
        self.conf_threshold  = conf_threshold
        self.redetect_patience = redetect_patience
        self.velocity_damping  = velocity_damping

        # Initialised in init()
        self._template_tensor: Optional[torch.Tensor] = None
        self._box: Optional[List[float]] = None          # raw last prediction
        self._ema_box: Optional[List[float]] = None      # smoothed prediction
        self._velocity = np.zeros(2, dtype=np.float32)   # [dvx, dvy] in px
        self._low_conf_count = 0
        self._search_expand  = 1.0                       # dynamic search scaling
        self._frame_h = 0
        self._frame_w = 0
        self._frame_count = 0

    # ── Initialisation ────────────────────────────────────────────────────────

    def init(self, frame: np.ndarray, box: List[float]):
        """
        Initialise tracker with first frame and ground-truth bounding box.
        box: [x, y, w, h] in pixel coordinates (top-left + size).
        """
        self._frame_h, self._frame_w = frame.shape[:2]
        self._box = list(box)
        self._ema_box = list(box)
        self._velocity = np.zeros(2, dtype=np.float32)
        self._low_conf_count = 0
        self._search_expand  = 1.0
        self._frame_count    = 0

        # Build template crop exactly as in dataset.py
        cx = box[0] + box[2] / 2
        cy = box[1] + box[3] / 2
        t_side = max(math.sqrt(box[2] * box[3]) * CONTEXT_TMPL, 4.0)
        t_crop, _, _, _ = crop_square(frame, cx, cy, t_side, TEMPLATE_SIZE)
        t_tensor = to_tensor(t_crop).unsqueeze(0).to(self.device)   # [1,3,128,128]
        self._template_tensor = t_tensor
        print(f"[Tracker] Initialised  box={[round(v,1) for v in box]}"
              f"  template_side={t_side:.1f}")

    # ── Per-frame update ──────────────────────────────────────────────────────

    @torch.no_grad()
    def update(self, frame: np.ndarray) -> Tuple[List[float], float]:
        """
        Process one frame and return (predicted_box, confidence).
        predicted_box: [x, y, w, h] in pixel coordinates (smoothed).
        confidence:    peak cls2 score in [0, 1].
        """
        assert self._template_tensor is not None, "Call init() first."
        self._frame_count += 1

        # ── 1. Compute search-crop centre using velocity prediction ──────────
        prev_cx = self._box[0] + self._box[2] / 2
        prev_cy = self._box[1] + self._box[3] / 2
        # Predicted centre = last centre + damped velocity
        pred_cx = prev_cx + self._velocity[0]
        pred_cy = prev_cy + self._velocity[1]
        # Clip to frame
        pred_cx = float(np.clip(pred_cx, 0, self._frame_w - 1))
        pred_cy = float(np.clip(pred_cy, 0, self._frame_h - 1))

        # ── 2. Compute search-crop side (with adaptive expansion) ────────────
        base_side = max(
            math.sqrt(self._box[2] * self._box[3]) * CONTEXT_SRCH,
            8.0,
        )
        s_side = base_side * self._search_expand

        # ── 3. Crop + preprocess search region ──────────────────────────────
        s_crop, s_scale, sox1, soy1 = crop_square(
            frame, pred_cx, pred_cy, s_side, SEARCH_SIZE
        )
        s_tensor = to_tensor(s_crop).unsqueeze(0).to(self.device)   # [1,3,256,256]

        # ── 4. Model inference ───────────────────────────────────────────────
        loc, _cls1, cls2 = self.model(self._template_tensor, s_tensor)
        # loc  : [1, 4, H, W]  — raw bbox map
        # cls2 : [1, 1, H, W]  — confidence heatmap

        # ── 5. Find peak in cls2 with sub-pixel precision ────────────────────
        cx_n, cy_n, conf = find_peak_cls2(cls2)

        # ── 6. Decode bbox at peak location ─────────────────────────────────
        pred_norm = decode_at_point(loc, cx_n, cy_n)
        # pred_norm: [cx_n, cy_n, w_n, h_n] in normalised search-crop coords

        # ── 7. Convert to frame coordinates ─────────────────────────────────
        new_box = norm_to_frame(
            pred_norm[0], pred_norm[1], pred_norm[2], pred_norm[3],
            search_cx=pred_cx, search_cy=pred_cy, search_side=s_side,
            frame_h=self._frame_h, frame_w=self._frame_w,
        )

        # ── 8. Confidence-gated update ───────────────────────────────────────
        if conf >= self.conf_threshold:
            # High confidence: update state normally
            new_cx = new_box[0] + new_box[2] / 2
            new_cy = new_box[1] + new_box[3] / 2

            # Update velocity (EMA)
            dx = new_cx - prev_cx
            dy = new_cy - prev_cy
            self._velocity[0] = (self.velocity_damping * self._velocity[0]
                                  + (1 - self.velocity_damping) * dx)
            self._velocity[1] = (self.velocity_damping * self._velocity[1]
                                  + (1 - self.velocity_damping) * dy)

            # EMA smoothing on the bbox itself
            a = self.ema_alpha
            self._ema_box = [
                a * new_box[i] + (1 - a) * self._ema_box[i]
                for i in range(4)
            ]
            self._box = new_box

            # Reset low-conf counter and search expansion
            self._low_conf_count = 0
            self._search_expand  = max(1.0, self._search_expand * 0.9)

        else:
            # Low confidence: don't update position, just expand search
            self._low_conf_count += 1
            self._search_expand = min(
                self._search_expand * SEARCH_EXPAND_RATIO,
                MAX_SEARCH_EXPAND,
            )
            # Decay velocity more aggressively when lost
            self._velocity *= 0.5

            # Re-detection: reset to global search after patience frames
            if self._low_conf_count >= self.redetect_patience:
                self._redetect(frame)

        return list(self._ema_box), conf

    # ── Re-detection ──────────────────────────────────────────────────────────

    @torch.no_grad()
    def _redetect(self, frame: np.ndarray):
        """
        Global re-detection: scan the full frame with a large search window.
        Strategy: try three large crops centred at frame-thirds + frame centre.
        Pick the crop with the highest cls2 peak as the new state.
        """
        H, W = frame.shape[:2]
        candidates = [
            (W * 0.5,  H * 0.5),   # centre
            (W * 0.25, H * 0.25),  # top-left quadrant
            (W * 0.75, H * 0.25),  # top-right quadrant
            (W * 0.25, H * 0.75),  # bottom-left quadrant
            (W * 0.75, H * 0.75),  # bottom-right quadrant
        ]
        best_conf   = -1.0
        best_box    = None
        large_side  = max(W, H) * 0.8   # cover ~80% of frame

        for (cx, cy) in candidates:
            s_crop, s_scale, sox1, soy1 = crop_square(
                frame, cx, cy, large_side, SEARCH_SIZE
            )
            s_tensor = to_tensor(s_crop).unsqueeze(0).to(self.device)
            loc, _, cls2 = self.model(self._template_tensor, s_tensor)
            cn_x, cn_y, conf = find_peak_cls2(cls2)

            if conf > best_conf:
                best_conf = conf
                pred_norm = decode_at_point(loc, cn_x, cn_y)
                best_box  = norm_to_frame(
                    pred_norm[0], pred_norm[1], pred_norm[2], pred_norm[3],
                    search_cx=cx, search_cy=cy, search_side=large_side,
                    frame_h=H, frame_w=W,
                )

        if best_box is not None and best_conf > self.conf_threshold * 0.5:
            self._box     = best_box
            self._ema_box = best_box
            self._velocity = np.zeros(2, dtype=np.float32)
            self._search_expand  = 1.0
            self._low_conf_count = 0
            print(f"[Tracker] Re-detected at frame {self._frame_count}"
                  f"  conf={best_conf:.3f}")
        else:
            # Keep last known position, reset expansion
            self._low_conf_count = 0
            self._search_expand  = 1.5
            print(f"[Tracker] Re-detection failed at frame {self._frame_count}"
                  f"  holding last position")


# ──────────────────────────────────────────────────────────────────────────────
# Visualisation helpers
# ──────────────────────────────────────────────────────────────────────────────

def draw_overlay(
    frame: np.ndarray,
    box: List[float],
    conf: float,
    frame_idx: int,
    fps: float,
    low_conf: bool,
) -> np.ndarray:
    """Draw bounding box + HUD on frame."""
    vis = frame.copy()
    x, y, w, h = [int(round(v)) for v in box]

    # Bounding box colour: green (high conf) → orange → red (low conf)
    if conf >= 0.5:
        colour = (0, 220, 0)
    elif conf >= CONF_THRESHOLD:
        colour = (0, 165, 255)
    else:
        colour = (0, 0, 220)

    # Draw box with rounded corners feel (thick border)
    cv2.rectangle(vis, (x, y), (x + w, y + h), colour, 2)

    # Corner markers
    arm = min(w, h) // 4
    for (rx, ry), (dx, dy) in [
        ((x,     y    ), ( 1,  1)),
        ((x + w, y    ), (-1,  1)),
        ((x,     y + h), ( 1, -1)),
        ((x + w, y + h), (-1, -1)),
    ]:
        cv2.line(vis, (rx, ry), (rx + dx * arm, ry), colour, 3)
        cv2.line(vis, (rx, ry), (rx, ry + dy * arm), colour, 3)

    # Crosshair at centre
    cx_i = x + w // 2; cy_i = y + h // 2
    cv2.drawMarker(vis, (cx_i, cy_i), colour, cv2.MARKER_CROSS, 10, 1)

    # HUD background
    hud_y = 10
    def _text(img, txt, row, col=10, scale=0.55, clr=(220, 220, 220), thick=1):
        cv2.putText(img, txt, (col, row), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
        cv2.putText(img, txt, (col, row), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, clr, thick, cv2.LINE_AA)

    _text(vis, f"Frame: {frame_idx:05d}",                hud_y + 20)
    _text(vis, f"FPS:   {fps:5.1f}",                     hud_y + 40)
    _text(vis, f"Conf:  {conf:.3f}",                     hud_y + 60,
          clr=(0, 220, 0) if conf >= CONF_THRESHOLD else (0, 80, 220))
    _text(vis, f"Box:   {x},{y},{w},{h}",                hud_y + 80)
    if low_conf:
        _text(vis, "LOW CONFIDENCE", hud_y + 105,
              clr=(0, 100, 255), scale=0.65, thick=2)

    return vis


# ──────────────────────────────────────────────────────────────────────────────
# Main inference loop
# ──────────────────────────────────────────────────────────────────────────────

def run(args):
    # ── Device ────────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"[submit.py] Device: {device}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = load_model(args.checkpoint, device)

    # ── Video ─────────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    native_fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
    vid_w        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h        = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[submit.py] Video: {args.video_path}")
    print(f"  Resolution : {vid_w}×{vid_h}  FPS: {native_fps:.1f}  Frames: {total_frames}")

    # ── Read first frame ──────────────────────────────────────────────────────
    ret, first_frame = cap.read()
    if not ret:
        raise RuntimeError("Cannot read first frame from video.")

    # ── Obtain initial bounding box ───────────────────────────────────────────
    if args.init_box:
        parts = [float(v) for v in args.init_box.split(",")]
        if len(parts) != 4:
            raise ValueError("--init_box must be x,y,w,h (4 comma-separated values)")
        init_box = parts
        print(f"[submit.py] Initial box from args: {init_box}")
    else:
        print("[submit.py] Select target: draw a rectangle with mouse, then press ENTER/SPACE.")
        roi = cv2.selectROI("Select Target (ENTER to confirm, C to cancel)",
                            first_frame, fromCenter=False, showCrosshair=True)
        cv2.destroyWindow("Select Target (ENTER to confirm, C to cancel)")
        if roi[2] <= 0 or roi[3] <= 0:
            raise RuntimeError("Invalid ROI selection.")
        init_box = list(map(float, roi))   # x, y, w, h
        print(f"[submit.py] Selected box: {init_box}")

    # ── Output video writer (optional) ───────────────────────────────────────
    writer = None
    if args.save_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            args.save_video, fourcc, native_fps, (vid_w, vid_h)
        )
        print(f"[submit.py] Saving output to: {args.save_video}")

    # ── Output TXT file (optional) ────────────────────────────────────────────
    txt_file = None
    if args.save_txt:
        txt_file = open(args.save_txt, "w")
        print(f"[submit.py] Saving predictions to: {args.save_txt}")

    # ── Initialise tracker ────────────────────────────────────────────────────
    tracker = HiFTTracker(
        model=model,
        device=device,
        ema_alpha=args.ema_alpha,
        conf_threshold=args.conf_threshold,
        redetect_patience=args.redetect_patience,
    )
    tracker.init(first_frame, init_box)

    # Frame 0: use init box as prediction
    predictions = [list(init_box)]
    if txt_file:
        txt_file.write(f"{init_box[0]:.2f},{init_box[1]:.2f},"
                       f"{init_box[2]:.2f},{init_box[3]:.2f}\n")

    # Draw frame 0
    vis0 = draw_overlay(first_frame, init_box, 1.0, 0, native_fps, False)
    if writer:
        writer.write(vis0)
    if not args.no_display:
        cv2.imshow("HiFT UAV Tracker  [Q to quit]", vis0)
        cv2.waitKey(1)

    # ── Main tracking loop ────────────────────────────────────────────────────
    frame_idx  = 0
    fps_smooth = native_fps
    t_start    = time.time()
    t_prev     = t_start

    print("\n[submit.py] Tracking started. Press Q to quit.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        # Track
        box, conf = tracker.update(frame)
        predictions.append(box)

        # Write prediction to txt
        if txt_file:
            txt_file.write(f"{box[0]:.2f},{box[1]:.2f},"
                           f"{box[2]:.2f},{box[3]:.2f}\n")

        # FPS measurement (exponential moving average)
        t_now  = time.time()
        dt     = t_now - t_prev
        t_prev = t_now
        fps_smooth = 0.9 * fps_smooth + 0.1 * (1.0 / max(dt, 1e-6))

        # Visualise
        low_conf = (conf < args.conf_threshold)
        vis = draw_overlay(frame, box, conf, frame_idx, fps_smooth, low_conf)

        if writer:
            writer.write(vis)

        if not args.no_display:
            cv2.imshow("HiFT UAV Tracker  [Q to quit]", vis)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):   # Q or ESC
                print("[submit.py] Quit by user.")
                break

        # Progress
        if frame_idx % 100 == 0:
            elapsed = time.time() - t_start
            pct     = 100.0 * frame_idx / max(total_frames, 1)
            print(f"  [{frame_idx:5d}/{total_frames}]  {pct:5.1f}%  "
                  f"fps={fps_smooth:.1f}  conf={conf:.3f}")

    # ── Cleanup ───────────────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    avg_fps = frame_idx / max(elapsed, 1e-6)
    print(f"\n[submit.py] Done.  {frame_idx} frames in {elapsed:.1f}s  "
          f"({avg_fps:.1f} fps avg)")

    cap.release()
    if writer:
        writer.release()
        print(f"[submit.py] Saved video: {args.save_video}")
    if txt_file:
        txt_file.close()
        print(f"[submit.py] Saved predictions: {args.save_txt}")
    if not args.no_display:
        cv2.destroyAllWindows()

    return predictions


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="AIC-4 HiFT UAV Tracker — Inference + Visualisation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive ROI selection:
  python submit.py --video_path drone.mp4 --checkpoint checkpoints/best.pth

  # Fixed initial box (x,y,w,h):
  python submit.py --video_path drone.mp4 --checkpoint checkpoints/best.pth \\
                   --init_box 320,240,64,48

  # Save output video + predictions:
  python submit.py --video_path drone.mp4 --checkpoint checkpoints/ema_best.pth \\
                   --save_video output.mp4 --save_txt results.txt

  # Headless (no display window):
  python submit.py --video_path drone.mp4 --checkpoint checkpoints/best.pth \\
                   --no_display --save_txt results.txt
""",
    )
    # Required
    p.add_argument("--video_path",   required=True,
                   help="Path to input video file")
    p.add_argument("--checkpoint",   required=True,
                   help="Path to trained checkpoint (.pth)")

    # Optional
    p.add_argument("--init_box",     default=None,
                   help="Initial bounding box as x,y,w,h (skip interactive selection)")
    p.add_argument("--save_video",   default=None,
                   help="Path to save output video (mp4)")
    p.add_argument("--save_txt",     default=None,
                   help="Path to save per-frame predictions (x,y,w,h, one per line)")
    p.add_argument("--no_display",   action="store_true",
                   help="Disable real-time display window (useful for headless servers)")
    p.add_argument("--cpu",          action="store_true",
                   help="Force CPU inference (default: use CUDA if available)")

    # Tracking hyper-parameters
    p.add_argument("--ema_alpha",         type=float, default=EMA_ALPHA,
                   help=f"Bbox EMA smoothing factor (default: {EMA_ALPHA})")
    p.add_argument("--conf_threshold",    type=float, default=CONF_THRESHOLD,
                   help=f"Confidence threshold for tracking failure (default: {CONF_THRESHOLD})")
    p.add_argument("--redetect_patience", type=int,   default=REDETECT_PATIENCE,
                   help=f"Low-conf frames before global re-detection (default: {REDETECT_PATIENCE})")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args)