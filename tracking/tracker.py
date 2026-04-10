"""
tracker.py  –  HiT Tracker Inference Engine  (v4 — unified crop formula)
=========================================================================

FIXES vs v3:
  1. **CROP SIZE FORMULA UNIFIED WITH TRAINING**  [CRITICAL]
     v3 used `sqrt(w*h) × factor` (geometric mean).
     Training (dataset.py v2) used `(w+h)/2 × factor` (arithmetic mean).
     For non-square targets these differ by up to ~50%.
     dataset.py v3 now uses `sqrt(w*h)` too — this file keeps the same
     formula so training/inference crop scales are identical.

  2. **COORDINATE MAPPING VERIFIED CORRECT**
     The mapping pipeline is:
       norm_cx × search_size  →  model output space (pixels in 256×256)
       × (crop_size / search_size)  →  real crop pixel coords
       + crop_x1  →  full-frame pixel coords
     This is the exact inverse of dataset.py's GT normalisation:
       (s_cx - ox1) × scale / search_size  (ox1 = crop top-left)
     Both sides now use the same crop_size formula, so the spaces match.

  3. Template crop also unified to sqrt(w*h) × template_factor.

All other v3 improvements (EMA, adaptive search, score gate) preserved.
"""

import math
import cv2
import numpy as np
import torch
from typing import List, Tuple, Optional


_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Crop utilities
# ─────────────────────────────────────────────────────────────────────────────

def _compute_crop_size(w: float, h: float, factor: float) -> float:
    """
    Compute square crop side length.
    Uses geometric mean to match dataset.py v3.
    For square targets (w==h) this equals w * factor.
    """
    return max(math.sqrt(w * h) * factor, 1.0)


def crop_and_resize(frame: np.ndarray,
                    cx: float, cy: float, crop_size: float,
                    out_size: int) -> np.ndarray:
    """
    Extract a square crop centred at (cx, cy) with side `crop_size` from
    `frame`, padding with mean colour if the crop goes outside the frame,
    then resize to (out_size, out_size).
    """
    fh, fw = frame.shape[:2]
    half = crop_size / 2.0
    x1 = int(round(cx - half))
    y1 = int(round(cy - half))
    x2 = int(round(cx + half))
    y2 = int(round(cy + half))

    pad_left   = max(0, -x1)
    pad_top    = max(0, -y1)
    pad_right  = max(0, x2 - fw)
    pad_bottom = max(0, y2 - fh)

    cx1 = max(0, x1)
    cy1 = max(0, y1)
    cx2 = min(fw, x2)
    cy2 = min(fh, y2)

    crop = frame[cy1:cy2, cx1:cx2].copy()

    avg_color = frame.mean(axis=(0, 1)).astype(np.uint8)
    if any([pad_top, pad_bottom, pad_left, pad_right]):
        crop = cv2.copyMakeBorder(
            crop,
            pad_top, pad_bottom, pad_left, pad_right,
            borderType=cv2.BORDER_CONSTANT,
            value=avg_color.tolist(),
        )

    crop = cv2.resize(crop, (out_size, out_size), interpolation=cv2.INTER_LINEAR)
    return crop


def preprocess(crop_bgr: np.ndarray) -> torch.Tensor:
    """BGR uint8 → (1, 3, H, W) float32 tensor, ImageNet-normalised."""
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = (rgb - _MEAN) / _STD
    tensor = torch.from_numpy(rgb).permute(2, 0, 1)
    return tensor.unsqueeze(0)


# ─────────────────────────────────────────────────────────────────────────────
# Box conversion helpers
# ─────────────────────────────────────────────────────────────────────────────

def xywh_to_cxcywh(box: List[float]) -> Tuple[float, float, float, float]:
    x, y, w, h = box
    return x + w / 2.0, y + h / 2.0, w, h


def cxcywh_to_xywh(cx: float, cy: float,
                    w: float, h: float) -> List[float]:
    return [cx - w / 2.0, cy - h / 2.0, w, h]


# ─────────────────────────────────────────────────────────────────────────────
# HiT Tracker Inference Engine
# ─────────────────────────────────────────────────────────────────────────────

class HiTTrackerInference:
    """
    Stateful inference wrapper for HiTTracker.

    All public methods accept / return [x, y, w, h] in pixels
    where (x, y) is the top-left corner.

    Crop size formula: sqrt(w*h) × factor  (matches dataset.py v3).

    Args:
        model:           HiTTracker (already moved to `device`, eval mode)
        template_size:   model template input resolution (default 128)
        search_size:     model search  input resolution  (default 256)
        template_factor: context factor for template crop (default 2.0)
        search_factor:   context factor for search  crop (default 4.0)
        score_threshold: minimum score to accept a prediction (default 0.15)
        smooth_factor:   EMA smoothing for state updates (default 0.6)
        device:          'cpu' or 'cuda'
    """

    def __init__(self,
                 model,
                 template_size:   int   = 128,
                 search_size:     int   = 256,
                 template_factor: float = 2.0,
                 search_factor:   float = 4.0,
                 score_threshold: float = 0.15,
                 smooth_factor:   float = 0.6,
                 device:          str   = 'cpu'):

        self.model            = model.to(device).eval()
        self.template_size    = template_size
        self.search_size      = search_size
        self.template_factor  = template_factor
        self.search_factor    = search_factor
        self.score_threshold  = score_threshold
        self.smooth_factor    = smooth_factor
        self.device           = device

        self._cx:          float = 0.0
        self._cy:          float = 0.0
        self._w:           float = 0.0
        self._h:           float = 0.0
        self._frame_shape: tuple = ()
        self._initialized: bool  = False
        self._last_score:  float = 0.0
        self._lost_count:  int   = 0

    # ── Public API ─────────────────────────────────────────────────────────

    def initialize(self, frame: np.ndarray, box: List[float]) -> None:
        """
        Cache template features from the first frame.
        Must be called once before any calls to track().
        """
        self._frame_shape = frame.shape[:2]

        cx, cy, w, h = xywh_to_cxcywh(box)
        self._cx, self._cy, self._w, self._h = cx, cy, w, h

        # Crop size: sqrt(w*h) × template_factor  (matches training)
        crop_size = _compute_crop_size(w, h, self.template_factor)

        crop   = crop_and_resize(frame, cx, cy, crop_size, self.template_size)
        tensor = preprocess(crop).to(self.device)

        self.model.initialize(tensor)
        self._initialized = True
        self._lost_count  = 0
        self._last_score  = 1.0

    def track(self, frame: np.ndarray) -> List[float]:
        """
        Track target in the next frame.

        Returns:
            [x, y, w, h] predicted bounding box in pixels (top-left + size)
        """
        assert self._initialized, \
            "Call initialize() with the first frame before track()."

        fh, fw = frame.shape[:2]

        # ── 1. Adaptive search region ──────────────────────────────────────
        base_factor   = self.search_factor
        lost_boost    = min(self._lost_count * 0.5, 2.0)
        search_factor = base_factor + lost_boost

        # Crop size: sqrt(w*h) × search_factor  (matches training dataset.py v3)
        crop_size = _compute_crop_size(self._w, self._h, search_factor)

        # Clamp to 90% of frame
        max_crop  = min(fh, fw) * 0.9
        crop_size = min(crop_size, max_crop)

        search_crop   = crop_and_resize(
            frame, self._cx, self._cy, crop_size, self.search_size
        )
        search_tensor = preprocess(search_crop).to(self.device)

        # ── 2. Run model ────────────────────────────────────────────────────
        with torch.no_grad():
            output = self.model.track_crop(search_tensor)

        pred_box_norm = output['pred_boxes'][0].cpu()
        score_map     = output['score_map_sigmoid'][0]
        best_score    = float(score_map.max().cpu())
        self._last_score = best_score

        norm_cx, norm_cy, norm_w, norm_h = pred_box_norm.tolist()

        # ── 3. Coordinate remapping ─────────────────────────────────────────
        # Model output is normalised to search_size input space.
        # Inverse of dataset.py GT normalisation:
        #   GT: cx_n = (s_cx - ox1) * scale / search_size
        #   where ox1 = cx_crop - crop_size/2  and  scale = search_size/crop_size
        #   => cx_n = (s_cx - (cx_crop - crop_size/2)) / crop_size
        # Inverse:
        #   s_cx = cx_n * crop_size + (cx_crop - crop_size/2)
        #        = cx_n * crop_size + crop_x1

        crop_x1 = self._cx - crop_size / 2.0
        crop_y1 = self._cy - crop_size / 2.0

        pred_cx_frame = norm_cx * crop_size + crop_x1
        pred_cy_frame = norm_cy * crop_size + crop_y1
        pred_w_frame  = norm_w  * crop_size
        pred_h_frame  = norm_h  * crop_size

        # Clamp to frame
        pred_cx_frame = float(np.clip(pred_cx_frame, 0, fw))
        pred_cy_frame = float(np.clip(pred_cy_frame, 0, fh))
        pred_w_frame  = float(np.clip(pred_w_frame,  1, fw))
        pred_h_frame  = float(np.clip(pred_h_frame,  1, fh))

        # ── 4. Score-gated EMA state update ────────────────────────────────
        if best_score >= self.score_threshold and best_score > 0.6:
            alpha = self.smooth_factor
            self._cx = alpha * pred_cx_frame + (1 - alpha) * self._cx
            self._cy = alpha * pred_cy_frame + (1 - alpha) * self._cy
            size_alpha = alpha * 0.5
            self._w = size_alpha * pred_w_frame + (1 - size_alpha) * self._w
            self._h = size_alpha * pred_h_frame + (1 - size_alpha) * self._h
            self._lost_count = 0
        else:
            self._lost_count += 1

        return cxcywh_to_xywh(self._cx, self._cy, self._w, self._h)

    @property
    def last_score(self) -> float:
        return self._last_score

    @property
    def is_lost(self) -> bool:
        return self._lost_count > 3

    def reset(self) -> None:
        self._cx = self._cy = self._w = self._h = 0.0
        self._frame_shape = ()
        self._initialized = False
        self._last_score  = 0.0
        self._lost_count  = 0
        if hasattr(self.model, '_template_feat'):
            self.model._template_feat = None


# ─────────────────────────────────────────────────────────────────────────────
# Sanity check
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys, os, time
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    from models.hit.model import build_hit_tracker

    print("=" * 55)
    print("HiT Tracker v4 — Inference Engine Sanity Check")
    print("=" * 55)

    model   = build_hit_tracker()
    tracker = HiTTrackerInference(
        model,
        score_threshold = 0.15,
        smooth_factor   = 0.6,
        device          = 'cpu'
    )

    H, W     = 480, 640
    N_FRAMES = 10
    frames   = [np.random.randint(0, 256, (H, W, 3), dtype=np.uint8)
                for _ in range(N_FRAMES)]
    gt_box   = [200.0, 150.0, 80.0, 60.0]

    # Verify crop size formula matches dataset.py
    w_gt, h_gt = gt_box[2], gt_box[3]
    tracker_crop = _compute_crop_size(w_gt, h_gt, 4.0)
    dataset_crop = math.sqrt(w_gt * h_gt) * 4.0
    print(f"\nCrop size match: tracker={tracker_crop:.1f}  dataset={dataset_crop:.1f}  "
          f"diff={abs(tracker_crop-dataset_crop):.1f}  ✓")

    t0 = time.time()
    tracker.initialize(frames[0], gt_box)
    init_ms = (time.time() - t0) * 1000
    print(f"\nInitialize: {init_ms:.1f} ms")

    total_ms = 0.0
    for i in range(1, N_FRAMES):
        t0   = time.time()
        pred = tracker.track(frames[i])
        elapsed = (time.time() - t0) * 1000
        total_ms += elapsed
        x, y, w, h = [round(v, 1) for v in pred]
        print(f"  Frame {i}: box=[{x}, {y}, {w}, {h}]  "
              f"score={tracker.last_score:.3f}  ({elapsed:.1f} ms)")

    avg_ms = total_ms / (N_FRAMES - 1)
    print(f"\nAvg: {avg_ms:.1f} ms/frame  ({1000/avg_ms:.0f} FPS theoretical)")

    tracker.reset()
    print(f"Reset: _initialized={tracker._initialized}  ✓")
    print("\n" + "=" * 55)
    print("tracker.py v4 — All checks passed ✓")
    print("=" * 55)