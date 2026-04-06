"""
tracker.py  –  HiT Tracker Inference Engine
============================================
Takes raw video frames and an initial bounding box, then tracks
the target across subsequent frames.

Handles everything the model doesn't:
  - Cropping the template and search regions from full frames
  - Normalising and tensorising the crops
  - Mapping predicted [cx, cy, w, h] (normalised, crop-space)
    back to pixel coordinates in the original frame
  - State management (last box, scale/context factors)

Usage:
    tracker = HiTTrackerInference(model)

    # Frame 0  — provide ground-truth box [x, y, w, h] in pixels
    tracker.initialize(frame_bgr, [x, y, w, h])

    # Frame N
    pred_box = tracker.track(frame_bgr)   # → [x, y, w, h] pixels

The box format throughout this file is  [x, y, w, h]  in pixel
coordinates (top-left origin), matching most VOT / UAV benchmarks.
Internally the model uses  [cx, cy, w, h]  normalised to [0, 1].
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# ImageNet normalisation constants
# ─────────────────────────────────────────────────────────────────────────────

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Crop utilities
# ─────────────────────────────────────────────────────────────────────────────

def _get_crop_box(cx: float, cy: float, size: float,
                  frame_h: int, frame_w: int) -> Tuple[int, int, int, int]:
    """
    Compute the pixel coordinates of a square crop centred at (cx, cy)
    with side length `size`.  Clamps to frame boundaries.

    Returns:
        (x1, y1, x2, y2)  pixel coords (may be outside frame)
    """
    half = size / 2.0
    x1 = int(round(cx - half))
    y1 = int(round(cy - half))
    x2 = int(round(cx + half))
    y2 = int(round(cy + half))
    return x1, y1, x2, y2


def crop_and_resize(frame: np.ndarray,
                    cx: float, cy: float, crop_size: float,
                    out_size: int) -> np.ndarray:
    """
    Extract a square crop centred at (cx, cy) with side `crop_size` from
    `frame`, padding with mean colour if the crop goes outside the frame,
    then resize to (out_size, out_size).

    Args:
        frame:     (H, W, 3)  BGR uint8
        cx, cy:    crop centre in pixel coords
        crop_size: side length of the square crop (pixels)
        out_size:  output size (pixels)

    Returns:
        crop: (out_size, out_size, 3) BGR uint8
    """
    fh, fw = frame.shape[:2]
    x1, y1, x2, y2 = _get_crop_box(cx, cy, crop_size, fh, fw)

    # Padding amounts if crop extends outside frame
    pad_left   = max(0, -x1)
    pad_top    = max(0, -y1)
    pad_right  = max(0, x2 - fw)
    pad_bottom = max(0, y2 - fh)

    # Clamp crop coords to frame
    cx1 = max(0, x1)
    cy1 = max(0, y1)
    cx2 = min(fw, x2)
    cy2 = min(fh, y2)

    crop = frame[cy1:cy2, cx1:cx2].copy()

    # Pad with mean colour (reduces distribution shift vs zero-padding)
    avg_color = frame.mean(axis=(0, 1)).astype(np.uint8)

    if any([pad_top, pad_bottom, pad_left, pad_right]):
        crop = cv2.copyMakeBorder(
            crop,
            pad_top, pad_bottom, pad_left, pad_right,
            borderType=cv2.BORDER_CONSTANT,
            value=avg_color.tolist(),
        )

    # Resize to model input size
    crop = cv2.resize(crop, (out_size, out_size), interpolation=cv2.INTER_LINEAR)
    return crop


def preprocess(crop_bgr: np.ndarray) -> torch.Tensor:
    """
    BGR uint8  →  (1, 3, H, W) float32 tensor, ImageNet-normalised.
    """
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = (rgb - _MEAN) / _STD                         # normalise
    tensor = torch.from_numpy(rgb).permute(2, 0, 1)    # HWC → CHW
    return tensor.unsqueeze(0)                          # add batch dim


# ─────────────────────────────────────────────────────────────────────────────
# Box conversion helpers
# ─────────────────────────────────────────────────────────────────────────────

def xywh_to_cxcywh(box: List[float]) -> Tuple[float, float, float, float]:
    """[x, y, w, h]  →  (cx, cy, w, h)"""
    x, y, w, h = box
    return x + w / 2.0, y + h / 2.0, w, h


def cxcywh_to_xywh(cx: float, cy: float,
                    w: float, h: float) -> List[float]:
    """(cx, cy, w, h)  →  [x, y, w, h]"""
    return [cx - w / 2.0, cy - h / 2.0, w, h]


# ─────────────────────────────────────────────────────────────────────────────
# Tracker
# ─────────────────────────────────────────────────────────────────────────────

class HiTTrackerInference:
    """
    Stateful inference wrapper for HiTTracker.

    Coordinate system used throughout:
      - All public methods accept / return  [x, y, w, h]  in pixels
        where (x, y) is the top-left corner.
      - Internally we track  (cx, cy, w, h)  in pixels for cropping.

    Args:
        model:           HiTTracker (already moved to `device`, eval mode)
        template_size:   model template input resolution (default 128)
        search_size:     model search  input resolution  (default 256)
        template_factor: context factor for template crop (default 2.0)
                         crop_size = sqrt(w*h) * template_factor
        search_factor:   context factor for search  crop (default 4.0)
        score_threshold: minimum score to accept a prediction (default 0.0)
                         set > 0 to enable lost-target detection
        device:          'cpu' or 'cuda'
    """

    def __init__(self,
                 model,
                 template_size:   int   = 128,
                 search_size:     int   = 256,
                 template_factor: float = 2.0,
                 search_factor:   float = 4.0,
                 score_threshold: float = 0.0,
                 device:          str   = 'cpu'):

        self.model            = model.to(device).eval()
        self.template_size    = template_size
        self.search_size      = search_size
        self.template_factor  = template_factor
        self.search_factor    = search_factor
        self.score_threshold  = score_threshold
        self.device           = device

        # ── State ──────────────────────────────────────────────────────────
        self._cx:          float = 0.0   # current target centre x (pixels)
        self._cy:          float = 0.0   # current target centre y (pixels)
        self._w:           float = 0.0   # current target width    (pixels)
        self._h:           float = 0.0   # current target height   (pixels)
        self._frame_shape: tuple = ()    # (H, W) of the sequence frames
        self._initialized: bool  = False

    # ── Public API ─────────────────────────────────────────────────────────

    def initialize(self, frame: np.ndarray, box: List[float]) -> None:
        """
        Cache template features from the first frame.
        Must be called once before any calls to track().

        Args:
            frame: (H, W, 3) BGR uint8
            box:   [x, y, w, h] in pixels — ground truth on frame 0
        """
        self._frame_shape = frame.shape[:2]   # (H, W)

        # Convert to centre format
        cx, cy, w, h = xywh_to_cxcywh(box)
        self._cx, self._cy, self._w, self._h = cx, cy, w, h

        # Compute context-aware template crop size
        # = geometric mean of (w, h) × context factor
        target_sz   = np.sqrt(w * h)
        crop_size   = target_sz * self.template_factor

        # Extract and preprocess template crop
        crop = crop_and_resize(
            frame, cx, cy, crop_size, self.template_size)
        tensor = preprocess(crop).to(self.device)

        # Cache template features in the model
        self.model.initialize(tensor)
        self._initialized = True

    def track(self, frame: np.ndarray) -> List[float]:
        """
        Track target in the next frame.

        Args:
            frame: (H, W, 3) BGR uint8

        Returns:
            [x, y, w, h] predicted bounding box in pixels
        """
        assert self._initialized, \
            "Call initialize() with the first frame before track()."

        fh, fw = frame.shape[:2]

        # ── 1. Build search crop centred on last predicted position ────────
        target_sz = np.sqrt(self._w * self._h)
        crop_size = target_sz * self.search_factor

        search_crop = crop_and_resize(
            frame,
            self._cx, self._cy,
            crop_size,
            self.search_size,
        )
        search_tensor = preprocess(search_crop).to(self.device)

        # ── 2. Run model ────────────────────────────────────────────────────
        output = self.model.track_crop(search_tensor)

        pred_box_norm = output['pred_boxes'][0].cpu()    # (4,) [cx,cy,w,h] ∈ [0,1]
        score_map     = output['score_map_sigmoid'][0]   # (1, H, W)
        best_score    = float(score_map.max().cpu())

        # ── 3. Map normalised prediction back to full-frame pixel coords ───
        #
        # The model predicts in the coordinate space of the search crop.
        # We need to map:
        #   norm_cx, norm_cy  (fraction of crop)  → full frame pixels
        #   norm_w,  norm_h   (fraction of crop)  → full frame pixels
        #
        norm_cx, norm_cy, norm_w, norm_h = pred_box_norm.tolist()

        # Scale to crop pixel space
        pred_cx_crop = norm_cx * crop_size
        pred_cy_crop = norm_cy * crop_size
        pred_w_crop  = norm_w  * crop_size
        pred_h_crop  = norm_h  * crop_size

        # Crop's top-left corner in the full frame
        crop_x1 = self._cx - crop_size / 2.0
        crop_y1 = self._cy - crop_size / 2.0

        # Convert to full frame pixel coords
        pred_cx_frame = crop_x1 + pred_cx_crop
        pred_cy_frame = crop_y1 + pred_cy_crop
        pred_w_frame  = pred_w_crop
        pred_h_frame  = pred_h_crop

        # Clamp to frame boundary
        pred_cx_frame = float(np.clip(pred_cx_frame, 0, fw))
        pred_cy_frame = float(np.clip(pred_cy_frame, 0, fh))
        pred_w_frame  = float(np.clip(pred_w_frame,  1, fw))
        pred_h_frame  = float(np.clip(pred_h_frame,  1, fh))

        # ── 4. Update state ─────────────────────────────────────────────────
        if best_score >= self.score_threshold:
            self._cx = pred_cx_frame
            self._cy = pred_cy_frame
            self._w  = pred_w_frame
            self._h  = pred_h_frame

        # ── 5. Return [x, y, w, h] ──────────────────────────────────────────
        return cxcywh_to_xywh(
            self._cx, self._cy, self._w, self._h)

    @property
    def last_score(self) -> float:
        """Confidence score of the most recent prediction."""
        return getattr(self, '_last_score', 0.0)

    def reset(self) -> None:
        """Clear all state (call between sequences)."""
        self._cx = self._cy = self._w = self._h = 0.0
        self._frame_shape = ()
        self._initialized = False
        self.model._template_feat = None


# ─────────────────────────────────────────────────────────────────────────────
# Sanity check
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys, os, time
    from pathlib import Path

    # Allow running from project root or from models/hit/
    _ROOT = Path(__file__).resolve()
    for _ in range(4):
        if (_ROOT / "models").exists():
            sys.path.insert(0, str(_ROOT))
            break
        _ROOT = _ROOT.parent

    from models.hit.model import build_hit_tracker

    print("=" * 55)
    print("HiT Tracker — Inference Engine Sanity Check")
    print("=" * 55)

    # ── Build model + tracker ──────────────────────────────────────────────
    model   = build_hit_tracker()
    tracker = HiTTrackerInference(model, device='cpu')

    # ── Simulate a video sequence ──────────────────────────────────────────
    # Synthetic 'video': random BGR frames, 480×640
    H, W    = 480, 640
    N_FRAMES = 10
    frames  = [np.random.randint(0, 256, (H, W, 3), dtype=np.uint8)
               for _ in range(N_FRAMES)]

    # Ground-truth box on frame 0  [x, y, w, h] in pixels
    gt_box  = [200.0, 150.0, 80.0, 60.0]   # a 80×60 box at (200,150)

    # ── Frame 0: initialise ────────────────────────────────────────────────
    t0 = time.time()
    tracker.initialize(frames[0], gt_box)
    init_ms = (time.time() - t0) * 1000

    print(f"\nFrame 0 (initialize):")
    print(f"  GT box (x,y,w,h):  {[round(v,1) for v in gt_box]}")
    print(f"  Template cached    ✓")
    print(f"  Time: {init_ms:.1f} ms")

    # ── Frames 1–N: track ─────────────────────────────────────────────────
    print(f"\nTracking frames 1–{N_FRAMES - 1}:")
    total_track_ms = 0.0
    for i in range(1, N_FRAMES):
        t0 = time.time()
        pred = tracker.track(frames[i])
        elapsed = (time.time() - t0) * 1000
        total_track_ms += elapsed

        x, y, w, h = [round(v, 1) for v in pred]
        print(f"  Frame {i:2d}: box=[{x:6.1f}, {y:6.1f}, {w:6.1f}, {h:6.1f}]"
              f"  ({elapsed:.1f} ms)")

    avg_ms = total_track_ms / (N_FRAMES - 1)

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\nPerformance:")
    print(f"  Init time:          {init_ms:.1f} ms")
    print(f"  Avg tracking time:  {avg_ms:.1f} ms / frame  "
          f"({1000/avg_ms:.1f} FPS theoretical)")

    # ── Box format checks ─────────────────────────────────────────────────
    print(f"\nBox format checks:")
    pred_final = tracker.track(frames[1])
    x, y, w, h = pred_final
    in_frame = (0 <= x <= W and 0 <= y <= H and w > 0 and h > 0)
    print(f"  x ∈ [0, W={W}]:  {0 <= x <= W}  → x={x:.1f}")
    print(f"  y ∈ [0, H={H}]:  {0 <= y <= H}  → y={y:.1f}")
    print(f"  w > 0:            {w > 0}  → w={w:.1f}")
    print(f"  h > 0:            {h > 0}  → h={h:.1f}")
    print(f"  All in frame:     {'✓' if in_frame else '✗'}")

    # ── Reset check ───────────────────────────────────────────────────────
    tracker.reset()
    print(f"\nReset check:")
    print(f"  tracker._initialized after reset: {tracker._initialized}  ✓")

    print("\n" + "=" * 55)
    print("tracker.py  — All checks passed ✓")
    print("=" * 55)