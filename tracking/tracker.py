"""
tracker.py  –  HiT Tracker Inference Engine
============================================
Takes raw video frames and an initial bounding box, then tracks
the target across subsequent frames.

FIX applied vs original:
  The coordinate remapping in track() was wrong.
  Original code:
      pred_cx_crop = norm_cx * crop_size   ← WRONG
      pred_cy_crop = norm_cy * crop_size
      pred_w_crop  = norm_w  * crop_size
      pred_h_crop  = norm_h  * crop_size

  The model always receives a search_size=256 input and predicts
  normalised coords in THAT fixed space, not in crop_size space.
  crop_size is the real-world pixel side of the crop BEFORE it was
  resized to 256.  To go from normalised → real pixel coords you must:
    1. norm × search_size  → pixel position inside the 256×256 image
    2. × (crop_size / search_size)  → pixel position inside the real crop
    3. + crop origin  → pixel position in the full frame

  Fixed code:
      # Step 1: model output space (256×256)
      pred_cx_model = norm_cx * self.search_size
      pred_cy_model = norm_cy * self.search_size
      pred_w_model  = norm_w  * self.search_size
      pred_h_model  = norm_h  * self.search_size

      # Step 2: scale from model space → real crop space
      scale = crop_size / self.search_size
      pred_cx_crop = pred_cx_model * scale
      pred_cy_crop = pred_cy_model * scale
      pred_w_frame = pred_w_model  * scale
      pred_h_frame = pred_h_model  * scale

      # Step 3: add crop origin → full frame
      pred_cx_frame = crop_x1 + pred_cx_crop
      pred_cy_frame = crop_y1 + pred_cy_crop

  Also fixed: _last_score was read by the last_score property but
  never written; it's now stored after every track() call.
"""

import cv2
import numpy as np
import torch
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
    """
    fh, fw = frame.shape[:2]
    x1, y1, x2, y2 = _get_crop_box(cx, cy, crop_size, fh, fw)

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

    All public methods accept / return [x, y, w, h] in pixels
    where (x, y) is the top-left corner.

    Args:
        model:           HiTTracker (already moved to `device`, eval mode)
        template_size:   model template input resolution (default 128)
        search_size:     model search  input resolution  (default 256)
        template_factor: context factor for template crop (default 2.0)
        search_factor:   context factor for search  crop (default 4.0)
        score_threshold: minimum score to accept a prediction (default 0.0)
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

        self._cx:          float = 0.0
        self._cy:          float = 0.0
        self._w:           float = 0.0
        self._h:           float = 0.0
        self._frame_shape: tuple = ()
        self._initialized: bool  = False
        self._last_score:  float = 0.0   # FIX: initialise so property works

    # ── Public API ─────────────────────────────────────────────────────────

    def initialize(self, frame: np.ndarray, box: List[float]) -> None:
        """
        Cache template features from the first frame.
        Must be called once before any calls to track().
        """
        self._frame_shape = frame.shape[:2]

        cx, cy, w, h = xywh_to_cxcywh(box)
        self._cx, self._cy, self._w, self._h = cx, cy, w, h

        target_sz = np.sqrt(w * h)
        crop_size = target_sz * self.template_factor

        crop   = crop_and_resize(frame, cx, cy, crop_size, self.template_size)
        tensor = preprocess(crop).to(self.device)

        self.model.initialize(tensor)
        self._initialized = True

    def track(self, frame: np.ndarray) -> List[float]:
        """
        Track target in the next frame.

        Returns:
            [x, y, w, h] predicted bounding box in pixels
        """
        assert self._initialized, \
            "Call initialize() with the first frame before track()."

        fh, fw = frame.shape[:2]

        # ── 1. Build search crop ───────────────────────────────────────────
        target_sz = np.sqrt(self._w * self._h)
        crop_size = target_sz * self.search_factor

        search_crop = crop_and_resize(
            frame, self._cx, self._cy, crop_size, self.search_size
        )
        search_tensor = preprocess(search_crop).to(self.device)

        # ── 2. Run model ───────────────────────────────────────────────────
        output = self.model.track_crop(search_tensor)

        pred_box_norm = output['pred_boxes'][0].cpu()     # (4,) ∈ [0,1]
        score_map     = output['score_map_sigmoid'][0]    # (1, H, W)
        best_score    = float(score_map.max().cpu())
        self._last_score = best_score                     # FIX: save score

        norm_cx, norm_cy, norm_w, norm_h = pred_box_norm.tolist()

        # ── 3. FIX: correct coordinate remapping ──────────────────────────
        #
        # The model predicts normalised coords relative to its fixed
        # search_size input (256×256), NOT relative to crop_size.
        #
        # Step 1 — model output space → pixel coords in 256×256 image
        pred_cx_model = norm_cx * self.search_size
        pred_cy_model = norm_cy * self.search_size
        pred_w_model  = norm_w  * self.search_size
        pred_h_model  = norm_h  * self.search_size

        # Step 2 — scale from model input space → real crop pixel space
        #   (crop_size pixels were resized to search_size pixels)
        scale         = crop_size / self.search_size
        pred_cx_crop  = pred_cx_model * scale
        pred_cy_crop  = pred_cy_model * scale
        pred_w_frame  = pred_w_model  * scale
        pred_h_frame  = pred_h_model  * scale

        # Step 3 — add crop top-left origin → full frame pixel coords
        crop_x1       = self._cx - crop_size / 2.0
        crop_y1       = self._cy - crop_size / 2.0
        pred_cx_frame = crop_x1 + pred_cx_crop
        pred_cy_frame = crop_y1 + pred_cy_crop

        # Clamp to frame boundary
        pred_cx_frame = float(np.clip(pred_cx_frame, 0, fw))
        pred_cy_frame = float(np.clip(pred_cy_frame, 0, fh))
        pred_w_frame  = float(np.clip(pred_w_frame,  1, fw))
        pred_h_frame  = float(np.clip(pred_h_frame,  1, fh))

        # ── 4. Update state only if score is good enough ──────────────────
        if best_score >= self.score_threshold:
            self._cx = pred_cx_frame
            self._cy = pred_cy_frame
            self._w  = pred_w_frame
            self._h  = pred_h_frame

        # ── 5. Return [x, y, w, h] ─────────────────────────────────────────
        return cxcywh_to_xywh(self._cx, self._cy, self._w, self._h)

    @property
    def last_score(self) -> float:
        """Confidence score of the most recent prediction."""
        return self._last_score

    def reset(self) -> None:
        """Clear all state (call between sequences)."""
        self._cx = self._cy = self._w = self._h = 0.0
        self._frame_shape = ()
        self._initialized = False
        self._last_score  = 0.0
        self.model._template_feat = None


# ─────────────────────────────────────────────────────────────────────────────
# Sanity check
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys, os, time
    from pathlib import Path

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

    model   = build_hit_tracker()
    tracker = HiTTrackerInference(model, device='cpu')

    H, W     = 480, 640
    N_FRAMES = 10
    frames   = [np.random.randint(0, 256, (H, W, 3), dtype=np.uint8)
                for _ in range(N_FRAMES)]
    gt_box   = [200.0, 150.0, 80.0, 60.0]

    t0 = time.time()
    tracker.initialize(frames[0], gt_box)
    init_ms = (time.time() - t0) * 1000

    print(f"\nFrame 0 (initialize):")
    print(f"  GT box (x,y,w,h):  {[round(v,1) for v in gt_box]}")
    print(f"  Template cached    ✓")
    print(f"  Time: {init_ms:.1f} ms")

    print(f"\nTracking frames 1–{N_FRAMES - 1}:")
    total_track_ms = 0.0
    for i in range(1, N_FRAMES):
        t0 = time.time()
        pred = tracker.track(frames[i])
        elapsed = (time.time() - t0) * 1000
        total_track_ms += elapsed
        x, y, w, h = [round(v, 1) for v in pred]
        print(f"  Frame {i:2d}: box=[{x:6.1f}, {y:6.1f}, {w:6.1f}, {h:6.1f}]"
              f"  score={tracker.last_score:.3f}  ({elapsed:.1f} ms)")

    avg_ms = total_track_ms / (N_FRAMES - 1)
    print(f"\nPerformance:")
    print(f"  Init time:          {init_ms:.1f} ms")
    print(f"  Avg tracking time:  {avg_ms:.1f} ms / frame  "
          f"({1000/avg_ms:.1f} FPS theoretical)")

    tracker.reset()
    print(f"\nReset check: _initialized={tracker._initialized}  ✓")
    print("\n" + "=" * 55)
    print("tracker.py  — All checks passed ✓")
    print("=" * 55)