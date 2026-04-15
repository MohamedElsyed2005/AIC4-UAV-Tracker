"""
submit.py  —  Production HiFT Tracker (Fixed v2)
Fixes applied:
  1. decode_prediction: uses cls2 peak + grid_sample on loc map (spatial decode).
  2. EMA_CENTRE lowered from 0.55 → 0.35 to reduce lag on fast UAV motion.
  3. Panic mode: runs both tmpl_orig and tmpl_dyn, picks best confidence.
  4. Added run_all_sequences() to produce a single submission.csv covering
     every sequence in every split — required for the actual competition submit.
"""
import argparse
import csv
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from data.dataset import InferenceSequence
from models.hift_full import HiFT

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────
TMPL_SIZE         = 128
SRCH_SIZE         = 256
CONTEXT_TMPL      = 2.0
CONTEXT_SRCH      = 4.0

CONF_THRESH       = 0.40
DIST_GATE_RATIO   = 3.0
SCALE_GATE_MAX    = 1.50   # tightened from 1.60
SCALE_GATE_MIN    = 0.60   # tightened from 0.55

# FIX: lowered from 0.55 → 0.35 to track fast-moving UAV targets without lag
EMA_CENTRE        = 0.35
EMA_SIZE          = 0.75   # slightly more responsive size updates

TMPL_UPDATE_K     = 60
# FIX: lowered from 0.75 → 0.58 so dynamic template actually updates once
#      the cls2 head is trained (previously rarely triggered)
TMPL_UPDATE_CONF  = 0.58

FROZEN_LIMIT      = 5
PANIC_FROZEN_LIMIT = 15
MAX_TRAVEL_BOXES  = 150.0

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# Image utilities
# ──────────────────────────────────────────────────────────────────────────────
def _crop_resize(frame: np.ndarray, cx: float, cy: float, s: float, out_size: int):
    H, W = frame.shape[:2]
    x1 = int(round(cx - s / 2));  y1 = int(round(cy - s / 2))
    x2 = int(round(cx + s / 2));  y2 = int(round(cy + s / 2))
    ox1, oy1 = x1, y1

    pt = max(0, -y1);  pl = max(0, -x1)
    pb = max(0, y2-H); pr = max(0, x2-W)

    if pt or pl or pb or pr:
        frame = cv2.copyMakeBorder(frame, pt, pb, pl, pr,
                                   cv2.BORDER_CONSTANT, value=(114, 114, 114))
        x1 += pl; x2 += pl; y1 += pt; y2 += pt

    patch = frame[y1:y2, x1:x2]
    side_real = max(patch.shape[0], patch.shape[1], 1)
    crop = cv2.resize(patch, (out_size, out_size))
    return crop, float(side_real), ox1, oy1


def _to_tensor(img: np.ndarray) -> torch.Tensor:
    img = img[:, :, ::-1].copy().astype(np.float32) / 255.0
    img = (img - _MEAN) / _STD
    return torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0)


def _box_floats(box) -> list:
    return [float(v) for v in box]


def draw_box(frame, box, color=(0, 255, 0), thickness=2):
    x, y, w, h = [int(round(float(v))) for v in box]
    out = frame.copy()
    cv2.rectangle(out, (x, y), (x + w, y + h), color, thickness)
    return out


# ──────────────────────────────────────────────────────────────────────────────
# FIXED: decode_prediction
# ──────────────────────────────────────────────────────────────────────────────
def decode_prediction(loc: torch.Tensor, cls2: torch.Tensor,
                      s_orig: float, ox1: float, oy1: float):
    """
    Decode HiFT output maps into a bounding box and confidence score.
    
    [FIX v3] Replaced hard argmax with soft-argmax (spatial expectation).
    
    Hard argmax: picks single pixel → noisy when confidence map is flat.
    Soft-argmax: weighted average of ALL pixels by confidence → smooth & stable.
    
    When cls2 is confident (peaked), soft≈hard.
    When cls2 is uncertain (flat), soft returns center → safe fallback.
    
    Also fixes grid shape using torch.stack for consistency with training code.

    loc  : [1, 4, H, W]  raw logits from localization head
    cls2 : [1, 1, H, W]  raw logits from classification head
    s_orig : crop side length in original frame pixels
    ox1, oy1 : top-left of crop in ORIGINAL (pre-padding) frame coords
    """
    with torch.no_grad():
        # ── Confidence: sigmoid on cls2 ───────────────────────────────────
        score_map = torch.sigmoid(cls2[0, 0]).cpu().float()   # [H, W]
        score_np  = score_map.numpy()
        conf      = float(score_np.max())

        H, W = score_np.shape

        # [FIX] Soft-argmax: spatial expectation instead of hard argmax
        # Normalize scores to sum=1 (softmax over spatial dims)
        scores_flat = score_map.view(-1)                        # [H*W]
        # Temperature=10 sharpens the distribution while keeping it differentiable
        weights = torch.softmax(scores_flat * 10.0, dim=0)     # [H*W]
        weights_2d = weights.view(H, W)                        # [H, W]

        # Coordinate grids in [0,1]
        ys = (torch.arange(H).float() + 0.5) / H              # [H]
        xs = (torch.arange(W).float() + 0.5) / W              # [W]
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij") # [H, W]

        # Weighted average position (soft-argmax)
        cx_peak_n = float((weights_2d * grid_x).sum())
        cy_peak_n = float((weights_2d * grid_y).sum())

        # ── Sample loc map at soft position ───────────────────────────────
        # [FIX] Use stack+view for correct grid shape [B, 1, 1, 2]
        gx = torch.tensor(2.0 * cx_peak_n - 1.0)
        gy = torch.tensor(2.0 * cy_peak_n - 1.0)
        grid = torch.stack([gx, gy], dim=-1).view(1, 1, 1, 2)

        # Sample loc at peak position: [1,4,H,W] → [1,4,1,1] → [4]
        loc_cpu = loc.cpu().float()
        pred = F.grid_sample(loc_cpu, grid, mode='bilinear',
                             padding_mode='border', align_corners=False)
        pred = torch.sigmoid(pred).view(4)                    # [cx_n,cy_n,w_n,h_n]

        cx_n, cy_n, w_n, h_n = pred.tolist()

        # ── Back-project to frame coordinates ──────────────────────────────
        cx_frame = cx_n * s_orig + ox1
        cy_frame = cy_n * s_orig + oy1
        w_frame  = w_n  * s_orig
        h_frame  = h_n  * s_orig

        x_frame  = cx_frame - w_frame / 2
        y_frame  = cy_frame - h_frame / 2

    return [x_frame, y_frame, w_frame, h_frame], conf


# ──────────────────────────────────────────────────────────────────────────────
# Tracker
# ──────────────────────────────────────────────────────────────────────────────
class HiFTTracker:
    def __init__(self, model: HiFT, device: str = "cuda"):
        self.model  = model
        self.device = device
        model.eval()

        self._tmpl_orig: torch.Tensor | None = None
        self._tmpl_dyn:  torch.Tensor | None = None

        self._raw_box    = None
        self._smooth_cx  = None
        self._smooth_cy  = None
        self._smooth_w   = None
        self._smooth_h   = None

        self._init_box = None
        self._init_cx  = None
        self._init_cy  = None

        self._last_good_box = None
        self._frozen_count  = 0
        self._frame_count   = 0

    def _make_template(self, frame: np.ndarray, box: list) -> torch.Tensor:
        x, y, w, h = box
        cx = x + w / 2;  cy = y + h / 2
        s  = max(math.sqrt(w * h) * CONTEXT_TMPL, 4.0)
        crop, *_ = _crop_resize(frame, cx, cy, s, TMPL_SIZE)
        return _to_tensor(crop).to(self.device)

    def _make_search(self, frame: np.ndarray, box: list, expand: float = 1.0):
        x, y, w, h = box
        cx = x + w / 2;  cy = y + h / 2
        s  = max(math.sqrt(w * h) * CONTEXT_SRCH * expand, 8.0)
        crop, s_orig, ox1, oy1 = _crop_resize(frame, cx, cy, s, SRCH_SIZE)
        return _to_tensor(crop).to(self.device), s_orig, ox1, oy1

    def initialize(self, frame: np.ndarray, box: list):
        box = _box_floats(box)
        x, y, w, h = box

        self._raw_box   = box
        self._smooth_cx = x + w / 2
        self._smooth_cy = y + h / 2
        self._smooth_w  = w
        self._smooth_h  = h

        self._init_box = box.copy()
        self._init_cx  = x + w / 2
        self._init_cy  = y + h / 2

        self._last_good_box = box
        self._frozen_count  = 0
        self._frame_count   = 0

        self._tmpl_orig = self._make_template(frame, box)
        self._tmpl_dyn  = self._tmpl_orig.clone()

    def _run_model(self, tmpl: torch.Tensor, srch: torch.Tensor,
                   s_orig: float, ox1: float, oy1: float):
        """Run one forward pass and return (box, conf)."""
        with torch.no_grad():
            loc, _, cls2 = self.model(tmpl, srch)
        return decode_prediction(loc, cls2, s_orig, ox1, oy1)

    def track(self, frame: np.ndarray):
        self._frame_count += 1
        H, W = frame.shape[:2]

        # ── SEARCH STRATEGY ─────────────────────────────────────────────────
        is_panic = False
        if self._frozen_count > PANIC_FROZEN_LIMIT:
            # PANIC MODE: scan entire frame
            s_full = max(H, W) * 1.1
            crop, s_orig, ox1, oy1 = _crop_resize(frame, W/2, H/2, s_full, SRCH_SIZE)
            srch_t = _to_tensor(crop).to(self.device)
            conf_thresh_use = 0.20
            is_panic = True
        elif self._frozen_count >= FROZEN_LIMIT * 3:
            search_box = self._last_good_box
            srch_t, s_orig, ox1, oy1 = self._make_search(frame, search_box, 2.5)
            conf_thresh_use = 0.30
        elif self._frozen_count >= FROZEN_LIMIT:
            search_box = self._last_good_box
            srch_t, s_orig, ox1, oy1 = self._make_search(frame, search_box, 1.8)
            conf_thresh_use = 0.35
        else:
            search_box = self._raw_box
            srch_t, s_orig, ox1, oy1 = self._make_search(frame, search_box, 1.0)
            conf_thresh_use = CONF_THRESH

        # ── INFERENCE ───────────────────────────────────────────────────────
        box_pred, conf = self._run_model(self._tmpl_dyn, srch_t, s_orig, ox1, oy1)

        # FIX: In panic mode, also try the original template and keep the best.
        # This helps re-detection when the dynamic template has drifted badly.
        if is_panic:
            box_orig, conf_orig = self._run_model(
                self._tmpl_orig, srch_t, s_orig, ox1, oy1
            )
            if conf_orig > conf:
                box_pred, conf = box_orig, conf_orig

        px, py, pw, ph = box_pred

        if pw < 5 or ph < 5:
            conf = 0.0

        # ── GATING ──────────────────────────────────────────────────────────
        rx, ry, rw, rh = self._raw_box
        prev_cx = rx + rw / 2;  prev_cy = ry + rh / 2
        new_cx  = px + pw / 2;  new_cy  = py + ph / 2

        dist = math.sqrt((new_cx - prev_cx) ** 2 + (new_cy - prev_cy) ** 2)

        if is_panic:
            dist_ok = True
        elif self._frozen_count > 0:
            dist_ok = dist <= max(rw, rh) * 6.0
        else:
            dist_ok = dist <= max(rw, rh) * DIST_GATE_RATIO

        scale_w  = pw / (rw + 1e-6)
        scale_h  = ph / (rh + 1e-6)
        scale_ok = (SCALE_GATE_MIN <= scale_w <= SCALE_GATE_MAX and
                    SCALE_GATE_MIN <= scale_h <= SCALE_GATE_MAX)

        dist_from_init = math.sqrt((new_cx - self._init_cx) ** 2 +
                                   (new_cy - self._init_cy) ** 2)
        max_travel = max(self._init_box[2], self._init_box[3]) * MAX_TRAVEL_BOXES
        anchor_ok  = dist_from_init <= max_travel

        conf_ok = conf >= conf_thresh_use

        if is_panic:
            prediction_good = conf_ok and pw > 10 and ph > 10
        else:
            prediction_good = conf_ok and dist_ok and scale_ok and anchor_ok

        if prediction_good:
            self._frozen_count = 0
            self._raw_box      = box_pred
            self._last_good_box = box_pred

            # FIX: EMA_CENTRE=0.35 (was 0.55) → tracks fast targets without lag
            self._smooth_cx = EMA_CENTRE * self._smooth_cx + (1 - EMA_CENTRE) * new_cx
            self._smooth_cy = EMA_CENTRE * self._smooth_cy + (1 - EMA_CENTRE) * new_cy
            self._smooth_w  = EMA_SIZE   * self._smooth_w  + (1 - EMA_SIZE)   * pw
            self._smooth_h  = EMA_SIZE   * self._smooth_h  + (1 - EMA_SIZE)   * ph

            if self._frame_count % TMPL_UPDATE_K == 0 and conf >= TMPL_UPDATE_CONF:
                self._tmpl_dyn = self._make_template(frame, box_pred)
        else:
            self._frozen_count += 1

        out_x = self._smooth_cx - self._smooth_w / 2
        out_y = self._smooth_cy - self._smooth_h / 2
        return [out_x, out_y, self._smooth_w, self._smooth_h], conf


# ──────────────────────────────────────────────────────────────────────────────
# Single-sequence runner (for debugging / visualising one sequence)
# ──────────────────────────────────────────────────────────────────────────────
def run_single_sequence(args, model):
    tracker = HiFTTracker(model, args.device)

    with open(args.manifest) as f:
        manifest = json.load(f)

    seq_info = None
    for split_seqs in manifest.values():
        if not isinstance(split_seqs, dict):
            continue
        for v in split_seqs.values():
            if f"{v['dataset']}/{v['seq_name']}" == args.seq_id:
                seq_info = v
                break
        if seq_info:
            break

    if seq_info is None:
        raise ValueError(f"Sequence '{args.seq_id}' not found in manifest.")

    seq    = InferenceSequence(seq_info, args.data_root)
    frame0, box0 = seq.get_init()
    tracker.initialize(frame0, _box_floats(box0))

    H, W = frame0.shape[:2]
    save_video = args.video_output.strip() != ""
    vout = None
    if save_video:
        vout = cv2.VideoWriter(args.video_output,
                               cv2.VideoWriter_fourcc(*"mp4v"), 30, (W, H))
        if not vout.isOpened():
            print("[WARN] Video writer failed")
            save_video = False

    csv_rows = []
    if save_video:
        vout.write(draw_box(frame0, box0))
    csv_rows.append((f"{args.seq_id}_0",
                     *[int(round(float(v))) for v in box0]))

    max_frames = args.max_frames or seq_info.get("n_frames", 999_999)
    count = 0

    for data in seq:
        if count >= max_frames:
            break

        if isinstance(data, (tuple, list)) and len(data) == 2:
            frame_idx, frame = data
        else:
            frame_idx = count
            frame = data if not isinstance(data, (tuple, list)) else data[-1]

        if frame is None:
            continue

        box, conf = tracker.track(frame)
        box_int   = [int(round(float(v))) for v in box]
        csv_rows.append((f"{args.seq_id}_{frame_idx}", *box_int))

        fc    = tracker._frozen_count
        color = (0,255,0) if fc==0 else ((0,165,255) if fc<FROZEN_LIMIT else (0,0,255))
        vis   = draw_box(frame, box_int, color=color)
        cv2.putText(vis, f"conf={conf:.2f} fr={fc}", (8, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        if save_video:
            vout.write(vis)
        count += 1

    if save_video and vout is not None:
        vout.release()
    seq.release()

    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "x", "y", "w", "h"])
        writer.writerows(csv_rows)

    print(f"Done — {len(csv_rows)} frames written to {args.output}")


# ──────────────────────────────────────────────────────────────────────────────
# FIX: All-sequence runner — produces the full competition submission CSV
# ──────────────────────────────────────────────────────────────────────────────
def run_all_sequences(args, model):
    """
    Loop over every sequence in every split of the manifest and write one
    unified submission.csv.  This is what the competition evaluator expects.

    Pass --split all  (default) to process every split, or e.g. --split public_lb
    to process only one split.
    """
    with open(args.manifest) as f:
        manifest = json.load(f)

    # Determine which splits to run
    if args.split == "all":
        splits_to_run = list(manifest.keys())
    else:
        splits_to_run = [args.split]

    all_rows = []

    for split_name in splits_to_run:
        split_seqs = manifest.get(split_name, {})
        if not isinstance(split_seqs, dict):
            continue

        seq_list = list(split_seqs.values())
        print(f"\n[split={split_name}] {len(seq_list)} sequences")

        for si, seq_info in enumerate(seq_list):
            seq_id = f"{seq_info['dataset']}/{seq_info['seq_name']}"
            print(f"  [{si+1}/{len(seq_list)}] {seq_id} ...")

            try:
                seq = InferenceSequence(seq_info, args.data_root)
            except (FileNotFoundError, RuntimeError) as e:
                print(f"    SKIP: {e}")
                continue

            # Re-initialise tracker for each sequence
            tracker = HiFTTracker(model, args.device)
            frame0, box0 = seq.get_init()
            tracker.initialize(frame0, _box_floats(box0))

            # First frame — use GT box
            all_rows.append((f"{seq_id}_0",
                              *[int(round(float(v))) for v in box0]))

            max_frames = seq_info.get("n_frames", 999_999)
            count = 0

            for data in seq:
                if count >= max_frames:
                    break

                if isinstance(data, (tuple, list)) and len(data) == 2:
                    frame_idx, frame = data
                else:
                    frame_idx = count
                    frame = data if not isinstance(data, (tuple, list)) else data[-1]

                if frame is None:
                    continue

                box, _ = tracker.track(frame)
                box_int = [int(round(float(v))) for v in box]
                all_rows.append((f"{seq_id}_{frame_idx}", *box_int))
                count += 1

            seq.release()
            print(f"    done ({count+1} frames)")

    # Write unified CSV
    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "x", "y", "w", "h"])
        writer.writerows(all_rows)

    print(f"\nSubmission CSV written: {args.output}  ({len(all_rows)} total rows)")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="HiFT tracker — inference and submission generation"
    )
    p.add_argument("--checkpoint", required=True,
                   help="Path to fine-tuned model checkpoint (.pth)")
    p.add_argument("--manifest",
                   default="data/contest_release/metadata/contestant_manifest.json")
    p.add_argument("--data_root", default="data/contest_release")
    p.add_argument("--output",    default="submission.csv")
    p.add_argument("--device",    default="cuda")

    # Mode selection
    p.add_argument("--mode", choices=["single", "all"], default="all",
                   help="'single': run one sequence (debug/visualise).  "
                        "'all': run all sequences (competition submission).")

    # Single-sequence options
    p.add_argument("--seq_id",      default=None,
                   help="Required when --mode single.  "
                        "Format: dataset/seq_name")
    p.add_argument("--video_output", default="",
                   help="Path for visualisation video (single mode only).")
    p.add_argument("--max_frames",  type=int, default=None)

    # All-sequence options
    p.add_argument("--split", default="all",
                   help="Which manifest split to process in --mode all. "
                        "Use 'all' to process every split (default).")

    return p.parse_args()


def main():
    args = parse_args()

    print(f"[load] HiFT on {args.device} ...")
    model = HiFT().to(args.device)
    model.load_pretrained(args.checkpoint, device=args.device)
    model.eval()

    if args.mode == "single":
        if args.seq_id is None:
            raise ValueError("--seq_id is required when --mode single")
        run_single_sequence(args, model)
    else:
        run_all_sequences(args, model)


if __name__ == "__main__":
    main()