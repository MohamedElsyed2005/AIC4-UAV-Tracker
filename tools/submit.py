"""
submit.py  –  HiFT Single Sequence Runner + Video Output
==========================================================

Usage:
python tools/submit.py \
  --checkpoint checkpoints/hift_finetuned_best.pth \
  --manifest data/contest_release/metadata/contestant_manifest.json \
  --data_root data/contest_release \
  --seq_id dataset1/Car_video \
  --output submission.csv \
  --video_output result.mp4 \
  --device cuda
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import cv2
import numpy as np

# ─────────────────────────────────────────────────────────────
# PATH SETUP (CRITICAL)
# ─────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from data.dataset import InferenceSequence
from models.hift_full import HiFT


def parse_args():
    p = argparse.ArgumentParser()
    
    p.add_argument("--checkpoint", default="checkpoints/hift_finetuned_best.pth")
    p.add_argument("--manifest", default="data/contest_release/metadata/contestant_manifest.json")
    p.add_argument("--data_root", default="data/contest_release")
    p.add_argument("--output", default="submission.csv")
    p.add_argument("--video_output", default="result.mp4")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--split", default="public_lb")
    p.add_argument("--seq_id", required=True)
    p.add_argument("--score_threshold", type=float, default=0.15)
    p.add_argument("--smooth_factor", type=float, default=0.6)
    p.add_argument("--max_frames", type=int, default=None, help="Limit frames for quick test")
    
    return p.parse_args()


def draw_box(frame, box, color=(0, 255, 0), thickness=2):
    """Draw bounding box on frame."""
    x, y, w, h = [int(round(v)) for v in box]
    x, y = max(0, x), max(0, y)
    return cv2.rectangle(frame.copy(), (x, y), (x + w, y + h), color, thickness)


def get_crop(frame, center, size):
    """
    Extract square crop centered at `center` with side `size`.
    Pads with gray (114) if crop goes outside frame.
    """
    h, w = frame.shape[:2]
    cx, cy = int(round(center[0])), int(round(center[1]))
    half = size // 2
    
    x1, x2 = cx - half, cx + half
    y1, y2 = cy - half, cy + half
    
    # Compute padding needed
    pad_l = max(0, -x1); pad_r = max(0, x2 - w)
    pad_t = max(0, -y1); pad_b = max(0, y2 - h)
    
    # Clamp to frame bounds
    x1_clamped = max(0, x1); x2_clamped = min(w, x2)
    y1_clamped = max(0, y1); y2_clamped = min(h, y2)
    
    crop = frame[y1_clamped:y2_clamped, x1_clamped:x2_clamped].copy()
    
    # Pad if needed
    if pad_t or pad_b or pad_l or pad_r:
        crop = cv2.copyMakeBorder(
            crop, pad_t, pad_b, pad_l, pad_r,
            cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )
    
    return crop


def preprocess_crop(crop_bgr, target_size):
    """BGR uint8 crop → normalized tensor [1, 3, target_size, target_size]."""
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    # ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    rgb = (rgb - mean) / std
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
    return tensor


def decode_loc(loc, search_size):
    """
    Decode loc output [B, 4, H, W] to box [x, y, w, h] in search-crop pixel coords.
    Model outputs normalized values in [0, 1] via sigmoid.
    """
    # Average over spatial dims to get single prediction [B, 4]
    pred = torch.sigmoid(loc.mean(dim=(2, 3)))  # [B, 4] in [0, 1]
    pred = pred.squeeze(0).cpu().numpy()  # [4]
    
    cx_norm, cy_norm, w_norm, h_norm = pred
    
    # Convert normalized to pixel coords in search crop
    cx_px = cx_norm * search_size
    cy_px = cy_norm * search_size
    w_px = w_norm * search_size
    h_px = h_norm * search_size
    
    # Convert center-format to top-left format
    x = cx_px - w_px / 2
    y = cy_px - h_px / 2
    
    return [x, y, w_px, h_px]


class HiFTTracker:
    """Simple, robust inference tracker for HiFT model."""
    
    def __init__(self, model, score_threshold=0.15, smooth_factor=0.6, device="cuda"):
        self.model = model
        self.score_threshold = score_threshold
        self.smooth_factor = smooth_factor
        self.device = device
        self.template = None
        self.prev_box = None
        self.tmpl_size = 127      # Must match model's expected template size
        self.search_size = 360    # Must match model's expected search size
        
    def reset(self):
        self.template = None
        self.prev_box = None
        
    def initialize(self, frame, box):
        """Initialize with first frame + GT box [x, y, w, h]."""
        # Convert to center format
        cx = box[0] + box[2] / 2.0
        cy = box[1] + box[3] / 2.0
        
        # Extract and preprocess template crop
        tmpl_crop = get_crop(frame, (cx, cy), size=self.tmpl_size)
        self.template = preprocess_crop(tmpl_crop, self.tmpl_size).to(self.device)
        
        # Store initial box
        self.prev_box = [float(v) for v in box]
        
    def track(self, frame):
        """Track object in new frame, return box [x, y, w, h] in frame coords."""
        if self.template is None:
            raise RuntimeError("Call initialize() before track()")
        
        H, W = frame.shape[:2]
        
        # Current estimated center
        cx = self.prev_box[0] + self.prev_box[2] / 2.0
        cy = self.prev_box[1] + self.prev_box[3] / 2.0
        
        # Extract and preprocess search crop
        search_crop = get_crop(frame, (cx, cy), size=self.search_size)
        search = preprocess_crop(search_crop, self.search_size).to(self.device)
        
        # Run model
        self.model.eval()
        with torch.no_grad():
            loc, cls1, cls2 = self.model(self.template, search)
        
        # Decode prediction in search-crop coords
        pred_box_crop = decode_loc(loc, self.search_size)
        
        # Map from search-crop coords to original frame coords
        # crop_x0, crop_y0 = top-left of search crop in original frame
        half = self.search_size // 2
        crop_x0 = cx - half
        crop_y0 = cy - half
        
        x = pred_box_crop[0] + crop_x0
        y = pred_box_crop[1] + crop_y0
        w = pred_box_crop[2]
        h = pred_box_crop[3]
        
        # Safety clamp to frame bounds
        x = max(0, min(x, W - 1))
        y = max(0, min(y, H - 1))
        w = max(10, min(w, W))
        h = max(10, min(h, H))
        
        # EMA smoothing with previous box
        if self.prev_box is not None:
            sf = self.smooth_factor
            x = sf * self.prev_box[0] + (1 - sf) * x
            y = sf * self.prev_box[1] + (1 - sf) * y
            w = sf * self.prev_box[2] + (1 - sf) * w
            h = sf * self.prev_box[3] + (1 - sf) * h
        
        # Final clamp after smoothing
        x = max(0, min(x, W - w))
        y = max(0, min(y, H - h))
        w = max(10, min(w, W - x))
        h = max(10, min(h, H - y))
        
        self.prev_box = [x, y, w, h]
        return [x, y, w, h]


def main():
    args = parse_args()

    print(f"\n[config]")
    print(f"  checkpoint : {args.checkpoint}")
    print(f"  seq_id     : {args.seq_id}")
    print(f"  device     : {args.device}")
    print(f"  output     : {args.output}")
    print(f"  video      : {args.video_output}\n")

    # ── Load model ────────────────────────────────────────────
    print("[load] Loading HiFT model...")
    model = HiFT().to(args.device)
    model.load_pretrained(args.checkpoint, device=args.device)
    model.eval()
    print("✅ Model loaded\n")

    # ── Init tracker ──────────────────────────────────────────
    tracker = HiFTTracker(
        model,
        score_threshold=args.score_threshold,
        smooth_factor=args.smooth_factor,
        device=args.device,
    )

    # ── Load sequence info ────────────────────────────────────
    with open(args.manifest) as f:
        manifest = json.load(f)

    sequences = manifest.get(args.split, {})
    
    seq_info = None
    for k, v in sequences.items():
        sid = f"{v['dataset']}/{v['seq_name']}"
        if sid == args.seq_id:
            seq_info = v
            break

    if seq_info is None:
        available = [f"{v['dataset']}/{v['seq_name']}" for v in sequences.values()]
        raise ValueError(f"Sequence '{args.seq_id}' not found.\nAvailable: {available[:10]}")

    seq_id = f"{seq_info['dataset']}/{seq_info['seq_name']}"
    print(f"[run] Sequence: {seq_id}")

    # ── Load sequence data ────────────────────────────────────
    seq = InferenceSequence(seq_info, args.data_root)
    init_frame, init_box = seq.get_init()
    
    tracker.reset()
    tracker.initialize(init_frame, init_box)
    print(f"✅ Initialized with box: {init_box}\n")

    # ── Video writer setup ────────────────────────────────────
    H, W = init_frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(args.video_output, fourcc, 30, (W, H))

    # ── CSV setup ─────────────────────────────────────────────
    all_rows = []

    # Frame 0: write GT box
    video_writer.write(draw_box(init_frame, init_box, color=(0, 0, 255), thickness=2))
    all_rows.append((f"{seq_id}_0", int(init_box[0]), int(init_box[1]),
                     int(init_box[2]), int(init_box[3])))

    # ── Tracking loop ─────────────────────────────────────────
    t0 = time.time()
    frame_count = 0
    max_frames = args.max_frames or seq_info.get("n_frames", 99999)

    for frame_idx, frame in seq:
        if frame_count >= max_frames:
            break
        frame_count += 1
        
        # Track
        pred_box = tracker.track(frame)
        pred_box_int = [int(round(v)) for v in pred_box]

        # Record for CSV
        seq.record(frame_idx, pred_box_int)
        all_rows.append((f"{seq_id}_{frame_idx}",
                         pred_box_int[0], pred_box_int[1],
                         pred_box_int[2], pred_box_int[3]))

        # Visualize and write video
        vis = draw_box(frame, pred_box_int, color=(0, 255, 0), thickness=2)
        cv2.putText(vis, f"Frame: {frame_idx}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        video_writer.write(vis)

        # Progress log
        if frame_count % 50 == 0:
            print(f"  📹 Processed {frame_count} frames...")

    # ── Cleanup ───────────────────────────────────────────────
    video_writer.release()
    seq.release()

    # ── Save CSV ──────────────────────────────────────────────
    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "x", "y", "w", "h"])
        writer.writerows(all_rows)

    elapsed = time.time() - t0
    fps = frame_count / elapsed if elapsed > 0 else 0

    # ── Summary ───────────────────────────────────────────────
    print(f"\n✅ Done!")
    print(f"  Sequence : {seq_id}")
    print(f"  Frames   : {frame_count}")
    print(f"  Time     : {elapsed:.2f}s")
    print(f"  FPS      : {fps:.2f}")
    print(f"  CSV      : {args.output}")
    print(f"  Video    : {args.video_output}\n")


if __name__ == "__main__":
    main()