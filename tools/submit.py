import argparse
import csv
import json
import sys
import time
from pathlib import Path

import torch
import cv2
import numpy as np

# ─────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from data.dataset import InferenceSequence
from models.hift_full import HiFT


# ─────────────────────────────────────────────
def to_float(x):
    """Force tensor/np/python → float"""
    if isinstance(x, torch.Tensor):
        return float(x.item())
    return float(x)


def to_box_list(box):
    """Ensure box is pure python list of floats"""
    return [to_float(v) for v in box]


# ─────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--manifest", default="data/contest_release/metadata/contestant_manifest.json")
    p.add_argument("--data_root", default="data/contest_release")
    p.add_argument("--output", default="submission.csv")
    p.add_argument("--video_output", default="result.mp4")
    p.add_argument("--device", default="cuda")
    p.add_argument("--split", default="public_lb")
    p.add_argument("--seq_id", required=True)
    p.add_argument("--max_frames", type=int, default=None)
    return p.parse_args()


# ─────────────────────────────────────────────
def draw_box(frame, box):
    x, y, w, h = map(int, box)
    return cv2.rectangle(frame.copy(), (x, y), (x + w, y + h), (0, 255, 0), 2)


# ─────────────────────────────────────────────
def get_crop(frame, center, size):
    """SAFE crop (forces float)"""
    cx = to_float(center[0])
    cy = to_float(center[1])

    h, w = frame.shape[:2]
    half = size // 2

    x1, x2 = int(cx - half), int(cx + half)
    y1, y2 = int(cy - half), int(cy + half)

    pad_l = max(0, -x1)
    pad_r = max(0, x2 - w)
    pad_t = max(0, -y1)
    pad_b = max(0, y2 - h)

    x1c, x2c = max(0, x1), min(w, x2)
    y1c, y2c = max(0, y1), min(h, y2)

    crop = frame[y1c:y2c, x1c:x2c]

    if pad_l or pad_r or pad_t or pad_b:
        crop = cv2.copyMakeBorder(
            crop, pad_t, pad_b, pad_l, pad_r,
            cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )

    return crop


# ─────────────────────────────────────────────
def preprocess_crop(crop, size):
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    rgb = (rgb - mean) / std
    return torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)


# ─────────────────────────────────────────────
def decode_loc(loc, size):
    pred = torch.sigmoid(loc.mean(dim=(2, 3))).squeeze(0)

    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()

    cx, cy, w, h = pred
    return [
        cx * size - (w * size) / 2,
        cy * size - (h * size) / 2,
        w * size,
        h * size
    ]


# ─────────────────────────────────────────────
class HiFTTracker:
    def __init__(self, model, device="cuda"):
        self.model = model
        self.device = device
        self.template = None
        self.prev_box = None
        self.tmpl_size = 128
        self.search_size = 256

    def initialize(self, frame, box):
        box = to_box_list(box)

        cx = box[0] + box[2] / 2
        cy = box[1] + box[3] / 2

        crop = get_crop(frame, (cx, cy), self.tmpl_size)
        self.template = preprocess_crop(crop, self.tmpl_size).to(self.device)

        self.prev_box = box

    def track(self, frame):
        cx = to_float(self.prev_box[0] + self.prev_box[2] / 2)
        cy = to_float(self.prev_box[1] + self.prev_box[3] / 2)

        search = get_crop(frame, (cx, cy), self.search_size)
        search = preprocess_crop(search, self.search_size).to(self.device)

        with torch.no_grad():
            loc, _, _ = self.model(self.template, search)

        box = decode_loc(loc, self.search_size)

        cx0 = cx - self.search_size / 2
        cy0 = cy - self.search_size / 2

        x = box[0] + cx0
        y = box[1] + cy0
        w = box[2]
        h = box[3]

        if self.prev_box is not None:
            sf = 0.6
            x = sf * self.prev_box[0] + (1 - sf) * x
            y = sf * self.prev_box[1] + (1 - sf) * y
            w = sf * self.prev_box[2] + (1 - sf) * w
            h = sf * self.prev_box[3] + (1 - sf) * h

        self.prev_box = [x, y, w, h]
        return self.prev_box


# ─────────────────────────────────────────────
def main():
    args = parse_args()

    print("[load] HiFT model")
    model = HiFT().to(args.device)
    model.load_pretrained(args.checkpoint, device=args.device)
    model.eval()

    tracker = HiFTTracker(model, args.device)

    with open(args.manifest) as f:
        manifest = json.load(f)

    sequences = manifest[args.split]

    seq_info = None
    for v in sequences.values():
        if f"{v['dataset']}/{v['seq_name']}" == args.seq_id:
            seq_info = v
            break

    seq = InferenceSequence(seq_info, args.data_root)
    frame0, box0 = seq.get_init()

    tracker.initialize(frame0, box0)

    H, W = frame0.shape[:2]
    out = cv2.VideoWriter(args.video_output,
                          cv2.VideoWriter_fourcc(*"mp4v"),
                          30, (W, H))

    csv_rows = []

    out.write(draw_box(frame0, box0))
    csv_rows.append((f"{args.seq_id}_0", *map(int, box0)))

    max_frames = args.max_frames or seq_info.get("n_frames", 99999)

    for data in seq:
        if isinstance(data, tuple) or isinstance(data, list):
            if len(data) == 2:
                frame_idx, frame = data
            else:
                frame_idx = None
                frame = data[-1]
        else:
            frame = data
            frame_idx = None

        if frame is None:
            continue

        box = tracker.track(frame)
        box = [int(round(float(v))) for v in box]

        if frame_idx is None:
            frame_idx = 0

        csv_rows.append((f"{args.seq_id}_{frame_idx}", *box))

        vis = draw_box(frame, box)
        out.write(vis)

    out.release()
    seq.release()

    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "x", "y", "w", "h"])
        writer.writerows(csv_rows)

    print("✅ Done!")


if __name__ == "__main__":
    main()