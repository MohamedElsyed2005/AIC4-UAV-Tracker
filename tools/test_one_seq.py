"""
submit.py  –  AIC-4 Single Sequence Runner + Video Output
==========================================================

NEW:
  - --seq_id to run only one sequence
  - saves prediction video with bounding boxes

Usage:
python submit.py \
  --checkpoint ../output/hit_run3/best.pth \
  --manifest ../data/contest_release/metadata/contestant_manifest.json \
  --data_root ../data/contest_release \
  --seq_id dataset1/Car_video/Car_video \ 
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
import cv2

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from data.dataset import InferenceSequence
from models.hit.model import HiTTracker
from tracking.tracker import HiTTrackerInference


def parse_args():
    p = argparse.ArgumentParser()

    # ── ROOT PATHS (DEFAULTS ADDED) ─────────────────────────────
    p.add_argument("--checkpoint", 
                   default="output/hit_run3/best.pth")

    p.add_argument("--manifest", 
                   default="data/contest_release/metadata/contestant_manifest.json")

    p.add_argument("--data_root", 
                   default="data/contest_release")

    # ── OUTPUTS ────────────────────────────────────────────────
    p.add_argument("--output", 
                   default="submission.csv")

    p.add_argument("--video_output", 
                   default="result.mp4")

    # ── RUN SETTINGS ────────────────────────────────────────────
    p.add_argument("--device", 
                   default="cuda" if torch.cuda.is_available() else "cpu")

    p.add_argument("--split", 
                   default="public_lb")

    p.add_argument("--seq_id", 
                   required=True)

    p.add_argument("--score_threshold", type=float, default=0.15)
    p.add_argument("--smooth_factor", type=float, default=0.6)

    return p.parse_args()


def draw_box(frame, box, color=(0, 255, 0)):
    x, y, w, h = [int(v) for v in box]
    return cv2.rectangle(frame.copy(), (x, y), (x + w, y + h), color, 2)


def main():
    args = parse_args()

    print(f"[load] checkpoint: {args.checkpoint}")
    model = HiTTracker.load(args.checkpoint, device=args.device)
    model.eval()

    tracker = HiTTrackerInference(
        model,
        score_threshold=args.score_threshold,
        smooth_factor=args.smooth_factor,
        device=args.device,
    )

    with open(args.manifest) as f:
        manifest = json.load(f)

    sequences = manifest[args.split]

    # ─────────────────────────────────────────────
    # find single sequence
    # ─────────────────────────────────────────────
    seq_info = None
    seq_key = None

    for k, v in sequences.items():
        sid = f"{v['dataset']}/{v['seq_name']}"
        if sid == args.seq_id:
            seq_info = v
            seq_key = k
            break

    if seq_info is None:
        raise ValueError(f"Sequence not found: {args.seq_id}")

    seq_id = f"{seq_info['dataset']}/{seq_info['seq_name']}"
    print(f"[run] sequence: {seq_id}")

    seq = InferenceSequence(seq_info, args.data_root)

    init_frame, init_box = seq.get_init()

    tracker.reset()
    tracker.initialize(init_frame, init_box)

    # ─────────────────────────────────────────────
    # VIDEO SETUP
    # ─────────────────────────────────────────────
    first_frame = init_frame
    h, w = first_frame.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(args.video_output, fourcc, 30, (w, h))

    all_rows = []

    # frame 0 GT
    video_writer.write(draw_box(init_frame, init_box, (0, 0, 255)))

    all_rows.append((f"{seq_id}_0",
                     int(init_box[0]), int(init_box[1]),
                     int(init_box[2]), int(init_box[3])))

    t0 = time.time()

    for frame_idx, frame in seq:
        pred_box = tracker.track(frame)
        pred_box = [int(round(v)) for v in pred_box]

        seq.record(frame_idx, pred_box)

        # save csv row
        all_rows.append((f"{seq_id}_{frame_idx}",
                         pred_box[0], pred_box[1],
                         pred_box[2], pred_box[3]))

        # draw + save video
        vis = draw_box(frame, pred_box, (0, 255, 0))
        video_writer.write(vis)

    video_writer.release()
    seq.release()

    # ─────────────────────────────────────────────
    # SAVE CSV
    # ─────────────────────────────────────────────
    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "x", "y", "w", "h"])
        writer.writerows(all_rows)

    elapsed = time.time() - t0

    print("\n[done]")
    print(f"Sequence: {seq_id}")
    print(f"Frames: {len(all_rows)}")
    print(f"Time: {elapsed:.2f}s")
    print(f"FPS: {len(all_rows)/elapsed:.2f}")
    print(f"CSV saved: {args.output}")
    print(f"Video saved: {args.video_output}")


if __name__ == "__main__":
    main()