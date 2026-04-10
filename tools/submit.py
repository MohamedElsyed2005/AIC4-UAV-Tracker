"""
submit.py  –  AIC-4 Submission Generator  (v4 — fixed imports)
===============================================================

CRITICAL FIX vs v3:
  Import paths corrected from package-style to flat project layout:
    from data.dataset         →  from dataset
    from models.hit.model     →  from model
    from tracking.tracker     →  from tracker

All v3 fixes preserved:
  - Frame 0 returns GT init_box
  - Integer output format
  - Score threshold tuning
  - ETA reporting

Usage:
    python submit.py \
        --checkpoint output/hit_run3/best.pth \
        --manifest   data/contest_release/metadata/contestant_manifest.json \
        --data_root  data/contest_release \
        --output     submission.csv \
        --device     cuda
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))

# ── FIXED IMPORTS ─────────────────────────────────────────────────────────────
from data.dataset import InferenceSequence     # was: from data.dataset
from models.hit.model   import HiTTracker            # was: from models.hit.model
from tracking.tracker import HiTTrackerInference   # was: from tracking.tracker


def parse_args():
    p = argparse.ArgumentParser(description="Generate AIC-4 submission CSV")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--manifest",
                   default=str(_ROOT / "data/contest_release/metadata/contestant_manifest.json"))
    p.add_argument("--data_root",
                   default=str(_ROOT / "data/contest_release"))
    p.add_argument("--output",  default="submission.csv")
    p.add_argument("--device",  default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--split",   default="public_lb")
    p.add_argument("--score_threshold", type=float, default=0.15)
    p.add_argument("--smooth_factor",   type=float, default=0.6)
    return p.parse_args()


def main():
    args = parse_args()

    print(f"[submit] Loading checkpoint: {args.checkpoint}")
    model = HiTTracker.load(args.checkpoint, device=args.device)
    model.eval()

    tracker = HiTTrackerInference(
        model,
        score_threshold = args.score_threshold,
        smooth_factor   = args.smooth_factor,
        device          = args.device,
    )

    with open(args.manifest) as f:
        manifest = json.load(f)

    sequences = manifest[args.split]
    n_seqs    = len(sequences)
    print(f"[submit] {n_seqs} sequences (split='{args.split}')")
    print(f"[submit] score_threshold={args.score_threshold}  smooth_factor={args.smooth_factor}")

    all_rows  = []
    t_total   = time.time()
    n_failed  = 0

    for seq_idx, (seq_key, seq_info) in enumerate(sequences.items()):
        seq_id   = f"{seq_info['dataset']}/{seq_info['seq_name']}"
        t_seq    = time.time()
        n_frames = seq_info["n_frames"]

        try:
            seq = InferenceSequence(seq_info, args.data_root)
        except (FileNotFoundError, RuntimeError) as e:
            print(f"  [SKIP] {seq_id}: {e}")
            n_failed += 1
            for fi in range(n_frames):
                all_rows.append((f"{seq_id}_{fi}", 0, 0, 0, 0))
            continue

        try:
            init_frame, init_box = seq.get_init()
            # Frame 0 always gets GT init_box (already recorded by get_init)

            tracker.reset()
            tracker.initialize(init_frame, init_box)

            for frame_idx, frame in seq:
                pred_box     = tracker.track(frame)
                pred_box_int = [int(round(v)) for v in pred_box]
                seq.record(frame_idx, pred_box_int)

        except Exception as e:
            print(f"  [ERROR] {seq_id}: {e}")
            n_failed += 1

        rows = seq.get_results()
        int_rows = []
        for row in rows:
            row_id = row[0]
            coords = [int(round(float(v))) for v in row[1:]]
            int_rows.append((row_id, *coords))
        all_rows += int_rows
        seq.release()

        elapsed = time.time() - t_seq
        fps     = n_frames / max(elapsed, 1e-6)
        done_frac     = (seq_idx + 1) / n_seqs
        elapsed_total = time.time() - t_total
        eta           = elapsed_total / done_frac * (1 - done_frac) if done_frac > 0 else 0

        print(f"  [{seq_idx+1:3d}/{n_seqs}] {seq_id:<45} "
              f"{n_frames:5d} frames  {fps:5.1f} fps  ETA {eta:.0f}s")

    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "x", "y", "w", "h"])
        for row in all_rows:
            writer.writerow(row)

    total_elapsed = time.time() - t_total
    print(f"\n[submit] Done in {total_elapsed:.1f}s")
    print(f"[submit] Sequences: {n_seqs - n_failed}/{n_seqs} succeeded")
    print(f"[submit] Rows written: {len(all_rows)}")
    print(f"[submit] Output: {args.output}")

    expected_rows = sum(info["n_frames"] for info in sequences.values())
    if len(all_rows) != expected_rows:
        print(f"[submit] WARNING: Expected {expected_rows} rows, wrote {len(all_rows)}")
    else:
        print(f"[submit] Row count OK: {len(all_rows)}")

    if n_failed > 0:
        print(f"[submit] WARNING: {n_failed} sequences failed.")


if __name__ == "__main__":
    main()