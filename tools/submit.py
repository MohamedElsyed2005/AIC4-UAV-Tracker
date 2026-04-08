"""
submit.py  –  AIC-4 Submission Generator
======================================================
Runs the trained HiT Tracker on every public_lb sequence and
writes a submission.csv ready for Kaggle upload.

Usage (from project root):
    python tools/generate_submission.py \
        --checkpoint output/hit_run1/best.pth \
        --manifest   data/contest_release/metadata/contestant_manifest.json \
        --data_root  data/contest_release \
        --output     submission.csv \
        --device     cuda

The output CSV has the exact format required by sample_submission.csv:
    id,x,y,w,h
    dataset2/basketball_player1_0,x,y,w,h
    ...
"""

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from data.dataset        import InferenceSequence
from models.hit.model    import HiTTracker
from tracking.tracker    import HiTTrackerInference

import json


def parse_args():
    p = argparse.ArgumentParser(description="Generate AIC-4 submission CSV")
    p.add_argument("--checkpoint", required=True,
                   help="Path to best.pth checkpoint")
    p.add_argument("--manifest",
                   default=str(_ROOT / "data/contest_release/metadata/contestant_manifest.json"))
    p.add_argument("--data_root",
                   default=str(_ROOT / "data/contest_release"))
    p.add_argument("--output",  default="submission.csv",
                   help="Output CSV path")
    p.add_argument("--device",  default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--split",   default="public_lb",
                   help="Which manifest split to run on")
    p.add_argument("--score_threshold", type=float, default=0.0,
                   help="Min score to update state (0 = always update)")
    return p.parse_args()


def main():
    args = parse_args()

    print(f"[submit] Loading checkpoint: {args.checkpoint}")
    model = HiTTracker.load(args.checkpoint, device=args.device)
    model.eval()

    tracker = HiTTrackerInference(
        model,
        device          = args.device,
        score_threshold = args.score_threshold,
    )

    with open(args.manifest) as f:
        manifest = json.load(f)

    sequences = manifest[args.split]
    n_seqs    = len(sequences)
    print(f"[submit] Running on {n_seqs} sequences (split='{args.split}')")

    all_rows  = []
    t_total   = time.time()
    n_failed  = 0

    for seq_idx, (seq_key, seq_info) in enumerate(sequences.items()):
        seq_id = f"{seq_info['dataset']}/{seq_info['seq_name']}"
        t_seq  = time.time()

        try:
            seq = InferenceSequence(seq_info, args.data_root)
        except (FileNotFoundError, RuntimeError) as e:
            print(f"  [SKIP] {seq_id}: {e}")
            n_failed += 1
            # Write zeros for every frame so CSV stays complete
            for fi in range(seq_info["n_frames"]):
                all_rows.append((f"{seq_id}_{fi}", 0, 0, 0, 0))
            continue

        try:
            init_frame, init_box = seq.get_init()
            tracker.reset()
            tracker.initialize(init_frame, init_box)

            for frame_idx, frame in seq:
                pred_box = tracker.track(frame)
                seq.record(frame_idx, pred_box)

        except Exception as e:
            print(f"  [ERROR] {seq_id} frame tracking failed: {e}")
            n_failed += 1

        rows      = seq.get_results()
        all_rows += rows
        seq.release()

        elapsed = time.time() - t_seq
        fps     = seq_info["n_frames"] / max(elapsed, 1e-6)
        print(f"  [{seq_idx+1:3d}/{n_seqs}] {seq_id:<40} "
              f"{seq_info['n_frames']:5d} frames  {fps:5.1f} fps  {elapsed:.1f}s")

    # ── Write CSV ──────────────────────────────────────────────────────────
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

    if n_failed > 0:
        print(f"[submit] WARNING: {n_failed} sequences failed — "
              "those rows contain zeros.")


if __name__ == "__main__":
    main()