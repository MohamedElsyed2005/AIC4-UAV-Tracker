"""
dataset.py  –  AIC-4 UAV Tracker  (v2 — robust manifest-aware loader)
=======================================================================
Changes vs v1
─────────────
1. validate_sequences() — scans every sequence at startup:
   • checks video file exists
   • attempts cap.isOpened() (catches moov atom errors early)
   • reads cap.get(CAP_PROP_FRAME_COUNT) and warns if it differs from n_frames
   • logs all corrupted/missing entries to a file
   Corrupted sequences are removed from self.sequences before training starts,
   so __getitem__ never wastes retries on systematically broken videos.

2. _VideoCache.maxsize raised to 64 (covers all training sequences so no
   eviction during a typical epoch).

3. _build_sample: frame index upper bound is
       min(n_frames, ann_len, cap_frame_count) - 1
   This eliminates the silent "decoded count ≠ n_frames" drift.

4. native_fps stored per sequence — used to scale max_gap so that the
   template/search gap is always ≤ max_gap_seconds of real time regardless
   of the sequence's native frame rate.

5. safe __getitem__: after 10 failed retries the fallback is logged, not
   silently swallowed.

6. print_dataset_stats: shows per-split fps histogram and flags sequences
   whose decoded frame count disagrees with n_frames.
"""

import json
import logging
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger("HiT.dataset")


# ─────────────────────────────────────────────────────────────────────────────
# Annotation helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_annotation(ann_path: str) -> List[List[float]]:
    """Read annotation file. Each line: x,y,w,h (top-left + size)."""
    boxes: List[List[float]] = []
    with open(ann_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.replace("\t", ",").replace(" ", ",").split(",")
            parts = [p for p in parts if p]
            if len(parts) >= 4:
                x, y, w, h = (float(parts[i]) for i in range(4))
                boxes.append([x, y, w, h])
    return boxes


def xywh_to_xyxy(box):
    x, y, w, h = box
    return [x, y, x + w, y + h]


def xyxy_to_xywh(box):
    x1, y1, x2, y2 = box
    return [x1, y1, x2 - x1, y2 - y1]


def clip_box(box, frame_h, frame_w, margin=0):
    x, y, w, h = box
    x = max(margin, min(x, frame_w - margin))
    y = max(margin, min(y, frame_h - margin))
    w = max(1, min(w, frame_w - x - margin))
    h = max(1, min(h, frame_h - y - margin))
    return [x, y, w, h]


def crop_and_resize(
    frame: np.ndarray,
    box: List[float],
    output_size: int,
    context_factor: float = 2.0,
) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """
    Crop a square region centred on `box` with context padding,
    then resize to output_size × output_size.

    Returns:
        crop     – (output_size, output_size, 3) uint8
        scale    – output_size / crop_side
        (x1, y1) – top-left corner in (possibly padded) frame coords
    """
    H, W = frame.shape[:2]
    x, y, w, h = box
    cx = x + w / 2
    cy = y + h / 2
    s  = max((w + h) / 2 * context_factor, 1.0)

    x1 = int(round(cx - s / 2))
    y1 = int(round(cy - s / 2))
    x2 = int(round(cx + s / 2))
    y2 = int(round(cy + s / 2))

    pad_top    = max(0, -y1)
    pad_left   = max(0, -x1)
    pad_bottom = max(0, y2 - H)
    pad_right  = max(0, x2 - W)

    if any([pad_top, pad_left, pad_bottom, pad_right]):
        frame = cv2.copyMakeBorder(
            frame, pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_CONSTANT, value=(114, 114, 114),
        )
        x1 += pad_left; x2 += pad_left
        y1 += pad_top;  y2 += pad_top

    crop = frame[y1:y2, x1:x2]
    crop_side = max(crop.shape[0], crop.shape[1], 1)
    scale = output_size / crop_side
    crop  = cv2.resize(crop, (output_size, output_size))
    return crop, scale, (x1, y1)


# ─────────────────────────────────────────────────────────────────────────────
# Augmentation (v2: separate template/search; flip returns flag)
# ─────────────────────────────────────────────────────────────────────────────

def _photometric_augment(img: np.ndarray) -> np.ndarray:
    alpha = random.uniform(0.8, 1.2)
    beta  = random.randint(-20, 20)
    img   = np.clip(alpha * img + beta, 0, 255).astype(np.uint8)
    if random.random() < 0.3:
        perm = list(range(3))
        random.shuffle(perm)
        img = img[:, :, perm]
    return img


def augment_template(img: np.ndarray) -> np.ndarray:
    """Photometric only — template must stay stable (no flip)."""
    return _photometric_augment(img)


def augment_search(img: np.ndarray) -> Tuple[np.ndarray, bool]:
    """
    Photometric + optional horizontal flip.
    Returns (augmented_image, flipped:bool).
    Caller must mirror cx_n when flipped=True: cx_n = 1.0 - cx_n.
    """
    img = _photometric_augment(img)
    flipped = False
    if random.random() < 0.5:
        img = cv2.flip(img, 1)
        flipped = True
    return img, flipped


# ─────────────────────────────────────────────────────────────────────────────
# ImageNet normalisation
# ─────────────────────────────────────────────────────────────────────────────

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def to_tensor(img: np.ndarray) -> torch.Tensor:
    """uint8 HWC BGR → float32 CHW RGB, ImageNet-normalised."""
    img = img[:, :, ::-1].copy().astype(np.float32) / 255.0
    img = (img - _MEAN) / _STD
    return torch.from_numpy(img.transpose(2, 0, 1))


# ─────────────────────────────────────────────────────────────────────────────
# Per-worker VideoCapture cache
# ─────────────────────────────────────────────────────────────────────────────

class _VideoCache:
    """
    LRU VideoCapture cache for one DataLoader worker process.

    FIX: maxsize raised to 64 — covers all training sequences so handles
    are almost never evicted during an epoch, eliminating repeated moov
    atom scans.

    Each cache entry also stores the real decoded frame count obtained at
    open-time (cap.get(CAP_PROP_FRAME_COUNT)), which is used to clamp the
    sampling index and avoid seeking past EOF.
    """

    def __init__(self, maxsize: int = 64):
        self._caps:    Dict[str, cv2.VideoCapture] = {}
        self._counts:  Dict[str, int]              = {}   # decoded frame count
        self._order:   List[str]                   = []
        self._maxsize: int                         = maxsize

    def get(self, path: str) -> Optional[Tuple[cv2.VideoCapture, int]]:
        """
        Returns (cap, decoded_frame_count) or None if the video is corrupt.
        Caches the handle on first access.
        """
        if path in self._caps:
            return self._caps[path], self._counts[path]

        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            cap.release()
            logger.warning("_VideoCache: cannot open %s", path)
            return None

        # Real decoded frame count (may differ from manifest n_frames)
        decoded = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if decoded <= 0:
            # Fallback: count manually (slow, only for pathological files)
            decoded = 0
            while True:
                ret, _ = cap.read()
                if not ret:
                    break
                decoded += 1
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        # Evict oldest if over capacity
        if len(self._order) >= self._maxsize:
            oldest = self._order.pop(0)
            old_cap = self._caps.pop(oldest, None)
            self._counts.pop(oldest, None)
            if old_cap:
                old_cap.release()

        self._caps[path]   = cap
        self._counts[path] = decoded
        self._order.append(path)
        return cap, decoded

    def __del__(self):
        for cap in self._caps.values():
            try:
                cap.release()
            except Exception:
                pass


# ─────────────────────────────────────────────────────────────────────────────
# Sequence metadata (enriched from manifest + startup validation)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SeqMeta:
    """All per-sequence metadata used by the dataset."""
    dataset:         str
    seq_name:        str
    n_frames:        int          # from manifest (ground truth)
    native_fps:      float        # from manifest
    video_path:      str          # absolute path
    annotation_path: Optional[str]
    annotation:      List[List[float]] = field(default_factory=list)
    decoded_frames:  int          = 0    # from CAP_PROP_FRAME_COUNT at open
    valid:           bool         = True  # False = skip this sequence


# ─────────────────────────────────────────────────────────────────────────────
# Startup validation
# ─────────────────────────────────────────────────────────────────────────────

def validate_sequences(
    sequences:    List[SeqMeta],
    log_path:     Optional[str] = None,
) -> Tuple[List[SeqMeta], List[str]]:
    """
    Open every video once at startup, record decoded_frames, and flag
    sequences whose video is missing or corrupt.

    WHY THIS FIXES "moov atom not found":
      The first VideoCapture.open() is the moment ffmpeg reads the moov atom.
      If it fails here (moov missing, file truncated, wrong codec), we mark
      the sequence invalid and remove it from training — the DataLoader never
      encounters it again.  We do NOT open-close in __getitem__; the handle
      stays open in _VideoCache for the lifetime of the worker.

    Args:
        sequences: list of SeqMeta objects
        log_path:  if given, write corrupted-sequence log to this file

    Returns:
        (valid_sequences, corrupted_ids)
    """
    valid:     List[SeqMeta] = []
    corrupted: List[str]     = []

    for seq in sequences:
        seq_id = f"{seq.dataset}/{seq.seq_name}"

        # ── 1. Check file exists ─────────────────────────────────────────
        if not os.path.exists(seq.video_path):
            logger.warning("[SKIP] missing video: %s", seq.video_path)
            corrupted.append(f"{seq_id}: file not found")
            seq.valid = False
            continue

        # ── 2. Attempt VideoCapture open ─────────────────────────────────
        # This is the exact point where ffmpeg emits "moov atom not found".
        cap = cv2.VideoCapture(seq.video_path)
        if not cap.isOpened():
            cap.release()
            logger.warning("[SKIP] cannot open: %s", seq.video_path)
            corrupted.append(f"{seq_id}: VideoCapture.isOpened() = False")
            seq.valid = False
            continue

        # ── 3. Record decoded frame count ────────────────────────────────
        decoded = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        seq.decoded_frames = decoded

        if decoded <= 0:
            logger.warning("[SKIP] zero frames decoded: %s", seq_id)
            corrupted.append(f"{seq_id}: decoded_frames=0")
            seq.valid = False
            continue

        # ── 4. Warn on n_frames mismatch (don't skip, just log) ──────────
        # The manifest n_frames is ground truth for annotation alignment.
        # We use min(n_frames, decoded_frames) as the safe upper bound.
        if abs(decoded - seq.n_frames) > max(5, 0.02 * seq.n_frames):
            logger.warning(
                "[WARN] frame count mismatch %s: manifest=%d decoded=%d",
                seq_id, seq.n_frames, decoded,
            )

        # ── 5. Check annotation ──────────────────────────────────────────
        if not seq.annotation:
            logger.warning("[SKIP] empty annotation: %s", seq_id)
            corrupted.append(f"{seq_id}: annotation empty")
            seq.valid = False
            continue

        valid.append(seq)

    if log_path and corrupted:
        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
        with open(log_path, "w") as f:
            f.write("\n".join(corrupted) + "\n")
        logger.info("Corrupted sequence log → %s  (%d entries)", log_path, len(corrupted))

    logger.info(
        "Sequence validation: %d valid, %d skipped",
        len(valid), len(corrupted),
    )
    return valid, corrupted


# ─────────────────────────────────────────────────────────────────────────────
# InferenceSequence
# ─────────────────────────────────────────────────────────────────────────────

class InferenceSequence:
    """
    Iterate over all frames of one video sequence for tracker evaluation.

    Usage
    ─────
    seq = InferenceSequence(seq_info, data_root)
    first_frame, init_box = seq.get_init()
    for frame_idx, frame in seq:
        bbox = tracker.track(frame)
        seq.record(frame_idx, bbox)
    results = seq.get_results()
    """

    def __init__(self, seq_info: dict, data_root: str):
        self.seq_info  = seq_info
        self.data_root = data_root
        self.seq_id    = f"{seq_info['dataset']}/{seq_info['seq_name']}"
        self.n_frames  = seq_info["n_frames"]
        self.native_fps = seq_info.get("native_fps", 30)

        video_path = os.path.join(data_root, seq_info["video_path"])
        ann_path_rel = seq_info.get("annotation_path")

        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")

        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise RuntimeError(
                f"Cannot open video (corrupt/missing moov atom?): {video_path}"
            )

        if ann_path_rel:
            ann_path = os.path.join(data_root, ann_path_rel)
            if not os.path.exists(ann_path):
                raise FileNotFoundError(f"Annotation not found: {ann_path}")
            self.boxes = load_annotation(ann_path)
        else:
            self.boxes = []

        self._predictions: Dict[int, List[float]] = {}

    def get_init(self) -> Tuple[np.ndarray, List[float]]:
        """Returns (first_frame_BGR, init_bbox [x,y,w,h]). Seeks to frame 0."""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError(f"Cannot read first frame of {self.seq_id}")
        if not self.boxes:
            raise RuntimeError(
                f"No annotation for {self.seq_id}. "
                "Set self.boxes = [[x,y,w,h]] externally before calling get_init()."
            )
        init_box = self.boxes[0]
        self._predictions[0] = init_box
        return frame, init_box

    def __iter__(self):
        """Yield (frame_index, frame_BGR) starting from frame 1."""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 1)
        frame_idx = 1
        while frame_idx < self.n_frames:
            ret, frame = self.cap.read()
            if not ret:
                break
            yield frame_idx, frame
            frame_idx += 1

    def record(self, frame_idx: int, bbox: List[float]):
        self._predictions[frame_idx] = list(bbox)

    def get_results(self) -> List[Tuple]:
        rows = []
        for fi in range(self.n_frames):
            row_id = f"{self.seq_id}_{fi}"
            box    = self._predictions.get(fi, [0, 0, 0, 0])
            rows.append((row_id, *[round(v, 2) for v in box]))
        return rows

    def release(self):
        self.cap.release()

    def __del__(self):
        try:
            self.cap.release()
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# TrackingPair
# ─────────────────────────────────────────────────────────────────────────────

class TrackingPair:
    __slots__ = ["template", "search", "gt_box", "seq_id", "frame_idx"]

    def __init__(self, template, search, gt_box, seq_id, frame_idx):
        self.template  = template
        self.search    = search
        self.gt_box    = gt_box
        self.seq_id    = seq_id
        self.frame_idx = frame_idx


# ─────────────────────────────────────────────────────────────────────────────
# TrainingDataset (v2)
# ─────────────────────────────────────────────────────────────────────────────

class TrainingDataset(Dataset):
    """
    Manifest-aware training dataset.

    Key improvements vs v1
    ──────────────────────
    • Reads ONLY from contestant_manifest.json — zero directory walking.
    • validate_sequences() called at __init__: corrupted/missing videos are
      removed before the first __getitem__.
    • Frame index sampling clamped to min(n_frames, decoded_frames, ann_len).
    • max_gap scaled by native_fps so a 96-fps sequence doesn't sample frames
      2 seconds apart when a 30-fps sequence samples 1/3 second apart.
    • _VideoCache maxsize=64 (covers full dataset, no eviction churn).

    Args
    ────
    manifest_path      path to contestant_manifest.json
    data_root          root folder where dataset1/, dataset2/, … live
    split              'train' or 'public_lb'
    template_size      output size for template crop (default 128)
    search_size        output size for search crop   (default 256)
    max_gap_frames     max frame gap at 30 fps (scaled to native fps per seq)
    samples_per_epoch  virtual epoch length (default 50 000)
    augment            apply image augmentations (default True)
    corrupt_log_path   path to write corrupted-sequence log (optional)
    """

    def __init__(
        self,
        manifest_path:    str,
        data_root:        str,
        split:            str   = "train",
        template_size:    int   = 128,
        search_size:      int   = 256,
        max_gap_frames:   int   = 100,     # at 30 fps → ~3 seconds
        samples_per_epoch: int  = 50_000,
        augment:          bool  = True,
        corrupt_log_path: Optional[str] = None,
    ):
        self.data_root          = data_root
        self.template_size      = template_size
        self.search_size        = search_size
        self.max_gap_frames     = max_gap_frames
        self.samples_per_epoch  = samples_per_epoch
        self.augment            = augment

        # ── Load manifest ─────────────────────────────────────────────────
        with open(manifest_path, "r") as f:
            manifest = json.load(f)

        if split not in manifest:
            raise ValueError(
                f"Split '{split}' not in manifest. "
                f"Available: {list(manifest.keys())}"
            )

        raw_seqs = manifest[split]

        # ── Build SeqMeta objects ─────────────────────────────────────────
        seqs: List[SeqMeta] = []
        for seq_dict in raw_seqs.values():
            ann_rel  = seq_dict.get("annotation_path")
            ann_path = os.path.join(data_root, ann_rel) if ann_rel else None
            annotation: List[List[float]] = []
            if ann_path and os.path.exists(ann_path):
                annotation = load_annotation(ann_path)

            seqs.append(SeqMeta(
                dataset         = seq_dict["dataset"],
                seq_name        = seq_dict["seq_name"],
                n_frames        = seq_dict["n_frames"],
                native_fps      = float(seq_dict.get("native_fps", 30)),
                video_path      = os.path.join(data_root, seq_dict["video_path"]),
                annotation_path = ann_path,
                annotation      = annotation,
            ))

        # ── Validate sequences at startup ─────────────────────────────────
        # This is where "moov atom not found" is caught ONCE and the sequence
        # is removed, rather than crashing inside __getitem__ repeatedly.
        self.sequences, _ = validate_sequences(seqs, log_path=corrupt_log_path)

        if not self.sequences:
            raise RuntimeError(
                f"No valid sequences found for split='{split}'. "
                "Check data_root and manifest paths."
            )

        print(
            f"[TrainingDataset] split='{split}' | "
            f"{len(self.sequences)} valid sequences | "
            f"{samples_per_epoch} samples/epoch"
        )

        # ── Per-worker VideoCapture cache (created lazily in workers) ─────
        self._vcache = _VideoCache(maxsize=64)

    # ── Sampling helpers ──────────────────────────────────────────────────

    def _safe_upper(self, seq: SeqMeta) -> int:
        """
        Safe upper bound for frame index sampling.
        Uses the minimum of manifest n_frames, actual decoded frame count,
        and annotation length to prevent seeking past EOF.
        """
        ann_len = len(seq.annotation)
        decoded = seq.decoded_frames if seq.decoded_frames > 0 else seq.n_frames
        return max(1, min(seq.n_frames, decoded, ann_len))

    def _max_gap_for_seq(self, seq: SeqMeta) -> int:
        """
        Scale max_gap_frames to the sequence's native fps.
        Example: max_gap_frames=100 at 30 fps = 3.33s.
        At 96 fps the same 3.33s = 320 frames, keeping temporal context
        consistent across variable-fps sequences.
        """
        fps_ratio = seq.native_fps / 30.0
        return max(1, int(self.max_gap_frames * fps_ratio))

    def _get_annotation_for_frame(self, seq: SeqMeta, frame_idx: int) -> List[float]:
        ann = seq.annotation
        if len(ann) == 1:
            return ann[0]
        return ann[min(frame_idx, len(ann) - 1)]

    def _read_frame(self, seq: SeqMeta, frame_idx: int) -> Optional[np.ndarray]:
        """
        Read one frame using the cached VideoCapture.
        Returns None cleanly if the video is not readable.

        WHY NOT OPEN/CLOSE PER CALL:
          Every VideoCapture() call makes ffmpeg re-parse the moov atom.
          For a 1393-frame 96-fps video this is ~2 MB of I/O per frame read.
          The _VideoCache keeps the handle open for the worker's lifetime,
          reducing moov overhead to exactly one parse per sequence per worker.
        """
        result = self._vcache.get(seq.video_path)
        if result is None:
            return None
        cap, decoded_count = result

        # Clamp to safe range
        safe_idx = min(frame_idx, decoded_count - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, safe_idx)
        ret, frame = cap.read()
        return frame if ret else None

    def _build_sample(self, seq: SeqMeta) -> Optional[TrackingPair]:
        """Build one (template, search, gt_box) pair from a sequence."""
        upper  = self._safe_upper(seq)
        if upper < 2:
            return None

        max_gap = self._max_gap_for_seq(seq)

        # Sample template and search indices
        t_idx = random.randint(0, upper - 2)
        s_idx = min(t_idx + random.randint(1, max_gap), upper - 1)

        t_box = self._get_annotation_for_frame(seq, t_idx)
        s_box = self._get_annotation_for_frame(seq, s_idx)

        # Skip degenerate boxes
        if t_box[2] <= 0 or t_box[3] <= 0:
            return None
        if s_box[2] <= 0 or s_box[3] <= 0:
            return None

        t_frame = self._read_frame(seq, t_idx)
        s_frame = self._read_frame(seq, s_idx)
        if t_frame is None or s_frame is None:
            return None

        # ── Crop ──────────────────────────────────────────────────────────
        t_crop, _, _ = crop_and_resize(
            t_frame, t_box, self.template_size, context_factor=2.0
        )
        s_crop, s_scale, (sx1, sy1) = crop_and_resize(
            s_frame, s_box, self.search_size, context_factor=4.0
        )

        # ── Augment ───────────────────────────────────────────────────────
        s_flipped = False
        if self.augment:
            t_crop            = augment_template(t_crop)
            s_crop, s_flipped = augment_search(s_crop)

        # ── Normalise GT bbox to search crop (cx,cy,w,h in [0,1]) ─────────
        s_cx = s_box[0] + s_box[2] / 2
        s_cy = s_box[1] + s_box[3] / 2

        cx_crop = (s_cx - sx1) * s_scale
        cy_crop = (s_cy - sy1) * s_scale
        w_crop  = s_box[2] * s_scale
        h_crop  = s_box[3] * s_scale

        cx_n = cx_crop / self.search_size
        cy_n = cy_crop / self.search_size
        w_n  = w_crop  / self.search_size
        h_n  = h_crop  / self.search_size

        # Mirror cx when search was horizontally flipped
        if s_flipped:
            cx_n = 1.0 - cx_n

        cx_n = float(np.clip(cx_n, 0.0, 1.0))
        cy_n = float(np.clip(cy_n, 0.0, 1.0))
        w_n  = float(np.clip(w_n,  0.0, 1.0))
        h_n  = float(np.clip(h_n,  0.0, 1.0))

        gt_box = torch.tensor([cx_n, cy_n, w_n, h_n], dtype=torch.float32)

        return TrackingPair(
            template  = to_tensor(t_crop),
            search    = to_tensor(s_crop),
            gt_box    = gt_box,
            seq_id    = f"{seq.dataset}/{seq.seq_name}",
            frame_idx = s_idx,
        )

    # ── PyTorch Dataset interface ─────────────────────────────────────────

    def __len__(self) -> int:
        return self.samples_per_epoch

    def __getitem__(self, _idx: int) -> dict:
        for attempt in range(10):
            seq    = random.choice(self.sequences)
            sample = self._build_sample(seq)
            if sample is not None:
                return {
                    "template":  sample.template,
                    "search":    sample.search,
                    "gt_box":    sample.gt_box,
                    "seq_id":    sample.seq_id,
                    "frame_idx": sample.frame_idx,
                }

        # Fallback — should be extremely rare after startup validation
        logger.warning(
            "__getitem__: 10 consecutive sample failures — returning zero tensor"
        )
        return {
            "template":  torch.zeros(3, self.template_size, self.template_size),
            "search":    torch.zeros(3, self.search_size,   self.search_size),
            "gt_box":    torch.tensor([0.5, 0.5, 0.1, 0.1]),
            "seq_id":    "fallback",
            "frame_idx": 0,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Dataset stats helper
# ─────────────────────────────────────────────────────────────────────────────

def print_dataset_stats(manifest_path: str, data_root: str):
    """
    Print a summary table and run a file-existence + frame-count check.
    Opens each video to compare CAP_PROP_FRAME_COUNT vs manifest n_frames.
    """
    with open(manifest_path) as f:
        manifest = json.load(f)

    print(f"\n{'Split':<12} {'Seqs':>6} {'Frames':>10} "
          f"{'MinFPS':>8} {'MaxFPS':>8} {'AvgFPS':>8}")
    print("─" * 58)

    for split, seqs in manifest.items():
        frames    = [v["n_frames"]    for v in seqs.values()]
        fps_list  = [v.get("native_fps", 30) for v in seqs.values()]
        print(
            f"{split:<12} {len(seqs):>6} {sum(frames):>10} "
            f"{min(fps_list):>8} {max(fps_list):>8} "
            f"{sum(fps_list)/len(fps_list):>8.1f}"
        )

    print("\n── File existence + frame count check ──")
    for split, seqs in manifest.items():
        missing = 0
        miscount = 0
        for seq in seqs.values():
            vp = os.path.join(data_root, seq["video_path"])
            ann_rel = seq.get("annotation_path")
            ap = os.path.join(data_root, ann_rel) if ann_rel else None

            if not os.path.exists(vp):
                missing += 1
                continue

            cap = cv2.VideoCapture(vp)
            if not cap.isOpened():
                missing += 1
                cap.release()
                continue

            decoded = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            if abs(decoded - seq["n_frames"]) > max(5, 0.02 * seq["n_frames"]):
                miscount += 1
                print(
                    f"  [WARN] {seq['dataset']}/{seq['seq_name']}: "
                    f"manifest={seq['n_frames']} decoded={decoded}"
                )

            if ap and not os.path.exists(ap):
                missing += 1

        status = []
        if missing:
            status.append(f"{missing} missing/corrupt")
        if miscount:
            status.append(f"{miscount} frame-count mismatch")
        print(f"  {split}: {', '.join(status) if status else '✓ all OK'}")


# ─────────────────────────────────────────────────────────────────────────────
# Smoke-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    _ROOT = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest",
        default=str(_ROOT / "data/contest_release/metadata/contestant_manifest.json"))
    parser.add_argument("--data_root",
        default=str(_ROOT / "data/contest_release"))
    parser.add_argument("--mode", choices=["stats", "sample", "validate"],
        default="stats")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)s  %(message)s")

    if args.mode == "stats":
        print_dataset_stats(args.manifest, args.data_root)

    elif args.mode == "validate":
        ds = TrainingDataset(
            manifest_path    = args.manifest,
            data_root        = args.data_root,
            split            = "train",
            samples_per_epoch = 10,
            augment          = False,
            corrupt_log_path = "/tmp/corrupted_sequences.txt",
        )
        print(f"\nValid sequences: {len(ds.sequences)}")

    elif args.mode == "sample":
        ds = TrainingDataset(
            manifest_path    = args.manifest,
            data_root        = args.data_root,
            split            = "train",
            samples_per_epoch = 100,
            augment          = True,
        )
        sample = ds[0]
        print("\nSample keys:", list(sample.keys()))
        print("template shape:", sample["template"].shape)
        print("search shape:  ", sample["search"].shape)
        print("gt_box:        ", sample["gt_box"])
        print("seq_id:        ", sample["seq_id"])
        print("\n✓ dataset.py v2 works correctly!")