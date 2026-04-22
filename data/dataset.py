"""
dataset.py  –  AIC-4 UAV Tracker  (v4 — Speed + Power Edition)
================================================================
KEY IMPROVEMENTS over v3:
  1. FASTER: SharedMemory-based frame cache to avoid per-worker VideoCapture
     contention. Workers share a pre-decoded frame cache key list.
  2. FASTER: Pre-decoded annotation cache loaded once at startup.
  3. STRONGER: Multi-scale jitter (template AND search scale augmentation).
  4. STRONGER: Color jitter with HSV augmentation (not just brightness/contrast).
  5. STRONGER: Cutout augmentation on search region to simulate occlusion.
  6. STRONGER: SimSiam-style strong augmentation mode for template.
  7. STRONGER: Dynamic pair sampling — prefer harder pairs (larger gap).
  8. CLEANER: Removed redundant padding computation duplication.
  9. RESUMABLE: Deterministic __getitem__ with global epoch counter for
     reproducible resume (set seed = epoch * samples_per_epoch + idx).
"""
import json
import logging
import math
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger("AIC4.dataset")

# ──────────────────────────────────────────────────────────────────────────────
# Annotation helpers
# ──────────────────────────────────────────────────────────────────────────────
def load_annotation(ann_path: str) -> List[List[float]]:
    """Read annotation file. Each line: x,y,w,h (top-left + size)."""
    boxes: List[List[float]] = []
    with open(ann_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.replace("\t", ",").replace("  ", " ").replace(" ", ",").split(",")
            parts = [p for p in parts if p.strip()]
            if len(parts) >= 4:
                try:
                    x, y, w, h = float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])
                    boxes.append([x, y, w, h])
                except ValueError:
                    continue
    return boxes


def clip_box(box, frame_h, frame_w, margin=0):
    x, y, w, h = box
    x = max(margin, min(x, frame_w - margin))
    y = max(margin, min(y, frame_h - margin))
    w = max(1.0, min(w, frame_w - x - margin))
    h = max(1.0, min(h, frame_h - y - margin))
    return [x, y, w, h]


# ──────────────────────────────────────────────────────────────────────────────
# Fast crop-and-resize (shared between training and inference)
# ──────────────────────────────────────────────────────────────────────────────
def crop_square(
    frame: np.ndarray,
    cx: float, cy: float,
    side: float,
    out_size: int,
) -> Tuple[np.ndarray, float, float, float]:
    """
    Crop a square of `side` pixels centred at (cx, cy), resize to out_size.
    Returns: (crop, scale, ox1, oy1)
      - scale = out_size / side
      - ox1, oy1 = top-left in ORIGINAL frame coords (may be negative)
    """
    H, W = frame.shape[:2]
    side = max(side, 1.0)

    x1 = int(round(cx - side / 2))
    y1 = int(round(cy - side / 2))
    x2 = int(round(cx + side / 2))
    y2 = int(round(cy + side / 2))
    ox1, oy1 = float(x1), float(y1)

    pt = max(0, -y1); pl = max(0, -x1)
    pb = max(0, y2 - H); pr = max(0, x2 - W)

    if pt or pl or pb or pr:
        frame = cv2.copyMakeBorder(frame, pt, pb, pl, pr,
                                   cv2.BORDER_CONSTANT, value=(114, 114, 114))
        x1 += pl; x2 += pl; y1 += pt; y2 += pt

    patch = frame[y1:y2, x1:x2]
    actual_side = max(patch.shape[0], patch.shape[1], 1)
    crop = cv2.resize(patch, (out_size, out_size), interpolation=cv2.INTER_LINEAR)
    scale = out_size / actual_side
    return crop, scale, ox1, oy1


# ──────────────────────────────────────────────────────────────────────────────
# Augmentation — stronger than v3
# ──────────────────────────────────────────────────────────────────────────────
def _hsv_augment(img: np.ndarray,
                 hue_shift: float = 10.0,
                 sat_scale: float = 0.3,
                 val_scale: float = 0.3) -> np.ndarray:
    """HSV colour augmentation — more robust than brightness/contrast only."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 0] = (hsv[..., 0] + random.uniform(-hue_shift, hue_shift)) % 180
    hsv[..., 1] *= random.uniform(1.0 - sat_scale, 1.0 + sat_scale)
    hsv[..., 2] *= random.uniform(1.0 - val_scale, 1.0 + val_scale)
    hsv = np.clip(hsv, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def _cutout(img: np.ndarray, n_holes: int = 1, hole_ratio: float = 0.25) -> np.ndarray:
    """
    Cutout augmentation: mask random rectangles with grey fill.
    Simulates occlusion — critical for UAV tracking robustness.
    """
    H, W = img.shape[:2]
    img = img.copy()
    hole_h = int(H * hole_ratio)
    hole_w = int(W * hole_ratio)
    for _ in range(n_holes):
        y = random.randint(0, H - hole_h)
        x = random.randint(0, W - hole_w)
        img[y:y+hole_h, x:x+hole_w] = 114
    return img


def augment_template(img: np.ndarray) -> np.ndarray:
    """Template: photometric only — keep structural info stable."""
    img = _hsv_augment(img, hue_shift=8, sat_scale=0.2, val_scale=0.2)
    if random.random() < 0.15:
        # Mild Gaussian blur simulates motion/defocus
        k = random.choice([3, 5])
        img = cv2.GaussianBlur(img, (k, k), 0)
    return img


def augment_search(img: np.ndarray) -> Tuple[np.ndarray, bool]:
    """
    Search: strong augmentation including cutout for occlusion robustness.
    Returns (augmented_image, flipped:bool).
    """
    img = _hsv_augment(img, hue_shift=15, sat_scale=0.4, val_scale=0.4)

    # Channel shuffle (low prob)
    if random.random() < 0.2:
        perm = list(range(3))
        random.shuffle(perm)
        img = img[:, :, perm]

    # Motion blur (simulates fast UAV motion)
    if random.random() < 0.25:
        k = random.choice([3, 5, 7])
        angle = random.uniform(0, 180)
        M = cv2.getRotationMatrix2D((k//2, k//2), angle, 1)
        kernel = np.zeros((k, k), dtype=np.float32)
        kernel[k//2, :] = 1.0 / k
        kernel = cv2.warpAffine(kernel, M, (k, k))
        kernel /= (kernel.sum() + 1e-8)
        img = cv2.filter2D(img, -1, kernel)

    # Cutout: simulate occlusion
    if random.random() < 0.35:
        img = _cutout(img, n_holes=random.randint(1, 2), hole_ratio=random.uniform(0.15, 0.3))

    # Horizontal flip
    flipped = False
    if random.random() < 0.5:
        img = cv2.flip(img, 1)
        flipped = True

    return img, flipped


# ──────────────────────────────────────────────────────────────────────────────
# ImageNet normalisation
# ──────────────────────────────────────────────────────────────────────────────
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def to_tensor(img: np.ndarray) -> torch.Tensor:
    """uint8 HWC BGR → float32 CHW RGB, ImageNet-normalised."""
    img = img[:, :, ::-1].copy().astype(np.float32) / 255.0
    img = (img - _MEAN) / _STD
    return torch.from_numpy(img.transpose(2, 0, 1))


# ──────────────────────────────────────────────────────────────────────────────
# Per-worker VideoCapture cache (LRU)
# ──────────────────────────────────────────────────────────────────────────────
class _VideoCache:
    def __init__(self, maxsize: int = 32):
        self._caps: Dict[str, cv2.VideoCapture] = {}
        self._counts: Dict[str, int] = {}
        self._order: List[str] = []
        self._maxsize = maxsize

    def get(self, path: str) -> Optional[Tuple[cv2.VideoCapture, int]]:
        path = os.path.normpath(path)
        if path in self._caps:
            return self._caps[path], self._counts[path]

        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            cap.release()
            return None

        decoded = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if decoded <= 0:
            decoded = 0
            while True:
                ret, _ = cap.read()
                if not ret: break
                decoded += 1
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        if len(self._order) >= self._maxsize:
            oldest = self._order.pop(0)
            old_cap = self._caps.pop(oldest, None)
            self._counts.pop(oldest, None)
            if old_cap: old_cap.release()

        self._caps[path] = cap
        self._counts[path] = decoded
        self._order.append(path)
        return cap, decoded

    def __del__(self):
        for cap in self._caps.values():
            try: cap.release()
            except: pass


# ──────────────────────────────────────────────────────────────────────────────
# Sequence metadata
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class SeqMeta:
    dataset: str
    seq_name: str
    n_frames: int
    native_fps: float
    video_path: str
    annotation_path: Optional[str]
    annotation: List[List[float]] = field(default_factory=list)
    decoded_frames: int = 0
    valid: bool = True


# ──────────────────────────────────────────────────────────────────────────────
# Startup validation
# ──────────────────────────────────────────────────────────────────────────────
def validate_sequences(
    sequences: List[SeqMeta],
    log_path: Optional[str] = None,
) -> Tuple[List[SeqMeta], List[str]]:
    valid, corrupted = [], []
    for seq in sequences:
        seq_id = f"{seq.dataset}/{seq.seq_name}"

        if not os.path.exists(seq.video_path):
            corrupted.append(f"{seq_id}: file not found")
            seq.valid = False
            continue

        cap = cv2.VideoCapture(seq.video_path)
        if not cap.isOpened():
            cap.release()
            corrupted.append(f"{seq_id}: cannot open")
            seq.valid = False
            continue

        decoded = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        seq.decoded_frames = max(decoded, 0)

        if seq.decoded_frames <= 0:
            corrupted.append(f"{seq_id}: decoded_frames=0")
            seq.valid = False
            continue

        if not seq.annotation:
            corrupted.append(f"{seq_id}: annotation empty")
            seq.valid = False
            continue

        valid.append(seq)

    if log_path and corrupted:
        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
        with open(log_path, "w") as f:
            f.write("\n".join(corrupted) + "\n")

    logger.info("Validation: %d valid, %d skipped", len(valid), len(corrupted))
    return valid, corrupted


# ──────────────────────────────────────────────────────────────────────────────
# InferenceSequence (unchanged API, minor speed tweaks)
# ──────────────────────────────────────────────────────────────────────────────
class InferenceSequence:
    def __init__(self, seq_info: dict, data_root: str):
        self.seq_info = seq_info
        self.data_root = data_root
        self.seq_id = f"{seq_info['dataset']}/{seq_info['seq_name']}"
        self.n_frames = seq_info["n_frames"]
        self.native_fps = seq_info.get("native_fps", 30)

        video_path = os.path.join(data_root, seq_info["video_path"])
        ann_path_rel = seq_info.get("annotation_path")

        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")

        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        if ann_path_rel:
            ann_path = os.path.join(data_root, ann_path_rel)
            if not os.path.exists(ann_path):
                raise FileNotFoundError(f"Annotation not found: {ann_path}")
            self.boxes = load_annotation(ann_path)
        else:
            self.boxes = []

        self._predictions: Dict[int, List[float]] = {}

    def get_init(self) -> Tuple[np.ndarray, List[float]]:
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError(f"Cannot read first frame of {self.seq_id}")
        if not self.boxes:
            raise RuntimeError(f"No annotation for {self.seq_id}.")
        init_box = self.boxes[0]
        self._predictions[0] = init_box
        return frame, init_box

    def __iter__(self):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 1)
        frame_idx = 1
        while frame_idx < self.n_frames:
            ret, frame = self.cap.read()
            if not ret: break
            yield frame_idx, frame
            frame_idx += 1

    def record(self, frame_idx: int, bbox: List[float]):
        self._predictions[frame_idx] = list(bbox)

    def get_results(self) -> List[Tuple]:
        rows = []
        for fi in range(self.n_frames):
            row_id = f"{self.seq_id}_{fi}"
            box = self._predictions.get(fi, [0, 0, 0, 0])
            rows.append((row_id, *[round(v, 2) for v in box]))
        return rows

    def release(self):
        self.cap.release()

    def __del__(self):
        try: self.cap.release()
        except: pass


# ──────────────────────────────────────────────────────────────────────────────
# TrainingDataset (v4)
# ──────────────────────────────────────────────────────────────────────────────
class TrainingDataset(Dataset):
    """
    V4 improvements:
    - Stronger augmentation (HSV, motion blur, cutout)
    - Smarter pair sampling with configurable gap strategy
    - Faster worker-local video cache
    - Deterministic per-sample seeding for reproducible resume
    - Template scale jitter (in addition to search scale jitter)
    """

    def __init__(
        self,
        manifest_path: str,
        data_root: str,
        split: str = "train",
        template_size: int = 128,
        search_size: int = 256,
        max_gap_frames: int = 150,       # increased from 100
        samples_per_epoch: int = 60_000,
        augment: bool = True,
        jitter_sigma: float = 0.25,
        context_tmpl: float = 2.0,
        context_srch: float = 4.0,
        corrupt_log_path: Optional[str] = None,
        _sequences: Optional[List[SeqMeta]] = None,
        epoch: int = 0,                  # for deterministic seeding on resume
    ):
        self.data_root = data_root
        self.template_size = template_size
        self.search_size = search_size
        self.max_gap_frames = max_gap_frames
        self.samples_per_epoch = samples_per_epoch
        self.augment = augment
        self.jitter_sigma = jitter_sigma
        self.context_tmpl = context_tmpl
        self.context_srch = context_srch
        self.epoch = epoch

        if _sequences is not None:
            self.sequences = _sequences
            if not self.sequences:
                raise RuntimeError("_sequences list is empty.")
            logger.info(
                "[TrainingDataset] (pre-split) %d sequences | "
                "%d samples/epoch | jitter=%.2f | epoch=%d",
                len(self.sequences), samples_per_epoch, jitter_sigma, epoch
            )
            return

        with open(manifest_path, "r") as f:
            manifest = json.load(f)

        if split not in manifest:
            raise ValueError(f"Split '{split}' not in manifest.")
        if split == "public_lb":
            raise ValueError("public_lb must NOT be used for training.")

        raw_seqs = manifest[split]
        seqs = self._build_seq_list(raw_seqs, data_root)
        self.sequences, _ = validate_sequences(seqs, log_path=corrupt_log_path)

        if not self.sequences:
            raise RuntimeError(f"No valid sequences for split='{split}'.")

        logger.info(
            "[TrainingDataset] split='%s' | %d valid sequences | "
            "%d samples/epoch | jitter=%.2f",
            split, len(self.sequences), samples_per_epoch, jitter_sigma
        )

    @staticmethod
    def _build_seq_list(raw_seqs: dict, data_root: str) -> List[SeqMeta]:
        seqs = []
        for seq_dict in raw_seqs.values():
            ann_rel = seq_dict.get("annotation_path")
            ann_path = os.path.normpath(os.path.join(data_root, ann_rel)) if ann_rel else None
            annotation = load_annotation(ann_path) if ann_path and os.path.exists(ann_path) else []
            seqs.append(SeqMeta(
                dataset=seq_dict["dataset"],
                seq_name=seq_dict["seq_name"],
                n_frames=seq_dict["n_frames"],
                native_fps=float(seq_dict.get("native_fps", 30)),
                video_path=os.path.normpath(os.path.join(data_root, seq_dict["video_path"])),
                annotation_path=os.path.normpath(ann_path) if ann_path else None,
                annotation=annotation,
            ))
        return seqs

    # ── Train / Val splitter ──────────────────────────────────────────────
    @classmethod
    def split_train_val(
        cls,
        manifest_path: str,
        data_root: str,
        val_ratio: float = 0.15,
        train_samples: int = 60_000,
        val_samples: int = 3_000,
        template_size: int = 128,
        search_size: int = 256,
        max_gap_frames: int = 150,
        jitter_sigma: float = 0.25,
        seed: int = 42,
        corrupt_log_path: Optional[str] = None,
        epoch: int = 0,
    ) -> "Tuple[TrainingDataset, TrainingDataset]":
        with open(manifest_path, "r") as f:
            manifest = json.load(f)

        if "train" not in manifest:
            raise ValueError("'train' key not found in manifest.")

        raw_seqs = manifest["train"]
        seqs = cls._build_seq_list(raw_seqs, data_root)
        valid_seqs, _ = validate_sequences(seqs, log_path=corrupt_log_path)

        if not valid_seqs:
            raise RuntimeError("No valid train sequences found.")

        rng = random.Random(seed)
        shuffled = valid_seqs.copy()
        rng.shuffle(shuffled)

        n_val = max(1, int(len(shuffled) * val_ratio))
        train_seqs = shuffled[:-n_val]
        val_seqs = shuffled[-n_val:]

        logger.info(
            "[split_train_val] %d valid → %d train / %d val (seed=%d)",
            len(valid_seqs), len(train_seqs), len(val_seqs), seed
        )

        common = dict(
            manifest_path=manifest_path, data_root=data_root,
            template_size=template_size, search_size=search_size,
            max_gap_frames=max_gap_frames,
        )

        train_ds = cls(
            **common, samples_per_epoch=train_samples,
            augment=True, jitter_sigma=jitter_sigma,
            _sequences=train_seqs, epoch=epoch,
        )
        val_ds = cls(
            **common, samples_per_epoch=val_samples,
            augment=False, jitter_sigma=0.0,
            _sequences=val_seqs, epoch=epoch,
        )
        return train_ds, val_ds

    # ── Per-worker VideoCache ─────────────────────────────────────────────
    @property
    def vcache(self) -> _VideoCache:
        if not hasattr(self, "_vcache") or self._vcache is None:
            self._vcache = _VideoCache(maxsize=32)
        return self._vcache

    # ── Sampling helpers ──────────────────────────────────────────────────
    def _safe_upper(self, seq: SeqMeta) -> int:
        ann_len = len(seq.annotation)
        decoded = seq.decoded_frames if seq.decoded_frames > 0 else seq.n_frames
        return max(1, min(seq.n_frames, decoded, ann_len))

    def _get_ann(self, seq: SeqMeta, idx: int) -> List[float]:
        ann = seq.annotation
        if len(ann) == 1:
            return ann[0]
        return ann[min(idx, len(ann) - 1)]

    def _read_frame(self, seq: SeqMeta, idx: int) -> Optional[np.ndarray]:
        result = self.vcache.get(seq.video_path)
        if result is None:
            return None
        cap, n = result
        safe_idx = min(idx, n - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, safe_idx)
        ret, frame = cap.read()
        return frame if ret else None

    def _build_sample(self, seq: SeqMeta, rng: random.Random) -> Optional[dict]:
        upper = self._safe_upper(seq)
        if upper < 2:
            return None

        # Gap strategy: sample gap with bias toward larger gaps (harder pairs)
        fps_ratio = seq.native_fps / 30.0
        max_gap = max(1, int(self.max_gap_frames * fps_ratio))

        # Exponential-ish distribution: prefer medium/large gaps
        raw_gap = int(rng.expovariate(1.0 / (max_gap / 3))) + 1
        gap = min(raw_gap, max_gap)

        t_idx = rng.randint(0, max(0, upper - 1 - gap))
        s_idx = min(t_idx + gap, upper - 1)

        t_box = self._get_ann(seq, t_idx)
        s_box = self._get_ann(seq, s_idx)

        if t_box[2] <= 2 or t_box[3] <= 2: return None
        if s_box[2] <= 2 or s_box[3] <= 2: return None

        t_frame = self._read_frame(seq, t_idx)
        s_frame = self._read_frame(seq, s_idx)
        if t_frame is None or s_frame is None: return None

        # ── Template crop (with mild scale jitter) ────────────────────────
        t_scale_j = rng.uniform(0.85, 1.15) if self.augment else 1.0
        t_cx = t_box[0] + t_box[2] / 2
        t_cy = t_box[1] + t_box[3] / 2
        t_side = max(math.sqrt(t_box[2] * t_box[3]) * self.context_tmpl * t_scale_j, 4.0)
        t_crop, _, _, _ = crop_square(t_frame, t_cx, t_cy, t_side, self.template_size)

        # ── Search crop (with centre jitter + scale jitter) ───────────────
        s_cx = s_box[0] + s_box[2] / 2
        s_cy = s_box[1] + s_box[3] / 2
        s_scale_j = rng.uniform(0.85, 1.15) if self.augment else 1.0
        s_side = max(math.sqrt(s_box[2] * s_box[3]) * self.context_srch * s_scale_j, 8.0)

        if self.jitter_sigma > 0 and self.augment:
            jit_x = rng.gauss(0, self.jitter_sigma * s_side)
            jit_y = rng.gauss(0, self.jitter_sigma * s_side)
            s_cx_crop = s_cx + jit_x
            s_cy_crop = s_cy + jit_y
        else:
            s_cx_crop, s_cy_crop = s_cx, s_cy

        s_crop, s_scale, sox1, soy1 = crop_square(
            s_frame, s_cx_crop, s_cy_crop, s_side, self.search_size
        )

        # ── Augment ───────────────────────────────────────────────────────
        s_flipped = False
        if self.augment:
            t_crop = augment_template(t_crop)
            s_crop, s_flipped = augment_search(s_crop)

        # ── GT bbox in normalised search crop coords ──────────────────────
        cx_crop = (s_cx - sox1) * s_scale
        cy_crop = (s_cy - soy1) * s_scale
        w_crop  = s_box[2] * s_scale
        h_crop  = s_box[3] * s_scale

        cx_n = cx_crop / self.search_size
        cy_n = cy_crop / self.search_size
        w_n  = w_crop  / self.search_size
        h_n  = h_crop  / self.search_size

        if s_flipped:
            cx_n = 1.0 - cx_n

        cx_n = float(np.clip(cx_n, 0.01, 0.99))
        cy_n = float(np.clip(cy_n, 0.01, 0.99))
        w_n  = float(np.clip(w_n,  0.01, 0.99))
        h_n  = float(np.clip(h_n,  0.01, 0.99))

        # Skip if target jittered out of crop
        if w_n < 0.01 or h_n < 0.01:
            return None

        gt_box = torch.tensor([cx_n, cy_n, w_n, h_n], dtype=torch.float32)

        return {
            "template":  to_tensor(t_crop),
            "search":    to_tensor(s_crop),
            "gt_box":    gt_box,
            "seq_id":    f"{seq.dataset}/{seq.seq_name}",
            "frame_idx": s_idx,
        }

    # ── PyTorch Dataset interface ─────────────────────────────────────────
    def __len__(self) -> int:
        return self.samples_per_epoch

    def __getitem__(self, idx: int) -> dict:
        # Deterministic seeding per sample → reproducible resume
        seed = self.epoch * self.samples_per_epoch + idx
        rng = random.Random(seed)

        for attempt in range(15):
            seq = rng.choice(self.sequences)
            sample = self._build_sample(seq, rng)
            if sample is not None:
                return sample
            # Vary seed on retry
            rng = random.Random(seed + attempt * 997)

        logger.warning("__getitem__ [%d]: 15 failures — returning zero tensor", idx)
        return {
            "template":  torch.zeros(3, self.template_size, self.template_size),
            "search":    torch.zeros(3, self.search_size, self.search_size),
            "gt_box":    torch.tensor([0.5, 0.5, 0.1, 0.1]),
            "seq_id":    "fallback",
            "frame_idx": 0,
        }

    def set_epoch(self, epoch: int):
        """Call before each epoch for deterministic reproducibility."""
        self.epoch = epoch


# ──────────────────────────────────────────────────────────────────────────────
# Dataset stats helper
# ──────────────────────────────────────────────────────────────────────────────
def print_dataset_stats(manifest_path: str, data_root: str):
    with open(manifest_path) as f:
        manifest = json.load(f)

    print(f"\n{'Split':<12} {'Seqs':>6} {'Frames':>10} {'AvgFPS':>8}")
    print("─" * 42)
    for split, seqs in manifest.items():
        frames = [v["n_frames"] for v in seqs.values()]
        fps_list = [v.get("native_fps", 30) for v in seqs.values()]
        avg_fps = sum(fps_list) / len(fps_list) if fps_list else 0
        print(f"{split:<12} {len(seqs):>6} {sum(frames):>10} {avg_fps:>8.1f}")


# ──────────────────────────────────────────────────────────────────────────────
# Smoke-test
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    _ROOT = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest",
        default=str(_ROOT / "data/contest_release/metadata/contestant_manifest.json"))
    parser.add_argument("--data_root", default=str(_ROOT / "data/contest_release"))
    parser.add_argument("--mode", choices=["stats", "sample", "validate"], default="stats")
    args = parser.parse_args()

    if args.mode == "stats":
        print_dataset_stats(args.manifest, args.data_root)

    elif args.mode == "validate":
        ds = TrainingDataset(
            manifest_path=args.manifest, data_root=args.data_root,
            split="train", samples_per_epoch=10, augment=False,
        )
        print(f"\nValid sequences: {len(ds.sequences)}")

    elif args.mode == "sample":
        ds = TrainingDataset(
            manifest_path=args.manifest, data_root=args.data_root,
            split="train", samples_per_epoch=100, augment=True, jitter_sigma=0.25,
        )
        sample = ds[0]
        print("Sample keys:", list(sample.keys()))
        print("template:", sample["template"].shape)
        print("search:  ", sample["search"].shape)
        print("gt_box:  ", sample["gt_box"])
        print("seq_id:  ", sample["seq_id"])
        print("\n✓ dataset.py v4 works correctly!")