"""
dataset.py  –  AIC-4 UAV Tracker  (v3 — jitter fix + padding fix)
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
            parts = line.replace("\t", ",").replace("  ", ",").split(",")
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
) -> Tuple[np.ndarray, float, Tuple[int, int], Tuple[int, int]]:
    """
    Crop a square region centred on `box` with context padding,
    then resize to output_size × output_size.
    FIX v3: now returns BOTH the pre-padding AND post-padding top-left
    so callers can use the correct origin for GT coordinate mapping.

    Returns:
        crop          – (output_size, output_size, 3) uint8
        scale         – output_size / crop_side
        (x1, y1)      – top-left in PADDED frame coords  (for indexing)
        (ox1, oy1)    – top-left in ORIGINAL frame coords (for GT mapping)
    """
    import math

    H, W = frame.shape[:2]
    x, y, w, h = box
    cx = x + w / 2
    cy = y + h / 2

    # FIX: use geometric mean to match tracker.py crop size formula
    s = max(math.sqrt(w * h) * context_factor, 1.0)

    x1 = int(round(cx - s / 2))
    y1 = int(round(cy - s / 2))
    x2 = int(round(cx + s / 2))
    y2 = int(round(cy + s / 2))

    # Remember original (pre-padding) origin for GT mapping
    ox1, oy1 = x1, y1

    pad_top = max(0, -y1)
    pad_left = max(0, -x1)
    pad_bottom = max(0, y2 - H)
    pad_right = max(0, x2 - W)

    if any([pad_top, pad_left, pad_bottom, pad_right]):
        frame = cv2.copyMakeBorder(
            frame, pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_CONSTANT, value=(114, 114, 114),
        )
        x1 += pad_left; x2 += pad_left
        y1 += pad_top; y2 += pad_top

    crop = frame[y1:y2, x1:x2]
    crop_side = max(crop.shape[0], crop.shape[1], 1)
    scale = output_size / crop_side
    crop = cv2.resize(crop, (output_size, output_size))
    return crop, scale, (x1, y1), (ox1, oy1)


# ──────────────────────────────────────────────────────────────────────────────
# Augmentation
# ──────────────────────────────────────────────────────────────────────────────
def _photometric_augment(img: np.ndarray) -> np.ndarray:
    alpha = random.uniform(0.8, 1.2)
    beta = random.randint(-20, 20)
    img = np.clip(alpha * img + beta, 0, 255).astype(np.uint8)
    if random.random() < 0.3:
        perm = list(range(3))
        random.shuffle(perm)
        img = img[:, :, perm]
    return img


def augment_template(img: np.ndarray) -> np.ndarray:
    """Photometric only — template must stay stable."""
    return _photometric_augment(img)


def augment_search(img: np.ndarray) -> Tuple[np.ndarray, bool]:
    """
    Photometric + optional horizontal flip.
    Returns (augmented_image, flipped:bool).
    """
    img = _photometric_augment(img)
    flipped = False
    if random.random() < 0.5:
        img = cv2.flip(img, 1)
        flipped = True
    return img, flipped


# ──────────────────────────────────────────────────────────────────────────────
# ImageNet normalisation
# ──────────────────────────────────────────────────────────────────────────────
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def to_tensor(img: np.ndarray) -> torch.Tensor:
    """uint8 HWC BGR → float32 CHW RGB, ImageNet-normalised."""
    img = img[:, :, ::-1].copy().astype(np.float32) / 255.0
    img = (img - _MEAN) / _STD
    return torch.from_numpy(img.transpose(2, 0, 1))


# ──────────────────────────────────────────────────────────────────────────────
# Per-worker VideoCapture cache
# ──────────────────────────────────────────────────────────────────────────────
class _VideoCache:
    """LRU VideoCapture cache for one DataLoader worker process."""

    def __init__(self, maxsize: int = 64):
        self._caps: Dict[str, cv2.VideoCapture] = {}
        self._counts: Dict[str, int] = {}
        self._order: List[str] = []
        self._maxsize: int = maxsize

    def get(self, path: str) -> Optional[Tuple[cv2.VideoCapture, int]]:
        path = os.path.normpath(path)
        if path in self._caps:
            return self._caps[path], self._counts[path]

        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            cap.release()
            logger.warning("_VideoCache: cannot open %s", path)
            return None

        decoded = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if decoded <= 0:
            decoded = 0
            while True:
                ret, _ = cap.read()
                if not ret:
                    break
                decoded += 1
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        if len(self._order) >= self._maxsize:
            oldest = self._order.pop(0)
            old_cap = self._caps.pop(oldest, None)
            self._counts.pop(oldest, None)
            if old_cap:
                old_cap.release()

        self._caps[path] = cap
        self._counts[path] = decoded
        self._order.append(path)
        return cap, decoded

    def __del__(self):
        for cap in self._caps.values():
            try:
                cap.release()
            except Exception:
                pass


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
    """
    Open every video once at startup, record decoded_frames, and flag
    sequences whose video is missing or corrupt.
    """
    valid: List[SeqMeta] = []
    corrupted: List[str] = []
    for seq in sequences:
        seq_id = f"{seq.dataset}/{seq.seq_name}"

        if not os.path.exists(seq.video_path):
            logger.warning("[SKIP] missing video: %s", seq.video_path)
            corrupted.append(f"{seq_id}: file not found")
            seq.valid = False
            continue

        cap = cv2.VideoCapture(seq.video_path)
        if not cap.isOpened():
            cap.release()
            logger.warning("[SKIP] cannot open: %s", seq.video_path)
            corrupted.append(f"{seq_id}: VideoCapture.isOpened() = False")
            seq.valid = False
            continue

        decoded = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        seq.decoded_frames = decoded

        if decoded <= 0:
            logger.warning("[SKIP] zero frames decoded: %s", seq_id)
            corrupted.append(f"{seq_id}: decoded_frames=0")
            seq.valid = False
            continue

        if abs(decoded - seq.n_frames) > max(5, 0.02 * seq.n_frames):
            logger.warning(
                "[WARN] frame count mismatch %s: manifest=%d decoded=%d",
                seq_id, seq.n_frames, decoded,
            )

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


# ──────────────────────────────────────────────────────────────────────────────
# InferenceSequence
# ──────────────────────────────────────────────────────────────────────────────
class InferenceSequence:
    """
    Iterate over all frames of one video sequence for tracker evaluation.
    Usage
    ─────
    seq = InferenceSequence(seq_info, data_root)
    first_frame, init_box = seq.get_init()
    for frame_idx, frame in seq:
        bbox = tracker.track(frame)
        seq.record(frame_id x, bbox)
    results = seq.get_results()
    """

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
            box = self._predictions.get(fi, [0, 0, 0, 0])
            rows.append((row_id, *[round(v, 2) for v in box]))
        return rows

    def release(self):
        self.cap.release()

    def __del__(self):
        try:
            self.cap.release()
        except Exception:
            pass


# ──────────────────────────────────────────────────────────────────────────────
# TrackingPair
# ──────────────────────────────────────────────────────────────────────────────
class TrackingPair:
    __slots__ = ["template", "search", "gt_box", "seq_id", "frame_idx"]

    def __init__(self, template, search, gt_box, seq_id, frame_idx):
        self.template = template
        self.search = search
        self.gt_box = gt_box
        self.seq_id = seq_id
        self.frame_idx = frame_idx


# ──────────────────────────────────────────────────────────────────────────────
# TrainingDataset (v3 — jitter fix)
# ──────────────────────────────────────────────────────────────────────────────
class TrainingDataset(Dataset):
    """
    Manifest-aware training dataset.
    CRITICAL FIX in v3:
    ───────────────────
    Search crop jitter: the search region is no longer always centred exactly
    on s_box.  Instead the centre is jittered by Gaussian noise with
    σ  = jitter_sigma × crop_side (default 0.25).  This forces the model to
    genuinely localise the target rather than always predicting the centre,
    which was the root cause of ~4% tracker performance.

    Args
    ────
    jitter_sigma  Standard deviation of search centre jitter as a fraction
                  of the crop side length.  0.25 is standard (SiamRPN++ etc).
                   Set to 0.0 to disable (not recommended).
    """

    def __init__(
        self,
        manifest_path: str,
        data_root: str,
        split: str = "train",
        template_size: int = 128,
        search_size: int = 256,
        max_gap_frames: int = 100,
        samples_per_epoch: int = 50_000,
        augment: bool = True,
        jitter_sigma: float = 0.25,
        corrupt_log_path: Optional[str] = None,
        _sequences: Optional[List["SeqMeta"]] = None,
    ):
        self.data_root = data_root
        self.template_size = template_size
        self.search_size = search_size
        self.max_gap_frames = max_gap_frames
        self.samples_per_epoch = samples_per_epoch
        self.augment = augment
        self.jitter_sigma = jitter_sigma

        if _sequences is not None:
            self.sequences = _sequences
            if not self.sequences:
                raise RuntimeError("_sequences list is empty.")
            print(
                f"[TrainingDataset] (pre-split) |  "
                f"{len(self.sequences)} sequences |  "
                f"{samples_per_epoch} samples/epoch |  "
                f"jitter_sigma={jitter_sigma}"
            )
            return

        with open(manifest_path, "r") as f:
            manifest = json.load(f)

        if split not in manifest:
            raise ValueError(
                f"Split '{split}' not in manifest.  "
                f"Available: {list(manifest.keys())}"
            )

        if split == "public_lb":
            raise ValueError(
                "The 'public_lb' split must NOT be used for training or  "
                "validation — it only has first-frame annotations."
            )

        raw_seqs = manifest[split]
        seqs: List[SeqMeta] = []
        for seq_dict in raw_seqs.values():
            ann_rel = seq_dict.get("annotation_path")
            ann_path = os.path.normpath(os.path.join(data_root, ann_rel)) if ann_rel else None
            annotation: List[List[float]] = []
            if ann_path and os.path.exists(ann_path):
                annotation = load_annotation(ann_path)

            seqs.append(SeqMeta(
                dataset=seq_dict["dataset"],
                seq_name=seq_dict["seq_name"],
                n_frames=seq_dict["n_frames"],
                native_fps=float(seq_dict.get("native_fps", 30)),
                video_path=os.path.normpath(os.path.join(data_root, seq_dict["video_path"])),
                annotation_path=os.path.normpath(ann_path) if ann_path else None,
                annotation=annotation,
            ))

        self.sequences, _ = validate_sequences(seqs, log_path=corrupt_log_path)

        if not self.sequences:
            raise RuntimeError(
                f"No valid sequences found for split='{split}'.  "
                "Check data_root and manifest paths."
            )

        print(
            f"[TrainingDataset] split='{split}' |  "
            f"{len(self.sequences)} valid sequences |  "
            f"{samples_per_epoch} samples/epoch |  "
            f"jitter_sigma={jitter_sigma}"
        )

    # ── Train / Val splitter ──────────────────────────────────────────────

    @classmethod
    def split_train_val(
        cls,
        manifest_path: str,
        data_root: str,
        val_ratio: float = 0.15,
        train_samples: int = 50_000,
        val_samples: int = 2_000,
        template_size: int = 128,
        search_size: int = 256,
        max_gap_frames: int = 100,
        jitter_sigma: float = 0.25,
        seed: int = 42,
        corrupt_log_path: Optional[str] = None,
    ) -> "Tuple[TrainingDataset, TrainingDataset]":
        with open(manifest_path, "r") as f:
            manifest = json.load(f)

        if "train" not in manifest:
            raise ValueError("'train' key not found in manifest.")

        raw_seqs = manifest["train"]
        seqs: List[SeqMeta] = []
        for seq_dict in raw_seqs.values():
            ann_rel = seq_dict.get("annotation_path")
            ann_path = os.path.normpath(os.path.join(data_root, ann_rel)) if ann_rel else None
            annotation: List[List[float]] = []
            if ann_path and os.path.exists(ann_path):
                annotation = load_annotation(ann_path)

            seqs.append(SeqMeta(
                dataset=seq_dict["dataset"],
                seq_name=seq_dict["seq_name"],
                n_frames=seq_dict["n_frames"],
                native_fps=float(seq_dict.get("native_fps", 30)),
                video_path=os.path.normpath(os.path.join(data_root, seq_dict["video_path"])),
                annotation_path=os.path.normpath(ann_path) if ann_path else None,
                annotation=annotation,
            ))

        valid_seqs, _ = validate_sequences(seqs, log_path=corrupt_log_path)

        if not valid_seqs:
            raise RuntimeError("No valid train sequences found.")

        rng = random.Random(seed)
        shuffled = valid_seqs.copy()
        rng.shuffle(shuffled)

        n_val = max(1, int(len(shuffled) * val_ratio))
        n_train = len(shuffled) - n_val
        train_seqs = shuffled[:n_train]
        val_seqs = shuffled[n_train:]

        print(
            f"[split_train_val] {len(valid_seqs)} valid sequences split into  "
            f"{len(train_seqs)} train / {len(val_seqs)} val   "
            f"(val_ratio={val_ratio:.0%}, seed={seed})"
        )

        common_kwargs = dict(
            manifest_path=manifest_path,
            data_root=data_root,
            template_size=template_size,
            search_size=search_size,
            max_gap_frames=max_gap_frames,
        )

        train_ds = cls(
            **common_kwargs,
            samples_per_epoch=train_samples,
            augment=True,
            jitter_sigma=jitter_sigma,
            _sequences=train_seqs,
        )
        val_ds = cls(
            **common_kwargs,
            samples_per_epoch=val_samples,
            augment=False,
            jitter_sigma=0.0,  # no jitter for val (measure true performance)
            _sequences=val_seqs,
        )

        return train_ds, val_ds

    # ── Per-worker VideoCache ─────────────────────────────────────────────

    @property
    def vcache(self):
        if not hasattr(self, "_vcache") or self._vcache is None:
            self._vcache = _VideoCache(maxsize=64)
        return self._vcache

    # ── Sampling helpers ──────────────────────────────────────────────────

    def _safe_upper(self, seq: SeqMeta) -> int:
        ann_len = len(seq.annotation)
        decoded = seq.decoded_frames if seq.decoded_frames > 0 else seq.n_frames
        return max(1, min(seq.n_frames, decoded, ann_len))

    def _max_gap_for_seq(self, seq: SeqMeta) -> int:
        fps_ratio = seq.native_fps / 30.0
        return max(1, int(self.max_gap_frames * fps_ratio))

    def _get_annotation_for_frame(self, seq: SeqMeta, frame_idx: int) -> List[float]:
        ann = seq.annotation
        if len(ann) == 1:
            return ann[0]
        return ann[min(frame_idx, len(ann) - 1)]

    def _read_frame(self, seq: SeqMeta, frame_idx: int) -> Optional[np.ndarray]:
        result = self.vcache.get(seq.video_path)
        if result is None:
            return None
        cap, decoded_count = result
        safe_idx = min(frame_idx, decoded_count - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, safe_idx)
        ret, frame = cap.read()
        return frame if ret else None

    def _crop_search_with_jitter(
        self,
        frame: np.ndarray,
        box: List[float],
        output_size: int,
        context_factor: float,
        jitter_sigma: float,
    ) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """
        Crop search region with optional Gaussian jitter on the crop centre.

        CRITICAL FIX:
        Without jitter the crop is always centred on the GT box, so the model
        always sees the target at normalised position (0.5, 0.5) and learns to
        ALWAYS predict the centre rather than to localise.  This destroys
        tracking performance.

        With jitter (sigma=0.25) the crop centre is perturbed by up to ~±0.5
        crop-sides, forcing the model to find the target at varying positions.

        Returns:
            crop          – resized crop
            scale         – output_size / crop_side
            (ox1, oy1)    – ORIGINAL (pre-padding) top-left; use for GT mapping
        """
        import math

        H, W = frame.shape[:2]
        x, y, w, h = box
        cx = x + w / 2
        cy = y + h / 2

        # Crop side using geometric mean (matches tracker.py)
        # Add scale jitter to Increase training data diversity
        scale_jitter = random.uniform(0.9, 1.1)
        s = max(math.sqrt(w * h) * context_factor * scale_jitter, 1.0)

        # ── JITTER ──────────────────────────────────────────────────────
        if jitter_sigma > 0:
            cx = cx + random.gauss(0, jitter_sigma * s)
            cy = cy + random.gauss(0, jitter_sigma * s)

        x1 = int(round(cx - s / 2))
        y1 = int(round(cy - s / 2))
        x2 = int(round(cx + s / 2))
        y2 = int(round(cy + s / 2))

        # Remember original origin for GT mapping
        ox1, oy1 = x1, y1

        pad_top = max(0, -y1)
        pad_left = max(0, -x1)
        pad_bottom = max(0, y2 - H)
        pad_right = max(0, x2 - W)

        if any([pad_top, pad_left, pad_bottom, pad_right]):
            frame = cv2.copyMakeBorder(
                frame, pad_top, pad_bottom, pad_left, pad_right,
                cv2.BORDER_CONSTANT, value=(114, 114, 114),
            )
            x1 += pad_left; x2 += pad_left
            y1 += pad_top; y2 += pad_top

        crop = frame[y1:y2, x1:x2]
        crop_side = max(crop.shape[0], crop.shape[1], 1)
        scale = output_size / crop_side
        crop = cv2.resize(crop, (output_size, output_size))
        return crop, scale, (ox1, oy1)

    def _build_sample(self, seq: SeqMeta) -> Optional[TrackingPair]:
        """Build one (template, search, gt_box) pair from a sequence."""
        import math

        upper = self._safe_upper(seq)
        if upper < 2:
            return None

        max_gap = self._max_gap_for_seq(seq)

        t_idx = random.randint(0, upper - 2)
        s_idx = min(t_idx + random.randint(1, max_gap), upper - 1)

        t_box = self._get_annotation_for_frame(seq, t_idx)
        s_box = self._get_annotation_for_frame(seq, s_idx)

        if t_box[2] <= 0 or t_box[3] <= 0:
            return None
        if s_box[2] <= 0 or s_box[3] <= 0:
            return None

        t_frame = self._read_frame(seq, t_idx)
        s_frame = self._read_frame(seq, s_idx)
        if t_frame is None or s_frame is None:
            return None

        # ── Template crop (no jitter — stable reference) ──────────────────
        t_sx = max(math.sqrt(t_box[2] * t_box[3]) * 2.0, 1.0)
        t_cx = t_box[0] + t_box[2] / 2
        t_cy = t_box[1] + t_box[3] / 2
        tx1 = int(round(t_cx - t_sx / 2))
        ty1 = int(round(t_cy - t_sx / 2))
        tx2 = int(round(t_cx + t_sx / 2))
        ty2 = int(round(t_cy + t_sx / 2))
        tH, tW = t_frame.shape[:2]
        tpad_t = max(0, -ty1); tpad_l = max(0, -tx1)
        tpad_b = max(0, ty2 - tH); tpad_r = max(0, tx2 - tW)
        if any([tpad_t, tpad_l, tpad_b, tpad_r]):
            t_frame_p = cv2.copyMakeBorder(t_frame, tpad_t, tpad_b, tpad_l, tpad_r,
                                           cv2.BORDER_CONSTANT, value=(114, 114, 114))
            tx1 += tpad_l; tx2 += tpad_l; ty1 += tpad_t; ty2 += tpad_t
        else:
            t_frame_p = t_frame
        t_patch = t_frame_p[ty1:ty2, tx1:tx2]
        t_side = max(t_patch.shape[0], t_patch.shape[1], 1)
        t_crop = cv2.resize(t_patch, (self.template_size, self.template_size))

        # ── Search crop WITH JITTER ────────────────────────────────────────
        s_crop, s_scale, (sox1, soy1) = self._crop_search_with_jitter(
            s_frame, s_box, self.search_size,
            context_factor=4.0,
            jitter_sigma=self.jitter_sigma if self.augment else 0.0,
        )

        # ── Augment ───────────────────────────────────────────────────────
        s_flipped = False
        if self.augment:
            t_crop = augment_template(t_crop)
            s_crop, s_flipped = augment_search(s_crop)

        # ── GT bbox in normalised search crop coords ──────────────────────
        # FIX: use ORIGINAL (pre-padding) top-left (sox1, soy1)  for mapping.
        # s_box is in original frame coords; sox1 is also in original (may be  < 0).
        # This makes the formula consistent regardless of whether crop needed padding.
        s_cx = s_box[0] + s_box[2] / 2
        s_cy = s_box[1] + s_box[3] / 2

        # Crop side length (same formula used in _crop_search_with_jitter)
        # After jitter the crop is still the same SIZE, just centred elsewhere.
        # We need the actual crop_side to compute scale —  s_scale was returned.
        cx_crop = (s_cx - sox1) * s_scale
        cy_crop = (s_cy - soy1) * s_scale
        w_crop = s_box[2] * s_scale
        h_crop = s_box[3] * s_scale

        cx_n = cx_crop / self.search_size
        cy_n = cy_crop / self.search_size
        w_n = w_crop / self.search_size
        h_n = h_crop / self.search_size

        if s_flipped:
            cx_n = 1.0 - cx_n

        cx_n = float(np.clip(cx_n, 0.0, 1.0))
        cy_n = float(np.clip(cy_n, 0.0, 1.0))
        w_n = float(np.clip(w_n, 0.0, 1.0))
        h_n = float(np.clip(h_n, 0.0, 1.0))

        # Skip degenerate GT (target jittered completely out of crop)
        if w_n < 0.01 or h_n < 0.01:
            return None

        gt_box = torch.tensor([cx_n, cy_n, w_n, h_n], dtype=torch.float32)

        return TrackingPair(
            template=to_tensor(t_crop),
            search=to_tensor(s_crop),
            gt_box=gt_box,
            seq_id=f"{seq.dataset}/{seq.seq_name}",
            frame_idx=s_idx,
        )

    # ── PyTorch Dataset interface ─────────────────────────────────────────

    def __len__(self) -> int:
        return self.samples_per_epoch

    def __getitem__(self, _idx: int) -> dict:
        for attempt in range(10):
            seq = random.choice(self.sequences)
            sample = self._build_sample(seq)
            if sample is not None:
                return {
                    "template": sample.template,
                    "search": sample.search,
                    "gt_box": sample.gt_box,
                    "seq_id": sample.seq_id,
                    "frame_idx": sample.frame_idx,
                }

        logger.warning(
            "__getitem__: 10 consecutive sample failures — returning zero tensor"
        )
        return {
            "template": torch.zeros(3, self.template_size, self.template_size),
            "search": torch.zeros(3, self.search_size, self.search_size),
            "gt_box": torch.tensor([0.5, 0.5, 0.1, 0.1]),
            "seq_id": "fallback",
            "frame_idx": 0,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Dataset stats helper
# ──────────────────────────────────────────────────────────────────────────────
def print_dataset_stats(manifest_path: str, data_root: str):
    with open(manifest_path) as f:
        manifest = json.load(f)
    print(f"\n{'Split': <12} {'Seqs': >6} {'Frames': >10}  "
          f"{'MinFPS': >8} {'MaxFPS': >8} {'AvgFPS': >8} ")
    print("─" * 58)

    for split, seqs in manifest.items():
        frames = [v["n_frames"] for v in seqs.values()]
        fps_list = [v.get("native_fps", 30) for v in seqs.values()]
        print(
            f"{split: <12} {len(seqs): >6} {sum(frames): >10}  "
            f"{min(fps_list): >8} {max(fps_list): >8}  "
            f"{sum(fps_list)/len(fps_list): >8.1f}"
        )

    print("\n── File existence + frame count check ── ")
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

            if ap and not os.path.exists(ap):
                missing += 1

        status = []
        if missing:
            status.append(f"{missing} missing/corrupt ")
        if miscount:
            status.append(f"{miscount} frame-count mismatch ")
        print(f"  {split}: {', '.join(status) if status else '✓ all OK'} ")


# ──────────────────────────────────────────────────────────────────────────────
# Quick jitter sanity check
# ──────────────────────────────────────────────────────────────────────────────
def _verify_jitter(n_samples: int = 1000):
    """
    Verify that with jitter enabled, GT positions are spread across [0,1]
    rather than concentrated at 0.5.
    """
    import math

    cx_vals = []
    cy_vals = []
    frame_hw = 1920
    box = [800.0, 400.0, 80.0, 60.0]  # typical box
    for _ in range(n_samples):
        w, h = box[2], box[3]
        s = max(math.sqrt(w * h) * 4.0, 1.0)
        cx_gt = box[0] + w / 2
        cy_gt = box[1] + h / 2

        cx_jit = cx_gt + random.gauss(0, 0.25 * s)
        cy_jit = cy_gt + random.gauss(0, 0.25 * s)

        ox1 = int(round(cx_jit - s / 2))
        oy1 = int(round(cy_jit - s / 2))

        scale = 256 / s
        cx_n = (cx_gt - ox1) * scale / 256
        cy_n = (cy_gt - oy1) * scale / 256
        cx_vals.append(cx_n)
        cy_vals.append(cy_n)

    cx_arr = np.array(cx_vals)
    cy_arr = np.array(cy_vals)
    print(f"Jitter test ({n_samples} samples): ")
    print(f"  cx: mean={cx_arr.mean():.3f}  std={cx_arr.std():.3f}   "
          f"range=[{cx_arr.min():.3f}, {cx_arr.max():.3f}] ")
    print(f"  cy: mean={cy_arr.mean():.3f}  std={cy_arr.std():.3f}   "
          f"range=[{cy_arr.min():.3f}, {cy_arr.max():.3f}] ")
    assert abs(cx_arr.mean() - 0.5) < 0.05, "Mean should be ~0.5"
    assert cx_arr.std() > 0.1, "Std should be  > 0.1 (target not always at center)"
    print("  ✓ Jitter is working correctly — target at varying positions")


# ──────────────────────────────────────────────────────────────────────────────
# Smoke-test
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    _ROOT = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest",
                        default=str(_ROOT / "data/contest_release/metadata/contestant_manifest.json"))
    parser.add_argument("--data_root",
                        default=str(_ROOT / "data/contest_release"))
    parser.add_argument("--mode", choices=["stats", "sample", "validate", "jitter"],
                        default="stats")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)s  %(message)s")

    if args.mode == "stats":
        print_dataset_stats(args.manifest, args.data_root)

    elif args.mode == "jitter":
        _verify_jitter(n_samples=2000)

    elif args.mode == "validate":
        ds = TrainingDataset(
            manifest_path=args.manifest,
            data_root=args.data_root,
            split="train",
            samples_per_epoch=10,
            augment=False,
            corrupt_log_path="/tmp/corrupted_sequences.txt",
        )
        print(f"\nValid sequences: {len(ds.sequences)}")

    elif args.mode == "sample":
        ds = TrainingDataset(
            manifest_path=args.manifest,
            data_root=args.data_root,
            split="train",
            samples_per_epoch=100,
            augment=True,
            jitter_sigma=0.25,
        )
        sample = ds[0]
        print("\nSample keys: ", list(sample.keys()))
        print("template shape: ", sample["template"].shape)
        print("search shape:   ", sample["search"].shape)
        print("gt_box:         ", sample["gt_box"])
        print("  (with jitter, cx/cy should NOT always be ~0.5)")
        print("seq_id:         ", sample["seq_id"])
        print("\n✓ dataset.py v3 works correctly!")