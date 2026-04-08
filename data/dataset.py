"""
dataset.py  –  AIC-4 UAV Tracker Competition
=============================================
Handles TWO use-cases:
  1. TrainingDataset  – yields (template_crop, search_crop, bbox_offset)
                        pairs used during model training.
  2. InferenceSequence – iterates over every frame of ONE sequence
                         for tracker evaluation / submission generation.

Data layout expected on disk
─────────────────────────────
<data_root>/
  dataset1/Car_video_2/Car_video_2.mp4
  dataset1/Car_video_2/annotation.txt   ← "x,y,w,h" per line
  ...

FIXES applied vs original:
  1. augment_image → augment_search: now returns (img, flipped:bool)
     so _build_sample can mirror cx_n when a horizontal flip occurred.
  2. _read_frame: opens VideoCapture once per (seq, worker) via an
     instance-level LRU cache instead of opening/closing every call.
  3. InferenceSequence.__init__: checks cap.isOpened() after
     VideoCapture() so corrupted / missing videos raise early.
  4. crop_and_resize return type hint fixed: tuple[ndarray, float, tuple].
  5. print_dataset_stats: guards against None annotation_path.
"""

import json
import os
import random
from pathlib import Path
from functools import lru_cache

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


# ─────────────────────────────────────────────────────────────────────────────
# Helper utilities
# ─────────────────────────────────────────────────────────────────────────────

def load_annotation(ann_path: str) -> list[list[float]]:
    """
    Read annotation file.
    Each line: x,y,w,h   (top-left corner + width/height).
    Returns list of [x, y, w, h] floats.
    """
    boxes = []
    with open(ann_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.replace("\t", ",").replace(" ", ",").split(",")
            parts = [p for p in parts if p]
            if len(parts) >= 4:
                x, y, w, h = float(parts[0]), float(parts[1]), \
                              float(parts[2]), float(parts[3])
                boxes.append([x, y, w, h])
    return boxes


def xywh_to_xyxy(box):
    """[x, y, w, h] → [x1, y1, x2, y2]"""
    x, y, w, h = box
    return [x, y, x + w, y + h]


def xyxy_to_xywh(box):
    """[x1, y1, x2, y2] → [x, y, w, h]"""
    x1, y1, x2, y2 = box
    return [x1, y1, x2 - x1, y2 - y1]


def clip_box(box, frame_h, frame_w, margin=0):
    """Clip [x,y,w,h] to frame boundaries."""
    x, y, w, h = box
    x = max(margin, min(x, frame_w - margin))
    y = max(margin, min(y, frame_h - margin))
    w = max(1, min(w, frame_w - x - margin))
    h = max(1, min(h, frame_h - y - margin))
    return [x, y, w, h]


# FIX 4: return type hint now matches actual 3-tuple return value
def crop_and_resize(frame: np.ndarray,
                    box: list,
                    output_size: int,
                    context_factor: float = 2.0) -> tuple[np.ndarray, float, tuple]:
    """
    Crop a square region centred on `box` with context padding,
    then resize to `output_size x output_size`.

    Returns:
        crop        - (output_size, output_size, 3)  uint8
        scale       - ratio output_size / crop_side
        (x1, y1)    - top-left corner of crop in (possibly padded) frame coords
    """
    H, W = frame.shape[:2]
    x, y, w, h = box
    cx = x + w / 2
    cy = y + h / 2

    s = (w + h) / 2 * context_factor
    s = max(s, 1.0)

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
            cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )
        x1 += pad_left; x2 += pad_left
        y1 += pad_top;  y2 += pad_top

    crop = frame[y1:y2, x1:x2]
    # FIX 4: explicit max(h, w) instead of passing tuple to max()
    crop_side = max(crop.shape[0], crop.shape[1])
    scale = output_size / max(crop_side, 1)
    crop  = cv2.resize(crop, (output_size, output_size))

    return crop, scale, (x1, y1)


# ─────────────────────────────────────────────────────────────────────────────
# FIX 1: augment_image split into two functions
#   - augment_template: photometric only (no flip — template must stay stable)
#   - augment_search:   photometric + flip, returns (img, flipped:bool)
# ─────────────────────────────────────────────────────────────────────────────

def _photometric_augment(img: np.ndarray) -> np.ndarray:
    """Brightness/contrast jitter + colour channel shuffle."""
    alpha = random.uniform(0.8, 1.2)
    beta  = random.randint(-20, 20)
    img   = np.clip(alpha * img + beta, 0, 255).astype(np.uint8)
    if random.random() < 0.3:
        perm = list(range(3))
        random.shuffle(perm)
        img = img[:, :, perm]
    return img


def augment_template(img: np.ndarray) -> np.ndarray:
    """Photometric augmentations for the template crop (no flip)."""
    return _photometric_augment(img)


def augment_search(img: np.ndarray) -> tuple[np.ndarray, bool]:
    """
    Augmentations for the search crop.

    Returns:
        img     - augmented image
        flipped - True if a horizontal flip was applied
                  (caller must mirror cx_n: cx_n = 1.0 - cx_n)
    """
    img = _photometric_augment(img)
    flipped = False
    # FIX 1: track the flip so caller can mirror the GT box cx
    if random.random() < 0.5:
        img = cv2.flip(img, 1)
        flipped = True
    return img, flipped


# ─────────────────────────────────────────────────────────────────────────────
# 1.  InferenceSequence  – used at test / submission time
# ─────────────────────────────────────────────────────────────────────────────

class InferenceSequence:
    """
    Iterate over all frames of a single video sequence.

    Usage
    ─────
    seq = InferenceSequence(seq_info, data_root)
    init_frame, init_box = seq.get_init()
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

        video_path = os.path.join(data_root, seq_info["video_path"])
        ann_path_rel = seq_info.get("annotation_path")

        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")

        # FIX 3: check isOpened() immediately — catches corrupted MP4s
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise RuntimeError(
                f"Cannot open video (corrupted or unsupported codec): {video_path}"
            )

        # Load annotations if present
        if ann_path_rel:
            ann_path = os.path.join(data_root, ann_path_rel)
            if not os.path.exists(ann_path):
                raise FileNotFoundError(f"Annotation not found: {ann_path}")
            self.boxes = load_annotation(ann_path)
        else:
            self.boxes = []   # submission sequences may have no annotation

        self._predictions: dict[int, list] = {}

    def get_init(self) -> tuple[np.ndarray, list]:
        """Returns (first_frame_BGR, init_bbox [x,y,w,h]). Resets to frame 0."""
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

    def record(self, frame_idx: int, bbox: list):
        """Store tracker output for one frame."""
        self._predictions[frame_idx] = list(bbox)

    def get_results(self) -> list[tuple]:
        """Returns list of (id_string, x, y, w, h) ready for CSV."""
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
# 2.  TrackingPair  – one (template, search) sample for training
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
# 3.  TrainingDataset  – PyTorch Dataset for training loop
# ─────────────────────────────────────────────────────────────────────────────

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def to_tensor(img: np.ndarray) -> torch.Tensor:
    """uint8 HWC BGR → float32 CHW RGB, ImageNet-normalised."""
    img = img[:, :, ::-1].copy().astype(np.float32) / 255.0
    img = (img - _MEAN) / _STD
    return torch.from_numpy(img.transpose(2, 0, 1))


# ─────────────────────────────────────────────────────────────────────────────
# FIX 2: per-worker VideoCapture cache
#   DataLoader workers are separate processes. Using an instance-level dict
#   keyed by video_path means each worker builds its own cache of open
#   VideoCapture handles — no file is opened more than once per worker.
#   This eliminates the O(N) open/seek/close overhead from the original code.
# ─────────────────────────────────────────────────────────────────────────────

class _VideoCache:
    """
    Simple LRU-style VideoCapture cache for one worker process.
    Keeps the last `maxsize` videos open.
    """
    def __init__(self, maxsize: int = 8):
        self._cache: dict[str, cv2.VideoCapture] = {}
        self._order: list[str] = []
        self._maxsize = maxsize

    def get(self, path: str) -> cv2.VideoCapture | None:
        if path in self._cache:
            return self._cache[path]
        cap = cv2.VideoCapture(path)
        # FIX 3: guard corrupted files here too
        if not cap.isOpened():
            cap.release()
            return None
        # Evict oldest if over capacity
        if len(self._order) >= self._maxsize:
            oldest = self._order.pop(0)
            self._cache.pop(oldest, None).release()
        self._cache[path] = cap
        self._order.append(path)
        return cap

    def __del__(self):
        for cap in self._cache.values():
            try:
                cap.release()
            except Exception:
                pass


class TrainingDataset(Dataset):
    """
    Builds (template_frame, search_frame, gt_bbox) pairs from
    all training sequences.

    Args
    ────
    manifest_path      path to contestant_manifest.json
    data_root          root folder where dataset1/, dataset2/, … live
    split              'train' or 'public_lb'
    template_size      output size for template crop (default 128)
    search_size        output size for search crop   (default 256)
    max_gap            max frame gap between template and search (default 100)
    samples_per_epoch  virtual epoch length (default 50_000)
    augment            apply image augmentations (default True)
    """

    def __init__(
        self,
        manifest_path: str,
        data_root: str,
        split: str = "train",
        template_size: int = 128,
        search_size: int   = 256,
        max_gap: int       = 100,
        samples_per_epoch: int = 50_000,
        augment: bool      = True,
    ):
        self.data_root         = data_root
        self.template_size     = template_size
        self.search_size       = search_size
        self.max_gap           = max_gap
        self.samples_per_epoch = samples_per_epoch
        self.augment           = augment

        with open(manifest_path, "r") as f:
            manifest = json.load(f)

        if split not in manifest:
            raise ValueError(
                f"Split '{split}' not in manifest. "
                f"Available: {list(manifest.keys())}"
            )

        self.sequences = list(manifest[split].values())

        self._annotations: dict[str, list] = {}
        for seq in self.sequences:
            ann_path_rel = seq.get("annotation_path")
            if ann_path_rel:
                ann_path = os.path.join(data_root, ann_path_rel)
                if os.path.exists(ann_path):
                    self._annotations[seq["seq_name"]] = load_annotation(ann_path)
                else:
                    self._annotations[seq["seq_name"]] = []
            else:
                self._annotations[seq["seq_name"]] = []

        self.sequences = [
            s for s in self.sequences
            if len(self._annotations[s["seq_name"]]) >= 1
            and s["n_frames"] >= 2
        ]

        print(f"[TrainingDataset] split='{split}' | "
              f"{len(self.sequences)} sequences | "
              f"{samples_per_epoch} samples/epoch")

        # FIX 2: video cache — one per Dataset instance (= one per worker)
        self._vcache = _VideoCache(maxsize=16)

    def _get_annotation_for_frame(self, seq: dict, frame_idx: int) -> list:
        ann = self._annotations[seq["seq_name"]]
        if len(ann) == 1:
            return ann[0]
        idx = min(frame_idx, len(ann) - 1)
        return ann[idx]

    def _read_frame(self, seq: dict, frame_idx: int) -> np.ndarray | None:
        """
        FIX 2: use cached VideoCapture instead of open/seek/close per call.
        FIX 3: returns None cleanly if video is corrupt/unreadable.
        """
        video_path = os.path.join(self.data_root, seq["video_path"])
        cap = self._vcache.get(video_path)
        if cap is None:
            return None   # corrupted or missing — _build_sample will skip
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        return frame if ret else None

    def _build_sample(self, seq: dict) -> TrackingPair | None:
        """Build one (template, search) pair from a sequence."""
        n       = seq["n_frames"]
        ann     = self._annotations[seq["seq_name"]]
        ann_len = len(ann)

        t_idx = random.randint(0, max(0, min(n - 2, ann_len - 2)))
        s_idx = min(t_idx + random.randint(1, self.max_gap), n - 1, ann_len - 1)

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

        # ── crop ──────────────────────────────────────────────────────────
        t_crop, _, _ = crop_and_resize(
            t_frame, t_box, self.template_size, context_factor=2.0
        )
        s_crop, s_scale, (sx1, sy1) = crop_and_resize(
            s_frame, s_box, self.search_size, context_factor=4.0
        )

        # ── augment ───────────────────────────────────────────────────────
        # FIX 1: separate template/search augment; capture flip flag
        s_flipped = False
        if self.augment:
            t_crop             = augment_template(t_crop)
            s_crop, s_flipped  = augment_search(s_crop)

        # ── normalise gt bbox to search crop (cx,cy,w,h in [0,1]) ────────
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

        # FIX 1: mirror cx when search was flipped
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
            seq_id    = f"{seq['dataset']}/{seq['seq_name']}",
            frame_idx = s_idx,
        )

    def __len__(self) -> int:
        return self.samples_per_epoch

    def __getitem__(self, _idx: int) -> dict:
        for _ in range(10):
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
        # Fallback (very rare — only if 10 consecutive videos are corrupted)
        return {
            "template":  torch.zeros(3, self.template_size, self.template_size),
            "search":    torch.zeros(3, self.search_size,   self.search_size),
            "gt_box":    torch.tensor([0.5, 0.5, 0.1, 0.1]),
            "seq_id":    "unknown",
            "frame_idx": 0,
        }


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Dataset stats helper
# ─────────────────────────────────────────────────────────────────────────────

def print_dataset_stats(manifest_path: str, data_root: str):
    """Print a summary table of the dataset."""
    with open(manifest_path) as f:
        manifest = json.load(f)

    print(f"\n{'Split':<12} {'Sequences':>10} {'Total Frames':>14} "
          f"{'Min':>6} {'Max':>6} {'Avg':>6}")
    print("─" * 60)
    for split, seqs in manifest.items():
        frames = [v["n_frames"] for v in seqs.values()]
        print(f"{split:<12} {len(seqs):>10} {sum(frames):>14} "
              f"{min(frames):>6} {max(frames):>6} "
              f"{sum(frames)//len(frames):>6}")

    print("\n── File existence check ──")
    for split, seqs in manifest.items():
        missing = 0
        for seq in seqs.values():
            vp = os.path.join(data_root, seq["video_path"])
            # FIX 5: guard against None annotation_path
            ann_rel = seq.get("annotation_path")
            ap = os.path.join(data_root, ann_rel) if ann_rel else None
            vp_ok = os.path.exists(vp)
            ap_ok = (ap is None) or os.path.exists(ap)
            if not vp_ok or not ap_ok:
                missing += 1
        status = "✓ all present" if missing == 0 else f"✗ {missing} missing"
        print(f"  {split}: {status}")


# ─────────────────────────────────────────────────────────────────────────────
# Quick smoke-test  (run:  python dataset.py)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    _ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    parser.add_argument("--manifest",
        default=os.path.join(_ROOT, "data", "contest_release", "metadata",
                             "contestant_manifest.json"))
    parser.add_argument("--data_root",
        default=os.path.join(_ROOT, "data", "contest_release"))
    parser.add_argument("--mode", choices=["stats", "sample"], default="stats")
    args = parser.parse_args()

    if args.mode == "stats":
        print_dataset_stats(args.manifest, args.data_root)

    elif args.mode == "sample":
        ds = TrainingDataset(
            manifest_path=args.manifest,
            data_root=args.data_root,
            split="train",
            samples_per_epoch=100,
            augment=True,
        )
        sample = ds[0]
        print("\nSample keys:", list(sample.keys()))
        print("template shape:", sample["template"].shape)
        print("search shape:  ", sample["search"].shape)
        print("gt_box:        ", sample["gt_box"])
        print("seq_id:        ", sample["seq_id"])
        print("\n✓ dataset.py works correctly!")