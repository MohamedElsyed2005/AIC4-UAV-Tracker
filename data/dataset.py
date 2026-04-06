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
  dataset1/Car_video_2/annotation.txt   ← "x,y,w,h" per line (one line = first frame only OR all frames)
  ...

The contestant_manifest.json tells us which sequences are in
'train' vs 'public_lb' splits.
"""

import json
import os
import random
from pathlib import Path

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
    Each line: x,y,w,h   (top-left corner + width/height, 1-indexed in some
    datasets - we keep as-is and trust the data).
    Returns list of [x, y, w, h] floats. May be just 1 line (first-frame init)
    or one line per frame.
    """
    boxes = []
    with open(ann_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # support comma or tab or space separated
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


def crop_and_resize(frame: np.ndarray,
                    box: list,
                    output_size: int,
                    context_factor: float = 2.0) -> tuple[np.ndarray, float]:
    """
    Crop a square region centred on `box` with context padding,
    then resize to `output_size x output_size`.

    Returns:
        crop        - (output_size, output_size, 3)  uint8
        scale       - ratio output_size / crop_side  (used to rescale coords)
    """
    H, W = frame.shape[:2]
    x, y, w, h = box
    cx = x + w / 2
    cy = y + h / 2

    # square crop side with context
    s = (w + h) / 2 * context_factor
    s = max(s, 1.0)

    x1 = int(round(cx - s / 2))
    y1 = int(round(cy - s / 2))
    x2 = int(round(cx + s / 2))
    y2 = int(round(cy + s / 2))

    # Pad frame if crop goes outside
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
    scale = output_size / max(crop.shape[:2], default=1)
    crop  = cv2.resize(crop, (output_size, output_size))

    return crop, scale, (x1, y1)   # also return top-left for offset calc


def augment_image(img: np.ndarray) -> np.ndarray:
    """Light augmentations safe for UAV tracking."""
    # Random brightness / contrast
    alpha = random.uniform(0.8, 1.2)   # contrast
    beta  = random.randint(-20, 20)    # brightness
    img   = np.clip(alpha * img + beta, 0, 255).astype(np.uint8)

    # Random horizontal flip (50 %)
    if random.random() < 0.5:
        img = cv2.flip(img, 1)

    # Random colour channel shuffle (30 %)
    if random.random() < 0.3:
        perm = list(range(3))
        random.shuffle(perm)
        img = img[:, :, perm]

    return img


# ─────────────────────────────────────────────────────────────────────────────
# 1.  InferenceSequence  – used at test / submission time
# ─────────────────────────────────────────────────────────────────────────────

class InferenceSequence:
    """
    Iterate over all frames of a single video sequence.

    Usage
    ─────
    seq = InferenceSequence(seq_info, data_root)
    init_frame, init_box = seq.get_init()     # first frame + gt bbox
    for frame_idx, frame in seq:              # subsequent frames
        bbox = tracker.update(frame)
        seq.record(frame_idx, bbox)
    results = seq.get_results()               # list of (seq_id_str, x, y, w, h)
    """

    def __init__(self, seq_info: dict, data_root: str):
        self.seq_info  = seq_info
        self.data_root = data_root
        self.seq_id    = f"{seq_info['dataset']}/{seq_info['seq_name']}"
        self.n_frames  = seq_info["n_frames"]

        video_path = os.path.join(data_root, seq_info["video_path"])
        ann_path   = os.path.join(data_root, seq_info["annotation_path"])

        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")
        if not os.path.exists(ann_path):
            raise FileNotFoundError(f"Annotation not found: {ann_path}")

        self.cap    = cv2.VideoCapture(video_path)
        self.boxes  = load_annotation(ann_path)   # may be 1 or N lines

        # Results storage: frame_index → [x, y, w, h]
        self._predictions: dict[int, list] = {}

    # ── public API ────────────────────────────────────────────────────────────

    def get_init(self) -> tuple[np.ndarray, list]:
        """
        Returns (first_frame_BGR, init_bbox [x,y,w,h]).
        Resets capture to frame 0.
        """
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError(f"Cannot read first frame of {self.seq_id}")
        init_box = self.boxes[0]   # always line 0
        # Store frame-0 prediction = init_box (tracker is given GT for frame 0)
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
        """
        Returns list of (id_string, x, y, w, h) ready for CSV.
        id_string = "dataset1/Car_video_0", "dataset1/Car_video_1", ...
        """
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
    """
    Lightweight container for a single training sample.
    Returned by TrainingDataset.__getitem__.
    """
    __slots__ = ["template", "search", "gt_box", "seq_id", "frame_idx"]

    def __init__(self, template, search, gt_box, seq_id, frame_idx):
        self.template  = template   # Tensor (3, Ht, Wt)
        self.search    = search     # Tensor (3, Hs, Ws)
        self.gt_box    = gt_box     # Tensor (4,) normalised [cx,cy,w,h] in [0,1]
        self.seq_id    = seq_id
        self.frame_idx = frame_idx


# ─────────────────────────────────────────────────────────────────────────────
# 3.  TrainingDataset  – PyTorch Dataset for training loop
# ─────────────────────────────────────────────────────────────────────────────

# ImageNet mean/std for normalisation (standard for pretrained backbones)
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def to_tensor(img: np.ndarray) -> torch.Tensor:
    """uint8 HWC BGR → float32 CHW RGB, normalised."""
    img = img[:, :, ::-1].copy().astype(np.float32) / 255.0   # BGR→RGB
    img = (img - _MEAN) / _STD
    return torch.from_numpy(img.transpose(2, 0, 1))            # HWC→CHW


class TrainingDataset(Dataset):
    """
    Builds (template_frame, search_frame, gt_bbox) pairs from
    all training sequences.

    For each sample:
      - Pick a random sequence.
      - Pick a random "template frame" (earlier in time).
      - Pick a "search frame" up to `max_gap` frames later.
      - Crop both around the annotated target with context padding.
      - Return as tensors + normalised gt bbox.

    Args
    ────
    manifest_path   path to contestant_manifest.json
    data_root       root folder where dataset1/, dataset2/, … live
    split           'train' or 'public_lb'
    template_size   output size for template crop (default 128)
    search_size     output size for search crop   (default 256)
    max_gap         max frame gap between template and search (default 100)
    samples_per_epoch  virtual epoch length (default 50 000)
    augment         apply image augmentations  (default True)
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
        self.data_root        = data_root
        self.template_size    = template_size
        self.search_size      = search_size
        self.max_gap          = max_gap
        self.samples_per_epoch = samples_per_epoch
        self.augment          = augment

        with open(manifest_path, "r") as f:
            manifest = json.load(f)

        if split not in manifest:
            raise ValueError(f"Split '{split}' not in manifest. "
                             f"Available: {list(manifest.keys())}")

        self.sequences = list(manifest[split].values())

        # Pre-load all annotations (fast, text only)
        self._annotations: dict[str, list] = {}
        for seq in self.sequences:
            ann_path = os.path.join(data_root, seq["annotation_path"])
            if os.path.exists(ann_path):
                self._annotations[seq["seq_name"]] = load_annotation(ann_path)
            else:
                self._annotations[seq["seq_name"]] = []

        # Filter out sequences with no annotation or < 2 frames
        self.sequences = [
            s for s in self.sequences
            if len(self._annotations[s["seq_name"]]) >= 1
            and s["n_frames"] >= 2
        ]

        print(f"[TrainingDataset] split='{split}' | "
              f"{len(self.sequences)} sequences | "
              f"{samples_per_epoch} samples/epoch")

    # ── internal ──────────────────────────────────────────────────────────────

    def _get_annotation_for_frame(self, seq: dict, frame_idx: int) -> list:
        """
        Return [x, y, w, h] for a given frame.
        If annotation file has 1 line only (first-frame init), the same
        bbox is used for ALL frames (simple proxy — still useful for training).
        If it has N lines, return line[frame_idx].
        """
        ann = self._annotations[seq["seq_name"]]
        if len(ann) == 1:
            return ann[0]
        idx = min(frame_idx, len(ann) - 1)
        return ann[idx]

    def _read_frame(self, seq: dict, frame_idx: int) -> np.ndarray | None:
        """Read a single frame from the video."""
        video_path = os.path.join(self.data_root, seq["video_path"])
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()
        return frame if ret else None

    def _build_sample(self, seq: dict) -> TrackingPair | None:
        """Build one (template, search) pair from a sequence."""
        n = seq["n_frames"]
        ann = self._annotations[seq["seq_name"]]
        ann_len = len(ann)

        # ── choose template frame ──────────────────────────────────────────
        t_idx = random.randint(0, max(0, min(n - 2, ann_len - 2)))
        s_idx = min(t_idx + random.randint(1, self.max_gap), n - 1,
                    ann_len - 1)

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

        # ── crop ──────────────────────────────────────────────────────────
        t_crop, _, _    = crop_and_resize(t_frame, t_box,
                                          self.template_size,
                                          context_factor=2.0)
        s_crop, s_scale, (sx1, sy1) = crop_and_resize(
            s_frame, s_box, self.search_size, context_factor=4.0
        )

        # ── augment ───────────────────────────────────────────────────────
        if self.augment:
            t_crop = augment_image(t_crop)
            s_crop = augment_image(s_crop)

        # ── normalise gt bbox to search crop (cx,cy,w,h in [0,1]) ────────
        # Search crop was centred on s_box centre; compute offset
        s_cx = s_box[0] + s_box[2] / 2
        s_cy = s_box[1] + s_box[3] / 2

        # In crop coordinates
        cx_crop = (s_cx - sx1) * s_scale
        cy_crop = (s_cy - sy1) * s_scale
        w_crop  = s_box[2] * s_scale
        h_crop  = s_box[3] * s_scale

        # Normalise to [0, 1]
        cx_n = cx_crop / self.search_size
        cy_n = cy_crop / self.search_size
        w_n  = w_crop  / self.search_size
        h_n  = h_crop  / self.search_size

        # Clamp
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

    # ── Dataset interface ─────────────────────────────────────────────────────

    def __len__(self) -> int:
        return self.samples_per_epoch

    def __getitem__(self, _idx: int) -> dict:
        """
        Returns a dict (easier to collate with DataLoader):
          {
            'template':  Tensor (3, template_size, template_size),
            'search':    Tensor (3, search_size,   search_size),
            'gt_box':    Tensor (4,)   [cx, cy, w, h] normalised,
            'seq_id':    str,
            'frame_idx': int,
          }
        """
        for _ in range(10):   # retry up to 10× on bad samples
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
        # Fallback: return zeros (very rare)
        return {
            "template":  torch.zeros(3, self.template_size, self.template_size),
            "search":    torch.zeros(3, self.search_size,   self.search_size),
            "gt_box":    torch.tensor([0.5, 0.5, 0.1, 0.1]),
            "seq_id":    "unknown",
            "frame_idx": 0,
        }


# ─────────────────────────────────────────────────────────────────────────────
# 4.  prepare_data.py  helper  –  quick sanity-check / dataset stats
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

    # Check how many videos actually exist
    print("\n── File existence check ──")
    for split, seqs in manifest.items():
        missing = 0
        for seq in seqs.values():
            vp = os.path.join(data_root, seq["video_path"])
            ap = os.path.join(data_root, seq["annotation_path"])
            if not os.path.exists(vp) or not os.path.exists(ap):
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
        default=os.path.join(_ROOT, "data", "contest_release", "metadata", "contestant_manifest.json"))
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