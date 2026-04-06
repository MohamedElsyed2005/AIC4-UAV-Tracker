import json
import shutil
from pathlib import Path

"""
Organize raw video datasets into train and test splits
based on contestant_manifest.json.

Assumes this script is in the same folder as 'data/'.
The 'train/' and 'test/' folders already exist inside 'data/'.
"""

DATA_ROOT = Path("data")
JSON_PATH = DATA_ROOT / "metadata/contestant_manifest.json"

# -----------------------------
# Load manifest
# -----------------------------
with open(JSON_PATH, "r") as f:
    manifest = json.load(f)

# Extract splits
train_ids = set(manifest.get("train", {}).keys())
test_ids  = set(manifest.get("public_lb", {}).keys())

train_root = DATA_ROOT / "train"
test_root  = DATA_ROOT / "test"

# -----------------------------
# Move videos
# -----------------------------
for dataset_dir in sorted(DATA_ROOT.iterdir()):
    if not dataset_dir.is_dir() or dataset_dir.name in ("train", "test", "metadata"):
        continue

    for video_dir in sorted(dataset_dir.iterdir()):
        if not video_dir.is_dir():
            continue

        full_id = f"{dataset_dir.name}/{video_dir.name}"

        if full_id in test_ids:
            split = "TEST"
            dest_root = test_root
        elif full_id in train_ids:
            split = "TRAIN"
            dest_root = train_root
        else:
            print(f"[WARNING] {full_id} not found in manifest → skipped")
            continue

        dest = dest_root / dataset_dir.name / video_dir.name
        dest.mkdir(parents=True, exist_ok=True)  # ensure folder exists

        shutil.move(str(video_dir), str(dest))
        print(f"[{split}] {full_id}")

    # Remove empty dataset folder
    if not any(dataset_dir.iterdir()):
        dataset_dir.rmdir()