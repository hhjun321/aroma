# AROMA Data Preparation Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Normalize ISP-AD, MVTec AD, VisA dataset directory structures to a unified format compatible with AROMA pipeline CLI arguments, and produce a dataset config that maps each experiment unit to ready-to-run CLI args.

**Architecture:** VisA requires filesystem reorganization based on `split_csv` CSVs (symlinks, not copies, to avoid duplicating ~30 GB). ISP-AD and MVTec are already compatible for Stage 1 but need a unified config entry per modality/category. A single `prepare_visa.py` script handles VisA reorganization; a `dataset_config.json` consolidates all three datasets into one reference file for running experiments.

**Tech Stack:** Python 3, `pathlib`, `csv`, `os.symlink`, `pytest`

---

## Dataset Structures (reference)

### ISP-AD — already compatible for Stage 1
```
isp/unsupervised/
├── ASM/train/good/          ← --image_dir (Stage 1)
├── ASM/test/area/           ← seed candidates (Stage 1b) — all defective
├── ASM/test/points/         ← seed candidates (Stage 1b) — all defective
├── ASM/ground_truth/{area,points}/
├── LSM_1/train/good/
├── LSM_1/train/good_reduced/
├── LSM_1/test/{area,good,points}/
├── LSM_2/train/good/
└── LSM_2/test/{area,good,points}/
```
No filesystem changes needed. Only config entry required.

### MVTec AD — fully compatible, no changes needed
```
mvtec/<category>/train/good/       ← --image_dir (Stage 1)
mvtec/<category>/test/<defect>/    ← seed candidates (Stage 1b)
mvtec/<category>/ground_truth/
```

### VisA — requires reorganization
```
Current:
visa/<category>/Data/Images/Normal/    ← all normal (train+test mixed)
visa/<category>/Data/Images/Anomaly/  ← all anomalous (train+test mixed)
visa/<category>/Data/Masks/Anomaly/
visa/split_csv/<category>.csv          ← defines train/test split

Target (AROMA-compatible):
visa/<category>/train/good/            ← symlinks to Normal/ (train split)
visa/<category>/test/good/             ← symlinks to Normal/ (test split)
visa/<category>/test/anomaly/          ← symlinks to Anomaly/ (test split)
visa/<category>/ground_truth/anomaly/  ← symlinks to Masks/Anomaly/
```

### VisA split_csv schema
Each `split_csv/<category>.csv` has columns:
```
object,split,label,image_path,mask_path
```
- `split`: `train` | `test`
- `label`: `normal` | `anomaly`
- `image_path`: relative path from visa root, e.g. `candle/Data/Images/Normal/0001.JPG`
- `mask_path`: relative path or empty string for normal images

---

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `prepare_visa.py` | Create | Reads split_csv per category, creates symlinks into AROMA-compatible structure |
| `dataset_config.json` | Create | Maps each dataset/category/modality to AROMA CLI args |
| `tests/test_prepare_visa.py` | Create | Tests for prepare_visa.py using a tmp-dir fixture |

---

## Task 1: `prepare_visa.py` — VisA filesystem reorganization

**Files:**
- Create: `prepare_visa.py`
- Test: `tests/test_prepare_visa.py`

### Context

`split_csv/<category>.csv` defines which images belong to train vs test. All actual images live in `Data/Images/Normal/` and `Data/Images/Anomaly/`. We create symlinks (not copies) to avoid duplicating data.

The script is idempotent: if symlinks already exist, skip them. If a symlink target does not exist, warn and skip.

### CLI usage

```bash
python prepare_visa.py --visa_dir /content/drive/MyDrive/data/Aroma/visa
```

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_prepare_visa.py
import csv
import os
from pathlib import Path
import pytest
from prepare_visa import reorganize_category, reorganize_all


@pytest.fixture
def fake_visa(tmp_path):
    """Create a minimal VisA-like directory structure with 4 images."""
    cat = "candle"
    img_dir = tmp_path / cat / "Data" / "Images"
    mask_dir = tmp_path / cat / "Data" / "Masks"
    (img_dir / "Normal").mkdir(parents=True)
    (img_dir / "Anomaly").mkdir(parents=True)
    (mask_dir / "Anomaly").mkdir(parents=True)

    # Create dummy image files
    for name in ("n001.png", "n002.png"):
        (img_dir / "Normal" / name).write_bytes(b"img")
    for name in ("a001.png",):
        (img_dir / "Anomaly" / name).write_bytes(b"img")
        (mask_dir / "Anomaly" / name).write_bytes(b"mask")

    # Create split CSV
    csv_dir = tmp_path / "split_csv"
    csv_dir.mkdir()
    rows = [
        {"object": cat, "split": "train", "label": "normal",
         "image_path": f"{cat}/Data/Images/Normal/n001.png", "mask_path": ""},
        {"object": cat, "split": "test", "label": "normal",
         "image_path": f"{cat}/Data/Images/Normal/n002.png", "mask_path": ""},
        {"object": cat, "split": "test", "label": "anomaly",
         "image_path": f"{cat}/Data/Images/Anomaly/a001.png",
         "mask_path": f"{cat}/Data/Masks/Anomaly/a001.png"},
    ]
    with open(csv_dir / f"{cat}.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["object","split","label","image_path","mask_path"])
        writer.writeheader()
        writer.writerows(rows)

    return tmp_path


def test_train_good_symlinks_created(fake_visa):
    reorganize_category(fake_visa, "candle")
    target = fake_visa / "candle" / "train" / "good" / "n001.png"
    assert target.is_symlink() or target.exists()


def test_test_good_symlinks_created(fake_visa):
    reorganize_category(fake_visa, "candle")
    target = fake_visa / "candle" / "test" / "good" / "n002.png"
    assert target.is_symlink() or target.exists()


def test_test_anomaly_symlinks_created(fake_visa):
    reorganize_category(fake_visa, "candle")
    target = fake_visa / "candle" / "test" / "anomaly" / "a001.png"
    assert target.is_symlink() or target.exists()


def test_ground_truth_symlinks_created(fake_visa):
    reorganize_category(fake_visa, "candle")
    target = fake_visa / "candle" / "ground_truth" / "anomaly" / "a001.png"
    assert target.is_symlink() or target.exists()


def test_idempotent_second_run_does_not_raise(fake_visa):
    reorganize_category(fake_visa, "candle")
    reorganize_category(fake_visa, "candle")  # second run — must not raise


def test_reorganize_all_processes_all_csv_categories(fake_visa):
    reorganize_all(fake_visa)
    assert (fake_visa / "candle" / "train" / "good" / "n001.png").exists()
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /content/drive/MyDrive/project/aroma
python -m pytest tests/test_prepare_visa.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'prepare_visa'`

- [ ] **Step 3: Implement `prepare_visa.py`**

```python
"""
prepare_visa.py — Reorganize VisA dataset for AROMA pipeline compatibility.

Reads split_csv/<category>.csv and creates symlinks:
  train/good/     ← Normal images (train split)
  test/good/      ← Normal images (test split)
  test/anomaly/   ← Anomaly images (test split)
  ground_truth/anomaly/ ← Anomaly masks (test split)

Usage:
    python prepare_visa.py --visa_dir /path/to/visa

Idempotent: existing symlinks are skipped without error.
"""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Optional


_SPLIT_MAP = {
    ("train", "normal"): ("train", "good"),
    ("test",  "normal"): ("test",  "good"),
    ("test",  "anomaly"): ("test", "anomaly"),
}


def _make_symlink(src: Path, dst: Path) -> None:
    """Create symlink dst → src. Skip if dst already exists."""
    if dst.exists() or dst.is_symlink():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    # On Windows, symlinks may require elevated privileges or developer mode.
    # Fall back to a file copy if symlink creation fails.
    try:
        os.symlink(src.resolve(), dst)
    except (OSError, NotImplementedError):
        import shutil
        shutil.copy2(src, dst)


def reorganize_category(visa_root: Path, category: str) -> None:
    """Process one VisA category using its split CSV."""
    csv_path = visa_root / "split_csv" / f"{category}.csv"
    if not csv_path.exists():
        print(f"  [WARN] CSV not found: {csv_path}, skipping")
        return

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            split = row["split"].strip()
            label = row["label"].strip()
            img_rel = row["image_path"].strip()
            mask_rel = row.get("mask_path", "").strip()

            folder = _SPLIT_MAP.get((split, label))
            if folder is None:
                continue

            img_src = visa_root / img_rel
            img_dst = visa_root / category / folder[0] / folder[1] / Path(img_rel).name
            if img_src.exists():
                _make_symlink(img_src, img_dst)
            else:
                print(f"  [WARN] Image not found: {img_src}")

            # Ground truth mask (anomaly only)
            if label == "anomaly" and mask_rel:
                mask_src = visa_root / mask_rel
                mask_dst = (
                    visa_root / category / "ground_truth" / "anomaly"
                    / Path(mask_rel).name
                )
                if mask_src.exists():
                    _make_symlink(mask_src, mask_dst)
                else:
                    print(f"  [WARN] Mask not found: {mask_src}")


def reorganize_all(visa_root: Path) -> None:
    """Process all categories found in split_csv/."""
    csv_dir = visa_root / "split_csv"
    if not csv_dir.exists():
        raise FileNotFoundError(f"split_csv directory not found: {csv_dir}")
    for csv_file in sorted(csv_dir.glob("*.csv")):
        category = csv_file.stem
        print(f"Processing: {category}")
        reorganize_category(visa_root, category)


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Reorganize VisA dataset for AROMA pipeline compatibility"
    )
    parser.add_argument(
        "--visa_dir", required=True,
        help="Path to VisA root (contains split_csv/ and category folders)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    reorganize_all(Path(args.visa_dir))
    print("Done.")
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_prepare_visa.py -v
```

Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add prepare_visa.py tests/test_prepare_visa.py
git commit -m "feat: prepare_visa — reorganize VisA into AROMA-compatible train/test structure"
```

---

## Task 2: `dataset_config.json` — Unified experiment config

**Files:**
- Create: `dataset_config.json`

### Context

Each experiment unit = one dataset category (or ISP-AD modality). The config maps each unit to the exact CLI args needed for Stages 1–4. This replaces hand-typing paths when running experiments.

Colab base path: `/content/drive/MyDrive/data/Aroma/`

### Format

```json
{
  "<unit_id>": {
    "domain": "mvtec|isp|visa",
    "image_dir": "<absolute path to train/good/>",
    "seed_dir": "<absolute path to defect images for Stage 1b seed selection>",
    "notes": "<optional>"
  }
}
```

- [ ] **Step 1: Write the config file**

Create `dataset_config.json` with the following content:

```json
{
  "_comment": "AROMA dataset config — maps each experiment unit to Stage 1/1b CLI args. Base: /content/drive/MyDrive/data/Aroma/",

  "isp_ASM": {
    "domain": "isp",
    "image_dir": "/content/drive/MyDrive/data/Aroma/isp/unsupervised/ASM/train/good",
    "seed_dir": "/content/drive/MyDrive/data/Aroma/isp/unsupervised/ASM/test/area",
    "notes": "256x256px. seed_dir contains real defect images annotated by area."
  },
  "isp_LSM_1": {
    "domain": "isp",
    "image_dir": "/content/drive/MyDrive/data/Aroma/isp/unsupervised/LSM_1/train/good",
    "seed_dir": "/content/drive/MyDrive/data/Aroma/isp/unsupervised/LSM_1/test/area",
    "notes": "512x512px. Largest ISP-AD split (3678 train patches). Recommended starting modality."
  },
  "isp_LSM_2": {
    "domain": "isp",
    "image_dir": "/content/drive/MyDrive/data/Aroma/isp/unsupervised/LSM_2/train/good",
    "seed_dir": "/content/drive/MyDrive/data/Aroma/isp/unsupervised/LSM_2/test/area",
    "notes": "512x512px."
  },

  "mvtec_bottle":     { "domain": "mvtec", "image_dir": "/content/drive/MyDrive/data/Aroma/mvtec/bottle/train/good",     "seed_dir": "/content/drive/MyDrive/data/Aroma/mvtec/bottle/test/broken_large" },
  "mvtec_cable":      { "domain": "mvtec", "image_dir": "/content/drive/MyDrive/data/Aroma/mvtec/cable/train/good",      "seed_dir": "/content/drive/MyDrive/data/Aroma/mvtec/cable/test/cut_outer_insulation" },
  "mvtec_capsule":    { "domain": "mvtec", "image_dir": "/content/drive/MyDrive/data/Aroma/mvtec/capsule/train/good",    "seed_dir": "/content/drive/MyDrive/data/Aroma/mvtec/capsule/test/scratch" },
  "mvtec_carpet":     { "domain": "mvtec", "image_dir": "/content/drive/MyDrive/data/Aroma/mvtec/carpet/train/good",     "seed_dir": "/content/drive/MyDrive/data/Aroma/mvtec/carpet/test/cut" },
  "mvtec_grid":       { "domain": "mvtec", "image_dir": "/content/drive/MyDrive/data/Aroma/mvtec/grid/train/good",       "seed_dir": "/content/drive/MyDrive/data/Aroma/mvtec/grid/test/broken" },
  "mvtec_hazelnut":   { "domain": "mvtec", "image_dir": "/content/drive/MyDrive/data/Aroma/mvtec/hazelnut/train/good",   "seed_dir": "/content/drive/MyDrive/data/Aroma/mvtec/hazelnut/test/crack" },
  "mvtec_leather":    { "domain": "mvtec", "image_dir": "/content/drive/MyDrive/data/Aroma/mvtec/leather/train/good",    "seed_dir": "/content/drive/MyDrive/data/Aroma/mvtec/leather/test/cut" },
  "mvtec_metal_nut":  { "domain": "mvtec", "image_dir": "/content/drive/MyDrive/data/Aroma/mvtec/metal_nut/train/good",  "seed_dir": "/content/drive/MyDrive/data/Aroma/mvtec/metal_nut/test/scratch" },
  "mvtec_pill":       { "domain": "mvtec", "image_dir": "/content/drive/MyDrive/data/Aroma/mvtec/pill/train/good",       "seed_dir": "/content/drive/MyDrive/data/Aroma/mvtec/pill/test/crack" },
  "mvtec_screw":      { "domain": "mvtec", "image_dir": "/content/drive/MyDrive/data/Aroma/mvtec/screw/train/good",      "seed_dir": "/content/drive/MyDrive/data/Aroma/mvtec/screw/test/scratch_head" },
  "mvtec_tile":       { "domain": "mvtec", "image_dir": "/content/drive/MyDrive/data/Aroma/mvtec/tile/train/good",       "seed_dir": "/content/drive/MyDrive/data/Aroma/mvtec/tile/test/crack" },
  "mvtec_toothbrush": { "domain": "mvtec", "image_dir": "/content/drive/MyDrive/data/Aroma/mvtec/toothbrush/train/good", "seed_dir": "/content/drive/MyDrive/data/Aroma/mvtec/toothbrush/test/defective" },
  "mvtec_transistor": { "domain": "mvtec", "image_dir": "/content/drive/MyDrive/data/Aroma/mvtec/transistor/train/good", "seed_dir": "/content/drive/MyDrive/data/Aroma/mvtec/transistor/test/bent_lead" },
  "mvtec_wood":       { "domain": "mvtec", "image_dir": "/content/drive/MyDrive/data/Aroma/mvtec/wood/train/good",       "seed_dir": "/content/drive/MyDrive/data/Aroma/mvtec/wood/test/scratch" },
  "mvtec_zipper":     { "domain": "mvtec", "image_dir": "/content/drive/MyDrive/data/Aroma/mvtec/zipper/train/good",     "seed_dir": "/content/drive/MyDrive/data/Aroma/mvtec/zipper/test/broken_teeth" },

  "visa_candle":      { "domain": "visa", "image_dir": "/content/drive/MyDrive/data/Aroma/visa/candle/train/good",      "seed_dir": "/content/drive/MyDrive/data/Aroma/visa/candle/test/anomaly",      "notes": "Run prepare_visa.py first" },
  "visa_capsules":    { "domain": "visa", "image_dir": "/content/drive/MyDrive/data/Aroma/visa/capsules/train/good",    "seed_dir": "/content/drive/MyDrive/data/Aroma/visa/capsules/test/anomaly",    "notes": "Run prepare_visa.py first" },
  "visa_cashew":      { "domain": "visa", "image_dir": "/content/drive/MyDrive/data/Aroma/visa/cashew/train/good",      "seed_dir": "/content/drive/MyDrive/data/Aroma/visa/cashew/test/anomaly",      "notes": "Run prepare_visa.py first" },
  "visa_chewinggum":  { "domain": "visa", "image_dir": "/content/drive/MyDrive/data/Aroma/visa/chewinggum/train/good",  "seed_dir": "/content/drive/MyDrive/data/Aroma/visa/chewinggum/test/anomaly",  "notes": "Run prepare_visa.py first" },
  "visa_fryum":       { "domain": "visa", "image_dir": "/content/drive/MyDrive/data/Aroma/visa/fryum/train/good",       "seed_dir": "/content/drive/MyDrive/data/Aroma/visa/fryum/test/anomaly",       "notes": "Run prepare_visa.py first" },
  "visa_macaroni":    { "domain": "visa", "image_dir": "/content/drive/MyDrive/data/Aroma/visa/macaroni1/train/good",   "seed_dir": "/content/drive/MyDrive/data/Aroma/visa/macaroni1/test/anomaly",   "notes": "Run prepare_visa.py first" },
  "visa_pcb":         { "domain": "visa", "image_dir": "/content/drive/MyDrive/data/Aroma/visa/pcb1/train/good",        "seed_dir": "/content/drive/MyDrive/data/Aroma/visa/pcb1/test/anomaly",        "notes": "Run prepare_visa.py first" },
  "visa_pipe_fryum":  { "domain": "visa", "image_dir": "/content/drive/MyDrive/data/Aroma/visa/pipe_fryum/train/good",  "seed_dir": "/content/drive/MyDrive/data/Aroma/visa/pipe_fryum/test/anomaly",  "notes": "Run prepare_visa.py first" }
}
```

- [ ] **Step 2: Verify the file is valid JSON**

```bash
python -c "import json; json.load(open('dataset_config.json')); print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add dataset_config.json
git commit -m "feat: add dataset_config.json — unified AROMA experiment config for all 3 datasets"
```

---

## Task 3: ISP-AD image format verification

**Files:**
- No new files — verification only

### Context

Stage 1 reads `*.png` only (`image_dir_path.glob("*.png")`). ISP-AD images may be PNG or another format. This task verifies format and documents any needed conversion.

- [ ] **Step 1: Check image file extensions in ISP-AD train/good directories (run on Colab)**

```python
from pathlib import Path
base = Path("/content/drive/MyDrive/data/Aroma/isp/unsupervised")
for mod in ["ASM", "LSM_1", "LSM_2"]:
    good = base / mod / "train" / "good"
    exts = {f.suffix.lower() for f in good.iterdir() if f.is_file()}
    print(f"{mod}: {exts}")
```

Expected: `{'.png'}` for all modalities.

- [ ] **Step 2: If non-PNG formats found**

If any modality contains `.bmp`, `.tiff`, `.jpg` etc., add a conversion step:

```python
# Conversion snippet (only if needed — do not implement preemptively)
import cv2
for img in good.glob("*"):
    if img.suffix.lower() != ".png":
        frame = cv2.imread(str(img))
        cv2.imwrite(str(img.with_suffix(".png")), frame)
        img.unlink()
```

- [ ] **Step 3: Check VisA Normal/ image extensions (run on Colab)**

```python
visa_base = Path("/content/drive/MyDrive/data/Aroma/visa")
for cat in visa_base.iterdir():
    normal = cat / "Data" / "Images" / "Normal"
    if normal.exists():
        exts = {f.suffix.lower() for f in normal.iterdir() if f.is_file()}
        if exts != {'.png'}:
            print(f"{cat.name}: {exts}")
```

Note: VisA images are often `.JPG`. If so, `prepare_visa.py` symlinks will point to `.JPG` files, but Stage 1 will not find them (globs `*.png` only). See Task 4 if conversion is needed.

---

## Task 4 (conditional): VisA JPG → PNG conversion

**Only execute if Task 3 confirms VisA images are non-PNG.**

**Files:**
- Modify: `prepare_visa.py` — add `--convert_png` flag

- [ ] **Step 1: Add conversion test**

```python
# Add to tests/test_prepare_visa.py
def test_jpg_images_converted_to_png_when_flag_set(fake_visa, tmp_path):
    """If source images are JPG, converted PNG symlink targets are created."""
    # Rename the normal image to .jpg
    cat_dir = fake_visa / "candle" / "Data" / "Images" / "Normal"
    old = cat_dir / "n001.png"
    new = cat_dir / "n001.jpg"
    old.rename(new)

    # Update CSV
    csv_path = fake_visa / "split_csv" / "candle.csv"
    text = csv_path.read_text().replace("n001.png", "n001.jpg")
    csv_path.write_text(text)

    reorganize_category(fake_visa, "candle", convert_png=True)

    # Converted PNG should exist in train/good/
    converted = fake_visa / "candle" / "train" / "good" / "n001.png"
    assert converted.exists()
```

- [ ] **Step 2: Implement `convert_png` option in `prepare_visa.py`**

Add `convert_png: bool = False` parameter to `reorganize_category()` and `reorganize_all()`.

When `convert_png=True`, instead of symlinking `.JPG` files directly, convert to `.png` via OpenCV and save to the destination path (not a symlink).

```python
import cv2

def _link_or_convert(src: Path, dst: Path, convert_png: bool) -> None:
    if dst.exists() or dst.is_symlink():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if convert_png and src.suffix.lower() != ".png":
        dst = dst.with_suffix(".png")
        if dst.exists():
            return
        frame = cv2.imread(str(src))
        if frame is not None:
            cv2.imwrite(str(dst), frame)
        return
    _make_symlink(src, dst)
```

Update `reorganize_category()` to use `_link_or_convert()` and update the dst filename accordingly.

Add `--convert_png` flag to `_parse_args()`:
```python
parser.add_argument("--convert_png", action="store_true",
                    help="Convert non-PNG images to PNG (needed if VisA images are JPG)")
```

- [ ] **Step 3: Run all tests**

```bash
python -m pytest tests/test_prepare_visa.py -v
```

Expected: all tests pass (including the new JPG conversion test)

- [ ] **Step 4: Commit**

```bash
git add prepare_visa.py tests/test_prepare_visa.py
git commit -m "feat: prepare_visa — add --convert_png flag for JPG→PNG conversion"
```

---

## Execution Order (on Colab)

```bash
# 1. Reorganize VisA (with PNG conversion if needed)
python prepare_visa.py \
    --visa_dir /content/drive/MyDrive/data/Aroma/visa \
    [--convert_png]

# 2. Run Stage 1 for each experiment unit (example: ISP-AD LSM_1)
python stage1_roi_extraction.py \
    --image_dir /content/drive/MyDrive/data/Aroma/isp/unsupervised/LSM_1/train/good \
    --output_dir /content/drive/MyDrive/data/Aroma/output/isp_LSM_1 \
    --domain isp \
    --roi_levels both \
    --workers -1

# 3. Run Stage 1b (pick one seed image from seed_dir)
python stage1b_seed_characterization.py \
    --seed_defect /content/drive/MyDrive/data/Aroma/isp/unsupervised/LSM_1/test/area/0001.png \
    --output_dir /content/drive/MyDrive/data/Aroma/output/isp_LSM_1
```

All `image_dir` and `seed_dir` paths are pre-listed in `dataset_config.json`.

---

## Summary of Changes

| Dataset | Changes |
|---|---|
| **MVTec AD** | None — already compatible |
| **ISP-AD** | None to filesystem. Config entry added. Format verified. |
| **VisA** | `prepare_visa.py` creates `train/good/`, `test/good/`, `test/anomaly/`, `ground_truth/anomaly/` via symlinks. Optional JPG→PNG conversion. |
