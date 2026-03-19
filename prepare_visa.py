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
