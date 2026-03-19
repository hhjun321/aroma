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


def _link_or_convert(src: Path, dst: Path, convert_png: bool) -> None:
    """Symlink src to dst, or convert to PNG if convert_png is True and src is not PNG."""
    if convert_png and src.suffix.lower() != ".png":
        dst_png = dst.with_suffix(".png")
        if dst_png.exists():
            return
        dst_png.parent.mkdir(parents=True, exist_ok=True)
        import cv2
        frame = cv2.imread(str(src))
        if frame is not None:
            cv2.imwrite(str(dst_png), frame)
        else:
            print(f"  [WARN] Could not read image for conversion: {src}")
    else:
        _make_symlink(src, dst)


def reorganize_category(visa_root: Path, category: str, convert_png: bool = False) -> None:
    """Process one VisA category using its split CSV."""
    csv_path = visa_root / "split_csv" / f"{category}.csv"
    if not csv_path.exists():
        print(f"  [WARN] CSV not found: {csv_path}, skipping")
        return

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                split = row["split"].strip()
                label = row["label"].strip()
                img_rel = row["image_path"].strip()
                mask_rel = row.get("mask_path", "").strip()
            except KeyError as e:
                print(f"  [WARN] Malformed CSV row in {csv_path}: missing column {e}, skipping")
                continue

            folder = _SPLIT_MAP.get((split, label))
            if folder is None:
                continue

            img_src = visa_root / img_rel
            img_dst = visa_root / category / folder[0] / folder[1] / Path(img_rel).name
            if img_src.exists():
                _link_or_convert(img_src, img_dst, convert_png)
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


def reorganize_all(visa_root: Path, convert_png: bool = False) -> None:
    """Process all categories found in split_csv/."""
    csv_dir = visa_root / "split_csv"
    if not csv_dir.exists():
        raise FileNotFoundError(f"split_csv directory not found: {csv_dir}")
    for csv_file in sorted(csv_dir.glob("*.csv")):
        category = csv_file.stem
        if not (visa_root / category).is_dir():
            print(f"Skipping:   {csv_file.name} (no matching category directory)")
            continue
        print(f"Processing: {category}")
        reorganize_category(visa_root, category, convert_png=convert_png)


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Reorganize VisA dataset for AROMA pipeline compatibility"
    )
    parser.add_argument(
        "--visa_dir", required=True,
        help="Path to VisA root (contains split_csv/ and category folders)"
    )
    parser.add_argument(
        "--convert_png", action="store_true",
        help="Convert non-PNG images to PNG (needed when VisA images are JPG)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    reorganize_all(Path(args.visa_dir), convert_png=args.convert_png)
    print("Done.")
