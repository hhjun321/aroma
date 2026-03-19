"""
prepare_visa.py — Reorganize VisA dataset for AROMA pipeline compatibility.

Supports two CSV layouts:

  1. Combined (standard VisA release):
       split_csv/1cls.csv — columns: object, split, label, image, mask

  2. Per-category (legacy / custom splits):
       split_csv/<category>.csv — columns: split, label, image_path, mask_path
       (also accepts 'image' / 'mask' column names)

Creates:
  train/good/            ← Normal images (train split)
  test/good/             ← Normal images (test split)
  test/anomaly/          ← Anomaly images (test split)
  ground_truth/anomaly/  ← Anomaly masks (test split)

Usage:
    python prepare_visa.py --visa_dir /path/to/visa [--convert_png]

Idempotent: existing symlinks/files are skipped without error.
"""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path


_SPLIT_MAP = {
    ("train", "normal"):  ("train", "good"),
    ("test",  "normal"):  ("test",  "good"),
    ("test",  "anomaly"): ("test",  "anomaly"),
}


def _make_symlink(src: Path, dst: Path) -> None:
    """Create symlink dst → src. Skip if dst already exists."""
    if dst.exists() or dst.is_symlink():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.symlink(src.resolve(), dst)
    except (OSError, NotImplementedError):
        import shutil
        shutil.copy2(src, dst)


def _link_or_convert(src: Path, dst: Path, convert_png: bool) -> None:
    """Symlink src to dst, or convert to PNG if convert_png and src is not PNG."""
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


def _process_rows(visa_root: Path, category: str, rows, convert_png: bool = False) -> None:
    """Place images/masks for one category given an iterable of CSV row dicts."""
    for row in rows:
        try:
            split = row["split"].strip()
            label = row["label"].strip()
            # Accept both 'image' (VisA standard) and 'image_path' (legacy)
            img_rel = (row.get("image") or row.get("image_path") or "").strip()
            mask_rel = (row.get("mask") or row.get("mask_path") or "").strip()
        except KeyError as e:
            print(f"  [WARN] Malformed CSV row: missing column {e}, skipping")
            continue

        if not img_rel:
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


def reorganize_category(visa_root: Path, category: str, convert_png: bool = False) -> None:
    """Process one VisA category using its per-category split CSV."""
    csv_path = visa_root / "split_csv" / f"{category}.csv"
    if not csv_path.exists():
        print(f"  [WARN] CSV not found: {csv_path}, skipping")
        return

    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))

    _process_rows(visa_root, category, rows, convert_png)


def reorganize_all(visa_root: Path, convert_png: bool = False) -> None:
    """Process all categories.

    Prefers split_csv/1cls.csv (standard VisA combined format, 'object' column).
    Falls back to per-category CSVs (split_csv/<category>.csv) if 1cls.csv absent.
    """
    csv_dir = visa_root / "split_csv"
    if not csv_dir.exists():
        raise FileNotFoundError(f"split_csv directory not found: {csv_dir}")

    one_cls = csv_dir / "1cls.csv"
    if one_cls.exists():
        _reorganize_from_combined_csv(visa_root, one_cls, convert_png)
        return

    # Legacy: per-category CSVs
    for csv_file in sorted(csv_dir.glob("*.csv")):
        category = csv_file.stem
        if not (visa_root / category).is_dir():
            print(f"Skipping:   {csv_file.name} (no matching category directory)")
            continue
        print(f"Processing: {category}")
        reorganize_category(visa_root, category, convert_png=convert_png)


def _reorganize_from_combined_csv(visa_root: Path, csv_path: Path, convert_png: bool = False) -> None:
    """Read 1cls.csv, group rows by 'object', dispatch to _process_rows."""
    rows_by_category: dict[str, list] = {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            obj = row.get("object", "").strip()
            if obj:
                rows_by_category.setdefault(obj, []).append(row)

    for category, rows in sorted(rows_by_category.items()):
        if not (visa_root / category).is_dir():
            print(f"Skipping:   {category} (no matching category directory)")
            continue
        print(f"Processing: {category}")
        _process_rows(visa_root, category, rows, convert_png)


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
