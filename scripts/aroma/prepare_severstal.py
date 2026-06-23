#!/usr/bin/env python3
"""
prepare_severstal.py — Severstal Steel Defect (Kaggle) → AROMA layout.

Severstal ships defect masks as **RLE** inside train.csv (long format:
ImageId, ClassId, EncodedPixels), images are 1600×256 (W×H), and there are
**4 defect classes**. Normal images are simply train_images files that never
appear in train.csv.

This script materializes an MVTec-like layout under <output_dir> so the rest of
the AROMA pipeline (which assumes binary-PNG masks) can consume it unchanged:

    Aroma/severstal/
      train/good/                       (normal images — context prototype input)
      test/class1/ .. test/class4/      (defect images grouped by ClassId → defect_type)
      masks/{ImageId}.png               (merged binary mask, all classes OR'd — single mode)
      masks/class1/ .. masks/class4/    (per-class binary masks — multi mode)
      severstal_manifest.json

Defect organization into test/class{1..4}/ lets distribution_profiling read the
subfolder name as `defect_type` (Path(d).name) → Class 2 rarity surfaces as a
real Neff<4 imbalance.

RLE convention (Kaggle Severstal / CASDA rle_utils):
  - 1-indexed pixel positions.
  - pairs = [start, length, start, length, ...].
  - column-major (Fortran, order='F') over a (H=256, W=1600) frame.

Idempotent: re-running overwrites masks/manifest and re-links images.

Usage (Colab):
    !python $AROMA_SCRIPTS/prepare_severstal.py \\
        --train_csv    $SEVERSTAL/train.csv \\
        --train_images $SEVERSTAL/train_images \\
        --output_dir   /content/drive/MyDrive/data/Aroma/severstal
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Severstal fixed geometry: width=1600, height=256.
SEVERSTAL_H = 256
SEVERSTAL_W = 1600
N_CLASSES = 4
_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


# ---------------------------------------------------------------------------
# RLE decode (prefer CASDA rle_utils if importable, else inline)
# ---------------------------------------------------------------------------

def _inline_rle_decode(rle: str, shape: Tuple[int, int] = (SEVERSTAL_H, SEVERSTAL_W)) -> np.ndarray:
    """Decode a Kaggle/Severstal RLE string → uint8 mask (H, W) of {0,1}.

    1-indexed starts, column-major (order='F'). Empty/NaN → all-zero.
    """
    h, w = shape
    mask = np.zeros(h * w, dtype=np.uint8)
    if not rle or not str(rle).strip():
        return mask.reshape((h, w), order="F")
    s = np.asarray(str(rle).split(), dtype=np.int64)
    starts = s[0::2] - 1  # 1-indexed → 0-indexed
    lengths = s[1::2]
    for start, length in zip(starts, lengths):
        end = start + length
        if start < 0:
            start = 0
        if end > mask.size:
            end = mask.size
        mask[start:end] = 1
    return mask.reshape((h, w), order="F")


def _get_rle_decoder():
    """Return an rle_decode(rle, shape) callable.

    Tries CASDA's src/utils/rle_utils.py for byte-identical decoding; on any
    import failure falls back to the inline implementation above.
    """
    try:
        from src.utils.rle_utils import rle_decode as _casda_decode  # type: ignore

        def _wrap(rle: str, shape: Tuple[int, int] = (SEVERSTAL_H, SEVERSTAL_W)) -> np.ndarray:
            m = _casda_decode(rle, shape)
            return np.asarray(m, dtype=np.uint8)

        return _wrap, "casda.rle_utils"
    except Exception:
        return _inline_rle_decode, "inline"


# ---------------------------------------------------------------------------
# CSV parsing
# ---------------------------------------------------------------------------

def _parse_train_csv(csv_path: str) -> Dict[str, List[Tuple[int, str]]]:
    """train.csv (ImageId, ClassId, EncodedPixels) → {ImageId: [(ClassId, RLE), ...]}.

    Rows with empty/NaN EncodedPixels are skipped (no defect annotation).
    """
    grouped: Dict[str, List[Tuple[int, str]]] = {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        # Normalize header names defensively.
        fields = {k.lower().strip(): k for k in (reader.fieldnames or [])}
        c_img = fields.get("imageid", "ImageId")
        c_cls = fields.get("classid", "ClassId")
        c_rle = fields.get("encodedpixels", "EncodedPixels")
        for row in reader:
            image_id = (row.get(c_img) or "").strip()
            rle = (row.get(c_rle) or "").strip()
            cls_raw = (row.get(c_cls) or "").strip()
            if not image_id or not rle or rle.lower() == "nan":
                continue
            try:
                cls = int(float(cls_raw))
            except (TypeError, ValueError):
                continue
            if cls < 1 or cls > N_CLASSES:
                continue
            grouped.setdefault(image_id, []).append((cls, rle))
    return grouped


def _list_image_files(directory: str) -> Dict[str, str]:
    """{filename: abs_path} for every image file directly under `directory`."""
    d = Path(directory)
    out: Dict[str, str] = {}
    if not d.exists():
        return out
    for p in sorted(d.iterdir()):
        if p.is_file() and p.suffix.lower() in _IMG_EXTS:
            out[p.name] = str(p)
    return out


# ---------------------------------------------------------------------------
# File staging
# ---------------------------------------------------------------------------

def _link_or_copy(src: str, dst: str) -> None:
    """Symlink src→dst; fall back to copy. Skip if dst exists."""
    dst_p = Path(dst)
    if dst_p.exists() or dst_p.is_symlink():
        return
    dst_p.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.symlink(os.path.abspath(src), dst)
    except (OSError, NotImplementedError):
        shutil.copy2(src, dst)


def _write_mask_png(mask: np.ndarray, out_path: str) -> None:
    """Write a {0,1} mask as a 0/255 grayscale PNG (lossless)."""
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    arr = (np.asarray(mask, dtype=np.uint8) > 0).astype(np.uint8) * 255
    try:
        import cv2  # type: ignore

        cv2.imwrite(out_path, arr)
    except Exception:
        # Fallback: Pillow (still lossless PNG).
        from PIL import Image  # type: ignore

        Image.fromarray(arr, mode="L").save(out_path)


# ---------------------------------------------------------------------------
# Main prepare
# ---------------------------------------------------------------------------

def prepare(
    train_csv: str,
    train_images: str,
    output_dir: str,
) -> Dict[str, object]:
    rle_decode, decoder_name = _get_rle_decoder()
    print(f"[prepare_severstal] RLE decoder: {decoder_name} "
          f"(shape={(SEVERSTAL_H, SEVERSTAL_W)} order='F', 1-indexed)")

    grouped = _parse_train_csv(train_csv)
    all_files = _list_image_files(train_images)
    print(f"[prepare_severstal] csv defect ImageIds={len(grouped)}  "
          f"train_images files={len(all_files)}")

    out = Path(output_dir)
    good_dir = out / "train" / "good"
    masks_dir = out / "masks"
    good_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    for c in range(1, N_CLASSES + 1):
        (out / "test" / f"class{c}").mkdir(parents=True, exist_ok=True)
        (masks_dir / f"class{c}").mkdir(parents=True, exist_ok=True)

    # --- Normal = real files NOT present in csv (real-file set difference) ---
    defect_ids = set(grouped.keys())
    normal_ids = [fn for fn in all_files if fn not in defect_ids]
    for fn in normal_ids:
        _link_or_copy(all_files[fn], str(good_dir / fn))

    # --- Defect images: per-class organization + masks ---
    per_class_counts = {c: 0 for c in range(1, N_CLASSES + 1)}
    manifest_defects: List[Dict[str, object]] = []
    n_missing_file = 0

    for image_id, ann in sorted(grouped.items()):
        src = all_files.get(image_id)
        if src is None:
            n_missing_file += 1
            continue

        classes_present = sorted({c for (c, _) in ann})
        # Primary class for test/ organization: the smallest ClassId present
        # (deterministic). per-class masks still capture every class separately.
        primary_class = classes_present[0]

        # Place the defect image under its primary class subfolder.
        dst_img = out / "test" / f"class{primary_class}" / image_id
        _link_or_copy(src, str(dst_img))
        per_class_counts[primary_class] += 1

        # Merged binary mask (all classes OR'd) for single mode.
        merged = np.zeros((SEVERSTAL_H, SEVERSTAL_W), dtype=np.uint8)
        # Per-class accumulation for multi mode.
        per_class_masks: Dict[int, np.ndarray] = {}
        for (cls, rle) in ann:
            m = rle_decode(rle, (SEVERSTAL_H, SEVERSTAL_W))
            m = (np.asarray(m, dtype=np.uint8) > 0).astype(np.uint8)
            merged |= m
            if cls in per_class_masks:
                per_class_masks[cls] |= m
            else:
                per_class_masks[cls] = m.copy()

        stem = Path(image_id).stem
        _write_mask_png(merged, str(masks_dir / f"{stem}.png"))
        for cls, m in per_class_masks.items():
            _write_mask_png(m, str(masks_dir / f"class{cls}" / f"{stem}.png"))

        manifest_defects.append({
            "image_id": image_id,
            "primary_class": primary_class,
            "classes": classes_present,
            "merged_mask": f"masks/{stem}.png",
            "per_class_masks": {
                str(cls): f"masks/class{cls}/{stem}.png" for cls in per_class_masks
            },
        })

    manifest = {
        "dataset": "severstal",
        "image_shape": {"height": SEVERSTAL_H, "width": SEVERSTAL_W},
        "rle_decoder": decoder_name,
        "n_classes": N_CLASSES,
        "counts": {
            "normal": len(normal_ids),
            "defect_images": len(manifest_defects),
            "per_primary_class": {str(c): per_class_counts[c] for c in per_class_counts},
            "missing_files": n_missing_file,
        },
        "layout": {
            "train_good": "train/good",
            "test_classes": [f"test/class{c}" for c in range(1, N_CLASSES + 1)],
            "masks_merged": "masks/{stem}.png",
            "masks_per_class": "masks/class{c}/{stem}.png",
        },
        "defects": manifest_defects,
    }
    (out / "severstal_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    print(f"[prepare_severstal] normal={len(normal_ids)}  "
          f"defect_images={len(manifest_defects)}  "
          f"per_primary_class={per_class_counts}  missing_files={n_missing_file}")
    print(f"[prepare_severstal] wrote layout + masks + manifest → {out}")
    return manifest


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Severstal Steel Defect → AROMA layout (RLE→PNG, per-class)"
    )
    p.add_argument("--train_csv", required=True,
                   help="train.csv (ImageId,ClassId,EncodedPixels)")
    p.add_argument("--train_images", required=True,
                   help="Severstal train_images directory")
    p.add_argument("--output_dir", required=True,
                   help="Output root, e.g. /content/drive/MyDrive/data/Aroma/severstal")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = _parse_args(argv)
    prepare(args.train_csv, args.train_images, args.output_dir)
    print("Done.")


if __name__ == "__main__":
    main()
