#!/usr/bin/env python3
"""
prepare_aitex.py — AITEX Fabric Image Database (Kaggle) → AROMA (MVTec-style) layout.

AITEX (Kaggle nexuswho/aitex-fabric-image-database) ships THREE top-level
folders, all 4096×256 PNG:

    Defect_images/    nnnn_ddd_ff.png        (defect image; ddd=defect code, ff=fabric)
    Mask_images/      nnnn_ddd_ff_mask.png   (binary mask, white=defect area)
    NODefect_images/  nnnn_000_ff.png        (defect-free image; defect code = 000)

Two AITEX quirks this script handles explicitly:
  - NOT every defect image has a mask. Mask_images/ has FEWER files than
    Defect_images/ — join by basename and SKIP+count defect images with no
    matching *_mask.png (masks are mandatory for AROMA morphology features).
  - masks live in a SEPARATE folder keyed by the '_mask' suffix.

This script materializes an MVTec-like layout under <output_dir> so the existing
distribution_profiling `_find_mask_path` mvtec/aitex branch resolves masks
unchanged (candidate = .../ground_truth/{defect_type}/{stem}_mask.png):

    Aroma/aitex/
      train/good/                          (NODefect_images → context prototype input)
      test/{ddd}/{stem}.png                (defect images grouped by defect code ddd → defect_type)
      ground_truth/{ddd}/{stem}_mask.png   (binary masks, matched by stem)
      aitex_manifest.json

Grouping defect images into test/{ddd}/ lets distribution_profiling read the
subfolder name as `defect_type` (Path(d).name) → each AITEX defect code surfaces
as a distinct defect_type. ddd is parsed as the 2nd underscore-token of the
filename nnnn_ddd_ff.png (kept as the literal token, e.g. '002', so config
seed_dirs match the on-disk folder name exactly).

Idempotent: re-running re-links images/masks and rewrites the manifest; existing
destination files are skipped.

Usage (Colab):
    !python $AROMA_SCRIPTS/prepare_aitex.py \\
        --defect_images   $AITEX_RAW/Defect_images \\
        --mask_images     $AITEX_RAW/Mask_images \\
        --nodefect_images $AITEX_RAW/NODefect_images \\
        --output_dir      /content/drive/MyDrive/data/Aroma/aitex
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_IMG_EXTS = {".png", ".PNG"}
_MASK_SUFFIX = "_mask"

# Guarded image libs (cv2 preferred, PIL+numpy fallback) — used ONLY to detect
# all-zero / unreadable masks so they are skipped, not silently passed to the
# profiling Otsu fallback (which would invent a fake defect region). When no lib
# is available we cannot inspect pixels → treat as "keep" (do not over-skip).
try:
    import numpy as _np  # type: ignore[import]
    _HAS_NUMPY = True
except Exception:
    _HAS_NUMPY = False
try:
    import cv2 as _cv2  # type: ignore[import]
    _HAS_CV2 = True
except Exception:
    _HAS_CV2 = False
try:
    from PIL import Image as _PILImage  # type: ignore[import]
    _HAS_PIL = True
except Exception:
    _HAS_PIL = False


def _mask_nonzero(mask_path: str) -> Optional[bool]:
    """True if mask readable AND has >=1 nonzero pixel; False if unreadable or
    all-zero; None if no image lib available (caller then keeps the mask)."""
    if _HAS_CV2:
        arr = _cv2.imread(mask_path, _cv2.IMREAD_GRAYSCALE)
        if arr is None:
            return False
        return bool(arr.max() > 0)
    if _HAS_PIL and _HAS_NUMPY:
        try:
            arr = _np.asarray(_PILImage.open(mask_path).convert("L"))
        except Exception:  # noqa: BLE001
            return False
        return bool(arr.max() > 0)
    return None


# ---------------------------------------------------------------------------
# Filename parsing
# ---------------------------------------------------------------------------

def _parse_defect_code(stem: str) -> Optional[str]:
    """Parse the defect code (ddd) from an AITEX filename stem 'nnnn_ddd_ff'.

    Returns the literal 2nd underscore-token (e.g. '002'), or None if the stem
    does not have at least 3 underscore-separated tokens.
    """
    parts = stem.split("_")
    if len(parts) < 3:
        return None
    return parts[1]


# ---------------------------------------------------------------------------
# File listing / staging
# ---------------------------------------------------------------------------

def _list_pngs(directory: str) -> List[Path]:
    """Sorted list of .png files directly under `directory` (empty if missing)."""
    d = Path(directory)
    if not d.exists():
        return []
    return sorted(p for p in d.iterdir() if p.is_file() and p.suffix in _IMG_EXTS)


def _link_or_copy(src: str, dst: str) -> None:
    """Symlink src→dst; fall back to copy. Skip if dst exists (idempotent)."""
    dst_p = Path(dst)
    if dst_p.exists() or dst_p.is_symlink():
        return
    dst_p.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.symlink(os.path.abspath(src), dst)
    except (OSError, NotImplementedError):
        shutil.copy2(src, dst)


# ---------------------------------------------------------------------------
# Main prepare
# ---------------------------------------------------------------------------

def prepare(
    defect_images: str,
    mask_images: str,
    nodefect_images: str,
    output_dir: str,
) -> Dict[str, object]:
    defect_files = _list_pngs(defect_images)
    nodefect_files = _list_pngs(nodefect_images)
    mask_files = _list_pngs(mask_images)
    print(f"[prepare_aitex] defect_images={len(defect_files)}  "
          f"nodefect_images={len(nodefect_files)}  mask_images={len(mask_files)}")

    out = Path(output_dir)
    good_dir = out / "train" / "good"
    good_dir.mkdir(parents=True, exist_ok=True)

    # --- Mask index: {defect_image_stem: mask_path} keyed by stem sans '_mask' ---
    #   mask file nnnn_ddd_ff_mask.png → key nnnn_ddd_ff
    mask_index: Dict[str, Path] = {}
    for m in mask_files:
        mstem = m.stem
        if mstem.endswith(_MASK_SUFFIX):
            key = mstem[: -len(_MASK_SUFFIX)]
        else:
            # Unexpected: mask without the '_mask' suffix — index by stem as-is.
            key = mstem
        mask_index[key] = m

    # --- Normal (defect-free) images → train/good/ ---
    for src in nodefect_files:
        _link_or_copy(str(src), str(good_dir / src.name))

    # --- Defect images grouped by defect code; masks matched by stem ---
    per_code_counts: Dict[str, int] = {}
    n_skipped_no_mask = 0
    n_skipped_empty_mask = 0
    n_unparsed_code = 0
    skipped_examples: List[str] = []
    manifest_defects: List[Dict[str, object]] = []

    for src in defect_files:
        stem = src.stem
        ddd = _parse_defect_code(stem)
        if ddd is None:
            n_unparsed_code += 1
            if len(skipped_examples) < 10:
                skipped_examples.append(f"unparsed_code:{src.name}")
            continue

        mask_src = mask_index.get(stem)
        if mask_src is None:
            # AITEX known quirk: some defect images have no mask. Skip (masks
            # are mandatory for AROMA morphology) and count for honest reporting.
            n_skipped_no_mask += 1
            if len(skipped_examples) < 10:
                skipped_examples.append(f"no_mask:{src.name}")
            continue

        # AITEX known quirk: some masks exist but are ALL-ZERO (no annotated
        # defect region) or unreadable. Staging them lets profiling's Otsu
        # fallback invent a fake defect (mask_source=fallback_otsu). Skip them
        # like no-mask so only genuine ground-truth defects seed the pipeline.
        if _mask_nonzero(str(mask_src)) is False:
            n_skipped_empty_mask += 1
            if len(skipped_examples) < 10:
                skipped_examples.append(f"empty_mask:{src.name}")
            continue

        # Stage image under test/{ddd}/{stem}.png
        dst_img = out / "test" / ddd / src.name
        _link_or_copy(str(src), str(dst_img))
        # Stage mask under ground_truth/{ddd}/{stem}_mask.png (mvtec convention).
        dst_mask = out / "ground_truth" / ddd / f"{stem}{_MASK_SUFFIX}.png"
        _link_or_copy(str(mask_src), str(dst_mask))

        per_code_counts[ddd] = per_code_counts.get(ddd, 0) + 1
        manifest_defects.append({
            "image_id": src.name,
            "defect_code": ddd,
            "image": f"test/{ddd}/{src.name}",
            "mask": f"ground_truth/{ddd}/{stem}{_MASK_SUFFIX}.png",
        })

    defect_codes = sorted(per_code_counts.keys())
    manifest = {
        "dataset": "aitex",
        "image_shape": {"height": 256, "width": 4096},
        "source_layout": {
            "defect_images": "Defect_images/nnnn_ddd_ff.png",
            "mask_images": "Mask_images/nnnn_ddd_ff_mask.png",
            "nodefect_images": "NODefect_images/nnnn_000_ff.png",
        },
        "counts": {
            "normal": len(nodefect_files),
            "defect_images_matched": len(manifest_defects),
            "defect_images_total": len(defect_files),
            "mask_files_total": len(mask_files),
            "skipped_no_mask": n_skipped_no_mask,
            "skipped_empty_mask": n_skipped_empty_mask,
            "unparsed_code": n_unparsed_code,
            "per_defect_code": per_code_counts,
        },
        "defect_codes": defect_codes,
        "layout": {
            "train_good": "train/good",
            "test_defect_codes": [f"test/{c}" for c in defect_codes],
            "ground_truth": "ground_truth/{ddd}/{stem}_mask.png",
        },
        "defects": manifest_defects,
    }
    (out / "aitex_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    print(f"[prepare_aitex] normal={len(nodefect_files)}  "
          f"defect_matched={len(manifest_defects)}  "
          f"per_defect_code={per_code_counts}")
    print(f"[prepare_aitex] skipped_no_mask={n_skipped_no_mask}  "
          f"skipped_empty_mask={n_skipped_empty_mask}  "
          f"unparsed_code={n_unparsed_code}")
    if skipped_examples:
        print(f"[prepare_aitex] skipped examples: {skipped_examples}")
    print(f"[prepare_aitex] defect_codes(defect_type)={defect_codes}")
    print(f"[prepare_aitex] wrote layout + masks + manifest → {out}")
    return manifest


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="AITEX Fabric Image Database → AROMA (MVTec-style) layout"
    )
    p.add_argument("--defect_images", required=True,
                   help="AITEX Defect_images/ directory (nnnn_ddd_ff.png)")
    p.add_argument("--mask_images", required=True,
                   help="AITEX Mask_images/ directory (nnnn_ddd_ff_mask.png)")
    p.add_argument("--nodefect_images", required=True,
                   help="AITEX NODefect_images/ directory (nnnn_000_ff.png)")
    p.add_argument("--output_dir", required=True,
                   help="Output root, e.g. /content/drive/MyDrive/data/Aroma/aitex")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = _parse_args(argv)
    prepare(args.defect_images, args.mask_images, args.nodefect_images, args.output_dir)
    print("Done.")


if __name__ == "__main__":
    main()
