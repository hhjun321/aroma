#!/usr/bin/env python3
"""
CASDA roi_metadata.csv → AROMA roi_selected.json adapter.

Converts CASDA Stage A ROI extraction output to the format expected by
AROMA's generate_defects.run() for fair copy-paste synthesis comparison.

Field mapping:
    roi_image_path   → image_path    (defect crop path, copy-paste source)
    class_id (int)   → cluster_id    (Severstal defect class 1-4)
    background_type  → cell_key      (context context label: smooth/vertical_stripe/...)
    suitability_score → roi_score    (CASDA's composite quality score)
    n/a              → deficit=0.0   (CASDA has no deficit concept)

Usage (CLI):
    python casda_roi_adapter.py \
        --metadata_csv  /path/to/roi_metadata.csv \
        --output_dir    /path/to/casda_roi/severstal \
        --min_suitability 0.5

Usage (import):
    from casda_roi_adapter import adapt
    rois = adapt(metadata_csv=..., output_dir=..., min_suitability=0.5)
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("aroma.casda_adapter")

# ---------------------------------------------------------------------------
# I/O bootstrap (same pattern as sibling scripts)
# ---------------------------------------------------------------------------

def _bootstrap_aroma_ref() -> None:
    ref = os.environ.get("AROMA_REF") or r"D:\project\aroma"
    ref_path = Path(ref)
    if ref_path.is_dir() and str(ref_path) not in sys.path:
        sys.path.insert(0, str(ref_path))


_bootstrap_aroma_ref()

try:
    from utils.io import save_json  # type: ignore[import]
except Exception:
    def save_json(data: Any, path: str) -> None:  # type: ignore[misc]
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

# Guarded PIL + numpy import (mirror generate_defects' HAS_PIL pattern).
# Used to derive a crop-relative defect_bbox from the nonzero region of the
# crop-aligned mask. If unavailable, we fall back to the CSV subtract-origin
# path (see _csv_fallback_bbox).
try:
    import numpy as np  # type: ignore[import]
    from PIL import Image  # type: ignore[import]
    HAS_PIL = True
except Exception:
    HAS_PIL = False


MIN_SUITABILITY = 0.5

# CASDA background_type values → used as cell_key
VALID_BACKGROUND_TYPES = {
    "smooth",
    "vertical_stripe",
    "horizontal_stripe",
    "complex_pattern",
    "unknown",
}


# ---------------------------------------------------------------------------
# defect_bbox computation
# ---------------------------------------------------------------------------
#
# generate_defects.py use_real_mask path (L350-384) requires:
#   - defect_mask_path: SAME SIZE as image_path; both cropped with the identical
#     box=(bx, by, bx+bw, by+bh). No independent resize -> size mismatch silently
#     corrupts the mask crop.
#   - defect_bbox: [x, y, w, h] (4 ints, top-left + width/height), CROP-RELATIVE.
#     It is REQUIRED (len==4) and bounds-checked against the defect IMAGE size
#     (bw>0, bh>0, bx>=0, by>=0, bx+bw<=dw, by+bh<=dh). Invalid -> ellipse fallback.
#
# For CASDA, roi_image_path is already a CROP and roi_mask_path is crop-aligned,
# so the shared coordinate system IS the crop. We derive defect_bbox from the
# nonzero region of the mask (most robust; auto-clamped inside the crop). We do
# NOT use the CSV defect_bbox column directly: it is in ORIGINAL 1600x256 coords
# and "(x1,y1,x2,y2)" format -> wrong space AND wrong format -> would fail the
# crop bounds check and silently ellipse-fall-back.


def _parse_xyxy(s: str) -> Optional[List[int]]:
    """Parse a "(x1, y1, x2, y2)" / "x1,y1,x2,y2" string into [x1,y1,x2,y2] ints.

    Returns None if the string is missing or not parseable into 4 ints.
    """
    if not s:
        return None
    cleaned = s.strip().strip("()[]").replace(" ", "")
    if not cleaned:
        return None
    parts = cleaned.split(",")
    if len(parts) != 4:
        return None
    try:
        return [int(round(float(p))) for p in parts]
    except (ValueError, TypeError):
        return None


def _mask_derived_bbox(mask_path: str) -> Optional[List[int]]:
    """Compute crop-relative [x, y, w, h] from the nonzero region of the mask.

    Returns None if PIL/numpy unavailable, the mask cannot be read, or the mask
    is empty (all-zero). An empty mask must NOT yield a zero-size bbox (it would
    be rejected by generate_defects and silently ellipse-fall-back).
    """
    if not HAS_PIL:
        return None
    try:
        m = np.asarray(Image.open(mask_path).convert("L"))
    except Exception as e:  # noqa: BLE001
        logger.warning("Failed to read mask %s: %s", mask_path, e)
        return None
    ys, xs = np.nonzero(m >= 128)
    if xs.size == 0 or ys.size == 0:
        return None  # empty mask -> caller falls back to ellipse
    x = int(xs.min())
    y = int(ys.min())
    w = int(xs.max() - xs.min() + 1)
    h = int(ys.max() - ys.min() + 1)
    # Defensive clamp to the mask dimensions (np indices are already in-bounds,
    # but keep w/h from ever exceeding the crop).
    mh, mw = m.shape[:2]
    w = min(w, mw - x)
    h = min(h, mh - y)
    if w <= 0 or h <= 0:
        return None
    return [x, y, w, h]


def _csv_fallback_bbox(
    defect_xyxy: Optional[List[int]],
    roi_xyxy: Optional[List[int]],
    crop_size: Optional[tuple],
) -> Optional[List[int]]:
    """Fallback when PIL/numpy is unavailable: CSV defect_bbox minus roi origin.

    CSV defect_bbox and roi_bbox are ORIGINAL-image "(x1,y1,x2,y2)". Subtract the
    roi_bbox origin to make it crop-relative, convert to [x,y,w,h], and clamp to
    the crop size. Returns None if inputs are insufficient or result is degenerate.
    """
    if defect_xyxy is None or roi_xyxy is None:
        return None
    dx1, dy1, dx2, dy2 = defect_xyxy
    ox, oy = roi_xyxy[0], roi_xyxy[1]
    x = dx1 - ox
    y = dy1 - oy
    w = dx2 - dx1
    h = dy2 - dy1
    # Clamp top-left to >= 0 (and shrink w/h accordingly).
    if x < 0:
        w += x
        x = 0
    if y < 0:
        h += y
        y = 0
    if crop_size is not None:
        cw, ch = crop_size
        w = min(w, cw - x)
        h = min(h, ch - y)
    if w <= 0 or h <= 0:
        return None
    return [int(x), int(y), int(w), int(h)]


# ---------------------------------------------------------------------------
# Core adapter
# ---------------------------------------------------------------------------

def adapt(
    metadata_csv: str,
    output_dir: str,
    min_suitability: float = MIN_SUITABILITY,
    per_class_cap: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Convert CASDA roi_metadata.csv to AROMA roi_selected.json.

    Args:
        metadata_csv:     Path to CASDA Stage A roi_metadata.csv.
        output_dir:       Directory to write roi_selected.json.
        min_suitability:  Minimum suitability_score threshold (CASDA default 0.5).
        per_class_cap:    Maximum ROIs per class_id (None = no cap).

    Returns:
        List of ROI dicts written to roi_selected.json.

    Raises:
        FileNotFoundError: metadata_csv does not exist.
        RuntimeError:      No valid ROIs pass the suitability filter.
    """
    csv_path = Path(metadata_csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"roi_metadata.csv not found: {metadata_csv}")

    rois: List[Dict[str, Any]] = []
    class_counts: Dict[int, int] = {}
    n_total = 0
    n_filtered_suitability = 0
    n_filtered_cap = 0
    n_missing_image = 0
    n_missing_mask = 0       # mask path empty / not on disk
    n_empty_mask = 0         # mask exists but all-zero
    n_size_mismatch = 0      # crop vs mask dimension mismatch
    n_with_mask = 0          # ROIs emitting defect_mask_path + defect_bbox
    n_csv_fallback = 0       # bbox derived via CSV subtract-origin path

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            n_total += 1

            # Suitability filter
            try:
                suitability = float(row["suitability_score"])
            except (KeyError, ValueError):
                n_filtered_suitability += 1
                continue
            if suitability < min_suitability:
                n_filtered_suitability += 1
                continue

            # Per-class cap
            try:
                class_id = int(row["class_id"])
            except (KeyError, ValueError):
                logger.warning("Invalid class_id in row: %s", row.get("image_id", "?"))
                continue

            if per_class_cap is not None:
                count = class_counts.get(class_id, 0)
                if count >= per_class_cap:
                    n_filtered_cap += 1
                    continue
                class_counts[class_id] = count + 1

            # Validate roi_image_path
            roi_image_path = row.get("roi_image_path", "").strip()
            if not roi_image_path:
                logger.warning("Empty roi_image_path for %s", row.get("image_id", "?"))
                n_missing_image += 1
                continue
            if not Path(roi_image_path).exists():
                logger.warning("roi_image_path not found: %s", roi_image_path)
                n_missing_image += 1
                continue

            # background_type → cell_key
            bg_type = row.get("background_type", "unknown").strip() or "unknown"
            if bg_type not in VALID_BACKGROUND_TYPES:
                logger.warning(
                    "Unexpected background_type '%s' for %s — using as-is",
                    bg_type, row.get("image_id", "?"),
                )

            # ----------------------------------------------------------------
            # Resolve defect_mask_path + defect_bbox so generate_defects'
            # use_real_mask path triggers for CASDA (identical synthesis to
            # random/aroma). On any failure we OMIT both fields for this ROI ->
            # generate_defects falls back to the synthetic ellipse mask (the row
            # is still usable, just not pixel-precise).
            # ----------------------------------------------------------------
            defect_mask_path: Optional[str] = None
            defect_bbox: Optional[List[int]] = None

            mask_path = row.get("roi_mask_path", "").strip()
            if not mask_path:
                logger.warning("Empty roi_mask_path for %s — ellipse fallback",
                               row.get("image_id", "?"))
                n_missing_mask += 1
            elif not Path(mask_path).exists():
                logger.warning("roi_mask_path not found: %s — ellipse fallback",
                               mask_path)
                n_missing_mask += 1
            else:
                # Enforce equal dimensions: generate_defects crops image and mask
                # with the SAME box and never resizes the mask. A mismatch would
                # silently misalign the mask crop, so drop the mask on mismatch.
                crop_size: Optional[tuple] = None
                size_ok = True
                if HAS_PIL:
                    try:
                        img_size = Image.open(roi_image_path).size
                        msk_size = Image.open(mask_path).size
                        crop_size = img_size
                        if img_size != msk_size:
                            logger.warning(
                                "crop/mask size mismatch for %s: image=%s mask=%s "
                                "— ellipse fallback",
                                row.get("image_id", "?"), img_size, msk_size,
                            )
                            n_size_mismatch += 1
                            size_ok = False
                    except Exception as e:  # noqa: BLE001
                        logger.warning("Failed to read image/mask size for %s: %s",
                                       row.get("image_id", "?"), e)
                        size_ok = False

                if size_ok:
                    bbox = _mask_derived_bbox(mask_path)
                    if bbox is None and HAS_PIL:
                        # PIL present but mask empty (all-zero) -> ellipse fallback.
                        logger.warning("Empty mask (all-zero) for %s — ellipse fallback",
                                       row.get("image_id", "?"))
                        n_empty_mask += 1
                    elif bbox is None and not HAS_PIL:
                        # No PIL/numpy: derive from CSV subtract-origin instead.
                        roi_xyxy = _parse_xyxy(row.get("roi_bbox", ""))
                        defect_xyxy = _parse_xyxy(row.get("defect_bbox", ""))
                        bbox = _csv_fallback_bbox(defect_xyxy, roi_xyxy, crop_size)
                        if bbox is not None:
                            n_csv_fallback += 1
                            logger.debug("CSV-fallback bbox for %s: %s",
                                         row.get("image_id", "?"), bbox)

                    if bbox is not None:
                        defect_mask_path = mask_path
                        defect_bbox = bbox
                        n_with_mask += 1

            roi_dict: Dict[str, Any] = {
                "image_id":    row.get("image_id", ""),
                "image_path":  roi_image_path,
                "cluster_id":  class_id,
                "cell_key":    bg_type,
                "roi_score":   round(suitability, 6),
                "morph_prior": 0.0,
                "ctx_prior":   0.0,
                "deficit":     0.0,
                "prompt":      row.get("prompt", ""),
                "morph_label": str(class_id),
                "ctx_label":   bg_type,
            }
            # Only emit mask fields when both are valid, so a partial/failed
            # resolution cleanly triggers generate_defects' ellipse fallback.
            if defect_mask_path is not None and defect_bbox is not None:
                roi_dict["defect_mask_path"] = defect_mask_path
                roi_dict["defect_bbox"] = defect_bbox

            rois.append(roi_dict)

    logger.info(
        "casda_roi_adapter: %d total → %d suitability-filtered, %d cap-filtered, "
        "%d missing-image → %d valid ROIs",
        n_total, n_filtered_suitability, n_filtered_cap, n_missing_image, len(rois),
    )
    logger.info(
        "casda_roi_adapter mask resolution: %d with real mask (%d via CSV-fallback), "
        "%d missing-mask, %d empty-mask, %d size-mismatch → %d will ellipse-fallback",
        n_with_mask, n_csv_fallback, n_missing_mask, n_empty_mask, n_size_mismatch,
        len(rois) - n_with_mask,
    )
    if not HAS_PIL:
        logger.warning(
            "PIL/numpy unavailable — used CSV subtract-origin bbox fallback. "
            "Install pillow + numpy for robust mask-derived bboxes."
        )

    if not rois:
        raise RuntimeError(
            f"No valid ROIs after filtering (min_suitability={min_suitability}). "
            f"Check roi_metadata.csv path and suitability scores."
        )

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "roi_selected.json"
    save_json(rois, str(out_path))

    logger.info("Wrote %d CASDA ROIs → %s", len(rois), out_path)
    return rois


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert CASDA roi_metadata.csv → AROMA roi_selected.json"
    )
    p.add_argument("--metadata_csv",    required=True,
                   help="Path to CASDA Stage A roi_metadata.csv")
    p.add_argument("--output_dir",      required=True,
                   help="Output directory for roi_selected.json")
    p.add_argument("--min_suitability", type=float, default=MIN_SUITABILITY,
                   help=f"Minimum suitability_score (default {MIN_SUITABILITY})")
    p.add_argument("--per_class_cap",   type=int, default=None,
                   help="Max ROIs per class_id (default: no cap)")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = _parse_args(argv)
    try:
        rois = adapt(
            metadata_csv=args.metadata_csv,
            output_dir=args.output_dir,
            min_suitability=args.min_suitability,
            per_class_cap=args.per_class_cap,
        )
        print(f"Adapted {len(rois)} CASDA ROIs → {args.output_dir}/roi_selected.json")
    except (FileNotFoundError, RuntimeError) as e:
        logger.error("%s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
