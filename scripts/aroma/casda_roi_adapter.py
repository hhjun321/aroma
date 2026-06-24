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
import shutil
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


def _bbox_from_mask_array(m: "np.ndarray") -> Optional[List[int]]:
    """Compute crop-relative [x, y, w, h] from the nonzero region of a decoded mask.

    Takes a PRE-DECODED grayscale mask array (caller decodes exactly once). Returns
    None if the mask is empty (all-zero). An empty mask must NOT yield a zero-size
    bbox (it would be rejected by generate_defects and silently ellipse-fall-back).
    """
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
# Local staging helpers (Colab Drive FUSE latency mitigation)
# ---------------------------------------------------------------------------
#
# The adapter touches every ROI crop + mask exactly once (a .size header read +
# a single full mask decode per row). On Colab, Drive is mounted via FUSE with
# high per-file latency, so thousands of random per-file reads dominate runtime.
# A single bulk copytree of the images/ + masks/ dirs is a sequential read that
# is dramatically cheaper, after which all reads hit local /content disk.

def _adapter_staging_dir(metadata_csv: str) -> Path:
    """Return a local staging path. Uses /content on Colab, sibling dir otherwise."""
    slug = Path(metadata_csv).parent.name
    colab_tmp = Path("/content/tmp")
    if colab_tmp.parent.exists():
        return colab_tmp / f"aroma_casda_roi_{slug}"
    return Path(metadata_csv).parent / f"_staging_tmp_{slug}"


def _stage_roi_dirs(drive_root: Path, staging_dir: Path) -> Path:
    """Bulk-copy drive_root/{images,masks} → staging_dir/{images,masks} once.

    Returns staging_dir (the new local_root). dirs_exist_ok=True so Colab --resume
    re-runs in the same session don't crash on FileExistsError. A single copytree
    is the sequential bulk read that beats thousands of random per-file Drive reads.
    """
    for sub in ("images", "masks"):
        src = drive_root / sub
        dst = staging_dir / sub
        try:
            shutil.copytree(src, dst, dirs_exist_ok=True)
            n = sum(1 for _ in dst.iterdir()) if dst.exists() else 0
            logger.info("Staged %s → %s (%d entries)", src, dst, n)
        except Exception as e:  # noqa: BLE001
            logger.warning("Failed to stage %s → %s: %s", src, dst, e)
    return staging_dir


def _remap(p: str, drive_root: Path, local_root: Path) -> str:
    """Remap an absolute Drive path under drive_root to local_root, else unchanged."""
    if not p:
        return p
    try:
        return str(local_root / Path(p).relative_to(drive_root))
    except ValueError:
        return p


# ---------------------------------------------------------------------------
# Core adapter
# ---------------------------------------------------------------------------

def adapt(
    metadata_csv: str,
    output_dir: str,
    min_suitability: float = MIN_SUITABILITY,
    per_class_cap: Optional[int] = None,
    local_staging: bool = False,
    staging_dir: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Convert CASDA roi_metadata.csv to AROMA roi_selected.json.

    Args:
        metadata_csv:     Path to CASDA Stage A roi_metadata.csv.
        output_dir:       Directory to write roi_selected.json.
        min_suitability:  Minimum suitability_score threshold (CASDA default 0.5).
        per_class_cap:    Maximum ROIs per class_id (None = no cap).
        local_staging:    Bulk-copy the referenced ROI images/ + masks/ dirs from
                          Drive to local /content disk ONCE, then read crops/masks
                          locally (Colab Drive FUSE latency mitigation). Default
                          False = byte-identical legacy behavior reading from Drive.
        staging_dir:      Override the local staging directory (default: derived
                          from metadata_csv parent name under /content/tmp).

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

    # Read all metadata rows up front (text-only, small) so we can derive the
    # Drive ROI root from the first usable row before optionally bulk-staging.
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    # ----------------------------------------------------------------------
    # Optional local staging (A): bulk-copy the ROI images/ + masks/ dirs from
    # Drive to local disk ONCE, then remap every absolute path to the local
    # copy. Graceful degrade: any failure falls back to identity (read Drive).
    # ----------------------------------------------------------------------
    remap = lambda p: p  # identity by default  # noqa: E731
    if local_staging:
        drive_root: Optional[Path] = None
        for r in rows:
            img = r.get("roi_image_path", "").strip()
            msk = r.get("roi_mask_path", "").strip()
            if img and msk:
                try:
                    drive_root = Path(
                        os.path.commonpath([str(Path(img).parent), str(Path(msk).parent)])
                    )
                except ValueError:
                    drive_root = None
                break
        if drive_root is None:
            logger.warning(
                "local_staging requested but no usable row with both "
                "roi_image_path and roi_mask_path — reading from Drive."
            )
        else:
            local_root = Path(staging_dir) if staging_dir else _adapter_staging_dir(metadata_csv)
            try:
                local_root = _stage_roi_dirs(drive_root, local_root)
                remap = lambda p, _dr=drive_root, _lr=local_root: _remap(p, _dr, _lr)  # noqa: E731
                logger.info(
                    "local_staging active: Drive root %s → local %s", drive_root, local_root
                )
            except Exception as e:  # noqa: BLE001
                logger.warning("Staging failed (%s) — reading from Drive.", e)

    # NOTE: loop body kept one indent level deep (under this guard) to minimize
    # the diff vs. the former `with open(...)` block. The guard is always true;
    # rows were fully read above so the file handle is already closed.
    if True:  # noqa: SIM103 — preserves block indentation, intentional
        for row in rows:
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

            # Validate roi_image_path (remap to local staging copy if active).
            roi_image_path = remap(row.get("roi_image_path", "").strip())
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

            mask_path = remap(row.get("roi_mask_path", "").strip())
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
                #
                # I/O collapse: decode the mask EXACTLY ONCE (single np.asarray)
                # and derive BOTH the size-check and the bbox from that one array.
                # The crop is touched only via a lazy .size header read. Net heavy
                # Drive ops per row drop from ~5 to ~2 (crop .size + one mask decode).
                crop_size: Optional[tuple] = None
                size_ok = True
                m = None  # decoded mask array (set below when HAS_PIL)
                if HAS_PIL:
                    try:
                        img_size = Image.open(roi_image_path).size           # crop dims (lazy header)
                        m = np.asarray(Image.open(mask_path).convert("L"))   # mask: single full decode
                        msk_h, msk_w = m.shape[:2]
                        msk_size = (msk_w, msk_h)
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
                        logger.warning("Failed to read image/mask for %s: %s",
                                       row.get("image_id", "?"), e)
                        size_ok = False

                if size_ok:
                    bbox = _bbox_from_mask_array(m) if (HAS_PIL and m is not None) else None
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
    p.add_argument("--local_staging",   action="store_true",
                   help="Bulk-copy ROI images/+masks/ dirs to /content once, then "
                        "read crops/masks locally (faster on Colab with Drive)")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = _parse_args(argv)
    try:
        rois = adapt(
            metadata_csv=args.metadata_csv,
            output_dir=args.output_dir,
            min_suitability=args.min_suitability,
            per_class_cap=args.per_class_cap,
            local_staging=args.local_staging,
        )
        print(f"Adapted {len(rois)} CASDA ROIs → {args.output_dir}/roi_selected.json")
    except (FileNotFoundError, RuntimeError) as e:
        logger.error("%s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
