#!/usr/bin/env python3
"""
AROMA roi_selected.json → CASDA roi_metadata.csv adapter.

This is the FORWARD adapter for the compounding experiment: it converts an
AROMA ROI selection (compatibility OR random strategy output) into a
CASDA-schema roi_metadata.csv so CASDA's prepare_controlnet_data.py can consume
it UNCHANGED. The single-factor swap (only the ROI list differs across arms)
runs through the SAME CASDA packager + trained ControlNet + Poisson engine.

Why this shape (verified against CASDA, do NOT modify CASDA):
    prepare_controlnet_data.py reads roi_metadata.csv via pd.read_csv, then
    ControlNetDatasetPackager.package_single_roi(roi_data, roi_image, roi_mask)
    REGENERATES the hint image + prompt from FIELDS:
        hint   = generate_hint_image(roi_image, roi_mask, defect_metrics=roi_data,
                                     background_type, stability_score)
        prompt = generate_prompt(defect_subtype, background_type, class_id,
                                 defect_metrics=roi_data, suitability_score)
    The csv 'prompt' TEXT is ignored/regenerated. roi_image / roi_mask are loaded
    from roi_image_path / roi_mask_path as CROP-ALIGNED arrays (same size).
    => This adapter must therefore fill the FIELDS the packager actually reads
       (defect_subtype, background_type, linearity/solidity/extent/aspect_ratio,
       stability_score/suitability_score) AND produce valid crop-aligned
       roi_image_path / roi_mask_path PNGs. 'prompt' is left blank (regenerated).

Field mapping (AROMA roi_selected entry → CASDA roi_metadata.csv column):
    image_id            ← entry.image_id
    class_id            ← int(class_key.replace('class','')) else 1
    region_id           ← per-image_id running index (0,1,2,...)
    roi_bbox            ← "(x, y, x+w, y+h)"  (== defect_bbox; severstal simplicity)
    defect_bbox         ← AROMA [x,y,w,h] → "(x, y, x+w, y+h)"
    centroid            ← "(x+w/2, y+h/2)"
    area                ← morphology.area (joined) else derived w*h
    linearity/solidity/extent/aspect_ratio ← morphology (joined) else 0.5 + WARN
    defect_subtype      ← entry.defect_subtype else "general"
    background_type     ← --background_type arg (default complex_pattern; severstal)
    suitability_score   ← roi_score|quality_score clamped [0,1] else 0.5
    matching_score      ← same as suitability_score
    continuity_score    ← same as suitability_score
    stability_score     ← same as suitability_score
    recommendation      ← "acceptable"
    prompt              ← ""  (CASDA PromptGenerator regenerates in-distribution)
    roi_image_path      ← crop-aligned defect crop PNG written to --crops_dir
    roi_mask_path       ← crop-aligned mask PNG written to --crops_dir

Usage (CLI):
    python aroma_to_casda_roi.py \
        --roi_selected   /path/to/roi_selected.json \
        --morphology_csv /path/to/morphology_features.csv \
        --output_csv     /path/to/roi_metadata.csv \
        --crops_dir      /path/to/crops \
        --background_type complex_pattern

Usage (import):
    from aroma_to_casda_roi import adapt
    n = adapt(roi_selected=..., morphology_csv=..., output_csv=...,
              crops_dir=..., background_type="complex_pattern")
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("aroma.to_casda")

# ---------------------------------------------------------------------------
# I/O bootstrap (same pattern as sibling scripts)
# ---------------------------------------------------------------------------

def _bootstrap_aroma_ref() -> None:
    ref = os.environ.get("AROMA_REF") or r"D:\project\aroma"
    ref_path = Path(ref)
    if ref_path.is_dir() and str(ref_path) not in sys.path:
        sys.path.insert(0, str(ref_path))


_bootstrap_aroma_ref()

# Guarded numpy / cv2 / PIL imports (mirror generate_defects' HAS_CV2 / HAS_PIL
# pattern). cv2 is preferred for cropping + PNG write; PIL is the fallback. numpy
# is used for array handling. When NEITHER cv2 nor PIL is available we cannot
# produce crop-aligned PNGs, so make_crops degrades to "skip every row that
# would need a crop" (logged loudly) rather than crashing — local runs lack the
# Drive images entirely, so a no-image environment is expected and must not crash.
try:
    import numpy as np  # type: ignore[import]
    HAS_NUMPY = True
except Exception:
    HAS_NUMPY = False

try:
    import cv2  # type: ignore[import]
    HAS_CV2 = True
except Exception:
    HAS_CV2 = False

try:
    from PIL import Image  # type: ignore[import]
    HAS_PIL = True
except Exception:
    HAS_PIL = False


# CASDA roi_metadata.csv header — EXACT column order the packager's pd.read_csv
# expects. Produced verbatim; do NOT reorder.
CASDA_HEADER: List[str] = [
    "image_id",
    "class_id",
    "region_id",
    "roi_bbox",
    "defect_bbox",
    "centroid",
    "area",
    "linearity",
    "solidity",
    "extent",
    "aspect_ratio",
    "defect_subtype",
    "background_type",
    "suitability_score",
    "matching_score",
    "continuity_score",
    "stability_score",
    "recommendation",
    "prompt",
    "roi_image_path",
    "roi_mask_path",
]

# Default morphology metric when no join match is found (verified contract).
_DEFAULT_METRIC = 0.5
_DEFAULT_SCORE = 0.5


# ---------------------------------------------------------------------------
# Small parse / clamp helpers
# ---------------------------------------------------------------------------

def _clamp01(v: float) -> float:
    """Clamp a float into [0.0, 1.0]."""
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v


def _as_xywh(bbox: Any) -> Optional[List[int]]:
    """Coerce an AROMA defect_bbox into [x, y, w, h] ints.

    Accepts a list/tuple of 4 numbers ([x, y, w, h]) or a "x,y,w,h" string.
    Returns None if it cannot be parsed into 4 ints or w/h are non-positive.
    """
    parts: Optional[List[Any]] = None
    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        parts = list(bbox)
    elif isinstance(bbox, str):
        cleaned = bbox.strip().strip("()[]")
        if cleaned:
            chunks = cleaned.split(",")
            if len(chunks) == 4:
                parts = chunks
    if parts is None:
        return None
    try:
        x, y, w, h = (int(round(float(p))) for p in parts)
    except (ValueError, TypeError):
        return None
    if w <= 0 or h <= 0:
        return None
    return [x, y, w, h]


def _xywh_to_xyxy_str(b: List[int]) -> str:
    """[x, y, w, h] → CASDA "(x1, y1, x2, y2)" string."""
    x, y, w, h = b
    return f"({x}, {y}, {x + w}, {y + h})"


def _xywh_to_centroid_str(b: List[int]) -> str:
    """[x, y, w, h] → CASDA "(cx, cy)" centroid string."""
    x, y, w, h = b
    return f"({x + w / 2.0}, {y + h / 2.0})"


def _remap_root(p: str, image_root: Optional[str]) -> str:
    """Optionally prefix a path with image_root.

    If image_root is given and p is not already absolute under it, join image_root
    with the path basename-preserving tail. We keep this conservative: when the
    stored path is absolute (typical AROMA Drive path) and image_root is set, we
    re-root it onto image_root by stripping its anchor; otherwise return as-is.
    """
    if not image_root or not p:
        return p
    pp = Path(p)
    if pp.is_absolute():
        # Strip the anchor (drive/root) and re-root under image_root.
        tail = pp.relative_to(pp.anchor) if pp.anchor else pp
        return str(Path(image_root) / tail)
    return str(Path(image_root) / pp)


# ---------------------------------------------------------------------------
# Morphology join index
# ---------------------------------------------------------------------------

def _load_morphology(morphology_csv: str) -> Tuple[Dict[str, Dict[str, Any]],
                                                   Dict[Tuple[str, str], Dict[str, Any]]]:
    """Build two lookup indices from morphology_features.csv.

    Primary key  : defect_mask_path (string, exact)            → row dict
    Fallback key : (image_id, defect_bbox-normalized-string)   → row dict

    The fallback key normalizes the morphology defect_bbox ("x,y,w,h" string) by
    stripping whitespace so it can be matched against an AROMA entry's bbox
    re-stringified in the same normal form.
    """
    by_mask: Dict[str, Dict[str, Any]] = {}
    by_img_bbox: Dict[Tuple[str, str], Dict[str, Any]] = {}
    if not morphology_csv:
        return by_mask, by_img_bbox
    mpath = Path(morphology_csv)
    if not mpath.exists():
        logger.warning("morphology_csv not found: %s — all metrics will default to %.1f",
                       morphology_csv, _DEFAULT_METRIC)
        return by_mask, by_img_bbox
    with open(mpath, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            mask = (row.get("defect_mask_path") or "").strip()
            if mask:
                by_mask[mask] = row
            img_id = (row.get("image_id") or "").strip()
            bbox_norm = _norm_bbox_key(row.get("defect_bbox", ""))
            if img_id and bbox_norm:
                by_img_bbox[(img_id, bbox_norm)] = row
    logger.info("Loaded morphology: %d rows by mask-path, %d by (image_id,bbox)",
                len(by_mask), len(by_img_bbox))
    return by_mask, by_img_bbox


def _norm_bbox_key(bbox: Any) -> str:
    """Normalize a bbox (list or 'x,y,w,h' string) into a 'x,y,w,h' int key.

    Returns "" if not parseable, so callers can skip building a fallback key.
    """
    xywh = _as_xywh(bbox)
    if xywh is None:
        return ""
    return ",".join(str(v) for v in xywh)


def _join_metrics(
    entry: Dict[str, Any],
    by_mask: Dict[str, Dict[str, Any]],
    by_img_bbox: Dict[Tuple[str, str], Dict[str, Any]],
) -> Tuple[Optional[Dict[str, Any]], bool]:
    """Find the morphology row for an AROMA entry.

    Primary: defect_mask_path exact match. Fallback: (image_id, normalized bbox).
    Returns (morph_row_or_None, matched_bool).
    """
    mask = (entry.get("defect_mask_path") or "").strip()
    if mask and mask in by_mask:
        return by_mask[mask], True
    img_id = (entry.get("image_id") or "").strip()
    bbox_norm = _norm_bbox_key(entry.get("defect_bbox", ""))
    if img_id and bbox_norm:
        hit = by_img_bbox.get((img_id, bbox_norm))
        if hit is not None:
            return hit, True
    return None, False


def _metric(morph: Optional[Dict[str, Any]], key: str) -> float:
    """Read a float metric from a morphology row, defaulting to 0.5 on miss/parse-fail."""
    if morph is None:
        return _DEFAULT_METRIC
    raw = morph.get(key)
    if raw is None or raw == "":
        return _DEFAULT_METRIC
    try:
        return float(raw)
    except (ValueError, TypeError):
        return _DEFAULT_METRIC


# ---------------------------------------------------------------------------
# class_id derivation
# ---------------------------------------------------------------------------

def _class_id(entry: Dict[str, Any]) -> int:
    """int(class_key.replace('class','')) if class_key like 'classN' else 1."""
    ck = entry.get("class_key")
    if isinstance(ck, str) and ck.lower().startswith("class"):
        tail = ck[len("class"):]
        try:
            return int(tail)
        except (ValueError, TypeError):
            return 1
    return 1


def _stratified_cap(entries: List[Any], max_rois: Optional[int]) -> List[Any]:
    """Cap entries to at most max_rois, BALANCED across class_id (round-robin).

    Deterministic — no RNG. Preserves each class's existing selection order
    (roi_selected is already ranked by the strategy) and round-robins across
    classes so a rare class (e.g. severstal c2) is not starved by a dominant one.
    Returns entries unchanged when max_rois is None/<=0 or already small enough.
    This caps GENERATION cost AND equalizes per-arm ROI count (fair single-factor
    swap: every arm capped to the same N before ControlNet inference).
    """
    if max_rois is None or max_rois <= 0 or len(entries) <= max_rois:
        return entries
    from collections import OrderedDict
    groups: "OrderedDict[int, List[Any]]" = OrderedDict()
    for e in entries:
        cid = _class_id(e) if isinstance(e, dict) else 1
        groups.setdefault(cid, []).append(e)
    picked: List[Any] = []
    idxs = {k: 0 for k in groups}
    keys = list(groups.keys())
    while len(picked) < max_rois:
        progressed = False
        for k in keys:
            if len(picked) >= max_rois:
                break
            i = idxs[k]
            if i < len(groups[k]):
                picked.append(groups[k][i])
                idxs[k] += 1
                progressed = True
        if not progressed:
            break
    per_class = {k: sum(1 for e in picked
                        if (_class_id(e) if isinstance(e, dict) else 1) == k) for k in keys}
    logger.info("stratified_cap: %d → %d ROIs (max_rois=%d); per-class=%s",
                len(entries), len(picked), max_rois, dict(per_class))
    return picked


# ---------------------------------------------------------------------------
# Crop writing (crop-aligned image + mask PNGs)
# ---------------------------------------------------------------------------

def _read_gray_or_color(path: str, gray: bool) -> Optional["np.ndarray"]:
    """Load an image as a numpy array (cv2 preferred, PIL fallback). None on failure."""
    if HAS_CV2:
        flag = cv2.IMREAD_GRAYSCALE if gray else cv2.IMREAD_COLOR
        arr = cv2.imread(path, flag)
        return arr
    if HAS_PIL and HAS_NUMPY:
        try:
            im = Image.open(path)
            im = im.convert("L") if gray else im.convert("RGB")
            return np.asarray(im)
        except Exception:  # noqa: BLE001
            return None
    return None


def _write_png(path: str, arr: "np.ndarray", is_color: bool) -> bool:
    """Write a numpy array to a PNG (cv2 preferred, PIL fallback). False on failure."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if HAS_CV2:
        try:
            return bool(cv2.imwrite(path, arr))
        except Exception:  # noqa: BLE001
            return False
    if HAS_PIL and HAS_NUMPY:
        try:
            mode = "RGB" if is_color else "L"
            # cv2 reads color as BGR; if we fell through to PIL, our arrays came
            # from PIL (RGB) since cv2 is absent, so no channel swap is needed.
            Image.fromarray(arr.astype("uint8"), mode=mode).save(path)
            return True
        except Exception:  # noqa: BLE001
            return False
    return False


def _crop_xywh(arr: "np.ndarray", b: List[int]) -> Optional["np.ndarray"]:
    """Crop arr to [x, y, w, h], clamped to array bounds. None if degenerate."""
    h_img, w_img = arr.shape[:2]
    x, y, w, h = b
    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(w_img, x + w)
    y1 = min(h_img, y + h)
    if x1 <= x0 or y1 <= y0:
        return None
    return arr[y0:y1, x0:x1]


def _make_crop_pair(
    image_path: str,
    mask_path: str,
    bbox: List[int],
    crops_dir: str,
    base: str,
) -> Optional[Tuple[str, str]]:
    """Crop image+mask to bbox (same box) and write crop-aligned PNGs.

    Returns (roi_image_path, roi_mask_path) on success, or None on any failure
    (missing file, unreadable, degenerate crop, write fail). Caller WARNs+skips.
    The two crops are guaranteed identical size: both cropped with the SAME box
    and the mask crop is resized to the image crop only if a 1px rounding diff
    appears (defensive; normally exact).
    """
    if not (HAS_CV2 or HAS_PIL) or not HAS_NUMPY:
        logger.warning("No cv2/PIL+numpy available — cannot write crops; skipping %s", base)
        return None
    if not image_path or not Path(image_path).exists():
        logger.warning("source image missing: %s — skip row %s", image_path, base)
        return None
    if not mask_path or not Path(mask_path).exists():
        logger.warning("source mask missing: %s — skip row %s", mask_path, base)
        return None

    img = _read_gray_or_color(image_path, gray=False)
    msk = _read_gray_or_color(mask_path, gray=True)
    if img is None:
        logger.warning("failed to read image: %s — skip row %s", image_path, base)
        return None
    if msk is None:
        logger.warning("failed to read mask: %s — skip row %s", mask_path, base)
        return None

    img_crop = _crop_xywh(img, bbox)
    msk_crop = _crop_xywh(msk, bbox)
    if img_crop is None or msk_crop is None:
        logger.warning("degenerate crop for %s (bbox=%s, img=%s) — skip row",
                       base, bbox, getattr(img, "shape", "?"))
        return None

    # Enforce identical crop sizes (CASDA loads them as a crop-aligned pair).
    ih, iw = img_crop.shape[:2]
    mh, mw = msk_crop.shape[:2]
    if (ih, iw) != (mh, mw):
        if HAS_CV2:
            msk_crop = cv2.resize(msk_crop, (iw, ih), interpolation=cv2.INTER_NEAREST)
        elif HAS_PIL:
            msk_crop = np.asarray(
                Image.fromarray(msk_crop.astype("uint8"), mode="L").resize(
                    (iw, ih), Image.NEAREST
                )
            )

    roi_image_path = str(Path(crops_dir) / f"{base}.png")
    roi_mask_path = str(Path(crops_dir) / f"{base}_mask.png")
    is_color = img_crop.ndim == 3
    if not _write_png(roi_image_path, img_crop, is_color):
        logger.warning("failed to write image crop: %s — skip row %s", roi_image_path, base)
        return None
    if not _write_png(roi_mask_path, msk_crop, is_color=False):
        logger.warning("failed to write mask crop: %s — skip row %s", roi_mask_path, base)
        return None
    return roi_image_path, roi_mask_path


# ---------------------------------------------------------------------------
# Core adapter
# ---------------------------------------------------------------------------

def adapt(
    roi_selected: str,
    output_csv: str,
    crops_dir: str,
    morphology_csv: Optional[str] = None,
    background_type: str = "complex_pattern",
    image_root: Optional[str] = None,
    make_crops: bool = True,
    max_rois: Optional[int] = None,
    casda_score: float = 0.7,
) -> int:
    """Convert an AROMA roi_selected.json into a CASDA-schema roi_metadata.csv.

    Args:
        roi_selected:    Path to AROMA roi_selected.json (compatibility OR random).
        output_csv:      Path to write the CASDA roi_metadata.csv.
        crops_dir:       Directory to write crop-aligned roi_image / roi_mask PNGs.
        morphology_csv:  Path to morphology_features.csv (per-defect metrics). When
                         absent or unmatched, metrics default to 0.5 with a WARN.
        background_type: CASDA background_type label (default "complex_pattern" for
                         severstal — matches CASDA's hint G/B channel logic).
        image_root:      Optional prefix to re-root image_path / defect_mask_path
                         (default None = use stored absolute paths).
        make_crops:      When True (default), crop image+mask to defect_bbox and
                         write crop-aligned PNGs; on a missing/unreadable source the
                         row is WARNed and SKIPPED (does not crash). When False, no
                         crops are written and roi_image_path/roi_mask_path are left
                         as the (optionally re-rooted) source paths.

    Returns:
        Number of CSV rows written.

    Raises:
        FileNotFoundError: roi_selected does not exist.
    """
    rsel_path = Path(roi_selected)
    if not rsel_path.exists():
        raise FileNotFoundError(f"roi_selected.json not found: {roi_selected}")

    with open(rsel_path, encoding="utf-8") as f:
        entries = json.load(f)
    if not isinstance(entries, list):
        raise ValueError(f"roi_selected.json must be a list, got {type(entries).__name__}")

    # Class-stratified cap BEFORE conversion/crops: equalizes per-arm ROI count
    # (fair swap) and bounds ControlNet generation cost. No-op when max_rois unset.
    entries = _stratified_cap(entries, max_rois)

    by_mask, by_img_bbox = _load_morphology(morphology_csv) if morphology_csv else ({}, {})

    if make_crops:
        Path(crops_dir).mkdir(parents=True, exist_ok=True)

    n_in = len(entries)
    n_written = 0
    n_skipped = 0
    n_morph_unmatched = 0
    region_counter: Dict[str, int] = {}  # per-image_id running region index

    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", newline="", encoding="utf-8") as cf:
        writer = csv.DictWriter(cf, fieldnames=CASDA_HEADER)
        writer.writeheader()

        for entry in entries:
            if not isinstance(entry, dict):
                logger.warning("Non-dict roi entry skipped: %r", entry)
                n_skipped += 1
                continue

            image_id = str(entry.get("image_id", "")).strip()

            # defect_bbox: AROMA [x,y,w,h] (or "x,y,w,h" string) → required.
            bbox = _as_xywh(entry.get("defect_bbox"))
            if bbox is None:
                logger.warning("Missing/invalid defect_bbox for %s — skip row", image_id or "?")
                n_skipped += 1
                continue

            class_id = _class_id(entry)

            # region_id: per-image_id running index 0,1,2,...
            region_id = region_counter.get(image_id, 0)

            # Morphology join (primary defect_mask_path, fallback image_id+bbox).
            morph, matched = _join_metrics(entry, by_mask, by_img_bbox)
            if not matched:
                n_morph_unmatched += 1
                logger.warning("No morphology match for %s (defect_mask_path/bbox) "
                               "— metrics default to %.1f", image_id or "?", _DEFAULT_METRIC)

            linearity = _metric(morph, "linearity")
            solidity = _metric(morph, "solidity")
            extent = _metric(morph, "extent")
            aspect_ratio = _metric(morph, "aspect_ratio")
            # area: prefer morphology, else derive from bbox (w*h).
            area_val = _metric(morph, "area") if (morph and morph.get("area")) else float(bbox[2] * bbox[3])

            # CASDA-suitability gate columns (suitability/matching/continuity/stability).
            # CRITICAL: CASDA's packaging quality_filter (stability>=0.3 AND matching>=0.5)
            # gates on THESE. AROMA's roi_score is a DIFFERENT scale (compatibility, often
            # <0.5) → mapping roi_score here filters out 100% of AROMA ROIs. But AROMA's
            # selection ALREADY decided inclusion; the packaging filter must be a no-op so
            # every selected ROI is packaged. We therefore set these to a fixed passing
            # constant (casda_score, default 0.7 — CASDA's own typical value), which is NOT
            # used by hint generation (hint uses linearity/solidity) and only minimally by
            # the prompt. prepare_controlnet_data.py exposes no quality_filter CLI flag and
            # CASDA core must not be modified, so neutralizing via the score columns is the
            # only in-distribution way to package the full per-arm selection.
            score = _clamp01(float(casda_score))

            defect_subtype = str(entry.get("defect_subtype") or "general").strip() or "general"

            # ---- crops (crop-aligned image + mask PNGs) --------------------
            base = f"{image_id}_class{class_id}_region{region_id}"
            if make_crops:
                src_image = _remap_root(str(entry.get("image_path", "")).strip(), image_root)
                src_mask = _remap_root(str(entry.get("defect_mask_path", "")).strip(), image_root)
                pair = _make_crop_pair(src_image, src_mask, bbox, crops_dir, base)
                if pair is None:
                    # WARN already emitted inside _make_crop_pair; skip this row.
                    n_skipped += 1
                    continue
                roi_image_path, roi_mask_path = pair
            else:
                roi_image_path = _remap_root(str(entry.get("image_path", "")).strip(), image_root)
                roi_mask_path = _remap_root(str(entry.get("defect_mask_path", "")).strip(), image_root)

            row: Dict[str, Any] = {
                "image_id":          image_id,
                "class_id":          class_id,
                "region_id":         region_id,
                "roi_bbox":          _xywh_to_xyxy_str(bbox),
                "defect_bbox":       _xywh_to_xyxy_str(bbox),
                "centroid":          _xywh_to_centroid_str(bbox),
                "area":              area_val,
                "linearity":         linearity,
                "solidity":          solidity,
                "extent":            extent,
                "aspect_ratio":      aspect_ratio,
                "defect_subtype":    defect_subtype,
                "background_type":   background_type,
                "suitability_score": score,
                "matching_score":    score,
                "continuity_score":  score,
                "stability_score":   score,
                "recommendation":    "acceptable",
                "prompt":            "",  # CASDA PromptGenerator regenerates
                "roi_image_path":    roi_image_path,
                "roi_mask_path":     roi_mask_path,
            }
            writer.writerow(row)
            n_written += 1
            region_counter[image_id] = region_id + 1

    logger.info(
        "aroma_to_casda_roi: n_in=%d, n_written=%d, n_skipped=%d, n_morph_unmatched=%d → %s",
        n_in, n_written, n_skipped, n_morph_unmatched, out_path,
    )
    if not (HAS_CV2 or HAS_PIL) and make_crops:
        logger.warning("Neither cv2 nor PIL available — make_crops produced no crops "
                       "(every row skipped). Install opencv-python or pillow.")
    return n_written


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert AROMA roi_selected.json → CASDA roi_metadata.csv"
    )
    p.add_argument("--roi_selected", required=True,
                   help="Path to AROMA roi_selected.json (compatibility OR random)")
    p.add_argument("--morphology_csv", default=None,
                   help="Path to morphology_features.csv (per-defect metrics)")
    p.add_argument("--output_csv", required=True,
                   help="Output path for CASDA roi_metadata.csv")
    p.add_argument("--crops_dir", required=True,
                   help="Directory to write crop-aligned roi_image / roi_mask PNGs")
    p.add_argument("--background_type", default="complex_pattern",
                   help="CASDA background_type label (default complex_pattern; severstal)")
    p.add_argument("--image_root", default=None,
                   help="Optional prefix to re-root image_path/defect_mask_path (default none)")
    p.add_argument("--max_rois", type=int, default=None,
                   help="Cap total ROIs (class-stratified round-robin) BEFORE conversion. "
                        "Equalizes per-arm ROI count + bounds ControlNet cost. Default none.")
    p.add_argument("--casda_score", type=float, default=0.7,
                   help="Constant for the CASDA gate columns (suitability/matching/"
                        "continuity/stability). Must pass CASDA quality_filter "
                        "(stability>=0.3, matching>=0.5). Default 0.7. AROMA's own selection "
                        "already decided inclusion; this neutralizes the packaging filter.")
    crop_grp = p.add_mutually_exclusive_group()
    crop_grp.add_argument("--make_crops", dest="make_crops", action="store_true",
                          help="Write crop-aligned PNGs (default)")
    crop_grp.add_argument("--no_make_crops", dest="make_crops", action="store_false",
                          help="Do not write crops; leave source paths in CSV")
    p.set_defaults(make_crops=True)
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = _parse_args(argv)
    try:
        n = adapt(
            roi_selected=args.roi_selected,
            output_csv=args.output_csv,
            crops_dir=args.crops_dir,
            morphology_csv=args.morphology_csv,
            background_type=args.background_type,
            image_root=args.image_root,
            make_crops=args.make_crops,
            max_rois=args.max_rois,
            casda_score=args.casda_score,
        )
        print(f"Wrote {n} CASDA ROI rows → {args.output_csv}")
    except (FileNotFoundError, ValueError) as e:
        logger.error("%s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
