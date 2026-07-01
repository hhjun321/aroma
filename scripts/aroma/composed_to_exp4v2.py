#!/usr/bin/env python3
"""
CASDA composed output → exp4v2 synth annotations.json converter.

This is the scriptified version of the compounding-guide §3d inline cell, with
one CRITICAL added guarantee: **entries exp4v2 cannot label are SKIPPED**.

Why (dev_note ⑧ 결함2 — parity 붕괴):
    CASDA `compose_casda_images.py` writes `{composed_dir}/images/*.png`,
    `{composed_dir}/masks/{name}` (full-frame masks) and a metadata.json. Many
    ControlNet-generated defects come out blank OR TINY, so their composed mask
    yields no bbox with area >= exp4v2's min_area (50). exp4v2's mask→bbox step
    (`_mask_to_bboxes`) extracts no bbox from such a mask, so those synth images
    silently DROP during labeling. The compounding run showed the casda arm =
    772/1013 labeled (241 dropped: 495/1939 masks were nonzero-but-tiny <50px)
    vs random/aroma = 1013/1013, breaking budget parity and confounding the
    aroma-vs-casda comparison. A naive ">=1 nonzero pixel" check does NOT catch
    the tiny masks (they have nonzero pixels). This converter instead applies the
    SAME bbox-area>=min_area criterion as exp4v2 BEFORE recording the entry, so
    exp4v2 samples from a labelable-only pool and every sampled image is labeled.

Output schema (matches exp4v2 `_load_synth_annotations`, verified):
    A list of dicts, each:
        image_path : absolute path to composed image
        mask_path  : absolute path to full-frame mask (exp4v2 derives bbox)
        source_roi : image file name (contains `_class{N}_` → class parse)
        cluster_id : Severstal class 1-4 (from `_class{N}` in name, fallback 1)
        dry_run    : False
    Written to `{output_dir}/{dataset_key}/annotations.json`.

Skip precedence (deterministic):
    - mask file missing            → SKIP, n_skipped_no_maskfile++
    - cv2/PIL available and mask yields no bbox with area >= _MIN_BBOX_AREA (50)
      → SKIP, n_skipped_no_bbox++  (matches exp4v2 `_mask_to_bboxes` min_area)
    - neither cv2 nor PIL available → cannot inspect pixels; pass on mask-EXISTS
      only (WARN once) so a no-image local env does not crash.

Usage (CLI):
    python composed_to_exp4v2.py \
        --composed_dir /path/to/compose_output \
        --output_dir   /path/to/synth_root \
        --dataset_key  severstal

Usage (import):
    from composed_to_exp4v2 import convert, adapt
    n_valid = convert(composed_dir=..., output_dir=..., dataset_key="severstal")
"""
from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("aroma.composed_to_exp4v2")

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
# and the sibling aroma_to_casda_roi adapter). cv2 is preferred for reading masks
# and counting nonzero pixels; PIL+numpy is the fallback. When NEITHER is present
# we cannot inspect mask pixels, so empty-mask detection degrades to "keep any
# mask file that EXISTS" (logged once) rather than crashing — local runs lack the
# Drive masks entirely, so a no-image environment is expected and must not crash.
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


# Regex for the `_class{N}` label CASDA embeds in composed file names. Matched
# once and reused (deterministic).
_CLASS_RE = re.compile(r"_class(\d+)")


# Minimum defect-bbox area (px) for a composed mask to count as labelable.
# MUST match exp4v2 `_mask_to_bboxes(min_area=...)` (default 50): a mask with
# nonzero pixels but whose every connected component's bbox is < this area yields
# NO bbox in exp4v2 and is silently dropped at labeling — breaking budget parity
# (the CASDA ControlNet arm produced ~25% such tiny-defect masks: 495/1939).
# Filtering them HERE makes annotations.json == exactly what exp4v2 will label.
_MIN_BBOX_AREA = 50


# ---------------------------------------------------------------------------
# Mask inspection
# ---------------------------------------------------------------------------

def _mask_has_valid_bbox(mask_path: str, min_area: int = _MIN_BBOX_AREA) -> Optional[bool]:
    """Return True if the mask yields a bbox with area >= min_area, else False.

    Mirrors exp4v2 `_mask_to_bboxes` (threshold >0 → external contours → keep a
    bbox iff bw*bh >= min_area) so this converter's validity criterion EXACTLY
    matches what exp4v2 will label. A mask with only sub-min_area (or zero)
    foreground passes a naive ">=1 nonzero" check but is DROPPED by exp4v2 at
    labeling → budget-parity break; catching it here prevents that.

    Returns None when the mask cannot be inspected (no cv2/PIL) — the caller then
    falls back to a mask-EXISTS-only check. cv2 is preferred (per-contour, exact
    match to exp4v2); PIL+numpy fallback approximates with the overall nonzero
    bounding box. A file that exists but fails to decode is treated as invalid.
    """
    if HAS_CV2:
        arr = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if arr is None:
            return False
        _, binm = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            x, y, bw, bh = cv2.boundingRect(c)
            if bw * bh >= min_area:
                return True
        return False
    if HAS_PIL and HAS_NUMPY:
        try:
            im = Image.open(mask_path).convert("L")
        except Exception:  # noqa: BLE001
            return False
        arr = np.asarray(im)
        ys, xs = np.nonzero(arr > 0)
        if xs.size == 0:
            return False
        # No cv2 → approximate exp4v2's per-contour test with the OVERALL nonzero
        # bounding box (conservative: may keep a mask exp4v2 later drops when its
        # pixels are scattered sub-min_area blobs; the cv2 path above is exact).
        bw = int(xs.max() - xs.min() + 1)
        bh = int(ys.max() - ys.min() + 1)
        return bw * bh >= min_area
    # No image library — cannot inspect pixels.
    return None


def _cluster_id_from_name(name: str) -> int:
    """Parse `_class{N}` from a composed file name → int, fallback 1."""
    m = _CLASS_RE.search(name)
    if m:
        try:
            return int(m.group(1))
        except (ValueError, TypeError):
            return 1
    return 1


# ---------------------------------------------------------------------------
# Core conversion
# ---------------------------------------------------------------------------

def adapt(
    composed_dir: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Build the exp4v2 annotation list from a CASDA composed directory.

    Scans `{composed_dir}/images/*.png`, pairs each with
    `{composed_dir}/masks/{name}`, and keeps only entries whose mask yields a
    bbox with area >= _MIN_BBOX_AREA (i.e. exp4v2 can label it). Returns
    (annotations, stats).

    Args:
        composed_dir: CASDA compose output root (images/ + masks/ + metadata.json).

    Returns:
        (annotations, stats) where annotations is the valid-only list and stats
        has keys n_total, n_valid, n_skipped_no_bbox, n_skipped_no_maskfile.

    Raises:
        FileNotFoundError: composed_dir/images does not exist.
    """
    cdir = Path(composed_dir)
    images_dir = cdir / "images"
    masks_dir = cdir / "masks"
    if not images_dir.is_dir():
        raise FileNotFoundError(f"composed images dir not found: {images_dir}")

    imgs = sorted(glob.glob(str(images_dir / "*.png")))
    ann: List[Dict[str, Any]] = []
    n_total = len(imgs)
    n_skipped_no_bbox = 0
    n_skipped_no_maskfile = 0
    warned_no_lib = False

    for ip in imgs:
        name = Path(ip).name
        cid = _cluster_id_from_name(name)
        # compose_casda_images.py writes masks as `{image_stem}_mask.png` (NOT the
        # bare image name). Fall back to the bare name for any legacy layout.
        mp = masks_dir / f"{Path(name).stem}_mask.png"
        if not mp.exists():
            legacy = masks_dir / name
            if legacy.exists():
                mp = legacy

        if not mp.exists():
            # No mask file at all → exp4v2 cannot derive a bbox → drop.
            n_skipped_no_maskfile += 1
            logger.debug("no mask file for %s — skip", name)
            continue

        ok = _mask_has_valid_bbox(str(mp))
        if ok is None:
            # No cv2/PIL — cannot inspect pixels; pass on mask-exists only.
            if not warned_no_lib:
                logger.warning(
                    "Neither cv2 nor PIL available — bbox-validity detection DISABLED; "
                    "keeping every entry whose mask file exists. Install "
                    "opencv-python or pillow for the parity-preserving skip guarantee."
                )
                warned_no_lib = True
        elif not ok:
            # No bbox >= min_area → exp4v2 drops it at labeling → SKIP here so
            # annotations.json == exactly what exp4v2 will label (parity).
            n_skipped_no_bbox += 1
            logger.debug("no bbox >= %dpx for %s — skip", _MIN_BBOX_AREA, name)
            continue

        ann.append({
            "image_path": str(Path(ip).resolve()),
            "mask_path":  str(mp.resolve()),
            "source_roi": name,        # `_class{N}_` → _parse_severstal_class
            "cluster_id": cid,         # 1-4 fallback for severstal multi mode
            "dry_run":    False,
        })

    stats = {
        "n_total": n_total,
        "n_valid": len(ann),
        "n_skipped_no_bbox": n_skipped_no_bbox,
        "n_skipped_no_maskfile": n_skipped_no_maskfile,
    }
    return ann, stats


def convert(
    composed_dir: str,
    output_dir: str,
    dataset_key: str = "severstal",
) -> int:
    """Convert a CASDA composed dir → exp4v2 annotations.json (valid-only).

    Writes `{output_dir}/{dataset_key}/annotations.json` containing only entries
    whose composed mask yields a bbox with area >= _MIN_BBOX_AREA (so exp4v2 can
    label it and every sampled synth image is labelable — preserves budget parity).

    Args:
        composed_dir: CASDA compose output root (images/ + masks/ + metadata.json).
        output_dir:   exp4v2 synth root; annotations go to
                      {output_dir}/{dataset_key}/annotations.json.
        dataset_key:  dataset subdir (default "severstal").

    Returns:
        Number of valid entries written (n_valid).

    Raises:
        FileNotFoundError: composed_dir/images does not exist.
    """
    ann, stats = adapt(composed_dir)

    out_dir = Path(output_dir) / dataset_key
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "annotations.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(ann, f, indent=2)

    logger.info(
        "composed_to_exp4v2: n_total=%d, n_valid=%d, n_skipped_no_bbox=%d, "
        "n_skipped_no_maskfile=%d (min_bbox_area=%d) → %s",
        stats["n_total"], stats["n_valid"], stats["n_skipped_no_bbox"],
        stats["n_skipped_no_maskfile"], _MIN_BBOX_AREA, out_path,
    )
    return stats["n_valid"]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert CASDA composed output → exp4v2 annotations.json "
                    "(skips masks with no bbox >= min_area so budget parity holds)."
    )
    p.add_argument("--composed_dir", required=True,
                   help="CASDA compose output root (images/ + masks/ + metadata.json)")
    p.add_argument("--output_dir", required=True,
                   help="exp4v2 synth root; annotations → {output_dir}/{dataset_key}/annotations.json")
    p.add_argument("--dataset_key", default="severstal",
                   help="Dataset subdir (default severstal)")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = _parse_args(argv)
    try:
        n = convert(
            composed_dir=args.composed_dir,
            output_dir=args.output_dir,
            dataset_key=args.dataset_key,
        )
        print(f"Wrote {n} valid exp4v2 annotations → "
              f"{Path(args.output_dir) / args.dataset_key / 'annotations.json'}")
    except FileNotFoundError as e:
        logger.error("%s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
