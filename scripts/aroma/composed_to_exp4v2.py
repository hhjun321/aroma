#!/usr/bin/env python3
"""
CASDA composed output → exp4v2 synth annotations.json converter.

This is the scriptified version of the compounding-guide §3d inline cell, with
one CRITICAL added guarantee: **empty-mask entries are SKIPPED**.

Why (dev_note ⑧ 결함2 — parity 붕괴):
    CASDA `compose_casda_images.py` writes `{composed_dir}/images/*.png`,
    `{composed_dir}/masks/{name}` (full-frame masks) and a metadata.json. Some
    ControlNet-generated defects come out blank, so their composed mask has ZERO
    nonzero pixels. exp4v2's mask→bbox step (`_mask_to_bboxes`) extracts no bbox
    from an empty mask, so those synth images silently DROP during labeling.
    The first compounding run showed casda arm = 771/1013 labeled (242 dropped)
    vs random/aroma = 1013/1013, breaking budget parity and confounding the
    aroma-vs-casda comparison. The inline §3d cell wrote every composed image
    (including empty-mask ones) into annotations.json, so the drop happened
    downstream inside exp4v2. This converter instead validates the mask has >=1
    nonzero pixel BEFORE recording the entry, so exp4v2 samples from a valid-only
    pool and every sampled image is labelable.

Output schema (matches exp4v2 `_load_synth_annotations`, verified):
    A list of dicts, each:
        image_path : absolute path to composed image
        mask_path  : absolute path to full-frame mask (exp4v2 derives bbox)
        source_roi : image file name (contains `_class{N}_` → class parse)
        cluster_id : Severstal class 1-4 (from `_class{N}` in name, fallback 1)
        dry_run    : False
    Written to `{output_dir}/{dataset_key}/annotations.json`.

Empty-mask skip precedence (deterministic):
    - mask file missing            → SKIP, n_skipped_no_maskfile++
    - cv2/PIL available and mask has 0 nonzero pixels → SKIP, n_skipped_empty++
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


# ---------------------------------------------------------------------------
# Mask inspection
# ---------------------------------------------------------------------------

def _mask_has_nonzero(mask_path: str) -> Optional[bool]:
    """Return True if the mask has >=1 nonzero pixel, False if all-zero.

    Returns None when the mask cannot be inspected (no cv2/PIL) — the caller then
    falls back to a mask-EXISTS-only check. cv2 is preferred; PIL+numpy fallback.
    A file that exists but fails to decode is treated as empty (False) so a
    corrupt/blank mask is skipped rather than passed through.
    """
    if HAS_CV2:
        arr = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if arr is None:
            return False
        # int() guards against numpy bool truthiness ambiguity.
        return int((arr > 0).sum()) > 0
    if HAS_PIL and HAS_NUMPY:
        try:
            im = Image.open(mask_path).convert("L")
        except Exception:  # noqa: BLE001
            return False
        arr = np.asarray(im)
        return int((arr > 0).sum()) > 0
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
    `{composed_dir}/masks/{name}`, and keeps only entries whose mask has a
    nonzero pixel (i.e. exp4v2 can derive a bbox). Returns (annotations, stats).

    Args:
        composed_dir: CASDA compose output root (images/ + masks/ + metadata.json).

    Returns:
        (annotations, stats) where annotations is the valid-only list and stats
        has keys n_total, n_valid, n_skipped_empty, n_skipped_no_maskfile.

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
    n_skipped_empty = 0
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

        nz = _mask_has_nonzero(str(mp))
        if nz is None:
            # No cv2/PIL — cannot inspect pixels; pass on mask-exists only.
            if not warned_no_lib:
                logger.warning(
                    "Neither cv2 nor PIL available — empty-mask detection DISABLED; "
                    "keeping every entry whose mask file exists. Install "
                    "opencv-python or pillow for the empty-mask skip guarantee."
                )
                warned_no_lib = True
        elif not nz:
            # Mask exists but is all-zero → no bbox → parity-breaking drop. SKIP.
            n_skipped_empty += 1
            logger.debug("empty mask (0 nonzero px) for %s — skip", name)
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
        "n_skipped_empty": n_skipped_empty,
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
    whose composed mask has a nonzero pixel (so exp4v2 can derive a bbox and
    every sampled synth image is labelable — preserves budget parity across arms).

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
        "composed_to_exp4v2: n_total=%d, n_valid=%d, n_skipped_empty=%d, "
        "n_skipped_no_maskfile=%d → %s",
        stats["n_total"], stats["n_valid"], stats["n_skipped_empty"],
        stats["n_skipped_no_maskfile"], out_path,
    )
    return stats["n_valid"]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert CASDA composed output → exp4v2 annotations.json "
                    "(skips empty-mask entries so budget parity holds)."
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
