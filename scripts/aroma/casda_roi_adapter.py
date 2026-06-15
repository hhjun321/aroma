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

            rois.append({
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
            })

    logger.info(
        "casda_roi_adapter: %d total → %d suitability-filtered, %d cap-filtered, "
        "%d missing-image → %d valid ROIs",
        n_total, n_filtered_suitability, n_filtered_cap, n_missing_image, len(rois),
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
