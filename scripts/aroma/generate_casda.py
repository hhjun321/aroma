#!/usr/bin/env python3
"""
AROMA Exp 1 — CASDA ROI selection → copy-paste synthesis.

Thin wrapper: adapts CASDA roi_metadata.csv → roi_selected.json,
then delegates to generate_defects.run() (same copy-paste engine used by AROMA).

All three methods (Random / CASDA / AROMA) use identical synthesis code;
only the ROI selection strategy differs — this isolates the ROI modeling contribution.

Usage (Colab):
    !python $AROMA_SCRIPTS/generate_casda.py \
        --metadata_csv  $AROMA_OUT_SEVERSTAL/casda/roi_metadata.csv \
        --normal_dir    $SEVERSTAL_DATA/train_normal \
        --output_dir    $AROMA_OUT_SEVERSTAL/synthetic_casda/severstal \
        --n_per_roi     3 \
        --seed          42
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("aroma.generate_casda")

# ---------------------------------------------------------------------------
# Bootstrap: add scripts/aroma/ to sys.path for sibling imports
# ---------------------------------------------------------------------------

_THIS_DIR = Path(__file__).resolve().parent


def _bootstrap() -> None:
    aroma_ref = os.environ.get("AROMA_REF") or r"D:\project\aroma"
    for p in (str(_THIS_DIR), str(Path(aroma_ref))):
        if p not in sys.path:
            sys.path.insert(0, p)


_bootstrap()

import casda_roi_adapter  # noqa: E402  (sibling, available after bootstrap)
import generate_defects   # noqa: E402  (sibling, available after bootstrap)


# ---------------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------------

def run(
    metadata_csv: str,
    normal_dir: str,
    output_dir: str,
    casda_roi_staging_dir: str | None = None,
    n_per_roi: int = 3,
    seed: int = 42,
    min_suitability: float = casda_roi_adapter.MIN_SUITABILITY,
    per_class_cap: int | None = None,
    local_staging: bool = False,
) -> Dict[str, Any]:
    """Run CASDA ROI adaptation then copy-paste synthesis.

    Args:
        metadata_csv:           CASDA Stage A roi_metadata.csv path.
        normal_dir:             Directory of normal (defect-free) background images.
        output_dir:             Destination for synthetic images + annotations.json.
        casda_roi_staging_dir:  Where to write intermediate roi_selected.json.
                                Defaults to {output_dir}/_casda_roi.
        n_per_roi:              Synthetic images per ROI.
        seed:                   Random seed for synthesis.
        min_suitability:        CASDA suitability filter threshold.
        per_class_cap:          Max ROIs per defect class (None = no cap).
        local_staging:          Copy inputs to /content/tmp before synthesis.

    Returns:
        generate_defects.run() result dict + n_rois_adapted.
    """
    # Step 1: CASDA ROI → AROMA roi_selected.json
    roi_dir = casda_roi_staging_dir or str(Path(output_dir) / "_casda_roi")
    try:
        rois = casda_roi_adapter.adapt(
            metadata_csv=metadata_csv,
            output_dir=roi_dir,
            min_suitability=min_suitability,
            per_class_cap=per_class_cap,
        )
    except (FileNotFoundError, RuntimeError) as e:
        logger.error("casda_roi_adapter failed: %s", e)
        sys.exit(1)

    logger.info("CASDA adapter: %d ROIs → %s/roi_selected.json", len(rois), roi_dir)

    # Step 2: copy-paste synthesis (same engine as AROMA)
    result = generate_defects.run(
        roi_dir=roi_dir,
        normal_dir=normal_dir,
        output_dir=output_dir,
        method="copy_paste",
        n_per_roi=n_per_roi,
        seed=seed,
        local_staging=local_staging,
    )
    result["n_rois_adapted"] = len(rois)
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate synthetic defects using CASDA ROI selection (copy-paste)"
    )
    p.add_argument("--metadata_csv",    required=True,
                   help="CASDA Stage A roi_metadata.csv path")
    p.add_argument("--normal_dir",      required=True,
                   help="Directory of normal (good) background images")
    p.add_argument("--output_dir",      required=True,
                   help="Output directory for synthetic images + annotations.json")
    p.add_argument("--casda_roi_dir",   default=None,
                   help="Intermediate roi_selected.json directory "
                        "(default: {output_dir}/_casda_roi)")
    p.add_argument("--n_per_roi",       type=int, default=3,
                   help="Synthetic images per ROI (default 3)")
    p.add_argument("--seed",            type=int, default=42)
    p.add_argument("--min_suitability", type=float,
                   default=casda_roi_adapter.MIN_SUITABILITY,
                   help="CASDA suitability filter (default 0.5)")
    p.add_argument("--per_class_cap",   type=int, default=None,
                   help="Max ROIs per defect class")
    p.add_argument("--local_staging",   action="store_true",
                   help="Copy inputs to /content/tmp (faster on Colab with Drive)")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = _parse_args(argv)
    result = run(
        metadata_csv=args.metadata_csv,
        normal_dir=args.normal_dir,
        output_dir=args.output_dir,
        casda_roi_staging_dir=args.casda_roi_dir,
        n_per_roi=args.n_per_roi,
        seed=args.seed,
        min_suitability=args.min_suitability,
        per_class_cap=args.per_class_cap,
        local_staging=args.local_staging,
    )
    status = result.get("status", "unknown")
    n_gen = result.get("n_generated", 0)
    print(f"[generate_casda] status={status}  n_generated={n_gen}  "
          f"n_rois_adapted={result.get('n_rois_adapted', 0)}")
    if status != "ok":
        sys.exit(1)


if __name__ == "__main__":
    main()
