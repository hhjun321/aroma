#!/usr/bin/env python3
"""
AROMA — Random ROI selection → copy-paste synthesis.

Thin wrapper: uniformly samples top_k ROIs from roi_candidates.json,
writes roi_selected.json, then delegates to generate_defects.run().

Used as baseline for both Exp 1 (Severstal) and Exp 2 (cross-domain).
All methods use identical synthesis code; only ROI selection differs.

Usage (Colab):
    !python $AROMA_SCRIPTS/generate_random.py \
        --candidates_json $AROMA_OUT/roi/severstal/roi_candidates.json \
        --normal_dir      $SEVERSTAL_DATA/train_normal \
        --output_dir      $AROMA_OUT_SEVERSTAL/synthetic_random/severstal \
        --top_k           200 \
        --seed            42
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("aroma.generate_random")

# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

_THIS_DIR = Path(__file__).resolve().parent


def _bootstrap() -> None:
    aroma_ref = os.environ.get("AROMA_REF") or r"D:\project\aroma"
    for p in (str(_THIS_DIR), str(Path(aroma_ref))):
        if p not in sys.path:
            sys.path.insert(0, p)


_bootstrap()

import generate_defects  # noqa: E402


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

try:
    from utils.io import load_json, save_json  # type: ignore[import]
except Exception:
    def load_json(p: str) -> Any:  # type: ignore[misc]
        with open(p, encoding="utf-8") as f:
            return json.load(f)

    def save_json(data: Any, p: str) -> None:  # type: ignore[misc]
        Path(p).parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Random selection
# ---------------------------------------------------------------------------

def select_random(
    candidates: List[Dict[str, Any]],
    top_k: int,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Uniformly sample top_k candidates without replacement."""
    n = len(candidates)
    if n == 0:
        return []
    k = min(top_k, n)
    rng = np.random.default_rng(seed)
    indices = rng.choice(n, size=k, replace=False)
    return [candidates[int(i)] for i in sorted(indices.tolist())]


# ---------------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------------

def run(
    candidates_json: str,
    normal_dir: str,
    output_dir: str,
    random_roi_staging_dir: str | None = None,
    top_k: int = 200,
    n_per_roi: int = 3,
    seed: int = 42,
    local_staging: bool = False,
    reject_clean_bg: bool = False,
    min_bg_quality: float = 0.7,
    bg_blur_threshold: float = 100.0,
) -> Dict[str, Any]:
    """Run random ROI selection then copy-paste synthesis.

    Args:
        candidates_json:        Path to AROMA roi_candidates.json (shared pool).
        normal_dir:             Directory of normal (defect-free) background images.
        output_dir:             Destination for synthetic images + annotations.json.
        random_roi_staging_dir: Where to write roi_selected.json.
                                Defaults to {output_dir}/_random_roi.
        top_k:                  Number of ROIs to select.
        n_per_roi:              Synthetic images per ROI.
        seed:                   Random seed (selection + synthesis).
        local_staging:          Copy inputs to /content/tmp (Colab Drive optimization).
        reject_clean_bg:        Forward black/flat-background gate to
                                generate_defects.run() (default OFF).
        min_bg_quality:         Min background quality for the gate (default 0.7).
        bg_blur_threshold:      Laplacian blur threshold for the gate (default 100.0).

    Returns:
        generate_defects.run() result dict + n_rois_selected.
    """
    cand_path = Path(candidates_json)
    if not cand_path.exists():
        logger.error("roi_candidates.json not found: %s", candidates_json)
        sys.exit(1)

    candidates: List[Dict[str, Any]] = load_json(str(cand_path))
    logger.info("Loaded %d candidates from %s", len(candidates), candidates_json)

    selected = select_random(candidates, top_k=top_k, seed=seed)
    logger.info("Randomly selected %d / %d ROIs (top_k=%d, seed=%d)",
                len(selected), len(candidates), top_k, seed)

    # Write roi_selected.json for generate_defects.run()
    roi_dir = random_roi_staging_dir or str(Path(output_dir) / "_random_roi")
    Path(roi_dir).mkdir(parents=True, exist_ok=True)
    save_json(selected, str(Path(roi_dir) / "roi_selected.json"))

    # Copy candidates alongside selection for ROI quality metrics
    save_json(candidates, str(Path(roi_dir) / "roi_candidates.json"))

    result = generate_defects.run(
        roi_dir=roi_dir,
        normal_dir=normal_dir,
        output_dir=output_dir,
        method="copy_paste",
        n_per_roi=n_per_roi,
        seed=seed,
        local_staging=local_staging,
        reject_clean_bg=reject_clean_bg,
        min_bg_quality=min_bg_quality,
        bg_blur_threshold=bg_blur_threshold,
    )
    result["n_rois_selected"] = len(selected)
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate synthetic defects using random ROI selection (copy-paste)"
    )
    p.add_argument("--candidates_json",  required=True,
                   help="AROMA roi_candidates.json path (shared candidate pool)")
    p.add_argument("--normal_dir",       required=True,
                   help="Directory of normal (good) background images")
    p.add_argument("--output_dir",       required=True,
                   help="Output directory for synthetic images + annotations.json")
    p.add_argument("--random_roi_dir",   default=None,
                   help="Staging dir for roi_selected.json "
                        "(default: {output_dir}/_random_roi)")
    p.add_argument("--top_k",            type=int, default=200,
                   help="Number of ROIs to select (default 200)")
    p.add_argument("--n_per_roi",        type=int, default=3,
                   help="Synthetic images per ROI (default 3)")
    p.add_argument("--seed",             type=int, default=42)
    p.add_argument("--local_staging",    action="store_true",
                   help="Copy inputs to /content/tmp (faster on Colab with Drive)")
    p.add_argument("--reject-clean-bg",  dest="reject_clean_bg",
                   action="store_true",
                   help="Reject black/flat (void) backgrounds at generation time "
                        "(default OFF)")
    p.add_argument("--min-bg-quality",   dest="min_bg_quality",
                   type=float, default=0.7,
                   help="Min background quality 0..1 for the clean-bg gate "
                        "(default 0.7)")
    p.add_argument("--bg-blur-threshold", dest="bg_blur_threshold",
                   type=float, default=100.0,
                   help="Laplacian-variance blur threshold for the clean-bg gate "
                        "(default 100.0)")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = _parse_args(argv)
    result = run(
        candidates_json=args.candidates_json,
        normal_dir=args.normal_dir,
        output_dir=args.output_dir,
        random_roi_staging_dir=args.random_roi_dir,
        top_k=args.top_k,
        n_per_roi=args.n_per_roi,
        seed=args.seed,
        local_staging=args.local_staging,
        reject_clean_bg=args.reject_clean_bg,
        min_bg_quality=args.min_bg_quality,
        bg_blur_threshold=args.bg_blur_threshold,
    )
    status = result.get("status", "unknown")
    n_gen = result.get("n_generated", 0)
    print(f"[generate_random] status={status}  n_generated={n_gen}  "
          f"n_rois_selected={result.get('n_rois_selected', 0)}")
    if status != "ok":
        sys.exit(1)


if __name__ == "__main__":
    main()
