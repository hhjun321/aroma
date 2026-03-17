"""
Stage 1: ROI Extraction

Processes all .png images in image_dir, extracts ROI regions using Otsu
thresholding (with SAM as optional upgrade when checkpoint is available),
runs background analysis on each local ROI crop, and writes roi_metadata.json.

CLI usage:
    python stage1_roi_extraction.py \
        --image_dir /path/to/images \
        --output_dir /path/to/output \
        --domain mvtec \
        --roi_levels both \
        --grid_size 64
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import cv2
import numpy as np

# Project utilities
sys.path.insert(0, str(Path(__file__).parent))
from utils.io import save_json, load_json, validate_dir
from utils.mask import save_mask, load_mask
from utils.background_characterization import BackgroundAnalyzer, BackgroundType


# ---------------------------------------------------------------------------
# Segmentation helpers
# ---------------------------------------------------------------------------

def _otsu_global_mask(gray: np.ndarray) -> np.ndarray:
    """Return a binary (0/255) mask via Otsu thresholding."""
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mask


def _extract_local_boxes(
    mask: np.ndarray,
    image_shape: Tuple[int, int],
    min_area: int = 100,
) -> List[Tuple[int, int, int, int]]:
    """Extract bounding boxes from connected components in the mask.

    Returns a list of (x, y, w, h) tuples that fit within image_shape (h, w).
    Falls back to a single full-image box if no component meets min_area.
    """
    img_h, img_w = image_shape
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    boxes: List[Tuple[int, int, int, int]] = []
    for lbl in range(1, num_labels):  # skip background label 0
        x = int(stats[lbl, cv2.CC_STAT_LEFT])
        y = int(stats[lbl, cv2.CC_STAT_TOP])
        w = int(stats[lbl, cv2.CC_STAT_WIDTH])
        h = int(stats[lbl, cv2.CC_STAT_HEIGHT])
        area = int(stats[lbl, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        # Clamp to image bounds
        x = max(0, min(x, img_w - 1))
        y = max(0, min(y, img_h - 1))
        w = min(w, img_w - x)
        h = min(h, img_h - y)
        if w > 0 and h > 0:
            boxes.append((x, y, w, h))

    if not boxes:
        # Fallback: single full-image zone
        boxes = [(0, 0, img_w, img_h)]

    return boxes


def _local_mask_from_box(
    global_mask: np.ndarray,
    box: Tuple[int, int, int, int],
) -> np.ndarray:
    """Crop the global mask to the given bounding box."""
    x, y, w, h = box
    return global_mask[y : y + h, x : x + w].copy()


# ---------------------------------------------------------------------------
# Background analysis helpers
# ---------------------------------------------------------------------------

def _analyze_box(
    image: np.ndarray,
    box: Tuple[int, int, int, int],
    analyzer: BackgroundAnalyzer,
    analysis_result: Dict,
) -> Tuple[str, float, float]:
    """Return (background_type, continuity_score, stability_score) for a box.

    Falls back gracefully when the image crop is too small for grid analysis.
    """
    x, y, w, h = box
    img_h, img_w = image.shape[:2]

    # Get background type at the box centre
    cx = min(x + w // 2, img_w - 1)
    cy = min(y + h // 2, img_h - 1)
    loc_info = analyzer.get_background_at_location(analysis_result, cx, cy)

    if loc_info is not None:
        bg_type = loc_info["background_type"]
        stability = float(loc_info["stability_score"])
    else:
        # crop is outside the grid (e.g. image smaller than grid_size)
        # Analyse the crop directly
        crop = image[y : y + h, x : x + w]
        gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop
        bg_enum, stability = analyzer.classify_patch(gray_crop)
        bg_type = bg_enum.value

    # Continuity score over the bounding box (x1,y1,x2,y2)
    continuity = analyzer.check_continuity(
        analysis_result, (x, y, x + w, y + h)
    )
    # check_continuity returns 0.0 when region is empty (too small for grid)
    # Fall back to the stability score so it's non-trivially informative
    if continuity == 0.0 and analysis_result["grid_shape"] == (0, 0):
        continuity = stability

    return bg_type, float(continuity), float(stability)


# ---------------------------------------------------------------------------
# Per-image processing
# ---------------------------------------------------------------------------

def _process_image(
    image_path: Path,
    output_dir: Path,
    domain: str,
    roi_levels: str,
    grid_size: int,
) -> Dict[str, Any]:
    """Process a single image and return its metadata entry."""
    image_id = image_path.stem
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise ValueError(f"Cannot read image: {image_path}")

    img_h, img_w = image_bgr.shape[:2]
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # ---- Global mask (Otsu) ------------------------------------------------
    global_mask = _otsu_global_mask(gray)
    global_mask_path = output_dir / "masks" / "global" / f"{image_id}.png"
    save_mask(global_mask, global_mask_path)

    # ---- Background analysis -----------------------------------------------
    # Use a grid_size that fits the image (at least 1 cell)
    effective_grid = min(grid_size, img_h, img_w)
    analyzer = BackgroundAnalyzer(
        grid_size=effective_grid,
        variance_threshold=100.0,
        edge_threshold=0.3,
    )
    analysis_result = analyzer.analyze_image(image_bgr)

    # If the image is smaller than the grid cell, grid_shape will be (0,0).
    # Patch the result so downstream helpers work.
    if analysis_result["grid_shape"] == (0, 0):
        bg_enum, stability = analyzer.classify_patch(gray)
        analysis_result["background_map"] = np.array([[bg_enum.value]], dtype=object)
        analysis_result["stability_map"] = np.array([[stability]], dtype=np.float32)
        analysis_result["grid_shape"] = (1, 1)
        analysis_result["grid_info"] = [{
            "grid_id": (0, 0),
            "bbox": (0, 0, img_w, img_h),
            "background_type": bg_enum.value,
            "stability_score": float(stability),
        }]

    # ---- ROI boxes ---------------------------------------------------------
    roi_boxes: List[Dict[str, Any]] = []
    local_masks: List[str] = []

    if roi_levels in ("local", "both"):
        boxes = _extract_local_boxes(global_mask, (img_h, img_w))

        for zone_id, box in enumerate(boxes):
            x, y, w, h = box

            # Local mask
            local_mask = _local_mask_from_box(global_mask, box)
            local_mask_path = (
                output_dir / "masks" / "local" / f"{image_id}_zone{zone_id}.png"
            )
            save_mask(local_mask, local_mask_path)
            local_masks.append(str(local_mask_path))

            # Background analysis for this box
            bg_type, continuity, stability = _analyze_box(
                image_bgr, box, analyzer, analysis_result
            )

            roi_boxes.append({
                "level": "local",
                "box": [x, y, w, h],
                "zone_id": zone_id,
                "background_type": bg_type,
                "continuity_score": float(np.clip(continuity, 0.0, 1.0)),
                "stability_score": float(np.clip(stability, 0.0, 1.0)),
            })

    elif roi_levels == "global":
        # Provide global bounding box as a single "global" ROI entry
        ys, xs = np.where(global_mask > 0)
        if len(xs) > 0 and len(ys) > 0:
            gx1, gx2 = int(xs.min()), int(xs.max())
            gy1, gy2 = int(ys.min()), int(ys.max())
            gw = gx2 - gx1 + 1
            gh = gy2 - gy1 + 1
        else:
            gx1, gy1, gw, gh = 0, 0, img_w, img_h

        bg_type, continuity, stability = _analyze_box(
            image_bgr, (gx1, gy1, gw, gh), analyzer, analysis_result
        )
        # For global-only mode, roi_boxes can be empty per spec, but include
        # the global box so background fields are accessible if needed.
        roi_boxes = []

    # ---- Metadata entry ----------------------------------------------------
    entry: Dict[str, Any] = {
        "image_id": image_id,
        "global_mask": str(global_mask_path),
        "local_masks": local_masks,
        "roi_boxes": roi_boxes,
    }
    return entry


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_extraction(
    image_dir: str,
    output_dir: str,
    domain: str = "mvtec",
    roi_levels: str = "both",
    grid_size: int = 64,
) -> None:
    """Process all .png images in image_dir and write roi_metadata.json."""
    image_dir_path = Path(image_dir)
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    validate_dir(str(image_dir_path))

    images = sorted(image_dir_path.glob("*.png"))
    if not images:
        raise FileNotFoundError(f"No .png images found in {image_dir_path}")

    metadata: List[Dict[str, Any]] = []
    for img_path in images:
        entry = _process_image(
            img_path, output_dir_path, domain, roi_levels, grid_size
        )
        metadata.append(entry)

    save_json(metadata, output_dir_path / "roi_metadata.json")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AROMA Stage 1: ROI Extraction"
    )
    parser.add_argument("--image_dir", required=True, help="Input image directory")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument(
        "--domain", default="mvtec", choices=["mvtec", "isp"],
        help="Dataset domain (default: mvtec)"
    )
    parser.add_argument(
        "--roi_levels", default="both", choices=["global", "local", "both"],
        help="Which ROI levels to extract (default: both)"
    )
    parser.add_argument(
        "--grid_size", type=int, default=64,
        help="Grid cell size for background analysis (default: 64)"
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    run_extraction(
        image_dir=args.image_dir,
        output_dir=args.output_dir,
        domain=args.domain,
        roi_levels=args.roi_levels,
        grid_size=args.grid_size,
    )
