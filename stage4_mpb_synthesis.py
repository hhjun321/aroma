"""stage4_mpb_synthesis.py — Modified Poisson Blending (MPB) synthesis stage.

Composites defect patches onto background images using cv2.seamlessClone
(Poisson blending) guided by a placement_map.json produced by Stage 3.
"""
import argparse
from pathlib import Path
from typing import List

import cv2
import numpy as np

from utils.io import load_json


def _transform_patch(patch: np.ndarray, scale: float, rotation: float) -> np.ndarray:
    """Apply scale and rotation to a patch image."""
    h, w = patch.shape[:2]
    if scale != 1.0:
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        patch = cv2.resize(patch, (new_w, new_h))
    if rotation != 0:
        h2, w2 = patch.shape[:2]
        M = cv2.getRotationMatrix2D((w2 / 2, h2 / 2), rotation, 1.0)
        patch = cv2.warpAffine(patch, M, (w2, h2))
    return patch


def _blend_patch(
    bg: np.ndarray,
    patch: np.ndarray,
    x: int,
    y: int,
) -> np.ndarray:
    """Blend patch into bg at position (x, y) using Poisson blending.

    Returns the composited image (same shape as bg).
    Falls back to direct copy if seamlessClone cannot be applied.
    """
    ph, pw = patch.shape[:2]
    ih, iw = bg.shape[:2]

    # Ensure patch fits within background dimensions
    if pw > iw or ph > ih:
        # Scale down patch so it fits
        scale_w = iw / pw if pw > iw else 1.0
        scale_h = ih / ph if ph > ih else 1.0
        scale = min(scale_w, scale_h) * 0.9  # slight margin
        new_w = max(1, int(pw * scale))
        new_h = max(1, int(ph * scale))
        patch = cv2.resize(patch, (new_w, new_h))
        ph, pw = patch.shape[:2]

    # Compute center of clone destination
    cx = x + pw // 2
    cy = y + ph // 2

    # Clamp center so that the patch lies fully within the background
    # seamlessClone requires patch to fit completely inside bg
    cx = max(pw // 2 + 1, min(iw - pw // 2 - 1, cx))
    cy = max(ph // 2 + 1, min(ih - ph // 2 - 1, cy))

    # Full white mask — blend entire patch
    mask = 255 * np.ones((ph, pw), dtype=np.uint8)

    try:
        result = cv2.seamlessClone(patch, bg, mask, (cx, cy), cv2.NORMAL_CLONE)
    except cv2.error:
        # Fallback: direct alpha composite using the patch region
        result = bg.copy()
        # Compute top-left corner from clamped center
        top = cy - ph // 2
        left = cx - pw // 2
        top = max(0, min(ih - ph, top))
        left = max(0, min(iw - pw, left))
        result[top:top + ph, left:left + pw] = patch

    return result


def run_synthesis(
    placement_map: str,
    image_dir: str,
    output_dir: str,
    format: str = "cls",
) -> None:
    """Composite defect patches onto backgrounds and write the augmented dataset.

    Args:
        placement_map: Path to placement_map.json produced by Stage 3.
        image_dir:     Directory containing background images (<image_id>.png).
        output_dir:    Root output directory.
        format:        Output format — 'cls' (classification) or 'yolo' (detection).
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    entries: List[dict] = load_json(placement_map)
    img_dir = Path(image_dir)

    for entry in entries:
        image_id: str = entry["image_id"]
        placements: List[dict] = entry.get("placements", [])

        # Load background image
        bg_path = img_dir / f"{image_id}.png"
        if not bg_path.exists():
            # Try without extension (in case it was already included)
            bg_path = img_dir / image_id
        if not bg_path.exists():
            continue

        bg = cv2.imread(str(bg_path))
        if bg is None:
            continue

        composited = bg.copy()
        ih, iw = bg.shape[:2]

        # Accumulate bounding boxes for YOLO labels
        yolo_boxes: List[str] = []

        for placement in placements:
            defect_path = placement["defect_path"]
            x: int = int(placement.get("x", 0))
            y: int = int(placement.get("y", 0))
            scale: float = float(placement.get("scale", 1.0))
            rotation: float = float(placement.get("rotation", 0))

            patch = cv2.imread(defect_path)
            if patch is None:
                continue

            # Apply geometric transforms
            patch = _transform_patch(patch, scale, rotation)
            ph, pw = patch.shape[:2]

            # Blend patch into composited background
            composited = _blend_patch(composited, patch, x, y)

            # Record bounding box (position of patch in background)
            if format == "yolo":
                # Clamped center used inside _blend_patch — replicate logic for bbox
                cx = x + pw // 2
                cy = y + ph // 2
                cx = max(pw // 2 + 1, min(iw - pw // 2 - 1, cx))
                cy = max(ph // 2 + 1, min(ih - ph // 2 - 1, cy))
                bx = cx - pw // 2
                by = cy - ph // 2
                # Normalize
                norm_cx = (bx + pw / 2) / iw
                norm_cy = (by + ph / 2) / ih
                norm_w = pw / iw
                norm_h = ph / ih
                yolo_boxes.append(f"0 {norm_cx:.6f} {norm_cy:.6f} {norm_w:.6f} {norm_h:.6f}")

        # Save output image and labels
        if format == "cls":
            cls_dir = out_path / "defect"
            cls_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(cls_dir / f"{image_id}.png"), composited)

        elif format == "yolo":
            images_dir = out_path / "images"
            labels_dir = out_path / "labels"
            images_dir.mkdir(parents=True, exist_ok=True)
            labels_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(images_dir / f"{image_id}.png"), composited)
            label_path = labels_dir / f"{image_id}.txt"
            label_path.write_text("\n".join(yolo_boxes))

        else:
            raise ValueError(f"Unknown format: {format!r}. Use 'cls' or 'yolo'.")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 4: Modified Poisson Blending (MPB) defect synthesis"
    )
    parser.add_argument("--placement_map", required=True, help="Path to placement_map.json")
    parser.add_argument("--image_dir", required=True, help="Directory of background images")
    parser.add_argument("--output_dir", required=True, help="Root output directory")
    parser.add_argument(
        "--format",
        default="cls",
        choices=["cls", "yolo"],
        help="Output format: 'cls' (classification) or 'yolo' (detection)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_synthesis(
        placement_map=args.placement_map,
        image_dir=args.image_dir,
        output_dir=args.output_dir,
        format=args.format,
    )
