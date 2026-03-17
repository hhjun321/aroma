"""stage3_layout_logic.py — Hybrid suitability-guided ROI selection for defect placement."""
import argparse
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from utils.io import load_json, save_json
from utils.suitability import SuitabilityEvaluator


def _gram_similarity(img1: np.ndarray, img2: np.ndarray) -> float:
    """Cosine similarity between Gram matrices of two patches."""
    def gram(x):
        h, w, c = x.shape
        x_flat = x.reshape(-1, c).astype(np.float32)
        return x_flat.T @ x_flat / (h * w)

    g1, g2 = gram(img1), gram(img2)
    g1_flat, g2_flat = g1.flatten(), g2.flatten()
    denom = np.linalg.norm(g1_flat) * np.linalg.norm(g2_flat)
    if denom < 1e-8:
        return 0.5
    return float(np.dot(g1_flat, g2_flat) / denom)


def _load_seeds(seeds_dir: str) -> list:
    """Return list of (path_str, image_array) for all PNG seeds."""
    seeds = []
    for p in sorted(Path(seeds_dir).glob("*.png")):
        img = cv2.imread(str(p))
        if img is not None:
            seeds.append((str(p), img))
    return seeds


def _crop_roi(image: np.ndarray, box: list) -> Optional[np.ndarray]:
    """Crop image using box [x, y, w, h], returns None if out of bounds."""
    x, y, w, h = box
    ih, iw = image.shape[:2]
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(iw, x + w), min(ih, y + h)
    if x2 <= x1 or y2 <= y1:
        return None
    return image[y1:y2, x1:x2]


def run_layout_logic(
    roi_metadata: str,
    defect_seeds_dir: str,
    output_dir: str,
    seed_profile: Optional[str] = None,
    domain: str = "mvtec",
    image_dir: Optional[str] = None,
) -> None:
    """Select best ROI for each defect seed using hybrid suitability score.

    Args:
        roi_metadata:    Path to roi_metadata.json produced by stage 1.
        defect_seeds_dir: Directory containing defect seed PNG images.
        output_dir:      Directory where placement_map.json will be written.
        seed_profile:    Optional path to seed_profile.json with 'subtype' field.
        domain:          Matching-rules domain ("isp" or "mvtec").
        image_dir:       Optional directory of source images for Gram similarity.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    meta_list = load_json(roi_metadata)
    seeds = _load_seeds(defect_seeds_dir)

    # Resolve defect subtype
    defect_subtype = "general"
    if seed_profile is not None:
        profile = load_json(seed_profile)
        defect_subtype = profile.get("subtype", "general")

    evaluator = SuitabilityEvaluator(domain=domain)

    placement_map = []

    for entry in meta_list:
        image_id = entry["image_id"]
        roi_boxes = entry.get("roi_boxes", [])

        # Load source image for Gram similarity if available
        source_image = None
        if image_dir is not None:
            img_path = Path(image_dir) / f"{image_id}.png"
            if img_path.exists():
                source_image = cv2.imread(str(img_path))

        placements = []

        for seed_path, seed_img in seeds:
            best_score = -1.0
            best_box = None
            best_bg_type = "unknown"

            for roi in roi_boxes:
                bg_type = roi.get("background_type", "smooth")
                continuity = float(roi.get("continuity_score", 0.5))
                stability = float(roi.get("stability_score", 0.5))

                # Gram similarity: use actual crop if source image available, else 0.5
                if source_image is not None:
                    crop = _crop_roi(source_image, roi["box"])
                    if crop is not None and crop.size > 0:
                        # Resize seed to crop size for comparison if needed
                        seed_resized = cv2.resize(seed_img, (crop.shape[1], crop.shape[0]))
                        gram_sim = _gram_similarity(seed_resized, crop)
                    else:
                        gram_sim = 0.5
                else:
                    gram_sim = 0.5

                score = evaluator.compute_suitability(
                    defect_subtype=defect_subtype,
                    background_type=bg_type,
                    continuity_score=continuity,
                    stability_score=stability,
                    gram_similarity=gram_sim,
                )

                if score > best_score:
                    best_score = score
                    best_box = roi["box"]
                    best_bg_type = bg_type

            # Compute placement coordinates from best ROI box center
            if best_box is not None:
                bx, by, bw, bh = best_box
                cx = int(bx + bw / 2)
                cy = int(by + bh / 2)
            else:
                cx, cy = 0, 0

            placements.append({
                "defect_path": seed_path,
                "x": cx,
                "y": cy,
                "scale": 1.0,
                "rotation": 0,
                "suitability_score": round(best_score, 6) if best_score >= 0 else 0.0,
                "matched_background_type": best_bg_type,
            })

        placement_map.append({
            "image_id": image_id,
            "placements": placements,
        })

    save_json(placement_map, out_path / "placement_map.json")


def _parse_args():
    parser = argparse.ArgumentParser(description="Stage 3: Suitability-guided layout logic")
    parser.add_argument("--roi_metadata", required=True, help="Path to roi_metadata.json")
    parser.add_argument("--defect_seeds_dir", required=True, help="Directory of defect seed PNGs")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--seed_profile", default=None, help="Path to seed_profile.json")
    parser.add_argument("--image_dir", default=None, help="Directory of source images (for Gram similarity)")
    parser.add_argument("--domain", default="mvtec", choices=["isp", "mvtec"], help="Matching rules domain")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_layout_logic(
        roi_metadata=args.roi_metadata,
        defect_seeds_dir=args.defect_seeds_dir,
        output_dir=args.output_dir,
        seed_profile=args.seed_profile,
        domain=args.domain,
        image_dir=args.image_dir,
    )
