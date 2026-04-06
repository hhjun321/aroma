"""stage3_layout_logic.py — Hybrid suitability-guided ROI selection for defect placement."""
import argparse
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from utils.io import load_json, save_json, validate_dir, validate_file
from utils.suitability import SuitabilityEvaluator


def _compute_placement_worker(args_tuple):
    """Module-level worker for placement computation (pickle-safe)."""
    image_id, roi_boxes, seed_paths_str, defect_subtype, domain, image_dir_str = args_tuple
    from utils.suitability import SuitabilityEvaluator
    import cv2
    evaluator = SuitabilityEvaluator()
    placements = []
    # For each seed, find best ROI by suitability score
    for seed_path_str in seed_paths_str:
        seed = cv2.imread(seed_path_str)
        if seed is None:
            continue
        best_score = -1.0
        best_box = roi_boxes[0] if roi_boxes else None
        for roi_box in roi_boxes:
            score = evaluator.compute_suitability(
                defect_subtype=defect_subtype,
                background_type=roi_box.get("background_type", "smooth"),
                continuity_score=float(roi_box.get("continuity_score", 0.5)),
                stability_score=float(roi_box.get("stability_score", 0.5)),
            )
            if score > best_score:
                best_score = score
                best_box = roi_box
        if best_box is None:
            continue
        x, y, w, h = best_box["box"]
        rotation = float(np.random.uniform(0, 360))
        # dominant_angle alignment
        if (best_box.get("background_type") == "directional"
                and defect_subtype in ("linear_scratch", "elongated")
                and best_box.get("dominant_angle") is not None):
            rotation = float(best_box["dominant_angle"])
        placements.append({
            "defect_path": seed_path_str,
            "x": x + w // 4,
            "y": y + h // 4,
            "scale": 1.0,
            "rotation": rotation,
            "suitability_score": round(best_score, 4),
            "matched_background_type": best_box.get("background_type", "smooth"),
        })
    return {"image_id": image_id, "placements": placements}


def _compute_placement_gpu(
    image_id: str,
    roi_boxes: list,
    seed_paths_str: list,
    defect_subtype: str,
    evaluator,          # GPUSuitabilityEvaluator instance
) -> dict:
    """GPU-accelerated placement: compute ROI scores once, reuse for all seeds."""
    import numpy as np
    if not roi_boxes or not seed_paths_str:
        return {"image_id": image_id, "placements": []}

    scores    = evaluator.compute_batch(defect_subtype, roi_boxes)
    best_idx  = int(np.argmax(scores))
    best_box  = roi_boxes[best_idx]
    bg_type   = best_box.get("background_type", "smooth")
    x, y, w, h = best_box["box"]

    if (bg_type == "directional"
            and defect_subtype in ("linear_scratch", "elongated")
            and best_box.get("dominant_angle") is not None):
        rotation = float(best_box["dominant_angle"])
    else:
        rotation = float(np.random.uniform(0, 360))

    placements = [
        {
            "defect_path":             seed_path_str,
            "x":                       x + w // 4,
            "y":                       y + h // 4,
            "scale":                   1.0,
            "rotation":                rotation,
            "suitability_score":       round(float(scores[best_idx]), 4),
            "matched_background_type": bg_type,
        }
        for seed_path_str in seed_paths_str
    ]
    return {"image_id": image_id, "placements": placements}


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
    workers: int = 0,
    use_gpu: bool = False,
) -> None:
    """Select best ROI for each defect seed using hybrid suitability score.

    Args:
        roi_metadata:    Path to roi_metadata.json produced by stage 1.
        defect_seeds_dir: Directory containing defect seed PNG images.
        output_dir:      Directory where placement_map.json will be written.
        seed_profile:    Optional path to seed_profile.json with 'subtype' field.
        domain:          Matching-rules domain ("isp" or "mvtec").
        image_dir:       Optional directory of source images for Gram similarity.
        workers:         Number of parallel workers (0=sequential, -1=auto, N>=2=N processes).
        use_gpu:         Use GPU-accelerated batch scoring (requires PyTorch).
    """
    # ── 전제조건 검증 ──────────────────────────────────────────────────────
    validate_file(roi_metadata, name="Stage 1 roi_metadata.json")
    validate_dir(defect_seeds_dir, name="Stage 2 defect_seeds_dir")
    if not any(Path(defect_seeds_dir).glob("*.png")):
        raise FileNotFoundError(
            f"Stage 2 defect_seeds_dir contains no PNG files: {defect_seeds_dir}"
        )
    if seed_profile is not None:
        validate_file(seed_profile, name="Stage 1b seed_profile.json")

    from utils.parallel import resolve_workers, run_parallel

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    meta_list = load_json(roi_metadata)
    seeds = _load_seeds(defect_seeds_dir)
    seed_paths = [p for p, _ in seeds]

    # Resolve defect subtype
    defect_subtype = "general"
    if seed_profile is not None:
        profile = load_json(seed_profile)
        defect_subtype = profile.get("subtype", "general")

    num_workers = resolve_workers(workers)

    if use_gpu:
        from utils.suitability import GPUSuitabilityEvaluator
        evaluator = GPUSuitabilityEvaluator()
        results = [
            _compute_placement_gpu(
                entry["image_id"],
                entry.get("roi_boxes", []),
                seed_paths,
                defect_subtype,
                evaluator,
            )
            for entry in meta_list
        ]
    else:
        tasks = [
            (
                entry["image_id"],
                entry.get("roi_boxes", []),
                seed_paths,
                defect_subtype,
                domain,
                image_dir,
            )
            for entry in meta_list
        ]
        results = run_parallel(
            _compute_placement_worker,
            tasks,
            num_workers,
            desc=f"Stage3 placement computation (workers={num_workers})",
        )

    # run_parallel may reorder results in parallel mode; preserve meta_list order
    id_to_result = {r["image_id"]: r for r in results}
    placement_map = [
        id_to_result.get(entry["image_id"], {"image_id": entry["image_id"], "placements": []})
        for entry in meta_list
    ]

    save_json(placement_map, out_path / "placement_map.json")


def _parse_args():
    parser = argparse.ArgumentParser(description="Stage 3: Suitability-guided layout logic")
    parser.add_argument("--roi_metadata", required=True, help="Path to roi_metadata.json")
    parser.add_argument("--defect_seeds_dir", required=True, help="Directory of defect seed PNGs")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--seed_profile", default=None, help="Path to seed_profile.json")
    parser.add_argument("--image_dir", default=None, help="Directory of source images (for Gram similarity)")
    parser.add_argument("--domain", default="mvtec", choices=["isp", "mvtec"], help="Matching rules domain")
    parser.add_argument("--workers", type=int, default=0,
                        help="병렬 워커 수 (0=순차 처리, -1=자동 감지, N>=2=N개 프로세스 병렬 처리)")
    parser.add_argument("--use_gpu", action="store_true",
                        help="GPU 가속 배치 평가 사용 (PyTorch 필요)")
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
        workers=args.workers,
        use_gpu=args.use_gpu,
    )
