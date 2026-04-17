import argparse
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple
from utils.io import save_json
from utils.mask import save_mask
from utils.defect_characterization import DefectCharacterizer
from utils.parallel import resolve_workers, run_parallel


def extract_seed_mask(image: np.ndarray, checkpoint: str = None) -> tuple:
    """Extract binary defect mask from seed image via SAM or Otsu fallback.

    Returns:
        (mask, method) where mask is a uint8 binary array and method is
        'sam' or 'otsu'.
    """
    try:
        from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
        ckpt = checkpoint or "sam_vit_b_01ec64.pth"
        sam = sam_model_registry["vit_b"](checkpoint=ckpt)
        generator = SamAutomaticMaskGenerator(sam)
        masks = generator.generate(image)
        if masks:
            # Select smallest non-background mask (likely the defect)
            sorted_masks = sorted(masks, key=lambda m: m["area"])
            return (sorted_masks[0]["segmentation"].astype(np.uint8)) * 255, "sam"
    except Exception:
        pass
    # Otsu fallback
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binary, "otsu"


def run_seed_characterization(seed_defect: str, output_dir: str,
                               model_checkpoint: str = None):
    image = cv2.imread(seed_defect)
    if image is None:
        raise FileNotFoundError(f"Seed image not found: {seed_defect}")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    mask_bin, seg_method = extract_seed_mask(image, model_checkpoint)
    mask_path = out / "seed_mask.png"
    save_mask(mask_bin, mask_path)

    # Convert to 0/1 binary for regionprops
    mask_01 = (mask_bin > 0).astype(np.uint8)

    dc = DefectCharacterizer()
    metrics = dc.analyze_defect_region(mask_01)
    if metrics is None:
        # Fallback: whole image as single region
        metrics = {"linearity": 0.0, "solidity": 1.0, "extent": 1.0,
                   "aspect_ratio": 1.0, "region_id": 0, "area": mask_01.sum(),
                   "bbox": (0, 0, image.shape[1], image.shape[0]),
                   "centroid": (image.shape[1] / 2, image.shape[0] / 2)}

    subtype = dc.classify_defect_subtype(metrics)
    profile = {
        "seed_path": str(seed_defect),
        "subtype": subtype,
        "linearity": round(metrics["linearity"], 4),
        "solidity": round(metrics["solidity"], 4),
        "extent": round(metrics["extent"], 4),
        "aspect_ratio": round(metrics["aspect_ratio"], 4),
        "mask_path": str(mask_path),
        "segmentation_method": seg_method,
    }
    save_json(profile, out / "seed_profile.json")


def _characterize_single_seed_worker(args_tuple: tuple) -> str | None:
    """Module-level worker for run_parallel (must be pickle-safe).

    Args:
        args_tuple: (seed_defect_str, output_dir_str, model_checkpoint_or_None)

    Returns:
        output_dir string on success, None on failure.
    """
    seed_defect, output_dir, model_checkpoint = args_tuple
    try:
        run_seed_characterization(seed_defect, output_dir, model_checkpoint)
        return output_dir
    except Exception:
        return None


def run_seed_characterization_batch(
    tasks: List[Tuple[str, str]],
    workers: int = 0,
    model_checkpoint: str = None,
) -> List[str]:
    """Run seed characterization on multiple seeds, optionally in parallel.

    Args:
        tasks: List of (seed_defect_path, output_dir) pairs.
        workers: 0=sequential, -1=auto, N=N workers.
        model_checkpoint: Optional SAM checkpoint path.

    Returns:
        List of output_dir strings that were successfully written.
    """
    if not tasks:
        return []
    num_workers = resolve_workers(workers)
    parallel_tasks = [
        (seed_defect, output_dir, model_checkpoint)
        for seed_defect, output_dir in tasks
    ]
    return run_parallel(
        _characterize_single_seed_worker,
        parallel_tasks,
        num_workers,
        desc="Stage1b seed characterization",
    )


def main():
    parser = argparse.ArgumentParser(description="AROMA Stage 1b: Seed Characterization")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--seed_defect", help="Single seed image path")
    group.add_argument(
        "--seed_dir",
        help="Directory containing seed images (*.png). Each seed gets its own "
        "output sub-directory under --output_dir named by the seed's stem.",
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_checkpoint", default=None)
    parser.add_argument(
        "--workers", type=int, default=0,
        help="병렬 워커 수 (0=순차 처리, -1=자동 감지, N>=2=N개 프로세스 병렬 처리). "
        "Single-seed mode (--seed_defect) ignores this. "
        "Batch mode (--seed_dir) uses it for parallel processing.",
    )
    args = parser.parse_args()

    if args.seed_defect:
        # Single-seed mode (backward-compatible)
        run_seed_characterization(
            args.seed_defect, args.output_dir, args.model_checkpoint
        )
    else:
        # Batch mode: discover seeds in seed_dir
        seed_dir = Path(args.seed_dir)
        seeds = sorted(seed_dir.glob("*.png"))
        if not seeds:
            print(f"No .png seeds found in {seed_dir}")
            return
        out_base = Path(args.output_dir)
        tasks = [
            (str(s), str(out_base / s.stem))
            for s in seeds
        ]
        results = run_seed_characterization_batch(
            tasks, workers=args.workers, model_checkpoint=args.model_checkpoint,
        )
        print(f"Stage 1b: {len(results)}/{len(tasks)} seeds characterized.")

if __name__ == "__main__":
    main()
