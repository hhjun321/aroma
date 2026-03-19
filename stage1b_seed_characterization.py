import argparse
import cv2
import numpy as np
from pathlib import Path
from utils.io import save_json
from utils.mask import save_mask
from utils.defect_characterization import DefectCharacterizer


def extract_seed_mask(image: np.ndarray, checkpoint: str = None) -> np.ndarray:
    """Extract binary defect mask from seed image via SAM or Otsu fallback."""
    try:
        from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
        ckpt = checkpoint or "sam_vit_b_01ec64.pth"
        sam = sam_model_registry["vit_b"](checkpoint=ckpt)
        generator = SamAutomaticMaskGenerator(sam)
        masks = generator.generate(image)
        if masks:
            # Select smallest non-background mask (likely the defect)
            sorted_masks = sorted(masks, key=lambda m: m["area"])
            return (sorted_masks[0]["segmentation"].astype(np.uint8)) * 255
    except Exception:
        pass
    # Otsu fallback
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binary


def run_seed_characterization(seed_defect: str, output_dir: str,
                               model_checkpoint: str = None):
    image = cv2.imread(seed_defect)
    if image is None:
        raise FileNotFoundError(f"Seed image not found: {seed_defect}")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    mask_bin = extract_seed_mask(image, model_checkpoint)
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
    }
    save_json(profile, out / "seed_profile.json")


def main():
    parser = argparse.ArgumentParser(description="AROMA Stage 1b: Seed Characterization")
    parser.add_argument("--seed_defect", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_checkpoint", default=None)
    parser.add_argument(
        "--workers", type=int, default=0,
        help="병렬 워커 수 (0=순차 처리, -1=자동 감지, N>=2=N개 프로세스 병렬 처리)"
        " — Stage 1b processes a single seed; this argument is accepted for CLI"
        " consistency but has no effect."
    )
    args = parser.parse_args()
    run_seed_characterization(args.seed_defect, args.output_dir, args.model_checkpoint)

if __name__ == "__main__":
    main()
