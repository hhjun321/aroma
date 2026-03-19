# ablation_random_placement.py
"""
Ablation Study 2: Random placement ignoring ROI suitability.
Baseline to isolate the contribution of suitability-guided placement strategy.
"""
import argparse
import cv2
import numpy as np
import random
from pathlib import Path
from utils.io import load_json


def _transform_patch(patch: np.ndarray, scale: float, rotation: float) -> np.ndarray:
    h, w = patch.shape[:2]
    if scale != 1.0:
        patch = cv2.resize(patch, (int(w * scale), int(h * scale)))
    if rotation != 0:
        h2, w2 = patch.shape[:2]
        M = cv2.getRotationMatrix2D((w2 / 2, h2 / 2), rotation, 1.0)
        patch = cv2.warpAffine(patch, M, (w2, h2))
    return patch


def run_random_placement(placement_map: str, image_dir: str, output_dir: str,
                         format: str = "cls", seed: int = 42) -> None:
    rng = random.Random(seed)
    placements = load_json(placement_map)
    img_dir = Path(image_dir)
    out = Path(output_dir)

    for entry in placements:
        image_id = entry["image_id"]
        bg_candidates = list(img_dir.glob(f"{image_id}.*"))
        if not bg_candidates:
            continue
        bg = cv2.imread(str(bg_candidates[0]))
        if bg is None:
            continue
        result = bg.copy()
        bboxes = []

        for p in entry["placements"]:
            patch = cv2.imread(p["defect_path"])
            if patch is None:
                continue
            patch = _transform_patch(patch, p.get("scale", 1.0), p.get("rotation", 0))
            ph, pw = patch.shape[:2]
            H, W = bg.shape[:2]

            # Random placement (ignore suitability)
            x = rng.randint(0, max(0, W - pw))
            y = rng.randint(0, max(0, H - ph))

            x1, y1 = x, y
            x2 = min(W, x + pw)
            y2 = min(H, y + ph)
            px2 = x2 - x1
            py2 = y2 - y1

            if x2 > x1 and y2 > y1:
                # Use Poisson blending for fair comparison (only placement differs)
                try:
                    center = (x1 + pw // 2, y1 + ph // 2)
                    center = (max(pw // 2, min(W - pw // 2 - 1, center[0])),
                              max(ph // 2, min(H - ph // 2 - 1, center[1])))
                    mask = 255 * np.ones(patch.shape[:2], dtype=np.uint8)
                    result = cv2.seamlessClone(patch, result, mask, center, cv2.NORMAL_CLONE)
                except cv2.error:
                    result[y1:y2, x1:x2] = patch[:py2, :px2]
                bboxes.append((x1, y1, x2 - x1, y2 - y1))

        if format == "yolo":
            img_out = out / "images"
            lbl_out = out / "labels"
            img_out.mkdir(parents=True, exist_ok=True)
            lbl_out.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(img_out / f"{image_id}.png"), result)
            H2, W2 = result.shape[:2]
            with open(lbl_out / f"{image_id}.txt", "w") as f:
                for (bx, by, bw, bh) in bboxes:
                    cx = (bx + bw / 2) / W2
                    cy = (by + bh / 2) / H2
                    nw = bw / W2
                    nh = bh / H2
                    f.write(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")
        else:
            defect_dir = out / "defect"
            defect_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(defect_dir / f"{image_id}.png"), result)


def main():
    parser = argparse.ArgumentParser(description="AROMA Ablation 2: Random Placement")
    parser.add_argument("--placement_map", required=True)
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--format", choices=["cls", "yolo"], default="cls")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_random_placement(args.placement_map, args.image_dir, args.output_dir,
                         args.format, args.seed)

if __name__ == "__main__":
    main()
