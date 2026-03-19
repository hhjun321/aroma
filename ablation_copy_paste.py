# ablation_copy_paste.py
"""
Ablation Study 1: Copy-paste synthesis without Poisson blending.
Baseline to isolate the contribution of MPB synthesis method.
"""
import argparse
import cv2
import numpy as np
from pathlib import Path
from utils.io import load_json, save_json


def _transform_patch(patch: np.ndarray, scale: float, rotation: float) -> np.ndarray:
    h, w = patch.shape[:2]
    if scale != 1.0:
        patch = cv2.resize(patch, (int(w * scale), int(h * scale)))
    if rotation != 0:
        h2, w2 = patch.shape[:2]
        M = cv2.getRotationMatrix2D((w2 / 2, h2 / 2), rotation, 1.0)
        patch = cv2.warpAffine(patch, M, (w2, h2))
    return patch


def run_copy_paste(placement_map: str, image_dir: str, output_dir: str,
                   format: str = "cls") -> None:
    placements = load_json(placement_map)
    img_dir = Path(image_dir)
    out = Path(output_dir)

    for entry in placements:
        image_id = entry["image_id"]
        # Find background image
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
            x, y = int(p["x"]), int(p["y"])

            # Clip to background bounds
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(bg.shape[1], x + pw)
            y2 = min(bg.shape[0], y + ph)
            px1 = x1 - x
            py1 = y1 - y
            px2 = px1 + (x2 - x1)
            py2 = py1 + (y2 - y1)

            if x2 > x1 and y2 > y1:
                # Direct copy-paste (no blending)
                result[y1:y2, x1:x2] = patch[py1:py2, px1:px2]
                bboxes.append((x1, y1, x2 - x1, y2 - y1))

        # Save output
        if format == "yolo":
            img_out = out / "images"
            lbl_out = out / "labels"
            img_out.mkdir(parents=True, exist_ok=True)
            lbl_out.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(img_out / f"{image_id}.png"), result)
            H, W = result.shape[:2]
            with open(lbl_out / f"{image_id}.txt", "w") as f:
                for (bx, by, bw, bh) in bboxes:
                    cx = (bx + bw / 2) / W
                    cy = (by + bh / 2) / H
                    nw = bw / W
                    nh = bh / H
                    f.write(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")
        else:
            defect_dir = out / "defect"
            defect_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(defect_dir / f"{image_id}.png"), result)


def main():
    parser = argparse.ArgumentParser(description="AROMA Ablation 1: Copy-Paste (no blending)")
    parser.add_argument("--placement_map", required=True)
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--format", choices=["cls", "yolo"], default="cls")
    args = parser.parse_args()
    run_copy_paste(args.placement_map, args.image_dir, args.output_dir, args.format)

if __name__ == "__main__":
    main()
