"""stage4_mpb_synthesis.py — Modified Poisson Blending (MPB) synthesis stage.

Composites defect patches onto background images using cv2.seamlessClone
(Poisson blending) guided by a placement_map.json produced by Stage 3.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional

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


def _blend_patch_fast(
    bg: np.ndarray,
    patch: np.ndarray,
    x: int,
    y: int,
) -> np.ndarray:
    """Blend patch into bg at (x, y) using Gaussian soft-mask alpha compositing.

    ~10-30× faster than seamlessClone (no Poisson iteration).
    Returns the composited image (same shape as bg).
    """
    ph, pw = patch.shape[:2]
    ih, iw = bg.shape[:2]

    # Scale down patch if it exceeds background
    if pw > iw or ph > ih:
        scale = min(iw / pw, ih / ph) * 0.9
        patch = cv2.resize(patch, (max(1, int(pw * scale)), max(1, int(ph * scale))))
        ph, pw = patch.shape[:2]

    cx = max(pw // 2 + 1, min(iw - pw // 2 - 1, x + pw // 2))
    cy = max(ph // 2 + 1, min(ih - ph // 2 - 1, y + ph // 2))
    top  = max(0, min(ih - ph, cy - ph // 2))
    left = max(0, min(iw - pw, cx - pw // 2))
    rh = min(ph, ih - top)
    rw = min(pw, iw - left)
    if rh <= 0 or rw <= 0:
        return bg

    ksize = max(3, (min(rh, rw) // 6) | 1)
    alpha = np.ones((rh, rw), dtype=np.float32)
    alpha = cv2.GaussianBlur(alpha, (ksize, ksize), 0)[..., np.newaxis]

    result = bg.copy().astype(np.float32)
    roi = result[top:top + rh, left:left + rw]
    roi[:] = patch[:rh, :rw].astype(np.float32) * alpha + roi * (1.0 - alpha)
    return result.astype(np.uint8)


def _synthesize_image(
    image_path_str: str,
    placements: List[dict],
    output_dir_str: str,
    fmt: str,
    use_fast_blend: bool = False,
) -> List[str]:
    """Composite all placements onto one background image and write output files.

    Returns a list of written file paths (may be empty if the image cannot be loaded).
    """
    bg_path = Path(image_path_str)
    image_id = bg_path.stem
    out_path = Path(output_dir_str)

    bg = cv2.imread(str(bg_path))
    if bg is None:
        return []

    composited = bg.copy()
    ih, iw = bg.shape[:2]
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

        patch = _transform_patch(patch, scale, rotation)
        ph, pw = patch.shape[:2]
        blend_fn = _blend_patch_fast if use_fast_blend else _blend_patch
        composited = blend_fn(composited, patch, x, y)

        if fmt == "yolo":
            cx = x + pw // 2
            cy = y + ph // 2
            cx = max(pw // 2 + 1, min(iw - pw // 2 - 1, cx))
            cy = max(ph // 2 + 1, min(ih - ph // 2 - 1, cy))
            bx = cx - pw // 2
            by = cy - ph // 2
            norm_cx = (bx + pw / 2) / iw
            norm_cy = (by + ph / 2) / ih
            norm_w = pw / iw
            norm_h = ph / ih
            yolo_boxes.append(f"0 {norm_cx:.6f} {norm_cy:.6f} {norm_w:.6f} {norm_h:.6f}")

    written: List[str] = []
    if fmt == "cls":
        cls_dir = out_path / "defect"
        cls_dir.mkdir(parents=True, exist_ok=True)
        out_img = str(cls_dir / f"{image_id}.png")
        cv2.imwrite(out_img, composited)
        written.append(out_img)
    elif fmt == "yolo":
        images_dir = out_path / "images"
        labels_dir = out_path / "labels"
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
        out_img = str(images_dir / f"{image_id}.png")
        cv2.imwrite(out_img, composited)
        written.append(out_img)
        label_path = labels_dir / f"{image_id}.txt"
        label_path.write_text("\n".join(yolo_boxes))
        written.append(str(label_path))
    else:
        raise ValueError(f"Unknown format: {fmt!r}. Use 'cls' or 'yolo'.")
    return written


def _synthesize_single_image_worker(args_tuple):
    """Module-level worker for MPB synthesis (pickle-safe)."""
    image_path_str, placements, output_dir_str, fmt, use_fast_blend = args_tuple
    image = cv2.imread(image_path_str)
    if image is None:
        return None
    return _synthesize_image(image_path_str, placements, output_dir_str, fmt, use_fast_blend)


def _synthesize_image_multi_seed_worker(args_tuple):
    """Worker that composites one background image against multiple seeds.

    Loads the background once and writes one output per seed — avoids redundant I/O.

    args_tuple: (bg_path_str, seed_entries, output_root_str, fmt, use_fast_blend)
        seed_entries: list of (seed_id, placements)
    """
    bg_path_str, seed_entries, output_root_str, fmt, use_fast_blend = args_tuple
    bg = cv2.imread(bg_path_str)
    if bg is None:
        return
    image_id = Path(bg_path_str).stem
    ih, iw = bg.shape[:2]

    for seed_id, placements in seed_entries:
        composited = bg.copy()
        yolo_boxes: List[str] = []

        for placement in placements:
            patch = cv2.imread(placement["defect_path"])
            if patch is None:
                continue
            patch = _transform_patch(
                patch,
                float(placement.get("scale", 1.0)),
                float(placement.get("rotation", 0)),
            )
            ph, pw = patch.shape[:2]
            x, y = int(placement.get("x", 0)), int(placement.get("y", 0))
            blend_fn = _blend_patch_fast if use_fast_blend else _blend_patch
            composited = blend_fn(composited, patch, x, y)

            if fmt == "yolo":
                cx = max(pw // 2 + 1, min(iw - pw // 2 - 1, x + pw // 2))
                cy = max(ph // 2 + 1, min(ih - ph // 2 - 1, y + ph // 2))
                bx, by = cx - pw // 2, cy - ph // 2
                yolo_boxes.append(
                    f"0 {(bx + pw/2)/iw:.6f} {(by + ph/2)/ih:.6f}"
                    f" {pw/iw:.6f} {ph/ih:.6f}"
                )

        seed_out = Path(output_root_str) / seed_id
        if fmt == "cls":
            d = seed_out / "defect"
            d.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(d / f"{image_id}.png"), composited)
        elif fmt == "yolo":
            (seed_out / "images").mkdir(parents=True, exist_ok=True)
            (seed_out / "labels").mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(seed_out / "images" / f"{image_id}.png"), composited)
            (seed_out / "labels" / f"{image_id}.txt").write_text("\n".join(yolo_boxes))


def run_synthesis_batch(
    image_dir: str,
    seed_placement_maps: List[tuple],
    output_root: str,
    format: str = "cls",
    use_fast_blend: bool = False,
    workers: int = 0,
) -> None:
    """Category-level batch synthesis with background caching.

    Loads each background image once and composites all seeds' placements in a
    single pass — reduces I/O by N_seeds factor compared to calling run_synthesis
    once per seed.

    Args:
        image_dir:          Directory containing background images (<image_id>.png).
        seed_placement_maps: List of (seed_id, placement_map_path) tuples.
        output_root:        Root output directory; outputs go to output_root/<seed_id>/.
        format:             'cls' or 'yolo'.
        use_fast_blend:     Use Gaussian soft-mask instead of seamlessClone.
        workers:            Thread workers (0=sequential, -1=auto, N=N threads).
    """
    from utils.parallel import resolve_workers

    # Build {image_id: [(seed_id, placements), ...]}
    image_seed_map: dict = {}
    for seed_id, pm_path in seed_placement_maps:
        for entry in load_json(pm_path):
            image_seed_map.setdefault(entry["image_id"], []).append(
                (seed_id, entry.get("placements", []))
            )

    img_dir = Path(image_dir)
    tasks = [
        (str(img_dir / f"{image_id}.png"),
         seed_entries,
         output_root,
         format,
         use_fast_blend)
        for image_id, seed_entries in image_seed_map.items()
        if (img_dir / f"{image_id}.png").exists()
    ]

    num_workers = resolve_workers(workers)
    if num_workers <= 1:
        for t in tasks:
            _synthesize_image_multi_seed_worker(t)
    else:
        with ThreadPoolExecutor(max_workers=num_workers) as ex:
            list(ex.map(_synthesize_image_multi_seed_worker, tasks))


def run_synthesis(
    image_dir: str,
    placement_map: str,
    output_dir: str,
    format: str = "cls",
    workers: int = 0,
    use_fast_blend: bool = False,
) -> None:
    """Composite defect patches onto backgrounds and write the augmented dataset.

    Args:
        image_dir:     Directory containing background images (<image_id>.png).
        placement_map: Path to placement_map.json produced by Stage 3.
        output_dir:    Root output directory.
        format:        Output format — 'cls' (classification) or 'yolo' (detection).
        workers:       Number of parallel workers (0=sequential, -1=auto, N>=2=N processes).
    """
    from utils.parallel import resolve_workers, run_parallel

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    entries: List[dict] = load_json(placement_map)
    img_dir = Path(image_dir)

    tasks = []
    for entry in entries:
        image_id: str = entry["image_id"]
        placements: List[dict] = entry.get("placements", [])

        bg_path = img_dir / f"{image_id}.png"
        if not bg_path.exists():
            bg_path = img_dir / image_id
        if not bg_path.exists():
            continue

        tasks.append((str(bg_path), placements, str(out_path), format, use_fast_blend))

    num_workers = resolve_workers(workers)
    run_parallel(
        _synthesize_single_image_worker,
        tasks,
        num_workers,
        desc=f"Stage4 MPB synthesis (workers={num_workers})",
    )


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
    parser.add_argument("--workers", type=int, default=0,
                        help="병렬 워커 수 (0=순차 처리, -1=자동 감지, N>=2=N개 프로세스 병렬 처리)")
    parser.add_argument("--fast_blend", action="store_true",
                        help="Gaussian soft-mask 합성 사용 (seamlessClone 대비 ~10-30× 빠름)")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_synthesis(
        image_dir=args.image_dir,
        placement_map=args.placement_map,
        output_dir=args.output_dir,
        format=args.format,
        workers=args.workers,
        use_fast_blend=args.fast_blend,
    )
