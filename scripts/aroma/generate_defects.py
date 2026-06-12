#!/usr/bin/env python3
"""
AROMA Step 4 — Defect Generation

Reads Step 3 ROI candidates and generates synthetic defect images by
compositing a defect mask/crop onto a normal background image.

Methods (--method):
    copy_paste    Alpha-composite paste (default, GPU-free)
    controlnet    Interface stub — implement in Colab with diffusers
    inpainting    Interface stub — implement in Colab with diffusers

Copy-paste blending:
    Poisson blending requires scipy and is used when available.
    Falls back to alpha-composite when scipy is absent or blend_mode=alpha.

Usage (Colab):
    !python $AROMA_SCRIPTS/generate_defects.py \
        --roi_dir        $AROMA_OUT/roi/isp_LSM_1 \
        --normal_dir     $AROMA_DATA_BASE/isp/LSM_1/train/good \
        --output_dir     $AROMA_OUT/synthetic/isp_LSM_1 \
        --method         copy_paste \
        --n_per_roi      3 \
        --blend_mode     alpha

Outputs (written to --output_dir):
    images/          synthetic defect images
    annotations.json list of {image_path, source_roi, prompt, method, ...}
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("aroma.generate")

# ---------------------------------------------------------------------------
# Optional imaging libraries — graceful fallback
# ---------------------------------------------------------------------------

try:
    from PIL import Image as _PILImage
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    _PILImage = None  # type: ignore[assignment]
    logger.warning("Pillow not found — copy_paste synthesis unavailable.")

try:
    from scipy.ndimage import gaussian_filter as _gauss
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    _gauss = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# I/O bootstrap
# ---------------------------------------------------------------------------

def _bootstrap_aroma_ref() -> str:
    ref = os.environ.get("AROMA_REF") or r"D:\project\aroma"
    ref_path = Path(ref)
    if ref_path.is_dir() and str(ref_path) not in sys.path:
        sys.path.insert(0, str(ref_path))
    return "aroma_ref" if ref_path.is_dir() else "inline"


_REF_SOURCE = _bootstrap_aroma_ref()

try:
    from utils.io import load_json, save_json  # type: ignore[import]
except Exception:
    def load_json(p):  # type: ignore[misc]
        with open(p, encoding="utf-8") as f:
            return json.load(f)

    def save_json(data, p):  # type: ignore[misc]
        Path(p).parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Blending utilities (copy_paste)
# ---------------------------------------------------------------------------

def _alpha_composite(
    background: "Image",
    defect_crop: "Image",
    mask: "Image",
    position: Tuple[int, int],
    feather_px: int = 4,
) -> "Image":
    """
    Alpha-composite defect_crop onto background at position.

    Args:
        background:   RGBA PIL Image (target normal image)
        defect_crop:  RGBA PIL Image (defect to paste)
        mask:         L-mode PIL Image (defect region mask)
        position:     (x, y) top-left paste position
        feather_px:   Gaussian sigma for mask edge softening

    Returns:
        RGBA PIL Image
    """
    from PIL import Image as PILImage

    bg = background.convert("RGBA")
    crop = defect_crop.convert("RGBA")
    msk = mask.convert("L")

    if feather_px > 0 and HAS_SCIPY:
        arr = np.array(msk, dtype=np.float32) / 255.0
        arr = _gauss(arr, sigma=feather_px)
        arr = np.clip(arr * 255, 0, 255).astype(np.uint8)
        msk = PILImage.fromarray(arr, mode="L")

    # Resize crop/mask to match if different size
    if crop.size != msk.size:
        msk = msk.resize(crop.size, PILImage.LANCZOS)

    crop.putalpha(msk)
    out = bg.copy()
    out.paste(crop, position, mask=msk)
    return out


def _random_paste_position(
    bg_size: Tuple[int, int],
    crop_size: Tuple[int, int],
    rng: random.Random,
) -> Tuple[int, int]:
    """Random valid top-left position so crop fits inside background."""
    bw, bh = bg_size
    cw, ch = crop_size
    x = rng.randint(0, max(0, bw - cw))
    y = rng.randint(0, max(0, bh - ch))
    return x, y


def copy_paste_synthesis(
    roi_entry: Dict[str, Any],
    normal_image_path: str,
    output_path: str,
    blend_mode: str = "alpha",
    feather_px: int = 4,
    rng: Optional[random.Random] = None,
) -> bool:
    """
    Paste the defect region from roi_entry onto the normal image.

    Args:
        roi_entry:          A single ROI dict (from roi_selected.json)
        normal_image_path:  Path to normal (good) background image
        output_path:        Destination path for the synthetic image
        blend_mode:         'alpha' (always) — 'poisson' reserved for future
        feather_px:         Edge feathering sigma in pixels
        rng:                Optional seeded random.Random for reproducibility

    Returns:
        True on success, False on error (image missing or PIL unavailable)
    """
    if not HAS_PIL:
        logger.error("Pillow required for copy_paste — install with: pip install Pillow")
        return False

    from PIL import Image as PILImage

    if rng is None:
        rng = random.Random(42)

    defect_path = roi_entry.get("image_path", "")
    if not defect_path or not Path(defect_path).exists():
        logger.warning("Defect image not found: %s", defect_path)
        return False

    if not Path(normal_image_path).exists():
        logger.warning("Normal image not found: %s", normal_image_path)
        return False

    try:
        defect_img = PILImage.open(defect_path).convert("RGB")
        normal_img = PILImage.open(normal_image_path).convert("RGBA")

        # Use full defect image as crop; create a simple elliptical mask
        cw, ch = defect_img.size
        mask_arr = np.zeros((ch, cw), dtype=np.uint8)
        cy, cx = ch // 2, cw // 2
        ry, rx = max(1, ch // 2 - 2), max(1, cw // 2 - 2)
        Y, X = np.ogrid[:ch, :cw]
        ellipse = ((X - cx) ** 2 / (rx ** 2) + (Y - cy) ** 2 / (ry ** 2)) <= 1.0
        mask_arr[ellipse] = 255
        mask = PILImage.fromarray(mask_arr, mode="L")

        defect_rgba = defect_img.convert("RGBA")
        position = _random_paste_position(normal_img.size, defect_rgba.size, rng)
        result = _alpha_composite(normal_img, defect_rgba, mask, position, feather_px)

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        result.convert("RGB").save(output_path)
        return True

    except Exception as exc:
        logger.error("copy_paste_synthesis failed for %s: %s", defect_path, exc)
        return False


# ---------------------------------------------------------------------------
# Method stubs
# ---------------------------------------------------------------------------

def controlnet_synthesis(
    roi_entry: Dict[str, Any],
    normal_image_path: str,
    output_path: str,
    **kwargs: Any,
) -> bool:
    """
    ControlNet-based defect generation stub.
    Implement in Colab using the diffusers library with a GPU runtime.
    """
    raise NotImplementedError(
        "controlnet_synthesis is not implemented. "
        "Install diffusers and implement in a Colab GPU cell."
    )


def inpainting_synthesis(
    roi_entry: Dict[str, Any],
    normal_image_path: str,
    output_path: str,
    **kwargs: Any,
) -> bool:
    """
    Inpainting-based defect generation stub.
    Implement in Colab using the diffusers library with a GPU runtime.
    """
    raise NotImplementedError(
        "inpainting_synthesis is not implemented. "
        "Install diffusers and implement in a Colab GPU cell."
    )


_SYNTHESIS_METHODS = {
    "copy_paste":  copy_paste_synthesis,
    "controlnet":  controlnet_synthesis,
    "inpainting":  inpainting_synthesis,
}


# ---------------------------------------------------------------------------
# Local staging helpers
# ---------------------------------------------------------------------------

def _default_staging_dir(roi_dir: str) -> Path:
    """Return a local staging path. Uses /content/tmp on Colab, sibling dir otherwise."""
    slug = Path(roi_dir).name
    colab_tmp = Path("/content/tmp")
    if colab_tmp.parent.exists():
        return colab_tmp / f"aroma_step4_{slug}"
    return Path(roi_dir).parent / f"_staging_tmp_{slug}"


def _stage_inputs(
    selected: List[Dict[str, Any]],
    normal_images: List[str],
    staging_dir: Path,
) -> tuple:
    """
    Copy defect and normal images to local staging dir.
    Returns (staged_selected, staged_normal_images) with updated paths.
    """
    defect_dir = staging_dir / "defects"
    normal_dir_local = staging_dir / "normals"
    defect_dir.mkdir(parents=True, exist_ok=True)
    normal_dir_local.mkdir(parents=True, exist_ok=True)

    # Stage defect images (deduplicated)
    defect_map: Dict[str, str] = {}
    for entry in selected:
        orig = entry.get("image_path", "")
        if orig and orig not in defect_map and Path(orig).exists():
            p = Path(orig)
            dst = defect_dir / p.name
            if dst.exists():
                dst = defect_dir / f"{p.stem}_{len(defect_map)}{p.suffix}"
            shutil.copy2(orig, str(dst))
            defect_map[orig] = str(dst)

    staged_selected = []
    for entry in selected:
        e = entry.copy()
        orig = e.get("image_path", "")
        if orig in defect_map:
            e["image_path"] = defect_map[orig]
        staged_selected.append(e)

    # Stage normal images (deduplicated)
    normal_map: Dict[str, str] = {}
    for orig in normal_images:
        if orig not in normal_map and Path(orig).exists():
            p = Path(orig)
            dst = normal_dir_local / p.name
            if dst.exists():
                dst = normal_dir_local / f"{p.stem}_{len(normal_map)}{p.suffix}"
            shutil.copy2(orig, str(dst))
            normal_map[orig] = str(dst)

    staged_normal = [normal_map.get(p, p) for p in normal_images]

    logger.info(
        "Staged %d defect images + %d normal images → %s",
        len(defect_map), len(normal_map), staging_dir,
    )
    return staged_selected, staged_normal


def _push_outputs(local_img_dir: Path, drive_img_dir: Path) -> int:
    """Copy synthesized images from local staging to Drive. Returns count copied."""
    drive_img_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for img_path in sorted(local_img_dir.iterdir()):
        if img_path.is_file():
            shutil.copy2(str(img_path), str(drive_img_dir / img_path.name))
            n += 1
    return n


# ---------------------------------------------------------------------------
# Normal image pool
# ---------------------------------------------------------------------------

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


def load_normal_images(normal_dir: str) -> List[str]:
    """Collect all image paths under normal_dir (recursive)."""
    d = Path(normal_dir)
    if not d.exists():
        logger.warning("Normal image directory not found: %s", normal_dir)
        return []
    paths = [str(p) for p in d.rglob("*") if p.suffix.lower() in _IMAGE_EXTS]
    logger.info("Found %d normal images in %s", len(paths), normal_dir)
    return paths


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run(
    roi_dir: str,
    normal_dir: str,
    output_dir: str,
    method: str = "copy_paste",
    n_per_roi: int = 3,
    blend_mode: str = "alpha",
    feather_px: int = 4,
    seed: int = 42,
    local_staging: bool = False,
) -> Dict[str, Any]:
    """
    Full Step 4 pipeline: load ROIs → synthesize → save annotations.

    Args:
        roi_dir:        Step 3 output directory (roi_selected.json)
        normal_dir:     Directory of normal (good) background images
        output_dir:     Destination for synthetic images + annotations.json
        method:         Synthesis method: copy_paste | controlnet | inpainting
        n_per_roi:      Number of synthetic images to generate per ROI
        blend_mode:     Blending mode for copy_paste ('alpha')
        feather_px:     Edge feather radius for alpha blend
        seed:           Random seed for reproducibility
        local_staging:  Copy inputs to /content/tmp before synthesis to avoid
                        per-image Drive I/O. Outputs are pushed back to Drive
                        output_dir at the end.

    Returns:
        dict with status, n_generated, annotations list
    """
    roi_path = Path(roi_dir) / "roi_selected.json"
    if not roi_path.exists():
        logger.error("roi_selected.json not found in %s", roi_dir)
        return {"status": "missing_roi", "n_generated": 0, "annotations": []}

    selected: List[Dict[str, Any]] = load_json(str(roi_path))
    if not selected:
        logger.warning("roi_selected.json is empty")
        return {"status": "empty_roi", "n_generated": 0, "annotations": []}

    normal_images = load_normal_images(normal_dir)
    if not normal_images:
        logger.warning("No normal images found — dry run only")

    if method not in _SYNTHESIS_METHODS:
        return {"status": f"unknown_method:{method}", "n_generated": 0, "annotations": []}

    # Local staging: copy inputs to fast local disk, run synthesis there
    staging_dir: Optional[Path] = None
    work_output_dir = output_dir

    if local_staging and normal_images:
        staging_dir = _default_staging_dir(roi_dir)
        selected, normal_images = _stage_inputs(selected, normal_images, staging_dir)
        work_output_dir = str(staging_dir / "synthetic")
        logger.info("Synthesis will run in local staging dir: %s", work_output_dir)

    synthesis_fn = _SYNTHESIS_METHODS[method]
    out = Path(work_output_dir)
    img_dir = out / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    annotations: List[Dict[str, Any]] = []
    n_ok = 0
    n_skip = 0

    drive_img_dir = Path(output_dir) / "images"

    for roi_idx, roi_entry in enumerate(selected):
        for rep_idx in range(n_per_roi):
            fname = f"syn_{roi_idx:05d}_{rep_idx:02d}.jpg"
            local_out_path = str(img_dir / fname)
            # Annotation stores the final Drive path
            final_out_path = str(drive_img_dir / fname) if local_staging else local_out_path

            if not normal_images:
                annotations.append({
                    "image_path":    final_out_path,
                    "source_roi":    roi_entry.get("image_path", ""),
                    "cluster_id":    roi_entry.get("cluster_id"),
                    "cell_key":      roi_entry.get("cell_key", ""),
                    "prompt":        roi_entry.get("prompt", ""),
                    "method":        method,
                    "blend_mode":    blend_mode,
                    "roi_score":     roi_entry.get("roi_score", 0.0),
                    "deficit":       roi_entry.get("deficit", 0.0),
                    "dry_run":       True,
                })
                n_ok += 1
                continue

            normal_path = rng.choice(normal_images)
            ok = synthesis_fn(
                roi_entry=roi_entry,
                normal_image_path=normal_path,
                output_path=local_out_path,
                blend_mode=blend_mode,
                feather_px=feather_px,
                rng=rng,
            )
            if ok:
                n_ok += 1
                annotations.append({
                    "image_path":    final_out_path,
                    "source_roi":    roi_entry.get("image_path", ""),
                    "normal_image":  normal_path,
                    "cluster_id":    roi_entry.get("cluster_id"),
                    "cell_key":      roi_entry.get("cell_key", ""),
                    "prompt":        roi_entry.get("prompt", ""),
                    "method":        method,
                    "blend_mode":    blend_mode,
                    "roi_score":     roi_entry.get("roi_score", 0.0),
                    "deficit":       roi_entry.get("deficit", 0.0),
                    "dry_run":       False,
                })
            else:
                n_skip += 1

    # Push synthesized images to Drive output_dir if staging was used
    if local_staging and staging_dir and img_dir.exists():
        n_copied = _push_outputs(img_dir, drive_img_dir)
        logger.info("Pushed %d images to Drive → %s", n_copied, drive_img_dir)

    drive_out = Path(output_dir)
    drive_out.mkdir(parents=True, exist_ok=True)
    save_json(annotations, str(drive_out / "annotations.json"))
    logger.info("Generated %d images (%d skipped) → %s", n_ok, n_skip, drive_out)

    return {
        "status":       "ok",
        "n_generated":  n_ok,
        "n_skipped":    n_skip,
        "annotations":  annotations,
        "method":       method,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="AROMA Step 4 — Synthetic Defect Generation"
    )
    p.add_argument("--roi_dir",     required=True,
                   help="Step 3 output directory (roi_selected.json)")
    p.add_argument("--normal_dir",  required=True,
                   help="Directory of normal/good background images")
    p.add_argument("--output_dir",  required=True,
                   help="Destination for synthetic images and annotations.json")
    p.add_argument("--method",      default="copy_paste",
                   choices=list(_SYNTHESIS_METHODS),
                   help="Synthesis method (default: copy_paste)")
    p.add_argument("--n_per_roi",   type=int, default=3,
                   help="Synthetic images per ROI (default: 3)")
    p.add_argument("--blend_mode",  default="alpha",
                   choices=["alpha"],
                   help="Blending mode for copy_paste (default: alpha)")
    p.add_argument("--feather_px",  type=int, default=4,
                   help="Edge feather sigma in pixels (default: 4)")
    p.add_argument("--seed",          type=int, default=42,
                   help="Random seed (default: 42)")
    p.add_argument("--local_staging", action="store_true",
                   help="Stage inputs to /content/tmp before synthesis to avoid "
                        "per-image Drive I/O (recommended on Colab)")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = _parse_args(argv)
    result = run(
        roi_dir=args.roi_dir,
        normal_dir=args.normal_dir,
        output_dir=args.output_dir,
        method=args.method,
        n_per_roi=args.n_per_roi,
        blend_mode=args.blend_mode,
        feather_px=args.feather_px,
        seed=args.seed,
        local_staging=args.local_staging,
    )
    if result.get("status") != "ok":
        sys.exit(1)


if __name__ == "__main__":
    main()
