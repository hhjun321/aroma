"""stage2_defect_seed_generation.py

Stage 2 of the AROMA pipeline: subtype-aware TF-IDG defect seed generation.

Given a single seed defect image, generate N augmented variants using
subtype-specific elastic warping strategies (Training-Free Industrial
Defect Generation — TF-IDG).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from utils.io import load_json


# ---------------------------------------------------------------------------
# Displacement-field helpers
# ---------------------------------------------------------------------------

def _base_grid(h: int, w: int):
    """Return identity remap grids (map_x, map_y) as float32 arrays."""
    map_x = np.tile(np.arange(w, dtype=np.float32), (h, 1))
    map_y = np.tile(np.arange(h, dtype=np.float32).reshape(h, 1), (1, w))
    return map_x, map_y


def _random_noise(h: int, w: int, amplitude: float, rng: np.random.Generator) -> np.ndarray:
    """Return a 2-D random displacement field, smoothed with a Gaussian."""
    noise = rng.uniform(-amplitude, amplitude, (h, w)).astype(np.float32)
    # Kernel size must be odd and at least 1; use ~1/4 of the smaller dimension
    k = max(3, (min(h, w) // 4) | 1)
    smoothed = cv2.GaussianBlur(noise, (k, k), 0)
    return smoothed


# ---------------------------------------------------------------------------
# Subtype-specific warp strategies
# ---------------------------------------------------------------------------

def _warp_linear_scratch(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Direction-preserving warp: small amplitude displacement in y only."""
    h, w = image.shape[:2]
    map_x, map_y = _base_grid(h, w)
    amplitude = max(1.0, min(h, w) * 0.15)
    map_y = map_y + _random_noise(h, w, amplitude, rng)
    return cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REFLECT_101)


def _warp_elongated(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Axis-aligned warp: stretch/compress along x-axis."""
    h, w = image.shape[:2]
    map_x, map_y = _base_grid(h, w)
    amplitude = max(1.0, w * 0.2)
    map_x = map_x + _random_noise(h, w, amplitude, rng)
    return cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REFLECT_101)


def _warp_compact_blob(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Isotropic warp: equal radial displacement in all directions."""
    h, w = image.shape[:2]
    map_x, map_y = _base_grid(h, w)
    amplitude = max(1.0, min(h, w) * 0.18)
    # Use the same noise source scaled equally for x and y
    base = _random_noise(h, w, amplitude, rng)
    map_x = map_x + base
    map_y = map_y + base
    return cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REFLECT_101)


def _warp_irregular(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Heavy elastic warp: high-amplitude random displacement field."""
    h, w = image.shape[:2]
    map_x, map_y = _base_grid(h, w)
    amplitude = max(2.0, min(h, w) * 0.35)
    map_x = map_x + _random_noise(h, w, amplitude, rng)
    map_y = map_y + _random_noise(h, w, amplitude, rng)
    return cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REFLECT_101)


def _warp_general(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Standard warp: random texture warp + optional flip and small rotation."""
    h, w = image.shape[:2]
    map_x, map_y = _base_grid(h, w)
    amplitude = max(1.5, min(h, w) * 0.25)
    map_x = map_x + _random_noise(h, w, amplitude, rng)
    map_y = map_y + _random_noise(h, w, amplitude, rng)
    result = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_REFLECT_101)

    # Random horizontal flip (50 % chance)
    if rng.random() < 0.5:
        result = cv2.flip(result, 1)

    # Small rotation ±15 °
    angle = rng.uniform(-15.0, 15.0)
    cx, cy = w / 2.0, h / 2.0
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    result = cv2.warpAffine(result, M, (w, h),
                            flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_REFLECT_101)
    return result


_WARP_STRATEGIES = {
    "linear_scratch": _warp_linear_scratch,
    "elongated": _warp_elongated,
    "compact_blob": _warp_compact_blob,
    "irregular": _warp_irregular,
    "general": _warp_general,
}


# ---------------------------------------------------------------------------
# Brightness / contrast jitter
# ---------------------------------------------------------------------------

def _brightness_contrast_jitter(image: np.ndarray,
                                 rng: np.random.Generator) -> np.ndarray:
    """Multiply each pixel by a random factor in [0.85, 1.15]."""
    factor = rng.uniform(0.85, 1.15)
    jittered = np.clip(image.astype(np.float32) * factor, 0, 255).astype(np.uint8)
    return jittered


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_seed_generation(
    seed_defect: str,
    num_variants: int,
    output_dir: str,
    seed_profile: str | None = None,
) -> None:
    """Generate *num_variants* augmented defect seeds from *seed_defect*.

    Parameters
    ----------
    seed_defect:
        Path to the source defect image (PNG/JPG).
    num_variants:
        Number of variant images to produce.
    output_dir:
        Directory where ``variant_XXXX.png`` files will be written.
    seed_profile:
        Optional path to a JSON profile produced by Stage 1b.  When
        supplied the ``subtype`` field selects the augmentation strategy.
    """
    # Load seed image
    image = cv2.imread(seed_defect)
    if image is None:
        raise FileNotFoundError(f"Cannot read seed image: {seed_defect}")

    # Determine subtype
    subtype = "general"
    if seed_profile is not None:
        profile = load_json(seed_profile)
        subtype = profile.get("subtype", "general")

    warp_fn = _WARP_STRATEGIES.get(subtype, _warp_general)

    # Prepare output directory
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng()

    for i in range(num_variants):
        variant = warp_fn(image, rng)
        variant = _brightness_contrast_jitter(variant, rng)
        filename = out_path / f"variant_{i:04d}.png"
        cv2.imwrite(str(filename), variant)


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Stage 2 — subtype-aware TF-IDG defect seed generation"
    )
    p.add_argument("--seed_defect", required=True,
                   help="Path to the input seed defect image.")
    p.add_argument("--seed_profile", default=None,
                   help="Path to the seed characterisation JSON (Stage 1b output).")
    p.add_argument("--num_variants", type=int, default=10,
                   help="Number of variant images to generate.")
    p.add_argument("--output_dir", required=True,
                   help="Directory to write variant images into.")
    return p


def main() -> None:
    args = _build_parser().parse_args()
    run_seed_generation(
        seed_defect=args.seed_defect,
        num_variants=args.num_variants,
        output_dir=args.output_dir,
        seed_profile=args.seed_profile,
    )


if __name__ == "__main__":
    main()
