#!/usr/bin/env python3
"""
AROMA Step 4 — Defect Generation

Reads Step 3 ROI candidates and generates synthetic defect images by
compositing a defect mask/crop onto a normal background image.

Methods (--method):
    copy_paste    Alpha-composite paste (default, GPU-free)
    controlnet    Trained-ControlNet texture generation → paste at the AROMA-
                  selected ROI (GT mask stays the real seed mask). Requires
                  --controlnet_path/--morphology_csv/--context_features/--config
                  and a GPU runtime (diffusers loads lazily).
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
import re
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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

try:
    import cv2  # type: ignore[import]
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    cv2 = None  # type: ignore[assignment]


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
        msk = PILImage.fromarray(arr)

    # Resize crop/mask to match if different size
    if crop.size != msk.size:
        msk = msk.resize(crop.size, PILImage.LANCZOS)

    crop.putalpha(msk)
    out = bg.copy()
    out.paste(crop, position, mask=msk)
    return out


def _reinhard_transfer(
    src_rgb: np.ndarray,
    ref_rgb: np.ndarray,
) -> np.ndarray:
    """
    Reinhard color/illumination transfer in Lab space.

    Matches the per-channel mean/std of ``src_rgb`` (the defect crop) to those of
    ``ref_rgb`` (the local background patch at the paste position) so the pasted
    defect picks up the local illumination/colour cast → placement becomes
    measurable in pixels.

    When the reference background patch is effectively grayscale (near-zero a/b
    chroma std, e.g. severstal steel), only the L (luminance) channel is matched
    — matching a/b against degenerate statistics would inject false colour.

    Deterministic (pure statistics, no RNG). Requires cv2; callers guard on
    HAS_CV2 and fall back before reaching here.

    Args:
        src_rgb: HxWx3 uint8 RGB defect crop.
        ref_rgb: hxwx3 uint8 RGB local background patch (any size > 0).

    Returns:
        HxWx3 uint8 RGB crop with transferred colour/illumination statistics.
    """
    # Lab via cv2 expects uint8 BGR/RGB; use RGB→LAB and operate in float.
    src_lab = cv2.cvtColor(src_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    ref_lab = cv2.cvtColor(ref_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)

    out = src_lab.copy()
    # Chroma flatness on the reference patch decides grayscale vs full transfer.
    ref_a_std = float(ref_lab[..., 1].std())
    ref_b_std = float(ref_lab[..., 2].std())
    grayscale_ref = (ref_a_std < 1.0) and (ref_b_std < 1.0)
    channels = (0,) if grayscale_ref else (0, 1, 2)

    for c in channels:
        s_mean = float(src_lab[..., c].mean())
        s_std = float(src_lab[..., c].std())
        r_mean = float(ref_lab[..., c].mean())
        r_std = float(ref_lab[..., c].std())
        if s_std < 1e-6:
            # Degenerate source channel — shift mean only, no scaling.
            out[..., c] = src_lab[..., c] - s_mean + r_mean
        else:
            out[..., c] = (src_lab[..., c] - s_mean) * (r_std / s_std) + r_mean

    out = np.clip(out, 0, 255).astype(np.uint8)
    return cv2.cvtColor(out, cv2.COLOR_LAB2RGB)


def _context_aware_composite(
    background: "Image",
    defect_crop: "Image",
    mask: "Image",
    position: Tuple[int, int],
    feather_px: int = 4,
) -> "Image":
    """
    Context-aware ('seamless') blend: Reinhard colour/illumination transfer from
    the local background patch into the defect crop, then gradient-domain
    cv2.seamlessClone(NORMAL_CLONE).

    The position decision (foreground / void / clean-bg gate) is made entirely by
    the caller; this function only changes *how* the already-chosen crop is
    composited at the already-chosen position. On ANY failure (cv2 missing, mask
    too small, clone centre at the border, OpenCV exception) it falls back to
    ``_alpha_composite`` so synthesis is never silently dropped and determinism
    is preserved.

    Args:
        background:   RGBA PIL Image (target normal image)
        defect_crop:  RGBA PIL Image (defect to paste)
        mask:         L-mode PIL Image (defect region mask, crop-sized)
        position:     (x, y) top-left paste position in background coords
        feather_px:   passed through to the alpha fallback only

    Returns:
        RGBA PIL Image.
    """
    from PIL import Image as PILImage

    if not HAS_CV2:
        return _alpha_composite(background, defect_crop, mask, position, feather_px)

    try:
        bg = background.convert("RGBA")
        crop = defect_crop.convert("RGB")
        msk = mask.convert("L")
        if crop.size != msk.size:
            msk = msk.resize(crop.size, PILImage.LANCZOS)

        crop_w, crop_h = crop.size
        bg_w, bg_h = bg.size
        x0, y0 = int(position[0]), int(position[1])

        # The crop must lie fully inside the background for seamlessClone; the
        # caller already guarantees crop fits, but clamp defensively.
        if (crop_w <= 0 or crop_h <= 0 or crop_w > bg_w or crop_h > bg_h
                or x0 < 0 or y0 < 0
                or x0 + crop_w > bg_w or y0 + crop_h > bg_h):
            return _alpha_composite(background, defect_crop, mask, position, feather_px)

        mask_arr = np.asarray(msk, dtype=np.uint8)
        # seamlessClone needs a non-trivial mask region; tiny masks (or all-zero)
        # make the Poisson solver degenerate → fall back.
        if int((mask_arr > 0).sum()) < 16:
            return _alpha_composite(background, defect_crop, mask, position, feather_px)

        bg_rgb = np.asarray(bg.convert("RGB"))
        crop_rgb = np.asarray(crop)

        # Local background patch at the paste position (crop-sized, in bounds).
        local_bg = bg_rgb[y0:y0 + crop_h, x0:x0 + crop_w]
        if local_bg.shape[0] == crop_h and local_bg.shape[1] == crop_w and local_bg.size > 0:
            crop_rgb = _reinhard_transfer(crop_rgb, local_bg)

        # cv2.seamlessClone composites src into dst at a CENTRE point.
        center = (x0 + crop_w // 2, y0 + crop_h // 2)
        # Binary mask (0/255) for the clone.
        clone_mask = np.where(mask_arr > 0, 255, 0).astype(np.uint8)

        # OpenCV works in BGR.
        src_bgr = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR)
        dst_bgr = cv2.cvtColor(bg_rgb, cv2.COLOR_RGB2BGR)
        blended_bgr = cv2.seamlessClone(
            src_bgr, dst_bgr, clone_mask, center, cv2.NORMAL_CLONE
        )
        blended_rgb = cv2.cvtColor(blended_bgr, cv2.COLOR_BGR2RGB)

        out = PILImage.fromarray(blended_rgb).convert("RGBA")
        # Preserve any alpha that was present in the original background.
        if "A" in bg.getbands():
            out.putalpha(bg.getchannel("A"))
        return out
    except Exception as exc:
        logger.warning(
            "seamless blend failed (%s) — falling back to alpha composite.", exc
        )
        return _alpha_composite(background, defect_crop, mask, position, feather_px)


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


# Foreground void-rejection thresholds (Mode A — see
# .claude/.dev_note/aroma_exp4v2_foreground-void-rejection.md). When the
# corner-vote selects a flat void region as "foreground" (large empty margins),
# a defect would otherwise be pasted onto the void. BRIGHTNESS-AGNOSTIC: the
# guard fires on FLATNESS (std) confirmed by a 2-D bbox-crop quality score —
# both invariant to whether the void is dark (severstal) or bright (aitex).
# Colab-tunable.
_FG_VOID_STD = 5.0       # selected-fg pixel std below this → flat void candidate
                         # (4-domain context_features: void collapses to
                         #  local_variance≈0 i.e. std<~3 regardless of brightness;
                         #  leather/mtd have no flat mode → 0% false reject)
_FG_VOID_MEAN = 25.0     # DEPRECATED (no longer used) — was a dark-only condition
                         # that silently passed bright/white voids (aitex). Kept
                         # only to avoid breaking any external import.
_FG_VOID_QUALITY = 0.5   # bbox-crop _background_quality_score below this → void
                         # (flat void ~0.43; below the 0.7 background gate so
                         #  textured objects are never misjudged)


def _foreground_mask(normal_img: np.ndarray) -> Optional[np.ndarray]:
    """
    Estimate the object foreground region of a normal (good) image.

    Strategy:
        1. Convert to grayscale and Otsu-threshold (binary split).
        2. Decide which class is BACKGROUND by majority vote over the four
           corner pixels — this handles both bright-object/dark-background and
           dark-object/bright-background datasets. Foreground = the other class.
        3. Keep only the largest connected component (drops speckle / texture).
        4. Reject degenerate splits: if foreground area is <2% or >90% of the
           frame, object/background separation is meaningless → return None so
           the caller falls back to random placement.
        5. Reject VOID foreground: if the selected foreground is flat AND dark
           (a void mis-selected as object, e.g. severstal margins), return None
           so the caller falls back — WITHOUT inverting the corner-vote polarity,
           so genuine dark-object datasets are unaffected. Effectiveness is
           maximized when the caller also passes --reject-clean-bg (default OFF),
           which keeps the random fallback from re-landing on the void.

    The result is deterministic given a fixed input (Otsu has no RNG), satisfying
    the reproducibility contract. Returns a uint8 mask (255 = foreground, 0 =
    background) or None when foreground estimation is not useful / cv2 missing.
    """
    if not HAS_CV2:
        return None

    try:
        arr = np.asarray(normal_img)
        if arr.ndim == 3:
            # PIL gives RGB; cv2 expects BGR but grayscale luminance is close
            # enough for thresholding — convert via cv2 for consistency.
            gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        else:
            gray = arr.astype(np.uint8)

        h, w = gray.shape[:2]
        if h < 2 or w < 2:
            return None

        # Otsu split: pixels >= threshold become 255 (the "bright" class).
        _thr, binary = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        # Corner majority vote → background class. Corners are the 4 pixels.
        corners = [
            int(binary[0, 0]), int(binary[0, w - 1]),
            int(binary[h - 1, 0]), int(binary[h - 1, w - 1]),
        ]
        bright_corners = sum(1 for c in corners if c >= 128)
        # If majority of corners are bright, background is the bright class →
        # foreground is the dark class, and vice versa.
        if bright_corners >= 2:
            fg = (binary < 128).astype(np.uint8) * 255
        else:
            fg = (binary >= 128).astype(np.uint8) * 255

        # Largest connected component only.
        n_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(
            fg, connectivity=8
        )
        if n_labels <= 1:
            return None
        # label 0 is background; pick the largest non-zero label by area.
        areas = stats[1:, cv2.CC_STAT_AREA]
        largest = int(np.argmax(areas)) + 1
        fg_mask = (labels == largest).astype(np.uint8) * 255

        # Step 4 — reject degenerate area splits first (cheap; also discards the
        # degenerate CCs before the void check below so it never wastes a quality
        # score on them). Matches the docstring step order.
        ratio = float(np.count_nonzero(fg_mask)) / float(h * w)
        if ratio < 0.02 or ratio > 0.90:
            return None

        # Step 5 — foreground void-rejection guard (Mode A), BRIGHTNESS-AGNOSTIC.
        # The corner-vote above only decides POLARITY; it never verifies the
        # chosen class is a real object. A flat empty margin (void) can win the
        # corner vote and its largest CC IS the void — the defect would be pasted
        # onto it. Void is FLAT regardless of brightness: severstal void is dark
        # (mean≈0), aitex void is bright/white (mean≈255) — 4-domain context_features
        # confirm BOTH collapse to local_variance≈0 (std<~3), while leather/mtd
        # have no such flat mode (0% void). So we gate on FLATNESS (std) + a 2-D
        # bbox-crop quality score (Laplacian sharpness + contrast, also
        # brightness-agnostic) and DROP the old `mean < _FG_VOID_MEAN` dark-only
        # condition that silently passed bright (white) voids. A genuine textured
        # object clears the std threshold, so real-object datasets are unaffected.
        sel = fg_mask >= 128
        if np.count_nonzero(sel) > 0:
            fg_pixels = gray[sel].astype(np.float32)
            if float(np.std(fg_pixels)) < _FG_VOID_STD:
                cc_left = int(stats[largest, cv2.CC_STAT_LEFT])
                cc_top = int(stats[largest, cv2.CC_STAT_TOP])
                cc_w = int(stats[largest, cv2.CC_STAT_WIDTH])
                cc_h = int(stats[largest, cv2.CC_STAT_HEIGHT])
                crop = gray[cc_top:cc_top + cc_h, cc_left:cc_left + cc_w]
                if (crop.size > 0
                        and _background_quality_score(crop.astype(np.float32))
                        < _FG_VOID_QUALITY):
                    return None  # void confirmed → fall back to random placement

        return fg_mask
    except Exception:
        return None


def _background_quality_score(gray: np.ndarray, blur_threshold: float = 100.0) -> float:
    """
    Compute a background-quality score (0..1) for a grayscale patch.

    Ported verbatim from CASDA ``extract_clean_backgrounds.compute_quality_score``
    so the same Severstal-validated weighting is reused here:

        quality = 0.30*blur + 0.30*contrast + 0.20*brightness + 0.20*noise

    - blur:       Laplacian variance >= blur_threshold → 1.0 else 0.3
                  (flat/black patch → low Laplacian → 0.3).
    - contrast:   min(std(gray)/128, 1.0) (black patch std≈0 → ≈0).
    - brightness: 1.0 if mean in [0.3, 0.7] of full range else 0.7.
    - noise:      1.0 - min(mean(local_var)/100, 1.0) (5x5 mean-filter based).

    Deterministic (no RNG). Caller guarantees ``HAS_CV2`` is True.
    """
    gray = gray.astype(np.float32) if gray.dtype != np.float32 else gray

    # 1. Blur check (30%)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    blur_score = 1.0 if laplacian_var >= blur_threshold else 0.3

    # 2. Contrast check (30%)
    contrast_score = min(float(np.std(gray)) / 128.0, 1.0)

    # 3. Brightness check (20%)
    mean_brightness = float(np.mean(gray)) / 255.0
    brightness_score = 1.0 if 0.3 <= mean_brightness <= 0.7 else 0.7

    # 4. Noise check (20%) — local variance via 5x5 mean filter
    kernel = np.ones((5, 5), np.float32) / 25.0
    local_mean = cv2.filter2D(gray, -1, kernel)
    local_var = cv2.filter2D((gray - local_mean) ** 2, -1, kernel)
    noise_level = float(np.mean(local_var)) / 100.0
    noise_score = 1.0 - min(noise_level, 1.0)

    quality = (
        0.30 * blur_score
        + 0.30 * contrast_score
        + 0.20 * brightness_score
        + 0.20 * noise_score
    )
    return float(quality)


def _is_clean_background(
    img_or_gray: np.ndarray,
    min_quality: float = 0.7,
    blur_threshold: float = 100.0,
) -> bool:
    """
    Return True if the patch is a usable (non black/flat) background.

    A clean background is one that is NOT a void/flat region — the weighted
    quality score must reach ``min_quality``. This matches CASDA's reference
    (``extract_clean_backgrounds``: accept iff ``quality >= min_quality``);
    ``blur_threshold`` flows only into the soft 30%-weighted blur term inside
    ``_background_quality_score`` (no separate hard Laplacian gate — that would
    double-penalize blur and diverge from the validated CASDA criterion).

    Accepts either a grayscale (H, W) or color (H, W, 3) ndarray. When cv2 is
    unavailable (``HAS_CV2`` False) the gate is disabled and True is returned —
    legacy behavior, same convention as ``_foreground_mask``.
    """
    if not HAS_CV2:
        return True

    try:
        arr = np.asarray(img_or_gray)
        if arr.ndim == 3:
            gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        else:
            gray = arr
        if gray.size == 0:
            return True

        gray_f = gray.astype(np.float32)
        return _background_quality_score(gray_f, blur_threshold) >= min_quality
    except Exception:
        # Evaluation failure must not drop an otherwise-usable background.
        return True


# ---------------------------------------------------------------------------
# Pair-level texture gate (see .claude/.dev_note/aroma_controlnet-arm_quality-filters.md)
# ---------------------------------------------------------------------------
# Motivation (severstal ControlNet pilot): the normal background is drawn
# uniformly at random, so a defect whose source surroundings are flat steel can
# land on a structurally periodic surface (checkerplate). seamlessClone only
# adapts colour/illumination — it cannot hide the phase break of a repeating
# pattern. The gate compares a texture descriptor of the source defect's
# surroundings (background strips around the bbox) against the candidate
# paste patch and resamples position → normal until compatible. Thresholds are
# Colab-tunable module constants; texture_dist_threshold=0.0 keeps the gate
# OFF and the rng stream byte-identical to legacy.

_TEXTURE_STRIP_PX = 24         # thickness of the background strips around the bbox
_TEXTURE_MIN_PIX = 64          # fewer selected pixels ⇒ descriptor None (gate passes)
_TEXTURE_MAX_NORMAL_REPICK = 5 # stage-2 budget: different-normal redraws
_TEXTURE_W = np.array([0.20, 0.20, 0.35, 0.25], dtype=np.float32)
                               # weights: std, lap-var, periodicity, orient-anisotropy


def _texture_descriptor(
    patch: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> Optional[np.ndarray]:
    """
    Rotation-invariant, scale-normalized texture vector in [0,1]^4:
    [contrast std, Laplacian variance, autocorrelation periodicity peak,
    gradient-orientation anisotropy].

    ``mask`` (bool, same HxW) optionally restricts the AGGREGATION of the
    per-pixel statistics to an irregular selection — the per-pixel operators
    themselves need the full rectangle, and the periodicity term always uses
    the full rectangle (callers therefore pass rectangular, defect-free
    patches; see _source_bg_descriptor).

    Returns None when cv2 is unavailable, the patch is degenerate, or the
    selection is too small — callers treat None as "cannot judge ⇒ gate
    passes" (same convention as ``_is_clean_background``).
    """
    if not HAS_CV2:
        return None
    try:
        arr = np.asarray(patch)
        if arr.ndim == 3:
            gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        else:
            gray = arr
        g = gray.astype(np.float32)
        h, w = g.shape[:2]
        if h < 8 or w < 8:
            return None
        sel = np.asarray(mask, dtype=bool) if mask is not None else None
        n_sel = int(sel.sum()) if sel is not None else g.size
        if n_sel < _TEXTURE_MIN_PIX:
            return None

        def _pick(a: np.ndarray) -> np.ndarray:
            return a[sel] if sel is not None else a.ravel()

        # Remove a fitted linear illumination plane FIRST so shading does not
        # masquerade as texture: a smooth ramp inflates std, dominates the
        # autocorrelation (measured: a texture-free 40-gray-level ramp scored
        # c_per=0.97 without detrending — indistinguishable from real
        # periodicity), and biases the orientation histogram with a constant
        # gradient vector. The Laplacian term is inherently plane-invariant.
        # This also matters because the gate inspects the RAW pre-blend patch
        # while Reinhard/seamlessClone would remove such lighting downstream.
        yy_c, xx_c = np.mgrid[0:h, 0:w]
        A = np.stack([xx_c.ravel(), yy_c.ravel(),
                      np.ones(h * w, dtype=np.float32)], axis=1).astype(np.float32)
        coef, *_ = np.linalg.lstsq(A, g.ravel(), rcond=None)
        g = g - (A @ coef).reshape(h, w)

        # Near-flat after detrending ⇒ nothing but quantization staircase
        # left — its autocorrelation is perfectly periodic (measured c_per=1.0
        # on a texture-free ramp) and would poison the comparison. Under one
        # gray level of residual std there is no texture to compare: None.
        if float(g.std()) < 1.0:
            return None

        # 1) contrast: std normalized by 128 (same scale as
        #    _background_quality_score's contrast term)
        c_std = min(float(_pick(g).std()) / 128.0, 1.0)

        # 2) edge energy: Laplacian variance squashed to [0,1)
        # (CV_32F: float32→CV_64F is not supported by every cv2 build)
        lap = cv2.Laplacian(g, cv2.CV_32F)
        c_lap = float(np.tanh(_pick(lap).var() / 500.0))

        # 3) periodicity: variance-normalized 2-D autocorrelation, height of
        #    the strongest non-central peak. Amplitude-invariant (chosen over
        #    a raw FFT power profile for that reason), but NOT size-invariant:
        #    a structureless patch of N pixels has a noise floor of roughly
        #    sqrt(2*ln N)/sqrt(N) (~0.02 at 256², ~0.15 for a 24px strip), so
        #    comparing very small windows inflates this term. Genuine periodic
        #    surfaces peak at 0.5-0.8 and stay well above the floor; treat
        #    sub-thousand-pixel comparisons as low-confidence when tuning.
        gz = g - float(g.mean())
        F = np.fft.rfft2(gz)
        ac = np.fft.irfft2(F * np.conj(F), s=g.shape)
        zero_lag = float(ac[0, 0])
        if zero_lag <= 1e-6:
            return None  # perfectly flat patch — no texture to compare
        ac = ac / zero_lag
        # Autocorr peaks near lag 0 wrap to the four array corners — blank a
        # central-lobe radius around each corner before taking the max.
        r = max(2, min(h, w) // 8)
        ac[:r, :r] = 0.0
        ac[:r, -r:] = 0.0
        ac[-r:, :r] = 0.0
        ac[-r:, -r:] = 0.0
        c_per = float(np.clip(ac.max(), 0.0, 1.0))

        # 4) orientation anisotropy: magnitude-weighted gradient-orientation
        #    histogram entropy, flipped so periodic/ridged surfaces score high
        gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
        ang = np.arctan2(gy, gx) % np.pi
        magw = np.hypot(gx, gy)
        hist, _ = np.histogram(_pick(ang), bins=8, range=(0.0, float(np.pi)),
                               weights=_pick(magw))
        total = float(hist.sum())
        if total <= 1e-6:
            p_hist = np.full(8, 1.0 / 8.0)
        else:
            p_hist = hist / total
        ent = float(-(p_hist * np.log(p_hist + 1e-12)).sum() / np.log(8))
        c_ani = 1.0 - min(max(ent, 0.0), 1.0)

        return np.array([c_std, c_lap, c_per, c_ani], dtype=np.float32)
    except Exception:
        # Descriptor failure must never reject an otherwise-usable pairing.
        return None


def _texture_distance(
    desc_a: Optional[np.ndarray],
    desc_b: Optional[np.ndarray],
) -> float:
    """Weighted L1 distance in [0,1]. Either side None ⇒ 0.0 (gate passes)."""
    if desc_a is None or desc_b is None:
        return 0.0
    return float((_TEXTURE_W * np.abs(desc_a - desc_b)).sum())


def _source_bg_descriptor(
    img_arr: np.ndarray,
    mask_arr: np.ndarray,
    bbox: Tuple[int, int, int, int],
) -> Optional[np.ndarray]:
    """
    Texture descriptor of the SOURCE defect's surroundings: the average
    descriptor of the rectangular background strips (thickness
    ``_TEXTURE_STRIP_PX``) directly above / below / left / right of the
    defect bbox.

    Rectangular strips — not a mask-dilation ring — on purpose: the
    periodicity term needs a filled rectangle, and a thin ring around a
    defect that dominates its own bbox leaves the autocorrelation dominated
    by the defect shape (measured: a noise background scored c_per=0.67
    through the ring approximation). Strips sit fully OUTSIDE the bbox, and
    ``defect_bbox`` is the bbox OF the mask, so they contain no defect
    pixels by construction.

    img_arr:  full defect image, (H,W) or (H,W,3) ndarray.
    mask_arr: full-frame defect mask, (H,W) ndarray (kept for signature
              stability; strips are defect-free by construction).
    bbox:     (x, y, w, h) of the defect.

    Returns None (gate passes) when cv2 is missing or no strip is usable —
    e.g. a bbox that fills the whole frame.
    """
    if not HAS_CV2:
        return None
    try:
        bx, by, bw, bh = (int(v) for v in bbox)
        ih, iw = np.asarray(mask_arr).shape[:2]
        img = np.asarray(img_arr)
        t = _TEXTURE_STRIP_PX
        strips = [
            img[max(0, by - t):by, bx:bx + bw],            # above
            img[by + bh:min(ih, by + bh + t), bx:bx + bw],  # below
            img[by:by + bh, max(0, bx - t):bx],             # left
            img[by:by + bh, bx + bw:min(iw, bx + bw + t)],  # right
        ]
        descs = []
        for s in strips:
            if s.shape[0] >= 8 and s.shape[1] >= 8:
                d = _texture_descriptor(s)
                if d is not None:
                    descs.append(d)
        if not descs:
            return None
        return np.mean(np.stack(descs), axis=0).astype(np.float32)
    except Exception:
        return None


_CTX_CELL_HELPERS: Optional[tuple] = None


def _load_context_cell_helpers() -> tuple:
    """Lazy-load (extract_context_features, context_cell_key) from the sibling
    distribution_profiling module, reused by the placement compat gate so the
    paste-patch cell_key is derived IDENTICALLY to how the compatibility matrix
    was built. Cached; returns (None, None) if the import fails (gate auto-off)."""
    global _CTX_CELL_HELPERS
    if _CTX_CELL_HELPERS is not None:
        return _CTX_CELL_HELPERS
    try:
        here = str(Path(__file__).resolve().parent)
        parent = str(Path(__file__).resolve().parent.parent)
        for pth in (here, parent):
            if pth not in sys.path:
                sys.path.insert(0, pth)
        import distribution_profiling as _dp  # noqa: PLC0415 — sibling, lazy
        _CTX_CELL_HELPERS = (_dp._extract_context_features, _dp._context_cell_key)
    except Exception as exc:
        logger.warning("compat gate: distribution_profiling import failed "
                       "(%s) — gate disabled", exc)
        _CTX_CELL_HELPERS = (None, None)
    return _CTX_CELL_HELPERS


# Compat-gate tiling (compat_mode=='symmetric' only). The compatibility matrix
# and bin_edges are learned on 64px patches (distribution_profiling
# _context_worker, GRID_SIZE=64), but the legacy gate query cropped the patch at
# the DEFECT crop size — a scale mismatch that misclassifies the cell on
# size-heterogeneous sets (devnote §2 finding E). When tiling is on the defect
# footprint is covered by full 64px windows queried at the SAME scale the matrix
# was built at, and their compat values are aggregated. 'mean' = average
# background suitability over the footprint (default); 'min' = strictest tile
# (safer against over-reject on large defects — alternative, not default).
_COMPAT_TILE = 64
_COMPAT_TILE_AGG = "mean"  # 'mean' (default) | 'min' (strict alternative)

# Symmetric positive placement (compat_mode=='symmetric' only — devnote
# aroma_compat_gate_clean-grounded_redesign §3-4/§7 positive steering). Instead
# of sampling a random/foreground position and reject-resampling low-compat
# cells (Scenario B), the placement SCANs every fitting position, RANKs each by
# the mean compatibility of the 64px tiles covering its footprint (same tiling
# scale the matrix was built at), drops footprints that straddle a void tile,
# and PLACEs by sampling from the top-K. _POS_STRIDE bounds both the candidate
# grid and the 64px feature-extraction cost; _POS_MAX_CAND is a safety cap that
# auto-coarsens the stride on very large images.
_POS_STRIDE = 32
_POS_TOPK = 8
_POS_MAX_CAND = 4096

# Image-level background selection by D_v-specific distribution similarity
# (compat_mode=='symmetric' only — devnote aroma_compat_gate_clean-grounded_
# redesign, "D_v-specific 배경 분포 유사도" redesign superseding the cluster-
# aggregate mean-compat ranking). The legacy main loop draws the background
# uniformly (`rng.choice(normals)`), so WHICH image a defect lands on is
# unconditioned on the source's background — only the WITHIN-image position was
# compat-ranked (_positive_place, unchanged). The prior image ranker scored each
# normal by the MEAN compatibility of its tiles for the defect CLUSTER, but
# cluster is a defect-morphology grouping (not a background type) and a mean
# collapses appearance heterogeneity — so similar mean-compat scores did not mean
# similar-looking backgrounds (devnote 원인2/원인3). This tier instead compares
# DISTRIBUTIONS: it builds the SOURCE defect image's background cell histogram
# (`_dv_bg_hist`, defect region excluded) and each clean image's non-void cell
# histogram (`_cell_hist` over the cached `_normal_tile_cells` tiles), ranks the
# clean pool by histogram-intersection similarity (`_hist_intersection`, [0,1]),
# and samples the background from the top-K (diversity preserved — argmax would
# collapse the pool). p_dv is per-SOURCE (cached by source key); the clean
# histogram is source-independent (cached ONCE per path); only the intersection +
# ranking repeats per distinct source.
_NORMAL_TOPK = 16          # sample the background from the top-K highest-scoring normals
_NORMAL_SAMPLE_CAP = 64    # max 64px tiles sampled per image (bounds cost on large pools)
_NORMAL_STRIDE = 64        # tile stride when scanning a normal (auto-coarsened above the cap)


def _tile_anchors(start: int, length: int, bound: int, tile: int = 64) -> List[int]:
    """Window start-coordinates that cover footprint [start, start+length) with
    full ``tile``-wide windows, each lying inside the image axis [0, bound).

    - Every returned anchor ``a`` satisfies 0 <= a <= bound-tile, so window
      [a, a+tile) is fully in-image (when bound >= tile).
    - The windows' union covers the whole footprint span; the final window is
      positioned to cover the footprint end.
    - ``length < tile``  → a single window centred on the footprint, clamped
      in-image.
    - ``bound <= tile``  → ``[0]`` (image is not wider than one tile on this
      axis; the caller slices what exists).

    Pure / deterministic — no rng. Anchors are de-duplicated, order preserved.
    """
    if bound <= tile:
        return [0]
    max_a = bound - tile
    if length < tile:
        center = start + length / 2.0
        a = int(round(center - tile / 2.0))
        return [max(0, min(a, max_a))]
    end = start + length
    anchors: List[int] = []
    a = start
    # Step by tile across the footprint; stop once a window would already reach
    # the footprint end (that tail is covered by the final anchor below).
    while a + tile < end:
        ca = max(0, min(a, max_a))
        if ca not in anchors:
            anchors.append(ca)
        a += tile
    last = max(0, min(end - tile, max_a))
    if last not in anchors:
        anchors.append(last)
    return anchors


def _foreground_paste_position(
    fg_mask: np.ndarray,
    bg_size: Tuple[int, int],
    crop_size: Tuple[int, int],
    mask_crop: "Image",
    rng: random.Random,
    max_tries: int = 20,
) -> Optional[Tuple[int, int]]:
    """
    Sample a paste position whose defect lands inside the object foreground.

    Args:
        fg_mask:    uint8 foreground mask (255=fg) for the background image,
                    shape (bg_h, bg_w) — as returned by _foreground_mask.
        bg_size:    (bg_w, bg_h) of the background image.
        crop_size:  (crop_w, crop_h) of the defect crop.
        mask_crop:  L-mode PIL Image of the defect-region mask (crop-sized).
        rng:        seeded random.Random — same seed → same sequence.
        max_tries:  number of random (x, y) samples to attempt.

    Returns:
        (x, y) top-left paste position such that the centroid of the defect
        mask, once pasted, lands on a foreground pixel; or None after max_tries.
    """
    bw, bh = bg_size
    cw, ch = crop_size
    if cw > bw or ch > bh:
        return None

    # Defect centroid within the crop (foreground pixels of the defect mask).
    msk_arr = np.asarray(mask_crop.convert("L"))
    ys, xs = np.nonzero(msk_arr >= 128)
    if xs.size == 0:
        # No defect foreground pixels — use crop geometric center.
        cx_off, cy_off = cw / 2.0, ch / 2.0
    else:
        cx_off = float(xs.mean())
        cy_off = float(ys.mean())

    x_max = max(0, bw - cw)
    y_max = max(0, bh - ch)
    fh, fw = fg_mask.shape[:2]

    for _ in range(max(1, max_tries)):
        x = rng.randint(0, x_max)
        y = rng.randint(0, y_max)
        px = int(round(x + cx_off))
        py = int(round(y + cy_off))
        if 0 <= px < fw and 0 <= py < fh and fg_mask[py, px] >= 128:
            return x, y
    return None


def _positive_place(
    nrgb: np.ndarray,
    crop_wh: Tuple[int, int],
    compat_row: Dict[str, float],
    bin_edges: Dict[str, List[float]],
    min_bg_quality: float,
    blur_threshold: float,
    rng: random.Random,
    stride: int = _POS_STRIDE,
    topk: int = _POS_TOPK,
    max_cand: int = _POS_MAX_CAND,
) -> Tuple[Optional[Tuple[int, int]], float, int]:
    """Scan-rank-place positive placement for the symmetric compat gate.

    Enumerates every top-left position where the crop fully fits (strided by
    ``stride``), scores each candidate by the MEAN compatibility of the 64px
    tiles that cover its footprint (``_tile_anchors`` / ``_COMPAT_TILE`` — the
    SAME scale the compatibility matrix was built at, and identical to the
    tiling in ``_compat_ok``), excludes any candidate whose footprint straddles
    a void tile, then samples the paste position from the top-K highest-compat
    candidates (``rng`` → per-ROI placement diversity under a fixed seed).

    Args:
        nrgb:            HxWx3 (or HxW) uint8 background array.
        crop_wh:         (crop_w, crop_h) of the defect crop (post fit-rescale).
        compat_row:      matrix[cluster] row (symmetric SGM). ``.get(cell, 0.5)``
                         → unobserved cell is neutral 0.5.
        bin_edges:       feature bin edges (shared with matrix construction).
        min_bg_quality:  void gate — a tile with quality below this is a void.
        blur_threshold:  passed through to the void quality score.
        rng:             seeded random.Random (top-K sample = the only new draw).
        stride:          candidate/grid stride (auto-coarsened above max_cand).
        topk:            sample the position from the top-K candidates by mean.
        max_cand:        safety cap on candidate count.

    Returns ``(pos_xy, best_mean, n_nonvoid)``:
        - ``pos_xy``:    chosen (x, y) top-left; ``None`` iff there is NO
                         non-void candidate (caller escapes to another normal).
        - ``best_mean``: footprint mean-compat of the chosen position, or — when
                         no non-void candidate exists — the best mean-compat over
                         ALL candidates (so the caller can still force a paste).
        - ``n_nonvoid``: number of candidates whose footprint has no void tile.

    Pure/deterministic given a fixed ``rng``. The profiling cell helpers are
    loaded lazily; when unavailable (or cv2 missing) every cell falls back to
    neutral compat 0.5 so the routine still ranks by voidness alone.
    """
    cw, ch = int(crop_wh[0]), int(crop_wh[1])
    h_img, w_img = nrgb.shape[:2]
    span_x = max(0, w_img - cw)
    span_y = max(0, h_img - ch)

    ctx_feat_fn, cell_key_fn = _load_context_cell_helpers()
    have_cells = ctx_feat_fn is not None and cell_key_fn is not None and HAS_CV2

    def _axis(span: int, st: int) -> List[int]:
        if span <= 0:
            return [0]
        pts = list(range(0, span + 1, st))
        if pts[-1] != span:
            pts.append(span)  # always include the far edge so it stays reachable
        return pts

    # Coarsen the stride until the candidate grid is under the safety cap.
    st = max(1, int(stride))
    xs_c = _axis(span_x, st)
    ys_c = _axis(span_y, st)
    while len(xs_c) * len(ys_c) > max_cand and st < max(span_x, span_y, 1):
        st *= 2
        xs_c = _axis(span_x, st)
        ys_c = _axis(span_y, st)

    # Per-anchor (64px tile) compat + void cache. cell_key is cluster-independent
    # and voidness depends only on the tile, so each distinct anchor is computed
    # once regardless of how many candidate footprints cover it → feature
    # extraction is O(distinct tiles), not O(candidates).
    tile_cache: Dict[Tuple[int, int], Tuple[float, bool]] = {}

    def _tile(ax: int, ay: int) -> Tuple[float, bool]:
        cached = tile_cache.get((ax, ay))
        if cached is not None:
            return cached
        win = nrgb[ay:ay + _COMPAT_TILE, ax:ax + _COMPAT_TILE]
        if win.size == 0:
            res = (0.5, True)
            tile_cache[(ax, ay)] = res
            return res
        if HAS_CV2 and win.ndim == 3:
            gray = cv2.cvtColor(win, cv2.COLOR_RGB2GRAY)
        else:
            gray = win if win.ndim == 2 else win[..., 0]
        try:
            is_void = not _is_clean_background(
                gray, min_quality=min_bg_quality, blur_threshold=blur_threshold
            )
        except Exception:
            is_void = False
        if have_cells:
            try:
                cell = cell_key_fn(ctx_feat_fn(gray), bin_edges)
                compat = float(compat_row.get(cell, 0.5))
            except Exception:
                compat = 0.5
        else:
            compat = 0.5
        res = (compat, is_void)
        tile_cache[(ax, ay)] = res
        return res

    best_all_mean = -1.0
    nonvoid: List[Tuple[float, Tuple[int, int]]] = []
    for y in ys_c:
        ay_anchors = _tile_anchors(y, ch, h_img, _COMPAT_TILE)
        for x in xs_c:
            ax_anchors = _tile_anchors(x, cw, w_img, _COMPAT_TILE)
            vals: List[float] = []
            has_void = False
            for ay in ay_anchors:
                for ax in ax_anchors:
                    compat, is_void = _tile(ax, ay)
                    vals.append(compat)
                    if is_void:
                        has_void = True
            mean = sum(vals) / len(vals) if vals else 0.5
            if mean > best_all_mean:
                best_all_mean = mean
            if not has_void:
                nonvoid.append((mean, (x, y)))

    if not nonvoid:
        return None, (best_all_mean if best_all_mean >= 0.0 else 0.5), 0

    # Rank by footprint mean-compat (desc) — stable sort keeps ties in scan
    # order — then sample from the top-K for placement diversity across the
    # n_per_roi repeats sharing one seeded rng.
    nonvoid.sort(key=lambda t: t[0], reverse=True)
    k = max(1, min(int(topk), len(nonvoid)))
    _chosen_mean, chosen_pos = rng.choice(nonvoid[:k])
    # Gate on the BEST non-void mean (nonvoid[0]), not the top-K-sampled one:
    # re-pick a normal only when NO non-void position reaches τ (devnote §3 step5).
    # Placement still uses the sampled top-K position for diversity.
    best_nonvoid_mean = float(nonvoid[0][0])
    return chosen_pos, best_nonvoid_mean, len(nonvoid)


def _normal_tile_cells(
    gray: np.ndarray,
    bin_edges: Dict[str, List[float]],
    stride: int = _NORMAL_STRIDE,
    cap: int = _NORMAL_SAMPLE_CAP,
    min_quality: float = 0.7,
    blur_threshold: float = 100.0,
) -> List[Tuple[Optional[str], bool]]:
    """Tile a normal image into ``(cell, void)`` pairs for image-level compat
    scoring (``_image_compat_score`` / ``_rank_normals``).

    Slides a full ``_COMPAT_TILE`` (64px) window over ``gray`` (already the SAME
    grayscale the placement gate uses) with step ``stride``; the step is
    auto-coarsened until the tile count is under ``cap`` so the samples stay
    spread across the whole image (never biased to the top-left corner) and the
    per-image cost is bounded on large pools. For each tile it records:

    - ``cell``: the profiling context cell key (``_extract_context_features`` →
      ``_context_cell_key``, IDENTICAL to how the compat matrix was built), or
      ``None`` when the profiling helpers / cv2 are unavailable (scored as a
      neutral 0.5 downstream).
    - ``void``: ``not _is_clean_background(tile)`` — a black/flat tile is a void
      and is excluded from the image's compat score.

    Cluster-independent (neither cell nor void depends on the defect cluster), so
    the caller computes it ONCE per normal path and reuses it across clusters.
    Pure / deterministic — no rng.
    """
    result: List[Tuple[Optional[str], bool]] = []
    if gray is None or getattr(gray, "size", 0) == 0:
        return result
    h, w = gray.shape[:2]
    tile = _COMPAT_TILE

    def _anchors(dim: int, st: int) -> List[int]:
        if dim <= tile:
            return [0]
        a = list(range(0, dim - tile + 1, st))
        if a[-1] != dim - tile:
            a.append(dim - tile)  # always include the far edge
        return a

    st = max(1, int(stride))
    ya = _anchors(h, st)
    xa = _anchors(w, st)
    # Coarsen the stride until the tile grid is under the cap — keeps the sample
    # spread across the image instead of truncating to the first `cap` tiles.
    while len(ya) * len(xa) > cap and st < max(h, w, 1):
        st *= 2
        ya = _anchors(h, st)
        xa = _anchors(w, st)

    ctx_feat_fn, cell_key_fn = _load_context_cell_helpers()
    have_cells = ctx_feat_fn is not None and cell_key_fn is not None and HAS_CV2

    for ay in ya:
        for ax in xa:
            if len(result) >= cap:
                return result
            win = gray[ay:ay + tile, ax:ax + tile]
            if win.size == 0:
                continue
            try:
                is_void = not _is_clean_background(
                    win, min_quality=min_quality, blur_threshold=blur_threshold
                )
            except Exception:
                is_void = False
            cell: Optional[str] = None
            if have_cells:
                try:
                    cell = cell_key_fn(ctx_feat_fn(win), bin_edges)
                except Exception:
                    cell = None
            result.append((cell, is_void))
    return result


def _image_compat_score(
    cells_voids: List[Tuple[Optional[str], bool]],
    compat_row: Optional[Dict[str, float]],
) -> float:
    """Compat score of one normal image for a defect cluster: the MEAN
    compatibility of its non-void tiles (``compat_row.get(cell, 0.5)`` — an
    unobserved / ``None`` cell is neutral 0.5, matching the gate's soft match).

    Returns ``-1.0`` when the image has NO non-void tile (fully black/flat →
    excluded from ranking by ``_rank_normals``). Pure / deterministic.
    """
    row = compat_row or {}
    vals = [float(row.get(cell, 0.5)) for cell, void in cells_voids if not void]
    if not vals:
        return -1.0
    return sum(vals) / len(vals)


def _rank_normals(
    scored: List[Tuple[float, str]],
    rng: random.Random,
    topk: int = _NORMAL_TOPK,
) -> Optional[str]:
    """Pick one normal path from ``scored`` ``[(score, path)]`` by compat rank.

    Drops excluded images (``score < 0`` → all-void, ``_image_compat_score``
    returned -1.0), sorts the rest by score descending (stable → ties keep input
    order for determinism), and samples ONE path from the top-``topk`` (``rng``
    → per-selection diversity under a fixed seed; argmax would collapse the pool
    to a single background). Returns ``None`` when nothing qualifies so the
    caller can fall back to the uniform draw. Pure given ``rng`` — exactly one
    rng draw when a path is returned, zero when ``None``.
    """
    valid = [(s, p) for s, p in scored if s >= 0.0]
    if not valid:
        return None
    valid.sort(key=lambda t: t[0], reverse=True)
    k = max(1, min(int(topk), len(valid)))
    return rng.choice(valid[:k])[1]


def _cell_hist(cells_voids: List[Tuple[Optional[str], bool]]) -> Dict[str, float]:
    """Normalized distribution over the NON-VOID context-cell keys of one image.

    Consumes the ``[(cell, void)]`` tiles produced by ``_normal_tile_cells``
    (clean images) and turns them into a probability histogram: count non-void
    tiles per cell (``None`` cells — profiling helpers unavailable — are
    skipped), then divide by the non-void total so the values sum to 1.0.
    Returns ``{}`` when there is no usable non-void cell (fully black/flat →
    scored -1.0 / dropped by the caller). Pure / deterministic — no rng.
    """
    counts: Dict[str, int] = {}
    total = 0
    for cell, void in cells_voids:
        if void or cell is None:
            continue
        counts[cell] = counts.get(cell, 0) + 1
        total += 1
    if total == 0:
        return {}
    return {c: n / total for c, n in counts.items()}


def _dv_bg_hist(
    gray: np.ndarray,
    mask: Optional[np.ndarray] = None,
    bbox: Optional[Tuple[int, int, int, int]] = None,
    bin_edges: Optional[Dict[str, List[float]]] = None,
    stride: int = _NORMAL_STRIDE,
    cap: int = _NORMAL_SAMPLE_CAP,
) -> Dict[str, float]:
    """Normalized BACKGROUND cell histogram of a SOURCE defect image (``D_v``).

    Slides a 64px window over ``gray`` (step ``stride``, auto-coarsened under
    ``cap`` exactly like ``_normal_tile_cells`` so the sample stays spread
    across the image and the cost stays bounded on large pools), EXCLUDES every
    tile that lies on the defect, and maps each surviving background tile to a
    profiling cell key (``_extract_context_features`` → ``_context_cell_key`` —
    the SAME discretization the compat matrix and the clean histogram use). The
    result is the normalized ``{cell: fraction}`` distribution of the defect
    image's background.

    Defect exclusion (mirrors distribution_profiling._context_worker):
      - ``mask`` given (full-frame binary, gray resolution — resized nearest if
        it differs): skip a tile whose defect-pixel fraction exceeds 0.5
        (``mask[tile].mean() > 0.5``), IDENTICAL to ``_context_worker``.
      - else ``bbox`` (x, y, w, h) given: skip a tile whose area-overlap with
        the bbox exceeds half the tile (fallback when no mask is persisted).

    Returns ``{}`` when no background tile survives, or when the profiling
    helpers / cv2 / ``bin_edges`` are unavailable. Pure / deterministic — no rng.
    """
    if gray is None or getattr(gray, "size", 0) == 0:
        return {}
    ctx_feat_fn, cell_key_fn = _load_context_cell_helpers()
    if ctx_feat_fn is None or cell_key_fn is None or not HAS_CV2 or bin_edges is None:
        return {}
    h, w = gray.shape[:2]
    tile = _COMPAT_TILE

    # Full-frame defect mask (bool, gray resolution) — matches _context_worker's
    # `raw > 0`. Resize nearest when a downscaled GT mask is passed.
    dmask: Optional[np.ndarray] = None
    if mask is not None and getattr(mask, "size", 0) > 0:
        m = mask
        if m.shape[:2] != (h, w):
            try:
                m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
            except Exception:
                m = None
        if m is not None:
            dmask = m > 0

    bx = by = bw_box = bh_box = None
    if dmask is None and isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        try:
            bx, by, bw_box, bh_box = (int(v) for v in bbox)
        except (TypeError, ValueError):
            bx = None

    def _anchors(dim: int, st: int) -> List[int]:
        if dim <= tile:
            return [0]
        a = list(range(0, dim - tile + 1, st))
        if a[-1] != dim - tile:
            a.append(dim - tile)  # always include the far edge
        return a

    st = max(1, int(stride))
    ya = _anchors(h, st)
    xa = _anchors(w, st)
    while len(ya) * len(xa) > cap and st < max(h, w, 1):
        st *= 2
        ya = _anchors(h, st)
        xa = _anchors(w, st)

    half_area = 0.5 * tile * tile
    counts: Dict[str, int] = {}
    total = 0
    for ay in ya:
        for ax in xa:
            y2, x2 = ay + tile, ax + tile
            if dmask is not None:
                if float(dmask[ay:y2, ax:x2].mean()) > 0.5:
                    continue  # majority-defect tile
            elif bx is not None:
                ix1, iy1 = max(ax, bx), max(ay, by)
                ix2 = min(x2, bx + bw_box)
                iy2 = min(y2, by + bh_box)
                inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                if inter > half_area:
                    continue  # tile mostly inside the defect bbox
            win = gray[ay:y2, ax:x2]
            if win.size == 0:
                continue
            # void/flat exclusion — mirror p_clean's rule (_normal_tile_cells
            # drops void tiles). Without this, the source's void tiles inject
            # cells absent from every clean histogram and bias the intersection
            # down / distort the ranking (review MEDIUM). Same defaults as
            # _normal_tile_cells (min_quality 0.7, blur 100.0).
            if not _is_clean_background(win, min_quality=0.7, blur_threshold=100.0):
                continue
            try:
                cell = cell_key_fn(ctx_feat_fn(win), bin_edges)
            except Exception:
                continue
            counts[cell] = counts.get(cell, 0) + 1
            total += 1
    if total == 0:
        return {}
    return {c: n / total for c, n in counts.items()}


def _hist_intersection(p: Dict[str, float], q: Dict[str, float]) -> float:
    """Histogram-intersection similarity of two normalized cell distributions:
    ``sum_cell min(p[cell], q[cell])`` over shared cells. Ranges [0, 1] (1.0 =
    identical distribution, 0.0 = disjoint supports). Returns 0.0 when either
    distribution is empty. Pure / deterministic — iterates the smaller dict."""
    if not p or not q:
        return 0.0
    if len(p) > len(q):
        p, q = q, p
    return sum(min(v, q[c]) for c, v in p.items() if c in q)


def _paste_and_finalize(
    normal_img: "Image",
    defect_crop: "Image",
    mask: "Image",
    output_path: str,
    blend_mode: str = "alpha",
    feather_px: int = 4,
    rng: Optional[random.Random] = None,
    mask_output_path: Optional[str] = None,
    reject_clean_bg: bool = False,
    min_bg_quality: float = 0.7,
    blur_threshold: float = 100.0,
    max_bg_tries: int = 20,
    defect_path_for_log: str = "",
    source_bg_desc: Optional[np.ndarray] = None,
    normal_pool: Optional[List[str]] = None,
    texture_dist_threshold: float = 0.0,
    compat_row: Optional[Dict[str, float]] = None,
    bin_edges: Optional[Dict[str, List[float]]] = None,
    compat_threshold: float = 0.0,
    compat_tile: bool = False,
) -> Dict[str, Any]:
    """Shared placement + blending + GT-mask tail used by every synthesis method.

    Extracted verbatim from copy_paste_synthesis so other generators (e.g.
    ControlNet) reuse the exact same machinery: fit rescale, foreground-
    constrained placement, clean-bg gate, seamless/alpha blend, full-frame GT
    mask persistence, and the bbox return contract.

    Texture gate (texture_dist_threshold > 0, opt-in): candidate patches must
    be within ``texture_dist_threshold`` of ``source_bg_desc``. Applied on
    BOTH placement paths — foreground-constrained and random fallback —
    because checkerplate-like normals produce a valid foreground mask and
    would bypass a fallback-only gate (the clean-bg gate, by contrast, stays
    fallback-only: legacy contract). Two stages: position resamples within
    the given normal (``max_bg_tries``), then up to
    ``_TEXTURE_MAX_NORMAL_REPICK`` re-picks from ``normal_pool``. Exhaustion
    pastes the last candidate (never a silent drop). When a re-pick happens
    the returned meta carries ``normal_image`` (the path actually used) and
    every meta carries a ``gate_stats`` dict the caller may pop/aggregate.
    """
    from PIL import Image as PILImage

    if rng is None:
        rng = random.Random(42)

    crop_w, crop_h = defect_crop.size
    bw_norm, bh_norm = normal_img.size  # (width, height) of background

    # Defect crop must fit inside the background. When it is larger (common on
    # size-heterogeneous sets like mtd/aitex), rescale it down — preserving the
    # aspect ratio — instead of dropping the sample. The 0.95 factor leaves a
    # placement margin so foreground/random sampling still has room to move.
    if crop_w > bw_norm or crop_h > bh_norm:
        scale = min(bw_norm / crop_w, bh_norm / crop_h) * 0.95
        new_w = max(1, int(crop_w * scale))
        new_h = max(1, int(crop_h * scale))
        defect_crop = defect_crop.resize((new_w, new_h), PILImage.LANCZOS)
        mask = mask.resize((new_w, new_h), PILImage.NEAREST)
        logger.info(
            "Defect crop (%dx%d) rescaled to (%dx%d) to fit normal (%dx%d): %s",
            crop_w, crop_h, new_w, new_h, bw_norm, bh_norm, defect_path_for_log,
        )
        crop_w, crop_h = new_w, new_h

    # Texture gate is active only when everything it needs is available; the
    # guard doubles as the rng-discipline switch — when False, NO new rng draw
    # happens anywhere below and the stream is byte-identical to legacy.
    texture_on = (
        texture_dist_threshold > 0.0
        and source_bg_desc is not None
        and HAS_CV2
    )
    # Placement-aware compat gate (--compat_threshold, opt-in): the candidate
    # paste patch's context cell must have compatibility >= threshold with this
    # defect cluster (compat_row from the compatibility_matrix). Makes placement
    # honor the background↔defect compatibility the selection score only
    # promised. Same rng-discipline as texture_on — off ⇒ no new rng draw.
    compat_on = (
        compat_threshold > 0.0
        and compat_row is not None
        and bin_edges is not None
        and HAS_CV2
    )
    _ctx_feat_fn = _cell_key_fn = None
    if compat_on:
        _ctx_feat_fn, _cell_key_fn = _load_context_cell_helpers()
        if _ctx_feat_fn is None or _cell_key_fn is None:
            compat_on = False  # profiling helpers unavailable → gate auto-off

    def _place_on(nrm: "Image") -> Tuple[np.ndarray, Tuple[int, int], bool]:
        """Placement decision for ONE normal — verbatim lift of the legacy
        foreground → random-fallback → clean-bg-resample block, with the
        texture check added. Returns (normal_rgb, position, gate_ok);
        gate_ok is False only when a gate loop exhausted max_bg_tries.

        Gate scopes differ BY DESIGN: the clean-bg gate stays confined to the
        random fallback (legacy contract — the foreground path is an object
        surface, not a void). The TEXTURE gate applies to BOTH paths:
        checkerplate-like normals yield a valid _foreground_mask (severstal
        pilot images: 24-34% foreground), so a fallback-only texture gate
        would silently bypass exactly the mismatch it exists to catch."""
        nrgb = np.asarray(nrm.convert("RGB"))

        def _tex_ok(pos_xy: Tuple[int, int]) -> bool:
            px, py = pos_xy
            patch = nrgb[py:py + crop_h, px:px + crop_w]
            return (_texture_distance(source_bg_desc,
                                      _texture_descriptor(patch))
                    <= texture_dist_threshold)

        def _compat_ok(pos_xy: Tuple[int, int]) -> bool:
            # Compatibility of the paste patch's context cell with this defect
            # cluster. compat_row.get(cell, 0.5): an unobserved cell is neutral
            # (0.5) so soft `>= threshold` accepts any row cell — the exact-match
            # form fails on sets whose defect context is rare in clean patches
            # (leather 4.7% coverage → ~95% fallback). Returns True on any
            # failure so a descriptor error never rejects a usable placement.
            #
            # compat_tile (symmetric mode only): the matrix/bin_edges were built
            # on 64px patches, so query at that SAME scale — cover the defect
            # footprint with full 64px windows (_tile_anchors, clamped in-image)
            # and aggregate each window's compat. Legacy (compat_tile=False):
            # query the crop-size patch, byte-identical to the historical gate.
            px, py = pos_xy
            try:
                if compat_tile:
                    h_img, w_img = nrgb.shape[:2]
                    ys = _tile_anchors(py, crop_h, h_img, _COMPAT_TILE)
                    xs = _tile_anchors(px, crop_w, w_img, _COMPAT_TILE)
                    vals: List[float] = []
                    for ay in ys:
                        for ax in xs:
                            win = nrgb[ay:ay + _COMPAT_TILE, ax:ax + _COMPAT_TILE]
                            g = (cv2.cvtColor(win, cv2.COLOR_RGB2GRAY)
                                 if win.ndim == 3 else win)
                            cell = _cell_key_fn(_ctx_feat_fn(g), bin_edges)
                            vals.append(float(compat_row.get(cell, 0.5)))
                    if not vals:
                        return True
                    agg = (min(vals) if _COMPAT_TILE_AGG == "min"
                           else sum(vals) / len(vals))
                    return agg >= compat_threshold
                patch = nrgb[py:py + crop_h, px:px + crop_w]
                gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY) if patch.ndim == 3 else patch
                cell = _cell_key_fn(_ctx_feat_fn(gray), bin_edges)
                return float(compat_row.get(cell, 0.5)) >= compat_threshold
            except Exception:
                return True

        def _gate_pass(pos_xy: Tuple[int, int]) -> bool:
            """Combined placement gate: texture AND compat (each skipped when
            its own gate is off)."""
            if texture_on and not _tex_ok(pos_xy):
                return False
            if compat_on and not _compat_ok(pos_xy):
                return False
            return True

        gate_active = texture_on or compat_on

        # Symmetric positive placement (compat_mode=='symmetric' → compat_tile).
        # Replace the random/foreground sample + reject-resample (Scenario B)
        # with scan-rank-place: enumerate fitting positions, score each by the
        # footprint mean-compat over 64px tiles, drop void-straddling footprints,
        # and sample from the top-K. `compat_on` (threshold>0 + row/bin_edges +
        # cv2 + helpers) guarantees everything `_positive_place` needs, so this
        # is the ONLY place a new rng draw enters the symmetric path. When
        # `compat_tile` is False (defect mode / legacy / texture-only) this
        # branch is skipped and the block below stays byte-identical to the
        # historical path — the compat_tile=False, threshold=0 rng stream is
        # untouched. NOTE: in symmetric positive mode the texture gate does not
        # steer placement (position is decided by compat/void alone).
        if compat_on and compat_tile:
            pos, best_mean, n_nonvoid = _positive_place(
                nrgb, (crop_w, crop_h), compat_row, bin_edges,
                min_bg_quality, blur_threshold, rng,
                stride=_POS_STRIDE, topk=_POS_TOPK,
            )
            if pos is None:
                # Every candidate footprint straddles a void tile — no usable
                # target on this normal. Keep a deterministic position so an
                # eventual last-candidate paste (stage-2 exhaustion) still lands
                # somewhere; gate_ok=False escapes to another normal first.
                pos = _random_paste_position(nrm.size, defect_crop.size, rng)
                return nrgb, pos, False
            # best_mean < τ triggers the same stage-2 re-pick / forced-paste
            # fallback the reject path used; the chosen (highest-ranked, top-K
            # sampled) position is retained for that forced paste.
            return nrgb, pos, best_mean >= compat_threshold

        # Constrain placement to the object foreground when one can be
        # estimated (object-centric datasets like visa_pcb). Falls back to
        # fully random placement when no useful foreground is found
        # (full-frame objects like mvtec_cable, detection failure, or cv2
        # unavailable) — preserving the legacy behavior exactly. Both draw
        # from `rng`, so a fixed seed yields a deterministic result.
        fgm = _foreground_mask(nrgb)
        pos = None
        if fgm is not None:
            pos = _foreground_paste_position(
                fgm, nrm.size, defect_crop.size, mask, rng
            )
        gate_ok = True
        if pos is not None and gate_active:
            # Foreground-path gate: resample the foreground position while the
            # local patch fails the texture/compat gate. All-fail → keep the
            # last candidate with gate_ok=False so stage 2 can escape to
            # another normal. gate_active=False consumes zero extra draws.
            gate_ok = _gate_pass(pos)
            for _ in range(max(1, max_bg_tries)):
                if gate_ok:
                    break
                cand = _foreground_paste_position(
                    fgm, nrm.size, defect_crop.size, mask, rng
                )
                if cand is None:
                    break
                pos = cand
                gate_ok = _gate_pass(pos)
        if pos is None:
            # Random-placement fallback. When a gate is on, resample the
            # position (same rng → deterministic) whenever the crop-sized
            # local background patch is black/flat (void), texture-incompatible
            # with the source surroundings, or a low-compat context cell.
            # All-fail → keep the last candidate so synthesis is still produced
            # (never a silent drop); the caller may then re-pick (stage 2).
            pos = _random_paste_position(nrm.size, defect_crop.size, rng)
            if reject_clean_bg or gate_active:
                gate_ok = False
                for _ in range(max(1, max_bg_tries)):
                    px, py = pos
                    patch = nrgb[py:py + crop_h, px:px + crop_w]
                    clean = (not reject_clean_bg) or _is_clean_background(
                        patch, min_quality=min_bg_quality,
                        blur_threshold=blur_threshold)
                    if clean and _gate_pass(pos):
                        gate_ok = True
                        break
                    pos = _random_paste_position(
                        nrm.size, defect_crop.size, rng
                    )
        return nrgb, pos, gate_ok

    used_normal_path: Optional[str] = None
    normal_rgb, position, gate_ok = _place_on(normal_img)

    # Stage 2 — different-normal re-pick. checkerplate-like mismatches are
    # position-invariant within one normal, so escaping to another normal is
    # the rescue path that actually works. Only ever loops when the texture
    # gate is on AND a pool was supplied; draws come from the same `rng`
    # (controlnet passes place_rng=Random(seed0) → content-addressed).
    n_repick = 0
    if (texture_on or compat_on) and not gate_ok and normal_pool:
        from PIL import Image as _PILImage
        for _ in range(_TEXTURE_MAX_NORMAL_REPICK):
            n_repick += 1
            cand_path = normal_pool[rng.randrange(len(normal_pool))]
            try:
                cand_img = _PILImage.open(cand_path).convert("RGBA")
            except Exception:
                continue
            # The fit-rescale above ran against the FIRST normal only; skip
            # re-picked candidates the crop no longer fits (rare — pools are
            # size-homogeneous on the texture datasets that reach this path).
            if crop_w > cand_img.size[0] or crop_h > cand_img.size[1]:
                continue
            normal_img = cand_img
            normal_rgb, position, gate_ok = _place_on(cand_img)
            # Record EVERY switch (not just accepted ones): on exhaustion the
            # paste lands on the LAST candidate, and the annotation must point
            # at the background the composite was actually made from.
            used_normal_path = cand_path
            if gate_ok:
                break
        if used_normal_path is not None:
            # Used normal changed → refresh the GT-mask canvas size below.
            bw_norm, bh_norm = normal_img.size

    if (texture_on or compat_on) and not gate_ok:
        logger.warning(
            "placement gate exhausted (%d positions x %d normal re-picks; "
            "tex>%.3f compat<%.2f) — pasting last candidate: %s",
            max_bg_tries, n_repick, texture_dist_threshold, compat_threshold,
            defect_path_for_log,
        )
    # Context-aware ('seamless') blending = Reinhard local-background colour/
    # illumination transfer + cv2.seamlessClone, falling back to the alpha
    # composite on any failure or when cv2 is unavailable. The position
    # decision above (foreground / void / clean-bg gate) is unchanged — only
    # HOW the chosen crop is composited at the chosen position differs.
    if blend_mode == "seamless":
        result = _context_aware_composite(
            normal_img, defect_crop, mask, position, feather_px
        )
    else:
        result = _alpha_composite(
            normal_img, defect_crop, mask, position, feather_px
        )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    result.convert("RGB").save(output_path)

    # bbox is reported in the COMPOSITE (background) coordinate system and
    # describes only the pasted defect region — fixing the prior bug where
    # the full image footprint [0,0,W,H] was returned.
    x_paste, y_paste = int(position[0]), int(position[1])
    bbox = [x_paste, y_paste, int(crop_w), int(crop_h)]

    # Persist GT mask in the composite coordinate system. Mask saving is
    # isolated so a write failure degrades to mask_path=None (legacy) rather
    # than discarding the already-saved composite image.
    saved_mask_path: Optional[str] = None
    if mask_output_path is not None:
        try:
            full_mask = PILImage.new("L", (bw_norm, bh_norm), 0)
            full_mask.paste(mask, (x_paste, y_paste))  # crisp pre-feather mask
            Path(mask_output_path).parent.mkdir(parents=True, exist_ok=True)
            full_mask.save(mask_output_path)  # PNG (lossless) — caller uses .png
            saved_mask_path = mask_output_path
        except Exception as mexc:
            logger.warning(
                "GT mask save failed for %s (composite kept, mask_path=None): %s",
                output_path, mexc,
            )

    meta: Dict[str, Any] = {"bbox": bbox, "mask_path": saved_mask_path}
    if used_normal_path is not None:
        meta["normal_image"] = used_normal_path
    # Per-call gate telemetry — run() pops and aggregates this (it must never
    # leak into annotations.json).
    meta["gate_stats"] = {
        "active": texture_on or compat_on,
        "texture_active": texture_on,
        "compat_active": compat_on,
        "repick": n_repick,
        "fallback": bool((texture_on or compat_on) and not gate_ok),
    }
    return meta


def copy_paste_synthesis(
    roi_entry: Dict[str, Any],
    normal_image_path: str,
    output_path: str,
    blend_mode: str = "alpha",
    feather_px: int = 4,
    rng: Optional[random.Random] = None,
    mask_output_path: Optional[str] = None,
    reject_clean_bg: bool = False,
    min_bg_quality: float = 0.7,
    blur_threshold: float = 100.0,
    max_bg_tries: int = 20,
    normal_pool: Optional[List[str]] = None,
    texture_dist_threshold: float = 0.0,
    compat_row: Optional[Dict[str, float]] = None,
    bin_edges: Optional[Dict[str, List[float]]] = None,
    compat_threshold: float = 0.0,
    compat_tile: bool = False,
) -> Optional[Dict[str, Any]]:
    """
    Paste the defect region from roi_entry onto the normal image.

    Args:
        roi_entry:          A single ROI dict (from roi_selected.json)
        normal_image_path:  Path to normal (good) background image
        output_path:        Destination path for the synthetic image
        blend_mode:         'alpha' (default) or 'seamless' (context-aware
                            Reinhard transfer + cv2.seamlessClone; alpha fallback)
        feather_px:         Edge feathering sigma in pixels
        rng:                Optional seeded random.Random for reproducibility
        mask_output_path:   Optional PNG path to persist the GT defect mask.
                            When given, a full-frame (background-sized) mask is
                            saved so downstream supervised pipelines can recover
                            an exact bbox instead of re-deriving one by image
                            differencing. None → no mask written (legacy).
        reject_clean_bg:    When True, the random-placement fallback rejects
                            candidate positions that land on a black/flat (void)
                            background patch and retries (same rng). The
                            foreground-constrained path is never gated. Default
                            False (legacy behavior).
        min_bg_quality:     Minimum background quality (0..1) for a candidate
                            position to be accepted when reject_clean_bg=True.
        max_bg_tries:       Max position resamples before giving up and pasting
                            on the last candidate (synthesis is preserved).
        normal_pool:        Optional list of alternative normal paths for the
                            texture gate's stage-2 re-pick.
        texture_dist_threshold: Pair-level texture gate threshold (0..1);
                            0.0 = OFF (default, legacy rng stream preserved).

    Returns:
        On success, a dict ``{"bbox": [x, y, w, h], "mask_path": <str|None>}``
        in the synthetic-image coordinate system, where bbox describes ONLY the
        pasted defect region. None on error (image missing, PIL unavailable, or
        the defect crop is larger than the background). When ``roi_entry``
        carries ``defect_bbox`` + ``defect_mask_path`` the real GT mask is used;
        otherwise a pre-feather ellipse fallback keeps bbox geometry independent
        of feathering / scipy availability.
    """
    if not HAS_PIL:
        logger.error("Pillow required for copy_paste — install with: pip install Pillow")
        return None

    from PIL import Image as PILImage

    if rng is None:
        rng = random.Random(42)

    defect_path = roi_entry.get("image_path", "")
    if not defect_path or not Path(defect_path).exists():
        logger.warning("Defect image not found: %s", defect_path)
        return None

    if not Path(normal_image_path).exists():
        logger.warning("Normal image not found: %s", normal_image_path)
        return None

    try:
        defect_img = PILImage.open(defect_path).convert("RGB")
        normal_img = PILImage.open(normal_image_path).convert("RGBA")

        # --- Resolve the defect crop + region mask ----------------------------
        # Preferred path (B-plan): an exact GT bbox + full-frame binary mask were
        # persisted upstream (distribution_profiling → roi_selection). Crop both
        # the defect and its mask to that bbox so we paste ONLY the real defect
        # region. Legacy fallback: full-image crop + synthetic ellipse mask.
        defect_bbox = roi_entry.get("defect_bbox")
        defect_mask_path = roi_entry.get("defect_mask_path") or ""

        use_real_mask = (
            isinstance(defect_bbox, (list, tuple)) and len(defect_bbox) == 4
            and defect_mask_path and Path(defect_mask_path).exists()
        )

        # Validate bbox lies within the defect image (stale roi_selected.json may
        # reference a different-resolution image). Out-of-bounds → ellipse fallback.
        if use_real_mask:
            try:
                bx, by, bw_box, bh_box = (int(v) for v in defect_bbox)
            except (TypeError, ValueError):
                use_real_mask = False
            else:
                dw, dh = defect_img.size  # (width, height)
                if not (
                    bw_box > 0 and bh_box > 0
                    and bx >= 0 and by >= 0
                    and bx + bw_box <= dw and by + bh_box <= dh
                ):
                    logger.warning(
                        "defect_bbox %s out of bounds for defect image %dx%d — "
                        "ellipse fallback for %s",
                        defect_bbox, dw, dh, defect_path,
                    )
                    use_real_mask = False

        source_bg_desc = None
        if use_real_mask:
            mask_full = PILImage.open(defect_mask_path).convert("L")
            box = (bx, by, bx + bw_box, by + bh_box)
            defect_crop = defect_img.crop(box).convert("RGBA")
            mask = mask_full.crop(box)
            # Texture gate source side: descriptor of the background band
            # around the real defect mask. Ellipse-fallback entries have no
            # real surroundings → descriptor stays None → gate auto-passes.
            if texture_dist_threshold > 0.0:
                source_bg_desc = _source_bg_descriptor(
                    np.asarray(defect_img), np.asarray(mask_full),
                    (bx, by, bw_box, bh_box),
                )
        else:
            # Legacy: use full defect image as crop with a simple elliptical mask
            cw, ch = defect_img.size
            mask_arr = np.zeros((ch, cw), dtype=np.uint8)
            cy, cx = ch // 2, cw // 2
            ry, rx = max(1, ch // 2 - 2), max(1, cw // 2 - 2)
            Y, X = np.ogrid[:ch, :cw]
            ellipse = ((X - cx) ** 2 / (rx ** 2) + (Y - cy) ** 2 / (ry ** 2)) <= 1.0
            mask_arr[ellipse] = 255
            mask = PILImage.fromarray(mask_arr)
            defect_crop = defect_img.convert("RGBA")

        return _paste_and_finalize(
            normal_img, defect_crop, mask, output_path,
            blend_mode=blend_mode,
            feather_px=feather_px,
            rng=rng,
            mask_output_path=mask_output_path,
            reject_clean_bg=reject_clean_bg,
            min_bg_quality=min_bg_quality,
            blur_threshold=blur_threshold,
            max_bg_tries=max_bg_tries,
            defect_path_for_log=defect_path,
            source_bg_desc=source_bg_desc,
            normal_pool=normal_pool,
            texture_dist_threshold=texture_dist_threshold,
            compat_row=compat_row,
            bin_edges=bin_edges,
            compat_threshold=compat_threshold,
            compat_tile=compat_tile,
        )

    except Exception as exc:
        logger.error("copy_paste_synthesis failed for %s: %s", defect_path, exc)
        return None


# ---------------------------------------------------------------------------
# ControlNet synthesis (generate defect texture → paste at AROMA-selected ROI)
# ---------------------------------------------------------------------------
# Design: the trained ControlNet renders only the defect-crop PIXELS (texture);
# the paste mask / GT mask stay the real seed mask, and placement + blending +
# recording go through _paste_and_finalize — the exact copy_paste machinery.
# The seed-shape GT mask keeps every accepted synthetic labelable (bbox from a
# real mask), structurally avoiding the dev_note ⑧ empty-mask parity collapse.
# Hint + prompt are rebuilt with the SAME generators/joins build_train_jsonl.py
# used for training, so inference stays in the training distribution.

_CN_BLANK_STD = 6.0      # masked-region std below this → blank (no defect drawn)
_CN_BLANK_RANGE = 12.0   # masked-region max-min below this → low-contrast collapse
_CN_MEAN_LO = 3.0        # near-all-black output
_CN_MEAN_HI = 252.0      # near-all-white output
_CN_MAX_ATTEMPTS = 3     # initial try + 2 seed-shifted retries

_CN_CTX: Optional[Dict[str, Any]] = None


def _cn_model_sig(controlnet_path: str) -> str:
    """Weights-file signature (size+mtime) so a retrained model at the SAME
    path invalidates the generation cache (fingerprint mismatch)."""
    for name in ("diffusion_pytorch_model.safetensors",
                 "diffusion_pytorch_model.bin"):
        p = Path(controlnet_path) / name
        if p.exists():
            st = p.stat()
            return f"{name}:{st.st_size}:{int(st.st_mtime)}"
    return controlnet_path  # remote hub id etc. — path-only fallback


def _configure_controlnet_context(
    controlnet_path: str,
    sd_base: str,
    morphology_csv: str,
    context_features: str,
    config_yaml: str,
    roi_candidates: Optional[str] = None,
    steps: int = 30,
    cond_scale: float = 0.7,
    resolution: int = 512,
    grayscale: bool = True,
    default_bg: str = "complex_pattern",
    use_cache: bool = True,
    ar_threshold: float = 2.5,
    ar_fallback: bool = True,
) -> Dict[str, Any]:
    """Build the ControlNet inference context ONCE (join tables + generators).

    Loads the same morphology/context/config/candidate inputs
    build_train_jsonl.py used for training so per-call work is dict lookups
    only. torch/diffusers load later, on the first generation
    (_get_cn_pipeline). Raises on missing inputs so a misconfigured run fails
    fast before the ROI loop.
    """
    global _CN_CTX
    here = str(Path(__file__).resolve().parent)
    if here not in sys.path:
        sys.path.insert(0, here)

    import aroma_to_casda_roi as _adapter  # noqa: PLC0415 — sibling, lazy
    import build_train_jsonl as _btj       # noqa: PLC0415 — sibling, lazy
    from utils.hint_generator import HintImageGenerator    # noqa: PLC0415
    from utils.prompt_generator import PromptGenerator     # noqa: PLC0415

    features, bin_edges = _btj._load_bin_edges(config_yaml)
    context_means = _btj._load_context_means(context_features, features)
    by_mask, by_img_bbox = _adapter._load_morphology(morphology_csv)

    # Training-side defect_subtype (morph_label) + stability (MAX ctx_prior)
    # come from roi_candidates.json (build_train_jsonl._load_candidate_join).
    # Optional: on a miss we fall back to the roi_selected entry fields.
    candidate_join: Dict[Any, Dict[str, Any]] = {}
    if roi_candidates and Path(roi_candidates).exists():
        candidate_join = _btj._load_candidate_join(roi_candidates)
    elif roi_candidates:
        logger.warning("roi_candidates not found: %s — prompt subtype/stability "
                       "fall back to roi_selected entry fields", roi_candidates)

    prompt_gen = PromptGenerator(style="technical")
    fingerprint = (
        f"{_cn_model_sig(controlnet_path)}|{sd_base}|{int(steps)}|"
        f"{float(cond_scale)}|{int(resolution)}|{bool(grayscale)}"
    )
    _CN_CTX = {
        "controlnet_path": controlnet_path,
        "sd_base": sd_base,
        "steps": int(steps),
        "cond_scale": float(cond_scale),
        "resolution": int(resolution),
        "grayscale": bool(grayscale),
        "default_bg": default_bg,
        "use_cache": bool(use_cache),
        "ar_threshold": float(ar_threshold),
        "ar_fallback": bool(ar_fallback),
        "fingerprint": fingerprint,
        "features": features,
        "bin_edges": bin_edges,
        "context_means": context_means,
        "by_mask": by_mask,
        "by_img_bbox": by_img_bbox,
        "candidate_join": candidate_join,
        "adapter": _adapter,
        "btj": _btj,
        "hint_gen": HintImageGenerator(),
        "prompt_gen": prompt_gen,
        "negative_prompt": prompt_gen.generate_negative_prompt(),
        "pipe": None,
        "torch": None,
        "device": None,
        "stats": {
            "calls": 0, "gen_ok": 0, "cache_hit": 0, "blank_retry": 0,
            "skip_blank": 0, "oom_retry": 0, "skip_oom": 0, "skip_no_mask": 0,
            "skip_ar": 0, "ar_fallback": 0, "skip_error": 0, "join_miss": 0,
            "bg_fallback": 0,
        },
    }
    logger.info(
        "ControlNet context ready: model=%s base=%s steps=%d scale=%.2f res=%d "
        "(morph rows=%d, context ids=%d, candidate keys=%d)",
        controlnet_path, sd_base, steps, cond_scale, resolution,
        len(by_mask), len(context_means), len(candidate_join),
    )
    return _CN_CTX


def _get_cn_pipeline(ctx: Dict[str, Any]):
    """Lazy-singleton diffusers pipeline — loaded once on the first generation."""
    if ctx["pipe"] is not None:
        return ctx["pipe"]
    import torch  # noqa: PLC0415 — GPU-only dependency, never at module top
    from diffusers import (  # noqa: PLC0415
        ControlNetModel,
        StableDiffusionControlNetPipeline,
        UniPCMultistepScheduler,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    controlnet = ControlNetModel.from_pretrained(
        ctx["controlnet_path"], torch_dtype=dtype
    )
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        ctx["sd_base"], controlnet=controlnet, safety_checker=None,
        torch_dtype=dtype,
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    pipe.enable_attention_slicing()
    pipe.enable_vae_slicing()
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception:
        pass  # recent torch falls back to SDPA
    ctx["pipe"] = pipe
    ctx["torch"] = torch
    ctx["device"] = device
    logger.info("ControlNet pipeline loaded on %s (%s)", device, dtype)
    return pipe


def _cn_rep_idx(output_path: str) -> int:
    """rep_idx from run()'s output naming ``syn_{roi:05d}_{rep:02d}``."""
    m = re.search(r"_(\d+)_(\d+)$", Path(output_path).stem)
    return int(m.group(2)) if m else 0


def _cn_content_seed(
    image_id: str, defect_bbox: Any, cell_key: str, rep_idx: int
) -> int:
    """Content-addressed diffusion seed: (defect identity, cell, repetition).

    Independent of ROI iteration order and normal-pool draws, so a resumed or
    partial rerun regenerates identical latents. cell_key disambiguates the
    same (image_id, bbox) selected into multiple context cells — without it
    those duplicates would collapse to identical latents AND identical
    placement rng.
    """
    import hashlib  # noqa: PLC0415
    key = f"{image_id}|{list(defect_bbox or [])}|{cell_key}|{rep_idx}"
    return int(hashlib.sha1(key.encode("utf-8")).hexdigest()[:15], 16)


def _cn_sidecar_path(output_path: str) -> Path:
    return Path(str(output_path) + ".meta.json")


def _cn_load_cached(
    output_path: str, mask_output_path: Optional[str], expected_fp: str
) -> Optional[Dict[str, Any]]:
    """Session-resume cache: reuse an already-generated composite without GPU.

    Valid only when the composite, its sidecar meta, and (if a mask was saved)
    the GT mask all exist AND the sidecar fingerprint matches the current
    sample identity + generation parameters + model weights signature — a
    retrained model or changed steps/scale/roi_selected invalidates stale
    outputs instead of silently reusing them.

    Returns the copy_paste-shaped meta dict (plus ``normal_image`` — the
    background actually used at generation time), else None.
    """
    out = Path(output_path)
    side = _cn_sidecar_path(output_path)
    if not (out.exists() and out.stat().st_size > 0 and side.exists()):
        return None
    try:
        cached = json.loads(side.read_text(encoding="utf-8"))
        if cached.get("fp") != expected_fp:
            return None
        bbox = cached.get("bbox")
        if not (isinstance(bbox, list) and len(bbox) == 4):
            return None
        meta = {"bbox": bbox, "mask_path": None,
                "normal_image": cached.get("normal_image")}
        if cached.get("mask_path"):
            if not (mask_output_path and Path(mask_output_path).exists()):
                return None
            meta["mask_path"] = mask_output_path
        return meta
    except Exception:
        return None


def _cn_is_blank(gen_img: "Image", mask_img: Optional["Image"]) -> bool:
    """True when the generated patch carries no defect signal in the seed region."""
    from PIL import Image as PILImage  # noqa: PLC0415

    arr = np.asarray(gen_img.convert("L"), dtype=np.float32)
    if not np.isfinite(arr).all():
        return True
    region = arr
    if mask_img is not None:
        m = np.asarray(mask_img.convert("L").resize(gen_img.size, PILImage.NEAREST))
        if int((m > 0).sum()) >= 16:
            region = arr[m > 0]
    mean = float(region.mean())
    if mean < _CN_MEAN_LO or mean > _CN_MEAN_HI:
        return True
    if float(region.std()) < _CN_BLANK_STD:
        return True
    if float(region.max() - region.min()) < _CN_BLANK_RANGE:
        return True
    return False


def _cn_generate(
    ctx: Dict[str, Any],
    prompt: str,
    hint_img: "Image",
    seed0: int,
    blank_mask: Optional["Image"],
    sample_label: str = "?",
) -> Optional["Image"]:
    """Diffusion call with blank/OOM handling: seed-shifted retries, then skip."""
    pipe = _get_cn_pipeline(ctx)
    torch = ctx["torch"]
    stats = ctx["stats"]
    oom_cls = getattr(getattr(torch, "cuda", None), "OutOfMemoryError", ())
    oom_seen = False
    last_fail = "blank"
    for attempt in range(_CN_MAX_ATTEMPTS):
        generator = torch.Generator(device=ctx["device"]).manual_seed(seed0 + attempt)
        try:
            with torch.inference_mode():
                image = pipe(
                    prompt=prompt,
                    negative_prompt=ctx["negative_prompt"],
                    image=hint_img,
                    num_inference_steps=ctx["steps"],
                    generator=generator,
                    height=ctx["resolution"],
                    width=ctx["resolution"],
                    controlnet_conditioning_scale=ctx["cond_scale"],
                ).images[0]
        except Exception as exc:
            is_oom = (isinstance(exc, oom_cls)
                      or "out of memory" in str(exc).lower())
            if is_oom:
                stats["oom_retry"] += 1
                last_fail = "oom"
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                if oom_seen:  # second OOM → give up on this sample
                    stats["skip_oom"] += 1
                    logger.warning("controlnet OOM twice — skip %s", sample_label)
                    return None
                oom_seen = True
                continue
            raise
        if _cn_is_blank(image, blank_mask):
            stats["blank_retry"] += 1
            last_fail = "blank"
            continue
        return image
    stats["skip_oom" if last_fail == "oom" else "skip_blank"] += 1
    logger.warning("controlnet %s after %d attempts — skip %s",
                   last_fail, _CN_MAX_ATTEMPTS, sample_label)
    return None


def controlnet_synthesis(
    roi_entry: Dict[str, Any],
    normal_image_path: str,
    output_path: str,
    **kwargs: Any,
) -> Optional[Dict[str, Any]]:
    """ControlNet defect generation → paste at the AROMA-selected ROI.

    Requires _configure_controlnet_context() to have run — run() wires it when
    ``--method controlnet`` is used. A direct unconfigured call keeps the
    legacy stub behavior (NotImplementedError).

    Returns the copy_paste contract: ``{"bbox": [x,y,w,h], "mask_path": ...}``
    or None (skip → run() counts n_skip). Blank/OOM shortfalls surface in the
    stats counters. AR-gated elongated ROIs fall back to copy_paste by default
    (count/class/bbox parity vs random/aroma-CP; meta tagged
    method=copy_paste_arfallback) — set ar_fallback=False for a pure arm.
    """
    ctx = _CN_CTX
    if ctx is None:
        raise NotImplementedError(
            "controlnet_synthesis requires _configure_controlnet_context(); "
            "use run(--method controlnet) with --controlnet_path/--morphology_csv/"
            "--context_features/--config."
        )
    if not HAS_PIL:
        logger.error("Pillow required for controlnet synthesis")
        return None

    from PIL import Image as PILImage

    stats = ctx["stats"]
    stats["calls"] += 1
    if stats["calls"] % 100 == 0:  # heartbeat for multi-hour Colab runs
        logger.info(
            "controlnet progress: %d calls (gen_ok=%d cache_hit=%d "
            "skip_blank=%d skip_oom=%d)",
            stats["calls"], stats["gen_ok"], stats["cache_hit"],
            stats["skip_blank"], stats["skip_oom"],
        )
    mask_output_path = kwargs.get("mask_output_path")

    try:
        # ---- 1) Resolve seed crop geometry (real GT mask REQUIRED) ----------
        # Unlike copy_paste there is no ellipse fallback: the hint R channel and
        # the GT mask both need the real seed shape.
        defect_path = roi_entry.get("image_path", "")
        defect_mask_path = roi_entry.get("defect_mask_path") or ""
        adapter = ctx["adapter"]
        bbox = adapter._as_xywh(roi_entry.get("defect_bbox"))
        if (not defect_path or not Path(defect_path).exists()
                or not defect_mask_path or not Path(defect_mask_path).exists()
                or bbox is None):
            stats["skip_no_mask"] += 1
            logger.warning("controlnet: missing defect image/mask/bbox for %s — skip",
                           roi_entry.get("image_id", "?"))
            return None

        defect_img = PILImage.open(defect_path).convert("RGB")
        dw, dh = defect_img.size
        bx, by, bw_box, bh_box = bbox
        if not (bw_box > 0 and bh_box > 0 and bx >= 0 and by >= 0
                and bx + bw_box <= dw and by + bh_box <= dh):
            stats["skip_no_mask"] += 1
            logger.warning("controlnet: defect_bbox %s out of bounds (%dx%d) — skip",
                           bbox, dw, dh)
            return None
        # ---- 1a) Aspect-ratio gate (elongated bbox → squash-unsquash smear) --
        # The full bbox is squashed to a resolution² square for generation and
        # un-squashed back (see the Geometry note below). severstal pilot: every
        # visual artifact had AR >= 3.1 (horizontal smear / flat patch), while
        # near-square bboxes were clean — so elongated ROIs are skipped instead
        # of visually gated by hand. Placed AFTER the bounds check (min >= 1,
        # no div-by-zero) and BEFORE the cache probe (no sidecar is written, so
        # a stale elongated composite can never be served from cache). '>' —
        # an AR exactly at the threshold passes. 0 disables the gate.
        ar_threshold = float(ctx.get("ar_threshold", 2.5))
        aspect = max(bw_box, bh_box) / min(bw_box, bh_box)
        if ar_threshold > 0 and aspect > ar_threshold:
            # Elongated bbox: ControlNet squash-unsquash smears it. Rather than
            # drop the ROI (which would strip elongated classes from THIS arm
            # only → composition confound vs random/aroma-CP), fall back to
            # copy_paste for this ROI so count + class + bbox stay at full
            # parity. aroma-CN vs aroma-CP then differ ONLY on the non-elongated
            # defects ControlNet actually synthesizes. Set ar_fallback=False to
            # restore the pure-arm skip. copy_paste inherits the SAME texture
            # gate so background distribution stays arm-consistent.
            if ctx.get("ar_fallback", True):
                stats["ar_fallback"] += 1
                logger.warning(
                    "controlnet: bbox AR %.2f > %.2f (%dx%d) — copy_paste "
                    "fallback %s", aspect, ar_threshold, bw_box, bh_box,
                    roi_entry.get("image_id", "?"),
                )
                fb = copy_paste_synthesis(
                    roi_entry=roi_entry,
                    normal_image_path=normal_image_path,
                    output_path=output_path,
                    blend_mode=kwargs.get("blend_mode", "alpha"),
                    feather_px=kwargs.get("feather_px", 4),
                    rng=kwargs.get("rng"),
                    mask_output_path=mask_output_path,
                    reject_clean_bg=kwargs.get("reject_clean_bg", False),
                    min_bg_quality=kwargs.get("min_bg_quality", 0.7),
                    blur_threshold=kwargs.get("blur_threshold", 100.0),
                    normal_pool=kwargs.get("normal_pool"),
                    texture_dist_threshold=kwargs.get("texture_dist_threshold", 0.0),
                )
                if fb is not None:
                    fb["method"] = "copy_paste_arfallback"
                return fb
            stats["skip_ar"] += 1
            logger.warning(
                "controlnet: bbox AR %.2f > %.2f (%dx%d) — skip %s",
                aspect, ar_threshold, bw_box, bh_box,
                roi_entry.get("image_id", "?"),
            )
            return None
        box = (bx, by, bx + bw_box, by + bh_box)
        mask = PILImage.open(defect_mask_path).convert("L").crop(box)

        image_id = str(roi_entry.get("image_id", ""))
        cell_key = str(roi_entry.get("cell_key", ""))
        seed0 = _cn_content_seed(image_id, bbox, cell_key, _cn_rep_idx(output_path))
        sample_fp = f"{ctx['fingerprint']}|{image_id}|{bbox}|{seed0}"

        # ---- 1b) Session-resume cache: skip the GPU on a fingerprint match --
        if ctx["use_cache"]:
            cached = _cn_load_cached(output_path, mask_output_path, sample_fp)
            if cached is not None:
                stats["cache_hit"] += 1
                return cached

        # ---- 2) Rebuild hint + prompt with the training-side generators -----
        btj = ctx["btj"]
        morph_row, matched = adapter._join_metrics(
            roi_entry, ctx["by_mask"], ctx["by_img_bbox"]
        )
        if not matched:
            stats["join_miss"] += 1
        metrics = btj._metrics_from_row(morph_row) if morph_row else {}

        # Training parity: build_train_jsonl sourced defect_subtype from the
        # candidate join's morph_label and stability from the defect's MAX
        # ctx_prior across cells — prefer those; entry fields are fallbacks.
        cand = ctx["candidate_join"].get((image_id, defect_mask_path))
        defect_subtype = ((cand or {}).get("morph_label")
                          or roi_entry.get("morph_label")
                          or roi_entry.get("defect_subtype") or "general")
        stability_raw = (cand or {}).get("ctx_prior")
        if stability_raw is None:
            stability_raw = roi_entry.get("ctx_prior")
        try:
            stability = float(stability_raw)
        except (TypeError, ValueError):
            stability = 0.5

        background_type = btj._derive_background_type(
            image_id, ctx["context_means"], ctx["features"], ctx["bin_edges"]
        )
        if background_type is None:
            stats["bg_fallback"] += 1
            background_type = ctx["default_bg"]

        roi_image_arr = adapter._read_gray_or_color(defect_path, gray=False)
        roi_mask_arr = adapter._read_gray_or_color(defect_mask_path, gray=True)
        if roi_image_arr is None or roi_mask_arr is None:
            stats["skip_error"] += 1
            logger.warning("controlnet: failed to reload image/mask for hint "
                           "(image_id=%s) — skip", image_id or "?")
            return None
        img_crop = adapter._crop_xywh(roi_image_arr, bbox)
        mask_crop = adapter._crop_xywh(roi_mask_arr, bbox)
        if img_crop is None or mask_crop is None:
            stats["skip_error"] += 1
            logger.warning("controlnet: degenerate hint crop (image_id=%s) — skip",
                           image_id or "?")
            return None

        hint_arr = ctx["hint_gen"].generate_hint_image(
            img_crop, mask_crop, metrics, background_type, stability
        )
        # Geometry: the FULL bbox is squashed to a resolution² square (and the
        # generation is un-squashed back to bbox size below) so hint ↔ output ↔
        # GT mask stay exactly aligned. Training used Resize+CenterCrop — for
        # near-square crops the two nearly coincide; for elongated crops no
        # square-generation scheme matches training exactly, and the full-bbox
        # square keeps the alignment invariant (center-crop inference would
        # paste fabricated texture outside the generated window). Pilot-gate
        # elongated defects visually (guide STEP 5).
        res = ctx["resolution"]
        hint_img = PILImage.fromarray(hint_arr).resize((res, res), PILImage.BILINEAR)
        blank_mask = mask.resize((res, res), PILImage.NEAREST)

        # NOTE: roi_entry["prompt"] (Stage 2 free-form) is deliberately NOT used
        # — it is out-of-distribution for the trained model. The technical
        # prompt below matches build_train_jsonl exactly.
        prompt = ctx["prompt_gen"].generate_prompt(
            defect_subtype=defect_subtype,
            background_type=background_type,
            stability_score=stability,
            defect_metrics=metrics,
            suitability_score=stability,
        )

        # ---- 3) Generate (content-addressed seed; blank/OOM retry) ----------
        gen_img = _cn_generate(ctx, prompt, hint_img, seed0, blank_mask,
                               sample_label=f"{image_id} rep{_cn_rep_idx(output_path)}")
        if gen_img is None:
            return None
        if ctx["grayscale"]:
            g = gen_img.convert("L")  # match force_grayscale_target training
            gen_img = PILImage.merge("RGB", (g, g, g))

        # ---- 4) Paste with the shared copy_paste machinery ------------------
        defect_crop = gen_img.resize((bw_box, bh_box), PILImage.LANCZOS).convert("RGBA")
        normal_img = PILImage.open(normal_image_path).convert("RGBA")
        # Texture gate source side: the band around the SEED defect mask in
        # the seed image — the surroundings the generated texture must match.
        texture_dist_threshold = float(kwargs.get("texture_dist_threshold", 0.0))
        source_bg_desc = None
        if texture_dist_threshold > 0.0:
            source_bg_desc = _source_bg_descriptor(
                roi_image_arr, roi_mask_arr, (bx, by, bw_box, bh_box)
            )
        place_rng = random.Random(seed0)  # placement content-addressed too
        meta = _paste_and_finalize(
            normal_img, defect_crop, mask, output_path,
            blend_mode=kwargs.get("blend_mode", "alpha"),
            feather_px=kwargs.get("feather_px", 4),
            rng=place_rng,
            mask_output_path=mask_output_path,
            reject_clean_bg=kwargs.get("reject_clean_bg", False),
            min_bg_quality=kwargs.get("min_bg_quality", 0.7),
            blur_threshold=kwargs.get("blur_threshold", 100.0),
            defect_path_for_log=defect_path,
            source_bg_desc=source_bg_desc,
            normal_pool=kwargs.get("normal_pool"),
            texture_dist_threshold=texture_dist_threshold,
            compat_row=kwargs.get("compat_row"),
            bin_edges=kwargs.get("bin_edges"),
            compat_threshold=kwargs.get("compat_threshold", 0.0),
            compat_tile=kwargs.get("compat_tile", False),
        )
        if meta:
            stats["gen_ok"] += 1
            # The background actually used — run() records it in annotations
            # even on a later cache hit (when its own rng draw may differ).
            # setdefault: a stage-2 texture re-pick already recorded the path
            # it actually pasted on — never clobber it.
            meta.setdefault("normal_image", normal_image_path)
            if ctx["use_cache"]:
                try:
                    _cn_sidecar_path(output_path).write_text(
                        json.dumps({"fp": sample_fp,
                                    "bbox": meta["bbox"],
                                    "mask_path": meta.get("mask_path"),
                                    "normal_image": meta["normal_image"]}),
                        encoding="utf-8",
                    )
                except Exception:
                    pass  # cache is best-effort; composite is already saved
        return meta

    except Exception as exc:
        stats["skip_error"] += 1
        logger.error("controlnet_synthesis failed for %s: %s",
                     roi_entry.get("image_id", "?"), exc, exc_info=True)
        return None


def _log_cn_stats() -> Optional[Dict[str, int]]:
    """One-line summary of the ControlNet counters (parity/quality evidence)."""
    if _CN_CTX is None:
        return None
    s = _CN_CTX["stats"]
    denom = max(1, s["gen_ok"] + s["skip_blank"])
    blank_rate = s["skip_blank"] / denom
    logger.info(
        "controlnet stats: gen_ok=%d cache_hit=%d blank_retry=%d skip_blank=%d "
        "oom_retry=%d skip_oom=%d skip_no_mask=%d skip_ar=%d ar_fallback=%d "
        "skip_error=%d join_miss=%d bg_fallback=%d (blank_rate=%.3f)",
        s["gen_ok"], s["cache_hit"], s["blank_retry"], s["skip_blank"],
        s["oom_retry"], s["skip_oom"], s["skip_no_mask"], s.get("skip_ar", 0),
        s.get("ar_fallback", 0), s["skip_error"], s["join_miss"],
        s["bg_fallback"], blank_rate,
    )
    if blank_rate > 0.2:
        logger.warning(
            "controlnet blank_rate=%.3f > 0.2 — undertrained model or "
            "hint/prompt distribution drift suspected", blank_rate,
        )
    return dict(s)


def inpainting_synthesis(
    roi_entry: Dict[str, Any],
    normal_image_path: str,
    output_path: str,
    **kwargs: Any,
) -> Optional[Dict[str, Any]]:
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
    Returns (staged_selected, staged_normal_images, normal_map) with updated
    paths. normal_map maps original Drive path -> local staged path (for
    annotations that must record the original Drive path).
    """
    defect_dir = staging_dir / "defects"
    normal_dir_local = staging_dir / "normals"
    defect_dir.mkdir(parents=True, exist_ok=True)
    normal_dir_local.mkdir(parents=True, exist_ok=True)

    # An upstream step (e.g. casda_roi_adapter --local_staging) may already have
    # staged the defect crops to the local /content disk. Re-copying a local file
    # to another local dir is pure waste, so detect already-local defect paths and
    # skip the redundant local->local copy (still record orig->orig so the remap
    # below is a no-op identity). Normal backgrounds are NEVER pre-staged by the
    # adapter, so their staging (below) is left untouched.
    _local_roots = ("/content/", "\\content\\")

    def _is_already_local(path: str) -> bool:
        return any(path.startswith(r) for r in _local_roots)

    # Stage defect images (deduplicated)
    defect_map: Dict[str, str] = {}
    n_defect_already_local = 0
    for entry in selected:
        orig = entry.get("image_path", "")
        if orig and orig not in defect_map and Path(orig).exists():
            if _is_already_local(orig):
                # Already on local disk — keep the path as-is, skip re-copy.
                defect_map[orig] = orig
                n_defect_already_local += 1
                continue
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
        "Staged %d defect images (%d already local, copy skipped) + %d normal "
        "images → %s",
        len(defect_map), n_defect_already_local, len(normal_map), staging_dir,
    )
    return staged_selected, staged_normal, normal_map


# Push-out copy concurrency. Same workload shape as the adapter's stage-IN
# (latency-bound FUSE round-trips, NOT CPU-bound), so we reuse the SAME
# AROMA_STAGE_WORKERS knob and deliberately do NOT key off os.cpu_count()
# (Colab = 2 vCPU). These helpers are duplicated (not imported) from
# casda_roi_adapter.py ON PURPOSE: the adapter imports generate_defects, so
# importing it back here would create a circular import. The ~15 duplicated
# lines are preferred over a new shared module.
_DEFAULT_STAGE_WORKERS = 16
_STAGE_WORKERS_MIN = 1
_STAGE_WORKERS_MAX = 64
_COPY_RETRIES = 3                 # total attempts per file
_COPY_BACKOFF = (0.5, 1.0, 2.0)   # sleep before retry attempts 2,3,...


def _push_workers() -> int:
    """Resolve push concurrency from AROMA_STAGE_WORKERS, clamped to a sane range."""
    raw = os.environ.get("AROMA_STAGE_WORKERS")
    if not raw:
        return _DEFAULT_STAGE_WORKERS
    try:
        n = int(raw)
    except (TypeError, ValueError):
        logger.warning("AROMA_STAGE_WORKERS=%r not an int — using default %d",
                       raw, _DEFAULT_STAGE_WORKERS)
        return _DEFAULT_STAGE_WORKERS
    clamped = max(_STAGE_WORKERS_MIN, min(_STAGE_WORKERS_MAX, n))
    if clamped != n:
        logger.warning("AROMA_STAGE_WORKERS=%d clamped to %d (range %d..%d)",
                       n, clamped, _STAGE_WORKERS_MIN, _STAGE_WORKERS_MAX)
    return clamped


def _push_one(src: str, dst: str) -> str:
    """Copy one file src → dst with retry/backoff. Returns 'staged'|'skipped'|'failed'.

    - The dst parent dir (drive_img_dir) is pre-created once by the caller (main
      thread) before any worker runs, so this performs NO mkdir.
    - Skip (cheap stat) if dst already exists on Drive with the same size — lets
      a Colab --resume re-run avoid re-pushing everything.
    - On transient FUSE errors, retry up to _COPY_RETRIES with exponential backoff.
    - Terminal failure: log a WARNING and return 'failed' (do NOT raise) so one
      failed file never aborts the whole push.
    """
    dst_path = Path(dst)

    # Same-size skip: identical file already present on Drive.
    try:
        if dst_path.exists() and dst_path.stat().st_size == os.path.getsize(src):
            return "skipped"
    except OSError:
        pass  # fall through to (re)copy

    last_err: Optional[Exception] = None
    for attempt in range(_COPY_RETRIES):
        try:
            shutil.copy2(src, dst)
            return "staged"
        except Exception as e:  # noqa: BLE001 — transient FUSE EIO/5xx
            last_err = e
            if attempt < _COPY_RETRIES - 1:
                sleep_s = _COPY_BACKOFF[min(attempt, len(_COPY_BACKOFF) - 1)]
                time.sleep(sleep_s)
    logger.warning("Failed to push %s → %s after %d attempts: %s — left on local staging",
                   src, dst, _COPY_RETRIES, last_err)
    return "failed"


def _push_outputs(local_img_dir: Path, drive_img_dir: Path) -> int:
    """Parallel-copy synthesized files from local staging to Drive.

    Mirror of the adapter's stage-IN: output volume = n_rois * n_per_roi (tens of
    thousands of files), each a high-latency per-file FUSE write. A sequential
    loop took tens of minutes to hours; copies are dispatched on a bounded
    ThreadPoolExecutor instead (FUSE round-trips, not CPU, are the bottleneck).

    Returns the number of files now PRESENT on Drive (staged + skipped) so the
    call-site "Pushed %d ..." logs stay truthful; terminally failed files are
    excluded from the count and surfaced in the breakdown log.
    """
    drive_img_dir.mkdir(parents=True, exist_ok=True)  # pre-created ONCE in main thread

    files = [p for p in sorted(local_img_dir.iterdir()) if p.is_file()]
    if not files:
        return 0

    workers = _push_workers()
    counts = {"staged": 0, "skipped": 0, "failed": 0}

    def _task(p: Path) -> str:
        return _push_one(str(p), str(drive_img_dir / p.name))

    with ThreadPoolExecutor(max_workers=workers) as ex:
        # Mutate counts ONLY in the main-thread as_completed loop.
        for fut in as_completed(ex.submit(_task, p) for p in files):
            counts[fut.result()] += 1

    logger.info(
        "push_outputs → %s: %d staged, %d skipped (already on Drive), %d failed "
        "(left on local staging) of %d files using %d workers",
        drive_img_dir, counts["staged"], counts["skipped"], counts["failed"],
        len(files), workers,
    )
    # Files now present on Drive = staged + skipped (failed are NOT on Drive).
    return counts["staged"] + counts["skipped"]


# ---------------------------------------------------------------------------
# Normal image pool
# ---------------------------------------------------------------------------

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


def load_normal_images(
    normal_dir: str,
    *,
    reject_clean_bg: bool = False,
    min_bg_quality: float = 0.7,
    blur_threshold: float = 100.0,
) -> List[str]:
    """Collect all image paths under normal_dir (recursive, sorted).

    Args:
        normal_dir:       Directory of normal/good background images.
        reject_clean_bg:  When True, evaluate each image once with
                          ``_is_clean_background`` and keep only non black/flat
                          (void) backgrounds. cv2 missing → gate disabled.
        min_bg_quality:   Minimum quality (0..1) to keep an image.
        blur_threshold:   Laplacian-variance blur gate threshold.

    The rglob result is sorted for deterministic pool ordering (so a fixed seed
    yields the same `rng.choice` selection). When the gate rejects EVERY image
    (e.g. an all-black normal dir) it is ignored and the original pool is
    returned — a non-empty pool is always preferred over a silent 0-output.
    """
    d = Path(normal_dir)
    if not d.exists():
        logger.warning("Normal image directory not found: %s", normal_dir)
        return []
    paths = sorted(str(p) for p in d.rglob("*") if p.suffix.lower() in _IMAGE_EXTS)
    logger.info("Found %d normal images in %s", len(paths), normal_dir)

    if not reject_clean_bg or not paths:
        return paths

    if not HAS_CV2:
        logger.info("reject_clean_bg requested but cv2 unavailable — gate disabled.")
        return paths
    if not HAS_PIL:
        logger.info("reject_clean_bg requested but Pillow unavailable — gate disabled.")
        return paths

    from PIL import Image as PILImage

    kept: List[str] = []
    for p in paths:
        try:
            arr = np.asarray(PILImage.open(p).convert("RGB"))
        except Exception as exc:
            # Unreadable here is surfaced later by copy_paste_synthesis; keep it
            # in the pool rather than dropping on an evaluation error.
            logger.warning("Background quality eval failed for %s: %s", p, exc)
            kept.append(p)
            continue
        if _is_clean_background(arr, min_quality=min_bg_quality,
                                blur_threshold=blur_threshold):
            kept.append(p)

    n_reject = len(paths) - len(kept)
    logger.info(
        "clean-bg pool gate: kept %d / %d normal images (%d rejected, "
        "min_quality=%.2f, blur_threshold=%.1f)",
        len(kept), len(paths), n_reject, min_bg_quality, blur_threshold,
    )
    if not kept:
        logger.warning(
            "clean-bg pool gate rejected ALL %d normal images — ignoring gate "
            "and using original pool (avoids empty pool / 0 output).",
            len(paths),
        )
        return paths
    return kept


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
    reject_clean_bg: bool = False,
    min_bg_quality: float = 0.7,
    bg_blur_threshold: float = 100.0,
    texture_dist_threshold: float = 0.0,
    compat_threshold: float = 0.0,
    compat_matrix_json: Optional[str] = None,
    compat_mode: str = "defect",
    clean_bg_json: Optional[str] = None,
    controlnet_path: Optional[str] = None,
    sd_base: str = "runwayml/stable-diffusion-v1-5",
    cn_steps: int = 30,
    cn_cond_scale: float = 0.7,
    cn_resolution: int = 512,
    cn_grayscale: bool = True,
    cn_default_bg: str = "complex_pattern",
    cn_cache: bool = True,
    cn_ar_threshold: float = 2.5,
    cn_ar_fallback: bool = True,
    morphology_csv: Optional[str] = None,
    context_features: Optional[str] = None,
    config_yaml: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Full Step 4 pipeline: load ROIs → synthesize → save annotations.

    Args:
        roi_dir:        Step 3 output directory (roi_selected.json)
        normal_dir:     Directory of normal (good) background images
        output_dir:     Destination for synthetic images + annotations.json
        method:         Synthesis method: copy_paste | controlnet | inpainting
        n_per_roi:      Number of synthetic images to generate per ROI
        blend_mode:     Blending mode for copy_paste ('alpha' | 'seamless')
        feather_px:     Edge feather radius for alpha blend
        seed:           Random seed for reproducibility
        local_staging:  Copy inputs to /content/tmp before synthesis to avoid
                        per-image Drive I/O. Outputs are pushed back to Drive
                        output_dir at the end.
        reject_clean_bg:  Enable the black/flat (void) background gate. Applied
                          at BOTH injection points: the normal pool (per-image,
                          once at load) and the random-placement fallback
                          position. Default False (legacy). cv2 missing →
                          gate auto-disabled.
        min_bg_quality:   Minimum background quality (0..1). CASDA default 0.7.
        bg_blur_threshold: Laplacian-variance blur gate threshold. CASDA
                           default 100.0.
        texture_dist_threshold: Pair-level texture gate (0..1 distance
                          between the source defect's surroundings and the
                          candidate paste patch). Random-fallback placement
                          only, both copy_paste and controlnet. 0.0 = OFF
                          (default, legacy rng stream preserved). Tuning band
                          ~0.15-0.35; cv2 missing → gate auto-disabled.
        cn_ar_threshold:  Skip ControlNet ROIs whose bbox aspect ratio
                          max(w,h)/min(w,h) exceeds this (squash-unsquash
                          smear on elongated crops; severstal pilot artifact
                          boundary AR>=3.1). Default 2.5; 0 disables.
        controlnet_path / sd_base / cn_*: ControlNet generation settings, used
                          only with method='controlnet'. controlnet_path plus
                          morphology_csv/context_features/config_yaml are
                          REQUIRED there (hint/prompt parity with
                          build_train_jsonl.py); missing ones fail fast with
                          status 'controlnet_missing_args:...'.

    Returns:
        dict with status, n_generated, annotations list (+ cn_stats for
        method='controlnet')
    """
    roi_path = Path(roi_dir) / "roi_selected.json"
    if not roi_path.exists():
        logger.error("roi_selected.json not found in %s", roi_dir)
        return {"status": "missing_roi", "n_generated": 0, "annotations": []}

    selected: List[Dict[str, Any]] = load_json(str(roi_path))
    if not selected:
        logger.warning("roi_selected.json is empty")
        return {"status": "empty_roi", "n_generated": 0, "annotations": []}

    normal_images = load_normal_images(
        normal_dir,
        reject_clean_bg=reject_clean_bg,
        min_bg_quality=min_bg_quality,
        blur_threshold=bg_blur_threshold,
    )
    if not normal_images:
        logger.warning("No normal images found — dry run only")

    if method not in _SYNTHESIS_METHODS:
        return {"status": f"unknown_method:{method}", "n_generated": 0, "annotations": []}

    # ControlNet: fail fast on missing inputs, then build the inference context
    # (join tables + generators) ONCE before the ROI loop. The diffusers
    # pipeline itself loads lazily on the first generation call.
    if method == "controlnet":
        missing = [name for name, val in (
            ("--controlnet_path", controlnet_path),
            ("--morphology_csv", morphology_csv),
            ("--context_features", context_features),
            ("--config", config_yaml),
        ) if not val]
        if missing:
            logger.error("controlnet method requires: %s", ", ".join(missing))
            return {"status": f"controlnet_missing_args:{','.join(missing)}",
                    "n_generated": 0, "annotations": []}
        if local_staging and cn_cache:
            logger.warning(
                "--local_staging discards the ControlNet generation cache across "
                "sessions — for controlnet runs prefer a Drive output_dir "
                "WITHOUT --local_staging (session-resume then skips the GPU for "
                "already-generated images)."
            )
        _configure_controlnet_context(
            controlnet_path=controlnet_path,
            sd_base=sd_base,
            morphology_csv=morphology_csv,
            context_features=context_features,
            config_yaml=config_yaml,
            roi_candidates=str(Path(roi_dir) / "roi_candidates.json"),
            steps=cn_steps,
            cond_scale=cn_cond_scale,
            resolution=cn_resolution,
            grayscale=cn_grayscale,
            default_bg=cn_default_bg,
            use_cache=cn_cache,
            ar_threshold=cn_ar_threshold,
            ar_fallback=cn_ar_fallback,
        )

    # Save original Drive paths before staging overwrites them
    orig_source_roi = [e.get("image_path", "") for e in selected]

    # Local staging: copy inputs to fast local disk, run synthesis there
    staging_dir: Optional[Path] = None
    work_output_dir = output_dir

    # Map staged-local normal path -> original Drive path so annotations record
    # the durable Drive path (not the ephemeral /content/tmp staging path).
    staged_to_drive: Dict[str, str] = {}

    if local_staging and normal_images:
        staging_dir = _default_staging_dir(roi_dir)
        selected, normal_images, normal_map = _stage_inputs(selected, normal_images, staging_dir)
        staged_to_drive = {staged: orig for orig, staged in normal_map.items()}
        work_output_dir = str(staging_dir / "synthetic")
        logger.info("Synthesis will run in local staging dir: %s", work_output_dir)

    synthesis_fn = _SYNTHESIS_METHODS[method]
    out = Path(work_output_dir)
    img_dir = out / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    mask_dir = out / "masks"
    mask_dir.mkdir(parents=True, exist_ok=True)

    # Placement-aware compat gate: load the compatibility matrix + context
    # bin_edges ONCE (copy_paste path). Both required; missing either → gate
    # stays off (warned). controlnet passes the same kwargs uniformly.
    compat_matrix: Optional[Dict[str, Dict[str, float]]] = None
    compat_bin_edges: Optional[Dict[str, List[float]]] = None
    if compat_threshold > 0.0:
        if compat_matrix_json and Path(compat_matrix_json).exists() and config_yaml:
            try:
                _here = str(Path(__file__).resolve().parent)
                if _here not in sys.path:
                    sys.path.insert(0, _here)
                import build_train_jsonl as _btj_cfg  # noqa: PLC0415
                _compat_json = load_json(compat_matrix_json)
                # compat_mode selects the compat_row source. 'symmetric' uses the
                # SGM clean-grounded matrix (matrix_symmetric); 'defect' (default)
                # uses the legacy defect-side image-mean matrix — byte-identical
                # legacy behaviour.
                #
                # NO silent fallback for symmetric: compat_threshold is calibrated
                # against the [0,1] max-normalized symmetric scale, whereas the
                # legacy 'matrix' holds raw probabilities (compat_max typically
                # <0.2). Applying a symmetric-scale threshold to raw matrix values
                # silently changes accept/reject behaviour and mislabels the run —
                # so an older profile lacking 'matrix_symmetric' hard-fails, forcing
                # a profiling re-run (or explicit --compat_mode defect) instead.
                if compat_mode == "symmetric":
                    compat_matrix = _compat_json.get("matrix_symmetric")
                    if compat_matrix is None:
                        raise ValueError(
                            "compat_mode=symmetric requires 'matrix_symmetric' in "
                            f"{compat_matrix_json}, but it is absent (older profile). "
                            "Re-run distribution_profiling to emit it, or use "
                            "--compat_mode defect."
                        )
                else:
                    compat_matrix = _compat_json.get("matrix", {})
                _, compat_bin_edges = _btj_cfg._load_bin_edges(config_yaml)
                logger.info("compat gate ON: threshold=%.2f, mode=%s, %d clusters in matrix",
                            compat_threshold, compat_mode, len(compat_matrix))
            except ValueError:
                # compat_mode=symmetric misconfiguration — hard user error, do
                # NOT silently disable the gate (would mislabel the run).
                raise
            except Exception as exc:
                logger.warning("compat gate: failed to load matrix/bin_edges "
                               "(%s) — gate disabled", exc)
                compat_matrix = compat_bin_edges = None
        else:
            logger.warning("compat gate: --compat_threshold>0 requires "
                           "--compat_matrix_json AND --config — gate disabled")

    rng = random.Random(seed)
    annotations: List[Dict[str, Any]] = []
    n_ok = 0
    n_skip = 0
    gate_agg = {"active": 0, "repick": 0, "fallback": 0,
                "texture_active": 0, "compat_active": 0}

    # Image-level background selection by D_v-specific distribution similarity
    # (symmetric mode only). When ON, the main loop replaces the uniform
    # `rng.choice(normal_images)` with a top-K sample over the clean pool ranked
    # by histogram-intersection similarity between the SOURCE defect image's
    # background cell histogram (`_dv_bg_hist`) and each clean image's non-void
    # cell histogram (`_cell_hist`). Clean tile lists AND their histograms are
    # cached per path (`normal_cells_cache` / `normal_hist_cache`, both
    # source-independent → computed once); each source's p_dv and its ranking
    # over the pool are cached per source key (`dv_hist_cache` / `dv_scored_cache`
    # / `dv_pool_cache`) so a large pool (e.g. mtd 956) is tiled once and every
    # later ROI/rep sharing a source is a cheap dict lookup. Gate OFF ⇒ the
    # uniform draw and rng stream are byte-identical to legacy.
    image_rank_on = (
        compat_matrix is not None
        and compat_bin_edges is not None
        and compat_mode == "symmetric"
        and compat_threshold > 0.0   # explicit: keep threshold=0 byte-identical even if matrix load decouples from threshold later
        and HAS_CV2
    )
    if image_rank_on:
        _irk_feat, _irk_cell = _load_context_cell_helpers()
        if _irk_feat is None or _irk_cell is None:
            image_rank_on = False  # profiling helpers unavailable → ranking off
        else:
            logger.info("image-level D_v-distribution ranking ON (symmetric): "
                        "top-%d of %d normals by background histogram-intersection, "
                        "cap=%d tiles/img, stride=%d",
                        _NORMAL_TOPK, len(normal_images),
                        _NORMAL_SAMPLE_CAP, _NORMAL_STRIDE)
    normal_cells_cache: Dict[str, List[Tuple[Optional[str], bool]]] = {}
    normal_hist_cache: Dict[str, Dict[str, float]] = {}
    dv_hist_cache: Dict[str, Dict[str, float]] = {}
    dv_scored_cache: Dict[str, List[Tuple[float, str]]] = {}
    dv_pool_cache: Dict[str, List[str]] = {}

    def _normal_cells_for(_p: str) -> List[Tuple[Optional[str], bool]]:
        """Load a normal, grayscale it exactly as the placement gate does (cv2
        RGB2GRAY), and return its cached [(cell, void)] tiles."""
        try:
            _arr = np.asarray(_PILImage.open(_p).convert("RGB"))
        except Exception:
            return []
        if HAS_CV2 and _arr.ndim == 3:
            _g = cv2.cvtColor(_arr, cv2.COLOR_RGB2GRAY)
        else:
            _g = _arr if _arr.ndim == 2 else _arr[..., 0]
        return _normal_tile_cells(
            _g, compat_bin_edges, stride=_NORMAL_STRIDE, cap=_NORMAL_SAMPLE_CAP,
            min_quality=min_bg_quality, blur_threshold=bg_blur_threshold,
        )

    def _normal_hist_for(_p: str) -> Dict[str, float]:
        """Cached non-void cell histogram (p_clean) for a normal path. Reuses the
        `normal_cells_cache` [(cell, void)] tiles → `_cell_hist`."""
        _h = normal_hist_cache.get(_p)
        if _h is None:
            _cells = normal_cells_cache.get(_p)
            if _cells is None:
                _cells = _normal_cells_for(_p)
                normal_cells_cache[_p] = _cells
            _h = _cell_hist(_cells)
            normal_hist_cache[_p] = _h
        return _h

    def _dv_hist_for(_re: Dict[str, Any]) -> Tuple[Optional[str], Dict[str, float]]:
        """Background histogram (p_dv) for a roi_entry's SOURCE defect image,
        cached by source key. Excludes the defect region via the full-frame GT
        mask (defect_mask_path) when present, else the defect_bbox. Returns
        (dv_key, p_dv); p_dv is {} when the source is missing/unreadable or no
        background tile survives."""
        _dpath = _re.get("image_path", "")
        if not _dpath or not Path(_dpath).exists():
            return None, {}
        _mpath = _re.get("defect_mask_path") or ""
        _bbox = _re.get("defect_bbox")
        _has_mask = bool(_mpath) and Path(_mpath).exists()
        # Key: mask-mode is bbox-independent (full-frame mask excludes all
        # defects) → key by mask; bbox-mode keys by the specific bbox.
        if _has_mask:
            _dv_key = f"{_dpath}|m:{_mpath}"
        else:
            _bb = list(_bbox) if isinstance(_bbox, (list, tuple)) else None
            _dv_key = f"{_dpath}|b:{_bb}"
        _p = dv_hist_cache.get(_dv_key)
        if _p is None:
            try:
                _arr = np.asarray(_PILImage.open(_dpath).convert("RGB"))
            except Exception:
                dv_hist_cache[_dv_key] = {}
                return _dv_key, {}
            if HAS_CV2 and _arr.ndim == 3:
                _g = cv2.cvtColor(_arr, cv2.COLOR_RGB2GRAY)
            else:
                _g = _arr if _arr.ndim == 2 else _arr[..., 0]
            _m = None
            if _has_mask:
                try:
                    _m = np.asarray(_PILImage.open(_mpath).convert("L"))
                except Exception:
                    _m = None
            _bb_arg = _bbox if (_m is None and isinstance(_bbox, (list, tuple))
                                and len(_bbox) == 4) else None
            _p = _dv_bg_hist(_g, mask=_m, bbox=_bb_arg, bin_edges=compat_bin_edges,
                             stride=_NORMAL_STRIDE, cap=_NORMAL_SAMPLE_CAP)
            dv_hist_cache[_dv_key] = _p
        return _dv_key, _p

    drive_img_dir = Path(output_dir) / "images"
    drive_mask_dir = Path(output_dir) / "masks"

    # --- Step 3.5 clean-bg assignment (optional, precomputed & data-driven) ---
    # When clean_bg_selection.py has emitted an assignment, background identity is
    # looked up from it (which normal + ranked pool) instead of re-ranking the raw
    # pool at generation time. Deterministic rep→pool indexing consumes ZERO
    # selection rng draws, so AROMA and its random-arm control (each reading their
    # own clean-bg json) share an IDENTICAL downstream placement/feather stream —
    # a clean symmetric control. Missing/invalid → legacy selection (below).
    clean_bg_map: Optional[Dict[int, Dict[str, Any]]] = None
    _cbg_path = (Path(clean_bg_json) if clean_bg_json
                 else Path(roi_dir) / "clean_bg_selected.json")
    _stem2path: Dict[str, str] = {}
    for _p in normal_images:
        _s = Path(_p).stem
        _stem2path.setdefault(_s, _p)
        _stem2path.setdefault(_s.lstrip("_"), _p)  # tolerate good-image "_{stem}" ids

    def _resolve_bg(_id: Any) -> Optional[str]:
        if not _id:
            return None
        return _stem2path.get(str(_id)) or _stem2path.get(str(_id).lstrip("_"))

    if normal_images and _cbg_path.exists():
        try:
            _cbg = load_json(str(_cbg_path))
            clean_bg_map = {int(e["roi_idx"]): e for e in _cbg
                            if e.get("roi_idx") is not None}
            logger.info("clean_bg assignment ON: %s (%d ROIs)",
                        _cbg_path.name, len(clean_bg_map))
        except Exception as _cbg_exc:
            logger.warning("clean_bg json load failed (%s) — legacy selection",
                           _cbg_exc)
            clean_bg_map = None
    # Resolve telemetry (surfaces stale json / id↔path mismatch / leather collision).
    _cbg_used = _cbg_fallback = _cbg_mismatch = 0

    for roi_idx, roi_entry in enumerate(selected):
        for rep_idx in range(n_per_roi):
            fname = f"syn_{roi_idx:05d}_{rep_idx:02d}.jpg"
            mask_fname = f"syn_{roi_idx:05d}_{rep_idx:02d}.png"
            local_out_path = str(img_dir / fname)
            local_mask_path = str(mask_dir / mask_fname)
            # Annotation stores the final Drive path
            final_out_path = str(drive_img_dir / fname) if local_staging else local_out_path
            final_mask_path = str(drive_mask_dir / mask_fname) if local_staging else local_mask_path

            if not normal_images:
                annotations.append({
                    "image_path":    final_out_path,
                    "source_roi":    orig_source_roi[roi_idx],
                    "image_id":      roi_entry.get("image_id", ""),
                    "cluster_id":    roi_entry.get("cluster_id"),
                    # Explicit defect-type string (roi_selection class_key). Lets
                    # multi-mode YOLO labelling resolve the class from a durable
                    # field instead of re-parsing the source_roi path. None for
                    # arms whose roi_entry lacks it (CASDA copy-paste) → downstream
                    # falls back to path parse / severstal cluster_id.
                    "class_key":     roi_entry.get("class_key"),
                    "cell_key":      roi_entry.get("cell_key", ""),
                    "prompt":        roi_entry.get("prompt", ""),
                    "method":        method,
                    "blend_mode":    blend_mode,
                    "roi_score":     roi_entry.get("roi_score", 0.0),
                    "deficit":       roi_entry.get("deficit", 0.0),
                    "mask_path":     None,
                    "bbox":          None,
                    "dry_run":       True,
                })
                n_ok += 1
                continue

            compat_row = (compat_matrix.get(str(roi_entry.get("cluster_id")), {})
                          if compat_matrix is not None else None)
            # Background selection. Symmetric mode: rank the clean pool by
            # histogram-intersection similarity between this ROI's SOURCE
            # background distribution (p_dv) and each clean image's non-void
            # distribution (p_clean), then top-K sample the background; the
            # stage-2 re-pick pool is narrowed to the top-K ranked normals so
            # escapes also stay rank-based. Clean images with no non-void cell
            # (empty p_clean) score -1.0 → dropped by `_rank_normals`, matching
            # the old all-void exclusion. Every other case (defect / legacy /
            # threshold=0 / empty p_dv) keeps the uniform draw — exactly one
            # rng.choice, byte-identical stream. compat_row is still passed to
            # synthesis for the WITHIN-image _positive_place (unchanged).
            stage2_pool = normal_images
            _cbg_entry = clean_bg_map.get(roi_idx) if clean_bg_map else None
            # Staleness guard: a clean_bg_selected.json produced against a DIFFERENT
            # roi_selected.json (regenerated seed/top_k) joins by positional roi_idx
            # only. Reject an entry whose recorded image_id disagrees with this ROI
            # → fall back to legacy for it (and count it) instead of pasting onto a
            # background chosen for a different defect.
            if (_cbg_entry is not None and _cbg_entry.get("image_id")
                    not in (None, "", roi_entry.get("image_id"))):
                _cbg_mismatch += 1
                _cbg_entry = None
            _cbg_pool = ([p for p in
                         (_resolve_bg(i) for i in (_cbg_entry.get("topk_pool") or []))
                         if p] if _cbg_entry else [])
            if clean_bg_map is not None and not _cbg_pool:
                _cbg_fallback += 1
            if _cbg_pool:
                # Precomputed identity → deterministic rep→pool index. Consume ONE
                # placeholder draw so EVERY (roi,rep) spends exactly one selection
                # draw whether it uses the precomputed pool or the legacy fallback —
                # this keeps the downstream placement/feather rng stream aligned even
                # under partial resolution (some ROIs precomputed, some fallback),
                # which the 0-draw variant did not guarantee.
                _ = rng.random()
                normal_path = _cbg_pool[rep_idx % len(_cbg_pool)]
                stage2_pool = _cbg_pool
                _cbg_used += 1
            elif image_rank_on:
                _dv_key, _p_dv = _dv_hist_for(roi_entry)
                if _p_dv:
                    _scored = dv_scored_cache.get(_dv_key)
                    if _scored is None:
                        _scored = []
                        for _npath in normal_images:
                            _p_clean = _normal_hist_for(_npath)
                            _sim = (_hist_intersection(_p_dv, _p_clean)
                                    if _p_clean else -1.0)
                            _scored.append((_sim, _npath))
                        dv_scored_cache[_dv_key] = _scored
                        _ranked = sorted((_t for _t in _scored if _t[0] >= 0.0),
                                         key=lambda _t: _t[0], reverse=True)
                        dv_pool_cache[_dv_key] = [_p for _, _p in _ranked[:_NORMAL_TOPK]]
                    _sel = _rank_normals(_scored, rng, _NORMAL_TOPK)
                    if _sel is not None:
                        normal_path = _sel
                        stage2_pool = dv_pool_cache[_dv_key] or normal_images
                    else:
                        normal_path = rng.choice(normal_images)
                else:
                    normal_path = rng.choice(normal_images)
            else:
                normal_path = rng.choice(normal_images)
            meta = synthesis_fn(
                roi_entry=roi_entry,
                normal_image_path=normal_path,
                output_path=local_out_path,
                blend_mode=blend_mode,
                feather_px=feather_px,
                rng=rng,
                mask_output_path=local_mask_path,
                reject_clean_bg=reject_clean_bg,
                min_bg_quality=min_bg_quality,
                blur_threshold=bg_blur_threshold,
                normal_pool=stage2_pool,
                texture_dist_threshold=texture_dist_threshold,
                compat_row=compat_row,
                bin_edges=compat_bin_edges,
                compat_threshold=compat_threshold,
                # 64px footprint tiling only in symmetric mode; defect mode keeps
                # the legacy crop-size query byte-identical (devnote §finding E).
                compat_tile=(compat_mode == "symmetric"),
            )
            if meta:
                n_ok += 1
                # Pop per-call gate telemetry so it never leaks into
                # annotations.json (cache-hit metas simply don't carry it).
                gs = meta.pop("gate_stats", None)
                if gs and gs.get("active"):
                    gate_agg["active"] += 1
                    gate_agg["repick"] += int(gs.get("repick", 0))
                    gate_agg["fallback"] += int(bool(gs.get("fallback")))
                    gate_agg["texture_active"] += int(bool(gs.get("texture_active")))
                    gate_agg["compat_active"] += int(bool(gs.get("compat_active")))
                # Record the durable Drive mask path only when a mask was saved.
                ann_mask_path = final_mask_path if meta.get("mask_path") else None
                # ControlNet cache hits and texture-gate stage-2 re-picks
                # return the background actually used — prefer it over this
                # loop's rng draw so the annotation never points at a
                # background the composite wasn't made from. copy_paste meta
                # carries normal_image only after a re-pick (else fallback).
                ann_normal = meta.get("normal_image") or normal_path
                annotations.append({
                    "image_path":    final_out_path,
                    "source_roi":    orig_source_roi[roi_idx],
                    "image_id":      roi_entry.get("image_id", ""),
                    "normal_image":  staged_to_drive.get(ann_normal, ann_normal),
                    "cluster_id":    roi_entry.get("cluster_id"),
                    # See dry-run branch: explicit defect-type for path-independent
                    # multi-mode class resolution (None → downstream fallback).
                    "class_key":     roi_entry.get("class_key"),
                    "cell_key":      roi_entry.get("cell_key", ""),
                    "prompt":        roi_entry.get("prompt", ""),
                    # AR-fallback controlnet samples carry method="copy_paste_
                    # arfallback"; every other path leaves method unset → run
                    # param. Distinguishes fallback samples in annotations.
                    "method":        meta.get("method", method),
                    "blend_mode":    blend_mode,
                    "roi_score":     roi_entry.get("roi_score", 0.0),
                    "deficit":       roi_entry.get("deficit", 0.0),
                    "mask_path":     ann_mask_path,
                    "bbox":          meta.get("bbox"),
                    "dry_run":       False,
                })
            else:
                n_skip += 1

    # Push synthesized images to Drive output_dir if staging was used
    if local_staging and staging_dir and img_dir.exists():
        n_copied = _push_outputs(img_dir, drive_img_dir)
        logger.info("Pushed %d images to Drive → %s", n_copied, drive_img_dir)
        if mask_dir.exists():
            n_masks = _push_outputs(mask_dir, drive_mask_dir)
            logger.info("Pushed %d masks to Drive → %s", n_masks, drive_mask_dir)

    drive_out = Path(output_dir)
    drive_out.mkdir(parents=True, exist_ok=True)
    save_json(annotations, str(drive_out / "annotations.json"))
    logger.info("Generated %d images (%d skipped) → %s", n_ok, n_skip, drive_out)
    if clean_bg_map is not None:
        _cbg_total = _cbg_used + _cbg_fallback
        logger.info("clean_bg resolve: used=%d fallback=%d mismatch=%d / %d (roi,rep)",
                    _cbg_used, _cbg_fallback, _cbg_mismatch, _cbg_total)
        if _cbg_total and _cbg_used < 0.9 * _cbg_total:
            logger.warning(
                "clean_bg resolve rate LOW (%.0f%%) — likely a stale "
                "clean_bg_selected.json vs roi_selected.json, an image_id<->path "
                "mismatch, or leather stem-collision. Most ROIs fell back to legacy "
                "selection; the symmetric-control invariant may not hold.",
                100.0 * _cbg_used / _cbg_total,
            )

    result: Dict[str, Any] = {
        "status":       "ok",
        "n_generated":  n_ok,
        "n_skipped":    n_skip,
        "annotations":  annotations,
        "method":       method,
    }
    if method == "controlnet":
        cn_stats = _log_cn_stats()
        if cn_stats is not None:
            result["cn_stats"] = cn_stats
    if texture_dist_threshold > 0.0 or compat_threshold > 0.0:
        act = max(1, gate_agg["active"])
        fb_rate = gate_agg["fallback"] / act
        logger.info(
            "placement-gate stats: active=%d (texture=%d compat=%d) "
            "repick_draws=%d fallback=%d (%.0f%%) [tex_thr=%.3f compat_thr=%.2f]",
            gate_agg["active"], gate_agg["texture_active"],
            gate_agg["compat_active"], gate_agg["repick"], gate_agg["fallback"],
            100 * fb_rate, texture_dist_threshold, compat_threshold,
        )
        if compat_threshold > 0.0 and fb_rate > 0.5:
            logger.warning(
                "compat gate fallback rate %.0f%% > 50%% — placement-aware is "
                "near-no-op on this dataset (most positions rejected → last "
                "candidate pasted). Report this; do NOT claim placement uplift.",
                100 * fb_rate,
            )
        result["gate_stats"] = gate_agg
    return result


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
    p.add_argument("--clean_bg_json", default=None,
                   help="Precomputed clean-bg assignment (Step 3.5 "
                        "clean_bg_selection.py). Default: auto-load "
                        "<roi_dir>/clean_bg_selected.json if present. Pass "
                        "clean_bg_random_arm.json to run the symmetric control arm. "
                        "Missing → legacy generation-time background selection.")
    p.add_argument("--blend_mode",  default="alpha",
                   choices=["alpha", "seamless"],
                   help="Blending mode for copy_paste: 'alpha' (default) or "
                        "'seamless' (Reinhard local-bg transfer + "
                        "cv2.seamlessClone; falls back to alpha if cv2 absent)")
    p.add_argument("--feather_px",  type=int, default=4,
                   help="Edge feather sigma in pixels (default: 4)")
    p.add_argument("--seed",          type=int, default=42,
                   help="Random seed (default: 42)")
    p.add_argument("--local_staging", action="store_true",
                   help="Stage inputs to /content/tmp before synthesis to avoid "
                        "per-image Drive I/O (recommended on Colab)")
    p.add_argument("--reject-clean-bg", dest="reject_clean_bg",
                   action="store_true",
                   help="Reject black/flat (void) backgrounds at generation time "
                        "(normal pool + random-placement fallback). Default OFF.")
    p.add_argument("--min-bg-quality", dest="min_bg_quality",
                   type=float, default=0.7,
                   help="Min background quality 0..1 for the clean-bg gate "
                        "(default 0.7, CASDA value)")
    p.add_argument("--bg-blur-threshold", dest="bg_blur_threshold",
                   type=float, default=100.0,
                   help="Laplacian-variance blur threshold for the clean-bg gate "
                        "(default 100.0, CASDA value)")
    p.add_argument("--texture-dist-threshold", dest="texture_dist_threshold",
                   type=float, default=0.0,
                   help="Pair-level texture gate: max descriptor distance (0..1) "
                        "between the source defect's surroundings and the paste "
                        "patch; resamples position then re-picks the normal on "
                        "mismatch (random-fallback placement only, both "
                        "copy_paste and controlnet). 0 = OFF (default). "
                        "Tuning band ~0.15-0.35.")
    p.add_argument("--compat_threshold", type=float, default=0.0,
                   help="Placement-aware compat gate: accept a paste position "
                        "only when its context cell has compatibility "
                        ">= threshold with the defect cluster "
                        "(compatibility_matrix). Both placement paths; soft "
                        "match (unobserved cell=0.5 neutral). 0 = OFF (default). "
                        "Requires --compat_matrix_json AND --config.")
    p.add_argument("--compat_matrix_json", default=None,
                   help="Path to compatibility_matrix.json (profiling output) "
                        "for --compat_threshold. bin_edges come from --config.")
    p.add_argument("--compat_mode", choices=["symmetric", "defect"],
                   default="defect",
                   help="compat_row source for the placement gate. 'defect' "
                        "(default) = legacy defect-side image-mean matrix "
                        "(byte-identical legacy). 'symmetric' = SGM "
                        "clean-grounded matrix_symmetric (falls back to legacy "
                        "matrix if that key is absent).")
    # --- ControlNet (--method controlnet) ---------------------------------
    p.add_argument("--controlnet_path", default=None,
                   help="Trained ControlNet dir (train_controlnet.py best_model/). "
                        "REQUIRED for --method controlnet")
    p.add_argument("--sd_base", default="runwayml/stable-diffusion-v1-5",
                   help="Stable Diffusion base model id/path "
                        "(default: runwayml/stable-diffusion-v1-5)")
    p.add_argument("--cn_steps", type=int, default=30,
                   help="Diffusion inference steps (default: 30)")
    p.add_argument("--cn_cond_scale", type=float, default=0.7,
                   help="controlnet_conditioning_scale (default: 0.7)")
    p.add_argument("--cn_resolution", type=int, default=512,
                   help="Generation resolution — must match training "
                        "(default: 512)")
    p.add_argument("--cn_no_grayscale", dest="cn_grayscale",
                   action="store_false", default=True,
                   help="Keep generated color. Default forces R==G==B to match "
                        "--force_grayscale_target training (use this flag when "
                        "the model was trained with --no_force_grayscale_target)")
    p.add_argument("--cn_default_bg", default="complex_pattern",
                   help="background_type fallback when the context join misses "
                        "(default: complex_pattern)")
    p.add_argument("--cn_no_cache", dest="cn_cache",
                   action="store_false", default=True,
                   help="Disable the session-resume generation cache "
                        "(sidecar .meta.json next to each composite)")
    p.add_argument("--cn_ar_threshold", type=float, default=2.5,
                   help="Gate ROIs whose bbox aspect ratio max(w,h)/min(w,h) "
                        "exceeds this — elongated crops smear under the "
                        "square squash-unsquash generation geometry "
                        "(default 2.5; 0 disables the gate)")
    p.add_argument("--cn_no_ar_fallback", dest="cn_ar_fallback",
                   action="store_false", default=True,
                   help="On an AR-gated ROI, skip it (pure ControlNet arm) "
                        "instead of the default copy_paste fallback. Default "
                        "falls back so output count + class + bbox stay at full "
                        "parity with the random/aroma-CP arms; the fallback "
                        "samples are tagged method=copy_paste_arfallback")
    p.add_argument("--morphology_csv", default=None,
                   help="morphology_features.csv — metrics join for hint/prompt "
                        "(REQUIRED for --method controlnet)")
    p.add_argument("--context_features", default=None,
                   help="context_features.csv — background_type derivation "
                        "(REQUIRED for --method controlnet)")
    p.add_argument("--config", dest="config_yaml", default=None,
                   help="recommended_config.yaml — context bin_edges "
                        "(REQUIRED for --method controlnet)")
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
        reject_clean_bg=args.reject_clean_bg,
        min_bg_quality=args.min_bg_quality,
        bg_blur_threshold=args.bg_blur_threshold,
        texture_dist_threshold=args.texture_dist_threshold,
        compat_threshold=args.compat_threshold,
        compat_matrix_json=args.compat_matrix_json,
        clean_bg_json=args.clean_bg_json,
        compat_mode=args.compat_mode,
        controlnet_path=args.controlnet_path,
        sd_base=args.sd_base,
        cn_steps=args.cn_steps,
        cn_cond_scale=args.cn_cond_scale,
        cn_resolution=args.cn_resolution,
        cn_grayscale=args.cn_grayscale,
        cn_default_bg=args.cn_default_bg,
        cn_cache=args.cn_cache,
        cn_ar_threshold=args.cn_ar_threshold,
        cn_ar_fallback=args.cn_ar_fallback,
        morphology_csv=args.morphology_csv,
        context_features=args.context_features,
        config_yaml=args.config_yaml,
    )
    if result.get("status") != "ok":
        sys.exit(1)


if __name__ == "__main__":
    main()
