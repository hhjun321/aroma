#!/usr/bin/env python3
"""
prepare_aitex.py — AITEX Fabric Image Database (Kaggle) → AROMA (MVTec-style) layout.

AITEX (Kaggle nexuswho/aitex-fabric-image-database) ships THREE top-level
folders, all 4096×256 PNG:

    Defect_images/    nnnn_ddd_ff.png        (defect image; ddd=defect code, ff=fabric)
    Mask_images/      nnnn_ddd_ff_mask.png   (binary mask, white=defect area)
    NODefect_images/  nnnn_000_ff.png        (defect-free image; defect code = 000)

Two AITEX quirks this script handles explicitly:
  - NOT every defect image has a mask. Mask_images/ has FEWER files than
    Defect_images/ — join by basename and SKIP+count defect images with no
    matching *_mask.png (masks are mandatory for AROMA morphology features).
  - masks live in a SEPARATE folder keyed by the '_mask' suffix.

Tiling (default ON, --tile 256 --stride 128):
    The raw 4096×256 aspect (16:1) collapses to 256×16 content under exp4v2's
    imgsz=256 letterbox, destroying every defect bbox (measured: baseline mAP50
    0.066). This script therefore slices each image into tile×tile windows with
    50% overlap BEFORE materializing the MVTec-style layout, so every downstream
    consumer (distribution_profiling / roi_selection / generate_* / exp4v2)
    reads ordinary square images with no code change:

      - defect image tile with >=1 mask bbox of area >= --min_tile_area
          → test/defect/{origstem}__tile_r{r}_c{c}.png
            + ground_truth/defect/{origstem}__tile_r{r}_c{c}_mask.png
      - defect image tile whose mask window is ALL-ZERO
          → train/good/  (background pool from defect-bearing fabrics)
      - defect image tile with only sub-min_area mask fragments
          → DISCARDED (neither labelable defect nor clean background)
      - NODefect image tile → train/good/

    The `__tile_` stem delimiter is the contract exp4v2's group-aware
    _split_defects uses to keep all tiles of one source image in the same
    train/val split (50% overlap would otherwise leak across the split).
    Tile files are real crops (written, not symlinked); masks are re-encoded
    per tile and bboxes re-derived from the mask window.

Single-class (default ON): all defect tiles are merged into test/defect/ +
ground_truth/defect/ — 71 train images / 11 codes gave ~6.5 img/class, which is
unlearnable; exp4v2 then degenerates to single-class detection naturally.
Pass --multi_class to keep per-code test/{ddd}/ folders instead.
Pass --tile 0 to restore the legacy non-tiled link/copy behavior.

Layout (tiled single-class default):

    Aroma/aitex_tiled/
      train/good/                              (normal tiles)
      test/defect/{stem}__tile_r0_c{c}.png     (defect tiles)
      ground_truth/defect/{...}_mask.png       (per-tile binary masks)
      aitex_manifest.json

Re-running: tile crops OVERWRITE existing files (deterministic re-generation).
Do NOT reuse an output_dir that holds the legacy non-tiled layout — stale
4096×256 files would coexist with tiles. Use a fresh dir (e.g. aitex_tiled).

Usage (Colab):
    !python $AROMA_SCRIPTS/prepare_aitex.py \\
        --defect_images   $AITEX_RAW/Defect_images \\
        --mask_images     $AITEX_RAW/Mask_images \\
        --nodefect_images $AITEX_RAW/NODefect_images \\
        --output_dir      /content/drive/MyDrive/data/Aroma/aitex_tiled \\
        --tile 256 --stride 128 --min_tile_area 50
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_IMG_EXTS = {".png", ".PNG"}
_MASK_SUFFIX = "_mask"

# Guarded image libs (cv2 preferred, PIL+numpy fallback) — used ONLY to detect
# all-zero / unreadable masks so they are skipped, not silently passed to the
# profiling Otsu fallback (which would invent a fake defect region). When no lib
# is available we cannot inspect pixels → treat as "keep" (do not over-skip).
try:
    import numpy as _np  # type: ignore[import]
    _HAS_NUMPY = True
except Exception:
    _HAS_NUMPY = False
try:
    import cv2 as _cv2  # type: ignore[import]
    _HAS_CV2 = True
except Exception:
    _HAS_CV2 = False
try:
    from PIL import Image as _PILImage  # type: ignore[import]
    _HAS_PIL = True
except Exception:
    _HAS_PIL = False


try:
    from scipy import ndimage as _ndimage  # type: ignore[import]
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


def _mask_nonzero(mask_path: str) -> Optional[bool]:
    """True if mask readable AND has >=1 nonzero pixel; False if unreadable or
    all-zero; None if no image lib available (caller then keeps the mask)."""
    if _HAS_CV2:
        arr = _cv2.imread(mask_path, _cv2.IMREAD_GRAYSCALE)
        if arr is None:
            return False
        return bool(arr.max() > 0)
    if _HAS_PIL and _HAS_NUMPY:
        try:
            arr = _np.asarray(_PILImage.open(mask_path).convert("L"))
        except Exception:  # noqa: BLE001
            return False
        return bool(arr.max() > 0)
    return None


# ---------------------------------------------------------------------------
# Tiling helpers (require numpy + (cv2 or PIL) — hard error if absent, unlike
# the mask-inspection fallback above: tiles are CROPS, we must read pixels)
# ---------------------------------------------------------------------------

_TILE_DELIM = "__tile_"


def _require_imaging() -> None:
    if not _HAS_NUMPY or not (_HAS_CV2 or _HAS_PIL):
        raise RuntimeError(
            "Tiling requires numpy + (cv2 or PIL). Install opencv-python or "
            "Pillow, or pass --tile 0 for the legacy non-tiled layout."
        )


def _read_image(path: str) -> "Optional[_np.ndarray]":
    """Read an image as HxWx3 BGR/RGB uint8 (channel order irrelevant — tiles
    are written back with the same library that read them)."""
    if _HAS_CV2:
        arr = _cv2.imread(path, _cv2.IMREAD_COLOR)
        return arr
    try:
        return _np.asarray(_PILImage.open(path).convert("RGB"))
    except Exception:  # noqa: BLE001
        return None


def _read_mask_arr(path: str) -> "Optional[_np.ndarray]":
    if _HAS_CV2:
        return _cv2.imread(path, _cv2.IMREAD_GRAYSCALE)
    try:
        return _np.asarray(_PILImage.open(path).convert("L"))
    except Exception:  # noqa: BLE001
        return None


def _write_image(path: Path, arr: "_np.ndarray") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if _HAS_CV2:
        ok = _cv2.imwrite(str(path), arr)
        if not ok:
            raise IOError(f"cv2.imwrite failed: {path}")
    else:
        _PILImage.fromarray(arr).save(str(path))


def _tile_coords(length: int, tile: int, stride: int) -> List[int]:
    """Top-left offsets covering [0, length) with tile-sized windows.

    Regular stride grid, plus a final offset clamped to `length - tile` so the
    right/bottom remainder is always covered (duplicate-free)."""
    if length <= tile:
        return [0]
    xs = list(range(0, length - tile + 1, stride))
    if xs[-1] != length - tile:
        xs.append(length - tile)
    return xs


def _label_components(binary: "_np.ndarray") -> "Optional[Tuple[_np.ndarray, int]]":
    """8-connected component labeling → (labels HxW int array, n_components).

    cv2 and scipy MUST agree (8-connectivity) or the tile set would depend on
    which library is installed. Returns None when neither is available."""
    if _HAS_CV2:
        n, labels = _cv2.connectedComponents(binary.astype("uint8"), connectivity=8)
        return labels, n - 1  # cv2 counts background as component 0
    if _HAS_SCIPY:
        labels, n = _ndimage.label(binary, structure=_np.ones((3, 3), dtype=int))
        return labels, n
    return None


def _mask_tile_bboxes(
    mask_tile: "_np.ndarray",
    min_area: int,
    open_borders: Tuple[bool, bool, bool, bool] = (False, False, False, False),
) -> Tuple[List[Tuple[int, int, int, int]], "Optional[_np.ndarray]", bool]:
    """Classify a tile's mask window → (bboxes, cleaned_mask, used_union).

    Component rules (mirrors exp4v2 _mask_to_bboxes semantics, tile-adapted):
      - component with bbox area >= min_area           → kept, own bbox
      - SMALL component touching an OPEN border        → border fragment of a
        defect that continues into the neighboring tile (50% overlap guarantees
        the neighbor holds it better) → ERASED from the cleaned mask
      - SMALL interior component (point/thread defect) → pixels kept; when no
        big component exists, their union bbox is returned (exp4v2's
        union-bbox fallback would label them anyway — dropping these tiles
        lost 35/102 AITEX sources and extinguished code 025 in measurement)

    open_borders = (left, top, right, bottom): True where the tile edge is
    INTERIOR to the source image (content continues beyond it). Tile edges that
    coincide with the image border are closed — a small defect there is a real
    point defect, not a fragment.

    Returns:
      bboxes       — [] means "no labelable defect": caller discards the tile
      cleaned_mask — 0/255 uint8 with border fragments erased (None iff bboxes
                     empty); this is what must be WRITTEN so exp4v2's own
                     union fallback cannot span an erased sliver
      used_union   — True when bboxes is the small-interior union fallback

    No cv2/scipy: cannot separate components — whole nonzero window is kept as
    one union bbox (permissive, matches exp4v2's lib-poor behavior)."""
    binary = (mask_tile > 0)
    if not bool(binary.any()):
        return [], None, False
    th, tw = binary.shape[:2]
    left_open, top_open, right_open, bottom_open = open_borders

    lab = _label_components(binary)
    if lab is None:
        ys, xs = _np.nonzero(binary)
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        return ([(x0, y0, x1 - x0, y1 - y0)],
                binary.astype("uint8") * 255, True)

    labels, n = lab
    boxes: List[Tuple[int, int, int, int]] = []
    cleaned = binary.copy()
    has_small_interior = False
    for i in range(1, n + 1):
        comp = labels == i
        ys, xs = _np.nonzero(comp)
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        w, h = x1 - x0, y1 - y0
        if w * h >= min_area:
            boxes.append((x0, y0, w, h))
            continue
        touches_open = ((x0 == 0 and left_open) or (y0 == 0 and top_open)
                        or (x1 == tw and right_open) or (y1 == th and bottom_open))
        if touches_open:
            cleaned[comp] = False   # border fragment — neighbor tile owns it
        else:
            has_small_interior = True
    used_union = False
    if not boxes:
        if not has_small_interior or not bool(cleaned.any()):
            return [], None, False
        ys, xs = _np.nonzero(cleaned)
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        boxes = [(x0, y0, x1 - x0, y1 - y0)]
        used_union = True
    return boxes, cleaned.astype("uint8") * 255, used_union


# ---------------------------------------------------------------------------
# Filename parsing
# ---------------------------------------------------------------------------

def _parse_defect_code(stem: str) -> Optional[str]:
    """Parse the defect code (ddd) from an AITEX filename stem 'nnnn_ddd_ff'.

    Returns the literal 2nd underscore-token (e.g. '002'), or None if the stem
    does not have at least 3 underscore-separated tokens.
    """
    parts = stem.split("_")
    if len(parts) < 3:
        return None
    return parts[1]


# ---------------------------------------------------------------------------
# File listing / staging
# ---------------------------------------------------------------------------

def _list_pngs(directory: str, recursive: bool = False) -> List[Path]:
    """Sorted .png files under `directory` (empty if missing).

    recursive=False: files directly under `directory` (Defect_images/Mask_images
    are flat). recursive=True: also descends into subfolders — AITEX
    NODefect_images groups defect-free images in per-fabric SUBFOLDERS, so a
    non-recursive scan there returns nothing and leaves train/good empty.
    """
    d = Path(directory)
    if not d.exists():
        return []
    it = d.rglob("*") if recursive else d.iterdir()
    return sorted(p for p in it if p.is_file() and p.suffix in _IMG_EXTS)


def _link_or_copy(src: str, dst: str) -> None:
    """Symlink src→dst; fall back to copy. Skip if dst exists (idempotent)."""
    dst_p = Path(dst)
    if dst_p.exists() or dst_p.is_symlink():
        return
    dst_p.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.symlink(os.path.abspath(src), dst)
    except (OSError, NotImplementedError):
        shutil.copy2(src, dst)


# ---------------------------------------------------------------------------
# Main prepare
# ---------------------------------------------------------------------------

def prepare(
    defect_images: str,
    mask_images: str,
    nodefect_images: str,
    output_dir: str,
    tile: int = 256,
    stride: int = 128,
    min_tile_area: int = 50,
    single_class: bool = True,
) -> Dict[str, object]:
    defect_files = _list_pngs(defect_images)
    nodefect_files = _list_pngs(nodefect_images, recursive=True)  # NODefect_images는 fabric별 서브폴더 중첩
    mask_files = _list_pngs(mask_images)
    print(f"[prepare_aitex] defect_images={len(defect_files)}  "
          f"nodefect_images={len(nodefect_files)}  mask_images={len(mask_files)}")

    out = Path(output_dir)
    good_dir = out / "train" / "good"
    good_dir.mkdir(parents=True, exist_ok=True)

    # Mixed-layout guard: a legacy (non-tiled) run leaves per-code test/{ddd}
    # folders and 4096-wide files; re-running tiled on the same dir would mix
    # generations. Warn loudly (the manifest records which mode wrote last).
    # NOTE/WARNING strings below are ASCII-only on purpose: on cp949 consoles
    # (Korean Windows) a U+2014/U+2192 in print() raises UnicodeEncodeError,
    # and an enclosing try would swallow the warning entirely (measured). The
    # try covers ONLY the json parse; prints happen outside it.
    prev_manifest = out / "aitex_manifest.json"
    prev: Optional[Dict[str, object]] = None
    if prev_manifest.exists():
        try:
            prev = json.loads(prev_manifest.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            prev = None
    if isinstance(prev, dict):
        prev_tiled = bool(prev.get("tiled", False))
        if prev_tiled != (tile > 0):
            print(f"[prepare_aitex] WARNING: output_dir already holds a "
                  f"{'tiled' if prev_tiled else 'NON-tiled'} layout but this run is "
                  f"{'tiled' if tile > 0 else 'non-tiled'} -- stale files will coexist. "
                  f"Use a fresh output_dir.")
        prev_tile = prev.get("tile") or {}
        if prev_tiled and tile > 0 and prev_tile != {
            "size": tile, "stride": stride, "min_tile_area": min_tile_area,
        }:
            print(f"[prepare_aitex] NOTE: tile params changed "
                  f"(prev={prev_tile} -> now size={tile} stride={stride} "
                  f"min_tile_area={min_tile_area}) -- stale tiles will be purged.")

    tiled = tile > 0
    if tiled:
        _require_imaging()
        if stride <= 0 or stride > tile:
            raise ValueError(f"--stride must be in (0, tile]: got stride={stride} tile={tile}")

    # Purge previously-generated tiles before a tiled run. Different params
    # (stride/min_tile_area) reclassify tiles between test/defect, train/good
    # and discarded — without a purge, a stale copy of the SAME pixels can sit
    # in train/good while the new copy lands in test/defect (result
    # contamination: exp4v2 enumerates both dirs straight from disk). The
    # `__tile_` delimiter is owned by this script, so deleting only those
    # files never touches legacy links or foreign data. Runs AFTER the
    # dependency/param validation above so a doomed run fails before deleting.
    if tiled:
        n_purged = 0
        for sub in ("train", "test", "ground_truth"):
            base = out / sub
            if not base.exists():
                continue
            for f in base.rglob(f"*{_TILE_DELIM}*"):
                if f.is_file():
                    f.unlink()
                    n_purged += 1
        if n_purged:
            print(f"[prepare_aitex] purged {n_purged} stale tile files from previous run")

    # --- Mask index: {defect_image_stem: mask_path} keyed by stem sans '_mask' ---
    #   mask file nnnn_ddd_ff_mask.png → key nnnn_ddd_ff
    mask_index: Dict[str, Path] = {}
    for m in mask_files:
        mstem = m.stem
        if mstem.endswith(_MASK_SUFFIX):
            key = mstem[: -len(_MASK_SUFFIX)]
        else:
            # Unexpected: mask without the '_mask' suffix — index by stem as-is.
            key = mstem
        mask_index[key] = m

    # --- Normal (defect-free) images → train/good/ ---
    n_normal_tiles = 0
    for src in nodefect_files:
        if not tiled:
            _link_or_copy(str(src), str(good_dir / src.name))
            continue
        img = _read_image(str(src))
        if img is None:
            print(f"[prepare_aitex] WARNING: unreadable normal image skipped: {src}")
            continue
        h, w = img.shape[:2]
        for r, y0 in enumerate(_tile_coords(h, tile, stride)):
            for c, x0 in enumerate(_tile_coords(w, tile, stride)):
                dst = good_dir / f"{src.stem}{_TILE_DELIM}r{r}_c{c}.png"
                _write_image(dst, img[y0:y0 + tile, x0:x0 + tile])
                n_normal_tiles += 1

    # --- Defect images; masks matched by stem ---
    per_code_counts: Dict[str, int] = {}
    n_skipped_no_mask = 0
    n_skipped_empty_mask = 0
    n_unreadable = 0
    n_unparsed_code = 0
    n_defect_tiles = 0
    n_bg_tiles_from_defect = 0
    n_discarded_fragment_tiles = 0
    n_union_fallback_tiles = 0
    sources_no_defect_tiles: Dict[str, int] = {}   # defect_code -> count
    sources_no_defect_examples: List[str] = []
    skipped_examples: List[str] = []
    manifest_defects: List[Dict[str, object]] = []

    for src in defect_files:
        stem = src.stem
        ddd = _parse_defect_code(stem)
        if ddd is None:
            n_unparsed_code += 1
            if len(skipped_examples) < 10:
                skipped_examples.append(f"unparsed_code:{src.name}")
            continue

        mask_src = mask_index.get(stem)
        if mask_src is None:
            # AITEX known quirk: some defect images have no mask. Skip (masks
            # are mandatory for AROMA morphology) and count for honest reporting.
            n_skipped_no_mask += 1
            if len(skipped_examples) < 10:
                skipped_examples.append(f"no_mask:{src.name}")
            continue

        # AITEX known quirk: some masks exist but are ALL-ZERO (no annotated
        # defect region) or unreadable. Staging them lets profiling's Otsu
        # fallback invent a fake defect (mask_source=fallback_otsu). Skip them
        # like no-mask so only genuine ground-truth defects seed the pipeline.
        if _mask_nonzero(str(mask_src)) is False:
            n_skipped_empty_mask += 1
            if len(skipped_examples) < 10:
                skipped_examples.append(f"empty_mask:{src.name}")
            continue

        # Class folder: single-class merges every code into 'defect'.
        cls = "defect" if single_class else ddd

        if not tiled:
            dst_img = out / "test" / cls / src.name
            _link_or_copy(str(src), str(dst_img))
            dst_mask = out / "ground_truth" / cls / f"{stem}{_MASK_SUFFIX}.png"
            _link_or_copy(str(mask_src), str(dst_mask))
            per_code_counts[ddd] = per_code_counts.get(ddd, 0) + 1
            manifest_defects.append({
                "image_id": src.name,
                "defect_code": ddd,
                "image": f"test/{cls}/{src.name}",
                "mask": f"ground_truth/{cls}/{stem}{_MASK_SUFFIX}.png",
            })
            continue

        img = _read_image(str(src))
        mask = _read_mask_arr(str(mask_src))
        if img is None or mask is None:
            n_unreadable += 1
            if len(skipped_examples) < 10:
                skipped_examples.append(f"unreadable:{src.name}")
            continue
        if img.shape[:2] != mask.shape[:2]:
            print(f"[prepare_aitex] WARNING: image/mask shape mismatch skipped: "
                  f"{src.name} img={img.shape[:2]} mask={mask.shape[:2]}")
            n_unreadable += 1
            continue

        h, w = mask.shape[:2]
        made_defect_tile = False
        for r, y0 in enumerate(_tile_coords(h, tile, stride)):
            for c, x0 in enumerate(_tile_coords(w, tile, stride)):
                mt = mask[y0:y0 + tile, x0:x0 + tile]
                tname = f"{stem}{_TILE_DELIM}r{r}_c{c}"
                if not bool((mt > 0).any()):
                    # Clean window from a defect-bearing fabric → background
                    # pool (adds fabric diversity matching the defect domain —
                    # fabric 08 appears ONLY in defect images).
                    _write_image(good_dir / f"{tname}.png",
                                 img[y0:y0 + tile, x0:x0 + tile])
                    n_bg_tiles_from_defect += 1
                    continue
                boxes, cleaned_mask, used_union = _mask_tile_bboxes(
                    mt, min_tile_area,
                    open_borders=(x0 > 0, y0 > 0, x0 + tile < w, y0 + tile < h),
                )
                if not boxes:
                    # Only border fragments of defects that continue into a
                    # neighboring tile: not labelable here, not clean — discard
                    # so neither split sees a mislabeled window.
                    n_discarded_fragment_tiles += 1
                    continue
                if used_union:
                    n_union_fallback_tiles += 1
                _write_image(out / "test" / cls / f"{tname}.png",
                             img[y0:y0 + tile, x0:x0 + tile])
                # cleaned_mask is 0/255 with border fragments erased, so
                # downstream threshold consumers (and exp4v2's own union
                # fallback) see exactly the labeled defect pixels.
                _write_image(out / "ground_truth" / cls / f"{tname}{_MASK_SUFFIX}.png",
                             cleaned_mask)
                n_defect_tiles += 1
                made_defect_tile = True
                manifest_defects.append({
                    "image_id": f"{tname}.png",
                    "origin": stem,
                    "defect_code": ddd,
                    "x0": int(x0), "y0": int(y0),
                    "n_bbox": len(boxes),
                    "union_fallback": used_union,
                    "image": f"test/{cls}/{tname}.png",
                    "mask": f"ground_truth/{cls}/{tname}{_MASK_SUFFIX}.png",
                })
        if made_defect_tile:
            per_code_counts[ddd] = per_code_counts.get(ddd, 0) + 1
        else:
            # Mask-bearing source that produced ZERO defect tiles — without
            # this counter the manifest arithmetic cannot explain the loss
            # (pre-union-fallback measurement: 35/102 sources vanished here).
            sources_no_defect_tiles[ddd] = sources_no_defect_tiles.get(ddd, 0) + 1
            if len(sources_no_defect_examples) < 10:
                sources_no_defect_examples.append(src.name)

    defect_codes = sorted(per_code_counts.keys())
    class_folders = ["defect"] if single_class else defect_codes
    manifest = {
        "dataset": "aitex",
        "tiled": tiled,
        "tile": ({"size": tile, "stride": stride, "min_tile_area": min_tile_area}
                 if tiled else None),
        "single_class": single_class,
        "classes": class_folders,
        "image_shape_original": {"height": 256, "width": 4096},
        "image_shape": ({"height": tile, "width": tile} if tiled
                        else {"height": 256, "width": 4096}),
        "source_layout": {
            "defect_images": "Defect_images/nnnn_ddd_ff.png",
            "mask_images": "Mask_images/nnnn_ddd_ff_mask.png",
            "nodefect_images": "NODefect_images/nnnn_000_ff.png",
        },
        "counts": {
            "normal_source_images": len(nodefect_files),
            "normal_tiles": n_normal_tiles,
            "bg_tiles_from_defect_images": n_bg_tiles_from_defect,
            "defect_tiles": n_defect_tiles,
            "union_fallback_tiles": n_union_fallback_tiles,
            "discarded_border_fragment_tiles": n_discarded_fragment_tiles,
            "defect_source_images_matched": sum(per_code_counts.values()),
            "defect_sources_zero_defect_tiles": sum(sources_no_defect_tiles.values()),
            "defect_sources_zero_defect_tiles_per_code": sources_no_defect_tiles,
            "defect_images_total": len(defect_files),
            "mask_files_total": len(mask_files),
            "skipped_no_mask": n_skipped_no_mask,
            "skipped_empty_mask": n_skipped_empty_mask,
            "unreadable": n_unreadable,
            "unparsed_code": n_unparsed_code,
            "per_defect_code_source_images": per_code_counts,
        },
        "defect_codes": defect_codes,
        "layout": {
            "train_good": "train/good",
            "test_defect_codes": [f"test/{c}" for c in class_folders],
            "ground_truth": "ground_truth/{cls}/{stem}_mask.png",
        },
        "defects": manifest_defects,
    }
    (out / "aitex_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    if tiled:
        print(f"[prepare_aitex] tiled: tile={tile} stride={stride} min_tile_area={min_tile_area}")
        print(f"[prepare_aitex] normal_tiles={n_normal_tiles} (+{n_bg_tiles_from_defect} bg tiles "
              f"from defect images)  defect_tiles={n_defect_tiles} "
              f"(union_fallback={n_union_fallback_tiles})  "
              f"discarded_border_fragments={n_discarded_fragment_tiles}")
        if sources_no_defect_tiles:
            print(f"[prepare_aitex] WARNING: {sum(sources_no_defect_tiles.values())} mask-bearing "
                  f"sources produced ZERO defect tiles (per code: {sources_no_defect_tiles}; "
                  f"examples: {sources_no_defect_examples})")
    print(f"[prepare_aitex] defect_source_images_matched={sum(per_code_counts.values())}  "
          f"per_defect_code={per_code_counts}")
    print(f"[prepare_aitex] skipped_no_mask={n_skipped_no_mask}  "
          f"skipped_empty_mask={n_skipped_empty_mask}  "
          f"unreadable={n_unreadable}  unparsed_code={n_unparsed_code}")
    if skipped_examples:
        print(f"[prepare_aitex] skipped examples: {skipped_examples}")
    print(f"[prepare_aitex] class folders(defect_type)={class_folders}")
    print(f"[prepare_aitex] wrote layout + masks + manifest → {out}")
    return manifest


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="AITEX Fabric Image Database → AROMA (MVTec-style) layout"
    )
    p.add_argument("--defect_images", required=True,
                   help="AITEX Defect_images/ directory (nnnn_ddd_ff.png)")
    p.add_argument("--mask_images", required=True,
                   help="AITEX Mask_images/ directory (nnnn_ddd_ff_mask.png)")
    p.add_argument("--nodefect_images", required=True,
                   help="AITEX NODefect_images/ directory (nnnn_000_ff.png)")
    p.add_argument("--output_dir", required=True,
                   help="Output root, e.g. /content/drive/MyDrive/data/Aroma/aitex_tiled")
    p.add_argument("--tile", type=int, default=256,
                   help="Tile size (default 256). 0 = legacy non-tiled layout.")
    p.add_argument("--stride", type=int, default=128,
                   help="Tile stride (default 128 = 50%% overlap).")
    p.add_argument("--min_tile_area", type=int, default=50,
                   help="Min bbox area (w*h px) for a tile to count as a defect "
                        "tile. Keep equal to exp4v2 _mask_to_bboxes min_area (50).")
    p.add_argument("--multi_class", action="store_true",
                   help="Keep per-code test/{ddd}/ folders instead of the "
                        "single merged test/defect/ (default: single-class).")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = _parse_args(argv)
    prepare(
        args.defect_images, args.mask_images, args.nodefect_images, args.output_dir,
        tile=args.tile, stride=args.stride, min_tile_area=args.min_tile_area,
        single_class=not args.multi_class,
    )
    print("Done.")


if __name__ == "__main__":
    main()
