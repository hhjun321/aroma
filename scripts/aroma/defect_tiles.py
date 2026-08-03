#!/usr/bin/env python3
"""
AROMA Step 3.4 — Defect Tile Occupancy (mask → tile map)

Emits, for every defect image, WHICH 64-pixel tiles carry defect pixels (core)
and which pure-background tiles are ADJACENT to them. This is the localized
query that clean_bg_selection.py needs: "the background right next to the
defect", as opposed to the whole-image background it currently uses.

Why a separate step: context_features.csv cannot express it. Profiling emits a
row for every tile whose defect fraction is <= 0.5, so a tile carrying up to
50% defect pixels is indistinguishable from a clean one, and a tile with ZERO
defect pixels is indistinguishable from a tile that was simply far away. Only
the mask separates them. Reading it once here keeps clean_bg_selection.py
offline (CSV + this JSON, never raw pixels).

Definitions (devnote aroma_adjacent_context_bg_selection.md §2-4):
    core          mask_frac(tile) >  0          — touches the defect
    adjacent_rN   mask_frac(tile) == 0 AND within N 8-neighbour dilations of
                  core AND present in context_features.csv
`adjacent` is the query set. It excludes `core` on purpose: a core tile's
context features are computed on a patch containing up to 50% defect pixels
(severstal mean 15.5%), while every clean-pool tile has 0%. Matching a
contaminated query against a clean corpus searches for backgrounds that
resemble defect texture — the opposite of the intent.

Grid policy (F1, devnote §7-1-5): the tile grid is TRUNCATED at W//64 x H//64,
byte-identical to distribution_profiling._context_worker, so tile coordinates
agree with context_features.csv exactly. On datasets whose sides are not
multiples of 64 this discards a right/bottom band (mtd 24.5% of image area,
kolektor 13.2%); mtd has 56/388 images whose defect lies ENTIRELY inside the
discarded band and therefore yields core = {} here. Those images are reported
as `core_empty` and the consumer falls back to the whole-image query. Fixing
the truncation instead (F2) would change compatibility_matrix.json and force a
tau recalibration, which this track deliberately avoids.

Usage (Colab):
    !python $AROMA_SCRIPTS/defect_tiles.py \
        --profiling_dir $AROMA_OUT/profiling/mtd \
        --output_dir    $AROMA_OUT/profiling/mtd

Local verification against the ORIGINAL ground-truth masks (the profiling
defect_masks/ directory only exists on Drive):
    python scripts/aroma/defect_tiles.py \
        --profiling_dir D:/project/aroma_dataset/profiling/profiling/mtd \
        --output_dir    D:/project/aroma_dataset/profiling/profiling/mtd \
        --gt_dir        D:/project/aroma_dataset/mtd/ground_truth

Outputs (written to --output_dir):
    defect_tiles.json         per-image core / adjacent_rN tile keys
    defect_tiles_summary.md   resolution, coverage, per-radius tile statistics
"""
import argparse
import csv
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

try:
    from PIL import Image
except ImportError:  # pragma: no cover - environment dependent
    Image = None  # type: ignore[assignment]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("aroma.defect_tiles")

TILE = 64
DEFECT_TILE_CUT = 0.5   # _context_worker drops a tile when mask.mean() > this


# ---------------------------------------------------------------------------
# CSV side — the authoritative tile grid
# ---------------------------------------------------------------------------

def _read_context_tiles(path: Path) -> Tuple[Dict[str, Set[str]], Dict[str, Tuple[int, int]]]:
    """Read context_features.csv → ({image_id: {patch_xy}}, {image_id: (W, H)}).

    Defect rows only. `patch_xy` is kept as the raw string so membership tests
    downstream need no parsing. Streamed row by row: severstal's file is ~120 MB
    and only three columns are needed.
    """
    tiles: Dict[str, Set[str]] = defaultdict(set)
    dims: Dict[str, Tuple[int, int]] = {}
    with path.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            if row.get("image_type") != "defect":
                continue
            iid = row.get("image_id", "")
            if not iid:
                continue
            pxy = row.get("patch_xy", "")
            if pxy:
                tiles[iid].add(pxy)
            if iid not in dims:
                try:
                    w, h = int(float(row["image_w"])), int(float(row["image_h"]))
                except (KeyError, TypeError, ValueError):
                    continue
                if w > 0 and h > 0:
                    dims[iid] = (w, h)
    return dict(tiles), dims


def _read_mask_paths(path: Path) -> Dict[str, str]:
    """image_id → defect_mask_path from morphology_features.csv."""
    out: Dict[str, str] = {}
    with path.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            iid = row.get("image_id", "")
            if iid:
                out[iid] = str(row.get("defect_mask_path", "") or "")
    return out


# ---------------------------------------------------------------------------
# Mask resolution
# ---------------------------------------------------------------------------

def _gt_candidates(gt_dir: Path, image_id: str) -> List[Path]:
    """Original ground-truth mask for an image_id, CLASS-AWARE.

    Profiling names its own masks `<image_id>.png`, but the source datasets keep
    a class subdirectory and sometimes a `_mask` suffix (mtd, aitex_tiled,
    kolektor). The class prefix must stay in play: every MVTec-Leather class
    directory holds a `000_mask.png`, and Severstal keeps BOTH a flat union mask
    and per-class masks, so a bare stem match silently picks the wrong defect on
    multi-class images. Resolution order — class directory, then flat stem, then
    the profiling `<image_id>` naming, then a unique prefix glob.
    """
    cls, stem = ("", image_id)
    if "_" in image_id:
        cls, stem = image_id.split("_", 1)

    exact: List[Path] = []
    if cls:
        exact += [gt_dir / cls / f"{stem}.png", gt_dir / cls / f"{stem}_mask.png"]
    exact += [gt_dir / f"{stem}.png", gt_dir / f"{stem}_mask.png",
              gt_dir / f"{image_id}.png", gt_dir / f"{image_id}_mask.png"]
    for p in exact:
        if p.exists():
            return [p]

    for base in ([gt_dir / cls] if cls else []) + [gt_dir]:
        if not base.is_dir():
            continue
        for pat in (f"{stem}*.png", f"{image_id}*.png"):
            hits = sorted(base.rglob(pat))
            if len(hits) == 1:
                return hits
            if len(hits) > 1:
                return hits              # caller flags it as ambiguous
    return []


def _resolve_mask(image_id: str, recorded: str, mask_dir: Optional[Path],
                  gt_dir: Optional[Path]) -> Tuple[Optional[Path], str]:
    """Locate one defect mask. Returns (path, how) — `how` feeds the summary."""
    if mask_dir is not None:
        p = mask_dir / f"{image_id}.png"
        if p.exists():
            return p, "mask_dir"
    if recorded:
        p = Path(recorded)
        if p.exists():
            return p, "recorded"
    if gt_dir is not None:
        hits = _gt_candidates(gt_dir, image_id)
        if len(hits) == 1:
            return hits[0], "gt_dir"
        if len(hits) > 1:
            return hits[0], "gt_dir_ambiguous"
    return None, "missing"


def _load_mask(path: Path, wh: Optional[Tuple[int, int]]) -> Optional[np.ndarray]:
    """Binary mask at the profiled image resolution (nearest resize on mismatch)."""
    if Image is None:
        return None
    try:
        img = Image.open(path).convert("L")
    except Exception as exc:            # noqa: BLE001 - unreadable mask is data, not a bug
        logger.warning("[defect_tiles] unreadable mask %s: %s", path, exc)
        return None
    if wh is not None and img.size != wh:
        img = img.resize(wh, Image.NEAREST)
    return np.asarray(img) > 0


# ---------------------------------------------------------------------------
# Tile occupancy
# ---------------------------------------------------------------------------

def _tile_fractions(mask: np.ndarray, gw: int, gh: int) -> Dict[Tuple[int, int], float]:
    """Defect-pixel fraction per tile over the TRUNCATED gw x gh grid (F1)."""
    if gw <= 0 or gh <= 0:
        return {}
    core = mask[:gh * TILE, :gw * TILE]
    # (gh, TILE, gw, TILE) → mean over the two tile axes; one pass, no Python loop.
    return {
        (i, j): float(v)
        for j, row in enumerate(core.reshape(gh, TILE, gw, TILE).mean(axis=(1, 3)))
        for i, v in enumerate(row)
    }


def _dilate(seed: Set[Tuple[int, int]], radius: int) -> Set[Tuple[int, int]]:
    """8-neighbour dilation applied `radius` times (Chebyshev ball)."""
    cur = set(seed)
    for _ in range(max(0, radius)):
        cur = {(i + dx, j + dy) for (i, j) in cur
               for dx in (-1, 0, 1) for dy in (-1, 0, 1)}
    return cur


def _xy(tile: Tuple[int, int]) -> str:
    """Tile index → the `patch_xy` key context_features.csv uses."""
    return f"{tile[0] * TILE}_{tile[1] * TILE}"


def analyze_image(
    mask: np.ndarray,
    have: Set[str],
    wh: Tuple[int, int],
    radii: Sequence[int],
) -> Dict[str, Any]:
    """core / adjacent_rN tile keys for one defect image. Pure — no I/O."""
    w, h = wh
    gw, gh = w // TILE, h // TILE
    frac = _tile_fractions(mask, gw, gh)
    core = {t for t, f in frac.items() if f > 0.0}
    # Tiles profiling would have emitted, derived independently of the CSV so a
    # drift between the two (stale profile vs newer mask) becomes visible.
    emit = {t for t, f in frac.items() if f <= DEFECT_TILE_CUT}

    entry: Dict[str, Any] = {
        "grid": [gw, gh],
        "core": sorted(_xy(t) for t in core),
        "n_csv_tiles": len(have),
        "n_grid_tiles": gw * gh,
        "csv_grid_mismatch": len({_xy(t) for t in emit} ^ have),
    }
    for r in radii:
        adj = {t for t in _dilate(core, r)
               if frac.get(t, 1.0) == 0.0 and _xy(t) in have}
        entry[f"adjacent_r{r}"] = sorted(_xy(t) for t in adj)
    return entry


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run(
    profiling_dir: str,
    output_dir: str,
    mask_dir: Optional[str] = None,
    gt_dir: Optional[str] = None,
    radii: Sequence[int] = (1, 2),
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    pd_dir = Path(profiling_dir)
    ctx_csv = pd_dir / "context_features.csv"
    morph_csv = pd_dir / "morphology_features.csv"
    missing = [str(p) for p in (ctx_csv, morph_csv) if not p.exists()]
    if missing:
        logger.error("[defect_tiles] missing inputs: %s", missing)
        return {"status": "missing_inputs", "missing": missing}
    if Image is None:
        logger.error("[defect_tiles] Pillow is required to read masks")
        return {"status": "no_pillow"}

    radii = sorted({int(r) for r in radii if int(r) >= 1})
    logger.info("[defect_tiles] reading %s", ctx_csv.name)
    have_by_img, dims = _read_context_tiles(ctx_csv)
    recorded = _read_mask_paths(morph_csv)
    ids = sorted(have_by_img)
    if limit:
        ids = ids[:limit]
    logger.info("[defect_tiles] %d defect images, radii=%s", len(ids), radii)

    md = Path(mask_dir) if mask_dir else None
    gd = Path(gt_dir) if gt_dir else None

    tiles: Dict[str, Any] = {}
    how_counts: Dict[str, int] = defaultdict(int)
    core_empty: List[str] = []
    mismatch: List[str] = []
    trunc_loss: List[float] = []
    per_r: Dict[int, List[int]] = {r: [] for r in radii}

    for n, iid in enumerate(ids, 1):
        wh = dims.get(iid)
        path, how = _resolve_mask(iid, recorded.get(iid, ""), md, gd)
        how_counts[how] += 1
        if path is None:
            continue
        mask = _load_mask(path, wh)
        if mask is None:
            how_counts["unreadable"] += 1
            continue
        if wh is None:
            wh = (mask.shape[1], mask.shape[0])
        entry = analyze_image(mask, have_by_img.get(iid, set()), wh, radii)
        gw, gh = entry["grid"]
        if wh[0] * wh[1] > 0:
            trunc_loss.append(1.0 - (gw * TILE * gh * TILE) / (wh[0] * wh[1]))
        if not entry["core"]:
            core_empty.append(iid)
        if entry["csv_grid_mismatch"]:
            mismatch.append(iid)
        for r in radii:
            per_r[r].append(len(entry[f"adjacent_r{r}"]))
        tiles[iid] = entry
        if n % 500 == 0:
            logger.info("[defect_tiles]   %d/%d", n, len(ids))

    stats: Dict[str, Any] = {
        "n_images": len(ids),
        "n_resolved": len(tiles),
        "resolution": dict(how_counts),
        "core_empty": len(core_empty),
        "core_empty_frac": (len(core_empty) / len(tiles)) if tiles else 0.0,
        "csv_grid_mismatch": len(mismatch),
        "grid_truncation_loss_mean": float(np.mean(trunc_loss)) if trunc_loss else 0.0,
        "grid_truncation_loss_max": float(np.max(trunc_loss)) if trunc_loss else 0.0,
        "radii": {},
    }
    for r in radii:
        v = np.asarray(per_r[r], dtype=float) if per_r[r] else np.zeros(0)
        stats["radii"][str(r)] = {
            "tiles_mean": float(v.mean()) if v.size else 0.0,
            "tiles_median": float(np.median(v)) if v.size else 0.0,
            "empty_frac": float((v == 0).mean()) if v.size else 0.0,
            "lt4_frac": float((v < 4).mean()) if v.size else 0.0,
        }

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": {
            "tile": TILE,
            "defect_tile_cut": DEFECT_TILE_CUT,
            "radii": radii,
            "grid_policy": "truncated (W//64 x H//64) — matches _context_worker (F1)",
            "key_format": "patch_xy string (x_y, top-left pixel) — joins context_features.csv",
            "stats": stats,
        },
        "core_empty_ids": core_empty,
        "csv_grid_mismatch_ids": mismatch[:200],
        "tiles": tiles,
    }
    out_json = out_dir / "defect_tiles.json"
    out_json.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    _write_summary(out_dir / "defect_tiles_summary.md", pd_dir.name, stats, radii)
    logger.info("[defect_tiles] wrote %s (%d images)", out_json, len(tiles))
    return {"status": "ok", "output": str(out_json), "stats": stats}


def _write_summary(path: Path, ds: str, stats: Dict[str, Any], radii: Sequence[int]) -> None:
    L = [f"# Defect tile occupancy — {ds}", ""]
    L.append(f"- images: {stats['n_images']}  resolved: {stats['n_resolved']}")
    L.append(f"- mask resolution: {stats['resolution']}")
    L.append(f"- **core empty** (defect outside the truncated grid): "
             f"{stats['core_empty']} ({100 * stats['core_empty_frac']:.1f}%) "
             f"→ consumer falls back to the whole-image query")
    L.append(f"- csv/grid tile-set mismatch: {stats['csv_grid_mismatch']} "
             f"(non-zero ⇒ profile and mask disagree — investigate before use)")
    L.append(f"- grid truncation loss: mean {100 * stats['grid_truncation_loss_mean']:.1f}%  "
             f"max {100 * stats['grid_truncation_loss_max']:.1f}% of image area")
    L += ["", "| radius | adjacent tiles (mean) | median | empty | <4 tiles |",
          "|---|---|---|---|---|"]
    for r in radii:
        s = stats["radii"][str(r)]
        L.append(f"| R{r} | {s['tiles_mean']:.1f} | {s['tiles_median']:.0f} | "
                 f"{100 * s['empty_frac']:.1f}% | {100 * s['lt4_frac']:.1f}% |")
    L += ["", "`adjacent_rN` = defect-pixel-free tiles within N 8-neighbour "
              "dilations of a defect-carrying tile, restricted to tiles present "
              "in context_features.csv. `core` is excluded from the query: its "
              "features are computed on patches holding up to 50% defect pixels."]
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AROMA Step 3.4 — defect tile occupancy")
    p.add_argument("--profiling_dir", required=True,
                   help="Profiling output directory (context_features.csv, morphology_features.csv)")
    p.add_argument("--output_dir", required=True,
                   help="Directory to write defect_tiles.json")
    p.add_argument("--mask_dir", default=None,
                   help="Directory of profiling-emitted masks named <image_id>.png. "
                        "Tried first; default is the path recorded in morphology_features.csv")
    p.add_argument("--gt_dir", default=None,
                   help="Original ground-truth mask root (recursive). Fallback for local "
                        "runs where the profiling defect_masks/ directory is unavailable")
    p.add_argument("--radii", type=int, nargs="+", default=[1, 2],
                   help="Adjacency radii to emit (8-neighbour dilations). Default: 1 2")
    p.add_argument("--limit", type=int, default=None,
                   help="Process only the first N image ids (smoke test)")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = _parse_args(argv)
    result = run(
        profiling_dir=args.profiling_dir,
        output_dir=args.output_dir,
        mask_dir=args.mask_dir,
        gt_dir=args.gt_dir,
        radii=args.radii,
        limit=args.limit,
    )
    if result.get("status") != "ok":
        sys.exit(1)


if __name__ == "__main__":
    main()
