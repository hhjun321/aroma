#!/usr/bin/env python3
"""
AROMA Step 3.5 — Clean-Background Selection

Mirrors roi_selection.py: turns PROFILING-DERIVED files (never raw good pixels)
into a per-ROI clean-background assignment, so generate_defects can look up
"which normal + which ranked pool" instead of re-scanning raw good images and
re-computing histograms at generation time.

Objective (Phase 1 — extract-first): for each selected ROI, rank clean (good)
background images by histogram-intersection between the source defect's
class-conditioned background cell distribution and each good image's non-void
cell distribution — the SAME similarity generate_defects computes at runtime
(_dv_bg_hist / _cell_hist / _hist_intersection), sourced offline from
context_features.csv. A void/quality prefilter (§1-0) and a bbox size-fit hard
gate (§1-a.2) determine the candidate pool first.

HONESTY (committed E1/E2/V2): the histogram matching is DOMAIN-CONDITIONAL
(aitex lift +0.78; severstal/mtd ~0 = indistinguishable from random) and
placement is geometry-blind (E2); context signal is weak overall (V2). The
CERTAIN value of this module is reproducibility + a clean symmetric control
(assign the SAME backgrounds to the random arm) + removing per-seed placement
variance. It does NOT by itself claim a general mAP improvement.

DATA-DRIVEN (no hardcoding): void/quality cut and the per-ROI pool size are
DERIVED from observed distributions when not passed on the CLI; the derived
values are recorded in the output meta / summary for auditability.

Usage (Colab):
    !python $AROMA_SCRIPTS/clean_bg_selection.py \
        --profiling_dir  $AROMA_OUT/profiling/mtd \
        --roi_dir        $AROMA_OUT/roi/mtd \
        --output_dir     $AROMA_OUT/roi/mtd \
        --emit_random_arm

Outputs (written to --output_dir):
    clean_bg_candidates.json   all (roi x good) scored candidates
    clean_bg_selected.json     per-ROI best + ranked top-K pool
    clean_bg_random_arm.json   (only with --emit_random_arm) symmetric control
    clean_bg_summary.md        human-readable table + derived thresholds
"""
import argparse
import bisect
import csv
import hashlib
import json
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("aroma.clean_bg")


# ---------------------------------------------------------------------------
# I/O bootstrap (verbatim from roi_selection.py — same scripts/aroma/ depth)
# ---------------------------------------------------------------------------

def _bootstrap_aroma_ref() -> str:
    try:
        repo_root = Path(__file__).resolve().parents[2]
        if repo_root.is_dir() and str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
    except (IndexError, OSError):
        pass
    ref = os.environ.get("AROMA_REF") or r"D:\project\aroma"
    ref_path = Path(ref)
    if ref_path.is_dir():
        if str(ref_path) not in sys.path:
            sys.path.insert(0, str(ref_path))
        return "aroma_ref"
    return "inline"


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
# Constants (structure only — NO tunable magic numbers; thresholds are derived)
# ---------------------------------------------------------------------------

# Fallback context-feature order + bin count, used ONLY if the compatibility
# matrix does not carry its own context_features list. Mirrors
# distribution_profiling.CONTEXT_FEATURES / N_CONTEXT_BINS so the cell keys are
# byte-identical to profiling's discretization.
_CONTEXT_FEATURES = [
    "local_variance", "edge_density", "texture_entropy",
    "frequency_energy", "orientation_consistency",
]
_N_CONTEXT_BINS = 3


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _parse_bbox(bbox: Any) -> Optional[List[int]]:
    """Accept a 'x,y,w,h' string OR a [x,y,w,h] list → [int,int,int,int] | None."""
    if bbox is None or bbox == "":
        return None
    if isinstance(bbox, (list, tuple)):
        parts = list(bbox)
    else:
        parts = str(bbox).split(",")
    if len(parts) != 4:
        return None
    try:
        return [int(round(float(p))) for p in parts]
    except (ValueError, TypeError):
        return None


def _load_discretizer(compat: Dict[str, Any]) -> Tuple[List[str], Dict[str, List[float]]]:
    """Return (feature_names, bin_edges) from the compatibility matrix so cell
    keys match profiling EXACTLY. bin_edges are re-used, never re-derived."""
    names = compat.get("context_features") or _CONTEXT_FEATURES
    bin_edges = compat.get("bin_edges") or {}
    return list(names), bin_edges


def _cell_key(feats: Dict[str, Any], names: List[str], bin_edges: Dict[str, List[float]]) -> str:
    """Discretize the 5 context features → 'b_b_b_b_b' — mirrors
    distribution_profiling._context_cell_key (bisect_right, clamp to N-1)."""
    bins = []
    for feat in names:
        try:
            val = float(feats.get(feat, 0.0) or 0.0)
        except (TypeError, ValueError):
            val = 0.0
        edges = bin_edges.get(feat, [0.0, 1.0])
        if len(edges) < 2 or edges[1] <= edges[0]:
            b = 0
        else:
            b = min(bisect.bisect_right(edges, val), _N_CONTEXT_BINS - 1)
        bins.append(str(b))
    return "_".join(bins)


def _hist_intersection(p: Dict[str, float], q: Dict[str, float]) -> float:
    """sum_cell min(p[cell], q[cell]) over shared cells, [0,1]. Extracted from
    generate_defects._hist_intersection (verbatim semantics)."""
    if not p or not q:
        return 0.0
    if len(p) > len(q):
        p, q = q, p
    return sum(min(v, q[c]) for c, v in p.items() if c in q)


# ---------------------------------------------------------------------------
# Input loading (status dict, never raises — mirrors roi_selection.load_inputs)
# ---------------------------------------------------------------------------

def load_inputs(profiling_dir: str, roi_dir: str) -> Dict[str, Any]:
    pd = Path(profiling_dir)
    rd = Path(roi_dir)
    required = [
        pd / "context_features.csv",
        pd / "compatibility_matrix.json",
        rd / "roi_selected.json",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        return {"status": "missing_inputs", "missing": missing}

    ctx_rows = _read_csv_rows(pd / "context_features.csv")
    compat = load_json(str(pd / "compatibility_matrix.json"))
    roi = load_json(str(rd / "roi_selected.json"))
    names, bin_edges = _load_discretizer(compat)

    # image_id → defect_type for the CLASS axis (Phase 2 class-conditioned dv).
    # morphology_features covers ALL defect images (not just the selected subset),
    # so the per-class background aggregate is complete. Optional: absent → class
    # signal simply unavailable (w_class derives to 0).
    iid_to_class: Dict[str, str] = {}
    iid_to_bbox: Dict[str, Optional[List[int]]] = {}
    for m in _read_csv_rows(pd / "morphology_features.csv"):
        _iid = m.get("image_id", "")
        iid_to_class[_iid] = m.get("defect_type", "")
        iid_to_bbox[_iid] = _parse_bbox(m.get("defect_bbox"))

    good_by_img: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    defect_rows: List[Dict[str, str]] = []
    for r in ctx_rows:
        it = r.get("image_type", "good")
        if it == "good":
            good_by_img[r.get("image_id", "")].append(r)
        elif it == "defect":
            defect_rows.append(r)

    return {
        "status": "ok",
        "compat": compat,
        "names": names,
        "bin_edges": bin_edges,
        "roi": roi,
        "good_by_img": dict(good_by_img),
        "defect_rows": defect_rows,
        "iid_to_class": iid_to_class,
        "iid_to_bbox": iid_to_bbox,
    }


# ---------------------------------------------------------------------------
# §1-0 — void / quality prefilter (offline, from context_features features)
# ---------------------------------------------------------------------------

def _patch_void(row: Dict[str, str], var_floor: float, edge_floor: float) -> bool:
    """A patch is void/flat when its texture energy is at/below the data-derived
    floor (near-zero local_variance AND edge_density) — the offline analogue of
    _is_clean_background's void detection (no pixels)."""
    try:
        lv = float(row.get("local_variance", 0.0) or 0.0)
        ed = float(row.get("edge_density", 0.0) or 0.0)
    except (TypeError, ValueError):
        return True
    return lv <= var_floor and ed <= edge_floor


def _derive_void_floors(good_by_img: Dict[str, List[Dict[str, str]]],
                        floor_pct: float = 15.0) -> Tuple[float, float]:
    """Data-driven void floors: a RAISED low-percentile (default p15) of the
    observed local_variance / edge_density distributions (NOT a hardcoded
    constant — the per-dataset percentile auto-adapts).

    Why p15 and not p1: dark border/void tiles cluster just ABOVE zero, not at
    it. On severstal the dark-void cluster sits at local_variance ~0.2 /
    edge_density ~1.0 (measured: the 88%-black normal _9edf820d2 has black-patch
    median local_variance=0.21, edge_density=0.98). The old p1 floor
    (var=0.10 / edge=0.65) lands BELOW that cluster and catches only ~1% of the
    black patches, so _patch_void reports a partial plate as ~0% void and it
    survives the gate. The pool's own p10 (var=0.48 / edge=3.21) already sits
    ABOVE the dark cluster yet far BELOW the valid full-plate cluster (var median
    61.7); p15 covers the full dark-border spread (var up to ~0.25) with margin.
    local_variance is the binding variable (black edge_density 0.98 is below even
    pool p5=1.27). Because it is a per-dataset percentile it does not over-flag
    textured pools (leather/mtd) and is not a severstal-tuned magic number; a
    dataset whose percentile still undershoots can be pinned via the absolute
    --var_floor / --edge_floor safety valve in run()."""
    lv, ed = [], []
    for rows in good_by_img.values():
        for r in rows:
            try:
                lv.append(float(r.get("local_variance", 0.0) or 0.0))
                ed.append(float(r.get("edge_density", 0.0) or 0.0))
            except (TypeError, ValueError):
                pass
    if not lv:
        return 0.0, 0.0
    var_floor = float(np.percentile(lv, floor_pct))
    edge_floor = float(np.percentile(ed, floor_pct))
    return var_floor, edge_floor


def valid_bg_pool(
    good_by_img: Dict[str, List[Dict[str, str]]],
    reject_clean_bg: bool,
    void_frac_max: Optional[float],
    var_floor: float,
    edge_floor: float,
    floor_pct: float = 15.0,
    floor_source: str = "percentile",
) -> Tuple[List[str], Dict[str, str], Dict[str, float]]:
    """Keep good images whose void_frac is at/below a data-derived cut. ALL-reject
    → fall back to the full pool (never a silent 0-output)."""
    void_frac: Dict[str, float] = {}
    for iid, rows in good_by_img.items():
        n = len(rows) or 1
        v = sum(1 for r in rows if _patch_void(r, var_floor, edge_floor))
        void_frac[iid] = v / n

    if void_frac_max is None:
        # Absolute majority-void cut: an image is a "partial plate" only when
        # MORE THAN HALF its patches are void. 0.5 = the majority (over-half)
        # boundary — a semantic threshold, NOT a dataset-tuned constant, and NOT
        # the old relative p90 that structurally kept ~90% of the pool no matter
        # how void-heavy the tail was (so it could never fully drop a partial
        # plate). severstal ref (with floors fixed): most images ~0% void, cut@0.5
        # drops only ~6.5% of the pool = just the worst partial plates.
        void_frac_max = 0.5

    reasons: Dict[str, str] = {}
    kept: List[str] = []
    if not reject_clean_bg:
        kept = list(good_by_img.keys())
        for iid in kept:
            reasons[iid] = "kept|gate_off"
    else:
        for iid, vf in void_frac.items():
            if vf <= void_frac_max:
                kept.append(iid)
                reasons[iid] = "kept|void_frac=%.4f" % vf
            else:
                reasons[iid] = "reject|void_frac=%.4f" % vf
        if not kept:
            logger.warning(
                "void/quality gate rejected ALL %d good images "
                "(void_frac_max=%.4f) — falling back to full pool.",
                len(good_by_img), void_frac_max,
            )
            kept = list(good_by_img.keys())
            for iid in kept:
                reasons[iid] = "fallback_all_reject"

    derived = {
        "var_floor": var_floor,
        "edge_floor": edge_floor,
        "void_floor_pct": float(floor_pct),
        "floor_source": floor_source,
        "void_frac_max": float(void_frac_max),
        "n_good": float(len(good_by_img)),
        "n_valid": float(len(kept)),
    }
    return kept, reasons, derived


# ---------------------------------------------------------------------------
# Cell histograms (offline, from context_features — no pixels)
# ---------------------------------------------------------------------------

def _image_hist(rows: List[Dict[str, str]], names, bin_edges,
                var_floor: float, edge_floor: float) -> Dict[str, float]:
    """Normalized NON-VOID cell histogram of one image's patches. Offline
    analogue of generate_defects._cell_hist (void tiles skipped)."""
    counts: Dict[str, int] = {}
    total = 0
    for r in rows:
        if _patch_void(r, var_floor, edge_floor):
            continue
        ck = _cell_key(r, names, bin_edges)
        counts[ck] = counts.get(ck, 0) + 1
        total += 1
    if total == 0:
        return {}
    return {c: n / total for c, n in counts.items()}


def _class_bg_hist(defect_rows: List[Dict[str, str]], names, bin_edges,
                   var_floor: float, edge_floor: float,
                   iid_to_class: Dict[str, str]) -> Dict[str, Dict[str, float]]:
    """Phase 2 (§1-a.1) — class-conditioned source background histograms:
    aggregate ALL defect-image patches BY class → {class: {cell: frac}}. Offline
    analogue of _dv_bg_hist generalized to the class axis. Profiling already
    excludes defect tiles (_context_worker); void tiles skipped (data floors)."""
    counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    total: Dict[str, int] = defaultdict(int)
    for r in defect_rows:
        cls = iid_to_class.get(r.get("image_id", ""))
        if not cls or _patch_void(r, var_floor, edge_floor):
            continue
        counts[cls][_cell_key(r, names, bin_edges)] += 1
        total[cls] += 1
    return {cls: {c: n / (total[cls] or 1) for c, n in cc.items()}
            for cls, cc in counts.items()}


# ---------------------------------------------------------------------------
# §1-a.2 — bbox size-fit (hard gate)
# ---------------------------------------------------------------------------

def _image_dim(rows: List[Dict[str, str]], tile: int = 64) -> Tuple[int, int]:
    """(W, H) of a good image. Prefer the exact pixel size emitted by profiling
    (image_w/image_h columns, same value on every patch row); fall back to the
    patch-grid estimate (max patch_xy + tile) for older CSVs without those
    columns. The grid estimate underestimates by up to one tile (truncated edge
    patches), so exact columns give clamp-free placement/scale."""
    for r in rows:                       # exact size — first valid row wins
        try:
            w = int(float(r.get("image_w", "")))
            h = int(float(r.get("image_h", "")))
        except (ValueError, TypeError):
            continue
        if w > 0 and h > 0:
            return w, h
    max_x = max_y = 0                     # fallback: patch-grid estimate
    for r in rows:
        pxy = str(r.get("patch_xy", ""))
        if "_" not in pxy:
            continue
        try:
            x, y = (int(v) for v in pxy.split("_", 1))
        except (ValueError, TypeError):
            continue
        max_x = max(max_x, x)
        max_y = max(max_y, y)
    return max_x + tile, max_y + tile


def _size_ok(defect_wh: Tuple[int, int], bg_dim: Tuple[int, int]) -> bool:
    (cw, ch), (bw, bh) = defect_wh, bg_dim
    return cw > 0 and ch > 0 and cw <= bw and ch <= bh


# Fit-rescale margin — MUST match generate_defects.copy_paste_synthesis (the 0.95
# factor at the `crop_w > bw_norm or crop_h > bh_norm` branch). This is a plumbing
# constant kept in lockstep with generation, NOT a tuned threshold (the size gate
# weight itself is data-derived). If generation's factor changes, change it here.
_FIT_MARGIN = 0.95


def _scale_to_fit(defect_wh: Tuple[int, int], bg_dim: Tuple[int, int]) -> float:
    """Rescale factor generation will apply so the crop fits the background:
    1.0 when it already fits (no distortion), <1 when it must shrink. Mirrors
    generate_defects exactly so precomputed positions stay valid post-rescale.
    Doubles as the Option-1 size-fit signal (higher = less distortion)."""
    cw, ch = defect_wh
    bw, bh = bg_dim
    if cw <= 0 or ch <= 0 or bw <= 0 or bh <= 0:
        return 1.0
    if cw <= bw and ch <= bh:
        return 1.0
    return min(bw / float(cw), bh / float(ch)) * _FIT_MARGIN


def _effective_wh(defect_wh: Tuple[int, int], bg_dim: Tuple[int, int]) -> Tuple[int, int]:
    """Crop (w,h) AFTER generation's fit-rescale — the size the paste position must
    be computed against so it isn't clamped at generation time (Option 2)."""
    s = _scale_to_fit(defect_wh, bg_dim)
    if s >= 1.0:
        return defect_wh
    cw, ch = defect_wh
    return (max(1, int(cw * s)), max(1, int(ch * s)))


# ---------------------------------------------------------------------------
# Phase 3 (§E2) — class-conditioned geometry prior for the paste POSITION
# ---------------------------------------------------------------------------
# E2 showed placement is geometry-blind (mtd break real edge 100% -> placed
# 46.5%; leather 0% -> 41.7%). Here we derive each class's real edge/surface
# tendency from morphology defect_bbox (pixel-free: source dim from the source
# image's context patch_xy grid) and precompute a paste position on the assigned
# background that RESPECTS that tendency. Opt-in (--geometry_prior); mAP effect
# is GPU-TBD (E2 caveat) so it defaults OFF.

_EDGE_MARGIN_FRAC = 0.08   # geometric border band (fraction of min side); E2 convention
_SPAN_FRAC = 0.80          # bbox covering >=80% of a side is a full-frame span


def _edge_surface(bbox_wh_xy: Tuple[int, int, int, int], dim: Tuple[int, int],
                  margin_frac: float = _EDGE_MARGIN_FRAC) -> str:
    """Classify a bbox placement as 'span' | 'edge' | 'surface' (mirrors E2)."""
    x, y, w, h = bbox_wh_xy
    W, H = dim
    if W <= 0 or H <= 0:
        return "surface"
    if w >= _SPAN_FRAC * W or h >= _SPAN_FRAC * H:
        return "span"
    m = margin_frac * min(W, H)
    if x <= m or y <= m or (W - (x + w)) <= m or (H - (y + h)) <= m:
        return "edge"
    return "surface"


def _class_edge_prior(iid_to_class, iid_to_bbox, src_dim_by_img,
                      margin_frac: float = _EDGE_MARGIN_FRAC
                      ) -> Tuple[Dict[str, float], float]:
    """Per-class 'edge+span' fraction of REAL defects + the global fraction —
    the data-driven target the placement should match (E2)."""
    by_class = defaultdict(lambda: [0, 0])  # class -> [edge_or_span, total]
    g_es = g_tot = 0
    for iid, bbox in iid_to_bbox.items():
        cls = iid_to_class.get(iid)
        dim = src_dim_by_img.get(iid)
        if not cls or not bbox or not dim:
            continue
        cat = _edge_surface((bbox[0], bbox[1], bbox[2], bbox[3]), dim, margin_frac)
        es = 1 if cat in ("edge", "span") else 0
        by_class[cls][0] += es
        by_class[cls][1] += 1
        g_es += es
        g_tot += 1
    prior = {c: (v[0] / v[1]) for c, v in by_class.items() if v[1]}
    global_es = (g_es / g_tot) if g_tot else 0.5
    return prior, global_es


def _place_position(wh: Tuple[int, int], dim: Tuple[int, int],
                    want_edge: bool, jitter01: float) -> Optional[List[int]]:
    """Deterministic paste top-left (x,y) that lands the crop at an EDGE (flush
    to one of the 4 borders, rotated by jitter for diversity) or on the SURFACE
    (interior, small jitter offset). None if the crop does not fit."""
    cw, ch = wh
    W, H = dim
    if cw <= 0 or ch <= 0 or cw > W or ch > H:
        return None
    xmax, ymax = W - cw, H - ch
    if want_edge:
        side = int(jitter01 * 4) % 4  # 0=left 1=top 2=right 3=bottom, rotated
        if side == 0:
            return [0, min(ymax, int(jitter01 * ymax))]
        if side == 1:
            return [min(xmax, int(jitter01 * xmax)), 0]
        if side == 2:
            return [xmax, min(ymax, int(jitter01 * ymax))]
        return [min(xmax, int(jitter01 * xmax)), ymax]
    # surface: interior, centred with a small deterministic offset for diversity
    cx, cy = xmax // 2, ymax // 2
    ox = int((jitter01 - 0.5) * 0.4 * xmax)
    oy = int((jitter01 - 0.5) * 0.4 * ymax)
    return [max(0, min(xmax, cx + ox)), max(0, min(ymax, cy + oy))]


# ---------------------------------------------------------------------------
# Candidate build + ranking
# ---------------------------------------------------------------------------

def _roi_class_axis(roi_entry: Dict[str, Any], multi_class: bool) -> Tuple[str, str]:
    """Return (class_axis_name, class_value). On single-class datasets (aitex,
    class_key all 'defect') fall back to defect_subtype/morph_label so the
    class-conditioned histogram still discriminates."""
    ck = str(roi_entry.get("class_key") or "_")
    if multi_class and ck not in ("", "_", "defect"):
        return "class_key", ck
    sub = str(roi_entry.get("defect_subtype") or roi_entry.get("morph_label") or "general")
    return "defect_subtype", sub


def _bg_jitter(source_key: str, normal_id: str) -> float:
    """Deterministic sub-score tie-break (blake2b, no salted hash) so equal
    intersections rotate the assigned image instead of always CSV-order."""
    h = hashlib.blake2b(("%s\x1f%s" % (source_key, normal_id)).encode("utf-8"), digest_size=8)
    return int.from_bytes(h.digest(), "big") / float(1 << 64) * 1e-7


def build_and_rank(
    data: Dict[str, Any],
    valid_ids: List[str],
    var_floor: float,
    edge_floor: float,
    pool_k: Optional[int],
    geometry_prior: bool = False,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, float]]:
    """Two signals per (ROI x valid good) candidate:
      src_fit   = hist∩(good, the ROI's SOURCE-image background)  [Phase 1, E1-faithful]
      class_fit = hist∩(good, the ROI class's aggregate background) [Phase 2, §1-a.1]
    Combine by DATA-DERIVED weights (no hardcoding): each signal is weighted by
    its measured discriminative lift (best − median within a ROI, averaged over
    ROIs) → a signal that is ~flat across backgrounds (weak, e.g. severstal/mtd
    context) gets ~0 weight; a discriminative one (aitex) dominates. Rank by the
    combined score; assign a data-cut top pool. Deterministic (no rng)."""
    names, bin_edges = data["names"], data["bin_edges"]
    roi = data["roi"]
    good_by_img = data["good_by_img"]
    defect_rows = data["defect_rows"]
    iid_to_class = data.get("iid_to_class", {})

    class_keys = {str(r.get("class_key") or "_") for r in roi}
    multi_class = len(class_keys - {"", "_", "defect"}) > 1

    # per-source dv (Phase 1) + class-conditioned dv (Phase 2)
    defect_by_img: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for r in defect_rows:
        defect_by_img[r.get("image_id", "")].append(r)
    src_hist_by_img = {iid: _image_hist(rows, names, bin_edges, var_floor, edge_floor)
                       for iid, rows in defect_by_img.items()}
    class_hist = _class_bg_hist(defect_rows, names, bin_edges, var_floor, edge_floor,
                                iid_to_class)

    # Guard: the per-source (E1) signal needs each ROI's image_id to resolve to a
    # defect row in context_features.csv. If most ROIs don't resolve, src_fit is
    # silently 0 for all of them (E1 gate reads 0, background diversity collapses)
    # — the classic stale-roi / new-profiling image_id mismatch. Make it LOUD.
    n_src_missing = sum(1 for r in roi
                        if str(r.get("image_id", "")) not in src_hist_by_img)
    src_match_frac = 1.0 - (n_src_missing / len(roi)) if roi else 0.0
    if roi and src_match_frac < 0.5:
        logger.warning(
            "image_id MISMATCH: %d/%d ROIs (%.0f%%) have no matching defect row in "
            "context_features.csv -> per-source (E1) signal is ~0 and background "
            "diversity will collapse. Likely a STALE roi_selected.json vs a freshly "
            "reprofiled context_features.csv (e.g. phase0 image_id unique-key rerun). "
            "FIX: re-run step3 (roi_selection) on the SAME profiling.",
            n_src_missing, len(roi), 100.0 * (1.0 - src_match_frac),
        )

    # Phase 3 (§E2) — class edge/surface prior from REAL defect geometry.
    class_edge = {}
    global_edge = 0.5
    if geometry_prior:
        src_dim_by_img = {iid: _image_dim(rows) for iid, rows in defect_by_img.items()}
        class_edge, global_edge = _class_edge_prior(
            iid_to_class, data.get("iid_to_bbox", {}), src_dim_by_img)

    good_hist: Dict[str, Dict[str, float]] = {}
    good_dim: Dict[str, Tuple[int, int]] = {}
    for iid in valid_ids:
        rows = good_by_img.get(iid, [])
        good_hist[iid] = _image_hist(rows, names, bin_edges, var_floor, edge_floor)
        good_dim[iid] = _image_dim(rows)

    # ---- Pass 1: score both signals per (ROI, good); collect per-ROI lift ----
    per_roi: List[Dict[str, Any]] = []
    lifts_src: List[float] = []
    lifts_cls: List[float] = []
    lifts_size: List[float] = []
    src_ceilings: List[float] = []
    for roi_idx, r in enumerate(roi):
        axis, cls_val = _roi_class_axis(r, multi_class)
        src_dv = src_hist_by_img.get(str(r.get("image_id", "")), {})
        cls_dv = class_hist.get(str(r.get("class_key") or ""), {})
        bbox = _parse_bbox(r.get("defect_bbox"))
        wh = (bbox[2], bbox[3]) if bbox else (0, 0)
        cand = []  # (normal_id, src_fit, class_fit, size_ok, size_fit)
        s_scores, c_scores, z_scores = [], [], []
        for iid in valid_ids:
            sf = _hist_intersection(good_hist[iid], src_dv)
            cf = _hist_intersection(good_hist[iid], cls_dv)
            so = _size_ok(wh, good_dim[iid]) if wh != (0, 0) else True
            # Option 1: size-fit signal = fit-rescale factor (1.0=no distortion,
            # <1=shrink). Constant across same-size backgrounds → 0 lift → 0 weight
            # (auto-downweighted); varies only when backgrounds differ in size.
            zf = _scale_to_fit(wh, good_dim[iid]) if wh != (0, 0) else 1.0
            cand.append((iid, sf, cf, so, zf))
            s_scores.append(sf); c_scores.append(cf); z_scores.append(zf)
        if s_scores:
            lifts_src.append(max(s_scores) - float(np.median(s_scores)))
            lifts_cls.append(max(c_scores) - float(np.median(c_scores)))
            lifts_size.append(max(z_scores) - float(np.median(z_scores)))
            src_ceilings.append(max(s_scores))
        per_roi.append({"roi_idx": roi_idx, "axis": axis, "cls_val": cls_val,
                        "cluster_id": r.get("cluster_id"), "cell_key": r.get("cell_key", ""),
                        "image_id": str(r.get("image_id", "")), "bbox": bbox,
                        "src_key": "%s\x1f%s" % (str(r.get("image_path", "")),
                                                 str(r.get("defect_bbox", ""))),
                        "cand": cand})

    # ---- Data-derived weights from mean lift (normalized) ----
    w_src = float(np.mean(lifts_src)) if lifts_src else 0.0
    w_cls = float(np.mean(lifts_cls)) if lifts_cls else 0.0
    w_size = float(np.mean(lifts_size)) if lifts_size else 0.0
    lift_src_m, lift_cls_m, lift_size_m = w_src, w_cls, w_size
    tot = w_src + w_cls + w_size
    if tot <= 0:                       # all signals flat → fall back to per-source
        w_src, w_cls, w_size = 1.0, 0.0, 0.0
    else:
        w_src, w_cls, w_size = w_src / tot, w_cls / tot, w_size / tot

    # ---- Pass 2: combine, rank, assign ----
    selected: List[Dict[str, Any]] = []
    pool_sizes: List[int] = []
    for pr in per_roi:
        src_key = pr["src_key"]
        scored = []  # (combined_jittered, normal_id, src_fit, class_fit, size_ok, size_fit)
        for iid, sf, cf, so, zf in pr["cand"]:
            comb = w_src * sf + w_cls * cf + w_size * zf
            scored.append((comb + _bg_jitter(src_key, iid), iid, sf, cf, so, zf))
        scored.sort(key=lambda t: t[0], reverse=True)
        if scored:
            if pool_k:
                top = scored[:max(1, pool_k)]
            else:
                cut = float(np.percentile([t[0] for t in scored], 95.0))
                top = [t for t in scored if t[0] >= cut] or scored[:1]
        else:
            top = []
        pool_sizes.append(len(top))
        base = {"roi_idx": pr["roi_idx"], "image_id": pr["image_id"],
                "class_axis": pr["axis"], "class_value": pr["cls_val"],
                "cluster_id": pr["cluster_id"], "cell_key": pr["cell_key"],
                "defect_bbox": pr["bbox"]}

        # Phase 3 — precompute a paste position per pool bg matching this class's
        # real edge/surface tendency (want_edge = class more edge-bound than the
        # global rate). Position depends on (bg dim, crop wh); deterministic
        # jitter for diversity. None (position at generation) when geometry off.
        # Option 2: compute against the EFFECTIVE (post fit-rescale) crop size so
        # the position stays valid — not clamped — when generation shrinks an
        # oversized crop to fit the background.
        def _pos_for(nid):
            if not geometry_prior or not pr["bbox"]:
                return None
            dim = good_dim.get(nid, (0, 0))
            wh = _effective_wh((pr["bbox"][2], pr["bbox"][3]), dim)
            want_edge = class_edge.get(str(pr["cls_val"]), global_edge) > global_edge
            j = (_bg_jitter(src_key, nid) / 1e-7)  # blake2b frac in [0,1)
            return _place_position(wh, dim, want_edge, j)

        if top:
            comb_j, best_id, sf, cf, so, zf = top[0]
            selected.append(dict(base,
                                 assigned_normal_id=best_id,
                                 topk_pool=[t[1] for t in top],
                                 topk_positions=[_pos_for(t[1]) for t in top],
                                 position=_pos_for(best_id),
                                 score=round(comb_j - _bg_jitter(src_key, best_id), 6),
                                 hist_intersection=round(sf, 6),   # per-source (E1-comparable)
                                 class_fit=round(cf, 6),           # class-conditioned (Phase 2)
                                 size_ok=bool(so),
                                 size_fit=round(zf, 4),            # Option 1 signal (1=no distortion)
                                 scale_factor=round(zf, 4),        # Option 2: generation's fit-rescale
                                 n_valid_bg=len(pr["cand"])))
        else:
            selected.append(dict(base, assigned_normal_id=None, topk_pool=[],
                                 topk_positions=[], position=None,
                                 score=0.0, hist_intersection=0.0, class_fit=0.0,
                                 size_ok=False, size_fit=0.0, scale_factor=0.0,
                                 n_valid_bg=0))

    derived = {
        "pool_cut": "p95" if not pool_k else ("k=%d" % pool_k),
        "mean_pool_size": float(np.mean(pool_sizes)) if pool_sizes else 0.0,
        "multi_class": float(multi_class),
        "w_src": round(w_src, 4), "w_class": round(w_cls, 4), "w_size": round(w_size, 4),
        "lift_src": round(lift_src_m, 4),
        "lift_class": round(lift_cls_m, 4),
        "lift_size": round(lift_size_m, 4),
        # per-source ceiling (E1 reproduction gate — independent of Phase-2 weights)
        "src_fit_ceiling_mean": round(float(np.mean(src_ceilings)), 4) if src_ceilings else 0.0,
        # fraction of ROIs whose image_id resolves to a defect row (E1 signal health);
        # <1.0 → stale roi / profiling image_id mismatch (see warning above)
        "src_match_frac": round(src_match_frac, 4),
        "geometry_prior": float(geometry_prior),
        "class_edge_prior": {c: round(v, 3) for c, v in class_edge.items()},
        "global_edge": round(global_edge, 3),
    }
    return selected, derived


def random_arm(selected: List[Dict[str, Any]], valid_ids: List[str], seed: int) -> List[Dict[str, Any]]:
    """Symmetric control: assign a uniformly-random valid background to the SAME
    ROI set, so a downstream random arm differs from AROMA ONLY in bg identity."""
    rng = np.random.default_rng(seed)
    out = []
    pool = list(valid_ids)
    for s in selected:
        if pool:
            idx = int(rng.integers(0, len(pool)))
            aid = pool[idx]
        else:
            aid = None
        entry = {k: s[k] for k in ("roi_idx", "image_id", "class_axis", "class_value",
                                   "cluster_id", "cell_key", "defect_bbox")}
        entry.update({"assigned_normal_id": aid, "topk_pool": [aid] if aid else [],
                      "arm": "random"})
        out.append(entry)
    return out


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def _build_summary(selected, derived_pool, derived_void, strategy) -> str:
    assigned = [s for s in selected if s.get("assigned_normal_id")]
    distinct_bg = len({s["assigned_normal_id"] for s in assigned})
    hi = [s["hist_intersection"] for s in assigned] or [0.0]
    lines = [
        "# AROMA Step 3.5 — Clean-Background Selection Summary",
        "",
        "> HONESTY: histogram matching is domain-conditional (aitex strong; "
        "severstal/mtd ~random). Value = reproducibility + symmetric control, "
        "NOT a general mAP win.",
        "",
        f"**Strategy**: `{strategy}`",
        f"**ROIs**: {len(selected)}  |  assigned: {len(assigned)}  |  "
        f"distinct backgrounds: {distinct_bg}",
        f"**hist_intersection**: mean {float(np.mean(hi)):.4f}  "
        f"median {float(np.median(hi)):.4f}  max {float(np.max(hi)):.4f}",
        "",
        "## Derived thresholds (data-driven, no hardcoding)",
        "",
        f"- void: var_floor={derived_void['var_floor']:.6g}  "
        f"edge_floor={derived_void['edge_floor']:.6g}  "
        f"(floor_pct={derived_void.get('void_floor_pct', 15.0):.1f}, "
        f"source={derived_void.get('floor_source', 'percentile')})  "
        f"void_frac_max={derived_void['void_frac_max']:.4f} (majority)  "
        f"→ valid {int(derived_void['n_valid'])}/{int(derived_void['n_good'])} good images",
        f"- pool_cut={derived_pool['pool_cut']}  "
        f"mean_pool_size={derived_pool['mean_pool_size']:.2f}  "
        f"multi_class={bool(derived_pool['multi_class'])}",
        f"- signal weights (data-derived from lift): "
        f"w_src={derived_pool.get('w_src')}  w_class={derived_pool.get('w_class')}  "
        f"w_size={derived_pool.get('w_size')}  "
        f"(lift_src={derived_pool.get('lift_src')}, lift_class={derived_pool.get('lift_class')}, "
        f"lift_size={derived_pool.get('lift_size')})",
        f"- src_fit_ceiling_mean={derived_pool.get('src_fit_ceiling_mean')}  "
        f"(E1 reproduction gate — compare to E1 sim_best)",
        f"- src_match_frac={derived_pool.get('src_match_frac')}  "
        f"(ROI image_id ↔ context defect 매칭율; <1.0이면 stale roi/profiling 불일치)",
        "",
        "## Sample assignments (top 30)",
        "",
        "| roi_idx | class | assigned_bg | hist∩ | pool | size_ok | scale |",
        "|---------|-------|-------------|-------|------|---------|-------|",
    ]
    for s in selected[:30]:
        lines.append(
            f"| {s['roi_idx']} | {s['class_value']} | {s['assigned_normal_id']} "
            f"| {s['hist_intersection']:.4f} | {len(s['topk_pool'])} | {s['size_ok']} "
            f"| {s.get('scale_factor', 1.0)} |"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run(
    profiling_dir: str,
    roi_dir: str,
    output_dir: str,
    strategy: str = "histogram",
    seed: int = 42,
    emit_random_arm: bool = False,
    reject_clean_bg: bool = True,
    void_frac_max: Optional[float] = None,
    void_floor_pct: float = 15.0,
    var_floor: Optional[float] = None,
    edge_floor: Optional[float] = None,
    pool_k: Optional[int] = None,
    geometry_prior: bool = False,
) -> Dict[str, Any]:
    logger.info("Loading inputs: profiling_dir=%s roi_dir=%s", profiling_dir, roi_dir)
    data = load_inputs(profiling_dir, roi_dir)
    if data["status"] != "ok":
        logger.error("Input loading failed: %s", data)
        return data

    # Data-driven percentile floors (default p15 — see _derive_void_floors), then
    # an absolute per-dataset safety valve: if --var_floor / --edge_floor are
    # passed they REPLACE the derived value (prescan-pinned override, no code
    # change). floor_source records which path decided each floor for audit.
    d_var, d_edge = _derive_void_floors(data["good_by_img"], void_floor_pct)
    var_override, edge_override = var_floor, edge_floor
    var_floor = d_var if var_override is None else float(var_override)
    edge_floor = d_edge if edge_override is None else float(edge_override)
    if var_override is None and edge_override is None:
        floor_source = "percentile"
    elif var_override is not None and edge_override is not None:
        floor_source = "override"
    else:
        floor_source = "override_partial"
    logger.info("Void floors: var_floor=%.6g edge_floor=%.6g (pct=%.1f, source=%s)",
                var_floor, edge_floor, void_floor_pct, floor_source)
    valid_ids, reasons, derived_void = valid_bg_pool(
        data["good_by_img"], reject_clean_bg, void_frac_max, var_floor, edge_floor,
        floor_pct=void_floor_pct, floor_source=floor_source,
    )
    logger.info("Valid clean-bg pool: %d / %d good images (void_frac_max=%.4f)",
                len(valid_ids), len(data["good_by_img"]), derived_void["void_frac_max"])

    selected, derived_pool = build_and_rank(
        data, valid_ids, var_floor, edge_floor, pool_k, geometry_prior=geometry_prior
    )
    # attach void provenance to each assignment
    for s in selected:
        aid = s.get("assigned_normal_id")
        s["valid_pool_reason"] = reasons.get(aid, "") if aid else "no_valid_bg"

    n_assigned = sum(1 for s in selected if s.get("assigned_normal_id"))
    logger.info("Assigned backgrounds to %d / %d ROIs (mean pool=%.2f)",
                n_assigned, len(selected), derived_pool["mean_pool_size"])

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    # NOTE: the full (roi x good) candidate set is O(n_roi x n_good) (millions on
    # severstal/aitex) — NOT persisted. selected.json keeps each ROI's ranked
    # top-pool + scores, which is the auditable record.
    save_json(selected, str(out / "clean_bg_selected.json"))
    rarm = None
    if emit_random_arm:
        rarm = random_arm(selected, valid_ids, seed)
        save_json(rarm, str(out / "clean_bg_random_arm.json"))

    (out / "clean_bg_summary.md").write_text(
        _build_summary(selected, derived_pool, derived_void, strategy), encoding="utf-8"
    )
    logger.info("Saved clean_bg_selected.json (%d) → %s", len(selected), out)

    return {
        "status": "ok",
        "n_selected": len(selected),
        "n_assigned": n_assigned,
        "derived_void": derived_void,
        "derived_pool": derived_pool,
        "emit_random_arm": bool(rarm is not None),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AROMA Step 3.5 — Clean-Background Selection")
    p.add_argument("--profiling_dir", required=True,
                   help="Phase 0 output (context_features.csv, compatibility_matrix.json)")
    p.add_argument("--roi_dir", required=True,
                   help="Step 3 output directory (roi_selected.json); outputs also written here by default")
    p.add_argument("--output_dir", required=True,
                   help="Directory to write clean_bg_selected.json")
    p.add_argument("--sampling_strategy", default="histogram", choices=["histogram"],
                   help="Background ranking strategy (Phase 1: histogram intersection)")
    p.add_argument("--seed", type=int, default=42, help="Seed for the random-arm control")
    p.add_argument("--emit_random_arm", action="store_true",
                   help="Also emit clean_bg_random_arm.json (symmetric control: same ROIs, random bg)")
    p.add_argument("--no_reject_clean_bg", dest="reject_clean_bg", action="store_false",
                   help="Disable the void/quality prefilter (keep all good images)")
    p.add_argument("--void_frac_max", type=float, default=None,
                   help="Max per-image void fraction to keep. Default: 0.5 "
                        "(majority-void boundary — an image whose more-than-half "
                        "patches are void is a partial plate). Override to pin a value.")
    p.add_argument("--void_floor_pct", type=float, default=15.0,
                   help="Percentile of observed local_variance/edge_density used to "
                        "derive the void floors (data-driven, per-dataset). Default 15 "
                        "sits above the dark-void cluster; raise/lower to tune.")
    p.add_argument("--var_floor", type=float, default=None,
                   help="Absolute local_variance void floor. Default: DATA-DRIVEN "
                        "(--void_floor_pct percentile). Set to REPLACE the derived "
                        "value (prescan safety valve — pins one dataset without code change).")
    p.add_argument("--edge_floor", type=float, default=None,
                   help="Absolute edge_density void floor. Default: DATA-DRIVEN "
                        "(--void_floor_pct percentile). Set to REPLACE the derived value.")
    p.add_argument("--pool_k", type=int, default=None,
                   help="Per-ROI ranked background pool size. Default: DATA-DRIVEN "
                        "(all size-fit candidates; generate_defects indexes rep→pool). "
                        "Set an int to cap.")
    p.add_argument("--geometry_prior", action="store_true",
                   help="Phase 3 (E2): also precompute a paste POSITION per pool bg "
                        "matching each class's real edge/surface tendency (from "
                        "morphology defect_bbox). Emits position/topk_positions; "
                        "generate_defects places there. Default OFF (mAP effect GPU-TBD).")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = _parse_args(argv)
    result = run(
        profiling_dir=args.profiling_dir,
        roi_dir=args.roi_dir,
        output_dir=args.output_dir,
        strategy=args.sampling_strategy,
        seed=args.seed,
        emit_random_arm=args.emit_random_arm,
        reject_clean_bg=args.reject_clean_bg,
        void_frac_max=args.void_frac_max,
        void_floor_pct=args.void_floor_pct,
        var_floor=args.var_floor,
        edge_floor=args.edge_floor,
        pool_k=args.pool_k,
        geometry_prior=args.geometry_prior,
    )
    if result.get("status") != "ok":
        sys.exit(1)


if __name__ == "__main__":
    main()
