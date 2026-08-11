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

    # Defect tile occupancy (defect_tiles.py, optional). Present → the per-source
    # query can be narrowed to the tiles ADJACENT to the defect instead of the
    # whole image background. Absent → --adjacent_radius is rejected in run().
    dt_path = pd / "defect_tiles.json"
    defect_tiles: Dict[str, Any] = {}
    defect_tiles_meta: Dict[str, Any] = {}
    if dt_path.exists():
        try:
            blob = load_json(str(dt_path))
            defect_tiles = blob.get("tiles", {}) or {}
            defect_tiles_meta = blob.get("meta", {}) or {}
        except Exception as exc:        # noqa: BLE001 - a bad cache must not kill the run
            logger.warning("defect_tiles.json unreadable (%s) — localization unavailable", exc)

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
        "defect_tiles": defect_tiles,
        "defect_tiles_meta": defect_tiles_meta,
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


# ---------------------------------------------------------------------------
# §5-4 — (k, c) 기반 신호: 1단계 k_fit + 2단계 ring_sgm 자리 산출
# (devnote aroma_adjacent_context_bg_selection.md §5-4, §6-9)
# ---------------------------------------------------------------------------

def _target_by_cluster(compat: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    """cluster k → L1 정규화한 matrix_symmetric[k] (목표 문맥 셀 분포).

    논문 수식 무수정: matrix_symmetric(k,c) ∝ sqrt(P_def(k,c)·P_clean(c)) 를 지지집합
    S_k 위에서 행 max 정규화한 값이다. 분포 매칭을 하려면 확률분포여야 하므로 L1 로
    다시 정규화한다 — 단조 rescale 이라 행의 상대 형태는 바뀌지 않는다.

    같은 목표를 두 단계가 공유한다:
      1단계 k_fit  : 배경 이미지의 셀 분포가 이 분포를 지원하는가 (P_clean 항이 정당 —
                     어떤 배경을 고를지 정하는 시점에는 가용성이 의미 있다)
      2단계 ring   : 후보 자리 둘레의 셀 분포가 이 분포와 겹치는가
    """
    out: Dict[str, Dict[str, float]] = {}
    for k, row in (compat.get("matrix_symmetric") or {}).items():
        s = sum(float(v) for v in row.values())
        if s > 0:
            out[str(k)] = {c: float(v) / s for c, v in row.items()}
    return out


def _tile_grid(rows: List[Dict[str, str]], names, bin_edges,
               var_floor: float, edge_floor: float,
               tile: int = 64) -> Tuple[Dict[Tuple[int, int], str], int, int]:
    """정상 이미지 → ({(i,j): cell}, gw, gh). void 타일은 제외한다.

    격자는 context_features.csv 가 emit 한 타일에서 그대로 읽는다 (절단 격자, F1).
    void 를 빼두면 footprint void-straddle 판정이 '타일 없음'과 같아진다.
    """
    grid: Dict[Tuple[int, int], str] = {}
    gw = gh = 0
    for r in rows:
        pxy = str(r.get("patch_xy", ""))
        if "_" not in pxy:
            continue
        try:
            x, y = (int(v) for v in pxy.split("_", 1))
        except (TypeError, ValueError):
            continue
        i, j = x // tile, y // tile
        gw, gh = max(gw, i + 1), max(gh, j + 1)
        if _patch_void(r, var_floor, edge_floor):
            continue
        grid[(i, j)] = _cell_key(r, names, bin_edges)
    return grid, gw, gh


def _np_quality(gray: np.ndarray, blur_threshold: float = 100.0) -> float:
    """CASDA 4-성분 배경 quality (0..1) — numpy 전용 구현.

    generate_defects._background_quality_score 와 같은 수식(blur 30% + contrast
    30% + brightness 20% + noise 20%)이되 cv2 를 쓰지 않는다(defect_tiles.py 와
    같은 이유 — cv2 dtype 지뢰 회피). Laplacian 은 3x3 커널의 interior 응답
    분산, noise 는 5x5 box-mean 잔차. 절대값이 cv2 구현과 미세하게 달라도
    무방하다 — 플로어가 같은 구현의 분위 분포에서 유도되므로 내부 정합만
    필요하다 (devnote aroma_site_quality_filter §1-b).
    """
    g = gray.astype(np.float32)
    if g.shape[0] < 7 or g.shape[1] < 7:
        return 1.0                      # 판정 불가 극소 영역 — 통과
    lap = (-4.0 * g[1:-1, 1:-1] + g[:-2, 1:-1] + g[2:, 1:-1]
           + g[1:-1, :-2] + g[1:-1, 2:])
    blur = 1.0 if float(lap.var()) >= blur_threshold else 0.3
    contrast = min(float(np.std(g)) / 128.0, 1.0)
    mb = float(np.mean(g)) / 255.0
    brightness = 1.0 if 0.3 <= mb <= 0.7 else 0.7
    k = 5
    c = np.pad(g, ((1, 0), (1, 0))).cumsum(axis=0).cumsum(axis=1)
    box = (c[k:, k:] - c[:-k, k:] - c[k:, :-k] + c[:-k, :-k]) / float(k * k)
    resid = g[k - 3:g.shape[0] - 2, k - 3:g.shape[1] - 2] - box
    noise = 1.0 - min(float(np.mean(resid ** 2)) / 100.0, 1.0)
    return 0.30 * blur + 0.30 * contrast + 0.20 * brightness + 0.20 * noise


def _ring_keys(si: int, sj: int, bw: int, bh: int) -> List[Tuple[int, int]]:
    """자리 사각형 [si..si+bw-1] x [sj..sj+bh-1] 의 8이웃 링 좌표."""
    i1, j1 = si + bw - 1, sj + bh - 1
    out = []
    for i in range(si - 1, i1 + 2):
        out.append((i, sj - 1))
        out.append((i, j1 + 1))
    for j in range(sj, j1 + 1):
        out.append((si - 1, j))
        out.append((i1 + 1, j))
    return out


def _best_ring_site(grid: Dict[Tuple[int, int], str], gw: int, gh: int,
                    bw: int, bh: int, tgt: Dict[str, float],
                    tile: int = 64,
                    allowed: Optional[set] = None,
                    ) -> Tuple[Optional[Tuple[int, int]], Optional[float]]:
    """ring_sgm: 링 셀 분포와 tgt 의 히스토그램 교집합이 최대인 자리 (픽셀 좌상단).

    footprint 에 void/결측 타일이 하나라도 있으면 그 자리는 버린다. generate 는
    이 위치를 forced_xy 로 소비하는데, 그 경로가 런타임 void 게이트와 tau 게이트를
    **우회**하므로(devnote §5-4-2) 여기서 걸러야 한다.

    Returns ``(best_xy, best_score)`` — the score is the argmax objective itself
    (배치 품질↔다운스트림 상관 분석 재료, devnote aroma_synth_provenance §4-1).
    ``(None, None)`` when nothing qualifies → 호출부가 position 을 비워 두고
    generate 가 기존 _positive_place 경로로 자연 폴백한다.
    """
    if not tgt or bw <= 0 or bh <= 0 or gw < bw or gh < bh:
        return None, None
    best, best_xy = -1.0, None
    for sj in range(gh - bh + 1):
        for si in range(gw - bw + 1):
            if allowed is not None and (si, sj) not in allowed:
                continue                      # site quality 필터 탈락 자리
            if any((si + a, sj + b) not in grid
                   for a in range(bw) for b in range(bh)):
                continue                      # void/결측 footprint
            ring = [grid[t] for t in _ring_keys(si, sj, bw, bh) if t in grid]
            if not ring:
                continue
            inv = 1.0 / len(ring)
            hist: Dict[str, float] = {}
            for c in ring:
                hist[c] = hist.get(c, 0.0) + inv
            score = sum(min(v, tgt[c]) for c, v in hist.items() if c in tgt)
            if score > best:
                best, best_xy = score, (si * tile, sj * tile)
    return best_xy, (best if best_xy is not None else None)


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


def _localize_defect_rows(
    defect_by_img: Dict[str, List[Dict[str, str]]],
    defect_tiles: Dict[str, Any],
    radius: int,
) -> Tuple[Dict[str, List[Dict[str, str]]], Dict[str, float]]:
    """Narrow each defect image's rows to the tiles ADJACENT to its defect.

    The query the research defines is "the background touching this defect", not
    "this image's background". `defect_tiles.json` carries, per image, the
    defect-pixel-free tiles within `radius` 8-neighbour steps of a defect-carrying
    tile, keyed by the same `patch_xy` string context_features.csv uses — so the
    narrowing is a set-membership filter, no coordinate arithmetic.

    Tiles CARRYING defect pixels are deliberately absent from that list: profiling
    keeps a tile whose defect fraction is <= 0.5, so its context features are
    computed on a patch holding up to 50% defect texture (severstal mean 15.5%),
    while every clean-pool tile holds 0%. Including them would search the clean
    pool for backgrounds resembling defect texture.

    Falls back to the whole image when the tile list is empty — an image whose
    defect lies entirely inside the grid's truncated right/bottom band has no
    adjacency at all (mtd 14.4%; see the F1 note in defect_tiles.py). Returns
    (rows_by_img, stats); stats feeds the summary so the fallback is never silent.
    """
    key = "adjacent_r%d" % radius
    out: Dict[str, List[Dict[str, str]]] = {}
    n_fallback = 0
    sizes: List[int] = []
    for iid, rows in defect_by_img.items():
        want = set(defect_tiles.get(iid, {}).get(key) or ())
        sel = [r for r in rows if r.get("patch_xy") in want] if want else []
        if not sel:
            n_fallback += 1
            sel = rows
        out[iid] = sel
        sizes.append(len(sel))
    arr = np.asarray(sizes, dtype=float) if sizes else np.zeros(0)
    stats = {
        "adjacent_radius": float(radius),
        "loc_fallback_frac": round(n_fallback / len(out), 4) if out else 0.0,
        "query_tiles_mean": round(float(arr.mean()), 2) if arr.size else 0.0,
        "query_tiles_lt4_frac": round(float((arr < 4).mean()), 4) if arr.size else 0.0,
    }
    return out, stats


def build_and_rank(
    data: Dict[str, Any],
    valid_ids: List[str],
    var_floor: float,
    edge_floor: float,
    pool_k: Optional[int],
    geometry_prior: bool = False,
    adjacent_radius: Optional[int] = None,
    k_fit: bool = False,
    site_selection: str = "off",
    site_pool_cap: int = 16,
    site_quality_pct: Optional[float] = None,
    image_dir: Optional[str] = None,
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

    # Query localization (devnote aroma_adjacent_context_bg_selection.md §5-2).
    # OFF by default: src_rows_by_img is defect_by_img and both histograms see the
    # same rows as before, so the legacy result is reproduced exactly.
    loc_stats: Dict[str, float] = {"adjacent_radius": 0.0}
    if adjacent_radius:
        src_rows_by_img, loc_stats = _localize_defect_rows(
            defect_by_img, data.get("defect_tiles") or {}, int(adjacent_radius))
        # Both signals are narrowed: the class aggregate is the same query pooled
        # over a class, so leaving it global would mix two spatial scales.
        defect_rows_q = [r for rows in src_rows_by_img.values() for r in rows]
        logger.info("Query localization ON: adjacent_r%d — %.1f tiles/query, "
                    "fallback %.1f%%, <4 tiles %.1f%%",
                    int(adjacent_radius), loc_stats["query_tiles_mean"],
                    100 * loc_stats["loc_fallback_frac"],
                    100 * loc_stats["query_tiles_lt4_frac"])
    else:
        src_rows_by_img, defect_rows_q = defect_by_img, defect_rows

    src_hist_by_img = {iid: _image_hist(rows, names, bin_edges, var_floor, edge_floor)
                       for iid, rows in src_rows_by_img.items()}
    class_hist = _class_bg_hist(defect_rows_q, names, bin_edges, var_floor, edge_floor,
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

    # §5-4: (k, c) 목표 분포. 1단계 k_fit 과 2단계 ring 자리 선택이 공유한다.
    tgt_by_k = _target_by_cluster(data.get("compat") or {}) if (k_fit or site_selection == "ring") else {}
    if (k_fit or site_selection == "ring") and not tgt_by_k:
        logger.warning("matrix_symmetric 이 없어 k_fit / ring 자리 선택을 끕니다 "
                       "(--compat_mode symmetric 프로파일링 필요)")
        k_fit, site_selection = False, "off"
    # 자리 선택용 타일 격자 — 정상 이미지당 1회, ROI 전체가 재사용
    grid_cache: Dict[str, Tuple[Dict[Tuple[int, int], str], int, int]] = {}

    def _grid_for(iid: str):
        g = grid_cache.get(iid)
        if g is None:
            g = _tile_grid(good_by_img.get(iid, []), names, bin_edges,
                           var_floor, edge_floor)
            grid_cache[iid] = g
        return g

    # ---- Pass 1: score both signals per (ROI, good); collect per-ROI lift ----
    per_roi: List[Dict[str, Any]] = []
    lifts_src: List[float] = []
    lifts_cls: List[float] = []
    lifts_size: List[float] = []
    lifts_k: List[float] = []
    src_ceilings: List[float] = []
    for roi_idx, r in enumerate(roi):
        axis, cls_val = _roi_class_axis(r, multi_class)
        src_dv = src_hist_by_img.get(str(r.get("image_id", "")), {})
        cls_dv = class_hist.get(str(r.get("class_key") or ""), {})
        # §5-4-1b — 형태 군집 축. class_fit(도메인 라벨 축)과 겹치지 않는 정보다:
        # severstal class 4 vs cluster 5, mvtec_leather class 5 vs cluster 3.
        k_dv = tgt_by_k.get(str(r.get("cluster_id"))) or {} if k_fit else {}
        bbox = _parse_bbox(r.get("defect_bbox"))
        wh = (bbox[2], bbox[3]) if bbox else (0, 0)
        cand = []  # (normal_id, src_fit, class_fit, size_ok, size_fit, k_fit)
        s_scores, c_scores, z_scores, k_scores = [], [], [], []
        for iid in valid_ids:
            sf = _hist_intersection(good_hist[iid], src_dv)
            cf = _hist_intersection(good_hist[iid], cls_dv)
            so = _size_ok(wh, good_dim[iid]) if wh != (0, 0) else True
            # Option 1: size-fit signal = fit-rescale factor (1.0=no distortion,
            # <1=shrink). Constant across same-size backgrounds → 0 lift → 0 weight
            # (auto-downweighted); varies only when backgrounds differ in size.
            zf = _scale_to_fit(wh, good_dim[iid]) if wh != (0, 0) else 1.0
            kf = _hist_intersection(good_hist[iid], k_dv) if k_dv else 0.0
            cand.append((iid, sf, cf, so, zf, kf))
            s_scores.append(sf); c_scores.append(cf); z_scores.append(zf)
            k_scores.append(kf)
        if s_scores:
            lifts_src.append(max(s_scores) - float(np.median(s_scores)))
            lifts_cls.append(max(c_scores) - float(np.median(c_scores)))
            lifts_size.append(max(z_scores) - float(np.median(z_scores)))
            lifts_k.append(max(k_scores) - float(np.median(k_scores)))
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
    w_k = float(np.mean(lifts_k)) if (k_fit and lifts_k) else 0.0
    lift_src_m, lift_cls_m, lift_size_m, lift_k_m = w_src, w_cls, w_size, w_k
    tot = w_src + w_cls + w_size + w_k
    if tot <= 0:                       # all signals flat → fall back to per-source
        w_src, w_cls, w_size, w_k = 1.0, 0.0, 0.0, 0.0
    else:
        w_src, w_cls, w_size, w_k = w_src / tot, w_cls / tot, w_size / tot, w_k / tot

    # ---- Pass 2: combine, rank, assign ----
    selected: List[Dict[str, Any]] = []
    pool_sizes: List[int] = []
    n_site_pos, n_site_none = 0, 0
    site_top_scores: List[float] = []   # top-1 site_score per ROI (derived stats)
    # 레코드별 배치 모드 라벨 — generate 측 position_source 판정 재료. derived 는
    # summary md 로만 나가 JSON 소비자가 접근할 수 없으므로 레코드에 직접 기록한다
    # (devnote aroma_synth_provenance §4-2b, §8-1 확정).
    _site_mode = ("ring" if site_selection == "ring"
                  else ("geometry_prior" if geometry_prior else "off"))
    # 자리 quality 필터 (devnote aroma_site_quality_filter): ring 전용 2차 게이트.
    # ON 이면 자리 확정을 루프 뒤로 미룬다(2-phase) — 플로어가 전체 자리 quality
    # 분포에서 나와야 하기 때문. OFF 경로는 기존 인라인 그대로 (byte-identical).
    site_q_on = (site_selection == "ring") and (site_quality_pct is not None)
    _deferred: List[Tuple[int, List[str], Dict[str, Any]]] = []
    for pr in per_roi:
        src_key = pr["src_key"]
        scored = []  # (combined_jittered, normal_id, src_fit, class_fit, size_ok, size_fit, k_fit)
        for iid, sf, cf, so, zf, kf in pr["cand"]:
            comb = w_src * sf + w_cls * cf + w_size * zf + w_k * kf
            scored.append((comb + _bg_jitter(src_key, iid), iid, sf, cf, so, zf, kf))
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
            """(position, site_score). ring 외 경로의 score 는 None."""
            if not pr["bbox"]:
                return None, None
            dim = good_dim.get(nid, (0, 0))
            wh = _effective_wh((pr["bbox"][2], pr["bbox"][3]), dim)
            if site_selection == "ring":
                # §5-4-1 ring_sgm: 링의 셀 분포가 tgt[k] 와 가장 겹치는 자리.
                # EFFECTIVE(fit-rescale 후) 크기로 타일 사각형을 잡아야 generation 이
                # 실제로 붙일 크기와 일치한다.
                tgt = tgt_by_k.get(str(pr["cluster_id"])) or {}
                grid, gw, gh = _grid_for(nid)
                bw = max(1, -(-int(wh[0]) // 64))
                bh = max(1, -(-int(wh[1]) // 64))
                return _best_ring_site(grid, gw, gh, bw, bh, tgt)
            if not geometry_prior:
                return None, None
            want_edge = class_edge.get(str(pr["cls_val"]), global_edge) > global_edge
            j = (_bg_jitter(src_key, nid) / 1e-7)  # blake2b frac in [0,1)
            return _place_position(wh, dim, want_edge, j), None

        # ring 모드는 자리 탐색이 비싸므로 pool 상위 N 개만 산출한다. generate 는
        # rep_idx % len(pool) 로 인덱싱하므로 앞쪽만 실제로 소비된다.
        def _positions_for(pool_ids):
            """(positions, site_scores) — 두 리스트는 동일 index 정렬."""
            cap = len(pool_ids) if site_selection != "ring" else max(1, site_pool_cap)
            pos_list, score_list = [], []
            for i, nid in enumerate(pool_ids):
                p, s = _pos_for(nid) if i < cap else (None, None)
                pos_list.append(p)
                score_list.append(round(s, 6) if s is not None else None)
            return pos_list, score_list

        if top:
            comb_j, best_id, sf, cf, so, zf, kf = top[0]
            pool_ids = [t[1] for t in top]
            if site_q_on:
                # 자리 확정은 phase B — 여기서는 자리 배열만 자리표시 (아래에서 패치)
                positions = [None] * len(pool_ids)
                site_scores: List[Optional[float]] = [None] * len(pool_ids)
                site_quals: List[Optional[float]] = [None] * len(pool_ids)
                _deferred.append((len(selected), pool_ids, pr))
            else:
                positions, site_scores = _positions_for(pool_ids)
                site_quals = [None] * len(positions)
            pos_best = positions[0] if positions else None
            site_best = site_scores[0] if site_scores else None
            if site_best is not None:
                site_top_scores.append(site_best)
            if site_selection == "ring" and not site_q_on:
                n_site_pos += sum(1 for p in positions[:max(1, site_pool_cap)] if p)
                n_site_none += sum(1 for p in positions[:max(1, site_pool_cap)] if not p)
            selected.append(dict(base,
                                 assigned_normal_id=best_id,
                                 topk_pool=pool_ids,
                                 topk_positions=positions,
                                 topk_site_scores=site_scores,     # index-aligned with topk_positions
                                 topk_site_quality=site_quals,     # index-aligned (필터 OFF 면 전부 None)
                                 position=pos_best,
                                 site_score=site_best,             # ring argmax objective (None off-ring)
                                 site_quality=(site_quals[0] if site_quals else None),
                                 site_mode=_site_mode,
                                 score=round(comb_j - _bg_jitter(src_key, best_id), 6),
                                 hist_intersection=round(sf, 6),   # per-source (E1-comparable)
                                 class_fit=round(cf, 6),           # class-conditioned (Phase 2)
                                 k_fit=round(kf, 6),               # §5-4-1b 형태 군집 축
                                 size_ok=bool(so),
                                 size_fit=round(zf, 4),            # Option 1 signal (1=no distortion)
                                 scale_factor=round(zf, 4),        # Option 2: generation's fit-rescale
                                 n_valid_bg=len(pr["cand"])))
        else:
            selected.append(dict(base, assigned_normal_id=None, topk_pool=[],
                                 topk_positions=[], topk_site_scores=[],
                                 topk_site_quality=[],
                                 position=None, site_score=None, site_quality=None,
                                 site_mode=_site_mode,
                                 score=0.0, hist_intersection=0.0, class_fit=0.0,
                                 k_fit=0.0,
                                 size_ok=False, size_fit=0.0, scale_factor=0.0,
                                 n_valid_bg=0))

    # ---- Phase B: 자리 quality 필터 (devnote aroma_site_quality_filter) ----
    # ① void 게이트(타일, admissibility — _grid_for 가 이미 반영) 위에서
    # ② admissible 자리별 crop-영역 quality 산출 → ③ 데이터셋 자기 분포의
    # 하위 site_quality_pct% 배제 → ④ 생존 자리 중 ring argmax.
    # 자리 단위(타일 아님)라 footprint 1타일 배제로 자리가 전멸하는 비선형이 없다
    # (타일 단위는 severstal 폴백 +20.2p 로 기각 — 시뮬레이션 2026-08-11).
    q_floor: Optional[float] = None
    n_q_filtered = 0
    n_q_unresolved_bg = 0
    if site_q_on and _deferred:
        from PIL import Image  # 가드된 임포트 — run() 이 사전 검증

        stem2p: Dict[str, Path] = {}
        for p in sorted(Path(image_dir).iterdir()):
            if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp"):
                stem2p[p.stem] = p
        gray_cache: Dict[str, Optional[np.ndarray]] = {}

        def _gray(nid: str) -> Optional[np.ndarray]:
            if nid in gray_cache:
                return gray_cache[nid]
            p = (stem2p.get(nid) or stem2p.get(nid.lstrip("_"))
                 or stem2p.get("_" + nid))
            g = None
            if p is not None:
                try:
                    g = np.asarray(Image.open(p).convert("L"), dtype=np.uint8)
                except Exception as exc:  # noqa: BLE001 — 깨진 파일은 필터만 건너뜀
                    logger.warning("site_quality: %s 로드 실패 (%s)", p, exc)
            gray_cache[nid] = g
            return g

        # (nid, ew, eh) → [((si,sj), quality)] admissible 자리 전수. None = 이미지
        # 미해석(그 배경만 필터 미적용 — 기존 무필터 자리 선택으로 동작).
        sites_cache: Dict[Tuple[str, int, int],
                          Optional[List[Tuple[Tuple[int, int], float]]]] = {}

        def _sites(nid: str, ew: int, eh: int, bw: int, bh: int):
            key = (nid, ew, eh)
            if key in sites_cache:
                return sites_cache[key]
            g = _gray(nid)
            if g is None:
                sites_cache[key] = None
                return None
            grid, gw, gh = _grid_for(nid)
            out: List[Tuple[Tuple[int, int], float]] = []
            if gw >= bw and gh >= bh:
                for sj in range(gh - bh + 1):
                    for si in range(gw - bw + 1):
                        if any((si + a, sj + b) not in grid
                               for a in range(bw) for b in range(bh)):
                            continue
                        if not any(t in grid for t in _ring_keys(si, sj, bw, bh)):
                            continue
                        x, y = si * 64, sj * 64
                        out.append(((si, sj), _np_quality(g[y:y + eh, x:x + ew])))
            sites_cache[key] = out
            return out

        def _slot_geo(pr_bbox, nid):
            dim = good_dim.get(nid, (0, 0))
            ew, eh = _effective_wh((pr_bbox[2], pr_bbox[3]), dim)
            bw = max(1, -(-int(ew) // 64))
            bh = max(1, -(-int(eh) // 64))
            return int(ew), int(eh), bw, bh

        cap = max(1, site_pool_cap)
        # 플로어 산출 — cap 이내 슬롯의 admissible 자리 quality 전수
        all_q: List[float] = []
        for _idx, pool_ids, pr in _deferred:
            if not pr["bbox"]:
                continue
            for nid in pool_ids[:cap]:
                ew, eh, bw, bh = _slot_geo(pr["bbox"], nid)
                s = _sites(nid, ew, eh, bw, bh)
                if s:
                    all_q.extend(q for _, q in s)
        if all_q:
            q_floor = float(np.percentile(np.asarray(all_q), float(site_quality_pct)))

        # 자리 확정 — 생존 자리 중 ring argmax
        for idx, pool_ids, pr in _deferred:
            rec = selected[idx]
            positions = rec["topk_positions"]
            site_scores = rec["topk_site_scores"]
            site_quals = rec["topk_site_quality"]
            tgt = tgt_by_k.get(str(pr["cluster_id"])) or {}
            if pr["bbox"]:
                for i, nid in enumerate(pool_ids[:cap]):
                    ew, eh, bw, bh = _slot_geo(pr["bbox"], nid)
                    s = _sites(nid, ew, eh, bw, bh)
                    grid, gw, gh = _grid_for(nid)
                    if s is None:
                        # 이미지 미해석 — 필터 없이 기존 자리 선택
                        n_q_unresolved_bg += 1
                        xy, sc = _best_ring_site(grid, gw, gh, bw, bh, tgt)
                        qv = None
                    else:
                        survivors = {xy for xy, q in s
                                     if q_floor is None or q >= q_floor}
                        n_q_filtered += len(s) - len(survivors)
                        if survivors:
                            xy, sc = _best_ring_site(grid, gw, gh, bw, bh, tgt,
                                                     allowed=survivors)
                            _qmap = {xy_: q for xy_, q in s}
                            qv = (_qmap.get((xy[0] // 64, xy[1] // 64))
                                  if xy is not None else None)
                        else:
                            xy, sc, qv = None, None, None   # 자리 전멸 → 폴백
                    positions[i] = xy
                    site_scores[i] = round(sc, 6) if sc is not None else None
                    site_quals[i] = round(qv, 6) if qv is not None else None
            rec["position"] = positions[0] if positions else None
            rec["site_score"] = site_scores[0] if site_scores else None
            rec["site_quality"] = site_quals[0] if site_quals else None
            if rec["site_score"] is not None:
                site_top_scores.append(rec["site_score"])
            n_site_pos += sum(1 for p in positions[:cap] if p)
            n_site_none += sum(1 for p in positions[:cap] if not p)
        logger.info(
            "site quality filter ON: pct=%.1f floor=%s — 자리 %d개 배제, "
            "미해석 배경 슬롯 %d, positions=%d fallback=%d",
            float(site_quality_pct),
            ("%.6f" % q_floor) if q_floor is not None else "n/a",
            n_q_filtered, n_q_unresolved_bg, n_site_pos, n_site_none,
        )

    derived = {
        "pool_cut": "p95" if not pool_k else ("k=%d" % pool_k),
        "mean_pool_size": float(np.mean(pool_sizes)) if pool_sizes else 0.0,
        "multi_class": float(multi_class),
        "w_src": round(w_src, 4), "w_class": round(w_cls, 4), "w_size": round(w_size, 4),
        "w_k": round(w_k, 4),
        "lift_src": round(lift_src_m, 4),
        "lift_class": round(lift_cls_m, 4),
        "lift_size": round(lift_size_m, 4),
        "lift_k": round(lift_k_m, 4),
        "k_fit": float(bool(k_fit)),
        "site_selection": site_selection,
        "site_pool_cap": float(site_pool_cap),
        "site_positions": float(n_site_pos),
        "site_fallback": float(n_site_none),
        "site_fallback_frac": (n_site_none / (n_site_pos + n_site_none)
                               if (n_site_pos + n_site_none) else 0.0),
        # top-1 ring site score distribution (None 제외; ring OFF 면 전부 0.0) —
        # 향후 임계/선별 판단 근거 (devnote aroma_synth_provenance §4-1c)
        "site_score_mean": (round(float(np.mean(site_top_scores)), 6)
                            if site_top_scores else 0.0),
        "site_score_p10": (round(float(np.percentile(site_top_scores, 10.0)), 6)
                           if site_top_scores else 0.0),
        "site_score_p90": (round(float(np.percentile(site_top_scores, 90.0)), 6)
                           if site_top_scores else 0.0),
        # 자리 quality 필터 (devnote aroma_site_quality_filter; OFF 면 0.0/-1)
        "site_quality_pct": float(site_quality_pct) if site_q_on else 0.0,
        "site_quality_floor": (round(q_floor, 6) if q_floor is not None else -1.0),
        "site_quality_filtered": float(n_q_filtered),
        "site_quality_unresolved_bg": float(n_q_unresolved_bg),
        # per-source ceiling (E1 reproduction gate — independent of Phase-2 weights)
        "src_fit_ceiling_mean": round(float(np.mean(src_ceilings)), 4) if src_ceilings else 0.0,
        # fraction of ROIs whose image_id resolves to a defect row (E1 signal health);
        # <1.0 → stale roi / profiling image_id mismatch (see warning above)
        "src_match_frac": round(src_match_frac, 4),
        "geometry_prior": float(geometry_prior),
        "class_edge_prior": {c: round(v, 3) for c, v in class_edge.items()},
        "global_edge": round(global_edge, 3),
    }
    derived.update(loc_stats)
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
        f"w_size={derived_pool.get('w_size')}  w_k={derived_pool.get('w_k')}  "
        f"(lift_src={derived_pool.get('lift_src')}, lift_class={derived_pool.get('lift_class')}, "
        f"lift_size={derived_pool.get('lift_size')}, lift_k={derived_pool.get('lift_k')})",
        (f"- **k_fit** (§5-4-1b 형태 군집 축): ON  w_k={derived_pool.get('w_k')}  "
         f"lift_k={derived_pool.get('lift_k')}  "
         f"(w_k≈0 이면 cluster 축이 배경을 구분하지 못한다는 뜻 — 자동 소거됨)"
         if derived_pool.get("k_fit") else "- k_fit: OFF"),
        (f"- **site_selection=ring** (§5-4-1 ring_sgm): positions "
         f"{int(derived_pool.get('site_positions', 0))}  "
         f"fallback {int(derived_pool.get('site_fallback', 0))} "
         f"({100 * float(derived_pool.get('site_fallback_frac', 0.0)):.1f}%)  "
         f"pool_cap={int(derived_pool.get('site_pool_cap', 0))}  "
         f"(fallback 은 position=None → generate 가 _positive_place 로 폴백)"
         if derived_pool.get("site_selection") == "ring" else "- site_selection: off"),
        f"- src_fit_ceiling_mean={derived_pool.get('src_fit_ceiling_mean')}  "
        f"(E1 reproduction gate — compare to E1 sim_best)",
        f"- src_match_frac={derived_pool.get('src_match_frac')}  "
        f"(ROI image_id ↔ context defect 매칭율; <1.0이면 stale roi/profiling 불일치)",
        (f"- **query localization**: adjacent_r{int(derived_pool['adjacent_radius'])}  "
         f"query_tiles_mean={derived_pool.get('query_tiles_mean')}  "
         f"fallback={100 * float(derived_pool.get('loc_fallback_frac', 0.0)):.1f}%  "
         f"<4tiles={100 * float(derived_pool.get('query_tiles_lt4_frac', 0.0)):.1f}%  "
         f"(질의=결함 인접 배경 타일; fallback은 전역 질의로 되돌아간 ROI 비율)"
         if derived_pool.get("adjacent_radius") else
         "- query localization: OFF (질의 = 결함 이미지 배경 전체 — legacy)"),
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
    adjacent_radius: Optional[int] = None,
    k_fit: bool = False,
    site_selection: str = "off",
    site_pool_cap: int = 16,
    site_quality_filter: bool = False,
    site_quality_pct: float = 15.0,
    image_dir: Optional[str] = None,
    output_tag: str = "",
) -> Dict[str, Any]:
    if site_selection == "ring" and geometry_prior:
        logger.error("--site_selection ring 과 --geometry_prior 는 배타입니다 "
                     "(둘 다 topk_positions 를 채운다). 하나만 쓰세요.")
        return {"status": "conflicting_position_sources"}
    # 자리 quality 필터 선결 검증 (devnote aroma_site_quality_filter §1-a)
    if site_quality_filter:
        if site_selection != "ring":
            logger.error("--site_quality_filter 는 --site_selection ring 전용입니다 "
                         "(자리 개념이 ring 에서만 정의됨).")
            return {"status": "site_quality_requires_ring"}
        if not image_dir or not Path(image_dir).is_dir():
            logger.error("--site_quality_filter 는 --image_dir <good 이미지 디렉터리> "
                         "가 필요합니다 (자리 quality 는 픽셀에서 산출). got: %s",
                         image_dir)
            return {"status": "site_quality_needs_image_dir"}
        try:
            from PIL import Image  # noqa: F401
        except ImportError:
            logger.error("--site_quality_filter 는 Pillow 가 필요합니다.")
            return {"status": "site_quality_needs_pillow"}
    logger.info("Loading inputs: profiling_dir=%s roi_dir=%s", profiling_dir, roi_dir)
    data = load_inputs(profiling_dir, roi_dir)
    if data["status"] != "ok":
        logger.error("Input loading failed: %s", data)
        return data

    # Localization is opt-in and must FAIL LOUD rather than degrade to the global
    # query: a silent fallback would look like "the change had no effect".
    if adjacent_radius:
        if not data.get("defect_tiles"):
            logger.error("--adjacent_radius %d needs defect_tiles.json in %s. "
                         "Run scripts/aroma/defect_tiles.py first.",
                         adjacent_radius, profiling_dir)
            return {"status": "missing_defect_tiles", "profiling_dir": profiling_dir}
        radii = [int(r) for r in (data.get("defect_tiles_meta", {}).get("radii") or [])]
        if radii and int(adjacent_radius) not in radii:
            logger.error("defect_tiles.json carries radii %s, not %d. Re-run "
                         "defect_tiles.py with --radii %d.",
                         radii, adjacent_radius, adjacent_radius)
            return {"status": "radius_unavailable", "available": radii}

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
        data, valid_ids, var_floor, edge_floor, pool_k, geometry_prior=geometry_prior,
        adjacent_radius=adjacent_radius, k_fit=k_fit,
        site_selection=site_selection, site_pool_cap=site_pool_cap,
        site_quality_pct=(site_quality_pct if site_quality_filter else None),
        image_dir=image_dir,
    )
    if derived_pool.get("site_selection") == "ring":
        logger.info("ring_sgm 자리 산출: %d positions, fallback %d (%.1f%%), pool_cap=%d",
                    int(derived_pool["site_positions"]), int(derived_pool["site_fallback"]),
                    100 * derived_pool["site_fallback_frac"], site_pool_cap)
    if k_fit:
        logger.info("k_fit ON: w_k=%.4f (lift_k=%.4f) — 0 에 가까우면 형태 군집 축이 "
                    "배경을 구분하지 못한다는 뜻", derived_pool["w_k"], derived_pool["lift_k"])
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
    # output_tag: 기존 산출물을 덮지 않고 나란히 두기 위한 접미사. 예 '_ring' →
    # clean_bg_selected_ring.json. generate_defects 는 --clean_bg_json 으로 지정한다.
    tag = output_tag or ""
    save_json(selected, str(out / f"clean_bg_selected{tag}.json"))
    rarm = None
    if emit_random_arm:
        rarm = random_arm(selected, valid_ids, seed)
        save_json(rarm, str(out / f"clean_bg_random_arm{tag}.json"))

    (out / f"clean_bg_summary{tag}.md").write_text(
        _build_summary(selected, derived_pool, derived_void, strategy), encoding="utf-8"
    )
    logger.info("Saved clean_bg_selected%s.json (%d) → %s", tag, len(selected), out)

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
    p.add_argument("--adjacent_radius", type=int, default=None,
                   help="Narrow the per-source query to background tiles ADJACENT to "
                        "the defect (N 8-neighbour steps), read from defect_tiles.json. "
                        "Requires scripts/aroma/defect_tiles.py to have run. Default OFF "
                        "= whole-image background query (legacy, byte-identical). "
                        "Recommended: 1 (severstal/kolektor/mvtec_leather), 2 (mtd/aitex)")
    p.add_argument("--output_tag", default="",
                   help="산출 파일명 접미사. 예 '_ring' → clean_bg_selected_ring.json / "
                        "clean_bg_random_arm_ring.json / clean_bg_summary_ring.md. 기존 "
                        "산출물을 덮지 않고 나란히 둘 때 사용하며, generate_defects 에는 "
                        "--clean_bg_json 으로 경로를 지정한다. 기본 '' (기존 파일명)")
    p.add_argument("--k_fit", action="store_true",
                   help="§5-4-1b — 배경 이미지 랭킹에 형태 군집(k) 축 신호를 추가한다: "
                        "hist∩(good_hist, L1norm(matrix_symmetric[k])). class_fit 은 도메인 "
                        "라벨 축이라 겹치지 않는다(severstal class4 vs cluster5, leather "
                        "class5 vs cluster3). 가중치는 기존 lift 자동 산출이 배분하므로 "
                        "신호가 평탄하면 w_k≈0 으로 스스로 소거된다. 권장 ON")
    p.add_argument("--site_selection", default="off", choices=["off", "ring"],
                   help="§5-4-1 — 'ring': 붙일 자리를 오프라인에서 확정한다. 자리 둘레의 "
                        "셀 분포와 L1norm(matrix_symmetric[k]) 의 히스토그램 교집합이 최대인 "
                        "위치(footprint 에 void 있는 자리는 배제). topk_positions/position 에 "
                        "기록되고 generate 가 forced_xy 로 소비 → generate 측 무변경. "
                        "--geometry_prior 와 배타. 기본 off (legacy 동일)")
    p.add_argument("--site_quality_filter", action="store_true",
                   help="ring 자리에 2차 quality 게이트 적용 (기본 OFF). "
                        "① void 게이트 → ② admissible 자리의 crop-영역 quality "
                        "분위 필터 → ③ 생존 자리 중 ring argmax. "
                        "--site_selection ring + --image_dir 필수 "
                        "(devnote aroma_site_quality_filter)")
    p.add_argument("--site_quality_pct", type=float, default=15.0,
                   help="자리 quality 분위 플로어 (하위 X%% 배제, 기본 15.0 — "
                        "void_floor_pct 와 통일). 절대 임계 아님 — 데이터셋 자기 "
                        "분포 기준이라 leather 포화 없음")
    p.add_argument("--image_dir", default=None,
                   help="good 이미지 디렉터리 (자리 quality 는 픽셀에서 산출). "
                        "--site_quality_filter ON 일 때 필수")
    p.add_argument("--site_pool_cap", type=int, default=16,
                   help="ring 모드에서 자리를 산출할 pool 상위 개수. generate 는 "
                        "rep_idx %% len(pool) 로 인덱싱하므로 앞쪽만 소비된다. 기본 16")
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
        adjacent_radius=args.adjacent_radius,
        k_fit=args.k_fit,
        site_selection=args.site_selection,
        site_pool_cap=args.site_pool_cap,
        site_quality_filter=args.site_quality_filter,
        site_quality_pct=args.site_quality_pct,
        image_dir=args.image_dir,
        output_tag=args.output_tag,
    )
    if result.get("status") != "ok":
        sys.exit(1)


if __name__ == "__main__":
    main()
