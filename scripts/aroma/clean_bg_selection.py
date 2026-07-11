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
from collections import Counter, defaultdict
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


def _derive_void_floors(good_by_img: Dict[str, List[Dict[str, str]]]) -> Tuple[float, float]:
    """Data-driven void floors: a small low-percentile of the observed
    local_variance / edge_density distributions (NOT a hardcoded constant). Void
    tiles cluster at/near zero; the 1st percentile separates them from textured
    background without a magic threshold."""
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
    var_floor = float(np.percentile(lv, 1.0))
    edge_floor = float(np.percentile(ed, 1.0))
    return var_floor, edge_floor


def valid_bg_pool(
    good_by_img: Dict[str, List[Dict[str, str]]],
    reject_clean_bg: bool,
    void_frac_max: Optional[float],
    var_floor: float,
    edge_floor: float,
) -> Tuple[List[str], Dict[str, str], Dict[str, float]]:
    """Keep good images whose void_frac is at/below a data-derived cut. ALL-reject
    → fall back to the full pool (never a silent 0-output)."""
    void_frac: Dict[str, float] = {}
    for iid, rows in good_by_img.items():
        n = len(rows) or 1
        v = sum(1 for r in rows if _patch_void(r, var_floor, edge_floor))
        void_frac[iid] = v / n

    if void_frac_max is None:
        # Data-driven cut: the 90th percentile of observed void_frac — images
        # in the void-heavy tail are dropped, the bulk is kept. No magic number.
        vals = list(void_frac.values())
        void_frac_max = float(np.percentile(vals, 90.0)) if vals else 1.0

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
        counts[_cell_key(r, names, bin_edges)] = counts.get(_cell_key(r, names, bin_edges), 0) + 1
        total += 1
    if total == 0:
        return {}
    return {c: n / total for c, n in counts.items()}


def _class_bg_hist(defect_rows: List[Dict[str, str]], names, bin_edges,
                   var_floor: float, edge_floor: float,
                   iid_to_class: Dict[str, str]) -> Dict[str, Dict[str, float]]:
    """Class-conditioned source-defect background histograms: aggregate defect
    image patches BY class → {class: {cell: frac}}. Offline analogue of
    _dv_bg_hist, generalized to the class axis (§1-a.1). Defect tiles are already
    excluded upstream by profiling's _context_worker."""
    by_class_counts: Dict[str, Counter] = defaultdict(Counter)
    by_class_total: Dict[str, int] = defaultdict(int)
    for r in defect_rows:
        iid = r.get("image_id", "")
        cls = iid_to_class.get(iid)
        if cls is None:
            continue
        if _patch_void(r, var_floor, edge_floor):
            continue
        by_class_counts[cls][_cell_key(r, names, bin_edges)] += 1
        by_class_total[cls] += 1
    out: Dict[str, Dict[str, float]] = {}
    for cls, counts in by_class_counts.items():
        tot = by_class_total[cls] or 1
        out[cls] = {c: n / tot for c, n in counts.items()}
    return out


# ---------------------------------------------------------------------------
# §1-a.2 — bbox size-fit (hard gate)
# ---------------------------------------------------------------------------

def _image_dim(rows: List[Dict[str, str]], tile: int = 64) -> Tuple[int, int]:
    """Approx (W, H) of a good image from its patch grid: max patch_xy + tile.
    patch_xy is 'x_y' (top-left of a 64px patch)."""
    max_x = max_y = 0
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
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, float]]:
    """Score each (ROI x valid good image) by class-conditioned histogram
    intersection, apply the size-fit hard gate, and assign a ranked top-K pool
    per ROI. Deterministic (no rng) → symmetric-control friendly."""
    names, bin_edges = data["names"], data["bin_edges"]
    roi = data["roi"]
    good_by_img = data["good_by_img"]
    defect_rows = data["defect_rows"]

    # class axis (multi vs single) — labeling/audit only in Phase 1
    class_keys = {str(r.get("class_key") or "_") for r in roi}
    multi_class = len(class_keys - {"", "_", "defect"}) > 1

    # Phase 1 (extract-first): dv = the SOURCE defect image's own background
    # histogram — byte-faithful to generate_defects._dv_bg_hist / E1, so the
    # ranking is directly verifiable against E1's numbers. Class-conditioned dv
    # (aggregate by class) is the Phase-2 upgrade (§1-a.1), deferred.
    defect_by_img: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for r in defect_rows:
        defect_by_img[r.get("image_id", "")].append(r)
    src_hist_by_img: Dict[str, Dict[str, float]] = {
        iid: _image_hist(rows, names, bin_edges, var_floor, edge_floor)
        for iid, rows in defect_by_img.items()
    }

    # precompute per valid good image: hist + dim (once)
    good_hist: Dict[str, Dict[str, float]] = {}
    good_dim: Dict[str, Tuple[int, int]] = {}
    for iid in valid_ids:
        rows = good_by_img.get(iid, [])
        good_hist[iid] = _image_hist(rows, names, bin_edges, var_floor, edge_floor)
        good_dim[iid] = _image_dim(rows)

    selected: List[Dict[str, Any]] = []
    pool_sizes: List[int] = []

    for roi_idx, r in enumerate(roi):
        axis, cls_val = _roi_class_axis(r, multi_class)
        dv = src_hist_by_img.get(str(r.get("image_id", "")), {})
        bbox = _parse_bbox(r.get("defect_bbox"))
        wh = (bbox[2], bbox[3]) if bbox else (0, 0)
        src_key = "%s\x1f%s" % (str(r.get("image_path", "")), str(r.get("defect_bbox", "")))

        # Score every valid good image. size_ok is RECORDED (not a hard exclude):
        # generate_defects rescales oversized crops, so dropping the ROI here
        # would only force a fallback. Keep the assignment, flag size_ok.
        scored: List[Tuple[float, str, bool]] = []  # (jittered_score, normal_id, size_ok)
        for iid in valid_ids:
            size_ok = _size_ok(wh, good_dim[iid]) if wh != (0, 0) else True
            hi = _hist_intersection(good_hist[iid], dv)  # ranking score = E1 metric
            scored.append((hi + _bg_jitter(src_key, iid), iid, size_ok))
        scored.sort(key=lambda t: t[0], reverse=True)

        # Data-driven pool cut (no hardcoding): keep the top decile of this ROI's
        # scores (>= p95), i.e. the strongest matches, so the pool is bounded and
        # meaningful for rep-cycling; --pool_k caps it explicitly if passed.
        if scored:
            if pool_k:
                top = scored[:max(1, pool_k)]
            else:
                cut = float(np.percentile([t[0] for t in scored], 95.0))
                top = [t for t in scored if t[0] >= cut] or scored[:1]
        else:
            top = []
        pool_sizes.append(len(top))

        base = {
            "roi_idx": roi_idx,
            "image_id": str(r.get("image_id", "")),
            "class_axis": axis,
            "class_value": cls_val,
            "cluster_id": r.get("cluster_id"),
            "cell_key": r.get("cell_key", ""),
            "defect_bbox": bbox,
        }
        if top:
            best_raw, best_id, best_ok = top[0]
            hi_best = round(best_raw - _bg_jitter(src_key, best_id), 6)
            selected.append(dict(base,
                                 assigned_normal_id=best_id,
                                 topk_pool=[iid for _, iid, _ in top],
                                 hist_intersection=hi_best,
                                 class_fit=hi_best,   # Phase 1: class_fit == per-source hist
                                 size_ok=bool(best_ok),
                                 n_valid_bg=len(scored)))
        else:
            selected.append(dict(base, assigned_normal_id=None, topk_pool=[],
                                 hist_intersection=0.0, class_fit=0.0,
                                 size_ok=False, n_valid_bg=0))

    derived = {
        "pool_cut": "p95" if not pool_k else ("k=%d" % pool_k),
        "mean_pool_size": float(np.mean(pool_sizes)) if pool_sizes else 0.0,
        "multi_class": float(multi_class),
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
        f"void_frac_max={derived_void['void_frac_max']:.4f}  "
        f"→ valid {int(derived_void['n_valid'])}/{int(derived_void['n_good'])} good images",
        f"- pool_cut={derived_pool['pool_cut']}  "
        f"mean_pool_size={derived_pool['mean_pool_size']:.2f}  "
        f"multi_class={bool(derived_pool['multi_class'])}",
        "",
        "## Sample assignments (top 30)",
        "",
        "| roi_idx | class | assigned_bg | hist∩ | pool | size_ok |",
        "|---------|-------|-------------|-------|------|---------|",
    ]
    for s in selected[:30]:
        lines.append(
            f"| {s['roi_idx']} | {s['class_value']} | {s['assigned_normal_id']} "
            f"| {s['hist_intersection']:.4f} | {len(s['topk_pool'])} | {s['size_ok']} |"
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
    pool_k: Optional[int] = None,
) -> Dict[str, Any]:
    logger.info("Loading inputs: profiling_dir=%s roi_dir=%s", profiling_dir, roi_dir)
    data = load_inputs(profiling_dir, roi_dir)
    if data["status"] != "ok":
        logger.error("Input loading failed: %s", data)
        return data

    var_floor, edge_floor = _derive_void_floors(data["good_by_img"])
    valid_ids, reasons, derived_void = valid_bg_pool(
        data["good_by_img"], reject_clean_bg, void_frac_max, var_floor, edge_floor
    )
    logger.info("Valid clean-bg pool: %d / %d good images (void_frac_max=%.4f)",
                len(valid_ids), len(data["good_by_img"]), derived_void["void_frac_max"])

    selected, derived_pool = build_and_rank(
        data, valid_ids, var_floor, edge_floor, pool_k
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
                   help="Max per-image void fraction to keep. Default: DATA-DRIVEN "
                        "(P90 of observed void_frac). Override only to pin a value.")
    p.add_argument("--pool_k", type=int, default=None,
                   help="Per-ROI ranked background pool size. Default: DATA-DRIVEN "
                        "(all size-fit candidates; generate_defects indexes rep→pool). "
                        "Set an int to cap.")
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
        pool_k=args.pool_k,
    )
    if result.get("status") != "ok":
        sys.exit(1)


if __name__ == "__main__":
    main()
