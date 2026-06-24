#!/usr/bin/env python3
"""
AROMA Step 3 — ROI Selection

Scores every (defect image × context bin) candidate using:
    ROI_score = 0.4 × P(M_i) + 0.4 × P(C_j) + 0.2 × Deficit(M_i, C_j)

Then selects ROIs via one of four sampling strategies:
    deficit_aware   Top-K Deficit Quantile oversampling (default)
    top_k           Highest ROI_score first
    weighted        Probability-weighted random draw
    random          Uniform random draw (baseline for Exp 2)

Usage (Colab):
    !python $AROMA_SCRIPTS/roi_selection.py \
        --profiling_dir      $AROMA_OUT/profiling/isp_LSM_1 \
        --prompts_dir        $AROMA_OUT/prompts/isp_LSM_1 \
        --sampling_strategy  deficit_aware \
        --top_k              200 \
        --output_dir         $AROMA_OUT/roi/isp_LSM_1

Outputs (written to --output_dir):
    roi_candidates.json    all scored candidates
    roi_selected.json      candidates chosen by the sampling strategy
    roi_summary.md         human-readable table
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("aroma.roi")


# ---------------------------------------------------------------------------
# I/O bootstrap
# ---------------------------------------------------------------------------

def _bootstrap_aroma_ref() -> str:
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
# ROI score weights (matches AROMA-sharpened-spec.md)
# ---------------------------------------------------------------------------

_W_MORPH   = 0.4
_W_CONTEXT = 0.4
_W_DEFICIT = 0.2

assert abs(_W_MORPH + _W_CONTEXT + _W_DEFICIT - 1.0) < 1e-9, "ROI weights must sum to 1"


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_inputs(profiling_dir: str, prompts_dir: str) -> Dict[str, Any]:
    """
    Load all Step 3 inputs.

    Required files in profiling_dir:
        morphology_features.csv, morphology_clusters.json,
        compatibility_matrix.json, deficit_analysis.json

    Required in prompts_dir:
        prompts.json

    Returns dict with status="ok" on success.
    """
    pd = Path(profiling_dir)
    rd = Path(prompts_dir)

    required = [
        pd / "morphology_features.csv",
        pd / "morphology_clusters.json",
        pd / "compatibility_matrix.json",
        pd / "deficit_analysis.json",
        rd / "prompts.json",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        return {"status": "missing_inputs", "missing": missing}

    morph_rows = _read_csv_rows(pd / "morphology_features.csv")
    clusters   = load_json(str(pd / "morphology_clusters.json"))
    compat     = load_json(str(pd / "compatibility_matrix.json"))
    deficit    = load_json(str(pd / "deficit_analysis.json"))
    prompts    = load_json(str(rd / "prompts.json"))

    return {
        "status":          "ok",
        "morph_rows":      morph_rows,
        "clusters":        clusters,
        "compat_matrix":   compat,
        "deficit_analysis": deficit,
        "prompts":         prompts,
    }


# ---------------------------------------------------------------------------
# ROI scoring
# ---------------------------------------------------------------------------

def score_roi(
    morph_prior: float,
    ctx_prior: float,
    deficit: float,
) -> float:
    """
    ROI_score = 0.4 × P(M_i) + 0.4 × P(C_j) + 0.2 × Deficit(M_i, C_j)

    All inputs clamped to [0, 1].
    """
    mp = max(0.0, min(1.0, float(morph_prior)))
    cp = max(0.0, min(1.0, float(ctx_prior)))
    d  = max(0.0, min(1.0, float(deficit)))
    score = _W_MORPH * mp + _W_CONTEXT * cp + _W_DEFICIT * d
    return float(np.clip(score, 0.0, 1.0))


def _parse_bbox(bbox_str: str) -> Any:
    """Parse a 'x,y,w,h' comma-separated bbox string → [int, int, int, int].

    Returns None for empty strings or any malformed input (wrong field count,
    non-integer tokens) so downstream synthesis can cleanly fall back to the
    legacy ellipse path.
    """
    if not bbox_str:
        return None
    parts = str(bbox_str).split(",")
    if len(parts) != 4:
        return None
    try:
        return [int(p.strip()) for p in parts]
    except (ValueError, TypeError):
        return None


def build_candidates(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Score all (morph_row × context_bin) candidates.

    Each morphology row is crossed with every occupied context bin in the
    compatibility matrix (cluster_row.items()), producing N candidates per
    image where N = number of occupied bins for that cluster.

    Each candidate:
        image_id, image_path, cluster_id, cell_key,
        roi_score, morph_prior, ctx_prior, deficit,
        prompt, morph_label, ctx_label
    """
    morph_rows    = data["morph_rows"]
    clusters_json = data["clusters"]
    compat_json   = data["compat_matrix"]
    deficit_json  = data["deficit_analysis"]
    prompts       = data["prompts"]

    # --- build lookup: image_id → cluster_id ---
    assignments: Dict[str, int] = {
        str(k): int(v)
        for k, v in clusters_json.get("cluster_assignments", {}).items()
    }
    cluster_labels: Dict[int, str] = {
        int(c["cluster_id"]): c.get("label", "")
        for c in clusters_json.get("clusters", [])
    }

    # --- compatibility matrix ---
    matrix: Dict[str, Dict[str, float]] = compat_json.get("matrix", {})

    # --- cluster priors and deficit ---
    cluster_priors: Dict[str, float] = {
        cid: float(info.get("prior", 0.0))
        for cid, info in deficit_json.items()
    }
    deficit_rows: Dict[str, Dict[str, float]] = {
        cid: info.get("deficit", {})
        for cid, info in deficit_json.items()
    }

    candidates: List[Dict[str, Any]] = []

    for row in morph_rows:
        image_id   = str(row.get("image_id", ""))
        image_path = str(row.get("image_path", ""))
        defect_bbox = _parse_bbox(row.get("defect_bbox", ""))
        defect_mask_path = str(row.get("defect_mask_path", ""))

        cluster_id = assignments.get(image_id)
        if cluster_id is None:
            continue

        cid_str     = str(cluster_id)
        morph_prior = cluster_priors.get(cid_str, 0.0)
        morph_label = cluster_labels.get(cluster_id, "")
        cluster_row = matrix.get(cid_str, {})

        # Iterate occupied context bins from the compatibility matrix.
        # This correctly pairs each image with every (cluster, cell_key)
        # combination that has observed context coverage.
        if not cluster_row:
            logger.warning("cluster %s has no context bins — using 'none' fallback for image %s",
                           cid_str, image_id)
        bin_iter = cluster_row.items() if cluster_row else {"none": 0.0}.items()
        for cell_key, ctx_prior in bin_iter:
            deficit      = float(deficit_rows.get(cid_str, {}).get(cell_key, 0.0))
            roi_score    = score_roi(morph_prior, float(ctx_prior), deficit)
            prompt_key   = f"{cluster_id}_{cell_key}"
            prompt_entry = prompts.get(prompt_key, {})

            candidates.append({
                "image_id":    image_id,
                "image_path":  image_path,
                "cluster_id":  cluster_id,
                "cell_key":    cell_key,
                "class_key":   str(row.get("defect_type") or "_"),
                "roi_score":   round(roi_score, 6),
                "morph_prior": round(morph_prior, 6),
                "ctx_prior":   round(float(ctx_prior), 6),
                "deficit":     round(deficit, 6),
                "prompt":      prompt_entry.get("prompt", ""),
                "morph_label": morph_label,
                "ctx_label":   prompt_entry.get("context_descriptor", ""),
                "defect_bbox": defect_bbox,
                "defect_mask_path": defect_mask_path,
            })

    logger.info("Scored %d ROI candidates from %d morph rows", len(candidates), len(morph_rows))
    return candidates


# ---------------------------------------------------------------------------
# Sampling strategies
# ---------------------------------------------------------------------------

def moderated_score(c: Dict[str, Any], rarity_temp: float = 1.0) -> float:
    """
    Ordering key for ROI selection.

    rarity_temp == 1.0 (default) → returns the raw ``roi_score`` verbatim, so
    every ordering decision is byte-identical to the pre-change behaviour.

    rarity_temp != 1.0 → recompute the score with only the Deficit term raised
    to ``rarity_temp`` (the morph / ctx prior anchors are kept as-is). This
    compresses or amplifies the deficit contribution without touching the raw
    ``roi_score`` stored on the candidate (provenance preserved for JSON).
    """
    if rarity_temp == 1.0:
        return float(c["roi_score"])
    mp = max(0.0, min(1.0, float(c.get("morph_prior", 0.0))))
    cp = max(0.0, min(1.0, float(c.get("ctx_prior", 0.0))))
    d  = max(0.0, min(1.0, float(c.get("deficit", 0.0))))
    score = _W_MORPH * mp + _W_CONTEXT * cp + _W_DEFICIT * (d ** rarity_temp)
    return float(np.clip(score, 0.0, 1.0))


def _pair_aware_allocation(
    candidates: List[Dict[str, Any]],
    top_k: int,
    per_pair_cap_frac: float | None = None,
    rarity_temp: float = 1.0,
    pair_cap: int | None = None,
) -> List[Dict[str, Any]]:
    """
    Two-phase ROI selection: Coverage-first + Quality-second.

    Phase 1 — Pair-aware allocation (Coverage + Deficit):
        1. Group candidates by (cluster_id, cell_key) pair.
           Compute PairDeficit = mean(deficit) for each pair.
        2. Assign base quota=1 to every pair (coverage guarantee).
        3. Distribute remaining quota (top_k - n_pairs) proportional to
           PairDeficit via Hamilton / Largest-Remainder method.
           Pairs with deficit=0 receive no extra quota beyond the base.
        3b. (cap-enabled only) Cap any single pair's quota at the effective
           cap (``pair_cap`` when supplied — an ABSOLUTE global int — else
           ceil(per_pair_cap_frac * top_k)) and round-robin the freed slots
           to under-saturated pairs by PairDeficit, breaking monoculture.
           Skipped entirely when both per_pair_cap_frac and pair_cap are None.
        4. Within each pair, select candidates by moderated_score descending.

    Phase 2 — Quality backfill:
        If Phase 1 selected fewer than top_k candidates (slack from
        under-populated pairs), fill remaining slots from the unselected
        pool ranked by global moderated_score. When a cap is active the
        backfill is cap-aware: it skips candidates whose pair is already at
        the effective cap (running per-pair counter), so an evicted pair's
        high-score candidates cannot re-flood the freed slots.

    Fallback: if top_k <= n_pairs, return moderated_score top-k globally
    (cap-aware when a cap is active).

    ``pair_cap`` (absolute int) overrides the per_pair_cap_frac-derived cap so
    the stratified caller can enforce a single GLOBAL ceiling rather than a
    per-bucket-local one. When None the cap (if any) is derived from
    per_pair_cap_frac * top_k as before.

    moderated_score collapses to the raw roi_score when rarity_temp == 1.0
    (default), so the default path is byte-identical to the prior behaviour.
    """
    if not candidates:
        return []

    # Effective per-pair cap: an absolute global int (pair_cap) takes
    # precedence; otherwise derive from per_pair_cap_frac * top_k. None ⇒
    # no cap ⇒ every cap branch below is skipped (byte-identical path).
    if pair_cap is not None:
        eff_cap: int | None = max(1, int(pair_cap))
    elif per_pair_cap_frac is not None:
        eff_cap = max(1, math.ceil(per_pair_cap_frac * top_k))
    else:
        eff_cap = None

    # --- Phase 1: pair-aware coverage + deficit allocation ---
    pair_groups: Dict[Any, List[Dict[str, Any]]] = defaultdict(list)
    for c in candidates:
        pair_groups[(c["cluster_id"], c["cell_key"])].append(c)

    pairs = list(pair_groups.keys())
    n_pairs = len(pairs)

    if top_k <= n_pairs:
        ranked = sorted(
            candidates,
            key=lambda c: moderated_score(c, rarity_temp),
            reverse=True,
        )
        if eff_cap is None:
            return ranked[:top_k]
        # cap-aware top-k: skip a candidate once its pair hits eff_cap
        picked: List[Dict[str, Any]] = []
        pair_count: Dict[Any, int] = defaultdict(int)
        for c in ranked:
            if len(picked) >= top_k:
                break
            key = (c["cluster_id"], c["cell_key"])
            if pair_count[key] >= eff_cap:
                continue
            picked.append(c)
            pair_count[key] += 1
        return picked

    pair_deficit = {
        p: float(np.mean([c["deficit"] for c in pair_groups[p]]))
        for p in pairs
    }

    quotas: Dict[Any, int] = {p: 1 for p in pairs}
    remaining = top_k - n_pairs

    deficit_pairs = {p: d for p, d in pair_deficit.items() if d > 0}
    total_deficit = sum(deficit_pairs.values())

    if total_deficit == 0:
        weight: Dict[Any, float] = {p: 1.0 for p in pairs}
        total_w = float(n_pairs)
    else:
        weight = dict(deficit_pairs)
        total_w = total_deficit

    ideal = {p: remaining * w / total_w for p, w in weight.items()}
    floor_q = {p: math.floor(v) for p, v in ideal.items()}
    shortfall = remaining - sum(floor_q.values())
    remainders = {p: ideal[p] - floor_q[p] for p in weight}
    for p in sorted(remainders, key=remainders.__getitem__, reverse=True)[:shortfall]:
        floor_q[p] += 1

    for p, extra in floor_q.items():
        quotas[p] += extra

    # --- Phase 1b: per-pair cap + redistribute (Fix1, opt-in) ---
    # Caps the slots any single (cluster_id, cell_key) pair can claim at the
    # effective ceiling (eff_cap — class-agnostic, GLOBAL when threaded by the
    # stratified caller), preventing one pair from monopolising the selection
    # (monoculture). Freed slots are handed, one at a time, to under-saturated
    # pairs ordered by PairDeficit. Any slots that cannot be re-placed are
    # absorbed by the Phase-2 backfill, so total selected stays <= top_k.
    # Skipped when no cap is active.
    if eff_cap is not None:
        cap = eff_cap
        freed = 0
        for p in pairs:
            avail = len(pair_groups[p])
            over = quotas[p] - cap
            # only the slots that the pair could actually have filled are freed
            redeemable = min(quotas[p], avail) - min(cap, avail)
            if over > 0 and redeemable > 0:
                quotas[p] = cap
                freed += redeemable
        if freed > 0:
            # round-robin freed slots to under-saturated pairs (PairDeficit desc),
            # each up to its own cap and candidate availability.
            order = sorted(pairs, key=lambda p: pair_deficit[p], reverse=True)
            progressed = True
            while freed > 0 and progressed:
                progressed = False
                for p in order:
                    if freed <= 0:
                        break
                    if quotas[p] < cap and quotas[p] < len(pair_groups[p]):
                        quotas[p] += 1
                        freed -= 1
                        progressed = True

    selected: List[Dict[str, Any]] = []
    for p, quota in quotas.items():
        top_in_pair = sorted(
            pair_groups[p],
            key=lambda c: moderated_score(c, rarity_temp),
            reverse=True,
        )
        selected.extend(top_in_pair[:quota])

    # --- Phase 2: quality backfill for slack slots ---
    slack = top_k - len(selected)
    if slack > 0:
        selected_ids = {id(c) for c in selected}
        rest = sorted(
            (c for c in candidates if id(c) not in selected_ids),
            key=lambda c: moderated_score(c, rarity_temp),
            reverse=True,
        )
        if eff_cap is None:
            backfill = rest[:slack]
        else:
            # cap-aware backfill: track the running per-pair count of what is
            # already selected, then skip any candidate whose pair is at
            # eff_cap. This stops an evicted (capped) pair's high-score
            # candidates from re-flooding the freed slots (the original bug).
            pair_count: Dict[Any, int] = defaultdict(int)
            for c in selected:
                pair_count[(c["cluster_id"], c["cell_key"])] += 1
            backfill = []
            for c in rest:
                if len(backfill) >= slack:
                    break
                key = (c["cluster_id"], c["cell_key"])
                if pair_count[key] >= eff_cap:
                    continue
                backfill.append(c)
                pair_count[key] += 1
        selected += backfill
        logger.info(
            "pair_aware_allocation: phase2 backfill %d / %d slack slots",
            len(backfill), slack,
        )

    return selected


def _stratified_pair_aware(
    candidates: List[Dict[str, Any]],
    top_k: int,
    per_pair_cap_frac: float | None = None,
    rarity_temp: float = 1.0,
) -> List[Dict[str, Any]]:
    """
    Class-stratified wrapper around :func:`_pair_aware_allocation` (Fix2).

    Buckets candidates by ``class_key`` and gives every class a symmetric
    floor (``top_k // K``) so no class can be starved to zero by another
    class winning the shared (cluster_id, cell_key) competition. The leftover
    budget (``top_k - K * floor``) is handed out by per-class deficit-mass via
    the same Hamilton / Largest-Remainder method used per-pair. Per-class
    quotas are clamped to each class's candidate count and the resulting
    surplus is redistributed to classes that can still absorb it. Each bucket
    is then allocated independently with :func:`_pair_aware_allocation`
    (preserving intra-class pair coverage + deficit awareness and the optional
    per-pair cap), the buckets are concatenated, and a global Phase-2 backfill
    tops the result up to exactly ``top_k``.

    With a single class (K <= 1) — i.e. every single-class dataset — this
    returns ``_pair_aware_allocation(...)`` verbatim, keeping that path
    byte-identical to the pre-change behaviour.

    Thesis / ablation caveat (cross-class layer is starvation-first, not
    deficit-proportional). The cross-class budget is dominated by the symmetric
    floor ``top_k // K``; only the leftover ``top_k % K`` slots (< K) flow
    through the per-class deficit-mass largest-remainder block below. When
    ``top_k`` is a multiple of ``K`` (e.g. top_k=200, K=4 → remainder 0) the
    deficit-mass branch is inert and every class receives an equal floor — the
    cross-class split is purely uniform. This is a deliberate anti-starvation
    design choice (no class can be driven to zero by another winning the shared
    pair competition), NOT class-level deficit-proportional oversampling.
    AROMA's deficit-proportional thesis is carried *within* each class by
    :func:`_pair_aware_allocation` (genuine PairDeficit Hamilton allocation,
    intact); at the cross-class granularity the shipped config trades
    deficit-proportionality for starvation-resistance. Any publication claim
    that the multi-class selection is "deficit-aware across classes" must be
    qualified accordingly, or explored via an ablation that shrinks the floor so
    a non-trivial remainder reaches the deficit-mass weighting.
    """
    if not candidates:
        return []

    buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for c in candidates:
        buckets[str(c.get("class_key") or "_")].append(c)

    keys = list(buckets.keys())
    K = len(keys)
    if K <= 1:
        return _pair_aware_allocation(candidates, top_k, per_pair_cap_frac, rarity_temp)

    # GLOBAL per-pair cap, derived ONCE from the REAL (top-level) top_k — NOT
    # from a per-bucket quota. Threaded into each bucket as an absolute int so
    # no single (cluster_id, cell_key) pair can claim more than pair_cap across
    # the whole concatenated selection. None ⇒ cap disabled (byte-identical).
    pair_cap = (
        max(1, math.ceil(per_pair_cap_frac * top_k))
        if per_pair_cap_frac is not None
        else None
    )

    # --- per-class quota: symmetric floor + deficit-mass largest-remainder ---
    floor = top_k // K
    quotas: Dict[str, int] = {k: floor for k in keys}
    remaining = top_k - floor * K

    if remaining > 0:
        class_deficit = {
            k: float(sum(c["deficit"] for c in buckets[k])) for k in keys
        }
        total_deficit = sum(class_deficit.values())
        if total_deficit == 0:
            weight: Dict[str, float] = {k: 1.0 for k in keys}
            total_w = float(K)
        else:
            weight = {k: d for k, d in class_deficit.items() if d > 0}
            total_w = total_deficit
        ideal = {k: remaining * w / total_w for k, w in weight.items()}
        floor_q = {k: math.floor(v) for k, v in ideal.items()}
        shortfall = remaining - sum(floor_q.values())
        remainders = {k: ideal[k] - floor_q[k] for k in weight}
        for k in sorted(remainders, key=remainders.__getitem__, reverse=True)[:shortfall]:
            floor_q[k] += 1
        for k, extra in floor_q.items():
            quotas[k] += extra

    # --- clamp to availability, then redistribute surplus to classes with room ---
    surplus = 0
    for k in keys:
        avail = len(buckets[k])
        if quotas[k] > avail:
            surplus += quotas[k] - avail
            quotas[k] = avail
    if surplus > 0:
        # hand surplus to under-filled classes (deficit-mass desc), one at a time
        order = sorted(
            keys,
            key=lambda k: float(sum(c["deficit"] for c in buckets[k])),
            reverse=True,
        )
        progressed = True
        while surplus > 0 and progressed:
            progressed = False
            for k in order:
                if surplus <= 0:
                    break
                if quotas[k] < len(buckets[k]):
                    quotas[k] += 1
                    surplus -= 1
                    progressed = True

    # --- per-class allocation, then concat ---
    # Per-class floor to protect during cap eviction: the symmetric floor
    # (top_k // K), clamped to each class's actual selected count (a class with
    # fewer than `floor` candidates can never be pushed below what it holds).
    class_floor_target: Dict[str, int] = {k: min(floor, quotas[k]) for k in keys}

    # --- per-class allocation, then concat ---
    # IMPORTANT: cap enforcement is moved ENTIRELY to the global post-concat
    # pass below (one of the two options the constraints explicitly permit).
    # The per-bucket calls therefore run WITHOUT the cap (pair_cap=None and
    # per_pair_cap_frac=None) so each class fills to its full quota q FIRST and
    # the class floor is always met before any cap trimming. Threading the cap
    # into the buckets would let a pair-diversity-starved class return fewer
    # than q (n_distinct_pairs * cap < q), breaching its floor before the
    # post-pass ever runs. With cap enforcement global-only, the only place a
    # pair can exceed pair_cap is the post-concat union of buckets, which the
    # eviction + class-aware backfill below resolves.
    selected: List[Dict[str, Any]] = []
    sel_class: Dict[int, str] = {}  # id(candidate) -> owning class bucket
    for k in keys:
        q = quotas[k]
        if q <= 0:
            continue
        picked = _pair_aware_allocation(buckets[k], q, None, rarity_temp)
        for c in picked:
            sel_class[id(c)] = k
        selected.extend(picked)

    if pair_cap is not None:
        # --- POST-CONCAT cap eviction (Fix1 global invariant) ---
        # The same (cluster_id, cell_key) pair can be filled inside multiple
        # class buckets, so a pair within cap per-bucket can exceed pair_cap
        # once concatenated (and, with cap-free buckets, even a single bucket's
        # dominating pair now exceeds it). Trim each over-cap pair down to
        # pair_cap by evicting its LOWEST-moderated_score members first.
        #
        # Unlike the previous (deadlocking) version, eviction is allowed to dip
        # a class BELOW its floor temporarily — the class-aware backfill that
        # follows restores every depleted class up to its floor from that
        # class's OWN non-over-cap candidates before any global fill. This is
        # what breaks the top_k % K == 0 deadlock: when every class sits exactly
        # at its floor, the old pre-backfill guard refused all trimming; here we
        # trim first and let the per-class refill recover the floor with other
        # pairs in the same class.
        by_pair: Dict[Any, List[Dict[str, Any]]] = defaultdict(list)
        for c in selected:
            by_pair[(c["cluster_id"], c["cell_key"])].append(c)

        evicted_ids: set = set()
        for key, members in by_pair.items():
            if len(members) <= pair_cap:
                continue
            # lowest moderated_score evicted first; stable tie-break by id()
            members_sorted = sorted(
                members,
                key=lambda c: (moderated_score(c, rarity_temp), -id(c)),
            )
            for c in members_sorted[: len(members) - pair_cap]:
                evicted_ids.add(id(c))

        if evicted_ids:
            selected = [c for c in selected if id(c) not in evicted_ids]

        # --- class-aware refill: restore each class to its floor first ---
        # For every class depleted below class_floor_target by eviction, refill
        # from that class's OWN unselected candidates, cap-aware (never push a
        # pair back over pair_cap), ordered by moderated_score. This guarantees
        # the floor survives the cap (the 50/50/50/50 invariant) using DIFFERENT
        # pairs than the over-cap one. A class that genuinely lacks enough
        # distinct pairs to both honour the cap AND reach its floor will fall
        # short here; that is logged as a cap-vs-floor conflict (no silent
        # violation), and the floor is preferred only insofar as its own pool
        # allows without breaking the cap.
        selected_ids = {id(c) for c in selected}
        pair_count: Dict[Any, int] = defaultdict(int)
        class_count: Dict[str, int] = defaultdict(int)
        for c in selected:
            pair_count[(c["cluster_id"], c["cell_key"])] += 1
            class_count[sel_class[id(c)]] += 1

        for k in keys:
            need = class_floor_target[k] - class_count[k]
            if need <= 0:
                continue
            pool = sorted(
                (c for c in buckets[k] if id(c) not in selected_ids),
                key=lambda c: moderated_score(c, rarity_temp),
                reverse=True,
            )
            added = 0
            for c in pool:
                if added >= need:
                    break
                pkey = (c["cluster_id"], c["cell_key"])
                if pair_count[pkey] >= pair_cap:
                    continue
                selected.append(c)
                sel_class[id(c)] = k
                selected_ids.add(id(c))
                pair_count[pkey] += 1
                class_count[k] += 1
                added += 1
            if class_count[k] < class_floor_target[k]:
                logger.info(
                    "stratified_pair_aware: class '%s' below floor (%d<%d) — "
                    "insufficient pair diversity to satisfy both cap (%d) and "
                    "floor; cap honoured, floor not fully met",
                    k, class_count[k], class_floor_target[k], pair_cap,
                )

    # --- global Phase-2 backfill to exactly top_k (cap-aware when capping) ---
    slack = top_k - len(selected)
    if slack > 0:
        selected_ids = {id(c) for c in selected}
        rest = sorted(
            (c for c in candidates if id(c) not in selected_ids),
            key=lambda c: moderated_score(c, rarity_temp),
            reverse=True,
        )
        if pair_cap is None:
            backfill = rest[:slack]
        else:
            # cap-aware: never let backfill re-flood a pair back over pair_cap.
            pair_count = defaultdict(int)
            for c in selected:
                pair_count[(c["cluster_id"], c["cell_key"])] += 1
            backfill = []
            for c in rest:
                if len(backfill) >= slack:
                    break
                key = (c["cluster_id"], c["cell_key"])
                if pair_count[key] >= pair_cap:
                    continue
                backfill.append(c)
                pair_count[key] += 1
        selected += backfill
        logger.info(
            "stratified_pair_aware: %d classes, global backfill %d / %d slack slots",
            K, len(backfill), slack,
        )

    return selected


def select_rois(
    candidates: List[Dict[str, Any]],
    strategy: str = "deficit_aware",
    top_k: int = 200,
    seed: int = 42,
    class_floor: bool = False,
    per_pair_cap_frac: float | None = None,
    rarity_temp: float = 1.0,
) -> List[Dict[str, Any]]:
    """
    Select top_k ROIs from candidates using the given strategy.

    Strategies:
        deficit_aware   Top-K Deficit Quantile oversampling (default)
        top_k           Highest roi_score first
        weighted        Probability-weighted random draw
        random          Uniform random draw (baseline for Exp 2)

    Multi-class options (deficit_aware only; default no-op):
        class_floor         When True and the candidate pool spans >1 class
                            (K>1), route through _stratified_pair_aware so each
                            class gets a symmetric floor. Single-class / K<=1
                            pools fall through to _pair_aware_allocation
                            unchanged (byte-identical).
        per_pair_cap_frac   Per-pair quota cap fraction (None = no cap).
        rarity_temp         Deficit-term temperature for the ordering key
                            (1.0 = raw roi_score, byte-identical).

    The random / top_k / weighted branches ignore the multi-class options.
    """
    if not candidates:
        return []

    top_k = max(1, min(top_k, len(candidates)))

    if strategy == "top_k":
        return sorted(candidates, key=lambda c: c["roi_score"], reverse=True)[:top_k]

    if strategy == "random":
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(candidates), size=top_k, replace=False)
        return [candidates[i] for i in sorted(indices.tolist())]

    if strategy == "weighted":
        scores = np.array([c["roi_score"] for c in candidates], dtype=np.float64)
        total = scores.sum()
        if total < 1e-12:
            return candidates[:top_k]
        probs = scores / total
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(candidates), size=top_k, replace=False, p=probs)
        return [candidates[i] for i in sorted(indices.tolist())]

    # deficit_aware (default)
    if class_floor:
        n_classes = len({str(c.get("class_key") or "_") for c in candidates})
        if n_classes > 1:
            return _stratified_pair_aware(
                candidates, top_k, per_pair_cap_frac, rarity_temp
            )
    return _pair_aware_allocation(candidates, top_k, per_pair_cap_frac, rarity_temp)


# ---------------------------------------------------------------------------
# Summary markdown
# ---------------------------------------------------------------------------

def _build_summary(
    candidates: List[Dict[str, Any]],
    selected: List[Dict[str, Any]],
    strategy: str,
) -> str:
    lines = [
        "# AROMA Step 3 — ROI Selection Summary",
        "",
        f"**Strategy**: `{strategy}`",
        f"**Total candidates**: {len(candidates)}",
        f"**Selected**: {len(selected)}",
        "",
        "## Selected ROIs (top 50)",
        "",
        "| image_id | cluster | cell_key | roi_score | deficit | prompt |",
        "|----------|---------|----------|-----------|---------|--------|",
    ]
    for c in selected[:50]:
        lines.append(
            f"| {c['image_id']} | {c['cluster_id']} ({c['morph_label']}) "
            f"| {c['cell_key']} | {c['roi_score']:.4f} | {c['deficit']:.4f} "
            f"| {c['prompt'][:60]}... |"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run(
    profiling_dir: str,
    prompts_dir: str,
    output_dir: str,
    strategy: str = "deficit_aware",
    top_k: int = 200,
    seed: int = 42,
    class_floor: bool = False,
    per_pair_cap_frac: float | None = None,
    rarity_temp: float = 1.0,
) -> Dict[str, Any]:
    """
    Full Step 3 pipeline: load → score → select → save.

    Returns dict with 'candidates' and 'selected' keys on success,
    or a status-only dict on failure.

    class_floor / per_pair_cap_frac / rarity_temp are multi-class options
    (default no-op) forwarded to select_rois; see its docstring.
    """
    logger.info("Loading inputs: profiling_dir=%s prompts_dir=%s", profiling_dir, prompts_dir)
    data = load_inputs(profiling_dir, prompts_dir)
    if data["status"] != "ok":
        logger.error("Input loading failed: %s", data)
        return data

    candidates = build_candidates(data)
    selected   = select_rois(
        candidates,
        strategy=strategy,
        top_k=top_k,
        seed=seed,
        class_floor=class_floor,
        per_pair_cap_frac=per_pair_cap_frac,
        rarity_temp=rarity_temp,
    )

    logger.info(
        "Strategy=%s  top_k=%d  candidates=%d  selected=%d",
        strategy, top_k, len(candidates), len(selected),
    )

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    save_json(candidates, str(out / "roi_candidates.json"))
    save_json(selected,   str(out / "roi_selected.json"))

    summary_md = _build_summary(candidates, selected, strategy)
    (out / "roi_summary.md").write_text(summary_md, encoding="utf-8")

    logger.info("Saved roi_candidates.json (%d), roi_selected.json (%d) → %s",
                len(candidates), len(selected), out)

    return {
        "status":     "ok",
        "candidates": candidates,
        "selected":   selected,
        "strategy":   strategy,
        "top_k":      top_k,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="AROMA Step 3 — ROI Selection"
    )
    p.add_argument("--profiling_dir", required=True,
                   help="Phase 0 output directory")
    p.add_argument("--prompts_dir",   required=True,
                   help="Step 2 output directory (prompts.json)")
    p.add_argument("--output_dir",    required=True,
                   help="Directory to write roi_candidates.json and roi_selected.json")
    p.add_argument("--sampling_strategy", default="deficit_aware",
                   choices=["deficit_aware", "top_k", "weighted", "random"],
                   help="ROI sampling strategy (default: deficit_aware)")
    p.add_argument("--top_k", type=int, default=200,
                   help="Number of ROIs to select (default: 200)")
    p.add_argument("--seed",  type=int, default=42,
                   help="Random seed for weighted sampling (default: 42)")
    p.add_argument("--class_mode", default="single",
                   choices=["single", "multi"],
                   help="Class mode (default: single). 'multi' is informational; "
                        "the actual stratification is gated by --class_floor.")
    p.add_argument("--nc", type=int, default=None,
                   help="Number of classes (optional, informational). If >1 it "
                        "implies a multi-class run; --class_floor still gates "
                        "the stratified allocation.")
    p.add_argument("--class_floor", action="store_true",
                   help="Enable per-class symmetric floor allocation (Fix2). "
                        "No-op on single-class (K<=1) pools. Default: off.")
    p.add_argument("--per_pair_cap_frac", type=float, default=None,
                   help="Per-pair quota cap as a fraction of top_k (Fix1). "
                        "None (default) disables the cap.")
    p.add_argument("--rarity_temp", type=float, default=1.0,
                   help="Deficit-term temperature for the ordering key (Fix3). "
                        "1.0 (default) = raw roi_score, byte-identical.")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = _parse_args(argv)
    result = run(
        profiling_dir=args.profiling_dir,
        prompts_dir=args.prompts_dir,
        output_dir=args.output_dir,
        strategy=args.sampling_strategy,
        top_k=args.top_k,
        seed=args.seed,
        class_floor=args.class_floor,
        per_pair_cap_frac=args.per_pair_cap_frac,
        rarity_temp=args.rarity_temp,
    )
    if result.get("status") != "ok":
        sys.exit(1)


if __name__ == "__main__":
    main()
