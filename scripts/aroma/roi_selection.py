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
import hashlib
import json
import logging
import math
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

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
    # Always add the repo root derived from this file's location so the
    # `utils.*` package (utils/io.py, utils/suitability.py,
    # utils/defect_characterization.py — sibling of scripts/) resolves
    # regardless of the AROMA_REF env or the runtime cwd. This file lives at
    # <repo>/scripts/aroma/roi_selection.py, so parents[2] is <repo>.
    # Without this, Colab clones (where AROMA_REF points elsewhere or is unset
    # and the default Windows path does not exist) raised 'No module named
    # utils', silently disabling the quality gate.
    try:
        repo_root = Path(__file__).resolve().parents[2]
        if repo_root.is_dir() and str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
    except (IndexError, OSError):
        pass

    # Preserve the existing AROMA_REF behavior (explicit override / default).
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
# Quality gate — subtype-matching proxy (opt-in pre-filter)
#
# Wires the EXISTING SuitabilityEvaluator.matching_score + DefectCharacterizer
# subtype classifier into ROI selection as a pre-filter that drops defect ROIs
# whose morphology is structurally unsuitable for the dataset background (e.g.
# compact_blob / irregular on a directional steel strip). This is ORTHOGONAL to
# the source-diversity (Fix4) / stratification (Fix2) / per-pair-cap (Fix1)
# machinery, which balance *which* ROIs are picked but never exclude an
# intrinsically low-suitability ROI.
#
# The full hybrid suitability score is intentionally NOT used: continuity /
# stability / gram and a semantic background_type are absent from AROMA-native
# profiling output, so only the subtype↔background MATCHING term is meaningful.
# background_type is supplied per-dataset (severstal steel strips → directional).
# Deps are optional: if the imports fail the gate is a no-op (score 1.0).
# ---------------------------------------------------------------------------

try:
    from utils.suitability import SuitabilityEvaluator  # type: ignore[import]
    from utils.defect_characterization import DefectCharacterizer  # type: ignore[import]
    _SUIT_EVAL: Any = SuitabilityEvaluator()
    _DEFECT_CHAR: Any = DefectCharacterizer()
    _HAS_QUALITY = True
except (ImportError, ModuleNotFoundError) as _qexc:  # pragma: no cover - env dependent
    _SUIT_EVAL = None
    _DEFECT_CHAR = None
    _HAS_QUALITY = False
    logger.warning(
        "Quality gate dependencies unavailable (%s) — quality filter disabled "
        "(all candidates pass)", _qexc,
    )


def _opt_float(value: Any):
    """Parse to float, or None when missing/empty/unparseable/NaN.

    Distinguishing None from 0.0 matters: a missing morphology metric must NOT
    be coerced to 0.0 (which would misclassify the defect as 'irregular' and
    silently drop it); it is treated as quality-unknown instead.
    """
    if value is None or value == "":
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else f


def quality_proxy(linearity, solidity, aspect_ratio, background_type: str) -> tuple:
    """Return ``(defect_subtype, quality_score)`` via the subtype-matching proxy.

    ``quality_score = SuitabilityEvaluator.matching_score(subtype, background_type)``
    where ``subtype = DefectCharacterizer.classify_defect_subtype(metrics)``.
    Returns ``('general', 1.0)`` when deps are unavailable OR any morphology
    metric is missing (quality-unknown → pass the gate, never misclassify).
    """
    if not _HAS_QUALITY:
        return "general", 1.0
    lin = _opt_float(linearity)
    sol = _opt_float(solidity)
    ar  = _opt_float(aspect_ratio)
    if lin is None or sol is None or ar is None:
        return "general", 1.0
    metrics = {"linearity": lin, "solidity": sol, "aspect_ratio": ar}
    subtype = _DEFECT_CHAR.classify_defect_subtype(metrics)
    return subtype, float(_SUIT_EVAL.matching_score(subtype, background_type))


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


def build_candidates(
    data: Dict[str, Any],
    background_type: str = "directional",
) -> List[Dict[str, Any]]:
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

        # Quality proxy — per-image (depends only on morphology), computed once
        # per row and shared across that image's context-bin candidates.
        defect_subtype, quality_score = quality_proxy(
            row.get("linearity"),
            row.get("solidity"),
            row.get("aspect_ratio"),
            background_type,
        )

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
                "defect_subtype": defect_subtype,
                "quality_score":  round(quality_score, 6),
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


def apply_quality_gate(
    candidates: List[Dict[str, Any]],
    min_quality: float,
) -> List[Dict[str, Any]]:
    """Drop candidates whose subtype-matching ``quality_score`` is below
    ``min_quality``. Returns the surviving subset.

    ``min_quality <= 0`` (default) disables the gate — returns ``candidates``
    unchanged (byte-identical to legacy). The gate is purely a pre-filter; the
    downstream allocator (stratification / per-pair cap / source-diversity cap)
    is untouched and operates on the surviving pool.

    A class whose candidates ALL fail the gate is logged as a warning (never
    silently dropped).
    """
    if min_quality <= 0.0:
        logger.info(
            "Quality gate disabled (min_quality=%.3f) — all %d candidates pass "
            "(legacy behavior)", min_quality, len(candidates),
        )
        return candidates
    if not _HAS_QUALITY:
        logger.warning(
            "min_quality=%.3f requested but quality deps unavailable — gate "
            "skipped (all candidates pass)", min_quality,
        )
        return candidates

    def _q(c) -> float:
        qs = float(c.get("quality_score", 1.0))
        return 0.0 if math.isnan(qs) else qs  # NaN → worst-case, never silently pass

    passing = [c for c in candidates if _q(c) >= min_quality]

    all_by: Dict[str, int] = defaultdict(int)
    pass_by: Dict[str, int] = defaultdict(int)
    for c in candidates:
        all_by[str(c.get("class_key", "_"))] += 1
    for c in passing:
        pass_by[str(c.get("class_key", "_"))] += 1

    logger.info(
        "Quality gate: min_quality=%.3f → %d/%d candidates pass",
        min_quality, len(passing), len(candidates),
    )
    for ck in sorted(all_by):
        logger.info("  class=%-8s passing=%-6d / %-6d", ck, pass_by.get(ck, 0), all_by[ck])
        if pass_by.get(ck, 0) == 0:
            logger.warning(
                "  class=%s has ZERO quality-passing ROIs (min_quality=%.3f) — "
                "no synthetic data will be generated for this class",
                ck, min_quality,
            )
    return passing


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


# ---------------------------------------------------------------------------
# Source-image diversity helpers (Fix4)
#
# The candidate pool emits ONE candidate per (cluster, cell_key) bin for every
# source image, all carrying the SAME (image_path, defect_bbox). Because the
# ROI score is image-blind, every candidate sharing a (cluster, cell) pair ties
# on score, and Python's stable sort surfaces the SAME head image in every pair,
# collapsing thousands of distinct source defects down to a few dozen repeated
# crops. These helpers (1) break score ties with a deterministic per-source
# jitter so different pairs surface different images, and (2) expose a stable
# per-source key so the allocator can cap how many times any one (image, bbox)
# is selected. The cap is ON by default (img_diversity_cap=1); pass a large int
# (or None) to restore the legacy uncapped, no-jitter path for an ablation.
# ---------------------------------------------------------------------------

def _source_key(c: Dict[str, Any]) -> Tuple[str, str]:
    """Stable identity of the underlying defect crop: (image_path, bbox).

    Two candidates with the same (image_path, defect_bbox) draw from the SAME
    pixels — selecting both adds zero defect-appearance diversity. Keyed on the
    raw path + bbox (not image_id) so it is robust regardless of id provenance.
    """
    bbox = c.get("defect_bbox")
    bbox_repr = ",".join(str(int(v)) for v in bbox) if isinstance(bbox, (list, tuple)) else str(bbox)
    return (str(c.get("image_path", "")), bbox_repr)


def _img_jitter(c: Dict[str, Any]) -> float:
    """Deterministic, sub-score tie-break keyed on (image_path, bbox, cell_key).

    Returns a value in [0, ~1e-6) derived from a fixed hashlib digest so that
    (a) ties between candidates resolve to DIFFERENT source images across
    different pairs (the head rotates instead of always being the same CSV-order
    image), and (b) results stay reproducible across runs and across the
    random/aroma arms (no Python salted hash). Magnitude << score granularity
    (roi_score is rounded to 1e-6) so it never reorders genuinely distinct
    scores — it only resolves exact ties.
    """
    # Build a stable string from (image_path, bbox, cell_key) and hash it.
    img_path, bbox_repr = _source_key(c)
    h = hashlib.blake2b(
        ("%s\x1f%s\x1f%s" % (img_path, bbox_repr, str(c.get("cell_key", "")))).encode("utf-8"),
        digest_size=8,
    )
    # Map the 64-bit digest into [0, 1) then scale below score granularity.
    frac = int.from_bytes(h.digest(), "big") / float(1 << 64)
    return frac * 1e-7


def _order_key(c: Dict[str, Any], rarity_temp: float, jitter: bool) -> float:
    """Combined ordering key: moderated_score (+ optional sub-score jitter)."""
    s = moderated_score(c, rarity_temp)
    return s + _img_jitter(c) if jitter else s


def _pair_aware_allocation(
    candidates: List[Dict[str, Any]],
    top_k: int,
    per_pair_cap_frac: float | None = None,
    rarity_temp: float = 1.0,
    pair_cap: int | None = None,
    img_diversity_cap: int | None = None,
    _src_counter: "Counter | None" = None,
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

    ``img_diversity_cap`` (Fix4) caps how many times any single source defect
    crop — keyed on (image_path, defect_bbox) — may be selected, ACROSS all
    pairs and phases. The cap is enabled by default (the caller passes 1);
    None ⇒ no cap, legacy byte-identical path. When set, a
    deterministic per-source jitter also breaks the score ties that otherwise
    collapse every pair's head onto the same CSV-order image, so the selection
    spreads across distinct sources instead of repeating a few dozen.
    ``_src_counter`` lets the stratified caller share ONE global source counter
    across class buckets so the cap is enforced globally, not per-bucket.
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

    # --- Source-image diversity cap (Fix4) ---
    # img_cap None ⇒ disabled (no-op, byte-identical). jitter is enabled iff
    # the cap is active. src_counter is SHARED when provided (stratified caller
    # threads one global Counter), else local to this call.
    img_cap = max(1, int(img_diversity_cap)) if img_diversity_cap is not None else None
    jitter = img_cap is not None
    src_counter: Counter = _src_counter if _src_counter is not None else Counter()

    def _src_ok(c: Dict[str, Any]) -> bool:
        """True if selecting c would not exceed the per-source img_cap."""
        if img_cap is None:
            return True
        return src_counter[_source_key(c)] < img_cap

    def _src_take(c: Dict[str, Any]) -> None:
        if img_cap is not None:
            src_counter[_source_key(c)] += 1

    # --- Phase 1: pair-aware coverage + deficit allocation ---
    pair_groups: Dict[Any, List[Dict[str, Any]]] = defaultdict(list)
    for c in candidates:
        pair_groups[(c["cluster_id"], c["cell_key"])].append(c)

    pairs = list(pair_groups.keys())
    n_pairs = len(pairs)

    if top_k <= n_pairs:
        # Budget < #distinct (cluster_id, cell_key) pairs. The legacy path
        # (ranked[:top_k]) spent slots on the highest-scoring candidates
        # regardless of pair, so several slots could land on the same pair and
        # the selection covered FEWER distinct pairs than uniform random
        # selection (observed: severstal context_coverage 0.71 < random 0.79 at
        # n=200, because no coverage floor applies in this branch).
        #
        # Coverage-first, deficit-ordered (B1): take at most ONE candidate per
        # pair, visiting pairs by descending PairDeficit, so the budget buys the
        # maximum number of DISTINCT pairs and prioritises rare (high-deficit)
        # ones — maximising morphology/context/rare-pair coverage at this budget.
        # img_cap / eff_cap are still honoured; any shortfall (pairs whose only
        # sources are img_cap-blocked) is backfilled below. Affects the
        # deficit_aware (AROMA) path only — the random strategy samples
        # uniformly and never enters this function.
        pair_deficit_local = {
            p: float(np.mean([c["deficit"] for c in pair_groups[p]]))
            for p in pairs
        }
        pair_order = sorted(
            pairs,
            key=lambda p: (pair_deficit_local[p], len(pair_groups[p])),
            reverse=True,
        )
        picked: List[Dict[str, Any]] = []
        for p in pair_order:
            if len(picked) >= top_k:
                break
            best = sorted(
                pair_groups[p],
                key=lambda c: _order_key(c, rarity_temp, jitter),
                reverse=True,
            )
            for c in best:
                if _src_ok(c):
                    picked.append(c)
                    _src_take(c)
                    break
        # Backfill if img_cap blocked some pairs and we are still short of top_k.
        if len(picked) < top_k:
            picked_ids = {id(c) for c in picked}
            pair_count: Dict[Any, int] = defaultdict(int)
            for c in picked:
                pair_count[(c["cluster_id"], c["cell_key"])] += 1
            rest = sorted(
                (c for c in candidates if id(c) not in picked_ids),
                key=lambda c: _order_key(c, rarity_temp, jitter),
                reverse=True,
            )
            for c in rest:
                if len(picked) >= top_k:
                    break
                key = (c["cluster_id"], c["cell_key"])
                if eff_cap is not None and pair_count[key] >= eff_cap:
                    continue
                if not _src_ok(c):
                    continue
                picked.append(c)
                pair_count[key] += 1
                _src_take(c)
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
            key=lambda c: _order_key(c, rarity_temp, jitter),
            reverse=True,
        )
        if img_cap is None:
            selected.extend(top_in_pair[:quota])
        else:
            # img_cap-aware: take up to `quota` candidates from this pair whose
            # source (image, bbox) has not yet hit img_cap. Over-cap sources are
            # skipped; the slots they would have filled fall through to Phase-2
            # backfill (which spreads them onto other distinct sources), so the
            # selection never repeats one crop beyond img_cap.
            taken = 0
            for c in top_in_pair:
                if taken >= quota:
                    break
                if not _src_ok(c):
                    continue
                selected.append(c)
                _src_take(c)
                taken += 1

    # --- Phase 2: quality backfill for slack slots ---
    slack = top_k - len(selected)
    if slack > 0:
        selected_ids = {id(c) for c in selected}
        rest = sorted(
            (c for c in candidates if id(c) not in selected_ids),
            key=lambda c: _order_key(c, rarity_temp, jitter),
            reverse=True,
        )
        if eff_cap is None and img_cap is None:
            backfill = rest[:slack]
        else:
            # cap-aware backfill: track the running per-pair count of what is
            # already selected, then skip any candidate whose pair is at
            # eff_cap OR whose source is at img_cap. This stops an evicted
            # (capped) pair's high-score candidates from re-flooding the freed
            # slots (the original bug) and stops one crop being repeated.
            pair_count: Dict[Any, int] = defaultdict(int)
            for c in selected:
                pair_count[(c["cluster_id"], c["cell_key"])] += 1
            backfill = []
            for c in rest:
                if len(backfill) >= slack:
                    break
                key = (c["cluster_id"], c["cell_key"])
                if eff_cap is not None and pair_count[key] >= eff_cap:
                    continue
                if not _src_ok(c):
                    continue
                backfill.append(c)
                pair_count[key] += 1
                _src_take(c)
        selected += backfill
        logger.info(
            "pair_aware_allocation: phase2 backfill %d / %d slack slots",
            len(backfill), slack,
        )

    # --- Bounded-repetition fallback (img_cap only) ---
    # If img_cap=1 (or low) and top_k exceeds the distinct sources available in
    # this pool, the distinct-only passes above cannot reach top_k. Rather than
    # silently return fewer than requested, relax the per-source cap one unit at
    # a time (still pair_cap-aware) until top_k is met, and LOG that bounded
    # repetition was necessary — i.e. this pool genuinely lacks enough distinct
    # sources. Only runs when an img_cap is active AND we are still short.
    if img_cap is not None and len(selected) < top_k:
        n_distinct = len({_source_key(c) for c in candidates})
        logger.info(
            "pair_aware_allocation: only %d distinct sources for top_k=%d under "
            "img_cap=%d — permitting bounded repetition to fill remaining %d slots",
            n_distinct, top_k, img_cap, top_k - len(selected),
        )
        ranked = sorted(
            candidates,
            key=lambda c: _order_key(c, rarity_temp, jitter),
            reverse=True,
        )
        pair_count = defaultdict(int)
        for c in selected:
            pair_count[(c["cluster_id"], c["cell_key"])] += 1
        relaxed_cap = img_cap
        while len(selected) < top_k:
            relaxed_cap += 1
            progressed = False
            for c in ranked:
                if len(selected) >= top_k:
                    break
                key = (c["cluster_id"], c["cell_key"])
                if eff_cap is not None and pair_count[key] >= eff_cap:
                    continue
                if src_counter[_source_key(c)] >= relaxed_cap:
                    continue
                selected.append(c)
                src_counter[_source_key(c)] += 1
                pair_count[key] += 1
                progressed = True
            if not progressed:
                # pair_cap also exhausted — cannot fill further without
                # violating monoculture control; stop (top_k may be unmet).
                logger.info(
                    "pair_aware_allocation: pair_cap exhausted at relaxed "
                    "img_cap=%d; selected=%d (top_k=%d not fully met)",
                    relaxed_cap, len(selected), top_k,
                )
                break

    return selected


def _stratified_pair_aware(
    candidates: List[Dict[str, Any]],
    top_k: int,
    per_pair_cap_frac: float | None = None,
    rarity_temp: float = 1.0,
    img_diversity_cap: int | None = None,
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
        return _pair_aware_allocation(
            candidates, top_k, per_pair_cap_frac, rarity_temp,
            img_diversity_cap=img_diversity_cap,
        )

    # GLOBAL source-image cap (Fix4): one Counter shared across all buckets so
    # the per-(image, bbox) cap is enforced GLOBALLY, not per-class. jitter is
    # enabled implicitly inside each bucket's _pair_aware_allocation when the
    # cap is set. None ⇒ disabled (no-op, byte-identical multi-class path).
    img_cap = max(1, int(img_diversity_cap)) if img_diversity_cap is not None else None
    jitter = img_cap is not None
    src_counter: Counter = Counter()

    def _src_ok(c: Dict[str, Any]) -> bool:
        if img_cap is None:
            return True
        return src_counter[_source_key(c)] < img_cap

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
        picked = _pair_aware_allocation(
            buckets[k], q, None, rarity_temp,
            img_diversity_cap=img_cap, _src_counter=src_counter,
        )
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
            # lowest order_key evicted first; stable tie-break by id()
            members_sorted = sorted(
                members,
                key=lambda c: (_order_key(c, rarity_temp, jitter), -id(c)),
            )
            for c in members_sorted[: len(members) - pair_cap]:
                evicted_ids.add(id(c))

        if evicted_ids:
            selected = [c for c in selected if id(c) not in evicted_ids]
            if img_cap is not None:
                # free the evicted candidates' source slots so the global cap
                # accounting reflects what is actually still selected.
                for c in candidates:
                    if id(c) in evicted_ids:
                        sk = _source_key(c)
                        if src_counter[sk] > 0:
                            src_counter[sk] -= 1

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
                key=lambda c: _order_key(c, rarity_temp, jitter),
                reverse=True,
            )
            added = 0
            for c in pool:
                if added >= need:
                    break
                pkey = (c["cluster_id"], c["cell_key"])
                if pair_count[pkey] >= pair_cap:
                    continue
                if not _src_ok(c):
                    continue
                selected.append(c)
                sel_class[id(c)] = k
                selected_ids.add(id(c))
                pair_count[pkey] += 1
                class_count[k] += 1
                if img_cap is not None:
                    src_counter[_source_key(c)] += 1
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
            key=lambda c: _order_key(c, rarity_temp, jitter),
            reverse=True,
        )
        if pair_cap is None and img_cap is None:
            backfill = rest[:slack]
        else:
            # cap-aware: never let backfill re-flood a pair back over pair_cap,
            # nor repeat a source (image, bbox) past img_cap.
            pair_count = defaultdict(int)
            for c in selected:
                pair_count[(c["cluster_id"], c["cell_key"])] += 1
            backfill = []
            for c in rest:
                if len(backfill) >= slack:
                    break
                key = (c["cluster_id"], c["cell_key"])
                if pair_cap is not None and pair_count[key] >= pair_cap:
                    continue
                if not _src_ok(c):
                    continue
                backfill.append(c)
                pair_count[key] += 1
                if img_cap is not None:
                    src_counter[_source_key(c)] += 1
        selected += backfill
        logger.info(
            "stratified_pair_aware: %d classes, global backfill %d / %d slack slots",
            K, len(backfill), slack,
        )

    # --- Bounded-repetition fallback (img_cap, multi-class) ---
    # The distinct-only passes above cannot reach top_k when the WHOLE pool has
    # fewer distinct sources than top_k. Relax the per-source cap one unit at a
    # time (still pair_cap-aware) until top_k is met, and LOG it. Only runs when
    # img_cap is active AND still short.
    if img_cap is not None and len(selected) < top_k:
        n_distinct = len({_source_key(c) for c in candidates})
        logger.info(
            "stratified_pair_aware: only %d distinct sources for top_k=%d under "
            "img_cap=%d — permitting bounded repetition to fill remaining %d slots",
            n_distinct, top_k, img_cap, top_k - len(selected),
        )
        ranked = sorted(
            candidates,
            key=lambda c: _order_key(c, rarity_temp, jitter),
            reverse=True,
        )
        pair_count = defaultdict(int)
        for c in selected:
            pair_count[(c["cluster_id"], c["cell_key"])] += 1
        relaxed_cap = img_cap
        while len(selected) < top_k:
            relaxed_cap += 1
            progressed = False
            for c in ranked:
                if len(selected) >= top_k:
                    break
                key = (c["cluster_id"], c["cell_key"])
                if pair_cap is not None and pair_count[key] >= pair_cap:
                    continue
                if src_counter[_source_key(c)] >= relaxed_cap:
                    continue
                selected.append(c)
                src_counter[_source_key(c)] += 1
                pair_count[key] += 1
                progressed = True
            if not progressed:
                logger.info(
                    "stratified_pair_aware: pair_cap exhausted at relaxed "
                    "img_cap=%d; selected=%d (top_k=%d not fully met)",
                    relaxed_cap, len(selected), top_k,
                )
                break

    return selected


def _stratified_compat(
    candidates: List[Dict[str, Any]],
    top_k: int,
    img_diversity_cap: int | None = None,
) -> List[Dict[str, Any]]:
    """
    Class-stratified allocation for the *compatibility* strategy (Fix1, §8).

    Mirrors the anti-starvation structure of :func:`_stratified_pair_aware`
    (bucket by ``class_key``, symmetric floor ``top_k // K`` per class, then a
    class-agnostic backfill) but the ranking key is the **compatibility score**

        compat_score = 0.6 * ctx_prior + 0.4 * morph_prior

    instead of the deficit-based ``moderated_score``. This exists because the
    plain ``compatibility`` branch ranks GLOBALLY by compat_score, letting a
    majority class (e.g. severstal class2) monopolise the global top-k and
    starving a rare class (class3) — ``--class_floor`` had no effect on that
    branch. Here every class is guaranteed a symmetric floor first.

    Determinism: no RNG. Ties are resolved by :func:`_img_jitter` (a fixed
    blake2b digest of the source crop), exactly as the global compat branch
    does, so results are reproducible and consistent across arms.

    With a single class (K <= 1) this returns the global compat top-k verbatim
    (identical to the plain ``compatibility`` branch), keeping the single-class
    path byte-identical.
    """
    if not candidates:
        return []

    def _compat_score(c: Dict[str, Any]) -> float:
        cp = max(0.0, min(1.0, float(c.get("ctx_prior", 0.0))))
        mp = max(0.0, min(1.0, float(c.get("morph_prior", 0.0))))
        return 0.6 * cp + 0.4 * mp

    # Deterministic combined key: compat_score desc, jitter breaks exact ties
    # by spreading across distinct source crops (same key the global branch uses).
    def _key(c: Dict[str, Any]) -> float:
        return _compat_score(c) + _img_jitter(c)

    # GLOBAL source-image cap (mirrors _stratified_pair_aware): one shared
    # Counter so the per-(image, bbox) cap is enforced GLOBALLY across buckets.
    # None ⇒ disabled.
    img_cap = max(1, int(img_diversity_cap)) if img_diversity_cap is not None else None
    src_counter: Counter = Counter()

    def _src_ok(c: Dict[str, Any]) -> bool:
        if img_cap is None:
            return True
        return src_counter[_source_key(c)] < img_cap

    def _take(c: Dict[str, Any]) -> None:
        if img_cap is not None:
            src_counter[_source_key(c)] += 1

    buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for c in candidates:
        buckets[str(c.get("class_key") or "_")].append(c)

    keys = list(buckets.keys())
    K = len(keys)

    if K <= 1:
        # Single class ⇒ global compat top-k (matches plain compatibility branch)
        # but still honour the source cap when one is set.
        ranked = sorted(candidates, key=_key, reverse=True)
        if img_cap is None:
            return ranked[:top_k]
        out: List[Dict[str, Any]] = []
        for c in ranked:
            if len(out) >= top_k:
                break
            if not _src_ok(c):
                continue
            out.append(c)
            _take(c)
        # bounded relaxation if distinct sources < top_k (mirror pair-aware)
        if len(out) < top_k:
            relaxed = img_cap
            while len(out) < top_k:
                relaxed += 1
                progressed = False
                for c in ranked:
                    if len(out) >= top_k:
                        break
                    if src_counter[_source_key(c)] >= relaxed:
                        continue
                    out.append(c)
                    src_counter[_source_key(c)] += 1
                    progressed = True
                if not progressed:
                    break
        return out

    # --- symmetric floor per class ---
    floor = top_k // K
    selected: List[Dict[str, Any]] = []
    selected_ids: set = set()

    if floor > 0:
        for k in keys:
            pool = sorted(buckets[k], key=_key, reverse=True)
            added = 0
            for c in pool:
                if added >= floor:
                    break
                if id(c) in selected_ids:
                    continue
                if not _src_ok(c):
                    continue
                selected.append(c)
                selected_ids.add(id(c))
                _take(c)
                added += 1

    # --- remainder backfill: class-agnostic, compat_score desc, cap-aware ---
    slack = top_k - len(selected)
    if slack > 0:
        rest = sorted(
            (c for c in candidates if id(c) not in selected_ids),
            key=_key,
            reverse=True,
        )
        for c in rest:
            if slack <= 0:
                break
            if not _src_ok(c):
                continue
            selected.append(c)
            selected_ids.add(id(c))
            _take(c)
            slack -= 1

    # --- bounded-repetition fallback (only if a cap blocked reaching top_k) ---
    if img_cap is not None and len(selected) < top_k:
        ranked_all = sorted(candidates, key=_key, reverse=True)
        relaxed = img_cap
        while len(selected) < top_k:
            relaxed += 1
            progressed = False
            for c in ranked_all:
                if len(selected) >= top_k:
                    break
                if src_counter[_source_key(c)] >= relaxed:
                    continue
                selected.append(c)
                src_counter[_source_key(c)] += 1
                progressed = True
            if not progressed:
                break

    return selected


def select_rois(
    candidates: List[Dict[str, Any]],
    strategy: str = "deficit_aware",
    top_k: int = 200,
    seed: int = 42,
    class_floor: bool = False,
    per_pair_cap_frac: float | None = None,
    rarity_temp: float = 1.0,
    img_diversity_cap: int | None = 1,
) -> List[Dict[str, Any]]:
    """
    Select top_k ROIs from candidates using the given strategy.

    Strategies:
        deficit_aware   Top-K Deficit Quantile oversampling (default)
        compatibility   Rank by compat_score = 0.6*ctx_prior + 0.4*morph_prior
                        (no deficit term); the quality gate is applied as an
                        upstream pre-filter in run(). ctx_prior IS the
                        compatibility value (compatibility_matrix).
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
        img_diversity_cap   (Fix4) Max times any single source defect crop
                            (image_path, defect_bbox) may be selected. Default
                            is 1 (each distinct source defect drawn at most
                            once), which forces the selection to draw DISTINCT
                            source defects and eliminates the diversity-collapse
                            confound where a few crops were repeated dozens of
                            times. Applies to deficit_aware ONLY (both single-
                            and multi-class). Pass a large int (e.g. 999) to
                            restore the legacy uncapped behaviour for an
                            ablation; None is also accepted as "no cap".

    The random / top_k / weighted branches ignore the multi-class options
    (random already samples WITHOUT replacement over distinct candidate rows,
    so it does not exhibit the deficit_aware tie-collapse).
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

    if strategy == "compatibility":
        # Compatibility-first selection (no deficit term). ctx_prior IS the
        # compatibility value (compatibility_matrix[cluster][cell]); rank by
        # compat_score = 0.6*ctx_prior + 0.4*morph_prior. The quality gate is
        # applied UPSTREAM in run() via apply_quality_gate (a pre-filter on
        # quality_score >= min_quality); here we only order by compatibility.
        # min_quality is enforced as a filter when the quality deps are
        # available — see run(); this branch operates on the surviving pool.
        def _compat_score(c: Dict[str, Any]) -> float:
            cp = max(0.0, min(1.0, float(c.get("ctx_prior", 0.0))))
            mp = max(0.0, min(1.0, float(c.get("morph_prior", 0.0))))
            return 0.6 * cp + 0.4 * mp
        # Fix1 (§8): when class_floor is requested and the pool spans >1 class,
        # route through class-stratified allocation so no majority class can
        # monopolise the global compat top-k and starve a rare class. Otherwise
        # keep the legacy global compat sort (byte-identical single-class path).
        if class_floor:
            n_classes = len({str(c.get("class_key") or "_") for c in candidates})
            if n_classes > 1:
                return _stratified_compat(
                    candidates, top_k, img_diversity_cap=img_diversity_cap,
                )
        # Deterministic ordering: compat_score desc, then the same per-source
        # jitter used elsewhere to spread ties across distinct source crops.
        return sorted(
            candidates,
            key=lambda c: _compat_score(c) + _img_jitter(c),
            reverse=True,
        )[:top_k]

    # deficit_aware (default)
    if class_floor:
        n_classes = len({str(c.get("class_key") or "_") for c in candidates})
        if n_classes > 1:
            return _stratified_pair_aware(
                candidates, top_k, per_pair_cap_frac, rarity_temp,
                img_diversity_cap=img_diversity_cap,
            )
    return _pair_aware_allocation(
        candidates, top_k, per_pair_cap_frac, rarity_temp,
        img_diversity_cap=img_diversity_cap,
    )


# ---------------------------------------------------------------------------
# Summary markdown
# ---------------------------------------------------------------------------

def _diversity_stats(selected: List[Dict[str, Any]]) -> Dict[str, int]:
    """Distinct-source-defect diversity of a selection (auditability covariate).

    Reports the count of distinct source crops actually drawn, so the
    diversity confound (a few crops repeated many times) is visible per arm
    and comparable across random / casda / aroma.
    """
    src_keys = [_source_key(c) for c in selected]
    counts = Counter(src_keys)
    return {
        "selected": len(selected),
        "distinct_source": len(counts),
        "distinct_image": len({c.get("image_path", "") for c in selected}),
        "max_per_source": max(counts.values()) if counts else 0,
    }


def _build_summary(
    candidates: List[Dict[str, Any]],
    selected: List[Dict[str, Any]],
    strategy: str,
) -> str:
    div = _diversity_stats(selected)
    lines = [
        "# AROMA Step 3 — ROI Selection Summary",
        "",
        f"**Strategy**: `{strategy}`",
        f"**Total candidates**: {len(candidates)}",
        f"**Selected**: {len(selected)}",
        "",
        "## Source-defect diversity",
        "",
        f"- Distinct source (image_path, defect_bbox): **{div['distinct_source']}** / {div['selected']} selected",
        f"- Distinct source images: **{div['distinct_image']}**",
        f"- Max repetition of any single source: **{div['max_per_source']}**",
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
    img_diversity_cap: int | None = 1,
    min_quality: float = 0.0,
    background_type: str = "directional",
) -> Dict[str, Any]:
    """
    Full Step 3 pipeline: load → score → quality-gate → select → save.

    Returns dict with 'candidates' and 'selected' keys on success,
    or a status-only dict on failure.

    class_floor / per_pair_cap_frac / rarity_temp / img_diversity_cap are
    deficit_aware options forwarded to select_rois; see its docstring.
    img_diversity_cap defaults to 1 (distinct source defects); the others
    default to no-op.

    min_quality (default 0.0 = OFF) opts into the subtype-matching quality gate:
    candidates with quality_score < min_quality are dropped BEFORE selection.
    roi_candidates.json keeps the full scored set (incl. quality_score) so the
    threshold can be tuned from the observed distribution. background_type is the
    dataset background used by the matching proxy (only used when min_quality>0).
    """
    logger.info("Loading inputs: profiling_dir=%s prompts_dir=%s", profiling_dir, prompts_dir)
    data = load_inputs(profiling_dir, prompts_dir)
    if data["status"] != "ok":
        logger.error("Input loading failed: %s", data)
        return data

    candidates = build_candidates(data, background_type=background_type)
    passing    = apply_quality_gate(candidates, min_quality)
    if not passing:
        logger.error(
            "Quality gate removed ALL %d candidates (min_quality=%.3f). "
            "roi_selected.json will be empty — lower --min_quality.",
            len(candidates), min_quality,
        )
    selected   = select_rois(
        passing,
        strategy=strategy,
        top_k=top_k,
        seed=seed,
        class_floor=class_floor,
        per_pair_cap_frac=per_pair_cap_frac,
        rarity_temp=rarity_temp,
        img_diversity_cap=img_diversity_cap,
    )

    logger.info(
        "Strategy=%s  top_k=%d  candidates=%d  selected=%d",
        strategy, top_k, len(candidates), len(selected),
    )
    div = _diversity_stats(selected)
    logger.info(
        "Source-defect diversity: distinct (image,bbox)=%d, distinct images=%d, "
        "max repetition per source=%d (img_diversity_cap=%s)",
        div["distinct_source"], div["distinct_image"], div["max_per_source"],
        img_diversity_cap,
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
                   choices=["deficit_aware", "compatibility", "top_k", "weighted", "random"],
                   help="ROI sampling strategy (default: deficit_aware). "
                        "'compatibility' ranks by 0.6*ctx_prior + 0.4*morph_prior "
                        "(no deficit term) and relies on --min_quality as the gate.")
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
    p.add_argument("--img_diversity_cap", type=int, default=1,
                   help="Max times any single source defect crop "
                        "(image_path, defect_bbox) may be selected (Fix4). "
                        "Default 1 = each distinct source defect drawn at most "
                        "once, removing the diversity-collapse confound "
                        "(deficit_aware only). Pass a large int (e.g. 999) to "
                        "restore the legacy uncapped behaviour for an ablation.")
    p.add_argument("--min_quality", type=float, default=0.0,
                   help="Minimum subtype-matching quality score to keep a ROI "
                        "candidate (default: 0.0 = gate DISABLED, legacy). Set "
                        ">0 (e.g. 0.5) to drop morphologically unsuitable ROIs "
                        "before selection. Tune from the quality_score "
                        "distribution in roi_candidates.json.")
    p.add_argument("--background_type", default="directional",
                   choices=["smooth", "directional", "periodic", "organic", "complex"],
                   help="Dataset background type for the subtype-matching "
                        "quality proxy (default: directional, suited to "
                        "severstal steel strips). Only used when --min_quality>0.")
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
        img_diversity_cap=args.img_diversity_cap,
        min_quality=args.min_quality,
        background_type=args.background_type,
    )
    if result.get("status") != "ok":
        sys.exit(1)


if __name__ == "__main__":
    main()
