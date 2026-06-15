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
import os
import sys
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
                "roi_score":   round(roi_score, 6),
                "morph_prior": round(morph_prior, 6),
                "ctx_prior":   round(float(ctx_prior), 6),
                "deficit":     round(deficit, 6),
                "prompt":      prompt_entry.get("prompt", ""),
                "morph_label": morph_label,
                "ctx_label":   prompt_entry.get("context_descriptor", ""),
            })

    logger.info("Scored %d ROI candidates from %d morph rows", len(candidates), len(morph_rows))
    return candidates


# ---------------------------------------------------------------------------
# Sampling strategies
# ---------------------------------------------------------------------------

def _rare_pair_deficit_quantile(
    candidates: List[Dict[str, Any]],
    top_k: int,
    quantile: float = 0.75,
) -> List[Dict[str, Any]]:
    """
    Deficit = 1 - Observed/Expected already pre-computed in deficit_analysis.
    Select top_k candidates weighted toward high-deficit pairs:
        1. Compute quantile threshold over nonzero-deficit candidates only.
           When 75%+ of candidates have deficit=0, p75(all)=0 collapses every
           candidate into the "rare" bucket and nullifies oversampling.
        2. Candidates above threshold get priority in selection.
        3. Fill remainder from full pool ranked by roi_score.
    """
    if not candidates:
        return []

    nonzero_def = [c["deficit"] for c in candidates if c["deficit"] > 0]
    if not nonzero_def:
        # No deficit signal — fall back to roi_score ranking.
        return sorted(candidates, key=lambda c: c["roi_score"], reverse=True)[:top_k]

    thr = float(np.quantile(nonzero_def, quantile))
    rare = [c for c in candidates if c["deficit"] >= thr]
    rest = [c for c in candidates if c["deficit"] < thr]

    rare.sort(key=lambda c: c["roi_score"], reverse=True)
    rest.sort(key=lambda c: c["roi_score"], reverse=True)

    # Fill from rare first, pad with rest
    selected = rare[:top_k]
    remaining = top_k - len(selected)
    if remaining > 0:
        selected += rest[:remaining]

    return selected[:top_k]


def select_rois(
    candidates: List[Dict[str, Any]],
    strategy: str = "deficit_aware",
    top_k: int = 200,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    Select top_k ROIs from candidates using the given strategy.

    Strategies:
        deficit_aware   Top-K Deficit Quantile oversampling (default)
        top_k           Highest roi_score first
        weighted        Probability-weighted random draw
        random          Uniform random draw (baseline for Exp 2)
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
    return _rare_pair_deficit_quantile(candidates, top_k)


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
) -> Dict[str, Any]:
    """
    Full Step 3 pipeline: load → score → select → save.

    Returns dict with 'candidates' and 'selected' keys on success,
    or a status-only dict on failure.
    """
    logger.info("Loading inputs: profiling_dir=%s prompts_dir=%s", profiling_dir, prompts_dir)
    data = load_inputs(profiling_dir, prompts_dir)
    if data["status"] != "ok":
        logger.error("Input loading failed: %s", data)
        return data

    candidates = build_candidates(data)
    selected   = select_rois(candidates, strategy=strategy, top_k=top_k, seed=seed)

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
    )
    if result.get("status") != "ok":
        sys.exit(1)


if __name__ == "__main__":
    main()
