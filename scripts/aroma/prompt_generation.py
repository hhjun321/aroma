#!/usr/bin/env python3
"""
AROMA Step 2 — Prompt Generation

Reads Phase 0 + Step 1 outputs and generates semantic defect prompts
per morphology cluster × context bin combination using fixed templates.
No LLM required.

Usage (Colab):
    !python $AROMA_SCRIPTS/prompt_generation.py \
        --profiling_dir  $AROMA_OUT/profiling/isp_LSM_1 \
        --complexity_dir $AROMA_OUT/complexity/isp_LSM_1 \
        --output_dir     $AROMA_OUT/prompts/isp_LSM_1

Outputs (written to --output_dir):
    prompts.json   dict keyed by "{cluster_id}_{cell_key}" →
                   {morphology_descriptor, context_descriptor,
                    prior_modifier, prompt, cluster_id, cell_key,
                    prior_prob, deficit}
    prompts_summary.md   human-readable table
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("aroma.prompt")


# ---------------------------------------------------------------------------
# I/O bootstrap (same pattern as compute_complexity.py)
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
# Phase 0 output constants (mirrors distribution_profiling.py)
# ---------------------------------------------------------------------------

CONTEXT_FEATURES = [
    "local_variance", "edge_density", "texture_entropy",
    "frequency_energy", "orientation_consistency",
]
MORPH_FEATURES = [
    "linearity", "solidity", "extent", "aspect_ratio", "eccentricity", "circularity",
]
N_CONTEXT_BINS = 3  # 0=low, 1=medium, 2=high


# ---------------------------------------------------------------------------
# Template tables
# ---------------------------------------------------------------------------

# Morphology: aspect_ratio thresholds
_AR_HIGH = 3.0
_AR_MED  = 1.5

# Morphology: linearity thresholds
_LIN_HIGH = 0.65
_LIN_LOW  = 0.30

# Morphology: solidity thresholds
_SOL_HIGH = 0.80
_SOL_MED  = 0.50

# Context: semantic labels per (feature, bin)
_CTX_LABELS: Dict[str, List[str]] = {
    "local_variance":        ["uniform",        "varied",       "highly varied"],
    "edge_density":          ["smooth",         "moderate edge","sharp edge"],
    "texture_entropy":       ["uniform texture","mixed texture","complex texture"],
    "frequency_energy":      ["low frequency",  "mid frequency","high frequency"],
    "orientation_consistency":["random grain",  "semi-directional","directional"],
}

# Prior modifier thresholds
_PRIOR_HIGH = 0.40   # dominant combination
_PRIOR_MED  = 0.20   # common combination


# ---------------------------------------------------------------------------
# Morphology descriptor
# ---------------------------------------------------------------------------

def _ar_label(ar: float) -> str:
    if ar > _AR_HIGH:
        return "highly elongated"
    if ar > _AR_MED:
        return "elongated"
    return "compact"


def _lin_label(lin: float) -> str:
    if lin > _LIN_HIGH:
        return "linear"
    if lin > _LIN_LOW:
        return "irregular"
    return "scattered"


def _sol_label(sol: float) -> str:
    if sol > _SOL_HIGH:
        return "solid"
    if sol > _SOL_MED:
        return "semi-solid"
    return "fragmented"


def generate_morphology_descriptor(centroid: dict) -> str:
    """
    Fixed-template morphology descriptor from cluster centroid.
    Uses aspect_ratio, linearity, solidity.

    Returns:
        str: e.g. "highly elongated linear defect with solid boundary"
    """
    ar  = float(centroid.get("aspect_ratio", 1.0))
    lin = float(centroid.get("linearity",    0.5))
    sol = float(centroid.get("solidity",     0.7))

    shape    = _ar_label(ar)
    linearity = _lin_label(lin)
    solidity  = _sol_label(sol)

    return f"{shape} {linearity} defect with {solidity} boundary"


# ---------------------------------------------------------------------------
# Context descriptor
# ---------------------------------------------------------------------------

def _parse_cell_key(cell_key: str) -> List[int]:
    """Convert '0_1_2_0_1' → [0, 1, 2, 0, 1]."""
    try:
        return [int(b) for b in cell_key.split("_")]
    except ValueError:
        return [0] * len(CONTEXT_FEATURES)


def _dominant_context_features(bins: List[int], top_n: int = 2) -> List[Tuple[str, int]]:
    """Return top_n features sorted by bin value descending."""
    ranked = sorted(
        enumerate(bins),
        key=lambda x: x[1],
        reverse=True,
    )[:top_n]
    return [(CONTEXT_FEATURES[i], b) for i, b in ranked]


def generate_context_descriptor(cell_key: str) -> str:
    """
    Fixed-template context descriptor from cell key like '0_1_2_0_1'.
    Labels each feature bin semantically and combines the dominant ones.

    Returns:
        str: e.g. "directional complex texture background"
    """
    bins = _parse_cell_key(cell_key)
    if len(bins) != len(CONTEXT_FEATURES):
        bins = (bins + [1] * len(CONTEXT_FEATURES))[:len(CONTEXT_FEATURES)]

    dominant = _dominant_context_features(bins, top_n=2)

    parts: List[str] = []
    for feat, b in dominant:
        label = _CTX_LABELS.get(feat, ["low", "medium", "high"])[min(b, 2)]
        parts.append(label)

    suffix = "surface background"
    return " ".join(parts) + " " + suffix


# ---------------------------------------------------------------------------
# Prior modifier
# ---------------------------------------------------------------------------

def generate_prior_modifier(
    cluster_id: int,
    cell_key: str,
    matrix: Dict[str, Dict[str, float]],
) -> str:
    """
    Describe how common this cluster × context combination is.

    Args:
        cluster_id:  morphology cluster integer id
        cell_key:    context bin key like '0_1_2_0_1'
        matrix:      compatibility matrix dict {cluster_str: {cell: prob}}

    Returns:
        str: e.g. "predominantly occurring on this surface type"
    """
    cluster_row = matrix.get(str(cluster_id), {})
    prob = float(cluster_row.get(cell_key, 0.0))

    if prob >= _PRIOR_HIGH:
        return "predominantly occurring on this surface type"
    if prob >= _PRIOR_MED:
        return "commonly found on this surface type"
    return "rarely observed on this surface type"


# ---------------------------------------------------------------------------
# Full prompt assembly
# ---------------------------------------------------------------------------

def assemble_prompt(
    morph_desc: str,
    ctx_desc: str,
    prior_mod: str,
) -> str:
    """Combine three descriptors into a final generation prompt."""
    return f"{morph_desc}, {ctx_desc}, {prior_mod}"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_inputs(
    profiling_dir: str,
    complexity_dir: str,
) -> Dict[str, Any]:
    """
    Load Phase 0 + Step 1 outputs.

    Returns dict with:
        clusters, compat_matrix, deficit_analysis, complexity_report, status
    """
    pd = Path(profiling_dir)
    cd = Path(complexity_dir)

    missing = []
    for p in [
        pd / "morphology_clusters.json",
        pd / "compatibility_matrix.json",
    ]:
        if not p.exists():
            missing.append(str(p))

    if missing:
        return {"status": "missing_inputs", "missing": missing}

    clusters_json  = load_json(str(pd / "morphology_clusters.json"))
    compat_json    = load_json(str(pd / "compatibility_matrix.json"))

    deficit_path = pd / "deficit_analysis.json"
    deficit_json = load_json(str(deficit_path)) if deficit_path.exists() else {}

    complexity_path = cd / "complexity_report.json"
    complexity_json = load_json(str(complexity_path)) if complexity_path.exists() else {}

    return {
        "status":           "ok",
        "clusters":         clusters_json,
        "compat_matrix":    compat_json,
        "deficit_analysis": deficit_json,
        "complexity_report": complexity_json,
    }


# ---------------------------------------------------------------------------
# Core generation
# ---------------------------------------------------------------------------

def generate_prompts(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate one prompt per (cluster_id × cell_key) combination.

    Returns:
        dict keyed by "{cluster_id}_{cell_key}" → prompt entry
    """
    clusters_json = data["clusters"]
    compat_json   = data["compat_matrix"]
    deficit_json  = data["deficit_analysis"]

    cluster_list: List[dict] = clusters_json.get("clusters", [])
    matrix: Dict[str, Dict[str, float]] = compat_json.get("matrix", {})

    if not cluster_list:
        logger.warning("No clusters found in morphology_clusters.json")
        return {}

    if not matrix:
        logger.warning("Compatibility matrix is empty")

    prompts: Dict[str, Any] = {}

    for cluster_info in cluster_list:
        cid  = int(cluster_info["cluster_id"])
        cid_str = str(cid)
        centroid: dict = cluster_info.get("centroid", {})
        n_samples = int(cluster_info.get("n_samples", 0))
        phase0_label = cluster_info.get("label", "")

        morph_desc = generate_morphology_descriptor(centroid)
        morph_prior = float(deficit_json.get(cid_str, {}).get("prior", 0.0))

        cluster_row: Dict[str, float] = matrix.get(cid_str, {})

        if not cluster_row:
            # No context data — emit a single prompt with neutral context
            key = f"{cid}_none"
            ctx_desc  = "surface background"
            prior_mod = "observed on this surface type"
            prompts[key] = {
                "morphology_descriptor": morph_desc,
                "context_descriptor":    ctx_desc,
                "prior_modifier":        prior_mod,
                "prompt":                assemble_prompt(morph_desc, ctx_desc, prior_mod),
                "cluster_id":            cid,
                "phase0_label":          phase0_label,
                "cell_key":              "none",
                "prior_prob":            morph_prior,
                "deficit":               0.0,
                "n_cluster_samples":     n_samples,
            }
            continue

        deficit_row: Dict[str, float] = deficit_json.get(cid_str, {}).get("deficit", {})

        for cell_key, cell_prob in cluster_row.items():
            ctx_desc  = generate_context_descriptor(cell_key)
            prior_mod = generate_prior_modifier(cid, cell_key, matrix)
            prompt    = assemble_prompt(morph_desc, ctx_desc, prior_mod)
            deficit   = float(deficit_row.get(cell_key, 0.0))

            key = f"{cid}_{cell_key}"
            prompts[key] = {
                "morphology_descriptor": morph_desc,
                "context_descriptor":    ctx_desc,
                "prior_modifier":        prior_mod,
                "prompt":                prompt,
                "cluster_id":            cid,
                "phase0_label":          phase0_label,
                "cell_key":              cell_key,
                "prior_prob":            round(float(cell_prob), 6),
                "deficit":               round(deficit, 6),
                "n_cluster_samples":     n_samples,
            }

    logger.info("Generated %d prompts across %d clusters", len(prompts), len(cluster_list))
    return prompts


# ---------------------------------------------------------------------------
# Summary markdown
# ---------------------------------------------------------------------------

def _build_summary(prompts: Dict[str, Any], complexity_report: dict) -> str:
    lines = [
        "# AROMA Step 2 — Prompt Summary",
        "",
        f"**Total combinations**: {len(prompts)}",
    ]

    mci = complexity_report.get("mci")
    cci = complexity_report.get("cci")
    pol = complexity_report.get("morphology_policy")
    if mci is not None:
        lines.append(f"**MCI**: {mci:.4f}  **CCI**: {cci:.4f}  **Policy**: {pol}")

    lines += ["", "## Prompts", ""]
    lines.append("| Key | Prompt | P(ctx\\|cluster) | Deficit |")
    lines.append("|-----|--------|-----------------|---------|")

    for key, entry in sorted(prompts.items()):
        p = entry.get("prompt", "")
        prob  = entry.get("prior_prob", 0.0)
        defic = entry.get("deficit",   0.0)
        lines.append(f"| `{key}` | {p} | {prob:.3f} | {defic:.3f} |")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run(
    profiling_dir: str,
    complexity_dir: str,
    output_dir: str,
) -> Dict[str, Any]:
    """
    Full Step 2 pipeline: load inputs → generate prompts → save.

    Returns the prompts dict (or a status-only dict on failure).
    """
    logger.info("Loading inputs from profiling_dir=%s complexity_dir=%s",
                profiling_dir, complexity_dir)
    data = load_inputs(profiling_dir, complexity_dir)

    if data["status"] != "ok":
        logger.error("Cannot proceed: %s", data)
        return data

    prompts = generate_prompts(data)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    save_json(prompts, str(out / "prompts.json"))
    logger.info("Saved prompts.json → %s (%d entries)", out, len(prompts))

    summary_md = _build_summary(prompts, data.get("complexity_report", {}))
    (out / "prompts_summary.md").write_text(summary_md, encoding="utf-8")
    logger.info("Saved prompts_summary.md")

    return prompts


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="AROMA Step 2 — Semantic Prompt Generation (fixed templates)"
    )
    p.add_argument("--profiling_dir",  required=True,
                   help="Phase 0 output directory (morphology_clusters.json etc.)")
    p.add_argument("--complexity_dir", required=True,
                   help="Step 1 output directory (complexity_report.json)")
    p.add_argument("--output_dir",     required=True,
                   help="Directory to write prompts.json and prompts_summary.md")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = _parse_args(argv)
    result = run(
        profiling_dir=args.profiling_dir,
        complexity_dir=args.complexity_dir,
        output_dir=args.output_dir,
    )
    if isinstance(result, dict) and result.get("status") not in (None, "ok"):
        sys.exit(1)


if __name__ == "__main__":
    main()
