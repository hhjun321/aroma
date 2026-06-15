#!/usr/bin/env python3
"""
AROMA Exp 2 — ROI 품질 비교 (AROMA vs Random)

AROMA의 deficit-aware ROI 선택이 무작위 선택 대비 더 나은
형태학적/컨텍스트 커버리지를 만드는지 5개 지표로 정량 비교한다.

지표:
    morphology_coverage   unique cluster_id in selected / unique in candidates
    context_coverage      unique cell_key in selected / unique in candidates
    rare_pair_coverage    deficit≥p75 unique (cluster,cell_key) in selected / total
    entropy               Shannon entropy of cluster_id dist, normalized H/log(n)
    gini                  Gini coefficient of cluster_id frequency distribution

입력 구조 (--aroma_roi_dir 기준):
    {aroma_roi_dir}/{dataset}/roi_candidates.json   — 전체 후보 (AROMA/random 공유)
    {aroma_roi_dir}/{dataset}/roi_selected.json     — AROMA 선택 결과
    {random_roi_dir}/{dataset}/roi_selected.json    — Random 선택 결과

Usage (Colab):
    !python $AROMA_SCRIPTS/experiments/exp2_roi_quality.py \
        --aroma_roi_dir   $AROMA_OUT/roi \
        --random_roi_dir  $AROMA_OUT/roi_random \
        --dataset_keys    isp_LSM_1 mvtec_cable visa_cashew visa_pcb \
        --output_dir      $AROMA_OUT/exp2

Outputs (written to --output_dir):
    exp2_results.json    per-dataset × strategy × metric
    exp2_summary.md      markdown comparison table
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("aroma.exp2")


# ---------------------------------------------------------------------------
# I/O bootstrap
# ---------------------------------------------------------------------------

def _bootstrap_aroma_ref() -> None:
    ref = os.environ.get("AROMA_REF") or r"D:\project\aroma"
    ref_path = Path(ref)
    if ref_path.is_dir() and str(ref_path) not in sys.path:
        sys.path.insert(0, str(ref_path))


_bootstrap_aroma_ref()

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
# Metric computation
# ---------------------------------------------------------------------------

def _gini(counts: List[int]) -> float:
    """Gini coefficient via mean absolute difference method. Range [0, 1]."""
    if not counts:
        return 0.0
    freqs = np.array(sorted(counts), dtype=np.float64)
    n = len(freqs)
    total = freqs.sum()
    if total == 0:
        return 0.0
    return float(
        (2 * np.sum(np.arange(1, n + 1) * freqs) / (n * total)) - (n + 1) / n
    )


def compute_metrics(
    selected: List[Dict[str, Any]],
    candidates: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Compute 5 ROI quality metrics for a single (dataset, strategy) pair.

    p75 threshold is computed from `candidates` so both AROMA and Random
    are evaluated against the same baseline.
    """
    n_sel = len(selected)
    n_cand = len(candidates)

    # --- morphology coverage ---
    sel_clusters = {c["cluster_id"] for c in selected}
    all_clusters = {c["cluster_id"] for c in candidates}
    morph_cov = len(sel_clusters) / len(all_clusters) if all_clusters else 0.0

    # --- context coverage ---
    sel_cells = {c["cell_key"] for c in selected}
    all_cells = {c["cell_key"] for c in candidates}
    ctx_cov = len(sel_cells) / len(all_cells) if all_cells else 0.0

    # --- rare pair coverage (deficit >= p75 of candidates) ---
    if not candidates:
        rare_pair_cov: float = 0.0
    else:
        deficits = np.array([c.get("deficit", 0.0) for c in candidates], dtype=np.float64)
        p75 = float(np.quantile(deficits, 0.75))
        if p75 == 0.0:
            # all deficit=0: no structural rare pairs — treat as fully covered
            rare_pair_cov = 1.0
        else:
            rare_cands = {
                (c["cluster_id"], c["cell_key"])
                for c in candidates if c.get("deficit", 0.0) >= p75
            }
            rare_sel = {
                (c["cluster_id"], c["cell_key"])
                for c in selected if c.get("deficit", 0.0) >= p75
            }
            # intersect to guard against selected not being a strict subset of candidates
            rare_sel = rare_sel & rare_cands
            rare_pair_cov = len(rare_sel) / len(rare_cands) if rare_cands else 0.0

    # --- entropy (Shannon, normalized H/log(n_clusters)) ---
    if not selected:
        entropy = 0.0
    else:
        cluster_counts = Counter(c["cluster_id"] for c in selected)
        n_clusters = len(cluster_counts)
        if n_clusters <= 1:
            entropy = 0.0
        else:
            total = sum(cluster_counts.values())
            probs = np.array(list(cluster_counts.values()), dtype=np.float64) / total
            probs = probs[probs > 0]
            H = float(-np.sum(probs * np.log(probs)))
            entropy = H / np.log(n_clusters)

    # --- gini ---
    if not selected:
        gini = 0.0
    else:
        cluster_counts = Counter(c["cluster_id"] for c in selected)
        gini = _gini(list(cluster_counts.values()))

    return {
        "morphology_coverage": round(float(morph_cov), 4),
        "context_coverage": round(float(ctx_cov), 4),
        "rare_pair_coverage": round(float(rare_pair_cov), 4),
        "entropy": round(float(entropy), 4),
        "gini": round(float(gini), 4),
        "n_selected": n_sel,
        "n_candidates": n_cand,
    }


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_json_safe(path: Path, label: str) -> Optional[List[Dict[str, Any]]]:
    if not path.exists():
        logger.warning("%s not found: %s", label, path)
        return None
    try:
        data = load_json(str(path))
        if not isinstance(data, list):
            logger.warning("%s is not a list: %s", label, path)
            return None
        return data
    except Exception as e:
        logger.warning("Failed to load %s (%s): %s", label, path, e)
        return None


def load_dataset(
    dataset_key: str,
    aroma_roi_dir: str,
    random_roi_dir: str,
) -> Optional[Dict[str, Any]]:
    """
    Load candidates, aroma_selected, random_selected for one dataset.
    Returns None if any required file is missing.
    """
    aroma_ds = Path(aroma_roi_dir) / dataset_key
    random_ds = Path(random_roi_dir) / dataset_key

    candidates = _load_json_safe(aroma_ds / "roi_candidates.json", "candidates")
    aroma_sel  = _load_json_safe(aroma_ds / "roi_selected.json",   "aroma_selected")
    random_sel = _load_json_safe(random_ds / "roi_selected.json",  "random_selected")

    if candidates is None or aroma_sel is None or random_sel is None:
        logger.warning("Skipping dataset '%s': missing files", dataset_key)
        return None

    logger.info(
        "Dataset '%s': %d candidates, %d aroma, %d random",
        dataset_key, len(candidates), len(aroma_sel), len(random_sel),
    )
    return {
        "candidates":    candidates,
        "aroma_selected": aroma_sel,
        "random_selected": random_sel,
    }


# ---------------------------------------------------------------------------
# Summary markdown
# ---------------------------------------------------------------------------

def _build_summary(results: Dict[str, Any]) -> str:
    metrics = [
        "morphology_coverage",
        "context_coverage",
        "rare_pair_coverage",
        "entropy",
        "gini",
    ]
    lines = [
        "# AROMA Exp 2 — ROI 품질 비교 (AROMA vs Random)",
        "",
        "entropy: Shannon entropy 정규화 (H / log(n_clusters), nats 기준, 0~1 스케일)",
        "gini: Gini coefficient of cluster_id frequency distribution (낮을수록 균등)",
        "",
        "| 데이터셋 | 전략 | morph_cov | ctx_cov | rare_pair_cov | entropy | gini | n_sel | n_cand |",
        "|---------|------|-----------|---------|---------------|---------|------|-------|--------|",
    ]
    for ds, ds_data in sorted(results.items()):
        for strategy in ("aroma", "random"):
            if strategy not in ds_data:
                continue
            m = ds_data[strategy]
            lines.append(
                f"| {ds} | {strategy.upper()} "
                f"| {m['morphology_coverage']:.4f} "
                f"| {m['context_coverage']:.4f} "
                f"| {m['rare_pair_coverage']:.4f} "
                f"| {m['entropy']:.4f} "
                f"| {m['gini']:.4f} "
                f"| {m['n_selected']} "
                f"| {m['n_candidates']} |"
            )

    # Delta section
    lines += ["", "## AROMA ↑ over Random (delta = AROMA − Random)", ""]
    delta_metrics = ["morphology_coverage", "context_coverage", "rare_pair_coverage", "entropy"]
    for ds, ds_data in sorted(results.items()):
        if "aroma" not in ds_data or "random" not in ds_data:
            continue
        deltas = {
            k: round(ds_data["aroma"][k] - ds_data["random"][k], 4)
            for k in delta_metrics
        }
        gini_delta = round(ds_data["random"]["gini"] - ds_data["aroma"]["gini"], 4)
        lines.append(
            f"- **{ds}**: morph_cov Δ{deltas['morphology_coverage']:+.4f}, "
            f"ctx_cov Δ{deltas['context_coverage']:+.4f}, "
            f"rare_pair Δ{deltas['rare_pair_coverage']:+.4f}, "
            f"entropy Δ{deltas['entropy']:+.4f}, "
            f"gini(↓좋음) Δ{gini_delta:+.4f}"
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run(
    aroma_roi_dir: str,
    random_roi_dir: str,
    dataset_keys: List[str],
    output_dir: str,
) -> Dict[str, Any]:
    results: Dict[str, Any] = {}

    for ds in dataset_keys:
        data = load_dataset(ds, aroma_roi_dir, random_roi_dir)
        if data is None:
            continue

        cands = data["candidates"]
        results[ds] = {
            "aroma":  compute_metrics(data["aroma_selected"],  cands),
            "random": compute_metrics(data["random_selected"], cands),
        }
        logger.info(
            "  %s  AROMA morph=%.4f ctx=%.4f rare=%.4f entropy=%.4f gini=%.4f",
            ds,
            results[ds]["aroma"]["morphology_coverage"],
            results[ds]["aroma"]["context_coverage"],
            results[ds]["aroma"]["rare_pair_coverage"],
            results[ds]["aroma"]["entropy"],
            results[ds]["aroma"]["gini"],
        )
        logger.info(
            "  %s Random morph=%.4f ctx=%.4f rare=%.4f entropy=%.4f gini=%.4f",
            ds,
            results[ds]["random"]["morphology_coverage"],
            results[ds]["random"]["context_coverage"],
            results[ds]["random"]["rare_pair_coverage"],
            results[ds]["random"]["entropy"],
            results[ds]["random"]["gini"],
        )

    if not results:
        logger.error("No datasets processed. Check paths and file existence.")
        return {"status": "no_data", "results": {}}

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    save_json(results, str(out / "exp2_results.json"))
    summary_md = _build_summary(results)
    (out / "exp2_summary.md").write_text(summary_md, encoding="utf-8")

    logger.info(
        "Saved exp2_results.json + exp2_summary.md → %s (%d datasets)",
        out, len(results),
    )
    return {"status": "ok", "results": results}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="AROMA Exp 2 — ROI 품질 비교 (AROMA vs Random)"
    )
    p.add_argument("--aroma_roi_dir",  required=True,
                   help="AROMA roi 출력 루트 디렉터리 (roi_candidates.json + roi_selected.json 포함)")
    p.add_argument("--random_roi_dir", required=True,
                   help="Random roi 출력 루트 디렉터리 (roi_selected.json 포함)")
    p.add_argument("--dataset_keys",   required=True, nargs="+",
                   help="비교할 데이터셋 키 목록 (예: isp_LSM_1 mvtec_cable)")
    p.add_argument("--output_dir",     required=True,
                   help="exp2_results.json, exp2_summary.md 저장 디렉터리")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = _parse_args(argv)
    result = run(
        aroma_roi_dir=args.aroma_roi_dir,
        random_roi_dir=args.random_roi_dir,
        dataset_keys=args.dataset_keys,
        output_dir=args.output_dir,
    )
    if result.get("status") != "ok":
        sys.exit(1)


if __name__ == "__main__":
    main()
