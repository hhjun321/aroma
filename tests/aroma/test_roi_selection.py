"""
Unit tests for scripts/aroma/roi_selection.py
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))

from aroma.roi_selection import (
    _context_cell_key,
    _rare_pair_deficit_quantile,
    build_candidates,
    load_inputs,
    run,
    score_roi,
    select_rois,
)
from aroma.prompt_generation import CONTEXT_FEATURES


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_BIN_EDGES = {f: [0.33, 0.67] for f in CONTEXT_FEATURES}
_CELL_HIGH = "2_2_2_2_2"
_CELL_LOW  = "0_0_0_0_0"
_CELL_MID  = "1_1_1_1_1"


def _make_morph_rows(n: int = 10) -> list:
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n):
        vals = rng.uniform(0, 1, len(CONTEXT_FEATURES) + 6)
        row = {
            "image_id":   f"img_{i}",
            "image_path": f"/data/img_{i}.png",
            "defect_type": "scratch", "domain": "isp", "mask_source": "gt",
            "area": "500",
        }
        for j, feat in enumerate(["linearity","solidity","extent","aspect_ratio","eccentricity","circularity"]):
            row[feat] = str(vals[j])
        for j, feat in enumerate(CONTEXT_FEATURES):
            row[feat] = str(vals[6 + j])
        rows.append(row)
    return rows


def _make_data(n_rows: int = 10, n_clusters: int = 2) -> dict:
    rows = _make_morph_rows(n_rows)
    assignments = {f"img_{i}": i % n_clusters for i in range(n_rows)}
    clusters = {
        "n_clusters": n_clusters,
        "clusters": [
            {"cluster_id": k, "n_samples": n_rows // n_clusters,
             "centroid": {f: 0.5 for f in ["linearity","solidity","extent","aspect_ratio","eccentricity","circularity"]},
             "label": f"cluster_{k}"}
            for k in range(n_clusters)
        ],
        "cluster_assignments": assignments,
    }
    matrix = {
        "n_clusters": n_clusters,
        "n_context_bins": 3,
        "context_features": CONTEXT_FEATURES,
        "bin_edges": _BIN_EDGES,
        "matrix": {
            str(k): {_CELL_MID: 0.5, _CELL_LOW: 0.3, _CELL_HIGH: 0.2}
            for k in range(n_clusters)
        },
    }
    deficit = {
        str(k): {
            "prior": 1.0 / n_clusters,
            "deficit": {_CELL_MID: 0.1, _CELL_LOW: 0.5, _CELL_HIGH: 0.8},
            "target_synthetic": {_CELL_MID: 0.1, _CELL_LOW: 0.4, _CELL_HIGH: 0.5},
        }
        for k in range(n_clusters)
    }
    prompts = {
        f"{k}_{_CELL_MID}": {"prompt": f"cluster {k} mid context", "context_descriptor": "mid surface background"}
        for k in range(n_clusters)
    }
    return {
        "status":          "ok",
        "morph_rows":      rows,
        "clusters":        clusters,
        "compat_matrix":   matrix,
        "deficit_analysis": deficit,
        "prompts":         prompts,
    }


def _make_disk_fixtures(tmp_path: Path, n_rows: int = 10) -> tuple:
    prof_dir = tmp_path / "profiling"
    prom_dir = tmp_path / "prompts"
    prof_dir.mkdir(parents=True)
    prom_dir.mkdir(parents=True)

    data = _make_data(n_rows)

    # morphology_features.csv
    fields = ["image_id", "image_path", "defect_type", "domain", "mask_source",
              "linearity","solidity","extent","aspect_ratio","eccentricity","circularity","area"]
    fields += CONTEXT_FEATURES
    with open(prof_dir / "morphology_features.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in data["morph_rows"]:
            # fill missing context cols from morph_rows (they have context cols too)
            w.writerow({k: row.get(k, "") for k in fields})

    for fname, obj in [
        ("morphology_clusters.json",  data["clusters"]),
        ("compatibility_matrix.json", data["compat_matrix"]),
        ("deficit_analysis.json",     data["deficit_analysis"]),
    ]:
        with open(prof_dir / fname, "w") as f:
            json.dump(obj, f)

    with open(prom_dir / "prompts.json", "w") as f:
        json.dump(data["prompts"], f)

    return prof_dir, prom_dir


# ---------------------------------------------------------------------------
# 1. ROI score formula
# ---------------------------------------------------------------------------

def test_score_roi_formula():
    s = score_roi(morph_prior=1.0, ctx_prior=1.0, deficit=1.0)
    assert abs(s - 1.0) < 1e-6

    s = score_roi(morph_prior=0.0, ctx_prior=0.0, deficit=0.0)
    assert abs(s - 0.0) < 1e-6


def test_score_roi_weights():
    s = score_roi(morph_prior=1.0, ctx_prior=0.0, deficit=0.0)
    assert abs(s - 0.4) < 1e-6

    s = score_roi(morph_prior=0.0, ctx_prior=1.0, deficit=0.0)
    assert abs(s - 0.4) < 1e-6

    s = score_roi(morph_prior=0.0, ctx_prior=0.0, deficit=1.0)
    assert abs(s - 0.2) < 1e-6


def test_score_roi_clamp():
    s = score_roi(-1.0, 2.0, -0.5)
    assert 0.0 <= s <= 1.0


# ---------------------------------------------------------------------------
# 2. Context cell key discretization
# ---------------------------------------------------------------------------

def test_context_cell_key_all_low():
    feat = {f: "0.0" for f in CONTEXT_FEATURES}
    key = _context_cell_key(feat, _BIN_EDGES, CONTEXT_FEATURES)
    assert key == _CELL_LOW


def test_context_cell_key_all_high():
    feat = {f: "1.0" for f in CONTEXT_FEATURES}
    key = _context_cell_key(feat, _BIN_EDGES, CONTEXT_FEATURES)
    assert key == _CELL_HIGH


def test_context_cell_key_missing_feature():
    feat = {}
    key = _context_cell_key(feat, _BIN_EDGES, CONTEXT_FEATURES)
    assert len(key.split("_")) == len(CONTEXT_FEATURES)


# ---------------------------------------------------------------------------
# 3. build_candidates
# ---------------------------------------------------------------------------

def test_build_candidates_count():
    data = _make_data(n_rows=10, n_clusters=2)
    cands = build_candidates(data)
    assert len(cands) == 10


def test_build_candidates_schema():
    data = _make_data(n_rows=5)
    cands = build_candidates(data)
    required = ["image_id","image_path","cluster_id","cell_key",
                "roi_score","morph_prior","ctx_prior","deficit","prompt"]
    for c in cands:
        for k in required:
            assert k in c, f"Missing {k!r}"
        assert 0.0 <= c["roi_score"] <= 1.0


def test_build_candidates_no_assignment():
    """Rows with no cluster assignment are skipped."""
    data = _make_data(n_rows=5)
    data["clusters"]["cluster_assignments"] = {}  # clear all
    cands = build_candidates(data)
    assert len(cands) == 0


def test_build_candidates_roi_score_finite():
    data = _make_data(n_rows=20)
    cands = build_candidates(data)
    for c in cands:
        assert 0.0 <= c["roi_score"] <= 1.0
        assert all(v is not None for v in c.values())


# ---------------------------------------------------------------------------
# 4. Sampling strategies
# ---------------------------------------------------------------------------

def test_select_rois_top_k():
    data = _make_data(n_rows=20)
    cands = build_candidates(data)
    selected = select_rois(cands, strategy="top_k", top_k=5)
    assert len(selected) == 5
    scores = [c["roi_score"] for c in selected]
    assert scores == sorted(scores, reverse=True)


def test_select_rois_weighted():
    data = _make_data(n_rows=20)
    cands = build_candidates(data)
    selected = select_rois(cands, strategy="weighted", top_k=5, seed=0)
    assert len(selected) == 5


def test_select_rois_deficit_aware():
    data = _make_data(n_rows=20)
    cands = build_candidates(data)
    selected = select_rois(cands, strategy="deficit_aware", top_k=5)
    assert len(selected) == 5


def test_select_rois_top_k_gt_candidates():
    data = _make_data(n_rows=5)
    cands = build_candidates(data)
    selected = select_rois(cands, strategy="top_k", top_k=100)
    assert len(selected) == len(cands)


def test_select_rois_empty():
    assert select_rois([], strategy="top_k", top_k=10) == []


# ---------------------------------------------------------------------------
# 5. Rare pair deficit quantile
# ---------------------------------------------------------------------------

def test_rare_pair_prefers_high_deficit():
    cands = [
        {"roi_score": 0.5, "deficit": 0.9},
        {"roi_score": 0.8, "deficit": 0.1},
        {"roi_score": 0.6, "deficit": 0.8},
    ]
    selected = _rare_pair_deficit_quantile(cands, top_k=2)
    # high-deficit items should be preferred
    deficits = [c["deficit"] for c in selected]
    assert max(deficits) >= 0.8


# ---------------------------------------------------------------------------
# 6. load_inputs
# ---------------------------------------------------------------------------

def test_load_inputs_missing(tmp_path):
    result = load_inputs(str(tmp_path / "nope"), str(tmp_path / "nope2"))
    assert result["status"] == "missing_inputs"


def test_load_inputs_ok(tmp_path):
    prof, prom = _make_disk_fixtures(tmp_path)
    result = load_inputs(str(prof), str(prom))
    assert result["status"] == "ok"
    assert len(result["morph_rows"]) == 10


# ---------------------------------------------------------------------------
# 7. Full run()
# ---------------------------------------------------------------------------

def test_run_output_schema(tmp_path):
    prof, prom = _make_disk_fixtures(tmp_path)
    out = tmp_path / "roi"

    result = run(
        profiling_dir=str(prof),
        prompts_dir=str(prom),
        output_dir=str(out),
        strategy="deficit_aware",
        top_k=5,
    )

    assert result["status"] == "ok"
    assert len(result["candidates"]) == 10
    assert len(result["selected"]) == 5

    assert (out / "roi_candidates.json").exists()
    assert (out / "roi_selected.json").exists()
    assert (out / "roi_summary.md").exists()


def test_run_all_strategies(tmp_path):
    for strategy in ("deficit_aware", "top_k", "weighted"):
        prof, prom = _make_disk_fixtures(tmp_path / strategy)
        out = tmp_path / f"roi_{strategy}"
        result = run(
            profiling_dir=str(prof),
            prompts_dir=str(prom),
            output_dir=str(out),
            strategy=strategy,
            top_k=5,
        )
        assert result["status"] == "ok", f"Strategy {strategy} failed"
        assert len(result["selected"]) == 5


def test_run_missing_inputs(tmp_path):
    result = run(
        profiling_dir=str(tmp_path / "missing"),
        prompts_dir=str(tmp_path / "missing2"),
        output_dir=str(tmp_path / "out"),
    )
    assert result["status"] == "missing_inputs"
