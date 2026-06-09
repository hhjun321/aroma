"""
Unit tests for scripts/aroma/compute_complexity.py

Fixtures use synthetic in-memory data (no disk Phase 0 execution required).
~10 test cases covering: normalization guards, MCI/CCI correctness,
policy selection routing, stability tie-break, output schema.
"""
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

# Make scripts/aroma importable from tests/
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))

from aroma.compute_complexity import (
    _clamp01,
    _cluster_context,
    _compute_silhouette,
    _eval_policy_silhouette,
    _get_candidate_policies,
    _label_entropy,
    _normalize_array,
    _normalize_scalar,
    _total_valley_count,
    _WEIGHT_PRESETS,
    compute_mci,
    compute_cci,
    load_config,
    load_phase0_outputs,
    run,
    run_meta_policy_generator,
    select_best_policy,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DEFAULT_CFG = load_config(None)  # built-in defaults, no yaml needed


def _make_dist_analysis(n_valleys_per_feature: int = 0) -> dict:
    features = ["linearity", "solidity", "extent", "aspect_ratio", "eccentricity", "circularity"]
    return {
        f: {"policy": "percentile", "distribution": "unimodal", "n_valleys": n_valleys_per_feature}
        for f in features
    }


def _make_phase0(
    n_defects: int = 30,
    n_clusters: int = 3,
    n_patches: int = 100,
    n_valleys_per_feature: int = 0,
    constant_morphology: bool = False,
) -> dict:
    rng = np.random.default_rng(42)
    if constant_morphology:
        morph_X = np.ones((n_defects, 6), dtype=np.float64) * 0.5
    else:
        morph_X = rng.uniform(0, 1, size=(n_defects, 6))

    # Assign labels cycling through clusters
    morph_labels = np.array(
        [i % max(n_clusters, 1) for i in range(n_defects)], dtype=np.int32
    )
    context_X = rng.uniform(0, 1, size=(n_patches, 5))

    return {
        "morph_X":          morph_X,
        "morph_ids":        [f"img_{i}" for i in range(n_defects)],
        "morph_labels":     morph_labels,
        "context_X":        context_X,
        "dist_analysis":    _make_dist_analysis(n_valleys_per_feature),
        "n_morph_clusters": n_clusters,
        "status":           "ok",
    }


def _make_profiling_dir(tmp_path: Path, n_valleys: int = 0) -> Path:
    """Create a minimal Phase 0 output directory in tmp_path."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    n_defects, n_patches = 20, 60

    # morphology_features.csv
    morph_path = tmp_path / "morphology_features.csv"
    fields = ["image_id", "image_path", "defect_type", "domain", "mask_source",
              "linearity", "solidity", "extent", "aspect_ratio", "eccentricity", "circularity", "area"]
    with open(morph_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for i in range(n_defects):
            vals = rng.uniform(0, 1, 6)
            w.writerow({
                "image_id": f"img_{i}", "image_path": f"/data/img_{i}.png",
                "defect_type": "scratch", "domain": "isp", "mask_source": "gt",
                **{feat: f"{v:.6f}" for feat, v in zip(
                    ["linearity", "solidity", "extent", "aspect_ratio", "eccentricity", "circularity"], vals
                )},
                "area": "500",
            })

    # context_features.csv
    ctx_path = tmp_path / "context_features.csv"
    fields_ctx = ["image_id", "image_type", "patch_xy",
                  "local_variance", "edge_density", "texture_entropy",
                  "frequency_energy", "orientation_consistency"]
    with open(ctx_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields_ctx)
        w.writeheader()
        for i in range(n_patches):
            vals = rng.uniform(0, 1, 5)
            w.writerow({
                "image_id": f"img_{i % n_defects}", "image_type": "good",
                "patch_xy": f"{(i*64) % 512}_{(i*64) // 512}",
                **{feat: f"{v:.6f}" for feat, v in zip(
                    ["local_variance", "edge_density", "texture_entropy",
                     "frequency_energy", "orientation_consistency"], vals
                )},
            })

    # distribution_analysis.json
    da = {f: {"policy": "percentile", "distribution": "unimodal", "n_valleys": n_valleys}
          for f in ["linearity", "solidity", "extent", "aspect_ratio", "eccentricity", "circularity"]}
    with open(tmp_path / "distribution_analysis.json", "w") as f:
        json.dump(da, f)

    # morphology_clusters.json — assign each defect to one of 3 clusters
    assignments = {f"img_{i}": i % 3 for i in range(n_defects)}
    clusters_json = {
        "n_clusters": 3,
        "method": "gmm_bic",
        "clusters": [{"cluster_id": k, "n_samples": n_defects // 3, "centroid": {}, "label": f"c{k}"}
                     for k in range(3)],
        "cluster_assignments": assignments,
    }
    with open(tmp_path / "morphology_clusters.json", "w") as f:
        json.dump(clusters_json, f)

    return tmp_path


# ---------------------------------------------------------------------------
# 1. Normalization guards: NaN/Inf → 0, constant array → 0
# ---------------------------------------------------------------------------

def test_clamp01_nan_inf():
    assert _clamp01(float("nan")) == 0.0
    assert _clamp01(float("inf")) == 0.0
    assert _clamp01(-1.0) == 0.0
    assert _clamp01(2.0) == 1.0
    assert 0.0 <= _clamp01(0.5) <= 1.0


def test_normalize_scalar_constant():
    """span=0 → returns 0.0 (no division by zero)."""
    assert _normalize_scalar(5.0, 5.0, 5.0, "minmax") == 0.0


def test_normalize_array_constant_column():
    """All-same column → normalized column is all 0."""
    X = np.full((10, 3), 0.5)
    Xn = _normalize_array(X, "zscore")
    assert np.all(Xn == 0.0)
    assert np.all(np.isfinite(Xn))


def test_normalize_array_nan_input():
    """NaN in input → corresponding row becomes 0, others finite."""
    X = np.array([[1.0, np.nan], [2.0, 3.0], [3.0, 5.0]])
    Xn = _normalize_array(X, "zscore")
    assert np.all(np.isfinite(Xn))


# ---------------------------------------------------------------------------
# 2. MCI: finite output, entropy≈0 for constant morphology
# ---------------------------------------------------------------------------

def test_mci_finite():
    phase0 = _make_phase0(n_defects=30, n_clusters=3)
    mci, comps = compute_mci(phase0, _DEFAULT_CFG)
    assert math.isfinite(mci), f"MCI is not finite: {mci}"
    assert 0.0 <= mci <= 1.0, f"MCI out of [0,1]: {mci}"
    assert math.isfinite(comps["raw"]["entropy"])
    assert math.isfinite(comps["raw"]["silhouette"])


def test_mci_constant_morphology_entropy_zero():
    """All-identical defects → label entropy ≈ 0 (or very small)."""
    phase0 = _make_phase0(n_defects=20, n_clusters=1, constant_morphology=True)
    # With single cluster, entropy should be 0
    phase0["morph_labels"] = np.zeros(20, dtype=np.int32)
    mci, comps = compute_mci(phase0, _DEFAULT_CFG)
    assert math.isfinite(mci)
    assert comps["raw"]["entropy"] < 0.1, f"Entropy not ≈ 0: {comps['raw']['entropy']}"


def test_mci_weight_modes_all_finite():
    """All 3 weight modes produce finite MCI, sum of weights = 1."""
    phase0 = _make_phase0(n_defects=30)
    for mode in ("equal", "entropy_heavy", "cluster_heavy"):
        mci, comps = compute_mci(phase0, _DEFAULT_CFG, weight_mode=mode)
        assert math.isfinite(mci), f"MCI not finite for mode={mode}"
        w = comps["weights"]
        assert abs(sum(w) - 1.0) < 1e-6, f"Weights don't sum to 1: {w}"


# ---------------------------------------------------------------------------
# 3. CCI: finite output
# ---------------------------------------------------------------------------

def test_cci_finite():
    phase0 = _make_phase0(n_patches=80)
    cci, comps = compute_cci(phase0, _DEFAULT_CFG)
    assert math.isfinite(cci), f"CCI is not finite: {cci}"
    assert 0.0 <= cci <= 1.0


def test_cci_empty_context():
    """Empty context → CCI = 0.0, no exception."""
    phase0 = _make_phase0(n_patches=0)
    cci, _ = compute_cci(phase0, _DEFAULT_CFG)
    assert cci == 0.0


# ---------------------------------------------------------------------------
# 4. Policy routing: unimodal → percentile, bimodal → otsu, multimodal → gmm
# ---------------------------------------------------------------------------

def test_policy_unimodal_returns_percentile():
    """n_valleys=0 (all unimodal) → candidates should include percentile."""
    da = _make_dist_analysis(n_valleys_per_feature=0)
    cands = _get_candidate_policies(da, mci=0.1, cfg=_DEFAULT_CFG)
    assert "percentile" in cands
    assert "gmm" not in cands


def test_policy_bimodal_path():
    """Exactly 1 total valley (1 feature with 1 valley, rest 0) → bimodal branch ["otsu","percentile"]."""
    features = ["linearity", "solidity", "extent", "aspect_ratio", "eccentricity", "circularity"]
    da = {f: {"policy": "percentile", "distribution": "unimodal", "n_valleys": 0} for f in features}
    da["linearity"]["n_valleys"] = 1  # total_valleys = 1 == valley_threshold
    cands = _get_candidate_policies(da, mci=0.4, cfg=_DEFAULT_CFG)
    assert "otsu" in cands
    assert "gmm" not in cands


def test_policy_multimodal_returns_gmm():
    """total_valleys > threshold → gmm in candidates."""
    da = _make_dist_analysis(n_valleys_per_feature=3)
    cands = _get_candidate_policies(da, mci=0.5, cfg=_DEFAULT_CFG)
    assert "gmm" in cands


def test_policy_max_candidates_respected():
    """candidates never exceed max_candidates."""
    da = _make_dist_analysis(n_valleys_per_feature=3)
    for mci_val in (0.0, 0.5, 0.9):
        cands = _get_candidate_policies(da, mci=mci_val, cfg=_DEFAULT_CFG)
        max_c = _DEFAULT_CFG["policy"]["max_candidates"]
        assert len(cands) <= max_c, f"Exceeded max_candidates: {cands}"


# ---------------------------------------------------------------------------
# 5. Silhouette join: partial mismatch → no crash
# ---------------------------------------------------------------------------

def test_silhouette_partial_label_mismatch(tmp_path):
    """cluster_assignments that covers only half of CSV rows → ok, no crash."""
    profiling_dir = _make_profiling_dir(tmp_path)

    # Overwrite morphology_clusters.json to cover only half of images
    mc_path = profiling_dir / "morphology_clusters.json"
    with open(mc_path) as f:
        clusters_json = json.load(f)
    assignments = {k: v for k, v in list(clusters_json["cluster_assignments"].items())[:10]}
    clusters_json["cluster_assignments"] = assignments
    with open(mc_path, "w") as f:
        json.dump(clusters_json, f)

    phase0 = load_phase0_outputs(str(profiling_dir))
    assert phase0["status"] in ("ok", "empty_context", "empty_morph")
    if phase0.get("morph_labels") is not None:
        assert np.all(np.isfinite(phase0["morph_labels"].astype(float)))


# ---------------------------------------------------------------------------
# 6. Stability tie-break path
# ---------------------------------------------------------------------------

def test_stability_tiebreak_reached():
    """Force two policies to same silhouette → stability tie-break logic runs."""
    cfg = _deep_merge_cfg({"policy": {"stability_margin": 999.0}})  # huge epsilon forces tie
    phase0 = _make_phase0(n_defects=40)
    morph_Xn = phase0["morph_X"]
    result_policy, margin, results = select_best_policy(
        ["percentile", "otsu"], morph_Xn, cfg
    )
    assert isinstance(result_policy, str)
    assert math.isfinite(margin)


def _deep_merge_cfg(override: dict) -> dict:
    from aroma.compute_complexity import _deep_merge
    return _deep_merge(_DEFAULT_CFG, override)


# ---------------------------------------------------------------------------
# 7. Full run() end-to-end: output schema
# ---------------------------------------------------------------------------

def test_run_output_schema(tmp_path):
    """run() writes complexity_report.json with all required keys."""
    profiling_dir = _make_profiling_dir(tmp_path / "profiling")
    output_dir = tmp_path / "output"

    report = run(
        profiling_dir=str(profiling_dir),
        output_dir=str(output_dir),
        cfg=_DEFAULT_CFG,
    )

    required_keys = [
        "mci", "cci", "morphology_policy", "context_policy",
        "stability_margin", "weight_mode", "mci_components",
        "cci_components", "candidate_policies", "evaluation_results",
    ]
    for k in required_keys:
        assert k in report, f"Missing key in report: {k!r}"

    assert math.isfinite(report["mci"]), "mci not finite"
    assert math.isfinite(report["cci"]), "cci not finite"
    assert isinstance(report["morphology_policy"], str)
    assert (output_dir / "complexity_report.json").exists()


def test_run_missing_files_returns_status(tmp_path):
    """run() on empty directory → status != ok, no crash."""
    report = run(
        profiling_dir=str(tmp_path / "nonexistent"),
        output_dir=str(tmp_path / "out"),
        cfg=_DEFAULT_CFG,
    )
    assert report["status"] not in ("ok",)
    assert report.get("mci") is None


# ---------------------------------------------------------------------------
# 8. Label entropy edge cases
# ---------------------------------------------------------------------------

def test_label_entropy_empty():
    assert _label_entropy(np.array([], dtype=np.int32)) == 0.0


def test_label_entropy_single_cluster():
    labels = np.zeros(20, dtype=np.int32)
    assert _label_entropy(labels) < 0.01


def test_label_entropy_uniform():
    labels = np.arange(4, dtype=np.int32).repeat(5)
    entropy = _label_entropy(labels)
    assert math.isfinite(entropy)
    assert entropy > 0.0


# ---------------------------------------------------------------------------
# 9. Total valley count
# ---------------------------------------------------------------------------

def test_total_valley_count_zero():
    da = _make_dist_analysis(0)
    assert _total_valley_count(da) == 0


def test_total_valley_count_sum():
    da = _make_dist_analysis(2)
    assert _total_valley_count(da) == 12  # 6 features × 2


# ---------------------------------------------------------------------------
# 10. compute_mci/cci never produce NaN or Inf
# ---------------------------------------------------------------------------

def test_no_nan_inf_mci_cci_edge_cases():
    """Single defect, no context patches → all outputs finite."""
    phase0_single = {
        "morph_X":          np.array([[0.5, 0.5, 0.5, 1.5, 0.3, 0.8]]),
        "morph_ids":        ["img_0"],
        "morph_labels":     np.array([0], dtype=np.int32),
        "context_X":        np.empty((0, 5)),
        "dist_analysis":    _make_dist_analysis(0),
        "n_morph_clusters": 1,
        "status":           "empty_context",
    }
    mci, mci_c = compute_mci(phase0_single, _DEFAULT_CFG)
    cci, cci_c = compute_cci(phase0_single, _DEFAULT_CFG)
    assert math.isfinite(mci)
    assert math.isfinite(cci)
    for v in mci_c["raw"].values():
        assert math.isfinite(v), f"MCI raw term not finite: {v}"
    for v in cci_c["raw"].values():
        assert math.isfinite(v), f"CCI raw term not finite: {v}"
