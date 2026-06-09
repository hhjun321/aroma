"""
Unit tests for scripts/aroma/prompt_generation.py
"""
import json
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))

from aroma.prompt_generation import (
    _parse_cell_key,
    generate_context_descriptor,
    generate_morphology_descriptor,
    generate_prior_modifier,
    assemble_prompt,
    generate_prompts,
    load_inputs,
    run,
    CONTEXT_FEATURES,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_profiling_dir(tmp_path: Path) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)

    # 2 clusters, 3 context bins
    clusters = {
        "n_clusters": 2,
        "method": "gmm_bic",
        "clusters": [
            {
                "cluster_id": 0,
                "n_samples": 20,
                "centroid": {
                    "linearity": 0.8, "solidity": 0.85,
                    "extent": 0.6, "aspect_ratio": 4.5,
                    "eccentricity": 0.9, "circularity": 0.3,
                },
                "label": "linear_scratch",
            },
            {
                "cluster_id": 1,
                "n_samples": 10,
                "centroid": {
                    "linearity": 0.2, "solidity": 0.4,
                    "extent": 0.5, "aspect_ratio": 1.1,
                    "eccentricity": 0.3, "circularity": 0.8,
                },
                "label": "compact_blob",
            },
        ],
        "cluster_assignments": {f"img_{i}": i % 2 for i in range(30)},
    }

    matrix = {
        "n_clusters": 2,
        "n_context_bins": 3,
        "context_features": CONTEXT_FEATURES,
        "bin_edges": {f: [0.33, 0.67] for f in CONTEXT_FEATURES},
        "matrix": {
            "0": {"0_1_2_0_1": 0.55, "1_0_1_2_0": 0.20, "2_2_0_1_1": 0.08},
            "1": {"0_1_2_0_1": 0.10, "1_0_1_2_0": 0.45, "2_2_0_1_1": 0.30},
        },
    }

    deficit = {
        "0": {
            "prior": 0.65,
            "deficit": {"0_1_2_0_1": 0.05, "1_0_1_2_0": 0.40, "2_2_0_1_1": 0.70},
            "target_synthetic": {"0_1_2_0_1": 0.05, "1_0_1_2_0": 0.40, "2_2_0_1_1": 0.55},
        },
        "1": {
            "prior": 0.35,
            "deficit": {"0_1_2_0_1": 0.60, "1_0_1_2_0": 0.10, "2_2_0_1_1": 0.20},
            "target_synthetic": {"0_1_2_0_1": 0.60, "1_0_1_2_0": 0.10, "2_2_0_1_1": 0.30},
        },
    }

    with open(tmp_path / "morphology_clusters.json", "w") as f:
        json.dump(clusters, f)
    with open(tmp_path / "compatibility_matrix.json", "w") as f:
        json.dump(matrix, f)
    with open(tmp_path / "deficit_analysis.json", "w") as f:
        json.dump(deficit, f)

    return tmp_path


def _make_complexity_dir(tmp_path: Path) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    report = {
        "mci": 0.72, "cci": 0.45,
        "morphology_policy": "gmm",
        "context_policy": "gmm",
        "stability_margin": 0.03,
        "weight_mode": "equal",
        "status": "ok",
    }
    with open(tmp_path / "complexity_report.json", "w") as f:
        json.dump(report, f)
    return tmp_path


# ---------------------------------------------------------------------------
# 1. cell key parsing
# ---------------------------------------------------------------------------

def test_parse_cell_key_normal():
    bins = _parse_cell_key("0_1_2_0_1")
    assert bins == [0, 1, 2, 0, 1]


def test_parse_cell_key_invalid():
    bins = _parse_cell_key("bad_key")
    assert all(b == 0 for b in bins)


# ---------------------------------------------------------------------------
# 2. Morphology descriptor templates
# ---------------------------------------------------------------------------

def test_morphology_highly_elongated_linear():
    centroid = {"aspect_ratio": 4.5, "linearity": 0.8, "solidity": 0.9}
    desc = generate_morphology_descriptor(centroid)
    assert "highly elongated" in desc
    assert "linear" in desc
    assert "solid" in desc


def test_morphology_compact_scattered():
    centroid = {"aspect_ratio": 1.0, "linearity": 0.1, "solidity": 0.3}
    desc = generate_morphology_descriptor(centroid)
    assert "compact" in desc
    assert "scattered" in desc
    assert "fragmented" in desc


def test_morphology_default_values():
    """Missing centroid keys → no crash, returns non-empty string."""
    desc = generate_morphology_descriptor({})
    assert isinstance(desc, str)
    assert len(desc) > 0


# ---------------------------------------------------------------------------
# 3. Context descriptor
# ---------------------------------------------------------------------------

def test_context_descriptor_format():
    desc = generate_context_descriptor("0_1_2_0_1")
    assert isinstance(desc, str)
    assert "background" in desc


def test_context_descriptor_all_high():
    """All bins = 2 → highest label for dominant features."""
    desc = generate_context_descriptor("2_2_2_2_2")
    assert isinstance(desc, str)
    assert len(desc) > 0


def test_context_descriptor_all_low():
    """All bins = 0 → produces a valid descriptor."""
    desc = generate_context_descriptor("0_0_0_0_0")
    assert isinstance(desc, str)
    assert len(desc) > 0


def test_context_descriptor_wrong_length():
    """Malformed cell key → no crash."""
    desc = generate_context_descriptor("0_1")
    assert isinstance(desc, str)


# ---------------------------------------------------------------------------
# 4. Prior modifier thresholds
# ---------------------------------------------------------------------------

def test_prior_modifier_high():
    matrix = {"0": {"0_1_2_0_1": 0.55}}
    mod = generate_prior_modifier(0, "0_1_2_0_1", matrix)
    assert "predominantly" in mod


def test_prior_modifier_medium():
    matrix = {"0": {"0_1_2_0_1": 0.25}}
    mod = generate_prior_modifier(0, "0_1_2_0_1", matrix)
    assert "commonly" in mod


def test_prior_modifier_low():
    matrix = {"0": {"0_1_2_0_1": 0.05}}
    mod = generate_prior_modifier(0, "0_1_2_0_1", matrix)
    assert "rarely" in mod


def test_prior_modifier_missing_cluster():
    mod = generate_prior_modifier(99, "0_1_2_0_1", {})
    assert "rarely" in mod


# ---------------------------------------------------------------------------
# 5. Prompt assembly
# ---------------------------------------------------------------------------

def test_assemble_prompt_joins():
    p = assemble_prompt("elongated defect", "smooth surface background", "rarely observed")
    assert "elongated defect" in p
    assert "smooth surface background" in p
    assert "rarely observed" in p


# ---------------------------------------------------------------------------
# 6. generate_prompts
# ---------------------------------------------------------------------------

def test_generate_prompts_count(tmp_path):
    profiling_dir = _make_profiling_dir(tmp_path / "profiling")
    data = {
        "status": "ok",
        "clusters": json.loads((profiling_dir / "morphology_clusters.json").read_text()),
        "compat_matrix": json.loads((profiling_dir / "compatibility_matrix.json").read_text()),
        "deficit_analysis": json.loads((profiling_dir / "deficit_analysis.json").read_text()),
    }
    prompts = generate_prompts(data)
    # 2 clusters × 3 cells each = 6
    assert len(prompts) == 6


def test_generate_prompts_schema(tmp_path):
    profiling_dir = _make_profiling_dir(tmp_path / "profiling")
    data = {
        "status": "ok",
        "clusters": json.loads((profiling_dir / "morphology_clusters.json").read_text()),
        "compat_matrix": json.loads((profiling_dir / "compatibility_matrix.json").read_text()),
        "deficit_analysis": json.loads((profiling_dir / "deficit_analysis.json").read_text()),
    }
    prompts = generate_prompts(data)
    required_keys = [
        "morphology_descriptor", "context_descriptor", "prior_modifier",
        "prompt", "cluster_id", "cell_key", "prior_prob", "deficit",
    ]
    for key, entry in prompts.items():
        for k in required_keys:
            assert k in entry, f"Missing key {k!r} in entry {key!r}"
        assert isinstance(entry["prompt"], str) and len(entry["prompt"]) > 0
        assert math.isfinite(entry["prior_prob"])
        assert math.isfinite(entry["deficit"])


def test_generate_prompts_empty_matrix():
    """No compatibility matrix → falls back to 'none' cell per cluster."""
    data = {
        "status": "ok",
        "clusters": {
            "clusters": [
                {"cluster_id": 0, "n_samples": 5,
                 "centroid": {"aspect_ratio": 2.0, "linearity": 0.5, "solidity": 0.6},
                 "label": "general"},
            ]
        },
        "compat_matrix": {"matrix": {}},
        "deficit_analysis": {},
    }
    prompts = generate_prompts(data)
    assert len(prompts) == 1
    key = list(prompts.keys())[0]
    assert key == "0_none"


def test_generate_prompts_empty_clusters():
    data = {
        "status": "ok",
        "clusters": {"clusters": []},
        "compat_matrix": {"matrix": {}},
        "deficit_analysis": {},
    }
    prompts = generate_prompts(data)
    assert prompts == {}


# ---------------------------------------------------------------------------
# 7. load_inputs — missing files
# ---------------------------------------------------------------------------

def test_load_inputs_missing(tmp_path):
    result = load_inputs(str(tmp_path / "missing"), str(tmp_path / "complexity"))
    assert result["status"] == "missing_inputs"
    assert len(result["missing"]) > 0


# ---------------------------------------------------------------------------
# 8. Full run() — output schema
# ---------------------------------------------------------------------------

def test_run_output_schema(tmp_path):
    profiling_dir = _make_profiling_dir(tmp_path / "profiling")
    complexity_dir = _make_complexity_dir(tmp_path / "complexity")
    output_dir = tmp_path / "output"

    result = run(
        profiling_dir=str(profiling_dir),
        complexity_dir=str(complexity_dir),
        output_dir=str(output_dir),
    )

    assert isinstance(result, dict)
    assert len(result) == 6  # 2 clusters × 3 cells

    assert (output_dir / "prompts.json").exists()
    assert (output_dir / "prompts_summary.md").exists()

    saved = json.loads((output_dir / "prompts.json").read_text())
    assert len(saved) == 6


def test_run_missing_profiling(tmp_path):
    result = run(
        profiling_dir=str(tmp_path / "nope"),
        complexity_dir=str(tmp_path / "complexity"),
        output_dir=str(tmp_path / "out"),
    )
    assert result.get("status") == "missing_inputs"


# ---------------------------------------------------------------------------
# 9. Prompt content sanity
# ---------------------------------------------------------------------------

def test_prompt_not_empty(tmp_path):
    profiling_dir = _make_profiling_dir(tmp_path / "profiling")
    data = {
        "status": "ok",
        "clusters": json.loads((profiling_dir / "morphology_clusters.json").read_text()),
        "compat_matrix": json.loads((profiling_dir / "compatibility_matrix.json").read_text()),
        "deficit_analysis": json.loads((profiling_dir / "deficit_analysis.json").read_text()),
    }
    prompts = generate_prompts(data)
    for key, entry in prompts.items():
        p = entry["prompt"]
        assert len(p) > 20, f"Prompt too short for {key}: {p!r}"
        assert ", " in p, f"Prompt missing separator for {key}: {p!r}"
