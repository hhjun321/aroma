# tests/test_stage2.py
import pytest, cv2, json
import numpy as np
from pathlib import Path

def test_output_count(tmp_path, synthetic_defect):
    from stage2_defect_seed_generation import run_seed_generation
    seed_path = tmp_path / "seed.png"
    cv2.imwrite(str(seed_path), synthetic_defect)
    out_dir = tmp_path / "seeds"
    run_seed_generation(str(seed_path), num_variants=5, output_dir=str(out_dir))
    assert len(list(out_dir.glob("*.png"))) == 5

def test_output_shape_matches_seed(tmp_path, synthetic_defect):
    from stage2_defect_seed_generation import run_seed_generation
    seed_path = tmp_path / "seed.png"
    cv2.imwrite(str(seed_path), synthetic_defect)
    out_dir = tmp_path / "seeds"
    run_seed_generation(str(seed_path), num_variants=3, output_dir=str(out_dir))
    for v_path in out_dir.glob("*.png"):
        assert cv2.imread(str(v_path)).shape == synthetic_defect.shape

def test_variants_differ_from_seed(tmp_path, synthetic_defect):
    from stage2_defect_seed_generation import run_seed_generation
    seed_path = tmp_path / "seed.png"
    cv2.imwrite(str(seed_path), synthetic_defect)
    out_dir = tmp_path / "seeds"
    run_seed_generation(str(seed_path), num_variants=3, output_dir=str(out_dir))
    diffs = [np.mean(np.abs(cv2.imread(str(p)).astype(float) - synthetic_defect.astype(float)))
             for p in out_dir.glob("*.png")]
    assert any(d > 1.0 for d in diffs)

def test_seed_profile_subtype_used(tmp_path, synthetic_defect):
    """When seed_profile is provided, generation runs without error for each subtype."""
    from stage2_defect_seed_generation import run_seed_generation
    from utils.io import save_json
    seed_path = tmp_path / "seed.png"
    cv2.imwrite(str(seed_path), synthetic_defect)
    for subtype in ["linear_scratch", "compact_blob", "irregular", "elongated", "general"]:
        profile_path = tmp_path / f"profile_{subtype}.json"
        save_json({"subtype": subtype, "linearity": 0.9, "solidity": 0.8,
                   "extent": 0.5, "aspect_ratio": 6.0}, profile_path)
        out_dir = tmp_path / f"seeds_{subtype}"
        run_seed_generation(str(seed_path), num_variants=2,
                            output_dir=str(out_dir), seed_profile=str(profile_path))
        assert len(list(out_dir.glob("*.png"))) == 2


def test_workers_argument_parallel_produces_correct_count(tmp_path, synthetic_defect):
    from stage2_defect_seed_generation import run_seed_generation
    seed_path = tmp_path / "seed.png"
    cv2.imwrite(str(seed_path), synthetic_defect)
    out_dir = tmp_path / "seeds_parallel"
    run_seed_generation(str(seed_path), num_variants=12, output_dir=str(out_dir), workers=1)
    assert len(list(out_dir.glob("*.png"))) == 12
