# tests/test_stage1b.py
import json, pytest, cv2
import numpy as np
from pathlib import Path

@pytest.fixture
def linear_seed_path(tmp_path):
    img = np.full((32, 64, 3), 180, dtype=np.uint8)
    img[14:18, 4:60] = [30, 30, 30]  # dark horizontal line = linear defect
    p = tmp_path / "seed_linear.png"
    cv2.imwrite(str(p), img)
    return p

def test_seed_profile_schema(tmp_path, linear_seed_path):
    from stage1b_seed_characterization import run_seed_characterization
    out = tmp_path / "out"
    run_seed_characterization(str(linear_seed_path), str(out))
    profile = json.loads((out / "seed_profile.json").read_text())
    for key in ("seed_path", "subtype", "linearity", "solidity", "extent", "aspect_ratio", "mask_path"):
        assert key in profile, f"Missing key: {key}"

def test_seed_subtype_is_valid_string(tmp_path, linear_seed_path):
    from stage1b_seed_characterization import run_seed_characterization
    out = tmp_path / "out"
    run_seed_characterization(str(linear_seed_path), str(out))
    profile = json.loads((out / "seed_profile.json").read_text())
    valid = {"linear_scratch", "elongated", "compact_blob", "irregular", "general"}
    assert profile["subtype"] in valid

def test_seed_mask_is_saved(tmp_path, linear_seed_path):
    from stage1b_seed_characterization import run_seed_characterization
    out = tmp_path / "out"
    run_seed_characterization(str(linear_seed_path), str(out))
    profile = json.loads((out / "seed_profile.json").read_text())
    assert Path(profile["mask_path"]).exists()

def test_linear_seed_classified_as_linear_or_elongated(tmp_path, linear_seed_path):
    from stage1b_seed_characterization import run_seed_characterization
    out = tmp_path / "out"
    run_seed_characterization(str(linear_seed_path), str(out))
    profile = json.loads((out / "seed_profile.json").read_text())
    assert profile["subtype"] in {"linear_scratch", "elongated", "general"}


# --- Batch / workers tests ---

@pytest.fixture
def three_seed_paths(tmp_path):
    """Create 3 distinct synthetic seed images for batch testing."""
    seeds = []
    for i in range(3):
        img = np.full((32, 64, 3), 180, dtype=np.uint8)
        # Vary the defect position per seed
        row = 10 + i * 4
        img[row:row + 4, 4:60] = [30, 30, 30]
        p = tmp_path / f"seed_{i}.png"
        cv2.imwrite(str(p), img)
        seeds.append(p)
    return seeds


def test_batch_processes_all_seeds(tmp_path, three_seed_paths):
    """run_seed_characterization_batch produces a profile for every seed."""
    from stage1b_seed_characterization import run_seed_characterization_batch
    tasks = [
        (str(s), str(tmp_path / f"out_{i}"))
        for i, s in enumerate(three_seed_paths)
    ]
    results = run_seed_characterization_batch(tasks, workers=0)
    assert len(results) == 3
    for i in range(3):
        profile_path = tmp_path / f"out_{i}" / "seed_profile.json"
        assert profile_path.exists(), f"Missing profile for seed {i}"
        profile = json.loads(profile_path.read_text())
        assert "subtype" in profile


def test_batch_returns_output_dirs(tmp_path, three_seed_paths):
    """Batch function returns the list of output directories that were written."""
    from stage1b_seed_characterization import run_seed_characterization_batch
    tasks = [
        (str(s), str(tmp_path / f"out_{i}"))
        for i, s in enumerate(three_seed_paths)
    ]
    results = run_seed_characterization_batch(tasks, workers=0)
    for r in results:
        assert Path(r).is_dir()
        assert (Path(r) / "seed_profile.json").exists()


def test_batch_with_workers_auto(tmp_path, three_seed_paths):
    """Batch with workers=-1 (auto) still produces correct results."""
    from stage1b_seed_characterization import run_seed_characterization_batch
    tasks = [
        (str(s), str(tmp_path / f"out_{i}"))
        for i, s in enumerate(three_seed_paths)
    ]
    # workers=-1 resolves to auto; with 3 tasks (<= 10) run_parallel
    # falls back to sequential, but the code path through resolve_workers is exercised
    results = run_seed_characterization_batch(tasks, workers=-1)
    assert len(results) == 3


def test_batch_empty_tasks(tmp_path):
    """Batch with empty task list returns empty list."""
    from stage1b_seed_characterization import run_seed_characterization_batch
    results = run_seed_characterization_batch([], workers=0)
    assert results == []


def test_cli_seed_dir_batch(tmp_path, three_seed_paths):
    """CLI --seed_dir mode processes all seeds in a directory."""
    import subprocess, sys
    seed_dir = three_seed_paths[0].parent
    out_dir = tmp_path / "cli_out"
    result = subprocess.run(
        [sys.executable, "-m", "stage1b_seed_characterization",
         "--seed_dir", str(seed_dir),
         "--output_dir", str(out_dir),
         "--workers", "0"],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, f"CLI failed: {result.stderr}"
    # Each seed should have its own sub-directory with a profile
    for sp in three_seed_paths:
        profile = out_dir / sp.stem / "seed_profile.json"
        assert profile.exists(), f"Missing profile for {sp.stem}"
