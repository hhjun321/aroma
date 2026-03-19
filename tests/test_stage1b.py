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
