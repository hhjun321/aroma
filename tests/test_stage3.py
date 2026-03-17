# tests/test_stage3.py
import json, pytest, cv2
import numpy as np
from pathlib import Path

def _make_roi_metadata(tmp_path, synthetic_image):
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[10:50, 10:50] = 255
    mask_path = tmp_path / "masks" / "local" / "img_001_zone0.png"
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(mask_path), mask)
    meta = [{"image_id": "img_001",
             "global_mask": "",
             "local_masks": [str(mask_path)],
             "roi_boxes": [{"level": "local", "box": [10, 10, 40, 40], "zone_id": 0,
                            "background_type": "smooth",
                            "continuity_score": 0.8, "stability_score": 0.75}]}]
    meta_path = tmp_path / "roi_metadata.json"
    meta_path.write_text(json.dumps(meta))
    return meta_path

def _make_seeds_dir(tmp_path, synthetic_defect):
    seeds_dir = tmp_path / "seeds"
    seeds_dir.mkdir()
    cv2.imwrite(str(seeds_dir / "variant_0000.png"), synthetic_defect)
    return seeds_dir

def _make_seed_profile(tmp_path):
    from utils.io import save_json
    profile = {"subtype": "compact_blob", "linearity": 0.1, "solidity": 0.95,
               "extent": 0.8, "aspect_ratio": 1.2}
    p = tmp_path / "seed_profile.json"
    save_json(profile, p)
    return p

def test_placement_map_schema(tmp_path, synthetic_image, synthetic_defect):
    from stage3_layout_logic import run_layout_logic
    meta = _make_roi_metadata(tmp_path, synthetic_image)
    seeds = _make_seeds_dir(tmp_path, synthetic_defect)
    profile = _make_seed_profile(tmp_path)
    out = tmp_path / "output"
    run_layout_logic(str(meta), str(seeds), str(out), seed_profile=str(profile), domain="isp")
    data = json.loads((out / "placement_map.json").read_text())
    assert isinstance(data, list)
    p = data[0]["placements"][0]
    for key in ("defect_path", "x", "y", "scale", "rotation", "suitability_score", "matched_background_type"):
        assert key in p, f"Missing key: {key}"

def test_suitability_score_in_range(tmp_path, synthetic_image, synthetic_defect):
    from stage3_layout_logic import run_layout_logic
    meta = _make_roi_metadata(tmp_path, synthetic_image)
    seeds = _make_seeds_dir(tmp_path, synthetic_defect)
    profile = _make_seed_profile(tmp_path)
    out = tmp_path / "output"
    run_layout_logic(str(meta), str(seeds), str(out), seed_profile=str(profile), domain="isp")
    data = json.loads((out / "placement_map.json").read_text())
    for entry in data:
        for p in entry["placements"]:
            assert 0.0 <= p["suitability_score"] <= 1.0

def test_placement_within_roi_bounds(tmp_path, synthetic_image, synthetic_defect):
    from stage3_layout_logic import run_layout_logic
    meta = _make_roi_metadata(tmp_path, synthetic_image)
    seeds = _make_seeds_dir(tmp_path, synthetic_defect)
    out = tmp_path / "output"
    run_layout_logic(str(meta), str(seeds), str(out))
    data = json.loads((out / "placement_map.json").read_text())
    for entry in data:
        for p in entry["placements"]:
            assert 0 <= p["x"] <= 64
            assert 0 <= p["y"] <= 64
