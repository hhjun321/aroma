# tests/test_stage1.py
import json, pytest, cv2
import numpy as np
from pathlib import Path

def test_roi_metadata_schema(tmp_path, temp_image_dir):
    from stage1_roi_extraction import run_extraction
    output_dir = tmp_path / "output"
    run_extraction(str(temp_image_dir), str(output_dir), domain="mvtec", roi_levels="both")
    meta = json.loads((output_dir / "roi_metadata.json").read_text())
    assert isinstance(meta, list)
    entry = meta[0]
    for key in ("image_id", "global_mask", "local_masks", "roi_boxes"):
        assert key in entry

def test_roi_boxes_have_background_fields(tmp_path, temp_image_dir):
    from stage1_roi_extraction import run_extraction
    output_dir = tmp_path / "output"
    run_extraction(str(temp_image_dir), str(output_dir), domain="mvtec", roi_levels="both")
    meta = json.loads((output_dir / "roi_metadata.json").read_text())
    for box in meta[0]["roi_boxes"]:
        assert "background_type" in box, "Missing background_type"
        assert "continuity_score" in box, "Missing continuity_score"
        assert "stability_score" in box, "Missing stability_score"
        assert 0.0 <= box["continuity_score"] <= 1.0
        assert 0.0 <= box["stability_score"] <= 1.0

def test_global_mask_is_binary(tmp_path, temp_image_dir):
    from stage1_roi_extraction import run_extraction
    output_dir = tmp_path / "output"
    run_extraction(str(temp_image_dir), str(output_dir), domain="mvtec", roi_levels="global")
    meta = json.loads((output_dir / "roi_metadata.json").read_text())
    mask = cv2.imread(meta[0]["global_mask"], cv2.IMREAD_GRAYSCALE)
    assert mask is not None
    assert set(np.unique(mask)).issubset({0, 255})

def test_roi_boxes_within_image_bounds(tmp_path, temp_image_dir):
    from stage1_roi_extraction import run_extraction
    output_dir = tmp_path / "output"
    run_extraction(str(temp_image_dir), str(output_dir), domain="isp", roi_levels="local")
    meta = json.loads((output_dir / "roi_metadata.json").read_text())
    for entry in meta:
        for box in entry["roi_boxes"]:
            x, y, w, h = box["box"]
            assert x >= 0 and y >= 0
            assert x + w <= 64 and y + h <= 64
