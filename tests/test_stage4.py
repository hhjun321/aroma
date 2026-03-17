# tests/test_stage4.py
import json, pytest, cv2
import numpy as np
from pathlib import Path

def _make_placement_map(tmp_path, synthetic_image, synthetic_defect):
    """Create a minimal placement_map.json and supporting files."""
    # Save background image
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    cv2.imwrite(str(img_dir / "img_001.png"), synthetic_image)

    # Save defect seed
    seeds_dir = tmp_path / "seeds"
    seeds_dir.mkdir()
    defect_path = seeds_dir / "variant_0000.png"
    cv2.imwrite(str(defect_path), synthetic_defect)

    # Create placement map
    placement_map = [
        {
            "image_id": "img_001",
            "placements": [
                {
                    "defect_path": str(defect_path),
                    "x": 20,
                    "y": 20,
                    "scale": 1.0,
                    "rotation": 0,
                    "suitability_score": 0.85,
                    "matched_background_type": "smooth"
                }
            ]
        }
    ]
    map_path = tmp_path / "placement_map.json"
    map_path.write_text(json.dumps(placement_map))
    return map_path, img_dir

def test_output_image_created(tmp_path, synthetic_image, synthetic_defect):
    from stage4_mpb_synthesis import run_synthesis
    map_path, img_dir = _make_placement_map(tmp_path, synthetic_image, synthetic_defect)
    out = tmp_path / "output"
    run_synthesis(str(map_path), str(img_dir), str(out), format="cls")
    output_images = list(out.rglob("*.png"))
    assert len(output_images) > 0, "No output images created"

def test_output_image_shape_matches(tmp_path, synthetic_image, synthetic_defect):
    from stage4_mpb_synthesis import run_synthesis
    map_path, img_dir = _make_placement_map(tmp_path, synthetic_image, synthetic_defect)
    out = tmp_path / "output"
    run_synthesis(str(map_path), str(img_dir), str(out), format="cls")
    output_images = list(out.rglob("*.png"))
    result = cv2.imread(str(output_images[0]))
    assert result.shape == synthetic_image.shape, "Output shape must match background"

def test_output_differs_from_background(tmp_path, synthetic_image, synthetic_defect):
    from stage4_mpb_synthesis import run_synthesis
    map_path, img_dir = _make_placement_map(tmp_path, synthetic_image, synthetic_defect)
    out = tmp_path / "output"
    run_synthesis(str(map_path), str(img_dir), str(out), format="cls")
    output_images = list(out.rglob("*.png"))
    result = cv2.imread(str(output_images[0]))
    diff = np.mean(np.abs(result.astype(float) - synthetic_image.astype(float)))
    assert diff > 0.5, "Output should differ from background after defect compositing"

def test_yolo_format_creates_labels(tmp_path, synthetic_image, synthetic_defect):
    from stage4_mpb_synthesis import run_synthesis
    map_path, img_dir = _make_placement_map(tmp_path, synthetic_image, synthetic_defect)
    out = tmp_path / "output_yolo"
    run_synthesis(str(map_path), str(img_dir), str(out), format="yolo")
    label_files = list(out.rglob("*.txt"))
    assert len(label_files) > 0, "YOLO format should produce label .txt files"
