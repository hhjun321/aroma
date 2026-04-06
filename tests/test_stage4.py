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
    run_synthesis(str(img_dir), str(map_path), str(out), format="cls")
    output_images = list(out.rglob("*.png"))
    assert len(output_images) > 0, "No output images created"

def test_output_image_shape_matches(tmp_path, synthetic_image, synthetic_defect):
    from stage4_mpb_synthesis import run_synthesis
    map_path, img_dir = _make_placement_map(tmp_path, synthetic_image, synthetic_defect)
    out = tmp_path / "output"
    run_synthesis(str(img_dir), str(map_path), str(out), format="cls")
    output_images = list(out.rglob("*.png"))
    result = cv2.imread(str(output_images[0]))
    assert result.shape == synthetic_image.shape, "Output shape must match background"

def test_output_differs_from_background(tmp_path, synthetic_image, synthetic_defect):
    from stage4_mpb_synthesis import run_synthesis
    map_path, img_dir = _make_placement_map(tmp_path, synthetic_image, synthetic_defect)
    out = tmp_path / "output"
    run_synthesis(str(img_dir), str(map_path), str(out), format="cls")
    output_images = list(out.rglob("*.png"))
    result = cv2.imread(str(output_images[0]))
    diff = np.mean(np.abs(result.astype(float) - synthetic_image.astype(float)))
    assert diff > 0.5, "Output should differ from background after defect compositing"

def test_yolo_format_creates_labels(tmp_path, synthetic_image, synthetic_defect):
    from stage4_mpb_synthesis import run_synthesis
    map_path, img_dir = _make_placement_map(tmp_path, synthetic_image, synthetic_defect)
    out = tmp_path / "output_yolo"
    run_synthesis(str(img_dir), str(map_path), str(out), format="yolo")
    label_files = list(out.rglob("*.txt"))
    assert len(label_files) > 0, "YOLO format should produce label .txt files"


def test_workers_argument_accepted(tmp_path, temp_image_dir, synthetic_defect):
    import json, cv2
    from stage4_mpb_synthesis import run_synthesis
    from utils.io import save_json

    defect_path = tmp_path / "defect_seeds" / "v_000.png"
    defect_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(defect_path), synthetic_defect)

    placement_map = [{"image_id": "img_001", "placements": [{
        "defect_path": str(defect_path),
        "x": 10, "y": 10, "scale": 1.0, "rotation": 0,
        "suitability_score": 0.8, "matched_background_type": "smooth",
    }]}]
    placement_path = tmp_path / "placement_map.json"
    save_json(placement_map, placement_path)

    out_dir = tmp_path / "output"
    run_synthesis(str(temp_image_dir), str(placement_path), str(out_dir),
                  format="cls", workers=1)
    assert len(list((out_dir / "defect").glob("*.png"))) > 0


# ---------------------------------------------------------------------------
# Tests for Method B: _blend_patch_fast (Gaussian soft-mask compositing)
# ---------------------------------------------------------------------------

def test_fast_blend_produces_output(synthetic_image, synthetic_defect):
    from stage4_mpb_synthesis import _blend_patch_fast
    result = _blend_patch_fast(synthetic_image, synthetic_defect, 20, 20)
    assert result is not None
    assert result.shape == synthetic_image.shape


def test_fast_blend_output_differs_from_background(synthetic_image, synthetic_defect):
    from stage4_mpb_synthesis import _blend_patch_fast
    result = _blend_patch_fast(synthetic_image, synthetic_defect, 20, 20)
    diff = np.mean(np.abs(result.astype(float) - synthetic_image.astype(float)))
    assert diff > 0.1, "Fast-blend output should differ from the background"


# ---------------------------------------------------------------------------
# Tests for Method C: run_synthesis_batch (background caching, multi-seed)
# ---------------------------------------------------------------------------

def _make_multi_seed_setup(tmp_path, synthetic_image, synthetic_defect):
    """Set up 2 seeds × 1 background image for batch test."""
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    cv2.imwrite(str(img_dir / "img_001.png"), synthetic_image)

    seeds_dir = tmp_path / "seeds"
    seeds_dir.mkdir()

    placement_maps = []
    for seed_id in ("seed_0001", "seed_0002"):
        seed_variant = seeds_dir / seed_id / "variant_0000.png"
        seed_variant.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(seed_variant), synthetic_defect)

        pm = [{"image_id": "img_001", "placements": [{
            "defect_path": str(seed_variant),
            "x": 10, "y": 10, "scale": 1.0, "rotation": 0,
            "suitability_score": 0.8, "matched_background_type": "smooth",
        }]}]
        pm_path = tmp_path / f"placement_map_{seed_id}.json"
        pm_path.write_text(json.dumps(pm))
        placement_maps.append((seed_id, str(pm_path)))

    return img_dir, placement_maps


def test_run_synthesis_batch_creates_outputs_for_all_seeds(
        tmp_path, synthetic_image, synthetic_defect):
    from stage4_mpb_synthesis import run_synthesis_batch
    img_dir, placement_maps = _make_multi_seed_setup(
        tmp_path, synthetic_image, synthetic_defect)
    out_root = tmp_path / "output"
    run_synthesis_batch(str(img_dir), placement_maps, str(out_root), format="cls")
    for seed_id, _ in placement_maps:
        out_img = out_root / seed_id / "defect" / "img_001.png"
        assert out_img.exists(), f"Expected output image for {seed_id} at {out_img}"


def test_run_synthesis_batch_use_fast_blend(
        tmp_path, synthetic_image, synthetic_defect):
    from stage4_mpb_synthesis import run_synthesis_batch
    img_dir, placement_maps = _make_multi_seed_setup(
        tmp_path, synthetic_image, synthetic_defect)
    out_root = tmp_path / "output_fast"
    run_synthesis_batch(str(img_dir), placement_maps, str(out_root),
                        format="cls", use_fast_blend=True)
    for seed_id, _ in placement_maps:
        out_img = out_root / seed_id / "defect" / "img_001.png"
        assert out_img.exists(), f"Expected fast-blend output for {seed_id}"


# ---------------------------------------------------------------------------
# Prerequisite validation tests (B1)
# ---------------------------------------------------------------------------

def test_missing_image_dir_raises(tmp_path, synthetic_image, synthetic_defect):
    """Stage 4 should raise FileNotFoundError for missing image_dir."""
    from stage4_mpb_synthesis import run_synthesis
    map_path, _ = _make_placement_map(tmp_path, synthetic_image, synthetic_defect)
    with pytest.raises(FileNotFoundError, match="Background image_dir"):
        run_synthesis(
            str(tmp_path / "nonexistent_images"),
            str(map_path),
            str(tmp_path / "out"),
        )


def test_missing_placement_map_raises(tmp_path, synthetic_image, synthetic_defect):
    """Stage 4 should raise FileNotFoundError for missing placement_map."""
    from stage4_mpb_synthesis import run_synthesis
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    cv2.imwrite(str(img_dir / "img_001.png"), synthetic_image)
    with pytest.raises(FileNotFoundError, match="Stage 3 placement_map"):
        run_synthesis(
            str(img_dir),
            str(tmp_path / "nonexistent_pm.json"),
            str(tmp_path / "out"),
        )


def test_batch_missing_placement_map_raises(tmp_path, synthetic_image, synthetic_defect):
    """run_synthesis_batch should raise for missing placement map files."""
    import warnings
    from stage4_mpb_synthesis import run_synthesis_batch
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    cv2.imwrite(str(img_dir / "img_001.png"), synthetic_image)
    bad_maps = [("seed_0001", str(tmp_path / "nonexistent.json"))]
    with pytest.raises(FileNotFoundError, match="Stage 3 placement_map"):
        run_synthesis_batch(str(img_dir), bad_maps, str(tmp_path / "out"))


# ---------------------------------------------------------------------------
# Format compatibility warning test (B6)
# ---------------------------------------------------------------------------

def test_yolo_format_emits_stage6_warning(tmp_path, synthetic_image, synthetic_defect):
    """Stage 4 should warn when format='yolo' about Stage 6 incompatibility."""
    import warnings
    from stage4_mpb_synthesis import run_synthesis
    map_path, img_dir = _make_placement_map(tmp_path, synthetic_image, synthetic_defect)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        run_synthesis(str(img_dir), str(map_path), str(tmp_path / "out_yolo"),
                      format="yolo")
    yolo_warnings = [x for x in w if "format='yolo'" in str(x.message)]
    assert len(yolo_warnings) >= 1, "Expected warning about yolo/Stage 6 incompatibility"
