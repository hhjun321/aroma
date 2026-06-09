"""
Unit tests for scripts/aroma/generate_defects.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))

from aroma.generate_defects import (
    _random_paste_position,
    copy_paste_synthesis,
    load_normal_images,
    run,
    HAS_PIL,
)

import random


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_roi_selected(tmp_path: Path, n: int = 3) -> Path:
    roi_dir = tmp_path / "roi"
    roi_dir.mkdir(parents=True)
    selected = [
        {
            "image_id":   f"img_{i}",
            "image_path": str(tmp_path / f"defect_{i}.jpg"),
            "cluster_id": i % 2,
            "cell_key":   "1_1_1_1_1",
            "roi_score":  0.5 + i * 0.1,
            "deficit":    0.3 + i * 0.1,
            "prompt":     f"test defect {i}",
        }
        for i in range(n)
    ]
    with open(roi_dir / "roi_selected.json", "w") as f:
        json.dump(selected, f)
    return roi_dir


def _make_pil_image(tmp_path: Path, name: str, size=(64, 64)) -> Path:
    """Create a small test image if PIL is available."""
    p = tmp_path / name
    if HAS_PIL:
        from PIL import Image as PILImage
        img = PILImage.fromarray(
            (np.random.randint(0, 255, (*size[::-1], 3), dtype=np.uint8))
        )
        img.save(str(p))
    else:
        p.write_bytes(b"fake")
    return p


# ---------------------------------------------------------------------------
# 1. Paste position
# ---------------------------------------------------------------------------

def test_random_paste_position_fits():
    rng = random.Random(0)
    for _ in range(20):
        x, y = _random_paste_position((100, 100), (30, 30), rng)
        assert 0 <= x <= 70
        assert 0 <= y <= 70


def test_random_paste_position_crop_larger():
    """Crop larger than background → position = (0, 0)."""
    rng = random.Random(0)
    x, y = _random_paste_position((10, 10), (50, 50), rng)
    assert x == 0
    assert y == 0


# ---------------------------------------------------------------------------
# 2. load_normal_images
# ---------------------------------------------------------------------------

def test_load_normal_images_empty_dir(tmp_path):
    imgs = load_normal_images(str(tmp_path / "nope"))
    assert imgs == []


def test_load_normal_images_finds_files(tmp_path):
    img_dir = tmp_path / "good"
    img_dir.mkdir()
    (img_dir / "a.jpg").write_bytes(b"fake")
    (img_dir / "b.png").write_bytes(b"fake")
    (img_dir / "c.txt").write_bytes(b"skip")
    imgs = load_normal_images(str(img_dir))
    assert len(imgs) == 2


# ---------------------------------------------------------------------------
# 3. run() — dry run (no normal images)
# ---------------------------------------------------------------------------

def test_run_dry_run_no_normal_dir(tmp_path):
    roi_dir = _make_roi_selected(tmp_path, n=3)
    out = tmp_path / "output"

    result = run(
        roi_dir=str(roi_dir),
        normal_dir=str(tmp_path / "no_normals"),
        output_dir=str(out),
        method="copy_paste",
        n_per_roi=2,
    )

    assert result["status"] == "ok"
    assert result["n_generated"] == 6  # 3 rois × 2
    assert len(result["annotations"]) == 6
    assert all(a["dry_run"] for a in result["annotations"])
    assert (out / "annotations.json").exists()


def test_run_missing_roi(tmp_path):
    result = run(
        roi_dir=str(tmp_path / "nope"),
        normal_dir=str(tmp_path / "normals"),
        output_dir=str(tmp_path / "out"),
    )
    assert result["status"] == "missing_roi"
    assert result["n_generated"] == 0


def test_run_empty_roi(tmp_path):
    roi_dir = tmp_path / "roi"
    roi_dir.mkdir()
    with open(roi_dir / "roi_selected.json", "w") as f:
        json.dump([], f)

    result = run(
        roi_dir=str(roi_dir),
        normal_dir=str(tmp_path / "normals"),
        output_dir=str(tmp_path / "out"),
    )
    assert result["status"] == "empty_roi"


def test_run_unknown_method(tmp_path):
    roi_dir = _make_roi_selected(tmp_path, n=2)
    result = run(
        roi_dir=str(roi_dir),
        normal_dir=str(tmp_path / "normals"),
        output_dir=str(tmp_path / "out"),
        method="nonexistent",
    )
    assert "unknown_method" in result["status"]


# ---------------------------------------------------------------------------
# 4. Annotation schema
# ---------------------------------------------------------------------------

def test_annotation_schema(tmp_path):
    roi_dir = _make_roi_selected(tmp_path, n=2)
    out = tmp_path / "output"
    result = run(
        roi_dir=str(roi_dir),
        normal_dir=str(tmp_path / "no_normals"),
        output_dir=str(out),
        n_per_roi=1,
    )
    required = [
        "image_path", "source_roi", "cluster_id", "cell_key",
        "prompt", "method", "roi_score", "deficit",
    ]
    for ann in result["annotations"]:
        for k in required:
            assert k in ann, f"Missing key {k!r} in annotation"


# ---------------------------------------------------------------------------
# 5. copy_paste_synthesis — missing files → False, no crash
# ---------------------------------------------------------------------------

def test_copy_paste_missing_defect(tmp_path):
    roi_entry = {"image_path": str(tmp_path / "nope.jpg")}
    ok = copy_paste_synthesis(
        roi_entry=roi_entry,
        normal_image_path=str(tmp_path / "normal.jpg"),
        output_path=str(tmp_path / "out.jpg"),
    )
    assert ok is False


def test_copy_paste_missing_normal(tmp_path):
    defect = _make_pil_image(tmp_path, "defect.jpg")
    roi_entry = {"image_path": str(defect)}
    ok = copy_paste_synthesis(
        roi_entry=roi_entry,
        normal_image_path=str(tmp_path / "no_normal.jpg"),
        output_path=str(tmp_path / "out.jpg"),
    )
    assert ok is False


@pytest.mark.skipif(not HAS_PIL, reason="Pillow not installed")
def test_copy_paste_produces_file(tmp_path):
    defect = _make_pil_image(tmp_path, "defect.jpg", size=(32, 32))
    normal = _make_pil_image(tmp_path, "normal.jpg", size=(128, 128))
    out = tmp_path / "result.jpg"

    roi_entry = {"image_path": str(defect)}
    ok = copy_paste_synthesis(
        roi_entry=roi_entry,
        normal_image_path=str(normal),
        output_path=str(out),
        feather_px=2,
    )
    assert ok is True
    assert out.exists()
    assert out.stat().st_size > 0


# ---------------------------------------------------------------------------
# 6. ControlNet / Inpainting stubs raise NotImplementedError
# ---------------------------------------------------------------------------

def test_controlnet_stub():
    from aroma.generate_defects import controlnet_synthesis
    with pytest.raises(NotImplementedError):
        controlnet_synthesis({}, "bg.jpg", "out.jpg")


def test_inpainting_stub():
    from aroma.generate_defects import inpainting_synthesis
    with pytest.raises(NotImplementedError):
        inpainting_synthesis({}, "bg.jpg", "out.jpg")
