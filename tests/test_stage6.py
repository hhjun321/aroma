# tests/test_stage6.py
import json
import shutil
import warnings
import pytest
import cv2
import numpy as np
from pathlib import Path


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

def _make_png(path: Path, size: int = 32) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = np.zeros((size, size, 3), dtype=np.uint8)
    cv2.imwrite(str(path), img)


def _make_stage4_seed(cat_dir: Path, seed_id: str, image_ids: list,
                      scores: dict = None) -> None:
    defect_dir = cat_dir / "stage4_output" / seed_id / "defect"
    defect_dir.mkdir(parents=True, exist_ok=True)
    for iid in image_ids:
        _make_png(defect_dir / f"{iid}.png")
    if scores is not None:
        data = {
            "weights": {"artifact": 0.5, "blur": 0.5},
            "scores": [{"image_id": iid, "artifact_score": s,
                        "blur_score": s, "quality_score": s}
                       for iid, s in scores.items()],
            "stats": {},
        }
        (cat_dir / "stage4_output" / seed_id / "quality_scores.json").write_text(
            json.dumps(data))


def _make_image_dir(cat_dir: Path, count: int = 3) -> Path:
    image_dir = cat_dir / "train" / "good"
    image_dir.mkdir(parents=True, exist_ok=True)
    for i in range(count):
        _make_png(image_dir / f"{i:03d}.png")
    return image_dir


def _make_seed_dirs(cat_dir: Path, defect_types: list) -> list:
    seed_dirs = []
    for dt in defect_types:
        sd = cat_dir / "test" / dt
        sd.mkdir(parents=True, exist_ok=True)
        _make_png(sd / "001.png")
        seed_dirs.append(str(sd))
    test_good = cat_dir / "test" / "good"
    test_good.mkdir(parents=True, exist_ok=True)
    _make_png(test_good / "001.png")
    return seed_dirs


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_baseline_only_good(tmp_path):
    from utils.dataset_builder import build_dataset_groups
    image_dir = _make_image_dir(tmp_path, count=3)
    seed_dirs = _make_seed_dirs(tmp_path, ["scratch"])

    result = build_dataset_groups(str(tmp_path), str(image_dir), seed_dirs)

    good_dir = tmp_path / "augmented_dataset" / "baseline" / "train" / "good"
    defect_dir = tmp_path / "augmented_dataset" / "baseline" / "train" / "defect"
    assert good_dir.exists()
    assert len(list(good_dir.glob("*.png"))) == 3
    assert not defect_dir.exists()
    assert result["baseline"]["defect_count"] == 0


def test_aroma_full_all_defects(tmp_path):
    from utils.dataset_builder import build_dataset_groups
    image_dir = _make_image_dir(tmp_path, count=2)
    seed_dirs = _make_seed_dirs(tmp_path, ["scratch"])
    _make_stage4_seed(tmp_path, "seed_001", ["000", "001"])

    result = build_dataset_groups(str(tmp_path), str(image_dir), seed_dirs)

    defect_dir = tmp_path / "augmented_dataset" / "aroma_full" / "train" / "defect"
    assert defect_dir.exists()
    assert len(list(defect_dir.glob("*.png"))) == 2
    assert result["aroma_full"]["defect_count"] == 2


def test_aroma_pruned_threshold(tmp_path):
    from utils.dataset_builder import build_dataset_groups
    image_dir = _make_image_dir(tmp_path, count=2)
    seed_dirs = _make_seed_dirs(tmp_path, ["scratch"])
    _make_stage4_seed(tmp_path, "seed_001", ["000", "001"],
                      scores={"000": 0.5, "001": 0.8})

    result = build_dataset_groups(str(tmp_path), str(image_dir), seed_dirs,
                                   pruning_threshold=0.6)

    defect_dir = tmp_path / "augmented_dataset" / "aroma_pruned" / "train" / "defect"
    assert defect_dir.exists()
    assert len(list(defect_dir.glob("*.png"))) == 1
    kept_files = list(defect_dir.glob("*.png"))
    assert any("001" in f.name for f in kept_files), \
        f"Expected image_id '001' to be kept, got: {[f.name for f in kept_files]}"
    assert result["aroma_pruned"]["defect_count"] == 1


def test_skip_existing(tmp_path):
    from utils.dataset_builder import build_dataset_groups
    image_dir = _make_image_dir(tmp_path, count=2)
    seed_dirs = _make_seed_dirs(tmp_path, ["scratch"])

    cached = {
        "pruning_threshold": 0.6,
        "baseline":     {"good_count": 2, "defect_count": 0},
        "aroma_full":   {"good_count": 2, "defect_count": 5},
        "aroma_pruned": {"good_count": 2, "defect_count": 3},
    }
    aug_dir = tmp_path / "augmented_dataset"
    aug_dir.mkdir(parents=True, exist_ok=True)
    (aug_dir / "build_report.json").write_text(json.dumps(cached))

    result = build_dataset_groups(str(tmp_path), str(image_dir), seed_dirs,
                                   pruning_threshold=0.6)
    assert result == cached


def test_skip_threshold_mismatch_reruns(tmp_path):
    from utils.dataset_builder import build_dataset_groups
    image_dir = _make_image_dir(tmp_path, count=2)
    seed_dirs = _make_seed_dirs(tmp_path, ["scratch"])
    _make_stage4_seed(tmp_path, "seed_001", ["000"], scores={"000": 0.8})

    aug_dir = tmp_path / "augmented_dataset"
    aug_dir.mkdir(parents=True, exist_ok=True)
    (aug_dir / "build_report.json").write_text(
        json.dumps({"pruning_threshold": 0.5, "baseline": {}, "aroma_full": {}, "aroma_pruned": {}}))

    result = build_dataset_groups(str(tmp_path), str(image_dir), seed_dirs,
                                   pruning_threshold=0.7)
    assert abs(result["pruning_threshold"] - 0.7) < 1e-6


def test_build_report_saved(tmp_path):
    from utils.dataset_builder import build_dataset_groups
    image_dir = _make_image_dir(tmp_path, count=3)
    seed_dirs = _make_seed_dirs(tmp_path, ["scratch"])
    _make_stage4_seed(tmp_path, "seed_001", ["000", "001"],
                      scores={"000": 0.8, "001": 0.9})

    build_dataset_groups(str(tmp_path), str(image_dir), seed_dirs)

    report_path = tmp_path / "augmented_dataset" / "build_report.json"
    assert report_path.exists()
    report = json.loads(report_path.read_text())
    assert report["baseline"]["good_count"] == 3
    assert report["aroma_full"]["defect_count"] == 2


def test_missing_quality_scores_skips_seed(tmp_path):
    from utils.dataset_builder import build_dataset_groups
    image_dir = _make_image_dir(tmp_path, count=2)
    seed_dirs = _make_seed_dirs(tmp_path, ["scratch"])
    # defect 이미지는 있지만 quality_scores.json 없음
    defect_dir = tmp_path / "stage4_output" / "seed_no_scores" / "defect"
    defect_dir.mkdir(parents=True, exist_ok=True)
    _make_png(defect_dir / "000.png")

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = build_dataset_groups(str(tmp_path), str(image_dir), seed_dirs,
                                       pruning_threshold=0.6)
        assert any(
            issubclass(warning.category, UserWarning)
            and "quality_scores.json" in str(warning.message)
            for warning in w
        )

    assert result["aroma_pruned"]["defect_count"] == 0


def test_empty_stage4_output(tmp_path):
    from utils.dataset_builder import build_dataset_groups
    image_dir = _make_image_dir(tmp_path, count=2)
    seed_dirs = _make_seed_dirs(tmp_path, ["scratch"])
    # stage4_output 디렉터리 없음

    result = build_dataset_groups(str(tmp_path), str(image_dir), seed_dirs)

    assert result["aroma_full"]["defect_count"] == 0
    assert result["aroma_pruned"]["defect_count"] == 0
