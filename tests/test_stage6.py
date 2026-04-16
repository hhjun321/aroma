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


def test_skip_not_applied_when_cached_defect_zero_and_stage4_now_exists(tmp_path):
    """stage4 미실행 상태로 캐시(defect_count=0)된 뒤 stage4 완료 시 재실행돼야 한다."""
    from utils.dataset_builder import build_dataset_groups
    image_dir = _make_image_dir(tmp_path, count=2)
    seed_dirs = _make_seed_dirs(tmp_path, ["scratch"])

    # 1차 실행 — stage4 없이 캐시 생성 (defect_count=0)
    stale_cache = {
        "pruning_threshold": 0.6,
        "baseline":     {"good_count": 2, "defect_count": 0},
        "aroma_full":   {"good_count": 2, "defect_count": 0},
        "aroma_pruned": {"good_count": 2, "defect_count": 0},
    }
    aug_dir = tmp_path / "augmented_dataset"
    aug_dir.mkdir(parents=True, exist_ok=True)
    (aug_dir / "build_report.json").write_text(json.dumps(stale_cache))

    # stage4 완료 후 상태 시뮬레이션
    _make_stage4_seed(tmp_path, "seed_001", ["000", "001"],
                      scores={"000": 0.8, "001": 0.9})

    # 2차 실행 — 캐시 무효화 후 재실행돼야 함
    result = build_dataset_groups(str(tmp_path), str(image_dir), seed_dirs,
                                   pruning_threshold=0.6)
    assert result["aroma_full"]["defect_count"] == 2, \
        "stage4 완료 후 재실행 시 defect_count 가 갱신돼야 함"


# ---------------------------------------------------------------------------
# Stage 4 precondition warning & stage4_status tests
# ---------------------------------------------------------------------------

def test_stage4_incomplete_emits_warning(tmp_path):
    """stage4_output에 seed 디렉터리만 있고 defect/ 하위 폴더가 없으면
    'Stage 4 미완료' UserWarning 과 stage4_status == 'incomplete'."""
    from utils.dataset_builder import build_dataset_groups
    image_dir = _make_image_dir(tmp_path, count=2)
    seed_dirs = _make_seed_dirs(tmp_path, ["scratch"])

    # seed 디렉터리만 생성, defect/ 서브디렉터리 없음
    for name in ("seed_001", "seed_002", "seed_003"):
        (tmp_path / "stage4_output" / name).mkdir(parents=True, exist_ok=True)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = build_dataset_groups(str(tmp_path), str(image_dir), seed_dirs)

        assert any(
            issubclass(warning.category, UserWarning)
            and "Stage 4 미완료" in str(warning.message)
            for warning in w
        ), f"Expected 'Stage 4 미완료' warning, got: {[str(x.message) for x in w]}"

    assert result["stage4_status"] == "incomplete"


def test_stage4_status_complete_in_report(tmp_path):
    """모든 seed에 defect 이미지가 있으면 stage4_status == 'complete'."""
    from utils.dataset_builder import build_dataset_groups
    image_dir = _make_image_dir(tmp_path, count=2)
    seed_dirs = _make_seed_dirs(tmp_path, ["scratch"])

    _make_stage4_seed(tmp_path, "seed_001", ["000", "001"])
    _make_stage4_seed(tmp_path, "seed_002", ["000"])

    result = build_dataset_groups(str(tmp_path), str(image_dir), seed_dirs)

    assert result["stage4_status"] == "complete"
    assert result["stage4_seeds_with_defects"] > 0


def test_stage4_status_partial_in_report(tmp_path):
    """일부 seed만 defect 이미지를 보유하면 stage4_status == 'partial'."""
    from utils.dataset_builder import build_dataset_groups
    image_dir = _make_image_dir(tmp_path, count=2)
    seed_dirs = _make_seed_dirs(tmp_path, ["scratch"])

    # seed_001: defect 이미지 있음
    _make_stage4_seed(tmp_path, "seed_001", ["000"])
    # seed_002: 디렉터리만 존재, defect 없음
    (tmp_path / "stage4_output" / "seed_002").mkdir(parents=True, exist_ok=True)

    result = build_dataset_groups(str(tmp_path), str(image_dir), seed_dirs)

    assert result["stage4_status"] == "partial"
    assert result["stage4_seeds_total"] == 2
    assert result["stage4_seeds_with_defects"] == 1


# ---------------------------------------------------------------------------
# Augmentation ratio tests
# ---------------------------------------------------------------------------

def test_augmentation_ratio_full_limits_defects(tmp_path):
    """augmentation_ratio_full이 설정되면 good_count * ratio 개수만큼만 선택."""
    from utils.dataset_builder import build_dataset_groups
    image_dir = _make_image_dir(tmp_path, count=10)  # 10 good images
    seed_dirs = _make_seed_dirs(tmp_path, ["scratch"])
    
    # 30개 defect 생성 (모두 quality_score=0.8)
    scores = {f"{i:03d}": 0.8 for i in range(30)}
    _make_stage4_seed(tmp_path, "seed_001", list(scores.keys()), scores=scores)

    # augmentation_ratio_full=1.5 → 10 * 1.5 = 15개 선택
    result = build_dataset_groups(
        str(tmp_path), str(image_dir), seed_dirs,
        augmentation_ratio_full=1.5
    )

    defect_dir = tmp_path / "augmented_dataset" / "aroma_full" / "train" / "defect"
    assert defect_dir.exists()
    assert len(list(defect_dir.glob("*.png"))) == 15
    assert result["aroma_full"]["defect_count"] == 15


def test_augmentation_ratio_pruned_limits_defects(tmp_path):
    """augmentation_ratio_pruned이 설정되면 threshold 적용 후 ratio만큼 선택."""
    from utils.dataset_builder import build_dataset_groups
    image_dir = _make_image_dir(tmp_path, count=10)
    seed_dirs = _make_seed_dirs(tmp_path, ["scratch"])
    
    # 20개 defect 생성: 10개는 quality=0.7, 10개는 quality=0.5
    scores = {f"{i:03d}": 0.7 if i < 10 else 0.5 for i in range(20)}
    _make_stage4_seed(tmp_path, "seed_001", list(scores.keys()), scores=scores)

    # pruning_threshold=0.6, augmentation_ratio_pruned=0.5 → 10 * 0.5 = 5개 선택
    # threshold 적용 후 10개 중 상위 5개 선택
    result = build_dataset_groups(
        str(tmp_path), str(image_dir), seed_dirs,
        pruning_threshold=0.6,
        augmentation_ratio_pruned=0.5
    )

    defect_dir = tmp_path / "augmented_dataset" / "aroma_pruned" / "train" / "defect"
    assert defect_dir.exists()
    assert len(list(defect_dir.glob("*.png"))) == 5
    assert result["aroma_pruned"]["defect_count"] == 5


def test_augmentation_ratio_selects_highest_quality(tmp_path):
    """augmentation_ratio 적용 시 quality_score가 높은 순으로 선택."""
    from utils.dataset_builder import build_dataset_groups
    image_dir = _make_image_dir(tmp_path, count=5)
    seed_dirs = _make_seed_dirs(tmp_path, ["scratch"])
    
    # 10개 defect 생성 (quality: 0.1~1.0)
    scores = {f"{i:03d}": 0.1 * (i + 1) for i in range(10)}
    _make_stage4_seed(tmp_path, "seed_001", list(scores.keys()), scores=scores)

    # augmentation_ratio_full=1.0 → 5 * 1.0 = 5개 선택 (상위 5개: 009~005)
    result = build_dataset_groups(
        str(tmp_path), str(image_dir), seed_dirs,
        augmentation_ratio_full=1.0
    )

    defect_dir = tmp_path / "augmented_dataset" / "aroma_full" / "train" / "defect"
    kept_files = sorted(defect_dir.glob("*.png"))
    assert len(kept_files) == 5
    
    # 상위 5개: 009 (1.0), 008 (0.9), 007 (0.8), 006 (0.7), 005 (0.6)
    expected_ids = ["009", "008", "007", "006", "005"]
    for exp_id in expected_ids:
        assert any(exp_id in f.name for f in kept_files), \
            f"Expected high-quality image {exp_id} to be selected"


def test_augmentation_ratio_warning_when_insufficient(tmp_path):
    """요청 개수가 가용 개수보다 많으면 warning 발생."""
    from utils.dataset_builder import build_dataset_groups
    image_dir = _make_image_dir(tmp_path, count=10)
    seed_dirs = _make_seed_dirs(tmp_path, ["scratch"])
    
    # 5개 defect만 생성
    scores = {f"{i:03d}": 0.8 for i in range(5)}
    _make_stage4_seed(tmp_path, "seed_001", list(scores.keys()), scores=scores)

    # augmentation_ratio_full=2.0 → 10 * 2.0 = 20개 요청, 5개만 가능
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = build_dataset_groups(
            str(tmp_path), str(image_dir), seed_dirs,
            augmentation_ratio_full=2.0
        )
        assert any(
            issubclass(warning.category, UserWarning)
            and "가용 defect" in str(warning.message)
            for warning in w
        )

    # 가용한 5개 모두 사용
    assert result["aroma_full"]["defect_count"] == 5


def test_augmentation_ratio_none_uses_all_defects(tmp_path):
    """augmentation_ratio=None이면 모든 defect 사용 (기존 동작)."""
    from utils.dataset_builder import build_dataset_groups
    image_dir = _make_image_dir(tmp_path, count=5)
    seed_dirs = _make_seed_dirs(tmp_path, ["scratch"])
    
    scores = {f"{i:03d}": 0.8 for i in range(20)}
    _make_stage4_seed(tmp_path, "seed_001", list(scores.keys()), scores=scores)

    result = build_dataset_groups(
        str(tmp_path), str(image_dir), seed_dirs,
        augmentation_ratio_full=None  # 명시적 None
    )

    # 20개 모두 사용
    assert result["aroma_full"]["defect_count"] == 20


def test_testset_includes_all_defect_types(tmp_path):
    """baseline/test/ 에 seed_dirs 에 없는 defect 유형도 포함되어야 한다.

    수정 전: seed_dirs(broken_large만) → test/broken_large/ 만 생성
    수정 후: test/ 전체 스캔 → test/broken_large/ + test/broken_small/ + test/contamination/ 모두 생성
    """
    from utils.dataset_builder import build_dataset_groups

    image_dir = _make_image_dir(tmp_path, count=3)
    # seed_dirs 에는 broken_large 만 등록
    seed_dirs = _make_seed_dirs(tmp_path, ["broken_large"])

    # test/ 에 broken_small, contamination 도 직접 추가 (seed_dirs 미등록)
    for extra_type in ["broken_small", "contamination"]:
        extra_dir = tmp_path / "test" / extra_type
        extra_dir.mkdir(parents=True, exist_ok=True)
        _make_png(extra_dir / "001.png")

    build_dataset_groups(str(tmp_path), str(image_dir), seed_dirs)

    baseline_test = tmp_path / "augmented_dataset" / "baseline" / "test"
    present = {d.name for d in baseline_test.iterdir() if d.is_dir()}
    assert "broken_large" in present, "seed_dirs 에 있는 유형은 포함돼야 함"
    assert "broken_small" in present, "seed_dirs 미등록 유형도 포함돼야 함"
    assert "contamination" in present, "seed_dirs 미등록 유형도 포함돼야 함"
    assert "good" in present, "good 이미지도 포함돼야 함"


def test_testset_fallback_to_seed_dirs_when_no_cat_test(tmp_path):
    """원본 test 디렉터리가 없으면 seed_dirs 폴백으로 테스트셋을 구성한다."""
    from utils.dataset_builder import build_dataset_groups

    # image_dir 을 test/ 없이 구성 (train/good 만)
    image_dir = tmp_path / "cat" / "train" / "good"
    image_dir.mkdir(parents=True, exist_ok=True)
    _make_png(image_dir / "000.png")

    # seed_dirs 는 별도 경로 (cat/test/ 아님)
    sd = tmp_path / "seeds" / "scratch"
    sd.mkdir(parents=True, exist_ok=True)
    _make_png(sd / "001.png")
    seed_dirs = [str(sd)]

    cat_dir = tmp_path / "cat"
    # cat/test/ 가 존재하지 않으므로 폴백 동작
    assert not (cat_dir / "test").exists()

    build_dataset_groups(str(cat_dir), str(image_dir), seed_dirs)

    baseline_test = cat_dir / "augmented_dataset" / "baseline" / "test"
    assert (baseline_test / "scratch").exists(), "폴백: seed_dirs 기반으로 test/scratch/ 생성돼야 함"


def test_augmentation_ratio_cached_in_report(tmp_path):
    """build_report.json에 augmentation_ratio 저장 및 캐시 검증."""
    from utils.dataset_builder import build_dataset_groups
    image_dir = _make_image_dir(tmp_path, count=5)
    seed_dirs = _make_seed_dirs(tmp_path, ["scratch"])
    
    scores = {f"{i:03d}": 0.8 for i in range(10)}
    _make_stage4_seed(tmp_path, "seed_001", list(scores.keys()), scores=scores)

    result = build_dataset_groups(
        str(tmp_path), str(image_dir), seed_dirs,
        augmentation_ratio_full=2.0,
        augmentation_ratio_pruned=1.5
    )

    # build_report.json 확인
    report_path = tmp_path / "augmented_dataset" / "build_report.json"
    assert report_path.exists()
    report = json.loads(report_path.read_text())
    assert report["augmentation_ratio_full"] == 2.0
    assert report["augmentation_ratio_pruned"] == 1.5


def test_skip_when_augmentation_ratio_matches(tmp_path):
    """캐시된 augmentation_ratio와 일치하면 skip."""
    from utils.dataset_builder import build_dataset_groups
    image_dir = _make_image_dir(tmp_path, count=5)
    seed_dirs = _make_seed_dirs(tmp_path, ["scratch"])

    cached = {
        "pruning_threshold": 0.6,
        "augmentation_ratio_full": 1.5,
        "augmentation_ratio_pruned": None,
        "baseline":     {"good_count": 5, "defect_count": 0},
        "aroma_full":   {"good_count": 5, "defect_count": 7},
        "aroma_pruned": {"good_count": 5, "defect_count": 3},
    }
    aug_dir = tmp_path / "augmented_dataset"
    aug_dir.mkdir(parents=True, exist_ok=True)
    (aug_dir / "build_report.json").write_text(json.dumps(cached))

    result = build_dataset_groups(
        str(tmp_path), str(image_dir), seed_dirs,
        pruning_threshold=0.6,
        augmentation_ratio_full=1.5,
        augmentation_ratio_pruned=None
    )
    
    # 캐시 반환
    assert result == cached


def test_rerun_when_augmentation_ratio_differs(tmp_path):
    """캐시된 augmentation_ratio와 다르면 재실행."""
    from utils.dataset_builder import build_dataset_groups
    image_dir = _make_image_dir(tmp_path, count=5)
    seed_dirs = _make_seed_dirs(tmp_path, ["scratch"])
    
    scores = {f"{i:03d}": 0.8 for i in range(10)}
    _make_stage4_seed(tmp_path, "seed_001", list(scores.keys()), scores=scores)

    # 이전 실행: ratio=1.0
    cached = {
        "pruning_threshold": 0.6,
        "augmentation_ratio_full": 1.0,
        "augmentation_ratio_pruned": None,
        "baseline":     {"good_count": 5, "defect_count": 0},
        "aroma_full":   {"good_count": 5, "defect_count": 5},
        "aroma_pruned": {"good_count": 5, "defect_count": 10},
    }
    aug_dir = tmp_path / "augmented_dataset"
    aug_dir.mkdir(parents=True, exist_ok=True)
    (aug_dir / "build_report.json").write_text(json.dumps(cached))

    # 새 실행: ratio=2.0 → 재실행
    result = build_dataset_groups(
        str(tmp_path), str(image_dir), seed_dirs,
        augmentation_ratio_full=2.0,  # 변경됨
        augmentation_ratio_pruned=None
    )
    
    # defect_count가 갱신됨
    assert result["aroma_full"]["defect_count"] == 10  # 5 * 2.0
    assert result["augmentation_ratio_full"] == 2.0


# ---------------------------------------------------------------------------
# Domain-based augmentation ratio tests
# ---------------------------------------------------------------------------

def test_augmentation_ratio_by_domain_applies_domain_specific_ratios(tmp_path):
    """도메인별 비율이 설정되면 도메인에 따라 다른 비율 적용."""
    from utils.dataset_builder import build_dataset_groups
    
    # mvtec 도메인 시뮬레이션: tmp_path/mvtec/bottle 구조 생성
    mvtec_dir = tmp_path / "mvtec"
    mvtec_dir.mkdir()
    bottle_dir = mvtec_dir / "bottle"
    bottle_dir.mkdir()
    
    image_dir = _make_image_dir(bottle_dir, count=10)  # 10 good images
    seed_dirs = _make_seed_dirs(bottle_dir, ["broken"])
    
    # 30개 defect 생성
    scores = {f"{i:03d}": 0.8 for i in range(30)}
    _make_stage4_seed(bottle_dir, "seed_001", list(scores.keys()), scores=scores)
    
    # 도메인별 비율 설정
    ratio_by_domain = {
        "isp": {"full": 1.0, "pruned": 0.5},
        "mvtec": {"full": 2.0, "pruned": 1.5},  # mvtec 적용됨
        "visa": {"full": 2.0, "pruned": 1.5},
    }
    
    result = build_dataset_groups(
        str(bottle_dir), str(image_dir), seed_dirs,
        augmentation_ratio_by_domain=ratio_by_domain
    )
    
    # mvtec 도메인의 비율 (2.0, 1.5) 적용
    assert result["domain"] == "mvtec"
    assert result["augmentation_ratio_full"] == 2.0
    assert result["augmentation_ratio_pruned"] == 1.5
    assert result["aroma_full"]["defect_count"] == 20  # 10 * 2.0
    assert result["aroma_pruned"]["defect_count"] == 15  # 10 * 1.5


def test_augmentation_ratio_by_domain_isp_applies_lower_ratio(tmp_path):
    """ISP 도메인은 낮은 비율 적용 (classification task)."""
    from utils.dataset_builder import build_dataset_groups
    
    # isp 도메인 시뮬레이션
    isp_dir = tmp_path / "isp"
    isp_dir.mkdir()
    asm_dir = isp_dir / "ASM"
    asm_dir.mkdir()
    
    image_dir = _make_image_dir(asm_dir, count=10)
    seed_dirs = _make_seed_dirs(asm_dir, ["area"])
    
    scores = {f"{i:03d}": 0.8 for i in range(30)}
    _make_stage4_seed(asm_dir, "seed_001", list(scores.keys()), scores=scores)
    
    ratio_by_domain = {
        "isp": {"full": 1.0, "pruned": 0.5},  # ISP: 낮은 비율
        "mvtec": {"full": 2.0, "pruned": 1.5},
    }
    
    result = build_dataset_groups(
        str(asm_dir), str(image_dir), seed_dirs,
        augmentation_ratio_by_domain=ratio_by_domain
    )
    
    assert result["domain"] == "isp"
    assert result["augmentation_ratio_full"] == 1.0
    assert result["augmentation_ratio_pruned"] == 0.5
    assert result["aroma_full"]["defect_count"] == 10  # 10 * 1.0
    assert result["aroma_pruned"]["defect_count"] == 5  # 10 * 0.5


def test_augmentation_ratio_by_domain_fallback_to_global(tmp_path):
    """도메인별 설정 없으면 전역 비율 fallback."""
    from utils.dataset_builder import build_dataset_groups
    
    # unknown 도메인
    unknown_dir = tmp_path / "unknown"
    unknown_dir.mkdir()
    cat_dir = unknown_dir / "category"
    cat_dir.mkdir()
    
    image_dir = _make_image_dir(cat_dir, count=10)
    seed_dirs = _make_seed_dirs(cat_dir, ["defect"])
    
    scores = {f"{i:03d}": 0.8 for i in range(30)}
    _make_stage4_seed(cat_dir, "seed_001", list(scores.keys()), scores=scores)
    
    ratio_by_domain = {
        "isp": {"full": 1.0, "pruned": 0.5},
        # "unknown" 도메인은 설정 없음
    }
    
    result = build_dataset_groups(
        str(cat_dir), str(image_dir), seed_dirs,
        augmentation_ratio_full=1.5,  # Fallback
        augmentation_ratio_pruned=1.0,
        augmentation_ratio_by_domain=ratio_by_domain
    )
    
    assert result["domain"] == "unknown"
    assert result["augmentation_ratio_full"] == 1.5  # Fallback 적용
    assert result["augmentation_ratio_pruned"] == 1.0
    assert result["aroma_full"]["defect_count"] == 15  # 10 * 1.5


def test_augmentation_ratio_by_domain_override_with_explicit_params(tmp_path):
    """도메인별 설정이 있어도 명시적 파라미터로 override 가능."""
    from utils.dataset_builder import build_dataset_groups
    
    mvtec_dir = tmp_path / "mvtec"
    mvtec_dir.mkdir()
    bottle_dir = mvtec_dir / "bottle"
    bottle_dir.mkdir()
    
    image_dir = _make_image_dir(bottle_dir, count=10)
    seed_dirs = _make_seed_dirs(bottle_dir, ["broken"])
    
    scores = {f"{i:03d}": 0.8 for i in range(30)}
    _make_stage4_seed(bottle_dir, "seed_001", list(scores.keys()), scores=scores)
    
    ratio_by_domain = {
        "mvtec": {"full": 2.0, "pruned": 1.5},
    }
    
    # 명시적 파라미터가 우선되지 않음 (도메인별 설정이 우선)
    result = build_dataset_groups(
        str(bottle_dir), str(image_dir), seed_dirs,
        augmentation_ratio_full=3.0,  # 무시됨
        augmentation_ratio_by_domain=ratio_by_domain
    )
    
    # 도메인별 설정이 우선 적용
    assert result["augmentation_ratio_full"] == 2.0
    assert result["aroma_full"]["defect_count"] == 20


def test_domain_extraction_from_cat_dir_path(tmp_path):
    """cat_dir 경로에서 도메인 추출 검증."""
    from utils.dataset_builder import build_dataset_groups
    
    # 다양한 도메인 구조 테스트
    for domain_name in ["isp", "mvtec", "visa"]:
        domain_dir = tmp_path / domain_name
        domain_dir.mkdir()
        cat_dir = domain_dir / "test_cat"
        cat_dir.mkdir()
        
        image_dir = _make_image_dir(cat_dir, count=5)
        seed_dirs = _make_seed_dirs(cat_dir, ["defect"])
        _make_stage4_seed(cat_dir, "seed_001", ["000"], scores={"000": 0.8})
        
        result = build_dataset_groups(
            str(cat_dir), str(image_dir), seed_dirs
        )
        
        assert result["domain"] == domain_name, \
            f"Expected domain '{domain_name}', got '{result['domain']}'"


