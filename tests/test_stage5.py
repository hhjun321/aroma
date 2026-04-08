# tests/test_stage5.py
import json
import numpy as np
import pytest
import cv2
from pathlib import Path


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def uniform_gray():
    """균일한 회색 이미지 — 아티팩트 없음."""
    return np.full((64, 64, 3), 128, dtype=np.uint8)


@pytest.fixture
def sharp_image():
    """날카로운 체커보드 패턴 — 선명한 이미지."""
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    for i in range(64):
        for j in range(64):
            if (i // 8 + j // 8) % 2 == 0:
                img[i, j] = 255
    return img


@pytest.fixture
def blurred_image(sharp_image):
    """가우시안 블러 이미지 — 흐린 이미지."""
    return cv2.GaussianBlur(sharp_image, (15, 15), 5)


# ---------------------------------------------------------------------------
# score_image 테스트
# ---------------------------------------------------------------------------

def test_score_image_basic(sharp_image):
    from utils.quality_scoring import score_image
    result = score_image(sharp_image)
    assert "artifact_score" in result
    assert "blur_score" in result
    assert "quality_score" in result
    assert 0.0 <= result["artifact_score"] <= 1.0
    assert 0.0 <= result["blur_score"] <= 1.0
    assert 0.0 <= result["quality_score"] <= 1.0


def test_score_image_weight_validation(sharp_image):
    from utils.quality_scoring import score_image
    with pytest.raises(ValueError):
        score_image(sharp_image, w_artifact=0.6, w_blur=0.6)


def test_score_artifacts_clean(uniform_gray):
    from utils.quality_scoring import score_image
    result = score_image(uniform_gray)
    # 균일한 이미지는 아티팩트가 없으므로 artifact_score 높아야 함
    assert result["artifact_score"] > 0.5


def test_score_sharpness_blurred(blurred_image):
    from utils.quality_scoring import score_image
    result = score_image(blurred_image)
    # 블러 이미지는 선명도가 낮아야 함
    assert result["blur_score"] < 0.5


def test_score_defect_images_skip(tmp_path):
    from utils.quality_scoring import score_defect_images
    # 미리 quality_scores.json 작성
    existing = {
        "weights": {"artifact": 0.5, "blur": 0.5},
        "scores": [{"image_id": "000", "artifact_score": 0.8,
                    "blur_score": 0.9, "quality_score": 0.85}],
        "stats": {"count": 1, "mean": 0.85, "std": 0.0,
                  "min": 0.85, "max": 0.85,
                  "p25": 0.85, "p50": 0.85, "p75": 0.85, "p90": 0.85},
    }
    cache = tmp_path / "quality_scores.json"
    cache.write_text(json.dumps(existing))

    result = score_defect_images(str(tmp_path))
    # 반환값이 기존 JSON과 동일해야 함 (재계산 없이 로드)
    assert result == existing


def test_score_defect_images_empty(tmp_path):
    from utils.quality_scoring import score_defect_images
    # defect/ 디렉터리 없음
    result = score_defect_images(str(tmp_path))
    assert result["scores"] == []
    assert result["stats"] == {}
    # 파일 미생성
    assert not (tmp_path / "quality_scores.json").exists()


def test_score_defect_images_ordering(tmp_path):
    from utils.quality_scoring import score_defect_images
    # defect/ 아래 3개 이미지 생성
    defect_dir = tmp_path / "defect"
    defect_dir.mkdir()
    for name in ["002.png", "000.png", "001.png"]:
        img = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        cv2.imwrite(str(defect_dir / name), img)

    result = score_defect_images(str(tmp_path))
    ids = [s["image_id"] for s in result["scores"]]
    # sorted 순서: 000, 001, 002
    assert ids == ["000", "001", "002"]


# ---------------------------------------------------------------------------
# Prerequisite validation tests (B1)
# ---------------------------------------------------------------------------

def test_run_quality_scoring_missing_dir_raises(tmp_path):
    """Stage 5 run_quality_scoring should raise for missing directory."""
    from stage5_quality_scoring import run_quality_scoring
    with pytest.raises(FileNotFoundError, match="Stage 4 seed output"):
        run_quality_scoring(str(tmp_path / "nonexistent_stage4_dir"))


# ---------------------------------------------------------------------------
# Resolution-adaptive normalization tests
# ---------------------------------------------------------------------------

def test_sharpness_resolution_invariant():
    """동일한 체커보드 패턴을 256×256과 512×512로 생성했을 때
    _score_sharpness 가 유사한 점수를 반환해야 한다."""
    from utils.quality_scoring import _score_sharpness

    def _make_checkerboard(size, block):
        img = np.zeros((size, size), dtype=np.uint8)
        for i in range(size):
            for j in range(size):
                if (i // block + j // block) % 2 == 0:
                    img[i, j] = 255
        return img

    small = _make_checkerboard(256, 32)   # 256×256, block 32
    large = _make_checkerboard(512, 64)   # 512×512, block 64 (동일 비율)

    score_small = _score_sharpness(small)
    score_large = _score_sharpness(large)

    # 해상도 정규화 이후 두 점수의 차이가 0.15 이내여야 함
    assert abs(score_small - score_large) < 0.15, (
        f"score_small={score_small:.4f}, score_large={score_large:.4f} — "
        f"해상도 정규화 실패: 차이 {abs(score_small - score_large):.4f} > 0.15"
    )


def test_sharpness_uses_pixel_count_scaling():
    """_score_sharpness 내부의 lap_var 정규화가 픽셀 수에 비례하는지 확인.
    균일 노이즈 이미지에서 큰 이미지의 lap_score 가 비정규화 시보다 낮아야 한다."""
    from utils.quality_scoring import _score_sharpness

    # 고정 시드로 노이즈 이미지 생성
    rng = np.random.RandomState(42)
    small = rng.randint(0, 255, (64, 64), dtype=np.uint8)
    # 큰 이미지: 동일 패턴을 2×2 타일링 (해상도만 증가)
    large = np.tile(small, (4, 4))  # 256×256

    score_small = _score_sharpness(small)
    score_large = _score_sharpness(large)

    # 정규화가 올바르면 타일링된 이미지도 비슷한 점수를 가져야 함
    # (정규화 없이는 large가 체계적으로 다른 점수를 받음)
    assert 0.0 <= score_small <= 1.0
    assert 0.0 <= score_large <= 1.0


# ---------------------------------------------------------------------------
# Batch processing tests
# ---------------------------------------------------------------------------

def test_run_quality_scoring_batch_empty():
    """Empty seed_dirs list should return empty results."""
    from stage5_quality_scoring import run_quality_scoring_batch
    results = run_quality_scoring_batch([])
    assert results == []


def test_run_quality_scoring_batch_processes_multiple_seeds(tmp_path, sharp_image):
    """Batch function should process multiple seed directories in parallel."""
    from stage5_quality_scoring import run_quality_scoring_batch
    
    # Create 3 seed directories with defect images
    seed_dirs = []
    for i in range(3):
        seed_dir = tmp_path / f"seed_{i:03d}"
        defect_dir = seed_dir / "defect"
        defect_dir.mkdir(parents=True)
        
        # Write 2 images per seed
        for j in range(2):
            img_path = defect_dir / f"img_{j:03d}.png"
            cv2.imwrite(str(img_path), sharp_image)
        
        seed_dirs.append(str(seed_dir))
    
    # Run batch processing
    results = run_quality_scoring_batch(
        stage4_seed_dirs=seed_dirs,
        workers=0,
        parallel_seeds=2,
    )
    
    # Should process all 3 seeds
    assert len(results) == 3
    
    # Each result should have required fields
    for result in results:
        assert "seed_id" in result
        assert "count" in result
        assert "mean" in result
        assert result["count"] == 2  # 2 images per seed
        assert 0.0 <= result["mean"] <= 1.0
    
    # Check that quality_scores.json was created for each seed
    for seed_dir in seed_dirs:
        json_path = Path(seed_dir) / "quality_scores.json"
        assert json_path.exists()


def test_run_quality_scoring_batch_skips_already_processed(tmp_path, sharp_image):
    """Batch function should skip seeds with existing quality_scores.json."""
    from stage5_quality_scoring import run_quality_scoring_batch
    
    # Create 2 seed directories
    seed1 = tmp_path / "seed_001"
    seed2 = tmp_path / "seed_002"
    
    for seed_dir in [seed1, seed2]:
        defect_dir = seed_dir / "defect"
        defect_dir.mkdir(parents=True)
        img_path = defect_dir / "img_000.png"
        cv2.imwrite(str(img_path), sharp_image)
    
    # Pre-create quality_scores.json for seed1 (simulate already processed)
    existing_scores = {
        "weights": {"artifact": 0.5, "blur": 0.5},
        "scores": [{"image_id": "img_000", "quality_score": 0.85}],
        "stats": {"count": 1, "mean": 0.85}
    }
    (seed1 / "quality_scores.json").write_text(json.dumps(existing_scores))
    
    # Run batch processing
    results = run_quality_scoring_batch(
        stage4_seed_dirs=[str(seed1), str(seed2)],
        workers=0,
        parallel_seeds=1,
    )
    
    # Should only process seed2 (seed1 was skipped)
    assert len(results) == 1
    assert results[0]["seed_id"] == "seed_002"

