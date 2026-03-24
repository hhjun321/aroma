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
