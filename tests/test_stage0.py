"""Stage 0 — in-place resize tests."""
import json
import shutil

import cv2
import numpy as np
import pytest
from pathlib import Path


@pytest.fixture
def tmp_category(tmp_path):
    """dataset_config 엔트리와 동일한 구조의 임시 카테고리 생성.

    구조:
      tmp_path/
        category/
          train/good/       <- image_dir (1024x768 이미지 3장)
          test/defect/      <- seed_dir  (800x600 이미지 2장)
          test/good/        <- test good (512x512 이미지 1장, skip 대상)
    """
    cat_dir = tmp_path / "category"

    image_dir = cat_dir / "train" / "good"
    image_dir.mkdir(parents=True)
    for i in range(3):
        img = np.random.randint(0, 255, (768, 1024, 3), dtype=np.uint8)
        cv2.imwrite(str(image_dir / f"img_{i:03d}.png"), img)

    seed_dir = cat_dir / "test" / "defect"
    seed_dir.mkdir(parents=True)
    for i in range(2):
        img = np.random.randint(0, 255, (600, 800, 3), dtype=np.uint8)
        cv2.imwrite(str(seed_dir / f"seed_{i:03d}.png"), img)

    test_good_dir = cat_dir / "test" / "good"
    test_good_dir.mkdir(parents=True)
    img_512 = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    cv2.imwrite(str(test_good_dir / "good_000.png"), img_512)

    entry = {
        "domain": "test",
        "image_dir": str(image_dir),
        "seed_dir": str(seed_dir),
    }
    return cat_dir, entry


def test_resize_images_to_target_size(tmp_category):
    """image_dir의 1024x768 이미지 3장이 512x512로 리사이즈되는지."""
    cat_dir, entry = tmp_category
    from stage0_resize import resize_directory
    image_dir = Path(entry["image_dir"])
    stats = resize_directory(image_dir, target_size=512)
    assert stats["resized"] == 3
    assert stats["skipped"] == 0
    assert stats["errors"] == 0
    for img_path in image_dir.glob("*.png"):
        img = cv2.imread(str(img_path))
        assert img.shape[:2] == (512, 512), f"{img_path.name} not resized"


def test_skip_already_correct_size(tmp_category):
    """이미 512x512인 이미지는 skip."""
    cat_dir, entry = tmp_category
    from stage0_resize import resize_directory
    test_good_dir = cat_dir / "test" / "good"
    stats = resize_directory(test_good_dir, target_size=512)
    assert stats["resized"] == 0
    assert stats["skipped"] == 1
    assert stats["errors"] == 0


def test_interpolation_area_for_downscale(tmp_category, monkeypatch):
    """축소 시 cv2.INTER_AREA 사용 확인."""
    cat_dir, entry = tmp_category
    from stage0_resize import resize_directory
    calls = []
    original_resize = cv2.resize
    def spy_resize(img, dsize, interpolation=None, **kw):
        calls.append(interpolation)
        return original_resize(img, dsize, interpolation=interpolation, **kw)
    monkeypatch.setattr(cv2, "resize", spy_resize)
    image_dir = Path(entry["image_dir"])
    resize_directory(image_dir, target_size=512)
    assert all(c == cv2.INTER_AREA for c in calls)


def test_sentinel_created_on_success(tmp_category):
    """리사이즈 성공 후 sentinel 파일이 생성되는지."""
    cat_dir, entry = tmp_category
    from stage0_resize import resize_category
    result = resize_category(entry, target_size=512)
    sentinel = cat_dir / ".stage0_resize_512_done"
    assert sentinel.exists()
    data = json.loads(sentinel.read_text())
    assert data["target_size"] == 512
    assert data["resized"] > 0
    assert "timestamp" in data


def test_sentinel_skip_on_rerun(tmp_category):
    """sentinel 있으면 카테고리 skip."""
    cat_dir, entry = tmp_category
    from stage0_resize import resize_category
    result1 = resize_category(entry, target_size=512)
    assert result1["resized"] > 0
    result2 = resize_category(entry, target_size=512)
    assert result2["skipped_category"] is True
    assert result2["resized"] == 0


# ---------------------------------------------------------------------------
# dry_run / missing directory 테스트
# ---------------------------------------------------------------------------

def test_dry_run_no_modification(tmp_category):
    """dry_run 시 파일이 변경되지 않아야 한다."""
    cat_dir, entry = tmp_category
    from stage0_resize import resize_category

    # 원본 사이즈 기록
    image_dir = Path(entry["image_dir"])
    original_sizes = {}
    for p in image_dir.glob("*.png"):
        img = cv2.imread(str(p))
        original_sizes[p.name] = img.shape[:2]

    result = resize_category(entry, target_size=512, dry_run=True)

    # 리사이즈 카운트는 있지만 실제 파일은 변경 안 됨
    assert result["resized"] > 0

    for p in image_dir.glob("*.png"):
        img = cv2.imread(str(p))
        assert img.shape[:2] == original_sizes[p.name], f"{p.name} was modified"

    # sentinel도 생성 안 됨
    sentinel = cat_dir / ".stage0_resize_512_done"
    assert not sentinel.exists()


def test_missing_directory_warns_not_errors(tmp_path):
    """미존재 디렉토리는 warning으로 처리, 에러 아님."""
    from stage0_resize import resize_directory

    stats = resize_directory(tmp_path / "nonexistent", target_size=512)

    assert stats["resized"] == 0
    assert stats["skipped"] == 0
    assert stats["errors"] == 0


# ---------------------------------------------------------------------------
# clean_category 테스트
# ---------------------------------------------------------------------------

def test_clean_removes_stage_outputs(tmp_category):
    """--clean이 stage1~6 출력물과 sentinel을 삭제하는지."""
    cat_dir, entry = tmp_category

    # stage 출력 디렉토리와 sentinel 생성
    (cat_dir / "stage1_output").mkdir()
    (cat_dir / "stage1_output" / "roi_metadata.json").touch()
    (cat_dir / "stage1b_output").mkdir()
    (cat_dir / "stage2_output").mkdir()
    (cat_dir / "stage3_output").mkdir()
    (cat_dir / "stage4_output").mkdir()
    (cat_dir / "augmented_dataset").mkdir()
    (cat_dir / ".stage0_resize_512_done").touch()

    from stage0_resize import clean_category

    deleted = clean_category(entry)

    assert not (cat_dir / "stage1_output").exists()
    assert not (cat_dir / "stage1b_output").exists()
    assert not (cat_dir / "stage2_output").exists()
    assert not (cat_dir / "stage3_output").exists()
    assert not (cat_dir / "stage4_output").exists()
    assert not (cat_dir / "augmented_dataset").exists()
    assert not (cat_dir / ".stage0_resize_512_done").exists()
    assert len(deleted) == 7  # 6 dirs + 1 sentinel


# ---------------------------------------------------------------------------
# workers 병렬 처리 테스트
# ---------------------------------------------------------------------------

def test_resize_directory_with_workers(tmp_category):
    """workers 파라미터가 resize_directory에서 병렬 처리를 수행하는지 확인."""
    cat_dir, entry = tmp_category
    from stage0_resize import resize_directory
    image_dir = Path(entry["image_dir"])

    # workers=2로 병렬 실행 — 결과는 순차와 동일해야 함
    stats = resize_directory(image_dir, target_size=512, workers=2)
    assert stats["resized"] == 3
    assert stats["skipped"] == 0
    assert stats["errors"] == 0

    # 실제 리사이즈 확인
    for img_path in image_dir.glob("*.png"):
        img = cv2.imread(str(img_path))
        assert img.shape[:2] == (512, 512), f"{img_path.name} not resized"


def test_resize_directory_workers_minus1(tmp_category):
    """workers=-1 (auto) 가 정상 동작하는지 확인."""
    cat_dir, entry = tmp_category
    from stage0_resize import resize_directory
    seed_dir = Path(entry["seed_dir"])

    stats = resize_directory(seed_dir, target_size=512, workers=-1)
    assert stats["resized"] == 2
    assert stats["errors"] == 0


def test_resize_category_passes_workers(tmp_category):
    """resize_category가 workers 파라미터를 resize_directory에 전파하는지 확인."""
    cat_dir, entry = tmp_category
    from stage0_resize import resize_category

    result = resize_category(entry, target_size=512, workers=2)
    assert result["resized"] > 0
    assert result["errors"] == 0

    # 실제 리사이즈 확인 (image_dir)
    for img_path in Path(entry["image_dir"]).glob("*.png"):
        img = cv2.imread(str(img_path))
        assert img.shape[:2] == (512, 512)
