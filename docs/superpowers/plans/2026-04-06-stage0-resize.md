# Stage 0 In-Place Resize — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `stage0_resize.py` + `tests/test_stage0.py` 구현 — 파이프라인 진입 전 원본 이미지를 512×512로 인플레이스 리사이즈하고, 기존 stage 출력물을 정리하는 Stage 0 스크립트.

**Architecture:** `stage0_resize.py`는 단일 스크립트로 resize + clean 기능을 모두 제공. `dataset_config.json`을 읽어 카테고리별 3개 디렉토리(image_dir, seed_dir, cat_dir/test/good/)를 처리. sentinel 파일 기반 resume 지원.

**Tech Stack:** Python, OpenCV, pathlib, argparse, `utils/io.py` (save_json, load_json)

**Spec:** `docs/superpowers/specs/2026-04-06-stage0-resize-design.md`

---

## File Structure

| 파일 | 역할 |
|------|------|
| Create: `stage0_resize.py` | `resize_directory`, `resize_category`, `clean_category` + CLI |
| Create: `tests/test_stage0.py` | 8 test cases (spec 정의 그대로) |
| Modify: `docs/작업일지/stage1_6_execute.md` | Stage 0 Colab 셀 추가 (Stage 1 앞) |

---

## Chunk 1: Core Implementation

### Task 1: `tests/test_stage0.py` — 테스트 fixtures + 첫 3개 테스트 작성

**Files:**
- Create: `tests/test_stage0.py`

- [ ] **Step 1-1: 테스트 파일 생성 — fixtures + resize 기본 테스트**

```python
# tests/test_stage0.py
"""Stage 0 — in-place resize tests."""
import json
import shutil

import cv2
import numpy as np
import pytest
from pathlib import Path


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_category(tmp_path):
    """dataset_config 엔트리와 동일한 구조의 임시 카테고리 생성.

    구조:
      tmp_path/
        category/
          train/good/       ← image_dir (1024×768 이미지 3장)
          test/defect/      ← seed_dir  (800×600 이미지 2장)
          test/good/        ← test good (512×512 이미지 1장, skip 대상)
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


# ---------------------------------------------------------------------------
# resize_directory 테스트
# ---------------------------------------------------------------------------

def test_resize_images_to_target_size(tmp_category):
    """image_dir의 1024×768 이미지 3장이 512×512로 리사이즈되는지."""
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
    """이미 512×512인 이미지는 skip."""
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

    # 1024×768 → 512×512는 축소 → INTER_AREA
    assert all(c == cv2.INTER_AREA for c in calls)
```

- [ ] **Step 1-2: 테스트 실행 — 실패 확인**

Run: `pytest tests/test_stage0.py -v --tb=short 2>&1 | head -30`
Expected: FAIL — `ModuleNotFoundError: No module named 'stage0_resize'`


### Task 2: `stage0_resize.py` — `resize_directory` 구현

**Files:**
- Create: `stage0_resize.py`

- [ ] **Step 2-1: `stage0_resize.py` 생성 — `resize_directory` 함수**

```python
"""stage0_resize.py

Stage 0 of the AROMA pipeline: in-place image resize.

Resizes all images in dataset directories to a uniform target size
(default 512×512) before the pipeline begins.  This ensures consistent
resolution across all downstream stages and reduces Google Drive storage.

Usage:
    python stage0_resize.py --config dataset_config.json --size 512
    python stage0_resize.py --config dataset_config.json --clean --dry-run
"""
from __future__ import annotations

import argparse
import glob as _glob
import logging
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2

sys.path.insert(0, str(Path(__file__).parent))
from utils.io import load_json, save_json

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_IMAGE_EXTENSIONS = ("*.png", "*.jpg", "*.jpeg", "*.bmp")

_STAGE_OUTPUT_DIRS = [
    "stage1_output",
    "stage1b_output",
    "stage2_output",
    "stage3_output",
    "stage4_output",    # Stage 5 output (quality_scores.json) lives here too
    "augmented_dataset",  # Stage 6 output
]


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------
def resize_directory(
    dir_path: Path | str,
    target_size: int = 512,
    dry_run: bool = False,
) -> dict[str, int]:
    """Resize all images in *dir_path* to *target_size* × *target_size* in-place.

    Returns dict with keys: resized, skipped, errors.
    """
    dir_path = Path(dir_path)
    stats = {"resized": 0, "skipped": 0, "errors": 0}

    if not dir_path.is_dir():
        logger.warning("Directory not found, skipping: %s", dir_path)
        return stats

    image_files: list[Path] = []
    for ext in _IMAGE_EXTENSIONS:
        image_files.extend(dir_path.glob(ext))
    image_files.sort()

    for img_path in image_files:
        try:
            img = cv2.imread(str(img_path))
            if img is None:
                logger.error("Failed to read image: %s", img_path)
                stats["errors"] += 1
                continue

            h, w = img.shape[:2]
            if h == target_size and w == target_size:
                stats["skipped"] += 1
                continue

            if dry_run:
                logger.info("[DRY-RUN] Would resize %s (%dx%d → %dx%d)",
                            img_path.name, w, h, target_size, target_size)
                stats["resized"] += 1
                continue

            # Choose interpolation: INTER_AREA for downscale, INTER_LINEAR for upscale
            needs_downscale = (h > target_size) or (w > target_size)
            interp = cv2.INTER_AREA if needs_downscale else cv2.INTER_LINEAR

            resized = cv2.resize(img, (target_size, target_size), interpolation=interp)
            cv2.imwrite(str(img_path), resized)
            stats["resized"] += 1

        except Exception:
            logger.exception("Error processing %s", img_path)
            stats["errors"] += 1

    return stats
```

- [ ] **Step 2-2: 처음 3개 테스트 실행 — 통과 확인**

Run: `pytest tests/test_stage0.py::test_resize_images_to_target_size tests/test_stage0.py::test_skip_already_correct_size tests/test_stage0.py::test_interpolation_area_for_downscale -v`
Expected: 3 passed


### Task 3: `resize_category` + sentinel 테스트 추가 및 구현

**Files:**
- Modify: `tests/test_stage0.py` — sentinel 테스트 추가
- Modify: `stage0_resize.py` — `resize_category` 함수 추가

- [ ] **Step 3-1: sentinel 테스트 추가 (tests/test_stage0.py 끝에 append)**

```python
# ---------------------------------------------------------------------------
# resize_category 테스트
# ---------------------------------------------------------------------------

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

    # 첫 실행
    result1 = resize_category(entry, target_size=512)
    assert result1["resized"] > 0

    # 두 번째 실행 — sentinel이 있으므로 skip
    result2 = resize_category(entry, target_size=512)
    assert result2["skipped_category"] is True
    assert result2["resized"] == 0
```

- [ ] **Step 3-2: 테스트 실행 — 실패 확인**

Run: `pytest tests/test_stage0.py::test_sentinel_created_on_success tests/test_stage0.py::test_sentinel_skip_on_rerun -v --tb=short`
Expected: FAIL — `ImportError: cannot import name 'resize_category'`

- [ ] **Step 3-3: `stage0_resize.py`에 `resize_category` 함수 추가**

`resize_directory` 함수 뒤에 추가:

```python
def resize_category(
    entry: dict[str, Any],
    target_size: int = 512,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Resize all images for a single dataset category.

    Processes three directories: image_dir, seed_dir, cat_dir/test/good/.
    Creates a sentinel file on success to enable resume.

    Returns dict with: category, resized, skipped, errors, skipped_category.
    """
    seed_dir = Path(entry["seed_dir"])
    cat_dir = seed_dir.parents[1]
    image_dir = Path(entry["image_dir"])
    test_good_dir = cat_dir / "test" / "good"

    sentinel = cat_dir / f".stage0_resize_{target_size}_done"

    result: dict[str, Any] = {
        "category": cat_dir.name,
        "resized": 0,
        "skipped": 0,
        "errors": 0,
        "skipped_category": False,
    }

    # ── Resume: sentinel이 있으면 skip ────────────────────────────
    if sentinel.exists():
        logger.info("Sentinel exists, skipping: %s", cat_dir.name)
        result["skipped_category"] = True
        return result

    # ── 3개 디렉토리 리사이즈 ─────────────────────────────────────
    for label, dir_path in [
        ("image_dir", image_dir),
        ("seed_dir", seed_dir),
        ("test/good", test_good_dir),
    ]:
        logger.info("Resizing %s: %s", label, dir_path)
        stats = resize_directory(dir_path, target_size=target_size, dry_run=dry_run)
        result["resized"] += stats["resized"]
        result["skipped"] += stats["skipped"]
        result["errors"] += stats["errors"]

    # ── Sentinel 생성 (dry_run이 아니고 에러 0건일 때) ────────────
    if not dry_run and result["errors"] == 0:
        save_json(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "target_size": target_size,
                "resized": result["resized"],
                "skipped": result["skipped"],
            },
            sentinel,
        )

    return result
```

- [ ] **Step 3-4: sentinel 테스트 2개 실행 — 통과 확인**

Run: `pytest tests/test_stage0.py::test_sentinel_created_on_success tests/test_stage0.py::test_sentinel_skip_on_rerun -v`
Expected: 2 passed

- [ ] **Step 3-5: 전체 Stage 0 테스트 실행 — 5개 통과 확인**

Run: `pytest tests/test_stage0.py -v`
Expected: 5 passed

- [ ] **Step 3-6: Commit**

```bash
git add stage0_resize.py tests/test_stage0.py
git commit -m "feat(stage0): add resize_directory + resize_category with sentinel resume"
```


### Task 4: dry_run + missing directory 테스트 추가 및 구현

**Files:**
- Modify: `tests/test_stage0.py` — dry_run + missing dir 테스트 추가

- [ ] **Step 4-1: dry_run + missing directory 테스트 추가**

```python
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
```

- [ ] **Step 4-2: 테스트 실행 — 통과 확인**

Run: `pytest tests/test_stage0.py::test_dry_run_no_modification tests/test_stage0.py::test_missing_directory_warns_not_errors -v`
Expected: 2 passed

- [ ] **Step 4-3: Commit**

```bash
git add tests/test_stage0.py
git commit -m "test(stage0): add dry_run and missing directory tests"
```


### Task 5: `clean_category` 테스트 추가 및 구현

**Files:**
- Modify: `tests/test_stage0.py` — clean 테스트 추가
- Modify: `stage0_resize.py` — `clean_category` 함수 추가

- [ ] **Step 5-1: clean 테스트 추가**

```python
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
    assert len(deleted) >= 7  # 6 dirs + 1 sentinel
```

- [ ] **Step 5-2: 테스트 실행 — 실패 확인**

Run: `pytest tests/test_stage0.py::test_clean_removes_stage_outputs -v --tb=short`
Expected: FAIL — `ImportError: cannot import name 'clean_category'`

- [ ] **Step 5-3: `stage0_resize.py`에 `clean_category` 함수 추가**

`resize_category` 뒤에 추가:

```python
def clean_category(
    entry: dict[str, Any],
    dry_run: bool = False,
) -> list[str]:
    """Delete stage 1–6 outputs and sentinel files for a category.

    Returns list of deleted (or would-be-deleted in dry_run) paths.
    """
    seed_dir = Path(entry["seed_dir"])
    cat_dir = seed_dir.parents[1]

    deleted: list[str] = []

    # ── Stage 출력 디렉토리 삭제 ──────────────────────────────────
    for dirname in _STAGE_OUTPUT_DIRS:
        target = cat_dir / dirname
        if target.exists():
            if dry_run:
                logger.info("[DRY-RUN] Would delete: %s", target)
            else:
                shutil.rmtree(target)
                logger.info("Deleted: %s", target)
            deleted.append(str(target))

    # ── Sentinel 파일 삭제 ────────────────────────────────────────
    for sentinel in cat_dir.glob(".stage0_resize_*_done"):
        if dry_run:
            logger.info("[DRY-RUN] Would delete: %s", sentinel)
        else:
            sentinel.unlink()
            logger.info("Deleted: %s", sentinel)
        deleted.append(str(sentinel))

    return deleted
```

- [ ] **Step 5-4: clean 테스트 실행 — 통과 확인**

Run: `pytest tests/test_stage0.py::test_clean_removes_stage_outputs -v`
Expected: 1 passed

- [ ] **Step 5-5: 전체 Stage 0 테스트 실행 — 8개 통과 확인**

Run: `pytest tests/test_stage0.py -v`
Expected: 8 passed

- [ ] **Step 5-6: Commit**

```bash
git add stage0_resize.py tests/test_stage0.py
git commit -m "feat(stage0): add clean_category to delete stage 1-6 outputs"
```


### Task 6: CLI (`_build_parser` + `main`) 구현

**Files:**
- Modify: `stage0_resize.py` — CLI 추가

- [ ] **Step 6-1: CLI 함수 추가**

`clean_category` 뒤에 추가:

```python
# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Stage 0 — in-place image resize to uniform target size"
    )
    p.add_argument("--config", default="dataset_config.json",
                   help="Path to dataset_config.json (default: dataset_config.json)")
    p.add_argument("--size", type=int, default=512,
                   help="Target image size (default: 512)")
    p.add_argument("--domain-filter", default=None,
                   choices=["isp", "mvtec", "visa"],
                   help="Process only the specified domain")
    p.add_argument("--dry-run", action="store_true",
                   help="Report only, do not modify files")
    p.add_argument("--clean", action="store_true",
                   help="Delete stage 1-6 outputs before resize")
    return p


def main() -> None:
    args = _build_parser().parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(message)s",
    )

    config = load_json(args.config)

    # Deduplicate by cat_dir (multiple entries may share the same category)
    seen: set[str] = set()
    entries: list[tuple[str, dict]] = []
    for key, entry in config.items():
        if key.startswith("_"):
            continue
        if args.domain_filter and entry.get("domain") != args.domain_filter:
            continue
        cat_dir = str(Path(entry["seed_dir"]).parents[1])
        if cat_dir in seen:
            continue
        seen.add(cat_dir)
        entries.append((key, entry))

    if not entries:
        print("No categories to process.")
        return

    # ── Clean (if requested) ──────────────────────────────────────
    if args.clean:
        print(f"{'[DRY-RUN] ' if args.dry_run else ''}Cleaning {len(entries)} categories...")
        for key, entry in entries:
            deleted = clean_category(entry, dry_run=args.dry_run)
            if deleted:
                print(f"  {key}: {len(deleted)} items {'would be ' if args.dry_run else ''}deleted")

    # ── Resize ────────────────────────────────────────────────────
    print(f"\n{'[DRY-RUN] ' if args.dry_run else ''}Resizing {len(entries)} categories to {args.size}×{args.size}...")
    total_resized, total_skipped, total_errors = 0, 0, 0

    for key, entry in entries:
        result = resize_category(entry, target_size=args.size, dry_run=args.dry_run)
        if result["skipped_category"]:
            print(f"  {key}: sentinel exists — skipped")
        else:
            print(f"  {key}: resized={result['resized']} skipped={result['skipped']} errors={result['errors']}")
        total_resized += result["resized"]
        total_skipped += result["skipped"]
        total_errors += result["errors"]

    print(f"\nTotal: resized={total_resized} skipped={total_skipped} errors={total_errors}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6-2: 전체 테스트 실행 — 8개 통과 확인 (CLI는 통합 테스트 불필요)**

Run: `pytest tests/test_stage0.py -v`
Expected: 8 passed

- [ ] **Step 6-3: Commit**

```bash
git add stage0_resize.py
git commit -m "feat(stage0): add CLI with --config, --size, --domain-filter, --dry-run, --clean"
```


### Task 7: Colab 실행 셀 추가

**Files:**
- Modify: `docs/작업일지/stage1_6_execute.md:32` — Stage 0 셀 삽입 (Stage 1 앞)

- [ ] **Step 7-1: `stage1_6_execute.md`에 Stage 0 셀 추가**

Line 32 (`---`) 뒤, line 34 (`## Stage 1: ROI 추출`) 앞에 삽입:

```markdown

## Stage 0: 이미지 리사이즈 (512×512)

**Sentinel:** `{cat_dir}/.stage0_resize_512_done`
**병렬:** 카테고리 단위 `ThreadPoolExecutor`

```python
import json, sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm.auto import tqdm

sys.path.insert(0, "/content/aroma")
from stage0_resize import resize_category, clean_category

REPO   = Path("/content/drive/MyDrive/project/aroma")
CONFIG = json.loads((REPO / "dataset_config.json").read_text(encoding="utf-8"))

DOMAIN_FILTER = "isp"   # "isp" / "mvtec" / "visa"
LABEL = {"isp": "ISP-AD", "mvtec": "MVTec AD", "visa": "VisA"}[DOMAIN_FILTER]
TARGET_SIZE = 512
CLEAN_FIRST = True  # True: stage1~6 출력물 먼저 삭제

# 병렬 설정
CAT_THREADS = 4   # I/O bound — 높게 설정 가능

# 처리 대상 수집
cat_tasks, skip = [], 0
seen = set()
for key, entry in CONFIG.items():
    if key.startswith("_") or entry["domain"] != DOMAIN_FILTER:
        continue
    cat_dir = Path(entry["seed_dir"]).parents[1]
    if str(cat_dir) in seen:
        continue
    seen.add(str(cat_dir))
    sentinel = cat_dir / f".stage0_resize_{TARGET_SIZE}_done"
    if sentinel.exists():
        skip += 1
    else:
        cat_tasks.append((key, entry))

if not cat_tasks:
    print(f"✓ {LABEL} Stage 0 모든 작업 완료 (skip {skip}건)")
else:
    print(f"{LABEL}: {len(cat_tasks)} categories 처리 예정 (skip {skip}건)")
    failed = []

    # Clean (optional)
    if CLEAN_FIRST:
        for key, entry in cat_tasks:
            deleted = clean_category(entry)
            if deleted:
                print(f"  🗑 {key}: {len(deleted)} items deleted")

    # Resize
    def _do(args):
        key, entry = args
        return key, resize_category(entry, target_size=TARGET_SIZE)

    with ThreadPoolExecutor(max_workers=CAT_THREADS) as ex:
        futs = {ex.submit(_do, t): t[0] for t in cat_tasks}
        for fut in tqdm(as_completed(futs), total=len(futs), desc="Stage 0"):
            key = futs[fut]
            try:
                _, result = fut.result()
                print(f"  ✓ {key}: resized={result['resized']} skip={result['skipped']}")
            except Exception as e:
                print(f"  ✗ {key}: {e}")
                failed.append(key)

    if failed:
        print(f"\n⚠ Failed: {failed}")
    else:
        print(f"\n✓ {LABEL} Stage 0 완료")
```

---
```

- [ ] **Step 7-2: Commit**

```bash
git add "docs/작업일지/stage1_6_execute.md"
git commit -m "docs: add Stage 0 resize cell to Colab execution playbook"
```


### Task 8: 전체 regression 테스트

**Files:** (none modified)

- [ ] **Step 8-1: 전체 테스트 실행**

Run: `pytest tests/ -v`
Expected: 133+ passed (기존 125 + 새로 추가된 8)

- [ ] **Step 8-2: 실패 시 수정 후 재실행**

Any failures → fix → re-run until all pass.

- [ ] **Step 8-3: Final commit (if any fixes)**

```bash
git add -A
git commit -m "fix: address test regression from stage0 integration"
```
