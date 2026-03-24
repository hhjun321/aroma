# Stage 6 Dataset Builder — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `utils/dataset_builder.py` + `stage6_dataset_builder.py` 구현 — Stage 4/5 출력을 baseline / aroma_full / aroma_pruned 3개 dataset group 으로 구성한다.

**Architecture:** `utils/dataset_builder.py` 는 순수 라이브러리; `stage6_dataset_builder.py` 는 CLI + 노트북 진입점 래퍼. Stage 5 (`utils/quality_scoring.py` + `stage5_quality_scoring.py`) 패턴과 동일한 분리 구조.

**Tech Stack:** Python, shutil, `utils/parallel.py` (resolve_workers/run_parallel)

**Spec:** `docs/superpowers/specs/2026-03-24-stage6-dataset-builder-design.md`

---

## File Structure

| 파일 | 역할 |
|------|------|
| Create: `utils/dataset_builder.py` | `_copy_worker`, `_collect_defect_paths`, `build_dataset_groups` |
| Create: `stage6_dataset_builder.py` | `run_dataset_builder` 래퍼 + CLI |
| Create: `tests/test_stage6.py` | 8 test cases |
| Modify: `docs/작업일지/2026-03-23.md` | Step 12-1/2a/2b/2c/3 셀 추가 |

---

## Task 1: 실패 테스트 먼저 작성

**Files:**
- Create: `tests/test_stage6.py`

- [ ] `tests/test_stage6.py` 생성:

```python
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
        assert any("quality_scores.json" in str(warning.message) for warning in w)

    assert result["aroma_pruned"]["defect_count"] == 0


def test_empty_stage4_output(tmp_path):
    from utils.dataset_builder import build_dataset_groups
    image_dir = _make_image_dir(tmp_path, count=2)
    seed_dirs = _make_seed_dirs(tmp_path, ["scratch"])
    # stage4_output 디렉터리 없음

    result = build_dataset_groups(str(tmp_path), str(image_dir), seed_dirs)

    assert result["aroma_full"]["defect_count"] == 0
    assert result["aroma_pruned"]["defect_count"] == 0
```

- [ ] 실패 확인: `pytest tests/test_stage6.py -v`
  - Expected: `ImportError: cannot import name 'build_dataset_groups' from 'utils.dataset_builder'`

---

## Task 2: `utils/dataset_builder.py` 구현

**Files:**
- Create: `utils/dataset_builder.py`

- [ ] `utils/dataset_builder.py` 생성:

```python
"""utils/dataset_builder.py

Stage 6 증강 데이터셋 구성 라이브러리.

baseline / aroma_full / aroma_pruned 3개 group 을 MVTec-style 폴더 구조로 생성.
파일 복사 방식 (symlink 미사용) — Colab/Google Drive 환경 호환.
"""
from __future__ import annotations

import json
import shutil
import warnings
from pathlib import Path


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _copy_worker(args_tuple: tuple) -> str | None:
    """ProcessPoolExecutor pickle-safe 복사 워커.

    args_tuple: (src_path_str, dst_path_str)
    dst 이미 존재하면 스킵 (파일 단위 resume).
    tasks 구성 예: tasks = [(str(src), str(dst)) for src, dst in copy_pairs]
    """
    src_path_str, dst_path_str = args_tuple
    dst = Path(dst_path_str)
    if dst.exists():
        return dst_path_str
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_path_str, dst_path_str)
    return dst_path_str


def _collect_defect_paths(
    cat_dir: str,
    pruning_threshold: float | None = None,
) -> list[tuple[str, str]]:
    """stage4_output/{seed_id}/defect/*.png 수집 (단일 깊이 *).

    Returns:
        List of (src_path_str, dst_filename) where
        dst_filename = "{seed_id}_{image_id}.png" (충돌 방지).
    """
    stage4_dir = Path(cat_dir) / "stage4_output"
    if not stage4_dir.exists():
        return []

    result = []
    for seed_dir in sorted(stage4_dir.iterdir()):
        if not seed_dir.is_dir():
            continue
        defect_dir = seed_dir / "defect"
        if not defect_dir.exists():
            continue

        img_paths = sorted(defect_dir.glob("*.png"))
        if not img_paths:
            continue

        if pruning_threshold is not None:
            quality_file = seed_dir / "quality_scores.json"
            if not quality_file.exists():
                warnings.warn(
                    f"quality_scores.json 없음, seed 스킵: {seed_dir.name}",
                    stacklevel=3,
                )
                continue
            quality_data = json.loads(quality_file.read_text())
            quality_map = {
                s["image_id"]: s["quality_score"]
                for s in quality_data.get("scores", [])
            }
            for p in img_paths:
                if quality_map.get(p.stem, 0.0) >= pruning_threshold:
                    result.append((str(p), f"{seed_dir.name}_{p.name}"))
        else:
            for p in img_paths:
                result.append((str(p), f"{seed_dir.name}_{p.name}"))

    return result


def _copy_images(src_dir: Path, dst_dir: Path, num_workers: int,
                 desc: str = "") -> int:
    """src_dir 아래 PNG/JPG 전체를 dst_dir 로 복사. 복사 파일 수 반환."""
    from utils.parallel import run_parallel
    imgs = (
        sorted(src_dir.glob("*.png"))
        + sorted(src_dir.glob("*.jpg"))
        + sorted(src_dir.glob("*.jpeg"))
    )
    if not imgs:
        return 0
    dst_dir.mkdir(parents=True, exist_ok=True)
    tasks = [(str(s), str(dst_dir / s.name)) for s in imgs]
    run_parallel(_copy_worker, tasks, num_workers, desc=desc)
    return len(imgs)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_dataset_groups(
    cat_dir: str,
    image_dir: str,
    seed_dirs: list[str],
    pruning_threshold: float = 0.6,
    workers: int = 0,
) -> dict:
    """3개 dataset group 을 구성하고 build_report.json 을 저장한다.

    Args:
        cat_dir: 카테고리 루트 디렉터리 (e.g. `.../mvtec/bottle`).
        image_dir: 원본 train/good 이미지 디렉터리.
        seed_dirs: dataset_config.json 에서 추출한 seed_dir 경로 목록.
            baseline/test/{defect_type}/ 구성에 사용.
        pruning_threshold: aroma_pruned 의 quality_score 최솟값.
        workers: 병렬 워커 수 (0=순차).

    Returns:
        build_report dict. build_report.json 으로도 저장.

    Skip:
        build_report.json 이 존재하고 저장된 pruning_threshold 와 일치하면
        재생성 없이 로드하여 반환. 불일치 시 전체 재생성.
    """
    cat_path = Path(cat_dir)
    aug_dir = cat_path / "augmented_dataset"
    report_path = aug_dir / "build_report.json"

    # Skip 조건 확인
    if report_path.exists():
        cached = json.loads(report_path.read_text())
        if abs(cached.get("pruning_threshold", -1) - pruning_threshold) < 1e-6:
            return cached

    from utils.parallel import resolve_workers
    num_workers = resolve_workers(workers)

    # ── baseline/train/good/ ───────────────────────────────────────────────
    baseline_good = aug_dir / "baseline" / "train" / "good"
    good_count = _copy_images(Path(image_dir), baseline_good, num_workers,
                               desc="baseline/train/good")

    # ── baseline/test/ ────────────────────────────────────────────────────
    # test/good/ (image_dir 기준: {cat}/train/good → {cat}/test/good)
    test_good_src = Path(image_dir).parents[1] / "test" / "good"
    if test_good_src.exists():
        _copy_images(test_good_src,
                     aug_dir / "baseline" / "test" / "good",
                     num_workers, desc="baseline/test/good")

    # test/{defect_type}/ (seed_dirs 에서 추출)
    for sd in seed_dirs:
        defect_type = Path(sd).name
        src = Path(sd)
        if src.exists():
            _copy_images(src,
                         aug_dir / "baseline" / "test" / defect_type,
                         num_workers, desc=f"baseline/test/{defect_type}")

    # ── aroma_full/train/ ─────────────────────────────────────────────────
    _copy_images(Path(image_dir),
                 aug_dir / "aroma_full" / "train" / "good",
                 num_workers, desc="aroma_full/train/good")

    full_defect_pairs = _collect_defect_paths(cat_dir, pruning_threshold=None)
    if full_defect_pairs:
        from utils.parallel import run_parallel
        full_defect_dst = aug_dir / "aroma_full" / "train" / "defect"
        full_defect_dst.mkdir(parents=True, exist_ok=True)
        tasks = [(src, str(full_defect_dst / dst_name))
                 for src, dst_name in full_defect_pairs]
        run_parallel(_copy_worker, tasks, num_workers, desc="aroma_full/train/defect")

    # ── aroma_pruned/train/ ───────────────────────────────────────────────
    _copy_images(Path(image_dir),
                 aug_dir / "aroma_pruned" / "train" / "good",
                 num_workers, desc="aroma_pruned/train/good")

    pruned_defect_pairs = _collect_defect_paths(cat_dir,
                                                 pruning_threshold=pruning_threshold)
    if pruned_defect_pairs:
        from utils.parallel import run_parallel
        pruned_defect_dst = aug_dir / "aroma_pruned" / "train" / "defect"
        pruned_defect_dst.mkdir(parents=True, exist_ok=True)
        tasks = [(src, str(pruned_defect_dst / dst_name))
                 for src, dst_name in pruned_defect_pairs]
        run_parallel(_copy_worker, tasks, num_workers, desc="aroma_pruned/train/defect")

    report = {
        "pruning_threshold": pruning_threshold,
        "baseline":     {"good_count": good_count, "defect_count": 0},
        "aroma_full":   {"good_count": good_count,
                         "defect_count": len(full_defect_pairs)},
        "aroma_pruned": {"good_count": good_count,
                         "defect_count": len(pruned_defect_pairs)},
    }
    aug_dir.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2))
    return report
```

- [ ] 전체 테스트 통과 확인: `pytest tests/test_stage6.py -v`
  - Expected: 8 passed

- [ ] 커밋:
```bash
git add utils/dataset_builder.py tests/test_stage6.py
git commit -m "feat: add utils/dataset_builder.py with 3-group dataset construction"
```

---

## Task 3: `stage6_dataset_builder.py` — CLI 래퍼

**Files:**
- Create: `stage6_dataset_builder.py`

- [ ] `stage6_dataset_builder.py` 생성:

```python
"""stage6_dataset_builder.py

Stage 6 of the AROMA pipeline: augmented dataset construction.

Builds baseline / aroma_full / aroma_pruned dataset groups from
Stage 4 synthesis outputs and Stage 5 quality scores.
"""
from __future__ import annotations

import argparse

from utils.dataset_builder import build_dataset_groups


def run_dataset_builder(
    cat_dir: str,
    image_dir: str,
    seed_dirs: list[str],
    pruning_threshold: float = 0.6,
    workers: int = 0,
) -> dict:
    """build_dataset_groups() 의 직접 래퍼.

    CLI 진입점과 노트북 셀 양쪽이 동일한 함수를 호출하도록 단일화.
    """
    return build_dataset_groups(
        cat_dir=cat_dir,
        image_dir=image_dir,
        seed_dirs=seed_dirs,
        pruning_threshold=pruning_threshold,
        workers=workers,
    )


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Stage 6 — augmented dataset construction"
    )
    p.add_argument("--cat_dir", required=True,
                   help="카테고리 루트 디렉터리.")
    p.add_argument("--image_dir", required=True,
                   help="원본 train/good 이미지 디렉터리.")
    p.add_argument("--seed_dirs", nargs="+", required=True,
                   help="dataset_config.json 의 seed_dir 경로 목록 (공백 구분).")
    p.add_argument("--pruning_threshold", type=float, default=0.6,
                   help="aroma_pruned quality_score 최솟값 (기본 0.6).")
    p.add_argument("--workers", type=int, default=0,
                   help="병렬 워커 수 (0=순차, -1=자동, N>=2=N 프로세스).")
    return p


def main() -> None:
    args = _build_parser().parse_args()
    result = run_dataset_builder(
        cat_dir=args.cat_dir,
        image_dir=args.image_dir,
        seed_dirs=args.seed_dirs,
        pruning_threshold=args.pruning_threshold,
        workers=args.workers,
    )
    print(f"baseline   good={result['baseline']['good_count']}")
    print(f"aroma_full defect={result['aroma_full']['defect_count']}")
    print(f"aroma_pruned defect={result['aroma_pruned']['defect_count']}")


if __name__ == "__main__":
    main()
```

- [ ] import 확인: `python -c "from stage6_dataset_builder import run_dataset_builder; print('OK')"`
  - Expected: `OK`

- [ ] 커밋:
```bash
git add stage6_dataset_builder.py
git commit -m "feat: add stage6_dataset_builder.py CLI wrapper"
```

---

## Task 4: 작업일지 Step 12-1/2a/2b/2c/3 셀 추가

**Files:**
- Modify: `docs/작업일지/2026-03-23.md`

- [ ] `docs/작업일지/2026-03-23.md` 의 `## 다음 액션` 섹션 앞에 다음 내용 삽입:

````markdown
---

## Stage 6 Colab 실행 셀

### Step 12-1: 단건 데이터셋 구성 테스트

```python
import json, sys
from pathlib import Path

sys.path.insert(0, "/content/aroma")
from stage6_dataset_builder import run_dataset_builder

REPO   = Path("/content/drive/MyDrive/project/aroma")
CONFIG = json.loads((REPO / "dataset_config.json").read_text())

# VisA candle 단건 테스트
TEST_KEY = "visa_candle"   # dataset_config.json 키 이름 확인 후 변경
entry = CONFIG[TEST_KEY]
cat_dir   = str(Path(entry["seed_dir"]).parents[1])
image_dir = entry["image_dir"]
seed_dirs = [v["seed_dir"] for k, v in CONFIG.items()
             if not k.startswith("_") and str(Path(v["seed_dir"]).parents[1]) == cat_dir]

result = run_dataset_builder(cat_dir=cat_dir, image_dir=image_dir, seed_dirs=seed_dirs)
print(result)
```

---

### Step 12-2a/b/c: 배치 데이터셋 구성

```python
import json, sys
from pathlib import Path
from collections import defaultdict
from tqdm.auto import tqdm

sys.path.insert(0, "/content/aroma")
from stage6_dataset_builder import run_dataset_builder

REPO   = Path("/content/drive/MyDrive/project/aroma")
CONFIG = json.loads((REPO / "dataset_config.json").read_text())

DOMAIN_FILTER = "visa"   # "isp" / "mvtec" / "visa"
LABEL = {"isp": "ISP-AD", "mvtec": "MVTec AD", "visa": "VisA"}[DOMAIN_FILTER]

cat_map = {}
cat_seed_dirs = defaultdict(list)
for key, entry in CONFIG.items():
    if key.startswith("_") or entry["domain"] != DOMAIN_FILTER:
        continue
    cat_dir   = str(Path(entry["seed_dir"]).parents[1])
    image_dir = entry["image_dir"]
    if cat_dir in cat_map and cat_map[cat_dir] != image_dir:
        raise ValueError(f"image_dir 불일치: {cat_dir}")
    cat_map[cat_dir] = image_dir
    cat_seed_dirs[cat_dir].append(entry["seed_dir"])

all_cats, skip = [], 0
for cat_dir, image_dir in cat_map.items():
    if (Path(cat_dir) / "augmented_dataset" / "build_report.json").exists():
        skip += 1
    else:
        all_cats.append((cat_dir, image_dir, cat_seed_dirs[cat_dir]))

if not all_cats:
    print(f"✓ {LABEL} 모든 작업 완료")
else:
    print(f"{LABEL}: {len(all_cats)} categories 처리 예정 (skip {skip}건)")
    failed = []
    for cat_dir, image_dir, seed_dirs in tqdm(all_cats, desc=f"Stage6 {LABEL}"):
        try:
            run_dataset_builder(cat_dir=cat_dir, image_dir=image_dir,
                                seed_dirs=seed_dirs)
        except Exception as e:
            failed.append({"category": Path(cat_dir).name,
                           "error": str(e), "type": type(e).__name__})
    print("\n" + (f"✓ {LABEL} 완료" if not failed else f"✗ {len(failed)}건 실패"))
    for f in failed:
        print(f"  [{f['category']}] {f['type']}: {f['error'][:100]}")
```

*(Step 12-2a: `DOMAIN_FILTER = "isp"`, Step 12-2b: `"mvtec"`, Step 12-2c: `"visa"`)*

---

### Step 12-3: 데이터셋 구성 검증

```python
import json
from pathlib import Path

REPO   = Path("/content/drive/MyDrive/project/aroma")
CONFIG = json.loads((REPO / "dataset_config.json").read_text())

for domain in ["isp", "mvtec", "visa"]:
    cat_dirs = set()
    for key, entry in CONFIG.items():
        if key.startswith("_") or entry["domain"] != domain:
            continue
        cat_dirs.add(str(Path(entry["seed_dir"]).parents[1]))

    done, missing = 0, []
    for cd in sorted(cat_dirs):
        report = Path(cd) / "augmented_dataset" / "build_report.json"
        if report.exists():
            done += 1
        else:
            missing.append(Path(cd).name)

    label = {"isp": "ISP-AD", "mvtec": "MVTec AD", "visa": "VisA"}[domain]
    status = f"✓ {done}/{len(cat_dirs)}" if not missing else f"✗ {missing}"
    print(f"{label}: {status}")
```

````

- [ ] 커밋:
```bash
git add docs/작업일지/2026-03-23.md
git commit -m "docs: add Stage 6 Colab cells (Step 12-1/2/3) to work journal"
```

---

## 검증

- [ ] 전체 테스트: `pytest tests/test_stage6.py -v`
  - Expected: 8 passed
- [ ] 회귀 없음: `pytest tests/ -q --ignore=tests/test_parallel.py`
  - Expected: 모든 기존 테스트 통과
