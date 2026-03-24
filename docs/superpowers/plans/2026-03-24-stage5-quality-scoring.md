# Stage 5 Quality Scoring — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `utils/quality_scoring.py` + `stage5_quality_scoring.py` + `tests/test_stage5.py` 구현 — Stage 4 합성 이미지에 대해 artifact_score + blur_score 2-metric 품질 점수를 계산하고 `quality_scores.json`으로 저장한다.

**Architecture:** `utils/quality_scoring.py`는 순수 라이브러리 (score_image, score_defect_images); `stage5_quality_scoring.py`는 CLI + 노트북 진입점 래퍼. 기존 `utils/suitability.py` → `stage3_layout_logic.py` 패턴과 동일한 분리 구조.

**Tech Stack:** Python, NumPy, OpenCV, `utils/parallel.py` (resolve_workers/run_parallel)

**Spec:** `docs/superpowers/specs/2026-03-24-stage5-quality-scoring-design.md`

---

## File Structure

| 파일 | 역할 |
|------|------|
| Create: `utils/quality_scoring.py` | `_score_artifacts`, `_score_sharpness`, `_score_single_worker`, `score_image`, `score_defect_images` |
| Create: `stage5_quality_scoring.py` | `run_quality_scoring` 래퍼 + CLI |
| Create: `tests/test_stage5.py` | 7 test cases (spec 정의 그대로) |

---

## Task 1: `utils/quality_scoring.py` — 라이브러리 구현

**Files:**
- Create: `utils/quality_scoring.py`

### 1-1. 실패 테스트 먼저 작성

- [ ] `tests/test_stage5.py` 생성 — 처음 3개 테스트만 (imports가 실패하도록)

```python
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
```

- [ ] 실패 확인: `pytest tests/test_stage5.py -v`
  - Expected: `ImportError: cannot import name 'score_image' from 'utils.quality_scoring'` (파일 없음)

### 1-2. `utils/quality_scoring.py` 구현

- [ ] `utils/quality_scoring.py` 생성:

```python
"""utils/quality_scoring.py

Stage 5 합성 품질 점수 계산 라이브러리.

2-metric: artifact_score (higher = fewer artifacts) + blur_score (higher = sharper).
CASDA score_casda_quality.py 에서 domain-agnostic 2개만 이식.
color_score 는 금속 표면 전용 캘리브레이션이므로 제거.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Internal metric functions
# ---------------------------------------------------------------------------

def _score_artifacts(gray: np.ndarray) -> float:
    """Gradient 기반 합성 아티팩트 감지.

    Sobel gradient magnitude의 3σ 이상치 비율(edge_score) +
    Laplacian energy / gradient mean 비율(hf_score).
    가중치: 0.6 * edge_score + 0.4 * hf_score.
    Returns: 1.0 = 아티팩트 없음 (higher = better).
    """
    gray_f = gray.astype(np.float32)

    # Sobel gradient magnitude
    gx = cv2.Sobel(gray_f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_f, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx ** 2 + gy ** 2)

    mean_mag = float(np.mean(mag))
    std_mag = float(np.std(mag))
    if mean_mag < 1e-6:
        return 1.0  # 균일 이미지 — 아티팩트 없음

    # edge_score: 3σ 이상치 픽셀 비율 (낮을수록 좋음 → 1 - ratio)
    threshold = mean_mag + 3.0 * std_mag
    outlier_ratio = float(np.mean(mag > threshold))
    edge_score = 1.0 - min(outlier_ratio * 10.0, 1.0)  # 10% 이상이면 최악

    # hf_score: Laplacian energy / gradient mean (높은 비율은 고주파 아티팩트)
    lap = cv2.Laplacian(gray_f, cv2.CV_32F)
    lap_energy = float(np.mean(np.abs(lap)))
    hf_ratio = lap_energy / (mean_mag + 1e-6)
    hf_score = 1.0 - min(hf_ratio / 5.0, 1.0)  # 5.0 이상이면 최악

    return float(np.clip(0.6 * edge_score + 0.4 * hf_score, 0.0, 1.0))


def _score_sharpness(gray: np.ndarray) -> float:
    """Laplacian variance + gradient contrast 기반 선명도.

    Laplacian variance(lap_score) + P90/P50 gradient contrast ratio(edge_sharpness).
    가중치: 0.5 * lap_score + 0.5 * edge_sharpness.
    Returns: 1.0 = 완전 선명 (higher = better).
    """
    gray_f = gray.astype(np.float32)

    # Laplacian variance — 선명한 이미지일수록 높음
    lap = cv2.Laplacian(gray_f, cv2.CV_32F)
    lap_var = float(np.var(lap))
    # 정규화: 1000을 기준값으로 사용 (경험적)
    lap_score = float(np.clip(lap_var / 1000.0, 0.0, 1.0))

    # Gradient contrast: P90 / P50 비율
    gx = cv2.Sobel(gray_f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_f, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx ** 2 + gy ** 2).flatten()
    p50 = float(np.percentile(mag, 50))
    p90 = float(np.percentile(mag, 90))
    if p50 < 1e-6:
        edge_sharpness = 0.0
    else:
        ratio = p90 / (p50 + 1e-6)
        edge_sharpness = float(np.clip((ratio - 1.0) / 9.0, 0.0, 1.0))  # ratio 1~10 → 0~1

    return float(np.clip(0.5 * lap_score + 0.5 * edge_sharpness, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Worker (pickle-safe for ProcessPoolExecutor)
# ---------------------------------------------------------------------------

def _score_single_worker(args_tuple: tuple) -> dict | None:
    """ProcessPoolExecutor pickle-safe 워커.

    args: (img_path_str,)
    Returns: {"image_id": str, "artifact": float, "blur": float} | None
    """
    (img_path_str,) = args_tuple
    img = cv2.imread(img_path_str)
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    image_id = Path(img_path_str).stem
    return {
        "image_id": image_id,
        "artifact": _score_artifacts(gray),
        "blur": _score_sharpness(gray),
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def score_image(
    img_bgr: np.ndarray,
    w_artifact: float = 0.5,
    w_blur: float = 0.5,
) -> dict:
    """단일 이미지 numpy array를 받아 품질 점수를 반환.

    Args:
        img_bgr: BGR 이미지 numpy array.
        w_artifact: artifact_score 가중치 (기본 0.5).
        w_blur: blur_score 가중치 (기본 0.5).

    Returns:
        {"artifact_score": float, "blur_score": float, "quality_score": float}

    Raises:
        ValueError: w_artifact + w_blur != 1.0
    """
    if abs(w_artifact + w_blur - 1.0) > 1e-6:
        raise ValueError(
            f"w_artifact + w_blur must equal 1.0, got {w_artifact + w_blur}"
        )
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    artifact = _score_artifacts(gray)
    blur = _score_sharpness(gray)
    quality = w_artifact * artifact + w_blur * blur
    return {
        "artifact_score": round(float(artifact), 4),
        "blur_score": round(float(blur), 4),
        "quality_score": round(float(quality), 4),
    }


def score_defect_images(
    stage4_seed_dir: str,
    w_artifact: float = 0.5,
    w_blur: float = 0.5,
    workers: int = 0,
) -> dict:
    """stage4_seed_dir/defect/*.png 전체를 평가하고 quality_scores.json 저장.

    Args:
        stage4_seed_dir: Stage 4 seed 출력 디렉터리 경로.
        w_artifact: artifact 가중치 (기본 0.5).
        w_blur: blur 가중치 (기본 0.5).
        workers: 병렬 워커 수 (0=순차, -1=자동, N>=2=N 프로세스).

    Returns:
        {"weights": {...}, "scores": [...], "stats": {...}}
        count == 0 이면 파일 미생성, 빈 stats 반환.
        quality_scores.json 이 이미 존재하면 재계산 없이 로드하여 반환 (skip).

    Raises:
        ValueError: w_artifact + w_blur != 1.0
    """
    if abs(w_artifact + w_blur - 1.0) > 1e-6:
        raise ValueError(
            f"w_artifact + w_blur must equal 1.0, got {w_artifact + w_blur}"
        )

    seed_dir = Path(stage4_seed_dir)
    cache_path = seed_dir / "quality_scores.json"

    # Skip: 이미 존재하면 로드하여 반환
    if cache_path.exists():
        return json.loads(cache_path.read_text())

    defect_dir = seed_dir / "defect"
    img_paths = sorted(defect_dir.glob("*.png")) if defect_dir.exists() else []

    # count == 0: 빈 stats 반환, 파일 미생성
    if not img_paths:
        return {
            "weights": {"artifact": w_artifact, "blur": w_blur},
            "scores": [],
            "stats": {},
        }

    from utils.parallel import resolve_workers, run_parallel

    num_workers = resolve_workers(workers)
    tasks = [(str(p),) for p in img_paths]
    raw_results = run_parallel(_score_single_worker, tasks, num_workers,
                               desc="Stage5 quality scoring")

    # 병렬 결과를 image_id 키 dict로 수집
    result_map = {r["image_id"]: r for r in raw_results if r is not None}

    # sorted 순서로 재조립
    scores = []
    for p in img_paths:
        image_id = p.stem
        r = result_map.get(image_id)
        if r is None:
            continue
        artifact = r["artifact"]
        blur = r["blur"]
        quality = w_artifact * artifact + w_blur * blur
        scores.append({
            "image_id": image_id,
            "artifact_score": round(float(artifact), 4),
            "blur_score": round(float(blur), 4),
            "quality_score": round(float(quality), 4),
        })

    # stats 계산
    q_values = [s["quality_score"] for s in scores]
    q_arr = np.array(q_values)
    stats = {
        "count": len(q_arr),
        "mean": round(float(np.mean(q_arr)), 4),
        "std": round(float(np.std(q_arr)), 4),
        "min": round(float(np.min(q_arr)), 4),
        "max": round(float(np.max(q_arr)), 4),
        "p25": round(float(np.percentile(q_arr, 25)), 4),
        "p50": round(float(np.percentile(q_arr, 50)), 4),
        "p75": round(float(np.percentile(q_arr, 75)), 4),
        "p90": round(float(np.percentile(q_arr, 90)), 4),
    }

    result = {
        "weights": {"artifact": w_artifact, "blur": w_blur},
        "scores": scores,
        "stats": stats,
    }
    cache_path.write_text(json.dumps(result, indent=2))
    return result
```

- [ ] 테스트 통과 확인: `pytest tests/test_stage5.py::test_score_image_basic tests/test_stage5.py::test_score_image_weight_validation tests/test_stage5.py::test_score_artifacts_clean -v`
  - Expected: 3 passed

---

## Task 2: 나머지 테스트 추가 및 통과

**Files:**
- Modify: `tests/test_stage5.py`

- [ ] 나머지 4개 테스트 추가 (파일 끝에 append):

```python
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
    import cv2
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
```

- [ ] 전체 테스트 통과 확인: `pytest tests/test_stage5.py -v`
  - Expected: 7 passed

- [ ] 커밋:
```bash
git add utils/quality_scoring.py tests/test_stage5.py
git commit -m "feat: add utils/quality_scoring.py with 2-metric artifact+blur scoring"
```

---

## Task 3: `stage5_quality_scoring.py` — CLI 래퍼 구현

**Files:**
- Create: `stage5_quality_scoring.py`

- [ ] `stage5_quality_scoring.py` 생성:

```python
"""stage5_quality_scoring.py

Stage 5 of the AROMA pipeline: synthesis quality scoring.

Computes artifact_score + blur_score for all defect images in a
Stage 4 seed output directory. Saves quality_scores.json alongside
the defect images.
"""
from __future__ import annotations

import argparse

from utils.quality_scoring import score_defect_images


def run_quality_scoring(
    stage4_seed_dir: str,
    w_artifact: float = 0.5,
    w_blur: float = 0.5,
    workers: int = 0,
) -> dict:
    """score_defect_images()의 직접 래퍼.

    CLI 진입점과 작업일지 셀 양쪽이 동일한 함수를 호출하도록 단일화.
    """
    return score_defect_images(
        stage4_seed_dir=stage4_seed_dir,
        w_artifact=w_artifact,
        w_blur=w_blur,
        workers=workers,
    )


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Stage 5 — synthesis quality scoring (artifact + blur)"
    )
    p.add_argument("--stage4_seed_dir", required=True,
                   help="Stage 4 seed output directory (contains defect/*.png).")
    p.add_argument("--w_artifact", type=float, default=0.5,
                   help="Weight for artifact_score (default 0.5).")
    p.add_argument("--w_blur", type=float, default=0.5,
                   help="Weight for blur_score (default 0.5).")
    p.add_argument("--workers", type=int, default=0,
                   help="병렬 워커 수 (0=순차, -1=자동, N>=2=N개 프로세스).")
    return p


def main() -> None:
    args = _build_parser().parse_args()
    result = run_quality_scoring(
        stage4_seed_dir=args.stage4_seed_dir,
        w_artifact=args.w_artifact,
        w_blur=args.w_blur,
        workers=args.workers,
    )
    stats = result.get("stats", {})
    count = stats.get("count", 0)
    print(f"Scored {count} images → quality_scores.json")
    if count:
        print(f"  mean={stats['mean']:.4f}  std={stats['std']:.4f}"
              f"  p50={stats['p50']:.4f}  min={stats['min']:.4f}  max={stats['max']:.4f}")


if __name__ == "__main__":
    main()
```

- [ ] import 동작 확인: `python -c "from stage5_quality_scoring import run_quality_scoring; print('OK')`
  - Expected: `OK`

- [ ] 커밋:
```bash
git add stage5_quality_scoring.py
git commit -m "feat: add stage5_quality_scoring.py CLI wrapper"
```

---

## Task 4: 작업일지 Step 11-1/2a/2b/2c/3 셀 추가

**Files:**
- Modify: `docs/작업일지/2026-03-23.md`

Step 11-1 ~ 11-3 셀을 작업일지에 추가한다 (spec의 Step 11-2a/b/c 패턴 그대로).

- [ ] `docs/작업일지/2026-03-23.md`의 `## 다음 액션` 섹션 앞에 다음 내용 삽입:

````markdown
---

## Colab 실행 셀

### Step 11-1: 단건 품질 점수 계산 테스트

```python
import sys
import cv2
from pathlib import Path

sys.path.insert(0, "/content/aroma")
from utils.quality_scoring import score_image

# VisA candle 첫 번째 seed의 첫 번째 defect 이미지로 테스트
SAMPLE = Path("/content/drive/MyDrive/project/aroma") \
    / "visa/candle/stage4_output" / sorted(
        (Path("/content/drive/MyDrive/project/aroma") / "visa/candle/stage4_output").iterdir()
    )[0].name / "defect" / "000.png"

img = cv2.imread(str(SAMPLE))
if img is None:
    print(f"이미지를 찾을 수 없음: {SAMPLE}")
else:
    result = score_image(img)
    print(result)
```

---

### Step 11-2a/b/c: 배치 품질 점수 계산

```python
import json, sys
from pathlib import Path
from tqdm.auto import tqdm

sys.path.insert(0, "/content/aroma")
from stage5_quality_scoring import run_quality_scoring

REPO   = Path("/content/drive/MyDrive/project/aroma")
CONFIG = json.loads((REPO / "dataset_config.json").read_text())

DOMAIN_FILTER = "visa"   # "isp" / "mvtec" / "visa"
LABEL = {"isp": "ISP-AD", "mvtec": "MVTec AD", "visa": "VisA"}[DOMAIN_FILTER]

ENTRIES = [(k, v) for k, v in CONFIG.items()
           if not k.startswith("_") and v["domain"] == DOMAIN_FILTER]

all_seeds, skip = [], 0
for key, entry in ENTRIES:
    cat_dir    = Path(entry["seed_dir"]).parents[1]
    stage4_dir = cat_dir / "stage4_output"
    if not stage4_dir.exists():
        continue
    for d in stage4_dir.iterdir():
        if not d.is_dir():
            continue
        if (d / "quality_scores.json").exists():
            skip += 1
        else:
            all_seeds.append((cat_dir, d.name))

if not all_seeds:
    print(f"✓ {LABEL} 모든 작업 완료")
else:
    print(f"{LABEL}: {len(all_seeds)} seeds 처리 예정 (skip {skip}건)")
    failed = []
    for cat_dir, seed_id in tqdm(all_seeds, desc=f"Stage5 {LABEL}"):
        try:
            run_quality_scoring(
                stage4_seed_dir = str(cat_dir / "stage4_output" / seed_id),
                workers = 0,
            )
        except Exception as e:
            failed.append({"category": cat_dir.name, "seed": seed_id,
                           "error": str(e), "type": type(e).__name__})
    print("\n" + (f"✓ {LABEL} 완료" if not failed else f"✗ {len(failed)}건 실패"))
    for f in failed:
        print(f"  [{f['category']}/{f['seed']}] {f['type']}: {f['error'][:100]}")
```

*(Step 11-2a: `DOMAIN_FILTER = "isp"`, Step 11-2b: `"mvtec"`, Step 11-2c: `"visa"`)*

---

### Step 11-3: 점수 분포 검증

```python
import json
from pathlib import Path
import numpy as np

REPO   = Path("/content/drive/MyDrive/project/aroma")
CONFIG = json.loads((REPO / "dataset_config.json").read_text())

all_scores = []
for key, entry in CONFIG.items():
    if key.startswith("_"):
        continue
    cat_dir    = Path(entry["seed_dir"]).parents[1]
    stage4_dir = cat_dir / "stage4_output"
    if not stage4_dir.exists():
        continue
    for d in stage4_dir.iterdir():
        cache = d / "quality_scores.json"
        if not cache.exists():
            continue
        data = json.loads(cache.read_text())
        all_scores.extend(s["quality_score"] for s in data.get("scores", []))

arr = np.array(all_scores)
print(f"전체 이미지 수: {len(arr)}")
print(f"mean={np.mean(arr):.4f}  std={np.std(arr):.4f}")
print(f"P25={np.percentile(arr,25):.4f}  P50={np.percentile(arr,50):.4f}"
      f"  P75={np.percentile(arr,75):.4f}  P90={np.percentile(arr,90):.4f}")
print(f"\nPruning threshold 후보:")
print(f"  P25 기준 ({np.percentile(arr,25):.3f}): 상위 75% 선택")
print(f"  P50 기준 ({np.percentile(arr,50):.3f}): 상위 50% 선택")
print(f"  P75 기준 ({np.percentile(arr,75):.3f}): 상위 25% 선택 (aggressive)")
```

````

- [ ] 커밋:
```bash
git add docs/작업일지/2026-03-23.md
git commit -m "docs: add Stage 5 Colab cells (Step 11-1/2/3) to work journal"
```

---

## 검증

- [ ] 전체 테스트 실행: `pytest tests/test_stage5.py -v`
  - Expected: 7 passed, 0 failed
- [ ] 기존 테스트 회귀 없음: `pytest tests/ -v --ignore=tests/test_parallel.py -q`
  - Expected: 모든 기존 테스트 통과
