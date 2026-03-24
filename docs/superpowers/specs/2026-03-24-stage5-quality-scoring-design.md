# Stage 5 Quality Scoring — Design Spec

**Date:** 2026-03-24
**Status:** Approved (rev 2 — spec reviewer 이슈 반영)

---

## Overview

Stage 4 (MPB synthesis) 결과물에 대해 2-metric 합성 품질 점수를 계산한다.
낮은 품질의 합성 이미지를 Stage 6에서 필터링하기 위한 사전 단계.

CASDA `score_casda_quality.py`에서 domain-specific color_score를 제거하고,
domain-agnostic artifact_score + blur_score 2-metric으로 AROMA에 적용한다.
(2026-03-23 작업일지의 3-metric 설계는 이 스펙으로 대체됨)

---

## Design Decisions

| 항목 | 결정 | 이유 |
|------|------|------|
| Metrics | artifact + blur (2-metric) | color_score는 금속 표면 전용 캘리브레이션 — AROMA 다양한 카테고리에 부적합 |
| 저장 위치 | `stage4_output/{seed_id}/quality_scores.json` | Stage 3 파일 불변 원칙 유지; Stage 4 결과물 자급자족 |
| 구현 구조 | `utils/quality_scoring.py` + `stage5_quality_scoring.py` | 기존 utils/suitability.py → stage3 패턴과 일치; 작업일지 셀 직접 import 가능 |
| 기본 가중치 | artifact: 0.5, blur: 0.5 | 동등 가중치; 추후 조정 가능 |
| 점수 방향성 | 모든 점수 higher = better | 1.0 = 아티팩트 없음 / 완전 선명; quality_score도 동일 |

---

## File Structure

```
utils/quality_scoring.py        — 2-metric 함수 + 공개 API (라이브러리)
stage5_quality_scoring.py       — CLI orchestrator
tests/test_stage5.py            — 단위/통합 테스트
```

---

## utils/quality_scoring.py

### 공개 API

#### `score_image()`

```python
def score_image(
    img_bgr: np.ndarray,
    w_artifact: float = 0.5,
    w_blur: float = 0.5,
) -> dict
```

- 단일 이미지 numpy array를 받아 점수를 반환
- 용도: Step 11-1 단건 테스트, 노트북 셀에서 직접 사용
- `w_artifact + w_blur != 1.0`이면 `ValueError` raise
- 반환: `{"artifact_score": float, "blur_score": float, "quality_score": float}`

#### `score_defect_images()`

```python
def score_defect_images(
    stage4_seed_dir: str,
    w_artifact: float = 0.5,
    w_blur: float = 0.5,
    workers: int = 0,
) -> dict
```

- `Path(stage4_seed_dir) / "defect"` 아래 `*.png` 전체 평가
- `w_artifact + w_blur != 1.0`이면 `ValueError` raise
- **Skip**: `stage4_seed_dir/quality_scores.json`이 이미 존재하면 재계산 없이 해당 JSON을 로드하여 반환
- **count == 0**: `defect/` 아래 유효 이미지가 없으면 `quality_scores.json`을 쓰지 않고 빈 stats dict 반환
- `quality_scores.json` 저장 후 동일 dict 반환
- **반환 형식 (fresh run, cache hit 모두 동일)**:
  ```json
  {
    "weights": {"artifact": 0.5, "blur": 0.5},
    "scores": [...],
    "stats": {...}
  }
  ```

### 내부 함수 (CASDA 로직 이식)

```python
def _score_artifacts(gray: np.ndarray) -> float
# Sobel gradient magnitude의 3σ 이상치 비율(edge_score) +
# Laplacian energy / gradient mean 비율(hf_score)
# 가중치: 0.6 * edge_score + 0.4 * hf_score
# 반환: 1.0 = 아티팩트 없음 (higher = better)

def _score_sharpness(gray: np.ndarray) -> float
# Laplacian variance(lap_score) + P90/P50 gradient contrast ratio(edge_sharpness)
# 가중치: 0.5 * lap_score + 0.5 * edge_sharpness
# 반환: 1.0 = 완전 선명 (higher = better)

def _score_single_worker(args_tuple: tuple) -> dict | None
# ProcessPoolExecutor pickle-safe 워커
# args: (img_path_str,)  ← index 없음; image_id는 Path.stem으로 직접 도출
# Returns: {"image_id": str, "artifact": float, "blur": float} | None
```

### image_id 도출 규칙

`image_id = Path(img_path_str).stem`
(Stage 4의 `_synthesize_image()`가 `{image_id}.png`로 저장하므로 stem이 곧 image_id)

### 병렬 결과 정렬

`score_defect_images()` 내부에서:
1. `defect/*.png`를 `sorted()` 로 수집 → 안정적 순서
2. 병렬 결과는 `{image_id: scores}` dict로 수집
3. 원래 sorted 순서대로 재조립하여 `scores` 리스트 구성

---

## Output Format

`stage4_output/{seed_id}/quality_scores.json`:

```json
{
  "weights": {"artifact": 0.5, "blur": 0.5},
  "scores": [
    {
      "image_id": "000",
      "artifact_score": 0.74,
      "blur_score": 0.91,
      "quality_score": 0.82
    }
  ],
  "stats": {
    "count": 50,
    "mean": 0.78,
    "std": 0.05,
    "min": 0.62,
    "max": 0.95,
    "p25": 0.74,
    "p50": 0.79,
    "p75": 0.85,
    "p90": 0.89
  }
}
```

---

## stage5_quality_scoring.py

### `run_quality_scoring()`

```python
def run_quality_scoring(
    stage4_seed_dir: str,
    w_artifact: float = 0.5,
    w_blur: float = 0.5,
    workers: int = 0,
) -> dict
```

`score_defect_images()`의 직접 래퍼.
CLI 진입점과 작업일지 셀 양쪽이 동일한 함수를 호출하도록 단일화.

### CLI

```bash
python stage5_quality_scoring.py \
    --stage4_seed_dir {path}/stage4_output/{seed_id} \
    [--w_artifact 0.5] [--w_blur 0.5] \
    [--workers 0]
```

---

## 작업일지 셀 패턴 (Step 11-2a/b/c)

Stage 3/4 셀과 동일한 구조:

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

---

## Step 11-3: 분포 검증

전 도메인 점수 분포 출력 + pruning threshold 참고값:
- P25 기준: 상위 75% 선택 → Stage 6 default threshold
- P50 기준: 상위 50% 선택
- P75 기준: 상위 25% 선택 (aggressive pruning)

---

## tests/test_stage5.py 커버리지

| 테스트 케이스 | 설명 |
|---|---|
| `test_score_image_basic` | 정상 이미지 → 0~1 범위 점수 반환 |
| `test_score_image_weight_validation` | w_artifact + w_blur ≠ 1.0 → ValueError |
| `test_score_artifacts_clean` | 균일한 이미지 → artifact_score 높음 |
| `test_score_sharpness_blurred` | 블러 이미지 → blur_score 낮음 |
| `test_score_defect_images_skip` | quality_scores.json 존재 시 재계산 없이 로드 |
| `test_score_defect_images_empty` | defect/ 비어있음 → 빈 stats 반환, 파일 미생성 |
| `test_score_defect_images_ordering` | 병렬 실행 결과가 sorted 순서로 정렬됨 |
