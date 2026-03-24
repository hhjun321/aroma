# Stage 6 Dataset Builder — Design Spec

**Date:** 2026-03-24
**Status:** Draft

---

## Overview

Stage 5 품질 점수를 활용하여 anomaly detection 벤치마크용 증강 데이터셋을 구성한다.
3개 dataset group (baseline / aroma_full / aroma_pruned) 을 MVTec-style 폴더 구조로 생성.
`dataset_config.json` 기준 ISP-AD 3개 / MVTec AD 15개 / VisA 12개 = 총 30개 cat_dir 처리.

---

## Design Decisions

| 항목 | 결정 | 이유 |
|------|------|------|
| 파일 처리 방식 | 복사 (shutil.copy2) | Colab/Google Drive 환경에서 symlink 불안정 — CASDA 동일 패턴 |
| 구현 구조 | `utils/dataset_builder.py` + `stage6_dataset_builder.py` | Stage 5 패턴 일치; 노트북 직접 import 가능 |
| Dataset groups | baseline / aroma_full / aroma_pruned | CASDA baseline_raw / casda_composed / casda_composed_pruning 대응 |
| test/ 구성 | baseline group 에만 포함 | Stage 7 에서 모든 group 이 동일 test set 공유 |
| Skip sentinel | `build_report.json` 존재 여부 | 디렉터리만 체크하면 부분 완료 시 재생성 방지 실패 |
| quality_scores.json 없는 seed | 해당 seed 스킵 + 경고 로그 | Stage 5 미실행 seed 에서 크래시 방지 |
| 결함 유형 이름 | `dataset_config.json`의 `seed_dir` 경로에서 추출 | seed_id 와 원본 결함 폴더명이 다를 수 있음 |

---

## File Structure

```
utils/dataset_builder.py        — 핵심 로직 (라이브러리)
stage6_dataset_builder.py       — CLI 오케스트레이터
tests/test_stage6.py            — 단위/통합 테스트
```

---

## Output Directory Structure

```
{cat_dir}/augmented_dataset/
├── build_report.json            ← 완료 sentinel + 통계
├── baseline/
│   ├── train/
│   │   └── good/               ← image_dir 원본 복사
│   └── test/
│       ├── good/               ← 원본 test/good 복사
│       └── {defect_type}/      ← 원본 test 결함 복사
├── aroma_full/
│   └── train/
│       ├── good/               ← image_dir 원본 복사
│       └── defect/             ← stage4_output 전체 defect 복사
└── aroma_pruned/
    └── train/
        ├── good/               ← image_dir 원본 복사
        └── defect/             ← quality_score >= threshold 인 것만 복사
```

`aroma_full` / `aroma_pruned` 에는 test/ 없음.
Stage 7 에서 모든 group 의 테스트는 `baseline/test/` 를 공통으로 사용.

---

## utils/dataset_builder.py

### 공개 API

#### `build_dataset_groups()`

```python
def build_dataset_groups(
    cat_dir: str,
    image_dir: str,
    seed_dirs: list[str],
    pruning_threshold: float = 0.6,
    workers: int = 0,
) -> dict
```

- `cat_dir`: 카테고리 루트 (e.g. `.../mvtec/bottle`)
- `image_dir`: 원본 train/good 이미지 디렉터리
- `seed_dirs`: 해당 cat_dir 의 모든 `seed_dir` 경로 목록 (dataset_config.json 에서 추출).
  `baseline/test/{defect_type}/` 구성 시 각 경로의 마지막 컴포넌트를 결함 유형명으로 사용.
- `pruning_threshold`: aroma_pruned group 의 quality_score 최솟값
- **Skip**: `build_report.json` 이 이미 존재하고 저장된 `pruning_threshold` 가 요청값과 일치하면
  재생성 없이 로드하여 반환. 불일치 시 전체 재생성.
- **반환 및 저장** (`build_report.json`):
  ```json
  {
    "pruning_threshold": 0.6,
    "baseline":     {"good_count": 209, "defect_count": 0},
    "aroma_full":   {"good_count": 209, "defect_count": 347},
    "aroma_pruned": {"good_count": 209, "defect_count": 261}
  }
  ```

#### `_collect_defect_paths()`

```python
def _collect_defect_paths(
    cat_dir: str,
    pruning_threshold: float | None = None,
) -> list[str]
```

- `cat_dir/stage4_output/{seed_id}/defect/*.png` 수집 (단일 깊이 `*`)
- `pruning_threshold` 지정 시:
  - `quality_scores.json` 없는 seed → 해당 seed 스킵 + `warnings.warn()` 로그
  - threshold 미만 이미지 제외
- 반환: 복사 대상 절대 경로 리스트

#### `_copy_worker()`

```python
def _copy_worker(args_tuple: tuple) -> str | None
# args_tuple: (src_path_str, dst_path_str)
# dst 이미 존재하면 스킵 (파일 단위 resume)
# tasks 구성: tasks = [(str(src), str(dst)) for src, dst in copy_pairs]
```

ProcessPoolExecutor pickle-safe 워커.

### test/ 결함 유형 추출 규칙

`baseline/test/{defect_type}/` 구성 시 결함 유형명은 `dataset_config.json` 항목의
`seed_dir` 경로 마지막 컴포넌트에서 추출.

예: `seed_dir = ".../bottle/test/broken_large"` → `defect_type = "broken_large"`

동일 `cat_dir` 를 가리키는 모든 config 항목의 `seed_dir` 를 순회하여
`test/{defect_type}/` 디렉터리 전체를 `baseline/test/` 아래 복사.

### `image_dir` 가정

동일 `cat_dir` 를 참조하는 모든 config 항목은 동일한 `image_dir` 를 가진다.
노트북 셀에서 첫 번째 항목의 `image_dir` 를 사용. 이 가정이 깨지면 ValueError raise.

---

## stage6_dataset_builder.py

### `run_dataset_builder()`

```python
def run_dataset_builder(
    cat_dir: str,
    image_dir: str,
    seed_dirs: list[str],
    pruning_threshold: float = 0.6,
    workers: int = 0,
) -> dict
```

`build_dataset_groups()` 의 직접 래퍼.

### CLI

```bash
python stage6_dataset_builder.py \
    --cat_dir {path}/{category} \
    --image_dir {path}/train/good \
    --seed_dirs {path}/test/broken_large {path}/test/broken_small \
    [--pruning_threshold 0.6] \
    [--workers 0]
```

---

## 작업일지 셀 패턴 (Step 12-2a/b/c)

```python
import json, sys
from pathlib import Path
from tqdm.auto import tqdm

sys.path.insert(0, "/content/aroma")
from stage6_dataset_builder import run_dataset_builder

REPO   = Path("/content/drive/MyDrive/project/aroma")
CONFIG = json.loads((REPO / "dataset_config.json").read_text())

DOMAIN_FILTER = "visa"   # "isp" / "mvtec" / "visa"
LABEL = {"isp": "ISP-AD", "mvtec": "MVTec AD", "visa": "VisA"}[DOMAIN_FILTER]

# cat_dir 단위 중복 제거 + image_dir 일관성 검사
from collections import defaultdict
cat_map: dict[str, str] = {}            # cat_dir → image_dir
cat_seed_dirs: dict[str, list] = defaultdict(list)  # cat_dir → [seed_dir, ...]

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

---

## tests/test_stage6.py 커버리지

| 테스트 케이스 | 설명 |
|---|---|
| `test_baseline_only_good` | baseline group — defect 없음, good 만 복사 |
| `test_aroma_full_all_defects` | aroma_full — quality 무관 전체 defect 포함 |
| `test_aroma_pruned_threshold` | quality_score < threshold 이미지 제외 확인 |
| `test_skip_existing` | `build_report.json` 존재 + threshold 일치 시 재생성 없이 스킵 |
| `test_skip_threshold_mismatch_reruns` | `build_report.json` 존재해도 threshold 불일치 시 전체 재생성 |
| `test_build_report_saved` | build_report.json 저장 및 count 정확성 확인 |
| `test_missing_quality_scores_skips_seed` | quality_scores.json 없는 seed 스킵 + 경고 |
| `test_empty_stage4_output` | stage4_output 없는 경우 defect_count=0 정상 처리 |
