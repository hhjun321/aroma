# AROMA Exp 4 — anomalib API 호환 수정 + resume 기능 추가

## (사용할 skills: micro-fix)

## 개요

설치된 anomalib 최신 버전과의 API 불일치로 exp4_downstream_ad.py 실행 불가 상태였음.
3가지 API 변경(class name, Folder 파라미터 2개 제거)으로 실행 가능하게 수정하고,
장시간 실행 중 crash 시 처음부터 재실행해야 하는 문제를 해결하기 위해 `--resume` 플래그와 incremental save를 추가함.

---

## 영향도 분석

### 이 기능이 변경하는 상태
- `exp4_results.json`: 조건 완료 시마다 incremental 저장 (이전: 전체 완료 후 1회 저장)
- CLI 인터페이스: `--resume` 플래그 추가

### 그 상태를 전제로 동작하는 기존 로직
- 없음 (평가 전용 스크립트)

---

## 수정 내용

### 1. `scripts/aroma/experiments/exp4_downstream_ad.py` — Simplenet → Supersimplenet

```python
# 변경 전
from anomalib.models import Patchcore, Simplenet, EfficientAd, ReverseDistillation

# 변경 후
from anomalib.models import Patchcore, Supersimplenet, EfficientAd, ReverseDistillation
```

anomalib 최신 버전에서 SimpleNet이 SuperSimpleNet으로 업그레이드됨. Patchcore, EfficientAd, ReverseDistillation은 이름 그대로.

### 2. `scripts/aroma/experiments/exp4_downstream_ad.py` — Folder `task` 파라미터 제거

anomalib 신버전 `Folder.__init__` 시그니처에서 `task` 파라미터 제거됨. task는 mask_dir 존재 여부로 자동 추론.

```python
# 제거된 것들
from anomalib import TaskType
task = TaskType.SEGMENTATION if mask_dir else TaskType.CLASSIFICATION
folder_kwargs["task"] = task
```

### 3. `scripts/aroma/experiments/exp4_downstream_ad.py` — Folder `image_size` 파라미터 제거

`Folder.__init__` 시그니처에 `image_size` 없음. 이미지 크기는 augmentation 또는 모델 기본값으로 처리.

```python
# 제거
image_size=(image_size, image_size),
```

### 4. `scripts/aroma/experiments/exp4_downstream_ad.py` — `--resume` + incremental save

**`_run_ad_mode()` 시그니처 추가:**
- `existing_results: Optional[Dict[str, Any]] = None`
- `output_path: Optional[str] = None`

**condition 루프 앞 skip 체크:**
```python
if cond in existing.get(ds, {}).get(model_name, {}):
    model_results[cond] = existing[ds][model_name][cond]
    logger.info("RESUME skip %s / %s / %s ...", ds, model_name, cond)
    continue
```

**조건 완료 시 incremental save:**
```python
if output_path:
    results[ds] = dict(ds_results)
    results[ds][model_name] = dict(model_results)
    save_json(results, output_path)
```

**`run()` 함수에 `resume: bool = False` 추가, 기존 results 로드:**
```python
if resume and Path(output_path).exists():
    existing_results = load_json(output_path)
```

**CLI `--resume` 플래그:**
```python
p.add_argument("--resume", action="store_true",
               help="기존 exp4_results.json에서 완료된 run을 skip하고 재개")
```

---

## 수정 대상 파일

- `scripts/aroma/experiments/exp4_downstream_ad.py`

---

## 환경 요구사항 (코드 외)

numpy 2.x + anomalib 조합에서 `ImportError: cannot import name '_center' from numpy._core.umath` 발생.

```python
!pip install "numpy<2.0" -q
# 런타임 재시작 필수
```

---

## resume 동작 범위

- **skip 단위**: `(dataset, model, condition)` 트리플 — 조건 완료 후 저장된 것만 skip
- **재실행 불가 범위**: 한 condition 내부 crash → 해당 condition 처음부터 재실행
- **skip 판단**: `existing_results[ds][model_name][cond]` 키 존재 여부 (에러 결과도 skip)
- **재시도 방법**: exp4_results.json에서 해당 키 삭제 후 `--resume` 실행

---

## 테스트

CLAUDE.md: pytest 금지. Colab 직접 검증.

```python
# smoke test: patchcore single dataset
!python $AROMA_SCRIPTS/experiments/exp4_downstream_ad.py \
    --model patchcore \
    --condition baseline \
    --dataset_keys isp_LSM_1 \
    --random_synthetic_dir $RANDOM_SYNTH_DIR \
    --aroma_synthetic_dir  $AROMA_SYNTH_DIR \
    --real_data_dir        $AROMA_DATA \
    --output_dir           $EXP4_OUT \
    --seed 42

# resume 검증: 위 실행 후 --resume 추가하면 "RESUME skip" 로그 확인
!python ... --resume
```
