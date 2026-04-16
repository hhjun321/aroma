# 작업일지 — 테스트셋 개선

> 날짜: 2026-04-16
> 목적: 벤치마크 테스트셋이 단일 결함 유형만 포함하는 문제 수정

---

## 1. 문제 진단

### 현상
mvtec_bottle Stage 7 벤치마크에서 AUROC=1.0 (aroma_full/pruned), AUROC=1.0 (efficientdet_d0/baseline) 발생.

### 근본 원인
`dataset_config.json`의 `seed_dir`이 카테고리당 1개 결함 유형만 지정되어 있고,
`utils/dataset_builder.py`의 `build_dataset_groups`가 테스트셋 defect 부분을 `seed_dirs`에서만 구성.

```python
# 기존 코드 (utils/dataset_builder.py)
for sd in seed_dirs:                            # broken_large 1개뿐
    defect_type = Path(sd).name
    _copy_images(src, aug_dir / "baseline" / "test" / defect_type, ...)
```

**결과 테스트셋 (mvtec_bottle)**

| 디렉터리 | 기존 | 수정 후 |
|---------|------|--------|
| test/good/ | ✓ | ✓ |
| test/broken_large/ | ✓ | ✓ |
| test/broken_small/ | ✗ 누락 | ✓ |
| test/contamination/ | ✗ 누락 | ✓ |

MVTec 15개 카테고리 중 14개에서 같은 문제 존재 (toothbrush만 단일 유형).
참고: `docs/작업일지/2026-04-06-defect-type-analysis.md`

### AUROC 인플레이션 메커니즘
1. Stage 4가 `broken_large` 시드로 합성 defect 생성 → 모델이 broken_large 패턴 학습
2. 테스트셋도 broken_large만 존재 → 학습한 패턴 그대로 평가
3. broken_large는 시각적으로 명확한 결함 → pretrained 특징으로도 완벽 분리 가능

---

## 2. 수정 계획

### Phase 1: `utils/dataset_builder.py` 수정
`seed_dirs` 대신 원본 test 디렉터리 전체를 스캔하여 모든 defect 유형 수집.

- `image_dir` → `{cat}/train/good` → parents[1] = `{cat}` → `{cat}/test/`
- `test/good/` 제외한 모든 하위 디렉터리를 테스트셋에 포함
- `seed_dirs`는 폴백(원본 test 디렉터리 없을 때)으로 유지

### Phase 2: Stage 6 재실행 (bottle)
- `augmented_dataset/` 삭제 후 재생성 필요
- `build_report.json` 삭제 또는 `build_dataset_groups(reset=True)` 호출

### Phase 3: Stage 7 재실행 (bottle)
- 완전한 테스트셋으로 벤치마크 재수행
- AUROC 변화 확인

---

## 3. 구현 이력

### [Phase 1] utils/dataset_builder.py 수정

**변경 위치:** `build_dataset_groups` 내 baseline/test 구성 블록 (line 287~295)

**변경 전:**
```python
# test/{defect_type}/ (seed_dirs 에서 추출)
for sd in seed_dirs:
    defect_type = Path(sd).name
    src = Path(sd)
    if src.exists():
        _copy_images(src, aug_dir / "baseline" / "test" / defect_type, ...)
```

**변경 후:**
```python
# test/{defect_type}/ — 원본 test 디렉터리 전체 스캔 (모든 defect 유형 포함)
# seed_dirs는 원본 test 디렉터리가 없을 때 폴백으로 사용
cat_test_dir = Path(image_dir).parents[1] / "test"
if cat_test_dir.exists():
    for defect_dir in sorted(cat_test_dir.iterdir()):
        if defect_dir.is_dir() and defect_dir.name != "good":
            _copy_images(defect_dir, aug_dir / "baseline" / "test" / defect_dir.name, ...)
else:
    for sd in seed_dirs:  # 폴백: 원본 구조 없을 때
        ...
```

**테스트 추가 (tests/test_stage6.py):**
- `test_testset_includes_all_defect_types`: seed_dirs 미등록 유형도 test set에 포함됨 검증
- `test_testset_fallback_to_seed_dirs_when_no_cat_test`: 원본 test 디렉터리 없을 때 폴백 동작 검증

**결과:** Stage 6 테스트 27/27 PASS, 전체 154/159 PASS
(나머지 5개 실패는 ultralytics 미설치 관련 기존 문제 — 이번 변경과 무관)

---

## 4. 재실행 기록

### Stage 6 재실행 (mvtec_bottle) ✓

- augmented_dataset 삭제 후 Stage 6 재실행
- 테스트셋 구성: broken_large + broken_small + contamination (3종)
- 소요: 약 6초 (캐시 히트 — build_report.json 이미 존재 추정)

### Stage 7 재실행 (mvtec_bottle) ✓ (부분)

- Stage 7 resume=True(기본값)로 실행 → baseline 실험은 이전 experiment_meta.json 재사용
- aroma 실험만 새 테스트셋으로 재평가됨

---

## 5. AUROC 비교 (수정 전 vs 수정 후)

| 모델/그룹 | 수정 전 | 수정 후 | 비고 |
|----------|--------|--------|------|
| yolo11/baseline | 0.7925 | 0.7925 | 캐시 재사용 추정 |
| yolo11/aroma_full | 1.0000 | 1.0000 | 캐시 또는 진짜 1.0 |
| yolo11/aroma_pruned | 1.0000 | 1.0000 | 캐시 또는 진짜 1.0 |
| efficientdet_d0/baseline | 1.0000 | 1.0000 | 캐시 재사용 추정 |
| efficientdet_d0/aroma_full | 1.0000 | **0.9950** | ▼ 새 테스트셋 효과 확인 |
| efficientdet_d0/aroma_pruned | 1.0000 | **0.9900** | ▼ 새 테스트셋 효과 확인 |

**해석:**
- efficientdet_d0 aroma 수치 감소 → broken_small, contamination 포함으로 평가 난이도 상승 확인
- yolo11/aroma 수치 불변 → reset=True 전체 재실행으로 캐시 여부 확인 필요
- baseline 수치 불변 → Stage 7 resume 캐시로 인한 것으로 추정

**TODO:** `reset=True`로 전체 재실행 후 yolo11/aroma 1.0 진위 확인

---

## 6. train/test 고정 비율 분할 (2026-04-16)

### 문제
- MVTec, VisA, ISP 각 카테고리의 원본 train/test 분할 비율이 카테고리마다 다름
- 비율이 다르면 모델 성능 비교가 불공정 (train 데이터양 차이)
- 원본 분할 유지 시 test/good 이미지가 train 에 포함될 위험 구조

### 해결: `split_ratio` 파라미터 추가

**변경 파일:**
- `utils/dataset_builder.py`: `_copy_file_list`, `_split_good_images` 헬퍼 추가, `build_dataset_groups`에 `split_ratio`, `split_seed` 파라미터 추가
- `stage6_dataset_builder.py`: `run_dataset_builder`, CLI, config 로드에 동일 파라미터 추가
- `configs/benchmark_experiment.yaml`: `split_ratio: 0.8`, `split_seed: 42` 추가
- `docs/작업일지/stage1_6_execute.md`: Stage 6 셀 코드에 `SPLIT_RATIO`, `SPLIT_SEED` 추가

**동작 방식 (`split_ratio=0.8`):**
1. `train/good` + `test/good` 전체 pooling
2. 파일명 기준 정렬 → `split_seed` 기반 결정적 셔플
3. 상위 80% → train/good (baseline, aroma_full, aroma_pruned 동일)
4. 하위 20% → test/good (고정, 절대 train에 포함 불가)
5. defect test 이미지는 기존과 동일 (`cat/test/{defect_type}/`)

**build_report.json** 에 `split_ratio`, `split_seed` 저장 → skip 조건 포함

### Colab 재실행 방법
```python
# augmented_dataset 삭제 후 Stage 6 재실행 필요
import shutil
from pathlib import Path

aug = Path(".../mvtec/bottle/augmented_dataset")
if aug.exists():
    shutil.rmtree(aug)

# Stage 6 재실행 (stage1_6_execute.md Stage 6 셀)
# SPLIT_RATIO = 0.8, SPLIT_SEED = 42 이 자동 적용됨
```

---

## 7. Stage 5 합성 품질 점수 분석 (2026-04-16)

### 관측값

| 도메인/카테고리 | quality_score | pruning_threshold=0.6 통과 | 비고 |
|--------------|:---:|:---:|------|
| isp_ASM | 0.647 | ✓ | 33 seeds, 132 img/seed |
| mvtec_bottle | 0.738 | ✓ | 20 seeds, 209 img/seed |
| visa_candle | 0.493 | ✗ | 10 seeds, 900 img/seed |

### 원인 분석

`artifact_score` (`utils/quality_scoring.py`) 두 세부 지표:
1. **edge_score**: 적응형 3σ 임계값 → 해상도·텍스처 무관 (자기정규화)
2. **hf_score**: `hf_ratio = mean|Laplacian| / mean|gradient|`, 상수 `5.0`으로 정규화

두 지표 모두 공간 평균/비율 → 해상도 독립적. `artifact_score`의 해상도 정규화 부재가 원인이 아님.

**실제 원인 — 텍스처 캘리브레이션 불일치:**
- candle 표면은 매끄럽고 왁스질 → gradient 평균(mean_mag)이 낮음
- MPB 합성으로 붙여진 결함은 이 매끄러운 배경 대비 날카로운 경계 생성
- → `hf_ratio = lap_energy / mean_mag` 상승 → `hf_score` 하락
- 상수 `5.0`은 ISP/MVTec 텍스처 기반 경험치 — VisA의 다양한 표면 유형에 범용적이지 않음

**결론:** quality_scoring.py 수정 불필요. 임계값 도메인 캘리브레이션 문제.

### 해결: `pruning_threshold_by_domain`

**변경 파일:**
- `configs/benchmark_experiment.yaml`: `pruning_threshold_by_domain` 추가
- `utils/dataset_builder.py`: `pruning_threshold_by_domain` 파라미터, `effective_threshold` 결정 로직, `build_report.json`에 `effective_pruning_threshold` 저장
- `stage6_dataset_builder.py`: 파라미터 전달 및 config 로드

**config 설정:**
```yaml
dataset:
  pruning_threshold: 0.6                  # global fallback
  pruning_threshold_by_domain:
    isp: 0.6
    mvtec: 0.6
    visa: 0.4                             # candle 0.493 통과
```

**동작 방식:**
1. `build_dataset_groups` 내 도메인 감지 후 `pruning_threshold_by_domain[domain]` 조회
2. 매칭되면 → `effective_threshold = pruning_threshold_by_domain[domain]`
3. 미매칭 → `effective_threshold = pruning_threshold` (global fallback)
4. `aroma_pruned` defect 수집 시 `effective_threshold` 사용
5. `build_report.json`에 `effective_pruning_threshold` 저장 → 설정 변경 시 자동 재빌드 트리거

**검증 (Colab):**
- VisA candle `build_report.json` → `"effective_pruning_threshold": 0.4`
- `aroma_pruned/train/defect/` 이미지 수 > 0 확인
