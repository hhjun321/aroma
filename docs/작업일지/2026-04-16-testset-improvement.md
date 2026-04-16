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
