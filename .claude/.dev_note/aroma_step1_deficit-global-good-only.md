# Step 1 deficit_analysis global_dist — good images만 사용하도록 수정

> WARNING: Verification pending. 4-dataset Phase0->Step3->Exp2 re-run needed to confirm rare_pair_coverage improvement.

---

## (사용할 skills: micro-fix)

## 개요

`step7_deficit_analysis()`의 `global_dist`를 all images(good+defect)로 계산하면
P(cell|global) ≈ P(cell|defect_cluster)가 되어 deficit이 거의 전부 0이 된다.
AROMA 설계 의도는 "normal(good) 배경에서 자주 나타나는 context인데 defect cluster가
커버하지 못하는 부분 = deficit"이므로 global_dist는 good images만으로 제한해야 한다.

isp_LSM_1 기준: cluster 0의 25 compat cell 중 nonzero deficit = 1개 (수정 전).

이 수정은 `aroma_step1_deficit-cellkey-mismatch.md` 수정 이후 추가 발견된 2차 수정.

---

## 영향도 분석

### 이 기능이 변경하는 상태
- `deficit_analysis.json`의 deficit 값 — good vs defect context 차이 반영
- nonzero deficit 비율 대폭 증가 기대

### 그 상태를 전제로 동작하는 기존 로직
- `roi_selection.py` `_rare_pair_deficit_quantile()` — p75 임계 기반 희귀쌍 선택 정상화
- `exp2_roi_quality.py` `compute_metrics()` — rare_pair_coverage 실제 값 산출

---

## 수정 내용

### 1. `scripts/distribution_profiling.py` — `step7_deficit_analysis()` good 이미지 필터 추가

**수정 전**: all images 포함
```python
for r in context_rows:
    iid = r.get("image_id", "")
    for feat in CONTEXT_FEATURES:
        ...
        img_ctx[iid][feat].append(float(v))
```

**수정 후**: good images만 필터링
```python
for r in context_rows:
    if r.get("image_type") != "good":
        continue
    iid = r.get("image_id", "")
    for feat in CONTEXT_FEATURES:
        ...
        img_ctx[iid][feat].append(float(v))
```

`image_type` 필드 확인: isp_LSM_1 기준 `Counter({'good': 58848, 'defect': 1520})`

---

## 수정 대상 파일

- `scripts/distribution_profiling.py` — `step7_deficit_analysis()` img_ctx for 루프 진입부

---

## 테스트

CLAUDE.md: pytest 금지. Colab에서 직접 검증.

1. Phase 0 재실행 (distribution_profiling.py, 4개 데이터셋)
2. Step 3 재실행 (roi_selection.py)
3. roi_candidates.json nonzero deficit 비율 확인:
   ```python
   arr = np.array([c["deficit"] for c in cands])
   print(f"nonzero={np.sum(arr>0)}, p75={np.quantile(arr,0.75):.6f}")
   ```
   기대: p75 > 0, nonzero 비율 대폭 증가
4. Step 4 재실행 → Exp 2 재실행
5. rare_pair_coverage가 1.0 고정에서 실제 값으로 변경 확인
   기대 방향: AROMA rare_pair_cov > Random (deficit-aware 전략의 핵심 효과 검증)
