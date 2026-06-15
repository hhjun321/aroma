# Exp2 rare_pair_coverage 지표 항상 1.0 버그 수정 — deficit nonzero 기반 임계값 산정

---

## (사용할 skills: feature-dev)

## 개요

`exp2_roi_quality.py`의 `compute_metrics()`와 `roi_selection.py`의 `_rare_pair_deficit_quantile()`가
`p75(all candidates)`를 rare 임계값으로 사용한다. isp_LSM_1/mvtec_cable/visa_cashew에서
candidates의 75%+ deficit=0이라 p75=0 → 전부 rare로 분류 → rare_pair_coverage=1.0 고정
(AROMA=Random 구분 불가).

근본 원인은 step7 deficit_analysis의 구조적 특성: good images의 context cell(77개, 넓게 분포)과
defect images의 context cell(7개, 좁게 집중)이 달라, compat에 있는 셀들은 defect 집중으로
p(cell|cluster)≈g(cell|global) → deficit≈0. 반면 compat에 없는 셀은 deficit>0이지만 candidates에
없어 선택 대상이 아니다.

두 버그는 같은 원인을 공유하지만, (1) ROI 선택 알고리즘과 (2) 평가 지표 정의라는 별개 관심사를
변경하므로 feature-dev로 처리한다.

---

## 영향도 분석

### 이 기능이 변경하는 상태
- `roi_selection.py` deficit_aware 전략의 실제 ROI 선택 결과 (rare 후보 우선순위 변경)
- `exp2_roi_quality.py` rare_pair_coverage 지표 값 (정의 자체 변경)

### 그 상태를 전제로 동작하는 기존 로직
- `exp2_summary.md` 생성 (비교 테이블 + delta 요약) — 재실행으로 덮어씀
- `exp2_results.json` 과거 결과 — rare_pair_coverage 값 비교 불가, 재실행 필요

---

## 수정 내용

### 1. `scripts/aroma/roi_selection.py` — `_rare_pair_deficit_quantile()` thr 계산 변경

**수정 전** (버그): p75(all)=0이면 전부 rare → oversampling 효과 소멸
```python
deficits = np.array([c["deficit"] for c in candidates], dtype=np.float64)
thr = float(np.quantile(deficits, quantile))
rare = [c for c in candidates if c["deficit"] >= thr]
rest = [c for c in candidates if c["deficit"] < thr]
```

**수정 후**: nonzero deficit만 대상으로 quantile 계산
```python
nonzero_def = [c["deficit"] for c in candidates if c["deficit"] > 0]
if len(nonzero_def) < 2:
    thr = 0.0
else:
    thr = float(np.quantile(nonzero_def, quantile))

if thr == 0.0:
    # nonzero가 없거나 너무 적으면 roi_score 기준 top_k (oversampling 생략)
    return sorted(candidates, key=lambda c: c["roi_score"], reverse=True)[:top_k]

rare = [c for c in candidates if c["deficit"] >= thr]
rest = [c for c in candidates if c["deficit"] < thr]
```

### 2. `scripts/aroma/experiments/exp2_roi_quality.py` — `compute_metrics()` rare_pair 정의 변경

**수정 전** (버그): p75(all)=0이면 rare_pair_cov=1.0 fallback (AROMA=Random 구분 불가)
```python
p75 = float(np.quantile(deficits, 0.75))
if p75 == 0.0:
    rare_pair_cov = 1.0
else:
    rare_cands = {
        (c["cluster_id"], c["cell_key"])
        for c in candidates if c.get("deficit", 0.0) >= p75
    }
    rare_sel = {
        (c["cluster_id"], c["cell_key"])
        for c in selected if c.get("deficit", 0.0) >= p75
    }
    rare_sel = rare_sel & rare_cands
    rare_pair_cov = len(rare_sel) / len(rare_cands) if rare_cands else 0.0
```

**수정 후**: deficit > 0인 (cluster_id, cell_key) 쌍을 rare로 정의
```python
rare_cands = {
    (c["cluster_id"], c["cell_key"])
    for c in candidates if c.get("deficit", 0.0) > 0
}
if not rare_cands:
    rare_pair_cov = 1.0  # 구조적 rare pair 없음 — deficit 정보 부재
else:
    rare_sel = {
        (c["cluster_id"], c["cell_key"])
        for c in selected if c.get("deficit", 0.0) > 0
    }
    rare_sel = rare_sel & rare_cands
    rare_pair_cov = len(rare_sel) / len(rare_cands)
```

**주의**: 두 파일의 임계값 정의가 미묘하게 다름 — roi_selection은 nonzero의 p75 기준 oversampling,
exp2는 deficit>0 전체를 rare로 측정. 이는 의도된 차이 (선택 vs 측정의 별도 기준).

### 3. `scripts/aroma/experiments/exp2_roi_quality.py` — docstring 갱신

모듈 docstring 11행 `rare_pair_coverage` 설명을 `deficit>=p75` → `deficit>0` 기반으로 수정.

---

## 수정 대상 파일

- `scripts/aroma/roi_selection.py` — `_rare_pair_deficit_quantile()` thr 계산 블록 (255-284행 부근)
- `scripts/aroma/experiments/exp2_roi_quality.py` — `compute_metrics()` rare_pair 계산 블록 (119-139행 부근) + 모듈 docstring

---

## 엣지 케이스

- **nonzero 후보 0개**: rare_cands 빈집합 → rare_pair_cov=1.0 (구조적 rare pair 없음, 의도된 동작)
- **nonzero 후보 1개**: `len(nonzero_def) < 2` → thr=0.0 → roi_score top_k (단일값 quantile의 의도치 않은 oversampling 방지)
- **selected ⊄ candidates**: `rare_sel & rare_cands` 교집합 가드 유지 필수 (cov > 1.0 방지)
- **deficit 부동소수 잡음**: `deficit > 0` 비교가 미세 양수값을 rare로 포함 — 현재는 허용

---

## 테스트

CLAUDE.md: pytest 금지. Colab에서 직접 검증.

1. Step 3 재실행 (roi_selection.py, deficit_aware 전략)
   - roi_selected.json에서 nonzero deficit 후보 비율 확인 (기대: random 대비 높음)
2. Exp 2 재실행 (exp2_roi_quality.py)
   - rare_pair_coverage가 1.0 고정에서 벗어남 확인
   - AROMA > Random on rare_pair_coverage 확인
   - isp_LSM_1 기대 수치: AROMA≈1.0, Random≈0.12 (비율 83/1542≈5.4% 기준)
3. 다른 지표(morphology_coverage, context_coverage, entropy, gini) 회귀 없음 확인
