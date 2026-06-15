# Step 1 deficit_analysis cell_key 인코딩 불일치 버그 수정

---

## (사용할 skills: micro-fix)

## 개요

`distribution_profiling.py`의 `step6_compatibility_learning()`과 `step7_deficit_analysis()`가
서로 다른 granularity로 cell_key를 생성하여 key 공간이 불일치한다.
step6는 이미지 단위 **평균 context feature**로 cell_key를 생성하지만,
step7는 **패치 단위 raw row**로 cell_key를 생성해 두 파일의 cell_key 집합이 다르다.
결과적으로 `roi_selection.py`의 `build_candidates()`에서 deficit lookup이 항상 0.0을 반환한다.

isp_LSM_1 실측: candidates 1282개 중 nonzero deficit 19개 (우연히 일치하는 key만).

---

## 영향도 분석

### 이 기능이 변경하는 상태
- `deficit_analysis.json`의 cell_key 공간이 `compatibility_matrix.json`과 일치하게 됨
- `deficit_analysis.json` 수치 변경 — Phase 0 재실행 필요

### 그 상태를 전제로 동작하는 기존 로직
- `roi_selection.py` `build_candidates()` — deficit_rows lookup 정상화
- `roi_selection.py` `_rare_pair_deficit_quantile()` — p75 임계 계산 정상화
- `exp2_roi_quality.py` `compute_metrics()` — rare_pair_coverage 정상 동작

---

## 수정 내용

### 1. `scripts/distribution_profiling.py` — `step7_deficit_analysis()` global_counts 계산 방식 변경

**현재 (버그)**: 패치 row 직접 사용
```python
global_counts: Dict[str, int] = defaultdict(int)
for r in context_rows:
    try:
        cell = _context_cell_key(r, bin_edges)   # 패치 단위 raw row
        global_counts[cell] += 1
    except Exception:
        pass
```

**수정**: 이미지 단위 mean context 사용 (step6와 동일 로직)
```python
# 이미지별 mean context 계산 (step6 defect_mean_ctx 패턴 동일 적용)
img_ctx: Dict[str, Dict[str, List[float]]] = defaultdict(
    lambda: {f: [] for f in CONTEXT_FEATURES}
)
for r in context_rows:
    iid = r.get("image_id", "")
    for feat in CONTEXT_FEATURES:
        v = r.get(feat)
        if v not in ("", None):
            try:
                img_ctx[iid][feat].append(float(v))
            except ValueError:
                pass

img_mean_ctx: Dict[str, Dict[str, float]] = {
    iid: {f: float(np.mean(feats[f])) for f in CONTEXT_FEATURES}
    for iid, feats in img_ctx.items()
    if all(img_ctx[iid][f] for f in CONTEXT_FEATURES)
}

global_counts: Dict[str, int] = defaultdict(int)
for iid, ctx in img_mean_ctx.items():
    try:
        cell = _context_cell_key(ctx, bin_edges)
        global_counts[cell] += 1
    except Exception:
        pass
```

---

## 수정 대상 파일

- `scripts/distribution_profiling.py` — `step7_deficit_analysis()` 내 `global_counts` 계산 블록

---

## 테스트

CLAUDE.md: pytest 금지. Colab에서 직접 검증.

1. `distribution_profiling.py` Phase 0 재실행 (4개 데이터셋)
2. 진단 코드 재실행:
   ```python
   compat_keys = set(cm.get("matrix", {}).get("4", {}).keys())
   deficit_keys = set(da.get("4", {}).get("deficit", {}).keys())
   print("intersection:", len(compat_keys & deficit_keys))
   ```
   → 교집합 > 0 확인 (이상적으로 compat_keys ⊆ deficit_keys)
3. `roi_selection.py` Step 3 재실행 → `roi_candidates.json` nonzero deficit 비율 확인
   - 기대: 대다수 nonzero (isp_LSM_1 기준 581/635 수준)
4. Exp 2 재실행 → `rare_pair_coverage` 값이 1.0 고정에서 실제 값으로 변경 확인
