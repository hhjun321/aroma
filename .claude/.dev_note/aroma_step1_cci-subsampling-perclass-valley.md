# CCI sub-sampling + valley_count per-class mean 구현

> WARNING: _perclass_valley_count() implemented here was removed in aroma_step1_mci-redesign-class-diversity.md. Only the CCI sub-sampling part remains valid.

---

## (사용할 skills: micro-fix)

## 개요

두 가지 독립적인 MCI/CCI 공정성 개선 작업.

1. **CCI sub-sampling**: `compute_cci()`의 `_col_mean`, `_col_var`가 현재 전체 `context_X` 사용. 고해상도 데이터셋이 저해상도보다 패치 수 더 많아 CCI가 과대평가될 수 있음. `_cluster_context()`는 이미 20k sub-sampling 적용 중이므로 나머지 CCI 통계도 동일한 기준으로 맞춤.

2. **valley_count per-class mean**: 현재 `_total_valley_count()`는 모든 인스턴스 pooled. 클래스 수가 많으면 (cable 8종) valley_count가 자동 증가. 클래스별 계산 후 mean → "결함 평균 형태 복잡도" 측정으로 클래스 수 편향 제거.

---

## 영향도 분석

### 변경하는 상태
- `complexity_report.json`의 CCI raw/normalized 수치 변경
- `complexity_report.json`의 MCI raw.valley_count 수치 변경

### 그 상태를 전제로 동작하는 기존 로직
- Meta Policy Generator(`run_meta_policy_generator`)가 MCI를 pruning 기준으로 사용 → valley_count 변경 시 MCI 변동으로 policy 선택 결과 달라질 수 있음
- 3개 데이터셋 이전 수치(isp=0.604, cable=0.646, cashew=0.607) 모두 재계산 필요

---

## 수정 내용

### 1. `scripts/aroma/compute_complexity.py` — CCI sub-sampling

`compute_cci()` 함수 내 `context_X` 사용 전 sub-sampling 적용:

```python
# 변경 전 (line ~562)
context_X: np.ndarray = phase0.get("context_X", np.empty((0, 5)))
n_ctx_clusters, _, _ = _cluster_context(context_X, cfg)

# 변경 후
context_X_full: np.ndarray = phase0.get("context_X", np.empty((0, 5)))
max_patches = int(cfg["cci"]["context_gmm"].get("max_patches", 20000))
if len(context_X_full) > max_patches:
    rng = np.random.default_rng(0)
    idx = rng.choice(len(context_X_full), max_patches, replace=False)
    context_X = context_X_full[idx]
else:
    context_X = context_X_full
n_ctx_clusters, _, _ = _cluster_context(context_X, cfg)
```

`_col_mean`, `_col_var` 내부 클로저가 `context_X`를 캡처하므로 별도 수정 불필요.

보고: `complexity_report.json`에 `n_context_patches_used` 필드 추가:
```python
return cci, {
    ...
    "n_context_patches_used": len(context_X),
    "n_context_patches_total": len(context_X_full),
}
```

### 2. `scripts/aroma/compute_complexity.py` — valley_count per-class mean

#### 2.1 `load_phase0_outputs()` — defect_type 반환 추가

```python
# morphology_features.csv 로딩 후
morph_defect_types = [r.get("defect_type", "unknown") for r in morph_rows]
# mask 적용 후 반환
morph_defect_types = [dt for dt, m in zip(morph_defect_types, mask) if m]

return {
    ...
    "morph_defect_types": morph_defect_types,  # 추가
}
```

#### 2.2 `_perclass_valley_count()` 함수 추가

```python
def _perclass_valley_count(
    morph_rows: List[Dict[str, str]],
    aggregation: str = "mean",
) -> float:
    """
    Per-class valley count: group morphology features by defect_type,
    compute total valley count per class, then mean/median across classes.
    
    Removes class-count bias: cable(8 types) vs isp(2 types) compared fairly.
    """
    from collections import defaultdict
    try:
        from scipy.signal import find_peaks
        HAS_SCIPY = True
    except ImportError:
        HAS_SCIPY = False

    if not HAS_SCIPY or not morph_rows:
        return 0.0

    # Group by defect_type
    class_rows: Dict[str, List] = defaultdict(list)
    for r in morph_rows:
        dt = r.get("defect_type", "unknown")
        class_rows[dt].append(r)

    class_counts = []
    for dt, rows in class_rows.items():
        total_valleys = 0
        for feat in MORPH_FEATURES:
            vals = []
            for r in rows:
                try:
                    vals.append(float(r[feat]))
                except (KeyError, ValueError):
                    continue
            if len(vals) < 3:
                continue
            # Simple histogram valley detection (mirror distribution_profiling logic)
            arr = np.array(vals)
            bins = min(50, len(arr) - 1)
            counts, _ = np.histogram(arr, bins=bins)
            # invert for peak detection
            inverted = counts.max() - counts
            noise_floor = np.sqrt(len(arr) / bins)
            prominence = max(noise_floor * 2, counts.max() * 0.1)
            peaks, _ = find_peaks(inverted, prominence=prominence)
            total_valleys += len(peaks)
        class_counts.append(total_valleys)

    if not class_counts:
        return 0.0

    if aggregation == "median":
        return float(np.median(class_counts))
    return float(np.mean(class_counts))
```

#### 2.3 `compute_mci()` — per-class valley count 사용

```python
# 변경 전
raw = {
    "valley_count": float(_total_valley_count(dist_analysis)),
    ...
}

# 변경 후
morph_rows_raw: List[Dict] = phase0.get("morph_rows_raw", [])
if morph_rows_raw:
    vc = _perclass_valley_count(morph_rows_raw, aggregation="mean")
else:
    # fallback: pooled (backward compat)
    vc = float(_total_valley_count(dist_analysis))

raw = {
    "valley_count": vc,
    ...
}
```

#### 2.4 `load_phase0_outputs()` — raw rows 반환 추가

```python
return {
    ...
    "morph_rows_raw": morph_rows,  # 추가 (defect_type 포함 원본 rows)
}
```

---

## 수정 대상 파일

- `scripts/aroma/compute_complexity.py` (단일 파일)

---

## 엣지 케이스

- defect_type이 하나뿐인 데이터셋(ISP LSM_1 = 2종): per-class mean = pooled과 유사하게 동작 (단 샘플 분할로 약간 다를 수 있음)
- scipy 없을 경우: fallback으로 pooled valley count 사용 (`_total_valley_count`)
- 클래스당 인스턴스 수 < 3: valley 계산 skip, 0으로 처리
- context_features.csv가 20k 미만인 경우: sub-sampling 없이 전체 사용

---

## 기대 효과

- cable(8종): valley_count pooled=16 → per-class mean 감소 예상 (각 클래스 개별 분포는 단순함)
- isp(2종): valley_count pooled=13 → 소폭 변화
- cashew(1종 = whole): pooled ≈ per-class (단일 클래스)

MCI 순서 cable > cashew > isp 유지 여부 재검증 필요.

---

## 테스트

Colab에서 실행 후 `complexity_report.json` 확인:
```python
!python $AROMA_SCRIPTS/compute_complexity.py \
    --profiling_dir $AROMA_OUT/profiling/isp_LSM_1 \
    --output_dir    $AROMA_OUT/complexity/isp_LSM_1_v2

!python $AROMA_SCRIPTS/compute_complexity.py \
    --profiling_dir $AROMA_OUT/profiling/mvtec_cable \
    --output_dir    $AROMA_OUT/complexity/mvtec_cable_v2

!python $AROMA_SCRIPTS/compute_complexity.py \
    --profiling_dir $AROMA_OUT/profiling/visa_cashew \
    --output_dir    $AROMA_OUT/complexity/visa_cashew_v2
```

확인 항목:
- `mci_components.raw.valley_count` — per-class mean 수치
- `cci_components.n_context_patches_used` vs `n_context_patches_total`
- MCI 순서: cable > cashew > isp 유지 여부
