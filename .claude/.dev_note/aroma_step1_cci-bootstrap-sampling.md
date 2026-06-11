# CCI context sub-sampling — bootstrap (n < 20k) 추가

---

## (사용할 skills: micro-fix)

## 개요

현재 `compute_cci()`의 sub-sampling은 n > 20000일 때만 작동(비복원 추출).
n < 20000인 소규모 데이터셋은 전체 패치 수 그대로 사용 → 데이터셋 간 통계 볼륨 불균일.

복원 추출(bootstrap)로 항상 정확히 20,000개 사용 → 모든 데이터셋이 동일한 통계 볼륨 기준.

Good patches only 원칙은 `load_phase0_outputs()`에서 이미 처리됨 — 별도 수정 불필요.

---

## 수정 내용

### `scripts/aroma/compute_complexity.py` — `compute_cci()` sub-sampling 블록

```python
# 변경 전
if len(context_X_full) > max_patches:
    _sub_rng = np.random.default_rng(0)
    _sub_idx = _sub_rng.choice(len(context_X_full), max_patches, replace=False)
    context_X = context_X_full[_sub_idx]
else:
    context_X = context_X_full

# 변경 후
n_ctx = len(context_X_full)
if n_ctx == 0:
    context_X = context_X_full
else:
    _sub_rng = np.random.default_rng(0)
    replace = n_ctx < max_patches  # bootstrap when fewer than target
    _sub_idx = _sub_rng.choice(n_ctx, max_patches, replace=replace)
    context_X = context_X_full[_sub_idx]
```

---

## 수정 대상 파일

- `scripts/aroma/compute_complexity.py` (5줄 교체)

---

## 엣지 케이스

- n_ctx == 0: sub-sampling 건너뜀 (빈 배열 → CCI=0.0 그대로)
- n_ctx < max_patches: 복원 추출로 볼륨 보완 (isp처럼 소규모 데이터셋)
- n_ctx >= max_patches: 비복원 추출 (기존 동작 유지)

## 테스트

```python
# complexity_report.json 확인 항목
# cci_components.n_context_patches_used == 20000 (항상)
# n_context_patches_total: 실제 패치 수 (참고용)
```
