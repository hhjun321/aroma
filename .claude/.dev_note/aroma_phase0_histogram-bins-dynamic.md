# valley 과감지 버그 수정 — 히스토그램 bins 동적 계산 적용

## (사용할 skills: micro-fix)

## 개요

`distribution_profiling.py`의 `_detect_valleys()`에서 `HISTOGRAM_BINS=50` 고정값을 사용해 92-100개 샘플에 대해 50개 bins를 생성하면 평균 1.84 샘플/bin으로 빈 bin이 다수 발생한다. 이 희소 bin들이 inverted histogram에서 잡음 peak로 감지되어 가짜 valley가 대량 생성된다.

실제 데이터 확인:
- mvtec_cable circularity: 자연 경계 ~1개 → 12개 감지
- visa_cashew eccentricity: 자연 경계 ~3개 → 11개 감지
- isp_LSM_1 extent: 자연 경계 ~3개 → 7개 감지

Sturges' rule 기반 동적 bins(`bins = ceil(log2(n)) + 1`)로 교체하면 92샘플 기준 50→8 bins로 감소, 잡음 peak 제거 가능.

---

## 영향도 분석

### 이 기능이 변경하는 상태
- `distribution_analysis.json`의 `n_valleys`, `valley_positions`, `boundaries` 값
- 결과적으로 Step 1 MCI 계산 시 valley_count 컴포넌트 값 변경

### 그 상태를 전제로 동작하는 기존 로직
- `compute_complexity.py` — `distribution_analysis.json` 읽어 valley_count 산출 → MCI 계산
- `aroma_step1.yaml` — `expected_range[valley_count]: [0.0, 18.0]` 설정값 (이 수정 후 재평가 필요)

---

## 수정 내용

### 1. `scripts/distribution_profiling.py` — bins 동적 계산 적용

**변경 위치**: line 86 (상수), line 326 (bins 계산)

변경 전:
```python
HISTOGRAM_BINS = 50  # line 86

# line 326 (_detect_valleys 내부)
bins = min(HISTOGRAM_BINS, max(len(work) - 1, 2))
```

변경 후:
```python
# line 86: 상수를 max_bins fallback으로 의미 변경 (대표본 상한)
MAX_HISTOGRAM_BINS = 50

# line 326
bins = max(int(np.ceil(np.log2(len(work)))) + 1, 5)
bins = min(bins, MAX_HISTOGRAM_BINS)  # 대표본 상한
```

- 92-100 샘플: bins 50 → 8
- 1000 샘플: bins 11
- 10000 샘플: bins 15 (상한 50 적용 전까지)
- `noise_floor`, `prominence` 계산식은 변경 없음 — bins 축소에 따라 자동 적정화

**docstring 갱신** (line 308-316):
`n_bins`가 고정이 아닌 Sturges' rule 기반 동적 값임을 반영.

---

## 수정 대상 파일

- `scripts/distribution_profiling.py`

---

## 테스트

Colab Phase 0 재실행 후 `distribution_analysis.json` 확인:

```python
# 각 데이터셋 재실행 후 valley_count 확인
import json
with open(f"{PROFILING_DIR}/distribution_analysis.json") as f:
    data = json.load(f)
for feat, info in data.items():
    print(f"{feat}: {info['n_valleys']} valleys")
```

예상 결과:
| feature | mvtec_cable 전→후 | visa_cashew 전→후 | isp_LSM_1 전→후 |
|---------|-----------------|-----------------|----------------|
| circularity | 12 → 1~3 | 5 → 유지 | 6 → 감소 |
| eccentricity | 7 → 감소 | 11 → ~3 | 3 → 유지 |
| extent | 6 → 감소 | 6 → 감소 | 7 → ~3 |

회귀 확인: log_transform 적용 feature(aspect_ratio)에서 valley 위치 expm1 역변환 후 정상 범위 유지.

---

## TODO

- [ ] bins 하한 5가 군집 4개 시나리오(valley 3개 필요)에서 충분한지 Colab 결과로 검증
- [ ] 수정 후 valley_count 합산 재계산 → `expected_range[valley_count]: [0.0, 18.0]` 유지 또는 재보정 결정
- [ ] Freedman-Diaconis(IQR 기반) 대안 검토 여부 — 현재 합의는 Sturges(A안) 유지
