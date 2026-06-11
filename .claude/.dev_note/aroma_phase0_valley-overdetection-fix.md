# valley 과감지 억제 — bounded feature 평탄도 기반 prominence 동적 조정

## (사용할 skills: micro-fix)

## 개요

`_detect_valleys()`에서 Sturges' rule bins 수정(#histogram-bins-dynamic) 이후에도 circularity(mvtec_cable=12), eccentricity(visa_cashew=11) 과감지가 동일하게 유지됨. bins 수가 아닌 `VALLEY_PROMINENCE_RATIO=0.1`이 실제 원인으로 확인. flat distribution에서 `counts.max() * 0.1` 항이 잡음 peak를 통과시킴.

B+C안 조합으로 수정:
- **B안**: 히스토그램 평탄도(CV = std/mean)가 낮으면 prominence ratio 자동 상향
- **C안**: bounded feature 집합으로 적용 범위 한정 (회귀 안전망)

---

## 영향도 분석

### 이 기능이 변경하는 상태
- `distribution_analysis.json`의 `n_valleys`, `valley_positions`, `boundaries`, `policy` 필드
- `policy` 변화 예상: 일부 feature가 `multimodal → unimodal` 전환 → `boundaries` 산출 방식 percentile로 전환 (의도된 변화)
- 하류: `compute_complexity.py`의 valley_count 컴포넌트 값 변경 → MCI 재계산 필요

### 그 상태를 전제로 동작하는 기존 로직
- `compute_complexity.py` — `distribution_analysis.json` 읽어 valley_count 산출 → MCI 계산
- `aroma_step1.yaml` — `expected_range[valley_count]` 설정값 (anchor 재계산 필요)
- `phase0_analysis.md` 섹션 2.1 valley_count 표 — 수정 후 재측정값으로 갱신 필요

---

## 수정 내용

### 1. `scripts/distribution_profiling.py` — 상수 추가 (line 87 부근)

```python
# 기존
VALLEY_PROMINENCE_RATIO = 0.1

# 추가
BOUNDED_FEATURES = {"circularity", "eccentricity", "solidity", "extent"}
FLAT_CV_THRESHOLD = 0.5          # TODO: 실측 CV 분포 보고 확정
BOUNDED_VALLEY_PROMINENCE_RATIO = 0.3   # TODO: 실측 후 보정
```

### 2. `scripts/distribution_profiling.py` — `_detect_valleys` 시그니처 및 prominence 계산 수정 (line 307~348)

**시그니처 변경** (line 307):
```python
# 기존
def _detect_valleys(values: np.ndarray) -> Tuple[int, List[float]]:

# 변경
def _detect_valleys(values: np.ndarray, feature_name: str = None) -> Tuple[int, List[float]]:
```

**prominence 계산 수정** (line 334-335):
```python
# 기존
noise_floor = np.sqrt(len(work) / bins)
prominence = max(noise_floor * 2, counts.max() * VALLEY_PROMINENCE_RATIO)

# 변경
noise_floor = np.sqrt(len(work) / bins)
cv = counts.std() / (counts.mean() + 1e-9)
is_bounded = feature_name in BOUNDED_FEATURES if feature_name else False
if is_bounded and cv < FLAT_CV_THRESHOLD:
    effective_ratio = BOUNDED_VALLEY_PROMINENCE_RATIO
else:
    effective_ratio = VALLEY_PROMINENCE_RATIO
prominence = max(noise_floor * 2, counts.max() * effective_ratio)
```

**docstring 갱신** (line 308-320):
- `VALLEY_PROMINENCE_RATIO` 고정 사용 설명 → `effective_ratio` 동적 조정 설명으로 교체
- `BOUNDED_FEATURES`, `FLAT_CV_THRESHOLD` 조건 명시

### 3. `scripts/distribution_profiling.py` — 호출부 수정 (line 624)

```python
# 기존
n_valleys, valley_pos = _detect_valleys(values)

# 변경
n_valleys, valley_pos = _detect_valleys(values, feat)
```

---

## 수정 대상 파일

- `scripts/distribution_profiling.py`

---

## 테스트

Phase 0 재실행 후 검증:

```python
import json
from pathlib import Path

AROMA_OUT = os.environ['AROMA_OUT']
datasets  = ['isp_LSM_1', 'mvtec_cable', 'visa_cashew']
features  = ['linearity','solidity','extent','aspect_ratio','eccentricity','circularity']

for ds in datasets:
    path = Path(AROMA_OUT) / 'profiling' / ds / 'distribution_analysis.json'
    with open(path) as f:
        data = json.load(f)
    print(f"\n{ds}")
    for feat in features:
        n = data.get(feat, {}).get('n_valleys', '-')
        print(f"  {feat:<20} {n}")
```

**목표 지표**:
| feature | 데이터셋 | 수정 전 | 목표 |
|---------|---------|--------|------|
| circularity | mvtec_cable | 12 | ~1 |
| eccentricity | visa_cashew | 11 | ~3 |

**회귀 확인** (genuine multi-modal 유지):
- aspect_ratio, linearity의 n_valleys 수정 전후 동일한지 비교

---

## TODO

- [ ] `FLAT_CV_THRESHOLD=0.5` 확정 전 실측 CV 출력 확인 (각 feature의 실제 CV 값)
- [ ] `BOUNDED_VALLEY_PROMINENCE_RATIO=0.3` 확정 전 과감지 억제 vs 진짜 valley 보존 밸런스 확인
- [ ] `BOUNDED_FEATURES`에서 solidity/extent 포함 여부 — 이 둘이 실제 flat 분포인지 데이터 확인
- [ ] 수정 후 valley_count 재집계 → anchor 재계산 (46.8에서 변경 가능)
- [ ] `phase0_analysis.md` 섹션 2.1 수치 갱신 필요
