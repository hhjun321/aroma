# ClassDiversity 정규화 — log-scale (ln(Neff)/ln(N_max))

---

## (사용할 skills: micro-fix)

## 개요

기존 min-max 정규화(range [1.0, 9.6])에서 로그 스케일 정규화로 변경.

**문제**: min-max 시 isp(Neff≈2.0) norm=0.116 → class_diversity가 MCI에 거의 기여 안 함 → isp≈cashew (0.389 vs 0.383, gap=0.006).

**해결**: ln(Neff)/ln(N_max) 적용.
- 2종 vs 1종 차이가 ln(2)/ln(8)=0.333으로 의미 있게 반영
- cable 오버슈팅 방지: 8종 → 0.997 ≈ 1.0 (상한 자연 수렴)
- N_max=8 고정 (현재 관측 최대 클래스 수, cable 기준)

**수식**:
$$\text{NormDiv} = \text{clamp}_{[0,1]}\left(\frac{\ln(N_{\text{eff}})}{\ln(N_{\text{max}})}\right)$$

**논문 참조 문구** (추후 게재 시 사용):
> "We observed that the diversity of defect classes in industrial datasets follows a non-linear complexity distribution; adding a second defect type increases the morphological challenge significantly more than adding an eighth. To capture this, we employ a logarithmic transformation of the effective class count ($N_{\text{eff}}$) for normalization. This enhances the sensitivity of MCI to differences between sparse-class datasets (e.g., 1 vs 2 classes), allowing AROMA to differentiate policy requirements even in simple industrial environments."

---

## 예상 결과

| | Neff | NormDiv | 예상 MCI |
|--|--|--|--|
| cable | 7.94 | 0.997 | **0.664** |
| isp | 2.00 | 0.333 | **0.443** |
| cashew | 1.00 | 0.000 | **0.383** |

순서: cable > isp > cashew (gap isp-cashew: 0.006 → **0.060**, 10×)

---

## 수정 내용

### 1. `scripts/aroma/compute_complexity.py`

#### 1.1 `_DEFAULT_CONFIG` — class_diversity expected_range 제거, n_max 추가

```python
"mci": {
    ...
    "expected_range": {
        "entropy":      [0.0, 4.0],
        "valley_count": [0.0, 41.0],
        # class_diversity 제거 — log-scale normalization은 별도 파라미터 사용
    },
    "class_diversity_n_max": 8,  # N_max for ln(Neff)/ln(N_max)
},
```

#### 1.2 `compute_mci()` — class_diversity 정규화 로직 교체

```python
# 변경 전
"class_diversity": _normalize_scalar(raw["class_diversity"], *rng["class_diversity"], norm_mode),

# 변경 후
n_max = float(cfg["mci"].get("class_diversity_n_max", 8.0))
log_n_max = math.log(n_max) if n_max > 1.0 else 1.0
neff = raw["class_diversity"]
norm_div = _clamp01(math.log(max(neff, 1.0)) / log_n_max)
# 그 후 normalized dict에 직접 삽입
```

### 2. `scripts/aroma/config/aroma_step1.yaml`

```yaml
mci:
  expected_range:
    entropy:         [0.0, 4.0]
    valley_count:    [0.0, 41.0]
    # class_diversity: 제거
  class_diversity_n_max: 8     # N_max for log-scale normalization
```

---

## 수정 대상 파일

- `scripts/aroma/compute_complexity.py`
- `scripts/aroma/config/aroma_step1.yaml`

---

## 엣지 케이스

- Neff < 1.0 (이론상 불가): max(neff, 1.0)으로 보호 → NormDiv=0.0
- N_max=1 (단일 클래스 데이터셋만 존재): log_n_max=0 방지 → fallback=1.0
- Neff > N_max (새 데이터셋에서 클래스 수 초과): clamp01로 1.0으로 처리

## 테스트

```python
# complexity_report.json 확인
# mci_components.normalized.class_diversity:
#   cable ≈ 0.997, isp ≈ 0.333, cashew = 0.000
# MCI: cable > isp > cashew (gap > 0.05)
```
