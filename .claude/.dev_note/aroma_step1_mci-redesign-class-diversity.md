# MCI 재설계 — ClusterCount → ClassDiversity (Neff), ValleyCount pooled 복귀

---

## (사용할 skills: micro-fix)

## 개요

세 데이터셋 결과 분석 후 MCI 컴포넌트 재설계.

**문제**: per-class valley_count 도입 시 cable(8종)이 소샘플 문제로 valley=1.25 → MCI 최하위. 기대 순서 역전.
**원인**: cable 8클래스 × ~20-30 인스턴스/클래스 → 히스토그램 샘플 부족 → valley 미감지.

**결론**: ClusterCount(GMM BIC k)를 ClassDiversity(Neff = e^H)로 교체.
- valley_count → pooled 복귀 (per-class 제거)
- class_diversity = exp(Shannon entropy of defect_type distribution)
  - cable(8종 균등) ≈ 8.0, isp(2종) ≈ 2.0, cashew(1종) = 1.0
- 논문 설명: "Effective number of defect categories estimated from the Shannon entropy of class distributions"

**케이블 MCI 설명 가능**: valley 수 낮아도 ClassDiversity=8.0 + InvSilhouette=1.0(군집화 불가) → 복잡도 정당화.

---

## 영향도 분석

### 변경하는 상태
- `complexity_report.json` MCI 구성: cluster_count 제거 → class_diversity 추가
- valley_count raw 값: per-class mean → pooled total
- MCI 수치 전면 변경 (3개 데이터셋 모두 재실행 필요)

### 기존 로직 영향
- Meta Policy Generator: MCI 값 변경 → hierarchical/otsu 선택 기준(0.6/0.3) 재검토 필요
- `_WEIGHT_PRESETS`: "cluster_heavy" → "diversity_heavy" 개명

---

## 수정 내용

### 1. `scripts/aroma/compute_complexity.py`

#### 1.1 `_DEFAULT_CONFIG` — expected_range 변경
```yaml
# 변경 전
valley_count:  [0.0, 18.0]
cluster_count: [1.0, 8.0]
orient_variance: [0.0, 1.0]

# 변경 후
valley_count:    [0.0, 41.0]   # anchor = 34(cashew 관측 max) × 1.2
class_diversity: [1.0, 9.6]   # anchor = 8 types × 1.2
orient_variance: [0.0, 1.6]   # anchor = 1.337(isp 관측 max) × 1.2
```

#### 1.2 `_WEIGHT_PRESETS` — "cluster_heavy" → "diversity_heavy"

#### 1.3 `_perclass_valley_count()` — 삭제 (소샘플 문제로 폐기)

#### 1.4 `_class_diversity_neff(morph_rows)` — 신규 추가
```python
H = -sum((c/total) * math.log(c/total) for c in counts.values())
return math.exp(H)
```

#### 1.5 `compute_mci()` — 컴포넌트 교체
```python
# 변경 전
raw = {valley_count: _perclass_valley_count(...), cluster_count: n_morph_clusters, ...}

# 변경 후
raw = {valley_count: _total_valley_count(dist_analysis),  # pooled 복귀
       class_diversity: _class_diversity_neff(morph_rows_raw), ...}
```

#### 1.6 docstring 업데이트: ClusterCount → ClassDiversity

### 2. `scripts/aroma/config/aroma_step1.yaml`

동일 range 변경 반영. weights 설명에 diversity_heavy 추가.

---

## 수정 대상 파일

- `scripts/aroma/compute_complexity.py`
- `scripts/aroma/config/aroma_step1.yaml`

---

## 기대 결과

| | valley_count | class_diversity | inv_silhouette | MCI 방향 |
|--|--|--|--|--|
| cable | 16 (pooled) | ~8.0 | 1.0 | 상승 예상 |
| isp | 13 (pooled) | ~2.0 | 0.572 | 하락 예상 |
| cashew | 15 (pooled) | 1.0 | 0.687 | 중간 예상 |

목표 순서: cable > cashew > isp (또는 cable > isp ≈ cashew)

## 테스트

```python
!python $AROMA_SCRIPTS/compute_complexity.py \
    --profiling_dir $AROMA_OUT/profiling/mvtec_cable \
    --output_dir    $AROMA_OUT/complexity/mvtec_cable

# 확인: mci_components.raw.class_diversity ≈ 8.0
# 확인: mci_components.raw.valley_count = pooled 값
# 확인: MCI 순서 cable > cashew > isp
```
