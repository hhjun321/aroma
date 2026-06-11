# AROMA Phase 0 — 데이터셋 3종 1차 분석

> 분석 대상: `isp_LSM_1` / `mvtec_cable` / `visa_cashew`
> Phase 0 (`distribution_profiling.py`) 출력 기준

---

## 1. 기본 지표 비교

| 항목 | isp_LSM_1 | mvtec_cable | visa_cashew |
|------|----------|------------|------------|
| 도메인 | ISP | MVTec | VisA |
| defect 인스턴스 수 | 95 | 92 | 100 |
| defect 종류 수 | **2** (area, points) | **8** | **1** (anomaly) |
| context patches | 60,368 | 79,868 | 35,124 |
| fallback masks | **95/95 (100% Otsu)** | 0/92 (GT mask) | 0/100 (GT mask) |
| 이미지 해상도 (추정) | 256px (16 patches/img) | 1024px (256 patches/img) | 512px (64 patches/img) |
| GMM 클러스터 수 | 4 | 4 | 4 |

**context patch 수 산출 근거**

- isp_LSM_1: 60,368 / 16 = **3,773** = 3,678(good) + 95(defect) ✓ → 256px 확인
- mvtec_cable: 79,868 / 256 ≈ 312 ≈ 224(good) + 92(defect) ✓ → 1024px 확인
- visa_cashew: 35,124 / 64 ≈ 549 ≈ ~500(good) + 100(defect) ✓ → 512px 확인

---

## 2. Valley Count 분석

### 2.1 Feature별 비교

| Feature | isp_LSM_1 | mvtec_cable | visa_cashew |
|---------|----------|------------|------------|
| linearity | 5 | 7 | 9 |
| solidity | 3 | 3 | 3 |
| extent | 1 | 1 | 0 |
| aspect_ratio | 3 | 4 | 3 |
| eccentricity | 1 | 0 | 0 |
| circularity | 0 | 1 | 0 |
| **합계** | **13** | **16** | **15** |

> bounded feature prominence 수정 적용 후 (circularity, eccentricity, extent 과감지 억제)

### 2.2 Expected Range 천장 문제

현재 설정: `expected_range[valley_count] = [0, 18]`  
근거: "6 features × max 3 valleys" 가정 (aroma_step1.yaml 주석)

**3종 모두 초과 → norm=1.0 고정 → MCI 판별력 없음**

```
isp_LSM_1:   27/18 = 1.50 → norm = 1.000
mvtec_cable: 39/18 = 2.17 → norm = 1.000
visa_cashew: 37/18 = 2.06 → norm = 1.000
```

### 2.3 과감지 의심 Feature

| Feature | 과감지 의심 데이터셋 | valleys | 근거 |
|---------|-----------------|---------|------|
| circularity | mvtec_cable | 12 | 92샘플에 12 valleys, [0,1] 균일 분포 경향 |
| eccentricity | visa_cashew | 11 | 100샘플에 11 valleys, [0.47~0.97] 밀집 구간 |
| linearity | visa_cashew | 9 | single anomaly class임에도 9 valleys |

[0,1] 범위 feature(circularity, eccentricity)는 분포가 평탄한 구간에서도 GMM이 valley를 잡는 경향. 노이즈 prominance 임계값이 상대적으로 낮기 때문.

### 2.4 Expected Range 재보정 시뮬레이션

| 데이터셋 | defect 종류 | [0,18] 현재 | [0,30] | [0,45] | [0,60] |
|---------|-----------|-----------|--------|--------|--------|
| isp_LSM_1 | 2종 | 1.000 | 0.900 | **0.600** | 0.450 |
| visa_cashew | 1종(다양) | 1.000 | 1.000 | **0.822** | 0.617 |
| mvtec_cable | 8종 | 1.000 | 1.000 | **0.867** | 0.650 |

**`[0, 45]` 권장**: isp_LSM_1(단순 2종)과 나머지를 구분하면서, 3종 모두 표현 가능.  
단, 과감지 문제 선결 없이 range만 바꾸면 노이즈 valley가 MCI를 과장할 수 있음.

---

## 3. 클러스터 품질 비교

### 3.1 클러스터 구성

**isp_LSM_1** (Otsu fallback 100%)

| Cluster | Label | N | aspect_ratio | linearity | circularity | 비고 |
|---------|-------|---|-------------|-----------|------------|------|
| 0 | general | **51** (54%) | 3.83 | 0.92 | 0.29 | 세장형 — area 34개 포함 |
| 1 | general | 12 | 2.58 | 0.68 | 0.10 | 중간 세장형 |
| 2 | compact_blob | 20 | 1.03 | 0.06 | 0.38 | 원형 — points 14개 포함 |
| 3 | general | 12 | 1.06 | 0.10 | **0.046** | 불규칙 blob — points 12개 전담 |

**mvtec_cable** (GT mask)

| Cluster | Label | N | aspect_ratio | linearity | circularity | 비고 |
|---------|-------|---|-------------|-----------|------------|------|
| 0 | elongated | 14 | 4.08 | 0.93 | 0.31 | bent_wire, missing_wire |
| 1 | general | 25 | 1.11 | 0.18 | 0.29 | cable_swap, cut_inner |
| 2 | compact_blob | 17 | 1.30 | 0.38 | **0.80** | poke, cut_outer |
| 3 | general | 36 (39%) | 2.03 | 0.72 | 0.30 | combined, missing_cable |

**visa_cashew** (GT mask)

| Cluster | Label | N | aspect_ratio | linearity | circularity | 비고 |
|---------|-------|---|-------------|-----------|------------|------|
| 0 | compact_blob | 35 (35%) | 1.47 | 0.53 | 0.67 | 중간 크기 blob |
| 1 | general | 31 | 2.43 | 0.81 | 0.55 | 중간 세장형 |
| 2 | **linear_scratch** | 17 | **10.05** | **0.97** | 0.17 | 극단적 세장형 — 완전 분리 |
| 3 | compact_blob | 17 | 1.11 | 0.18 | **0.77** | 소형 원형 blob |

### 3.2 클러스터 균형도 비교

| 데이터셋 | 최대 점유율 | 분포 균형 | 품질 평가 |
|---------|-----------|---------|---------|
| isp_LSM_1 | 54% (cl.0) | 불균형 | **낮음** — Otsu 노이즈로 과집중 |
| mvtec_cable | 39% (cl.3) | 중간 | 중상 — 8종이 4그룹으로 합리적 분리 |
| visa_cashew | 35% (cl.0) | 양호 | **높음** — linear_scratch 완전 분리 |

예상 silhouette 순서: `visa_cashew > mvtec_cable > isp_LSM_1`

---

## 4. Fallback Mask 영향 분석

### 4.1 isp_LSM_1 — Otsu 왜곡 사례

isp_LSM_1의 `points` 결함은 물리적으로 소형 원형 점이어야 함 (circularity ≈ 0.8~1.0 예상).  
그러나 cluster 3 (points 12개 전담)의 circularity = **0.046** — 거의 0.

**원인**: Otsu 임계값이 points 결함 경계를 과도하게 확장하거나 노이즈 픽셀을 포함시켜 비원형 mask 생성.

```
GT mask가 있었다면:
  points → circularity ≈ 0.8~1.0, compact_blob 클러스터 집중
  area   → circularity ≈ 0.1~0.5, elongated/general 클러스터

실제 Otsu 결과:
  cluster 0에 area(34) + points(17) 혼재 → 형태학 분리 실패
  cluster 3: points가 circularity=0.046의 불규칙 blob으로 왜곡
```

### 4.2 GT mask vs Otsu — 신뢰도 비교

| 항목 | GT mask (cable, cashew) | Otsu fallback (isp_LSM_1) |
|------|------------------------|--------------------------|
| 형태학 feature 신뢰도 | 높음 | **낮음** |
| circularity 왜곡 | 없음 | 심각 (points: 0.046) |
| 클러스터 분리도 | 중~상 | 낮음 |
| MCI 해석 | 데이터셋 복잡도 반영 | Otsu 노이즈 포함 |

**결론**: isp_LSM_1의 Phase 0 결과는 Otsu 노이즈를 포함하므로 Step 1 MCI 해석 시 주의 필요.  
SAM으로 재실행 시 클러스터 품질 및 valley 분포 변화 예상.

---

## 5. MCI 예측 및 개선 방향

### 5.1 현재 expected_range 기준 MCI 예측

valley_count 3종 모두 norm=1.0 → valley_count 컴포넌트가 MCI를 지배.  
차이는 `inv_silhouette` (= 1 - silhouette) 에서만 발생.

```
예상 MCI 순서 (expected_range [0,18]):
  isp_LSM_1 ≥ mvtec_cable ≈ visa_cashew

이유:
  - isp_LSM_1: inv_silhouette 높음 (클러스터 분리 불량) → MCI 오히려 높을 수 있음
  - visa_cashew: linear_scratch 분리 우수 → silhouette 높음 → inv_silhouette 낮음 → MCI 낮음
```

**역설**: 단순한(2종) 데이터셋이 Otsu 노이즈로 인해 복잡한 데이터셋보다 MCI가 높게 나올 수 있음.

### 5.2 개선 방향

**단기 (expected_range 재보정)**

```yaml
# aroma_step1.yaml 수정 제안
mci:
  expected_range:
    valley_count: [0.0, 45.0]   # 기존 18.0 → 45.0
```

효과:
- isp_LSM_1 valley norm: 1.000 → 0.600
- mvtec_cable valley norm: 1.000 → 0.867  
- visa_cashew valley norm: 1.000 → 0.822
- defect 종류 수 차이(2종 vs 8종)를 MCI로 구분 가능해짐

**중기 (valley 과감지 개선)**

과감지 우선 검토 대상:
1. `circularity`: prominence 임계값 상향 또는 최대 valleys/feature 제한
2. `eccentricity`: [0,1] bounded feature 전용 감도 파라미터 분리

**장기 (isp 도메인 SAM 재실행)**

isp_LSM_1을 SAM으로 재실행 후 비교:
- points 결함 circularity 복구 확인
- 클러스터 분리도 개선 여부 확인

---

## 요약

| 평가 항목 | isp_LSM_1 | mvtec_cable | visa_cashew |
|---------|----------|------------|------------|
| valley_count | 27 (현재 범위 초과) | 39 (최고) | 37 |
| 과감지 의심 | extent=7 | circularity=12 | eccentricity=11 |
| 클러스터 품질 | 낮음 (Otsu) | 중상 | **높음** |
| GT mask | ❌ Otsu | ✅ | ✅ |
| MCI 신뢰도 | 낮음 | 높음 | 높음 |
| **재보정 후 예상 복잡도** | 낮음 (2종) | **높음** (8종) | 중간 (1종 다양) |

---

## 6. Data-Driven Anchor 설계

### 6.1 Feature 분류 — Bounded vs Unbounded

| 분류 | Feature | 현재 expected_range | 처리 방법 |
|------|---------|-------------------|---------|
| **Bounded** | inv_silhouette | [0, 1] | 고정 유지 |
| **Bounded** | cluster_count | [2, 10] | 고정 유지 |
| **Unbounded** | valley_count | [0, 18] ← **문제** | Data-driven anchor |
| **Unbounded** | entropy | [0, ?] | Data-driven anchor |
| **Unbounded** | FreqComplexity (CCI) | [0, ?] | Data-driven anchor |
| **Unbounded** | OrientVariance (CCI) | [0, ?] | Data-driven anchor |

Bounded feature는 수학적·설정 기반 상한이 존재하므로 anchor 관리 불필요.  
Unbounded feature는 해상도·defect 종류·텍스처에 따라 값이 달라져 고정 범위 유지 불가.

### 6.2 Anchor 산출 공식

```
n_datasets < 100:  anchor = max(dataset_values) × 1.2
n_datasets ≥ 100:  anchor = percentile(dataset_values, 95)
```

- 소규모(n<100): 관측 최댓값에 여유 20% 부여
- 대규모(n≥100): 극단값 무시, p95로 robust 상한 산출

### 6.3 anchor.json 스키마 제안

```json
{
  "version": "1.0",
  "created_from": ["isp_LSM_1", "mvtec_cable", "visa_cashew"],
  "n_datasets": 3,
  "anchors": {
    "valley_count": {
      "method": "max_x1.2",
      "value": 46.8,
      "raw_max": 39.0
    },
    "entropy": {
      "method": "max_x1.2",
      "value": "<TODO: 실측 후 기입>",
      "raw_max": "<TODO>"
    }
  }
}
```

### 6.4 시뮬레이션 결과 (valley_count 기준)

입력: valley_counts = [13, 16, 15] (isp / cable / cashew)  
anchor = max(16) × 1.2 = **19.2**

| 데이터셋 | valley_count | norm (÷19.2) | 기존 norm (÷18) |
|---------|-------------|-------------|---------------|
| isp_LSM_1 | 13 | **0.677** | 1.000 |
| mvtec_cable | 16 | **0.833** | 1.000 |
| visa_cashew | 15 | **0.781** | 1.000 |

기존 판별력 0 → 3종 간 상대 순위 복원.  
8종 defect cable이 2종 isp보다 높게 측정 — 직관과 일치.

> 과감지 수정 적용 후 실측값 기준 (bounded feature CV 기반 prominence 동적 조정)

### 6.5 미해결 이슈

1. ~~**histogram bins fix 선행 조건**~~ → ✅ **완료**: bounded feature CV 기반 prominence 동적 조정 적용. anchor 19.2로 확정.
2. **threshold_n=100 현실성**: 현재 보유 데이터셋 3종 → n<100 경로만 사용. p95 경로는 향후 대규모 실험 후 검증 필요.
3. **entropy/FreqComplexity 단위**: 실측값 범위 미확인. Step 1 실행 후 `distribution_analysis.json`에서 entropy 분포 확인 후 anchor 추가 기입 필요.
