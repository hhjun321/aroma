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

### 2.2 Expected Range 천장 문제 → 해결

초기 설정: `expected_range[valley_count] = [0, 18]` (6 features × max 3 valleys 가정)

과감지 수정(bounded feature CV 기반 prominence 조정) 후 pooled 기준 실측값:
```
isp_LSM_1:   13  (2종)
mvtec_cable: 16  (8종)
visa_cashew: 15  (1종)
```

**확정 range: `[0.0, 41.0]`**  
근거: 중간 과정에서 per-class mean 최대값 34 × 1.2 = 40.8 → 41로 설정 후 유지 (pooled 복귀 후에도 보수적 상한으로 유지, 향후 대규모 데이터셋 대비)

### 2.3 과감지 의심 Feature

| Feature | 과감지 의심 데이터셋 | valleys | 근거 |
|---------|-----------------|---------|------|
| circularity | mvtec_cable | 12 | 92샘플에 12 valleys, [0,1] 균일 분포 경향 |
| eccentricity | visa_cashew | 11 | 100샘플에 11 valleys, [0.47~0.97] 밀집 구간 |
| linearity | visa_cashew | 9 | single anomaly class임에도 9 valleys |

[0,1] 범위 feature(circularity, eccentricity)는 분포가 평탄한 구간에서도 GMM이 valley를 잡는 경향. 노이즈 prominance 임계값이 상대적으로 낮기 때문.

### 2.4 Expected Range 재보정 시뮬레이션 (참고용, 과거 분석)

> ⚠️ 아래는 과감지 수정 전 (valley_count 27/39/37) 기준 시뮬레이션 — 참고용 보존.  
> **확정 range: `[0.0, 41.0]`** (Section 6.4 실측 결과 참조)

| 데이터셋 | defect 종류 | [0,18] 구버전 | [0,30] | [0,45] | [0,60] |
|---------|-----------|-----------|--------|--------|--------|
| isp_LSM_1 | 2종 | 1.000 | 0.900 | 0.600 | 0.450 |
| visa_cashew | 1종(다양) | 1.000 | 1.000 | 0.822 | 0.617 |
| mvtec_cable | 8종 | 1.000 | 1.000 | 0.867 | 0.650 |

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

## 5. Step 1 확정 결과 (2026-06-12)

### 5.1 MCI 설계 확정

**4-component 등가중치**:

```
MCI = 0.25 × Entropy_norm
    + 0.25 × ValleyCount_norm
    + 0.25 × ClassDiversity_norm
    + 0.25 × InvSilhouette
```

| 컴포넌트 | 정규화 방법 | 범위 |
|---------|----------|------|
| Entropy | minmax [0, 4.0] | |
| ValleyCount | minmax [0, 41.0] | pooled 전체 인스턴스 기준 |
| ClassDiversity | log-scale: ln(Neff)/ln(8) | Neff=e^H, N_max=8 |
| InvSilhouette | 1 - silhouette | silhouette ∈ [-1,1] → clamp01 |

**ClassDiversity (Hill 다양성 지수)**:  
Neff = e^H (Shannon 엔트로피 기반 유효 클래스 수). 클래스 분포가 균일할수록 Neff → K.  
로그 스케일 정규화: 2번째 defect 종류 추가 시 복잡도 증가폭이 8번째 추가 시보다 훨씬 큼.

### 5.2 MCI 확정 결과

| 데이터셋 | Entropy (raw/norm) | ValleyCount (raw/norm) | ClassDiversity (Neff/norm) | InvSilhouette | **MCI** |
|---------|-------------------|----------------------|--------------------------|---------------|---------|
| isp_LSM_1 | 2.202 / 0.551 | 13 / 0.317 | 1.999 / 0.333 | 0.572 | **0.4432** |
| mvtec_cable | 1.071 / 0.268 | 16 / 0.390 | 7.940 / 0.996 | 1.000 | **0.6636** |
| visa_cashew | 1.923 / 0.481 | 15 / 0.366 | 1.000 / 0.000 | 0.687 | **0.3834** |

**MCI 순서**: `mvtec_cable (0.664) > isp_LSM_1 (0.443) > visa_cashew (0.383)` ✓ 직관 일치

cable이 최고인 이유: 8종 defect → ClassDiversity ≈ 1.0, InvSilhouette=1.0 (silhouette=-0.179)

### 5.3 CCI 확정 결과

20,000 패치 bootstrap 서브샘플링 적용 (해상도 공정성).

| 데이터셋 | TextureEntropy (raw/norm) | ClusterCount_ctx (raw/norm) | FreqComplexity (raw/norm) | OrientVariance (raw/norm) | **CCI** |
|---------|--------------------------|----------------------------|--------------------------|--------------------------|---------|
| isp_LSM_1 | 2.403 / 0.300 | 5.0 / 0.571 | 0.0145 / 0.014 | 1.337 / 0.835 | **0.4304** |
| mvtec_cable | 2.825 / 0.353 | 5.0 / 0.571 | 0.0013 / 0.001 | 0.079 / 0.049 | **0.2438** |
| visa_cashew | 3.253 / 0.407 | 5.0 / 0.571 | 0.0082 / 0.008 | 0.011 / 0.007 | **0.2483** |

CCI 서브샘플 수: isp=20,000/58,848, cable=20,000/57,344, cashew=20,000/28,800

### 5.4 정책 결정

| 데이터셋 | MCI | CCI | Morphology Policy | Context Policy |
|---------|-----|-----|-------------------|----------------|
| isp_LSM_1 | 0.443 | 0.430 | **otsu** (stability tie-break) | gmm |
| mvtec_cable | 0.664 | 0.244 | **hierarchical** (MCI>0.6) | gmm |
| visa_cashew | 0.383 | 0.248 | **otsu** | gmm |

cable: MCI=0.664 → hierarchical 정책 (다중 수준 임계값)  
isp/cashew: MCI<0.5 → otsu 정책 (단순 이진 임계값)

### 5.5 향후 과제

- isp_LSM_1 SAM 재실행 후 재평가 (Otsu 노이즈로 inv_silhouette 과대 추정 가능성)
- valley_count range [0,41] 타당성: 현재 pooled max=16 → 보수적 상한. 더 많은 데이터셋 추가 후 anchor 갱신 예정

---

## 요약

| 평가 항목 | isp_LSM_1 | mvtec_cable | visa_cashew |
|---------|----------|------------|------------|
| valley_count (pooled) | **13** | **16** | **15** |
| defect 종류 수 | 2 | 8 | 1 |
| ClassDiversity Neff | 1.999 | 7.940 | 1.000 |
| 클러스터 품질 | 낮음 (Otsu) | 중상 | **높음** |
| GT mask | ❌ Otsu | ✅ | ✅ |
| **MCI** | 0.443 | **0.664** | 0.383 |
| **CCI** | **0.430** | 0.244 | 0.248 |
| **정책** | otsu | **hierarchical** | otsu |

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
      "method": "conservative_max",
      "value": 41.0,
      "raw_max": 16.0,
      "note": "per-class 과정에서 관측된 max=34 × 1.2 = 41 (pooled 복귀 후 보수적 상한 유지)"
    },
    "orient_variance": {
      "method": "max_x1.2",
      "value": 1.6,
      "raw_max": 1.337
    },
    "entropy": {
      "method": "fixed",
      "value": 4.0,
      "note": "이론적 최대값 (6 features × log2 bins 기준)"
    },
    "texture_entropy": {
      "method": "fixed",
      "value": 8.0,
      "note": "이론적 최대값"
    }
  }
}
```

### 6.4 확정 normalization 결과 (Step 1 실측)

| 데이터셋 | valley_count (raw) | norm (÷41.0) | orient_var (raw) | norm (÷1.6) |
|---------|-------------------|-------------|-----------------|------------|
| isp_LSM_1 | 13 | **0.317** | 1.337 | **0.835** |
| mvtec_cable | 16 | **0.390** | 0.079 | **0.049** |
| visa_cashew | 15 | **0.366** | 0.011 | **0.007** |

valley_count [0,41]로 3종 간 순위 복원.  
orient_variance: isp가 현저히 높음 (다양한 결함 방향) → CCI isp=0.430 > cable/cashew의 주요 요인.

### 6.5 미해결 이슈

1. ~~**histogram bins fix 선행 조건**~~ → ✅ **완료**: bounded feature CV 기반 prominence 동적 조정 적용. anchor 19.2로 확정.
2. **threshold_n=100 현실성**: 현재 보유 데이터셋 3종 → n<100 경로만 사용. p95 경로는 향후 대규모 실험 후 검증 필요.
3. **entropy/FreqComplexity 단위**: 실측값 범위 미확인. Step 1 실행 후 `distribution_analysis.json`에서 entropy 분포 확인 후 anchor 추가 기입 필요.
