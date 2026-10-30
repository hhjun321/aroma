# AROMA 특징(feature) 6종 — 수식·구현·논문 대응

> 원본 메모(특징 → 등장 위치)를 코드·논문으로 조사해 확장. 조사일 2026-10-30.
> 코드 기준: `scripts/distribution_profiling.py:213 _extract_context_features`, `scripts/aroma/compute_complexity.py:580 compute_cci`, `utils/background_characterization.py:93 classify_patch`.
> 논문 기준: `Article/text/section3_2.txt` §3.2.1 식(1)·Table 2, §3.2.3 Table 3 + 정의 문단.

| 특징 | 등장 위치 |
|---|---|
| TextureEntropy | CCI 식 (1), 텍스처 범주(Organic) |
| ContextClusterCount | CCI 식 (1) |
| FreqComplexity | CCI 식 (1), 텍스처 범주(Directional·Periodic) |
| OrientVariance | CCI 식 (1) |
| LocalVariance | 텍스처 범주(Smooth·Organic) |
| OrientEntropy | 텍스처 범주(Directional) |

---

## 0. 먼저 알아야 할 것 — 이름은 같아도 두 계열은 다른 양이다

6개 특징은 **두 파이프라인**에 걸쳐 있고, 세 이름(TextureEntropy·FreqComplexity·Orient*)은 **동명이인**이다.

| 계열 | 단위 | 코드 | 논문 | 역할 |
|---|---|---|---|---|
| **A. CCI 성분** (식 1, Table 2) | 데이터셋 1개 → 스칼라 1개 | `distribution_profiling.py`(패치 특징) → `compute_complexity.py`(집계) | §3.2.1 | 데이터셋 난이도 지표. §4.2–4.3에서 "언제 문맥 배치가 이득인가" 조건 변수 |
| **B. 텍스처 범주 판정** (Table 3) | 64px 패치 1개 → 5범주 라벨 | `utils/background_characterization.py` `classify_patch` | §3.2.3 | ROI의 `background_type` 라벨. **core placement 미사용**, `--min_quality>0`일 때 subtype 매칭 quality proxy에만 입력 |

이름 충돌 때문에 논문 Table 3은 2026-07-29에 **개명**됐다(`aroma_table3_background_descriptor_definitions.md`). 원본 메모의 "텍스처 범주" 열 이름은 개명 전 이름이다.

| 메모(구명) | Table 3 현재 이름 | CCI 쪽 동명 성분과의 차이 |
|---|---|---|
| TextureEntropy(Organic) | **LBPEntropy** | 같은 LBP 식. Table 3=패치값, CCI=패치 **평균** |
| FreqComplexity(Directional·Periodic) | **AutocorrPeak** | 완전히 다른 양. Table 3=자기상관 peak, CCI=FFT 고주파 에너지비의 패치간 **분산** |
| OrientEntropy(Directional) | **GradOrientEntropy** | 둘 다 그라디언트 방향 엔트로피지만 bin·범위·가중 전부 다름(§2.3 참조) |

---

## 1. 공통 재료 — 5개 패치 문맥 특징 (`_extract_context_features`)

정상 이미지를 **64px 절단 격자**로 자르고, 결함 마스크 과반 패치는 제외한 뒤 각 grayscale 패치 p에서 5개 값을 뽑는다. 이 5개가 CCI 성분과 문맥 셀(P33/P66 tertile 키, 예 `"2_2_1_2_0"`)의 공통 원료다.

| # | 특징 | 수식 | 코드 |
|---|---|---|---|
| 1 | `local_variance` | Var(p) — 픽셀 강도 분산 | `np.var(p)` |
| 2 | `edge_density` | mean(√(Sx²+Sy²)) — Sobel 3×3 크기 평균 | `cv2.Sobel` |
| 3 | `texture_entropy` | H(q_LBP) = −Σ q_k log₂ q_k, q = LBP(P=8,R=1,uniform) 10-bin 정규화 히스토그램 | `local_binary_pattern` |
| 4 | `frequency_energy` | Σ_{r>R/4}\|F\|/Σ\|F\|, F = fftshift(fft2(p)), 중심 반경 min(h,w)/4 **밖**(고주파) 크기 비율 | `np.fft.fft2` |
| 5 | `orientation_consistency` | H(q_θ), θ = atan2(Sy,Sx) ∈ [−π,π], 18-bin, **비가중** 히스토그램. 이름과 달리 엔트로피라 **낮을수록 방향 일관** | `np.arctan2` |

Figure 3.2.2-2 라벨도 이 정의 그대로: `texture_entropy (LBP)`, `frequency_energy (HF ratio)`, `orientation_consistency (entropy; low = coherent)`.

---

## 2. 계열 A — CCI 식 (1) 네 성분

### 2.1 식

논문 §3.2.1:

```
CCI = Mean(TextureEntropy, ContextClusterCount, FreqComplexity, OrientVariance)
```

코드 `compute_cci`:

```
CCI = Σ_i w_i · norm_i(raw_i),   w = equal(¼씩),   norm = minmax clip to expected_range
```

표본: 데이터셋 전체 문맥 패치에서 **20,000개 서브샘플**(부족하면 bootstrap 복원추출). 논문의 "20,000 sampled image patches per dataset"와 일치.

### 2.2 성분별 정의

| CCI 성분 | raw 값 (코드) | 원료 특징 | 집계 | minmax 범위 | 의미(논문) |
|---|---|---|---|---|---|
| **TextureEntropy** | `mean(texture_entropy)` | §1-3 LBP 엔트로피 | 패치 **평균** | [0, 8] | texture disorder |
| **ContextClusterCount** | `n_ctx_clusters` | 5특징 z-score 벡터 | GMM(k=1..5) **BIC 최소 k** | [1, 8] | 배경 문맥 군집 수 |
| **FreqComplexity** | `var(frequency_energy)` | §1-4 고주파 에너지비 | 패치간 **분산** | [0, 1] | frequency-domain complexity |
| **OrientVariance** | `var(orientation_consistency)` | §1-5 방향 엔트로피 | 패치간 **분산** | [0, 1.6] | gradient-orientation variability |

핵심: TextureEntropy만 **평균**, Freq/Orient는 **분산**이다. 즉 FreqComplexity·OrientVariance는 "고주파가 많다/방향이 흐트러졌다"가 아니라 **패치마다 그 값이 얼마나 들쭉날쭉한가**(배경 이질성)를 잰다. 논문 제목 "background heterogeneity"와 정합.

ContextClusterCount 세부(`_cluster_context`, `_fit_gmm_bic`): 5특징 z-score → GaussianMixture k=1..max_k(5), n_init=1로 BIC 탐색 → best k를 n_init=3으로 재적합. 반환 k가 성분값. (morphology 군집도 같은 GMM+BIC이지만 그쪽은 MCI 재료.)

### 2.3 Table 2 실측값 해석

| Dataset | CCI | TextureEntropy | Clusters | FreqComplexity | OrientVariance |
|---|---|---|---|---|---|
| AITeX | 0.440 | 2.848 | 5 | 0.0158 | 1.3085 |
| Kolektor | 0.224 | 2.578 | 5 | 0.0002 | 0.0035 |
| Severstal | 0.306 | 3.008 | 5 | 0.0073 | 0.4326 |
| MTD | 0.246 | 3.215 | 5 | 0.0065 | 0.0093 |
| MVTec Leather | 0.200 | 2.981 | 4 | 0.0002 | 0.0005 |

- Clusters는 5/5/5/5/4 → max_k=5 **천장**에 붙어 변별력 없음.
- TextureEntropy 2.58–3.22로 폭 좁음(논문: "varies only modestly").
- 실질 변별은 **OrientVariance·FreqComplexity**(AITeX ≫ Severstal ≫ 나머지). 논문: "structural rather than intensity-based features dominate".
- 단, minmax 범위가 [0,1]·[0,1.6]인데 raw 최대가 0.0158·1.31이므로 FreqComplexity는 정규화 후 거의 0에 수렴 → CCI 차이는 사실상 OrientVariance + TextureEntropy가 끈다.

---

## 3. 계열 B — Table 3 텍스처 범주 4 디스크립터

### 3.1 논문 Table 3 (개명 후) + lazy cascade

| 순서 | Category | 논문 기준 | 코드 실제 판정(`classify_patch`) |
|---|---|---|---|
| 1 | Smooth | LocalVariance ≤ P25 | `np.var(patch) < 100.0` |
| 2 | Directional | GradOrientEntropy ≤ P25 **AND** AutocorrPeak ≤ P25 | grad-orient entropy `< 1.0` (AND 조건 없음) |
| 3 | Periodic | AutocorrPeak ≥ P75 | autocorr peak `> 0.15` |
| 4 | Organic | LBPEntropy ≥ P50 **AND** LocalVariance ≥ P75 | LBP entropy `> 2.5` (단일 조건) |
| 5 | Complex | else | else |

cascade = 위에서부터 **첫 매치 승**. 각 조건은 앞 조건의 부정과 암묵 AND(논문도 이렇게 서술).

### 3.2 디스크립터 수식 (논문 §3.2.3 정의 문단 = 코드 기준으로 작성됨)

공통 연산자: `Entropy(q) = − Σ_k q_k · log₂ q_k`

| 디스크립터 | 수식 | 코드 세부 | 판정 방향 |
|---|---|---|---|
| **LocalVariance** | Var(p) | `np.var(patch)` — §1-1과 동일 양 | 낮음 → Smooth |
| **GradOrientEntropy** | Entropy(q_θ), θ = atan2(Sy,Sx) **mod 180°**, **8 bins × 22.5°**, **크기 가중** | `_compute_gradient_direction_entropy` | 낮음(단일 방향 지배) → Directional. 임계 1.0 bit은 "dot grid(≈1.67)가 directional로 오분류"돼 2.0에서 하향한 **수동 튜닝값** |
| **LBPEntropy** | Entropy(q_LBP), LBP^riu2(P=8,R=1) 10-bin | `_compute_lbp_entropy` — §1-3과 **동일 식** | 높음(미세 패턴 공존) → Organic |
| **AutocorrPeak** | max_{(u,v)≠0} R(u,v)/R(0,0), R = ℱ⁻¹(\|ℱ(p)\|²) (Wiener–Khinchin) | `_compute_autocorrelation_peak`: `ifft2(|fft2|²)`, 원점 정규화 후 `ac[0,0]=0`, max | 높음(비영 lag에서 자기복제) → Periodic |

레퍼런스: Shannon[36], Dalal&Triggs HOG[37](크기 가중+unsigned 방향), Ojala LBP[38], Haralick[39](자기상관 주기성). 번호는 임시.

### 3.3 §1-5 `orientation_consistency`와 GradOrientEntropy 비교 (OrientEntropy 혼동 주의)

| | `orientation_consistency` (CCI 원료) | `GradOrientEntropy` (Table 3) |
|---|---|---|
| 각도 범위 | [−π, π] (방향 부호 유지) | [0°, 180°) (부호 접음) |
| bin | 18 | 8 |
| 가중 | 없음(픽셀 수) | Sobel 크기 가중 |
| 쓰임 | 패치간 Var → OrientVariance | 패치 단독 임계 → Directional |

스트라이프는 두 부호의 그라디언트가 섞이므로 부호 유지 18-bin에서는 2봉, 접은 8-bin에서는 1봉 → 같은 표면이라도 두 값은 다르게 나온다.

---

## 4. 논문 ↔ 구현 불일치 (기록만, 코드 미수정 — 사용자 결정 2026-07-29)

1. **임계가 percentile이 아니라 고정 상수**: 100.0 / 1.0 / 0.15 / 2.5. 코드베이스 어디에도 P25/P50/P75 유도 로직 없음. 논문 "per-dataset percentile cascade" 서술과 불일치. "no hand-set constants" 주장과 긴장 — 단 core placement 경로가 아니라 quality proxy라 헤드라인 주장 직접 훼손은 아님.
2. **Directional의 `AND AutocorrPeak ≤ P25`는 코드에 대응물 없음** — autocorr은 directional 판정 **다음**에 계산됨.
3. **Organic의 `AND LocalVariance ≥ P75`도 단일 조건으로 구현** — 다만 cascade상 Smooth 탈락(var ≥ 100)은 이미 통과했으니 약한 형태로는 암묵 충족.
4. 후속 선택지: (a) 논문을 고정 임계로 정정(최소 비용) / (b) `background_characterization.py`를 percentile로 개조 후 재프로파일링. 미결.

---

## 5. 결함 쪽 — 형태 특징 벡터 6종 (GMM 입력 · MCI 원료)

> 코드: `utils/defect_characterization.py` `DefectCharacterizer.analyze_defect_region`(linearity·solidity·extent·aspect_ratio) + `scripts/distribution_profiling.py:262 _morph_worker`(eccentricity·circularity 추가). 집계: `distribution_profiling.py:876 step5_morphology_clustering`(GMM), `compute_complexity.py:450 compute_mci`.
> 논문: §3.2.1(MCI 서술), §3.2.2 Figure 3.2.2-1, §3.2.3 Table 4/4b + 정의 문단, Figure 3.2.3-1 캡션(6종 명시).

### 5.1 측정 대상

결함 이미지 1장 → 마스크(GT 우선, 없으면 SAM → Otsu 폴백) → **8-연결 성분 중 최대 면적 blob 1개**의 `regionprops`. 6종 전부 이 최대 blob만 기술한다(`defect_bbox`·저장 마스크는 전 성분 — 합성 crop 범위와 측정 대상이 다름, Figure 3.2.2-3 spec 참조).

### 5.2 6종 정의 (`MORPH_FEATURES` 순서)

| # | 특징 | 수식 | 코드 | 범위 | 의미 |
|---|---|---|---|---|---|
| 1 | `linearity` | 1 − λ_min/λ_max, λ = 픽셀 좌표 2×2 공분산 고유값 | `np.cov(coords)` → `eigvalsh` | [0,1) | 1≈선, 0≈원 |
| 2 | `solidity` | Area / Area(convex hull) | `region.solidity` | (0,1] | 낮을수록 경계 요철·오목 |
| 3 | `extent` | Area / Area(최대 blob 자신의 bbox) | `region.extent` | (0,1] | bbox 채움률 |
| 4 | `aspect_ratio` | major/minor — 2차 중심모멘트 등가 타원 축 길이비 | `axis_major_length / axis_minor_length` (minor≈0 → 99.0) | [1,∞) | 세장비. 롱테일 |
| 5 | `eccentricity` | √(1 − (minor/major)²) = **√linearity** | `region.eccentricity` | [0,1) | 등가 타원 이심률 |
| 6 | `circularity` | 4π·Area / Perimeter² | `4*np.pi*region.area/region.perimeter**2` | (0,1] | 1=원 |

**독립 축이 아니다**: 1·4·5는 같은 등가 타원의 함수. linearity = 1 − AR⁻², eccentricity = √linearity → Spearman 1.000(논문 §3.2.3: "Every linearity threshold reduces to an aspect-ratio threshold … Eccentricity … redundant"). 실질 자유도 ≈ 4(세장·solidity·extent·circularity).

### 5.3 GMM 군집 K* (Step 5)

```
X ∈ ℝ^{N×6}  → per-feature min-max [0,1]  → GaussianMixture(k=1..5)  → K* = argmin BIC
```

- `--n_gmm_components` 기본 5. sklearn 없으면 aspect_ratio percentile 3분할 폴백.
- CCI 쪽 `_cluster_context`는 **z-score**, 여기는 **min-max** — 정규화 방식 다름.
- min-max 부작용(Figure 3.2.2-3 실측, severstal): AR max=74.7 극단값 1개가 축을 압축 → medoid AR=2.57(분위 30%)이 입력 0.021, 분산 점유 3.5%(6종 최저). 실제 지배 축은 circularity(31.4%).
- 군집 라벨 `_auto_label`(centroid 휴리스틱: lin>0.7∧AR>5 → linear_scratch / AR>4 → elongated / AR<2∧sol>0.85 → compact_blob / sol<0.65 → irregular / else general)은 **표시용**. 군집 k 자체가 `morph_prior P(k)=n_k/N`와 호환성 행렬 행 인덱스.
- ppt_guide.md:135 "K* = argmax BIC" → 코드는 **argmin BIC**(sklearn `bic()`는 낮을수록 좋음). 표기 정정 필요.

### 5.4 MCI 4성분 (`compute_mci`) — CCI §2와 대칭

```
MCI = Σ w_i · norm_i,  w = equal(¼),  minmax clip
```

| MCI 성분 | raw (코드) | 원료 | 정규화 | 논문 서술(§3.2.1) |
|---|---|---|---|---|
| **Entropy** | H(군집 라벨 분포) = −Σ p_k log₂ p_k | Step 5 라벨 | minmax [0, 4] | "intensity variation"(※ 실체는 군집 크기 균형도) |
| **ValleyCount** | Σ_{6특징} n_valleys — 반전 히스토그램 `find_peaks`(prominence = max(2√(n/bins), 0.1·max)) | 6특징 각 분포 | minmax [0, 41] | distribution multimodality |
| **ClassDiversity** | N_eff = exp(H_ln(defect_type 분포)) (Hill 수) | GT 결함 클래스 라벨 | ln(N_eff)/ln(8) | defect-type diversity |
| **1 − Silhouette** | 1 − silhouette(z-score X, 라벨), ≤5000 샘플 | 6특징 + 라벨 | clamp01 | cluster separability(낮을수록 분리 안 됨 → 복잡) |

ValleyCount 세부: bounded 특징(circularity·eccentricity·extent)은 Sturges bins + 평탄(CV<0.5) 시 prominence 비율 0.3으로 상향. 우편포(p90−p50 > 2·(p50−p10))는 log1p 후 탐지.

주의: 논문의 "intensity variation" 표현과 코드 Entropy(군집 라벨 엔트로피) 불일치 — 픽셀 강도와 무관. ClassDiversity는 6종 벡터가 아닌 **GT 클래스명**을 씀 → MCI 4성분 중 1개는 형태 벡터 외부 정보.

### 5.5 Table 1 실측 MCI

| Dataset | Defects | MCI | CCI |
|---|---|---|---|
| AITeX | 352 | 0.392 | 0.440 |
| Kolektor | 52 | 0.328 | 0.224 |
| Severstal | 3620 | 0.455 | 0.306 |
| MTD | 388 | 0.561 | 0.246 |
| MVTec Leather | 92 | 0.595 | 0.200 |

MCI↔CCI 역상관 경향(Leather: MCI 최고·CCI 최저). Defects 수는 Table 4b.

### 5.6 Table 4 subtype (참고 — GMM과 별개 경로)

6종 중 **aspect_ratio·solidity 2개만** 사용, 데이터셋별 P33/P66 cascade: linear_scratch(AR>P66) → compact_blob(AR<P33 ∧ sol>P66) → irregular(sol<P33) → general. 중간 tertile < 0.15σ면 고정 폴백(Kolektor solidity 0.70/0.90). 용도 = elastic-warping 변형 파라미터. 군집 k(GMM)와 subtype(Table 4)은 **다른 라벨**.

---

---

## 6. 한 줄 요약

- **CCI 4성분**: 5개 패치 특징 중 3개(LBP 엔트로피, 고주파 비율, 방향 엔트로피)를 20k 패치에서 **평균/분산/분산**으로 집계 + GMM-BIC 군집 수. equal-weight minmax 평균.
- **Table 3 4디스크립터**: 패치 단독 **LocalVariance·GradOrientEntropy·AutocorrPeak·LBPEntropy**를 고정 임계 lazy cascade로 5범주화. 코드는 `utils/background_characterization.py`, 소비처는 quality proxy뿐.
- 메모의 TextureEntropy/FreqComplexity/OrientEntropy는 **계열에 따라 다른 양**이다 — 논문 Table 3는 이미 LBPEntropy/AutocorrPeak/GradOrientEntropy로 개명됨.
- **결함 형태 6종**: 최대 blob의 linearity·solidity·extent·aspect_ratio·eccentricity·circularity. min-max → GMM(k≤5) argmin BIC = K*. 1·4·5는 같은 타원 → 실질 4축. MCI = 라벨 엔트로피·valley 수·클래스 N_eff·(1−silhouette) 평균.
