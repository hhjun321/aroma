# AROMA 특징(feature) 6종 — 수식·구현·논문 대응

> 원본 메모(특징 → 등장 위치)를 코드·논문으로 조사해 확장. 조사일 2026-10-30, 졸업논문 기준 갱신 2026-11-03.
> 코드 기준: `scripts/distribution_profiling.py:213 _extract_context_features`, `scripts/aroma/compute_complexity.py:580 compute_cci` / `:450 compute_mci`, `utils/background_characterization.py:93 classify_patch`.
> 논문 기준 (2종):
> - **졸업논문** `GRADUATION/20261102_졸업논문.md` 제5장 제2절 — 식 (5-1)~(5-8), 표 5-1~5-3, 그림 5-2~5-5. **스터디 1순위 문서.**
> - 영문 Article `Article/text/section3_2.txt` §3.2.1 식(1)·Table 2, §3.2.3 Table 3(개명본)·Table 4.

| 특징 | 등장 위치 (졸업논문 기준) |
|---|---|
| TextureEntropy | CCI 식 (5-1) · 표 5-2 Organic 조건 식 (5-6) |
| ContextClusterCount | CCI 식 (5-1) |
| FreqComplexity | CCI 식 (5-1) · 표 5-2 Directional·Periodic 조건 식 (5-4)(5-5) |
| OrientVariance | CCI 식 (5-1) |
| LocalVariance | 표 5-2 Smooth·Organic 조건 식 (5-3)(5-6) |
| OrientationEntropy | 표 5-2 Directional 조건 식 (5-4) |

---

## 0. 먼저 알아야 할 것 — 이름은 같아도 두 계열은 다른 양이다

6개 특징은 **두 파이프라인**에 걸쳐 있고, 세 이름(TextureEntropy·FreqComplexity·Orient*)은 **동명이인**이다.

| 계열 | 단위 | 코드 | 졸업논문 | 역할 |
|---|---|---|---|---|
| **A. CCI 성분** (식 5-1, 표 5-1) | 데이터셋 1개 → 스칼라 1개 | `distribution_profiling.py`(패치 특징) → `compute_complexity.py`(집계) | 제5장 2절 도입부 + 1항 | 데이터셋 난이도 지표. 5.3절에서 "언제 문맥 배치가 이득인가"를 조건화하는 축 (AITeX +4.96 → Severstal +1.06 → MTD −0.25 → Leather −4.91 pp 순서 = CCI 내림차순) |
| **B. 텍스처 범주 판정** (표 5-2, 식 5-3~5-6) | 64px 패치 1개 → 5범주 라벨 | `utils/background_characterization.py` `classify_patch` | 제5장 2절 2항 | ROI의 `background_type` 라벨. 논문 표현 "해석·보고용 라벨"(1항 말미). **core placement 미사용**, `--min_quality>0`일 때 subtype 매칭 quality proxy에만 입력 |

### 0.1 이름 3중 대응표 (메모 → 영문 Article → 졸업논문)

영문 Article Table 3는 2026-07-29에 CCI와의 충돌을 피하려고 **개명**됐다(`aroma_table3_background_descriptor_definitions.md`). **졸업논문 표 5-2는 개명 전 이름을 그대로 쓰되**, 식 (5-6) 아래 문단에서 "네 디스크립터는 모두 패치 하나의 값 … CCI 동명 성분은 데이터셋 전체 집계"라고 명시해 혼동을 산문으로 해소한다. 스터디 시 **졸업논문 이름으로 말하고, 계열(A/B)을 함께 붙여** 구분한다.

| 메모(구명) | 졸업논문 표 5-2 | 영문 Article Table 3 | CCI(계열 A) 동명 성분과의 차이 |
|---|---|---|---|
| LocalVariance | LocalVariance | LocalVariance | CCI에 동명 없음. 5개 패치 특징 §1-1과 같은 양 |
| OrientEntropy(Directional) | **OrientationEntropy** | GradOrientEntropy | 둘 다 그라디언트 방향 엔트로피지만 bin·범위·가중 전부 다름(§3.3) |
| FreqComplexity(Directional·Periodic) | **FreqComplexity** | AutocorrPeak | 완전히 다른 양. 표 5-2 = 자기상관 peak, CCI = FFT 고주파 에너지비의 패치간 **분산**. 졸업논문 자체가 "명칭과 달리 … 반복 패턴의 강도"라고 인정 |
| TextureEntropy(Organic) | **TextureEntropy** | LBPEntropy | 같은 LBP 식. 표 5-2 = 패치값, CCI = 패치 **평균** (졸업논문 (4) 아래 문단에 명시) |

### 0.2 개념 모델 — 디스크립터 · filter · 라벨 · 벡터 (스터디 합의 2026-11-03)

**용어 고정.**

| 용어 | 정의 | 예 |
|---|---|---|
| 국소 연산자 | 픽셀 단위 응답. 신호처리 의미의 filter | Sobel, LBP 코드, FFT |
| **디스크립터** | 영역(64px 패치 / 최대 blob) 1개 → 스칼라 1개. 판단 없음, **자(ruler)** | Var(p), Entropy(q), AutocorrPeak, solidity, circularity |
| **filter** | 디스크립터 + 부등호 + 임계 + **검사 순서**를 모두 포함한 판정식 | `LocalVariance ≤ P25` (5-3), cascade 전체 |
| 라벨 | filter 출력 | Smooth…Complex, linear_scratch…general |
| 벡터 | 디스크립터 **raw 값**을 묶은 것. 라벨 아님 | 배경 5차원, 결함 6차원 |

디스크립터 혼자는 구분 못 한다. 같은 축이 반대 방향으로 두 범주에 들어간다.

| 디스크립터 | 낮을 때 | 높을 때 |
|---|---|---|
| LocalVariance | Smooth (≤P25) | Organic 조건 중 하나 (≥P75) |
| FreqComplexity | Directional 조건 중 하나 (≤P25) | Periodic (≥P75) |
| OrientationEntropy | Directional (≤P25) | — |
| TextureEntropy | — | Organic (≥P50) |

디스크립터 1개 = 축 1개. 범주 = 축 위 구간 + cascade 순서(Organic은 "Smooth·Directional·Periodic 아님" 암묵 포함).

**흐름은 두 갈래. 벡터 갈래와 라벨 갈래가 갈라진다.**

```
패치 / blob
  │
  └─ 디스크립터 raw 값
        ├─ [A] 벡터로 묶음 ─→ tertile 맥락 셀 · GMM K* · CCI/MCI   ← 배치·지수에 실제 쓰임
        └─ [B] filter(부등호·임계·순서) ─→ 라벨 ─→ 메타데이터 · warping 파라미터  ← 옆길
```

| 벡터 | 내용물 | 라벨 포함? | 소비처 |
|---|---|---|---|
| 배경 5차원 (§1) | local_variance, edge_density, texture_entropy, frequency_energy, orientation_consistency **raw** | ✗ | tertile → 맥락 셀 c (식 5-10 ctx_prior), CCI 집계 (5-1) |
| 결함 6차원 (§5.2) | linearity, solidity, extent, AR, eccentricity, circularity **raw** | ✗ | min-max → GMM K* (5-7), morph_prior P(k), MCI (5-2) |

Smooth/Directional 라벨은 어느 벡터에도 안 들어간다 — `roi_metadata.background_type` 문자열로만 저장. subtype 라벨도 벡터 아님 — elastic-warping 파라미터 선택 키.

**틀리기 쉬운 순서.**
- ✗ 측정 → filter → 라벨 → 벡터 → 분석
- ✓ 측정 → **벡터 → 분석**(GMM·셀·호환성). filter → 라벨은 옆길, 분석에 안 들어감

왜 이렇게 설계됐나: 라벨로 묶으면 5범주·4subtype 사람이 그은 구간에 정보가 뭉개진다. "데이터셋별 고정 상수 배제"(4항) 주장은 raw 벡터 위에서 **데이터가 구간을 긋기**(GMM·tertile) 때문에 성립. filter 라벨은 사람 구간 → 논문이 "해석·보고용"으로 격리한 이유.

**이 모델이 깨지는 곳.** filter→범주 그림은 표 5-2·표 5-3에서만 성립. CCI는 같은 디스크립터를 20k 패치 평균/분산으로 **집계**할 뿐 범주 없음. GMM은 축만 주고 구간은 데이터가 긋는다 — 어느 특징이 어느 군집 만드는지 사람이 안 정함. 심사장에서 CCI·GMM까지 "filter"로 묶어 말하면 반박 들어옴.

**한 줄.** filter(부등호 포함 수식) 출력 = 라벨, 보고·보조용. 분석·배치에 쓰는 벡터 = filter 거치기 **전** 디스크립터 raw 값.

---

## 1. 공통 재료 — 5개 패치 문맥 특징 (`_extract_context_features`)

정상 이미지를 **64px 절단 격자**로 자르고, 결함 마스크 과반 패치는 제외한 뒤 각 grayscale 패치 p에서 5개 값을 뽑는다. 이 5개가 (a) CCI 성분의 원료이자 (b) 배경 **맥락 셀**(졸업논문 2절 4항, 그림 5-5: 특성별 P33/P66 tertile → 셀 키 예 `"2_2_1_2_0"`)의 원료다.

| # | 특징 | 수식 | 코드 |
|---|---|---|---|
| 1 | `local_variance` | Var(p) — 픽셀 강도 분산 | `np.var(p)` |
| 2 | `edge_density` | mean(√(Sx²+Sy²)) — Sobel 3×3 크기 평균 | `cv2.Sobel` |
| 3 | `texture_entropy` | H(q_LBP) = −Σ q_k log₂ q_k, q = LBP(P=8,R=1,uniform) 10-bin 정규화 히스토그램 | `local_binary_pattern` |
| 4 | `frequency_energy` | Σ_{r>R/4}\|F\|/Σ\|F\|, F = fftshift(fft2(p)), 중심 반경 min(h,w)/4 **밖**(고주파) 크기 비율 | `np.fft.fft2` |
| 5 | `orientation_consistency` | H(q_θ), θ = atan2(Sy,Sx) ∈ [−π,π], 18-bin, **비가중** 히스토그램. 이름과 달리 엔트로피라 **낮을수록 방향 일관** | `np.arctan2` |

졸업논문 그림 5-5("프로파일된 다섯 개 배경 맥락 특성과 P33/P66 tertile 경계")의 다섯 축이 이 5개다. 영문 Figure 3.2.2-2 라벨도 같은 정의: `texture_entropy (LBP)`, `frequency_energy (HF ratio)`, `orientation_consistency (entropy; low = coherent)`.

### 1.1 저장 구조 — 이미지 → row 리스트 (`_context_worker`, `:355`)

- 격자는 **비중첩**, `h//64 × w//64` 개. 나머지 픽셀 절단. 예: Severstal 1600×256 → 25×4 = 100 패치. 결함 이미지는 마스크 과반(>0.5) 패치 스킵.
- 패치당 row 1개. 내용 = **raw float 5개 + 메타**: `{local_variance, edge_density, texture_entropy, frequency_energy, orientation_consistency, image_id, image_type(good|defect), patch_xy:"128_64", image_w, image_h}`.
- 셀 코드(`2_2_1_2_0`)는 **저장 안 함**. P33/P66은 데이터셋 전 row 풀에서 산출(`_compute_bin_edges`, `:497`)되므로 추출 시점엔 정할 수 없다. 집계 때 `_context_cell_key(row, bin_edges)`(`:480`)로 그때그때 문자열 생성.
- bin 규칙: `np.searchsorted(edges, val, side="right")` → `0: val < P33`, `1: P33 ≤ val < P66`, `2: val ≥ P66`. 자리 순서 = `CONTEXT_FEATURES` 순서(①~⑤). 5번째 자리는 엔트로피라 **0 = 방향 일관**, 2 = 흐트러짐(뒤집혀 읽힘). 3⁵ = 243 셀 가능, 실제 등장 셀만 ctx_prior 열. 퇴화 경계(P66 ≤ P33)면 그 자리 전부 0 → 축 하나 죽음(Leather 류 균질 표면).
- 셀 코드 쓰는 단위 둘: **패치 단위** 카운트(P_clean(c), P_def(k,c) → 식 5-10) / **이미지 단위**(이미지 내 패치 raw 평균 → 셀 1개, global/deficit 경로). 식 (5-11)~(5-13)의 h_g = 이미지 내 패치 셀 히스토그램, 식 (5-16)의 h_s = ring 패치 셀 히스토그램.

### 1.2 브리지 — §1 패치값이 §2 CCI 성분으로 접히는 길

같은 row가 두 갈래로 소비된다. **CCI는 셀 코드를 쓰지 않는다** — raw 열을 직접 집계.

```
rows (N × 5 raw)
  ├─ P33/P66 → 셀 코드 → P_clean, P_def → ctx_prior (식 5-10)   ← 배치
  └─ 20k 서브샘플 → mean / GMM / var / var → CCI (식 5-1)        ← 지수
```

§1의 5값은 패치 한 장의 자(ruler) 눈금이고, §2는 그 눈금을 데이터셋 전체에서 **평균(수준)·분산(이질성)·군집수(종류 수)**로 접은 것이다. 5 중 3개만 단독 성분이 되고 2개는 군집수 안에서만 산다.

```
패치 특징 (§1, 패치당 1값)          집계 (20k 패치)                 CCI 성분 (§2, 데이터셋당 1값)
─────────────────────────          ──────────────────────          ─────────────────────────
① local_variance          ─┐
② edge_density            ─┤  z-score → GMM k=1..5 → argmin BIC k → ContextClusterCount
③ texture_entropy         ─┤
④ frequency_energy        ─┤
⑤ orientation_consistency ─┘
③ texture_entropy          ──  mean  ─────────────────────────────→ TextureEntropy
④ frequency_energy         ──  var   ─────────────────────────────→ FreqComplexity
⑤ orientation_consistency  ──  var   ─────────────────────────────→ OrientVariance
```

| 포인트 | 설명 |
|---|---|
| 왜 5→4 | ①②는 단독 성분 없음. GMM 입력으로만 기여. ③④⑤는 단독 성분 + GMM 입력 **이중 역할** |
| 왜 평균 vs 분산 | TextureEntropy는 "이 표면이 얼마나 복잡한가"(수준 → 평균). Freq·Orient는 "패치마다 얼마나 다른가"(이질성 → 분산). CCI 정의 "배경 이질성"에 맞추려면 분산 |
| 왜 군집수 | 5차원 공간에서 패치가 몇 덩어리로 뭉치나 = 배경 종류 수. 단 실측 5/5/5/5/4 천장 |

이름 꼬임 주의: `orientation_consistency`(실체는 엔트로피)의 분산 → **OrientVariance**. `frequency_energy`(고주파비)의 분산 → **FreqComplexity**(complexity = 분산이지 에너지 아님).

---

## 2. 계열 A — CCI 식 (5-1) 네 성분

### 2.1 식

졸업논문 식 (5-1):

```
CCI = Mean(TextureEntropy, ContextClusterCount, FreqComplexity, OrientVariance)      (5-1)
```

본문: "네 가지 성분을 min-max 정규화한 뒤 동일 가중으로 평균". 코드 `compute_cci`:

```
CCI = Σ_i w_i · norm_i(raw_i),   w = equal(¼씩),   norm = minmax clip to expected_range
```

표본: 데이터셋 전체 문맥 패치에서 **20,000개 서브샘플**(부족하면 bootstrap 복원추출). 졸업논문 "각 데이터셋에서 추출한 2만 개의 샘플 패치"와 일치.

### 2.2 성분별 정의

| CCI 성분 | raw 값 (코드) | 원료 특징 | 집계 | minmax 범위 | 졸업논문 표현 |
|---|---|---|---|---|---|
| **TextureEntropy** | `mean(texture_entropy)` | §1-3 LBP 엔트로피 | 패치 **평균** | [0, 8] | 텍스처 엔트로피 |
| **ContextClusterCount** | `n_ctx_clusters` | 5특징 z-score 벡터 | GMM(k=1..5) **BIC 최소 k** | [1, 8] | 배경 문맥 군집 수 |
| **FreqComplexity** | `var(frequency_energy)` | §1-4 고주파 에너지비 | 패치간 **분산** | [0, 1] | 주파수 복잡도 |
| **OrientVariance** | `var(orientation_consistency)` | §1-5 방향 엔트로피 | 패치간 **분산** | [0, 1.6] | 방향 분산 |

핵심: TextureEntropy만 **평균**, Freq/Orient는 **분산**이다. 즉 FreqComplexity·OrientVariance는 "고주파가 많다/방향이 흐트러졌다"가 아니라 **패치마다 그 값이 얼마나 들쭉날쭉한가**(배경 이질성)를 잰다. 졸업논문 "배경의 이질성을 나타내는 문맥 복잡도 지수"와 정합.

ContextClusterCount 세부(`_cluster_context`, `_fit_gmm_bic`): 5특징 z-score → GaussianMixture k=1..max_k(5), n_init=1로 BIC 탐색 → best k를 n_init=3으로 재적합. 반환 k가 성분값. (결함 형태 군집도 같은 GMM+BIC이지만 그쪽은 MCI 재료, §5.3.)

### 2.3 실측값 해석 (영문 Table 2 전체값 + 졸업논문 1항 본문 인용값)

졸업논문 표 5-1은 MCI·CCI 합산값만 싣고, 성분값은 1항 본문에 산문으로 인용한다(굵게 = 본문 인용). 영문 Table 2가 전체표.

| Dataset | CCI | TextureEntropy | Clusters | FreqComplexity | OrientVariance |
|---|---|---|---|---|---|
| AITeX | 0.440 | 2.848 | 5 | **0.0158** | **1.3085** |
| Kolektor | 0.224 | 2.578 | 5 | **0.0002** | **0.0035** |
| Severstal | 0.306 | 3.008 | 5 | 0.0073 | 0.4326 |
| MTD | 0.246 | 3.215 | 5 | 0.0065 | 0.0093 |
| MVTec Leather | 0.200 | 2.981 | 4 | **0.0002** | **0.0005** |

- Clusters는 5/5/5/5/4 → max_k=5 **천장**에 붙어 변별력 없음(졸업논문 미언급).
- TextureEntropy **2.58–3.22**로 폭 좁음(졸업논문: "변동이 작아, 배경 구분을 지배하는 것은 밝기 기반 특성이 아니라 구조적 특성").
- 실질 변별은 **OrientVariance·FreqComplexity**(AITeX ≫ Severstal ≫ 나머지). 졸업논문: "방향성 분산과 주파수 복잡도가 데이터셋을 가장 크게 구분".
- 단, minmax 범위가 [0,1]·[0,1.6]인데 raw 최대가 0.0158·1.31이므로 FreqComplexity는 정규화 후 거의 0에 수렴 → CCI 차이는 사실상 OrientVariance + TextureEntropy가 끈다. (논문 서술 "주파수 복잡도가 구분"은 raw 순위 기준이지 CCI 기여도 기준이 아님 — 질문 들어오면 이 구분으로 답한다.)

---

## 3. 계열 B — 표 5-2 텍스처 범주 4 디스크립터

### 3.0 §1·§2와의 관계 — 옆 갈래, AROMA 경로에선 실행 안 됨 (호출처 추적 2026-11-06)

| | §1 패치 (CCI·셀) | 표 5-2 디스크립터 |
|---|---|---|
| 코드 | `distribution_profiling.py` | `utils/background_characterization.py` |
| 패치 출처 | 정상 이미지 64px 절단 격자 전체 | ROI 후보 crop (Otsu+연결요소) |
| 저장 | rows (raw 5값) | `roi_metadata.background_type` 문자열 |
| 수식 공유 | — | LocalVariance = ①, TextureEntropy = ③ 같은 식 **재구현**. OrientationEntropy ≠ ⑤, FreqComplexity ≠ ④ |

호출처 전수:

| 호출 | 어디 | 규칙 |
|---|---|---|
| `classify_patch` (표 5-2 cascade) | `stage1_roi_extraction.py:121,173` **만** — CASDA Stage 1 (제6장) | 고정 임계 4개 |
| AROMA `roi_selection.py:111` | `background_type` = **데이터셋별 상수** 인자(기본 `"directional"`) | 판정 없음 |
| `build_train_jsonl.py:281 _derive_background_type` | §1 셀 bin으로 **제3의 규칙**: ⑤bin==2 → directional / ③or①bin==2 → complex_pattern / ②∧①bin==0 → smooth / else complex_pattern | 3범주, 데이터 유도 |

**AROMA 배치 수식 (5-9)~(5-16) 어디에도 라벨 입력 없음.** 배경 성격은 셀 코드가 대신 짊어짐(243 셀 > 5범주). 유일 접점 `roi_selection.py:509` quality gate는 `--min_quality` 기본 0 = 꺼짐(논문 실험 = legacy 경로). 켜도 `background_type` 상수라 subtype만 거름, 공간 선택 영향 0.

**라벨이 실제 쓰이는 곳 = CASDA(제6장).** `stage3_layout_logic.py:30-91` (SuitabilityEvaluator matching 0.4 + continuity 0.3 + stability 0.2 + gram 0.1, `directional`이면 각도 정렬 분기) · `utils/prompt_generator.py:98` (라벨 → 확산 프롬프트). 표 5-2가 제5장에 있는 이유 = "AROMA가 CASDA의 수작업 `MATCHING_RULES`를 데이터 유도 ctx_prior로 대체했다"는 대비 설명.

한 줄: CASDA = 라벨 → 수작업 매칭표 → 배치. AROMA = 라벨 버리고 셀 → 공존 통계 → 배치.

### 3.1 졸업논문 표 5-2 · 식 (5-3)~(5-6) + lazy cascade

| 순서 | Category | 졸업논문 기준 (식) | 코드 실제 판정(`classify_patch`) |
|---|---|---|---|
| 1 | Smooth | LocalVariance ≤ P25 (5-3) | `np.var(patch) < 100.0` |
| 2 | Directional | OrientationEntropy ≤ P25 **AND** FreqComplexity ≤ P25 (5-4) | grad-orient entropy `< 1.0` (AND 조건 없음) |
| 3 | Periodic | FreqComplexity ≥ P75 (5-5) | autocorr peak `> 0.15` |
| 4 | Organic | TextureEntropy ≥ P50 **AND** LocalVariance ≥ P75 (5-6) | LBP entropy `> 2.5` (단일 조건) |
| 5 | Complex | else | else |

cascade = 위에서부터 **첫 매치 승**(졸업논문: "식 5-3에서 5-6 순서로 조건을 검사하여 처음 만족하는 범주를 부여"). 각 조건은 앞 조건의 부정과 암묵 AND.

졸업논문 표 5-2의 Example Surface 열(bare metal / textile weave / tiles / leather / boundaries)은 예시일 뿐 "실제 라벨은 백분위 규칙으로만 결정"이라고 명시.

### 3.2 디스크립터 수식 (졸업논문 2항 (1)~(4) 산문 = 코드 기준)

졸업논문은 수식 없이 산문 4단락으로 정의한다. 영문 Article §3.2.3 정의 문단이 display 수식본. 공통 연산자: `Entropy(q) = − Σ_k q_k · log₂ q_k`

| 졸업논문 번호·이름 | 수식 | 코드 세부 | 판정 방향 |
|---|---|---|---|
| **(1) 밝기 변동 LocalVariance** | Var(p) | `np.var(patch)` — §1-1과 동일 양 | 낮음 → Smooth |
| **(2) 방향성 OrientationEntropy** | Entropy(q_θ), θ = atan2(Sy,Sx) **mod 180°**, **8 bins × 22.5°**, **크기 가중** | `_compute_gradient_direction_entropy` | 낮음(단일 방향 지배) → Directional. 임계 1.0 bit은 "dot grid(≈1.67)가 directional로 오분류"돼 2.0에서 하향한 **수동 튜닝값** |
| **(3) 주기성 FreqComplexity** | max_{(u,v)≠0} R(u,v)/R(0,0), R = ℱ⁻¹(\|ℱ(p)\|²) (Wiener–Khinchin) | `_compute_autocorrelation_peak`: `ifft2(|fft2|²)`, 원점 정규화 후 `ac[0,0]=0`, max | 높음(비영 lag에서 자기복제) → Periodic. 졸업논문 산문 "원점의 자기상관 값을 기준으로 정규화한 뒤 원점을 제외한 위치에서 가장 높은 상관값" = 이 식 그대로 |
| **(4) 미세 질감 다양성 TextureEntropy** | Entropy(q_LBP), LBP^riu2(P=8,R=1) 10-bin | `_compute_lbp_entropy` — §1-3과 **동일 식** | 높음(미세 패턴 공존) → Organic |

레퍼런스(영문 Article만, 졸업논문엔 없음): Shannon[36], Dalal&Triggs HOG[37](크기 가중+unsigned 방향), Ojala LBP[38], Haralick[39](자기상관 주기성). 번호는 임시.

### 3.3 §1-5 `orientation_consistency`와 OrientationEntropy 비교 (OrientEntropy 혼동 주의)

| | `orientation_consistency` (CCI 원료, 맥락 셀 축) | `OrientationEntropy` (표 5-2) |
|---|---|---|
| 각도 범위 | [−π, π] (방향 부호 유지) | [0°, 180°) (부호 접음) |
| bin | 18 | 8 |
| 가중 | 없음(픽셀 수) | Sobel 크기 가중 |
| 쓰임 | 패치간 Var → OrientVariance; tertile → 맥락 셀 | 패치 단독 임계 → Directional |

스트라이프는 두 부호의 그라디언트가 섞이므로 부호 유지 18-bin에서는 2봉, 접은 8-bin에서는 1봉 → 같은 표면이라도 두 값은 다르게 나온다.

---

## 4. 논문 ↔ 구현 불일치 (기록만, 코드 미수정 — 사용자 결정 2026-07-29, 졸업논문 20261102에도 그대로 잔존)

1. **임계가 percentile이 아니라 고정 상수**: 100.0 / 1.0 / 0.15 / 2.5. 코드베이스 어디에도 P25/P50/P75 유도 로직 없음. 졸업논문은 한 발 더 나가 "P25, P50, P75는 … **데이터셋마다 자동으로 계산된다** … 동일한 규칙이 금속·직물·가죽 데이터셋에 수정 없이 적용"이라고 서술 → 심사 질문 1순위 리스크. 단 같은 절에서 표 5-2를 "해석·보고용 라벨"로 못박았고 core placement 경로(식 5-9~5-16)가 아니므로 "데이터셋별 고정 상수 배제"(4항) 헤드라인 주장은 직접 훼손되지 않는다.
2. **Directional의 `AND FreqComplexity ≤ P25`(식 5-4)는 코드에 대응물 없음** — autocorr은 directional 판정 **다음**에 계산됨.
3. **Organic의 `AND LocalVariance ≥ P75`(식 5-6)도 단일 조건으로 구현** — 다만 cascade상 Smooth 탈락(var ≥ 100)은 이미 통과했으니 약한 형태로는 암묵 충족.
4. 후속 선택지: (a) 논문을 고정 임계로 정정(최소 비용) / (b) `background_characterization.py`를 percentile로 개조 후 재프로파일링. 미결. 심사 전 (a) 권장.
5. **(2026-11-06 추가) 졸업논문 4항 "추출된 후보 영역에는 2항의 기준(표 5-2)에 따라 배경 텍스처 범주 라벨을 부여한다" — AROMA 코드에 대응물 없음.** `classify_patch` 호출처는 CASDA `stage1_roi_extraction.py`뿐(§3.0). 1~3보다 상위 리스크("안 돈다" 수준). 방어선 = 1항 말미 "해석·보고용 라벨". 정정 시 4항 문장을 "CASDA(제6장)에서는 … AROMA는 셀 코드로 대체" 로 바꾸는 게 최소 비용.
6. **(2026-11-06 추가) 졸업논문 3항 "하위유형 라벨은 합성 시 elastic-warping 변형 파라미터 선택에 사용" — AROMA 코드에 elastic 0건.** `elastic` 은 CASDA `stage2_defect_seed_generation.py`·`stage4_diffusion_synthesis.py`(제6장)에만 존재. copy-paste 주 실험에서 subtype이 실제 닿는 곳 = 단일클래스 셋 class_fit (5-13) 의 클래스 축 대체(`clean_bg_selection.py:679`) + 기본 꺼진 quality gate. "ROI 순위(5-9, 5-10)에 쓰이지 않는다"는 참이나 "(5-13)에는 쓰인다(단일클래스)"가 빠짐. 또 `roi_selection.py:380 subtype_mode` 기본 `fixed`(CASDA 상수) — 논문 정본 percentile은 `--subtype_mode percentile` 명시 전달 시에만(step3 가이드).

**해소된 불일치(갱신 2026-11-03)**: 영문 Article §3.2.1의 MCI "intensity variation" 표현은 졸업논문에서 "형태 군집 라벨 분포의 엔트로피(Entropy)"로 바로잡혔다(§5.4). ppt_guide "argmax BIC"도 argmin으로 정정 완료(§5.3).

---

## 5. 결함 쪽 — 형태 특징 벡터 6종 (GMM 입력 · MCI 원료)

> 코드: `utils/defect_characterization.py` `DefectCharacterizer.analyze_defect_region`(linearity·solidity·extent·aspect_ratio) + `scripts/distribution_profiling.py:262 _morph_worker`(eccentricity·circularity 추가). 집계: `distribution_profiling.py:876 step5_morphology_clustering`(GMM), `compute_complexity.py:450 compute_mci`.
> 졸업논문: 2절 도입부 식 (5-2) MCI, 3항 "결함 형태 군집화 (GMM·BIC)" 식 (5-7)(5-8), 그림 5-3, 표 5-3 + 그림 5-4.

### 5.1 측정 대상

결함 이미지 1장 → 마스크(코드: GT 우선, 없으면 SAM → Otsu 폴백; 졸업논문은 "SAM[57]을 활용하여 마스크를 추출"로 단순화) → **8-연결 성분 중 최대 면적 blob 1개**의 `regionprops`(졸업논문: "추출된 결함 마스크 중 면적이 가장 큰 성분을 대상으로"). 6종 전부 이 최대 blob만 기술한다(`defect_bbox`·저장 마스크는 전 성분 — 합성 crop 범위와 측정 대상이 다름).

### 5.2 6종 정의 (`MORPH_FEATURES` 순서 = 졸업논문 나열 순서)

| # | 졸업논문 이름 | 수식 | 코드 | 범위 | 의미 |
|---|---|---|---|---|---|
| 1 | 선형도 `linearity` | 1 − λ_min/λ_max, λ = 픽셀 좌표 2×2 공분산 고유값 | `np.cov(coords)` → `eigvalsh` | [0,1) | 1≈선, 0≈원 |
| 2 | 솔리디티 `solidity` | Area / Area(convex hull) | `region.solidity` | (0,1] | 낮을수록 경계 요철·오목 |
| 3 | 점유율 `extent` | Area / Area(최대 blob 자신의 bbox) | `region.extent` | (0,1] | bbox 채움률 |
| 4 | 종횡비 `aspect_ratio` | major/minor — 2차 중심모멘트 등가 타원 축 길이비 | `axis_major_length / axis_minor_length` (minor≈0 → 99.0) | [1,∞) | 세장비. 롱테일 |
| 5 | 이심률 `eccentricity` | √(1 − (minor/major)²) = **√linearity** | `region.eccentricity` | [0,1) | 등가 타원 이심률 |
| 6 | 원형도 `circularity` | 4π·Area / Perimeter² | `4*np.pi*region.area/region.perimeter**2` | (0,1] | 1=원 |

**독립 축이 아니다**: 1·4·5는 같은 등가 타원의 함수. linearity = 1 − AR⁻², eccentricity = √linearity → Spearman 1.000(영문 §3.2.3: "Every linearity threshold reduces to an aspect-ratio threshold … Eccentricity … redundant"). 실질 자유도 ≈ 4(세장·solidity·extent·circularity). 졸업논문은 이 중복을 언급하지 않고 "신장도(elongation)를 중심으로 연속적인 분포"라고만 서술 — 질문 대비.

### 5.3 GMM 군집 K* (Step 5) — 졸업논문 식 (5-7)(5-8)

```
X ∈ ℝ^{N×6}  → per-feature min-max [0,1]  → GaussianMixture(k=1..5)  → K* = argmin_k BIC(k)      (5-7)
BIC(k) = −2 ln L̂_k + p_k ln n                                                                      (5-8)
```

- 졸업논문: "Min-Max 정규화 … 군집 수를 1~5개로 설정한 GMM … BIC가 최소가 되는 군집 수로 자동 결정 … 별도의 수동 임계값 설정이 필요하지 않다." 코드와 일치.
- `--n_gmm_components` 기본 5. sklearn 없으면 aspect_ratio percentile 3분할 폴백.
- CCI 쪽 `_cluster_context`는 **z-score**, 여기는 **min-max** — 정규화 방식 다름(졸업논문 미언급).
- min-max 부작용(영문 Figure 3.2.2-3 실측, severstal): AR max=74.7 극단값 1개가 축을 압축 → medoid AR=2.57(분위 30%)이 입력 0.021, 분산 점유 3.5%(6종 최저). 실제 지배 축은 circularity(31.4%).
- 군집 라벨 `_auto_label`(centroid 휴리스틱: lin>0.7∧AR>5 → linear_scratch / AR>4 → elongated / AR<2∧sol>0.85 → compact_blob / sol<0.65 → irregular / else general)은 **표시용**. 군집 k 자체가 `morph_prior P(k)=n_k/N`(졸업논문 4항: "해당 형태 클러스터가 차지하는 비율")와 호환성 행렬 `ctx_prior(k,c)`(식 5-10) 행 인덱스.
- 그림 5-3(Severstal·AITeX 형태 클러스터, 별표 = 클러스터 중심)이 이 결과.
- argmax/argmin: 졸업논문 (5-7)·ppt_guide_21:202·308 모두 **argmin** — 과거 ppt 표기 오류는 정정 완료.

### 5.4 MCI 4성분 (`compute_mci`) — 졸업논문 식 (5-2), CCI §2와 대칭

```
MCI = Mean(Entropy, ValleyCount, ClassDiversity, 1 − Silhouette)      (5-2)
    = Σ w_i · norm_i,  w = equal(¼),  minmax clip
```

| MCI 성분 | raw (코드) | 원료 | 정규화 | 졸업논문 표현 |
|---|---|---|---|---|
| **Entropy** | H(군집 라벨 분포) = −Σ p_k log₂ p_k | §5.3 라벨 | minmax [0, 4] | "형태 군집 라벨 분포의 엔트로피" ✓ (영문 Article의 "intensity variation"은 오기였음 — 졸업논문에서 수정) |
| **ValleyCount** | Σ_{6특징} n_valleys — 반전 히스토그램 `find_peaks`(prominence = max(2√(n/bins), 0.1·max)) | 6특징 각 분포 | minmax [0, 41] | "형태 특징 분포의 골짜기 수" |
| **ClassDiversity** | N_eff = exp(H_ln(defect_type 분포)) (Hill 수) | GT 결함 클래스 라벨 | ln(N_eff)/ln(8) | "결함 클래스 다양성" |
| **1 − Silhouette** | 1 − silhouette(z-score X, 라벨), ≤5000 샘플 | 6특징 + 라벨 | clamp01 | "군집 분리도의 보수" |

졸업논문 명시: "Entropy와 Silhouette은 형태 군집 라벨 위에서 계산되며, 군집 수는 본 절 3항의 GMM·BIC로 결정된다." → MCI는 §5.3 K*에 종속.

ValleyCount 세부: bounded 특징(circularity·eccentricity·extent)은 Sturges bins + 평탄(CV<0.5) 시 prominence 비율 0.3으로 상향. 우편포(p90−p50 > 2·(p50−p10))는 log1p 후 탐지.

주의: ClassDiversity는 6종 벡터가 아닌 **GT 클래스명**을 씀 → MCI 4성분 중 1개는 형태 벡터 외부 정보. 단일클래스 데이터셋(AITeX·Kolektor)은 N_eff=1 → 이 성분 0.

### 5.5 표 5-1 실측 MCI·CCI

| Dataset | Domain | Defects | MCI | CCI |
|---|---|---|---|---|
| AITeX | Textile | 352 | 0.392 | 0.440 |
| Kolektor | Metal commutator | 52 | 0.328 | 0.224 |
| Severstal | Steel | 3620 | 0.455 | 0.306 |
| MTD | Magnetic tile | 388 | 0.561 | 0.246 |
| MVTec Leather | Leather | 92 | 0.595 | 0.200 |

졸업논문: "MCI 0.33~0.60, CCI 0.20~0.44의 넓은 복잡도 범위". MCI↔CCI 역상관 경향(Leather: MCI 최고·CCI 최저). Defects 수는 표 5-4(Defect Instances). 그림 5-2 = MCI–CCI 평면 좌표계 → 5.3절 결과 해석의 축.

### 5.6 표 5-3 subtype (참고 — GMM과 별개 경로)

6종 중 **aspect_ratio·solidity 2개만** 사용, 데이터셋별 P33/P66 cascade: linear_scratch(AR>P66) → compact_blob(AR<P33 ∧ sol>P66) → irregular(sol<P33) → general. 중간 tertile < 0.15σ면 고정 폴백(졸업논문: "백분위 구간이 퇴화한 경우(Kolektor의 solidity)에는 안전장치로 0.70/0.90 고정값"). 용도 = elastic-warping 변형 파라미터 선택, **ROI 순위(식 5-9, 5-10)에는 미사용**. 섭동 실험: P33/P66 ±5pp → 라벨 3.1~14.5% 변동, 배치 결과 불변. 그림 5-4 = 두 특성 분포 위 붉은 점선 경계(AR 로그축). 군집 k(GMM)와 subtype(표 5-3)은 **다른 라벨**.

cascade 순서가 결과를 바꾼다(코드 주석 "Order is load-bearing"): AR 9.0·sol 0.65 인 mask는 규칙 1(AR>P66)에 먼저 걸려 linear_scratch — 순서 바꾸면 irregular.

**mask 1장 = 꼬리표 2개, 독립.** row에 `cluster_id`(k)·`morph_label`(k 사후이름)·`defect_subtype`(표 5-3) 셋 다 저장. k는 전체 N장 동시 계산(GMM), subtype은 mask 단독 + 분위 2쌍. 같은 바구니 k 안에 subtype 여러 개 섞이고, 같은 subtype이 여러 k에 흩어짐. `morph_label`과 `defect_subtype`은 **같은 어휘·다른 출처** — `generate_defects.py:2278`은 morph_label 우선.

**실행 모드**: `roi_selection.py:380 subtype_mode` 기본 `fixed` = CASDA 상수(lin>0.85∧AR>5 / AR>5∧lin>0.6 → elongated / AR<2∧sol>0.9 / sol<0.7). 논문 정본 = `percentile` 명시 전달(`colab_execute_new/step3_execute.md:83`).

**소비처 실측 (2026-11-06, copy-paste 주 실험 기준)**:

| 소비처 | 사용 여부 |
|---|---|
| ROI_score / ctx_prior / site_score (5-9, 5-10, 5-16) | ✗ — k가 흐름 |
| class_fit (5-13) | **단일클래스 셋(AITeX·Kolektor)만** — `clean_bg_selection.py:679 _roi_class_axis`가 클래스 축을 `defect_subtype`/`morph_label`로 대체 → bg_score 순위에 영향 |
| quality gate (`roi_selection.py:509`) | `--min_quality` 기본 0 = 꺼짐 |
| ControlNet 프롬프트 (`generate_defects.py:2330`) | ControlNet arm만, 주 실험 밖. 출처는 morph_label 우선 |
| elastic-warping | **AROMA 코드 없음**. CASDA stage2/stage4(제6장) 것 → §4-6 |

---

## 6. 소비처 지도 — 6특징이 졸업논문 4·5항 어디로 흘러가는가

| 산출물 | 원료 | 졸업논문 위치 | 코드 |
|---|---|---|---|
| 맥락 셀 c (tertile 키) | §1 5개 패치 특징 | 4항, 그림 5-5 | `distribution_profiling.py` |
| morph_prior P(k) | §5.3 GMM 군집 | 식 (5-9) 우변 | roi_selection |
| ctx_prior(k,c) ∝ √(P_def·P_clean) | 군집 k × 셀 c 공존 통계 | 식 (5-10), 그림 5-6 | roi_selection |
| ROI_score = ctx_prior + morph_prior | 위 둘 | 식 (5-9), 그림 5-7 | roi_selection |
| bg_score = src_fit + class_fit + size_fit | 셀 히스토그램 교차 ∩. **class_fit의 클래스 축: 다중클래스=GT 클래스, 단일클래스=subtype/morph_label**(§5.6) | 식 (5-11)~(5-15), 그림 5-8 | clean_bg_selection |
| site_score = ∩(h_s, tgt[k]) | ring 셀 히스토그램 | 식 (5-16), 그림 5-9 | generate_defects |
| 품질 게이트 (blur·contrast·brightness·noise 0.3/0.3/0.2/0.2) | 패치 픽셀 통계 — **6특징과 무관** | 4항 "품질 게이트" 문단 | `generate_defects.py:440 _background_quality_score` |
| `background_type` × subtype 매칭(0.3~1.0 표) | §3 표 5-2 라벨 + §5.6 subtype | 졸업논문 **미서술** | `utils/suitability.py MATCHING_RULES` ← `roi_selection.py:249` (`--min_quality>0` 시만) |

핵심: 졸업논문 본문의 "품질 게이트"는 픽셀 통계 4항목이고, 표 5-2 라벨이 들어가는 subtype 매칭은 논문에 안 나온다. 두 "품질"을 섞지 말 것. §3 범주 라벨의 논문상 역할은 1항 말미 "해석·보고용"이 전부.

**k vs subtype — 주 흐름은 k.** 결함 쪽에서 (5-9)(5-10)(5-16)을 흐르는 건 군집 번호 k. subtype은 단일클래스 class_fit과 꺼진 quality gate 두 곁길뿐(§5.6 소비처 표).

**k는 "행 번호".** 값(morph_prior[k], ctx_prior[k][·], tgt[k])은 바구니 k의 mask 전체가 만든 **군집 단위** 양이고, 개별 mask는 자기 번호로 그 행을 꺼내 쓴다. 같은 k의 mask들은 같은 선호 셀 분포를 공유. mask **개별** 정보가 배치에 들어가는 길은 따로: (5-12) src_fit의 p_src(v) = 그 결함 자신의 원본 주변 셀 히스토그램, (5-14) size_fit의 크롭 w×h.

```
배치 입력
  ├─ 군집 단위: ctx_prior 행 k, morph_prior[k]          (식 5-9, 5-10, 5-16)
  └─ mask 단위: p_src(v) 자기 주변, 크롭 크기 w×h        (식 5-12, 5-14)
```

### 6.1 결함 1개가 자리 하나로 가는 길 — mask C (k=3, 선형 긁힘) 추적

```
[0] 재료     k=3, 셀 코드 체계(243), P_clean(c), P_def(3,c)
[1] 어디에    ctx_prior 행 3 + morph_prior[3] → ROI_score → 후보 쌍 순위 → 상위 K + 쿼터
[2] 어느 사진  bg_score = src_fit + class_fit + size_fit → 정상 이미지 g 확정
[3] 어느 자리  site_score = ∩(h_s, tgt[3]) → g 안의 픽셀 위치 s 확정
[4] 게이트    품질 4항목 ≥ 0.7
[5] 합성      copy-paste blend (seamlessClone + Reinhard)
```

**[0] 재료 — 데이터셋 1회.** P_clean(c) = 정상 패치 전부 셀 세기. P_def(3,c) = k=3 바구니 mask 전체의 **원본 주변 패치** 셀 세기. ctx_prior(3,c) ∝ √(P_def·P_clean), 행 max로 나눠 최고 셀 = 1.0. 기하평균 = 결함 주변에도, 정상 배경에도 흔한 셀만 높음. 한쪽만 높으면 √가 깎음(논문 "어느 한쪽 확률만 높은 후보 과선택 방지"). 가상: P_def=0.30, P_clean=0.12 → √0.036 = 0.19.

**[1] 어디에 — ROI_score (5-9).** 후보 = (결함, 셀) 쌍. `ROI_score(C,c) = ctx_prior(3,c) + morph_prior[3]`. morph_prior는 C에겐 상수 → C의 셀 순위는 ctx_prior 행이 결정. morph_prior 역할은 **결함 간** 비교(희소 바구니가 밀리는 건 클러스터별 기본 할당 + 클래스별 하한 쿼터가 막음). 그림 5-6 히트맵 = 이 행렬, 붉은 상자 = 행 max.

**[2] 어느 사진 — bg_score (5-11~5-15).** 후보 정상 이미지 g마다 셀 히스토그램 h_g(g 안 패치들의 셀 분포). ∩(a,b) = Σ_c min(a_c, b_c).

| 적합도 | 비교 대상 | 단위 |
|---|---|---|
| src_fit = ∩(h_g, p_src(v)) | **C 자신** 원본 주변 셀 분포 | mask |
| class_fit = ∩(h_g, p_cls) | C 클래스 전체 주변 셀 분포. 단일클래스면 subtype/morph_label | 클래스 |
| size_fit = min(1, 0.95·W_g/w, 0.95·H_g/h) | 크롭 w×h가 g에 들어가나 | mask |

셋 합산 최대 g 채택. 그림 5-8.

**[3] 어느 자리 — site_score (5-16).** g 안 후보 위치 s마다 크롭 둘레 **ring** 패치들의 셀 히스토그램 h_s. tgt[3] = ctx_prior 행 3. `s* = argmax_s ∩(h_s, tgt[3])` — "긁힘이 원래 앉던 셀 분포"와 "이 자리 둘레 셀 분포"가 가장 겹치는 곳. 평균 아닌 **분포 매칭**이라 흔한 셀 몰빵 차단(ppt_guide_21:298 — footprint→ring은 4/5 악화, 평균→분포 매칭은 5/5 개선). 그림 5-9.

**[4][5]** 품질 게이트(blur·contrast·brightness·noise 0.3/0.3/0.2/0.2 ≥ 0.7) 통과 자리만 → 블렌딩.

| 단계 | 군집 k가 주는 것 | mask C가 주는 것 |
|---|---|---|
| [1] | ctx_prior 행, morph_prior | — |
| [2] | — (class_fit은 클래스) | p_src(v), w×h |
| [3] | tgt[k] | 크롭 크기(ring 범위) |

읽을 때 주의 2개:
1. 논문 ROI_score = 단순 합. canonical 코드는 가중 realism 모드(0.5/0.3/0.2) — 논문의 **의도적 단순화**(메모 `project_paper_324_simplification`). 감사 시 결함 아님.
2. 셀 코드가 [0]~[3] 전부 관통. §1 raw 5값이 배치의 유일한 배경 언어. 표 5-2 라벨 어디에도 없음.

한 줄: k가 "어떤 셀을 좋아하나"(행)를 주고, mask가 "내 원래 주변·내 크기"를 주고, 둘을 셀 히스토그램 겹침으로 맞춰 사진 → 자리 순으로 좁힌다.

---

## 7. 한 줄 요약

- **CCI 4성분 (5-1)**: 5개 패치 특징 중 3개(LBP 엔트로피, 고주파 비율, 방향 엔트로피)를 20k 패치에서 **평균/분산/분산**으로 집계 + GMM-BIC 군집 수. equal-weight minmax 평균. 5.3절 효과 조건화 축.
- **표 5-2 4디스크립터 (5-3~5-6)**: 패치 단독 **LocalVariance·OrientationEntropy·FreqComplexity·TextureEntropy**를 lazy cascade로 5범주화. 논문은 percentile, 코드는 고정 임계(§4). 소비처는 quality proxy뿐, 논문상 "해석·보고용".
- 메모의 TextureEntropy/FreqComplexity/OrientEntropy는 **계열에 따라 다른 량** — 졸업논문은 구명 유지 + 산문 구분, 영문 Article은 LBPEntropy/AutocorrPeak/GradOrientEntropy로 개명.
- **결함 형태 6종**: 최대 blob의 linearity·solidity·extent·aspect_ratio·eccentricity·circularity. min-max → GMM(k≤5) argmin BIC = K* (5-7)(5-8). 1·4·5는 같은 타원 → 실질 4축. MCI (5-2) = 라벨 엔트로피·valley 수·클래스 N_eff·(1−silhouette) 평균.

---

## 8. 스터디 자문 (답은 위 §에 있음)

1. 졸업논문 표 5-2 FreqComplexity와 식 (5-1) FreqComplexity는 왜 같은 이름인데 다른 양인가? 각각 무엇을 재는가? (§0.1, §2.2, §3.2)
2. CCI 실측에서 FreqComplexity raw 최대 0.0158은 minmax [0,1] 뒤 얼마인가? CCI 차이를 실제로 끄는 성분은? (§2.3)
3. 식 (5-4) Directional의 AND 조건이 코드에서 성립하지 않는 이유를 cascade 순서로 설명하라. (§3.1, §4-2)
4. 심사위원이 "P25/P50/P75가 데이터셋마다 자동 계산된다"의 근거를 물으면? (§4-1)
5. linearity·aspect_ratio·eccentricity의 수학적 관계와 GMM 입력 실질 자유도는? (§5.2)
6. MCI의 Entropy는 무엇의 엔트로피인가? 영문 Article과 졸업논문 표현 차이는? (§5.4)
7. 표 5-2 라벨과 4항 품질 게이트는 같은 "품질"인가? 각각 어디서 쓰이는가? (§6)
8. "디스크립터 = filter"가 틀린 이유 한 줄. filter의 정확한 범위는? Smooth 라벨은 어느 벡터에 들어가는가? (§0.2)
9. 셀 코드 `2_2_1_2_0`의 자리 순서와 0/1/2 뜻은? 5번째 자리 0이 "일관"인 이유는? 저장 시점에 셀 코드가 없는 이유는? (§1.1)
10. CCI 검산: Severstal 성분 raw(3.008 / 5 / 0.0073 / 0.4326) → 0.306이 나오는 계산을 써라. 어느 성분이 사실상 죽어 있나? (§2.3, §1.2)
11. 표 5-2 cascade가 AROMA 코드에서 실행되는가? `background_type`은 어디서 오나? 라벨이 실제 쓰이는 장은? (§3.0)
12. mask 1장에서 나오는 것은 실수 몇 개이고 정수 k는 언제 생기나? K*=5일 때 "k1~k5"는 각각 무엇으로 표현되나? (§5.2, §5.3)
13. k와 subtype이 독립인 이유, 같은 어휘(`linear_scratch`)가 두 출처에서 나오는 문제, 주 실험에서 subtype이 실제 닿는 곳은? (§5.6)
14. §6 지도에서 k는 mask 정보인가 군집 정보인가? mask 개별 정보가 배치에 들어가는 식은? (§6, §6.1)

---

## 9. 스터디 진행 기록

| 일자 | 통과 | 다음 |
|---|---|---|
| 2026-11-03 | §0 개념 모델 합의(디스크립터/filter/라벨/벡터) | §1 |
| 2026-11-06 | §1 패치·셀 구조, §1.2 브리지, §2 CCI 검산, §3.0 표 5-2 호출처, §5 결함 6종·k·K*·MCI·subtype, §6 소비처 지도·6.1 추적 | §6.1 [2]~[3] 수식 세부(5-11~5-16) 또는 §4 불일치 처리 방향 결정 |
