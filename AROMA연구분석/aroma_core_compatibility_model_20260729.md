# AROMA 핵심 — 대칭 호환성 모델 `ctx_prior(k, c)` 정리

> **목적**: AROMA의 기여가 집약된 `ctx_prior(k, c) ∝ √(P_def(k, c) · P_clean(c))`가 **무엇을 세어 어떻게 확률이 되는지**를 코드·실측 기준으로 정리한다. 논문 §3.2.2·§3.2.4 서술의 근거 문서이며, 설명 시 이 문서를 중심으로 삼는다.
>
> **작성**: 2026-07-29. 코드 = `scripts/distribution_profiling.py`, `scripts/aroma/generate_defects.py`, `scripts/aroma/roi_selection.py`. 실측 = `D:/project/aroma_dataset/profiling/profiling/<ds>/`(로컬 미러, Colab Drive `sym_final/profiling`과 동일 버전 확인됨).
>
> **개정 2026-07-31**: §4-1~§4-4(`P_def` 관측 범위·support 실측·원 카운트·**cluster 무차별성**), **§4-5 개선 제안**(`P_def` 계수 범위를 결함 국소로 — 미구현), §5 본문(교환 대칭·기하평균 선택 근거), §5-1~§5-7(역할 분담·재순위 실측·ε 발동 규모·기여 균등 검증·효과 크기 데이터셋 의존·**`S_k` 비대칭과 neutral 0.5 편향**), **§5-8 개선 제안**(`P_clean` 곱셈 형태 재검토 — 미구현), §6-1(void 게이트 fail-open), §6-2(배경 이미지 선택은 히스토그램 교집합), **§7 신설**(legacy `matrix` vs `matrix_symmetric` 소비 경로 분기) 추가. 정직성 11~19 추가. 이에 따라 구 §7→§8, §8→§9, §9→§10.

---

## 0. 한 줄 요약

`ctx_prior(k, c)`는 **결함 형태 군집 `k`**(결함 영역 단위)와 **배경 context cell `c`**(64px 패치 단위)의 **공출현 빈도**를, 두 방향에서 대칭으로 본 기하평균이다. 두 인자 모두 수작업 상수가 아니라 데이터셋 자체 패치 계수에서 나온다.

---

## 1. 먼저 구분해야 하는 라벨 5종

파이프라인에는 결함 쪽 3종·배경 쪽 2종의 라벨이 있고 **어휘가 겹쳐 혼동을 부른다**. `ctx_prior`가 쓰는 것은 굵게 표시한 두 축뿐이다.

### 결함 쪽

| 기호 | 정체 | 단위 | severstal 값 | 소비처 |
|---|---|---|---|---|
| `class_key` / `defect_type` | **GT 결함 클래스**(주석) | (class, 이미지) 쌍 | class1~4 | multi-class YOLO 라벨, `class_floor` |
| **`k` / `cluster_id`** | **GMM/BIC 형태 군집**(비지도) | **결함 영역(bbox)** | 0~4 (K=5) | `morph_prior`, **compat matrix 행** |
| `defect_subtype` | Table 4 규칙 출력 | 결함 영역 | linear_scratch / compact_blob / irregular / general | `quality_score` |

### 배경 쪽

| 기호 | 정체 | 단위 | 개수 | 소비처 |
|---|---|---|---|---|
| **`c` / `cell_key`** | **context feature tertile 코드** | **64px 패치** | 3⁵ = 243 | **compat matrix 열** |
| `background_type` | Table 3 텍스처 범주 | ROI | 5 (smooth/directional/periodic/organic/complex) | `quality_score` |

⚠ **주의 2건**
- `background_type`을 만드는 `BackgroundAnalyzer`는 **현행 파이프라인에서 실행되지 않는다**(호출부 `stage1_roi_extraction.py`가 `colab_execute_new/`에서 미참조, `roi_metadata` 산출물 부재). `roi_selection.py`는 CLI 기본값 `"directional"` 상수를 쓴다.
- cluster 라벨(`_auto_label`)이 `linear_scratch`·`elongated` 등 Table 4 subtype과 **같은 어휘**를 쓴다. 값이 겹쳐 보여도 별개 분류계다.

---

## 2. `k` — 결함 형태 군집

### 2-1. 대상 단위 = 결함 영역(bbox), 64px 패치 아님

`_morph_worker`(`distribution_profiling.py:296-349`):

```python
metrics = char.analyze_defect_region(mask)          # 6종 특징 ← 최대면적 성분 1개
bx, by, bw, bh = cv2.boundingRect(binary)           # bbox ← 전 연결성분
return {"image_id": f"{defect_type}_{image_path.stem}", ..., "defect_bbox": f"{bx},{by},{bw},{bh}"}
```

- **행 1개 = (defect_type, 이미지) 1쌍.** 한 이미지에 두 클래스 결함이 있으면 두 행이 되도록 `image_id`에 defect_type을 접두한다(단 severstal 실측 중복은 0 — §2-5).
- 검산: severstal 3,620행 = class1 477 + class2 111 + class3 2,748 + class4 284.

⚠ **측정 대상 불일치 (load-bearing)**: 형태 특징 **6종 전부**가 `analyze_defect_region`(`utils/defect_characterization.py:45` `region = max(props, key=lambda r: r.area)`)과 그 뒤 regionprops 블록(`:301-309`)을 통해 **최대면적 연결성분 1개**만 기술한다. 반면 `defect_bbox`와 저장되는 마스크 PNG는 **전 성분**을 포함한다(코드 주석 `:313-315`가 의도 명시 — 합성 crop이 결함 전체를 담도록).

⇒ **`k`는 최대 blob의 형태로 결정되지만, 합성 시 붙는 crop은 모든 blob을 담는다.** 다중 성분이 흔한 데이터셋에서 이 괴리가 커진다(§2-5).

### 2-2. data-driven인 지점 — 군집 수까지 데이터가 정한다

| 요소 | 근거 |
|---|---|
| 군집 수 `K` | `_fit_gmm_bic`(`:463-474`)가 `k ∈ [1, n_gmm_max]`를 순회하며 **BIC 최소** 선택 |
| 군집 경계 | `GaussianMixture(n_components=k, random_state=42, n_init=3)`를 표준화 행렬 `X_norm`에 적합(`:905`) |
| 입력 특징 | `MORPH_FEATURES`(`:76-78`) 6종 — linearity, solidity, extent, aspect_ratio, eccentricity, circularity |
| 배정 저장 | `cluster_assignments[image_id] = k` |

**BIC가 실제로 데이터셋별로 다른 K를 고른다** — leather만 3, 나머지 4종은 5:

| 데이터셋 | 결함 수 | **K** | 군집 (k: n, P(k), label) |
|---|---|---|---|
| AITeX | 352 | 5 | 0: 119, .338, linear_scratch / 1: 29, .082, compact_blob / 2: 79, .224, linear_scratch / 3: 43, .122, general / 4: 82, .233, compact_blob |
| Kolektor | 52 | 5 | 0: 5, .096, linear_scratch / 1: 16, .308, linear_scratch / 2: 15, .288, general / 3: 14, .269, linear_scratch / 4: 2, .038, general |
| Severstal | 3,620 | 5 | 0: 522, .144, linear_scratch / 1: 859, .237, **elongated** / 2: 861, .238, linear_scratch / 3: 438, .121, compact_blob / 4: 940, .260, general |
| MTD | 388 | 5 | 0: 127, .327, general / 1: 61, .157, compact_blob / 2: 79, .204, compact_blob / 3: 85, .219, linear_scratch / 4: 36, .093, linear_scratch |
| MVTec Leather | 92 | **3** | 0: 22, .239, compact_blob / 1: 46, .500, **elongated** / 2: 24, .261, compact_blob |

`P(k) = n_k / N`이 §3.2.4 `ROI_score`의 `morph_prior`다.

### 2-3. `k` ≠ GT 결함 클래스 (실측 증거)

비지도 군집이므로 GT 클래스와 정렬되지 않는다. **모든 군집이 복수 클래스를 혼재**한다.

**Severstal** (K=5, GT 4 classes):

| k | class1 | class2 | class3 | class4 | 합 | 혼재 |
|---|---|---|---|---|---|---|
| 0 | 49 | 42 | 425 | 6 | 522 | 4 |
| 1 | 131 | 11 | 665 | 52 | 859 | 4 |
| 2 | 16 | 58 | 787 | 0 | 861 | 3 |
| 3 | 78 | 0 | 284 | 76 | 438 | 3 |
| 4 | 203 | 0 | 587 | 150 | 940 | 3 |

**MVTec Leather** (K=3, GT 5 classes):

| k | color | cut | fold | glue | poke | 혼재 |
|---|---|---|---|---|---|---|
| 0 | 8 | 0 | 3 | 6 | 5 | 4 |
| 1 | 9 | 19 | 14 | 4 | 0 | 4 |
| 2 | 2 | 0 | 0 | 9 | 13 | 3 |

논문 Figure 3.2.2-1 캡션의 *"k is a shape cluster, not the defect class"* 가 이 사실을 가리킨다.

### 2-4. ⚠ 알려진 한계 — 특징 중복, min-max 왜곡, 라벨 부분 사영

세 가지가 겹쳐 있다. 서로 다른 문제이므로 분리해 기록한다.

#### (a) `linearity`·`eccentricity`가 실질 중복

실측(4,504건 전량, 기계 정밀도):

```
linearity    ≡ 1 − AR⁻²          max|err| = 4.4e-15
eccentricity ≡ √linearity        max|err| = 3.4e-15
Spearman(linearity, AR) = 1.000  (5종 전부)
```

세 특징이 같은 2차 중심모멘트에서 나온다. 단 **GMM은 순위가 아니라 선형 관계에 반응**하므로(Gaussian/유클리드 기하) 표준화 후 Pearson으로 봐야 한다:

| 쌍 | severstal | aitex | leather |
|---|---|---|---|
| linearity ~ eccentricity | **+0.992** | **+0.973** | **+0.990** |
| linearity ~ aspect_ratio | +0.563 | +0.516 | +0.799 |
| aspect_ratio ~ eccentricity | +0.518 | +0.455 | +0.744 |

⇒ **`linearity`·`eccentricity`만 2중 가중**이다. `lin = 1 − AR⁻²`가 강한 비선형이라 `aspect_ratio`는 Spearman 1.000에도 선형 중복이 아니다. "3중 가중"은 과장.

#### (b) min-max 표준화가 `aspect_ratio`를 죽인다

표준화가 z-score가 아니라 **min-max**다(`:900-902`):

```python
X_norm = (X - X.min(0)) / (X.max(0) - X.min(0) + 1e-6)
```

min-max 후 분산 점유율(= 유클리드 거리 기여):

| 특징 | severstal | aitex | leather |
|---|---|---|---|
| circularity | **31.4%** | 19.4% | 20.4% |
| linearity | 23.0% | 18.3% | 23.0% |
| extent | 18.8% | **22.0%** | 15.9% |
| eccentricity | 13.0% | 9.7% | 18.0% |
| solidity | 10.3% | 16.4% | 11.7% |
| **aspect_ratio** | **3.5%** | 14.3% | 11.0% |

severstal에서 `aspect_ratio`가 **3.5%로 6종 최저**다. Table 4가 유일한 판정 기준으로 쓰는 특징이 군집화에서는 거의 무력하다. 원인은 우편향 + min-max:

| 데이터셋 | AR p50 | p90 | p99 | max | p99/max |
|---|---|---|---|---|---|
| **severstal** | 4.73 | 13.70 | 26.04 | **74.74** | **0.35** |
| aitex | 5.13 | 46.70 | 82.21 | 90.51 | 0.91 |
| mtd | 2.48 | 8.79 | 21.43 | 25.75 | 0.83 |
| leather | 2.45 | 6.05 | 8.63 | 10.35 | 0.83 |
| kolektor | 5.17 | 6.87 | 8.90 | 9.84 | 0.90 |

severstal만 `p99/max = 0.35` — **극단값 1개가 축의 65%를 차지**해 나머지가 하단 1/3에 압축된다. z-score였다면 없을 문제이고, 나머지 4종(0.83~0.91)은 정상이다. **severstal 특유의 이상치 민감성.**

즉 severstal에서 실제 지배 축은 `aspect_ratio`가 아니라 **`circularity`(31.4%)**다.

#### (c) cluster 라벨이 군집의 부분 사영

`_auto_label(centroid)`(`:1405-1418`)은 6종 중 **3종만** 본다:

```python
ar, sol, lin = centroid['aspect_ratio'], centroid['solidity'], centroid['linearity']
if lin > 0.7 and ar > 5.0:   return "linear_scratch"
if ar > 4.0:                 return "elongated"
if ar < 2.0 and sol > 0.85:  return "compact_blob"
if sol < 0.65:               return "irregular"
return "general"
```

`extent`·`eccentricity`·`circularity`는 라벨에 **전혀 들어가지 않고**, `lin`은 `ar`의 함수이므로 실질 독립 축은 `ar`·`sol` **2개**다. ⇒ **6차원으로 군집화하고 2차원으로 이름을 붙인다.**

**AITeX 실례 — `linear_scratch`가 두 번, `compact_blob`이 두 번 나온다:**

| k | n | label | linearity | solidity | extent | aspect_ratio | eccentricity | circularity |
|---|---|---|---|---|---|---|---|---|
| 0 | 119 | linear_scratch | 0.9991 | 0.5642 | 0.3411 | **42.39** | 0.9995 | **0.0509** |
| 1 | 29 | compact_blob | 0.3097 | 0.9361 | 0.8111 | 1.36 | 0.4771 | **1.3113** |
| 2 | 79 | linear_scratch | 0.9678 | 0.6711 | 0.4671 | **8.28** | 0.9836 | **0.2469** |
| 3 | 43 | general | 0.9323 | 0.9700 | 0.9147 | 3.90 | 0.9655 | 0.5014 |
| 4 | 82 | compact_blob | 0.5524 | 0.8884 | 0.7170 | 1.58 | 0.7364 | **0.9622** |

규칙 경로:

```
k=0  ar=42.392 lin=0.999  → 규칙1) lin>0.7 AND ar>5.0   → linear_scratch
k=2  ar= 8.278 lin=0.968  → 규칙1) 동일                  → linear_scratch   ← 중복
k=1  ar= 1.361 sol=0.936  → 규칙3) ar<2.0 AND sol>0.85  → compact_blob
k=4  ar= 1.577 sol=0.888  → 규칙3) 동일                  → compact_blob     ← 중복
```

규칙 1의 `ar > 5.0`에 **상한이 없어** k=0(AR 42.4)과 k=2(AR 8.3)이 함께 통과한다. 두 군집은 실제로 매우 다르다 — AR 5.1배, circularity 4.9배(0.051 vs 0.247). k=0은 극단적으로 길고 얇은 실 형태, k=2는 중간 길이 긁힘이다. **정작 둘을 가장 크게 갈라놓은 `circularity`가 라벨 판정에 안 들어간다.**

⇒ **`k`를 라벨로 지칭하면 안 된다.** AITeX의 "linear_scratch 군집"은 두 개이고 서로 5배 다르다. Figure 3.2.2-1 범례가 `k{n} · label` 형식으로 번호와 라벨을 함께 쓰는 것이 이 점에서 옳다.

별건 TODO: `.claude/.dev_note/aroma_subtype_percentile_thresholds.md` §10.

### 2-5. 데이터셋별 관측 단위 — AITeX vs Severstal 대조

`k`의 대상은 "결함 하나"가 아니라 **한 행**이고, 그 행이 무엇을 뜻하는지는 데이터셋마다 다르다.

| | AITeX | Severstal |
|---|---|---|
| 행 1개 | 결함 포함 **타일** 1장 (256×256, single-class) | (class, 이미지) 1쌍 ≈ 이미지 1장 (1600×256) |
| 행 수 | 352 | 3,620 |
| 클래스 축 | single (`defect_type` = `defect` 뿐) | multi (class1~4) |
| 다중 연결성분 비율 | **6.5%** (23/352) | **27.9~73.5%** (class별) |
| 최대 성분 수 | 4 | **17** |
| 이미지당 context 패치 | 16.0 (min 10 / max 16) | **98.4** (min 5 / max 100) |
| `P_def` support (관측 cell) | **22~60** / 243 | **192~197** / 243 |
| K (BIC) | 5 | 5 |

#### ① 다중클래스 라벨이 있으나 실제 이미지 중복은 0

| | 값 |
|---|---|
| (class, image) 쌍 | 477 + 111 + 2,748 + 284 = **3,620** |
| distinct image stem | **3,620** |
| 2개 이상 클래스에 걸친 이미지 | **0 / 3,620 (0.0%)** |

`image_id = f"{defect_type}_{stem}"`는 다중클래스 대비 안전장치이지만 severstal에서 실제로 발동하지 않는다. `prepare_severstal.py`가 각 이미지를 **하나의 클래스에만** 배정했다.

⚠ **그 흔적으로 참조되지 않는 마스크가 남는다** — `masks/class{n}` 파일 수가 `test/class{n}`보다 많다:

| | test/ | masks/ | 차이 |
|---|---|---|---|
| class1 | 477 | 477 | 0 |
| class2 | 111 | 129 | +18 |
| class3 | 2,748 | 2,798 | +50 |
| class4 | 284 | 430 | +146 |
| **합** | 3,620 | 3,834 | **+214** |

`_find_mask_path`(`:164-175`)가 `masks/{defect_type}/{stem}.png`를 **우선** 조회하므로 형태 특징은 항상 해당 클래스 마스크에서 산출된다(merged `masks/{stem}.png`는 폴백, 미사용). 다만 **다른 클래스에 배정된 이미지의 결함 214건은 형태 통계에 진입하지 않는다.**

#### ② 다중 성분이 흔하고, 형태 특징은 그중 하나만 본다

클래스별 표본 측정(연결성분 수 분포):

| class | 표본 | 성분 1개 | **2개 이상** | 최대 |
|---|---|---|---|---|
| class1 | 200 | 95 | **52.5%** | 16 |
| class2 | 129 | 93 | 27.9% | 4 |
| class3 | 200 | 53 | **73.5%** | 17 |
| class4 | 200 | 82 | **59.0%** | 9 |

§2-1의 측정 대상 불일치가 여기서 실질 문제가 된다. class3은 **73.5%**에서 성분이 여러 개인데:

- **`k` 결정** ← 최대면적 blob 1개의 형태 (6종 특징 전부)
- **합성 crop** ← `defect_bbox` = 전 성분을 감싸는 상자 + 전 성분 마스크 PNG

즉 **군집이 기술하는 대상과 실제로 붙여지는 대상이 다르다.** 최대 blob이 길쭉한 스크래치라 `k`가 `linear_scratch` 군집에 들어가도, 붙는 crop은 스트립에 흩어진 17개 덩어리 전체일 수 있다. AITeX(6.5%)에서는 거의 발생하지 않던 문제다.

#### ③ 그래서 `k`의 해석이 달라진다

AITeX `k`는 대체로 **단일 결함의 형태**를 가리킨다. Severstal `k`는 **최대 blob의 형태**를 가리키며, 그 행이 대표하는 실체(=합성될 crop)는 결함 앙상블이다. §2-3 교차표에서 모든 cluster가 3~4개 GT class를 혼재한 것도 이와 무관하지 않다 — 최대 blob 형태는 클래스 정체성보다 약한 신호다.

#### ④ `ctx_prior` 추정 밀도가 두 배 이상 다르다

이미지당 context 패치가 AITeX 16개(256×256 ÷ 64px 격자 = 4×4) vs Severstal 98.4개(1600×256 → 25×4 = 100)다. 그 결과:

- Severstal `P_def(k, ·)`는 cluster당 수만~수십만 패치로 추정되어 **관측 cell이 192~197 / 243**(거의 전 공간)
- AITeX는 **22~60 / 243**으로 훨씬 희소 → `S_k`가 작고, 게이트 질의 시 미관측 cell(neutral 0.5, §6)로 빠질 확률이 크다

**같은 `ctx_prior` 공식이 두 데이터셋에서 표본 밀도가 전혀 다른 추정치를 만든다.** compat 게이트 진단에서 AITeX가 중간 케이스(관측 cell 전멸거부 + neutral fallback 수락)로 분류된 것과 정합한다(`.claude/.dev_note/aroma_compat_gate_clean-grounded_redesign.md`).

---

## 3. `c` — 배경 context cell

### 3-1. 대상 단위 = 64px 패치. 단 `c`는 패치가 아니라 **범주**

```
64px 패치 1개 → 5개 context feature 측정 → 각 tertile bin(0/1/2) → cell 코드 c
```

**다대일**이다. 패치는 세는 단위(관측), cell은 담는 칸(범주).

`distribution_profiling.py`:
- `CONTEXT_FEATURES`(`:80-83`) 순서 = `local_variance, edge_density, texture_entropy, frequency_energy, orientation_consistency`
- `N_CONTEXT_BINS = 3`(`:85`) — 주석 `P33 / P66 → bins 0, 1, 2`
- `_compute_bin_edges`(`:497-515`) → `edges[feat] = [p33, p66]` (데이터셋별 유도)
- `_context_cell_key`(`:480-494`) → `np.searchsorted(edges, val, side="right")` + 상한 clamp
  ⇒ **bin 0 = 하위 tertile, 1 = 중간, 2 = 상위.** 다섯 자리를 `_`로 연결. 공간 = **3⁵ = 243**

예: `c = 0_0_0_1_0` = local_variance 하위 / edge_density 하위 / texture_entropy 하위 / frequency_energy **중간** / orientation_consistency 하위.

### 3-2. 실측 규모 — cell 하나에 수천 패치가 담긴다

| 데이터셋 | context 패치 | 관측 clean cell | cluster별 P_def cell |
|---|---|---|---|
| AITeX | 120,140 | 128 / 243 | 22–60 |
| Kolektor | 52,943 | 242 / 243 | 105–203 |
| Severstal | 936,769 | 208 / 243 | 192–197 |
| MTD | 27,512 | 216 / 243 | 116–187 |
| MVTec Leather | 86,106 | 191 / 243 | 147–180 |

Severstal good 패치 590,200개가 208 cell에 담김 → **평균 약 2,838 patches/cell**. `P_clean(c)`는 특정 패치가 아니라 **히스토그램 한 칸의 높이**다.

---

## 4. 두 확률 — 무엇을 세는가

`_build_symmetric`(`distribution_profiling.py:551-627`). 결함 영역과 겹치는 패치(mask>0.5)는 `_context_worker`가 **상류에서 제외**하므로 계수 대상은 전부 배경이다(docstring `:566-568`).

```python
for r in context_rows:
    if r["image_type"] == "good":
        clean_counts[_context_cell_key(r, bin_edges)] += 1          # :590-592
    elif r["image_type"] == "defect":
        cid = cluster_assignments.get(r["image_id"])                # :594  ← 이미지의 k 상속
        def_counts[int(cid)][_context_cell_key(r, bin_edges)] += 1  # :598

clean_dist   = {cell: cnt / clean_total}                            # :601-604  → P_clean
P_def_patch[k] = {cell: cnt / total_k}                              # :606-612  → P_def
```

$$P_{\text{clean}}(c) = \frac{n_{\text{clean}}(c)}{\sum_{c'} n_{\text{clean}}(c')}, \qquad P_{\text{def}}(k, c) = \frac{n_{\text{def}}(k, c)}{\sum_{c'} n_{\text{def}}(k, c')}$$

| | 모집단 | 정규화 |
|---|---|---|
| `P_clean(c)` | 전 **normal 이미지**의 배경 패치 | 합 = 1 (실측 1.0000) |
| `P_def(k, c)` | 결함 영역이 cluster `k`인 **결함 이미지**의 배경 패치 | cluster별 합 = 1 (실측 1.0000) |

**두 granularity의 접합점이 여기다.** 패치가 개별적으로 `k`로 분류되는 게 아니라 **소속 이미지의 `k`를 물려받는다**. 따라서 `P_def(k, c)`는 "cluster `k` 결함이 **어떤 배경 위에 놓여 있었는가**"의 분포이며, 결함 픽셀의 분포가 아니다.

`P_clean`은 `k`에 무관하므로 **모든 cluster 행에 같은 열 가중**이 걸린다. "clean-grounded"의 grounding이 이것이다.

### 4-1. ⚠ `P_def`가 재는 것 — 오해 4건

**① 결함 국소 이웃이 아니라 이미지 전역 배경이다.**
severstal 1600×256 이미지 1장의 배경 패치 ~98개가 **전부 같은 `k`**를 상속한다. 결함이 좌측 끝에 있어도 우측 끝 패치까지 `P_def(k, ·)`에 들어간다. "결함 주변 배경"이 아니라 "결함이 있던 이미지의 배경 전체" — **국소성이 없다.**
AITeX 타일(256×256, 16패치)은 상대적으로 국소적이다. §2-5④의 *표본 밀도* 차이와 **별개의 의미 차이**다.

🔧 **이 지점이 연구 의도와 어긋난다 — 개선 항목 §4-5 참조.** ROI 배치 기준은 결함 국소 배경이어야 한다.

**② 이미지 1장 1표가 아니라 배경 면적 비례다.**
결함이 큰 이미지는 제외되는 패치가 많아 기여가 작아진다. 가중이 암묵적으로 배경 넓이에 비례한다.

**③ 결함 형태 정보가 들어 있지 않다.**
`k`는 라벨로만 들어온다. 분포 자체는 순수 배경 텍스처 통계다.

**④ cluster 배정이 없으면 통째로 드롭된다.**
`if cid is None: continue`(`:595-596`). morph 행이 없는 결함 이미지(마스크 부재·처리 실패)의 배경 패치는 **어느 `P_def`에도 들어가지 않는다**. §2-5①의 미참조 마스크 214건과는 별개 경로다.

### 4-2. 실측 support (2026-07-31)

`compatibility_matrix.json` 직접 계수:

| 데이터셋 | `clean_dist` cell | `P_def_patch` cell (cluster별) | 정규화 검산 |
|---|---|---|---|
| severstal | 208 | 195 / 196 / 192 / 192 / 197 | 전부 합 1.0 |
| kolektor | 242 | 132 / 186 / 203 / 191 / 105 | 〃 |
| mtd | 216 | 187 / 166 / 166 / 168 / 116 | 〃 |
| aitex | 128 | 37 / 22 / 60 / 23 / 60 | 〃 |
| mvtec_leather | 191 | 147 / 180 / 154 | 〃 |

### 4-3. 확률값의 원 카운트 — severstal `0_0_0_1_0`

`context_features.csv` + `morphology_clusters.json`에서 `_build_symmetric`을 재현한 결과(§5-2 값과 일치):

| | 분자 | 분모 | 분모 정체 |
|---|---|---|---|
| `P_clean(c)` | 30,494 | **590,200** | normal **5,902장 × 100패치** (1600×256 → 25×4 격자, 제외 0) |
| `P_def(1, c)` | 13,007 | **82,025** | cluster 1 결함 **859장**의 배경 패치 |

859 × 100 = 85,900 중 **3,875개(4.5%)가 결함 과반 타일로 제외**됐다(이미지당 평균 95.5 배경 패치). ⇒ severstal 결함이 프레임의 약 4.5%를 덮는다.

`c = 0_0_0_1_0` = `local_variance` 하위 / `edge_density` 하위 / `texture_entropy` 하위 / `frequency_energy` 중간 / `orientation_consistency` 하위 = **매끈하고 특징 없는 강판 면**. 243칸 균등이면 0.41%인데 `P_clean = 5.17%` — **12.6배 과대표**, 정상 강판의 지배적 텍스처다.

### 4-4. ⚠ `P_def`가 cluster를 거의 구분하지 못한다 (severstal)

같은 cell `0_0_0_1_0`에서 5개 cluster 전부:

| k | `P_def(k, c)` | 이미지 | 배경 패치 | `P_def / P_clean` |
|---|---|---|---|---|
| 0 | 0.146049 | 522 | 49,141 | 2.83 |
| **1** | **0.158574** | 859 | 82,025 | **3.07** |
| 2 | 0.072370 | 861 | 85,229 | 1.40 |
| 3 | 0.071595 | 438 | 41,176 | 1.39 |
| 4 | 0.101137 | 940 | 88,998 | 1.96 |

**전부 1보다 크다.** "cluster 1 결함이 매끈한 배경을 선호한다"가 아니라 **"severstal 결함 이미지 전반이 정상 이미지보다 배경이 매끈하다"**에 가깝다 — pool 수준 편향이지 결함 형태별 신호가 아니다.

cluster 행끼리의 log `P_def` 상관으로 확장 측정:

| 데이터셋 | min | median | max |
|---|---|---|---|
| **severstal** | **0.956** | **0.973** | **0.991** |
| mvtec_leather | 0.890 | 0.890 | 0.958 |
| kolektor | 0.451 | 0.741 | 0.849 |
| mtd | 0.507 | 0.632 | 0.795 |
| **aitex** | **0.070** | **0.330** | 0.709 |

severstal은 5개 행이 사실상 같은 분포다(r ≈ 0.97). ⇒ **`ctx_prior(k, c)`가 `k`에 거의 의존하지 않는다.** 어느 cluster를 붙이든 같은 자리를 추천한다.

원인은 §4-1①과 직결된다 — `P_def(k, ·)`가 이미지 **전역** 배경이고 severstal 이미지 1장이 평균 26.8 cell에 흩어지므로, cluster별 차이가 이미지 전체 배경 통계에 희석된다. AITeX(r = 0.07~0.71)는 타일 단위라 cluster별로 뚜렷이 갈린다.

**§3.2.4를 "결함 형태별로 적합한 배경을 고른다"로 서술하면 severstal에서는 실증되지 않는다.** §5-6(대칭 효과가 severstal에서 가장 약함)과 같은 방향의 발견이다.

### 4-5. 🔧 개선 필요 — `P_def`의 계수 범위를 결함 주변으로 좁혀야 한다

> **상태: 미구현 / 설계 제안.** 본 연구의 의도와 현행 구현이 어긋나는 지점이라 개선 항목으로 기록한다.
>
> 📋 **패치 노트: `.claude/.dev_note/aroma_ctxprior_localization_and_normalization.md`**
> 본 절(계수 범위 국소화)과 **§5-8**(`P_clean` 곱셈 형태 재검토)을 하나의 착수 단위로 묶어 정리했다. 순서 의존(국소화 → 상관 재측정 → 정규화 형태 결정 → τ 재캘리브레이션)이 있으므로 구현 시 그 문서를 기준으로 삼는다.

#### 연구 의도 vs 현행 동작

본 연구에서 ROI 배치 기준은 **결함이 실제로 놓여 있던 국소 배경**이어야 한다. "이 형태의 결함은 이런 표면 위에 생긴다"가 `ctx_prior`가 담아야 할 명제다.

현행 `_context_worker`는 결함 이미지의 **모든 비-결함 패치**를 방출하고 `_build_symmetric`이 그것을 전부 그 이미지의 `k`로 계수한다. 결함이 좌측 끝에 있어도 1600px 반대편 패치까지 `P_def(k, ·)`에 들어간다. 그 결과가 §4-4의 cluster 무차별성이다.

#### 학습–추론 스케일 불일치

| 단계 | 관측 범위 |
|---|---|
| 학습 (`P_def` 계수) | 이미지 **전역** 배경 (severstal ~95 타일) |
| 추론 (§6 3단계 판정) | crop **footprint** (160×96 → **6 타일**) |

같은 `ctx_prior`를 두 스케일에서 쓴다. 전역 통계로 학습한 값을 6타일 국소 질의에 적용하는 구조다.

#### 개선안

| 안 | 계수 대상 | 비고 |
|---|---|---|
| A. 반경 링 | 결함 bbox를 `R`만큼 팽창한 영역과 겹치는 배경 타일 | `R = _COMPAT_TILE`(1링)이 최소안 |
| B. 인접 타일만 | bbox에 접하는 8-이웃 타일 | 가장 좁음, 표본 급감 |
| C. 거리 가중 | 전 배경 타일을 쓰되 결함 중심거리로 감쇠 가중 | 표본 유지, hard cut 없음 |
| **D. footprint 정합** | **결함 bbox 크기의 창을 결함 중심에 두고 그 안의 배경 타일** | **추론 스케일과 일치 — 권장** |

D가 §6 3단계의 판정 단위와 정확히 맞물린다. §2-1의 "측정 대상 불일치"(형태=최대 blob / crop=전 blob)와 같은 계열의 정합 문제다.

#### 구현 메모

- `_context_worker`는 이미 `patch_xy`를 방출한다(`:387`). 결함 bbox는 `morphology_features.csv`의 `defect_bbox`에 있다 → **거리/포함 판정에 필요한 정보가 이미 산출돼 있다.**
- `_build_symmetric`에 반경 파라미터를 추가하는 방식이면 legacy `matrix` 경로와 분리된 채 additive로 갈 수 있다(현행 SGM 도입 방식과 동일).
- 재-profiling 필요. `matrix_symmetric`만 바뀌므로 `--compat_mode defect` 경로는 무영향.

#### 예상 효과·위험

- **효과**: cluster 행 간 상관이 내려가 `k` 조건화가 실제로 작동. severstal r ≈ 0.97 → 하락 기대.
- **위험 1**: 표본 급감. severstal k1 82,025 → 1링이면 이미지당 ~10-20 타일 수준으로 축소. support cell 수(현 192~197)도 감소.
- **위험 2**: AITeX는 이미 support 22~60으로 희소(§2-5④). 좁히면 §5-7(b)의 neutral 0.5 조우율(현 14~64%)이 더 악화될 수 있다. **데이터셋별 사전 스캔 후 반경 확정 필요.**
- **위험 3**: `P_clean`은 정상 이미지에 결함이 없어 같은 방식으로 좁힐 수 없다 → 두 항의 관측 범위가 비대칭이 된다(결함 국소 vs 정상 전역). 이 비대칭을 허용할지 별도 판단 필요.

---

## 5. SGM — 기하평균·support·정규화

`:614-627`:

```python
raw[cell] = ((p_def + epsilon) * (p_clean + epsilon)) ** 0.5   # epsilon = 1e-3
matrix_symmetric[k] = {cell: raw[cell] / max(raw.values())}
```

$$\text{ctx\_prior}(k, c) = \frac{\sqrt{(P_{\text{def}}(k,c)+\varepsilon)(P_{\text{clean}}(c)+\varepsilon)}}{\max_{c' \in S_k} \sqrt{(P_{\text{def}}(k,c')+\varepsilon)(P_{\text{clean}}(c')+\varepsilon)}}$$

| 요소 | 의미 |
|---|---|
| `√( · )` | **기하평균** — 두 인자에 대칭. 어느 한쪽이 0이면 전체 붕괴 |
| `ε = 10⁻³` | additive smoothing — 위 0-붕괴 방지. `symmetric_epsilon`으로 provenance 저장 |
| `S_k = {c : P_def(k,c) > 0}` | **관측 support** — 그 cluster에서 실제 관측된 cell만 대상 |
| `max` 나눗셈 | 행 정규화 → cluster별 최적합 cell = 1.0. 논문 식이 `=`가 아니라 `∝`인 이유 |

**대칭성의 의미**: 결함 쪽엔 흔하지만 clean에는 드문 cell이 그 **반대와 동등하게 감점**된다. 이 점이 결함-조건부 빈도 `P_def` 단독과 다른 지점이다.

엄밀히는 **교환 대칭** `f(a,b) = f(b,a)`다. `(P_def=0.100, P_clean=0.001)`과 `(P_def=0.001, P_clean=0.100)`이 정확히 같은 raw 값을 받는다. log를 취하면 구조가 드러난다:

$$\log \text{SGM} = \tfrac{1}{2}\log(P_{\text{def}}+\varepsilon) + \tfrac{1}{2}\log(P_{\text{clean}}+\varepsilon)$$

**log 공간의 균등가중 산술평균** — 두 항이 정확히 반반이다.

왜 기하평균인지는 대안과 비교하면 분명하다:

| 결합 | 교환 대칭 | 한쪽이 0에 가까울 때 |
|---|---|---|
| 산술평균 `(a+b)/2` | ○ | 다른 쪽이 크면 **살아남음** — 감점이 안 됨 |
| `min(a, b)` | ○ | 완전 붕괴 — 큰 쪽 정보를 전부 버림 |
| **기하평균 `√(ab)`** | ○ | **곱셈적 감점** — 붕괴하되 큰 쪽 정보는 보존 |
| `P_def` 단독 | ✗ | clean 희소성을 무시 |

⚠ 단, 이 대칭은 **support 안에서만** 성립한다. `S_k`의 정의가 비대칭이라 밖에서는 깨진다 — §5-7.

### 5-1. 두 항의 역할 분담

| 항 | 묻는 것 |
|---|---|
| `P_def(k, c)` | **적합성** — 이 cluster 결함이 놓여 있을 법한 배경인가 |
| `P_clean(c)` | **가용성** — 실제로 붙일 clean pool에 그런 자리가 존재하는가 |

기하평균이라 한쪽이 작으면 전체가 죽는다. `P_def` 단독이면 "결함 이미지엔 흔하지만 clean pool엔 없는 자리"를 1순위로 추천해 **붙일 데가 없고**, `P_clean` 단독이면 그냥 텍스처 빈도표라 **결함 정보가 사라진다.**

### 5-2. 실측 예시 — Severstal cluster 1 peak

| 항목 | 값 |
|---|---|
| peak cell | `0_0_0_1_0` |
| `P_def(1, c)` | 0.158574 |
| `P_clean(c)` | 0.051667 |
| 정규화 전 SGM | `√((0.158574+1e-3)(0.051667+1e-3))` = 0.091675 |
| 행 정규화 후 | **1.0000** |

### 5-3. 대칭이 실제로 순위를 바꾼다 (severstal k=1, 실측 2026-07-31)

peak 하나만 보면 `P_def` 최대 = `ctx_prior` 최대라 대칭의 효과가 안 드러난다. 2위 이하에서 갈린다:

| cell | `P_def` | `P_clean` | `ctx_prior` | `P_def` 순위 → `ctx_prior` 순위 |
|---|---|---|---|---|
| `0_0_0_1_0` | 0.15857 | 0.05167 | **1.0000** | 1 → 1 |
| `0_0_0_2_0` | 0.02683 | 0.04649 | 0.3966 | 4 → **2** ↑ |
| `1_1_0_2_2` | 0.01919 | **0.05260** | 0.3588 | 7위권 → **3** ↑↑ |
| `0_0_2_0_1` | 0.03249 | 0.03089 | 0.3565 | 2 → 4 ↓ |
| `2_2_1_2_2` | 0.02789 | 0.02358 | 0.2907 | 3 → 5 ↓ |

최대 강등 3건:

| cell | `P_def` | `P_clean` | 순위 |
|---|---|---|---|
| `1_0_2_1_1` | 0.00445 | **0.000539** | 52 → **91** (−39) |
| `1_0_1_1_0` | 0.00182 | **0.000469** | 91 → 118 (−27) |
| `2_1_2_1_1` | 0.00344 | 0.001201 | 62 → 88 (−26) |

`1_1_0_2_2`는 결함 쪽 빈도가 평범한데 정상 표면에 흔해서 3위로 올라오고, `1_0_2_1_1`은 결함 쪽엔 관측됐지만 clean pool에 거의 없어 39계단 내려간다. **"결함 옆에 자주 있었다"만으로는 부족하고 "정상 표면에도 흔한 자리"여야 점수가 유지된다.**

### 5-4. `P_clean = 0` 케이스 — ε이 실제로 일하는 지점

`S_k`는 `P_def > 0`으로만 정의되므로, support 안에 있으면서 `P_clean(c) = 0`인 cell이 생길 수 있다. 그 경우 SGM = `√((p_def+ε)·ε)`로 붕괴하고, **ε이 없으면 정확히 0**이다.

데이터셋별 발생 건수(실측 2026-07-31):

| 데이터셋 | clean cell | cluster별 *clean support 밖* `P_def` cell 수 |
|---|---|---|
| severstal | 208 | 0 / 0 / 0 / 0 / 0 |
| kolektor | 242 | 0 / 0 / 0 / 0 / 1 |
| mtd | 216 | 1 / 1 / 3 / 3 / 0 |
| aitex | 128 | 0 / 0 / 3 / 3 / 2 |
| **mvtec_leather** | 191 | **5 / 5 / 6** |

severstal은 0건 — `P_def` support가 clean support에 완전히 포함된다. leather가 가장 많다(§6-1의 leather washout·포화와 같은 방향).

⇒ ε의 실질 역할은 "0-붕괴 방지"이며, **그 발동이 데이터셋별로 0~6 cell 수준**이다. §9 정직성 7번(하드코딩 상수)을 방어할 때 이 규모를 함께 제시할 수 있다.

### 5-5. 실측 — 두 항의 기여가 정말 균등한가 (2026-07-31)

기하평균은 log 산술평균이므로 **log 변동폭이 큰 쪽이 순위를 지배**한다. support 내에서 측정:

| 데이터셋 | `sd log P_def` / `sd log P_clean` | corr(log ctx, log P_def) | corr(log ctx, log P_clean) |
|---|---|---|---|
| severstal | 0.98 ~ 1.03 | 0.964 ~ 0.972 | 0.962 ~ 0.972 |
| mtd | 0.88 ~ 1.13 | 0.898 ~ 0.957 | 0.868 ~ 0.959 |
| kolektor | 0.76 ~ 1.05 | 0.792 ~ 0.977 | 0.886 ~ 0.978 |
| aitex | 0.83 ~ 1.19 | 0.813 ~ 0.937 | 0.846 ~ 0.942 |
| mvtec_leather | 1.01 ~ 1.04 | 0.820 ~ 0.850 | 0.814 ~ 0.841 |

비율이 전부 1 근처이고 두 상관계수가 거의 같다. ⇒ **어느 한쪽도 순위를 지배하지 않는다.** "동등하게 감점"이 형식뿐 아니라 실측으로도 성립한다.

### 5-6. 대칭의 *효과 크기*는 데이터셋 속성에 좌우된다

대칭이 성립하는 것과 대칭이 무언가를 바꾸는 것은 다르다. `P_def` 단독 순위 대비:

| 데이터셋 | corr(log `P_def`, log `P_clean`) | Spearman(`P_def`, `ctx_prior`) | 평균 \|Δrank\| | 최대 \|Δrank\| |
|---|---|---|---|---|
| severstal | **0.855 ~ 0.889** | 0.966 ~ 0.970 | ~10 / 195 | 57 |
| kolektor k1~k3 | 0.895 ~ 0.911 | 0.965 ~ 0.969 | ~10 / 190 | 54 |
| mtd | 0.561 ~ 0.836 | 0.844 ~ 0.944 | ~14 / 170 | 74 |
| aitex k0 / k3 | **0.429 / 0.430** | 0.792 / 0.817 | 5.6 / 3.4 | 15 / 8 |
| **mvtec_leather** | **0.336 ~ 0.429** | **0.716 ~ 0.833** | **23 ~ 25 / 150** | **105** |

⇒ **대칭 항의 기여도 = 결함 이미지 배경 분포와 정상 이미지 배경 분포가 얼마나 다른가.**

- severstal은 두 pool이 같은 강판이라 배경 분포가 이미 닮았고(r ≈ 0.87) 대칭이 순위를 거의 안 바꾼다(Spearman 0.97).
- leather는 두 분포가 크게 달라(r ≈ 0.37) 실질 재배치가 일어난다(Spearman 0.72, 최대 105계단).

**대칭 항은 결함/정상 pool의 배경이 다른 데이터셋에서만 일한다.** §5-3의 severstal 재순위 예시는 그중 효과가 가장 약한 케이스임에 유의.

### 5-7. ⚠ 대칭이 깨지는 지점 — `S_k` 정의와 neutral 0.5

#### (a) support 정의 자체가 비대칭

`S_k = {c : P_def(k,c) > 0}` — `P_clean`은 support 결정에 관여하지 않는다.

| 상황 | 행렬에서 | 런타임 값 |
|---|---|---|
| `P_def > 0`, `P_clean = 0` | 행에 **있음**, `√((p+ε)·ε)` | ≈ 0 (최하위) |
| `P_def = 0`, `P_clean > 0` | 행에 **없음** | `.get(cell, 0.5)` → **0.5** |

두 경우가 전혀 동등하지 않다. 후자가 압도적으로 유리하다. **"반대와 동등하게 감점"은 support 안에서만 참이다.**

#### (b) neutral 0.5는 중립이 아니라 관대하다

행 max 정규화 때문에 1.0은 정의상 1개뿐이고 나머지는 급락한다. 실측:

| 데이터셋 | row median `ctx_prior` | `ctx_prior < 0.5` 비율 | **런타임 조우율** (clean 확률질량이 row 밖) |
|---|---|---|---|
| severstal | 0.037 | 98 ~ 99% | **0.0 ~ 0.1%** |
| mvtec_leather | 0.053 | 98% | 0.9 ~ 11.8% |
| mtd | 0.141 | 90 ~ 94% | 0.8 ~ **23.3%** (k4) |
| kolektor | 0.265 | 85 ~ 94% | 1.8 ~ **25.0%** (k4) |
| **aitex** | 0.053 | 90 ~ 95% | **14.2 ~ 63.7%** |

severstal k1: 관측 196 cell 중 **195개가 0.5 미만**(median 0.027). 행에 없는 clean cell 12개는 런타임에 0.5를 받아 **관측 195개 중 194개보다 높은 점수**가 된다.

⇒ `_positive_place`의 footprint 평균에서 **미관측 타일 1개가 관측 타일 여러 개를 압도**한다(0.5 vs median 0.027 ≈ 18배). 배치가 "결함 이미지에서 한 번도 관측되지 않은 배경" 쪽으로 끌린다 — **의도와 반대 방향**이다.

심각도는 조우율에 비례한다:

- **severstal 안전**(0.0~0.1%) — support가 clean 공간을 사실상 덮는다
- **aitex 심각**(최대 63.7%) — k3는 clean 패치 확률질량의 2/3가 미관측 cell에 떨어져 0.5를 받는다. §2-5④에서 AITeX가 "관측 cell 전멸거부 + neutral fallback 수락"으로 분류된 것의 정량 근거
- mtd k4·kolektor k4(23~25%), leather k0/k2(10~12%)도 무시할 수 없다

§6 2단계의 "미관측 cell은 거부가 아니라 중립 0.5다" 한 줄이 실제로는 이 규모의 편향을 뜻한다. **τ 값에 따라 배치 결과가 뒤집힐 수 있으므로 별건 검토 대상.**

### 5-8. 🔧 개선 필요 — `P_clean`을 곱하는 것이 매칭인가

> **상태: 미구현 / 설계 제안.**
> 📋 **패치 노트: `.claude/.dev_note/aroma_ctxprior_localization_and_normalization.md` §2**
> §4-5(계수 범위 국소화)와 **하나의 착수 단위**다. 국소화가 선행 조건이므로 순서는 그 문서 §4를 따른다.

연구 핵심은 "결함 국소 배경의 특징을 clean pool에서 찾아 그 자리에 합성"이다. 이 정의에서 세 항의 지위는:

| 항 | 지위 |
|---|---|
| `P_def(k, ·)` | **질의(query)** — 찾고자 하는 표면 프로파일 |
| clean pool | **탐색 대상(corpus)** |
| `P_clean(c)` | corpus의 **주변 분포** — 질의의 한쪽 항이 **아님** |

⇒ §4-5 위험 3(관측 범위 비대칭)은 **문제가 아니다.** query는 국소여야 하고 corpus 통계는 전역이어야 한다.

정보검색 비유가 정확하다 — `P_def` = query term 분포, `P_clean` = document frequency. SGM은 `query × DF`이며 **TF-IDF와 반대 방향**이다. IDF 논리대로면 나눠야 한다: `lift(k,c) = P_def / P_clean`.

**실측 — SGM 순위와 lift 순위가 직교한다** (Spearman, cluster별 범위):

| 데이터셋 | Spearman(lift, `ctx_prior`) |
|---|---|
| severstal | −0.084 ~ +0.121 |
| aitex | −0.256 ~ +0.049 |
| mtd | −0.276 ~ +0.040 |
| kolektor | −0.272 ~ +0.101 |
| mvtec_leather | −0.077 ~ +0.008 |

severstal k=1 개별 cell(§5-3과 같은 대상):

| cell | `P_def` | `P_clean` | **lift** | lift 순위 | `ctx_prior` 순위 |
|---|---|---|---|---|---|
| `0_0_0_1_0` | 0.15857 | 0.051667 | 3.07 | 11 | 1 |
| `1_1_0_2_2` | 0.01919 | 0.052596 | **0.37** | **161** | **3** |
| `1_0_2_1_1` | 0.00445 | 0.000539 | **8.26** | **2** | **92** |

`1_1_0_2_2`는 결함 쪽에 정상보다 **덜** 흔한데(lift 0.37) SGM이 3위로 올리고, `1_0_2_1_1`은 cluster 1에 가장 특이한 표면인데(lift 8.26, 2위) 92위로 내린다. 노이즈 아니다 — 각각 결함 ~365 / clean ~318 패치.

**`P_clean`의 위치가 뒤바뀌어 있다:**

| 단계 | 가용성 항이 정당한가 | 현재 |
|---|---|---|
| ROI 후보 열거 | **정당** — pool에 없는 cell 추천은 슬롯 낭비 | legacy `matrix`, SGM 미적용 |
| 배치(§6) | **부당** — 타일이 이미 손에 있어 가용성 확보됨 | SGM 적용 |

도입 경위상 검증된 lever는 `P_def` patch-gran화(support 확장)와 max-norm(τ 정합)이며, **`P_clean` 곱셈 자체의 기여는 분리 입증되지 않았다**(`aroma_compat_gate_clean-grounded_redesign.md` §3-1·§5-2).

대안 4안(`P_def` 단독 / lift / cluster-특이 lift / 역할 분리)과 각 trade-off는 패치 노트 §2-5. **lift 채택 시 중립점이 1.0으로 정의되어 §5-7(b)의 neutral 0.5 자의성도 함께 해소된다.**

---

## 6. 배치 — `ctx_prior`가 실제 좌표로 바뀌는 5단계

`_positive_place`(`generate_defects.py:888-1030`). **판정 단위는 cell이 아니라 crop footprint다.** 같은 cell 위에 놓인 두 후보라도 crop이 함께 덮는 이웃 타일이 달라 점수가 갈린다.

> **범위 주의**: 본 절은 "**이미지 한 장 안에서 어디**"만 다룬다. 그 앞 단계인 "**어느 normal 이미지**"는 `ctx_prior`를 쓰지 않는다 — §6-2 참조.

![위치 결정 5단계](Article/figure/image/%5Bfigure%203.2.4%202%5D%20placement_footprint.png)

> 생성: `Article/figure/script/[figure 3.2.4 2] placement_footprint.py` — 운영 함수(`_extract_context_features`/`_context_cell_key`/`_is_clean_background`/`_tile_anchors`)와 상수(`_COMPAT_TILE`/`_POS_STRIDE`/`_POS_TOPK`)를 직접 import(재구현 금지). 대상 = severstal normal `00031f466`, cluster 1, crop 160×96.

### 1단계 — 후보 격자 열거

crop이 완전히 들어가는 모든 좌상단 위치를 `stride = _POS_STRIDE = 32`로 훑는다. 후보 수가 `_POS_MAX_CAND`를 넘으면 stride를 2배씩 늘려 상한 이하로 맞춘다. 우·하단 끝 좌표는 항상 포함시켜 가장자리가 도달 불가능해지지 않게 한다(`_axis`). 예시에서 **후보 276개**.

### 2단계 — 타일별 (compat, void) 캐시

각 64px 앵커에 대해 **한 번만** 계산한다:

```python
cell    = cell_key_fn(ctx_feat_fn(gray), bin_edges)
compat  = float(compat_row.get(cell, 0.5))     # S_k 밖 = 미관측 → neutral 0.5
is_void = not _is_clean_background(gray, ...)
```

cell·void 모두 cluster와 무관하므로 캐시가 성립 → 비용이 O(후보 수)가 아니라 **O(서로 다른 타일 수)**.

**미관측 cell은 거부가 아니라 중립 0.5다.** 게이트를 하드 필터로 오해하면 안 된다.

그림 **패널 A**가 이 단계다 — 100개 타일(4×25) 각각의 compat(위 숫자)과 cell key(아래). 같은 강판 한 장 안에서 compat이 **0.01~0.36**으로 갈리고, cell도 `0_0_1_0_1`·`1_1_1_0_1`·`0_0_2_0_1` 등으로 흩어진다(§3-1의 "이미지 1장이 평균 27 cell 점유"가 눈으로 보이는 지점).

### 3단계 — 후보 점수 = footprint를 덮는 타일들의 compat 평균

```python
for ay in _tile_anchors(y, ch, h_img, _COMPAT_TILE):
    for ax in _tile_anchors(x, cw, w_img, _COMPAT_TILE):
        compat, is_void = _tile(ax, ay)
        vals.append(compat)
        if is_void: has_void = True
mean = sum(vals) / len(vals)
```

`_COMPAT_TILE_AGG = "mean"`이 기본(대안 `'min'`은 상수에만 존재, 미사용).

그림 **패널 B**가 핵심이다. 동일 crop 160×96이 각각 **6개 타일**(x 3 × y 2)을 덮는데, 위치에 따라 평균이:

| 후보 | 위치 | footprint 타일 | mean-compat |
|---|---|---|---|
| best | (1376, 128) | 6 | **0.357** |
| median | (416, 160) | 6 | 0.135 |
| worst | (512, 64) | 6 | **0.028** |

**13배 차이**다. 타일 수가 같아도 어떤 타일 조합을 덮는지가 점수를 정한다.

### 4단계 — void 배제 (하드 필터)

footprint가 void 타일을 **하나라도** 포함하면 후보에서 제외한다. compat이 높아도 탈락이다. 비-void 후보가 0개면 `None`을 반환해 호출부가 **다른 normal로 도망친다**.

⚠ **현 환경에서 이 단계는 무력하다** — §6-1 참조. 예시에서 void 탈락 **0개**.

### 5단계 — top-K 샘플링

```python
nonvoid.sort(key=lambda t: t[0], reverse=True)   # 안정 정렬 → 동점은 스캔 순서 유지
k = max(1, min(_POS_TOPK, len(nonvoid)))         # _POS_TOPK = 8
_chosen_mean, chosen_pos = rng.choice(nonvoid[:k])
best_nonvoid_mean = float(nonvoid[0][0])         # τ 판정은 최선값으로
return chosen_pos, best_nonvoid_mean, len(nonvoid)
```

argmax가 아니라 상위 8개 무작위인 이유는 `n_per_roi` 반복이 하나의 seeded rng를 공유하므로 **배치 다양성**을 얻기 위함이다. 고정 seed면 결정론이 유지된다.

**τ 임계 비교는 `nonvoid[0]`(최선)로, 배치는 top-K 샘플 위치로** 한다 — "좋은 자리가 하나라도 있으면 이 normal을 쓴다, 단 붙이는 자리는 상위 8개 중 추첨". 그림 패널 C의 초록 밴드가 그 8개, 점선이 τ 판정 기준(0.357)이다.

### 정리 — 같은 cell 안에서 자리를 가르는 요인

| 요인 | 작동 |
|---|---|
| **footprint 조합** | crop이 덮는 이웃 타일들의 compat 평균 (동일 cell 타일이어도 이웃이 다름) |
| void 배제 | footprint에 void 1개라도 있으면 탈락 — compat 무관 (**현재 무력**) |
| top-K rng | 상위 8개 동급 후보 중 추첨 → 최종 좌표 |

`ctx_prior`는 **후보 순위를 매기는 재료**이고, 좌표는 `footprint 평균 → void 필터 → top-8 추첨`으로 정해진다. cell 하나만으로는 자리가 정해지지 않는다.

### 6-1. ⚠ void 게이트가 현 환경에서 무력화돼 있다 (실측 2026-07-31)

`_background_quality_score`(`:457-460`)가 입력을 `float32`로 변환한 뒤 `cv2.Laplacian(gray, cv2.CV_64F)`를 호출한다. **OpenCV 4.13.0은 `CV_32F → CV_64F` 조합을 지원하지 않아 예외를 던진다.**

```python
# _is_clean_background :517-521
        gray_f = gray.astype(np.float32)
        return _background_quality_score(gray_f, blur_threshold) >= min_quality
    except Exception:
        return True          # ← 예외를 삼키고 "clean" 반환 (fail-open)
```

임계 비교 **전에** 예외가 나므로 `min_quality` 값과 무관하다. `_positive_place`도 같은 패턴으로 한 번 더 감싼다(`:981-984` → `is_void = False`).

검증: `_is_clean_background(np.zeros((64,64), np.uint8))` → **`True`**. 순수 검은 패치가 clean으로 판정된다. **로컬과 Colab 모두 cv2 4.13.0에서 동일 재현.**

Laplacian 조합별 지원 여부:

| src | dst | 결과 |
|---|---|---|
| uint8 | CV_64F | OK |
| uint8 | CV_32F | OK |
| **float32** | **CV_64F** | **RAISES** |
| float32 | CV_32F | OK |

**영향 범위** — `_is_clean_background`/`_background_quality_score`를 쓰는 모든 게이트:

| 위치 | 게이트 |
|---|---|
| `:431` | `_foreground_mask` void 전경 거부 가드(`_FG_VOID_QUALITY`) |
| `:980` | `_positive_place` void 배제 (본 절 4단계) |
| `:1096` | `_normal_tile_cells` void |
| `:1271` | 타일 창 void 검사 |
| `:1552` | random fallback 위치 게이트 |
| `:2743` | `load_normal_images` pool 게이트(`--reject-clean-bg`) |

실측 void율(5종 good 타일): **전부 0.0%**. 데이터에 void가 없어서가 아니라 검출기가 fail-open이기 때문이다.

**수정 시 임계 재산정이 함께 필요하다.** `ddepth`를 `CV_32F`로 고쳐 되살리면 `min_quality=0.7`에서 void율이 98~100%로 반대 극단이 된다(가중식의 `blur`·`contrast` 항이 산업 표면 64px 타일에서 구조적으로 낮아 quality median 0.43~0.60). **운영 설정은 `min_quality=0.5`** 이며 그 경우:

| 데이터셋 | void@0.7 | void@0.5 |
|---|---|---|
| severstal | 99.8% | **23.4%** |
| kolektor | 100.0% | 78.4% |
| mtd | 98.4% | **22.0%** |
| mvtec_leather | 100.0% | 99.6% |
| aitex_tiled | 98.5% | **11.5%** |

severstal·mtd·aitex는 11~23%로 합리적, leather 99.6%는 포화(가죽 표면 quality median 0.432) — 별건.

과거 devnote에 게이트가 작동한 실측이 있으므로(검은배경 9.1%→6.0%, `aroma_exp4v2_foreground-void-rejection.md`), **cv2 업그레이드 시점 이후 조용히 죽은 것**으로 보인다. 결과 재현성에 직결되므로 별도 dev_note로 분리 처리한다.

### 6-2. 그 앞 단계 — 어느 normal을 쓸지는 `ctx_prior`가 정하지 않는다

§6 1~5단계는 "이미지 한 장 **안에서** 어디"다. 그보다 앞에 "**어느 이미지**"가 있고, **여기엔 compat 행렬 값이 쓰이지 않는다.**

배경 선택 우선순위 3단(`generate_defects.py:3199-3260`):

| 순위 | 경로 | 기준 | compat 값 사용 |
|---|---|---|---|
| 1 | `clean_bg_selected.json` 사전계산 pool | `clean_bg_selection.py` 별도 산출(roi_idx 조인, `image_id` 불일치 시 폐기) | ✗ |
| 2 | image-rank (`image_rank_on`) | **히스토그램 교집합** — 아래 | ✗ |
| 3 | `rng.choice(normal_images)` | 균일 추첨 | ✗ |

#### 기준 = cell 분포 유사도 (배경 ↔ 배경)

```python
_sim = _hist_intersection(_p_dv, _p_clean)          # :3246
_sel = _rank_normals(_scored, rng, _NORMAL_TOPK)    # :3253  top-16 추첨
```

| 항목 | 정의 |
|---|---|
| `p_dv` | **소스 결함 이미지의 배경** cell 히스토그램(`_dv_bg_hist:1174`). 결함 타일 제외 — mask>0.5, 없으면 bbox 과반 겹침(`_context_worker` 규칙 미러) |
| `p_clean` | clean 이미지의 non-void cell 히스토그램(`_cell_hist:1152` over `_normal_tile_cells:1033`) |
| 유사도 | `_hist_intersection = Σ_c min(p[c], q[c])` ∈ [0,1] (`:1284`) |
| 선택 | 내림차순 정렬 → **top-16**(`_NORMAL_TOPK`) rng 추첨. stage-2 re-pick pool도 top-16으로 좁힘 |
| 샘플링 | stride 64, **cap 64 타일/이미지** — 초과 시 stride 2배씩(좌상단 편향 방지) |

**`compat_row`를 한 번도 읽지 않는다.** 공유하는 것은 `bin_edges`와 cell 어휘뿐이다. 즉 `ctx_prior`(결함 cluster × 배경 cell)가 아니라 **배경 대 배경** 분포 유사도다.

#### 왜 compat 평균이 아닌가

이전 티어는 normal을 cluster에 대한 **mean compat**으로 점수 매겼다(`_image_compat_score:1111`). 주석 `:778-791`이 교체 이유를 명시한다 — cluster는 결함 형태 군집이지 배경 유형이 아니고, 평균이 외양 이질성을 뭉갠다. **mean-compat이 비슷하다고 배경이 닮은 것이 아니다.** 그래서 분포 비교로 교체했다.

⚠ `_image_compat_score`는 **정의만 남고 런타임 호출이 없다**(호출부는 `.claude/.etc/positive_place_viz/` 스모크·시각화 스크립트뿐). 코드를 읽고 "이미지 순위도 compat으로 매긴다"고 서술하면 틀린다.

⇒ 두 단계가 서로 다른 신호를 쓴다: **소스 결함이 놓여 있던 배경과 닮은 배경을 고르고(§6-2), 그 안에서 cluster 호환 좌표를 고른다(§6 1~5단계).**

#### 활성 조건 — §6과 동일

`compat_matrix ≠ None` AND `bin_edges ≠ None` AND `compat_mode == "symmetric"` AND `compat_threshold > 0` AND cv2(`:3018-3024`). 하나라도 빠지면 균일 추첨으로 되돌아가고 rng 스트림이 legacy와 byte-identical하다.

#### ⚠ §6-1 결함이 이 경로에도 걸린다

`p_dv`·`p_clean` 둘 다 void 타일을 빼도록 설계됐으나 `_is_clean_background`가 fail-open이라 **현재 아무 것도 빠지지 않는다.**

임계 비대칭도 있다:

| 히스토그램 | void 판정 `min_quality` |
|---|---|
| `p_dv` (`_dv_bg_hist:1268`) | **0.7 하드코딩** |
| `p_clean` (`_normal_cells_for:3054`) | `min_bg_quality` — 운영 **0.5** |

cv2 dtype 버그를 고치면 §6-1 표대로 `void@0.7`이 98~100%가 되어 **`p_dv`가 대부분 `{}`가 되고 → `image_rank`가 통째로 균일 추첨으로 폴백**한다. 수정 시 이 경로의 임계까지 함께 재산정해야 한다.

---

## 7. ⚠ 소비 경로 — 행렬이 2개고 소비처가 갈린다

`compatibility_matrix.json`에는 **서로 다른 두 행렬**이 들어 있고, 파이프라인 단계마다 다른 쪽을 읽는다. 문서 §4~§6이 기술하는 것은 `matrix_symmetric`뿐이다.

### 7-1. 파일 안의 두 행렬

| 키 | 산출 방식 | 관측 단위 | 정규화 | 스케일 | 코드 |
|---|---|---|---|---|---|
| `matrix` (legacy) | 결함 이미지의 **패치 평균** context → 이미지당 cell 1개 → 카운트 → `P(cell\|k)` | 이미지 1장 = cell **1개** | 행 합 = 1 (확률) | max 보통 <0.2 | `:995-1013` |
| **`matrix_symmetric`** (SGM) | 패치 개별 계수 + clean 접합 기하평균 (§4·§5) | **64px 패치** | 행 **max = 1** | [0, 1] | `_build_symmetric` |
| `clean_dist` / `P_def_patch` / `symmetric_epsilon` | SGM 재료·provenance | 패치 | 합 = 1 | — | 〃 |

**두 스케일은 호환되지 않는다.** τ(`--compat_threshold`)는 symmetric의 max-normalized [0,1]을 전제로 캘리브레이션돼 있어, legacy raw 확률에 그대로 먹이면 전량 거부된다. 코드가 `compat_mode=symmetric`인데 키가 없으면 **hard-fail**시키는 이유다(`generate_defects.py:2959-2965`, silent fallback 금지).

### 7-2. 단계별 소비처

| 단계 | 코드 | 읽는 키 | 쓰임 |
|---|---|---|---|
| ROI 후보 생성·랭킹 | `roi_selection.py:429` | **`matrix`** (legacy) | `ctx_prior` → `ROI_score`; `--strategy compatibility`는 `0.6·ctx_prior + 0.4·morph_prior` |
| deficit 분석 | `distribution_profiling.py:1056` | **`matrix`** (legacy) | `max(0, P(c\|good) − P(c\|k))`. roi_selection에 전달되나 현재 weight 0 (provenance) |
| **합성 배치** | `generate_defects.py:2958` | **`matrix_symmetric`** (`--compat_mode symmetric`) | §6 — footprint 점수·τ 판정 |
| 배경 이미지 선택 | `generate_defects.py:3238-3253` | **없음** (`bin_edges`만) | §6-2 — 히스토그램 교집합 |
| 학습 JSONL | `build_train_jsonl.py:355` | 없음 (`roi_candidates.json` 경유) | `ctx_prior` MAX → `stability_score` |
| clean-bg 사전선택 | `clean_bg_selection.py:142-146` | `bin_edges`만 | 자체 히스토그램 재구성 |

### 7-3. 그래서 `ctx_prior`라는 이름이 두 산출을 가리킨다

| 부르는 곳 | 실체 |
|---|---|
| §3.2.4 식(§5의 SGM) | `matrix_symmetric[k][c]` — 배치 게이트 전용 |
| §3.2.5 `ROI_score`의 `ctx_prior` | `matrix[k][c]` — legacy 이미지평균 확률 |

**같은 기호가 서로 다른 스케일·다른 관측 단위의 값을 가리킨다.** legacy는 결함 이미지 1장이 cell 1개에만 기여하므로 support가 결함 이미지 수로 상한되고, §2-5④에서 인용한 192~197 / 22~60은 **SGM 쪽 수치**다.

**이 분기는 사고가 아니라 의도된 설계다.** `aroma_compat_gate_clean-grounded_redesign.md` §4가 `roi_selection` 미전환을 명시하고 근거를 둘 든다 — (a) cluster당 ~191 cell로 candidate 폭증, (b) selection↔placement 직교 확인. 결론은 "**clean-grounding은 게이트 전용**". 다만 §5-8은 그 결정이 가용성 항을 가장 부당한 쪽(배치)에만 남겼다고 본다.

논문에서 "선택과 배치가 같은 `ctx_prior`를 쓴다"로 읽히면 부정확하다. 활성화 조건도 다르다 — 배치 쪽은 `--compat_threshold > 0` **AND** `--compat_mode symmetric`(둘 다 기본 OFF: `0.0` / `"defect"`)일 때만 동작하고, ROI 선택 쪽은 항상 legacy를 읽는다.

**미확인 사항**: 실제 실험 실행 커맨드의 `--strategy` / `--compat_mode` / `--compat_threshold` 조합. 결과 해석 전 확인 필요.

---

## 8. 논문 대응 위치

| 문서 위치 | 대응 내용 |
|---|---|
| §3.2.2 | `k`(GMM/BIC 군집), `c`(P33/P66 tertile cell), Figure 3.2.2-1(군집 산점도·`P(k)`), Figure 3.2.2-2(context feature 분포·tertile 경계) |
| §3.2.4 | `ctx_prior` 식·산출 설명, Figure 3.2.4-1(cluster × cell 히트맵), Figure 3.2.5-3(흐름도) — **`matrix_symmetric` 기준** (§7-3) |
| §3.2.5 | `ROI_score`의 `ctx_prior` — **legacy `matrix` 기준** (§7-3) |
| §3.2.3 | `defect_subtype`(Table 4/4b), `background_type`(Table 3) — **`ctx_prior`와 무관한 별개 축** |

---

## 9. 서술 시 지켜야 할 정직성

1. **`k`는 GT 클래스가 아니다** — §2-3 교차표(severstal·leather 모두 전 군집이 복수 클래스 혼재).
2. **`c`는 개별 패치가 아니다** — §3-1 다대일, §3-2 평균 ~2,838 patches/cell.
3. **`k`와 `c`는 관측 단위가 다르다** — 결함 영역 vs 64px 패치, 접합은 이미지 수준 상속(§4).
4. **형태 특징은 최대 blob만, 합성 crop은 전 blob** — §2-1·§2-5②. severstal class3은 73.5%가 다중 성분이라 군집이 기술하는 대상과 붙여지는 대상이 어긋난다. "결함 형태로 군집화한다"고만 쓰면 과장이다.
5. **행 1개의 의미가 데이터셋마다 다르다** — §2-5 대조표. AITeX는 타일, severstal은 이미지. "결함 하나당 하나의 k"가 아니다.
6. **`background_type`(Table 3)은 `c`가 아니며 현행 파이프라인에서 산출되지 않는다** — §1 주의.
7. **`ε = 10⁻³`은 하드코딩 상수다** — "no hand-set constants" 서술과 함께 쓸 때 주의. smoothing은 통상 방법론 기본값이고 행 정규화가 뒤따라 순위 영향은 제한적이라는 것이 방어 논거.
8. **`ctx_prior` 추정 밀도가 데이터셋 간 6배 차이** — §2-5④. AITeX `S_k` 22~60 vs severstal 192~197. 같은 공식이지만 통계적 신뢰도가 다르다.
9. **GMM 특징 중복·min-max 왜곡·라벨 부분 사영** — §2-4. 특히 **`k`를 cluster 라벨로 지칭하면 안 된다**(AITeX는 `linear_scratch` 군집이 2개, 서로 AR 5배 차이).
10. **severstal 마스크 214건이 형태 통계에 진입하지 않는다** — §2-5①.
11. **void 게이트가 현 환경에서 무력화돼 있다** — §6-1. cv2 4.13.0 dtype 예외 + fail-open. "clean-bg 게이트 항상 ON" 정책을 인용할 때 반드시 함께 밝혀야 한다.
12. **`ctx_prior`는 배경 이미지 선택에 관여하지 않는다** — §6-2. "어느 normal"은 배경↔배경 히스토그램 교집합(`p_dv` vs `p_clean`), "어디에"만 `ctx_prior`다. `_image_compat_score`는 정의만 남고 런타임 미호출. "compat 행렬로 배경을 고른다"고 쓰면 틀린다.
13. **`ctx_prior`가 두 산출을 가리킨다** — §7-3. 배치 게이트 = `matrix_symmetric`(패치 단위, max-norm [0,1]), ROI_score = legacy `matrix`(이미지평균 확률, max<0.2). 스케일도 관측 단위도 다르다. "선택과 배치가 같은 값을 쓴다"고 쓰면 틀린다.
14. **`P_def`는 결함 국소 이웃이 아니라 이미지 전역 배경이다** — §4-1①. severstal 1600×256 이미지의 배경 패치 ~98개가 전부 같은 `k`를 상속한다. "결함 주변 컨텍스트"라고 쓰면 과장이다.
15. **`P_def` 가중은 이미지 균등이 아니라 배경 면적 비례다** — §4-1②. 결함이 큰 이미지일수록 기여가 작다.
16. **ε 발동 규모는 데이터셋별 0~6 cell** — §5-4. severstal 0건, leather 5~6건. 7번(하드코딩 상수) 방어 시 이 수치를 함께 제시하면 "순위 영향 제한적"이 실측으로 뒷받침된다.
17. **대칭은 `S_k` 안에서만 성립한다** — §5-7(a). `P_def=0, P_clean>0` cell은 행에 없어 런타임 **0.5**, `P_def>0, P_clean=0` cell은 행에 있어 **≈0**. 전혀 동등하지 않다.
18. **neutral 0.5는 중립이 아니라 관대하다** — §5-7(b). 관측 cell의 85~99%가 0.5 미만(severstal median 0.027)이라 미관측 타일이 footprint 평균을 끌어올린다. 런타임 조우율 severstal 0.1% / **aitex 최대 63.7%**. "미관측 = 중립"으로만 서술하면 편향 규모를 감춘다.
19. **대칭 항의 효과는 데이터셋 의존** — §5-6. severstal Spearman 0.97(거의 무변화) vs leather 0.72(최대 105계단 재배치). "대칭이 순위를 바로잡는다"를 severstal 결과로 뒷받침할 수 없다.
20. **severstal에서 `ctx_prior`는 `k`를 거의 구분하지 못한다** — §4-4. cluster 행 간 log `P_def` 상관 0.956~0.991. 어느 cluster를 붙이든 같은 자리를 추천한다. "결함 형태별로 적합한 배경을 고른다"는 aitex(r=0.07~0.71)에서는 성립하나 **severstal에서는 실증되지 않는다.**
21. **`P_def` 계수 범위가 연구 의도와 어긋난다 (미해결)** — §4-5. 의도는 결함 국소 배경, 구현은 이미지 전역 배경. 학습(전역 ~95타일)–추론(footprint 6타일) 스케일 불일치. 개선 전까지 §3.2.4를 "결함 주변 컨텍스트"로 서술하면 안 된다. → `.claude/.dev_note/aroma_ctxprior_localization_and_normalization.md`
22. **`P_clean` 곱셈이 매칭 논리가 아니다 (미해결)** — §5-8. `P_def`/`P_clean` = lift 순위와 `ctx_prior` 순위의 Spearman이 −0.28 ~ +0.12로 직교. 도입 경위상 검증된 lever는 patch-gran화·max-norm이고 곱셈 자체의 기여는 분리 입증되지 않았다. "결함 특징에 맞는 배경을 찾는다"로 서술할 근거가 현행 식에는 없다. → 동일 패치 노트 §2

---

## 10. 관련 문서

- **`Article/figure/script/[figure 3.2.4 2] placement_footprint.py`** — §6 시각화 생성 스크립트(운영 함수 import). 출력 `Article/figure/image/[figure 3.2.4 2] placement_footprint.png`, 사양 `.../script/[figure 3.2.4 2] placement_footprint.md`
- **`.claude/.dev_note/aroma_ctxprior_localization_and_normalization.md`** — 🔧 **§4-5 + §5-8 착수 패치 노트.** `P_def` 계수 범위 국소화(개선안 A1~A4) + 정규화 형태 재검토(B1~B4), 순서 의존·위험·미확인 사항 정리
- **`.claude/.dev_note/aroma_cleanbg_gate_cv2_dtype_failopen.md`** — §6-1 void 게이트 무력화 결함 상세·수정 시 결정 사항
- `.claude/.dev_note/aroma_compat_gate_clean-grounded_redesign.md` — SGM + patch-granularity 재설계(본 모델의 도입 경위), τ 사전스캔
- `.claude/.dev_note/aroma_table3_background_descriptor_definitions.md` — Table 3 지표 정의, `background_type` 미실행 발견
- `.claude/.dev_note/aroma_subtype_percentile_thresholds.md` — Table 4/4b percentile 전환, linearity 중복 증명
- `pivot_local_validation_20260711.md` — E1(clean-bg 매칭 도메인 조건부: aitex만 강함)
- `class_vs_cluster_validation_20260711.md` — cluster vs class 재포지셔닝
- `Article/figure/script/[figure 3.2.5 3] roi_selection_flow.md` — severstal linear-scratch 예시로 본 선택→배치 흐름
