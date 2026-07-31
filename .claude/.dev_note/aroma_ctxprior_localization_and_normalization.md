# `ctx_prior` 재설계 — `P_def` 국소화 + 정규화 형태 재검토

## (성격: 설계 제안 — 미착수. 실측 근거 포함)

연구 핵심은 **"결함 국소 배경의 특징을 clean pool에서 찾아 그 자리에 합성"**이다. 현행 `ctx_prior(k,c) ∝ √(P_def·P_clean)`는 두 지점에서 이 정의와 어긋난다.

| # | 어긋남 | 절 |
|---|---|---|
| **A** | `P_def`가 결함 국소가 아니라 **이미지 전역 배경**을 센다 | §1 |
| **B** | `P_clean`을 **곱해서** 흔한 자리를 선호한다 — 매칭이 아니라 가용성 가중 | §2 |

A가 B의 선행 조건이다(§4 순서). 배경 분석은 `AROMA연구분석/aroma_core_compatibility_model_20260729.md` §4-1·§4-4·§4-5·§5-7 참조.

---

## 1. 문제 A — `P_def` 계수 범위

### 1-1. 현행 동작

`distribution_profiling.py:_context_worker`가 결함 이미지의 **모든 비-결함 패치**(mask 과반 타일만 제외)를 방출하고, `_build_symmetric`이 그것을 전부 그 이미지의 `k`로 계수한다. 결함이 좌측 끝에 있어도 1600px 반대편 패치가 `P_def(k,·)`에 들어간다.

### 1-2. 결과 — cluster를 구분하지 못한다 (실측 2026-07-31)

cluster 행끼리의 log `P_def` 상관:

| 데이터셋 | min | median | max |
|---|---|---|---|
| **severstal** | **0.956** | **0.973** | **0.991** |
| mvtec_leather | 0.890 | 0.890 | 0.958 |
| kolektor | 0.451 | 0.741 | 0.849 |
| mtd | 0.507 | 0.632 | 0.795 |
| aitex | 0.070 | 0.330 | 0.709 |

severstal 5개 행이 사실상 동일 분포다 ⇒ **어느 cluster를 붙이든 같은 자리를 추천한다.**

같은 cell `0_0_0_1_0`에서 전 cluster가 clean 대비 enriched:

| k | `P_def(k,c)` | `P_def/P_clean` |
|---|---|---|
| 0 | 0.146049 | 2.83 |
| 1 | 0.158574 | 3.07 |
| 2 | 0.072370 | 1.40 |
| 3 | 0.071595 | 1.39 |
| 4 | 0.101137 | 1.96 |

"cluster 1이 매끈한 배경을 선호"가 아니라 **"severstal 결함 이미지 전반이 정상 이미지보다 배경이 매끈"** — pool 수준 편향이다.

원인: severstal 이미지 1장이 평균 26.8 cell에 흩어지므로 cluster별 차이가 전역 통계에 희석된다. AITeX(타일 단위, r=0.07~0.71)는 갈린다.

### 1-3. 학습–추론 스케일 불일치

| 단계 | 관측 범위 |
|---|---|
| 학습 (`P_def` 계수) | 이미지 **전역** (severstal ~95 타일) |
| 추론 (`_positive_place` footprint) | crop **footprint** (160×96 → **6 타일**) |

`aroma_compat_gate_clean-grounded_redesign.md` §2 발견 E가 build(64px) vs query(crop) 불일치를 지적하고 타일링으로 고쳤으나, **build 측의 공간 범위(전역 vs 국소)는 그대로 남았다.** 본 노트가 그 잔여 축이다.

### 1-4. 개선안

| 안 | 계수 대상 | 비고 |
|---|---|---|
| A1. 반경 링 | 결함 bbox를 `R`만큼 팽창한 영역과 겹치는 배경 타일 | `R = _COMPAT_TILE`(1링)이 최소안 |
| A2. 인접 타일만 | bbox에 접하는 8-이웃 타일 | 가장 좁음, 표본 급감 |
| A3. 거리 가중 | 전 배경 타일 유지, 결함 중심거리로 감쇠 가중 | hard cut 없음, 표본 유지 |
| **A4. footprint 정합** | **결함 bbox 크기 창을 결함 중심에 두고 그 안의 배경 타일** | **추론 스케일과 일치 — 권장** |

A4가 `_positive_place` 3단계 판정 단위와 정확히 맞물린다.

### 1-5. 구현 메모

- `_context_worker`는 이미 `patch_xy`를 방출한다(`distribution_profiling.py:387`).
- 결함 bbox는 `morphology_features.csv`의 `defect_bbox`에 있다(`:49`).
- ⇒ **거리·포함 판정에 필요한 정보가 이미 산출돼 있다.** 신규 측정 불필요.
- `_build_symmetric`에 반경 파라미터를 추가하는 additive 방식이면 legacy `matrix` 경로와 분리 유지(현행 SGM 도입 방식과 동일).
- 재-profiling 필요. `matrix_symmetric`만 변하므로 `--compat_mode defect` 경로 무영향.

### 1-6. 위험

| # | 위험 |
|---|---|
| 1 | **표본 급감.** severstal k1 82,025 → 1링이면 이미지당 ~10-20 타일. support cell(현 192~197)도 감소 |
| 2 | **aitex 악화.** 이미 support 22~60으로 희소 → §3-2의 neutral 0.5 조우율(현 14.2~63.7%)이 더 오른다. **데이터셋별 사전 스캔 후 반경 확정 필수** |
| 3 | `P_clean`은 정상 이미지에 결함이 없어 같은 방식으로 좁힐 수 없다 → 관측 범위 비대칭 발생. **§2에서 해소됨**(비대칭은 정상) |

---

## 2. 문제 B — `P_clean`을 곱하는 것이 맞는가

### 2-1. 역할 재정의

연구 핵심을 매칭으로 놓으면 세 항의 지위가 이렇다:

| 항 | 지위 |
|---|---|
| `P_def(k, ·)` | **질의(query)** — 찾고자 하는 표면 프로파일 |
| clean pool | **탐색 대상(corpus)** |
| `P_clean(c)` | corpus의 **주변 분포** — 질의의 한쪽 항이 아님 |

⇒ **§1-6 위험 3(관측 범위 비대칭)은 문제가 아니다.** query는 국소여야 하고 corpus 통계는 전역이어야 한다. 같은 방식으로 잴 이유가 없다.

정보검색 비유: `P_def` = query term 분포, `P_clean` = document frequency. SGM은 `query × DF`다 — **TF-IDF와 반대 방향.** IDF 논리대로면 나눠야 한다.

`lift(k,c) = P_def(k,c) / P_clean(c)` = "정상 대비 이 표면이 얼마나 결함과 결합하는가". 이게 "특징을 찾는다"에 부합하는 양이다.

### 2-2. 실측 — SGM 순위와 lift 순위가 직교한다 (2026-07-31)

Spearman(lift, `ctx_prior`), cluster별:

| 데이터셋 | 범위 |
|---|---|
| severstal | −0.084 ~ +0.121 |
| aitex | −0.256 ~ +0.049 |
| mtd | −0.276 ~ +0.040 |
| kolektor | −0.272 ~ +0.101 |
| mvtec_leather | −0.077 ~ +0.008 |

**전부 0 근처 또는 음수.** severstal k=1 개별 cell:

| cell | `P_def` | `P_clean` | **lift** | lift 순위 | `ctx_prior` | ctx 순위 |
|---|---|---|---|---|---|---|
| `0_0_0_1_0` | 0.15857 | 0.051667 | 3.07 | 11 | 1.0000 | 1 |
| `1_1_0_2_2` | 0.01919 | 0.052596 | **0.37** | **161** | 0.3588 | **3** |
| `0_0_2_0_1` | 0.03249 | 0.030895 | 1.05 | 92 | 0.3565 | 4 |
| `1_0_2_1_1` | 0.00445 | 0.000539 | **8.26** | **2** | 0.0316 | **92** |
| `1_0_1_1_0` | 0.00182 | 0.000469 | 3.87 | 5 | 0.0222 | 119 |

- `1_1_0_2_2`: lift 0.37 — 결함 쪽에 정상보다 **덜** 흔한데 SGM이 3위로 올린다.
- `1_0_2_1_1`: lift 8.26(2위) — cluster 1에 가장 특이한 표면인데 SGM이 92위로 내린다.

노이즈 아님 — `1_0_2_1_1`은 결함 ~365패치 / clean ~318패치.

⇒ **연구 의도 기준으로 SGM이 정반대로 정렬하는 사례가 실재한다.**

### 2-3. 그럼 `P_clean`은 왜 들어갔나 — 도입 경위

`aroma_compat_gate_clean-grounded_redesign.md` 확인. 목적은 **over-accept 치료**(게이트 무력화)였고 검증된 lever는 둘이다:

| lever | 효과 | 근거 |
|---|---|---|
| `P_def` **patch-granularity화** | support KEY 확장 (leather 5 → ~191 cell) | 동 문서 §3-1, "clean_dist만 patch-gran화하면 KEY를 안 늘려 over-accept 유지" |
| **max-norm** | cluster별 raw compat_max 3~8배 편차 해소 → 단일 τ 성립 | 동 문서 §5-2 |

**`P_clean` 곱셈 자체의 기여는 분리 입증되지 않았다.** SGM은 3개 설계안 중 `deficit_conflict=false`·`realism_break=false`를 만족해 채택됐고, 시맨틱은 "결함군집 AND 순수 normal 둘 다 흔한 cell만 고득점" — **가용성 논리**지 매칭 논리가 아니다.

### 2-4. 가용성 항의 위치가 뒤바뀌어 있다

| 단계 | 가용성 항이 정당한가 | 현재 |
|---|---|---|
| ROI 후보 열거 (`roi_selection.py:429`) | **정당** — pool에 없는 cell 추천은 슬롯 낭비 | legacy `matrix` 사용, SGM 미적용 (동 문서 §4에서 **의도적 배제**: 후보 폭증 + selection↔placement 직교) |
| 배치 (`_positive_place`) | **부당** — 타일이 이미 손에 있어 가용성 확보됨 | SGM 적용 |

**가용성 항이 가장 정당한 곳엔 없고, 가장 부당한 곳에 있다.**

### 2-5. 개선안

| 안 | 식 | 얻는 것 | 잃는 것 |
|---|---|---|---|
| B1. `P_def` 단독 | `P_def(k,c) / max_c P_def(k,c)` | 의도 직결, 최소 변경 | §1-2 pool 편향 잔존 |
| B2. **lift** | `(P_def+ε) / (P_clean+ε)` | pool 편향 제거. **중립점이 1.0으로 자연스러움** → §3-2 neutral 0.5 자의성 동시 해소 | 희소 cell 분산 폭발 → count 기반 shrinkage 필수 |
| B3. cluster-특이 lift | `P_def(k,c) / P_def(·,c)` | §1-2 cluster 무차별성 **직접 겨냥** | clean pool 가용성 완전 무시 |
| **B4. 역할 분리 (권장)** | 점수 = B1~B3 중 하나, `P_clean`은 **feasibility 필터**로만 (곱셈 아님) | 각 항이 제 자리에서 작동 | 구현 2곳(selection·placement) |

---

## 3. 함께 걸리는 기존 결함

### 3-1. void 게이트 fail-open

`aroma_cleanbg_gate_cv2_dtype_failopen.md`. `P_def`/`P_clean` 계수 자체에는 영향 없으나(`_context_worker`는 void 필터를 안 쓴다), 배치 4단계와 배경 이미지 선택(`_dv_bg_hist`/`_normal_tile_cells`)에 걸린다. 본 재설계 검증 시 void율이 0%인 상태임을 전제로 읽어야 한다.

### 3-2. neutral 0.5가 관측 cell 대부분을 압도

행 max 정규화 탓에 관측 cell의 **85~99%**가 0.5 미만이다. 미관측 cell은 `.get(cell, 0.5)`로 그보다 높은 점수를 받는다.

| 데이터셋 | row median `ctx_prior` | `<0.5` 비율 | 런타임 조우율(clean 질량이 row 밖) |
|---|---|---|---|
| severstal | 0.037 | 98~99% | 0.0~0.1% |
| mvtec_leather | 0.053 | 98% | 0.9~11.8% |
| mtd | 0.141 | 90~94% | 0.8~23.3% |
| kolektor | 0.265 | 85~94% | 1.8~25.0% |
| **aitex** | 0.053 | 90~95% | **14.2~63.7%** |

배치가 "결함 이미지에서 한 번도 관측되지 않은 배경" 쪽으로 끌린다. **B2(lift)를 채택하면 중립점이 1.0으로 정의되어 이 자의성이 함께 해소된다.**

---

## 4. 착수 순서 (순서 의존)

1. **`P_def` 국소화** (§1-4 안 A4)
2. **cluster 행 상관 재측정** — severstal 0.97이 내려가는지. **안 내려가면 3단계 이후 무의미** (국소화 전에는 어떤 정규화도 pool 통계를 잴 뿐)
3. **정규화 형태 결정** (§2-5) — 1·2 결과 보고 판단
4. **τ 재캘리브레이션** — 스케일이 바뀌므로 필수. 데이터셋별 사전 스캔(발동률 CPU 측정) 후 확정
5. Colab 재-profiling → 합성 → 다운스트림 비교

각 단계에서 aitex를 관측 지표로 삼는다 — support가 가장 희소해 열화가 먼저 드러난다.

---

## 5. 미확인

- 실제 실험 실행 커맨드의 `--strategy` / `--compat_mode` / `--compat_threshold` 조합. `--compat_mode` 기본값이 `defect`, `--compat_threshold` 기본값이 `0.0`이라 **`matrix_symmetric`이 실제로 소비됐는지 자체가 미확인.** 본 재설계의 영향 범위를 확정하려면 선행 확인 필요.
- 국소화 후 표본 수 — §1-6 위험 1의 실제 규모.

---

## 6. 관련 문서

- `AROMA연구분석/aroma_core_compatibility_model_20260729.md` — `ctx_prior` 전체 모델. 본 노트의 §1 = 그 문서 §4-1·§4-4·§4-5, §2 = §5-1·§5-3·§5-6, §3-2 = §5-7
- `aroma_compat_gate_clean-grounded_redesign.md` — SGM 도입 경위, 발견 E(스케일 불일치), roi_selection 미전환 근거
- `aroma_cleanbg_gate_cv2_dtype_failopen.md` — void 게이트 fail-open
- `aroma_subtype_percentile_thresholds.md` — linearity 중복 증명(형태 특징 쪽 별건)
