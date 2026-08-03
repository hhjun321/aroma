# 논문 서술 ↔ 구현 간극 — 배치/배경선택 계열

## (성격: 논문 갱신 트랙. 구현 트랙은 `aroma_adjacent_context_bg_selection.md`)

`AROMA연구분석/Article/text` 확정본과 실제 구현이 어긋나는 지점을 모은다. 구현 논의 중 발견된 것이므로 근거는 그쪽 노트를 가리킨다.

**원칙: `Article/text` 수정은 결과가 온전히 도출된 뒤에 한다.** 아래 §2는 이미 반영한 것, §3은 대기 중인 것.

---

## 1. 논문이 서술하는 절차

| 절 | 서술 |
|---|---|
| **§3.2.3** | normal 이미지에서 ROI 추출 — Otsu → connected-component → 최소 면적 필터. 주변 텍스처를 5범주(smooth/directional/periodic/organic/complex)로 Table 3 percentile 캐스케이드 분류. continuity score · stability score 산출 |
| **§3.2.4 ①** | `ROI_score = 0.6·ctx_prior + 0.4·morph_prior`, top-K, 클러스터별 base quota, 클래스별 floor, 소스 crop 1회 사용 |
| **§3.2.4 ②** | 합성 전 각 ROI에 clean 배경 할당 — profiling 유래 histogram-matching cue |
| **§3.2.4 ③** | `ctx_prior(k,c) ∝ √(P_def·P_clean)`, ε=10⁻³, 행 max 정규화. 배치는 "64-pixel tiled neighborhood of the target ROI" 평가 |
| **§3.2.5** | 선택된 crop을 지정 bbox에 배치 |
| **§3.2.6** | quality ≥ 0.7 게이트가 clean-background 할당 **이전에** 배경 패치를 거르고 foreground void도 거부 |
| **§3.3.4** | 배경 선택 실험 — "histogram-intersection similarity to the ROI's context signature" |

---

## 2. 이미 반영한 수정 (2026-08-03)

이론·방법 서술만 손댔다. 실측값은 다운스트림 후.

| # | 절 | 수정 내용 |
|---|---|---|
| A1 | §3.2.4 ② | 배경 할당 cue를 3 → **4개**로. (iii) morphology-cluster compatibility 추가. class 축과 cluster 축이 다른 분할임을 명시(severstal 4 class vs 5 cluster, leather 5 vs 3). 가중치가 measured discriminative lift에서 유도되어 평탄한 신호는 가중치 0 |
| A2 | §3.2.4 ③ | **핵심 수정.** 수식 무수정. 호환성 행을 per-tile score가 아니라 **target context distribution** `q_k`로 소비한다고 재정의. `score(k,position) = Σ_c min(h(c), q_k(c))`, `h` = 링의 셀 히스토그램. 두 설계 근거를 이론으로 서술 — (1) footprint는 결함이 덮어쓰므로 ring을 읽는다 (2) 평균은 "가장 흔한 표면"으로 몰기 때문에 분포 매칭을 한다 |
| A3 | Figure 3.2.4-1 캡션 | 행이 target distribution으로 소비된다는 점 추가. severstal처럼 행이 거의 동일한 표면에서 평균은 붕괴하지만 분포 매칭은 자리를 구분한다는 설명 |
| A4 | Figure 3.2.4-2 캡션 | flow를 4-cue 배경 할당 + ring 매칭 기준으로 갱신 |
| A5 | §3.2.6 | 배치가 selection 시점에 결정되므로 void 거부도 거기서 적용됨을 명시. footprint에 void 타일이 있으면 후보 폐기, 링에서도 void 제외. 텍스처 에너지 기반 + 데이터셋별 저분위 floor(하드코딩 없음) |
| A6 | §3.3.4 | "ROI's context signature"가 구현과 불일치했다. 4개 cue를 정확히 열거하고, **세 히스토그램 cue는 전부 이미지 전역 분포를 비교하며 국소화는 배치 단계에서만 들어간다**고 명시 |

A6이 논문 내부 충돌(§3.2.4 ②의 "source overlap" vs §3.3.4의 "ROI's context signature")을 해소한다.

---

## 3. 미반영 — 남은 간극

### D1. §3.2.3 — Table 3 캐스케이드가 산출되지 않는다 ★

**2026-08-03 정정.** 처음에 "§3.2.3 전체가 미구현"으로 판단해 삭제했다가 **되돌렸다**. `background_type`은 실제로 소비된다.

```
roi_selection.quality_proxy(..., background_type)
  → SuitabilityEvaluator.matching_score(defect_subtype, background_type)
  → quality_score → roi_selected.json 기록
  → apply_quality_gate(candidates, min_quality)   후보 사전 필터
```

정확한 간극은 **값의 출처**다:

| 논문 | 구현 |
|---|---|
| Table 3의 per-dataset percentile 캐스케이드로 5범주를 **산출** | `--background_type` **CLI 인자의 데이터셋별 고정 상수** (기본 `directional`) |
| continuity score · stability score 산출 | **없음** — `roi_selection.py:109-111`이 직접 명시: *"continuity / stability / gram and a semantic background_type are absent from AROMA-native profiling output"* |
| ROI를 normal 이미지에서 Otsu+CC로 열거 | `roi_selected.json` 항목은 **결함 인스턴스 + context cell** 쌍. clean 이미지 좌표 없음. `generate_defects:373-400`의 Otsu+CC는 정상 이미지 최대 연결성분을 잡는 **foreground constraint**이지 후보 위치 열거가 아니다 |

**Table 3의 4 descriptor 중 3개는 프로파일링 문맥 특징과 실질 동일**하다 — LocalVariance, LBPEntropy(=`texture_entropy`), GradOrientEntropy(≈`orientation_consistency`). 다른 하나 `AutocorrPeak`만 AROMA 경로에 없다(논문도 frequency-energy와 구별된다고 적고 있다).

⇒ 서술 자체가 허구는 아니고 **산출 경로가 없는 것**이다. 조치 후보:
- (a) Table 3을 "이 캐스케이드로 라벨을 정한다" → "라벨은 데이터셋 단위로 지정하며, 지정 기준이 Table 3"으로 서술 완화
- (b) 캐스케이드를 실제로 구현해 서술을 참으로 만든다
- (c) continuity/stability 언급만 삭제

### D2. `MATCHING_RULES`가 하드코딩 표다 ★ 민감

`utils/suitability.py`의 subtype × background_type 5×5 상수:

| | smooth | directional | periodic | organic | complex |
|---|---|---|---|---|---|
| linear_scratch | 0.5 | **1.0** | 0.7 | 0.3 | 0.3 |
| elongated | 0.6 | 0.9 | 0.7 | 0.4 | 0.4 |
| compact_blob | **0.9** | 0.4 | 0.7 | 0.6 | 0.5 |
| irregular | 0.5 | 0.4 | 0.5 | 0.8 | **0.9** |
| general | 0.7 | 0.7 | 0.7 | 0.7 | 0.7 |

가중치 `W_MATCHING 0.4 / W_CONTINUITY 0.3 / W_STABILITY 0.2 / W_GRAM 0.1`도 하드코딩.

논문 §3.2 서두는 *"AROMA re-derives its defect–background compatibility model directly from dataset statistics"* 이고 §3.2.4는 *"automatically replaces CASDA's handcrafted compatibility matrix"* 라고 주장한다. **이 표가 정확히 그 handcrafted matrix다.**

완화 요인 — `roi_selection`은 `compute_suitability`가 아니라 `matching_score`만 쓴다(continuity/stability/gram이 없으므로). 그리고 `apply_quality_gate`는 `min_quality <= 0`이면 **비활성**이다.

**선결 확인**: 실제 실행의 `--min_quality` 값. 0(기본)이면 게이트 무동작이고 `quality_score`는 JSON 기록용에 그쳐 실질 영향이 없다. >0이면 하드코딩 표가 후보를 실제로 거른다. severstal `roi_selected.json` 샘플이 `quality_score: 1.0`(linear_scratch × directional)이었다 — 전체 분포로 판정 가능. **미확인.**

### D3. §3.2.4 ③ "the resulting compatibility score determines the placement location"

A2로 서술은 갱신했으나, **현행 구현**(`_positive_place`)은 여전히 다르다 — 정상 이미지 **전역**을 32px stride로 훑고(최대 4096 후보) **top-8 무작위 샘플**한다. 오프라인 ring 경로(`--site_selection ring`)를 실제 실험에 쓰기 전까지 논문 서술과 구현이 어긋난 상태다.

### D4. §3.2.4 ① ROI 후보 선택은 미측정

`ROI_score = 0.6·ctx_prior + 0.4·morph_prior`. P1 벤치는 **자리 선택만** 쟀다. ROI 후보 선택에서 `ctx_prior`가 어떤지는 모른다. 가중치 0.6/0.4도 하드코딩이다.

### D5. §4 기전 귀속 ★ 가장 민감

`ctx_prior` footprint 평균이 자리 선택에서 random보다 나쁘다(severstal P1 −0.270). 그런데 exp4v2에서 AROMA는 Random과 비슷하거나 나았다. 세 가지가 겹친다:

1. `_positive_place`가 argmax가 아니라 **top-8 무작위 샘플** → 나쁜 신호가 희석
2. τ 게이트가 거의 발동하지 않는다 (prescan accept 0.75)
3. 실제 이득은 **void 거부 게이트** 몫으로 이미 관찰돼 있다 (메모리 `project_cleanbg_gate_policy`)

⇒ downstream 수치는 유효하되 **그 수치를 만든 기전이 논문이 지목한 기전이 아닐 가능성**이 높다. §4 서술 재검토 필요.

---

## 4. 반영 계획

| 항목 | 대상 절 | 조치 | 시점 |
|---|---|---|---|
| D3 | §3.2.4 ③ | ring 경로를 실험 기본으로 승격하면 해소 | 다운스트림 후 |
| D5 | §4 | 기전 귀속 재검토 | 다운스트림 후 · **민감** |
| D1 | §3.2.3 | 3안 중 택일 | 별건 |
| D2 | §3.2 서두 · §3.2.4 | `--min_quality` 확인 후 판단 | **선결 확인 필요** |
| D4 | §3.2.4 ① | ROI 선택 P1 측정 후 | 미착수 |

---

## 5. 관련 문서

- `.claude/.dev_note/aroma_adjacent_context_bg_selection.md` — 구현 트랙. §2의 근거 전부
- `.claude/.dev_note/aroma_ctxprior_localization_and_normalization.md` — `matrix_symmetric` 정의 개선안(P1으로 기각)
- `AROMA연구분석/aroma_core_compatibility_model_20260729.md` — `ctx_prior` 전체 모델·정직성 목록
