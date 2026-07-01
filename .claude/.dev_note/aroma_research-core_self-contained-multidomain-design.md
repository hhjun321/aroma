# AROMA self-contained 다중도메인 증강 프레임워크 — 설계 spec

## (성격: 연구 전략·설계 노트 — 코드 패치 노트 아님)

> 이 노트는 코드 구현 노트가 아니라 AROMA 핵심 연구의 **파이프라인 설계 + 비교 방법론(Approach A) spec**이다. [[aroma_research-core_thesis-and-compounding]]가 thesis scoping과 compounding 실험을 다뤘다면, 본 노트는 그 위에서 "AROMA = self-contained 다중도메인 증강 프레임워크"를 어떻게 구성하고 무엇을 방어할지 확정한다. 관련: [[aroma_research-core_thesis-and-compounding]], [[aroma_exp4v2_foreground-void-rejection]], [[aroma_exp2_roi-quality]], [[aroma_severstal_flat_diagnosis_and_direction]].
>
> 출처: `design-review` 워크플로우(9 에이전트: grounding 3 + analyze 3 + redteam 2[1 fail] + synth 1) + 사용자 대면 검토(5개 섹션 순차 승인). 워크플로우가 잡은 2개 CRITICAL 발견을 코드로 직접 재검증한 뒤 설계에 반영.

---

## 개요

AROMA를 **self-contained 다중도메인 증강 프레임워크**로 정의한다: (1) 도메인별 real 결함으로 ControlNet 학습, (2) 학습 ControlNet으로 결함 생성, (3) AROMA type-aware ROI 선택이 **어디/무엇**을 배치할지 결정, (4) ROI 영역에 Poisson blend → 증강셋. 논문 일차 축 = **다중도메인 일반성(breadth) = 적용성(applicability) + ROI 품질(quality)**.

이 설계의 핵심 동기는 [[aroma_research-core_thesis-and-compounding]]에서 확인된 문제 — CASDA와 파이프라인을 억지로 병합하면 confound가 발생 — 를 피하고, AROMA를 CASDA에 의존하지 않는 자체 파이프라인으로 재구성하는 것이다. CASDA는 Severstal 전용 참조로만 남긴다(비교자·병합 아님).

---

## ★ 설계 전 검증된 2개 CRITICAL 발견 (코드행 근거)

### 발견 ① "self-contained"는 오늘 기준 사실이 아니다 → P0로 해소

`aroma_to_casda_roi.py` 헤더(3–24행) 직접 확인: 이 파일은 AROMA `roi_selected.json` → **CASDA `roi_metadata.csv`로 넘기는 FORWARD adapter**다. ControlNet 학습에 필요한 `hint`(R=mask/G=structure/B=texture)와 `prompt`는 **CASDA의 `prepare_controlnet_data.py` / `ControlNetDatasetPackager`가 재생성**한다(헤더 명시: "do NOT modify CASDA" = CASDA는 외부 레포). `def generate_hint_image` 정의는 AROMA scripts 어디에도 **없다**(grep 0건; 15행 호출은 docstring 내 CASDA 설명).

→ **open question 종결**: analyze:pipeline 에이전트가 "hint 생성기 존재"로 본 것은 오판, feasibility 분석("부재")이 맞음. 오늘 "AROMA ControlNet 학습"의 유일한 실경로는 CASDA 경유. 이 상태로 "self-contained, 병합 아님"을 논문에 쓰면 리뷰어가 이 파일 헤더만 읽어도 사실 오류로 잡음.

→ **결정(사용자 승인)**: **P0 self-contained train.jsonl+hint 빌더를 신규 구현**해 CASDA 의존을 제거한다. "AROMA가 자체 ControlNet 파이프라인 소유"를 진실화.

### 발견 ② type-awareness가 diffusion에 도달하지 않는다 → P1로 해소

`stage4_diffusion_synthesis.py`가 `control_pil`을 seed당 1회(248–251행) 계산해 모든 ROI에 재사용(267행). 다중 ROI가 동일 Canny를 받아 각 ROI의 morphology(linearity/solidity/aspect_ratio)가 생성 조건화에 도달하지 못함. → **P1(per-ROI control) 구현 전에는 ControlNet 경로로 "type-aware ROI가 무엇을 생성할지 결정"을 주장 불가.** ([[aroma_research-core_thesis-and-compounding]] ④ 치명결함1과 동일 이슈)

---

## 섹션 1 — 목표 / 비-목표

### 목표
- 4개 도메인(steel/texture/object/color)에 동일 파이프라인 적용성 실증
- deficit-aware ROI 선택이 random 대비 intrinsic coverage(morphology/context/rare-pair/entropy/gini) 우위임을 **통계적으로** 방어
- downstream 검출 이득은 **정직 보고**(real 희소 도메인만 이득, real 충분 도메인은 무해/노이즈)

### 비-목표
- CASDA와의 병합/compounding (참조로만)
- ControlNet **생성 품질** 개선 주장 (AROMA는 배치·선택만; 생성 엔진은 표준 ControlNet)
- GPU 비용 실측 (load-test 정책상 추정만)

---

## 섹션 2 — 파이프라인 설계 (step 매핑 기준)

실제 `AROMA연구분석/colab_execute/step*_execute.md` 실행 가이드 기준 매핑:

| Step | 스크립트 | 이번 설계 | 자원 |
|------|---------|----------|------|
| **Step 1** `compute_complexity.py` | MCI/CCI → 정책 | **무수정 재사용** | CPU |
| **Step 2** `prompt_generation.py` | cell별 프롬프트 | **무수정 재사용** (단, `PromptGenerator` 유틸을 P0 빌더가 재사용) | CPU |
| **Step 3** `roi_selection.py` | `roi_selected.json` (AROMA 핵심) | **무수정 재사용** | CPU |
| **Step 3.5 (신규, P0)** `build_train_jsonl.py` | real 결함 crop + mask/morphology + `PromptGenerator` → **hint PNG 신규생성 + train.jsonl** | **신규 구현** — CASDA `prepare_controlnet_data.py` 의존 대체 | CPU |
| **Step 3.7 (신규, 재사용)** `train_controlnet.py` | 도메인별 ControlNet 학습 | **스크립트 존재, 배선만 신규** | GPU |
| **Step 4** `generate_defects.py` | copy_paste(완료) / controlnet(stub :777) | **controlnet_synthesis stub 구현 + P1 per-ROI control** | GPU(controlnet) / CPU(copy_paste) |
| **Step 4 blend** `_context_aware_composite` | Reinhard + `seamlessClone`(Poisson) | **무수정 재사용** (`blend_mode=seamless`) | CPU |

**정확한 표현**: "Step 1–3 = 무수정 재사용, Step 3.5·3.7 = 신규 삽입(P0 빌더 / 학습 배선), Step 4 = controlnet 경로 개선." Tier 0(copy_paste+Poisson)만 볼 때는 Step 4 copy_paste 경로가 이미 완성돼 "Step 1–4 그대로"지만, ControlNet 서사를 실으려면 3.5/3.7이 반드시 신규로 들어간다.

### P0 빌더 상세 (신규 핵심)
각 real 결함 → (a) `image_path`에서 `defect_bbox` crop → target PNG, (b) mask+morphology → **hint PNG 신규 생성기(R=mask/G=structure/B=texture 3채널)**, (c) `PromptGenerator.generate_prompt(...)` → prompt/negative_prompt, (d) `train.jsonl` 한 줄 append. 도메인별 별도 `train.jsonl`. 출력 스키마 = `train_controlnet.py:79-84`의 `{target, hint, prompt, negative_prompt, source}`.
재사용 가능: `PromptGenerator`(도메인중립), `DefectCharacterizer.analyze_defect_region(mask)`(morphology 즉석 산출 fallback), `aroma_to_casda_roi.py`의 crop-align 로직(`_make_crop_pair`, `_join_metrics`).

### P1 필수 개조 (경로 무관)
- **결함1**: `control_pil`을 placement 루프 안 per-ROI 계산으로 이동(각 ROI의 defect crop+mask/morphology에서).
- **결함2**: `seed=seed+i`(loop-index) → **content-hash seed** `seed=int(sha256((image_id+cell_key+str(defect_bbox)+str(rep_idx)).encode()).hexdigest()[:8],16)`. morphology(control)와 noise(seed) 직교 분리 + 순서 무관 결정론.
- **결함3**: prompt가 `placements[0]`만 사용 → placement별 prompt.

### 권장 경로A (생성 통합 진입점)
`generate_defects.controlnet_synthesis` stub(:777, NotImplementedError) 구현 — `copy_paste_synthesis`와 동일 시그니처(`roi_entry, normal_image_path, output_path, rng, mask_output_path`)라 `run()`의 `_SYNTHESIS_METHODS` dispatch에 자연 편입. 학습 ControlNet으로 결함 patch 생성 → Step 4 `seamlessClone` Poisson으로 배치. "생성=ControlNet, 배치=AROMA ROI, blend=Poisson" 서사에 정확 부합. **ROI별 독립 patch 생성이라 per-ROI 미흐름(결함1)을 구조적으로 회피.**
- 경로B(대안): `stage4_diffusion_synthesis.py` inpaint 직접 사용. diffusion 자체가 경계 부드러움이라 Poisson 불요이나 annotations 스키마·seed 결정론을 stage4에 재이식 필요. → 이번엔 경로A 우선.

---

## 섹션 3 — 비교 방법론 (Approach A)

### 일차 층: intrinsic ROI-품질 (`exp2_roi_quality.py`) — CPU, 다중도메인, ControlNet 불요
- arm: `aroma`(sampling_strategy=deficit_aware) vs `random`. 둘 다 공유 `roi_candidates.json` 대비 스코어링. budget = min(aroma,random) 균등화(`_equalize_budget`) → 품질(물량 아님).
- 지표 5개(`compute_metrics`): morphology_coverage, context_coverage, **rare_pair_coverage(deficit>0, load-bearing)**, entropy, gini.
- 도메인 세트: mvtec_carpet/leather(texture), mvtec_cable/hazelnut(object), visa_cashew/pcb(object·color), isp_LSM_1. 패밀리당 ≥2. Severstal = 참조만.
- ⚠️ **치명적 측정 결함(P0 방법론)**: exp2가 도메인당 **단일 결정론적 delta(n=1)만 산출** — p_value/bootstrap/wilcoxon/std/n_seed 없음(grep 확인). random branch(`roi_selection.py` L1396-1399)는 단일 고정 seed. AROMA deficit_aware는 결정론적(blake2b는 sort-tiebreak jitter일 뿐 RNG 없음)이므로 **random arm만 ≥5 distinct seed 재실행** → 도메인·지표별 mean±std + 도메인 세트 전반 aroma-vs-random-mean **paired Wilcoxon**. 이거 없이는 "aroma>random"은 통계적으로 방어 불가. rare_pair_coverage의 p75 collapse 버그도 함께 수정(deficit>0 직접 사용). [[aroma_exp2_rarepair-deficit-nonzero-fix]]

### 이차 층: 동일엔진 downstream (`exp4_v2_supervised_detection.py`) — 정직 맥락 (corroborating 아님)
- 엔진 parity 확인: aroma/random synthetic 모두 동일 `generate_defects.py` copy_paste_synthesis + Poisson, blend_mode/reject_clean_bg/n_per_roi/seed 동일, 차이는 `--roi_dir`(roi_selected.json)만 → ROI-효과 clean attribution.
- arm(`ALL_CONDITION_KEYS` L138): baseline(real-only floor) / random-ROI+엔진 / aroma-ROI+엔진. ROI-효과 = aroma vs random, baseline = 바닥.
- shippable 엔진 = copy_paste+Poisson만. ControlNet arm은 P0/P1 전까지 미래 tier caveat.
- **정직 보고 (누적 사실 준수)**: aug>baseline는 real 희소 도메인만, Severstal aug≤baseline(real 충분→무해/노이즈), carpet만 aroma>random 유의·leather 역전, rare-oversample backfire(aroma c2 5×→c2 최악). downstream을 corroborating으로 제시 **금지**, 엄격히 secondary/contextual.

### CASDA — Severstal 참조만
`roi_metadata.csv` 파이프라인이 Severstal 전용(`extract_rois` 단일도메인) → 참조 행 + "ROI 추출이 도메인 종속이라 steel 밖 확장 불가" 구조적 한계 한 줄. **cross-domain 비교자·병합 절대 금지.** exp4v2 casda arm parity 이미 붕괴(empty-mask budget drop) 함께 표기. ([[aroma_exp4v2_casda-condition]])

---

## 섹션 4 — Claim Scoping (방어선)

### 방어 가능 (evidence-backed)
1. "AROMA는 도메인별 real 결함 + type-aware ROI 선택을 4도메인에 동일 파이프라인으로 적용하는 self-contained 프레임워크" — Tier 0 + exp2로 오늘 실증(P0 빌더 구현 시 ControlNet 포함 self-contained).
2. "deficit-aware ROI가 동일 budget에서 random보다 rare-pair/morphology/context coverage 높음" — exp2 5지표. **단 ≥5 seed + paired Wilcoxon 후에만.**
3. "CASDA는 ROI 추출이 Severstal 종속이라 구조적으로 steel 밖 확장 불가; AROMA는 벗어남."
4. "Poisson+AROMA ROI는 real 희소 도메인서 baseline 이득" — 정직 보고 범위(Severstal은 real 충분→무해/노이즈로 함께 보고).

### 방어 불가 / over-claim 금지
- ❌ "AROMA가 CASDA를 다중도메인으로 확장·개선했다" — P0 전에는 유일 학습 경로가 CASDA 경유(발견 ①). P0 구현 후에만 성립.
- ❌ "type-aware ROI가 무엇을(morphology) 생성할지 조건화한다" — P1(per-ROI control) 전에는 배치+noise에만 귀속(발견 ②).
- ❌ "AROMA가 ControlNet 생성 품질을 개선" — 엔진은 표준 ControlNet, AROMA는 배치·선택만.
- ❌ "downstream이 breadth를 입증" — 이차/맥락. Severstal aug≤baseline, leather 역전, rare-oversample backfire 정직 보고.
- ❌ Tier 1 해석: "ROI 효과 없음" vs "생성 OOD 품질저하"를 시각 pilot 없이 구분 불가. zero-shot Canny fallback을 "도메인별 학습"으로 오기 금지.

### ⚠️ redteam hole (t: a→m, HIGH)
**적용성(applicability)을 방법론적 유의성(significance)과 등치 금지.** AROMA가 4도메인에 "돌아간다"(구조적, 오늘 성립)와 "random보다 유의하게 낫다"(통계적, ≥5 seed + paired test 후)는 별개. 현 exp2 n=1·단일 seed random은 applicability만 보이고 significance는 못 봄. 서사에서 둘을 명시적으로 분리 진술하고, 후자는 significance machinery 완비 전까지 **"예비/미검증"** 표기.

---

## 섹션 5 — Scope · Tier · 우선순위

### 우선순위 (verified-need)
- **P0** = AROMA self-contained `train.jsonl` + hint-image 빌더 (신규 prepare 스크립트: real 결함 crop→target, mask/structure/texture→3ch hint, PromptGenerator→prompt). 없으면 CASDA 경유 외 AROMA ControlNet 학습 경로 자체가 없음. **순수 CPU-side glue, GPU 무관.**
- **P1** = stage4/generate per-ROI control-image flow + content-hash seed. **순수 CPU-side glue.**
- **P2** = generate_defects `controlnet_synthesis` stub un-stub (경로A).
- **P3** = content-hash seed를 Tier 0에도 바인딩 (Tier-1 arm 비교 필요 시).

### Tier 전략 (evidence-vs-cost)
- **Tier 0** (copy_paste+Poisson) — 오늘 실행 가능, CPU, 4도메인, 결정론적, 일차 breadth 증거. intrinsic ROI-품질(exp2)은 합성 자체 불요.
- **Tier 1** (pretrained/zero-shot Canny ControlNet inpaint = stage4 as-is + P1) — 중간 GPU, 비-steel OOD 위험, **시각 pilot gate 필수**. 주의: stage4는 `controlnet_model` 미지정 시 `lllyasviel/sd-controlnet-canny` fallback(zero-shot) → "도메인별 학습" 아님. 제안 서사 실증하려면 반드시 Tier 2 산출물을 `--controlnet_model`로 전달.
- **Tier 2** (도메인별 fine-tune) — 최고 비용, **P0 빌더 선행 필요**, 최고 품질이나 carpet/leather/VisA 결함-crop 희소가 병목. Tier 1 도메인별 시각 pilot으로 게이트.

### 비용 (추정만 — GPU 실측 정책 금지)
ControlNet SD1.5 fine-tune/도메인이 지배 비용. Colab T4/A100 대략 수백~2k step/도메인 × 4도메인, 도메인당 single-digit GPU-hour(512px fp16, 기존 SNR/cosine/early-stop). **진짜 리스크 driver = carpet/leather/VisA 결함-crop 희소성(수십~저수백 인스턴스), compute 아님.**

---

## 영향도 분석

### 이 설계가 변경/신설하는 상태
- **신규 파일**: `build_train_jsonl.py`(P0) + hint 생성기(신규 모듈 또는 함수).
- **수정**: `generate_defects.py`(`controlnet_synthesis` stub 구현 + per-ROI control), `stage4_diffusion_synthesis.py`(경로B 택 시 per-ROI control + seed), `exp2_roi_quality.py`(random seed sweep + rare_pair p75 버그), 신규 집계 스크립트(paired Wilcoxon).
- **신규 데이터**: 도메인별 `train.jsonl`, hint/target PNG, 도메인별 `controlnet_model` 디렉터리, 도메인별 synthetic set.

### 그 상태를 전제로 동작하는 기존 로직
- Step 1–3(`compute_complexity`/`prompt_generation`/`roi_selection`)은 무수정 → 회귀 0.
- exp4v2 harness는 condition-agnostic(synthetic_dir만 교체) → ControlNet arm 추가 시 harness 무변경.
- `generate_defects.copy_paste_synthesis`(Tier 0)는 단일 `rng=random.Random(seed)`(:1179) 순서 의존 결정론 — Tier 0 clean, arm 비교 가능. controlnet 추가 시 GPU generator seed도 content-hash 바인딩 필수(안 하면 재실행/arm 순서 변경 시 confound).

### 신규 코드가 만드는 상태의 하위 호환
- P0/P1/P2는 모두 **신규 경로 추가**이지 기존 copy_paste 경로 변경 아님 → Tier 0 재현성 보존.
- ⚠️ **stale-grounding 정정**: class-balance Fix1(`_stratified_compat`, `roi_selection.py:1186`) 이제 구현됨, 단 `--class_floor` 게이트 뒤(default False:1346,1633) → compounding/compatibility run은 반드시 `--class_floor` 전달(안 하면 class3 기아 재발). deficit_aware는 이미 `_stratified_pair_aware` 보유. composed_to_exp4v2 bbox>=50 parity filter(`_mask_has_valid_bbox`)는 uncommitted(유실 위험) — self-contained Tier 0에는 불요, CASDA 참조 arm에만 필요.

---

## 미확정 사항 (open questions — 구현 착수 전 결정)

1. **TODO: train.jsonl 입력 범위** — `roi_selected.json`(선택된 ROI subset) vs **전체 real 결함 pool**. ControlNet 학습은 통상 전체 결함 pool이 품질상 유리 → 후자 유력하나 확정 필요. (워크플로우는 roi_selected로 봤음)
2. **TODO: hint 생성기 morphology 소스 통일** — morphology_features.csv join(`_join_metrics`) vs `DefectCharacterizer.analyze_defect_region(mask)` 즉석 계산이 동일 metric 스케일을 내는가? HintImageGenerator는 linearity>0.7에서 skeletonize 분기라 오차가 hint 형태를 바꿈. 단일 소스로 통일할지.
3. **TODO: exp2 random seed sweep 구현 방식** — `sampling_strategy=random`을 distinct `--seed`로 ≥5회 재실행 vs seed loop 추가. paired Wilcoxon을 exp2 내부 vs 외부 집계 스크립트로. (새 테스트코드 금지 규칙과의 관계 — 실험 스크립트는 테스트코드 아님)
4. **TODO: 도메인별 background_type 매핑 정책** — carpet/leather→periodic/organic, bottle/hazelnut→smooth/complex를 dataset_config.json 확장 vs CLI. min_quality gate를 breadth 스토리에서 쓸지(쓰면 매핑이 prerequisite).
5. **TODO: n_selected parity 사전점검** — img_diversity_cap bounded-repetition 부족(L803/L1144)으로 budget 외 이유로 aroma/random n이 다르면 `_equalize_budget`이 불균등 유효 다양성을 숨김. 도메인별 n_selected parity assert 방법.
6. **TODO: content-hash seed를 Tier 0에도 적용할지** — 현재 copy_paste는 순서 의존 결정론(단일 rng). arm 순서·ROI 순서 변경 내성을 위해 content-hash로 갈지, 순서 고정으로 충분한지.
7. **TODO: CASDA empty-mask 근본 원인** — ControlNet blank 생성 vs compose 실패 미진단. mask filter 후에도 valid pool < cap 가능성 — CASDA 참조 행조차 신뢰 가능한 budget인지.

---

## 다음 단계

- 본 노트는 **설계 spec**(read-only 검토 결과)이므로 코드 편집 없음.
- P0–P3 구현 착수 시 각 단계는 `devnote-guard.md` 상 별도 dev_note + `feature-dev` 대상.
- 우선 착수 후보: **P0(train.jsonl+hint 빌더)** — 이것이 self-contained 서사의 gating. 그다음 exp2 random seed sweep(일차 증거의 통계적 방어).
- 검증은 Colab 전용(새 테스트코드 금지, CLAUDE.md). 환경변수는 `$VAR` 형식(`${VAR}` 금지), AROMA self-contained run은 자체 env 매핑 셀 + 경로 존재 assertion 필요(CASDA env 표 재사용 금지).
