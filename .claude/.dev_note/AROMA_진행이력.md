# AROMA 연구 진행 이력 (2026-06-09 ~ 2026-07-13)

> **문서 목적**: `.claude/.dev_note/` 하위 패치 노트 80건을 시간순으로 정리해 AROMA 연구의 진행 내역을 파악한다.
> **범위**: `.dev_note/*.md` 전체 80건 (`.completed` 목록 52건 + 미등재 28건 — `.completed`는 07-13 이후 갱신되지 않아 stale).
> **날짜**: 각 노트의 git 최초 커밋 날짜(월-일). 2026년.
> **깊이**: 노트당 문제 → 변경 → 결과/영향 3~5줄. 코드 diff는 원본 노트 참조.
> **폐기·전환된 결정**은 §9에 별도 정리 (연구 흐름 이해의 핵심).

---

## 시기 구분 요약

| 시기 | 기간 | 성격 | 노트 |
|------|------|------|------|
| §1 | 06-09 ~ 06-12 | 파이프라인 골격(Phase 0 → Step 4) + Phase 0 지표 정비 | 14 |
| §2 | 06-15 ~ 06-17 | deficit 버그 체인 수정 + 실험 프레임(exp1~5) 정의 | 16 |
| §3 | 06-19 ~ 06-23 | 평가 패러다임 전환(one-class AD → supervised YOLO) + 라벨 파이프라인 구축 | 9 |
| §4 | 06-24 ~ 06-29 | multi-class 공정성·품질 게이트 + 하위성능 진단 | 12 |
| §5 | 06-29 ~ 07-03 | 방법론 전환(compat+blend), self-contained ControlNet 설계, 다중도메인 확장 | 5 |
| §6 | 07-04 ~ 07-07 | 4종 일반화, 저비용 증거실험(exp5/exp6), ControlNet arm, H1 확정 | 10 |
| §7 | 07-08 ~ 07-13 | placement-aware 전환, compat 게이트 재설계, Step 3.5 신설, 학습 속도 | 14 |
| §8 | 07-13 이후 | (dev_note 없음) 논문 원고 정리 — git 이력 기준 | — |

---

## §1. 06-09 ~ 06-12 — 파이프라인 골격과 Phase 0 지표 정비

**`aroma_project_roadmap`** (06-09) — 전체 스크립트 로드맵. `AROMA-sharpened-spec.md`(명확성 91/100) 기준으로 파이프라인 구현 계획 수립. 핵심 발견: 기존 `scripts/distribution_profiling.py`가 형태학/컨텍스트 특징 추출·분포 분석·클러스터링·호환성 행렬·Deficit 분석을 이미 완전 구현 → AROMA는 그 위에 MCI/CCI 스칼라 + Meta Policy Generator + Prompt Generation만 추가하는 구조로 축소. Step 4 blending 방식(Poisson vs alpha)은 이 시점 미확정.

**`aroma_step1_complexity-analysis`** (06-10) — Step 1 구현. CASDA의 수작업 형태학 카테고리·고정 임계값·전문가 정의 호환성 매트릭스 의존을 데이터 구동으로 대체하는 것이 AROMA의 출발점. `distribution_profiling.py`를 Phase 0으로 호출한 뒤 출력을 읽어 MCI/CCI 스칼라와 정책 라우터만 구현하는 `compute_complexity.py` 신설.

**`aroma_step1_colab-guide`** (06-10) — Step 1의 Colab 실행 가이드 작성. Phase 0 → Step 1 전체를 셀 단위로 실행 가능하게 문서화.

**`aroma_step2_prompt-generation`** (06-10) — Step 2. 형태학 클러스터 × 컨텍스트 빈 조합별 자연어 프롬프트를 LLM 없이 고정 템플릿으로 생성.

**`aroma_step3_roi-selection`** (06-10) — Step 3. (결함 이미지 × 컨텍스트 빈) 후보를 스코어링하고 샘플링 전략에 따라 합성용 ROI 목록 선별.

**`aroma_step4_generate-defects`** (06-10) — Step 4. Step 3 ROI를 정상 배경에 합성. Copy-Paste를 기본 구현하고 ControlNet/Inpainting은 인터페이스 stub만 둠 (이 stub이 이후 07월 self-contained 설계까지 미구현 상태로 남는다).

**`aroma_dataset_phase0_analysis`** (06-11) — isp_LSM_1 / mvtec_cable / visa_cashew 3종의 Phase 0 출력 비교 분석 문서. 이후 지표 개선의 근거 데이터.

**`aroma_phase0_histogram-bins-dynamic`** (06-11) — `_detect_valleys()`의 `HISTOGRAM_BINS=50` 고정값이 92~100 샘플에서 평균 1.84 샘플/bin → 빈 bin이 잡음 peak로 감지되어 가짜 valley 대량 발생(cable circularity 자연경계 1개 → 12개 감지). Sturges' rule(`ceil(log2 n)+1`)로 동적 계산 전환 → 50→8 bins. MCI valley_count 컴포넌트 값이 바뀌어 재실행 필요.

**`aroma_phase0_valley-overdetection-fix`** (06-11) — bins 수정 후에도 과감지가 유지됨 → 실제 원인은 `VALLEY_PROMINENCE_RATIO=0.1`로 확정. (B) 히스토그램 평탄도(CV)가 낮으면 prominence를 자동 상향 + (C) bounded feature로 적용 범위 한정(회귀 안전망). 일부 feature가 multimodal→unimodal로 전환되며 boundaries 산출이 percentile로 바뀜 — 의도된 변화.

**`aroma_step1_cci-bootstrap-sampling`** (06-12) — CCI sub-sampling이 n>20000일 때만 작동해 소규모 데이터셋은 전체 패치 사용 → 데이터셋 간 통계 볼륨 불균일. 복원추출(bootstrap)로 항상 정확히 20,000 패치 사용하도록 통일.

**`aroma_step1_mci-redesign-class-diversity`** (06-12) — per-class valley_count 도입 시 cable(8종, 클래스당 20~30 인스턴스)이 소샘플로 valley=1.25 → MCI 최하위로 기대 순서 역전. MCI 컴포넌트를 ClusterCount(GMM BIC k) → ClassDiversity(`Neff = e^H`, Shannon entropy 기반)로 교체하고 valley_count는 pooled로 복귀. cable의 복잡도를 ClassDiversity=8.0으로 정당화.

**`aroma_step1_class-diversity-log-norm`** (06-12) — min-max 정규화에서 isp(Neff≈2.0)의 norm이 0.116으로 MCI 기여가 소실 → isp≈cashew(gap 0.006). `ln(Neff)/ln(N_max)` 로그 정규화로 전환 → gap 0.060(10배), 순서 cable > isp > cashew 확립. cable은 0.997로 상한에 자연 수렴.

**`aroma_step3_roi-candidates-cellkey-bugfix`** (06-12) — `build_candidates()`가 context feature 없는 morphology row로 cell_key를 만들어 항상 `"0_0_0_0_0"` 고정 → `prompts.json` 키 불일치로 대부분 cluster에서 프롬프트 공백. morphology row당 cluster_row 전체를 순회해 (image × cell_key) 조합별 candidate 생성으로 수정. candidate 수 `n_images × 1` → `n_images × n_occupied_bins`.

**`aroma_colab_execute_batch-parallel-execution`** (06-12) — phase0/step1~step4 5개 실행 가이드가 단일 `DATASET_KEY` 하드코딩만 제공 → 28개 데이터셋 순회 실행 불가. `dataset_config.json` 자동 로드 + 선행 산출물 존재 체크(skip) + ThreadPoolExecutor 병렬 실행 섹션 추가(문서 전용).

---

## §2. 06-15 ~ 06-17 — deficit 버그 체인과 실험 프레임 정의

이 시기의 핵심은 **deficit 신호가 사실상 죽어 있었다는 사실의 발견과 연쇄 수정**이다. AROMA의 초기 헤드라인 원리가 deficit-aware ROI 선택이었으므로, 이 체인은 이후 방법론 전환(§5)의 직접적 전사(前史)다.

**`aroma_exp2_roi-quality`** (06-15) — Steps 1~4가 4종(isp_LSM_1, mvtec_cable, visa_cashew, visa_pcb)에서 완료됨. deficit-aware 선택이 무작위 대비 형태학/컨텍스트 커버리지에서 우월한지 정량 검증하는 exp2 신설. `roi_selection.py --sampling_strategy random` 옵션과 `scripts/aroma/experiments/` 디렉터리 신규 생성.

**`aroma_step1_deficit-cellkey-mismatch`** (06-15) — `distribution_profiling.py`의 step6(호환성)은 이미지 단위 **평균** context로 cell_key를, step7(deficit)은 **패치 단위 raw row**로 cell_key를 만들어 키 공간이 불일치 → `build_candidates()`의 deficit lookup이 항상 0.0 반환(isp 실측: candidates 1282개 중 nonzero 19개). granularity 통일. Phase 0 재실행 필요.

**`aroma_exp2_rarepair-deficit-nonzero-fix`** (06-15) — `p75(all candidates)`를 rare 임계로 쓰는데 candidates의 75%+가 deficit=0 → p75=0 → 전부 rare → `rare_pair_coverage`가 1.0으로 고정(AROMA/Random 구분 불가). 구조적 원인: good 이미지의 context cell(77개, 넓게 분포)과 defect 이미지의 cell(7개, 좁게 집중)이 달라 compat 내 셀은 deficit≈0, deficit>0인 셀은 candidates에 없음. nonzero 기반 임계로 재정의.

**`aroma_exp1_severstal-casda-comparison`** (06-15) — 논문 Exp 1 정의: Severstal(CASDA의 홈그라운드)에서 AROMA vs CASDA vs Random. **모든 방법이 동일 copy-paste 합성을 쓰고 ROI 선택만 다르게** 해서 "합성 개선"이 아닌 "ROI modeling 기여"를 분리하는 원칙 확립 — 이 원칙이 이후 모든 공정성 논쟁의 기준이 된다.

**`aroma_step1_cci-subsampling-perclass-valley`** (06-17) — 두 가지 공정성 개선: (1) `compute_cci()`의 `_col_mean`/`_col_var`가 전체 `context_X`를 써서 고해상도 데이터셋의 CCI가 과대평가 → 20k sub-sampling 기준 통일. (2) valley_count를 클래스별 계산 후 평균해 클래스 수 편향 제거. 3종 이전 수치(isp 0.604 / cable 0.646 / cashew 0.607) 전부 재계산 필요.

**`aroma_phase0_data-driven-anchor`** (06-17) — `expected_range`가 수작업 고정값(`valley_count: [0,18]`)이라 실측 27~39가 전부 상한 초과 → norm=1.0 고정, MCI 판별력 소실. `anchor = max×1.2` 데이터 기반 상한 설계를 문서화(시뮬레이션: 46.8 적용 시 3종 상대 순위 복원). 문서 전용, 코드 변경 없음.

**`aroma_step3_pair-aware-allocation`** (06-17) — exp2 결과에서 AROMA가 Random보다 morph/ctx/rare_pair **모두 낮음**(isp: AROMA morph 0.50·ctx 0.14 vs Random 1.00·0.91). 원인: rare 21개 선택 후 나머지 179개를 roi_score 상위로 패딩 → ctx_prior 지배 cell_key가 상위를 독점 → 특정 (cluster, cell_key) pair에 depth-first 집중. **Coverage-first + Quality-second 2-stage** 구조로 전략 재설계(모든 pair 최소 1개 보장 → deficit 기반 quota → 잔여 슬롯은 roi_score).

**`aroma_step1_deficit-global-good-only`** (06-17) — `step7`의 `global_dist`를 good+defect 전체로 계산하면 `P(cell|global) ≈ P(cell|defect_cluster)` → deficit이 거의 전부 0. 설계 의도("good 배경에서 흔한데 defect cluster가 못 덮는 부분")대로 **good 이미지만** 사용하도록 수정. cellkey-mismatch 수정 이후 발견된 2차 결함.

**`aroma_exp3_generation-quality`** (06-17) — 논문 Exp 2(구현 exp3): AROMA vs Random의 cross-domain 생성 품질(FID). CASDA는 Severstal 전용이라 4종에는 미적용. copy-paste 한계를 논문에 명시하는 문구 확정 — "원 결함 외형을 보존하므로 새로운 형태를 생성할 수 없다. 본 연구 목적은 합성 모델 개선이 아니라 adaptive ROI modeling이 합성 학습데이터 품질을 높이는지 평가하는 것".

**`aroma_exp_number_mapping`** (06-17) — 논문 Exp 번호(E1~E4)와 개발 Roadmap 번호(Exp 1~5) 불일치 정리. 논문 E1(MCI/CCI 유효성)·E2(Policy 검증)는 Roadmap에 대응 구현이 **없음**(미구현), 논문 E3=Roadmap exp2, 논문 E4 일부=Roadmap exp3. E1/E2 대응 방안은 미결로 남김.

**`aroma_exp5_crossdomain`** (06-17) — ISP(LSM_1/LSM_2/ASM) + MVTec cable + VisA(cashew/pcb) 도메인에서 AROMA 우위가 유지되는지 검증하는 exp5 정의. (이 데이터셋 구성은 이후 §5~§6에서 severstal/leather/aitex/mtd 4종으로 전면 교체된다.)

**`aroma_exp3_kid-lpips-addition`** (06-17) — 논문이 FID/KID/LPIPS 3종을 요구하나 구현은 FID만 → KID(소표본 신뢰성), LPIPS(지각 유사도) 추가.

**`aroma_exp4_downstream-ad`** (06-17) — 하류 이상탐지 평가 exp4 정의. baseline(정상만) / random / aroma × AD 모델 4종(PatchCore, SimpleNet, EfficientAD, RD++).

**`aroma_exp3_fid-unit-mismatch-fix`** (06-17) — FID의 real side는 GT mask bbox crop(작은 패치), synth side는 512×512 전체 이미지를 비교 → 분포 단위 불일치로 FID 147~326의 비정상값. `annotations.json`에 mask 필드가 없어 patch-level 정렬이 원천 불가했던 것이 근인. 기존 exp3 결과 전량 무효화·재실행.

**`aroma_exp4_api-fix-resume`** (06-17) — anomalib 최신 버전 API 불일치 3건으로 exp4 실행 불가 → 수정. 장시간 실행 crash 대응으로 `--resume` + 조건별 incremental save 추가.

**`aroma_exp4_local-cache`** (06-17) — symlink가 Google Drive를 가리켜 DataLoader I/O 병목. real 이미지를 데이터셋당 1회 `/tmp`로 복사(isp 기준 Drive I/O 12회 → 1회). 부수 수정: incremental save 순서 오류로 같은 데이터셋의 이전 model 결과가 누락되던 버그 정정.

---

## §3. 06-19 ~ 06-23 — 평가 패러다임 전환과 라벨 파이프라인 구축

**`aroma_exp4v2_supervised-yolo-detection`** (06-19) — **exp4의 근본 설계 결함 발견**: `_prepare_ad_dataset_with_masks()`가 합성 결함을 `train/good/`에 복사 → one-class 모델이 결함을 "정상"으로 학습 → AUROC 붕괴(cable/simplenet: baseline 0.7646 vs random 0.5015 vs aroma 0.4996). 올바른 패러다임은 **합성 결함 + bbox → supervised YOLOv8 detection**. 지표가 Image AUROC → mAP@0.5로 전환되며 이후 모든 downstream 평가가 exp4v2 위에서 이뤄진다.

**`aroma_exp4v2_synth-ratio-optimization`** (06-19) — cable에서 AROMA precision 0.19 붕괴. 원인 두 가지: synth:real = 600:46 = 13:1 과잉 주입으로 도메인 shift, val_frac=0.5로 train 46장 소표본. 재합성 없이 `--max_synth_per_ds` subsampling + val_frac 기본값 조정으로 대응.

**`aroma_exp4v2_synth-mask-bbox-persistence`** (06-22) — 진단 결과 합성이 baseline을 못 이기는 근본 원인이 **합성 GT 라벨 품질 붕괴**로 확정. `generate_defects.py`가 합성에 쓴 타원 마스크와 paste 좌표를 저장하지 않아 downstream이 라벨을 추정해야 했음. 마스크 PNG + `bbox`/`mask_path` 키를 `annotations.json`에 영속화(additive). `generate_random.py`도 같은 엔진에 위임하므로 한 곳 수정으로 양쪽 적용.

**`aroma_exp4v2_defect-bbox-pipeline`** (06-22) — 더 심각한 결함: copy_paste가 결함 영역을 크롭하지 않고 **full test 이미지를 통째로 paste** → `bbox=[0,0,1024,1024]`, 타원 mask가 전체를 덮음. 결함 mask가 profiling에서 계산되고도 어디에도 저장되지 않아 전파 경로가 없던 것이 근인. B안(정공법): profiling 시점에 mask PNG + bbox 영속화 → CSV 컬럼 additive 추가 → roi_selection 전파 → generate_defects가 타이트 크롭 + 실제 mask 형태 blend.

**`aroma_exp4v2_foreground-placement`** (06-22) — B안 적용 후 크롭·마스크·bbox는 정확해졌으나 **결함이 객체 밖 배경(void)에 paste**됨(visa_pcb: PCB가 아닌 하단 검은 배경). `_random_paste_position`이 이미지 전체 균등 무작위였던 탓. paste 위치를 foreground(객체) 영역 내부로 제약. DEFICIT 기반 context-aware 배치(논문 thesis)는 후속 TODO로 명시.

**`aroma_exp4v2_multiseed-aggregation`** (06-23) — 단일 seed 결과가 작은 val(18~42장)에서 노이즈가 커 AROMA vs RANDOM 차이(평균 +1.7pp)의 유의성 판정 불가. `--seeds`(nargs+) + mean±std + 95% CI 집계. top-level은 평균값 유지해 기존 reader 호환, `per_seed`/`std`/`ci95` 추가.

**`aroma_exp4v2_severstal-integration`** (06-23) — CASDA 비교용 Severstal을 exp4v2에 통합. 난점: mask가 RLE(PNG 아님), 4-class, 1600×256 비정사각, config 미등록. `prepare_severstal.py` 신설(RLE→PNG) + single/multi class_mode 양쪽 지원(single 기본 → 기존 MVTec/VisA 경로 byte-identical).

**`aroma_exp4v2_severstal-synth-multiclass`** (06-23) — multi 모드에서 real은 per-class 라벨이나 **synth는 `class_id=0` 고정** → 합성 결함 전량이 class1로 오라벨. `source_roi` 경로의 `class{N}`을 파싱해 `class_id = N-1` 부여. single 모드는 불변.

**`aroma_exp4v2_train-speed-knobs`** (06-23) — `model.train()`에 batch/cache/rect 미전달로 Ultralytics 기본값 사용. Severstal(1600×256, 6.25:1)에서 rect=False가 ~84% 회색 패딩 연산 낭비, cache=False가 매 epoch 재디코드. 세 인자를 CLI 노출(기본값=현행이라 미지정 시 byte-identical), 전 조건 동일 적용으로 비교 공정성 유지.

---

## §4. 06-24 ~ 06-29 — multi-class 공정성, 품질 게이트, 하위성능 진단

**`aroma_exp4v2_per-class-metrics`** (06-24) — multi 모드에서 macro 평균 스칼라만 저장돼 클래스별 기여가 안 보임. Ultralytics `val_results.box`의 per-class 데이터를 `out["per_class"]`로 기록(multi 한정, single/타 데이터셋 byte-identical). rare-class 분석의 기반이 된다.

**`aroma_step3_multiclass-allocation-fix`** (06-24) — severstal c2만 AROMA가 baseline·random 양쪽에 회귀(c2 map50: baseline 0.2617 / random 0.2821 / aroma 0.2304). 산출물 직접 분석으로 원인 확정: `_pair_aware_allocation`의 **class-blind 할당**. top_k=200 결과 c1 131(65.5%)/c2 11/c3 58/c4 **0**(완전 starvation). 두 증상 = starvation-to-0(c4) + monoculture(c2: 단일 pair가 quota 독점 → 동일 morphology 11개 → 한 archetype 과적합). 처방: generic class-floor + diversity-cap(single-class는 K=1이라 구조적으로 면역). **severstal 특이 대응이 아닌 multi-class 일반 대응**이라는 원칙 명시.

**`aroma_exp4v2_casda-condition`** (06-24) — CASDA를 4번째 조건으로 추가. 학습 코어가 condition-agnostic이고 CASDA 합성 피더가 이미 완성돼 있어 소수정으로 가능. **프레이밍 명시**: 이것은 "공유 copy-paste 엔진 안에서의 ROI 선택 ablation"이지 CASDA 네이티브 파이프라인(Poisson + ControlNet) 전체가 아니다 → 논문 표기 "CASDA ROI selection inside AROMA's shared synthesis engine". 이 caveat이 이후 exp7 필요성의 근거가 된다.

**`aroma_severstal_flat_diagnosis_and_direction`** (06-25, 전략 노트) — 검출 mAP가 조건 간 **평탄**(baseline 0.3821 > aroma 0.3595 > random 0.3379, 3-seed)한 현상 진단. per-seed delta(aroma−random) +0.057/+0.010/−0.002 → 신뢰구간이 0을 포함. 세 원인 복합: (1) real_train≈2534가 풍부해 synth:real 0.24:1의 포화 구간, (2) rare 클래스 c2의 하드 다양성 천장(117 ROI/95 distinct source/2 subtype — 어떤 선택 전략도 c2 다양성을 창출 못 함), (3) aggregate가 이미 클래스 동등가중. 인프라 스케일업 커밋(top_k 200→1690 등)으로 3 arm 예산 동일화. 즉시 가능한 detector-decoupled 진단 2건(train-label 클래스 히스토그램, intrinsic diversity audit) 제안.

**`aroma_exp4v2_source-diversity-fairness`** (06-25) — 평탄 원인 조사에서 **독립 confound 3건 + 인프라 결함 3건** 발견. 치명적인 것은 #1: AROMA ROI 선택이 3620 distinct 소스를 **88 distinct crop**으로 붕괴(CASDA 1692 대비 ~19배 다양성 기아) → `img_diversity_cap` 도입. #2: AROMA/random이 **결함 포함 배경**(context_select)으로 합성 → 무라벨 결함이 false-negative로 학습 → `train/good`으로 전환(배경 선택은 원래 context-blind 랜덤이었으므로 기능 손실 0). **논문의 "context-matched 배경 합성" 주장 수정 필요**를 명시. 이전 모든 AROMA 결과가 #1로 오염됐을 수 있음을 경고.

**`aroma_exp4v2_roi-quality-gate`** (06-25) — 옛 코드 기준 진단표(baseline 0.4163 / casda 0.4330 / **aroma 0.3132**, c2는 0.0675로 붕괴). AROMA가 c2에 429장(CASDA 175) 과투입했으나 붕괴 — CASDA suitability≥0.5 통과 c2 ROI는 전체 117개뿐인데 AROMA엔 품질 컷오프가 없었음. origin/main이 이미 Fix1~4(img_diversity_cap, class_floor, per_pair_cap, rarity_temp)를 가진 것을 확인해 중복 계획(공급량 연동 per-class cap) 폐기하고 **품질 게이트만** 얹음.

**`aroma_exp4v2_roi-quality-gate_colab-guide`** (06-25) — 품질 게이트(`--min_quality`, `--background_type`) Colab 부록. Pass 1(게이트 OFF로 분포 확인) → Pass 2(임계 적용) 2-pass 절차. `--min_quality 0.0`이 OFF 기본값.

**`aroma_exp4v2_clean-background-gate`** (06-26) — 배경 풀 적재에 품질 필터가 전무해 검은/평탄(void) 이미지가 배경 후보가 되고, foreground 추정 실패 시 폴백으로 결함이 검은 영역에 붙음. CASDA의 검증된 `compute_quality_score`를 이식하되 전 데이터셋 공통 적용. **구현 완료**: pool-level + position-level 이중 주입, CLI 3개(`--reject-clean-bg`/`--min-bg-quality 0.7`/`--bg-blur-threshold 100`), 리뷰 반영으로 하드 Laplacian 게이트 제거하고 CASDA식 단일 기준으로 통일.

**`aroma_exp4v2_aroma-underperformance-diagnosis`** (06-28, 전략 노트) — clean-bg 게이트의 Colab 실측 결과가 예상과 다름: **pool 게이트 R=0**(severstal train/good에 전역 검은 이미지 없음 → no-op), OFF==ON에서 검은배경 비율 동일. Cell 4B 추적으로 **진짜 원인 확정**: 검은배경 결함 35장 중 33장(94%)이 `_foreground_mask`의 corner-vote가 검은 void를 "전경"으로 오분류한 경로(via_fg)로 발생. aroma==random 동일 수치 → **공유 paste 엔진 레벨 문제, ROI 전략 무관**. 우선순위: 1순위 void 오검출 수정(~94% 해소), 2순위 pool 게이트 유효면적 기준(~6%).

**`aroma_exp4v2_foreground-void-rejection`** (06-28) — `_foreground_mask`에 전경 void 거부 가드를 additive 추가(선택된 전경이 "평탄+검음"이면 극성을 뒤집지 않고 `None` 반환 → 기존 random-placement 경로로). **Colab 실측**: None 7.8%→18.7%, 어두운 전경 26.8%→17.0%, 검은배경 via_fg 33→9(aroma/random 동일), 검은배경 비율 aroma 9.1%→6.0%(−34%). 판정 = 부분 성공(Mode A 제거). 잔여 6~7%는 Mode B(폴백이 거대-void normal에 재착지). **전략적 함의**: 수정이 3조건 대칭이라 "증강 vs baseline"에는 기여하나 "aroma vs casda" 상대순위는 거의 안 바꿈 → Mode B는 보류.

**`aroma_exp3_cleanbg-rerun-guide`** (06-29) — 게이트가 구현됐지만 CLI 기본값이 OFF(`store_true`)이고 exp3/step4 가이드에 플래그가 없어 **게이트 미적용으로 합성**되는 상태. 두 가이드의 합성 명령에 플래그 추가 + 재생성 전 기존 디렉터리 삭제 안내. 평가 스크립트는 읽기 전용이라 변경 없음.

---

## §5. 06-29 ~ 07-03 — 방법론 전환과 self-contained 재구성

**`aroma_roi-synthesis_compatibility-context-blend`** (06-29) — **방법론 전환의 분기점**. exp2 실증으로 두 사실 확인: (1) **deficit-aware는 object-centric에서 무의미** — carpet/leather/wood/metal_nut의 rare_pair(deficit>0)=0, severstal만 신호 있음. (2) 합성이 배경 context에 의존하지 않으면 "어디 두나"가 픽셀에 반영되지 않음. → deficit-aware를 헤드라인 원리에서 **폐기**하고 선택(L1: compatibility + quality)과 합성(L3: seamlessClone + Reinhard 색 전이)을 함께 개선. training-free 유지. clean-bg 게이트·foreground 가드와는 직교하도록 설계.

**`aroma_exp4v2_visa-config-driven-loader`** (06-29) — exp4v2의 `_load_detection_dataset`이 dataset_key를 하드코딩 분기로 해석해 `visa_cashew`/`visa_pcb`만 지원, generic 핸들러 부재. `dataset_config.json` 기반 config-driven `visa_*` 로더 추가(exp3에 이미 있던 패턴 재사용). 기존 하드코딩 분기·severstal multi 경로는 무수정.

**`aroma_research-core_thesis-and-compounding`** (07-01, 전략 노트) — cross-domain 결과 재검토 결과 **AROMA 기여를 "baseline 상시 격파"로 framing하면 severstal 자기 반례로 즉시 reject**된다는 결론. 정정된 thesis: AROMA = 생성과 직교하는 **결함×배경 type 인지 ROI 선택 컴포넌트**, CASDA는 단일 도메인 특화 대비군. 반복 명시할 caveat: **exp4v2의 'casda' arm은 진짜 CASDA가 아니다** — `generate_casda.py`가 `method="copy_paste"`를 하드와이어하고 `controlnet_synthesis`는 `NotImplementedError` 스텁이라 어떤 arm에서도 diffusion이 실행되지 않았다.

**`aroma_research-core_self-contained-multidomain-design`** (07-02, 설계 spec) — AROMA를 **self-contained 다중도메인 증강 프레임워크**로 재정의: 도메인별 real 결함으로 ControlNet 학습 → 생성 → AROMA type-aware ROI가 어디/무엇을 배치할지 결정 → Poisson blend. 논문 일차 축 = 다중도메인 일반성(breadth). CASDA와 파이프라인을 억지로 병합하면 confound가 생기므로 CASDA는 Severstal 전용 참조로만 남긴다. Step 1~3은 무수정(회귀 0), exp4v2 harness도 condition-agnostic이라 무변경.

**`aroma_selfcontained_p0-train-jsonl-builder`** (07-02) — 현재 ControlNet 학습데이터를 만드는 **유일 경로가 CASDA 경유**(`aroma_to_casda_roi.py` → CASDA `prepare_controlnet_data.py`)라는 발견에 대한 대응. real 결함 → target crop + 3채널 hint PNG + prompt를 AROMA 안에서 직접 생산하는 `scripts/aroma/build_train_jsonl.py` 신규 작성. 소비자 `train_controlnet.py`는 무수정. 순수 신규 경로 추가로 기존 상태 변경 없음.

**`aroma_multidomain_aitex-mtd-integration`** (07-03) — breadth 실증 데이터셋을 **4종으로 확정**: Severstal(강판)·MVTec leather(가죽)·AITEX(텍스타일)·MTD(자성타일). 도메인 축 분산(경성표면 2 + 연성/유기 2)으로 "강철 편중" 탈피. 앞 2종은 네이티브(검증만), AITEX(Kaggle)/MTD(Dataset Ninja Supervisely 배포판)는 `prepare_aitex.py`/`prepare_mtd.py` + `_find_mask_path` 분기 추가로 신규 통합. 기존 도메인 분기는 `elif` 추가만이라 회귀 0.

---

## §6. 07-04 ~ 07-07 — 4종 일반화, 저비용 증거실험, ControlNet arm

**`aroma_exp4v2_multiclass-all-datasets`** (07-04) — `dataset_config.json`은 4종 모두 `class_mode:"multi"`인데 코드가 `ds_class_mode = class_mode if ds == "severstal" else "single"`로 severstal에만 적용 → 나머지 3종은 per-class AP 관측 불가. rare-class별 AROMA 기여가 연구 crux이므로 multi 경로를 데이터셋-일반화: gate 하드코딩 제거, `_enumerate_defect_classes` 헬퍼 신설, YAML `nc`/`names` 동적화, real/synth class id 동적 매핑. `--class_mode` 미지정 시 전 데이터셋 byte-identical.

**`aroma_exp4v2_quality-gate-fairness`** (07-04) — quality gate가 **aroma 선택 경로에만** 적용되어 random은 un-gated 풀에서 샘플 → confound. 그대로 켜면 aroma 풀만 축소돼 `n_synth_train` 불일치로 공정성 붕괴. 보정 = **random도 동일 게이트 통과 풀에서 샘플**("selection 전략만 다르다" 불변식 유지). casda는 자체 `min_suitability=0.5`가 전략 정체성이므로 유지.

**`aroma_exp4v2_epoch-pilot`** (07-05) — `--baseline_epochs 300 --patience 50`이 보수적 상한. 실측 수렴점(best_epoch)으로 재산정하는 파일럿. 워크플로 비판 반영으로 severstal 1셀 대신 **큰 것(severstal) + 작은 것(mtd) 2셀 × 3조건**(소규모는 epoch당 스텝이 적어 늦게 수렴할 수 있어 단일 외삽 위험). 이 run은 real_frac 커브의 100% 지점과 겸용 → 순증 GPU ≈ 0. 코드 수정 0.

**`aroma_exp4v2_real-frac-curve`** (07-05) — downstream 인과를 단일 full-budget 점 비교에서 **real 25/50/100% × 3조건 커브**(severstal+mtd)로 재구성. "real이 희소할수록 AROMA-선택 합성이 gap을 더 메운다"는 증강 연구 표준 논증 축 획득. 주 주장은 **Δ(aroma−random) 커브로 한정**(제외된 real이 합성 소스에 포함되는 leakage는 양 arm 대칭). seed 정책 반전: 분산 작은 severstal은 1~2 seed + bootstrap CI 보조, 분산 큰 mtd는 3 seed 유지.

**`aroma_exp5_prdc-coverage`** (07-05) — AROMA 핵심 주장을 **선택에 쓰지 않은 외부 좌표계(DINOv2)** 에서 반증 가능하게 검증. exp2의 순환성(자체 cluster/cell 라벨 위 지표)과 n=1 약점을 동시 해소하는 L2 증거. **사전 등록 가설**: 동일 copy-paste 엔진이므로 Precision/Density는 동등, Recall/Coverage만 aroma > random. 둘 다 오르거나 Precision이 깨지면 기각 — 이 비대칭 예측이 사후합리화를 차단. 전체 30분 이내.

**`aroma_exp6_knn-test-coverage`** (07-05) — copy-paste 증강이 검출기를 돕는 **기제**를 측정. held-out val의 real 결함 crop에서 세 학습 풀(real / real+random / real+aroma)까지의 최근접 cosine distance 비교. min-over-pool이라 단조 감소는 자명하므로 **비교 대상은 random 대비 aroma의 추가 감소분**. exp5 임베딩 캐시 재사용 시 CPU 수 분.

**`aroma_exp6_rare-mode-coverage`** (07-05) — "부족한 (형태×배경) 조합 타겟팅" 문구와 직결되는 이산 모드 증거. DINOv2 k-means로 모드를 독립 정의(순환성 제거)하고 rare 모드 hit rate를 **random 재선택 30-seed null 분포** 대비 검정. random 재선택이 메타데이터에서 seed만 바꾸는 CPU 밀리초 연산이라 30-seed가 사실상 공짜.

**`aroma_controlnet-arm_aitex-integration`** (07-06) — `controlnet_aroma_arm_execute.md`가 aitex를 "baseline 학습 실패"로 제외했으나 타일링(256×256/stride128) + 단일클래스 전환으로 정상화 → 편입. 단 20260705 결과에 aitex가 없어 **이식 소스가 존재하지 않으므로** 다른 3종(이식 + aroma arm만 학습)과 달리 fresh `--condition all`로 실행. 이식 루프·parity 딕셔너리·비교표에서 aitex를 별도 처리해 KeyError 회피.

**`aroma_step4_h1-recombination-no-info`** (07-07, 증거 노트) — **severstal flat의 원인 확정**. H1: copy-paste는 train 실 결함 crop의 **재조합**이라 외형 신규성이 0이고, 신규성은 (결함, 배경) 조합에서만 나온다. 데이터가 충분한 도메인에서는 모델이 이미 모든 외형을 학습했으므로 재조합의 추가 정보가 없다. 예측(합성 효과 ∝ 1/real 크기)이 실측과 일치: **leather +7pp / mtd +1.3pp / severstal 0pp**. 대안 가설 반증: H2 학습동역학 이상 기각, H4 라벨 품질 기각, 순수 데이터 포화는 부분 기각(100ep에서 아직 상승 중 — "합성이 못 돕는 것"이지 "더 배울 게 없는 것"이 아님).

**`aroma_controlnet-arm_quality-filters`** (07-07) — ControlNet arm 파일럿의 품질 결함 2종 대응. (1) **AR 게이트**: bbox를 512² 정사각으로 squash→생성→un-squash하는 구조라 고종횡비 결함에서 텍스처 붕괴 → AR 초과 ROI는 생성 스킵. (2) **pair-level 텍스처 배경 재추첨**: checkerplate(무늬강판) normal에 paste하면 seamlessClone이 색만 동화하고 구조적 반복을 못 맞춤 → 텍스처 거리 기반 재추첨. 리뷰(15 에이전트, 적대적 검증): 발견 11건 중 확정 2건 — 조명 구배 오인(descriptor 계산 전 선형 조명 평면 detrend로 수정), c_per 크기 의존 노이즈 바닥(문서화). RNG stream-shift는 결정론 유지하되 필터 OFF 대비 byte-identical은 깨짐을 명시.

---

## §7. 07-08 ~ 07-13 — placement-aware 전환, compat 재설계, Step 3.5 신설

**`aroma_placement_aware_score_redesign`** (07-08) — 반증된 deficit 항을 점수에서 제거하고(`--score_mode realism` = 0.5·ctx + 0.3·morph + 0.2·quality), 선택이 약속한 배경 호환을 배치가 물리적으로 이행하도록 `_place_on`에 위치-조건화 compat 게이트(`--compat_threshold τ`) 추가. 둘 다 **opt-in 기본 OFF**로 기존 실험 byte-identical 보존. 목적은 Placement×Selection 2×2 ablation. 주의: `rarity_temp≠1.0`이면 legacy 가중치로 재계산되어 realism과 불일치.

**`aroma_exp4v2_class-key-propagation`** (07-08) — multi 모드 synth 클래스를 `source_roi` 경로 **정규식 재파싱**으로 복원하는데, 원본 `defect_type`은 이미 `roi_selected.json`까지 전파돼 있었음. `class_key` 필드를 annotation에 명시 전파하고 경로 재파싱은 fallback으로 유지. 결정적 함정: `_load_synth_annotations`가 로드 시 5개 키만 통과시켜 새 필드를 버리므로 로더도 수정 필요. 경로 재파싱이 실패해 class 0으로 강등되던 케이스만 라벨이 교정됨(의도된 버그 수정).

**`aroma_step1_cci-adaptive-range`** (07-08) — `expected_range` 하드코딩 상수를 per-dataset 데이터드리븐으로 전환(opt-in, 기본 OFF). **핵심 위험 식별**: MCI는 정책 선택에서 절대 임계 비교(`high_complexity_mci=0.6`, `prune_low_mci=0.3`)에 쓰여 범위를 바꾸면 클러스터링 정책 → 하위 stage 전체로 캐스케이드 → **1차는 CCI-only**(CCI는 리포트·로그 전용이라 안전), MCI adaptive는 별도 노트로 분리.

**`aroma_future_generator-backbone-alternatives`** (07-09, 참조 노트) — SD1.5+ControlNet 512² 정사각 학습이 AR 왜곡의 근본 원인이나, **backbone 교체는 값싼 수정이 아니라는 판단**. 이유: (1) SD1.5의 가벼운 prior가 grayscale·소규모 산업 데이터에 오히려 적합, (2) 파이프라인 전면 재작성, (3) study 중간 교체 시 기존 run과 comparability 붕괴, (4) AROMA 기여는 ROI 선택이지 생성기가 아님(헤드라인에 넣으면 "생성기 벤치마크"로 변질). 후보 비교 결론: SDXL은 AR만 해결하고 데이터 부적합, defect-GAN 계열이 AR 관점 최강이나 ControlNet의 구조적 조건화를 잃으므로 base 교체가 아닌 **별도 생성 arm**으로 다룰 사안. 현 시점 비권장.

**`aroma_compat_gate_clean-grounded_redesign`** (07-10) — compat 기준분포 `P(cell|cluster)`가 **결함 이미지의 image-mean context**에서 학습돼 (a) 결함 존재가 배경 통계를 오염, (b) granularity가 게이트 질의와 불일치 → 순수 normal 배경 paste 시 domain-shift + 게이트 무력화. 무력화의 단일 기제 규명: τ=0.5에서 관측 cell(compat<0.5)은 전멸 거부되고 미관측 cell(0.5 fallback)만 통과 → 데이터셋에 따라 net 결과만 다름(leather over-accept, severstal over-reject, aitex 중간). **CPU 사전진단 실측**: TV(good, defect) = leather 0.6955 / aitex 0.4327 / severstal 0.2637 / mtd 0.1281, patch-gran 재구축 시 coverage 헤드룸 +95%p(leather)~+12%p(severstal). 처방 = **SGM(대칭 기하평균) + P_def·clean_dist 양쪽 patch-granularity화**. τ는 compat_max 미만(~0.05)에서 스캔해야 한다는 제약 확정.

**`aroma_exp7_severstal-casda-native-comparison`** (07-10, **BLOCKED**) — 논문 갭 "CASDA 프레임워크 비교 미실행" 해소 계획. 기존 `casda` 조건(shared 엔진 ablation)과 별개로 CASDA **네이티브 풀파이프라인**(ControlNet 생성 + Poisson Blending) 산출물과 비교. 사용자 정정에 따른 공정 비교 원칙: 두 프레임워크 모두 ControlNet 기반이므로 AROMA arm도 copy-paste가 아니라 `--method controlnet --blend_mode seamless` 산출을 사용 → 차이는 blend 방식(Poisson vs seamlessClone) + AROMA의 ROI modeling. severstal `sym_final` 전체 체인 완료 전 착수 금지.

**`aroma_exp4v2_perclass-parity-cap`** (07-11, **보류**) — arm 간 클래스 분포·라벨화 수율을 강제 동일화하는 per-class stratified cap 설계. **결정: 구현하지 않음**. 방법론적 근거: exp4v2의 정당한 통제는 총 학습 예산 동수까지이며, 합성 결함의 클래스 분포·라벨화 수율은 AROMA 선택·배치의 **결과(post-treatment)** 이므로 강제 동일화는 인과추론상 bad control / post-treatment bias. 당장 권장 = 강제 균등화 대신 arm별 분포·수율 계측 보고 + per-class AP 제시. 리뷰어가 명시적으로 요구할 때만 `--label_parity` 선택 플래그로 구현.

**`aroma_phase0_image-id-unique-key`** (07-11) — MVTec 계열은 `defect_type` 하위폴더마다 동일 stem(`000`,`001`…)이 존재하는데 `image_id = stem`이라 **클래스 간 충돌** → leather 클러스터링이 degenerate하고 `assignments.get(image_id)`가 동일 stem 5개 결함에 단일 클러스터만 반환(실제 파이프라인 결함). `f"{defect_type}_{stem}"` 고유키로 수정(morph·context 두 워커 락스텝). leather cluster_assignments 19→92, 타 3종은 논리 불변. 부수 작업 B: `_image_dim()`이 patch 격자로 이미지 dim을 추정하는데 비-overlap truncate 타일링 탓에 956장 중 879장 과소추정(mean −31px) → clamp-safe이나 edge-flush 위치가 ~30px 안쪽에 놓여 geometry prior 품질 저하.

**`aroma_step3_5_clean-bg-selection`** (07-11) — 배경 선정이 **생성 시점에 raw good 픽셀을 재스캔**(ROI마다·시드마다 히스토그램 재계산)하던 것을, `roi_selection.py`가 `profiling → roi_selected.json`을 만드는 것과 **대칭으로** 프로파일링 파생 파일에서 한 번 precompute하는 Step 3.5 모듈로 분리. **정직성 제약 명시**: 히스토그램 매칭은 도메인 조건부(aitex lift +0.78 강, severstal/mtd ≈0) → 확실한 가치는 재현성 + 대칭 대조군(random arm에 동일 배경 배정) + per-seed placement variance 제거. 현 placement가 geometry-blind임도 기록(mtd break 실제 edge 100% → 배치 46.5%). rng 소비가 offline으로 이동하면서 발생하는 스트림 시프트 처리 필요.

**`aroma_step3_5_clean-bg-selection_design`** (07-11) — 위 모듈의 구현 설계서. `roi_selection.py`의 top-to-bottom 레이아웃을 미러(bootstrap/load_json/optional-dependency gate verbatim 복사), 이산화는 `compatibility_matrix.json`의 `bin_edges` 재사용(재유도 금지), 입력 로드는 status dict로 절대 raise 안 함.

**`aroma_exp4v2_localcache-mask-nrm-skip`** (07-13) — 학습 전 Drive→`/tmp` 스테이징이 severstal seed 42에서 **10434 파일 / 568.1s**. 분석: merged mask와 synth `normal_image`는 epoch마다 읽히지 않고(mask는 label 빌드 1회, YoloCache hit이면 0회) 스테이징 정당성이 없음. mask는 Drive 원본 경로 유지하고 `mask_map` **key만 rewrite**, `nrm_`은 mask가 있으면 스킵. 10434 → ~5212(**~50% 감소**, 장수 산수 기준). mask 없는 legacy/CASDA synth는 `normal_image` fallback이 필요하므로 스킵 조건을 `mask_path && exists()`로 한정.

**`aroma_cleanbg_void-floor-fix`** (07-13) — exp4v2 AROMA class2 composite가 평균 **48%(최대 88%) BLACK**. 로컬 실측: `corr(composite black, normal black) = 1.000` → black은 paste/blend가 아니라 **선정된 normal 배경**(부분/가장자리 강판)에서 100% 유래. 근인 두 가지: (1) `_derive_void_floors()`가 void floor를 관측값의 1st percentile로 유도해 dark-void 클러스터보다 한참 낮음 → black 패치의 1%만 검출, (2) `valid_bg_pool()`의 void_frac 컷이 90th percentile **상대 컷**이라 tail이 아무리 void-heavy해도 항상 ~90%를 유지 → partial plate 완전 제거 불가. void floor 재유도 + 절대 void_frac 컷으로 수정.

**`aroma_naive-random-placement`** (07-13) — 의도한 대비("AROMA grounded vs random naive")가 무력화된 상태 발견: 현재 random arm도 `_foreground_mask`로 placement가 제약되고(severstal good 45/60 = 75%에서 mask 반환) 가이드가 `--reject-clean-bg`를 넘겨 void 게이트·pool 게이트가 ON → **naive가 아님**. placement 비대칭을 AROMA 기여로 분리하는 naive random-placement baseline 추가.

**`aroma_exp4v2_workers-compile-knobs`** (07-13) — A100 학습시간 단축. `--workers`(미전달로 Ultralytics 기본 8 사용 중)와 `--compile`(torch.compile inductor, ~14% 속도↑, opt-in) 노출 + 전파 체인 5지점 수정. 가이드 파라미터 상향(`--batch 64→128`, `--patience 50→25`, `--workers 12`). 웹 확인 결과 YOLOv8은 2026 Ultralytics 완전 지원 → **모델 교체 없음**(논문 비교실험 정체성 보존).

---

## §8. 07-13 이후 — 원고 정리 (dev_note 없음)

dev_note 기록은 07-13에서 멈춘다. 이후 작업은 git 커밋 이력 기준으로 **논문 원고 정리**다: coverletter 작성 → `colab_execute_new` 문서를 v2-1 5종(severstal / mvtec_leather / aitex / mtd / kolektor) 현황으로 현행화 → `AROMA.txt`를 개별 section 파일 최신본으로 재동기화 → reference sync. 코드 변경은 없다.

---

## §9. 폐기·전환된 결정

연구 흐름을 이해하려면 **무엇을 왜 버렸는지**가 남긴 것보다 중요하다.

### 9.1 헤드라인 원리 교체: deficit-aware → compatibility + quality

| 시점 | 상태 |
|------|------|
| 06-09 ~ 06-17 | deficit-aware ROI 선택이 AROMA의 헤드라인. `0.4·morph + 0.4·ctx + 0.2·deficit` |
| 06-15 ~ 06-17 | deficit 신호가 사실상 죽어 있었음이 3연쇄 버그로 확인 (cell_key granularity 불일치 → rare 임계 p75=0 → global_dist에 defect 혼입) |
| 06-29 | exp2 점유 분석: object-centric(carpet/leather/wood/metal_nut)에서 rare_pair(deficit>0) = **0**. severstal만 신호. → **deficit-aware 폐기** |
| 06-29 ~ 07-08 | compatibility + quality 선택(L1) + seamlessClone/Reinhard 블렌딩(L3)으로 전환. `--score_mode realism`은 deficit 가중 0, provenance로만 JSON에 잔존 |

### 9.2 평가 패러다임 교체: one-class AD → supervised detection

exp4(PatchCore/SimpleNet/EfficientAD/RD++)는 합성 결함을 `train/good/`에 넣는 구조적 결함으로 AUROC이 붕괴(baseline 0.765 >> 증강 0.50). 06-19에 supervised YOLOv8 detection(exp4v2)으로 전면 전환, 지표도 Image AUROC → mAP@0.5. exp4 스크립트는 삭제하지 않고 남겼으나 이후 인용되지 않는다.

### 9.3 MCI 컴포넌트 왕복

- ClusterCount(GMM BIC k) → **ClassDiversity(Neff)** 로 교체 (06-12)
- valley_count: pooled → per-class → **pooled 복귀** (06-12) → 다시 per-class mean 시도 (06-17). 클래스 수 편향과 소샘플 문제의 트레이드오프를 반복 조정한 흔적
- ClassDiversity 정규화: min-max → **log-scale** (06-12)
- `expected_range`: 하드코딩 → data-driven anchor 설계(06-17) → 실제 전환은 **CCI-only opt-in**으로 축소(07-08, MCI는 절대 임계 캐스케이드 위험으로 제외)

### 9.4 "CASDA 비교"의 의미 재정의

- 06-24: exp4v2에 `casda` 조건 추가. 프레이밍은 "공유 copy-paste 엔진 내 ROI 선택 ablation"
- 07-01: **`casda` arm은 진짜 CASDA가 아님**을 명시 — `generate_casda.py`가 `method="copy_paste"`를 하드와이어하고 `controlnet_synthesis`는 스텁이라 어떤 arm에서도 diffusion이 실행되지 않았다
- 07-10: 진짜 프레임워크 비교는 exp7(CASDA 네이티브 ControlNet + Poisson vs AROMA ControlNet + seamlessClone)로 분리. 선행 체인 미완으로 **BLOCKED**

### 9.5 논문 주장 철회·수정

- **"context-matched 배경 합성"**: 배경 선택은 실제로 context-blind 랜덤(`rng.choice`)이었음. AROMA의 context 강점은 ROI/prompt 선택에 있다고 수정 (06-25)
- **"baseline 상시 격파"**: severstal 자기 반례로 즉시 reject되는 framing → "생성과 직교하는 type-aware ROI 선택 컴포넌트"로 정정 (07-01)
- **clean-bg 게이트의 기대 효과**: pool 게이트가 severstal에서 R=0 no-op임이 실측으로 드러남. 진짜 원인은 `_foreground_mask`의 void 오검출(94%) (06-28)

### 9.6 계획했다가 흡수·폐기된 설계

- **공급량 연동 per-class cap**: origin/main의 Fix1~4(img_diversity_cap, class_floor, per_pair_cap, rarity_temp)에 이미 흡수됨을 확인해 폐기, 품질 게이트만 별도 구현 (06-25)
- **`--bg-injection` 셀렉터**: injection=both가 확정이라 별도 토글 불필요 → 미구현 (06-26)
- **ControlNet backbone 교체(SDXL/PixArt/defect-GAN)**: AR 꼬리를 위한 비용-효과 역전 + comparability 붕괴로 **현 시점 비권장**, future work 참조 노트로만 보존 (07-09)
- **per-class stratified parity cap**: post-treatment bias라는 방법론적 근거로 **구현 보류**, 리뷰어 요청 시에만 (07-11)

### 9.7 데이터셋 유니버스 교체

| 시기 | 구성 |
|------|------|
| 06-09 ~ 06-17 | isp_LSM_1 / mvtec_cable / visa_cashew / visa_pcb (+ isp_LSM_2, isp_ASM, visa_macaroni…) |
| 06-23 | severstal 추가 (CASDA 비교용) |
| 07-03 | **4종 확정**: severstal / mvtec_leather / aitex / mtd (도메인 축 분산 — 경성표면 2 + 연성/유기 2) |
| 07-14 이후 | kolektor 추가 → **v2-1 5종** (dev_note 없음, `dataset_config.json`·git 이력 기준) |

---

## §10. 미완·보류 항목 (07-13 기준)

| 항목 | 상태 | 근거 노트 |
|------|------|----------|
| exp7 (CASDA-native 3-arm 비교) | **BLOCKED** — severstal `sym_final` 체인 완료 대기 | `aroma_exp7_severstal-casda-native-comparison` |
| per-class parity cap | 보류 — 리뷰어 요청 시에만 구현 | `aroma_exp4v2_perclass-parity-cap` |
| 검은배경 Mode B (폴백이 거대-void normal 재착지) | 보류 — 3조건 대칭이라 상대순위 불변, exp4v2가 binding constraint로 지목할 때만 | `aroma_exp4v2_foreground-void-rejection` |
| MCI adaptive expected_range | 보류 — 정책 선택 절대 임계 캐스케이드 위험, 별도 노트 필요 | `aroma_step1_cci-adaptive-range` |
| 논문 E1(MCI/CCI 유효성)·E2(Policy 검증) 실험 | **미구현** — Roadmap에 대응 스크립트 없음, 대응 방안 미결 | `aroma_exp_number_mapping` |
| ControlNet backbone 대체 | 비권장(재스코프 시 defect-GAN 1순위) | `aroma_future_generator-backbone-alternatives` |
| H1 완결용 real_frac 곡선 | 미확정 — baseline `--real_frac 0.25/0.5/0.75` 필요 | `aroma_step4_h1-recombination-no-info` |
| compat 게이트 SGM + patch-gran 재설계 | 착수 게이트 통과 후 진행 (leather는 no-op 확정 시 제외) | `aroma_compat_gate_clean-grounded_redesign` |

---

## 부록 — 노트 유형 분류

80건 중 순수 구현 패치가 아닌 노트:

- **전략·진단 노트** (코드 패치 아님): `aroma_severstal_flat_diagnosis_and_direction`, `aroma_exp4v2_aroma-underperformance-diagnosis`, `aroma_research-core_thesis-and-compounding`, `aroma_step4_h1-recombination-no-info`
- **설계 spec**: `aroma_research-core_self-contained-multidomain-design`, `aroma_step3_5_clean-bg-selection_design`
- **문서 전용**: `aroma_project_roadmap`, `aroma_step1_colab-guide`, `aroma_dataset_phase0_analysis`, `aroma_colab_execute_batch-parallel-execution`, `aroma_phase0_data-driven-anchor`, `aroma_exp_number_mapping`, `aroma_exp3_cleanbg-rerun-guide`, `aroma_exp4v2_roi-quality-gate_colab-guide`, `aroma_controlnet-arm_aitex-integration`
- **참조/future work**: `aroma_future_generator-backbone-alternatives`
- **보류·차단**: `aroma_exp7_severstal-casda-native-comparison`, `aroma_exp4v2_perclass-parity-cap`
