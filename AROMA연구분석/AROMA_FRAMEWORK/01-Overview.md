# 01 — 프로젝트 개요

> **Claude 요약:** AROMA(Adaptive ROI-based Morphology-Aware Augmentation)는 CASDA에서 진화한 8-stage 산업 결함 합성 프레임워크로, 데이터셋 복잡도 통계(MCI/CCI)로부터 ROI 모델링·배치 정책을 **자동 선택**하여 CASDA가 수작업으로 만들던 호환성 행렬과 형태 규칙을 데이터셋별로 재유도한다. 핵심 novelty는 프로파일링에서 산출한 대칭 호환성 게이트(symmetric compatibility gate)이며, ControlNet + seamless blending을 주 생성 엔진으로 쓴다. 다운스트림 이득은 엔진·헤드룸 조건부로, ControlNet 하에서 가장 어려운 Severstal에서만 일관된 개선(+0.096 mAP@0.5)이 확인되었다.

## Abstract

산업 AVI(Automated Visual Inspection)의 이상 탐지는 결함 샘플의 희소성과 클래스 불균형으로 제약된다. CASDA 같은 기존 context-aware 합성 프레임워크는 결함–배경 관계를 인코딩해 이 부족을 완화하지만, 도메인별 수작업 규칙과 수동으로 구성한 호환성 행렬에 의존하여 새 데이터셋마다 재설계가 필요하다. AROMA는 8-stage 프레임워크로, 그 핵심 기여는 방법론적이다 — 데이터셋 복잡도 통계로부터 ROI 모델링 정책을 자동 선택하여 수작업 재설계를 제거한다. 자동 프로파일링 단계가 각 데이터셋의 결함-형태 및 배경-컨텍스트 분포를 측정하고 공출현으로부터 결함–배경 호환성 모델을 계산하며, 여기서 MCI(Morphology Complexity Index)와 CCI(Context Complexity Index)를 산출한다. Meta Policy Generator가 이를 distribution diagnostics · candidate pruning · empirical evaluation을 통해 클러스터링·배치 정책으로 변환한다. 이 두 데이터-주도 구성요소가 CASDA가 손으로 만든 두 산출물(호환성 행렬 + 형태 규칙)을 데이터셋마다 자동으로 재유도한다. 다운스트림 결과는 엔진 의존적이다: 학습된 ControlNet + seamless blending 엔진 하에서 context-aware ROI 선택은 가장 어렵고 헤드룸이 큰 Severstal에서만 최대·유일하게 통계적으로 일관된 이득(+0.096 mAP@0.5, 3개 seed 일관)을 내며, 천장에 가까운 데이터셋은 random augmentation과 수렴한다. 동일 선택 정책이 training-free copy-paste 엔진 하에서는 (Severstal 포함) random과 통계적으로 구별되지 않는다 — 최고-deficit (형태, 컨텍스트) 조합은 복사할 소스 패치가 없기 때문(deficit이 소스 가용성과 직교)이며, context-aware 선택은 생성 엔진이 대상 조합을 합성할 수 있을 때에만 효과를 낸다.

## 핵심 특징 (Highlights)

1. **복잡도-인지 정책 자동 선택** — MCI/CCI 스칼라 지표로 morphology(otsu/gmm) · context(gmm/percentile) 정책을 데이터셋별로 자동 결정. 도메인 지식이 아닌 분포 통계에서 분기 → "AROMA도 handcrafted" 공격 방어.
2. **대칭 호환성 게이트 (core novelty)** — 프로파일링이 데이터셋 자체 패치 공출현에서 `matrix_symmetric`을 계산. 64px 타일 query로 배치 후보를 채점하고 데이터셋별 임계 τ로 accept/reject. CASDA 수작업 호환성 행렬의 자동 재유도 대응물.
3. **Deficit-aware ROI 선택의 한계 규명** — 최고-deficit 조합은 복사할 실 소스 crop이 없어 copy-paste 하에서 구조적으로 무효(deficit ⟂ source availability). 생성 엔진이 있어야 context/deficit-aware 선택이 작동함을 실증.
4. **ControlNet 주 엔진 + AR 폴백** — SD1.5 backbone freeze, ControlNet만 fine-tune. 3-채널 구조 hint(hint_generator.py)로 color shortcut 차단. bbox 종횡비 > τ_AR인 경우 copy_paste_arfallback으로 안전 폴백.
5. **엔진·헤드룸 조건부 정직 평가** — 4개 이종 데이터셋(steel/organic/textile/machined-ceramic)에서 cherry-picking 없이 평가. ControlNet 하 Severstal +0.096 mAP@0.5(전 seed 일관), 천장 근접 데이터셋은 random과 수렴.
6. **단일변수 비교 원칙** — arm 간 차이는 ROI 선택(random vs aroma) 또는 생성법(copy_paste vs controlnet)만. 배치·blend·GT mask·normal 풀·합성 예산(n_synth parity) 고정.

## 전체 파이프라인

```
prepare_datasets (step -1)         v2-1 4종을 AROMA 레이아웃 정규화 + dataset_config.json 등록
        │                          (prepare_severstal / prepare_mtd / prepare_aitex; leather는 준비 불요)
        ▼
phase0  distribution_profiling     morphology/context 분포 측정 + compatibility_matrix.json
        │                          (matrix_symmetric · P_def_patch · clean_dist · symmetric_epsilon)
        ▼
step1   compute_complexity         MCI / CCI 스칼라 + morph/context 정책 선택 (complexity_report.json)
        ▼
step2   prompt_generation          "{cluster}_{cell}" → prompt + prior + deficit (prompts.json)
        ▼
step3   roi_selection              deficit_aware + realism score, top_k=200 (roi_selected.json)
        │                          score = 0.5·ctx_prior + 0.3·morph_prior + 0.2·quality
        ▼
step4   [CN 학습 서브트랙]          build_train_jsonl → train_controlnet (SD1.5 + ControlNet)
        │                          + τ 사전스캔 (compat_gate CPU 진단, τ≠0.5, aitex는 AR/TEX 추가)
        ▼
step5   generate_defects           ┌─ AROMA arm: --method controlnet + --compat_mode symmetric
        │  (생성)                   │             + clean-bg 게이트 + seamless blend
        │                          └─ random arm: generate_random (통제, 동일 clean-bg 게이트)
        ▼
exp*    exp3 / exp4v2 / exp5 / exp6  exp4v2 = 헤드라인(YOLO 검출, baseline/random/aroma fresh 학습)
                                     exp3=생성품질(FID) · exp5=PRDC · exp6=임베딩 커버리지
```

## 리포지토리 구조

```
scripts/
├── distribution_profiling.py        # phase0 — 루트에 위치 (aroma/ 아님)
├── train_controlnet.py              # step4b — 루트에 위치 (SD1.5 + ControlNet fine-tune)
├── plot_*.py / fig_common.py        # 논문 figure 생성
└── aroma/
    ├── compute_complexity.py        # step1
    ├── prompt_generation.py         # step2
    ├── roi_selection.py             # step3 ★ score_roi / _compat_score
    ├── build_train_jsonl.py         # step4a (CN 학습 데이터 빌드)
    ├── generate_defects.py          # step5 ★ 합성 엔진 (_paste_and_finalize)
    ├── generate_random.py           # step5 random arm (통제 baseline)
    ├── generate_casda.py            # CASDA arm copy_paste
    ├── select_context_prototypes.py # 배경 풀 대표 선택 (CLIP→KMeans→medoid)
    ├── clean_bg_selection.py        # clean-bg 게이트 유틸
    ├── aroma_to_casda_roi.py        # compounding 실험 forward 어댑터
    ├── casda_roi_adapter.py         # CASDA roi_metadata 어댑터
    ├── composed_to_exp4v2.py        # CASDA composed → exp4v2 annotations
    ├── prepare_severstal.py / prepare_mtd.py / prepare_aitex.py   # step -1 데이터 준비
    └── experiments/
        ├── exp1_casda_comparison.py         # AROMA vs CASDA
        ├── exp2_roi_quality.py              # ROI 선택 품질
        ├── exp3_generation_quality.py       # 생성 품질 (FID 등)
        ├── exp4_downstream_ad.py            # 하류 이상 탐지 (구)
        ├── exp4_v2_supervised_detection.py  # ★ 헤드라인 (YOLO 검출)
        ├── exp5_prdc.py                     # Precision/Recall/Density/Coverage
        └── exp6_embedding_coverage.py       # 임베딩 커버리지
utils/
├── hint_generator.py                # 3-채널 구조 hint (학습·추론 공용)
└── prompt_generator.py              # technical-style 결정적 prompt (학습·추론 공용)
```

## CASDA → AROMA 진화

| 측면 | CASDA | AROMA (추가/변경) |
|------|-------|-------------------|
| 호환성 행렬 | 도메인 전문가 수작업 구성 | 프로파일링이 데이터셋 공출현에서 `matrix_symmetric` 자동 계산 |
| 형태 규칙 | steel용 handcrafted morphological rules | MCI/CCI → Meta Policy Generator가 정책 자동 선택 |
| 복잡도 분석 | 없음 | MCI = Mean(Entropy, ValleyCount, ClassDiversity, 1−Silhouette) / CCI = Mean(TextureEntropy, ContextClusterCount, FreqComplexity, OrientVariance) |
| ROI 선택 | — | deficit_aware 2-phase 할당 + realism score (compat gate) |
| 배치 | 고정 규칙 | 대칭 호환성 게이트(scan→rank→place, τ 사전스캔) + clean-bg 게이트 |
| 생성 backbone | ControlNet | 동일 ControlNet backbone 유지 (SD1.5 freeze) + AR 폴백 |
| 확장성 | 새 데이터셋마다 전체 재설계 | 데이터셋별 상수 없이 자동 재유도 (per-domain re-engineering 제거) |
| 평가 범위 | 단일 도메인(steel) | 4개 이종 데이터셋 (severstal · mvtec_leather · mtd · aitex tiled/single-class) |

## 관련 노트

[[00-INDEX]] | [[02-Stage0-Prepare-Profiling]] | [[05-Stage3-ControlNet-Generation]] | [[09-Compatibility-Gate]]
