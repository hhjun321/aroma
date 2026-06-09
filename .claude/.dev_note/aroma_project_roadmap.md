# AROMA 프로젝트 — 전체 스크립트 로드맵

---

## (사용할 skills: feature-dev)

## 개요

AROMA-sharpened-spec.md (명확성 91/100) 기준 전체 파이프라인을 구현한다.

**핵심 발견**: `D:\project\aroma\scripts\distribution_profiling.py` (2026-06-04)가
형태학 특징 추출 / 컨텍스트 특징 추출 / 분포 분석 / 형태학 클러스터링 / 호환성 행렬 / Deficit 분석을 **이미 완전 구현**함.
aroma-plus는 이 위에 MCI/CCI 스칼라 지표 + Meta Policy Generator + Prompt Generation을 추가하는 형태로 구성한다.

참조:
- `D:\project\aroma-plus\AROMA-sharpened-spec.md` (91/100)
- `D:\project\aroma-plus\.claude\.dev_note\aroma_step1_complexity-analysis.md` (Step 1 상세)
- `D:\project\aroma\scripts\distribution_profiling.py` (재활용 기존 구현)
- `D:\project\aroma-plus\.claude\.etc\dataset_config.json` (Google Drive 실제 경로)

---

## 실제 데이터셋 구성

Base: `/content/drive/MyDrive/data/Aroma/`

| 도메인 | 카테고리 | 비고 |
|--------|---------|------|
| **ISP** | ASM (256px), LSM_1 (512px, 3678 train), LSM_2 (512px) | 권장 시작: LSM_1 |
| **MVTec** | bottle, cable, capsule, carpet, grid, hazelnut, leather, metal_nut, pill, screw, tile, toothbrush, transistor, wood, zipper (15개) | |
| **VisA** | candle, capsules, cashew, chewinggum, fryum, macaroni, pcb4, pipe_fryum (8개) | 사전에 `prepare_visa.py` 필요 |

구조 패턴:
```
{base}/{domain}/{category}/train/good/      ← image_dir (정상 이미지)
{base}/{domain}/{category}/test/{defect}/   ← seed_dirs (결함 seed 이미지)
```

> Severstal: `/content/drive/MyDrive/data/Severstal/` — CASDA 비교 실험(Exp 1)에만 사용.
> CASDA ROI: `$SEVERSTAL/roi_patches_v5.1/roi_metadata.csv`

---

## 기존 구현 재활용 (`distribution_profiling.py`)

`D:\project\aroma\scripts\distribution_profiling.py` 가 커버하는 범위:

| 기존 단계 | 출력 파일 | 내용 |
|----------|----------|------|
| Step 2 (형태학 특징) | `morphology_features.csv` | linearity, solidity, extent, aspect_ratio, eccentricity, circularity |
| Step 3 (컨텍스트 특징) | `context_features.csv` | local_variance, edge_density, texture_entropy, frequency_energy, orientation_consistency (64px 패치) |
| Step 4 (분포 분석) | `distribution_analysis.json` | valley 검출 → Unimodal/Bimodal/Multimodal → Percentile/Otsu/GMM 자동 선택 |
| Step 5 (형태학 클러스터링) | `morphology_clusters.json` | GMM/BIC 자동 k 선택 (max_k=5) |
| Step 6 (호환성 행렬) | `compatibility_matrix.json` | P(context_cell \| cluster) |
| Step 7 (Deficit 분석) | `deficit_analysis.json` | 클러스터별 deficit + target_synthetic 가중치 |
| Step 8/9 (정책/리포트) | `threshold_policies.json`, `recommended_config.yaml`, `analysis_report.md` | |

**aroma-plus가 추가 구현할 것:**
1. **MCI/CCI 스칼라** — distribution_analysis.json 출력을 읽어 데이터셋 단위 복잡도 지표 산출
2. **Meta Policy Generator** — MCI/CCI → 데이터셋 단위 정책 선택 (기존은 feature별 개별 정책)
3. **Semantic Prompt Generation** — 클러스터 레이블 + context 특성 → 자연어 프롬프트
4. **ROI Selection** — deficit_analysis.json + prior → deficit-aware ROI 스코어링 (기존 deficit은 있으나 ROI 스코어링 없음)

---

## 전체 파이프라인 구조

```
Dataset
  ↓ [Phase 0 — 기존] distribution_profiling.py
  │   → morphology_features.csv, context_features.csv
  │   → distribution_analysis.json, morphology_clusters.json
  │   → compatibility_matrix.json, deficit_analysis.json
  │
  ↓ [Step 1 — 신규] compute_complexity.py
  │   → MCI, CCI (스칼라)
  │   → morphology_policy, context_policy (데이터셋 단위)
  │
  ↓ [Step 2 — 신규] prompt_generation.py
  │   → semantic prompts per cluster × context combination
  │
  ↓ [Step 3 — 신규] roi_selection.py
  │   → deficit_aware ROI 후보 목록 (mask crop + score + prompt)
  │
  ↓ [Step 4 — 신규] generate_defects.py
      → synthetic defect images
```

Colab 실행 순서:
```python
# Phase 0 (기존 스크립트 — dataset_key별 1회 실행)
!python $AROMA_REF/scripts/distribution_profiling.py \
    --dataset_config $DATASET_CONFIG \
    --dataset_key    isp_LSM_1 \
    --output_dir     $AROMA_OUT/profiling/isp_LSM_1

# Step 1 (신규)
!python $AROMA_SCRIPTS/compute_complexity.py \
    --profiling_dir  $AROMA_OUT/profiling/isp_LSM_1 \
    --output_dir     $AROMA_OUT/complexity/isp_LSM_1

# Step 2 (신규)
!python $AROMA_SCRIPTS/prompt_generation.py \
    --profiling_dir  $AROMA_OUT/profiling/isp_LSM_1 \
    --complexity_dir $AROMA_OUT/complexity/isp_LSM_1 \
    --output_dir     $AROMA_OUT/prompts/isp_LSM_1

# Step 3 (신규)
!python $AROMA_SCRIPTS/roi_selection.py \
    --profiling_dir  $AROMA_OUT/profiling/isp_LSM_1 \
    --prompts_dir    $AROMA_OUT/prompts/isp_LSM_1 \
    --sampling_strategy deficit_aware \
    --output_dir     $AROMA_OUT/roi/isp_LSM_1
```

---

## 신규 구현 항목

### Step 1: `scripts/aroma/compute_complexity.py`

입력: `distribution_analysis.json` (Phase 0 출력)

구현 항목:
- `compute_mci(distribution_analysis) -> float`
  - distribution_analysis.json 읽어 feature별 통계 추출
  - `MCI = Mean(z-score(Entropy), z-score(ValleyCount), z-score(ClusterCount), z-score(1-Silhouette))`
  - Silhouette: morphology_clusters.json에서 클러스터 품질 추출
- `compute_cci(context_features_csv) -> float`
  - 동일 패턴: TextureEntropy, ClusterCount, FrequencyComplexity, OrientationVariance
- `run_meta_policy_generator(mci, cci, distribution_analysis) -> dict`
  - MCI/CCI → Distribution Diagnostics → Candidate Policies (max 2~3개 pruning) → Empirical Evaluation → Best Policy
  - **Best Policy 선택 기준**: Highest Silhouette Score
  - **동점/근접 처리**: Δ < ε → Stability Score로 tie-break (BIC/AIC·복합 AutoML Score 사용 안 함)
- 출력: `complexity_report.json` (MCI, CCI, morphology_policy, context_policy, stability_margin)
- **단계 완료 기준**: `complexity_report.json` 파일 생성 확인 → Step 2 착수
- Ablation 지원: `--weight_mode equal|entropy_heavy|cluster_heavy`

상세 명세: `aroma_step1_complexity-analysis.md`

---

### Step 2: `scripts/aroma/prompt_generation.py`

입력: `morphology_clusters.json`, `compatibility_matrix.json` (Phase 0), `complexity_report.json` (Step 1)

구현 항목:
- `generate_morphology_descriptor(cluster_centroid) -> str`
  - centroid의 aspect_ratio, linearity, solidity 기반 자연어 설명
  - 예: "Highly elongated metallic scratch"
- `generate_context_descriptor(context_bin) -> str`
  - context 5개 feature 분위수 기반
  - 예: "Directional brushed metal texture"
- `generate_prior_modifier(morph_cluster_id, ctx_bin, compatibility_matrix) -> str`
  - P(ctx|cluster) 높은 조합 → "Aligned with dominant texture direction"
- 최종 프롬프트: `morphology + context + prior_modifier`
- **템플릿 방식**: 고정 템플릿 (LLM 생성 사용 안 함) ✓

---

### Step 3: `scripts/aroma/roi_selection.py`

입력: `morphology_features.csv`, `morphology_clusters.json`, `deficit_analysis.json`, `prompts/` (Phase 0 + Step 2)

구현 항목:
- `score_roi(mask_features, morph_cluster_id, ctx_bin, deficit) -> float`
  - `ROI_score = 0.4 × P(M_i) + 0.4 × P(C_j) + 0.2 × Deficit(M_i, C_j)` ✓
- **Rare Pair 정의**: `Deficit(M_i, C_j) = 1 - Observed(M_i, C_j) / Expected(M_i, C_j)` ✓
- **Rare Pair 선택**: Top-K Deficit Quantile ✓
- 샘플링 전략:
  - `deficit_aware`: Top-K Deficit Quantile 기반 오버샘플링 (기본값)
  - `top_k`: 상위 K개
  - `weighted`: 확률 비례
- 출력: ROI 목록 (mask_crop_path, morph_label, ctx_label, score, prompt, prior_prob)

---

### Step 4: `scripts/aroma/generate_defects.py`

입력: Step 3 ROI 목록

구현 항목:
- 합성 방법 플러그인: `--method copy_paste | controlnet | inpainting`
- `copy_paste_synthesis(roi, target_image)` — 기본값, GPU 불필요
- ControlNet 연동은 인터페이스만 정의 (Colab GPU 환경 별도)
- 출력: synthetic image + annotation
- TODO: Copy-Paste blending 방식 (Poisson vs alpha composite)

---

## 실험 스크립트

### Exp 1: `scripts/aroma/experiments/exp1_roi_distribution.py`

CASDA vs AROMA ROI 분포 비교 (Severstal only — CASDA 비교 데이터셋)
지표: KL Divergence, Wasserstein Distance, Earth Mover Distance

```python
!python $AROMA_SCRIPTS/experiments/exp1_roi_distribution.py \
    --casda_roi    $CASDA_ROI_DIR \
    --aroma_roi    $AROMA_OUT/roi/severstal \
    --output_dir   $AROMA_OUT/exp1
```

---

### Exp 2: `scripts/aroma/experiments/exp2_roi_quality.py`

ROI 품질 분석 (ISP + MVTec + VisA)
지표: Morphology Coverage, Context Coverage, Rare Pair Coverage, Entropy, Gini

```python
!python $AROMA_SCRIPTS/experiments/exp2_roi_quality.py \
    --aroma_roi    $AROMA_OUT/roi \
    --random_roi   $RANDOM_ROI_DIR \
    --dataset_config $DATASET_CONFIG \
    --output_dir   $AROMA_OUT/exp2
```

---

### Exp 3: `scripts/aroma/experiments/exp3_synthetic_quality.py`

합성 품질 평가 (단일 대표 데이터셋 — ISP LSM_1 또는 MVTec 카테고리 1개, ControlNet 고정)
⚠️ FID/KID 계산 = GPU 집약 — 부하 측정 항목, 자동 실행 금지 (load-test-policy.md)
지표: FID, KID, LPIPS

---

### Exp 4: `scripts/aroma/experiments/exp4_downstream_ad.py`

하류 이상 탐지 성능 평가
모델: PatchCore, SimpleNet, EfficientAD, RD++
지표: Image AUROC, Pixel AUROC, PRO

---

### Exp 5: `scripts/aroma/experiments/exp5_crossdomain.py`

크로스 도메인 일반화 (retraining 없음)
도메인: ISP (ASM/LSM_1/LSM_2), MVTec (대표), VisA (pcb4 포함)

```python
!python $AROMA_SCRIPTS/experiments/exp5_crossdomain.py \
    --dataset_config $DATASET_CONFIG \
    --domains        isp mvtec visa \
    --aroma_scripts  $AROMA_SCRIPTS \
    --output_dir     $AROMA_OUT/exp5
```

---

## 전체 디렉토리 구조

```
scripts/aroma/                        ← aroma-plus 신규 스크립트
├── compute_complexity.py             [Step 1 — MCI/CCI + Meta Policy]
├── prompt_generation.py              [Step 2]
├── roi_selection.py                  [Step 3]
├── generate_defects.py               [Step 4]
├── experiments/
│   ├── exp1_roi_distribution.py
│   ├── exp2_roi_quality.py
│   ├── exp3_synthetic_quality.py
│   ├── exp4_downstream_ad.py
│   └── exp5_crossdomain.py
└── config/
    ├── aroma_step1.yaml
    └── dataset_config.json           ← .claude/.etc/에서 복사

D:\project\aroma\scripts\             ← 기존 AROMA 스크립트 (재활용)
└── distribution_profiling.py         ← Phase 0, 직접 호출
```

---

## 환경변수 (Colab 셀 추가 필요)

```python
import os
# aroma-plus 신규 스크립트
os.environ['AROMA_SCRIPTS']   = "/content/AROMA_PLUS/scripts/aroma"
os.environ['AROMA_OUT']       = "/content/drive/MyDrive/data/Aroma/aroma_output"
os.environ['AROMA_DATA_BASE'] = "/content/drive/MyDrive/data/Aroma"
os.environ['DATASET_CONFIG']  = "/content/AROMA_PLUS/scripts/aroma/config/dataset_config.json"
os.environ['RANDOM_ROI_DIR']  = "/content/drive/MyDrive/data/Aroma/random_roi"

# 기존 AROMA 스크립트 (Phase 0)
os.environ['AROMA_REF']       = "/content/AROMA"   # D:\project\aroma clone 경로

# Severstal (Exp 1 CASDA 비교 전용)
SEVERSTAL = '/content/drive/MyDrive/data/Severstal'
os.environ['SEVERSTAL']        = SEVERSTAL
os.environ['SEVERSTAL_IMAGES'] = f'{SEVERSTAL}/train_images'
os.environ['SEVERSTAL_CSV']    = f'{SEVERSTAL}/train.csv'
os.environ['CASDA_ROI_DIR']    = f'{SEVERSTAL}/roi_patches_v5.1'
```

---

## 미확정 사항 (TODO 요약)

- ~~Step 1: Empirical Policy Evaluation 평가 지표~~ → Silhouette 확정, Δ<ε시 Stability ✓
- ~~Step 2: 프롬프트 템플릿 방식~~ → 고정 템플릿 확정 ✓
- ~~Step 3: ROI_score 수식~~ → 0.4M + 0.4C + 0.2D 확정 ✓
- ~~Step 3: Rare Pair threshold~~ → Top-K Deficit Quantile 확정 ✓
- **Step 4**: Copy-Paste blending 방식 (Poisson vs alpha composite) — 미확정
- **Exp 3**: 단일 대표 데이터셋 확정 (ISP LSM_1 vs MVTec 카테고리) — 미확정
- VisA: 실험 전 `prepare_visa.py` 실행 자동화 방안
- ISP 도메인: ground-truth 마스크 없음 → SAM/Otsu fallback, 형태학 통계 신뢰도 제한
