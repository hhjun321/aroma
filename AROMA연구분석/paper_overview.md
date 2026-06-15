# AROMA: Adaptive ROI Modeling for Anomaly Synthesis in Industrial Inspection

> **상태**: 연구 방향 정리 문서 (추후 SCI 제출 예정)  
> **작성일**: 2026-06-15  
> **포함 실험**: Exp 2 (Cross-Domain Generalization) — Exp 1 (CASDA Comparison) 제외 (미확립)

---

## Abstract (Draft)

Anomaly detection in industrial inspection is fundamentally limited by the scarcity of defect training data.
Existing synthetic data augmentation methods either rely on deep generative models requiring large training datasets,
or apply fixed ROI selection strategies that ignore the statistical distribution of the target domain.
We propose **AROMA (Adaptive ROI Modeling for Anomaly synthesis)**, a data-driven framework that automatically
analyzes the morphological and contextual complexity of a target domain and performs deficit-aware ROI selection
to maximize the diversity of synthesized defect training data.

AROMA operates without generative model training:
it characterizes defect morphology through **Morphology Complexity Index (MCI)** and image context
through **Context Complexity Index (CCI)**, computes a deficit signal representing underrepresented
(morphology, context) pairs, and allocates ROIs via a **pair-aware two-stage quota assignment** that
guarantees coverage of all pair types while prioritizing underrepresented ones.

We evaluate AROMA on four industrial datasets across multiple domains
(isp\_LSM\_1, MVTec Cable, VisA Cashew, VisA PCB) using copy-paste synthesis as a controlled generation baseline.
Experimental results show that AROMA improves synthetic data diversity (higher morphology/context coverage,
rare pair coverage) and downstream anomaly detection performance (Image-level AUROC, Pixel-level AUROC
with PaDiM) compared to uniform random ROI selection, **without any manual domain-specific redesign**.

---

## 1. Introduction

### 1.1 Problem Statement

Industrial visual inspection systems based on anomaly detection require large, diverse training datasets.
However, defect samples are inherently rare in production environments.
This data scarcity problem motivates synthetic defect augmentation:
generating artificial defect images to supplement real training data.

Current approaches to synthetic defect generation fall into three categories:

| Category | Approach | Limitation |
|----------|----------|------------|
| Deep generative models | GAN, Diffusion model | Requires large defect datasets to train; domain-specific |
| Copy-paste / cut-paste | Paste real defect crops onto normal backgrounds | ROI selection is random or manual |
| Domain-specific ROI modeling | CASDA: manually designed compatibility matrices | Requires per-domain expert redesign |

The central hypothesis of this work is:
**the quality of synthetic training data is more sensitive to WHERE defects are placed (ROI selection)
than to HOW they are generated (synthesis model).**

### 1.2 Motivation: ROI Modeling as the Key Variable

Consider two extreme ROI selection strategies with identical copy-paste synthesis:

- **Random selection**: uniform sampling from all candidate ROIs → high coverage by chance, low quality awareness
- **Score-top selection**: greedy selection by ROI quality score → high individual quality, low diversity (dominant context monopolizes)

Both strategies fail to address the fundamental problem:
**underrepresented (morphology, context) pairs receive disproportionately few ROIs**,
creating a biased synthetic training distribution.

AROMA addresses this by:
1. Quantifying domain complexity (MCI, CCI) from normal images — no defect labels required
2. Computing a deficit signal per (morphology cluster, context bin) pair
3. Allocating ROI quotas proportional to deficit, with a base coverage guarantee

### 1.3 Contribution

This paper makes the following contributions:

1. **Adaptive domain characterization**: MCI and CCI automatically extract morphological and contextual
   complexity from normal images, enabling zero-shot adaptation to new industrial domains.

2. **Deficit-aware ROI modeling**: We introduce a per-pair deficit metric that quantifies the
   underrepresentation of each (defect morphology, background context) combination.
   ROI selection is driven by this signal rather than by individual ROI quality scores alone.

3. **Pair-aware two-stage allocation**: A Hamilton-method quota assignment (Phase 1) guarantees
   coverage of all pair types, followed by a quality-aware global backfill (Phase 2) that ensures
   the exact requested number of ROIs.

4. **Domain-agnostic pipeline**: AROMA requires no manual category definitions, compatibility matrices,
   or domain-specific tuning.
   The same pipeline runs on all evaluated datasets without modification.

5. **Controlled evaluation framework**: By fixing the synthesis model (copy-paste) across all comparison
   methods and varying only ROI selection, we isolate the ROI modeling contribution to synthetic data quality.

### 1.4 Scope and Limitation

Copy-paste synthesis preserves the original defect appearance and therefore cannot generate novel
defect morphologies. The objective of this study is not to improve the synthesis model itself,
but to evaluate whether adaptive ROI modeling improves the quality of synthesized training data.

This limitation applies equally to all compared methods (Random, AROMA), ensuring a fair comparison.

---

## 2. Methodology Overview

### 2.1 Pipeline

```
Normal Images + Defect Images
        ↓
Phase 0: Distribution Profiling
        (morphology clustering, context binning, compatibility analysis)
        ↓
Step 1: Complexity Analysis
        MCI (Morphology Complexity Index)
        CCI (Context Complexity Index)
        deficit per (cluster_id, cell_key) pair
        ↓
Step 2: Prompt Generation
        per-ROI text prompts for annotation
        ↓
Step 3: ROI Selection — Pair-Aware 2-Stage Allocation
        Phase 1: quota per pair (Hamilton method, deficit-proportional)
        Phase 2: quality backfill (global roi_score top-k)
        → roi_selected.json
        ↓
Step 4: Copy-Paste Synthesis
        → synthetic defect images + annotations.json
```

### 2.2 Key Concepts

**Morphology Complexity Index (MCI)**  
Measures the diversity of defect morphology across the dataset.
Computed from defect shape features (linearity, solidity, aspect ratio, area) clustered into K groups.
High MCI = high morphological variety among defects.

**Context Complexity Index (CCI)**  
Measures the diversity of normal background context.
Computed from background patch features (gradient entropy, frequency content, texture) discretized into grid cells.
High CCI = diverse background patterns where defects may occur.

**Deficit Signal**  
For each (morphology cluster, context cell) pair:
```
deficit(M_i, C_j) = max(0, P(C_j | normal) - P(C_j | defect near M_i))
```
High deficit = context C_j is common in normal images but rarely paired with morphology M_i in defect instances.
ROI selection should allocate more slots to high-deficit pairs to bridge this gap.

**Pair-Aware Allocation**  
Phase 1 (Coverage + Deficit):
- Base quota = 1 for every (cluster, cell) pair (guarantees coverage)
- Remaining quota distributed proportionally to pair-level mean deficit via Hamilton's Largest Remainder method

Phase 2 (Quality Backfill):
- Unfilled slots (when a pair has fewer candidates than quota) filled by global roi_score top-k
- Guarantees n_selected = top_k exactly

---

## 3. Experimental Setup

### 3.1 Datasets (Exp 2: Cross-Domain Generalization)

| Dataset | Domain | Defect Types | Image Size |
|---------|--------|-------------|------------|
| isp\_LSM\_1 | ISP sensor inspection | Multiple | Variable |
| MVTec Cable | Industrial cable | Cut, bent, missing cable, ... | ~1024px |
| VisA Cashew | Agricultural product | Crack, dent, ... | 800×600 |
| VisA PCB | PCB board | Burn, missing component, ... | 800×600 |

No domain-specific configuration changes between datasets.

### 3.2 Compared Methods

| Method | ROI Selection | Synthesis |
|--------|-------------|-----------|
| Random | Uniform random from roi_candidates.json | Copy-paste (identical) |
| AROMA | Pair-aware deficit-aware allocation | Copy-paste (identical) |

### 3.3 Evaluation Metrics

**ROI Modeling Quality** (pre-synthesis, CPU-only):
- `morphology_coverage`: fraction of unique cluster IDs covered
- `context_coverage`: fraction of unique cell keys covered
- `rare_pair_coverage`: fraction of deficit>0 pairs covered
- `entropy`: normalized Shannon entropy of cluster distribution
- `gini`: Gini coefficient of cluster frequency (lower = more uniform)

**Synthesis Quality** (FID):
- FID(real defect patches, synthetic defect patches)
- Patch = defect bbox crop; lower FID = better distribution match

**Downstream Anomaly Detection** (PaDiM, ResNet-18):
- Train: Real Normal + Synthetic defects (per method)
- Test: Real Only (synthetic NEVER in test set)
- Metrics: Image-level AUROC, Pixel-level AUROC

### 3.4 Hyperparameters

| Parameter | Value |
|-----------|-------|
| n_selected (top_k) | 200 per dataset |
| n_per_roi | 3 synthetic images per ROI |
| PaDiM backbone | ResNet-18 |
| PaDiM seed | 42 |
| FID feature | Inception v3 pool3 (dim=2048) |

---

## 4. Expected Results

Based on the AROMA design principles, the following outcomes are expected:

**Claim 1**: AROMA achieves higher `rare_pair_coverage` than Random (deficit-aware selection guarantees minority pair representation).

**Claim 3**: AROMA achieves lower FID than Random (more contextually appropriate ROIs → more realistic defect composites).

**Claim 4**: AROMA improves downstream anomaly detection (Image AUROC, Pixel AUROC) over Random ROI baseline.

**Claim 5**: AROMA generalizes across all 4 datasets without domain-specific redesign (same pipeline, same hyperparameters).

---

## 5. Related Work (Placeholder)

- Industrial anomaly detection: PaDiM [Defard+21], PatchCore [Roth+22], FastFlow [Yu+21]
- Copy-paste augmentation: [Ghiasi+21], CutPaste [Li+21]
- Domain-specific ROI modeling: CASDA [internal reference]
- Dataset complexity: MVTec AD [Bergmann+19], VisA [Zou+22]

---

## 6. Implementation Notes

- Pipeline code: `D:\project\aroma\scripts\aroma\`
- Key scripts: `distribution_profiling.py`, `compute_complexity.py`, `prompt_generation.py`, `roi_selection.py`, `generate_defects.py`
- Experiment: `scripts/aroma/experiments/exp2_roi_quality.py` (ROI quality), `exp1_casda_comparison.py` (FID + AD)
- Colab guides: `AROMA연구분석/colab_execute/`

---

## 7. TODO (연구 완료 전 필요)

- [ ] Exp 2 전체 실행 및 수치 확인 (isp_LSM_1, mvtec_cable, visa_cashew, visa_pcb)
- [ ] FID 수치 확보 (torchmetrics, Colab GPU)
- [ ] PaDiM 학습 + AUROC 수치 확보
- [ ] Abstract 수치 placeholder `[X.X]` 채우기
- [ ] Related Work 섹션 실제 citation 추가
- [ ] 타겟 저널 결정 (IEEE TII / Pattern Recognition / Sensors 등 검토)
