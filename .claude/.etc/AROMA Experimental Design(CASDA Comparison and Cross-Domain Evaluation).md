# AROMA Experimental Design: CASDA Comparison and Cross-Domain Evaluation

## 1. Experimental Philosophy

AROMA is proposed as a **domain-adaptive ROI modeling framework**.

The objective is not to improve the synthetic generation model itself, but to improve:

- ROI modeling
- Morphology-context matching
- ROI selection
- Synthetic training data quality

through adaptive dataset-driven analysis.

Therefore, all synthesis experiments use the same generation pipeline, while only the ROI selection strategy is changed.

---

# 2. Why CASDA Cannot Be Applied to All Datasets

CASDA was originally developed and validated on:

**Severstal Steel Defect Detection Dataset**

Reference:

- Kaggle Severstal Steel Defect Detection

CASDA relies on manually engineered:

- Defect morphology categories
- Background categories
- Compatibility matrices
- Prompt generation rules

These components were specifically designed for steel surface inspection.

Applying CASDA to new domains would require:

- Redefining defect categories
- Redefining background categories
- Manually rebuilding compatibility matrices

for every dataset.

Such redesign would violate the purpose of evaluating domain-adaptive ROI modeling.

Therefore, CASDA comparisons are restricted to the dataset domain for which it was originally designed.

---

# 3. Experimental Structure

The evaluation is divided into two complementary experiments.

---

# Experiment A: CASDA Baseline Comparison

## Dataset

Severstal Steel Defect Detection

## Compared Methods

### Random ROI Selection

Randomly selects ROI candidates.

---

### CASDA

Original manually designed ROI modeling strategy.

Uses:

- Defect Type
- Background Type
- Compatibility Matrix

for ROI scoring.

---

### AROMA

Proposed adaptive ROI modeling framework.

Uses:

- MCI
- CCI
- Meta Policy Generator
- Dynamic Morphology Modeling
- Dynamic Context Modeling
- Deficit-aware ROI Selection

---

## Purpose

Evaluate whether AROMA can outperform CASDA even on the dataset for which CASDA was originally designed.

---

## Evaluation Metrics

### ROI Modeling Quality

- Morphology Coverage
- Context Coverage
- Rare Pair Coverage
- Diversity Entropy

---

### Synthetic Quality

FID

Comparison:

```text
Real Defect Patch
vs
Synthetic Defect Patch
```

---

### Downstream AD Performance

PaDiM

Training:

```text
Normal Images
+
Synthetic Images
```

Testing:

```text
Real Test Images Only
```

Metrics:

- Image-level AUROC
- Pixel-level AUROC
- PRO Score (optional)

---

# Experiment B: Cross-Domain Generalization

## Datasets

### MVTec AD

- Cable

### VisA

- Cashew
- PCB

---

## Compared Methods

### Random ROI Selection

Baseline.

### AROMA

Adaptive ROI Modeling.

---

## CASDA Exclusion

CASDA is not evaluated on these datasets.

Reason:

CASDA requires manually engineered:

- Morphology definitions
- Background definitions
- Compatibility matrices

for each domain.

Such redesign contradicts the objective of evaluating domain-independent ROI modeling.

---

## Purpose

Evaluate whether AROMA can operate on unseen industrial datasets without any manual redesign.

---

## Evaluation Metrics

### ROI Modeling Quality

- Morphology Coverage
- Context Coverage
- Rare Pair Coverage
- Diversity Entropy

---

### Synthetic Quality

FID

Comparison:

```text
Real Defect Patch
vs
Synthetic Defect Patch
```

---

### Downstream AD Performance

PaDiM

Training:

```text
Normal Images
+
Synthetic Images
```

Testing:

```text
Real Test Images Only
```

Metrics:

- Image-level AUROC
- Pixel-level AUROC

---

# 4. Synthesis Pipeline

All methods use the same synthesis procedure.

Only ROI selection differs.

```text
ROI Selection
        ↓
Copy-Paste Synthesis
        ↓
Synthetic Dataset
        ↓
PaDiM Training
        ↓
Evaluation
```

---

# 5. Copy-Paste Limitation Statement

The objective of AROMA is ROI modeling rather than synthesis model improvement.

Therefore, Copy-Paste is intentionally used as a controlled generation mechanism.

Limitation:

```text
Copy-Paste preserves original defect appearance
and cannot generate novel defect morphologies.
```

This limitation applies equally to:

- Random
- CASDA
- AROMA

ensuring fair comparison of ROI modeling quality.

---

# 6. Expected Claims

## Claim 1

AROMA outperforms Random ROI Selection.

---

## Claim 2

AROMA outperforms CASDA on the Severstal dataset.

---

## Claim 3

AROMA improves synthetic data quality as measured by FID.

---

## Claim 4

AROMA improves downstream anomaly detection performance.

---

## Claim 5

AROMA generalizes to multiple industrial domains without manual redesign.

---

# Final Experimental Scope

| Dataset | Random | CASDA | AROMA | Purpose |
|----------|----------|----------|----------|----------|
| Severstal | ✓ | ✓ | ✓ | CASDA Comparison |
| MVTec Cable | ✓ | ✗ | ✓ | Cross-Domain Validation |
| VisA Cashew | ✓ | ✗ | ✓ | Cross-Domain Validation |
| VisA PCB | ✓ | ✗ | ✓ | Cross-Domain Validation |

This design ensures a fair comparison against the original CASDA framework while simultaneously validating AROMA's domain-adaptive capability.



