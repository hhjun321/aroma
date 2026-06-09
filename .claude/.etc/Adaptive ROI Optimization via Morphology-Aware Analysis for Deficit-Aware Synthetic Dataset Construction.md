아래는 지금까지 확정된 **Deficit-Aware Augmentation 기반 AROMA 최종 연구 구조**를 반영한 Markdown 초안입니다.

# AROMA: Adaptive ROI Optimization via Morphology-Aware Analysis for Deficit-Aware Synthetic Dataset Construction

## 1. Research Motivation

Industrial anomaly detection often suffers from severe data imbalance and limited defect diversity.

Recent ROI-based synthetic defect generation methods such as CASDA demonstrated that selecting context-aware Regions of Interest (ROIs) improves synthetic image quality by considering relationships between defect morphology and background characteristics.

However, CASDA relies heavily on manually designed components:

* Fixed defect categories
* Fixed background categories
* Expert-defined compatibility matrices
* Manually tuned thresholds
* Dataset-specific rules

These limitations hinder scalability across different industrial domains, including:

* Severstal Steel Defect Dataset
* MVTec AD
* VisA
* PCB Inspection
* Semiconductor Wafer Inspection
* Surface Inspection Datasets

Furthermore, existing approaches primarily focus on selecting suitable ROIs rather than addressing imbalance in morphology-context distributions.

---

# 2. Research Question

Traditional ROI-based methods ask:

> Which ROI should be selected for defect synthesis?

AROMA addresses a broader question:

> How can dataset complexity be utilized to identify and compensate underrepresented morphology-context patterns during synthetic defect generation?

The objective is not merely to find valid ROIs, but to construct balanced synthetic datasets that improve representation of rare defect-context combinations.

---

# 3. Core Hypothesis

### H1

Industrial datasets exhibit different levels of morphology complexity and context complexity.

### H2

A single ROI modeling strategy is suboptimal across all datasets.

### H3

Dataset complexity can be quantified through statistical analysis of morphology and context distributions.

### H4

Complexity-aware policy selection improves morphology-context modeling quality.

### H5

Deficit-aware ROI sampling improves coverage of underrepresented morphology-context combinations.

### H6

Improved morphology-context coverage leads to higher-quality synthetic datasets.

---

# 4. Overall Framework

```text
Dataset
    ↓
Morphology Complexity Analysis
    ↓
Context Complexity Analysis
    ↓
MCI / CCI Computation
    ↓
Complexity-Aware Policy Selection
    ↓
Morphology Modeling
    ↓
Context Modeling
    ↓
Morphology-Context Prior Learning
    ↓
Deficit Analysis
    ↓
Deficit-Aware ROI Sampling
    ↓
Prototype-Based Prompt Generation
    ↓
ControlNet Synthesis
    ↓
Balanced Synthetic Dataset
```

Unlike CASDA, AROMA explicitly models distribution imbalance and allocates synthesis resources toward underrepresented morphology-context combinations.

---

# 5. Morphology Complexity Analysis

## Input Features

For each defect mask:

* Aspect Ratio
* Log Aspect Ratio
* Circularity
* Solidity
* Extent
* Eccentricity
* Fill Ratio

---

## Complexity Indicators

### Distribution Entropy

Measures diversity of morphology features.

### Valley Count

Measures the number of significant modes in feature distributions.

### Estimated Cluster Count

Represents morphology diversity.

### Cluster Recoverability

Measured using Silhouette Score.

High Silhouette:

```text
Well-separated morphology groups
```

Low Silhouette:

```text
Highly mixed morphology groups
```

---

## Morphology Complexity Index (MCI)

All indicators are normalized and equally weighted.

```text
MCI =
Mean(
Normalized Entropy,
Normalized Valley Count,
Normalized Cluster Count,
Normalized (1 - Silhouette)
)
```

Equal weighting is adopted to avoid introducing dataset-specific hyperparameters.

---

# 6. Context Complexity Analysis

## Input Features

Normal background patches are analyzed using:

* Local Variance
* Texture Entropy
* Edge Density
* Frequency Energy
* Orientation Consistency

---

## Context Indicators

### Texture Entropy

Texture diversity.

### Context Cluster Count

Estimated number of context groups.

### Frequency Complexity

Variation of spectral characteristics.

### Orientation Variance

Variation of dominant texture directions.

---

## Context Complexity Index (CCI)

```text
CCI =
Mean(
Normalized Texture Entropy,
Normalized Context Cluster Count,
Normalized Frequency Complexity,
Normalized Orientation Variance
)
```

---

# 7. Complexity-Aware Policy Selection

The primary contribution of AROMA.

Rather than applying a fixed ROI modeling strategy, AROMA selects policies according to dataset complexity.

---

## Example Policies

### Low Complexity

* Percentile Partition
* Otsu Thresholding

### Medium Complexity

* Gaussian Mixture Model (GMM)

### High Complexity

* Hierarchical Clustering

---

## Outputs

Policy selection determines:

* Clustering algorithm
* Number of clusters
* Prior estimation method
* ROI sampling strategy

---

# 8. Morphology Modeling

Morphology groups are automatically discovered.

## Input Features

* Log Aspect Ratio
* Circularity
* Solidity
* Extent
* Eccentricity

---

## Candidate Algorithms

* K-Means
* GMM
* Hierarchical Clustering

---

## Example Outputs

```text
M1: Compact Defect

M2: Elongated Defect

M3: Irregular Defect

M4: Diffuse Contamination

M5: Structural Defect
```

The number of morphology groups is automatically determined.

---

# 9. Context Modeling

Context groups are automatically discovered from texture characteristics.

## Input Features

* Local Variance
* Texture Entropy
* Edge Density
* Frequency Energy
* Orientation Consistency

---

## Example Outputs

```text
C1: Smooth Surface

C2: Directional Texture

C3: Repetitive Pattern

C4: High-Frequency Texture

C5: Complex Background
```

---

# 10. Morphology-Context Prior Learning

AROMA learns relationships directly from data.

## Objective

Estimate:

```text
P(Morphology, Context)
```

---

## Example

| Morphology | Context | Probability |
| ---------- | ------- | ----------- |
| M1         | C1      | 0.31        |
| M2         | C3      | 0.22        |
| M3         | C5      | 0.18        |

The learned prior captures realistic morphology-context co-occurrence patterns.

---

# 11. Deficit Analysis

A key extension beyond CASDA.

Rather than reproducing the observed distribution, AROMA identifies underrepresented morphology-context combinations.

---

## Observed Distribution

| Pair  | Count |
| ----- | ----- |
| M1-C1 | 450   |
| M2-C2 | 210   |
| M3-C4 | 27    |
| M4-C5 | 9     |

---

## Target Distribution

Possible strategies:

### Uniform Balancing

```text
Target(M,C) = constant
```

### Square-root Balancing

```text
Target(M,C)
∝ sqrt(Observed(M,C))
```

---

## Deficit Computation

```text
Deficit(M,C)

=
Target(M,C)
-
Observed(M,C)
```

---

## Example

| Pair  | Observed | Target | Deficit |
| ----- | -------- | ------ | ------- |
| M1-C1 | 450      | 250    | -200    |
| M3-C4 | 27       | 250    | +223    |
| M4-C5 | 9        | 250    | +241    |

Large positive deficits indicate combinations that require additional synthesis.

---

# 12. Deficit-Aware ROI Sampling

The learned deficits guide synthesis budget allocation.

---

## CASDA

```text
Compatibility Rule
↓
ROI Selection
```

---

## AROMA

```text
Deficit Analysis
↓
ROI Sampling Budget Allocation
↓
ROI Selection
```

---

### Example

Rare pair:

```text
M4-C5
```

receives higher synthesis priority than overrepresented pairs.

---

## Output

A balanced ROI candidate pool emphasizing rare morphology-context combinations.

---

# 13. Prototype-Based Prompt Generation

Unlike CASDA's manually assigned defect/background labels, AROMA derives prompts from learned morphology and context prototypes.

---

## Morphology Prototype

```text
Fragmented irregular defect
```

---

## Context Prototype

```text
Complex textured metallic surface
```

---

## Generated Prompt

```text
A fragmented irregular defect appearing on a complex textured metallic surface.
```

Prompts are automatically generated from learned cluster characteristics.

---

# 14. Synthetic Dataset Construction

The selected ROIs and generated prompts are used for:

* ControlNet-based synthesis
* Diffusion-based generation
* Inpainting
* Copy-Paste augmentation

AROMA is independent of any specific synthesis model.

---

# 15. Experimental Design

The experimental design prioritizes ROI modeling validation and deficit-aware augmentation quality.

---

# Experiment 1: MCI / CCI Validation

## Objective

Validate whether complexity indices reflect actual dataset complexity.

---

## E1-1 Cluster Recoverability

Measure:

* MCI
* CCI
* Silhouette Score

Expected relationship:

```text
MCI ↑
↓
Silhouette ↓
```

---

## E1-2 Complexity Ranking Consistency

Example ranking:

```text
Severstal
<
VisA
<
MVTec
<
PCB
```

Verify whether MCI and CCI produce consistent ordering.

---

# Experiment 2: Policy Validation

## Objective

Validate complexity-aware policy selection.

---

## E2-1 Policy Selection Analysis

| Dataset   | Complexity | Selected Policy |
| --------- | ---------- | --------------- |
| Severstal | Low        | Otsu            |
| VisA      | Medium     | GMM             |
| PCB       | High       | Hierarchical    |

---

## E2-2 Alternative Policy Comparison

| Policy       | KL Divergence ↓ |
| ------------ | --------------- |
| Otsu         | 0.44            |
| GMM          | 0.21            |
| Hierarchical | 0.10            |

Expected outcome:

```text
Selected Policy
=
Best ROI Modeling Performance
```

---

# Experiment 3: Deficit-Aware ROI Modeling

## Objective

Evaluate how effectively AROMA balances morphology-context distributions.

Methods:

* Random ROI
* CASDA
* AROMA

---

## E3-1 Morphology Coverage

Coverage of morphology clusters.

---

## E3-2 Context Coverage

Coverage of context clusters.

---

## E3-3 Rare Pair Coverage

Coverage of underrepresented morphology-context combinations.

---

## E3-4 Distribution Balance

Metrics:

* Normalized Entropy
* Gini Coefficient
* Coverage Ratio

Expected outcome:

```text
AROMA > CASDA > Random
```

---

# Experiment 4: Synthetic Quality Verification

## Objective

Verify that improved ROI modeling leads to better synthetic quality.

---

## Configuration

Same:

* ControlNet architecture
* Training schedule
* Hyperparameters

Methods:

* Random ROI
* CASDA ROI
* AROMA ROI

---

## Metrics

* FID
* KID
* LPIPS

This experiment serves as downstream validation rather than the primary contribution.

---

# 16. Expected Contributions

## C1

Morphology Complexity Index (MCI)

## C2

Context Complexity Index (CCI)

## C3

Complexity-Aware ROI Policy Selection

## C4

Automatic Morphology Modeling

## C5

Automatic Context Modeling

## C6

Data-Driven Morphology-Context Prior Learning

## C7

Deficit-Aware ROI Sampling Strategy

## C8

Balanced Synthetic Dataset Construction Framework

---

# Final Definition

AROMA is a Complexity-Aware ROI Augmentation Framework that automatically analyzes morphology complexity, context complexity, and morphology-context imbalance to guide deficit-aware synthetic defect generation.

Unlike CASDA, which relies on manually designed compatibility rules, AROMA learns morphology-context structures directly from data and allocates synthesis resources toward underrepresented defect patterns, enabling scalable and domain-adaptive synthetic dataset construction across diverse industrial inspection domains.
