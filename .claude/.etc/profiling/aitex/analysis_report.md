# AROMA Distribution Profiling Report

**Dataset key**: `aitex` | **Domain**: `aitex`
**Defect instances**: 311 | **Context patches**: 114174
**Fallback masks**: 0

## Morphology Clustering

Method: `gmm_bic` | Clusters: **5**

| Cluster ID | Label | N samples | aspect_ratio | linearity | solidity |
|-----------|-------|-----------|-------------|-----------|---------|
| 0 | compact_blob | 80 | 1.66 | 0.59 | 0.88 |
| 1 | linear_scratch | 103 | 41.76 | 1.00 | 0.56 |
| 2 | elongated | 57 | 4.43 | 0.94 | 0.88 |
| 3 | linear_scratch | 46 | 11.28 | 0.98 | 0.55 |
| 4 | compact_blob | 25 | 1.14 | 0.22 | 0.95 |

## Distribution Policies

| Feature | Distribution | Policy | N valleys |
|---------|-------------|--------|----------|
| linearity | unimodal | percentile | 0 |
| solidity | multimodal | gmm | 6 |
| extent | bimodal | otsu | 1 |
| aspect_ratio | multimodal | gmm | 5 |
| eccentricity | unimodal | percentile | 0 |
| circularity | multimodal | gmm | 2 |

## Figures

See `figures/` directory for histograms and heatmaps.
