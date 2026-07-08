# AROMA Distribution Profiling Report

**Dataset key**: `mvtec_leather` | **Domain**: `mvtec`
**Defect instances**: 92 | **Context patches**: 86106
**Fallback masks**: 0

## Morphology Clustering

Method: `gmm_bic` | Clusters: **3**

| Cluster ID | Label | N samples | aspect_ratio | linearity | solidity |
|-----------|-------|-----------|-------------|-----------|---------|
| 0 | compact_blob | 22 | 1.70 | 0.63 | 0.92 |
| 1 | elongated | 46 | 4.88 | 0.94 | 0.82 |
| 2 | compact_blob | 24 | 1.17 | 0.25 | 0.96 |

## Distribution Policies

| Feature | Distribution | Policy | N valleys |
|---------|-------------|--------|----------|
| linearity | multimodal | gmm | 10 |
| solidity | multimodal | gmm | 4 |
| extent | unimodal | percentile | 0 |
| aspect_ratio | multimodal | gmm | 9 |
| eccentricity | unimodal | percentile | 0 |
| circularity | bimodal | otsu | 1 |

## Figures

See `figures/` directory for histograms and heatmaps.
