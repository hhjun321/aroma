# AROMA Distribution Profiling Report

**Dataset key**: `severstal` | **Domain**: `severstal`
**Defect instances**: 3620 | **Context patches**: 936769
**Fallback masks**: 0

## Morphology Clustering

Method: `gmm_bic` | Clusters: **5**

| Cluster ID | Label | N samples | aspect_ratio | linearity | solidity |
|-----------|-------|-----------|-------------|-----------|---------|
| 0 | linear_scratch | 861 | 10.87 | 0.99 | 0.97 |
| 1 | compact_blob | 437 | 1.33 | 0.41 | 0.94 |
| 2 | linear_scratch | 522 | 14.85 | 0.99 | 0.83 |
| 3 | general | 939 | 2.32 | 0.79 | 0.89 |
| 4 | elongated | 861 | 4.78 | 0.95 | 0.86 |

## Distribution Policies

| Feature | Distribution | Policy | N valleys |
|---------|-------------|--------|----------|
| linearity | unimodal | percentile | 0 |
| solidity | unimodal | percentile | 0 |
| extent | unimodal | percentile | 0 |
| aspect_ratio | multimodal | gmm | 4 |
| eccentricity | unimodal | percentile | 0 |
| circularity | unimodal | percentile | 0 |

## Figures

See `figures/` directory for histograms and heatmaps.
