# AROMA Distribution Profiling Report

**Dataset key**: `mtd` | **Domain**: `mtd`
**Defect instances**: 388 | **Context patches**: 27512
**Fallback masks**: 0

## Morphology Clustering

Method: `gmm_bic` | Clusters: **5**

| Cluster ID | Label | N samples | aspect_ratio | linearity | solidity |
|-----------|-------|-----------|-------------|-----------|---------|
| 0 | compact_blob | 61 | 1.17 | 0.26 | 0.87 |
| 1 | linear_scratch | 85 | 7.85 | 0.97 | 0.88 |
| 2 | general | 127 | 2.78 | 0.85 | 0.85 |
| 3 | linear_scratch | 36 | 10.21 | 0.96 | 0.59 |
| 4 | compact_blob | 79 | 1.54 | 0.57 | 0.93 |

## Distribution Policies

| Feature | Distribution | Policy | N valleys |
|---------|-------------|--------|----------|
| linearity | multimodal | gmm | 2 |
| solidity | multimodal | gmm | 3 |
| extent | unimodal | percentile | 0 |
| aspect_ratio | multimodal | gmm | 4 |
| eccentricity | unimodal | percentile | 0 |
| circularity | unimodal | percentile | 0 |

## Figures

See `figures/` directory for histograms and heatmaps.
