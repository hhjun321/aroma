# AROMA Exp 4 — Downstream Anomaly Detection 평가

비교: Baseline (real only) vs Random ROI vs AROMA ROI | copy-paste synthesis 동일

## mvtec_cable

### Image AUROC

| 조건         |      patchcore |      simplenet |   efficient_ad |   rd_plus_plus |
|------------|-------------- | -------------- | -------------- | --------------|
| baseline   |            N/A |         0.7646 |            N/A |            N/A |
| random     |            N/A |         0.5015 |            N/A |            N/A |
| aroma      |            N/A |         0.4996 |            N/A |            N/A |

### Delta (AROMA − Random) per model

- **patchcore**: N/A
- **simplenet**: image_auroc -0.0019, pixel_auroc -0.0628
- **efficient_ad**: N/A
- **rd_plus_plus**: N/A
