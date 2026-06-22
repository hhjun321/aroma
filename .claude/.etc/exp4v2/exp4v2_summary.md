exp4v2_summary.md
1
100%
# AROMA Exp 4 v2 — Supervised YOLOv8 Defect Detection 평가

비교: Baseline (real-only) vs Random ROI (real+synth) vs AROMA ROI (real+synth)
모든 조건이 real labeled defect (train split) 위에서 학습.
baseline: COCO→real-only 학습 후 best.pt 저장. random/aroma: baseline best.pt 에서 finetune.
Val = real defect (GT mask → bbox), train 과 disjoint.

## mvtec_cable

### yolov8n

| 조건         |        map50 |     map50_95 |    precision |       recall | n_real_train | n_synth_train |
|------------|------------ | ------------ | ------------ | ------------ | ------------ | ------------|
| baseline   |       0.5846 |       0.3664 |       0.6707 |       0.5610 |           64 |            0 |
| random     |       0.6210 |       0.3801 |       0.7061 |       0.5366 |           64 |          128 |
| aroma      |       0.5920 |       0.3295 |       0.5772 |       0.6661 |           64 |          128 |

**Delta (AROMA − Baseline)**: map50 +0.74pp, map50_95 -3.69pp, precision -9.35pp, recall +10.51pp
**Delta (AROMA − Random)**:   map50 -2.90pp, map50_95 -5.06pp, precision -12.89pp, recall +12.95pp