# AROMA Exp 4 v2 — Supervised YOLOv8 Defect Detection 평가

비교: Baseline (background-only) vs Random ROI vs AROMA ROI
Train = synthetic defect (labeled), Val = real defect (GT mask → bbox)

## mvtec_cable

### yolov8n

| 조건         |      map50 |   map50_95 |  precision |     recall |    n_train |
|------------|---------- | ---------- | ---------- | ---------- | ----------|
| baseline   |     0.0000 |     0.0000 |     0.0000 |     0.0000 |        224 |
| random     |     0.0000 |     0.0000 |     0.0000 |     0.0000 |        600 |
| aroma      |     0.0000 |     0.0000 |     0.0000 |     0.0000 |          0 |

**Delta (AROMA − Random)**: map50 +0.0000, map50_95 +0.0000, precision +0.0000, recall +0.0000
