# AROMA Exp 4 v2 — Supervised YOLOv8 Defect Detection 평가 (multi-seed)

seeds = [42, 1, 2]  (n_seeds=3)
각 셀 = mean ± std (sample std, ddof=1; n_seeds<2면 std=0). 95% CI는 JSON ci95 참조.
비교: Baseline (real-only) vs Random ROI (real+synth) vs AROMA ROI (real+synth)
Val = real defect (GT mask → bbox), train 과 disjoint. seed별 독립 split.

## severstal

### yolov8n

| 조건         |            map50 |         map50_95 |        precision |           recall |     n_real_train |    n_synth_train |          n_seeds |
|------------|---------------- | ---------------- | ---------------- | ---------------- | ---------------- | ---------------- | ----------------|
| baseline   |              N/A |              N/A |              N/A |              N/A |              N/A |              N/A |              N/A |
| random     |              N/A |              N/A |              N/A |              N/A |              N/A |              N/A |              N/A |
| casda      |              N/A |              N/A |              N/A |              N/A |              N/A |              N/A |              N/A |
| aroma      |    0.4991±0.0311 |    0.2402±0.0149 |    0.5513±0.0777 |    0.5082±0.0086 |             2534 |             2534 |                3 |

**Delta (AROMA − Baseline)**: map50 N/A, map50_95 N/A, precision N/A, recall N/A
**Delta (AROMA − Random)**:   map50 N/A, map50_95 N/A, precision N/A, recall N/A

#### per-class map50 (multi)

| class    |   baseline |     random |      casda |      aroma |     Δ(A-R) |
|----------|---------- | ---------- | ---------- | ---------- | ----------|
| c1       |        N/A |        N/A |        N/A |     0.4406 |        N/A |
| c2       |        N/A |        N/A |        N/A |     0.3157 |        N/A |
| c3       |        N/A |        N/A |        N/A |     0.5930 |        N/A |
| c4       |        N/A |        N/A |        N/A |     0.6472 |        N/A |
