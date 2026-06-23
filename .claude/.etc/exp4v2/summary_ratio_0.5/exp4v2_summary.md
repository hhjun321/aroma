# AROMA Exp 4 v2 — Supervised YOLOv8 Defect Detection 평가 (multi-seed)

seeds = [42, 1, 2]  (n_seeds=3)
각 셀 = mean ± std (sample std, ddof=1; n_seeds<2면 std=0). 95% CI는 JSON ci95 참조.
비교: Baseline (real-only) vs Random ROI (real+synth) vs AROMA ROI (real+synth)
Val = real defect (GT mask → bbox), train 과 disjoint. seed별 독립 split.

## mvtec_cable

### yolov8n

| 조건         |            map50 |         map50_95 |        precision |           recall |     n_real_train |    n_synth_train |          n_seeds |
|------------|---------------- | ---------------- | ---------------- | ---------------- | ---------------- | ---------------- | ----------------|
| baseline   |    0.7224±0.0511 |    0.4752±0.0236 |    0.7556±0.0486 |    0.6926±0.0295 |               64 |                0 |                3 |
| random     |    0.7344±0.0815 |    0.4978±0.0493 |    0.8227±0.0201 |    0.6747±0.0791 |               64 |               32 |                3 |
| aroma      |    0.7400±0.0742 |    0.4937±0.0485 |    0.8320±0.0615 |    0.6677±0.0520 |               64 |               32 |                3 |

**Delta (AROMA − Baseline)**: map50 +1.76±9.01pp, map50_95 +1.85±5.39pp, precision +7.64±7.84pp, recall -2.49±5.98pp
**Delta (AROMA − Random)**:   map50 +0.56±11.02pp, map50_95 -0.41±6.92pp, precision +0.93±6.47pp, recall -0.70±9.47pp

## mvtec_pill

### yolov8n

| 조건         |            map50 |         map50_95 |        precision |           recall |     n_real_train |    n_synth_train |          n_seeds |
|------------|---------------- | ---------------- | ---------------- | ---------------- | ---------------- | ---------------- | ----------------|
| baseline   |    0.8932±0.0517 |    0.5738±0.0433 |    0.8679±0.0606 |    0.8565±0.0639 |               99 |                0 |                3 |
| random     |    0.9493±0.0413 |    0.6435±0.0456 |    0.9242±0.0374 |    0.9109±0.0501 |               99 |               49 |                3 |
| aroma      |    0.9280±0.0347 |    0.6311±0.0509 |    0.9231±0.0221 |    0.8870±0.0125 |               99 |               48 |                3 |

**Delta (AROMA − Baseline)**: map50 +3.48±6.23pp, map50_95 +5.73±6.68pp, precision +5.52±6.45pp, recall +3.05±6.51pp
**Delta (AROMA − Random)**:   map50 -2.13±5.39pp, map50_95 -1.24±6.83pp, precision -0.11±4.34pp, recall -2.39±5.16pp

## mvtec_wood

### yolov8n

| 조건         |            map50 |         map50_95 |        precision |           recall |     n_real_train |    n_synth_train |          n_seeds |
|------------|---------------- | ---------------- | ---------------- | ---------------- | ---------------- | ---------------- | ----------------|
| baseline   |    0.9085±0.0238 |    0.5917±0.0491 |    0.9390±0.0346 |    0.8094±0.0397 |               42 |                0 |                3 |
| random     |    0.9583±0.0171 |    0.6787±0.0114 |    0.9523±0.0179 |    0.8835±0.0176 |               42 |               21 |                3 |
| aroma      |    0.9507±0.0119 |    0.6657±0.0153 |    0.9392±0.0475 |    0.8901±0.0406 |               42 |               21 |                3 |

**Delta (AROMA − Baseline)**: map50 +4.22±2.66pp, map50_95 +7.40±5.14pp, precision +0.02±5.88pp, recall +8.07±5.68pp
**Delta (AROMA − Random)**:   map50 -0.76±2.08pp, map50_95 -1.30±1.91pp, precision -1.31±5.08pp, recall +0.66±4.43pp
