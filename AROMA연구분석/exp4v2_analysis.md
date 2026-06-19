## Exp4v2 Results Analysis

YOLOv8 Supervised Defect Detection — Baseline vs Random vs AROMA

Source data: `.claude/.etc/exp4v2/exp4v2_results.json`
Reference (one-class AD): `.claude/.etc/exp4/exp4_results.json`

---

### Summary

**This run is invalid.** Every metric is `0.0000` across all three conditions
(baseline, random, aroma) for the single executed cell `mvtec_cable / yolov8n`.
This is not a scientific result (e.g., "AROMA ties random") — it is a **pipeline
failure**. No conclusion about the AROMA hypothesis can be drawn from this data.

Two independent faults are visible in the raw numbers:

1. **AROMA arm never trained** — `n_train = 0`. The synthetic defect set for the
   AROMA condition was empty, so its `0.0` is fully explained by having no data.
2. **Baseline and Random also scored `0.0` despite having data** (224 and 600
   training images respectively). This is the more alarming signal: two arms that
   *did* have training data still produced zero detections, which points to a
   systemic evaluation/label-pipeline bug rather than a modeling outcome.

Recommended action: **discard these numbers**, fix the validation-label pipeline
first (explains the all-arms zero), then fix AROMA synthetic generation (explains
`n_train = 0`), and re-run after a single-arm smoke test passes.

---

### Results Table (all cells)

Only one cell exists: 1 dataset (`mvtec_cable`) × 1 model (`yolov8n`). The result
schema is nested to scale to more datasets/models, but only this cell was run.

#### mvtec_cable × yolov8n

| Condition | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall | n_train |
|-----------|---------|--------------|-----------|--------|---------|
| baseline  | 0.0000  | 0.0000       | 0.0000    | 0.0000 | 224     |
| random    | 0.0000  | 0.0000       | 0.0000    | 0.0000 | 600     |
| aroma     | 0.0000  | 0.0000       | 0.0000    | 0.0000 | **0**   |

**Deltas** (degenerate — both arms produced nothing):

| Cell | AROMA − baseline | AROMA − random |
|------|------------------|----------------|
| mvtec_cable / yolov8n | 0.0000 (all metrics) | 0.0000 (all metrics) |

---

### Key Findings

1. **Uniform zeros across all arms and all metrics.** mAP@0.5, mAP@0.5:0.95,
   precision, and recall are *exactly* `0.0` for every condition. Exact zeros
   across precision *and* recall simultaneously are characteristic of a code/label
   fault, not a soft domain-gap degradation (which would produce small but nonzero,
   noisy values).

2. **AROMA `n_train = 0`** — a distinct, upstream failure. The AROMA ROI-extraction
   / synthetic-defect generation step produced no training images for `mvtec_cable`.
   This is independent of, and additional to, the all-arms zero problem.

3. **Data-bearing arms still failed.** Baseline (224) and Random (600) had training
   data and still scored zero. This is the single most diagnostic fact in the run:
   the failure is downstream of having data — most likely in the validation label
   set (GT mask → bbox conversion) or in train/val class-ID alignment.

4. **Single-cell scope.** With one dataset and one model, there is no cross-cell
   structure to analyze; win-rate and "which dataset benefits most" questions are
   not answerable.

---

### Hypothesis Assessment (AROMA > random?)

**Indeterminate — the experiment did not execute.** The hypothesis can be neither
supported nor rejected from this data.

| Question | Answer |
|----------|--------|
| 2. AROMA > random? | **Indeterminate.** Both `0.0`; the `+0.0000` delta exists only because both arms produced nothing. |
| 3. Does synthetic data help vs baseline? | **Indeterminate.** Baseline is also `0.0` — there is no working reference point. |
| 4. Which datasets/models benefit most? | **N/A** — single cell, all zero. |
| 5. Surprising failures? | **Yes — the entire run.** See Root-Cause below. |

**Aggregate win rates** (meaningless given all-zero inputs, reported for
completeness):

- AROMA > random:  0 / 1 cells (0%) — a tie at zero, not a win.
- AROMA > baseline: 0 / 1 cells (0%) — a tie at zero, not a win.

---

### Comparison to Exp4 (one-class AD)

Exp4 (one-class anomaly detection, SimpleNet on `mvtec_cable`) produced
**discriminating, nonzero** numbers:

| Condition | image_auroc | pixel_auroc |
|-----------|-------------|-------------|
| baseline  | 0.7646      | 0.6285      |
| random    | 0.5015      | 0.6304      |
| aroma     | 0.4996      | 0.5676      |

So supervised detection does **not** tell a cleaner story than AD here — it tells
**no** story. The contrast is itself diagnostic: the AD pipeline runs end-to-end and
yields separable values, whereas the supervised-detection pipeline does not execute.

Note on the prior: in exp4 AD, synthetic data *hurts* relative to baseline
(random / aroma both ≈ 0.50 image-AUROC vs baseline 0.7646), and **AROMA ≈ random**
(0.4996 vs 0.5015). If exp4v2 were ever fixed, that prior would predict a hard road
for the "AROMA helps" hypothesis — synthetic ROIs were not additive in the AD
setting, and there is no evidence yet that supervised detection reverses that.

---

### Weaknesses / Limitations

- **No valid measurement.** All reported metrics are degenerate zeros; nothing here
  measures model quality.
- **Single cell only.** 1 dataset × 1 model — no generalization, no cross-architecture
  or cross-dataset comparison possible even if the numbers were valid.
- **Two confounded failures.** The all-arms eval/label fault and the AROMA
  `n_train = 0` generation fault are separate problems; one must not be mistaken for
  the other when debugging.
- **No smoke test gating.** The run scaled to three arms without a single-arm
  baseline-only check that would have surfaced the zero-mAP problem immediately.
- **Possible silent failures.** Zero values may also mask a missing `best.pt`, empty
  prediction set, or path error — none of which are distinguishable from the summary
  table alone.

#### Root-Cause Assessment (ranked)

1. **Validation label conversion bug (most likely, explains all arms).** GT mask →
   bbox conversion may have produced empty or malformed val labels. Zero true boxes
   → mAP undefined → reported as `0.0`. This single fault drives *every* arm to zero
   simultaneously, matching the observed pattern.
2. **Class-ID mismatch** between train and val labels (predictions never match a GT
   class) — also drives all arms to zero.
3. **AROMA synthetic-generation failure** (`n_train = 0`) — independent, AROMA-only.
4. **Synthetic → real domain collapse** — possible but should *not* drive precision
   and recall to *exactly* `0.0` across all arms; exact zeros indicate a code/label
   fault, not a soft domain gap.
5. **Silent training/inference failure** (no `best.pt`, empty predictions, path error).

---

### Recommended next steps

Diagnostic order — fix the all-arms cause first, then the AROMA-specific cause:

1. **Verify val label conversion (GT mask → bbox).** Inspect a few generated YOLO
   `.txt` label files; confirm nonzero box counts and correct class IDs. A broken val
   set drives all arms to `0.0` simultaneously and is the most likely culprit.
2. **Confirm the baseline run produced a `best.pt`** and that predictions are nonempty
   on val — run `model.val()` verbosely on a single image.
3. **Fix AROMA synthetic generation** so `n_train > 0` for the aroma arm
   (ROI-extraction / synthetic-defect pipeline for `mvtec_cable`).
4. **Check train/val class-ID alignment** (single-class "defect" vs per-defect-type
   IDs).
5. **Re-run with a single-arm smoke test first** — get baseline-only to a nonzero
   mAP before scaling back to all three arms, then add datasets/models.

**Bottom line:** discard these numbers, fix the val-label pipeline first (explains
all-arms zero), then close the AROMA data-generation gap (explains aroma
`n_train = 0`), and re-run.
