# Figure — Background-selection compatibility (AROMA vs Random) across datasets

**Script**: `[figure 4.1 3] bg_similarity_datasets.py`
**Output**: `../image/[figure 4.1 3] bg_similarity_datasets.png`
**Data root**: `D:/project/aroma_dataset` (override with env `AROMA_DATASET_ROOT`)

## What the figure shows

One grouped violin/box pair **per dataset** (five datasets, ordered by descending
Context Complexity Index, CCI): the **background compatibility** of the ROIs'
assigned normal-background images, for **AROMA** (blue, compatibility-selected)
vs **Random** (orange, uniform). Per dataset it annotates
Δ = mean(AROMA) − mean(Random), a one-sided Mann-Whitney U p-value
(H1: AROMA > Random), and n (number of ROIs).

## Why this measurement is valid under copy-paste

Under the training-free copy-paste engine both arms paste the **same real defect
pixels** and share the **same defect ROIs**; they differ only in **which normal
background** each ROI is assigned to. Distribution-fidelity metrics (FID/KID/LPIPS,
PRDC) are therefore near-identical between arms *by construction* and cannot
discriminate the arms. Background selection, however, **does** differ. This figure
isolates exactly that factor — it is an **engine-independent** test of AROMA's
placement/selection mechanism, meaningful precisely where fidelity metrics are not.

## Metric (independent, dataset-pooled reference)

- **Reference** (per dataset): mean **texture histogram** of the background region
  (mask-excluded pixels) pooled over the real defect images — class-agnostic.
- **Per ROI**: the assigned normal image's texture histogram ∩ the dataset reference
  (histogram intersection, 0–1).
- **Texture histogram** = concatenation of `[intensity | gradient-magnitude |
  local-variance]` histograms (32 bins each, renormalised to sum 1). Gradient and
  local variance capture the texture cues AROMA's symmetric compatibility gate uses;
  plain intensity alone under-represents the effect on near-uniform steel.
- The metric is **independent of the pipeline's own class_fit score** (which AROMA
  maximises by construction), so a positive Δ is evidence that the compatibility
  gate recovers the *real* defect–background association, not a tautology.

Inputs: `roi/<ds>/clean_bg_selected.json` (AROMA), `roi/<ds>/clean_bg_random_arm.json`
(Random), real defect images + masks, and `train/good` normals.

## Reading

- **Δ > 0 with ***** ⟹ AROMA assigns backgrounds whose texture matches the real
  defect background significantly better than Random. The n is in the hundreds to
  thousands of ROIs, so significance here is **robust** — in contrast to the n=3
  seed downstream comparison (§4.4), which is directional but underpowered.
- Measured result: the background-selection mechanism is **significant on four of
  five datasets** (AITeX, Severstal, Kolektor, MVTec Leather; MTD n.s.).

### Crucial interpretation — mechanism ≠ downstream benefit

This figure measures the **selection mechanism**, not the detection payoff, and the
two must not be conflated. AROMA selects measurably more compatible backgrounds even
on **MVTec Leather** and **Kolektor** — datasets whose *downstream* Δ(AROMA−Random)
is null/negative (§4.4). There is no contradiction:

- The mechanism (does AROMA pick a texture-closer background?) operates almost
  everywhere — even a "near-uniform" leather surface carries enough texture variation
  for the compatibility gate to select a closer match than a uniform draw.
- The downstream benefit (does that closer background improve YOLO mAP?) is
  **separately gated** by (i) baseline headroom — near-ceiling datasets (MTD 0.91,
  Kolektor 0.97, Leather 0.83) leave no room to improve — and (ii) whether the
  background variation actually confounds detection; on a background-agnostic surface
  a well-selected background does not change what the detector learns.

So this figure should be read as evidence that **AROMA's placement policy does what it
is designed to do (engine-independently)**, complementing — not duplicating — the
CCI/headroom-conditional *downstream* story in §4.4. Selecting a compatible background
is necessary for the policy to matter, but not sufficient for a detection gain.

## Companion (qualitative, not in this script)

A qualitative montage (`bg_pool_compare`, per-dataset representative examples) shows
the same effect visually: AROMA's top-K background pool is a consistent match to the
baseline surface, whereas Random's pool mixes incompatible surfaces (checkerplate,
voids, bright cracks). That montage is illustrative; **this** figure is the
statistical, all-sample aggregate.

## Reproduce

```bash
python "[figure 4.1 3] bg_similarity_datasets.py"     # writes ../image/[figure 4.1 3] bg_similarity_datasets.png
```
Optional `scipy` gives exact Mann-Whitney U; without it the script falls back to a
1000-permutation p-value. Deterministic (fixed reference sampling seed 42).
