# Figure 8 — AITeX ROI Selection Comparison

## Purpose
Visualize AROMA's contextual ROI selection strategy versus uniform random placement in defect-to-background augmentation. Show that AROMA selects distinct background contexts compared to random sampling, enabling qualitative validation of the method's design intent.

## Content
**Three columns × Two rows**
- **Columns**: Baseline (original defect) | AROMA (AROMA-selected background) | Random (random background)
- **Rows**: Two representative defect instances
  - Row 1: Low roi_score (weakest compatibility confidence, closest to baseline quality)
  - Row 2: High roi_score (strongest compatibility confidence, most selective AROMA choice)

## Data Source
- Dataset: AITeX (single-class tiled-pattern textile defects)
- Original defects: `D:\project\aroma_dataset\aitex\test\{YYY}\*.png`
- AROMA synthetics: `D:\project\aroma_dataset\synth_aroma\aitex\images\*.jpg`
- Random synthetics: `D:\project\aroma_dataset\synth_random\aitex\images\*.jpg`
- ROI selections: `D:\project\aroma_dataset\roi\aitex\roi_selected.json` (AROMA), `clean_bg_random_arm.json` (Random)

## Image Specifications
- **Resolution**: 1500 px × 2800 px (dpi=150)
- **Layout**: GridSpec(2, 3) with hspace=0.3, wspace=0.05
- **Subplot titles**: "Baseline", "AROMA", "Random" (each column, top only)
- **Font**: matplotlib default, fontsize=12 for titles
- **Color**: RGB native (no special colormap required)

## Script
Generate via `viz_roi_comparison.py` with dataset="aitex", output dpi=150.

## Caption (Draft)
**Figure 8.** AITeX ROI selection comparison: original defects versus synthetic augmentations via AROMA and random background placement. Rows represent low (top) and high (bottom) AROMA compatibility confidence. Columns show baseline (original), AROMA-selected background context, and random context. Visual texture differences illustrate AROMA's distinct context selection strategy.

**Word count**: 44 words (exceeds 25; condense for final version below)

**Caption (Final, ≤25 words)**:
AITeX ROI selection comparison. Original defects (left) and synthetic augmentations via AROMA-selected (middle) versus random (right) background contexts. Rows represent low and high AROMA compatibility confidence.

**Word count**: 23 words ✓

## Section Reference
- **Section**: 4.3 (Downstream Detection Performance)
- **Placement**: After Table 8 (before discussion of AROMA vs Random performance on AITeX)
- **Callout in Text**: "As shown in Figure 8, AROMA and Random select visually distinct background contexts..."
