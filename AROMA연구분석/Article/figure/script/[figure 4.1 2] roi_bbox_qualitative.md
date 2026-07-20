# Figure 7 — ROI Placement Qualitative Comparison (Baseline vs AROMA vs Random)

## Purpose
Visually confirm the ROI placement policy's behavior on a single representative image per dataset: where a real defect actually occurs (Baseline), where AROMA's compatibility-scan placement (`clean_bg_selection` + `roi_selection` scan-rank-place logic) would place candidate defects, and where uniform-random placement would place the same candidates. Qualitative eyeball check only — not a quantitative metric (that is Figure 6).

## Content
**5 rows (datasets) × 3 columns**
- **Columns**: Baseline | AROMA | Random (labels on top row only)
- **Rows**: AITeX, Kolektor, Severstal, MTD, MVTec Leather (dataset name on left, no other labeling)
- All 3 columns in a row show the **same underlying image** (one representative real defect-image per dataset), with different bbox overlays:
  - **Baseline**: 1 box — the real ground-truth defect bbox (`defect_bbox` field of the representative `roi_selected.json` entry).
  - **AROMA**: 5 boxes — candidate placements computed by calling the actual production scan-rank-place function (`_positive_place` in `scripts/aroma/generate_defects.py`) on the representative image, using 5 real defect bbox *sizes* drawn from the same dataset's `roi_selected.json` pool (diverse subtypes/aspect ratios) and the real per-cluster compatibility row from `compatibility_matrix.json` (`matrix_symmetric`). Void tiles are excluded via `_is_clean_background`, identical to the production gate.
  - **Random**: same 5 bbox sizes, positioned via the actual production `_random_paste_position` (uniform-random valid top-left), no compatibility/void consideration.
- Box sizes are real (never invented); only the AROMA/Random *positions* on this specific representative image are computed by re-running the real placement functions, since the stored ROI JSONs do not persist multiple candidate positions per image (`position` field is `null` for every entry in every dataset — verified 2026-07-16).
- **Candidate-size target (2026-07-16 revision)**: for AITeX the 5 candidate sizes still target ~40% of the image area (large, clearly visible candidates on a small tile). For Kolektor, Severstal, MTD, and MVTec Leather the target was changed to the **representative image's own baseline defect bbox area** instead of 40% — at 40% the AROMA/Random candidates on those four datasets were all similarly oversized and their placement difference was hard to see; matching the real baseline defect's scale makes the AROMA-vs-Random positional spread visible.
- **Display aspect (2026-07-16 revision)**: every row's display cell uses the same 1:1 width:height ratio, matching MVTec Leather's own 1024×1024 aspect, regardless of the source image's true aspect ratio (e.g. Severstal 1600×256, Kolektor 500×1255). This is a display-only convenience (`imshow` + `aspect="auto"`) for a uniform grid — the underlying pixel content and bbox coordinates are untouched.

## Data Source
- Representative image + baseline bbox + candidate bbox sizes: `D:\project\aroma_dataset\roi\<dataset>\roi_selected.json` (`image_path`, `defect_bbox`, `cluster_id`, `defect_subtype`)
- Compatibility rows: `D:\project\aroma_dataset\profiling\profiling\<dataset>\compatibility_matrix.json` (`matrix_symmetric[str(cluster_id)]`, `bin_edges`)
- Placement functions (imported, not reimplemented): `scripts/aroma/generate_defects.py` — `_positive_place`, `_random_paste_position`, `_is_clean_background`, `_load_context_cell_helpers`; `scripts/aroma/clean_bg_selection.py` — `_effective_wh` (real generation-time fit-rescale, applied before placement so an oversized real bbox is rescaled exactly like production rather than silently clipped)
- Image files: local mirror under `D:\project\aroma_dataset\{aitex_tiled,kolektor,severstal,mtd,mvtec_leather}\test\...` (Colab path → local path mapping per dataset root)

## Image Specifications
- **Resolution**: ≥300 dpi, ≥1000 px per side (per FIGURE_TABLE_WORKFLOW.md §6.4)
- **Layout**: GridSpec(5, 3), row = dataset, column = arm, uniform 1:1 cell aspect (see above)
- **Box colors**: Baseline = red, AROMA = green, Random = blue — outline only, thickness 2px, same color across the 5 boxes within a column (kept simple per instruction)
- **Labels**: column titles ("Baseline"/"AROMA"/"Random") on top row only; dataset name as row label (left margin) only. No other annotation (score, size, etc.) — per instruction to keep it simple.
- **RNG seed**: fixed per dataset (`42 + dataset_index`) for reproducibility of both the AROMA top-K sample and the Random draw.

## Caption (Draft)
**Figure 7.** Qualitative ROI placement comparison across the five datasets. Each row shows one representative image with the real defect region (Baseline), five AROMA-scored candidate placements, and five uniform-random placements of identical box sizes.

**Word count**: 34 words (condense for final ≤25-word form when inserted).

**Caption (Final, ≤25 words)**:
Qualitative ROI placement comparison. Baseline shows the real defect region; AROMA and Random overlay five identical-size candidate boxes scored by compatibility versus placed uniformly at random.

**Word count**: 24 words ✓

## Section Reference
- **Section**: 4.1 (ROI Quality Evaluation)
- **Placement**: After the Figure 6 discussion, as a qualitative complement to the quantitative coverage/entropy/Gini metrics.
- **Callout in Text**: "Figure 7 illustrates this behavior qualitatively on one representative image per dataset: ..."
