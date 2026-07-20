# §4 Qualitative figures — author TODO (moved from section4_3.txt per FIGURE_TABLE_WORKFLOW §9.1)

% =====================================================================
% TODO (AUTHOR): QUALITATIVE FIGURES — NOT YET GENERATED
% ---------------------------------------------------------------------
% A qualitative-results subsection (proposed §4.5) was requested but could
% NOT be produced from the local checkout. No qualitative figure was
% generated and NONE was fabricated, because there are no usable image
% pixels on this machine.
%
% WHY BLOCKED (verified by full disk scan, not assumed):
%   - Total raster images under the project on disk: 11.
%       * 9 are ALREADY-RENDERED paper plots in
%         AROMA연구분석/Article/figure/ (AUROC curves, complexity
%         landscape, ROI coverage, pipeline diagram, etc.) — finished
%         vector-style plots, not raw image crops.
%       * 2 are degenerate test fixtures:
%             tests/fixtures/images/img_001.png  (64x64)
%             tests/fixtures/seed_defect.png     (16x16)
%         These are placeholder fixtures, NOT real defect / normal /
%         synthetic crops.
%   - Real defect, normal/background, random-synth, and AROMA-synth
%     images live exclusively on Google Drive / Colab
%     (e.g. /content/drive/MyDrive/data/Aroma/...) and are unreachable
%     from this local Windows checkout.
%   - docs/data_dir/*.txt are directory-tree listings (paths only,
%     no pixels); .claude/.etc/ holds only JSON/markdown metric reports.
%
% WHAT THE FOUR FIGURES SHOULD SHOW (once pixels are available):
%   Figure A — Side-by-side grid: real defect | random-synth | AROMA-synth
%              (+ matching normal background) per dataset. Caption should
%              highlight AROMA's context-aware defect placement on
%              appropriate background texture vs. random placement.
%   Figure B — Pipeline-stage visualization: seed defect -> elastic-warp
%              variant (TF-IDG) -> Poisson-blended composite -> quality-gate
%              decision, for one representative sample.
%   Figure C — Synthesis diversity: multiple TF-IDG variants from a single
%              seed across the five subtypes (linear_scratch, elongated,
%              compact_blob, irregular, general).
%   Figure D — Failure cases: blending artifacts, quality-gate rejects,
%              and/or degenerate AROMA output (see CAUTION below).
%
% RE-RUNS / EXPORTS NEEDED (do this in Colab, where pixels exist):
%   1. Add an image-export step after Stage 1b/synthesis that dumps a
%      sample of {real, random, AROMA} defect crops plus the matching
%      {normal} background per dataset into
%      .../qualitative_samples/<dataset>/.
%   2. Target a complexity spectrum across the canonical roster to make
%      Figure A informative: MVTec Leather (MCI 0.645, high morphology),
%      Severstal (MCI 0.455), AITeX (MCI 0.398, high context CCI 0.434),
%      MTD (MCI 0.561). Values from Table 1.
%   3. Per project rule (no new .ipynb), write the matplotlib grid builder
%      as a .md Colab execution guide under colab_execute/.
%
% CAUTION (data integrity): before building the side-by-side grid,
%   verify that non-empty AROMA and Random synthetic crops actually exist
%   on Drive for each canonical dataset (Severstal, MVTec Leather, AITeX,
%   MTD). Note Severstal copy-paste was doubly null (synthesis inert at the
%   reported budget; Table 12), so a Severstal panel may be uninformative —
%   prefer AITeX (the only positive) for the context-matched-placement
%   contrast, and use any empty/degenerate output as a Figure D failure
%   case rather than presenting an empty panel.
%
% Once qualitative_samples/ are exported and the grid figures are rendered:
%   - Insert a new subsection "##4.5. Qualitative Results" here (before §5).
%   - Add \begin{figure}...\includegraphics{figure/fig_qualitative_*}...
%     blocks with \label{fig:qual_*} and honest captions (including
%     limitations / failure modes).
%   - Add callout sentences in §4.5 and a forward reference from §4.2
%     (FID discussion) noting that the placement advantage FID cannot
%     resolve is made visible qualitatively in §4.5.
% =====================================================================