### Sequential Placement Decisions

Synthesis placement decomposes into three sequential decisions: (i) **which defect to use as the source (ROI selection)**, (ii) **onto which normal background image it should be placed (background assignment)**, and (iii) **at which position on that background (site resolution)**. All three stages consult a single compatibility model, defined first.

### Compatibility Model: `matrix_symmetric`

`matrix_symmetric` models the compatibility between a morphology cluster (k) and a 64-px discrete context cell (c). It is built from two observed distributions over the same context-cell space — P_def(k, c), the context cells observed in images containing cluster-(k) defects, and P_clean(c), the context cells observed across all normal images — combined by their geometric mean:

ctx_prior(k, c) ∝ √( P_def(k, c) · P_clean(c) )

so that a cell scores high only when defects of that cluster are actually observed near it *and* the cell exists in normal images. Each row is then normalised per cluster, so that a row expresses the **relative compatibility** of every context cell with cluster (k), peaking at 1 for the most compatible cell.

### 1. ROI Selection

This stage determines **which defects to use as sources**. Each candidate is scored by combining its context-based compatibility with the cluster prior,

ROI_score = 0.5 · ctx_prior + 0.3 · morph_prior + 0.2 · quality

where ctx_prior is read from `matrix_symmetric`, morph_prior is the cluster prior P(k) of §3.2.2, and quality is a graded subtype-suitability score inherited unchanged from CASDA's ROI suitability evaluation (crop subtype of §3.2.3 → discrete plausibility score for the inspected surface). The top (K) candidates by `ROI_score` become the sources for the placement stages that follow.

### 2. Background Assignment

This stage determines the **normal background image** for each selected ROI. Candidates are ranked by three fitness measures, all built on the histogram intersection ∩(a, b) = Σ_c min( a(c), b(c) ):

src_fit(g) = ∩( h_g, p_src(v) )
class_fit(g) = ∩( h_g, p_cls )
size_fit(g) = min( 1, 0.95 · W_g / w, 0.95 · H_g / h )

where h_g is the context-cell histogram of candidate background (g). `src_fit` asks how strongly the candidate exhibits the context of the image the defect was extracted from; `class_fit` asks the same for the defect class as a whole, remaining applicable to multi-label datasets; `size_fit` asks whether the ROI fits into the candidate at an appropriate size. The three are summed,

bg_score = src_fit + class_fit + size_fit

and the normal image with the highest score is assigned as the background.

### 3. Site Resolution

This stage determines **where on the assigned background the defect is placed**. A candidate position (s) is the top-left coordinate of the defect crop; the **ring region** around the crop represents the local context that would surround the defect after insertion. The ring of each candidate is converted into a context-cell histogram (h_s) and compared against the `matrix_symmetric` row of the defect's cluster (k) as the target profile:

site_score = ∩( h_s, tgt[k] )

The position with the highest `site_score` is selected. Placement therefore relies neither on geometric heuristics nor on random selection, but on **the position whose surroundings most closely resemble the context in which that defect morphology was observed in the real data**.

In summary, ROI selection decides *which defect*, background assignment decides *which normal image*, and site resolution decides *which position* — with `matrix_symmetric` providing the shared representation of defect–context compatibility across all three, so that AROMA jointly considers the defect's shape, the context it was originally observed in, and the local context of its new location.
