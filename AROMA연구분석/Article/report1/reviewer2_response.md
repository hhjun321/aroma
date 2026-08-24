# Response to Reviewer 2

We thank the reviewer for the careful reading and constructive comments. The central subject of AROMA is the transition from our earlier CASDA pipeline, in which structural settings were specified manually, to a pipeline in which these settings are derived from each dataset's own statistics. To keep the revision aligned with this data-driven thesis, we have strengthened the method in three areas.

(i) learning-based parameter determination for the categorical structure
morphology clusters, context cells, and subtype thresholds are all estimated from the profiled data (revised section 3.2.2–3.2.3).

(ii) removal of manually specified weighting from the ROI-scoring equation
the revised Eq. (3) combines its two normalized terms as an unweighted sum, leaving no coefficient to tune (revised section 3.2.4).

(iii) a ring-based site-resolution step for ROI selection 
each defect is placed at the position whose surrounding context distribution best matches the context observed around that defect morphology in the real data, replacing geometric heuristics and random choice alike (revised section 3.2.4).

---

## Comment 1

> *The paper repeatedly claims to be "free of manual tuning" and "without hand-set constants," yet the hyperparameters in Equations (2) and (4) are manually set.*

Response.

We appreciate the opportunity to clarify the scope of this claim, and we have sharpened the wording in the revised manuscript so that it cannot be misread.

The revised text (section 3.2.2–3.2.4) now makes the claim exact: every quantity entering the placement decision is either derived from the dataset's own statistics or an unweighted combination of such quantities; the manually set coefficients the reviewer identified (Eq. (2) weights, Eq. (4) weights) have both been removed from the method.

Manuscript changes: Eq. (2) reduced to an unweighted, constant-free combination with a role-based rationale (§3.2.4); §3.2.6 (Quality Gate, Eq. (4)) removed; new boundary-sensitivity subsection (§4.5); data-derived partition descriptions in §3.2.2–3.2.3.

---

## Comment 2

> *Only YOLOv8n is used as the downstream detector; it is recommended to supplement the experiments with more recent mainstream detectors.*

Response.

Following this recommendation, we supplemented the evaluation with YOLOv11n, a recent mainstream detector, under the identical three-arm (Baseline / Random / AROMA), three-seed protocol on Severstal and AITeX — the heterogeneous-surface datasets where the placement effect is the operative question (new Tables 13–14, section 4.3).

Manuscript changes: new detector-generality paragraph and tables in §4.3 (YOLOv11n, Tables 13–14: Severstal and AITeX, three-arm × three-seed).

---

## Comment 3

> *Fixed thresholds are used for defect subtype classification without providing a threshold sensitivity analysis.*

Response.

We address this in three parts.

- The subtype thresholds are no longer fixed constants.
In the revised manuscript (section 3.2.3), subtype boundaries are derived per dataset from the observed morphology distributions as percentile boundaries, reported in Table 5. The derived values vary by up to a factor of 4.5 across datasets (aspect-ratio boundary 3.62 on MTD vs. 16.43 on AITeX), demonstrating precisely the cross-dataset variation that a fixed threshold would absorb silently.

- Equation (4) has been removed, and the quality criterion is disclosed as inherited.
In revising the quality-gating description, we deleted the subsection presenting Eq. (4), including its fixed component weights and absolute acceptance threshold, rather than attempting to justify these manually specified settings. The underlying gate is a coarse admissibility pre-filter inherited unchanged from our earlier CASDA pipeline. It discards unusable patches before any placement decision and is applied identically to both the AROMA and random arms. Thus, it is common to all comparisons in the paper, operates upstream of and independently from the placement scoring in Eq. (2), and is not part of the claimed data-driven contribution.

The revised text (section 3.2.2–3.2.4) now makes the claim exact: every quantity entering the placement decision is either derived from the dataset's own statistics or an unweighted combination of such quantities; the manually set coefficients the reviewer identified (Eq. (2) weights, Eq. (4) weights) have both been removed from the method.

Manuscript changes: Eq. (2) reduced to an unweighted, constant-free combination with a role-based rationale (§3.2.4); §3.2.6 (Quality Gate, Eq. (4)) removed; new boundary-sensitivity subsection (§4.5); data-derived partition descriptions in §3.2.2–3.2.3.

---

## Comment 4

> *The CCI formula lists only three components, whereas the text explicitly states "four normalized components."*

Response.

We thank the reviewer for catching this inconsistency. The formula in the submitted version omitted one component; the text's statement of "four normalized components" was correct. The revised §3.2.1 now states the complete formula,

CCI = Mean(TextureEntropy, ContextClusterCount, FreqComplexity, OrientVariance),

and we have additionally added Table 2, which decomposes the CCI of each of the five datasets into these four measured components (texture entropy, context cluster count, frequency-domain complexity, and gradient-orientation variance), so the formula, the text, and the reported values can be verified against one another directly.

Manuscript changes: §3.2.1 (complete four-component formula); new Table 2 (per-dataset CCI decomposition).

---

## Comment 5

> *Quantitative results for Shannon entropy and the Gini coefficient are not reported.*

Response.

We now report these values quantitatively. The revised section 4.1 adds Table 7 with the normalized Shannon entropy and Gini coefficient of the selected-ROI morphology-cluster distribution, for AROMA and for an equal-budget uniform-random selection from the same candidate pool, on all five datasets:

Two observations accompany the table in the revised text. First, AROMA's selection is at least as even as uniform-random selection on four of the five datasets, with higher entropy and lower Gini coefficients on Severstal, MTD, and Kolektor, and a tie on MVTec Leather. Second, on AITeX—the most heterogeneous surface—AROMA deliberately trades distributional evenness for compatibility (entropy: 0.815 vs. 0.859; Gini: 0.406 vs. 0.336), concentrating placements on compatible pairs. This breadth-for-compatibility trade-off is an intended behavior, and its downstream consequence is evaluated in section 4.2, where AITeX shows the largest gain of AROMA over random placement (+2.84 pp). We also clarify that entropy and Gini measure distributional evenness rather than placement quality. Accordingly, the purpose-aligned selection metrics remain the coverage statistics reported in section 4.1, alongside which the new distributional measures are provided.

Manuscript changes: §4.1 (new Table 5b, entropy and Gini for both arms on all five datasets, with interpretation paragraph).

---

## Comment 6

> *The keywords include "ControlNet" and "Stable Diffusion," despite neither technique being used in the proposed method.*

Response.

We agree and have corrected the keyword list. "ControlNet" and "Stable Diffusion" described related work rather than the proposed method and have been removed. The revised keywords reflect what the paper actually contributes and uses:

Manuscript changes: keyword list replaced.

---

## Comment 7

> *The references lack relevant works published within the last three years.*

Response.

We have strengthened the recency of the bibliography. The revised manuscript adds three works from 2024–2025, integrated into the text rather than appended:

- Hu et al., AAAI 2024 (AnomalyDiffusion) — few-shot anomaly image generation with diffusion models, cited in §2.3 as the current representative of diffusion-based defect generation;
- Zhang et al., CVPR 2024 (RealNet) — realistic synthetic anomaly generation with strength-controllable diffusion, cited in §2.3 as recent evidence that synthetic-anomaly realism drives downstream detection;
- Yang et al., Knowledge-Based Systems 2025 — balance recovery and collaborative adaptation for federated fault diagnosis, cited in the new data-decentralization discussion in §2.1.

Manuscript changes: §2.1 and §2.3 (new citations in context); References (three works added, 2024–2025).

---
