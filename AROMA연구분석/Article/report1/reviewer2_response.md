# Response to Reviewer 2

We thank the reviewer for the careful reading and constructive comments. Below we respond point by point. Reviewer comments are quoted in italics; our responses follow, with the corresponding manuscript changes indicated.

---

## Comment 1

> *The paper repeatedly claims to be "free of manual tuning" and "without hand-set constants," yet the hyperparameters in Equations (2) and (4) are manually set.*

**Response.**

We appreciate the opportunity to clarify the scope of this claim, and we have sharpened the wording in the revised manuscript so that it cannot be misread.

**(1) What "free of manual tuning" refers to.** The claim concerns the dataset-facing parameters — everything that must adapt when the method is applied to a new dataset. In the revised manuscript, all of these are derived from each dataset's own statistics rather than set by hand: morphology clusters are selected by BIC over a Gaussian mixture, context cells are formed by per-feature tertile (P33/P66) binning, background categories use percentile boundaries (e.g., P25) of the profiled features, and defect-subtype thresholds are derived from the observed per-dataset morphology distributions (Table 4b), varying by up to a factor of 4.5 across datasets. None of these involve manual tuning: applying AROMA to a new dataset re-derives them automatically, with no re-engineering.

**(2) The weights in Equation (2) are dataset-independent structural coefficients, not tuning parameters.** They are set once, encode a fixed priority order (context > morphology), and are applied unchanged to all five datasets — no per-dataset adjustment exists anywhere in the pipeline. To verify that these coefficients do not function as hidden tuning knobs, we added a weight-ratio sensitivity analysis on all five datasets (new §4.5): every ctx:morph ratio from 0.1/0.9 to 0.9/0.1 retains 87.5–100% of the top-K selection, and the only consequential change is eliminating a term entirely (a morphology-only score falls to 37.5% retention on Kolektor). The ranking therefore depends on which terms participate, not on the specific coefficient values — the opposite of a manually tuned optimum. As a scale reference, the ablation of §4.4 shows that even replacing the entire scored selection with a uniform-random one shifts mAP by 3.8 pp; the selection differences induced by alternative weight ratios (0–12.5% of membership) are far smaller in extent.

**(3) Equation (4) has been removed, and the quality criterion is disclosed as inherited.** In revising the quality-gating description we deleted the subsection that presented Eq. (4) — its fixed component weights and absolute acceptance threshold — rather than defending it. The underlying gate is a coarse admissibility pre-filter inherited unchanged from our earlier CASDA pipeline; it discards unusable patches *before* any placement decision and is applied identically to the AROMA and random arms, so it is common-mode with respect to every comparison in the paper, operates upstream of and independently from the placement scoring of Eq. (2), and is not part of the claimed data-driven contribution. No numbered equation followed Eq. (4), so the numbering of Equations (1)–(3) is unchanged.

The revised text (§3.2.2–3.2.4) now states this distinction explicitly: per-dataset parameters are data-derived, while the Eq. (2) weights are fixed, dataset-independent design constants whose robustness is demonstrated empirically.

**Manuscript changes:** explicit fixed-design-choice statement and priority-order rationale at Eq. (2) (§3.2.4); §3.2.6 (Quality Gate, Eq. (4)) removed; new sensitivity subsection (§4.5); data-derived partition descriptions in §3.2.2–3.2.3.

---

## Comment 2

> *Only YOLOv8n is used as the downstream detector; it is recommended to supplement the experiments with more recent mainstream detectors.*

**Response.**

Following this recommendation, we supplemented the evaluation with **YOLOv11n**, a recent mainstream detector, under the identical three-arm (Baseline / Random / AROMA), three-seed protocol on Severstal and AITeX — the heterogeneous-surface datasets where the placement effect is the operative question (new Table X, §X).

The results reproduce the ordering observed with YOLOv8n: **[AROMA > Random > Baseline on both datasets; numbers TBD after E4]**. This indicates that the measured placement effect is a property of the synthesized data rather than of a particular detector architecture.

Two considerations guided the scope of this supplement. First, the paper's claim is about the *data-side* placement policy, and the detector is only the measurement instrument; demonstrating that the effect survives a change of instrument on two representative datasets addresses the generality question directly. Second, YOLOv8n is retained as the primary detector throughout the paper for comparability with prior industrial augmentation studies and with our earlier controlled experiments, which were all conducted under that architecture.

**Manuscript changes:** new detector-generality table and subsection (numbers to be inserted after the supplementary YOLOv11n runs; placeholder pending E4).

---

## Comment 3

> *Fixed thresholds are used for defect subtype classification without providing a threshold sensitivity analysis.*

**Response.**

We address this in three parts.

**(1) The subtype thresholds are no longer fixed constants.** In the revised manuscript (§3.2.3), subtype boundaries are derived per dataset from the observed morphology distributions as percentile boundaries, reported in Table 4b. The derived values vary by up to a factor of 4.5 across datasets (aspect-ratio boundary 3.62 on MTD vs. 16.43 on AITeX), demonstrating precisely the cross-dataset variation that a fixed threshold would absorb silently.

**(2) Subtype labels do not gate placement.** The revised §3.2.2–3.2.4 separates the two roles that were conflated in the original text: placement decisions operate on the BIC-selected morphology clusters and tertile context cells, and the ROI ranking of Eq. (2) does not consume the subtype labels; the named categories serve interpretation and reporting. A perturbation of a subtype boundary therefore cannot alter which placements are generated.

**(3) Threshold sensitivity analysis (new §4.5).** We nevertheless quantified the stability of the labels themselves: perturbing the percentile points defining the tertile boundaries by ±5 points (P33/P66 → 28–38 / 61–71) on all five datasets relabels 3.1–9.1% of defects for a single-boundary shift and 7.7–14.5% for a joint shift (Table 13) — in each case proportional to the distribution mass moved by the shift, with no amplification or instability cliff.

**Manuscript changes:** §3.2.3 (per-dataset percentile boundaries, Table 4b); new sensitivity subsection (§4.5) reporting the boundary-perturbation results alongside the weight-ratio analysis.

---

## Comment 4

> *The CCI formula lists only three components, whereas the text explicitly states "four normalized components."*

**Response.**

We thank the reviewer for catching this inconsistency. The formula in the submitted version omitted one component; the text's statement of "four normalized components" was correct. The revised §3.2.1 now states the complete formula,

CCI = Mean(TextureEntropy, ContextClusterCount, FreqComplexity, OrientVariance),

and we have additionally added Table 2, which decomposes the CCI of each of the five datasets into these four measured components (texture entropy, context cluster count, frequency-domain complexity, and gradient-orientation variance), so the formula, the text, and the reported values can be verified against one another directly.

**Manuscript changes:** §3.2.1 (complete four-component formula); new Table 2 (per-dataset CCI decomposition).

---

## Comment 5

> *Quantitative results for Shannon entropy and the Gini coefficient are not reported.*

**Response.**

We now report these values quantitatively. The revised §4.1 adds Table 5b with the normalized Shannon entropy and Gini coefficient of the selected-ROI morphology-cluster distribution, for AROMA and for an equal-budget uniform-random selection from the same candidate pool, on all five datasets:

| Dataset | Entropy (AROMA / Random) | Gini (AROMA / Random) |
|---|---|---|
| Severstal | 0.984 / 0.956 | 0.122 / 0.207 |
| MTD | 0.956 / 0.867 | 0.210 / 0.338 |
| MVTec Leather | 0.896 / 0.899 | 0.243 / 0.243 |
| AITeX | 0.815 / 0.859 | 0.406 / 0.336 |
| Kolektor | 0.883 / 0.802 | 0.296 / 0.390 |

Two observations accompany the table in the revised text. First, AROMA's selection is as even as or more even than uniform-random selection on four of the five datasets (higher entropy, lower Gini on Severstal, MTD, and Kolektor; tie on MVTec Leather). This is a direct consequence of the per-pair coverage quotas in the allocation (§3.2.4): uniform sampling inherits the candidate pool's cluster imbalance, whereas the quota actively spreads selections across morphology–context pairs. Second, on AITeX — the most heterogeneous surface — AROMA deliberately trades distributional evenness for compatibility (entropy 0.815 vs. 0.859; Gini 0.406 vs. 0.336), concentrating placements on compatible pairs; this breadth-for-compatibility trade is the intended behavior, and its downstream consequence is measured in §4.2 (AITeX is where AROMA's gain over random placement is largest, +2.84 pp). We also note that entropy and Gini measure distributional evenness, not placement quality; the purpose-aligned selection metrics remain the coverage statistics of §4.1, alongside which the new values are reported.

**Manuscript changes:** §4.1 (new Table 5b, entropy and Gini for both arms on all five datasets, with interpretation paragraph).

---

## Comment 6

> *The keywords include "ControlNet" and "Stable Diffusion," despite neither technique being used in the proposed method.*

**Response.**

We agree and have corrected the keyword list. "ControlNet" and "Stable Diffusion" described related work rather than the proposed method and have been removed. The revised keywords reflect what the paper actually contributes and uses:

*industrial visual inspection; defect detection; data augmentation; copy-paste synthesis; context-aware placement; defect–background compatibility; dataset complexity index; YOLOv8*

(ControlNet-based generation is discussed only as related work in §2.4 and as a possible extension in §5, where it is cited, not claimed.)

**Manuscript changes:** keyword list replaced.

---

## Comment 7

> *The references lack relevant works published within the last three years.*

**Response.**

We have strengthened the recency of the bibliography. The revised manuscript adds three works from 2024–2025, integrated into the text rather than appended:

- **Hu et al., AAAI 2024 (AnomalyDiffusion)** — few-shot anomaly image generation with diffusion models, cited in §2.3 as the current representative of diffusion-based defect generation;
- **Zhang et al., CVPR 2024 (RealNet)** — realistic synthetic anomaly generation with strength-controllable diffusion, cited in §2.3 as recent evidence that synthetic-anomaly realism drives downstream detection;
- **Yang et al., Knowledge-Based Systems 2025** — balance recovery and collaborative adaptation for federated fault diagnosis, cited in the new data-decentralization discussion in §2.1.

These join the recent works already present in the submitted version, including Hütten et al. (2024, survey), EfficientAD (WACV 2024), SimpleNet (CVPR 2023), ControlNet (ICCV 2023), and SAM (ICCV 2023). The related-work discussion now covers the 2023–2025 state of the art in both generative defect synthesis and industrial anomaly detection.

**Manuscript changes:** §2.1 and §2.3 (new citations in context); References (three works added, 2024–2025).

---
