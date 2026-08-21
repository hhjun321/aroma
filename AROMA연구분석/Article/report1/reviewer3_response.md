# Response to Reviewer 3

We thank the reviewer for the careful reading and constructive comments. Below we respond point by point. Reviewer comments are quoted in italics; our responses follow, with the corresponding manuscript changes indicated.

---

## Comment 1

> *The introduction and methodology heavily criticize existing frameworks (like CASDA) for relying on "domain-specific handcrafted rules" and claim that AROMA completely replaces manual tuning with a "data-driven" approach. However, this claim is fundamentally false based on the authors' own methodology. Table 3 explicitly uses hardcoded, manually engineered percentile cascades to define background categories (e.g., "Smooth" requires Local Variance ≤ P25). Furthermore, Table 4 dictates entirely arbitrary, hardcoded rules for defect subtypes (e.g., "linear_scratch" is strictly defined as Linearity > 0.9 AND AspectRatio > 5). These are manual, hand-set constants that contradict the paper's central methodological claim of being entirely data-driven.*

**Response.**

We appreciate the opportunity to address this directly, because the two tables the reviewer cites work differently in the revised manuscript than the comment assumes — and where the criticism applied to the submitted version, we have fixed the text.

**(1) The cited Table 4 rule no longer exists.** The fixed cascade quoted by the reviewer ("linear_scratch: Linearity > 0.9 AND AspectRatio > 5") has been replaced: subtype boundaries are now derived per dataset from the observed morphology distributions as tertile boundaries, with the derived values reported in Table 4b. These values vary by up to a factor of 4.5 across the five datasets (aspect-ratio boundary 3.62 on MTD vs. 16.43 on AITeX) — variation that the old fixed rule would have absorbed silently, which is exactly why it was removed. A homogeneity safeguard additionally reverts a feature to a conservative default when its middle tertile is degenerate (below 15% of the feature's standard deviation), so a dataset that is genuinely homogeneous in one feature is not split on measurement noise.

**(2) Percentile rules are not hand-set constants.** A rule such as "LocalVariance ≤ P25" (Table 3) contains no hardcoded threshold: P25 denotes a position in each dataset's own feature distribution, so the operative numerical threshold is recomputed for every dataset and differs across them. What is fixed is only the partition convention (quartiles for background categories, tertiles for context cells and subtypes) — dataset-agnostic machinery, analogous to choosing a histogram binning, not domain knowledge. We verified this convention is not a hidden tuning knob: perturbing the tertile boundary positions by ±5 percentile points changes final ROI selections proportionally and without instability (top-K overlap 64–98% across all perturbations and datasets; new §4.5).

**(3) What "data-driven" claims — and what it replaces in CASDA.** CASDA's compatibility matrix encodes domain semantics by hand: an engineer asserts which defect types belong on which background types. AROMA derives exactly this object from patch-level co-occurrence statistics (§3.2.4), and its categorical structure from per-dataset estimation (BIC-selected Gaussian-mixture morphology clusters; tertile context cells). The revised manuscript states the claim at this scope: applying AROMA to a new dataset requires no re-engineering, while the remaining dataset-independent design constants (the scoring weights of Eq. 2) are explicitly acknowledged as fixed choices whose robustness is demonstrated by sensitivity analysis (±0.1 weight perturbations retain 95.5–100% of selections). We have also separated the two roles the submitted version conflated: the named categories of Tables 3–4 serve interpretation and reporting; placement decisions operate on the estimated clusters and cells.

**(4) Empirical confirmation.** The new ablation study (§4.4) shows the downstream gain is produced by the estimated compatibility chain itself: replacing any stage with its random counterpart drops mAP below even uniform-random augmentation, confirming that the measured benefit comes from the data-derived structure rather than from any fixed constant.

**Manuscript changes:** §3.2.2–3.2.3 (per-dataset percentile derivation, Table 4b, homogeneity safeguard); Eq. (2) fixed-design-choice statement (§3.2.4); new sensitivity subsection (§4.5: boundary perturbation and weight simplex); new ablation study (§4.4); claim-scope wording revised throughout.

---

## Comment 2

> *The downstream detection results do not demonstrate the superiority of the AROMA framework. On the Kolektor dataset, AROMA (0.9870 mAP@0.5) is actively worse than the Random baseline (0.9938 mAP@0.5). On the MTD dataset, AROMA (0.9440 mAP@0.5) again underperforms the Random baseline (0.9465 mAP@0.5). Most egregiously, on the MVTec Leather dataset, AROMA (0.8052 mAP@0.5) degrades performance severely, losing to both the original Baseline (0.8321 mAP@0.5) and the Random approach (0.8543 mAP@0.5). A proposed pipeline that fails to beat a naive uniform-random placement baseline on 3 out of the 5 evaluated datasets cannot be claimed as a robust advancement.*

**Response.**

The reviewer's reading of the submitted numbers was fair, and this comment prompted the most substantial revision in the paper, on two levels: the placement method itself was improved, and the evaluation protocol was corrected.

**(1) The placement mechanism of §3.2.4 was revised, and all results were re-measured.** The revised §3.2.4 reformulates the final placement decision as ring-context distribution matching: each candidate position's surrounding context histogram is matched against the compatibility model's target profile for the defect's morphology cluster, with void and unobserved tiles excluded before scoring. All downstream experiments were re-run with this revised mechanism under a unified multi-seed protocol (n = 3 seeds, identical training configuration across all arms; Tables 6–10). The submitted single-seed numbers additionally contained noise artifacts: in near-ceiling regimes (Kolektor baseline ≈ 0.95, MTD ≈ 0.91) sub-1-pp single-seed differences are dominated by seed variance (the corrected Kolektor baseline alone has seed std 0.0398, four times the gap the reviewer cites), and the MVTec Leather run had suffered a training collapse under a configuration ill-suited to that small dataset. The revised protocol removes these artifacts and reports per-seed sign consistency.

**(2) Under the revised method and protocol, the pattern the reviewer criticizes no longer exists.** AROMA vs. Random (mAP@0.5): AITeX +2.84 pp, Severstal +1.32 pp, MTD +0.41 pp, Kolektor +0.29 pp, MVTec Leather −0.11 pp. AROMA outperforms the real-only Baseline on **5 of 5** datasets, and on no dataset does it fall below Random beyond seed noise. In particular Kolektor — cited as "actively worse" — now shows AROMA 0.9866 vs. Random 0.9837.

**(3) What the paper claims — and where AROMA's methodology does not help.** We do not claim universal superiority over random placement; the revised manuscript states the boundary of the method's effectiveness as a finding. On a monotone background such as MVTec Leather, the background offers no informative compatibility signal: the compatibility ranking collapses onto a near-uniform pool, so the placements AROMA produces barely differ from random placements, and the resulting performance difference is accordingly not meaningful (−0.11 pp, within seed noise). We state this explicitly: **for datasets with largely homogeneous backgrounds, the AROMA methodology offers little benefit** — augmentation itself still helps (both arms gain ≈ +9 pp over baseline on Leather), but the placement policy is not the operative variable there. Across the roster the AROMA–random gap decreases monotonically with background-context complexity (CCI, §5), from +2.84 pp on the most heterogeneous surface to parity on the most homogeneous one, and the worst case observed anywhere is parity. This calibrated claim — consistent gains where baseline headroom and contextual diversity coexist, neutrality on monotone backgrounds — is the robust and reproducible form of the contribution.

**Manuscript changes:** §3.2.4 (revised ring-context site resolution); Tables 6–10 (re-measured, 3-seed mean ± std, unified protocol); interpretation in §4.2–4.3; §5 (monotonic CCI relationship, effectiveness boundary); Abstract and §6.

---

## Comment 3

> *The ROI scoring equation (Equation 2) dictates a score based on 0.6⋅ctx_prior+0.4⋅ The authors state that "the ranking introduces no hand-set constants," yet the weights 0.6 and 0.4 are literally hand-set constants. There is no empirical justification or ablation study provided to prove why these specific weights are optimal. Similarly, Equation 4 assigns arbitrary weights (0.30 for blur, 0.30 for contrast, 0.20 for brightness, 0.20 for noise) to calculate a quality score.*

**Response.**

**(1) The quoted sentence was wrong and has been removed.** The reviewer is right that "the ranking introduces no hand-set constants" contradicted the visible weights. The revised §3.2.4 states the opposite, explicitly: the weights are fixed, dataset-independent design choices encoding a priority order (context > morphology > quality). We have also corrected Eq. (2) to the full three-component score used by the implementation, ROI_score = 0.5·ctx_prior + 0.3·morph_prior + 0.2·quality; the two-term form in the submitted version was an oversimplified rendering.

**(2) Empirical justification is now provided — for robustness, not optimality.** We do not claim these weights are optimal, and for a rank-based top-K selection they do not need to be: what must be shown is that the outcome does not hinge on the specific values. The new sensitivity analysis (§4.5) sweeps the full weight simplex in 0.1 steps on all five datasets. Perturbing the adopted weights by ±0.1 retains 95.5–100% of the top-K selection (mean 98.5–100%); across the broad interior of the simplex the selection remains stable; the only changes of consequence occur when a component is removed entirely (e.g., zeroing the quality term alters 35–47% of the selection on four of the five datasets, and no single-component score reproduces the selection on all five — retention falls to 34–66% on at least one dataset per zeroed configuration). The weights therefore act as a priority ordering over participating terms, not as a tuned optimum — the ranking is insensitive to their exact values but sensitive to which terms participate.

**(3) Ablation study is now provided.** The new §4.4 ablates the placement pipeline stage by stage (ROI selection, background assignment, site resolution) with downstream mAP as the endpoint: the full pipeline (0.5197) outperforms every leave-one-out variant, and removing the compatibility-based ROI selection causes the largest drop (−3.80 pp, consistent across all three seeds). This complements the sensitivity analysis: the mechanism's presence is what carries the gain; the coefficient values are non-critical.

**(4) Equation (4) has been removed.** We agree these weights were arbitrary in the sense that matters: they were ported unchanged from CASDA's background-extraction criterion rather than derived. In the revision we deleted the quality-gate subsection presenting Eq. (4) — its fixed weights and absolute threshold — instead of defending it. The gate itself is a coarse admissibility pre-filter that discards unusable patches before any placement decision and is applied identically to the AROMA and random arms, so it is common-mode with respect to every comparison in the paper and is not part of the claimed contribution; the revised §3.2.4 references the crop quality score with its CASDA provenance where it enters Eq. (2), without restating the formula. No numbered equation followed Eq. (4), so the numbering of Equations (1)–(3) is unchanged.

**Manuscript changes:** offending sentence removed and Eq. (2) corrected to the three-component score (§3.2.4); new sensitivity subsection (§4.5); new ablation study (§4.4, Table 11); §3.2.6 (Quality Gate, Eq. (4)) removed, with the quality score's CASDA provenance noted at Eq. (2).

---

## Comment 4

> *The visual presentation of the data is severely lacking. Figures 3, 4, 5, and 6 feature text, axis labels, legends, and annotations that are far too small to be legible.*

**Response.**

We agree and have regenerated the figures. All plots in the revised manuscript are re-rendered with legibility as an explicit constraint: axis labels, tick labels, legends, and in-figure annotations are enforced to a minimum effective size of 7–8 pt at the final single-column print width, line widths and marker sizes are increased accordingly, and dense annotation overlays are either enlarged or moved into captions. We note additionally that a substantial fraction of the figures were replaced or newly created in the course of this revision — the placement mechanism of §3.2.4 was revised (new pipeline, compatibility-heatmap, background-assignment, and site-resolution figures), the results of §4 were re-measured under the multi-seed protocol, and new figures accompany the sensitivity and ablation analyses — so the figures the reviewer cites have been superseded rather than merely reformatted.

**Manuscript changes:** all figures regenerated under an explicit legibility standard; new figures for §3.2.4, §4.4, and the sensitivity subsection.

---

## Comments 5–7 (literature recommendations)

> *The literature review can also be improved please find attached some recommendations to improve your literature review:*
> *Panagiotis Stavropoulos, Alexios Papacharalampopoulos, Dimitris Petridis, A vision-based system for real-time defect detection: a rubber compound part case study, Procedia CIRP, Volume 93, 2020, Pages 1230–1235.*
> *Bergmann, P., Batzner, K., Fauser, M. et al. The MVTec Anomaly Detection Dataset: A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection. Int J Comput Vis 129, 1038–1059 (2021).*

**Response.**

We thank the reviewer for the recommendations; both works are now reflected in the manuscript.

**Bergmann et al. (IJCV 2021)** was already cited in the submitted version as the source of the MVTec Anomaly Detection dataset — reference [3] — and MVTec Leather, one of the five datasets in our evaluation roster, is drawn from it (§3.1). In the revised version we additionally cite it where one-class anomaly detection is introduced in §2.1, so the dataset's role in establishing that paradigm is properly attributed.

**Stavropoulos et al. (Procedia CIRP 2020)** has been added to §2.1: as a vision-based real-time defect detection case study on rubber compound parts, it illustrates the supervised, deployment-oriented end of industrial inspection that motivates our problem setting, and it now anchors the discussion of practical inspection systems alongside the existing surveys.

Beyond these two recommendations, the literature review was broadened in this revision: §2.3 adds recent diffusion-based defect generation (AnomalyDiffusion, AAAI 2024; RealNet, CVPR 2024), §2.5 adds context-aware placement in natural scenes (Dvornik et al., ECCV 2018; InstaBoost, ICCV 2019) with an explicit statement of how our industrial setting differs, and §2.1 adds a discussion of data decentralization and federated fault diagnosis (Yang et al., Knowledge-Based Systems 2025).

**Manuscript changes:** §2.1 (Stavropoulos et al. added; Bergmann et al. cited at the one-class paradigm introduction); References updated; broader literature additions in §2.1, §2.3, §2.5.

---
