# Response to Reviewer 3

We thank Reviewer 3 for the rigorous and detailed evaluation. We have revised the manuscript substantially and address all seven comments below. The central subject of AROMA is the transition from our earlier CASDA pipeline, in which structural settings were specified manually, to a pipeline in which these settings are derived from each dataset's own statistics. To keep the revision aligned with this data-driven thesis, we have strengthened the method in three areas.

(i) learning-based parameter determination for the categorical structure
Morphology clusters, context cells, and subtype thresholds are all estimated from the profiled data (revised section 3.2.2–3.2.3).

(ii) removal of manually specified weighting from the ROI-scoring equation
The revised Eq. (3) combines its two normalized terms as an unweighted sum, leaving no coefficient to tune (revised section 3.2.4).

(iii) a ring-based site-resolution step for ROI selection
Each defect is placed at the position whose surrounding context distribution best matches the context observed around that defect morphology in the real data, replacing geometric heuristics and random choice alike (revised section 3.2.4).

Comment 1: The introduction and methodology heavily criticize existing frameworks (like CASDA) for relying on "domain-specific handcrafted rules" and claim that AROMA completely replaces manual tuning with a "data-driven" approach. However, this claim is fundamentally false based on the authors' own methodology. Table 3 explicitly uses hardcoded, manually engineered percentile cascades to define background categories (e.g., "Smooth" requires Local Variance ≤ P25). Furthermore, Table 4 dictates entirely arbitrary, hardcoded rules for defect subtypes (e.g., "linear_scratch" is strictly defined as Linearity > 0.9 AND AspectRatio > 5). These are manual, hand-set constants that contradict the paper's central methodological claim of being entirely data-driven.

Response 1: We appreciate the opportunity to address this directly, because the two tables the reviewer cites work differently in the revised manuscript than the comment assumes — and where the criticism applied to the submitted version, we have fixed the text.

- The cited Table 4 rule no longer exists.
The fixed cascade quoted by the reviewer ("linear_scratch: Linearity > 0.9 AND AspectRatio > 5") has been replaced: subtype boundaries are now derived per dataset from the observed morphology distributions as tertile boundaries, with the derived values reported in Table 5.

- Empirical confirmation.
The new ablation study (section 4.4) shows the downstream gain is produced by the estimated compatibility chain itself: replacing any stage with its random counterpart drops mAP below even uniform-random augmentation, confirming that the measured benefit comes from the data-derived structure rather than from any fixed constant.

Manuscript changes: section 3.2.2–3.2.3 (per-dataset percentile derivation, Table 5, homogeneity safeguard); ROI-scoring equation rewritten as an unweighted sum (section 3.2.4); new sensitivity subsection (section 4.5: boundary perturbation); new ablation study (section 4.4); claim-scope wording revised throughout.

Comment 2: The downstream detection results do not demonstrate the superiority of the AROMA framework. On the Kolektor dataset, AROMA (0.9870 mAP@0.5) is actively worse than the Random baseline (0.9938 mAP@0.5). On the MTD dataset, AROMA (0.9440 mAP@0.5) again underperforms the Random baseline (0.9465 mAP@0.5). Most egregiously, on the MVTec Leather dataset, AROMA (0.8052 mAP@0.5) degrades performance severely, losing to both the original Baseline (0.8321 mAP@0.5) and the Random approach (0.8543 mAP@0.5). A proposed pipeline that fails to beat a naive uniform-random placement baseline on 3 out of the 5 evaluated datasets cannot be claimed as a robust advancement.

Response 2: The reviewer's reading of the submitted numbers was fair, and this comment prompted the most substantial revision in the paper, on two levels: the placement method itself was improved, and the evaluation protocol was corrected.

- The placement mechanism of section 3.2.4 was revised, and all results were re-measured.
The revised section 3.2.4 reformulates the final placement decision as ring-context distribution matching: each candidate position's surrounding context histogram is matched against the compatibility model's target profile for the defect's morphology cluster, with void and unobserved tiles excluded before scoring. All downstream experiments were re-run with this revised mechanism under a unified multi-seed protocol (n = 3 seeds, identical training configuration across all arms).

AROMA outperforms Random in mAP@0.5 on four of the five datasets, with gains of +2.84 pp on AITeX, +1.32 pp on Severstal, +0.41 pp on MTD, and +0.29 pp on Kolektor; the difference on MVTec Leather is −0.11 pp. AROMA also outperforms the real-only Baseline on all five datasets, and no dataset shows a meaningful degradation relative to Random beyond seed-level variation. In particular, Kolektor—the dataset cited as showing that AROMA was "actively worse"—now yields an mAP@0.5 of 0.9866 for AROMA versus 0.9837 for Random.

The revised manuscript explicitly states the boundary of the method's effectiveness as a finding. On a monotone background such as MVTec Leather, the background provides little informative compatibility signal: the compatibility ranking becomes nearly uniform, so the placements produced by AROMA differ only marginally from random placements, and the resulting performance difference is correspondingly negligible (−0.11 pp, within seed-level variation). We state this limitation explicitly: for datasets with largely homogeneous backgrounds, the AROMA methodology offers limited benefit.

Manuscript changes: section 3.2.4 (revised ring-context site resolution); Tables 6–10 (re-measured, 3-seed mean ± std, unified protocol); interpretation in section 4.2–4.3; section 5 (monotonic CCI relationship, effectiveness boundary); Abstract and section 6.

Comment 3: The ROI scoring equation (Equation 2) dictates a score based on 0.6⋅ctx_prior+0.4⋅ The authors state that "the ranking introduces no hand-set constants," yet the weights 0.6 and 0.4 are literally hand-set constants. There is no empirical justification or ablation study provided to prove why these specific weights are optimal. Similarly, Equation 4 assigns arbitrary weights (0.30 for blur, 0.30 for contrast, 0.20 for brightness, 0.20 for noise) to calculate a quality score.

Response 3: We address this in two parts.

- The quoted sentence was wrong, and both it and the weights themselves have been removed.
The reviewer is right that "the ranking introduces no hand-set constants" contradicted the visible 0.6/0.4 weights. The revised ROI-scoring equation (Eq. (3) in the revised manuscript) combines the two normalized terms as an unweighted sum, ROI_score = ctx_prior + morph_prior, with a role-based rationale in section 3.2.4 — the context term carries the placement signal, spans the full [0, 1] range after row normalization, and therefore naturally dominates the ranking, while the bounded cluster prior orders candidates of equal compatibility.

- Ablation study is now provided.
The new section 4.4 ablates the placement pipeline stage by stage (ROI selection, background assignment, site resolution) with downstream mAP as the endpoint: the full pipeline (0.5197) outperforms every leave-one-out variant, and removing the compatibility-based ROI selection causes the largest drop (−3.80 pp, consistent across all three seeds). This complements the sensitivity analysis: the mechanism's presence is what carries the gain; the coefficient values are non-critical.

The revised text (section 3.2.2–3.2.4) now makes the claim exact: every quantity entering the placement decision is either derived from the dataset's own statistics or an unweighted combination of such quantities; the manually set coefficients the reviewer identified (the Eq. (2) and Eq. (4) weights of the submitted version) have both been removed from the method.

Manuscript changes: offending sentence removed; ROI-scoring equation reduced to an unweighted, constant-free combination with a role-based rationale (section 3.2.4); new boundary-sensitivity subsection (section 4.5); new ablation study (section 4.4); Quality Gate subsection (Eq. (4)) removed.

Comment 4: The visual presentation of the data is severely lacking. Figures 3, 4, 5, and 6 feature text, axis labels, legends, and annotations that are far too small to be legible.

Response 4: We agree and have regenerated the figures. All plots in the revised manuscript have been improved for better readability, with clearer labels, legends, annotations, line widths, and marker sizes.

Manuscript changes: all figures regenerated under an explicit legibility standard; new figures for the revised section 3.2.4; new tables for section 4.4 and section 4.5.

Comments 5–7: The literature review can also be improved please find attached some recommendations to improve your literature review:
Panagiotis Stavropoulos, Alexios Papacharalampopoulos, Dimitris Petridis, A vision-based system for real-time defect detection: a rubber compound part case study, Procedia CIRP, Volume 93, 2020, Pages 1230–1235.
Bergmann, P., Batzner, K., Fauser, M. et al. The MVTec Anomaly Detection Dataset: A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection. Int J Comput Vis 129, 1038–1059 (2021).

Response 5–7: We thank the reviewer for the recommendations; both works are now reflected in the manuscript.

- Stavropoulos et al. (Procedia CIRP 2020) has been added to section 2.1.
The reference and its relevance to vision-based industrial defect detection have been appropriately incorporated into the discussion in section 2.1.

- Bergmann et al. (IJCV 2021)
We additionally cite this reference in section 2.1 when introducing one-class anomaly detection, providing appropriate attribution in this context.

Manuscript changes: section 2.1 (Stavropoulos et al. added; Bergmann et al. cited at the one-class paradigm introduction); References updated; broader literature additions in section 2.1, 2.3, and 2.5.
