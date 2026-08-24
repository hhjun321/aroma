# Response to Reviewer 1

We thank Reviewer 1 for the rigorous and detailed evaluation. We have revised the manuscript substantially and address all five comments below. The central subject of AROMA is the transition from our earlier CASDA pipeline, in which structural settings were specified manually, to a pipeline in which these settings are derived from each dataset's own statistics. To keep the revision aligned with this data-driven thesis, we have strengthened the method in three areas.

(i) learning-based parameter determination for the categorical structure
Morphology clusters, context cells, and subtype thresholds are all estimated from the profiled data (revised section 3.2.2–3.2.3).

(ii) removal of manually specified weighting from the ROI-scoring equation
The revised Eq. (3) combines its two normalized terms as an unweighted sum, leaving no coefficient to tune (revised section 3.2.4).

(iii) a ring-based site-resolution step for ROI selection
Each defect is placed at the position whose surrounding context distribution best matches the context observed around that defect morphology in the real data, replacing geometric heuristics and random choice alike (revised section 3.2.4).

Comment 1: The AROMA framework still incorporates a significant number of pre-defined structures and empirical rules. It is necessary to demonstrate that these manually set parameters do not dominate the final performance through systematic sensitivity analysis or learning-based parameter determination methods.

Response 1: We thank the reviewer for this important point, and we have addressed it with both learning-based parameter determination and systematic sensitivity analysis.

- Learning-based parameter determination (revised section 3.2.2–3.2.3).
The parameters that define AROMA's categorical structure are estimated from each dataset's own statistics rather than hand-set: morphology clusters are obtained by a Gaussian mixture whose cluster count is selected per dataset by the Bayesian Information Criterion; background context cells are formed by per-feature tertile (P33/P66) binning of the profiled context features; and the subtype thresholds are derived per dataset from the observed morphology distributions (Table 5), varying by up to a factor of 4.5 across datasets — variation that any fixed constant would silently absorb.

- Ablation study (new section 4.4).
A leave-one-out ablation over the three placement stages shows that the full pipeline (0.5197 mAP@0.5) outperforms every partial variant, and that disabling any single stage drops performance below even uniform-random augmentation. The measured gains are thus a property of the integrated pipeline design, not of any individually tuned constant.

In addition, we have made the scope of the "data-driven" claim exact throughout the manuscript: every quantity entering a placement decision is either estimated from the dataset's own statistics or an unweighted combination of such estimates, with no hand-set coefficients remaining in the placement path.

Manuscript changes: revised section 3.2.2–3.2.3 (data-driven partitions, derived thresholds); ROI-scoring equation reduced to an unweighted, constant-free combination with a role-based rationale (section 3.2.4); new boundary-sensitivity subsection (section 4.5); new ablation study (section 4.4).

Comment 2: CCI is further used to explain when AROMA is effective, yet this conclusion is actually based on only five datasets. Five data points are insufficient to demonstrate a consistent relationship between CCI and the performance gains achieved by AROMA.

Response 2: We agree. Five datasets cannot statistically establish a quantitative relationship between CCI and the magnitude of AROMA's gain, and we have revised the manuscript so that no such claim is made.

- The evidence is not limited to five (CCI, gain) points.
The cross-dataset pattern is supported by within-dataset mechanistic measurements (section 4.1): the background-compatibility signal that AROMA exploits is directly measured per dataset, and is statistically significant on four of the five datasets (e.g., Severstal Δ = +0.043, p = 3.8×10⁻⁴³), while on MVTec Leather — the dataset where AROMA shows no advantage — the compatibility ranking measurably collapses onto a near-uniform background pool, removing the positional signal (section 4.3).
The claimed mechanism (contextual heterogeneity → informative placement signal → downstream gain) is therefore observed at the level of the mechanism itself, not merely inferred from a five-point correlation.

- Explicit limitation.
We have added a statement to the Discussion/Limitations acknowledging that the CCI–gain relationship is a hypothesis whose quantitative form requires validation on a substantially larger dataset roster, which we identify as future work.

Manuscript changes: tone adjustments in Abstract, section 4.3, section 5, section 6 (conditional-effectiveness framing); new limitation sentence in section 5 (CCI–gain relationship as hypothesis, larger roster as future work).

Comment 3: It is recommended to at least include comparisons with more representative methods, such as copy-paste, context-aware placement, hard-sample augmentation, and relevant generative defect augmentation techniques.

Response 3: All experimental arms composite identical real defect pixels with the identical blending operator and the identical synthesis budget; the placement policy is the only manipulated variable (Introduction, section 3.1). This design attributes measured differences specifically to placement, which is the paper's claimed contribution.

- Copy-paste.
The canonical copy-paste method [Dwibedi et al., ICCV 2017; Ghiasi et al., CVPR 2021] is already included: our Random arm is uniform copy-paste — authentic defect pixels pasted at uniformly random valid positions through the same compositing engine. It is evaluated on all five datasets. We have clarified this equivalence in the revised section 3.1.

- Context-aware placement.
The revised manuscript adds a leave-one-out ablation (section 4.4, Table 15) that decomposes context-awareness into its three constituent decisions (ROI selection, background assignment, site resolution) and evaluates each partial policy downstream — a finer-grained comparison than a single monolithic context-aware baseline. We have also expanded section 2.5 to discuss prior context-aware placement work [Dvornik et al., ECCV 2018; InstaBoost, ICCV 2019] and to state the setting difference: those methods relocate annotated instances or insert general objects within natural scenes, whereas AROMA composites defects from a cross-image defect pool onto clean industrial backgrounds.

Manuscript changes: section 3.1 (Random arm identified as canonical uniform copy-paste, with citations); section 2.5 (context-aware placement discussion expanded, InstaBoost added); section 4.4 (new ablation study); section 5 (limitation statement on generative comparison scope and follow-up work).

Comment 4: Please further explain why AROMA is ineffective—or even leads to a performance decline—on certain datasets.

Response 4: We have both re-examined these cases experimentally and explained them mechanistically in the revision.

- Corrected results: no dataset-level performance decline remains.
All downstream experiments were re-run under a unified multi-seed protocol. Under this protocol, AROMA improves over the real-only baseline on all five datasets, and against random placement the gap is positive in direction on four (AITeX +2.84 pp, Severstal +1.32 pp, MTD +0.41 pp, Kolektor +0.29 pp) and null on one (MVTec Leather, −0.11 pp). The corrected protocol removes this pathology, and seed-level variance is now reported throughout.

Importantly, in no dataset does AROMA fall below random placement or the baseline beyond seed noise: the worst observed case is parity. The revised Abstract and Conclusion state this "beneficial when both headroom and contextual diversity are present, otherwise neutral rather than harmful" characterization explicitly.

Manuscript changes: Tables 6–10 rewritten (3-seed mean ± std, unified protocol); interpretation paragraphs in section 4.2–4.3; section 5 (monotonic CCI relationship, conditional effectiveness); Abstract and section 6 (neutrality statement).

Comment 5: Data decentralization is an important issue in the field. The authors could discuss this by considering some works. -Balance recovery and collaborative adaptation approach for federated fault diagnosis of inconsistent machine groups.

Response 5: We thank the reviewer for pointing to this line of work. We have added a discussion of this issue to Section 2, citing the suggested work on balance recovery and collaborative adaptation for federated fault diagnosis of inconsistent machine groups (Yang et al., Knowledge-Based Systems, 2025). We also note the connection to AROMA's design: because AROMA derives its compatibility model and all categorical structure exclusively from each dataset's own local statistics — with no cross-dataset information required — it is directly deployable as a site-local augmentation module within such federated settings, complementing collaborative model adaptation with local data-side balance recovery.

Manuscript changes: section 2.1 (new discussion of data decentralization and federated fault diagnosis, with its complementarity to local data-side augmentation); References (Yang et al., Knowledge-Based Systems 2025, 317, 113480 added).
