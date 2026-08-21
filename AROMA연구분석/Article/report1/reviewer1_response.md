# Response to Reviewer 1

We thank the reviewer for the careful reading and constructive comments. Below we respond point by point. Reviewer comments are quoted in italics; our responses follow, with the corresponding manuscript changes indicated.

---

## Comment 1

> *The AROMA framework still incorporates a significant number of pre-defined structures and empirical rules. It is necessary to demonstrate that these manually set parameters do not dominate the final performance through systematic sensitivity analysis or learning-based parameter determination methods.*

**Response.**

We thank the reviewer for this important point, and we have addressed it with both learning-based parameter determination and systematic sensitivity analysis.

**(1) Learning-based parameter determination (revised §3.2.2–3.2.3).** The parameters that define AROMA's categorical structure are estimated from each dataset's own statistics rather than hand-set: morphology clusters are obtained by a Gaussian mixture whose cluster count is selected per dataset by the Bayesian Information Criterion; background context cells are formed by per-feature tertile (P33/P66) binning of the profiled context features; and the subtype thresholds are derived per dataset from the observed morphology distributions (Table 4b), varying by up to a factor of 4.5 across datasets — variation that any fixed constant would silently absorb.

**(2) Systematic sensitivity analysis (new §4.5).** For the remaining fixed design choices — the ROI-scoring weights — we swept the ctx:morph weight ratio exhaustively on all five datasets. Every ratio from 0.1/0.9 to 0.9/0.1 retains 87.5–100% of the top-K ROI selection, while removing the context term entirely causes substantial divergence (retention falls to 37.5% on Kolektor). The exact weight values therefore do not dominate the outcome; what matters is the participation and priority of the context-compatibility term.

**(3) Ablation study (new §4.4).** A leave-one-out ablation over the three placement stages shows that the full pipeline (0.5197 mAP@0.5) outperforms every partial variant, and that disabling any single stage drops performance below even uniform-random augmentation. The measured gains are thus a property of the integrated pipeline design, not of any individually tuned constant.

In addition, we have narrowed the scope of the "data-driven" claim throughout the manuscript: it now refers specifically to the derivation of the defect–background compatibility model, while the scoring-weight values are explicitly acknowledged as fixed design choices whose robustness is demonstrated by the sensitivity analysis above.

**Manuscript changes:** revised §3.2.2–3.2.3 (data-driven partitions, Table 4b); fixed-design-choice statement and priority-order rationale at Eq. (2) in §3.2.4; new sensitivity subsection (§4.5); new ablation study (§4.4, Table 11).

---

## Comment 2

> *CCI is further used to explain when AROMA is effective, yet this conclusion is actually based on only five datasets. Five data points are insufficient to demonstrate a consistent relationship between CCI and the performance gains achieved by AROMA.*

**Response.**

We agree. Five datasets cannot statistically establish a quantitative relationship between CCI and the magnitude of AROMA's gain, and we have revised the manuscript so that no such claim is made.

**(1) Claim demoted from predictor to characterization.** In the revised manuscript, the CCI is presented as a descriptive complexity characterization of the evaluation roster, and the conditional-effectiveness statement is framed as an observed pattern with a mechanistic explanation — not as a fitted or validated predictive relationship. The Abstract and Conclusion now state that context-aware placement is beneficial "when both improvement headroom and contextual diversity are present," offered as practical guidance rather than a law.

**(2) The evidence is not limited to five (CCI, gain) points.** The cross-dataset pattern is supported by within-dataset mechanistic measurements (§4.1): the background-compatibility signal that AROMA exploits is directly measured per dataset, and is statistically significant on four of the five datasets (e.g., Severstal Δ = +0.032, p = 3.1×10⁻²³), while on MVTec Leather — the dataset where AROMA shows no advantage — the compatibility ranking measurably collapses onto a near-uniform background pool, removing the positional signal (§4.3). The claimed mechanism (contextual heterogeneity → informative placement signal → downstream gain) is therefore observed at the level of the mechanism itself, not merely inferred from a five-point correlation.

**(3) Explicit limitation.** We have added a statement to the Discussion/Limitations acknowledging that the CCI–gain relationship is a hypothesis whose quantitative form requires validation on a substantially larger dataset roster, which we identify as future work.

**Manuscript changes:** tone adjustments in Abstract, §4.3, §5, §6 (conditional-effectiveness framing); new limitation sentence in §5 (CCI–gain relationship as hypothesis, larger roster as future work).

---

## Comment 3

> *It is recommended to at least include comparisons with more representative methods, such as copy-paste, context-aware placement, hard-sample augmentation, and relevant generative defect augmentation techniques.*

**Response.**

We appreciate this recommendation and respond for each of the four method families, noting first the design principle of our evaluation: it is a controlled placement experiment. All experimental arms composite identical real defect pixels with the identical blending operator and the identical synthesis budget; the placement policy is the only manipulated variable (Introduction, §3.1). This design attributes measured differences specifically to placement, which is the paper's claimed contribution.

**(1) Copy-paste.** The canonical copy-paste method [Dwibedi et al., ICCV 2017; Ghiasi et al., CVPR 2021] is already included: our Random arm *is* uniform copy-paste — authentic defect pixels pasted at uniformly random valid positions through the same compositing engine. It is evaluated on all five datasets (Tables 6–10). We have clarified this equivalence in the revised §3.1.

**(2) Context-aware placement.** The revised manuscript adds a leave-one-out ablation (§4.4, Table 11) that decomposes context-awareness into its three constituent decisions (ROI selection, background assignment, site resolution) and evaluates each partial policy downstream — a finer-grained comparison than a single monolithic context-aware baseline. We have also expanded §2.5 to discuss prior context-aware placement work [Dvornik et al., ECCV 2018; InstaBoost, ICCV 2019] and to state the setting difference: those methods relocate annotated instances or insert general objects within natural scenes, whereas AROMA composites defects from a cross-image defect pool onto clean industrial backgrounds.

**(3) Hard-sample and generative defect augmentation.** These families change *what is synthesized* (defect appearance), not only *where it is placed*. Including them in the same table would vary appearance and placement simultaneously, confounding the attribution the controlled design exists to protect: any observed difference could no longer be assigned to the placement policy. They are, moreover, orthogonal to AROMA — a generative or hard-sample synthesizer could be combined with AROMA's placement gate, which we consider a promising extension. We now state explicitly in the Limitations that a head-to-head downstream comparison against generative defect synthesis is outside the scope of the placement-controlled design, and we note our ongoing work integrating a ControlNet-based generation arm into the same placement framework as follow-up research.

**Manuscript changes:** §3.1 (Random arm identified as canonical uniform copy-paste, with citations); §2.5 (context-aware placement discussion expanded, InstaBoost added); §4.4 (new ablation study); §5 (limitation statement on generative comparison scope and follow-up work).

---

## Comment 4

> *Please further explain why AROMA is ineffective—or even leads to a performance decline—on certain datasets.*

**Response.**

We have both re-examined these cases experimentally and explained them mechanistically in the revision.

**(1) Corrected results: no dataset-level performance decline remains.** All downstream experiments were re-run under a unified multi-seed protocol (n = 3 seeds, mean ± std, identical training configuration across all arms; Tables 6–10). Under this protocol, AROMA improves over the real-only baseline on all five datasets, and against random placement the gap is positive in direction on four (AITeX +2.84 pp, Severstal +1.32 pp, MTD +0.41 pp, Kolektor +0.29 pp) and null on one (MVTec Leather, −0.11 pp). The decline reported in the original submission — most visibly on MVTec Leather — traced to single-seed runs in which a training configuration ill-suited to the small dataset (large batch with aggressive early stopping, effectively terminating training within a few gradient updates) caused training collapse in individual arms. The corrected protocol removes this pathology, and seed-level variance is now reported throughout.

**(2) Why the advantage narrows to parity on some datasets.** The revised §4.2–4.3 and §5 identify two conditions under which context-aware placement stops adding value over random placement, and both are now stated as part of the paper's contribution rather than as anomalies. (a) *Baseline headroom exhaustion*: on Kolektor (baseline 0.9503) and MTD (0.9113), detection is near ceiling and any placement policy converges — augmentation gains shrink to a few pp for both arms. (b) *Uninformative compatibility signal*: on MVTec Leather's near-homogeneous surface, the similarity-ranked background assignment collapses onto a small background pool, so the compatibility ranking carries no positional signal and AROMA reduces to a diversity-restricted variant of random placement (§4.3); augmentation itself remains highly effective (both arms ≈ +9 pp over baseline), but *which* background is chosen is not the operative variable. Across the roster the AROMA–random gap decreases monotonically with the CCI (§5), consistent with this mechanism. At class level, offsetting movements persist in the near-ceiling regime (e.g., MTD: Crack +3.28 pp vs. Uneven −3.02 pp against random), which we now report explicitly.

Importantly, in no dataset does AROMA fall below random placement or the baseline beyond seed noise: the worst observed case is parity. The revised Abstract and Conclusion state this "beneficial when both headroom and contextual diversity are present, otherwise neutral rather than harmful" characterization explicitly.

**Manuscript changes:** Tables 6–10 rewritten (3-seed mean ± std, unified protocol); interpretation paragraphs in §4.2–4.3; §5 (monotonic CCI relationship, conditional effectiveness); Abstract and §6 (neutrality statement).

---

## Comment 5

> *Data decentralization is an important issue in the field. The authors could discuss this by considering some works. -Balance recovery and collaborative adaptation approach for federated fault diagnosis of inconsistent machine groups.*

**Response.**

We thank the reviewer for pointing to this line of work. Data decentralization is indeed a growing constraint in industrial inspection: defect samples are not only scarce but also fragmented across production sites and machine groups, and privacy or operational barriers often prevent pooling them centrally. We have added a discussion of this issue to §2, citing the suggested work on balance recovery and collaborative adaptation for federated fault diagnosis of inconsistent machine groups (Yang et al., Knowledge-Based Systems, 2025). We also note the connection to AROMA's design: because AROMA derives its compatibility model and all categorical structure exclusively from each dataset's own local statistics — with no cross-dataset information required — it is directly deployable as a site-local augmentation module within such federated settings, complementing collaborative model adaptation with local data-side balance recovery.

**Manuscript changes:** §2.1 (new discussion of data decentralization and federated fault diagnosis, with its complementarity to local data-side augmentation); References (Yang et al., Knowledge-Based Systems 2025, 317, 113480 added).

---
