I now have all the committed evidence I need. I have enough grounding to produce the deliverable. The key committed facts:

- **Datasets with committed copy-paste results**: Table 12 roster = mvtec_carpet/mvtec_leather/visa_macaroni/visa_fryum (`.claude/.etc/exp4v2_mp/`); consolidated master = aitex_single(tiled)/severstal(multi)/mtd(multi)/mvtec_leather. These are **different rosters** — flagged as author-decision.
- **Only positive**: aitex_single Δ+0.097, t=4.51, n=1. baseline 0.372 (headroom).
- **deficit refuted**: aitex won with deficit 0/200; mtd lost with deficit 25%. Real driver = ctx_prior(+0.427) + quality_score(+0.30).
- **clean-patch coverage**: leather 4.7% / mtd 67% / aitex 78%.
- **0.923 AUROC frozen** — must stay deleted.

Now producing the markdown deliverable.

# AROMA 논문 개선 산출물

세션 확정 증거(exp4v2 consolidated, aitex 재검증, deficit 진단, placement 재설계)와 audit/검증 findings를 종합한 논문 개선안입니다. **정직성 제약 준수**: 어떤 정량 수치도 창작하지 않았고, 미검증 항목은 `[committed 파일에서 refill]` placeholder로 두되 정성적 방향만 서술했습니다. 되살리면 안 되는 수치(특히 AUROC 0.923)는 §4에 명시했습니다.

---

## 1. 개선 요약 (한국어)

### 핵심 문제
초안 전체가 **"AROMA가 downstream 성능을 개선한다"**는 서사로 관통되어 있으나, 이 서사는 현 committed 증거로 방어 불가합니다.

1. **데이터셋 불일치(3중)**: 본문 §3.1은 구버전 로스터(isp_LSM_1/mvtec_cable/visa_cashew/visa_pcb4). committed 결과는 두 개의 서로 다른 로스터로 존재 — copy-paste consolidated(aitex/severstal/mtd/leather)와 Table 12(carpet/leather/macaroni/fryum). setup↔results가 논문 내에서 서로 불일치.
2. **deficit-aware placement 반증**: 초안이 핵심 기여로 제시하나, deficit은 소스 가용성과 직교(deficit ⟂ availability) — 채울 수 없는 조합에만 높음. 유일 positive인 aitex는 선택 ROI deficit=0/200으로 이겼고, deficit 최다(25%)인 mtd는 null. copy-paste에서 구조적으로 무용.
3. **downstream 이득 = n=1**: 4종 중 aroma>random은 aitex_single(tiled) 1건뿐(Δ+0.097, t=4.51). severstal(평탄)·mtd(천장)·leather(천장) null. "superior downstream" 일반화 불가.
4. **정직성 위반 잔재**: abstract가 "제거했다"고 선언한 AUROC 0.923이 §5·§6에 생존 — 자기모순.
5. **frozen 위반 서술**: §4.1(coverage dominate)·§4.3(16셀 monotonic, gain∝complexity)·§4.2(FID −0.97)이 TBD/폐기 프로토콜인데도 본문·캡션이 단정.

### 재프레이밍 방향 (3축)
"성능 개선"을 기여 축에서 내리고, 결과와 독립적으로 성립하는 방법론 + 정직한 negative로 재구성:

- **일차 기여 = 방법론**: MCI/CCI 복잡도 통계로 ROI 정책을 per-dataset 재engineering 없이 자동 선택하는 프레임워크(CASDA 수작업 대비 확장성). 결과와 무관하게 성립.
- **결과 = 정직한 조건부/부정**: 다중 도메인 downstream에서 aroma≈random이 지배적. 이득은 "측정 정상 + baseline 헤드룸 + 호환 배치 가능"(aitex) 조건에서만. 천장 데이터셋은 random 포화.
- **메커니즘 발견(negative를 기여로 승격)**: deficit-aware copy-paste의 구조적 한계(deficit ⟂ 소스 가용성) — 왜 rare-combo 타겟팅이 copy-paste에서 작동 못하는가. 이것이 가장 방어 가능한 강한 기여.

---

## 2. 섹션별 조치표

| 섹션 (line) | 문제 | 조치 | 사유 / 근거 |
|---|---|---|---|
| **Abstract (L2)** | 이미 [RESULTS FROZEN]. deficit-aware가 파이프라인 selling point로 나열 | **유지 + 연화** | 방법론 서술 정합. deficit-aware를 중립적 'placement stage'로 격하, 결과 문장을 조건부/negative 예고로. **수치 되살리기 금지**(FROZEN 유지) |
| **§1 기여 bullet 3 (L17)** | "deficit-aware가 coverage를 random보다 개선" 핵심 기여 | **삭제 후 대체** | aitex_positive_reverification: deficit ↔ 승패 무관. '메커니즘 발견(deficit⟂availability)' negative 기여로 대체 |
| **§1 기여 bullet 4 (L18)** | "superior downstream detection performance across four datasets" | **수정(조건부화)** | consolidated: positive n=1(aitex). 'superior' 삭제, 'architecture-dependent'도 삭제(16셀 run 미존재) |
| **§1 본문 (L13)** | "optimizes placement through deficit-aware sampling" | **연화** | '구현 구성요소 + 한계 forward-ref'로. '최적화/개선' 성능 어휘 제거 |
| **§2.4 CASDA (L31)** | CASDA 대비 확장성 | **유지** | 방법론 확장성 주장 성립. 단 성능 우위 함의 인접 제거 |
| **§2.5 Copy-Paste (L33-34)** | copy-paste 구조적 한계(deficit⟂소스, placement-blind) 미언급 | **보강** | 분야 공통 한계로 중립 서술 추가 → §4 negative·ControlNet 동기와 연결 |
| **§2.6 (L37)** | "generalize across heterogeneous datasets" | **유지** | 방법론 확장성. 인접 성능 암시만 제거 |
| **§3.1 + Table 1 (L42-51)** | 구버전 4종 로스터, Train/Test 수치 | **삭제 후 교체 + 저자 결정 플래그** | committed 로스터 2종 불일치(consolidated vs Table 12). 저자가 canonical 셋 확정 필요. 수치 TBD(창작 금지) |
| **§3.1 조건정의 (L53)** | AUROC/one-class 문맥 잔재 | **정합** | §4.4 supervised YOLO/mAP/3-seed와 정렬 |
| **§3.2 파이프라인 (L56,61)** | Figure1 캡션 'deficit-aware placement' 자동 모듈 | **연화** | 중립적 'placement optimization stage'로. 인과 주장 제거 |
| **§3.2 Figure 캡션 (L74,79,89)** | 'real statistics (24 reports)' 단언 | **저자 확인 게이팅** | complexity_report.json/policy JSON이 canonical 셋에 실존·정합 확인 후 게재. MCI/CCI 특정 수치 본문 삽입 금지 |
| **§3.2.6 Stage 3 (L112-113)** | deficit-aware를 핵심 이점으로 제시 | **연화 + 한계 명시** | 구현 사실 유지, '설계했으나 구조적 무용 발견'으로 전환. driver=ctx_prior+quality 명시 |
| **§3.3.1 Split (L124)** | 구버전명, single-seed | **수정** | 확정 4종 split + 3-seed(mean±std) 정정 |
| **§3.3.2 AD 아키텍처 (L127)** | 4개 one-class AD | **재정렬/이관 서술** | §4.3 폐기됨. supervised YOLOv8로 재정렬 또는 'confound로 이관' 정직 서술. Table 11 refill 금지 |
| **§4.1 ROI Quality (L151,156)** | 'AROMA dominates all four datasets' 단정 | **삭제 → TBD** | Table 8 frozen. coverage 우위 반증(저다양성·단일클러스터). placement 실측(4.7/67/78%) 인용 |
| **§4.2 Synthesis (L177,184)** | FID −0.97/+0.05 구체 수치, 구버전명 | **삭제** | Table 9/10 frozen. 정성 논거(픽셀 동일→FID 둔감)만 유지, 결론부는 §4.4 mAP로 |
| **§4.3 Downstream AUROC (L189-223)** | 16셀 monotonic, gain∝complexity, 4 Figure | **전면 삭제 → §4.4로 대체** | Table 11 frozen(유일 실측 셀 AROMA 패). gain∝complexity 반증(driver=헤드룸). AD AUROC refill 금지 |
| **§4.4 Table 12 (L233-241)** | 'AROMA≈Random' 결론 (정직) | **유지 + 로스터 정합화** | 결론 정직. 단 Table 12(carpet/macaroni/fryum) vs consolidated(aitex/severstal/mtd) 별개 파일 — **한 표에 섞지 말 것**. 저자 canonical 확정 |
| **§4.4 aitex positive** | 논문에서 누락 | **별도 행/표로 추가** | 유일 방어가능 positive. 단 n=1·tiled 특수성·confound 즉시 병기(cherry-pick 금지) |
| **§4.4 Table 13 Severstal (L243-252)** | 'placement 공간 커서 AROMA 가치 드러날 것' 예단 | **PENDING 유지 + 예단 삭제** | copy-paste severstal 이중 null. ControlNet 부족조합 충전 미검증, aitex 세장형 AR 부적합 명시 |
| **§5 Discussion (L322)** | **"average AUROC 0.923"**, deficit coverage 우위 | **즉시 삭제 + 재작성** | 정직성 정면 위반(abstract가 제거 선언한 값). aroma≈random 지배로 재작성 |
| **§5 (L324)** | 'anticipated to reveal' architecture 민감도 | **삭제** | 미래시제 추정. AD 4-아키텍처 run 미존재 |
| **§5 (L326)** | MCI/CCI ρ≥0.95, pruning=exhaustive 검증 | **TBD/연화** | committed 근거 미확인. 설계 서술만 유지, 'validated' 단정 제거 |
| **§5 Limitations (L328)** | single-seed(42) | **정정 + 4대 한계 추가** | 실제 3-seed. n=1/deficit무작동/천장null/confound/tiled 추가 |
| **§6 Conclusion (L331)** | **"AUROC 0.923", "outperforms both", four datasets 우월** | **삭제 후 재작성** | 정직성 위반 + 반증. 3축 결론으로 |
| **전역 Figures (L156,203,210,221)** | dominate/monotonic/complexity 상관 단정 | **캡션 단정 삭제** | TBD/폐기 프로토콜. gain_vs_complexity는 인과 반증→삭제 |
| **References (L354)** | copy-paste/synthetic-eval 관행 인용 부족 | **선택적 보강(검증 후만)** | 실존 확인된 문헌만. 창작 금지 — 미확인 시 skip |

---

## 3. 재작성된 핵심 섹션 (영어, 논문 문체)

> 아래 재작성문의 모든 정량 결과는 committed 파일에서 refill하도록 placeholder 처리했습니다. 정성적 방향(aroma≈random 지배, aitex 조건부)만 서술합니다.

### (a) Abstract (rewritten)

> Anomaly detection in industrial Automated Visual Inspection (AVI) is constrained by severe data sparsity and class imbalance, as defect samples are rare and morphologically heterogeneous. Prior context-aware synthesis frameworks such as CASDA mitigate this scarcity by encoding defect–background relationships, but rely on a manually constructed compatibility matrix and domain-specific rules that must be re-engineered for each new dataset, limiting scalability. **We propose AROMA (Adaptive ROI-based Morphology-Aware Augmentation), an eight-stage framework whose primary contribution is methodological: it automatically selects ROI modeling policies from dataset complexity statistics, eliminating the per-dataset re-engineering required by handcrafted approaches.** AROMA computes a Morphology Complexity Index (MCI) and a Context Complexity Index (CCI), which a Meta Policy Generator translates into clustering and placement policies through distribution diagnostics, candidate pruning, and empirical evaluation. **Alongside this framework, we report two honest empirical findings that qualify when synthetic ROI selection helps. First, in multi-domain downstream detection, AROMA is largely statistically indistinguishable from random ROI placement; a measurable benefit appears only on a single dataset that combines valid measurement, ample baseline headroom, and physically feasible context-matched placement, while datasets whose detectors are already near-ceiling converge with random augmentation. Second, we show that the deficit-aware, rare-combination targeting motivating much of this line of work is structurally inert under copy-paste synthesis, because the combinations with the highest deficit are precisely those for which no source patch exists to copy (deficit is orthogonal to source availability). We frame this negative result as a mechanistic contribution explaining why rare-combination targeting cannot operate without a generative engine.** [RESULTS FROZEN — quantitative results are reported only in §4.4 from committed result files; prior projected numbers, including a previously stated "Image AUROC of 0.923," were unbacked and have been permanently removed.]

### (b) Contributions list (rewritten)

> The principal contributions of this work are:
>
> - **A data-driven complexity policy framework.** We introduce MCI and CCI statistics that, through a Meta Policy Generator, automatically determine ROI modeling and placement policies, eliminating the per-dataset handcrafted compatibility matrix and morphological rules required by prior context-aware frameworks such as CASDA. This contribution is a design/scalability advance and is independent of downstream accuracy: the same pipeline is applied across heterogeneous datasets without per-domain re-engineering.
> - **An auditable policy-selection mechanism.** We design a Meta Policy Generator that selects among candidate clustering and context-partitioning policies through distribution diagnostics, complexity-ranked candidate pruning, and empirical policy evaluation, making the otherwise opaque policy decision reproducible from measured statistics.
> - **A mechanistic negative result on deficit-aware copy-paste.** We empirically demonstrate that deficit-aware, rare-combination targeting is structurally inert under copy-paste synthesis: the highest-deficit (morphology, context) combinations are those with no real source crop to copy, so deficit is orthogonal to source availability. We show that the single dataset on which AROMA outperformed random placement did so with zero deficit in its selected ROIs, whereas the dataset with the highest selected deficit showed no benefit. This clarifies why rare-combination targeting requires a generative engine rather than patch reuse.
> - **An honest, conditional downstream evaluation.** We evaluate AROMA across multiple industrial datasets and report, without cherry-picking, that AROMA and random ROI placement are statistically indistinguishable in the dominant case; a downstream benefit is observed only under the conjunction of valid measurement, baseline headroom, and feasible context-matched placement (a single dataset, n=1), and we analyze the confounds and ceiling effects that bound this result.

### (c) §3.1 Experimental Setup — dataset paragraph (rewritten, 확정 4종 + 저자 결정 플래그)

> AROMA was evaluated on publicly available industrial anomaly detection datasets selected to span a broad range of morphological and contextual complexity, so that the automatic policy-selection mechanism is exercised across distinct complexity regimes rather than a single visual regime. The evaluated datasets are **Severstal** (full-frame steel surface with multiple defect classes and large foreground extent), **MVTec Leather** (an organic surface texture), **AITeX** (a directional, periodic woven-fabric texture, evaluated under a tiled protocol; see below), and **MTD** (machined magnetic-tile ceramic surface). Together they cover metal, organic, textile, and machined-ceramic surfaces, providing a heterogeneous test of the framework's claim to generalize without per-dataset re-engineering. The per-dataset statistics are summarized in Table 1.
>
> **[AUTHOR DECISION REQUIRED — dataset roster reconciliation.]** The committed downstream results are currently split across two rosters that must be reconciled before publication: the consolidated copy-paste analysis reports Severstal / MVTec Leather / AITeX / MTD, whereas the object-centric copy-paste table (Table 12) reports MVTec Carpet / MVTec Leather / VisA Macaroni / VisA Fryum from a separate committed run. §3.1 must name exactly the datasets that appear in the reported results tables; the setup and results rosters cannot disagree. Table 1 counts are to be filled only from the corresponding committed `complexity_report.json` for each finalized dataset. **No dataset statistic is entered without a committed source; unavailable values remain TBD.**

**Table 1.** Dataset statistics and domain characteristics. `[FROZEN — refill from committed complexity_report.json per finalized roster; no fabricated counts.]`

| Dataset | Surface / domain | Train (normal) | Test (good) | Test (defect) |
|---------|------------------|----------------|-------------|---------------|
| TBD | Steel (full-frame, multi-class) | TBD | TBD | TBD |
| TBD | Leather (organic texture) | TBD | TBD | TBD |
| TBD | Textile (directional/periodic, tiled) | TBD | TBD | TBD |
| TBD | Magnetic tile (machined ceramic) | TBD | TBD | TBD |

### (d) §3.2.6 Stage 3 — Deficit-Aware Placement, honest transition (rewritten)

> **3.2.6 Stage 3: Placement Optimization and the Limits of Deficit-Aware Targeting**
>
> Placement of synthetic defects was scored over the Cartesian product of ROI candidates and seed variants. The pipeline originally incorporated a deficit-aware sampling component intended to prioritize underrepresented (morphology, context) pairs and thereby steer synthesis toward rare combinations. During evaluation, however, we found this component to be **structurally inert under copy-paste synthesis**, and we report this finding rather than presenting deficit-aware sampling as a source of gain.
>
> The reason is a fundamental property of patch reuse: the (morphology, context) combinations with the highest deficit are, by construction, those that are rarely observed in the real data — and therefore those for which no source crop exists to copy. Deficit is thus orthogonal to source availability: high-deficit bins are unfillable, and fillable bins have low deficit. Concretely, in the profiled datasets the highest-deficit bins contained no candidate crops `[committed: deficit_actionability_diagnosis — leather 0 actionable of 134 non-zero deficit bins; mtd 40 of 430, max ≈0.051]`, so the deficit term made no difference to selection. Consistent with this, the only dataset on which AROMA outperformed random placement did so with **zero deficit among its selected ROIs**, whereas the dataset with the largest selected deficit showed no benefit `[committed: aitex_positive_reverification — selected deficit>0: aitex 0/200, mtd 50/200]`. The signals that actually dominated the selection there were contextual compatibility (`ctx_prior`) and per-candidate realism (`quality_score`), not deficit `[committed: aitex_positive_reverification — ctx_prior +0.427, quality_score +0.30 from candidate to selected median]`.
>
> Accordingly, the deficit term contributes no measurable effect under copy-paste, and its intended value — filling rare combinations with sources that do not otherwise exist — could only be realized in combination with a generative engine (e.g., ControlNet), a direction we discuss in §5. Defect composites are formed by Poisson blending onto the assigned background (Stage 4).

### (e) New Limitations paragraph (rewritten)

> **Limitations.** Several limitations bound the interpretation of our results and are stated explicitly to forestall over-generalization.
>
> First, **the sole downstream benefit is a single-dataset observation (n=1).** Among the evaluated datasets, AROMA outperformed random ROI placement only on the tiled AITeX configuration `[committed: exp4v2 consolidated — Δ(AROMA−Random) ≈ +0.097 mAP@0.5, paired t ≈ 4.51, 3 seeds]`; the remaining datasets showed no significant difference. A single positive cannot support a claim that AROMA improves detection in general.
>
> Second, **the deficit-aware mechanism does not drive downstream gains and is refuted as a source of benefit under copy-paste** (§3.2.6): deficit is orthogonal to source availability, so rare-combination targeting cannot operate through patch reuse.
>
> Third, **near-ceiling datasets saturate under random augmentation.** On datasets whose baseline detection is already high `[committed: consolidated — MTD baseline ≈0.92, MVTec Leather ≈0.83]`, random synthetic augmentation already reaches the achievable range, leaving no headroom for ROI selection to exploit; the observed benefit is confined to a dataset with substantial baseline headroom `[committed: AITeX baseline ≈0.372]`.
>
> Fourth, **the AITeX positive is confounded between selection strategy and candidate pool.** We have not yet isolated whether the gain arises from AROMA's ROI *selection* policy or from the composition of the candidate pool it draws from; a symmetric-filter control (applying the same compatibility/quality top-filter to the random arm) is required before attributing the gain to the selection strategy. The result may also reflect the specific tiled, single-class AITeX configuration and may not transfer to other domains.
>
> Fifth, **placement is only physically feasible on a subset of datasets.** Current placement is placement-blind at the pixel level, and measured clean-background coverage of in-matrix context cells is highly uneven `[committed: placement_aware_score_redesign — leather 4.7%, mtd 67.2%, aitex 77.8%]`, so context-matched placement is realizable primarily on AITeX; on leather, exact-match placement falls back to legacy placement in the great majority of cases. For full-frame datasets such as Severstal, where foreground occupies a large fraction of the frame, clean-background gating is not meaningful and coverage was not measured, so Severstal placement claims are withheld pending separate verification.
>
> Finally, all downstream results reported here use copy-paste synthesis over 3 seeds (mean ± standard deviation, paired testing); the ControlNet-based generative arm is in progress and its ability to actually fill deficit combinations is unverified, with the elongated aspect ratios of AITeX defects posing a known difficulty.

### (f) Honest Conclusion (rewritten)

> **6. Conclusion.** We presented AROMA, a framework that automatically selects ROI modeling and placement policies from dataset complexity statistics (MCI and CCI) via a Meta Policy Generator, removing the per-domain handcrafted re-engineering required by prior context-aware synthesis such as CASDA. The principal contribution is this **methodological framework for automatic, per-dataset policy selection**, which holds independently of downstream accuracy and was applied across heterogeneous industrial surfaces without per-dataset re-engineering.
>
> Our empirical findings are reported honestly and conditionally rather than as a uniform performance improvement. In multi-domain downstream detection, **AROMA and random ROI placement are statistically indistinguishable in the dominant case**; a measurable benefit was observed only on a single dataset (tiled AITeX, n=1) that combined valid measurement, ample baseline headroom, and feasible context-matched placement, while near-ceiling datasets converged with random augmentation `[committed: exp4v2 consolidated]`. We further contribute a **mechanistic negative result**: deficit-aware, rare-combination targeting is structurally inert under copy-paste synthesis because high-deficit combinations lack any real source patch to copy (deficit ⟂ source availability). Rather than a demonstration that AROMA improves detection, this work therefore delimits **when and why ROI selection helps or fails** for patch-based synthesis, and motivates coupling data-driven placement with a generative engine as the direction in which rare-combination targeting could become effective. Future work should reconcile the evaluation roster, run the symmetric-filter control to separate selection from pool effects, and complete the generative arm.

---

## 4. 금지 확인 (되살리면 안 되는 수치/주장)

**절대 부활 금지 (정직성 제약 정면 위반):**

1. **"Image AUROC of 0.923"** (모든 형태·모든 섹션) — abstract가 제거 선언한 미검증/창작 수치. §5 L322·§6 L331에 잔존 중 → 삭제 필수.
2. **"AROMA > Random > Baseline" 일반 확인** — 반증됨(4종 중 positive 1건, n=1).
3. **"superior downstream detection performance across four datasets"** — breadth 과장. positive는 aitex 1종.
4. **16-cell monotonic AROMA ≥ Random > Baseline** (§4.3 L198,210) — 폐기 프로토콜, 유일 실측 셀에서 AROMA 패(0.4996 vs 0.7646).
5. **gain ∝ MCI/CCI (양의 상관)** (§4.3 L216, fig:gain_vs_complexity) — 인과 반증. driver는 complexity가 아니라 헤드룸+compatibility.
6. **"deficit-aware가 coverage/rare-pair를 개선" / "deficit이 작동"** — 반증(deficit⟂availability, aitex deficit 0으로 승).
7. **"AROMA dominates Random on every metric, all four datasets"** (§4.1 L151,156) — Table 8 frozen + coverage 우위 반증(저다양성·단일클러스터).
8. **FID delta 구체 수치 (−0.97, +0.05)** (§4.2 L177,184) — Table 9/10 frozen(미계산).
9. **MCI/CCI ablation ρ≥0.95, pruning=exhaustive 동등** (§5 L326) — committed 근거 미확인. 확인 전 부활 금지.
10. **구버전 데이터셋명** (isp_LSM_1/mvtec_cable/visa_cashew/visa_pcb4) — committed 결과 없음.
11. **4개 one-class AD 아키텍처 AUROC 표 (Table 11) refill** — frozen 지시. AD AUROC로 채우지 말 것(exp4v2 mAP만).
12. **"single seed (seed=42)" 한계** — 실제 3-seed. 낡은 서술.
13. **"Severstal에서 placement 가치가 드러날 것"** (§4.4 L241) — 미검증 예단. copy-paste severstal 이중 null.
14. **'architecture-dependent gains'** (§1 L18, §5 L324) — 16셀 4-아키텍처 AUROC run 미존재.

**허용 (committed 경로 병기 시):** aitex_single Δ+0.097 / t≈4.51 (§4.4 본문 한정, n=1·tiled·confound 즉시 병기 필수 — abstract 헤드라인 승격 금지); clean-patch coverage 4.7/67.2/77.8% (placement 실현가능성 제약 인용); ctx_prior+0.427 / quality_score+0.30 (driver 분석).

**저자 결정 필요 플래그:** (1) canonical 데이터셋 로스터 확정 — consolidated(aitex/severstal/mtd/leather) vs Table 12(carpet/leather/macaroni/fryum)는 별개 파일이므로 한 표에 섞지 말 것; (2) Figure의 complexity_report.json(24)·policy JSON이 확정 셋에 실존·정합하는지 확인 후 게재, 미확인 시 TBD/frozen; (3) References 보강은 실존 검증된 문헌만(창작 시 데스크 리젝).

**참조한 committed 근거 파일 (절대경로):**
- `D:/project/aroma/AROMA연구분석/exp4v2_copypaste_consolidated_20260707.md`
- `D:/project/aroma/AROMA연구분석/aitex_positive_reverification_20260708.md`
- `D:/project/aroma/AROMA연구분석/placement_aware_score_redesign_20260708.md`
- `D:/project/aroma/AROMA연구분석/deficit_actionability_diagnosis_20260707.md`
- `D:/project/aroma/AROMA연구분석/Article/AROMA.txt` (개선 대상 초안)