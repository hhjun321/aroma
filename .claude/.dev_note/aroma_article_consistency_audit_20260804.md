# Article/text 정합 감사 — `ring_sgm` 검증 결과 대조 (2026-08-04)

## (성격: 감사 결과 · **미착수**. 항목별 원문 인용·제안 문장 포함)

`ring_sgm` + `k_fit` 개정과 실제 산출물 검증(`aroma_adjacent_context_bg_selection.md` §3-5)에 비춰
`AROMA연구분석/Article/text/*.txt` 에서 고쳐야 할 지점을 전수 감사한 결과다.

---

## 산출 방법

Workflow 병렬 감사 — **13 에이전트 · 21분 · 토큰 1.37M · 검증 통과 63건 → 실행 항목 36건**

| 단계 | 구성 |
|---|---|
| **Read** (6 병렬) | 영역별로 dev_note 2건 + 담당 section 을 읽고 finding 추출. 각 finding 은 **파일 + 원문 verbatim 인용 + 문제 + 제안 + status + severity** 필수. 인용 없는 finding 은 폐기 |
| **Verify** (6, effort=high) | 영역별 **적대적 검증**. 기본값 REFUTED. 4항 판정 — (1) 인용이 파일에 verbatim 존재하나 (2) 근거가 dev_note 에 실제로 있나 (3) 필수 제약 위반인가 (4) 종합 |
| **Synthesize** (1, effort=high) | 중복 제거 · 충돌 병기 · 우선순위 · already_applied 분리 · 결과대기 분리 |

영역 분할: `core-3.2.4` / `method-rest` / `exp-3.3` / `results-4` / `front` / `back`

pipeline 구조 — 영역별로 Read 끝나는 즉시 Verify 가 붙는다(배리어 없음).

재실행: `Workflow({scriptPath: '.../article-consistency-vs-ringsgm-wf_4358a0e2-458.js', resumeFromRunId: 'wf_4358a0e2-458'})`
에이전트별 원 반환값: 해당 run 의 `journal.jsonl`

---

## 감사에 부과한 필수 제약 (위반 제안은 자동 REFUTED)

| # | 제약 | 이유 |
|---|---|---|
| 1 | 다운스트림 mAP 관련 **새 주장 금지** | exp4v2 미수행 |
| 2 | 실측 수치 **새로 박기 금지** (단 *기존 수치가 개정으로 틀리게 된 경우는 지적*) | 이론 서술만 손대는 단계 |
| 3 | 밝기/intensity 특징 추가 제안 금지 | 텍스처 전용 연구 |
| 4 | §3.2.3 Table 3 · `background_type` **삭제 재제안 금지** | 2026-08-03 삭제 후 **되돌린** 결정. `background_type` 이 `roi_selection` 의 `quality_proxy → matching_score → apply_quality_gate` 로 실제 소비됨 |

---

## 착수 전 미결정 5건

1. **착수 순서** — 파일별(§3.2 → §3.3 → §4 → front/back) vs 성격별(사실오류 → 기전귀속 → 누락)
2. **충돌 C1** — `MATCHING_RULES` 하드코딩 서술을 지금 고칠지, `--min_quality` 실제값 확인 후로 미룰지
3. **충돌 C2** — §3.3.4 통제 주장을 통제 강화(재실행)로 갈지 서술 완화로 갈지
4. **#5 Table 5** — rep 수를 지금 박을지, §4 수치를 만든 실행분이 확정될 때까지 §3.3.5 참조로 걸어둘지 (**감사는 후자 권고**)
5. **#7 Figure 4.1-2** — 개정 규칙으로 재생성할지, "구 top-K 스캔 산출"로 명시·격하할지

---

## 관련 문서

- `.claude/.dev_note/aroma_adjacent_context_bg_selection.md` — 개정 근거·검증 결과 **정본**
- `.claude/.dev_note/aroma_paper_gaps_placement.md` — 논문 갱신 트랙. §2 반영 6건 / §3 미반영 5건. **본 감사와 중복 항목 대조 필요**
- `AROMA연구분석/Article/figure/FIGURE_TABLE_WORKFLOW.md` — figure/table 작업 규칙 (#4 콜아웃 오름차순, #7 재생성 시 3단계 준수)
- `AROMA연구분석/Article/figure/FIGURE_NAMING.md` — figure 파일명·섹션 매핑

---

## 요약

| 우선순위 | 건수 |
|---|---|
| 수정 필요 — high | 15 |
| 수정 필요 — medium | 10 |
| 수정 필요 — low | 3 |
| 신설 필요 | 8 (high 1 / medium 2 / low 5) |
| 충돌 — 판단 필요 | 2 |
| 이미 반영됨 (확인 완료) | 12 |
| 결과 대기 (범위 밖) | 6 |
| **실행 항목 합계** | **36** |

| 파일 | 관여 건수 (그룹 항목은 파일마다 중복 계상) |
|---|---|
| `section3_2.txt` | 17 |
| `section3_3.txt` | 5 |
| `section5.txt` | 5 |
| `section4_1.txt` | 4 |
| `section4_2.txt` | 2 |
| `section4_3.txt` | 2 |
| `section6.txt` | 2 |
| `Abstract.txt` | 2 (+충돌 1) |
| `Introduction.txt` | 2 |
| `Reference.txt` | 2 |
| `coverletter(aroma).md` | 2 |
| `section2.txt` | 1 |

---

## 수정 필요 — high

### 1. section3_2.txt §3.2.3 첫 문장 — 후보 자리 열거 기제가 논문 안에 두 개 있다
- **현재**: "Regions of interest (ROIs) were extracted from normal images to identify plausible defect placement locations: Otsu thresholding produces a foreground mask, connected-component analysis yields candidate ROIs, and components below a minimum connected-component area are discarded as noise."
- **문제**: 개정된 §3.2.4 는 후보 자리를 64px 격자 위 타일 오프셋 열거 + `valid(s)` 필터로 정의한다("A candidate position s is any tile offset at which the defect's bounding box fits"). §3.2.3 첫 문장은 같은 대상("plausible defect placement locations")을 Otsu+CC 가 열거한다고 서술해, 어느 것이 `s*` 를 낳는지 판정 불가. gaps note D1 이 "generate_defects:373-400 의 Otsu+CC 는 정상 이미지 최대 연결성분을 잡는 foreground constraint 이지 후보 위치 열거가 아니다"로 이미 기록.
- **제안**: Table 3·background_type 은 그대로 두고 목적절만 재정의 — "…to delimit the usable foreground region a paste must fall inside, and to characterise the texture surrounding it: …" + "The enumeration of candidate positions within that region is carried out on the 64-pixel tile grid at placement time (Section 3.2.4); the extraction here supplies the region and its texture category, not the positions themselves."
- **지적 영역**: method-rest

### 2. section3_2.txt §3.2.4 (line 103-105) — 'no hand-set constants' 가 거짓
- **현재**: "Both priors are derived directly from the profiling statistics, so the ranking introduces no hand-set constants"
- **문제**: 바로 위 줄의 `ROI_score = 0.6 · ctx_prior + 0.4 · morph_prior` 에서 0.6/0.4 는 `roi_selection.py:1327 / 1349 / 1551` 에 리터럴로 3회 박힌 하드코딩이다. 두 prior 는 프로파일링 유래지만 혼합 가중치는 손으로 정한 값. 같은 절에서 배경 cue 가중치는 λ_j 로 자동 유도한다고 서술해 대비가 두드러진다. gaps note D4 가 같은 지적("가중치 0.6/0.4도 하드코딩이다").
- **제안**: "…the two mixing weights are fixed at 0.6 and 0.4 for every dataset, so the ranking introduces no dataset-specific tuning even though the mixture itself is not derived." 로 축소. (Abstract 의 `per-dataset` 한정어와 정합 — 항목 A-2 참조.)
- **지적 영역**: core-3.2.4

### 3. section3_2.txt §3.2.6 첫 문단 마지막 — void 거부 기제 두 개의 적용 범위 미분리
- **현재**: "the same score also rejects flat/void foreground regions (dark or bright) that would otherwise be mistaken for a real defect object."
- **문제**: §3.2.6 에는 (1) quality≥0.7 점수(밝기 항 포함)와 (2) 텍스처 에너지 기반 오프라인 게이트가 함께 서술되는데 역할 분담이 없다. 개정 후 자리 admissibility 를 실제로 결정하는 것은 (2)뿐이고, (2)는 텍스처 전용이라 밝고 매끈한 이상을 원리적으로 못 잡는다(dev_note 위험 5, §3-4 육안 실사례). 현행 문장은 "dark or bright"까지 걸러진다는 능력을 배치 단계로 확장 오독시킨다.
- **제안**: 밝기 특징 추가가 아니라 범위 분리 — 첫 문단 끝에 "This score operates on the pool of candidate background patches; it is a pool-level pre-filter and does not decide where a defect may be placed." 추가, 둘째 문단 도입부에 "Placement admissibility is decided instead by the texture-energy void test described below, … discriminates flat, edge-free tiles rather than tiles of a particular intensity." 추가.
- **지적 영역**: method-rest

### 4. section3_2.txt §3.2.4 (line 132) — 'Figure 7' dangling reference
- **현재**: "(As shown in Figure 7, the row-normalized values are stored as matrix_symmetric)"
- **문제**: 논문은 절-국소 번호 체계(3.2.4-1 등)를 쓰고 'Figure 7' 은 어느 section 문서에도 없다(구 평면 번호 잔재). 내용상 Figure 3.2.4-1 을 가리키므로 같은 그림에 대한 콜아웃이 line 132(잘못된 번호, 먼저 등장)와 line 156(정상)에 두 번 있다. FIGURE_TABLE_WORKFLOW.md §6.2 최초 콜아웃 오름차순 규칙 위반.
- **제안**: 괄호 콜아웃 삭제, 사실 서술만 남김 — "… the row-normalized values are what the profiling stage stores as matrix_symmetric." 최초 콜아웃은 line 156 하나로 유지.
- **지적 영역**: core-3.2.4

### 5. section3_3.txt §3.3.1 Table 5 'Synth per Condition' 열 — 균일 600 이 틀렸다
- **현재**: "| Severstal | Steel | 5902 | 3620 | 4131 / 1771 | 600 | 4 |"
- **문제**: 5행 전부 600 인데 실제 pool 은 `top_k × n_per_roi` 이고 top_k 가 데이터셋별로 다르다(`step3_execute.md:96` severstal 1000 / 나머지 200). 2026-08-04 검증 실행의 annotations 총량은 severstal 2000 / 나머지 400(dev_note §3-5(a)). `synth_pool_sizing.md` §1 당시 관측값(severstal 400 / mtd 400 / aitex 400 / leather 600)과도 어긋나 — **어느 실행에서도 5종 모두 600 인 적이 없다.** 게다가 exp4v2 는 `cap = real_train × synth_ratio` 로 subsample 하므로 '학습 파티션에 추가된 개수'가 애초에 이 값이 아니다.
- **제안**: 열 이름을 'Synth Pool (top_K × reps)' 로 바꾸고 top_K 를 데이터셋별 명기(실행 설정값), 본문에 "The synthesis pool is top_K selected ROIs × a per-ROI repetition count shared by both augmented arms; the number actually added to the training partition is capped at real_train × synth_ratio…" 추가. **rep 수는 §4.2/4.3 수치를 만든 실행분이 확정될 때까지 표에 박지 말고 §3.3.5 참조로 걸어둔다**(결과 대기 항목 참조).
- **지적 영역**: exp-3.3

### 6. section3_3.txt §3.3.1 조건 (ii) — random arm 정의가 실제 arm 과 다르다
- **현재**: "(ii) random-ROI: synthetic defects placed uniformly at random locations; (iii) AROMA: synthetic defects placed by the data-driven compatibility-aware policy (Section 3.2.4)"
- **문제**: `generate_random.py` 는 (a) `roi_candidates.json` 에서 무랭킹 균일 표본으로 top_k 를 재선택해 `roi_selected.json` 을 덮어쓰고, (b) `random_placement=True` 기본으로 placement grounding 전부 우회, (c) `step5_execute.md` STEP 4 가 clean-bg 게이트 인자를 넘기지 않아 게이트 미적용. 즉 ROI 선택·자리·게이트 3개가 동시에 다르다. 조건 이름('random-ROI')과 설명('placed uniformly at random locations')이 서로 다른 것을 가리키고, 같은 파일 §3.3.5 의 정확한 서술 및 §4.1 의 세 arm 구분과도 어긋난다.
- **제안**: "(ii) random: a deliberately naive copy-paste baseline that re-samples its ROI set uniformly from the same candidate pool without compatibility ranking and pastes each defect at a uniformly random position, with no foreground constraint and no placement or clean-background gating; this is the ungated re-selected arm of Section 4.1." + 조건 이름을 'random (naive)' 로 통일.
- **지적 영역**: exp-3.3

### 7. section4_1.txt Figure 4.1-2 캡션·직후 문단 — 구방식 함수를 호출한 figure 를 §3.2.4 예시로 제시
- **현재**: "Figure 4.1-2 illustrates this behavior qualitatively by directly invoking the same compatibility-scan and uniform-random placement functions on one fixed representative image per dataset"
- **문제**: 이 figure 가 호출하는 함수는 구방식 `_positive_place`(전역 32px stride → footprint 평균 → top-8 무작위)다. 캡션의 'overlay five identical-size candidate boxes scored by compatibility' 가 top-K 샘플 구조를 드러내며, `s* = argmax_{s : valid(s)} score(k, s)` 단일해를 규정한 §3.2.4 와 정면 충돌. dev_note §3-1(b)(평균→분포 매칭 개선 5/5, footprint→링 단독 악화 4/5), §3-4(두 방식 자리 일치 1/30 = 3%)가 두 방식이 다른 산출을 낸다고 실측.
- **제안**: (a) 개정 규칙으로 재생성 — 'the single ring-matched argmax position s* with its footprint F(s*) and ring R(s*)' 로 캡션 재작성, 'five identical-size candidate boxes' 삭제. (b) 재생성 불가 시 예시용 구성물로 격하 — "…generated with the runtime top-K compatibility scan rather than the offline ring-matching rule of Section 3.2.4" 를 붙이고 'the same compatibility-scan … functions' 표현 제거.
- **지적 영역**: results-4

### 8. section4_1.txt Figure 4.1-2 직후 문단 — void 회피를 compatibility 점수에 오귀속
- **현재**: "The compatibility-scored candidates concentrate away from void or texture-degenerate regions, whereas the uniformly placed candidates disregard local context and land on such regions."
- **문제**: 그림이 실제 호출한 구방식에서는 반증됐다 — dev_note §3-5(b) void 침범률 severstal old 20.4% vs `rand` 7.5%(구방식이 무작위보다 2.7배 나쁨), kolektor old 22.2% vs 20.9%(사실상 동일). 개정 후 참이 되지만 그 이유는 점수가 아니라 `valid(s)` footprint void 배제와 오프라인 게이트(§3-5(c) 주장 2: 런타임 `_is_clean_background` fail-open 을 ring 경로가 메웠다).
- **제안**: 인과를 admissibility 로 재귀속 — "The AROMA panel's positions lie inside observed, non-void texture because the admissibility condition valid(s) of Section 3.2.4 discards any candidate whose footprint touches a void or unobserved tile before scoring, whereas the uniformly placed candidates are subject to no such constraint." 'compatibility-scored candidates concentrate away from void' 형태의 인과 서술 삭제.
- **지적 영역**: results-4

### 9. section4_1.txt Figure 4.1-3 문단 — Δ·p 가 3-cue 실행 산출인데 §3.3.4 는 4-cue 를 규정
- **현재**: "AROMA's assigned backgrounds score higher than Random's on four of the five datasets (AITeX: Δ = +0.071, p = 1.7×10⁻¹¹; Severstal: Δ = +0.032, p = 3.1×10⁻²³; Kolektor: Δ = +0.015, p = 1.2×10⁻⁴; MVTec Leather: Δ = +0.079, p = 2.8×10⁻⁴²), with MTD the exception (Δ = +0.005, not significant, p = 0.25)"
- **문제**: 이 값은 배경 cue 3개(source/class/size) 실행분이다. 개정으로 §3.2.4 ②·§3.3.4 가 4-cue 로 재서술됐고("the four profiling cues of Section 3.2.4"), dev_note §3-3 은 w_k 가 5/5 에서 가중치를 벌었음을 실측(kolektor 0.247, severstal 0.219, aitex 0.166) — 배경 할당이 바뀌면 hist∩ 분포와 Δ·p 가 모두 바뀐다. **'개정으로 기존 논문 수치가 틀리게 된' 경우**에 해당.
- **제안**: (a) 4-cue 배경 할당으로 Figure 4.1-3 재산출 후 값 교체. (b) 재산출 전이면 조건절 삽입 — "Measured on the background assignment produced by the source, class, and size cues; the morphology-cluster cue u_mor was introduced after this measurement and its effect on the assignment is not reflected here." 새 수치를 창작하지 않는다.
- **지적 영역**: results-4

### 10. section4_3.txt §4.3 MTD 문단 — null 결과를 'near-ceiling' 단일 원인으로 귀속
- **현재**: "AROMA's Fray (0.9499) is −0.75 pp below Random (0.9574), suggesting that context-aware placement offers no benefit in the near-ceiling regime."
- **문제**: MTD 는 기전 자체가 상당 비율 미적용이다 — dev_note 위험 4(64px 격자 절단으로 mtd 면적 24.5%, max 54.9%, 결함 56/388 은 100% 격자 밖), 위험 4b(build/runtime 격자 불일치 → 런타임 조회가 중립 0.5, "mtd·kolektor 결과 해석 시 감안"). regime 단일 귀속은 이 기하 한계를 은폐한다.
- **제안**: 후보 원인 병렬로 — "Two explanations are not separated by this experiment: the near-ceiling baseline leaves little headroom for any placement policy, and MTD's tile geometry is the least favourable of the roster — a substantial share of its defects falls outside the 64-pixel profiling grid…" **주의**: 18.9% 폴백률은 ring 실행분 통계이고 Table 9 는 런타임 경로 산출분이므로, 폴백 문구의 근거는 위험 4·4b(격자 절단·중립 0.5)로 쓰고 비율 수치는 넣지 않는다.
- **지적 영역**: results-4

### 11. section4_3.txt §4.3 결론 + Introduction 기여 2번 — 기전 귀속을 arm 단위로 되돌려야 한다
- **현재 (section4_3.txt)**: "the pattern confirms context-aware placement operates conditionally on both sufficient headroom and informative surface heterogeneity."
- **현재 (Introduction.txt)**: "The same engine and synthesis budget are applied to both the AROMA and random arms, making the comparison a controlled ablation of the compat-ibility gate."
- **문제**: 두 문장 모두 AROMA−Random 격차를 compatibility 랭킹 단일 성분에 귀속한다. 그런데 같은 논문 §3.3.5 가 "the Random arm is a deliberately naive baseline with no placement gating, so the comparison measures the full data-driven placement framework rather than any single component" 라고 명시 — 내부 충돌이다. 게다가 dev_note §3-1(ctx_prior severstal −0.270 / mtd −0.119 / kolektor −0.053), §3-2("ctx_prior 가 5종 중 4종에서 chance보다 나쁘다")가 랭킹 단독 기여를 반증하고, 관찰된 이득은 void 거부 게이트 몫(§3-5(c) 주장 2)일 가능성이 높다. gaps note D5 와 동일 지적.
- **제안**: §4.3 — "the full data-driven placement framework — compatibility ranking, background assignment, and the void/quality gates together — helps conditionally … Because the Random arm is ungated (§3.3.5), these results do not isolate the contribution of the compatibility ranking from that of the gates". Introduction — "a controlled ablation of the full data-driven placement policy — background assignment, admissibility, and compatibility-based positioning — against unguided placement" + §3.3.5 참조. 기여 2번의 "so that the measured effect is attributable to context-aware placement rather than to gener-ative-model quality" 는 생성 품질 배제 주장으로 유효하므로 유지.
- **지적 영역**: results-4, front

### 12. section5.txt 세번째 한계 — 커버리지→'배치 신뢰 불가' 추론이 개정으로 성립하지 않는다
- **현재**: "Third, context-matched placement is constrained by clean-background feasibility: measured clean-background coverage of in-matrix context cells is highly uneven across datasets (AITeX 77.8%, MTD 67.2%, MVTec Leather 4.7%), so realistic ROI placement is reliable only on AITeX"
- **문제**: 개정 후 가용성은 커버리지가 아니라 footprint 유효성으로 정의된다("valid(s) ⟺ ∀ t ∈ F(s) : t is observed and not void"). q_k 는 지지집합 밖 0 이므로 미관측 셀은 교집합 기여 0 일 뿐 자리가 배제되지 않는다(중립 0.5 대체 소멸). 실측도 반대다 — leather(커버리지 4.7%)는 폴백 0%로 전량 확정, mtd(67.2%)가 폴백 최악(dev_note §3-3, §3-5(a)). 수치 77.8/67.2/4.7 자체는 프로파일링 통계로 유효하나 결론이 틀리게 됐다.
- **제안**: '가용성' → '판별력' 재서술 — "Third, the sharpness of context matching is bounded by clean-background coverage of the cells on which q_k is supported: where clean backgrounds rarely supply those cells (measured coverage AITeX 77.8%, MTD 67.2%, MVTec Leather 4.7%), the attainable value of ∩(h_s, q_k) is compressed and the score separates candidate positions less, even though admissibility itself is governed by footprint validity (Section 3.2.6)." 'reliable only on AITeX' 절은 삭제 또는 'offer a narrower dynamic range for the matching score' 로 완화.
- **지적 영역**: back

### 13. section5.txt 첫 문단 — 존재하지 않는 구성요소 'compatibility threshold'
- **현재**: "the gate's individual components (the compatibility threshold and the clean-background gate) are not separately ablated"
- **문제**: 개정된 §3.2.4 에는 임계가 없다 — 배치는 `s* = argmax_{s : valid(s)} score(k, s)` 이고 게이트 역할은 footprint void 배제(§3.2.6)가 맡는다. §3.2.4 전체(99–166행)에 'threshold' 가 한 번도 나오지 않는다(§3.2.6 quality ≥ 0.7 은 배경 패치 게이트로 별개).
- **제안**: "the gate's individual components (the ring-matching placement rule of Section 3.2.4, the four-cue background assignment, and the footprint void rejection of Section 3.2.6) are not separately ablated in the downstream comparison" — 'threshold' 표현 제거.
- **지적 영역**: back

### 14. Abstract.txt / section5.txt / section6.txt / coverletter(aroma).md — 구방식 산출 mAP 수치의 귀속 한정 (4곳 묶음)
- **현재 (Abstract)**: "Performance gains are consistently observed on datasets sat-isfying both conditions (AITeX and Severstal), whereas the advantage disappears on datasets with near-ceiling baseline accuracy"
- **현재 (section5.txt 2문단)**: "Under the training-free copy-paste engine (see §§4.2–4.3), AROMA's context-aware ROI selection yields a consistent positive direction over random placement on the high-headroom datasets (AITeX +4.96 pp, Severstal +1.06 pp mAP@0.5)"
- **현재 (section6.txt 2문단)**: "Under a training-free copy-paste engine, AROMA consistently outperformed random ROI selection on datasets with both sufficient baseline headroom and high context complexity, including AITeX (+4.96 pp) and Severstal (+1.06 pp mAP@0.5)."
- **현재 (coverletter)**: "Experimental results demonstrate that AROMA provides an effective and generalizable framework for industrial defect data augmentation."
- **문제**: 세 가지가 겹친다. (1) 이 수치는 개정 이전 배치 경로(전역 32px stride + top-8 무작위) 산출물이고, gaps note D3 이 "오프라인 ring 경로를 실제 실험에 쓰기 전까지 논문 서술과 구현이 어긋난 상태다"로 기록 — '방법 = 링 매칭 / 결과 = 구방식' vintage 불일치. (2) §6 과 coverletter 는 §5 자신의 한정("does not reach statistical significance (paired t, df=2)", "a directional trend rather than a proven improvement", leather −4.91 pp, "AROMA's downstream benefit is conditional … rather than a uniform improvement")보다 강하게 서술. (3) D3 추적표의 대상 절이 §3.2.4 ③ 뿐이어서 Abstract·Introduction 기여 3·4번이 재실행 시 함께 갱신되지 않는다.
- **제안**: 수치는 그대로 두고 귀속만 한정. §5 — 문단 첫 문장 뒤에 "These runs were produced by the placement path in effect at the time of the experiment" 삽입(단, '…validated at the level of placement geometry only' 류 표현은 논문에 보고 근거가 없으므로 완화). §6 — 'consistently outperformed' → "showed a consistent positive direction … — not a statistically significant improvement at n=3 seeds — … under the placement path in effect at the time of those runs". coverletter — 방법 기여 우위 + 조건부 서술로 재작성('effective and generalizable' 삭제). Abstract 결과 3문장은 **지금 손대지 않고** gaps note §4 의 D3 행 '대상 절' 에 Abstract(결과 3문장)·Introduction(기여 3·4번)을 추가해 hold 등록. 어느 곳에도 'ring'/'distribution matching' 어휘를 끌어오지 않는다(링 배치로 이 gains 를 얻었다는 함의 방지).
- **지적 영역**: front, back

### 15. Abstract.txt / section5.txt / section6.txt / coverletter(aroma).md / Introduction.txt — 개정된 방법 요약 누락 (5곳 묶음)
- **현재 (Abstract)**: "learns a symmetric compatibility gate that scores placement locations without any per-dataset hand-tuning"
- **현재 (section5.txt 첫 문단)**: "a symmetric compatibility gate scores placement locations directly from each dataset's own profiling, removing the per-domain handcrafted compatibility matrix and morphological rules that CASDA required"
- **현재 (section6.txt)**: "AROMA computes a symmetric compatibility gate directly from patch-level defect–background co-occurrence statistics for each dataset, enabling context-aware placement without per-dataset handcrafting."
- **현재 (coverletter)**: "* Automatic compatibility modeling from patch-level defect–background co-occurrence statistics, eliminating handcrafted, per-dataset compatibility rules."
- **현재 (Introduction 기여 1번)**: "This eliminates domain-specific hand-tuning and provides a unified, data-driven placement mechanism applica-ble across diverse industrial datasets."
- **문제**: 개정의 두 축이 다섯 요약 서술 어디에도 없다 — (i) 호환성 행을 target context distribution q_k 로 소비해 링 히스토그램과 매칭("The compatibility row of a cluster is consumed as a **target context distribution** rather than as a per-tile score"), (ii) 형태 군집 cue 를 포함한 4-cue 배경 할당(가중치는 measured lift 유도). dev_note §3-1(b)는 **개선을 만드는 것이 바로 score 정의(평균→분포 매칭)** 임을 기여 분리로 확인했다(footprint→ring 단독은 악화 4/5). 또 §3.2.4 L138 "so background assignment and placement are driven by one distribution rather than two quantities" 라는 구조적 통일성이 요약에서 사라진다. Abstract·section6 은 7/27–28 이후 미수정.
- **제안**: 네 요약 문장을 §3.2.4 어휘('target context distribution', 'ring/neighbourhood', 'rather than averaging', 'measured discriminative lift')로 확장. Introduction 은 새 bullet 을 만들지 말고 기여 1번 안에 한 문장 흡수 — "The same learned distribution serves both stages of placement — it ranks which defect-free image a defect is composited onto and where within that image it is positioned…". coverletter 는 bullet 을 둘로 분리. **주의**: Introduction 제안문의 "the separate, hand-specified rules prior work used for each stage" 는 노트 미근거 확장이므로 CASDA 배치 규칙 한정으로 축소. 수치는 넣지 않는다.
- **지적 영역**: front, back

---

## 수정 필요 — medium

### 16. section3_2.txt §3.2.4 (line 144-146) — h_s 분모에 void 타일이 포함된다
- **현재**: "h_s(c) = | { t ∈ R(s) : cell(t) = c } | / | { t ∈ R(s) } |"
- **문제**: R(s) 는 'the eight-neighbour tiles of that rectangle' 로만 정의돼 void/미관측을 포함하므로 h_s 가 1로 합해지지 않고, void 이웃이 있는 자리는 score 가 기계적으로 깎인다. 바로 앞 산문은 'a normalised histogram over its non-void tiles' 라 수식과 산문이 어긋난다. 구현은 `ring = [grid[t] … if t in grid]` + `inv = 1.0 / len(ring)`(clean_bg_selection.py:604-607)로 살아남은 타일 수로만 정규화. 같은 절 h_g 는 올바르게 써 놨다.
- **제안**: 관측 링을 집합으로 도입 — "R̃(s) = { t ∈ R(s) : t is observed and not void }" 정의 후 "h_s(c) = | { t ∈ R̃(s) : cell(t) = c } | / | R̃(s) |", 산문은 'over the observed, non-void tiles of the ring, so that h_s is a distribution'. R̃(s) = ∅ 인 자리 폐기도 한 절 추가(`if not ring: continue`).
- **지적 영역**: core-3.2.4

### 17. section3_2.txt §3.2.4 (line 116-118) — u_siz 수식이 컴포지터 동작과 어긋난다
- **현재**: "u_siz(g) = min( 1, 0.95 · W_g / w, 0.95 · H_g / h ),"
- **문제**: `_scale_to_fit` 은 `if cw <= bw and ch <= bh: return 1.0` 로 들어가기만 하면 정확히 1.0 이고, `_FIT_MARGIN = 0.95` 는 축소 분기에만 곱한다. 논문 수식대로면 crop 이 한 변의 95% 를 넘게 차지하는 경우(왜곡 없이 들어가는데도) u_siz < 1 — severstal 처럼 결함이 한 변을 관통하는 데이터셋에서 실제 발생. 자기 산문('equal to one when the crop fits without distortion')과도 모순. 이 값은 F(s) 크기 산정에도 쓰여 배치까지 전파된다.
- **제안**: 조건부 정의로 — "u_siz(g) = 1 if w ≤ W_g and h ≤ H_g, and 0.95 · min( W_g / w, H_g / h ) otherwise," + "the 0.95 margin is the fixed safety factor the compositor applies when it must shrink a crop … and equals one whenever no rescale is needed".
- **지적 영역**: core-3.2.4

### 18. section3_2.txt §3.2.4 (line 128) + §3.2.5 첫 문장 — 자리 결정 시점을 selection 으로 통일
- **현재 (§3.2.4)**: "During composition, placement is governed by a symmetric compatibility gate that scores a defect morphology cluster k against a 64-pixel context cell c"
- **현재 (§3.2.5)**: "Each selected defect crop is placed at the designated bounding box on the target normal image."
- **문제**: 개정 후 자리는 배경 할당 단계에서 오프라인 확정되고 합성은 `forced_xy` 로 소비한다(dev_note §2-3: `generate_defects.py:1415-1421` 이 `_positive_place` 앞에서 단락, §3-5(a) 좌표 불일치 0/5종). 논문 §3.2.6 자신이 "Because placement is resolved at selection time (Section 3.2.4)" 라고 못박아 §3.2.4 의 'During composition' 과 정면 충돌 — 폐기된 런타임 탐색 프레이밍의 잔재. §3.2.5 의 'designated' 는 사실과 맞지만 그 bbox 가 §3.2.4 의 s* 라는 출처가 없어 합성이 자리를 다시 찾는다고 읽힐 여지가 남는다.
- **제안**: §3.2.4 — "Placement is resolved at selection time, before composition, by a symmetric compatibility gate that scores…"; line 140 도 "The position is then resolved on the same 64-pixel tile grid" 로 통일. §3.2.5 — "Each selected defect crop is placed at the bounding box fixed for it at selection time (position s* of Section 3.2.4); composition performs no further position search."
- **지적 영역**: core-3.2.4, method-rest

### 19. section3_2.txt §3.2.2 첫 문단 — 호환성 모델의 소비처가 배치 하나로만 예고된다
- **현재**: "These morphology clusters and context cells form the row and column indices of the symmetric compatibility model used for placement (Section 3.2.4); because both partitions are estimated from each dataset's own statistics, no dataset-specific constants are introduced."
- **문제**: 개정으로 호환성 행은 q_k 로 재정규화되어 두 곳에서 소비된다 — 자리 선택과 배경 랭킹 네번째 cue u_mor = ∩(h_g, q_k). 배경 랭킹의 세 히스토그램 cue 가 모두 이산 셀 공간 위에서 정의되므로 §3.2.2 의 셀 공간은 배경 선택 단계 전체의 정의역이다. 'used for placement' 만 적으면 u_mor 의 정의역이 §3.2.4 에서 갑자기 등장한다(§3.2.4 L138 이 사후 연결하고 있어 실질은 전방 참조 정합성 수준).
- **제안**: "…the symmetric compatibility model that drives both clean-background ranking and placement (Section 3.2.4)" + 선택적으로 "the same cell space also carries the histogram cues by which a clean background is ranked."
- **지적 영역**: method-rest

### 20. section3_2.txt §3.2.2 / section3_3.txt §3.3.3 / section4_1.txt §4.1 — ROI 선택 단계와 자리 선택 단계의 용어 혼용 (3곳 묶음)
- **현재 (§3.2.2)**: "each cluster k carries a prior P(k) = n_k / N, its share of the defect population, that serves as the morph_prior when ranking placements (Section 3.2.4)."
- **현재 (§3.3.3)**: "The ability of the ROI selection policy to explore the defect-placement space was evaluated by comparing two ROI sets generated from the same candidate pool: AROMA's compatibility-aware selection and a uniform-random selection (Section 3.2.4)."
- **현재 (§4.1)**: "three coverage statistics were computed over AROMA's final placements and an independently, uniformly re-selected Random set drawn from the same candidate pool, at equalized selection size: morphology coverage, context coverage, and rare-pair coverage"
- **문제**: 개정으로 두 단계가 분리됐는데 세 곳이 뒤섞여 있다. (a) 자리 점수 `score(k,s) = ∩(h_s, q_k)` 에는 morph_prior 항이 없다 — morph_prior 는 `ROI_score = 0.6·ctx_prior + 0.4·morph_prior` 한 곳뿐인데 §3.2.2 는 "when ranking placements" 라 쓴다. (b) §3.2.4 가 'per-tile score 가 아니라 target distribution' 을 **무조건형**으로 선언했으나 ROI 랭킹은 여전히 (k,c) 스칼라를 쓴다(`roi_selection.py:1327`, `:475-494`) — §3.3.3 coverage 지표는 그 스칼라 소비를 전제로 성립하므로 리뷰어가 '분포로 소비한다면서 왜 셀별 coverage 를 세는가'로 읽는다. (c) §4.1 coverage 는 선택 단계 통계인데 'final placements' 라 써서 링 개정으로 수치가 바뀌어야 하는 것처럼 오인된다(§3.3.3 은 "the fractions of morphology clusters and context cells represented in the selection" 으로 정의).
- **제안**: §3.2.2 — "when ranking ROI candidates (Section 3.2.4); the position within the chosen background is then decided by the compatibility row alone, without the cluster prior." §3.3.3 — "The compatibility model enters ROI ranking as a scalar score per defect–context pair, whereas placement consumes the same row as a target distribution (Section 3.2.4); the coverage comparison therefore concerns the former and is unaffected by the placement rule." §4.1 — 'final placements' → 'final ROI selection', line 8 도 'AROMA concentrates its ROI selection on compatible cells' + "These statistics characterise the ROI selection stage (§3.2.4 ①) and are independent of the placement rule…".
- **지적 영역**: method-rest, exp-3.3, results-4

### 21. section3_3.txt §3.3.4 마지막 두 문장 — 평가 서술자가 랭킹 기준 공간과 다르다(그리고 intensity 채널)
- **현재**: "Background-selection quality is evaluated for each ROI by computing the histogram intersection between the assigned background texture (intensity, gradient magnitude, and local variance) and a dataset-level reference histogram pooled from the background regions of real defect images."
- **문제**: (1) 4 cue 는 전부 이산 context-cell 공간의 ∩ 인데 평가는 픽셀 히스토그램(`[figure 4.1 3] bg_similarity_datasets.py:53` `_texture_desc`)이고 겹치는 축은 local variance 하나뿐. 실은 장점(자기참조 평가 회피)인데 논문이 말하지 않아, u_mor 추가 후 '기준과 지표가 어긋난다'는 지적을 받기 쉬운 형태가 됐다. (2) 서술자 첫 항이 intensity 인데 §3.2.1 은 "structural rather than intensity-based features dominate background differentiation" 이라 반대로 논증 — 텍스처 전용 프레이밍과 충돌.
- **제안**: "Background-selection quality is evaluated with a descriptor that is deliberately independent of the context-cell space the four cues optimise — a pixel-level texture histogram (gradient magnitude and local variance) — so the evaluation is not a restatement of the selection criterion." intensity 채널은 평가 전용 서술자에서 **제거**(권장), 유지 시 "the intensity channel enters the evaluation descriptor only, not AROMA's context features" 를 붙여 §3.2.1 과 봉합. (파이프라인에 특징 추가가 아니라 평가 서술자에서 빼는 방향.)
- **지적 영역**: exp-3.3

### 22. section3_3.txt §3.3.5 — 동수 보장의 기전과 게이트 재배치가 서술되지 않았다
- **현재**: "For each dataset, both augmented conditions (Random and AROMA) add the same number of synthetic defect images to the real training partition under an identical synthesis budget (see §3.3.1), so that the two conditions differ only in the placement mechanism and not in the amount of synthetic data."
- **문제**: 동수를 만드는 것은 arm 공유 파라미터(`step5_execute.md:157` `N_PER_ROI = 2` 주석: "AROMA arm(STEP 3/3B)과 random arm(STEP 4)이 같은 값을 써야 … 공정하다", 동일 top_k/seed)인데 'identical synthesis budget' 으로 뭉갠다. 그러면 두 질문에 답이 없다 — (i) ring 자리 산출 실패 ROI 에서 AROMA arm 개수가 줄지 않는가(줄지 않는다: `position=None` → `_positive_place` 폴백으로 이미지는 그대로), (ii) 'clean-background gates' 가 이제 어디서 걸리는가(개정 A5 로 void 거부가 선택 시점으로 이동, 런타임 `_is_clean_background` 는 fail-open).
- **제안**: (1) "…both arms share the same number of selected ROIs (top_K), the same per-ROI repetition count, and the same seed…" (2) "Because placement is resolved at selection time for the AROMA arm, the offline footprint-void rejection and ring void exclusion of Section 3.2.6 are part of this AROMA-only gating; a ROI for which no admissible position exists still contributes an image through the runtime placement search, so the gating never reduces the AROMA arm's image count below the Random arm's." 실측 폴백률은 넣지 않는다.
- **지적 영역**: exp-3.3

### 23. section4_2.txt Kolektor 문단 — parity 를 regime 단일 원인으로 확정 서술
- **현재**: "AROMA's performance parity with Random—despite its compatibility-aware design—indicates that in near-ceiling, low-headroom regimes (Kolektor; see also MTD in Section 4.3), the placement policy contributes less than baseline dataset characteristics."
- **문제**: '기전은 정상 작동했으나 regime 이 여지를 주지 않았다'를 'indicates' 로 확정한다. dev_note §3-2 kolektor `ctx_prior` 0.562 vs chance 0.535(chance 보다 나쁨), §3-5(b) 구방식 kolektor void 22.2% vs `rand` 20.9%. 자리 선택 기전이 무작위 대비 우위가 없었다는 독립 측정이 있으므로 대안 설명(랭킹 신호 부재)을 배제한 서술이 된다. 위험 10 은 kolektor ROI 52건으로 신뢰구간이 넓다고 기록.
- **제안**: "…is consistent with two readings that this experiment does not separate: a near-ceiling, low-headroom regime leaves little for any placement policy to recover, or the placement rule itself supplies little ranking signal on this surface." 새 수치는 넣지 않는다.
- **지적 영역**: results-4

### 24. section4_3.txt MVTec Leather 문단 — 배경 다양성 붕괴의 원인 오귀속
- **현재**: "on this near-uniform surface the similarity-ranked background assignment collapses onto a small background pool (15 distinct normal backgrounds across 400 AROMA composites, versus 202 for Random), so AROMA loses background diversity precisely where placement position carries no compatibility signal"
- **문제**: 현상은 개정 후에도 재현되나(dev_note §3-5(d) leather 16 vs `rand_arm` 202) 원인이 표면 균일성이 아니다 — 위험 7 "배경 재사용 — severstal distinct backgrounds 264/1000, 평균 3.8회. pool top-1만 쓰기 때문(`topk_pool`에 14+개 있음)". §3-5(d)는 kolektor 53 vs 249 로 near-uniform 아닌 데이터셋에서도 같은 붕괴를 보고. 또 '15 distinct' 는 3-cue 실행값(항목 9 와 동일한 조건 문제).
- **제안**: 원인을 랭킹 소비 구조로 이동 — "the similarity-ranked background assignment concentrates on the highest-ranked candidates of the pool, so a small set of normal backgrounds is reused across many composites (15 … versus 202 for Random, under the three-cue ranking of this run). This concentration is a property of consuming only the top of the ranked pool rather than of the leather surface itself; it is harmful here because the near-uniform surface offers no compensating compatibility signal…" 타 데이터셋 재사용 수치는 본문에 넣지 않는다.
- **지적 영역**: results-4

### 25. section4_2.txt AITeX 문단 — 배경 선택 / 자리 선택 단계 모호 ⚠ 제안 수정 필요
- **현재**: "AROMA's ability to identify and prioritize compatible background contexts within the tile geometry yields measurable gains."
- **문제**: 'compatible background contexts' 가 배경 이미지 선택(§3.2.4 ②)인지 자리의 국소 문맥(§3.2.4 ③)인지 구분되지 않는다. 개정으로 두 단계가 명확히 분리됐으므로(§3.3.4: "the localisation to the defect's immediate neighbourhood enters at the placement stage, not here.") 이 모호성이 기전 오귀속으로 읽힌다. 자리 선택 쪽 근거는 오히려 부정적이다 — dev_note §3-1 aitex `ctx_prior` −0.009±.046, §3-1(c) 표본 확대 시 +0.011 → −0.032 부호 반전, §3-2 0.638 vs chance 0.590.
- **제안**: 배경 할당 단계로 확정하되 **원안의 'largest measured advantage on AITeX' 는 삭제** — 같은 파일 §4.1 이 MVTec Leather Δ = +0.079 > AITeX +0.071 을 보고하므로 '최대치' 는 거짓이고 그대로 반영하면 논문에 허위 비교급을 새로 박는다. "AROMA's background-assignment stage, which ranks candidate normal tiles by their whole-image context-cell distributions (§3.2.4 ②), attains a significant advantage over Random on AITeX (Figure 4.1-3) … The placement-position rule is not isolated by this experiment." 수준으로.
- **지적 영역**: results-4

---

## 수정 필요 — low

### 26. section3_2.txt §3.2.6 — 'without a hand-set constant' 가 과장
- **현재**: "with the acceptance floor taken as a low percentile of each dataset's own observed distribution, so the criterion adapts per dataset without a hand-set constant."
- **문제**: 적응하는 것은 임계값이고 분위 수준 자체는 로스터 공통 상수다 — `clean_bg_selection.py:264` `floor_pct: float = 15.0`, `:265` 주석 "a RAISED low-percentile (default p15)", `:1229` `--void_floor_pct default=15.0`. 상대 분위 컷이라 void 가 없는 데이터셋에서도 하위 15% 가 잘린다(dev_note 위험 5). 'no hand-set constants' 를 반복 주장하는 맥락에서 코드를 보면 즉시 걸린다.
- **제안**: "…so the cutoff value adapts per dataset without a hand-set absolute energy threshold; the percentile level at which the floor is taken is a single setting shared across the roster, and being a relative cut it removes a low fraction of tiles even on datasets that contain no void region." (15 라는 값은 논문에 넣지 않는다.)
- **지적 영역**: method-rest

### 27. section3_2.txt §3.2.1 — CCI 성분이 Table 3 판정량이라는 함의
- **현재**: "These measurements provide the quantitative basis for the background categories used during ROI extraction (Section 3.2.3)."
- **문제**: §3.2.1 이 측정한 것은 CCI 4성분(TextureEntropy, ContextClusterCount, FreqComplexity, OrientVariance)이고 Table 3 의 4 descriptor 는 LocalVariance, GradOrientEntropy, AutocorrPeak, LBPEntropy 로 집합이 다르다. §3.2.3 자신이 "therefore distinct from the frequency-energy component of the CCI" 라고 구별한다. gaps note D1(라벨이 `--background_type` CLI 고정 상수)까지 겹치면 'quantitative basis' 가 실제보다 강하다.
- **제안**: Table 3 유지, 연결 강도만 완화 — "These measurements motivate the background categories used during ROI extraction (Section 3.2.3), whose own descriptors are defined there."
- **지적 영역**: method-rest

### 28. Reference.txt PROVISIONAL 블록 주석 — AROMA.txt 를 조건부 동기화 대상으로만 지정
- **현재**: "# 재정렬 시 함께 갱신할 파일: section3_2.txt, section3_3.txt, section5.txt, AROMA.txt"
- **문제**: `Article/AROMA.txt`(2026-07-28)는 개정 이전 병합본이다 — u_mor·q_k·footprint·ring·target context distribution·[36]–[41] 전부 0건, `:364` 'the compatibility threshold and the clean-background gate', `:374` 'consistently outperformed random ROI selection' 로 §5/§6 개정 전 상태. '레퍼런스 재정렬 시' 조건부 대상으로만 적혀 있어 실제 괴리(개정분 전량 미반영)가 드러나지 않고, §5/§6 을 위 항목대로 고치면 더 벌어진다.
- **제안**: "# AROMA.txt 는 2026-07-28 병합본으로 §3.2.3/§3.2.4 개정분과 [36]–[41]이 모두 미반영이다. 확정본은 text/*.txt 이며, AROMA.txt 는 제출 직전 text/ 로부터 재생성한다." 이후 §5/§6 수정은 section5/6.txt 에만 반영.
- **지적 영역**: back

---

## 신설 필요

### 29. [high] section3_2.txt §3.2.4 / §3.2.5 / section5.txt — admissible 공집합 폴백 경로가 전혀 서술되지 않았다 (3곳 묶음)
- **현재 (§3.2.4)**: "score(k, s) = ∩( h_s, q_k ),   s* = argmax_{ s : valid(s) } score(k, s)."
- **현재 (§3.2.6, 단언 측)**: "so no position that straddles an unusable region can be emitted"
- **현재 (§5 한계 목록)**: "**Limitations and Future Work.** Three limitations bound the interpretation of these results."
- **문제**: { s : valid(s) } 가 공집합일 수 있는데 동작이 어디에도 없다. `_best_ring_site` 는 (i) 격자 < footprint, (ii) 모든 footprint 가 void/결측, (iii) 링 전부 void 일 때 `None` 을 반환하고 호출부가 `position` 을 비워 컴포지터 자체 탐색으로 폴백한다(docstring: "Returns None when nothing qualifies → 호출부가 position 을 비워 두고 generate 가 기존 `_positive_place` 경로로 자연 폴백한다"). 비율이 무시할 수 없다 — dev_note §3-3 mtd 604/2,596(18.9%), leather 0.1%, §3-5(a) '위치없음' 열이 예측 폴백률과 일치. 원인은 위험 4 의 64px 격자 절단. 즉 방법 서술이 전 사례를 덮지 못하고, AROMA arm 일부는 논문이 서술한 규칙으로 배치되지 않는다.
- **제안**: 세 곳에 각각 한 문단. §3.2.4 (line 150 뒤, 신설) — "When no position satisfies valid(s) … the ring rule emits no position and the ROI falls back to the compositor's own placement search, so the rule is a preference over admissible positions rather than a hard requirement." §3.2.5 (신설) — "When a background admits no valid position for a crop … the selection stage emits no position and composition falls back to its own placement search for that crop." §5 — 한계를 'Three'→'Four' 로 늘리고 "Fourth, the footprint-validity condition is absolute, so on datasets whose defects are large relative to the tile grid a fraction of ROIs admits no valid position at all; those ROIs revert to the ungated placement path … largest where the 64-pixel grid truncates a substantial share of the frame." **세 곳 모두 백분율을 넣지 않는다.**
- **지적 영역**: core-3.2.4, method-rest, back

### 30. [medium] section3_2.txt §3.2.2 — observed / unobserved / void 술어가 정의 없이 쓰인다
- **현재**: "Background context is discretized into cells by per-feature tertile (P33/P66) binning of the profiled context features. These morphology clusters and context cells form the row and column indices of the symmetric compatibility model used for placement (Section 3.2.4)"
- **문제**: §3.2.4 의 admissibility("∀ t ∈ F(s) : t is observed and not void")와 ring 히스토그램이 이 세 술어에 의존하는데 §3.2.2 에 개념이 없다. 특히 'unobserved' 는 논문 전체에서 §3.2.6 line 178 단 1회 등장하고 정의절이 전무 — 프로파일링 격자가 W//64 × H//64 로 절단되어 우측·하단 나머지(최대 63px)에 셀이 없어 생기는 개념이다(dev_note 위험 4: mtd 면적 24.5%, max 54.9%, kolektor 13.2%; `defect_tiles.json` meta `grid_policy: "truncated (W//64 x H//64) — matches _context_worker"`). 정의 없는 술어가 개정 수식의 정의역을 제한하고 있다.
- **제안**: 셀 정의 문장 뒤 두 문장 — "The grid is truncated to whole tiles, so a right- or bottom-edge remainder narrower than one tile carries no context cell; such tiles are unobserved and are excluded wherever the model is queried." + "Tiles that are observed but carry no usable surface are labelled void by the criterion of Section 3.2.6."
- **지적 영역**: method-rest

### 31. [medium] section3_2.txt §3.2.2 세번째 문단 — 셀 공간의 희소성·미관측 셀 처리 미서술
- **현재**: "the five profiled context features and their P33/P66 tertile boundaries differ markedly across datasets, so the same discretization yields dataset-specific context cells"
- **문제**: 셀 공간은 5특징 × 3수준 곱공간이라 데이터셋별 관측 셀은 일부다. 개정 score 가 교집합이므로 이 희소성이 load-bearing — q_k 는 지지집합 밖 0 이고 미관측 셀은 0 을 기여한다(dev_note §2-1: "목표는 지지집합 S_k 밖에서 0 → 미관측 셀은 교집합 기여 0. 런타임 `_positive_place` 의 `.get(cell, 0.5)` 중립 처리와 다르며, neutral 0.5 편향이 자연 해소된다"). 논문은 셀 공간 크기·희소성·미관측 셀 처리를 한 번도 언급하지 않아 개정 수식이 왜 중립값 없이 잘 정의되는지 알 수 없다.
- **제안**: 문단 끝에 — "The cell space is the product of the per-feature tertile levels, of which only a minority is populated in any one dataset; a cell is defined only where the profiling actually observed it." + "A compatibility row therefore has support on the observed cells alone, so a cell absent from a cluster's profile contributes nothing to the intersection scores of Section 3.2.4 and needs no neutral default."
- **지적 영역**: method-rest

### 32. [low] section3_2.txt §3.2.4 (line 140) — 절단 격자와 rescale-후 크기 누락
- **현재**: "Placement then proceeds on the same 64-pixel tile grid at which the model was built. A candidate position s is any tile offset at which the defect's bounding box fits"
- **문제**: (1) 격자는 프로파일링이 emit 한 절단 격자다(`_tile_grid` docstring: "격자는 context_features.csv 가 emit 한 타일에서 그대로 읽는다 (절단 격자, F1)") — 변 길이가 64 배수가 아닌 데이터셋의 경계 밴드는 후보 공간에 아예 없는데 'any tile offset … fits' 는 전역 훑기로 읽힌다. (2) F(s) 크기는 원본이 아니라 fit-rescale 후 크기(`wh = _effective_wh(...)` → `bw = max(1, -(-int(wh[0]) // 64))`) — 이 정합이 §3-5(a) 좌표 불일치 0 을 만든 부분이라 서술 가치가 있다.
- **제안**: "…the whole-tile grid the profiling stage emits, so the sub-tile remainder at the right and bottom borders carries no context cell and is not enumerated. A candidate position s is any whole-tile offset at which the crop fits after the u_siz rescale, and it determines two tile sets: the footprint F(s), the tile-aligned rectangle the rescaled crop would cover, and the ring R(s), …"
- **지적 영역**: core-3.2.4

### 33. [low] section3_2.txt §3.2.4 (line 134-138) — q_k 의 지지집합 규약 명시
- **현재**: "q_k(c) = ctx_prior(k, c) / Σ_{c'} ctx_prior(k, c'),"
- **문제**: line 109 은 ∩(a,b) = Σ_c min(a(c), b(c)) 라고만 쓰고 c 의 범위·미관측 셀 처리를 밝히지 않아 독자가 0 으로 볼지 중립값으로 볼지 판단 못 한다. 구현은 `score = sum(min(v, tgt[c]) for c, v in hist.items() if c in tgt)` 로 정확히 0 기여. 이 규약이 개정의 이론적 이득 하나(중립 기본값 편향 제거)다.
- **제안**: line 138 뒤 — "The row is supported only on the cells actually observed with cluster k, so q_k(c) = 0 elsewhere and a ring tile whose cell never co-occurred with cluster k contributes exactly zero to the intersection; unobserved cells are neither rewarded nor assigned a neutral default value." (항목 31 과 함께 처리하면 중복 없이 봉합된다.)
- **지적 영역**: core-3.2.4

### 34. [low] section3_2.txt §3.2.4 (line 148-150) — argmax 동점 처리 미정의
- **현재**: "and the position is scored, and selected, by"
- **문제**: Figure 3.2.4-1 캡션 자신이 "on a surface such as Severstal, where the rows are nearly identical" / "share a dominant cell on the near-uniform Severstal surface" 라고 쓰는데, 그런 표면에서는 링이 지배 셀 하나로 채워진 자리가 다수 나와 score 가 정확히 같아진다. `_best_ring_site` 는 `if score > best` 엄격 비교 + 행-우선 순회이므로 동점은 항상 최상단-최좌측으로 해소된다 — argmax 표기가 감추는 계통 편향.
- **제안**: 수식 뒤 한 절 — "Ties are resolved deterministically by scan order, which matters on near-uniform surfaces where many admissible rings share the same cell composition and therefore the same score." (또는 시드 고정 균등 추출로 바꿀 경우 그 방식을 명시.)
- **지적 영역**: core-3.2.4

### 35. [low] section2.txt §2.5 — '무엇을 읽어야 하는가' 축이 related work 에 없다 ⚠ 근거 축소 필요
- **현재**: "To date, principled, data-driven learning of ROI placement policies has received little attention, leaving the choice of placement strategy largely heuristic."
- **문제**: §3.2.4 는 배치 규칙의 두 설계를 "Two properties of this rule are deliberate" 로 이론 정당화하는데, §2 는 배치 정책이 heuristic 이라는 공백만 지적하고 '덮이는 영역 vs 살아남는 경계' 축을 세우지 않아 그 논증이 related work 상 어디에 대응하는지 근거가 없다. §2.2 에 hook 이 이미 있다 — "achieving natural alignment between the pasted defect and the surrounding background texture, illumination, and geometry remains a challenging and largely unsolved problem [16]".
- **제안**: §2.5 마지막 문장 앞에 한 문장 신설. 단 **원안의 "existing compatibility criteria are typically evaluated on the region the defect will occupy" 는 선행연구 일반화 주장인데 [16] 도 dev_note 도 이를 뒷받침하지 않으므로 축소**해야 한다 — 예: 붙여넣기가 덮어쓰는 영역과 합성 후 남는 둘레가 구별된다는 사실 서술 + §2.2 [16] 참조까지만. 새 인용문헌은 만들지 않는다.
- **지적 영역**: front

### 36. [low] Reference.txt — 히스토그램 교집합의 정본 인용 부재
- **현재**: "# 미인용 잔여 항목(이번 범위 밖): Otsu thresholding(§3.2.3), BIC(§3.2.2)"
- **문제**: 개정으로 ∩ 가 방법의 핵심 연산자로 승격됐다 — §3.2.4 "All cues are histogram intersections over the discrete context-cell space" 및 배치 점수 `score(k, s) = ∩( h_s, q_k )`. 그런데 정본 출처(Swain & Ballard, Color Indexing, IJCV 1991)가 Reference.txt 에 없고 위 대기 목록에도 없다(`text/` 전체에 'Swain' 0건). 개정 전에는 cue 하나였으나 이제 배치 규칙 자체가 이 연산자다.
- **제안**: PROVISIONAL 블록에 "Swain, M. J.; Ballard, D. H. Color Indexing. International Journal of Computer Vision. 1991, 7(1), 11–32. DOI: 10.1007/BF00130487." 를 추가하고 ∩ 정의(line 109) 직전에 인용. 등재만 할 경우 주석을 "…, histogram intersection(§3.2.4)" 로 갱신.
- **지적 영역**: back

---

## 충돌 — 판단 필요

### C1. MATCHING_RULES(D2) 관련 서술을 지금 고칠지, `--min_quality` 확인 후로 미룰지
두 영역이 **반대 시점**을 제안한다. 임의 선택하지 않았다.

| | method-rest 안 | front 안 |
|---|---|---|
| 대상 인용 | section3_2.txt §3.2 서두: "Unlike CASDA, which relies on a handcrafted, do-main-specific compatibility matrix and morphological rules, AROMA re-derives its de-fect–background compatibility model directly from dataset statistics." | Abstract.txt: "auto-matically replacing the handcrafted compatibility matrices required by previous approaches" |
| 제안 | **지금** 주장 범위를 좁힌다 — "…the defect–background compatibility model **that governs placement** directly from dataset statistics", §3.2.4 의 "automatically replaces CASDA's handcrafted compatibility matrix" 도 "at the placement stage" 로 한정 | **본문은 지금 고치지 않는다**(D2 선결 확인 미완). gaps note §3 D2 및 §4 반영 계획의 '대상 절' 에 Abstract(4번째 문장)·Introduction(기여 1번)·section2 §2.6 을 추가해 blast radius 만 등재하고, `--min_quality > 0` 판정 후 세 곳을 동시 처리 |
| 공통 사실 | `utils/suitability.py` MATCHING_RULES 5×5 상수 + `W_MATCHING 0.4 / W_CONTINUITY 0.3 / W_STABILITY 0.2 / W_GRAM 0.1` 하드코딩이 `roi_selection.quality_proxy → matching_score → apply_quality_gate` 로 소비된다(gaps D2). `apply_quality_gate` 는 `min_quality ≤ 0` 이면 비활성 |
| 공통 제약 | 양측 모두 Table 3·background_type 삭제를 요구하지 않는다 |

판단 축: `--min_quality` 실값 확인 전에 §3.2 서두를 선제 완화할 것인가(안전하지만 게이트 비활성이면 불필요한 약화), 아니면 확인까지 현행 유지할 것인가(그동안 Abstract·Intro·§2.6 이 추적표에 없던 상태는 front 안이 해소).

### C2. section3_3.txt §3.3.4 통제 주장 — 통제 강화(재실행) vs 서술 완화 (exp-3.3, high)
- **현재**: "Both AROMA and Random use the identical ROI set and defect instances, differing only in the assignment of defect-free background images."
- **문제**: `--site_selection ring` 을 켜면 AROMA arm 은 `position`/`topk_positions` 를 채우고 `forced_xy` 분기가 `_positive_place` 를 단락하는데(dev_note §2-3, §3-5(a) 좌표 불일치 0), `random_arm()` 은 entry 에 `position`·`topk_positions` 를 **넣지 않아** 런타임 `_positive_place` 로 자리를 잡는다. 두 arm 이 배경 identity 뿐 아니라 배치 규칙까지 갈리므로 'differing only in …' 및 'share the same … synthesis pipeline' 이 성립하지 않는다.

| (a) 통제 강화 | (b) 서술 완화 |
|---|---|
| `random_arm` entry 에도 동일 ring 자리 산출을 적용(같은 q_k, 배경만 무작위)한 뒤 — "differing only in the identity of the assigned background image; both arms resolve the paste position by the same ring-matching rule of Section 3.2.4" | "differing in the assignment of defect-free background images; the random control additionally resolves its paste position by the runtime search rather than the offline ring rule, so this experiment **bounds** the background-assignment effect from above rather than isolating it" + §4.1 해석 문장도 함께 완화 |
| §3.3.4 의 목적(성분 격리)에 부합하나 재실행 필요 | 재실행 없이 즉시 반영 가능하나 성분 격리 주장을 포기 |

보조 사실: §3.3.4 가 보고하는 지표 자체는 배경 이미지 히스토그램이라 배치와 무관하므로, 실제 결함은 'same synthesis pipeline' 문구 정확성에 국한될 수 있다(검증 note).

---

## 이미 반영됨 (확인 완료)

재작업 금지 — 아래는 개정 설계·구현과 정합함을 인용 대조로 확인한 항목이다.

1. **section3_2.txt §3.2.4 ② 4 cue 열거** — "u_src(g) = ∩( h_g, p_src(v) ), / u_cls(g) = ∩( h_g, p_cls ), / u_mor(g) = ∩( h_g, q_k )," 가 dev_note §2-2 `k_fit`·λ_j 유도(`w = mean(best − median)`)·class≠cluster 근거(severstal 4 vs 5, leather 5 vs 3)와 일치. (u_siz 수식만 항목 17 로 별건.)  [core-3.2.4]
2. **section3_2.txt §3.2.4 target distribution 재정의** — "The compatibility row of a cluster is consumed as a **target context distribution** rather than as a per-tile score. Renormalising the row to sum to one — a monotone rescaling that leaves its relative shape unchanged — gives" 가 dev_note §2-1 주의사항까지 그대로 반영. 수식 √(P_def·P_clean)·ε·행 max 정규화·P_def 전역·k 조건화 전부 무수정 보존.  [core-3.2.4]
3. **section3_2.txt §3.2.4 valid(s)** — "valid(s) ⟺ ∀ t ∈ F(s) : t is observed and not void," 가 'footprint 에 void/결측 타일이 하나라도 있으면 자리 자체를 버린다' 를 수식 수준으로 반영, 'is observed' 로 결측까지 포함. 점수 계산 전 폐기임도 명시.  [core-3.2.4]
4. **section3_2.txt §3.2.4 두 설계 근거** — "Two properties of this rule are deliberate. First, the ring is read rather than the footprint, because the footprint is precisely the region the pasted defect overwrites; what survives composition, and what borders the defect after blending, is the ring." + 평균 argmax 가 '가장 평범한 표면' 으로 수렴하는 기전. 폐기된 런타임 기전(32px stride, top-8, τ)의 잔재 0건. ⚠ 향후 §4 기전 귀속 시 첫 근거를 성능 근거로 승격하지 말 것(§3-1(b): footprint→ring 단독은 4/5 악화).  [core-3.2.4]
5. **section3_2.txt Figure 3.2.4-1/-2/-3 캡션** — "Each row is consumed at placement time as a target distribution to be matched by the ring around a candidate position (Section 3.2.4), not as a per-tile score to be averaged" 외 3종 캡션이 본문·spec·생성 스크립트(`adm[0]`/`adm[len(adm)//2]` = best·median)·w_src 0.407~0.576(5/5 최대)·severstal w_size 0.0 vs mtd 0.069 와 정합. 남은 조치는 항목 4(Figure 7 콜아웃)뿐.  [core-3.2.4]
6. **section3_2.txt §3.2.6 두번째 문단 (A5)** — "Because placement is resolved at selection time (Section 3.2.4), void rejection is applied there as well … Void tiles are likewise excluded from the ring before its histogram is formed" 가 개정 설계와 정확히 일치(void 침범 severstal 20.4%→1.2%, kolektor 22.2%→0.0%). 잔여 후속은 'unobserved' 정의 신설(항목 30)뿐.  [method-rest]
7. **section3_3.txt §3.3.4 4 cue + 국소화 귀속** — "AROMA ranks candidate backgrounds by the four profiling cues of Section 3.2.4 — … together with size compatibility" 및 'the localisation … enters at the placement stage, not here' 가 A1·A6 요구를 충족, 부록 R1(전역 질의 MRR 6.7배)과도 정합.  [exp-3.3]
8. **section4_1.txt §4 서두 세 Random arm 구분** — "Random is instantiated differently per experiment an independently re-selected ROI set, the same ROIs with only background assignment randomized, or an ungated re-selected ROI set so a Random result in one figure does not carry over to another." 가 §3.3.3/§3.3.4/§3.3.5 및 dev_note 의 `rand`/`rand_arm` 구분과 1:1 대응. (편집상 'per experiment' 뒤 콜론 누락만 권고.)  [results-4]
9. **section4_1.txt Figure 4.1-3 설계 서술** — "To test the background-selection mechanism independently of the copy-paste engine, AROMA's ROI set is given a symmetric random-background control that keeps the same ROIs and randomizes only the background assignment (§3.3.4)" 의 방향성 결론이 §3-5(c) 주장 3(`rand_arm` 5/5 최악)으로 재확인. Δ·p 값의 cue 조건만 항목 9 로 별건. ⚠ `rand_arm` 은 배치 게이팅까지 없앤 프레임워크 수준 arm 이라 §3.3.4 의 배경-only 대조와 완전히 같지는 않다는 교란은 기록해 둘 것.  [results-4]
10. **Abstract.txt 'per-dataset' 한정어** — "without any per-dataset hand-tuning, auto-matically replacing the handcrafted compatibility matrices required by previous approaches" 는 개정 후에도 수정 불필요(4번째 cue 가중치도 lift 자동 산출에 편입, w_k 5/5 획득 · w_size 4/5 자동 0). ⚠ **한정어 'per-dataset' 를 삭제하거나 'no hand-set constants' 로 강화하지 말 것** — 전역 하드코딩(0.6/0.4, MATCHING_RULES)이 존재하므로 이 한정어가 문장을 참으로 유지하는 유일한 장치다. (다만 `--background_type` 이 데이터셋별 고정 상수라는 D1 기록이 있어 근거는 완전하지 않고 D2 선결 확인에 종속.)  [front]
11. **section2.txt §2.6** — "AROMA introduces a measure-then-derive mechanism: a profiling stage measures each dataset's per-feature defect-morphology and background-context distributions and computes a defect–background compatibility model directly from their patch-level co-occurrence" 는 유도 단계만 서술하므로 개정 전후 모두 참. 단 "In contrast to CASDA's handcrafted, single-domain compatibility matrix" 는 충돌 C1 의 D2 blast radius 목록에 등재 대상.  [front, back]
12. **Reference.txt [39]** — "[39] Haralick, R. M. Statistical and Structural Approaches to Texture. Proceedings of the IEEE. 1979, 67(5), 786–804. DOI: 10.1109/PROC.1979.11328." 는 고아가 아니다(section3_2.txt:52 AutocorrPeak 정의에서 인용, [36]:48 · [37][38]:52 · [40][41]:73 도 전부 인용 확인). 권고: §3.2.3 삭제→복원 이력이 있으므로 "[36]–[39]는 §3.2.3 배경 텍스처 지표 문단에 전적으로 의존 — 해당 문단 삭제 시 4건 동시 고아" 주석 한 줄 추가.  [back]

---

## 결과 대기 (범위 밖)

다운스트림(exp4v2) 미수행. 아래는 결과가 나와야 판단·확정 가능하다. **지금 mAP 관련 새 주장을 넣지 않는다.**

1. **Abstract 결과 3문장 + Introduction 기여 3·4번** — 수치 자체는 손대지 않고 gaps note §4 D3 행 '대상 절' 에 hold 등재만 (항목 14).
2. **Figure 4.1-3 Δ·p 4-cue 재산출** — 항목 9 의 선택지 (a). 재산출 전에는 조건절(b)로 봉합.
3. **Table 5 per-ROI repetition 수** — §4.2/4.3 수치를 만든 실행의 rep 수가 확정될 때까지 표에 박지 않고 §3.3.5 참조로 (항목 5).
4. **D2 `--min_quality` 실값 확인** — 충돌 C1 의 선결 조건. `> 0` 이면 MATCHING_RULES 가 후보를 실제로 거르므로 Abstract·Introduction 기여 1·§2.6·§3.2 서두·§3.2.4 를 동시 처리.
5. **D3 — ring 경로를 실험 기본으로 승격** — 승격 시 §3.2.4 ③ 서술↔구현 불일치가 해소되고 항목 14 의 vintage 한정이 불필요해진다.
6. **D5 — §4 기전 귀속 최종 판정** — 항목 8·10·11·23·24 는 지금 '단일 원인 귀속 완화' 까지만 가능. compatibility 랭킹 / 게이트 / 배경 할당의 기여 분해는 다운스트림 성분 실험이 있어야 확정된다. 폴백 비율·void 침범률 등 실측 수치는 본문에 넣지 않는다.