# Report1 리뷰어 답변서 작성 세션 — 답변 3건 + 민감도·entropy 실측 + 원고 수정 (2026-08-21)

## (성격: 세션 기록 · **완료분 + 잔여 작업 목록**. 답변서는 완성, 실험 E4·figure·재정렬은 미착수)

> **🔴 최종 번복 B안 (2026-08-21 후속 세션) — 아래 A안을 대체. Eq.2 = 무가중 합, §4.5 sweep 제거**
> R2-1·R3-3 공통 지적(0.6/0.4 hand-set constants)에 대해 **가중치 자체를 제거**하는 것으로 최종 결정:
> - **Eq.2 = `ROI_score = ctx_prior + morph_prior` (상수항 없음)**. 논리: §3.2.4 3단계 재편으로 compat model이 선택·배경할당·위치해상 모두에서 직접 소비되므로 선택 점수 내 priority weighting 불필요. §3.2.4 정당화 문단 재작성(항 필요성은 §4.4 ablation 인용).
> - **§4.5 가중치 sweep(구 Table 12) 통째 제거** — 방어할 상수가 없어 존재 이유 소멸. §4.5는 "Sensitivity of the Tertile Partition Boundaries"로 개제, tertile 섭동 표가 **Table 12** 승계. (아래 2b의 sweep 실측은 재심 대비 예비 데이터로 보존)
> - **E4 완료 및 반영**: YOLO11n 3-seed 결과(`.claude/.etc/exp4v2/20260821/yolo11/exp4v2_results_{1,2,42}.json`, casda arm은 오수행으로 제외) → §4.3 말미 "Cross-architecture validation (YOLO11n)" 소절 신설. **Table 13**(Severstal, Table 8 형식 per-class) + **Table 14**(AITeX, Table 6 형식 mAP@0.5 단일 열). Severstal AROMA 3-seed 전승(+1.48pp, c2/c4 집중), AITeX comparable(+0.97pp, seed 분산 이내). R2-2 placeholder 실수치로 채움.
> - 답변서 3건 모두 동기화: R1-1(2) "가중치 제거+tertile 민감도", R2-1(2)·R3-3(1,2) "removed rather than defended", R3-1(2) 인용 수치를 §4.5 실제 수치(3.1–14.5% relabel)로 정정.
> - **잔존 리스크 갱신**: 논문 무가중 vs production 코드 0.5/0.3/0.2 불일치는 여전(형태는 다르나 동일 지뢰). sweep 실측("값 비민감·항 참여 민감")이 완화 근거로 유효.
> - **잔여**: figure `[figure 3.2.4 1] roi_selection_flow*.py`(0.6/0.4 하드코딩) 수정·재생성, `[figure 4.5 1] weight_simplex*` retired 처리, section3_2_4_eng/kor.md 초안·AROMA.txt 조립본 미반영, SOP "0.6/0.4 유지 결정" 기록 갱신.
>
> **⚠️ (구) 최종 결정 A안 (2026-08-21 세션 말미, 위 B안으로 대체됨) — Eq.2는 2-항 표기 유지**: 세션 중 Eq.2를 3-성분(0.5/0.3/0.2)으로 전환했다가 **최종적으로 제출본 2-항(0.6·ctx + 0.4·morph) 표기로 회귀** (사용자 결정: 수식 단순화 취지 유지, 수식 변경 가시성 리스크 회피). 이에 따라:
> - §3.2.4 Eq.2 = 0.6/0.4 + fixed-design·priority-order(context>morphology) 이론 근거 (quality 서술 제거)
> - §4.5 = **2-항 비율 sweep** (`sensitivity_0604_results.json` 기준: 비율 0.1/0.9~0.9/0.1 → 87.5–100%, ctx-only 83.5–100%, morph-only kolektor 37.5% 붕괴), Table 13 = subtype 라벨 안정성만(재배정률; retention 열 제거 — 2-항 세계에선 Eq.2가 subtype 미소비)
> - 답변서 3건에서 "three-component 정정" 주장 전부 소거, 비율-inoperative 논증으로 대체
> - 아래 본문 §2·4의 3-성분 전환 서술은 **중간 이력**로 읽을 것. 3-성분 sweep 산출물(`sensitivity_results.json`)·임계 섭동 retention 수치는 보존 — 재심 대비 예비 데이터
> - **잔존 리스크 (SOP §4.5(a) 재부상, 저자 인지 하 수용)**: 논문 표기 0.6/0.4 vs production 코드 0.5/0.3/0.2(quality 포함) 불일치 유지. 코드 공개 시 노출 가능. 완화 근거: 비율 비민감 실측 + §4.4 스케일(전면 교체도 3.8pp)

Applied Sciences 1차 심사(Reviewer 1/2/3, Major Revision) 19개 항목에 대한 point-by-point 영문 답변서를
`대응계획_SOP.md` 기반으로 작성 완료. 답변에 필요한 정량 근거 2건(subtype 임계 섭동, entropy/Gini)을
로컬 CPU로 신규 실측했고, 답변이 약속하는 원고 수정을 같은 세션에서 반영했다.

---

## 1. 산출물 — 답변서 3건

| 파일 | 항목 수 | 핵심 전략 |
|---|---|---|
| `Article/report1/reviewer1_response.md` | 5 | R1-1 sensitivity+ablation 3중 방어 · R1-2 CCI 주장 강등(기전 증거 병기) · R1-3 통제변인 4단 사슬(D5) · R1-4 재측정 해명 · R1-5 federated 인용 수용 |
| `Article/report1/reviewer2_response.md` | 7 | R2-1 "no manual tuning" 범위 명확화(**양보 없음** — 사용자 확정: percentile 도출은 수동 튜닝 아님) · R2-2 YOLOv11n 보강(E4, 수치 TBD) · R2-3 임계 섭동 실측 · R2-4 CCI 4-성분 기해소 · R2-5 entropy/Gini 정량표 · R2-6 keywords · R2-7 최신 문헌 3편 |
| `Article/report1/reviewer3_response.md` | 7(5–7 통합) | R3-1 percentile≠hardcoded 논증 · R3-2 **§3.2.4 방법 개선 귀속** + 단조 배경 한계 명시(사용자 지시) · R3-3 강건성(≠최적성) 재프레임 · R3-4 figure 재생성 · R3-5~7 문헌 수용 |

답변서 공통 수치 출처: Tables 6–10(3-seed exp4v2), §4.4 Table 11(ablation), §4.5 Tables 12–13(이 세션 실측).

---

## 2. 신규 실측 ① — subtype 임계 섭동 (R2-3/R3-1 근거)

- 스크립트: `D:\project\AROMA_DATASET\roi_weight_sensitivity\subtype_threshold_perturbation.py`
- 결과: `subtype_perturbation_results.json`
- 설계: P33/P66 → ±5pt 8개 구성. 경로 = 임계 → `_percentile_subtype` 재배정 → quality=matching_score(subtype, "directional") → roi_score(0.5/0.3/0.2, round 6) → production allocator 재실행 → top-K overlap. `quality_proxy`·`_subtype_percentiles`·allocator를 `aroma.roi_selection`에서 직접 import (재구현 없음). 동질성 가드((p_hi−p_lo)<0.15·sd → fixed 폴백)는 섭동점에도 동일 적용.
- 결과: 단일 경계 ±5pt → 재배정 3.1–9.1% / overlap 67.5–98.0%. 양 경계 동시 → 7.7–14.5% / 64.0–94.5%. **비례적, 절벽 없음.** subtype은 0.2 가중 이산 quality 항({0.4, 0.7, 1.0})으로만 진입 → score 최대 이동 0.12로 구조적 유계.

### ⚠️ 핵심 발견 — 로컬 roi/ 미러는 fixed-임계 시대 판

`D:\project\AROMA_DATASET\roi\{ds}\roi_candidates.json`의 저장 subtype이 **fixed 임계(2.0/5.0, 0.7/0.9) 재계산과 0 mismatch**, percentile 재계산과 대량 mismatch(severstal 20k 중 12,084), `subtype_mode` 필드 부재. 즉 로컬 미러는 step3 percentile 전환(`--subtype_mode percentile`) **이전** 산출물. percentile 정본은 Drive에만 있다.

**대응**: percentile baseline을 로컬에서 **재구성** — 후보 풀(rows×bins)·ctx/morph prior는 subtype_mode와 무관하게 동일하고 allocator는 결정적이므로, 재구성 선택 = production percentile 선택과 동등. sanity: S0a(임계 함수 대조) · S0b(fixed 재계산 == 저장값, 미러 무결성) · S1(저장 점수 → allocator == roi_selected.json, allocator 충실도) 전 데이터셋 통과. fixed-era 선택과 percentile 재구성 선택의 overlap: severstal 42.7% / mtd 49.5% / leather 51.0% / aitex 73.0% / kolektor 87.5% (참고용 — 두 시대 선택이 크게 다름을 의미).

**후속 주의**: 로컬 미러 기반의 다른 분석(예: 기존 weight sensitivity의 S1)은 유효하나, 저장 subtype/quality를 그대로 쓰는 분석은 fixed-era임을 인지할 것. Drive 정본과의 byte 대조는 미수행.

## 2b. 신규 실측 — 0.6/0.4 기준 sweep (2026-08-21 추가, 사용자 지시)

- 스크립트: 같은 폴더 `sensitivity_0604_baseline.py` → `sensitivity_0604_results.json`
- 설계: 기준 선택을 제출본 표기(0.6·ctx+0.4·morph, quality 무)로 생성 후 (a) w_q=0 평면 ctx:morph 비율 0.1/0.9~0.9/0.1 (b) ±0.1 이웃 (c) 참조점 스윕.
- 결과: **(a) 비율은 무의미** — 0.9/0.1↔0.1/0.9 반전에도 87.5–100% 유지(aitex 100%·kolektor 98.5% 전 구간). **(b/c) quality 항 참여가 유일한 분기** — w_q=0.1만 넣어도 50–63%(kolektor 86.5 예외), production/등가중 동일 overlap. 기존(0.5/0.3/0.2 기준) 분석의 거울상 — "값 비민감·항 참여 민감" 결론이 기준 선택에 독립.
- 반영: §4.5 Table 12를 2-블록(값 변경=worst-case 2행 / 항 참여 변경 3행)으로 재구성 + 문단 재서술. R3-3 답변에 "0.6/0.4 비율 자체가 inoperative" 정면 문장 추가.

## 3. 신규 실측 ② — entropy/Gini 재계산 (R2-5/D6 근거)

- 스크립트: 같은 폴더 `entropy_gini_recompute.py` (exp2 `compute_metrics` 재사용, percentile 재구성 AROMA arm vs 동일 풀 균등 Random arm seed 42, equal budget)
- 결과: `entropy_gini_results.json` → 원고 §4.1 **Table 5b**
- **D6의 우려("균등성 지표는 random 근최적") 실측 반증**: AROMA가 4/5에서 동등 이상 — severstal 0.984/0.956·mtd 0.956/0.867·kolektor 0.883/0.802 (entropy A/R, gini도 A 우위), leather tie, **aitex만 열위**(0.815/0.859 — compat 집중의 의도적 trade, §4.2 최대 이득과 연결해 서술). 기제: 균등 Random은 풀의 클러스터 불균형을 상속, AROMA per-pair quota가 능동 평탄화.

---

## 4. 원고 수정 (이 세션 반영분)

| 파일 | 수정 |
|---|---|
| `section3_2.txt` | **Eq.2 → 3-성분 전환**: `ROI_score = 0.5·ctx + 0.3·morph + 0.2·quality` + fixed-design 문장 + §4.5 cross-ref. Figure 3.2.4-1/-3 캡션 동기화 (**figure 이미지 재생성 필요**). **§3.2.6 전체 삭제 (T4 원안 확정 — 사용자 직접 삭제, 2026-08-21)**: §3.2.4:143 "(Section 3.2.6)" cross-ref 제거, void admissibility 문장·figure 캡션 void 언급은 유지(D1). **Eq.2 quality 항 서술 정정 주의**: 처음 넣은 인라인(blur/contrast/brightness/noise photometric)은 삭제된 §3.2.6 배경 게이트 설명으로 **오기** — Eq.2의 quality 실체는 `matching_score(subtype, background_type)` = CASDA SuitabilityEvaluator의 subtype-적합성 이산 점수({0.4,0.7,1.0}; roi_selection.quality_proxy). 최종 서술 = "graded subtype-suitability score inherited unchanged from CASDA's ROI suitability evaluation [8]" (CASDA.txt:53 'matching quality' 성분과 대응 — 인용 안전). production background_type은 전 데이터셋 "directional"(step3 기본값)이므로 per-dataset 적응 주장 금지 |
| `section3_2_4_eng.md` / `_kor.md` | Eq.2 동기화 |
| `section2.txt` | §2.1 Stavropoulos[46]·one-class에 [3]·data decentralization 문단[43] / §2.3 AnomalyDiffusion[44]·RealNet[45] / §2.5 InstaBoost[42] + 설정 차이 문장 |
| `section3_1.txt` | Random-ROI = canonical uniform copy-paste [21,22] 명시 |
| `section4_1.txt` | **Table 5b** (entropy/Gini) + 해석 문단 |
| `section4_5.txt` | **신설** — §4.5 Sensitivity of the Fixed Design Choices: Table 12(성분-제거 3행만 — 사용자 지시로 이웃/interior 행 삭제, 본문 산문으로 대체), Table 13(임계 섭동), Figure 4.5-1 선언 |
| `section5.txt` | CCI n=5 limitation 문장 + 생성형 비교 범위밖 limitation 문단(ControlNet 후속 언급) |
| `Reference.txt` | [42] InstaBoost / [43] Yang KBS 2025 / [44] AnomalyDiffusion / [45] RealNet(DOI 미확정—조판 시) / [46] Stavropoulos — 임시번호, 등장순 재정렬 대기 |

**Eq.2 표기 지뢰 해소**: 논문 0.6/0.4(2-성분) vs 코드 0.5/0.3/0.2(3-성분) 불일치(consistency audit 2026-08-12 미해결 항목)를 **논문을 코드에 맞추는 방향**으로 종결 (사용자 확정). sensitivity 실측이 이 선택 지지: 0.6/0.4/0.0은 production 대비 severstal 35% 이탈.

**수치 정직성 정정 2건**: 메모리 기반 초안 수치를 실측 대조 후 정정 — "mean 98.7–100%" → 98.5–100%, "single-component 31–93% divergence" → "quality 제거 시 4/5에서 35–47%, zeroed 구성마다 최소 1개 데이터셋 34–66%".

## 5. 코드/execute 변경 (E4 준비)

- `scripts/aroma/experiments/exp4_v2_supervised_detection.py`: `ALL_MODEL_KEYS`·`--model` choices에 **`yolo11n`** 추가 (2줄 — 미추가 시 argparse 거부 + 집계 누락). ultralytics 명명 주의: `yolo11n.pt` (v 없음).
- `AROMA연구분석/colab_execute_new/exp4v2_execute.md`: **STEP 6 신설** — E4 detector-generality (yolo11n, severstal+aitex, 그룹 파라미터 그대로, 별도 `$EXP4V2_Y11` output, 판정 규칙 사전 등록: 순서 재현 여부만, 절대값 해석 금지, Δ 부호 뒤집힘 은폐 금지).

---

## 6. 잔여 작업

1. ~~**E4 Colab 실행** → placeholder 채우기 + detector-generality 소절 신설~~ **완료 (B안 배너 참조)** — §4.3 Tables 13–14, R2-2 실수치 반영. 단, YOLO11 casda arm `no_synth_annotations` 오류는 원인 미수정(비교군에서 제외로 종결)
2. **Figure**: 전면 재생성(R3-4 legibility 7–8pt)만 잔여. **Figure 4.5-1도 미채택(2026-08-21)** — §4.5는 Table 12 참조로 종결(0.6/0.4/0.0 행이 가중치 영향 전달), 생성본은 figure/_retired/ 보존. **Figure 3.2.4-3은 삭제 확정(2026-08-21)** — 근거: 구판은 row-normalized matrix_symmetric로 후보를 합성했으나 production 후보 ctx_prior는 raw 스케일(≈0.08대)이라 "ctx=1 saturation·compatibility dominates" 시각 서사가 채점 실체와 불일치 위험(코드 공개 시 대조 공격 표면), 실후보 재렌더는 top-N 동률 퇴화. 본문 해당 문단·캡션 삭제, 스크립트·png는 figure/_retired/ 이동. **Figure 3.2.4-4/-5 번호 재정렬 안 함** (조판 전역 재번호 전제, 작업 라벨 gap 유지 — F1에서 확인)
3. **keywords 교체** (제출 시스템): R2-6 확정 8개 — industrial visual inspection; defect detection; data augmentation; copy-paste synthesis; context-aware placement; defect–background compatibility; dataset complexity index; YOLOv8
4. ~~**Reference 등장순 재정렬**~~ **완료 (2026-08-21 후속 세션)** — 전 46건 등장순 재번호. 주요 이동: 구46(Stavropoulos)→9, 구43(Yang)→15, 구44(AnomalyDiffusion)→22, 구45(RealNet)→23, 구42(InstaBoost)→28, 구31(SAM)→40, 구32(YOLO)→43, 구33–35(FID/KID/LPIPS)→44–46; 구36–39(Shannon~Haralick)는 우연히 번호 불변. 본문 갱신: Introduction/section2/3_1/3_2/3_3/5. 검증: 등장순 == 1..46 단조 확인. **AROMA.txt 조립본은 미갱신**(구 본문 기준이라 재조립 시 text/ 정본에서 반영). 답변서 3건에는 번호 인용 없음 확인. 미인용 잔여 후보(Otsu, BIC)는 Reference.txt 말미 주석 유지
5. **F1 최종 검수**: Abstract–§6 정합 감사 + 삼중 대조. 확인 포인트 — **Eq.2 무가중 표기 전파(B안)** + 표 번호 최종 확인(§4.4 T11 → §4.5 T12 → §4.3 말미 T13–14), §4 도입부의 §4.4/4.5 안내, **Eq.4 삭제 후 수식 번호**(답변서가 "Eq.(1)–(3) 불변, Eq.(4) 뒤 번호 수식 없음"이라 단정 — 조판본에서 실제 번호 매김 확인 필수), RealNet DOI 확정, AROMA.txt(조립본)에 §3.2.6 잔존 — 재조립 시 text/ 정본 반영

## 7. 관련 파일

- 답변서: `Article/report1/reviewer{1,2,3}_response.md`
- 분석: `D:\project\AROMA_DATASET\roi_weight_sensitivity\{subtype_threshold_perturbation, entropy_gini_recompute}.py` + 결과 json 2건
- 계획 정본: `Article/report1/대응계획_SOP.md` (T4 §3.2.6 삭제 — 세션 중 일시적으로 유지+공개인정으로 갔다가 **최종 T4 원안 복귀**: 사용자 직접 삭제. 답변서 R2-1(3)/R3-3(4)는 삭제 경위 문안으로 교체됨 — F1 2-1 침묵삭제 금지 이행)
- 메모리: `memory/project_report1_responses.md`
