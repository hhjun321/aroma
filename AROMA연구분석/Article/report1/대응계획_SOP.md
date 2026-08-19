# AROMA 논문 Report1 대응계획 — 인덱스 및 공통작업 SOP

- 작성일: 2026-08-19
- 대상: Applied Sciences 1차 심사 결과 (Reviewer 1/2/3, Major Revision)
- 원본: `reviewer1.txt` ~ `reviewer3.txt` (번역본: `*_kor.txt`)

## 문서 구성

본 문서가 단일 수행 지점이다 — 공통 실험(E1–E5)·공통 텍스트 작업(T1–T7)·최종 검수(F1). (리뷰어별 분리 SOP는 2026-08-19 폐기·본 문서로 통합 — 리뷰어 고유 항목은 §1.1 확정 결정사항 및 T/F 항목의 "요구 리뷰어" 열로 추적한다.)

---

## 1. 목적

리뷰어 3명이 공통으로 요구한 실증 근거와 텍스트 수정을 **한 번만 수행**하여 리뷰어별 대응의 입력으로 제공한다. 핵심 방어 전략:

- **"data-driven / no hand-set constants" 주장 범위 축소**: 데이터 기반 주장은 compatibility model 도출에 한정, Eq.2 가중치·Table 3/4 임계값은 고정 설계 선택으로 인정 후 민감도 분석으로 견고성 입증 (가중치 표기 0.6/0.4 유지 결정).
- **음성 결과 정면 해명**: Leather는 학습 붕괴 오염값 재실험, Kolektor/MTD는 headroom 소진 프레임 전진 배치.
- **실험 보강**: Leather 재실행, ablation mAP, sensitivity sweep, 추가 detector, multi-seed. (ControlNet arm 실행은 범위 제외 — 텍스트 대응)
- **§3.2.6 전체 삭제(T4)**: Eq.4·절대 0.7 임계 서술을 논문에서 제거 — R2-1/R3-3의 Eq.4 공격 표면 자체를 소멸.

### 1.1 확정 결정사항 (2026-08-19)

| # | 결정 | 효과 |
|---|------|------|
| D1 | **void gate 메커니즘은 논문에 제시하지 않는다** | §3.2.6의 offline void 게이트(텍스처 에너지, p15 분위) 서술 삭제. 단 §3.2.4:143의 "void tiles ... excluded before scoring" admissibility 문장과 Figure 3.2.4-4/5 캡션의 void 언급은 **유지** — 작동 원리만 미서술 |
| D2 | **quality 0.7 문제는 §3.2.4로 완전 해소** | §3.2.6 통째 삭제 (Eq.4, 절대 0.7, void 서술 전부). §3.2.4 본문 무수정 — cross-ref 정리만. site 분위 필터도 논문 미서술 (논문은 ①void 배제 → ③ring argmax만 서술) |
| D3 | **하드코딩 분포 대응은 개선된 §3.2.1–3.2.3에서 기해소** | Table 4 tertile 전환(P33/P66 + Table 4b)·CCI 4-성분 수식·BIC-GMM partition이 R3-1 후반·R2-4를 선제 해소. 신규 원고 작업 불필요 — response letter에서 개정 절 지목 |
| D4 | **논문 결과 table은 ring 결과(exp4v2 20260813) 기준으로 재작성** | `.claude/.etc/exp4v2/20260813/exp4v2_results.json` (seed1, gate ON, ring 방법 = 개정 §3.2.4)을 §4 결과표의 기준 데이터로 승격. Leather는 random arm 붕괴(0.0492)로 **재수행 예정** — 치유값 확보 후 표 확정. R3-2 "3/5 패배" 프레임을 "4/5 우위 + MTD tie"로 역전 (T5) |
| D5 | **R1-3(비교 방법)은 ablation 수행(`exp_ablation_execute.md` = E2)으로 대체** | "recommend" 수위이므로 신규 비교 arm(ControlNet/CASDA) 없이 부분 수용. 논리 = **통제변인 구조**: 본 연구는 placement 통제 실험(전 arm 동일 결함 픽셀+동일 blend, placement만 조작 변인). 같은-변인 계열(copy-paste, context-aware)은 실험으로, 다른-변인 계열(hard-sample, 생성형)은 confound 논거+limitation으로 응답 (T6). CASDA arm 재실험 안 함 — 구 데이터(casda_aroma) CASDA 2/3 우세, 자충수 위험. ControlNet arm(f45c689)은 미소진 예비 카드로 보존 (재심 재요구 대비) |
| D6 | **R2-5(entropy·Gini 미보고)는 수치 보고 + 해석 통제(C안)로 대응** | 정량치를 표로 제시하되 프레임 선점: entropy/Gini는 선택 분포의 **균등성** 지표라 uniform random이 구성상 근최적 — AROMA의 낮은 entropy·높은 Gini는 호환 pair 의도적 과표집의 설계상 귀결. 목적 정렬 지표는 coverage·rare-pair(§3.3.3). 배경: entropy/Gini는 exp2 5-지표에서 부적합 판정으로 강등된 이력(`AROMA_worklist.md` P0 — "균등성 보상 = random 강점"), 제출본에 언급만 잔존해 지적 발생. 수치는 **현행 compat+ring 산출물 기준 CPU 재계산** (구 deficit 시대 값 사용 금지) — T7 |

### 1.2 기해소(Already-Resolved) 현황 — 개선된 §3.2.1–3.2.4의 리뷰 대응도

리뷰어들은 개선 전 제출본을 심사했다. R3가 인용한 Table 4 규칙("Linearity > 0.9 AND AspectRatio > 5")은 현행 원고에 존재하지 않는다.

| 지적 | 해소 수단 | 잔여 작업 |
|------|----------|----------|
| R3-1 후반 (Table 4 하드코딩) | §3.2.3 tertile 전환 + Table 4b 실측치 + "고정 상수 불가" 논증 + homogeneity safeguard | 답변 문안만 (개정 절 지목) |
| R2-4 (CCI 수식 3 vs 4) | §3.2.1 수식 4-성분 완비 + Table 2 분해 | 제출본 렌더링 대조만 (R2 SOP 고유) |
| R3-3 전반 ("no hand-set constants" 문장) | 해당 문장 §3.2.4에서 이미 제거됨 | 잔존 2곳(§3.2.2:24, §3.2.6:163) — §3.2.6은 T4 삭제로 소멸, §3.2.2는 T1에서 범위 한정 |
| R3-3 후반·R2-1 중 Eq.4 | **T4 삭제로 수식 자체 소멸** | 답변 경위 문안 (F1) |
| R2-3 (임계 민감도) | 임계값이 데이터 도출로 전환되어 전제 약화 | E3 Table 3/4 섭동으로 정량 보강 (유지) |

## 2. 적용범위

| 구분 | 범위 |
|------|------|
| 실험 (Colab) | E1–E5 전체 |
| 원고 공통 수정 | "manual tuning" 계열 문구 전수 교체(T1), conditional-effectiveness 전진 배치(T2), References 통합 갱신(T3), §3.2.6 전체 삭제(T4), §4 결과표 재작성(T5), R1-3 논리 사슬(T6), entropy·Gini 정량표(T7) |
| 최종 검수 | 전체 원고 일관성 감사 + 수치 삼중 대조(F1) |
| 제외 | ControlNet arm Colab 실행(구현 f45c689 유지, 후속), 신규 데이터셋 추가(CCI n=5는 주장 강등으로 대응) |

## 3. 수행절차

### 3.1 공통 실험 (E1–E5)

#### 3-1-1. 목적
민감도 분석·음성 결과 해명·검출기 일반성·시드 안정성 근거를 단일 Colab 캠페인으로 확보한다.

#### 3-1-2. 수행내용

| ID | 실험 | 요구 리뷰어 | 내용 |
|----|------|------------|------|
| **E1** | Leather 재수행 (random arm 붕괴 치유) | R1-4, R3-2 | 20260813 ring 결과에서 leather **random arm이 붕괴**(mAP 0.0492, precision 0.0002 — batch128+patience25 병리: 128장=1 batch→1 update/epoch, 25 updates 내 사살). **1차: random arm 단독 재수행** (kolektor 치유 knob 동일: batch 16 / patience 0 / rect 제거) — 실행 가이드 `.claude/.dev_note/exp4v2_leather_random_rerun_execute.md`. 치유값 확인 후 **표 확정 전 baseline·aroma 동일 knob 재수행 권장** (0812에서 baseline 1/3·aroma 1/3도 collapse — arm 간 프로토콜 혼재는 공정성 공격 표면). 치유값 확보 전 leather A−R 비교 인용 금지. 최종 판정은 3-seed 집계본(`$EXP4V2_OUT/exp4v2_results.json`)으로 — 20260813 파일은 seed1 단독 |
| **E2** | Ablation mAP arm (5-arm) | R1-1, **R1-3(D5 대체)**, R3-3 | 로컬 proxy 완결분의 Colab 다운스트림 mAP 실행 (3-seed). `exp_ablation_execute.md` 준수. **5-arm 스펙트럼**: full AROMA → A1 ROI-random → A2 BG-random → A3 Site-random → all-random(=uniform copy-paste, §4 기측정). 로컬 proxy 근거: Stage2 Δ+0.163 (p=6.7×10⁻⁵³), Stage3 누적 분해 0.131→0.081→0.061 (site/bg 기여 독립·누적). proxy 결과만으로 mAP 주장 금지 (정직성 원칙) |
| **E3** | Sensitivity sweep | R1-1, R2-1, R2-3, R3-1, R3-3 | (a) Eq.2 가중치 그리드 (0.5/0.5)·(0.6/0.4)·(0.7/0.3), (b) Table 3/4 임계값 ±섭동 시 카테고리 배정 변화율·성능 영향. ~~Eq.4 sweep~~ — T4로 수식 자체가 삭제되어 제외 (D2) |
| **E4** | 추가 detector | R2-2 | 최신 주류 1종(YOLOv11n 또는 RT-DETR 계열), Severstal + AITeX 최소 2종에서 baseline/random/AROMA 재현 |
| **E5** | Multi-seed | R1-4, R3-2 | 주요 비교(최소 Severstal/AITeX/Leather) seed 3개, 평균±표준편차 보고 |

※ 모든 실행 명령 `colab-execution.md` 규칙($VAR, !python) 준수. 임계값 관련 필터는 데이터셋별 발동률 CPU 사전 스캔 후 확정.

#### 3-1-3. 산출물
- E1: Leather 치유 결과표 (baseline/random/AROMA mAP@0.5)
- E2: Ablation mAP 결과표
- E3: Sensitivity 결과표 + figure (가중치 그리드, 임계값 섭동)
- E4: 추가 detector 결과표
- E5: Multi-seed 평균±표준편차 표
- Colab 실행 가이드 .md (dev_note 연계)

### 3.2 공통 텍스트 작업 (T1–T3)

#### 3-2-1. 목적
복수 리뷰어가 지적한 원고 수정을 단일 편집 패스로 처리하여 상호 불일치를 예방한다.

#### 3-2-2. 수행내용

| ID | 작업 | 요구 리뷰어 | 내용 |
|----|------|------------|------|
| **T1** | 주장 범위 축소 | R1-1, R2-1, R3-1, R3-3 | "free of manual tuning", "without hand-set constants", "no hand-set constants" 전수 검색 → compatibility model 도출 한정 표현으로 일괄 교체 (§3.2.2:24 포함; §3.2.6:163은 T4 삭제로 자동 소멸). Eq.2 가중치는 고정 설계 선택 명시 + E3 결과로 견고성 서술. §3.2.2–3.2.3에 Table 3/4 역할 분리(카테고리 라벨링은 해석용, 배치 결정은 BIC-GMM·tertile partition) 서술 |
| **T2** | 조건부 유효성 전진 배치 + 프레임 교체 | R1-4, R3-2 | D4 반영: "3/5 패배 해명"이 아니라 **"4/5 우위 + MTD tie" 서사로 교체**. §5의 headroom-exhaustion·conditional-effectiveness 프레임을 §4 결과 제시부로 전진 배치 — MTD(−0.009, near-ceiling 0.9282)는 E5 multi-seed로 "유의차 없음(tie)" 정리, Kolektor는 신 결과에서 +0.025 역전이므로 해명 불필요. E1 Leather 치유값 반영 |
| **T3** | References 통합 갱신 | R1-5, R2-7, R3-5~7 | 단일 패스로: (a) federated fault diagnosis 1문장 인용(R1-5), (b) 2024–2026 최신 문헌 3편 이상(R2-7), (c) Stavropoulos 2020·Bergmann 2021 (R3). 번호 재정렬 1회로 종결 |
| **T4** | §3.2.6 전체 삭제 | R2-1(Eq.4 파트), R3-3(Eq.4 파트) | D1·D2 확정 반영: (a) §3.2.6 Quality Gate 절 통째 삭제 — Eq.4(0.30/0.30/0.20/0.20), 절대 0.7 임계, "fixed absolute threshold, not a per-dataset adaptive fraction" 자기모순 문장, void 게이트 메커니즘 서술 전부 제거. (b) §3.2.4:143 "(Section 3.2.6)" cross-ref 제거 — admissibility 문장 자체와 Figure 3.2.4-4/5 캡션의 void 언급은 유지 (D1). (c) 절 번호 재정렬 (§3.2.5 Blending이 마지막 절이 됨) + **수식 번호 재정렬** — Eq.4 소멸로 이후 수식 번호 변동, 리뷰어가 "Equations (2) and (4)"로 지목했으므로 response letter에 구→신 번호 매핑 필수. (d) site 분위 필터는 논문 미서술 (D2) — 코드 공개 시 질의 대비 답변 문안만 F1에 준비 |
| **T6** | R1-3 대응 텍스트 (D5 논리 사슬) | R1-3 | Response letter + 원고 반영 4단 구성: (1) **통제변인 프레임 명시** — §3 또는 §4 도입부에 "전 arm 동일 결함 픽셀·동일 blend, placement만 조작 변인" 1문단. (2) **copy-paste** — §4의 all-random(Random) arm을 "uniform copy-paste [Dwibedi et al.]"로 재명명·인용 (외부 canonical baseline임을 명시). (3) **context-aware placement** — E2 ablation이 context-awareness를 성분 분해(ROI/BG/Site)함을 §4 신설 소절로 제시 + **§2에 Dvornik et al. (ECCV 2018)·InstaBoost (ICCV 2019) 인용 필수** — 설정 차이 명시(그들=annotated instance 재배치/일반 객체 삽입, 우리=결함 풀→clean 배경 cross-image 합성). 인용 누락 시 "상대는 CASDA뿐" 주장이 사실 오류로 반격당함 (T3 문헌 갱신과 통합). (4) **hard-sample·생성형** — confound 논거(외형+배치 동시 변화 → placement 귀속 붕괴) 우선, orthogonality·결합 가능성 보조, limitation에 "생성형 downstream 정면 비교는 범위 밖" 정직 명시 + ControlNet arm 구현 존재를 후속 확장으로 언급 |
| **T7** | entropy·Gini 정량표 + 해석 문단 (D6) | R2-5 | 절차 4단: (1) 제출본에서 Shannon entropy·Gini 언급 위치 특정 (R2-4 제출본 대조와 동일 패스 — 현행 §3.3.3에는 이미 부재, coverage 3종만 정의됨). (2) 현행 compat+ring 산출물(step3 roi_selected.json) 기준 CPU 재계산 — Colab GPU 불필요, severstal은 ablation k200 로컬 산출물로 즉시 가능, exp1/exp2 스크립트 재사용. (3) §4.1에 데이터셋 × arm(AROMA/Random) × (coverage 3종, entropy, Gini) 정량표 + D6 해석 문단 (균등성 지표 프레임 선점 + breadth-vs-depth tradeoff 정직 병기 — context_coverage random 우위도 구조적임을 함께 서술). (4) §4.1 산문에 "AROMA 전 지표 우위" 식 과잉 주장 잔존 여부 전수 확인·수정 (worklist P0 모순 이력) — 표 추가가 자기모순 노출하지 않도록 선행 필수 |
| **T5** | §4 결과 table 재작성 (ring 기준) | R3-2, R1-4 | D4 확정 반영: §4 다운스트림 결과표를 exp4v2 20260813 ring 결과로 전면 교체. 기준 데이터는 아래 기준표. 조건: (a) Leather 행은 E1 치유값 확보 후 기입 — 그 전까지 placeholder, (b) E5 multi-seed 완료 시 평균±표준편차 형식으로 최종화 (seed1 단독 수치로 논문 확정 금지 — 단일 샘플 단정 금지), (c) per-class 표(Severstal c1–c4 등)와 n_synth_per_class 배분도 신 결과 기준으로 갱신, (d) 본문·Abstract·§5·§6에 흩어진 구 수치 전수 교체 — F1 삼중 대조 대상 |

**[T5 기준표] exp4v2 ring 결과 (20260813, seed1, mAP@0.5)**
출처: `.claude/.etc/exp4v2/20260813/exp4v2_results.json` (gate ON, ring = 개정 §3.2.4 방법)

| Dataset | Baseline | Random | AROMA | A−R | 상태 |
|---------|----------|--------|-------|-----|------|
| Severstal | 0.4939 | 0.5185 | **0.5337** | +0.0152 | 확정 후보 (E5 대기) |
| Kolektor | 0.9188 | 0.9610 | **0.9855** | +0.0245 | 확정 후보 (E5 대기) — 제출본 열세에서 역전 |
| MTD | 0.8857 | **0.9282** | 0.9192 | −0.0090 | E5로 tie 여부 판정 (near-ceiling) |
| AITeX | 0.4083 | 0.4476 | **0.4516** | +0.0040 | E5로 우위 유의성 판정 |
| MVTec Leather | 0.8148 | ~~0.0492~~ 붕괴 | 0.8674 | 비교 무효 | **E1 재수행 대기** — random arm 치유 후 기입 |

제출본 대비: R3-2가 지목한 3개 열세 데이터셋 중 Kolektor 역전, Leather 재측정 예정, MTD만 근소 열세(−0.009) 잔존. AROMA > Baseline은 5/5.

#### 3-2-3. 산출물
- T1: 문구 교체 diff 목록 + §3.2.2–3.2.4 수정 원고 + sensitivity 소절 신설
- T2: §4/§5 수정 원고 (4/5 우위 + MTD tie 프레임)
- T3: References 갱신본 + §2/§5 인용 문장
- T4: §3.2.6 삭제 diff + 절·수식 번호 재정렬본 + 구→신 수식 번호 매핑표
- T5: §4 결과표 갱신본 (E1·E5 완료 후 최종) + 구 수치 전수 교체 diff
- T6: 통제변인 문단 + all-random→copy-paste 재명명 diff + §2 context-aware placement 인용 확장 (Dvornik·InstaBoost) + limitation 문단 + R1-3 답변 논리 사슬 문안
- T7: entropy·Gini 정량표 (§4.1) + 균등성-지표 해석 문단 + §4.1 과잉 주장 수정 diff + CPU 재계산 산출 로그

### 3.3 최종 검수 및 Response Letter (F1)

#### 3-3-1. 목적
19개 항목 point-by-point 답변서 작성, 수정 원고 정합성 최종 검증.

#### 3-3-2. 수행내용
1. 리뷰어별 항목 × (답변 요지 / 수정 위치 / 신규 근거) 매핑표 작성 — 각 리뷰어 SOP의 산출물 취합.
2. 범위 제외 항목(ControlNet 직접 비교, 데이터셋 추가)은 한계 인정 + 후속 과제 논거.
2-1. **Eq.4 삭제 경위 문안** (R2-1·R3-3 답변): 리뷰어가 명시 인용한 수식이므로 침묵 삭제 금지 — "품질 게이팅 서술을 정비하며 고정 가중치·절대 임계 수식을 제거했다"는 경위 + 구→신 수식 번호 매핑 제시. site 분위 필터 관련 질의 대비 예비 문안 별도 준비 (논문 미서술이므로 능동 언급은 하지 않음).
2-2. **기해소 항목 답변** (§1.2 표 기반): R3-1 후반·R2-4는 개정 §3.2.1/3.2.3 지목으로 답변 — 신규 실험·수정 없이 종결.
2-3. **R1-3 답변 사슬** (D5·T6): ① placement 통제 실험 프레임 → ② copy-paste = all-random arm 재명명(기포함) → ③ context-aware = ablation 성분 분해로 상회 + Dvornik/InstaBoost 인용·설정 구분 → ④ hard-sample/생성형 = confound 논거 + 결합 가능성 + limitation. 재심에서 생성형 재요구 시 ControlNet arm(구현 완료) 실행이 예비 카드.
3. 전체 일관성 감사: Abstract–Introduction–Results–Discussion–Conclusion 간 수치·CCI 톤·조건부 유효성 프레임 정합 (기존 consistency audit 절차 재적용).
4. 수치 교차 검증: 본문 ↔ 갱신 표 ↔ 실험 로그 삼중 대조.

#### 3-3-3. 산출물
- Response Letter (point-by-point, 영문)
- 지적-수정 매핑표
- 최종 수정 원고 (변경 추적본 + 클린본)
- 일관성 감사 노트

## 4. 기대효과

1. 3명 공통 급소(hand-set constants 모순)를 T1+T4+E2+E3 4중 근거로 단일 수행 방어 — 리뷰어별 답변 간 표현 불일치 원천 차단.
2. **R3-2 프레임 역전 (D4·T5)**: ring 결과 기준 AROMA > Random 4/5 예상(Kolektor 제출본 열세 → +0.025 역전, Leather는 E1 재수행 후 판정) + AROMA > Baseline 5/5 — "naive random조차 못 이긴다"는 R3-2 핵심 주장을 해명이 아닌 **우위 데이터로 직접 반박**. 잔여 열세는 MTD −0.009 하나뿐이며 near-ceiling tie로 정리.
3. **T4(§3.2.6 삭제)로 Eq.4 공격 표면 소멸** — R2-1·R3-3의 절반이 실험 없이 종결, E3 범위 축소로 Colab 비용 절감. §1.2 기해소 항목(Table 4 tertile, CCI 4-성분)까지 합치면 hand-set 계열 지적 중 실험이 필요한 것은 Eq.2 가중치와 Table 3/4 섭동뿐.
4. 공통 작업 단일화로 편집 충돌·번호 재정렬 반복 등 중복 비용 제거.
5. 잔존 리스크 명시: (a) 논문 Eq.2 가중치 표기(0.6/0.4) vs canonical 코드(0.5/0.3/0.2) 불일치 유지 — 코드 공개 요구 시 노출 위험, E3가 "가중치 비민감" 결론이면 완화. (b) site 분위 필터·타일 void 게이트(절대 0.7, `generate_defects.py`)는 코드에 존재하나 논문 미서술(D1·D2) — 코드 공개 시 "논문에 없는 필터" 질의 가능, F1 예비 문안으로 대비.
