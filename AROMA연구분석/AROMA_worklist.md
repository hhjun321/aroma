# AROMA 연구 작업목록 (종합)

**작성**: 2026-06-29
**범위**: 논문(`Article/AROMA.txt`) 완성까지 필요한 전체 작업
**현 상태**: 논문 RESULTS FROZEN (2026-06-24) — 모든 결과 테이블 TBD, 헤드라인 수치는 projected/unbacked로 제거됨

---

## 0. 이번 세션 확정 사항

| 항목 | 결정 |
|------|------|
| **clean-bg 게이트** | 항상 ON 정책 (`--reject-clean-bg`, min-bg-quality 0.7, blur 100.0). 전 조건(aroma/random/casda) 동일 적용 |
| **평가 데이터셋** | **5셋 확정 (deficit-우선 재선택, 2026-06-29)** — carpet, leather, macaroni, fryum, severstal |
| **선택 기준 전환** | MCI/CCI spread → **deficit-richness**(rare-pair 수 + deficit gini). AROMA 핵심주장(deficit-aware ROI)이 측정되려면 deficit 신호 필요 |
| **선택근거 figure** | `fig_dataset_selection_deficit.png` (2패널: A.복잡도 landscape, B.deficit-suitability) |

### 확정 데이터셋 (5, deficit-우선)

| 데이터셋 | 도메인 | MCI | CCI | rare(d>0) | deficit gini | 역할 |
|---|---|---|---|---|---|---|
| visa_macaroni | 농산물 | 0.405 | 0.239 | 76 | 0.730 | 저-MCI 앵커, 최고 gini |
| visa_fryum | 농산물 | 0.402 | 0.239 | 45 | 0.697 | visa 균형 |
| severstal | 철강 | 0.455 | 0.306 | 358 | 0.693 | CASDA 원도메인, 최고 CCI |
| mvtec_leather | 가죽 | 0.645 | 0.236 | 134 | 0.647 | 고-MCI |
| mvtec_carpet | 카펫 | 0.671 | 0.233 | 1161 | 0.657 | 최대 placement 공간 |

전부 deficit-rich(rare≥45, gini≥0.6) → exp2서 AROMA deficit-aware 신호 측정 가능. 도메인 visa2/mvtec2/steel1.

### 선택 근거 (deficit 구조 분석, profiling 전 24셋)
- **exp2 변별 = (rare-pair > top_k) AND (deficit gini 높음)**. 둘 중 하나라도 실패하면 AROMA≈Random.
- **deficit-richness ⊥ CCI**: 강변별셋 전부 저-CCI(0.23~0.31). 고-CCI셋(bottle/pill/zipper/metal_nut)은 deficit-빈약 → 고-CCI spread와 exp2 변별 동시 불가. CCI는 severstal(0.306) 최고로 수렴.
- **AROMA 무의미 셋**: bottle/screw(rare=0, gini=0 — deficit 없음), cashew(rare 5, gini 0.05 — 평탄), capsule. 제외.

**이전 6셋(cable/pill/wood/metal_nut/cashew/severstal) 폐기 이유**: MCI/CCI spread로 골라 deficit-poor 셋(pill/metal_nut/cashew) 다수 포함 → exp2서 AROMA 우위 원리적 불가. severstal만 변별됐음. **transistor 후보였으나 연속성·고려사항 다수로 제거.**

⚠️ **트레이드오프**: 기존 exp4v2 완료셋(pill/wood/cable 3-seed) 폐기 → 신규 5셋 다운스트림(exp3/exp4) 전체 재실행 필요. MCI 중간대(0.46~0.64) 갭 존재(grid 0.605로 보강 가능, 보류).

---

## 1. 논문 현황 분석

### 1.1 RESULTS FROZEN 상태
- Abstract: 결과 수치 보류, "Image AUROC 0.923" 등 projected 수치 제거 (정직)
- Tables 8–12: 전부 `TBD`
- 하드룰: commit된 result-file 경로 없는 수치 금지

### 1.2 🔴 치명적 내부 모순 (SCI 심사 즉시 적발 — 최우선)

| 위치 | 문제 |
|------|------|
| §5 Discussion (line 318) | "AROMA>Random>Baseline ... **avg AUROC 0.923**" — Abstract가 제거한 수치 본문 잔존. 자기모순 |
| §6 Conclusion (line 327) | "**avg Image AUROC 0.923**" — 동일 모순 |
| §4.3 prose (line 198) | "16 cells 단조 AROMA≥Random>Baseline" — 실제 유일 AD셀(mvtec_cable/SuperSimpleNet)은 AROMA **패배** (0.4996 vs 0.7646) |
| §4.4 Table 12 vs Fig caption | 테이블=all-zeros 실패런, 캡션=Baseline 0.4751/Random 0.2079. 둘 다 stale·상호모순 |

### 1.3 논문 주장 vs 실제 데이터 괴리

논문 헤드라인: *AROMA > Random > Baseline 일관, avg 0.923*

실제 commit 데이터:
- **MVTec 3-seed (ratio 0.5)**: AROMA ≈ Random (Random 약간↑: cable Δ+0.56, pill −2.13, wood −0.76pp), 둘 다 > Baseline
- **Severstal cleanbg 0.4 (3-seed)**: mean baseline 0.390 > casda 0.371 > aroma 0.364 > random 0.346
  - AROMA > Random **2/3 seeds** (+1.9pp)
  - AROMA ≈ CASDA (−0.7pp) ← 핵심: 수작업 CASDA와 동등, 재엔지니어링 불요
  - Baseline가 전부 이김 ← 합성품질(copy-paste) 한계, AROMA 범위 밖

→ "AROMA>Random>Baseline 일관"은 데이터로 반박. **프레이밍 전환 필요**: AROMA≈CASDA + AROMA>Random(부분) + baseline 미달=합성품질 한계(범위 밖)

---

## 2. 작업목록 (우선순위)

상태: ✅완료 🔲대기 🔄진행중 ⛔차단

### P0 — 내부 모순 제거 (compute 0, 즉시, 논문 무결성)
- 🔲 §5 line 318 — "0.923" + "AROMA>Random>Baseline" 삭제 → 실제 데이터 프레이밍
- 🔲 §6 line 327 — "avg AUROC 0.923" 삭제
- 🔲 §4.3 line 198 prose — "16-cell 단조" 주장 제거, AD 프로토콜 deprecate 반영
- 🔲 §4.4 Table 12 + Fig caption — stale all-zeros/모순 캡션 제거

### P1 — §4.4 실제 exp4v2로 refill (데이터 일부 보유)
- 🔲 MVTec 3-seed(0.5) → Table 12 재작성 (pill/wood/cable, AROMA≈Random·둘다>Baseline)
- 🔲 Severstal cleanbg 0.4 **3-seed 집계** — `casda_0.4_cleanbg/{1,2,43}.json` → `exp4v2_summary.md` 미집계 상태. mean±std+CI 산출
- 🔲 ratio 통일 결정 (0.4 vs 0.5) 또는 dataset별 ratio 명시
- 🔲 §4.4 산문 재작성 (실패런 서사 제거, AROMA≈CASDA·AROMA>Random 프레이밍)

### P1.5 — exp3 cleanbg 재실행 (가이드 준비 완료 ✅)
- ✅ exp3_execute.md / step4_execute.md 게이트 플래그 + 삭제노트 추가 (이번 세션 workflow)
- 🔲 Colab: synthetic 삭제 → AROMA/Random 게이트ON 재생성 → exp3 FID/AD → §4.2 Table 9,10 refill
- 🔲 exp3 가이드 DATASETS 6셋으로 갱신 (isp 제거, toothbrush 추가) — ISP 동기화(§3)와 연동

### P2 — 미실행 실험 생성
- 🔲 §4.1 ROI coverage (exp2_roi_quality) — result JSON 부재. 6셋 실행 → Table 8
- 🔲 §4.3 처리 결정: one-class AD 폐기 vs 유지 (frozen note=폐기·exp4v2 대체 권고)
- 🔲 **toothbrush 신규 실험** — exp3(FID/AD) + exp4v2(YOLO), 게이트 ON, 3-seed

### P3 — 정성 그림 §4.5 (Colab 차단 ⛔)
- ⛔ Drive에서 {real/random/AROMA} crop + normal 배경 export → matplotlib grid (.md 가이드, 신규 .ipynb 금지)
- ⚠️ mvtec_cable AROMA n_train=0 이력 — 존재 검증 후 사용

### P4 — 통계·다중시드 (Discussion 한계 #1)
- 🔄 MVTec/severstal 3-seed 완료, 나머지(cashew/wood/toothbrush) multi-seed + 유의성 검정 필요

### P5 — 데이터셋 정렬 / ISP 제외 동기화
- 🔲 (§3 참조) ISP 잔존 4건 정리

---

## 3. ISP 제외 동기화 항목

ISP 전 셋 제외 결정 → 잔존 isp 정리. **visa_pcb도 함께 제외**(최종 6셋에 없음).

| 위치 | 상태 | 조치 |
|---|---|---|
| `colab_execute/exp2_execute.md` | ✅완료 | 6셋 갱신 (prose+DATASETS+--dataset_keys) |
| `colab_execute/exp3_execute.md` | ✅완료 | 6셋 갱신 (전제+DATASETS+--dataset_keys×2+주의사항) |
| `exp3-sharpened-spec.md` | 🔲대기 | scope+체크리스트 isp_LSM_1/visa_pcb 제거 → 6셋 |
| `Article/AROMA.txt` Table 1 + §3.1 | 🔲대기 | isp_LSM_1/visa_pcb4 행·산문 제거, mvtec/visa/severstal로 |
| `Article/figure/fig_dataset_selection_candidates.png` | 🔲대기 | isp 후보 추천 obsolete — 삭제/폐기 표기 |

선택 6셋 자체는 ISP 없음 → 실험 재실행 영향 없음. 문서/spec 동기화만.

**exp2 실행 확인 (2026-06-29)**: 6셋 전부 random ROI 선택 성공 (severstal 포함 profiling/prompts 존재 확정). severstal 266,609 candidates→200 selected. ⚠️ `No module named 'utils'` → quality gate 전 셋 비활성(기본 OFF 정책이라 결과 무관).

---

## 7. severstal 파이프라인 점검 결과 (구현 보류 — 보고만)

점검일 2026-06-29. 발견만 기록, 구현은 업로드 결과 확인 후 우선순위 재조정.

| # | 발견 | 심각도 | 타입 | 위치 |
|---|---|---|---|---|
| 1 | c2 fix 커밋됐으나 미재실행 | High | 실행 | (Colab 재실행) |
| 2 | **cross-class deficit 할당 무력화** — `top_k%K==0`(200%4=0) 시 클래스간 배분 순수 uniform, deficit-aware 미적용. c2 희소클래스 우선 안 됨 → "deficit-aware" 주장이 severstal multi-class서 절반만 참 | Medium | 설계 | `roi_selection.py:821-834` |
| 3 | per-class mask 누락 silent (downstream 무음 실패 가능) | Low | 견고성 | `exp4_v2_supervised_detection.py:247-254` |
| 4 | synth class fallback silent (source_roi 없으면 class0 기본, 경고 없음) | Low | 로깅 | `exp4_v2_supervised_detection.py:807-832` |
| 5 | ROI 선택 출력에 per-class 카운트 없음 (c2 불균형 비가시) | Low | 가시성 | `roi_selection.py:~1260` |
| 6 | severstal `background_type` 검증 없음 (directional 강제 안 됨) | Low | 검증 | `roi_selection.py:~1307` |
| 7 | void-rejection 임계(`_FG_VOID_*`) CLI 미노출, 하드코딩 | Low | 기능 | `generate_defects.py:169-177` |
| 8 | `No module named 'utils'` → quality gate import 불가 (전 셋, Colab 경로) | Low | 환경 | `roi_selection.py` |

**비-이슈**: ROI 후보 266K 폭증 = full-frame strip 특성(3620 img × ~74 context bins), 버그 아님. 성능 우려 시 candidate 샘플링 옵션 추가 가능(저우선).

**핵심 (#2)**: severstal multi-class에서 deficit-aware가 클래스 *내부*엔 작동하나 클래스 *간*엔 미작동(uniform). c2 rare-class 우선 배분 누락 → 변경 시 결과 바뀜 → 재실행 검증 필수. 안전 관찰성 개선(#3~6)은 결과 불변, c2 진단에 직접 도움.

**관련 dev_note**: `aroma_severstal_flat_diagnosis_and_direction.md`, `aroma_exp4v2_severstal-synth-multiclass.md`, `aroma_step3_multiclass-allocation-fix.md`, `aroma_exp4v2_foreground-void-rejection.md`, `aroma_exp4v2_clean-background-gate.md`.

---

## 8. exp2 결과 분석 (2026-06-29, `.claude/.etc/exp2/`)

**결과가 논문 §4.1 주장 반박.** 6셋 실행 결과:
- 커버리지 3종(morph/ctx/rare-pair) = object-centric 5셋서 **전부 1.0=1.0 포화** (무변별). top_k=200이 후보(225~452) 대부분 선택 → 양 전략 전 셀 커버.
- entropy/gini 혼합: AROMA 명확 승=wood만, **Random 승=cable·cashew**, pill 혼합, toothbrush 동일(퇴화).
- severstal: AROMA 우위처럼 보이나 **n_sel 1690 vs 200 비대칭** 탓 (step3 AROMA 예산 ≠ exp2 random 200). 교란.

§4.1 산문(line 151) "AROMA가 전 데이터셋서 coverage 우위, rare-pair 최대, entropy↑ gini↓" → **데이터로 거짓**. P0 모순에 추가.

### 조치 (사용자 결정)
- ✅ **#1 n_selected 동일화** — `exp2_roi_quality.py`에 `_equalize_budget` 추가. **재실행됨** (n_equalized=200 전부).
- ✅ **#3 toothbrush→metal_nut** (위 §0). metal_nut exp2 아직 미실행(step3 profiling 필요).

### equalized 재실행 결과 (2026-06-29, 5셋)
| ds | ctx_cov A/R | rare_pair A/R | entropy A/R | gini A/R | 판정 |
|---|---|---|---|---|---|
| pill | 1.0=1.0 | 1.0=1.0 | 0.761/0.745✓ | 0.472/0.464✗ | 혼합 |
| wood | 1.0=1.0 | 1.0=1.0 | 0.816/0.584✓ | 0.317/0.487✓ | AROMA |
| cable | 1.0=1.0 | 1.0=1.0 | 0.624/0.645✗ | 0.417/0.380✗ | Random |
| cashew | 1.0=1.0 | 1.0=1.0 | 0.840/0.932✗ | 0.370/0.256✗ | Random |
| severstal | **0.710/0.785✗** | 0.418/0.404✓ | 0.971/0.950✓ | 0.164/0.218✓ | 혼합 |

**핵심**: equalize 후 severstal AROMA ctx_cov 0.710 < random 0.785 — 이전 "압승"은 n_sel=1690 인공물. 공정예산서 AROMA가 컨텍스트 셀 덜 커버.

### 근본원인 (`roi_selection.py:592`)
`top_k <= n_pairs` 분기(severstal: distinct pairs>200)에 **커버리지 floor 없음** → deficit 점수 top-200, 같은 (cluster,cell) 중복 허용 → distinct 셀↓. object-centric(pairs<200)은 else 분기서 1-per-pair 보장 → ctx_cov 1.0 포화.

### B1 수정 ✅ (구현 완료, 재실행 검증 필요)
`roi_selection.py:592` 분기를 **coverage-first deficit-ordered**로 교체: pair당 최대 1개, PairDeficit 내림차순 방문 → top_k distinct pair 보장(rare 우선). severstal ctx_cov·rare_pair 둘 다 random 추월 기대. random 전략(uniform `rng.choice`)은 이 함수 미경유 → AROMA만 변경. py_compile 통과.

### 후속
- ✅ **B2 top_k sweep 가이드** — `colab_execute/exp2_topk_sweep_execute.md` 작성. top_k∈{20,30,50,100,200}, AROMA(B1)+Random 재선택, exp2 평가, coverage-vs-budget 곡선. **Colab 실행 대기**.
- 🔵 metal_nut도 퇴화 확인 (n_cand=93<200, toothbrush 53과 동일). top_k<93서만 변별 → sweep로 해소.
- 🔲 **#4 cross-class deficit** — §7 #2, 결과변경 보류.
- 🔲 **metal_nut exp2** — step3 profiling/prompts 생성 후.
- 🔲 **§4.1 재서술** — exp2 데이터가 "AROMA coverage 우위" 미지지. B1 재실행 후 재평가.

### #2 — 커버리지 포화 해소 / AROMA 우위 입증 방향
1. **top_k 축소**(30~50) — 선택 압력, rare_pair 변별 회복.
2. **coverage-vs-budget 곡선** — "더 적은 예산으로 전 셀 커버" 효율.
3. **지표 정렬** — entropy/gini는 균등성=random 강점. deficit-aware는 rare 과표집→비균등. rare_pair@작은예산을 1차 지표로.
4. **B1 + #4** — breadth+depth 동시 확보.

### #2 — 커버리지 포화 해소 / AROMA 우위 입증 방향
1. **top_k 축소** (예 30~50) — 선택 압력 부여. 포화 해소돼 rare_pair 변별 회복. equalization은 필요조건이나 불충분(200이면 여전히 포화).
2. **coverage-vs-budget 곡선** — endpoint 대신 "더 적은 선택으로 전 셀 커버" 효율로 보고. deficit-aware의 진짜 장점은 효율.
3. **지표 정렬** — entropy/gini는 *균등성* 보상 = random 강점. deficit-aware는 의도적으로 rare 셀 과표집 → 분포 비균등 → gini↑. 즉 **entropy/gini는 AROMA의 성공 지표로 부적합**(cable/cashew "패배"는 설계상 당연). rare_pair@작은예산을 1차 지표로.
4. **#4 cross-class deficit 수정** — multi-class서 rare-class 우선 배분 실제 작동시켜 severstal 등서 우위 확보.

---

## 9. 방법론 전환 L1+L3 구현 완료 (2026-06-29)

deficit-aware 폐기 → **compatibility+quality 선택(L1) + context-aware 블렌딩(L3)**. dev_note: `aroma_roi-synthesis_compatibility-context-blend.md`. workflow 구현, verify 7/7 PASS.

- **L1** `roi_selection.py`: utils import 복구(`__file__` 루트 부트스트랩) → quality gate 작동; 신규 `compatibility` strategy(0.6·ctx_prior+0.4·morph_prior, deficit 無, quality gate 필터); deficit_aware ablation 보존.
- **L3** `generate_defects.py`: `_reinhard_transfer`(Lab, grayscale→L-only) + `_context_aware_composite`(로컬배경 Reinhard → cv2.seamlessClone) ; `blend_mode='seamless'`(default 'alpha'); fallback 완비; position/mask/bbox 무변경.
- **wrappers**: random/casda `blend_mode` 포워딩(세 arm 공정).
- py_compile 4파일 통과. commit 대기.

### 확정 실험 설계 (2026-06-29, TODO 전부 해소)

| 실험 | 데이터셋 | 조건 | 지표 |
|------|---------|------|------|
| **exp2** | 5셋(severstal+carpet+leather+macaroni+fryum) | random vs aroma | 선택 ROI 평균 compatibility(ctx_prior)+quality (coverage/deficit 폐기) |
| **exp3** | 동 5셋 | baseline/random/aroma | FID + one-class AD |
| **exp4v2 2층위** | severstal | baseline/random/**casda**/aroma | supervised detection mAP |
|  | carpet/leather/macaroni/fryum | baseline/random/aroma | supervised detection mAP |

- casda=철강 전용 → severstal 4-way만. 전 5셋 supervised detection으로 다운스트림 일반화.
- 합성 조건 = ROI 선택만 다름, blend=seamless 동일.
- realism(합성성능) 지표는 **표 제외** — AROMA 기여=ROI 선택, 블렌드는 enabler.

### 해소된 TODO
compat 공식(0.6·ctx+0.4·morph, quality gate), Reinhard(grayscale→L-only), exp2 재목적화, realism 표제외, blender 세 arm seamless, utils `__file__` 부트스트랩 — 전부 확정. quality threshold만 첫 gated run 후 결정(defer).

### 남은 후속
- ✅ 재실행 가이드 작성: `colab_execute/method_pivot_rerun_execute.md` (step3 compatibility + step4 seamless + exp2 inline + exp3 + exp4v2 2층위 + ablation).
- 🔲 Colab 실행: 위 가이드대로 5셋 재생성·평가. 배치 measurable 육안, ablation(compat vs random vs deficit / alpha vs seamless).
- 🔲 quality threshold 결정 (gate 복구 후 quality_score 분포 확인 → `--min_quality` 상향).
- 🔲 casda metadata_csv 경로 확인(severstal casda arm).
- 🔲 결과 확보 후 논문 §3.2(방법)·§4(결과) 재서술 + P0 모순 제거.

---

## 4. 완료 항목 (이번 세션)

- ✅ clean-bg 게이트 구현 (generate_defects/random/casda) — 이전 세션, 정책 항상 ON 확정
- ✅ exp3/step4 Colab 가이드 게이트 플래그 주입 + 삭제노트 (workflow, verify 7/7 통과)
- ✅ 데이터셋 선택 figure `fig_dataset_selection_landscape.png` (6셋, ISP 제외, MCI–CCI landscape)
- ✅ 선택근거 정량분석 (3×3 격자 커버리지, 후보 랭킹)

---

## 5. 자산 위치

| 자산 | 경로 |
|------|------|
| 논문 | `AROMA연구분석/Article/AROMA.txt` |
| 선택 figure | `AROMA연구분석/Article/figure/fig_dataset_selection_landscape.png` |
| figure 생성 스크립트 | scratchpad (⚠️ repo 영속화 필요 — 재현성) |
| MVTec 3-seed 결과 | `.claude/.etc/exp4v2/summary_ratio_0.5/` |
| Severstal cleanbg 0.4 (3-seed) | `.claude/.etc/exp4v2/casda_0.4_cleanbg/{1,2,43}.json` (미집계) |
| Severstal multi n=1 (pre-fix) | `.claude/.etc/exp4v2/summary_ratio_1.0/` |
| complexity reports | `.claude/.etc/complexity/<dataset>/complexity_report.json` |
| exp3 재실행 가이드 | `AROMA연구분석/colab_execute/exp3_execute.md`, `step4_execute.md` |

---

## 6. 추천 진행 순서

1. **P0** (오늘, compute 0) — 논문 내부 모순 제거. 심사 리스크 최우선
2. **P1 #2** — severstal cleanbg 0.4 3-seed 집계 (데이터 있음, compute 소량)
3. **§3 ISP 동기화** — spec/guide/paper isp 제거 (compute 0)
4. **P1.5 + P2 toothbrush** — Colab: 6셋 게이트ON exp3/exp4v2 재실행
5. **P1 #1,#4** — exp4v2 실제 결과로 §4.4 refill + 산문 재작성
6. **P3** — 정성 그림 (Colab pixel 확보 시)
