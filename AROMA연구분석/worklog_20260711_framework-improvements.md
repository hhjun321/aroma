# AROMA 프레임워크 개선 작업 정리 — 2026-07-11

> **목적**: 오늘 수행한 개선(아이디어 도출·설계·구현)과 테스트·점검 결과를 논문 작성 참조용으로 정리한다.
> **범위**: 커밋 `9978826`(class vs cluster 재포지셔닝) … `688a495`(execute 문서 개선), 총 15커밋.
> **원칙**: AROMA 연구는 **data-driven(no-hardcoding)** 을 기본으로 하며, 결과는 **정직하게**(과장·창작 금지) 보고한다.
> **관련 문서**: 로컬 재검증 실측 `local_revalidation_mtd20_20260711.md`, 피벗 검증 `pivot_local_validation_20260711.md`, 클래스/클러스터 검증 `class_vs_cluster_validation_20260711.md`.

---

## 0. 한눈에 보기

오늘의 개선은 **"결함 crop을 어느 clean 배경에, 어디에, 어떤 크기로 붙일지"를 생성 시점 재계산이 아니라 프로파일링-파생 사전선정으로 전환**하고, 그 선정을 **데이터-유도 신호**로 결정하도록 만든 것이다. 부수적으로 프로파일링의 두 무결성 결함(image_id 충돌, dim 과소추정)을 고쳤다.

| 축 | 개선 전 | 개선 후 |
|---|---|---|
| 배경 선택 | 생성 시점 원본 good 재스캔·랭킹(매 ROI·매 seed) | **profiling-파생 사전선정**(`clean_bg_selected.json`), 재현·대칭대조 |
| 선택 신호 | 단일 히스토그램 | **3-신호 데이터-유도 가중**(src / class / size) |
| 배치 위치 | geometry-blind(클래스 무관) | **클래스 기하 prior**(E2 레버, opt-in) |
| 크기 대응 | fit 이진 게이트(기록만) | size-fit 신호 + scale 기록 + 위치 재계산 |
| 이미지 dim | patch-격자 추정(≤63px 과소) | **실제 dim 컬럼**(gap 0) |
| image_id | stem(클래스 충돌 → leather 붕괴) | **클래스 고유키** |

---

## 1. 전략적 배경 — 아이디어 도출

### 1-1. 재포지셔닝: multi-domain data-driven ROI 검출 + context-aware 배치 (training-free)
- 기존 CASDA(단일 도메인 + ControlNet) 대비, AROMA를 **다중 도메인·데이터-유도 ROI 영역 검출 + 문맥 인지 배치를 copy-paste(무학습) 기판 위에서** 수행하는 방법론으로 재정의.
- ControlNet은 future work로 격하. 단 **deficit⟂availability 음성 결과**(결핍과 가용성의 비상관)는 생성(generation)을 future work로 정당화하는 근거로 **보존**.
- 자동화 주장은 **context/background 축으로 한정**. 결함 그룹핑은 클래스 라벨 사용. class_ratio 매칭은 주장에서 **제외**.

### 1-2. class vs cluster 실증 (커밋 `9978826`, `24d2c30`)
- 비지도 morphology 클러스터가 결함 클래스를 얼마나 대변하는지 4종 로컬 검증(V1–V4). 결과는 도메인-의존적 — 클래스 라벨을 1차 그룹핑 축으로 사용하는 근거.

### 1-3. copy-paste 피벗 + 정직한 신호 특성화 (커밋 `da0ae23`)
- **E1(clean-bg 히스토그램 매칭)**: 변별력이 **도메인-조건부**. aitex는 강함(hist∩ best ~0.89), **severstal/mtd는 랜덤 배경과 사실상 구분 불가**(lift≈0).
- **E2(배치 기하)**: 현행 배치는 **geometry-blind** — 클래스별 실제 edge/surface 경향(mtd break edge 100% 등)이 배치에서 ~50%로 평탄화.
- **함의**: clean_bg 사전선정의 **확정 가치 = 재현성 + 대칭 대조군 + 목적함수 강화 여지**. "일반적 mAP 향상"은 무검증 주장 금지 → GPU 실험으로 판정.

### 1-4. no-hardcoding 원칙
- 임계·가중·경계는 **관측 분포에서 유도**(void P90/P1, pool P95, 신호 가중은 측정된 lift 비례). magic number 금지, 유도값은 summary에 기록.

---

## 2. 개선 항목 — 설계와 구현

### 2-1. phase0 무결성 (커밋 `31ee0aa`, `b1bb497`)

**(A) image_id 클래스-충돌 고유키** — MVTec leather는 `defect_type` 폴더 간 stem(`000`…) 충돌 → `cluster_assignments` last-wins 덮어쓰기로 92 결함이 19로 붕괴. `image_id = f"{defect_type}_{stem}"`로 수정(morph·context 두 워커 락스텝). mtd/severstal/aitex는 stem이 이미 고유라 문자열만 변경(로직 무영향); leather가 실효 수혜자.

**(B) 실제 이미지 dim 컬럼** — `context_features.csv`의 patch_xy 격자로 dim을 추정하면 비-overlap truncate 타일링 탓에 **항상 과소추정**(로컬 mtd 실측 mean −31~−32px, max −63px, 956/956 과소). `_context_worker`가 이미 보유한 `h,w=img.shape`를 `image_w/image_h`로 방출(재읽기 0). 소비측 `clean_bg._image_dim`은 컬럼 있으면 실제값, 없으면 격자 추정 fallback(하위호환).

### 2-2. clean_bg_selection 모듈 신설 (커밋 `ce640a9`, `76be646`, `7a86875`, `b1267f8`, `b2ebd76`)

`roi_selection.py`를 미러링한 신규 전담 단계(step3.5). profiling-파생 파일(`context_features.csv`·`compatibility_matrix.json`·`roi_selected.json`)로 clean 배경을 **사전선정** → `clean_bg_selected.json`. 원본 good 픽셀 재스캔 금지.

- **선행 void/품질 필터**: non-void 셀 기준 데이터-유도 컷(P90 void_frac, P1 floor). 유효 pool만 후보.
- **후보 3-기준**: (1) defect-type/class 적합, (2) bbox size-fit, (3) bg-type↔defect 매칭.
- **Phase 1 (E1 충실)**: per-source 히스토그램 매칭. `src_fit_ceiling_mean`을 E1 재현 게이트로 emit(Phase 2 가중과 독립).
- **Phase 2 (class-fit, 데이터-유도 가중)**: 클래스-조건부 배경 히스토그램을 추가 신호로. 각 신호의 측정 lift에 비례한 가중 → **약한 신호 자동 downweight**.
- **Phase 3 (E2 레버, opt-in)**: 클래스별 실제 edge/surface/span prior를 morphology bbox+실제 dim에서 유도 → 배정 배경 위 **paste 위치 precompute**. 기본 OFF(mAP 효과 GPU-TBD).
- **크기 대응(옵션1·2)**: size-fit 신호(fit-rescale 계수)를 3번째 데이터-유도 신호로 편입 → 배경 크기 편차 있는 셋만 자동 가중(균일 셋은 w_size≈0). `scale_factor` 기록(투명성) + Phase3 위치를 **effective(리스케일 후) 크기**로 계산해 generation clamp 회피.
- **대칭 대조군**: `random_arm()`이 동일 ROI에 **uniform 무작위 배경** 배정(`clean_bg_random_arm.json`) → 배경 정체성만 다른 tight 계측기.

### 2-3. generate_defects 소비 통합 (커밋 `7ecdc5c`, `bfe0264`)
- `clean_bg_selected.json` 자동 로드 → per-ROI 배정 배경·위치 소비. rep→pool 결정론적 인덱싱, rng 스트림 정렬(placeholder draw), staleness(image_id) 가드, `used/fallback/mismatch` telemetry + `<90%` WARNING.
- Phase3 위치는 `_paste_and_finalize(forced_xy=…)` → `_place_on` early-return으로 소비(copy_paste·controlnet 양 경로).

### 2-4. 코드 리뷰 반영 (커밋 `76be646`)
- CRITICAL: rng 부분 해소 → placeholder draw로 스트림 정렬; staleness image_id 가드; resolve telemetry + <90% 경고. MED/LOW 정리.

---

## 3. 테스트 및 점검

> CLAUDE.md: pytest 금지·부하테스트 자동실행 금지. 검증은 **로컬 CPU 실행 + 재측정 셀 + 서브에이전트 코드리뷰**로 수행.

### 3-1. E1/E2 로컬 검증 (피벗 단계)
- E1 `src_fit_ceiling`(sim_best 근접): mtd 0.50 / aitex 0.89 / severstal 0.62 재현.
- E2 배치 prior: Phase3 위치가 클래스 edge/surface 경향 복원(break/crack/fray edge, blowhole surface).

### 3-2. mtd 20-ROI end-to-end 재검증 (오늘 최종, `local_revalidation_mtd20_20260711.md`)

로컬 `AROMA_DATASET/mtd`(good 956, defect 388)로 profiling→roi_selection(20)→clean_bg(`--geometry_prior`)→generate_defects(40 synth) 신 코드 전량 재생성.

| 점검 | 결과 |
|---|---|
| dim 컬럼 gap 폐쇄 | grid mean L1 **63.2px**(0/956 정확) → exact **0.0px**(956/956) |
| E1 재현 게이트 | src_fit_ceiling **0.476**(mtd~0.50, PASS) |
| 3-신호 데이터-유도 가중 | w_src **0.559** / w_class **0.282** / w_size **0.160** |
| size-fit(exact-dim) | 리스케일 3/20, mean scale 0.976 |
| clean_bg 소비 telemetry | used=**40** fallback=**0** mismatch=**0** / 40 |
| 배경 정체성(rep→pool) | **40/40** |
| 위치 정확 반영(clamp-free) | **40/40** |
| geometry per-class edge+span | blowhole **0%** / break·crack·fray·uneven **100%** |

**해석**: 위치 40/40 정확은 dim 컬럼(작업 B)의 직접 성과(구 grid dim은 리스케일 후 clamp 유발). geometry는 E2 평탄화를 실제 클래스 기하로 복원. mtd 배경 매칭 hist∩ 0.46 = ~random(정직성 유지).

### 3-3. 서브에이전트 코드리뷰
- dim 컬럼 변경: code+python 리뷰 **CRITICAL/HIGH/MEDIUM 0건**(헤더/emit 정합, 소비처 무영향, (W,H) 축 일치, 예외 커버리지 `float('')·nan·None`, 하위호환).
- 소비처 교차검증: `compute_complexity`(CONTEXT_FEATURES 순서-인덱싱)·`build_train_jsonl`·`roi_selection` 모두 컬럼 append에 안전(전부 DictReader 키 접근).

### 3-4. 산출물 점검 (Q1)
- `roi_candidates.json` ~13MB vs `roi_selected.json` ~15KB(~889×). candidates는 random arm(무작위 샘플 풀)·ControlNet train jsonl·exp1/2/6이 재소비 → 삭제 불가, 슬림화가 최적화 여지.

---

## 4. 정직성 / 방법론적 판단 (논문 서술 핵심)

### 4-1. 도메인-조건부 신호 (과대주장 금지)
- clean-bg 히스토그램 매칭은 **aitex에서만 강한 양성**, severstal/mtd는 ~random. → clean_bg 사전선정의 확정 가치는 **재현성·대칭대조·목적함수 강화 여지**로 서술하고, 일반적 mAP 향상은 GPU 실험 결과에 종속.

### 4-2. full-stack vs naive (교란이 아니라 의도된 대비)
- exp4v2 random arm은 **순수 무작위(무처리)**. AROMA vs random = **AROMA 파이프라인 전체의 기여**를 측정하는 의도된 대비(bg 정체성만 격리하는 tight control은 별도 `clean_bg_random_arm` 계측기).

### 4-3. parity는 총량까지만 — post-treatment control 회피
- arm 간 통제는 **총 학습 예산 동수**까지가 표준. 합성 결함의 **클래스 분포·라벨화 수율은 AROMA 선택·배치의 결과(post-treatment)**라, 강제 동일화는 **bad control**(AROMA의 정당한 이득 경로 제거 → null 편향). per-class stratified cap은 **리뷰어 요청 시에만** 도입(보존 스펙 존재). 현행은 계측·per-class AP 보고로 분해 제시.

### 4-4. dim 정밀도의 잔여 한계
- 실제 dim 컬럼 도입 전(구 profiling)은 grid fallback이라 edge-flush가 실제 가장자리보다 최대 ~63px 안쪽. 신 profiling(`b1bb497`)에서 완전 해소. 혼용 금지.

---

## 5. 논문 반영 포인트

1. **방법론 프레임**: 다중 도메인 데이터-유도 ROI 검출 + 문맥 인지 배치(무학습 기판). 자동화는 context/background 축 한정.
2. **데이터-유도 설계**: 3-신호 가중이 관측 lift에 비례해 자동 조정(균일 배경 셋은 크기 신호 자동 소거) — no-hardcoding의 구체 사례.
3. **정직한 조건부/음성 결과**: 히스토그램 매칭 도메인-조건부, deficit⟂availability, 배치 geometry-blind → 각각 (a)한계 명시, (b)생성 future work 정당화, (c)Phase3 기하 prior 개선으로 연결.
4. **재현성·통제 설계**: 사전선정으로 seed간 배치 분산 제거, 대칭 대조군, post-treatment control 회피(공정성 논증).
5. **무결성 수정**: image_id 고유키(비지도 클러스터링 정상화), 실제 dim(배치·스케일 정밀도) — 재현 가능한 파이프라인 근거.

## 6. 남은 작업

- **B0 Colab phase0 전체 재실행**(4종): 고유키 + dim 컬럼 흡수 → leather V1/V2 정상화, 전 데이터셋 dim 컬럼.
- **B1–B3**: step1→3.5→4 (AROMA arm + random arm) 신 profiling 기반 재생성.
- **B4 GPU exp4v2**: context-placement의 mAP 판정(핵심 미해결 — 헤드라인은 headroom 있는 데이터셋이 arbiter, near-ceiling(mtd) 단독 판정 금지).
- **B5**: aitex T1(유일 조건부 양성) 재확인.
- **선택 최적화**: roi_candidates 슬림 스키마.
- **리뷰어 대응 대기**: per-class stratified cap + 라벨화-후 parity(`aroma_exp4v2_perclass-parity-cap.md`).
- **C1(최종)**: 위 실험 완료 후 AROMA.txt 개선.
