# AROMA Step 3.5 — clean_bg_selection 모듈 신설 (프로파일링 파생 파일 기반 clean-background 사전 선정)

## 사용할 skills: feature-dev

> **근거**: 신규 파일(`clean_bg_selection.py`) + 새 추상화(생성-시점 픽셀 스캔 → 프로파일링 파생 히스토그램 소싱으로 이관) + 신규 파이프라인 스테이지(step3.5) + `generate_defects.py` 통합 다수 + 신규 Colab 문서 → micro-fix 조건(단일 관심사·50줄 미만·새 파일 없음) 전부 위반 → **feature-dev**.

---

## 개요

현재 배경 선정은 **생성 시점에 raw good 픽셀을 직접 재스캔**한다(`generate_defects.py`: `_collect_normals` rglob+게이트, `_dv_bg_hist`로 픽셀에서 배경 셀 히스토그램 재계산, `_rank_normals`로 top-K 랭크, `_positive_place`로 위치 스캔). ROI마다·시드마다 픽셀을 다시 읽고 히스토그램을 재계산한다.

사용자 지적: 선정은 **원본 good을 직접 뒤지지 말고**, 이미지셋 분석으로 도출한 **프로파일링 파생 파일**(`context_features.csv`·`compatibility_matrix.json`)과 `roi_selected.json`을 활용해, `roi_selection.py`가 `profiling → roi_selected.json`을 만드는 것과 **대칭으로 한 번 precompute**해야 한다. 신규 `clean_bg_selection.py`가 `clean_bg_selected.json`을 산출하고, `generate_defects`는 이를 소비한다.

**정직성 제약 (반드시 유지):**
- **E1** (`AROMA연구분석/pivot_local_validation_20260711.md`): 현 히스토그램 매칭은 **도메인 조건부** — aitex lift +0.78(강함), severstal/mtd lift≈0(랜덤과 구분 불가). → 본 모듈의 **확실한 가치 = (a) 재현성 + 대칭 대조군(random arm에 동일 배경 배정) 구성 + per-seed placement variance 제거, (b) objective 업그레이드 지점.** **일반적 mAP 향상 주장 금지.**
- **E2**: 현 placement는 **geometry-blind**(mtd break 실제 edge 100%→배치 46.5%; leather 0%→41.7%). objective 상향 레버 = 배정 위치를 각 `class_key` 실제 edge/surface prior로 제약.
- **V2**: context 신호 전반 약함(I/H 1~4%) → objective 상향 상한 불확실, **가정 말고 검증**.

---

## 설계 원칙 (필수) — no-hardcoding / data-driven

**AROMA는 data-driven이 기본 원칙. 이 모듈의 어떤 임계·가중·경계·유형도 매직넘버로 하드코딩하지 않는다.**
- **임계값**(void/quality 컷, size-fit 비율, 히스토그램 top-K, class 적합도 컷 등)은 **관측 분포에서 유도**(예: 데이터셋별 percentile·분위수·분포 갭)한다. 불가피한 상수는 **CLI 인자 + 데이터 유도 기본값**으로 노출하고 근거를 기록한다.
- **3-기준 결합 가중치**는 고정 상수 금지 — 데이터에서 유도하거나(관측 분리도 기반) 최소한 데이터셋별 프로파일에서 산출. (roi_selection의 `roi_score` 가중치가 상수인 것을 답습하지 말 것.)
- **bg-type 유형·경계**(기준 3)도 고정 5-범주 하드코딩보다 **context feature 분포 기반 유도/클러스터링**을 우선 검토(불가 시 데이터셋 프로파일로 매핑).
- 승계하는 기존 파라미터(`--min-bg-quality` 등)도 **데이터 유도 기본값** 지향. 데이터셋별 산출값을 `clean_bg_selected.json` 메타에 기록해 auditable·재현 가능하게.

> 원칙 위반 리스크: `_POS_TOPK=8`, `min_bg_quality=0.7`, `cn_ar_threshold=2.5` 처럼 파이프라인에 이미 존재하는 고정 상수를 그대로 복제하지 말 것. 신규 임계는 프로파일링 통계에서 도출한다.

---

## 영향도 분석

### 이 기능이 변경/생성하는 상태
- 신규 산출물 `clean_bg_selected.json` (roi_dir 내) — generate_defects의 신뢰 소스가 됨.
- (선택) random-arm 배정 파일 — 대칭 대조군용.

### 그 상태를 전제로 동작하는 기존 로직 (생성-시점 선정 가정 붕괴 지점)
- `_collect_normals`(~2646, **void/품질 게이트 포함**) / `_rank_normals`(~1129) / `_positive_place`(~888) / `_dv_bg_hist`(~1174) / `_cell_hist`(~1152) / `_hist_intersection`(~1284) — 전부 "생성 시점에 픽셀·rng로 선정/필터"를 전제. json 경로에서 우회 필요(현행은 fallback 잔존). void/품질 필터링(§1-0)이 선행 단계로 이관되면 `_collect_normals`의 `reject_clean_bg` 게이트는 fallback으로만 남음.
- **rng 스트림/결정성**: 현 symmetric 경로는 ROI당 정확히 1 `rng.choice`(~3091, byte-identical stream 주석). 선정을 offline으로 옮기면 이 rng 소비가 사라져 **후속 rng 스트림 시프트** → 배치·feather 등 다운스트림 변동. 시드/소비 순서 재정렬로 결정성 재현 필요.
- `dv_hist_cache`/`dv_scored_cache`/`dv_pool_cache`(~2921–3094) 캐시 구조 — json 경로에선 불필요.

### "없음(0개)" 상태
- 게이트가 good 전부 reject / json missing → **silent 0-output 금지**, fallback으로 비어있지 않은 풀 확보.
- 호환 clean bg 없는 소스 defect → neutral/uniform 배정 fallback.

---

## 수정 내용

### 1. 신규 `scripts/aroma/clean_bg_selection.py` — roi_selection.py 미러

- **입력**:
  - `profiling/{ds}/context_features.csv` — `image_type=='good'` 행 = 배경 후보(per-image non-void cell 히스토그램 집계, `patch_xy`로 위치 후보). `image_type=='defect'` 행 = 소스 배경 분포(`_dv_bg_hist` 재구성).
  - `compatibility_matrix.json` — `bin_edges`로 5 context feature를 cell로 이산화, `matrix`.
  - `roi_selected.json` — 배경이 필요한 defect 목록(source key, class_key, cluster_id, cell_key).
- **출력** `clean_bg_selected.json`: ROI별(또는 (cluster/class,cell)별) 배정 `normal image_id` + optional `position`(cell region / `patch_xy`). 옵션 `--emit_random_arm`로 random-arm 배정 병행.
- **CLI**: `--profiling_dir --roi_dir --output_dir --seed --topk [--emit_random_arm --granularity]` — `_parse_args`/`main`/`run` 패턴 미러.
- **추출(refactor) 대상 함수** (generate_defects → 본 모듈, 픽셀 대신 context_features 소싱):
  - `_cell_hist`(1152) — good 이미지 non-void cell 분포
  - `_dv_bg_hist`(1174) — 소스 defect 배경 분포 (context_features defect 행에서 재구성)
  - `_hist_intersection`(1284)
  - `_rank_normals`(1129) — top-K 샘플
  - 위치는 `_positive_place`(888) 로직을 `patch_xy` 기반으로 재구성
  - → generate_defects의 원본 함수는 **fallback용으로 잔존**.

#### 1-0. 선행 — void/품질 필터로 유효 후보 풀 확정 (3-기준보다 먼저)

현재 `generate_defects.py::_collect_normals`(~2646)가 생성 시점에 raw good을 rglob·픽셀로 읽어 `_is_clean_background(min_quality=0.7, blur_threshold)`로 **void/flat/저품질(검거나 흐린) 배경 이미지를 제거**한다. 이 필터링을 **선행 단계로 이관**하여, 3-기준·랭킹 이전에 **유효 clean-bg 후보 풀을 먼저 확정**한다(픽셀 재스캔 없이 분석 파생 정보 사용).

- **소스(픽셀 대신)**: `context_features.csv`의 good 이미지 패치 features(`local_variance`/`edge_density` 등)로 void/flat 유도(저-variance·저-edge = void/flat), per-image 집계로 품질/void 판정. (또는 distribution_profiling이 per-good-image quality/void 플래그를 emit하도록 확장 — TODO 11.)
- **동작**: void/저품질 good 이미지를 후보에서 **선(先)배제** → 남은 풀에 대해서만 §1-a 3-기준·랭킹 수행. 전부 배제되면 게이트 우회 후 원본 풀 반환(`_collect_normals` 현행 all-reject fallback 정책 준수).
- **파라미터 승계**: `--reject-clean-bg`/`--min-bg-quality`/`--bg-blur-threshold`를 본 모듈 CLI로 승계(임계 일관). 결과를 `clean_bg_selected.json`(또는 별도 valid-pool)에 기록.
- ⚠️ `_is_clean_background`는 픽셀 기반 → context_features feature 유도치와 **불일치 가능**. 오프라인 유도가 픽셀 게이트를 충분히 재현하는지 검증 필요(TODO 11).

#### 1-a. 후보군 결정 — 3-기준 사전 필터/스코어 (합리적 후보군 판정)

랭킹(히스토그램 교차) **이전에**, 각 clean-bg가 해당 defect의 합리적 후보인지 다음 3기준으로 판정하여 후보 풀을 좁힌다. 전부 **profiling 파생 정보만** 사용(원본 픽셀 재스캔 금지).

1. **defect-type(class 라벨)별 적합도** — 배경이 이 결함 클래스에 맞는가.
   - 소스: `roi_selected.json`의 `class_key`(=defect_type) + `context_features.csv`의 defect 행을 **클래스별로 집계한 실제 배경 셀 분포**(class-conditioned `_dv_bg_hist`).
   - 판정: 후보 good 이미지의 셀 히스토그램이 **그 클래스의 실제 배경 분포**와 유사한지(히스토그램 교차). E1의 소스-단위 매칭을 **클래스-조건부**로 일반화. (E2의 class-conditioned 방향과 정합.)
   - ⚠️ V2 경고: context 신호 약함 → 클래스 적합도의 변별력이 도메인-조건부일 수 있음(aitex 강, severstal/mtd 약). 스코어로 쓰되 하드 컷 신중.

2. **결함 bbox 크기별 적합도 (size-fit)** — 이 크기 crop이 배경/청정영역에 물리적으로 들어가는가.
   - 소스: `roi_selected.json`의 `defect_bbox`(w,h) + 후보 배경 크기(= `context_features`의 `patch_xy` 그리드 최대 범위 + tile, 또는 profiling에 이미지 dim 기록 시 그 값).
   - 판정: crop (w,h)가 배경 dim(및 non-void 청정 영역 범위) 안에 들어가는 후보만 유효. 안 들어가면 후보 제외(또는 리스케일 가능성 별도). **하드 게이트** 성격(현 generate_defects의 oversized-crop 리스케일/스킵 로직과 정합).

3. **bg-type ↔ defect-type 매칭 적합도** — 이 배경 유형이 이 결함 유형에 적합한가.
   - 소스: **후보 good 이미지별 bg-type 라벨**(현재 미보존 → context_features 셀/features에서 유도해야 함; 또는 per-region `ctx_label` 활용) × 결함 클래스/subtype.
   - 판정: 기존 `SuitabilityEvaluator.matching_score(subtype, background_type)`(roi_selection.py:147)를 **데이터셋-단일 background_type → per-이미지 bg-type**으로 확장한 (defect-type × bg-type) 적합도. 부적합 유형 배경은 후보에서 배제/감점.
   - ⚠️ 현재 bg-type은 `--background_type` 데이터셋 단일값(choices: smooth/directional/periodic/organic/complex). **per-good-image bg-type 분류는 신규 유도 작업**(TODO 7).

> 후보군 결정 = (2) size-fit 하드 게이트 통과 → (3) bg-type 매칭 + (1) class 적합도 스코어로 후보 풀 확정 → 그 위에서 히스토그램 교차 랭킹·top-K 배정. 세 기준은 `clean_bg_selected.json`에 후보 판정 근거(class_fit/size_ok/bgtype_score)를 함께 기록해 auditable하게.

### 2. `scripts/aroma/generate_defects.py` — clean_bg_selected.json 소비

- `run`/copy_paste_synthesis(~2709) 에서 `roi_dir/clean_bg_selected.json` 로드 추가(`roi_selected.json` 로드 ~2774 옆).
- 선정 루프(~3062–3098): json 존재 시 ROI별 배정 image_id/position 조회 → `normal_path`/`stage2_pool` 대체, 생성-시점 랭킹(`_dv_hist_for`/`_rank_normals`, ~3076–3098) **우회**. json 없으면 현행 로직 **fallback**.
- 배정된 normal 픽셀은 여전히 **열어서** paste(seamless/alpha) — "**어느** normal·**어디**"만 json에서 옴.
- `_collect_normals`/`_rank_normals`/`_positive_place`는 fallback으로 유지.

### 3. 신규 파이프라인 스테이지 + Colab 문서
- profiling → roi_selection(step3) → **clean_bg_selection(step3.5, NEW)** → generate_defects(step4).
- 신규 `AROMA연구분석/colab_execute_new/step3_5_execute.md`.

---

## 수정 대상 파일

- 신규: `scripts/aroma/clean_bg_selection.py`
- 수정: `scripts/aroma/generate_defects.py` (로드 ~2774 인근, 선정 루프 ~3062–3098, fallback 유지)
- 신규: `AROMA연구분석/colab_execute_new/step3_5_execute.md`
- 참고(미수정, 구조 템플릿): `scripts/aroma/roi_selection.py`

---

## 테스트 / 검증

CLAUDE.md 정책: **pytest 금지** — Colab 실행 + 로컬 재측정.

1. **Colab step3.5 실행**: `clean_bg_selection.py`가 `clean_bg_selected.json` 생성(4종).
2. **오프라인 재현 검증**: `clean_bg_selected`의 히스토그램 교차 수치가 E1의 생성-시점 `_dv_bg_hist`/`_hist_intersection` 결과와 일치하는가 — **aitex lift +0.78 재현**(`AROMA연구분석/scripts_local/e1.py` 방식으로 재측정). 불일치 시 context_features 소싱 ↔ 픽셀 소싱의 discretization 갭 조사.
3. **대칭 대조군 실현성**: symmetric arm·random arm에 동일 배경 배정이 emit되고, generate_defects가 양 arm에서 json 소비로 동일 배경 사용하는지.
4. (E2 후속, 별도 범위) class_key edge/surface prior 배치 시 mtd/leather placement 개선 여부 — 본 노트 범위 밖, 검증 항목으로만 기록.

---

## TODO / 미확정

1. **Granularity**: per selected-ROI vs per (cluster/class, cell).
2. **Objective**: 현행 D_v 히스토그램 매칭 **추출 먼저**(재현 + 대칭 대조군 계측기) vs class-conditioned geometry prior(개선) — **추출-우선 권장**.
3. **Position precompute**: `context_features` `patch_xy`로 위치까지 precompute vs 이미지만 배정하고 위치는 생성 시점 유지.
4. **Random-arm 배정 emit**: 대칭 대조군용 random_arm clean_bg 배정 병행 출력 여부.
5. **선행 의존**: leather `image_id` stem-collision fix(`aroma_phase0_image-id-unique-key.md`) + phase0 재실행 — leather `context_features` image_id 충돌로 그 전엔 leather 부정확.
6. **전제**: copy-paste pivot 하 진행 (ControlNet은 future work로 강등).
7. **bg-type per-이미지 유도 (기준 3 선행)**: 현재 bg-type은 `--background_type` 데이터셋 단일값뿐 → 후보 good 이미지별 bg-type을 `context_features` 셀/features(또는 `ctx_label`)에서 분류·유도하는 방법 확정 필요. 유도 불가 시 기준 3은 데이터셋-단일 fallback.
8. **배경 크기 소스 (기준 2)**: 배경 dim을 `context_features` `patch_xy` 최대범위로 근사할지 vs profiling에 이미지 dim을 명시 저장하도록 distribution_profiling 확장할지.
9. **3-기준 결합 방식**: (2) 하드 게이트 + (1)(3) 가중 스코어 → 후보 풀. 가중치·컷오프는 데이터-드리븐(관측 분포)으로 정할지 고정할지. class 적합도(1)는 V2 약신호라 하드 컷 대신 소프트 스코어 권장.
10. **후보 근거 기록**: `clean_bg_selected.json`에 class_fit/size_ok/bgtype_score를 per-candidate로 남겨 auditable + 대칭대조 재현.
11. **void/품질 필터 오프라인화 (§1-0)**: `_is_clean_background`(픽셀) 게이트를 `context_features` feature 유도로 대체할지 vs distribution_profiling이 per-good-image quality/void 플래그를 emit하도록 확장할지. 유도치가 픽셀 게이트를 재현하는지 검증 필요. `--reject-clean-bg/--min-bg-quality/--bg-blur-threshold` 임계 승계.
