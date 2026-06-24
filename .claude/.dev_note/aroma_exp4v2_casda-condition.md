# Exp4v2 — CASDA를 4번째 조건으로 추가 (CASDA vs AROMA 공정 비교)

> 설계 근거: design workflow 산출. exp1 dev note(FID+PaDiM, exp4v2 이전)를 supervised mAP 패러다임으로 reconcile.
> 효과 주장은 multi-seed CI로 확정.

---

## (사용할 skills: feature-dev)

## 개요

CASDA vs AROMA를 **공정 비교**하기 위해 live harness `exp4_v2_supervised_detection.py`에 `casda` 조건을 추가한다. CASDA 자기 지표(detection mAP@0.5 + per-class AP, severstal nc=4)로 정면 비교. 모든 조건(random/casda/aroma)이 **동일 copy-paste 엔진**을 쓰고 **ROI 선택 전략만 다름** — AROMA의 ROI-modeling 기여 격리(AROMA 핵심 원칙).

**핵심 발견**: exp4v2 학습 코어가 condition-name-agnostic(baseline만 특례). CASDA 합성 피더(`generate_casda.py` → `casda_roi_adapter` → 동일 `generate_defects.run(copy_paste)`)는 이미 완성. → 한 파일 소수정으로 `casda` 조건 추가 가능.

**부 지표**: exp1 `--mode roi`(ROI 품질 5지표)로 *왜* AROMA가 이기는지 메커니즘 설명. FID/AD(PaDiM)는 제외.

## 확정 결정 (사용자)

1. **synth_ratio 동일** — 3 조건 동일 비율, 글로벌 적용. 동량 보장.
2. **n_per_roi 동일** — 3 생성기 동일값.
3. **seeds 3개** — multi-seed CI (예: 42 1337 2025).
4. **CASDA native (Option A)**: `min_suitability=0.5`(CASDA 기본), `per_class_cap=None`(무제한). class balance 통제 안 함 → CASDA 충실 재현. **per-class n_synth 로깅 필수** (c4/c2 starvation이 격차 만들었는지 투명화). reviewer 지적 시 per_class_cap ablation으로 후속 개선.
5. **부 지표 roi만** (FID/AD 제외).
6. **cross-dataset 안 함** — severstal multi-class 단독.

## 프레이밍 (overclaim 방지)

이 비교는 **공유 copy-paste 엔진 안에서 ROI 선택 전략 ablation** — CASDA 네이티브 파이프라인(Poisson Blending + ControlNet) 전체가 아님. 논문 표기: "CASDA ROI selection inside AROMA's shared synthesis engine". AROMA의 class_floor는 이제 AROMA 방법의 정당한 일부 → "AROMA의 ROI 전략(deficit-aware 선택 + class-balanced 할당)이 CASDA의 suitability 선택을 능가" 프레이밍.

---

## 영향도 분석

### 이 기능이 변경하는 상태
- `exp4v2_results.json`에 `casda` 조건 추가 (조건별 map50/per_class/n_synth_train).
- summary.md 표에 casda 열.

### 그 상태를 전제로 동작하는 기존 로직
- baseline/random/aroma 조건: **불변**. 학습 코어가 condition-agnostic(baseline만 특례), per-class·multi-seed·weights 로직 무수정.
- baseline/random은 `_pair_aware_allocation` 미진입 → 비교 무결성 유지.

### 공정성 (load-bearing)
- synth-count cap이 **literal `("random","aroma")` 튜플** 순회 (L1809 max_synth, L1853 synth_ratio). **casda를 양쪽 다 추가 안 하면 casda synth 무제한 → 더 많은 synth로 학습 = 은밀한 불공정.** 비협상 필수 수정.

---

## 수정 내용

### 1. `scripts/aroma/experiments/exp4_v2_supervised_detection.py` (단일 파일, ~6+1곳)

1. **L138** — `ALL_CONDITION_KEYS = ["baseline", "random", "casda", "aroma"]` (summary 표도 자동 포함).
2. **~L2583** — `--condition` choices에 `'casda'` 추가.
3. **~L2598** — CLI `--casda_synthetic_dir` 신규 (random/aroma 옆).
4. **L1801-1805** — `synth_by_cond`에 `"casda": _load_synth_annotations(casda_synthetic_dir, ds)` 추가.
5. **★L1809 + L1853 (공정성 핵심)** — cap 튜플 `("random","aroma")` → `("random","casda","aroma")` **양쪽 다**.
6. **threading** — `casda_synthetic_dir`를 `main()` → `run()` → `_run_detection_mode()` 시그니처+호출부(~2446-2534, ~1731-1754)로 전달.
7. **per-class n_synth 로깅 (결정 4)** — multi 모드서 조건별 synth의 class별 카운트를 로그 및/또는 results에 기록. `_parse_severstal_class(source_roi)`로 class 집계 → 결과 JSON에 `n_synth_per_class` 또는 로그 라인. (c4/c2 starvation 투명화 목적)

선택(I/O 가속, correctness 아님): /tmp 스테이징 튜플(~942/988/992) + `_local_cache_for_yolo` 시그니처에 casda 추가.

**무수정**: `_run_yolo_condition`, `_write_yolo_labels`, `_parse_severstal_class`, weights 분기, `_aggregate_seeds`, multi-class 모드 — 전부 condition-agnostic.
**무수정 (재사용)**: `generate_casda.py`, `casda_roi_adapter.py`, `generate_defects.py` — 완성 검증됨.

### 2. `AROMA연구분석/colab_execute/exp4v2_execute.md` (또는 신규 CASDA 가이드)

CASDA Stage A + generate_casda 셀 추가:
- **Cell A — CASDA Stage A** (CPU, clone + ROI 추출):
  ```python
  !git clone --single-branch -b approve https://github.com/hhjun321/CASDA.git /content/CASDA
  !python $SCRIPTS/extract_rois.py --image_dir $TRAIN_IMAGES --train_csv $TRAIN_CSV \
      --output_dir $ROI_DIR --roi_size 256 --min_suitability 0.5
  # → $ROI_DIR/roi_metadata.csv + images/*.png
  ```
- **Cell B — CASDA synth (동일 엔진)**:
  ```python
  !python $AROMA_SCRIPTS/generate_casda.py --metadata_csv $ROI_DIR/roi_metadata.csv \
      --normal_dir $NORMAL_DIR --output_dir $AROMA_OUT/synthetic_casda \
      --n_per_roi $N_PER_ROI --seed 42
  ```
  (adapter는 min_suitability=0.5, per_class_cap 미지정=None 기본 → 결정 4)
- **exp4v2 run** (4 조건):
  ```python
  !python $AROMA_SCRIPTS/experiments/exp4_v2_supervised_detection.py \
      --condition all --dataset_keys severstal --class_mode multi \
      --random_synthetic_dir $AROMA_OUT/synthetic_random \
      --casda_synthetic_dir  $AROMA_OUT/synthetic_casda \
      --aroma_synthetic_dir  $AROMA_OUT/synthetic \
      --synth_ratio 1.0 --n_per_roi 3 --seeds 42 1337 2025 \
      --imgsz 640 --val_frac 0.3 --baseline_epochs 100 --patience 20 \
      --batch 64 --cache ram --rect --yolo_cache_dir $AROMA_OUT/yolo_cache --resume
  ```

> CASDA Drive 루트 `/content/drive/MyDrive/data/Severstal` (AROMA `.../data/Aroma`와 다름). 둘 다 Colab 마운트. (colab-execution.md env 참조)

---

## Feasibility 게이트 (Colab/Drive)

| 전제 | 상태 |
|------|------|
| CASDA clone in Colab (`/content/CASDA`, branch approve) | ✅ |
| Stage A `extract_rois.py` (CPU, Stage B/C/D ControlNet/GPU **불필요**) | ✅ |
| **`roi_metadata.csv` + crops on Drive** | ★**blocking — Stage A 1회 실행 필요** |
| CASDA ROI class 보유(`_class{N}_` → `_parse_severstal_class` 매칭) | ✅ per-class AP 무배선 |
| AROMA측 자산(adapter/generate_casda/generate_defects) | ✅ 완성 |

**단일 hard 전제**: severstal에 대해 CASDA Stage A 미실행이면 1회 실행해 `roi_metadata.csv` + crops를 Drive에 생성 (adapter와 동일 Colab 세션 권장 — 절대경로 drift 방지).

---

## 수정 대상 파일

- `scripts/aroma/experiments/exp4_v2_supervised_detection.py` (L138, ~1801-1805, **1809 & 1853**, ~2583/2598, ~1731-1754/2446-2534, per-class synth 로깅)
- `AROMA연구분석/colab_execute/exp4v2_execute.md` (CASDA Stage A + generate_casda 셀 + 4조건 run)
- 재사용(무수정): `generate_casda.py`, `casda_roi_adapter.py`, `generate_defects.py`, `exp1_casda_comparison.py`(`--mode roi` 부지표)

---

## 암묵적 요구사항 / 엣지

| 상황 | 처리 |
|------|------|
| casda를 cap 튜플에 누락 | casda synth 무제한 → 불공정. **양쪽(1809/1853) 필수** |
| n_synth 조건간 불일치 | synth_ratio 동일 + n_per_roi 동일로 동량. 결과에 `n_synth_train` 조건별 보고로 parity 증명 |
| c4/c2 starvation (CASDA native) | 정당한 발견(조작 아님). per-class n_synth 로깅으로 명시 |
| adapter 0 ROI 통과 | RuntimeError (기존 동작) |
| roi_image_path 절대경로 drift | Stage A + adapter 동일 세션, adapter `n_missing_image≈0` 확인 |
| background_type 'textured' 등 미정의 | 경고 후 cell_key로 사용(기존) |
| summary 표 | L138 확장으로 casda 자동 포함 |

---

## 테스트 (Colab, pytest 금지)

1. **공정성 parity**: 결과 JSON에서 random/casda/aroma의 `n_synth_train` **동일** 확인. 다르면 cap 튜플 누락 의심.
2. **per-class 투명성**: 조건별 per-class synth 카운트 로그/JSON 확인 (CASDA c4/c2 starve 여부).
3. **adapter sanity**: `casda_roi_adapter` 로그 `n_missing_image≈0`, class별 ROI 수 c3/c4 포함 확인.
4. **결과**: per-condition map50 + per-class AP(c1..c4) + 3-seed CI.

**성공 기준** ("AROMA > CASDA on Severstal"):
- AROMA mAP@0.5 > CASDA, **3-seed 95% CI 비겹침**, **동일 n_synth_train**.
- AROMA per-class AP ≥ CASDA, 특히 **소수 클래스 c3·c4** (CASDA H4 주장 영역).
- (뒷받침) exp1 `--mode roi`서 AROMA rare_pair/morphology coverage 우위.

---

## 미확정 / 후속 (TODO)

- reviewer가 "deficit 기여만 분리" 지적 시 → CASDA에 `per_class_cap` 부여한 Option B ablation 추가 (현재 안 함).
- exp1 `--mode roi` 부지표 실행 셀은 exp1_execute.md 기존 활용.
- per-class n_synth 로깅의 정확한 출력 위치(로그 vs JSON 필드)는 구현 시 기존 로깅 패턴 따라 결정.

---

## CASDA mask 매핑 (공정성) — 후속 수정

> 위 dev_note는 `casda_roi_adapter.py`를 "재사용(무수정)"으로 기재했으나, 공정성 검토 중 **합성 단계의 비공정성**이 발견되어 adapter를 수정함. 본 절이 그 변경을 기록한다.

### 발견한 격차 (ellipse fallback)

`generate_defects.py`의 copy_paste(`use_real_mask` 게이트, ~L350-384)는 roi_entry가 **`defect_mask_path`(파일 존재) + `defect_bbox`(4-int, crop-relative, bounds 통과)** 를 모두 보유할 때만 **실제 결함 마스크**를 정밀 paste한다. 둘 중 하나라도 없거나 bounds 검증 실패 시 → crop 전체에 **합성 타원(ellipse) 마스크**로 fallback.

- random/aroma의 roi_entry(`roi_selection.py` 산출)는 `defect_mask_path` + `defect_bbox`를 보유 → **정밀 paste**.
- 기존 `casda_roi_adapter.adapt()` 출력 dict는 image_path/cluster_id/cell_key/roi_score/priors/deficit/prompt/morph_label/ctx_label만 emit하고 **mask 필드는 없었음** → CASDA만 **타원 blob을 paste**.
- 결과: "동일 copy-paste 엔진, ROI 선택만 다름"이라는 공정성 원칙 위반. CASDA는 합성 품질 자체가 다른 상태로 비교됐을 것.

### 수정 (roi_mask_path → defect_mask_path + crop-relative defect_bbox)

`casda_roi_adapter.py`에서 CSV `roi_mask_path`를 `defect_mask_path`로 매핑하고, crop-relative `defect_bbox`를 산출해 emit:

- **`defect_mask_path`** = CSV `roi_mask_path` (crop과 정렬된 마스크). 경로 비었거나 디스크 부재 시 mask 필드 omit → ellipse fallback.
- **`defect_bbox` = [x, y, w, h]** (top-left + width/height), **crop 좌표계**. 산출 방식:
  - 주 경로(`_mask_derived_bbox`): crop-aligned 마스크의 nonzero(≥128) 영역 bounding box. `x=xs.min(), y=ys.min(), w=xs.max()-xs.min()+1, h=ys.max()-ys.min()+1`. 마스크 차원으로 clamp. all-zero 마스크 → None → ellipse fallback(`n_empty_mask`).
  - 차선(`_csv_fallback_bbox`, PIL/numpy 부재 시만): CSV `defect_bbox`(원본 1600×256 "(x1,y1,x2,y2)") − `roi_bbox` origin → crop-relative [x,y,w,h], crop 크기로 clamp.
  - **CSV `defect_bbox` 컬럼을 직접 쓰지 않음**: 원본 이미지 좌표 + (x1,y1,x2,y2) 포맷이라 crop bounds 검증을 통과 못 하고 조용히 ellipse fallback함.
- **size-mismatch 가드**: generate_defects는 image와 mask를 동일 box로 crop하고 mask를 resize하지 않으므로, crop≠mask 크기면 마스크 정렬이 깨짐 → 크기 불일치 시 mask drop(`n_size_mismatch`). mask-derived bbox가 crop bounds 안에 있음이 이 가드로 보장됨.
- mask 필드는 **둘 다 유효할 때만 emit** — 부분/실패는 깔끔히 ellipse fallback. 진단 로그에 `n_with_mask / n_missing_mask / n_empty_mask / n_size_mismatch / n_csv_fallback` 집계.

### Stage A 산출물 재사용 (Cell A 생략)

CASDA Stage A(`extract_rois.py`)는 **이미 1회 실행되어 Drive에 `roi_metadata.csv` + `roi_patches_v5.1/{images,masks}/*.png`가 존재**(사용자 확정: CSV가 참조하는 모든 `roi_image_path` AND `roi_mask_path` 파일이 Drive에 존재). 따라서 이번 실행에서는 **Cell A(CASDA clone + Stage A) 생략**, 기존 `roi_metadata.csv`를 그대로 재사용 → `generate_casda` → 4조건 exp4v2.

### scope 정정 (검토 지적 반영)

본 작업은 `casda_roi_adapter.py` 외에 **`exp4_v2_supervised_detection.py`도 수정**됨(+130/-15). 이는 위 dev_note 본문 "수정 내용 §1"에 계획된 `casda` 4번째 조건 추가(`ALL_CONDITION_KEYS`, `--casda_synthetic_dir`, `_resolve_synth_class` cluster_id fallback, per-class synth 카운팅, `_local_cache_for_yolo` casda 스테이징, `_aggregate_seeds` n_synth_per_class)와 동일 작업이다. 변경은 **순수 additive + `--casda_synthetic_dir` gated**(생략 시 no_synth 거부)이며 random/aroma의 label-writing·caching·capping 출력은 byte-identical → 회귀 없음. 단, adapter 구현 보고가 "단일 파일만 수정"으로 기재한 것은 부정확 — 두 파일 모두 본 effort의 일부임.
