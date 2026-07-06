# ControlNet aroma arm 실행 가이드에 aitex(tiled/single) 데이터셋 추가

## (사용할 skills: feature-dev)

> 문서(`controlnet_aroma_arm_execute.md`) 다중 STEP(0/3~6/8/9) 수정 + aitex 전용 분기(경로·seed·imgsz·grayscale·이식정책)가 다른 3종과 상이 → 단일 관심사 아님, micro-fix 조건 위반 → feature-dev. 코드 변경은 없고 Colab 실행 가이드(.md)만 수정. 선행 컨텍스트: [[aroma_multidomain_aitex-mtd-integration]], [[aroma_selfcontained_p0-train-jsonl-builder]].

## 개요

`controlnet_aroma_arm_execute.md`는 현재 leather/mtd/severstal 3종만 대상으로 하고 aitex를 명시적으로 제외한다("aitex 제외 — baseline 학습 실패 상태"). aitex는 이후 `aitex_tiled_rerun_execute.md`로 타일링(256×256/stride128) + 단일클래스 전환하여 baseline이 정상화되었으므로, ControlNet aroma arm 실험에 편입한다.

핵심 차이: 다른 3종은 20260705 결과에서 baseline/random을 **이식**하고 aroma(ControlNet) arm만 학습한다. 그러나 **20260705에는 aitex가 없어 이식 소스가 존재하지 않는다.** 또한 이번 실험에서 aitex는 **aroma-CP(copy-paste) 비교를 하지 않기로 확정**했으므로, 이식할 이점(GPU 절약·기존 baseline 앵커)이 성립하지 않는다. 따라서 aitex는 **이식 없이 `--condition all`로 baseline/random/aroma-CN을 한 run에서 fresh 학습**한다 — 한 run 내 세 조건이 동일 seed·split을 공유하여 단일변수 비교가 내부적으로 완결된다.

### 확정 결정 (사용자)

- **aitex = aroma-CN 단독 확정**, aroma-CP 비교 **없음**.
- aitex는 **이식 스킵 + `--condition all` fresh 학습** (다른 3종은 기존 20260705 이식 유지).
- **seed = 1 2 42** — `aitex_tiled_rerun_execute.md`의 `1 2 43`은 오실행이므로 정정. aitex single 실험의 올바른 seed는 1 2 42.
- aitex는 single class(nc=1), fabric grayscale → **grayscale target ON**(leather처럼 OFF 아님).
- **imgsz 256** (타일 원해상도 1:1; 다른 3종은 640 rect).

---

## 영향도 분석

### 이 기능이 변경하는 상태
- 실행 가이드 문서 `controlnet_aroma_arm_execute.md`의 STEP 0/3~6/8/9 셀·표·설명.
- 코드/구성 변경 **없음** — 문서만 수정.

### 그 상태를 전제로 동작하는 기존 로직 (깨지면 안 됨)
- 기존 3종(leather/mtd/severstal)의 **이식 경로(STEP 8-A) 및 seed 42 1 2 규약은 그대로 유지**되어야 함. aitex를 `DATASETS` 리스트에 무분별하게 추가하면 STEP 8-A 이식 루프가 aitex를 20260705에서 찾다가 실패하거나 잘못 이식할 위험 → aitex는 이식 루프에서 **제외**하고 별도 셀로 분리해야 함.
- STEP 6의 `N_PER_ROI`, STEP 7의 `RANDOM_N` parity 딕셔너리 등 데이터셋별 상수에 aitex 키 누락 시 KeyError 위험 → aitex 포함 여부를 각 셀에서 일관되게 처리.
- STEP 9 비교표는 `ref[ds]["yolov8n"]["aroma"]`(copy-paste)를 참조 → aitex는 ref에 없으므로 aroma-CP 열 접근 시 KeyError → aitex는 비교표에서 CP 열 제외.

### delete/bulk 아님
- 문서에 aitex 분기를 **추가**만 함. 기존 3종 셀 삭제 없음.

---

## 수정 내용

대상 파일: `AROMA연구분석/colab_execute/controlnet_aroma_arm_execute.md`

### 1. 헤더/개요 — aitex 편입 반영
- "대상 데이터셋" 줄의 "(aitex 제외 — baseline 학습 실패 상태)"를 수정: aitex는 tiled/single 전환으로 편입, 단 **이식 없이 fresh(`--condition all`), seed 1 2 42** 임을 명시.
- 워크플로우 개요 표에 aitex 특수 처리(이식 스킵) 주석 추가.

### 2. STEP 0 — aitex 전용 경로 변수
- `DATASETS`에 aitex를 넣되, aitex의 `normal_dir`·profiling/roi 경로가 다름을 처리.
  - `AITEX_TILED = $DRIVE/aitex_tiled`, normal_dir = `$AITEX_TILED/train/good`.
  - profiling/roi는 표준 키(`profiling/aitex`, `roi/aitex`) 그대로 사용 가능(aitex_tiled_rerun이 이 경로에 산출).
  - `normal_dir(ds)`는 `dataset_config.json`의 aitex entry(`image_dir` = `.../aitex_tiled/train/good`)를 읽으므로 기존 `normal_dir()` 헬퍼로 해소되는지 확인. 해소되면 별도 분기 불요.
- TODO: aitex를 `DATASETS`에 통합할지, 별도 `DATASETS_GRAFT`(3종) / aitex 분리 리스트로 둘지 STEP별로 결정. STEP 3~6는 통합 가능, STEP 8은 분리 필수.

### 3. STEP 3 — build_train_jsonl (aitex 포함)
- aitex도 동일하게 `build_train_jsonl.py` 실행. 입력 경로(morphology/roi_candidates/context/config)는 `profiling/aitex`, `roi/aitex`.
- single class·grayscale은 build 단계에는 영향 없음(build는 target/hint/prompt만 생성). 그대로 3종과 동일 루프에 포함 가능.

### 4. STEP 4 — train_controlnet (aitex 차등)
- 데이터셋별 차등 표에 aitex 행 추가: 규모(수백 타일), epochs(mtd 유사 150 제안, TODO 확정), `--augment` ON, **grayscale target 기본 ON**(별도 플래그 불요), single class.
- aitex 전용 학습 셀 추가(다른 3종 셀과 병렬 구조).

### 5. STEP 5 — pilot (aitex 포함)
- pilot 루프에 aitex 포함. `GRAY_FLAG`는 leather만 `--cn_no_grayscale`이므로 aitex는 grayscale 유지(빈 플래그) — 기존 조건문 `ds == "mvtec_leather"` 그대로면 aitex 자동으로 grayscale ON.

### 6. STEP 6 — 전량 생성 (aitex 포함)
- generate 루프에 aitex 포함. `N_PER_ROI`에 aitex 키 추가(기존 합성 annotations 역산 셀이 aitex도 계산하도록).
- `GRAY_FLAG` 동일(aitex는 grayscale 유지).

### 7. STEP 7 — parity/cn_stats (aitex 포함)
- `RANDOM_N` 딕셔너리에 aitex 키 추가 — aitex random arm의 n_synth_train은 **aitex_tiled_rerun 결과**에서 확인해 기입(TODO: 실측값). 이식이 아니라 fresh 학습이므로 parity 개념은 "aroma-CN pool ≥ random pool"로 동일 적용.

### 8. STEP 8 — aitex 전용 fresh 학습 셀 (핵심 분기)
- **8-A 이식 루프에서 aitex 제외** — 이식 대상은 leather/mtd/severstal 3종만.
- **aitex 전용 별도 셀 신설**: `--condition all`, `--dataset_keys aitex`, `--seeds 1 2 42`, `--imgsz 256`, `--aroma_synthetic_dir $SYN_CN`(ControlNet 합성), `--random_synthetic_dir $AROMA_OUT/synthetic_random`, `--output_dir $EXP4V2_CN`. `--resume`는 이식이 없으므로 baseline/random/aroma 세 조건 모두 fresh 학습(seed당 3 run).
- 다른 3종의 8-B(이식 후 aroma만 학습, `--seeds 42 1 2`, `--imgsz 640 --rect`)와 **완전 분리된 셀**로 유지.
- ⚠️ aitex 셀의 파라미터(val_frac/synth_ratio/epochs/batch)는 `aitex_tiled_rerun_execute.md` §7-B와 **동일하게** 맞춰 tiled 재실행과 비교 가능성 유지(단 seed는 1 2 42로 정정).

### 9. STEP 9 — 비교표 aitex 행 (CP 열 제외)
- aitex는 aroma-CP가 없으므로 비교표에서 **aroma-CP / Δ(CN-CP) 열 제외**. aitex 행은 baseline·random·aroma-CN·Δ(CN-R)만 출력.
- 3종 표(CP 포함)와 aitex 표(CP 제외)를 분리하거나, ref 조회를 `ds != "aitex"` 가드로 감싸 KeyError 회피.

---

## 수정 대상 파일

- `AROMA연구분석/colab_execute/controlnet_aroma_arm_execute.md` (유일 수정 대상, 문서)

---

## 미확정 사항 (TODO)

- **STEP 4 aitex epochs/augment 값** — mtd(150, augment ON) 준용 제안이나 aitex 타일 규모에 맞춰 확정 필요.
- **STEP 7 `RANDOM_N[aitex]`** — aitex_tiled_rerun의 random arm n_synth_train 실측값 기입.
- **DATASETS 리스트 구조** — STEP 3~6는 4종 통합 루프, STEP 8만 분리가 최소 변경인지, 아니면 전 STEP에서 aitex를 상수/분기로 특수 처리할지 문서 작성 시 확정.
- **normal_dir(aitex) 해소 경로** — `dataset_config.json` aitex entry가 `aitex_tiled/train/good`을 가리키는지 확인(가이드 §3에서 이미 확인됨) → 기존 `normal_dir()` 헬퍼로 자동 해소 여부.

---

## 테스트

- 코드 변경 없음 → 빌드/유닛 테스트 불요(프로젝트 정책: 테스트 미작성).
- 검증은 Colab에서 문서 셀을 순서대로 실행하여:
  - STEP 3: aitex `train.jsonl` 라인 수 > 0.
  - STEP 8 aitex 셀: `--condition all`로 baseline/random/aroma 세 조건 모두 학습 로그 확인, seed 1 2 42.
  - STEP 9: aitex 행이 CP 열 없이 정상 출력(KeyError 없음).
