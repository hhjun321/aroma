# sym_final 파이프라인 정본 스펙 (SPEC) — 전 문서 공유

> 이 파일은 `colab_execute_new/`의 모든 실행 가이드(phase0/step1~4/exp3/exp4v2/exp5/exp6)가 **공유하는 정본**이다. 각 문서는 아래 §1 공통 환경 셀을 **그대로 복사**하고, 자기 스테이지에 해당하는 §3 명령만 사용한다. **env·output 루트 규약을 문서마다 재발명 금지**.

## 0. 최종 규격 요약 (aroma-sym)

`exp4v2_mtd_symmetric_execute.md`를 정본 규격으로 삼는다:
- **selection**: `roi_selection.py --sampling_strategy deficit_aware --score_mode realism` (multi 3종은 class 게이트 추가, aitex single은 미사용)
- **생성(AROMA arm)**: `generate_defects.py --method controlnet` + `--compat_mode symmetric --compat_threshold <τ>` (SGM matrix_symmetric + 64px 타일링 query + positive placement) + clean-bg 게이트 + `--blend_mode seamless`. aitex는 elongated용 AR/텍스처 게이트 추가.
- **random arm**: `generate_random.py` (통제, 동일 clean-bg 게이트)
- **데이터셋 준비(step -1)**: `prepare_datasets_execute.md` — v2-1 4종을 AROMA 레이아웃으로 준비 + `dataset_config.json` 등록. phase0 앞. (mvtec_leather는 Drive에 표준 레이아웃으로 이미 존재 → 준비 불요, 경로 확인만.)
- **생성 선결 준비(step4)**: `build_train_jsonl.py` + `train_controlnet.py`(ControlNet 학습) **+ τ 사전스캔**(`compat_gate_cpu_diagnosis §10`). step5 생성이 소비하는 CN 모델·τ를 이 단계에서 모두 확정. step4는 step0·step3 산출(profiling·roi_candidates)을 입력으로 쓰므로 step3 뒤·step5 앞.
- **실행 순서 체인 = prepare_datasets(step -1) → phase0 → step1 → step2 → step3 → step4(CN 학습+τ) → step5(생성) → exp\***
- **다운스트림**: `exp4_v2_supervised_detection.py` — **fresh 전조건 학습**(baseline/random/aroma 모두 처음부터, graft 미사용)
- **데이터셋**: v2-1 4종 `severstal · mvtec_leather · mtd · aitex`. **aitex = tiled(256×256/stride128, single-class)**.

## 1. 공통 환경 셀 (모든 문서 STEP 0에 그대로)

```python
import os, json

# ===== 공통 환경 (sym_final 전 문서 동일 — 수정 금지) =====
os.environ['DRIVE']          = '/content/drive/MyDrive/data/Aroma'
os.environ['AROMA_REF']      = '/content/AROMA'
os.environ['AROMA_SCRIPTS']  = '/content/AROMA/scripts/aroma'
os.environ['AROMA_OUT']      = f"{os.environ['DRIVE']}/aroma_output"
os.environ['AROMA_DATA']     = f"{os.environ['DRIVE']}"
os.environ['DATASET_CONFIG'] = os.environ.get('DATASET_CONFIG', '/content/AROMA/dataset_config.json')
# ===== 단일 버전 루트 (stage-first: {stage}/{ds}) =====
os.environ['SYM_ROOT'] = f"{os.environ['AROMA_OUT']}/sym_final"
os.environ['CN_MODELS'] = f"{os.environ['SYM_ROOT']}/controlnet_models"   # ControlNet 학습본(step4 산출, step5 소비)
def S(stage, ds=None):
    p = f"{os.environ['SYM_ROOT']}/{stage}"
    return f"{p}/{ds}" if ds else p

DATASETS = ["severstal", "mvtec_leather", "mtd", "aitex"]   # v2-1 4종
with open(os.environ['DATASET_CONFIG']) as f: CFG = json.load(f)
def normal_dir(ds): return CFG[ds]["image_dir"]                 # aitex → aitex_tiled/train/good
def is_multi(ds):   return CFG[ds].get("class_mode") == "multi" # aitex=single (자동)
```

## 2. output 루트 규약 (stage-first — 반드시 준수)

전 산출물은 `sym_final/{stage}/{ds}` 아래. **stage-first**여야 `--dataset_keys` 다중 + `--aroma_synthetic_dir`/`--roi_dir_root` 루트 규약(root/{ds})과 정합한다(ds-first면 exp3/5/6 깨짐).

| stage | 경로 `S(stage, ds)` | 생성 문서 | 소비 문서 |
|-------|--------------------|-----------|-----------|
| `profiling` | `sym_final/profiling/{ds}` | phase0 | step1/2/3/4, exp* |
| `complexity` | `sym_final/complexity/{ds}` | step1 | step2 |
| `prompts` | `sym_final/prompts/{ds}` | step2 | step3 |
| `roi` | `sym_final/roi/{ds}` | step3 | step5/step4, exp6(rare) |
| `cn_data` | `sym_final/cn_data/{ds}` | step4 | step4 |
| `controlnet_models` | `sym_final/controlnet_models/{ds}/best_model` | step4 | step5 |
| `compat_gate` | `sym_final/compat_gate/{ds}` | step4(τ 사전스캔) | step5 |
| `synth_aroma` | `sym_final/synth_aroma/{ds}` | step5 | exp4v2/exp3/exp5/exp6 |
| `synth_random` | `sym_final/synth_random/{ds}` | step5 | exp4v2/exp3/exp5/exp6 |
| `exp4v2` | `sym_final/exp4v2` | exp4v2 | — |
| `exp3` | `sym_final/exp3` | exp3 | — |
| `exp5` | `sym_final/exp5` | exp5 | — |
| `exp6` | `sym_final/exp6` | exp6 | — |
| `embed_cache` | `sym_final/embed_cache/{ds}` | exp5/exp6 (최초 임베딩 추출 시) | exp5·exp6 공유 (DINOv2 캐시 재사용) |

- **CN 모델**은 `sym_final/controlnet_models/{ds}/best_model`(step4 산출). step5 선결 assert.
- exp* 루트 인자: `--aroma_synthetic_dir $(S('synth_aroma'))` `--random_synthetic_dir $(S('synth_random'))` `--roi_dir_root $(S('roi'))` — 스크립트가 `/{ds}`를 붙임.

## 3. 스테이지별 정본 명령 (핵심 = 변경된 부분)

### phase0 — distribution_profiling (+ symmetric 키 검증)
```python
for DS in DATASETS:
    os.environ['DS'] = DS; os.environ['PROF'] = S('profiling', DS)
    # ⚠️ distribution_profiling.py 는 scripts/(루트)에 있음 — $AROMA_SCRIPTS(scripts/aroma) 아님
    !python $AROMA_REF/scripts/distribution_profiling.py \
        --dataset_config $DATASET_CONFIG --dataset_key $DS \
        --output_dir $PROF --num_workers -1
```
- 검증: `compatibility_matrix.json`에 `matrix_symmetric`·`P_def_patch`·`clean_dist`·`symmetric_epsilon` 존재 assert(없으면 `--compat_mode symmetric` hard-fail). drift 대비 기존 profiling 백업.

### step1 — compute_complexity
```python
!python $AROMA_SCRIPTS/compute_complexity.py \
    --profiling_dir $PROF --output_dir $CPLX --weight_mode equal --local_staging
```

### step2 — prompt_generation
```python
!python $AROMA_SCRIPTS/prompt_generation.py \
    --profiling_dir $PROF --complexity_dir $CPLX --output_dir $PROMPTS
```

### step3 — roi_selection (aroma-sym selection)
```python
# 3종(multi):
!python $AROMA_SCRIPTS/roi_selection.py \
    --profiling_dir $PROF --prompts_dir $PROMPTS \
    --sampling_strategy deficit_aware --score_mode realism \
    --top_k 200 --img_diversity_cap 1 \
    --class_mode multi --class_floor --per_pair_cap_frac 0.05 \
    --output_dir $ROI
# aitex(single): 위에서 --class_mode multi --class_floor --per_pair_cap_frac 0.05 제거
```

### step4 — ControlNet 학습 (step5 선결; 순서상 step3 뒤·step5 앞)
```python
# 4a. 학습 데이터 빌드 (CPU)
!python $AROMA_SCRIPTS/build_train_jsonl.py \
    --morphology_csv $PROF/morphology_features.csv \
    --roi_candidates $ROI/roi_candidates.json \
    --context_features $PROF/context_features.csv \
    --config $PROF/recommended_config.yaml \
    --output_dir $CN_DATA --style technical         # CN_DATA = S('cn_data', DS)
# 4b. 학습 (GPU, 데이터셋당 1세션)
!python /content/AROMA/scripts/train_controlnet.py \
    --data_dir $CN_DATA --output_dir $CN_MODELS/$DS \
    --mixed_precision fp16 --gradient_checkpointing \
    --num_train_epochs <E> [--augment] [--no_force_grayscale_target] \
    --early_stopping_patience <P> --checkpointing_steps 500 \
    --save_optimizer_state --save_fp16 --resume_from_checkpoint latest
```
데이터셋별 차등(controlnet_aroma_arm STEP 4 규약):

| ds | epochs | --augment | grayscale target |
|----|--------|-----------|------------------|
| severstal | 100 | OFF | 기본 ON |
| mtd | 150 | ON | 기본 ON |
| aitex (tiled) | 150 | ON | 기본 ON (mtd 준용) |
| mvtec_leather | 300 | ON | **OFF** (`--no_force_grayscale_target`) |
> 산출: `$CN_MODELS/{ds}/best_model/`. 세션 끊김 → 동일 셀 재실행(`--resume_from_checkpoint latest`).

```python
# 4c. τ 사전스캔 (compat_gate_cpu_diagnosis §10) — step5 생성 선결. DS별.
#     출력 S('compat_gate',ds)/compat_tau_prescan_{ds}.json 의 ds_tau. τ=0.5 금지. 신 profiling(matrix_symmetric) 필수.
#     (aitex 전용) 추가로 AR/텍스처 게이트 임계 사전스캔(controlnet_aroma_arm STEP 5-0/5-0b) → AR_T/TEX_T 확정.
```
> τ·AR·TEX는 step5가 소비하므로 **step4 말미에서 모두 확정**한다(생성 직전 재발명 금지).

### step5 — AROMA arm 생성 (controlnet + symmetric)
```python
out = f"{S('synth_aroma', DS)}"   # exp4v2가 /{ds} 붙이므로 실제 생성은 이 경로에 그대로
!python $AROMA_SCRIPTS/generate_defects.py \
    --roi_dir     $ROI \
    --normal_dir  $NORMAL \
    --output_dir  {out} \
    --method      controlnet \
    --controlnet_path  $CN_MODELS/$DS/best_model \
    --morphology_csv   $PROF/morphology_features.csv \
    --context_features $PROF/context_features.csv \
    --config           $PROF/recommended_config.yaml \
    --n_per_roi 3 --seed 42 --blend_mode seamless \
    --reject-clean-bg --min-bg-quality 0.7 --bg-blur-threshold 100.0 \
    --compat_mode symmetric --compat_threshold $TAU \
    --compat_matrix_json $PROF/compatibility_matrix.json \
    [aitex 전용] --cn_ar_threshold $AR_T --texture-dist-threshold $TEX_T
```
> **주의**: `--aroma_synthetic_dir`가 exp4v2에서 `S('synth_aroma')`이고 `/{ds}`를 붙이므로, 생성 `--output_dir`는 반드시 `S('synth_aroma', DS)`(=`sym_final/synth_aroma/{ds}`)여야 한다. mvtec_leather는 `--cn_no_grayscale` 추가(컬러 가죽). aitex AR/텍스처 임계는 사전스캔+pilot 확정.

### step5 (계속) — random arm 생성 (통제)
```python
!python $AROMA_SCRIPTS/generate_random.py \
    --candidates_json $ROI/roi_candidates.json \
    --normal_dir      $NORMAL \
    --output_dir      {S('synth_random', DS)} \
    --top_k 200 --n_per_roi 3 --seed 42 \
    --reject-clean-bg --min-bg-quality 0.7 --bg-blur-threshold 100.0
```

### exp4v2 — downstream detection (fresh 전조건)
2 그룹(파라미터 상이). **graft 미사용, `--condition all` fresh**.
```python
# 3종 (multi, 640/rect/100, seeds 42 1 2)
!python $AROMA_SCRIPTS/experiments/exp4_v2_supervised_detection.py \
    --model yolov8n --condition all --dataset_keys severstal mvtec_leather mtd --class_mode multi \
    --random_synthetic_dir $(S('synth_random')) --aroma_synthetic_dir $(S('synth_aroma')) \
    --real_data_dir $AROMA_DATA --output_dir $(S('exp4v2')) --yolo_cache_dir $AROMA_OUT/yolo_cache \
    --imgsz 640 --val_frac 0.3 --synth_ratio 1.0 --baseline_epochs 100 --patience 50 \
    --batch 64 --cache ram --rect --seeds 42 1 2 --resume
# aitex (single, 256/no-rect/300, seeds 1 2 42) — --class_mode 미지정, --rect 미사용
!python ... --dataset_keys aitex --imgsz 256 --baseline_epochs 300 --seeds 1 2 42 (--rect 없음) ...
```

### exp3 / exp5 / exp6 — 입력 루트만 sym_final로 교체
legacy 명령 구조 유지, `--dataset_keys severstal mvtec_leather aitex mtd`, 그리고:
- `--aroma_synthetic_dir $(S('synth_aroma'))` `--random_synthetic_dir $(S('synth_random'))`
- exp6 rare: `--roi_dir_root $(S('roi'))`
- `--output_dir` = `S('exp3')`/`S('exp5')`/`S('exp6')`, embed_cache = `S('embed_cache')` 공유.

## 4. 데이터셋 규약

| ds | class_mode | normal_dir | exp4v2 파라미터 |
|----|-----------|-----------|-----------------|
| severstal | multi | dataset_config | imgsz 640, rect, 100ep, seeds 42 1 2 |
| mvtec_leather | multi | dataset_config | imgsz 640, rect, 100ep, seeds 42 1 2. 생성 시 `--cn_no_grayscale` |
| mtd | multi | dataset_config | imgsz 640, rect, 100ep, seeds 42 1 2 |
| aitex | **single (tiled)** | `aitex_tiled/train/good` | imgsz 256, **no rect**, 300ep, seeds 1 2 42. 생성 시 AR/텍스처 게이트 |

## 5. 공통 무결성 / 정직 (전 문서 말미)

- 사후 튜닝 금지(τ·seed·synth_ratio·epochs 결과 보고 후 변경 금지).
- prescan 필수: τ·AR·텍스처 임계는 CPU 사전스캔 확정값. mtd 값 aitex 무검증 전용 금지.
- aitex는 tile-level·single-class → 절대값 타 데이터셋 직접 비교 금지, Δ만 유효.
- 테스트 코드 신규 작성·pytest 금지(CLAUDE.md). 검증은 Colab 실행으로.
- `--local_staging`: CPU copy_paste/random·complexity엔 사용 가능. **ControlNet 생성엔 미사용**(sidecar 캐시 Drive 직결 필요).
