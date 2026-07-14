# MVTec zipper(multi) · transistor(single) — headroom baseline 확인 가이드

> **목적(확인용)**: 로스터 편입 **전에** 두 신규 후보의 **real-only YOLO baseline mAP**를 측정해 **near-ceiling 여부(headroom)** 판정. AROMA ROI 선택은 (1) 배경 compat 레버(CCI)와 (2) baseline headroom 둘 다 있어야 신호가 난다. CCI 는 profiling 으로 확인됨(zipper 0.472 강 · transistor 0.317 중); **headroom 은 complexity 로 안 나오므로 여기서 실측**한다.
> **실행 환경**: GPU (A100 권장). baseline 은 real-only 라 합성(synth) 불필요.
> **후보 배정 근거(확정)**: zipper → **multi**(7타입, 타입당 16~19 균형, 119 defect, CCI 0.472 최고) · transistor → **single**(4타입이나 타입당 10개로 multi 하기엔 희소 → nc=1 collapse, CCI 0.317 중레버).

---

## 0. 원리 (재구성 0)

`exp4_v2_supervised_detection.py`의 generic mvtec 분기가 `{real_data_dir}/mvtec/{category}`의 `test/{type}`를 모두 `test_defect`로 수집하고, 마스크는 `ground_truth/{type}/{stem}_mask.png`로 해석한다(MVTec 네이티브 레이아웃, `prepare_*` 재구성 불요). `--class_mode`만 토글:
- transistor → `--class_mode single`(기본) → 라벨 전부 `class_id=0`, **nc=1**.
- zipper → `--class_mode multi` → 폴더명 열거, **nc=7**.

**baseline 은 `--condition baseline`** — real train split 만 학습, synth 미사용(합성 생성 선행 불요).

> ⚠️ **patience 0(full epoch) 사용**: aitex 에서 `--patience 25`가 데이터 굶은 baseline 을 조기중단해 붕괴(0.204±0.114)시킨 전례가 있다. headroom 은 **신뢰 가능한 천장값**이 필요하므로 여기서는 early-stop 없이 full epoch 학습한다.

---

## 1. 환경변수

```python
import os
# DRIVE 는 세션 표준(_SPEC §1)과 동일하다고 가정
os.environ['DRIVE']         = '/content/drive/MyDrive/data/Aroma'
os.environ['AROMA_REF']     = '/content/AROMA'
os.environ['AROMA_SCRIPTS'] = '/content/AROMA/scripts/aroma'
os.environ['AROMA_OUT']     = f"{os.environ['DRIVE']}/aroma_output"
os.environ['AROMA_DATA']    = f"{os.environ['DRIVE']}"      # {AROMA_DATA}/mvtec/{cat} 를 소비
os.environ['BCHECK_OUT']    = f"{os.environ['AROMA_OUT']}/sym_final/exp4v2_baseline_check"
os.environ['YOLO_CACHE']    = f"{os.environ['AROMA_OUT']}/yolo_cache"
# --aroma/random_synthetic_dir 는 argparse 필수 인자다(baseline-only 여도).
# baseline 은 synth 를 로드하지 않으므로 기존 sym_final 루트를 지정만 하면 무해
# (zipper/transistor annotations 부재해도 condition=baseline 에선 참조 안 함).
os.environ['SYNTH_AROMA']  = f"{os.environ['AROMA_OUT']}/sym_final/synth_aroma"
os.environ['SYNTH_RANDOM'] = f"{os.environ['AROMA_OUT']}/sym_final/synth_random"

for k in ('AROMA_DATA','BCHECK_OUT','YOLO_CACHE','SYNTH_AROMA','SYNTH_RANDOM'):
    print(f"{k:<12} {os.environ[k]}")
```

## 2. 전제 확인 — real 이미지 + 마스크 존재

```python
import os, glob
base = f"{os.environ['AROMA_DATA']}/mvtec"
for c in ('zipper','transistor'):
    tg = len(glob.glob(f"{base}/{c}/train/good/*"))
    types = [t for t in os.listdir(f"{base}/{c}/test") if t!='good' and os.path.isdir(f"{base}/{c}/test/{t}")]
    ndef = sum(len(glob.glob(f"{base}/{c}/test/{t}/*")) for t in types)
    gt   = os.path.isdir(f"{base}/{c}/ground_truth")
    print(f"{c:<11} train_good={tg:4}  types={len(types)}  defect_imgs={ndef:4}  ground_truth={'OK' if gt else 'MISSING'}")
```

> 기대: zipper train_good=240 types=7 defect=119 / transistor train_good=213 types=4 defect=40. `ground_truth=OK`(마스크→val GT)여야 mAP 산출 가능.

## 3. baseline 실행 — zipper (multi, nc=7)

```python
!python $AROMA_SCRIPTS/experiments/exp4_v2_supervised_detection.py \
    --model yolov8n --condition baseline \
    --dataset_keys mvtec_zipper \
    --class_mode multi \
    --aroma_synthetic_dir  $SYNTH_AROMA \
    --random_synthetic_dir $SYNTH_RANDOM \
    --real_data_dir $AROMA_DATA \
    --output_dir    $BCHECK_OUT \
    --yolo_cache_dir $YOLO_CACHE \
    --imgsz 640 --val_frac 0.3 \
    --baseline_epochs 100 --patience 0 \
    --batch 128 --workers 12 --cache ram \
    --seeds 42 \
    --resume
```

## 4. baseline 실행 — transistor (single, nc=1)

```python
!python $AROMA_SCRIPTS/experiments/exp4_v2_supervised_detection.py \
    --model yolov8n --condition baseline \
    --dataset_keys mvtec_transistor \
    --aroma_synthetic_dir  $SYNTH_AROMA \
    --random_synthetic_dir $SYNTH_RANDOM \
    --real_data_dir $AROMA_DATA \
    --output_dir    $BCHECK_OUT \
    --yolo_cache_dir $YOLO_CACHE \
    --imgsz 640 --val_frac 0.3 \
    --baseline_epochs 100 --patience 0 \
    --batch 128 --workers 12 --cache ram \
    --seeds 42 \
    --resume
```

> **공유 `--output_dir` 안전**: ds 키가 달라(`mvtec_zipper` / `mvtec_transistor`) 결과 키 충돌 없음. `--resume`가 완료분 skip.
> transistor 는 `--class_mode` 생략 = single 기본(nc=1). `--rect` 미사용(MVTec 정사각).
> `--compile` 미사용(불필요 + mtd dynamo 붕괴 전례 회피).
> seed 1개(42) = headroom 빠른 확인용. 최종 로스터 편입 시 42 1 2 로 확장.

## 5. headroom 판정

```python
import json, os
r = json.load(open(f"{os.environ['BCHECK_OUT']}/exp4v2_results.json"))
for ds in ('mvtec_zipper','mvtec_transistor'):
    cell = r.get(ds,{}).get('yolov8n',{}).get('baseline',{})
    m = cell.get('map50'); m95 = cell.get('map50_95')
    print(f"{ds:<18} baseline map50={m}  map50_95={m95}")
```

**판정 기준** (headroom):

| baseline map50 | 해석 | arbiter 적합성 |
|---|---|---|
| ≳ 0.90 | near-ceiling (mtd·leather형) | ❌ placement 이득 측정 불가 → flat 예상 |
| ~0.5–0.85 | 충분한 headroom | ✅ AROMA 신호 측정 가능 |
| ≲ 0.3 | 과소 or 미수렴 | ⚠️ patience 0 인데도 낮으면 데이터/난이도 확인 |

> CCI(레버)와 **교차**: headroom 있고(≤0.85) **CCI 있는**(zipper 0.472 / transistor 0.317) 셋이 유효 arbiter. 둘 중 near-ceiling 이면 로스터에서 제외 판단.
> ⚠️ aitex 교훈: baseline 이 비정상적으로 낮고 seed 편차 크면 under-train 의심 — 여기서는 patience 0 라 그 confound 는 제거됨.

---

## 6. 다음 단계 (headroom 통과 시)

baseline 확인 후 유효 판정된 셋에 대해:
1. **로스터 통합 방식 확정**(leather 교체 vs 추가) — 미결.
2. 합성 파이프라인 선행: `prepare`(dataset_config 등록) → profiling(완료됨) → `generate_defects`(synth_aroma/synth_random) → exp4v2 3조건(seeds 42 1 2, batch128/patience25/workers12).
3. zipper=multi / transistor=single 로 exp4v2 실행.

> 본 문서는 **headroom 확인 전용**(확인용). 정식 실행 가이드는 통합 방식 확정 후 별도 작성.

---

## 7. dataset_config 등록 (완료)

`dataset_config.json`의 `mvtec_zipper`·`mvtec_transistor` 엔트리는 **이미 존재**(domain/image_dir/seed_dirs). 이번에 `class_mode`만 추가:
- `mvtec_zipper`: `"class_mode": "multi"` (7타입)
- `mvtec_transistor`: `"class_mode": "single"` (nc=1)

```python
import json
CFG = json.load(open('/content/AROMA/dataset_config.json'))
for c in ('mvtec_zipper','mvtec_transistor'):
    print(c, '->', CFG[c].get('class_mode'), '| image_dir:', CFG[c]['image_dir'])
```

> exp4v2 의 `mvtec_*` 분기는 `{real_data_dir}/mvtec/{category}`를 **직접** 해석하므로 real 데이터에 대한 config 의존은 없다. `class_mode`는 `is_multi(ds)` 자동판정·생성 파이프라인용 정합 필드.

## 8. prepare — 재구성 없음

MVTec zipper/transistor 는 정사각 네이티브 레이아웃(`train/good`, `test/{type}`, `ground_truth/{type}/{stem}_mask.png`) → **aitex 같은 타일링/재구성 불요**. §2 전제확인 통과로 prepare 완료 간주.

## 9. 생성 파이프라인 (self-contained, 순서대로 실행)

**baseline 은 real-only(§3-4 완료). `random`·`aroma` arm 은 synth 필요** → zipper/transistor 는 미생성이므로 아래 체인을 순서대로 실행해 `synth_aroma`/`synth_random`을 만든 뒤 §10 exp4v2 3-arm 을 돌린다.

체인: **step2(prompt) → step3(roi) → step3_5(clean_bg) → step4c(τ 사전스캔) → step5(생성 copy_paste + random)**. step1(profiling/complexity)은 ✅ 완료(업로드됨). **copy-paste 전용(논문 정합) → step4 의 ControlNet 학습(4a/4b) SKIP, 4c τ-스캔만.**

### 9-0. 공통 환경 (`_SPEC §1` 규약)

```python
import os, json, pathlib
os.environ['DRIVE']          = '/content/drive/MyDrive/data/Aroma'
os.environ['AROMA_REF']      = '/content/AROMA'
os.environ['AROMA_SCRIPTS']  = '/content/AROMA/scripts/aroma'
os.environ['AROMA_OUT']      = f"{os.environ['DRIVE']}/aroma_output"
os.environ['AROMA_DATA']     = f"{os.environ['DRIVE']}"
os.environ['DATASET_CONFIG'] = os.environ.get('DATASET_CONFIG', '/content/AROMA/dataset_config.json')
os.environ['SYM_ROOT'] = f"{os.environ['AROMA_OUT']}/sym_final"
def S(stage, ds=None):
    p = f"{os.environ['SYM_ROOT']}/{stage}"
    return f"{p}/{ds}" if ds else p

DATASETS = ["mvtec_zipper", "mvtec_transistor"]   # 신규 2종만
with open(os.environ['DATASET_CONFIG']) as f: CFG = json.load(f)
def normal_dir(ds): return CFG[ds]["image_dir"]
def is_multi(ds):   return CFG[ds].get("class_mode") == "multi"   # zipper=multi, transistor=single
for ds in DATASETS: print(ds, '->', 'multi' if is_multi(ds) else 'single')
```

### 9-1. step2 — prompt 생성 (CPU)

```python
for DS in DATASETS:
    os.environ['DS']=DS; os.environ['PROF']=S('profiling',DS); os.environ['CPLX']=S('complexity',DS); os.environ['PROMPTS']=S('prompts',DS)
    assert pathlib.Path(f"{os.environ['CPLX']}/complexity_report.json").exists(), f"{DS} complexity 없음(step1 선행)"
    print(f"\n===== prompts: {DS} =====")
    !python $AROMA_SCRIPTS/prompt_generation.py \
        --profiling_dir  $PROF \
        --complexity_dir $CPLX \
        --output_dir     $PROMPTS
```

### 9-2. step3 — roi_selection (CPU) — multi/single 분기

```python
# 공통: --sampling_strategy deficit_aware --score_mode realism --top_k 200 --img_diversity_cap 1
# multi(zipper)만 --class_mode multi --class_floor --per_pair_cap_frac 0.05 추가.
# transistor(single)는 그 플래그 없이 = 전역 compat top-k (single-class path, --class_mode 기본 single).
for DS in DATASETS:
    os.environ['DS']=DS; os.environ['PROF']=S('profiling',DS); os.environ['PROMPTS']=S('prompts',DS); os.environ['ROI']=S('roi',DS)
    print(f"\n===== roi_selection: {DS} ({'multi' if is_multi(DS) else 'single'}) =====")
    if is_multi(DS):
        !python $AROMA_SCRIPTS/roi_selection.py \
            --profiling_dir $PROF --prompts_dir $PROMPTS \
            --sampling_strategy deficit_aware --score_mode realism \
            --top_k 200 --img_diversity_cap 1 \
            --class_mode multi --class_floor --per_pair_cap_frac 0.05 \
            --output_dir $ROI
    else:
        !python $AROMA_SCRIPTS/roi_selection.py \
            --profiling_dir $PROF --prompts_dir $PROMPTS \
            --sampling_strategy deficit_aware --score_mode realism \
            --top_k 200 --img_diversity_cap 1 \
            --output_dir $ROI
```

### 9-3. step3_5 — clean_bg_selection (CPU) — random arm 동시 생성

```python
for DS in DATASETS:
    os.environ['DS']=DS; os.environ['PROF']=S('profiling',DS); os.environ['ROI']=S('roi',DS)
    print(f"\n===== clean_bg_selection: {DS} =====")
    !python $AROMA_SCRIPTS/clean_bg_selection.py \
        --profiling_dir $PROF \
        --roi_dir       $ROI \
        --output_dir    $ROI \
        --emit_random_arm
```

> `--emit_random_arm` → `clean_bg_random_arm.json`(대칭 대조군). void floor 는 기본(p15) — 필요시 `--void_floor_pct`/`--var_floor`/`--edge_floor`로 데이터셋별 핀(step3_5_execute.md §옵션 참고).

### 9-4. step4c — τ 사전스캔 (CPU) — ⚠️ 필수, ControlNet 학습 SKIP

τ(compat_threshold)는 `S('compat_gate',DS)/compat_tau_prescan_{DS}.json`의 `ds_tau`(0<τ<0.5)로 확정된다. **generate_defects 가 `--compat_mode symmetric`에서 τ 없으면/0.5면 hard-fail.** copy-paste 전용이라 step4 의 4a(build_train_jsonl)·4b(ControlNet 학습, GPU)는 **건너뛰고 4c τ-스캔만** 실행한다.

> **실행 방법**: `colab_execute_new/step4_execute.md`의 **§STEP 4c 셀 [4c-0]~[4c-5]** 를 `DS='mvtec_zipper'`로 한 번, `DS='mvtec_transistor'`로 한 번 실행한다(각 DS당 5셀 순차). 그 τ-스캔 코드(≈120줄, symmetric 스케일 + 64px 타일링-aware, R=0.25 percentile)는 게이트와 동일 경로를 재현해야 하므로 **정본 셀을 그대로 재사용**한다(여기 전사 시 drift 위험 → 정본 참조가 안전). aitex 전용 AR/TEX 스캔은 **해당 없음**(신규 2종 무관).

```python
# 확정 확인 (두 DS 모두 ds_tau 로드돼야 §9-5 진행)
TAU_BY_DS={}
for DS in DATASETS:
    p=f"{S('compat_gate',DS)}/compat_tau_prescan_{DS}.json"
    tau=json.load(open(p)).get('ds_tau') if pathlib.Path(p).exists() else None
    assert tau is not None and 0.0<tau<0.5, f"{DS}: τ 미확정({tau}) — step4c 재확인(τ=0.5/None 금지)"
    TAU_BY_DS[DS]=float(tau); print(f"{DS} ds_tau={tau}")
```

### 9-5. step5 — AROMA arm 생성 (copy_paste, CPU)

```python
for DS in DATASETS:
    os.environ['DS']=DS; os.environ['ROI']=S('roi',DS); os.environ['NORMAL']=normal_dir(DS)
    os.environ['OUT']=S('synth_aroma',DS)
    os.environ['COMPAT']=f"{S('profiling',DS)}/compatibility_matrix.json"
    os.environ['TAU']=str(TAU_BY_DS[DS])
    print(f"\n===== AROMA copy_paste gen {DS} (τ={os.environ['TAU']}) =====")
    !python $AROMA_SCRIPTS/generate_defects.py \
        --roi_dir     $ROI \
        --normal_dir  $NORMAL \
        --output_dir  $OUT \
        --method      copy_paste \
        --n_per_roi 3 --seed 42 --blend_mode seamless \
        --reject-clean-bg --min-bg-quality 0.7 --bg-blur-threshold 100.0 \
        --compat_mode symmetric --compat_threshold $TAU \
        --compat_matrix_json $COMPAT \
        --local_staging
```

> 로그 필수 신호: `clean_bg assignment ON` + `compat gate ON: threshold=… mode=symmetric` + `placement-gate stats fallback<50%`. `matrix_symmetric` 없으면 hard-fail(phase0 symmetric emit 확인). CN 전용 인자(`--controlnet_path`/`--morphology_csv`/`--cn_*`) **미사용**. `--config` 불요(bin_edges 가 compat matrix 에 포함).

### 9-6. step5 — random arm 생성 (naive baseline, CPU)

```python
for DS in DATASETS:
    os.environ['DS']=DS; os.environ['ROI']=S('roi',DS); os.environ['NORMAL']=normal_dir(DS); os.environ['OUT_R']=S('synth_random',DS)
    print(f"\n===== random gen {DS} =====")
    !python $AROMA_SCRIPTS/generate_random.py \
        --candidates_json $ROI/roi_candidates.json \
        --normal_dir      $NORMAL \
        --output_dir      $OUT_R \
        --top_k 200 --n_per_roi 3 --seed 42 \
        --local_staging
```

> random arm 은 naive placement 기본 ON → clean-bg 게이트 플래그 **넘기지 않는다**(uniform-random 위치, grounding 없음 = 의도된 placement 비대칭 ablation).

### 9-7. parity 확인 (exp4v2 진입 전)

```python
for DS in DATASETS:
    for lbl, root in [("aroma", S('synth_aroma',DS)), ("random", S('synth_random',DS))]:
        ann=pathlib.Path(root)/"annotations.json"
        n=sum(1 for a in json.load(open(ann)) if not a.get('dry_run') and a.get('bbox')) if ann.exists() else 0
        print(f"{DS:<18} {lbl:<7} labelable={n}")
```

> 두 arm 모두 **labelable > 0** 이어야 exp4v2 유효. 0이면 해당 step5 재확인(τ 과대·clean_bg fallback 등).

## 10. exp4v2 3-arm 실행 (§9 생성 완료 후)

```python
os.environ['SYNTH_AROMA']  = S('synth_aroma')     # 루트 — 스크립트가 /{ds} 붙임
os.environ['SYNTH_RANDOM'] = S('synth_random')
os.environ['EXP4V2_OUT']   = f"{S('exp4v2')}_mvtec_new"   # 기존 4셋과 분리(resume 오염 방지)
os.environ['YOLO_CACHE']   = f"{os.environ['AROMA_OUT']}/yolo_cache"

# zipper — multi
!python $AROMA_SCRIPTS/experiments/exp4_v2_supervised_detection.py \
    --model yolov8n --condition all \
    --dataset_keys mvtec_zipper --class_mode multi \
    --aroma_synthetic_dir  $SYNTH_AROMA \
    --random_synthetic_dir $SYNTH_RANDOM \
    --real_data_dir $AROMA_DATA \
    --output_dir    $EXP4V2_OUT \
    --yolo_cache_dir $YOLO_CACHE \
    --imgsz 640 --val_frac 0.3 --synth_ratio 1.0 \
    --baseline_epochs 100 --patience 25 \
    --batch 128 --workers 12 --cache ram \
    --seeds 42 1 2 --resume

# transistor — single (--class_mode 생략 = single 기본)
!python $AROMA_SCRIPTS/experiments/exp4_v2_supervised_detection.py \
    --model yolov8n --condition all \
    --dataset_keys mvtec_transistor \
    --aroma_synthetic_dir  $SYNTH_AROMA \
    --random_synthetic_dir $SYNTH_RANDOM \
    --real_data_dir $AROMA_DATA \
    --output_dir    $EXP4V2_OUT \
    --yolo_cache_dir $YOLO_CACHE \
    --imgsz 640 --val_frac 0.3 --synth_ratio 1.0 \
    --baseline_epochs 100 --patience 25 \
    --batch 128 --workers 12 --cache ram \
    --seeds 42 1 2 --resume
```

> `--rect`·`--compile` 미사용(MVTec 정사각 + dynamo 붕괴 회피). batch128/patience25/workers12 = 현 exp4v2 표준.
> ⚠️ 이 3-arm 의 baseline 은 patience 25 로 재학습되어 §3-4(patience 0) headroom 확인값과 **다르다** — 3-arm 내부 baseline 을 정본으로 쓰고 혼용 금지.
> 판정: aroma vs random per-seed Δ(mAP50 + mAP50-95), 부호 일관성. zipper(headroom+고CCI)가 주 arbiter, transistor 보조.
