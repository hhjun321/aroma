# exp_ablation — 3단계 leave-one-out Ablation의 downstream mAP 실행 (Colab)

> **목적**: `.claude/.dev_note/exp_ablation_execute.md`의 잔여 단계. 로컬 proxy 검증(severstal, 2026-08-18/19)으로 유의 분리가 확인된 4개 arm(full / A1 ROI-random / A2 BG-random / A3 Site-random)의 **downstream mAP 기여를 exp4v2 프로토콜로 분해**한다.
> **실행 환경**: 합성(STEP 1~3)은 CPU, exp4v2(STEP 4)는 GPU.
> **대상**: 우선 `severstal` 단독 (로컬 proxy와 동일 조건). 확장 시 A2는 `mvtec_leather` 우선 — §4.3 diversity-loss(15/400 collapse) 직접 검증 서사.
> **정직성**: 로컬 proxy 수치(hist∩·site_score)는 메커니즘 증거일 뿐이다. mAP 주장은 본 문서 STEP 4 결과로만 한다.

---

## Arm 정의 (dev_note와 동일)

| arm | Stage 1 (ROI) | Stage 2 (BG) | Stage 3 (Site) | 분리 방법 |
|---|---|---|---|---|
| full | compat | fitness | ring | 기존 step3→3.5→5 체인 그대로 (= 기존 aroma arm, **재합성 불필요 — 기존 synth_aroma_tobe 재사용**) |
| A1 | **random** | fitness | ring | `roi_selection --sampling_strategy random` 산출에 step3.5/5 체인 |
| A2 | compat | **random** | (position 無→랜덤) | step3.5의 `clean_bg_random_arm.json`을 `--clean_bg_json`으로 소비 |
| A3 | compat | fitness | **random** | `--site_selection off` 산출을 `--clean_bg_json`으로 소비 |
| all-random | random | random | random | 기존 random arm (= 기존 synth_random, 재합성 불필요) |

---

## STEP 0 — 공통 환경 셀 (sym_final 전 문서 동일 — 그대로 복사)

```python
import os, json

os.environ['DRIVE']          = '/content/drive/MyDrive/data/Aroma'
os.environ['AROMA_REF']      = '/content/AROMA'
os.environ['AROMA_SCRIPTS']  = '/content/AROMA/scripts/aroma'
os.environ['AROMA_OUT']      = f"{os.environ['DRIVE']}/aroma_output"
os.environ['DATASET_CONFIG'] = os.environ.get('DATASET_CONFIG', '/content/AROMA/dataset_config.json')
os.environ['SYM_ROOT'] = f"{os.environ['AROMA_OUT']}/sym_final"
def S(stage, ds=None):
    p = f"{os.environ['SYM_ROOT']}/{stage}"
    return f"{p}/{ds}" if ds else p

with open(os.environ['DATASET_CONFIG']) as f: CFG = json.load(f)
def normal_dir(ds): return CFG[ds]["image_dir"]

DS = "severstal"          # 본 문서는 단일 데이터셋 단위 실행
N_PER_ROI = 3             # ⚠️ step5 full run(NREP=3)과 동일해야 함 — 2026-08-19 2차 parity 결함: 2로 돌리면 pool 2000 < exp4v2 cap 2534라 full(3000→2534 소비) 대비 수량 confound
os.environ['DS'] = DS
os.environ['PROF']   = S('profiling', DS)
os.environ['NORMAL'] = normal_dir(DS)
os.environ['COMPAT'] = f"{S('profiling', DS)}/compatibility_matrix.json"

# τ — step4c 확정값 소비 (재스캔 금지, severstal ds_tau=0.1381)
import pathlib
tau = json.load(open(f"{S('compat_gate', DS)}/compat_tau_prescan_{DS}.json"))['ds_tau']
assert tau is not None and 0.0 < tau < 0.5
os.environ['TAU'] = str(tau)

# ablation 전용 산출 루트 (기존 roi/synth_* 절대 불변)
os.environ['ABL'] = f"{os.environ['SYM_ROOT']}/ablation"
print(f"DS={DS}  τ={tau}  ABL={os.environ['ABL']}")
```

> ⚠️ Drive의 profiling은 경로가 native라 **로컬 실행 때 필요했던 image_path/mask remap이 불필요**하다.

---

## STEP 1 — A1용 random ROI 선정 (CPU, ~수십 초)

기존 `roi/{ds}`는 건드리지 않고 ablation 루트에 새로 만든다. `--seed 42` 고정.

> ⚠️ **parity (2026-08-19 Colab 1차 실행에서 실측된 결함)**: `--top_k`는 반드시 **기존 `roi/{ds}/roi_selected.json`의 ROI 수와 동일**해야 한다. Drive의 sym_final full run은 **1000 ROI**(× n_per_roi 2 = 2000장)다. 로컬 검증값(top_k=200)을 그대로 쓰면 A1만 400장, A2/A3는 기존 1000-ROI 파일을 소비해 2000장이 되어 **arm 간 수량 parity가 깨진다** (1차 실행 실측: A1=400 vs A2=2000). top_k를 먼저 확인하고 맞출 것:

```python
import json
N_ROI = len(json.load(open(os.path.expandvars("$SYM_ROOT/roi/$DS/roi_selected.json"))))
os.environ['TOPK'] = str(N_ROI)
print(f"기존 full arm ROI 수 = {N_ROI} → A1 top_k 동일 적용")
```

```python
!python $AROMA_SCRIPTS/roi_selection.py \
    --profiling_dir $PROF \
    --prompts_dir   $SYM_ROOT/prompts/$DS \
    --output_dir    $ABL/roi_a1_random/$DS \
    --sampling_strategy random --top_k $TOPK --seed 42
```

> 검수: `roi_selected.json (<N_ROI>)` 저장 로그 — 수가 full arm과 동일해야 함. full arm의 ROI는 기존 `roi/{ds}/roi_selected.json`(compatibility)을 그대로 쓴다.

---

## STEP 2 — clean_bg_selection 2회 (CPU, ~2분/회)

```python
# (a) A1: random ROI에 AROMA bg+site 부여 (ring + qf run — step3.5 canonical)
!python $AROMA_SCRIPTS/clean_bg_selection.py \
    --profiling_dir $PROF \
    --roi_dir       $ABL/roi_a1_random/$DS \
    --output_dir    $ABL/roi_a1_random/$DS \
    --k_fit --site_selection ring --emit_random_arm \
    --site_quality_filter --image_dir $NORMAL

# (b) A3: full ROI에 bg fitness만 부여, site 미해결(position=None)
!python $AROMA_SCRIPTS/clean_bg_selection.py \
    --profiling_dir $PROF \
    --roi_dir       $SYM_ROOT/roi/$DS \
    --output_dir    $ABL/cbg_siteoff/$DS \
    --k_fit --site_selection off --emit_random_arm
```

> A2의 입력(`clean_bg_random_arm.json`)은 기존 `roi/{ds}`에 이미 있다(step3.5 `--emit_random_arm` 산출). 없으면 (b)와 같은 방식으로 ring run을 ablation 루트에 재실행해 얻는다.
> 검수: `Assigned backgrounds to 200 / 200`, (a)는 `ring_sgm 자리 …positions` 로그.

---

## STEP 3 — 합성 3 arm (CPU, ~5–15분/arm)

full·all-random은 기존 산출(`synth_aroma_tobe`, `synth_random`) 재사용 — 여기서는 A1/A2/A3만 만든다. 공통 플래그는 step5 copy_paste run과 동일.

```python
ARMS = [
    # (tag,            roi_dir,                    clean_bg_json)
    ("a1_roirand", f"$ABL/roi_a1_random/$DS", None),                                            # A1: 자기 clean_bg_selected 자동 로드
    ("a2_bgrand",  f"$SYM_ROOT/roi/$DS",      f"$SYM_ROOT/roi/$DS/clean_bg_random_arm.json"),   # A2
    ("a3_siteoff", f"$SYM_ROOT/roi/$DS",      f"$ABL/cbg_siteoff/$DS/clean_bg_selected.json"),  # A3
]
for tag, roi, cbj in ARMS:
    os.environ['ROI_D'] = os.path.expandvars(roi)
    os.environ['OUT_D'] = os.path.expandvars(f"$ABL/synth_{tag}/$DS")
    os.environ['CBJ']   = ("--clean_bg_json " + os.path.expandvars(cbj)) if cbj else ""
    print(f"\n===== ablation gen {tag} =====")
    !python $AROMA_SCRIPTS/generate_defects.py \
        --roi_dir     $ROI_D \
        --normal_dir  $NORMAL \
        --output_dir  $OUT_D \
        --method      copy_paste $CBJ \
        --n_per_roi 3 --seed 42 --blend_mode seamless \
        --reject-clean-bg --min-bg-quality 0.7 --bg-blur-threshold 100.0 \
        --compat_mode symmetric --compat_threshold $TAU \
        --compat_matrix_json $COMPAT
```

> 검수 (T = N_ROI × n_per_roi. sym_final 1000 ROI × 3 = **T=3000** — full `synth_aroma_tobe`와 동일해야 함. 2026-08-19 2차 실행이 n_per_roi=2로 돌아 2000이 나온 것이 parity 결함 2호. rep **상향**(2→3) 재실행은 copy_paste가 전량 재생성하지만 파일명이 superset(`_00`·`_01` 덮어씀 + `_02` 추가)이고 annotations.json도 전체 재작성이라 **디렉터리 비우기 불필요** — 반대로 rep을 **낮출** 때만 잔재가 남으므로 비워야 한다. 소요는 1차 실행의 ~1.5배):
> - 공통: `Generated T images (0 skipped)`, `clean_bg resolve: used=T fallback=0 mismatch=0`
> - A1: `position_source: ring≈T` (1차 실측 397/400 — ring-무효 자리 소수 폴백 정상) / A2·A3: `position_source: fallback=T`
> - A2: `repick_draws>0` + `placement gate exhausted` WARNING 소수 정상 (compat 게이트가 랜덤 배경 재추첨, 소진 시 마지막 후보 paste — Colab 실측 repick 140, exhausted 28/2000 = 1%). **exhausted가 수 % 를 크게 넘으면 τ 재확인**
>
> **STEP 4 진입 전 parity 검수 (필수)** — 4 arm 합성 수가 전부 동일해야 exp4v2 비교가 성립:
>
> ```python
> import json
> for tag in ["a1_roirand", "a2_bgrand", "a3_siteoff"]:
>     n = len(json.load(open(os.path.expandvars(f"$ABL/synth_{tag}/$DS/annotations.json"))))
>     print(tag, n)
> n_full = len(json.load(open(os.path.expandvars(f"$SYM_ROOT/synth_aroma_tobe/$DS/annotations.json"))))
> print("full", n_full)   # 4개 값 전부 동일(=3000)해야 함. ablation arm이 부족하면 n_per_roi=3으로 동일 셀 재실행(전량 재생성, 파일명 superset이라 dir 비우기 불필요). full보다 크면 해당 디렉터리 비우고 재합성
> # parity 기준은 pool ≥ exp4v2 cap(severstal 2534)이 4 arm 공통으로 성립하는 것 — full은 기존 학습에서 3000 중 2534를 소비했다

> ```

---

## STEP 4 — exp4v2 downstream mAP (GPU, 그룹 A 프로토콜, 3 seeds)

exp4v2의 condition 축은 `baseline/random/casda/aroma`로 고정이므로, **ablation arm은 `--aroma_synthetic_dir` 교체 + `--condition aroma` 단독 실행**으로 넣고 output_dir을 arm별로 분리한다. baseline·random·full-aroma는 기존 exp4v2 결과를 그대로 쓴다(재학습 금지 — 비교 기준 동결).

> ⚠️ **프로토콜 결함 3호 (2026-08-19/20 1차 GPU 실행 실측)**: 아래 커맨드에 그룹 A 플래그가 없던 판으로 실행되어 **exp4v2 기본값(epochs 50·batch 16·imgsz 256·patience 0·rect off·synth cap 미적용)으로 학습**됨 → Table 8 비교 무효. 1차 결과(A1 .3061 > A3 .3033 > A2 .2826, 상대 순위만 유효)는 `.claude/.dev_note/exp_ablation_execute.md` "downstream 상대 결과" 절에 보존. **현재 커맨드는 그룹 A 플래그 전체를 내장** — 기본값에 의존하는 인자가 하나도 없어야 한다.

```python
os.environ['YOLO_CACHE'] = f"{os.environ['AROMA_OUT']}/yolo_cache"
ABL_ARMS = ["a1_roirand", "a2_bgrand", "a3_siteoff"]

for tag in ABL_ARMS:
    os.environ['SYNTH'] = os.path.expandvars(f"$ABL/synth_{tag}")       # {dir}/{ds}/annotations.json 규약
    os.environ['OUT_E'] = os.path.expandvars(f"$ABL/exp4v2_ga_{tag}")   # ga = 그룹 A 판. 1차(기본값 판) exp4v2_{tag}/seed*와 분리 보존
    print(f"\n===== exp4v2 groupA {tag} =====")
    !python $AROMA_SCRIPTS/experiments/exp4_v2_supervised_detection.py \
        --model yolov8n \
        --condition aroma \
        --dataset_keys $DS \
        --class_mode multi \
        --aroma_synthetic_dir  $SYNTH \
        --random_synthetic_dir $SYM_ROOT/synth_random \
        --real_data_dir        $DRIVE \
        --output_dir           $OUT_E \
        --yolo_cache_dir       $YOLO_CACHE \
        --imgsz 640 \
        --val_frac 0.3 \
        --synth_ratio 1.0 \
        --baseline_epochs 100 \
        --patience 25 \
        --batch 128 \
        --cache ram \
        --rect \
        --workers 12 \
        --compile \
        --seeds 42 1 2 \
        --resume
```

> - **arm당 1회 호출 + `--seeds 42 1 2`** (per-seed 루프 아님) — 집계(mean/std/ci95)가 results.json에 자동 생성되고 resume가 seed 단위로 skip한다.
> - 플래그는 `exp4v2_execute.md` STEP 2-1(severstal 그룹 A)과 자구 동일. severstal은 고정 해상도(1600×256)라 `--compile` 안전. **소형셋 확장 시(kolektor/leather) batch16+patience0 강제 — batch128 collapse 회피, `--compile`은 mtd 금지.**
> - `--random_synthetic_dir`는 required라 전달하지만 `--condition aroma`만 돌므로 소비되지 않는다.

### STEP 4 종료 후 프로토콜 검증 셀 (필수 — 결함 1·2·3호 전부 "가정 ≠ 실상"에서 발생)

```python
import yaml, json, glob, os
for tag in ["a1_roirand", "a2_bgrand", "a3_siteoff"]:
    for seed in [42, 1, 2]:
        a = yaml.safe_load(open(os.path.expandvars(
            f"$ABL/exp4v2_ga_{tag}/_seeds/seed{seed}/{DS}/yolov8n/aroma/args.yaml")))
        assert a["epochs"] == 100 and a["imgsz"] == 640 and a["batch"] == 128 \
           and a["patience"] == 25 and a["rect"] is True, (tag, seed, a)
    m = json.load(open(os.path.expandvars(f"$ABL/exp4v2_ga_{tag}/exp4v2_results.json")))[DS]["yolov8n"]["aroma"]
    assert m["n_synth_train"] == 2534 and m["n_real_train"] == 2534, (tag, m["n_synth_train"])
    print(tag, "OK — protocol=groupA, cap=2534, n_seeds=", m["n_seeds"])
```

> **Provenance 확인** (학습 로그에서): A1은 `[Provenance] severstal/aroma: ring=2534 fallback=0` (ring 풀 2919 ≥ cap이라 성립해야 정상). A2·A3는 풀 전량이 fallback이라 uniform subsample — **confound 아님, position-source 구성 자체가 arm의 처치(treatment)다.**

---

## STEP 5 — 결과 취합

arm당 1개 results.json에 3-seed 집계가 이미 들어있다 (`--seeds 42 1 2` 단일 호출의 산출):

```python
import json, os
for tag in ["a1_roirand", "a2_bgrand", "a3_siteoff"]:
    m = json.load(open(os.path.expandvars(f"$ABL/exp4v2_ga_{tag}/exp4v2_results.json")))[DS]["yolov8n"]["aroma"]
    print(f"{tag:12s} mAP@0.5 = {m['map50']:.4f} ± {m['std']['map50']:.4f}  "
          f"per-seed={ {k: v['map50'] for k, v in m['per_seed'].items()} }")
```

비교표 (기존 값과 병합 — 그룹 A 프로토콜 동일이므로 병기 성립):

| Method | 구성 | mAP@0.5 |
|---|---|---|
| Baseline | real-only | 기존 Table 8 (.5033) |
| all-random | R+R+R | 기존 Table 8 (.5065) |
| A1 | **R**+A+A | 본 실행 |
| A2 | A+**R**+(**R** 연쇄) | 본 실행 |
| A3 | A+A+**R** | 본 실행 |
| full AROMA | A+A+A | 기존 Table 8 (.5197) |

### 해석 가정 (명문화 — 결과 보고 시 함께 기술)

1. **A2는 순수 Stage 2 제거가 아니다**: random 배경엔 ring 위치 정보가 없어 Stage 3도 연쇄 붕괴(`position_source: fallback=T`). full−A2 = **BG+site 결합 기여**. 순수 Stage 2 단독은 **A3−A2 산술 추정**이며, 이는 **stage 간 가산성 가정**을 요구한다 — 근거는 로컬 proxy의 가산성 실측(site_score 0.131→0.081(−site)→0.061(−site−bg), 독립·누적 분리, p<1e-12).
2. **full−A1 = ROI(crop) selection 기여, full−A3 = site resolution 단독 기여** — 이 둘은 연쇄 없이 깨끗함.
3. **subsample 구성 차이는 confound가 아니다**: aroma-계열 arm(A1)은 ring-우선 소비, A2·A3는 전량 fallback → uniform. position-source 구성이 곧 arm의 처치다.
4. **주장 한계**: n=3 seed — 유의성 주장은 per-seed 부호 일치(3/3) + 로컬 proxy 삼각측량 병기로 한정. 미세 격차(1차 실행의 A1↔A3 0.3pt 규모)는 방향성 언급도 하지 않는다.
5. 1차(기본값 프로토콜) 상대 결과 **A1 .3061 > A3 .3033 > A2 .2826**은 프로토콜 강건성의 보조 증거로만 인용 (절대치 인용 금지).

---

## 참조

- 설계·로컬 proxy 수치: `.claude/.dev_note/exp_ablation_execute.md`
- canonical 체인: `step3_5_execute.md`(clean_bg), `step5_execute.md`(generate 플래그·검수 로그)
- 로컬 산출: `D:/project/AROMA_DATASET/ablation_k200/severstal/` (proxy_metrics.json, analyze_smoke_arms.py)
