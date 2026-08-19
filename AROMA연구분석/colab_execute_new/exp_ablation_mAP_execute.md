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
N_PER_ROI = 2             # step5와 동일 — arm 간 수량 parity 필수
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
        --n_per_roi 2 --seed 42 --blend_mode seamless \
        --reject-clean-bg --min-bg-quality 0.7 --bg-blur-threshold 100.0 \
        --compat_mode symmetric --compat_threshold $TAU \
        --compat_matrix_json $COMPAT
```

> 검수 (T = N_ROI × n_per_roi. sym_final 1000 ROI 기준 T=2000 — Colab 1차 실행 실측치):
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
> print("full", n_full)   # 4개 값 전부 동일해야 함. 불일치 arm은 출력 디렉터리 비우고 재합성
> ```

---

## STEP 4 — exp4v2 downstream mAP (GPU, 3 seeds)

exp4v2의 condition 축은 `baseline/random/casda/aroma`로 고정이므로, **ablation arm은 `--aroma_synthetic_dir` 교체 + `--condition aroma` 단독 실행**으로 넣고 output_dir을 arm별로 분리한다. baseline·random·full-aroma는 기존 exp4v2 결과를 그대로 쓴다(재학습 금지 — 비교 기준 동결).

```python
SEEDS = [42, 1, 2]
ABL_ARMS = ["a1_roirand", "a2_bgrand", "a3_siteoff"]

for tag in ABL_ARMS:
    for seed in SEEDS:
        os.environ['SYNTH'] = os.path.expandvars(f"$ABL/synth_{tag}")     # {dir}/{ds}/annotations.json 규약
        os.environ['OUT_E'] = os.path.expandvars(f"$ABL/exp4v2_{tag}/seed{seed}")
        os.environ['SEED']  = str(seed)
        print(f"\n===== exp4v2 {tag} seed={seed} =====")
        !python $AROMA_SCRIPTS/experiments/exp4_v2_supervised_detection.py \
            --condition aroma \
            --dataset_keys $DS \
            --aroma_synthetic_dir  $SYNTH \
            --random_synthetic_dir $SYM_ROOT/synth_random \
            --real_data_dir $DRIVE \
            --output_dir    $OUT_E \
            --seed $SEED --resume
```

> - epochs·imgsz·batch 등 나머지 하이퍼파라미터는 **기존 severstal exp4v2 run과 동일 값**을 명시 전달할 것 (기존 exp4v2_execute 문서의 severstal 값 확인 — 프로토콜 divergence는 곧 비교 무효).
> - `--random_synthetic_dir`는 required라 전달하지만 `--condition aroma`만 돌므로 소비되지 않는다.
> - 소형셋 확장 시(kolektor/leather) batch16+patience0 강제 — batch128 collapse 회피.
> - resume는 per-seed JSON — 세션 끊기면 동일 셀 재실행.

---

## STEP 5 — 결과 취합

```python
import json, glob, statistics
rows = {}
for tag in ["a1_roirand", "a2_bgrand", "a3_siteoff"]:
    vals = []
    for seed in [42, 1, 2]:
        p = os.path.expandvars(f"$ABL/exp4v2_{tag}/seed{seed}/exp4v2_results.json")
        r = json.load(open(p))
        vals.append(r[DS]["aroma"]["yolov8n"]["map50"])   # 실제 키 구조는 파일서 확인
    rows[tag] = (statistics.mean(vals), statistics.stdev(vals))
for k, (m, s) in rows.items():
    print(f"{k:12s} mAP@0.5 = {m:.4f} ± {s:.4f}")
```

비교표 (기존 값과 병합):

| Method | 구성 | mAP@0.5 |
|---|---|---|
| Baseline | real-only | 기존 Table 8 |
| all-random | R+R+R | 기존 Table 8 |
| A1 | **R**+A+A | 본 실행 |
| A2 | A+**R**+(rand pos) | 본 실행 |
| A3 | A+A+**R** | 본 실행 |
| full AROMA | A+A+A | 기존 Table 8 |

해석 축: full−A1 = ROI selection 기여, full−A2 = BG assignment(+site 연쇄) 기여, full−A3 = site resolution 단독 기여. n=3 시드라 유의성 주장은 방향성+로컬 proxy 병기로 한정.

---

## 참조

- 설계·로컬 proxy 수치: `.claude/.dev_note/exp_ablation_execute.md`
- canonical 체인: `step3_5_execute.md`(clean_bg), `step5_execute.md`(generate 플래그·검수 로그)
- 로컬 산출: `D:/project/AROMA_DATASET/ablation_k200/severstal/` (proxy_metrics.json, analyze_smoke_arms.py)
