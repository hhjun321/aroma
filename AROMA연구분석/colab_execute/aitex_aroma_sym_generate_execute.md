# aitex aroma-symmetric 생성 (tiled + SGM symmetric 게이트) — downstream 선결

> **목적**: `exp4v2_symmetric_downstream_execute.md` 트랙 B가 요구하는 **aitex aroma-symmetric 합성물**을 생성한다. `newpipe/aitex/synth_aroma_sym/aitex/`에 산출 → downstream STEP 1의 `avail['aitex']`가 ✓가 되어 트랙 B가 활성화된다.
> **방식**: `exp4v2_mtd_symmetric_execute.md` §1~§5.1과 **동일**(ControlNet 생성 + `--compat_mode symmetric`: SGM matrix_symmetric + 64px 타일링 query + positive placement). **aitex 특화** = tiled(256/stride128) · **단일클래스** · elongated 결함용 **AR/텍스처 게이트**.
> **실행 환경**: selection·사전스캔 = CPU | 생성(ControlNet) = **GPU**(A100 권장).

---

## 0. 선결 산출물 (없으면 먼저 생성 — 3종)

aitex는 mtd와 달리 상류 산출물이 별도 트랙(tiled)이라 아래 3개가 **모두** 있어야 한다:

| 선결 | 산출물 | 생성 가이드 |
|------|--------|------------|
| ① profiling (symmetric 키) | `profiling/aitex/compatibility_matrix.json`에 `matrix_symmetric` | `profiling_symmetric_rebuild_verify_execute.md` (DS=`aitex`) |
| ② ControlNet 모델 | `controlnet_models/aitex/best_model/` | `controlnet_aroma_arm_execute.md` STEP 3~4 (aitex, tiled, augment ON, grayscale ON) |
| ③ τ 사전스캔 | `diagnostics/aitex/compat_gate/compat_tau_prescan_aitex.json`의 `ds_tau` | `compat_gate_cpu_diagnosis_execute.md` §10 (DS=`aitex`) |

> ①②③는 이 가이드 §2/§4에서 존재를 assert한다. 없으면 위 가이드 먼저.

---

## 1. 환경변수

```python
import os, json

os.environ['DRIVE'] = '/content/drive/MyDrive/data/Aroma'
os.environ['AROMA_SCRIPTS'] = '/content/AROMA/scripts/aroma'
os.environ['AROMA_REF']     = '/content/AROMA'
os.environ['AROMA_OUT']     = f"{os.environ['DRIVE']}/aroma_output"
os.environ['DATASET_CONFIG']= os.environ.get('DATASET_CONFIG', '/content/AROMA/dataset_config.json')
os.environ['CN_MODELS']     = f"{os.environ['AROMA_OUT']}/controlnet_models"

DS = 'aitex'
os.environ['DS'] = DS
os.environ['PROFILING_DIR'] = f"{os.environ['AROMA_OUT']}/profiling/{DS}"
os.environ['PROMPTS_DIR']   = f"{os.environ['AROMA_OUT']}/prompts/{DS}"
os.environ['COMPAT_JSON']   = f"{os.environ['PROFILING_DIR']}/compatibility_matrix.json"
os.environ['AROMA_CONFIG']  = f"{os.environ['PROFILING_DIR']}/recommended_config.yaml"

# tiled normal (256/stride128) — dataset_config aitex entry로 해소 (image_dir = aitex_tiled/train/good)
with open(os.environ['DATASET_CONFIG']) as f: _cfg = json.load(f)
os.environ['NORMAL_DIR'] = _cfg[DS]['image_dir']
print("aitex class_mode =", _cfg[DS].get('class_mode'), "| NORMAL_DIR =", os.environ['NORMAL_DIR'])

# 출력 (downstream 트랙 B가 참조하는 경로와 정확히 일치해야 함)
os.environ['SEL_AROMA']     = f"{os.environ['AROMA_OUT']}/newpipe/{DS}/sel_aroma"
os.environ['AROMA_SYM_DIR'] = f"{os.environ['AROMA_OUT']}/newpipe/{DS}/synth_aroma_sym"   # 생성 → /{DS}
os.environ['OUT_DIR']       = f"{os.environ['AROMA_OUT']}/diagnostics/{DS}/compat_gate"    # τ 사전스캔 위치
print("AROMA_SYM_DIR/DS =", f"{os.environ['AROMA_SYM_DIR']}/{DS}  ← downstream avail['aitex'] 대상")
```

> `class_mode`가 `single`, `image_dir`이 `.../aitex_tiled/train/good`으로 떠야 정상. 다르면 dataset_config aitex entry가 tiled를 가리키는지 확인(`aitex_tiled_rerun_execute.md` §3).

## 2. 선행 확인 (matrix_symmetric + CN 모델 필수)

```python
import os, json, pathlib
for k in ('PROFILING_DIR','COMPAT_JSON','AROMA_CONFIG','NORMAL_DIR'):
    print(f"{k:<14} {'OK' if pathlib.Path(os.environ[k]).exists() else 'MISSING':<8} {os.environ[k]}")
cn = f"{os.environ['CN_MODELS']}/{DS}/best_model"
print(f"CN_MODEL       {'OK' if pathlib.Path(cn).exists() else 'MISSING'}  {cn}")
for k in ('morphology_features.csv','context_features.csv'):
    p = f"{os.environ['PROFILING_DIR']}/{k}"
    print(f"{k:<24} {'OK' if pathlib.Path(p).exists() else 'MISSING'}")

c = json.load(open(os.environ['COMPAT_JSON']))
have = [k for k in ('matrix','matrix_symmetric','P_def_patch','clean_dist','symmetric_epsilon') if k in c]
print("compat keys:", have)
assert 'matrix_symmetric' in c, "matrix_symmetric 없음 → 선결 ① profiling_symmetric_rebuild_verify_execute.md (DS=aitex) 먼저"
assert pathlib.Path(cn).exists(), "CN 모델 없음 → 선결 ② controlnet_aroma_arm_execute.md STEP 3~4 (aitex) 먼저"
```

## 3. selection (single-class — aitex는 multi 플래그 미사용)

```python
!python $AROMA_SCRIPTS/roi_selection.py \
    --profiling_dir     $PROFILING_DIR \
    --prompts_dir       $PROMPTS_DIR \
    --sampling_strategy deficit_aware --score_mode realism \
    --top_k 200 --img_diversity_cap 1 \
    --output_dir        $SEL_AROMA
```

> ⚠️ **mtd(§3)와 차이**: aitex는 `class_mode=single`(결함 폴더 1개)이라 `--class_mode multi`·`--class_floor`·`--per_pair_cap_frac`을 **쓰지 않는다**(single 기본값 축퇴). `--rarity_temp` 미전달(realism 정합).
> `roi_candidates.json` + `roi_selected.json`이 `$SEL_AROMA`에 생성됐는지 확인.

## 4. τ 사전스캔 (symmetric 스케일, 타일링-aware) — 선결 ③

`compat_gate_cpu_diagnosis_execute.md §10`을 **DS=aitex**로 실행 → `compat_tau_prescan_aitex.json`의 `ds_tau`. **τ=0.5 금지.** aitex는 大 데이터셋 → Colab에서만.

```python
import json, os
p = f"{os.environ['OUT_DIR']}/compat_tau_prescan_{DS}.json"
assert os.path.exists(p), f"τ 사전스캔 없음 → 선결 ③ compat_gate_cpu_diagnosis §10 (DS=aitex) 먼저: {p}"
TAU = json.load(open(p)).get('ds_tau')
os.environ['TAU'] = str(TAU)
print("compat_threshold τ =", TAU)
assert TAU and 0.0 < TAU < 0.5, "τ 이상 — §10 재확인 (mtd 로컬 참고치 ≈0.23; aitex는 실측값 사용)"
```

> mtd처럼 하드코딩 폴백을 두지 않는다 — aitex τ 로컬 참고치가 없으므로 **반드시 §10 실측값**을 쓴다(prescan 필수 원칙, aitex 98% 폴백 사후 교훈).

## 5. AR / 텍스처 게이트 사전스캔 (aitex elongated 특화) — CPU

aitex 결함은 세장형(elongated)이 많아 ControlNet 512² squash에서 왜곡·수평 스미어가 생긴다. `controlnet_aroma_arm_execute.md` **STEP 5-0 / 5-0b**를 DS=aitex로 실행해 `AR_THRESHOLD`·`TEX_THRESHOLD`를 확정한다(발동률 CPU 예측 후 pilot 육안).

```python
import json, numpy as np, pathlib
sel = json.load(open(f"{os.environ['SEL_AROMA']}/roi_selected.json"))
ars = []
for e in sel:
    bb = e.get("defect_bbox")
    if bb and len(bb) == 4 and min(bb[2], bb[3]) > 0:
        ars.append(max(bb[2], bb[3]) / min(bb[2], bb[3]))
ars = np.array(ars)
print(f"aitex n={len(ars)}  AR med={np.median(ars):.2f} p90={np.percentile(ars,90):.2f} max={ars.max():.1f}")
for t in (2.0, 2.5, 3.0, 4.0, 6.0, 8.0):
    print(f"  AR>{t}: fallback {100*(ars>t).mean():.0f}%")
```

```python
# 사전스캔 + pilot 육안으로 확정 (controlnet STEP 5-0/5-0b 판정 규칙 준수)
AR_T  = 2.5     # ← AR 게이트: 초과 ROI는 copy_paste 폴백(개수·bbox parity 유지). 0=OFF
TEX_T = 0.25    # ← 텍스처 게이트(0.15~0.35 밴드). 0=OFF
os.environ['AR_T'], os.environ['TEX_T'] = str(AR_T), str(TEX_T)
print(f"AR_T={AR_T}  TEX_T={TEX_T}")
```

> ⚠️ aitex는 AR 상위 비중이 커 fallback이 높을 수 있다(과거 임계 2.5에서 98% 실측). fallback이 높으면 aroma-sym 대부분이 copy_paste가 되어 **"ControlNet 생성 기여 측정 제한"**으로 정직 보고한다 — 폴백 자체는 개수·클래스·bbox parity를 유지하므로 downstream은 정상 동작. 임계를 무리하게 올려(왜곡물 허용) fallback을 낮추는 것은 pilot 육안 통과 없이는 금지.

## 6. aroma-symmetric 생성 (ControlNet + symmetric 게이트 + AR/텍스처) — GPU

```python
import os
out = f"{os.environ['AROMA_SYM_DIR']}/{DS}"
!python $AROMA_SCRIPTS/generate_defects.py \
    --roi_dir     $SEL_AROMA \
    --normal_dir  $NORMAL_DIR \
    --output_dir  {out} \
    --method      controlnet \
    --controlnet_path  $CN_MODELS/$DS/best_model \
    --morphology_csv   $PROFILING_DIR/morphology_features.csv \
    --context_features $PROFILING_DIR/context_features.csv \
    --config           $AROMA_CONFIG \
    --n_per_roi 3 --seed 42 --blend_mode seamless \
    --reject-clean-bg --min-bg-quality 0.7 --bg-blur-threshold 100.0 \
    --compat_mode symmetric --compat_threshold $TAU \
    --compat_matrix_json $COMPAT_JSON \
    --cn_ar_threshold $AR_T \
    --texture-dist-threshold $TEX_T
```

> **활성 확인(로그)**: `compat gate ON: threshold=… mode=symmetric` + `placement-gate stats: fallback=M%`(positive placement) + `controlnet stats`(gen_ok/ar_fallback/blank_rate) + `texture-gate stats`. matrix_symmetric 없으면 **hard-fail**(설계).
> **aitex grayscale**: CN 모델이 grayscale ON으로 학습됨 → 별도 플래그 불요(leather만 `--cn_no_grayscale`). aitex는 직물 grayscale이므로 기본 동작.
> **`--local_staging` 미사용**: ControlNet 경로는 sidecar 캐시로 세션 재개 → Drive 직결 유지. 세션 끊기면 동일 셀 재실행(이어서 생성).
> mtd §5.1 대비 추가된 것은 `--cn_ar_threshold`·`--texture-dist-threshold`(§5 확정값)뿐 — 나머지 게이트/블렌드는 동일.

## 7. parity / 확인 (downstream 진입 전)

```python
import json, os
ann = json.load(open(f"{os.environ['AROMA_SYM_DIR']}/{DS}/annotations.json"))
live = [a for a in ann if not a.get("dry_run") and a.get("bbox")]
with_mask = sum(1 for a in live if a.get("mask_path"))
n_arfb = sum(1 for a in live if a.get("method") == "copy_paste_arfallback")
# downstream 트랙 B random arm n_synth_train (20260706_aitex 실측)
NEED = 246
print(f"labelable={len(live)} (mask={with_mask}, ar_fallback={n_arfb} = {100*n_arfb/max(1,len(live)):.0f}%)  needed>={NEED}")
print("✓ parity OK" if len(live) >= NEED else "✗ 부족 → n_per_roi 상향 후 §6 재실행(캐시로 부족분만 추가)")
```

> `ar_fallback` 비율을 **기록**한다 — downstream/논문에 "aitex aroma-sym 중 X%는 elongated이라 copy_paste 처리, ControlNet 생성 기여는 나머지"로 명시. 이 비율이 매우 높으면(예: >80%) aitex는 "생성 arm 측정 제한" 데이터셋으로 정직 보고.
> labelable < 246이면 `--n_per_roi 4`로 §6 재실행 — sidecar 캐시로 기존 생성분 skip, 부족분만 추가.

## 8. 다음 단계

`newpipe/aitex/synth_aroma_sym/aitex/annotations.json`이 생성되면 → `exp4v2_symmetric_downstream_execute.md` STEP 1 재실행 시 `avail['aitex']=True` → **트랙 B(aitex, tiled single, 256/no-rect/300, seeds 1 2 42)** 자동 활성. baseline/random은 `20260706_aitex`에서 이식, aroma-sym만 학습.

---

## 무결성 / 정직

- **downstream 정합**: 본 산출물은 downstream 트랙 B 규약(tiled single-class, `20260706_aitex` 이식 소스)과 짝이다. selection·seed·게이트가 mtd와 다른 것은 aitex tiled 특성 때문이며, 이는 트랙 B가 별도 이식 소스를 쓰는 것과 일관된다.
- **AR 폴백 정직 보고**: aroma-sym의 ControlNet 생성 비중 = `1 - ar_fallback비율`. aitex는 이 비중이 낮을 수 있으므로 결과 해석 시 반드시 병기(생성 novelty 기여 vs copy_paste 재조합 분리).
- **prescan 필수**: τ·AR·텍스처 임계는 전부 사전스캔 확정값 — 결과 보고 후 변경 금지. mtd 값 무검증 전용 금지(aitex 98% 폴백 교훈).
- **tile-level**: aitex는 256 타일 단위라 downstream mAP가 tile-level(50% overlap 중복 계수). 절대값을 타 데이터셋과 직접 비교 금지.
- **테스트 코드·pytest 금지**(CLAUDE.md). 검증은 Colab 실행으로.
