# step4 — AROMA arm + random arm 생성 (sym_final)

> 이 문서는 `colab_execute_new/_SPEC.md`를 **정본**으로 따른다. STEP 0 환경 셀은 `_SPEC §1`을 그대로 복사한 것이며, 명령은 `_SPEC §3 step4`만 사용한다. env·output 루트 규약을 재발명하지 않는다.

**목적**: exp4v2/exp3/exp5/exp6가 소비할 **합성물 2종**을 생성한다.
- **AROMA arm** (`generate_defects.py --method controlnet` + `--compat_mode symmetric`): ControlNet 생성 + SGM matrix_symmetric + 64px 타일링 query + positive placement + clean-bg 게이트 + seamless 블렌딩. → `S('synth_aroma', ds)`
- **random arm** (`generate_random.py`, 통제군): 동일 clean-bg 게이트. → `S('synth_random', ds)`

**중요 — step4는 소비만 한다 (재사전스캔 금지)**: τ 사전스캔·(aitex) AR/텍스처 게이트 임계·ControlNet 학습은 **step5에서 이미 확정**됐다. step4는 이 확정값을 **읽어서 소비만** 한다. step4에서 τ·AR·TEX를 다시 스캔하지 않는다(`_SPEC §5` prescan 필수 원칙, aitex 98% 폴백 교훈).

**실행 환경**: AROMA arm 생성(ControlNet) = **GPU 필수**(Colab Pro A100 권장) | random arm = CPU.

**전제 — step5 완료**: 데이터셋별로 아래가 모두 존재해야 한다(아래 선결 assert가 검증).
- `$CN_MODELS/{ds}/best_model/` (step5 ControlNet 학습본)
- `S('profiling',ds)/compatibility_matrix.json` 에 `matrix_symmetric` 키 (없으면 symmetric 모드 hard-fail)
- `S('compat_gate',ds)/compat_tau_prescan_{ds}.json` 의 `ds_tau` (τ 사전스캔 확정값)
- (**aitex 전용**) `S('compat_gate','aitex')/ar_tex_prescan_aitex.json` 의 `ar_threshold`·`tex_threshold` (step5에서 확정한 AR/텍스처 게이트 임계)

**데이터셋**: v2-1 4종 `severstal · mvtec_leather · mtd · aitex`. aitex = tiled(256×256/stride128, single-class).

---

## 실행 순서 (체인)

```
phase0(profiling) → step1(complexity) → step2(prompts) → step3(roi_selection) → step5(CN 학습 + τ·AR/TEX 사전스캔) → [step4(생성, 이 문서)] → exp4v2/exp3/exp5/exp6
```

step4는 **step5 뒤·exp* 앞**이다. step5가 확정한 CN 모델·τ·(aitex)AR/TEX를 소비해 합성물을 만들고, 그 산출(`synth_aroma`/`synth_random`)을 exp4v2/exp3/exp5/exp6가 소비한다.

---

## STEP 0 — 공통 환경 셀 (`_SPEC §1` 그대로 — 수정 금지)

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
os.environ['CN_MODELS'] = f"{os.environ['SYM_ROOT']}/controlnet_models"   # ControlNet 학습본(step5 산출, step4 소비)
def S(stage, ds=None):
    p = f"{os.environ['SYM_ROOT']}/{stage}"
    return f"{p}/{ds}" if ds else p

DATASETS = ["severstal", "mvtec_leather", "mtd", "aitex"]   # v2-1 4종
with open(os.environ['DATASET_CONFIG']) as f: CFG = json.load(f)
def normal_dir(ds): return CFG[ds]["image_dir"]                 # aitex → aitex_tiled/train/good
def is_multi(ds):   return CFG[ds].get("class_mode") == "multi" # aitex=single (자동)
```

---

## STEP 1 — 패키지 설치 (AROMA arm 생성용, GPU)

```python
!pip install diffusers transformers accelerate safetensors -q
!pip install scikit-image opencv-python-headless -q
# xformers는 선택 (없으면 SDPA로 자동 fallback)
```

```python
!nvidia-smi
```

---

## STEP 2 — 선결 assert (step5 산출 확인 + τ·AR/TEX 로드)

step5가 확정한 값을 **읽어들이고**(재스캔 금지), 없으면 hard-fail한다. τ는 데이터셋별 `ds_tau`를 `TAU_BY_DS`에 담아 STEP 3 생성 루프가 소비한다. **τ=0.5 금지**(사전스캔 미실행 폴백 방지).

```python
import pathlib, json

TAU_BY_DS = {}
AR_T = TEX_T = None
missing = []

for DS in DATASETS:
    prof = S('profiling', DS)
    checks = {
        "CN best_model":         f"{os.environ['CN_MODELS']}/{DS}/best_model",
        "compatibility_matrix":  f"{prof}/compatibility_matrix.json",
        "morphology_features":   f"{prof}/morphology_features.csv",
        "context_features":      f"{prof}/context_features.csv",
        "recommended_config":    f"{prof}/recommended_config.yaml",
        "roi_selected":          f"{S('roi', DS)}/roi_selected.json",
        "roi_candidates":        f"{S('roi', DS)}/roi_candidates.json",
        "tau_prescan":           f"{S('compat_gate', DS)}/compat_tau_prescan_{DS}.json",
    }
    print(f"\n=== {DS} ({'multi' if is_multi(DS) else 'single'}) ===")
    for name, p in checks.items():
        ok = pathlib.Path(p).exists()
        if not ok: missing.append(f"{DS}:{name} → {p}")
        print(f"  {'✓' if ok else '✗'} {name:<22} {p}")

    # profiling: matrix_symmetric 키 필수 (symmetric 모드 hard-fail 방지)
    cj_p = checks["compatibility_matrix"]
    if pathlib.Path(cj_p).exists():
        cj = json.load(open(cj_p))
        have = [k for k in ('matrix','matrix_symmetric','P_def_patch','clean_dist','symmetric_epsilon') if k in cj]
        print(f"  compat keys: {have}")
        assert 'matrix_symmetric' in cj, \
            f"{DS}: matrix_symmetric 없음 → phase0(symmetric emit) 재실행 필요. --compat_mode symmetric는 hard-fail."

    # τ 로드 (ds_tau) — 재스캔 금지, step5 확정값 소비. τ=0.5 금지.
    tp = checks["tau_prescan"]
    if pathlib.Path(tp).exists():
        tau = json.load(open(tp)).get('ds_tau')
        assert tau is not None and 0.0 < tau < 0.5, \
            f"{DS}: τ 이상({tau}) — step5 §τ 사전스캔 재확인 (τ=0.5/None 금지)"
        assert abs(tau - 0.5) > 1e-9, f"{DS}: τ=0.5 금지 (사전스캔 미실행 폴백 의심)"
        TAU_BY_DS[DS] = float(tau)
        print(f"  τ(ds_tau) = {tau}")

# aitex 전용: AR/텍스처 게이트 임계 (step5 확정값) — 재스캔 금지
ar_p = f"{S('compat_gate', 'aitex')}/ar_tex_prescan_aitex.json"
print(f"\n=== aitex AR/TEX 게이트 임계 ===\n  {'✓' if pathlib.Path(ar_p).exists() else '✗'} {ar_p}")
if pathlib.Path(ar_p).exists():
    _at = json.load(open(ar_p))
    AR_T, TEX_T = _at.get('ar_threshold'), _at.get('tex_threshold')
    assert AR_T is not None and TEX_T is not None, "aitex AR_T/TEX_T 누락 → step5 AR/텍스처 사전스캔 재확인"
    print(f"  AR_T={AR_T}  TEX_T={TEX_T}  (step5 확정값 — step4 재스캔 금지)")
else:
    missing.append(f"aitex:ar_tex_prescan → {ar_p}")

print("\nMISSING:", len(missing))
for m in missing: print("  -", m)
assert not missing, "선결 산출 누락 — step5(CN 학습 + τ·AR/TEX 사전스캔) 먼저 완료할 것"
assert set(TAU_BY_DS) == set(DATASETS), f"τ 미로드 데이터셋 존재: {set(DATASETS) - set(TAU_BY_DS)}"
print("\n✓ 선결 OK — TAU_BY_DS =", {k: round(v, 4) for k, v in TAU_BY_DS.items()})
```

> - `matrix_symmetric` 없으면 `--compat_mode symmetric`가 설계상 **hard-fail**한다 — phase0에서 symmetric 키를 emit한 profiling인지 확인.
> - τ는 step5 사전스캔 확정값을 **그대로** 쓴다. 로컬 참고치(mtd≈0.23)는 참고용일 뿐, Colab에서는 `ds_tau` 실측을 사용한다.
> - **aitex AR/TEX는 step5에서 확정**해 `ar_tex_prescan_aitex.json`(키 `ar_threshold`·`tex_threshold`)으로 남긴 값을 소비한다. step4에서 `roi_selected.json`으로 AR을 다시 스캔하지 않는다.

---

## STEP 3 — AROMA arm 생성 (ControlNet + symmetric 게이트, GPU)

`_SPEC §3 step4`. 4종 공통 명령에 **데이터셋별 추가 플래그**만 붙는다.

- **공통**: `--method controlnet --controlnet_path $CN_MODELS/$DS/best_model` + `--compat_mode symmetric --compat_threshold $TAU --compat_matrix_json <compatibility_matrix.json>` + clean-bg 게이트(`--reject-clean-bg --min-bg-quality 0.7 --bg-blur-threshold 100.0`) + `--blend_mode seamless` + `--n_per_roi 3 --seed 42`.
- **mvtec_leather**: `--cn_no_grayscale` 추가(컬러 가죽 — grayscale 강제 해제).
- **aitex**: `--cn_ar_threshold $AR_T --texture-dist-threshold $TEX_T` 추가(elongated 왜곡 방지, step5 확정 임계).
- **`--local_staging` 미사용**(ControlNet): sidecar 캐시가 Drive 직결이어야 세션 재개 시 살아남는다.
- **출력 `--output_dir` = `S('synth_aroma', DS)`**(= `sym_final/synth_aroma/{ds}`). exp*가 `--aroma_synthetic_dir $(S('synth_aroma'))`에 `/{ds}`를 붙이므로 이 경로여야 정합한다.

> ⚠️ **GPU 세션 관리**: severstal은 수천 장이라 최장이다. 세션이 끊기면 **동일 셀 재실행** = sidecar 캐시로 이어서 생성(재-GPU 스킵). 한 번에 한 데이터셋만 돌리려면 `DATASETS_GEN = ["severstal"]`처럼 좁혀 아래 루프의 대상을 바꾼다.

```python
DATASETS_GEN = DATASETS   # 세션 분리 시 ["severstal"] 등으로 좁힘

for DS in DATASETS_GEN:
    os.environ['DS']     = DS
    os.environ['PROF']   = S('profiling', DS)
    os.environ['ROI']    = S('roi', DS)
    os.environ['NORMAL'] = normal_dir(DS)
    os.environ['OUT']    = S('synth_aroma', DS)                       # ← 반드시 이 경로
    os.environ['COMPAT'] = f"{S('profiling', DS)}/compatibility_matrix.json"
    os.environ['TAU']    = str(TAU_BY_DS[DS])                          # step5 확정 τ 소비
    # 데이터셋별 추가 플래그 (step5 확정값 소비 — 재사전스캔 금지)
    if DS == "mvtec_leather":
        os.environ['EXTRA'] = "--cn_no_grayscale"
    elif DS == "aitex":
        os.environ['EXTRA'] = f"--cn_ar_threshold {AR_T} --texture-dist-threshold {TEX_T}"
    else:
        os.environ['EXTRA'] = ""
    print(f"\n===== AROMA gen {DS}  (τ={os.environ['TAU']}, extra='{os.environ['EXTRA']}') =====")
    !python $AROMA_SCRIPTS/generate_defects.py \
        --roi_dir     $ROI \
        --normal_dir  $NORMAL \
        --output_dir  $OUT \
        --method      controlnet \
        --controlnet_path  $CN_MODELS/$DS/best_model \
        --morphology_csv   $PROF/morphology_features.csv \
        --context_features $PROF/context_features.csv \
        --config           $PROF/recommended_config.yaml \
        --n_per_roi 3 --seed 42 --blend_mode seamless \
        --reject-clean-bg --min-bg-quality 0.7 --bg-blur-threshold 100.0 \
        --compat_mode symmetric --compat_threshold $TAU \
        --compat_matrix_json $COMPAT \
        $EXTRA
```

> **활성 확인(로그)**:
> - `clean_bg assignment ON: clean_bg_selected.json (N ROIs)` — step3.5 사전선정 배경 소비(파일 없으면 legacy 생성-시점 선정으로 자동 fallback, 무해).
> - `clean_bg resolve: used=U fallback=F mismatch=M / T (roi,rep)` — **U가 T에 근접(F·M≈0)해야 정상**. `<90%` 시 WARNING(경로 불일치/staleness). 로컬 mtd 20-ROI 재검증: used=40 fallback=0 mismatch=0.
> - (step3.5를 `--geometry_prior`로 실행한 경우) precompute된 `position`을 소비해 클래스 기하대로 배치. phase0가 `image_w/image_h`를 방출하면 **clamp-free**(로컬 40/40 정확); 없으면 grid 추정이라 edge-flush가 실제 가장자리보다 안쪽에 놓일 수 있음.
> - `compat gate ON: threshold=… mode=symmetric` — symmetric 게이트 활성(matrix_symmetric 없으면 hard-fail).
> - `placement-gate stats: fallback=M%` — positive placement(scan·rank·place). fallback 과다(>50%)면 τ 과대 의심 → step5 사전스캔 재확인(step4에서 튜닝 금지).
> - (aitex) `controlnet stats`(gen_ok / ar_fallback / blank_rate) + `texture-gate stats` — AR 폴백·텍스처 게이트 동작.
> - severstal 수천 장 = 수 시간. 캐시 재개 가능(동일 셀 재실행).

---

## STEP 4 — random arm 생성 (통제군, CPU)

`_SPEC §3 step4 random`. AROMA arm과 **동일 clean-bg 게이트**·동일 `top_k`/`n_per_roi`/`seed`. positive placement·compat 게이트는 미적용(무변경 통제군). CPU라 `--local_staging` 사용 가능(`_SPEC §5`).

```python
for DS in DATASETS:
    os.environ['DS']     = DS
    os.environ['ROI']    = S('roi', DS)
    os.environ['NORMAL'] = normal_dir(DS)
    os.environ['OUT_R']  = S('synth_random', DS)                      # ← exp*가 /{ds} 붙임
    print(f"\n===== random gen {DS} =====")
    !python $AROMA_SCRIPTS/generate_random.py \
        --candidates_json $ROI/roi_candidates.json \
        --normal_dir      $NORMAL \
        --output_dir      $OUT_R \
        --top_k 200 --n_per_roi 3 --seed 42 \
        --local_staging \
        --reject-clean-bg --min-bg-quality 0.7 --bg-blur-threshold 100.0
```

> random은 개선(positive placement·compat) 무관이지만 clean-bg 게이트는 AROMA와 대칭이어야 비교가 공정하다. 출력 `--output_dir`은 반드시 `S('synth_random', DS)`.

---

## STEP 5 — parity / 확인 (exp* 진입 전)

두 arm 모두 labelable(dry_run 아니고 bbox 있는 항목) > 0인지 확인하고, **aitex는 AR 폴백 비율을 기록**한다(정직 보고용).

```python
import json

def _live(path):
    ann = json.load(open(path))
    return [a for a in ann if not a.get("dry_run") and a.get("bbox")]

print(f"{'dataset':<14} {'aroma':>8} {'random':>8}  note")
print("-" * 48)
for DS in DATASETS:
    a = _live(f"{S('synth_aroma', DS)}/annotations.json")
    r = _live(f"{S('synth_random', DS)}/annotations.json")
    with_mask = sum(1 for x in a if x.get("mask_path"))
    note = f"mask={with_mask}"
    if DS == "aitex":
        n_arfb = sum(1 for x in a if x.get("method") == "copy_paste_arfallback")
        note += f"  ar_fallback={n_arfb} ({100*n_arfb/max(1,len(a)):.0f}%)"
    print(f"{DS:<14} {len(a):>8} {len(r):>8}  {note}")
    assert len(a) > 0 and len(r) > 0, f"{DS}: labelable 0 — 생성 실패 (STEP 3/4 로그 확인)"
print("\n✓ parity OK — 두 arm 모두 labelable > 0")
```

> - **aitex `ar_fallback` 비율을 반드시 기록**한다. 이 비율이 높으면(예: >80%) aroma-sym의 대부분이 copy_paste 폴백이 되어 "ControlNet 생성 기여 측정 제한" 데이터셋으로 정직 보고한다(폴백은 개수·클래스·bbox parity를 유지하므로 downstream은 정상 동작). ControlNet 생성 비중 = `1 - ar_fallback비율`.
> - labelable이 exp*의 `synth_ratio=1.0` cap을 못 채우면(random arm n_synth_train 미만) `--n_per_roi`를 올려 해당 STEP을 재실행 — sidecar 캐시로 기존 생성분은 skip되고 부족분만 추가된다.

---

## 무결성 / 정직 (`_SPEC §5`)

- **step4는 소비만 — 재사전스캔 금지**: τ·AR·TEX는 전부 step5 사전스캔 확정값. step4에서 다시 스캔하거나 τ=0.5로 폴백하지 않는다(aitex 98% 폴백 사후 교훈).
- **사후 튜닝 금지**: τ·seed·n_per_roi·blend/게이트 설정을 결과 보고 후 변경하지 않는다. fallback이 과다해도 step4에서 임계를 손대지 말고 step5 사전스캔을 재검한다.
- **AR 폴백 정직 보고**: aitex aroma-sym의 ControlNet 생성 비중을 `1 - ar_fallback비율`로 병기한다(생성 novelty 기여 vs copy_paste 재조합 분리). 이 값을 임의로 높이려고 AR 임계를 올려 왜곡물을 허용하는 것은 pilot 육안 통과(step5) 없이 금지.
- **clean-bg 게이트 대칭**: AROMA·random 두 arm에 동일 게이트(`--reject-clean-bg --min-bg-quality 0.7 --bg-blur-threshold 100.0`)를 적용해야 비교가 공정하다.
- **aitex는 tile-level·single-class** → 절대값을 타 데이터셋과 직접 비교 금지, Δ만 유효.
- **`--local_staging`**: random(CPU)에는 사용 가능, **ControlNet 생성(AROMA arm)에는 미사용**(sidecar 캐시 Drive 직결 필요).
- **테스트 코드 신규 작성·pytest 금지**(CLAUDE.md). 검증은 Colab 실행으로.
- **시간 실측·처리량 벤치 미수행**(load-test policy).
