# step5 — AROMA arm + random arm 생성 (sym_final)

> 이 문서는 `colab_execute_new/_SPEC.md`를 **정본**으로 따른다. STEP 0 환경 셀은 `_SPEC §1`을 그대로 복사한 것이며, 명령은 `_SPEC §3 step5`만 사용한다. env·output 루트 규약을 재발명하지 않는다.

**목적**: exp4v2/exp3/exp5/exp6가 소비할 합성물을 생성한다. AROMA arm은 **두 생성 방법**을 모두 지원하며(reviewer 대응 — 무학습 copy_paste vs 생성형 ControlNet 비교), 여기에 통제군 random arm을 더해 최대 3종을 만든다.

- **AROMA arm — ControlNet** (STEP 3, `--method controlnet` + `--compat_mode symmetric`, **GPU**): ControlNet 생성 + SGM matrix_symmetric + 64px 타일링 query + positive placement + clean-bg 게이트 + seamless. → `S('synth_aroma', ds)` (또는 비교 시 `synth_aroma_cn`).
- **AROMA arm — copy_paste** (STEP 3B, `--method copy_paste`, **CPU·무학습**): 원본 crop을 clean-bg에 직접 합성. **동일한** clean-bg 게이트·compat symmetric placement·clean_bg_selected 소비. ControlNet 학습 불요. → 비교 시 `S('synth_aroma_cp', ds)`.
- **random arm** (STEP 4, `generate_random.py`, naive baseline): naive 배치(placement grounding·게이트 없음 — compat/foreground/clean-bg·void 모두 미적용, uniform-random 위치). → `S('synth_random', ds)`.

> **두 방법 대응(reviewer)**: 논문 피벗은 무학습 copy_paste가 주 방법, ControlNet은 생성형 대조(future-work 근거). reviewer가 "생성형 대비 무학습의 기여"를 물으면 **동일 ROI·동일 clean_bg·동일 placement 게이트** 하에 두 arm을 각각 생성해 exp4v2/exp3에서 나란히 보고한다(placement 게이트를 맞춰야 공정). AROMA ROI-선택·배치 기여는 두 방법 공통이고, ControlNet은 여기에 생성 novelty만 더한다.

**중요 — step5는 소비만 한다 (재사전스캔 금지)**: τ 사전스캔·(aitex) AR/텍스처 게이트 임계·ControlNet 학습은 **step4에서 이미 확정**됐다. step5는 이 확정값을 **읽어서 소비만** 한다(`_SPEC §5` prescan 필수 원칙, aitex 98% 폴백 교훈).

**실행 환경**: ControlNet arm = **GPU 필수**(Colab Pro A100 권장) | copy_paste arm·random arm = **CPU**.

**전제 (방법별로 다름)**:
- **공통(두 방법)**: `S('profiling',ds)/compatibility_matrix.json`의 `matrix_symmetric` 키(없으면 symmetric hard-fail), `S('roi',ds)/roi_selected.json`·`clean_bg_selected.json`(step3·step3.5), `compat_mode symmetric`일 때 `S('compat_gate',ds)/compat_tau_prescan_{ds}.json`의 `ds_tau`(step4c).
- **ControlNet 전용 추가**: `$CN_MODELS/{ds}/best_model/`(step4b 학습본), (**aitex**) `ar_tex_prescan_aitex.json`의 `ar_threshold`·`tex_threshold`.
- **copy_paste는 CN 학습본·AR 임계 불요** — ControlNet 학습(step4b)을 건너뛴다. (aitex 텍스처 게이트 `tex_threshold`만 선택 사용.)

**데이터셋**: v2-1 4종 `severstal · mvtec_leather · mtd · aitex`. aitex = tiled(256×256/stride128, single-class).

---

## 실행 순서 (체인)

```
phase0(profiling) → step1(complexity) → step2(prompts) → step3(roi_selection) → step4(CN 학습 + τ·AR/TEX 사전스캔) → [step5(생성, 이 문서)] → exp4v2/exp3/exp5/exp6
```

step5는 **step4 뒤·exp* 앞**이다. step4가 확정한 CN 모델·τ·(aitex)AR/TEX를 소비해 합성물을 만들고, 그 산출(`synth_aroma`/`synth_random`)을 exp4v2/exp3/exp5/exp6가 소비한다.

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
os.environ['CN_MODELS'] = f"{os.environ['SYM_ROOT']}/controlnet_models"   # ControlNet 학습본(step4 산출, step5 소비)
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

## STEP 2 — 선결 assert (step4 산출 확인 + τ·AR/TEX 로드)

step4가 확정한 값을 **읽어들이고**(재스캔 금지), 없으면 hard-fail한다. τ는 데이터셋별 `ds_tau`를 `TAU_BY_DS`에 담아 STEP 3 생성 루프가 소비한다. **τ=0.5 금지**(사전스캔 미실행 폴백 방지).

```python
import pathlib, json

# 생성할 AROMA arm 방법. 둘 다면 ["controlnet","copy_paste"], 무학습만이면 ["copy_paste"].
METHODS = ["controlnet", "copy_paste"]
USE_CN = "controlnet" in METHODS          # CN 학습본·AR 임계 필요 여부
USE_COMPAT = True                          # compat_mode symmetric 사용(τ 필요). False면 순수 clean_bg 배치.

TAU_BY_DS = {}
AR_T = TEX_T = None
missing = []

for DS in DATASETS:
    prof = S('profiling', DS)
    checks = {
        "compatibility_matrix":  f"{prof}/compatibility_matrix.json",
        "morphology_features":   f"{prof}/morphology_features.csv",
        "context_features":      f"{prof}/context_features.csv",
        "recommended_config":    f"{prof}/recommended_config.yaml",
        "roi_selected":          f"{S('roi', DS)}/roi_selected.json",
        "roi_candidates":        f"{S('roi', DS)}/roi_candidates.json",
        "clean_bg_selected":     f"{S('roi', DS)}/clean_bg_selected.json",   # step3.5 (두 방법 공통)
    }
    if USE_CN:      # ControlNet 경로만 학습본 필수
        checks["CN best_model"] = f"{os.environ['CN_MODELS']}/{DS}/best_model"
    if USE_COMPAT:  # compat symmetric placement 쓸 때만 τ 필수
        checks["tau_prescan"] = f"{S('compat_gate', DS)}/compat_tau_prescan_{DS}.json"
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

    # τ 로드 (ds_tau) — compat symmetric 쓸 때만. 재스캔 금지, τ=0.5 금지.
    if USE_COMPAT:
        tp = checks["tau_prescan"]
        if pathlib.Path(tp).exists():
            tau = json.load(open(tp)).get('ds_tau')
            assert tau is not None and 0.0 < tau < 0.5, \
                f"{DS}: τ 이상({tau}) — step4c τ 사전스캔 재확인 (τ=0.5/None 금지)"
            TAU_BY_DS[DS] = float(tau)
            print(f"  τ(ds_tau) = {tau}")

# aitex 전용 AR/텍스처 임계 — ControlNet 경로만 AR 사용(copy_paste는 텍스처만 선택). 재스캔 금지.
ar_p = f"{S('compat_gate', 'aitex')}/ar_tex_prescan_aitex.json"
print(f"\n=== aitex AR/TEX 게이트 임계 ===\n  {'✓' if pathlib.Path(ar_p).exists() else '✗'} {ar_p}")
if pathlib.Path(ar_p).exists():
    _at = json.load(open(ar_p))
    AR_T, TEX_T = _at.get('ar_threshold'), _at.get('tex_threshold')
    print(f"  AR_T={AR_T}  TEX_T={TEX_T}  (step4c 확정값 — step5 재스캔 금지)")
elif USE_CN:   # ControlNet + aitex면 AR 임계 필수
    missing.append(f"aitex:ar_tex_prescan → {ar_p}")

print("\nMISSING:", len(missing))
for m in missing: print("  -", m)
assert not missing, f"선결 산출 누락 — METHODS={METHODS}: 필요한 상류(step3.5/step4c{'/step4b CN' if USE_CN else ''}) 먼저 완료할 것"
if USE_COMPAT:
    assert set(TAU_BY_DS) == set(DATASETS), f"τ 미로드 데이터셋 존재: {set(DATASETS) - set(TAU_BY_DS)}"
print("\n✓ 선결 OK — TAU_BY_DS =", {k: round(v, 4) for k, v in TAU_BY_DS.items()})
```

> - `matrix_symmetric` 없으면 `--compat_mode symmetric`가 설계상 **hard-fail**한다 — phase0에서 symmetric 키를 emit한 profiling인지 확인.
> - τ는 step4 사전스캔 확정값을 **그대로** 쓴다. 로컬 참고치(mtd≈0.23)는 참고용일 뿐, Colab에서는 `ds_tau` 실측을 사용한다.
> - **aitex AR/TEX는 step4에서 확정**해 `ar_tex_prescan_aitex.json`(키 `ar_threshold`·`tex_threshold`)으로 남긴 값을 소비한다. step5에서 `roi_selected.json`으로 AR을 다시 스캔하지 않는다.

---

## STEP 3 — AROMA arm 생성 (ControlNet + symmetric 게이트, GPU) — 2차 (reviewer 대응)

> **이 단계는 2차**다. 1차(무학습 copy_paste = STEP 3B)로 AROMA vs random을 먼저 확정하고, reviewer가 "생성형 대비 무학습의 기여"를 물을 때 이 ControlNet arm을 추가 생성해 나란히 비교한다. 선결로 step4b(ControlNet 학습, GPU)가 완료돼야 한다. 1차만 진행 중이면 이 STEP은 건너뛴다.

`_SPEC §3 step5`. 4종 공통 명령에 **데이터셋별 추가 플래그**만 붙는다.

- **공통**: `--method controlnet --controlnet_path $CN_MODELS/$DS/best_model` + `--compat_mode symmetric --compat_threshold $TAU --compat_matrix_json <compatibility_matrix.json>` + clean-bg 게이트(`--reject-clean-bg --min-bg-quality 0.7 --bg-blur-threshold 100.0`) + `--blend_mode seamless` + `--n_per_roi 3 --seed 42`.
- **mvtec_leather**: `--cn_no_grayscale` 추가(컬러 가죽 — grayscale 강제 해제).
- **aitex**: `--cn_ar_threshold $AR_T --texture-dist-threshold $TEX_T` 추가(elongated 왜곡 방지, step4 확정 임계).
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
    os.environ['TAU']    = str(TAU_BY_DS[DS])                          # step4 확정 τ 소비
    # 데이터셋별 추가 플래그 (step4 확정값 소비 — 재사전스캔 금지)
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
> - `clean_bg assignment ON: clean_bg_selected.json (N ROIs)` — step3.5 사전선정 배경 소비. **파일이 없으면 legacy 생성-시점 선정으로 자동 fallback되는데, 이는 "무해"가 아니다**: legacy는 원본 good 픽셀 재스캔 휴리스틱이라 step3.5의 profiling-파생 배경 선정과 **배경 히스토그램 분포가 다르다** → aroma arm의 배경이 의도와 달라지고 exp4v2 aroma/random 대조의 공정성이 저하될 수 있다. **`clean_bg assignment ON`이 반드시 떠야** 하며, 안 뜨면 step3.5 산출(`S('roi',ds)/clean_bg_selected.json`) 존재·경로를 먼저 확인한다(fallback 상태로 생성 금지).
> - `clean_bg resolve: used=U fallback=F mismatch=M / T (roi,rep)` — **U가 T에 근접(F·M≈0)해야 정상**. `<90%` 시 WARNING(경로 불일치/staleness). 로컬 mtd 20-ROI 재검증: used=40 fallback=0 mismatch=0.
> - (step3.5를 `--geometry_prior`로 실행한 경우) precompute된 `position`을 소비해 클래스 기하대로 배치. phase0가 `image_w/image_h`를 방출하면 **clamp-free**(로컬 40/40 정확); 없으면 grid 추정이라 edge-flush가 실제 가장자리보다 안쪽에 놓일 수 있음.
> - `compat gate ON: threshold=… mode=symmetric` — symmetric 게이트 활성(matrix_symmetric 없으면 hard-fail).
> - `placement-gate stats: fallback=M%` — positive placement(scan·rank·place). fallback 과다(>50%)면 τ 과대 의심 → step4 사전스캔 재확인(step5에서 튜닝 금지).
> - (aitex) `controlnet stats`(gen_ok / ar_fallback / blank_rate) + `texture-gate stats` — AR 폴백·텍스처 게이트 동작.
> - severstal 수천 장 = 수 시간. 캐시 재개 가능(동일 셀 재실행).

---

## STEP 3B — AROMA arm 생성 (copy_paste, 무학습·CPU) — ★ 1차 주 경로

> **진행 순서(계획)**: **1차 = 이 STEP 3B(copy_paste)** 로 AROMA arm 생성 → STEP 4 random arm과 함께 exp4v2에서 **AROMA(copy_paste) vs random** 비교. **2차 = reviewer report 대응 시 STEP 3(controlnet)** 로 AROMA arm을 추가 생성해 "무학습 vs 생성형" 비교. 1차는 GPU·ControlNet 학습(step4b) 불요.

copy-paste 피벗(무학습 기판). ControlNet 생성 대신 **결함 crop을 clean-bg에 직접 합성**한다. STEP 3(controlnet)과 **동일한 clean-bg 게이트·compat symmetric placement·clean_bg_selected 소비**를 쓰되, 생성 방식만 copy_paste다.

- **선결이 가볍다**: ControlNet 학습(step4b, GPU) **불요**. phase0·step1~3·step3.5 + (compat 쓰면) step4c τ만 있으면 된다. **GPU 불필요(CPU)**.
- **공통**: `--method copy_paste` + clean-bg 게이트(`--reject-clean-bg --min-bg-quality 0.7 --bg-blur-threshold 100.0`) + `--compat_mode symmetric --compat_threshold $TAU --compat_matrix_json <…>`(positive placement, method-무관) + `--blend_mode seamless`(또는 `alpha`) + `--n_per_roi 3 --seed 42`.
- **CN 전용 인자 미사용**: `--controlnet_path`·`--morphology_csv`·`--context_features`(CN conditioning), `--cn_ar_threshold`·`--cn_no_grayscale`(ControlNet squash/그레이스케일 전용)는 **넣지 않는다**(copy_paste는 squash가 없어 AR 폴백 자체가 없음).
- **`--config` 불요**: compat 게이트의 `bin_edges`는 `--compat_matrix_json`(compatibility_matrix.json)에 이미 들어있어 **matrix JSON만으로 자기완결**한다(코드 `299c5b0`). `--config`를 넣어도 무해하나 copy_paste엔 불필요. (구버전 코드는 `--config`를 요구했으니 저장소가 `299c5b0` 이상인지 확인.)
- **aitex 텍스처 게이트만** 선택 적용: `--texture-dist-threshold $TEX_T`(텍스처 이질 배치 거부는 method-무관). AR 게이트는 미적용.
- **`--local_staging` 사용 가능**(CPU 경로).
- **출력**: 1차 copy_paste는 `S('synth_aroma', DS)`. 2차에 controlnet과 **비교**하려면 각각 `S('synth_aroma_cp', DS)`·`S('synth_aroma_cn', DS)`로 분리해 exp*에서 `--aroma_synthetic_dir`로 지정.

```python
DATASETS_GEN = DATASETS   # 세션 분리 시 좁힘

for DS in DATASETS_GEN:
    os.environ['DS']     = DS
    os.environ['ROI']    = S('roi', DS)
    os.environ['NORMAL'] = normal_dir(DS)
    os.environ['OUT']    = S('synth_aroma', DS)      # 2차 controlnet과 비교 시 'synth_aroma_cp'
    os.environ['COMPAT'] = f"{S('profiling', DS)}/compatibility_matrix.json"
    os.environ['TAU']    = str(TAU_BY_DS[DS])         # step4c 확정 τ (compat_mode symmetric일 때)
    os.environ['EXTRA']  = f"--texture-dist-threshold {TEX_T}" if DS == "aitex" else ""  # 텍스처 게이트만
    print(f"\n===== AROMA copy_paste gen {DS}  (τ={os.environ['TAU']}) =====")
    !python $AROMA_SCRIPTS/generate_defects.py \
        --roi_dir     $ROI \
        --normal_dir  $NORMAL \
        --output_dir  $OUT \
        --method      copy_paste \
        --n_per_roi 3 --seed 42 --blend_mode seamless \
        --reject-clean-bg --min-bg-quality 0.7 --bg-blur-threshold 100.0 \
        --compat_mode symmetric --compat_threshold $TAU \
        --compat_matrix_json $COMPAT \
        --local_staging \
        $EXTRA
```

> **compat 없이(순수 clean_bg 배치)** 돌리려면 `--compat_mode`·`--compat_threshold`·`--compat_matrix_json`을 빼면 된다(그러면 step4c τ도 불요 → phase0·step1~3·step3.5만으로 완결). 단 2차에서 controlnet arm과 비교할 때는 **placement 게이트를 맞춰야** 공정하다.
> **활성 확인(로그)**: `clean_bg assignment ON` + `clean_bg resolve used/fallback/mismatch`(U≈T, F·M≈0) + `compat gate ON: threshold=… mode=symmetric` + `placement-gate stats: active=N compat=N fallback=M%`. `--config` 없이도 `compat gate ON`이 떠야 정상(로컬 mtd 20-ROI: active=40 fallback=0, clean_bg used=40/40). `controlnet stats`·`ar_fallback`은 copy_paste에선 안 나온다(정상).
> **정직성**: copy_paste는 생성 novelty가 없다(원본 crop 재조합). AROMA ROI-선택·배치 기여를 exp4v2 mAP로 판정한다. exp3(FID/생성품질)에서 2차 controlnet arm과 절대비교하지 않는다.

---

## STEP 4 — random arm 생성 (통제군, CPU)

`_SPEC §3 step5 random`. random arm은 **naive copy-paste baseline**(무검사 무작위 위치)이다 — AROMA arm과 동일 `top_k`/`n_per_roi`/`seed`(ROI-선택 무작위성 동일)를 쓰되, **placement grounding·게이트는 의도적으로 없다**: compat/positive placement, foreground 제약, clean-bg/void 게이트 모두 미적용. 이것이 **의도된 placement 비대칭**으로, AROMA의 smart-placement 프레임워크(올바른 ROI·context·void 제외 랭킹)라는 기여를 격리 측정하는 정당한 ablation이다(naive random-position copy-paste는 문헌의 표준 증강 baseline). CPU라 `--local_staging` 사용 가능(`_SPEC §5`).

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
        --local_staging
```

> **random arm은 naive placement가 기본 ON**이다(`generate_random.py`의 `random_placement` 기본값 True → `generate_defects`에 전달). 따라서 `--reject-clean-bg`/`--min-bg-quality`/`--bg-blur-threshold`(clean-bg 게이트)를 **넘기지 않는다** — naive 분기가 모든 placement grounding(compat/`_positive_place`, foreground 제약, void/clean-bg 게이트)을 우회하고 uniform-random 위치에 붙인다. `--reject-clean-bg`를 다시 켜면 `load_normal_images()` **pool 게이트**가 void normal을 풀에서 제거해(=grounding) 대비가 흐려지므로 넣지 않는다. grounded 경로로 되돌리려면 `--no-random-placement`(디버그용). 출력 `--output_dir`은 반드시 `S('synth_random', DS)`.

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
> - **naive 배치 검증(재실행 후)**: (1) random 로그에 `placement-gate stats …`가 **찍히지 않아야** 한다 — naive 분기가 compat/positive placement·foreground·void 게이트를 전부 우회한 증거(AROMA arm에서는 계속 출력됨). (2) random 합성물을 육안 확인하면 결함이 **검은/void/edge 영역에 착지**하는 사례가 섞여 있어야 정상이다(AROMA arm은 grounded라 그렇지 않다). (3) 재합성·재평가 후 **AROMA vs random mAP 격차가 벌어지는** 방향이 기대값이다(placement grounding 기여가 격리되어 드러남). 근거·설계 의도: `.claude/.dev_note/aroma_naive-random-placement.md`. exp4v2 YOLO 학습은 사용자 명시 요청 시에만 실행(load-test policy).

---

## 무결성 / 정직 (`_SPEC §5`)

- **step5는 소비만 — 재사전스캔 금지**: τ·AR·TEX는 전부 step4 사전스캔 확정값. step5에서 다시 스캔하거나 τ=0.5로 폴백하지 않는다(aitex 98% 폴백 사후 교훈).
- **사후 튜닝 금지**: τ·seed·n_per_roi·blend/게이트 설정을 결과 보고 후 변경하지 않는다. fallback이 과다해도 step5에서 임계를 손대지 말고 step4 사전스캔을 재검한다.
- **AR 폴백 정직 보고**: aitex aroma-sym의 ControlNet 생성 비중을 `1 - ar_fallback비율`로 병기한다(생성 novelty 기여 vs copy_paste 재조합 분리). 이 값을 임의로 높이려고 AR 임계를 올려 왜곡물을 허용하는 것은 pilot 육안 통과(step4) 없이 금지.
- **placement 비대칭(의도) — 옛 "게이트 대칭=공정" 지침 supersede**: AROMA arm = grounded smart placement(compat/positive placement + clean-bg·void 게이트, `--reject-clean-bg` 등), random arm = naive baseline(게이트·grounding OFF, `random_placement` 기본 ON). 이 비대칭이 **AROMA의 기여(placement 프레임워크 전체)를 격리하는 정당한 ablation**이며, 숨기는 역전이 아니라 의도된 대비다(naive random-position copy-paste는 표준 증강 baseline). 이전 문서의 "random에도 동일 clean-bg 게이트를 걸어야 공정하다"는 지침은 이 프레이밍으로 대체된다. (단 STEP 3 vs STEP 3B의 controlnet↔copy_paste 비교는 **둘 다 AROMA arm**의 생성 novelty ablation이므로 placement 게이트를 맞추는 것이 여전히 옳다 — L11/L256 유지.)
- ⚠️ **step5 재실행(합성 재생성) 시 exp5/exp6 임베딩 캐시 무효화**: exp5(PRDC)·exp6(kNN)의 DINOv2 임베딩 캐시 키는 **경로/파일명 기반**이라, 재생성이 동일 파일명(`syn_00000_00.jpg`)으로 내용만 덮어쓰면 **stale 임베딩이 재사용**된다. 재합성한 데이터셋은 exp5/exp6 실행 전 `!rm -rf $EMBED_CACHE_DIR/{ds}` 후 재실행한다(상세: exp5/exp6_execute.md).
- **aitex는 tile-level·single-class** → 절대값을 타 데이터셋과 직접 비교 금지, Δ만 유효.
- **`--local_staging`**: random(CPU)에는 사용 가능, **ControlNet 생성(AROMA arm)에는 미사용**(sidecar 캐시 Drive 직결 필요).
- **테스트 코드 신규 작성·pytest 금지**(CLAUDE.md). 검증은 Colab 실행으로.
- **시간 실측·처리량 벤치 미수행**(load-test policy).
