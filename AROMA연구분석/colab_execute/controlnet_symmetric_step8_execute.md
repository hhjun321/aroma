# controlnet arm STEP 8 (symmetric 개선판) — leather / severstal / aitex

> **관계**: `controlnet_aroma_arm_execute.md` STEP 8을 **symmetric 개선(SGM 대칭 matrix + 타일링 + positive placement)**으로 실행하는 부록. aroma arm 합성만 symmetric으로 교체하고 baseline/random은 그대로 이식.
> **개선 활성**: `generate_defects ... --compat_mode symmetric --compat_threshold <τ> --compat_matrix_json <profiling>/compatibility_matrix.json` (`exp4v2_mtd_symmetric_execute.md` §5.1과 동일).
> **왜 이식이 유효한가**: symmetric은 **aroma arm에만** 영향 → 이식하는 baseline/random 불변 → graft 유효.

---

## 0. 범위 · 전제 (읽고 시작)

### 대상
- **실작업(3종)**: `mvtec_leather` · `severstal` · `aitex`
- **mtd = 보류(완료)**: `exp4v2_mtd_symmetric_execute.md`(epochs 100, imgsz 640, rect, seeds 42/1/2, synth_ratio 1.0)로 **이미 학습·평가 완료**. STEP 8에서 재실행하지 않고 **성능표에 aroma-CN-sym 값만 이식**(§F). mtd 값: baseline 0.9204 / random 0.9343 / **aroma-sym 0.9322** (self-contained). 4종 통합표에서는 §F 규약대로 삽입.

### 전제 (3종 모두)
1. **matrix_symmetric emit 완료**: symmetric profiling은 **기존 경로 `$AROMA_OUT/profiling/{ds}`에 그대로 작성**됨(별도 경로 없음). `--compat_mode symmetric`은 `matrix_symmetric` 없으면 **hard-fail**.
2. **per-dataset τ 사전스캔**: τ는 데이터셋마다 스케일 상이 → **재사용·τ=0.5 금지**. leather/severstal/aitex **각각 prescan 필요(현 TBD)**.
3. **CN 모델**: `$CN_MODELS/{ds}/best_model` (STEP 4 완료).

---

## 1. 환경변수

```python
import os, json
os.environ['AROMA_SCRIPTS'] = '/content/AROMA/scripts/aroma'
os.environ['AROMA_REF']     = '/content/AROMA'
os.environ['AROMA_OUT']     = f"{os.environ['DRIVE']}/aroma_output"
os.environ['AROMA_DATA']    = f"{os.environ['DRIVE']}/Aroma"
os.environ['DATASET_CONFIG']= os.environ.get('DATASET_CONFIG', '/content/AROMA/dataset_config.json')
os.environ['CN_MODELS']     = f"{os.environ['AROMA_OUT']}/controlnet_models"

# 이식 소스 (baseline/random) — controlnet STEP 8과 동일
os.environ['EXP4V2_REF']       = f"{os.environ['AROMA_OUT']}/exp4v2/_ref_20260705"       # leather/severstal
os.environ['EXP4V2_AITEX_REF'] = f"{os.environ['AROMA_OUT']}/exp4v2/_ref_20260706_aitex" # aitex
os.environ['AROMA_OUT_RANDOM'] = f"{os.environ['AROMA_OUT']}/synthetic_random"           # random 이식원 합성 dir

# symmetric aroma 산출 (비대칭 run과 충돌 방지 위해 별도 dir)
os.environ['SYN_CN_SYM']  = f"{os.environ['AROMA_OUT']}/synthetic_cn_sym"
os.environ['EXP4V2_CN_SYM'] = f"{os.environ['AROMA_OUT']}/newpipe/exp4v2_cn_sym"

DATASETS = ["mvtec_leather", "severstal", "aitex"]   # mtd 제외(보류)
with open(os.environ['DATASET_CONFIG']) as f: _cfg = json.load(f)

# per-dataset τ (prescan 후 채움; mtd 0.2348은 참고, 여기선 미사용)
TAU = {"mvtec_leather": None, "severstal": None, "aitex": None}   # ← §A prescan 결과로 교체
print("대상:", DATASETS)
```

---

## 2. STEP A — τ 사전스캔 (3종, CPU)

`compat_gate_cpu_diagnosis_execute.md §10`을 `DS`별로 실행 → `compat_tau_prescan_{ds}.json`의 `ds_tau`. **τ=0.5 금지.**

```python
import json, os
for ds in DATASETS:
    p = f"{os.environ['AROMA_OUT']}/diagnostics/{ds}/compat_gate/compat_tau_prescan_{ds}.json"
    if os.path.exists(p):
        TAU[ds] = json.load(open(p)).get('ds_tau')
    print(f"{ds:14s} τ = {TAU[ds]}")
    assert TAU[ds] and 0.0 < TAU[ds] < 0.5, f"{ds} τ 미확정/이상 — §10 prescan 먼저"
```

---

## 3. STEP B — selection (aroma 공통 후보, multi realism) ★ 필수 선행

aroma arm은 **selection → synthesis** 순서. 신 profiling 기반 선택을 데이터셋별로 먼저 생성해야 STEP C가 읽을 `sel_aroma`가 생긴다. (random arm은 이식되므로 selection 불필요 — aroma용 전용.)

```python
for ds in DATASETS:
    prof = f"{os.environ['AROMA_OUT']}/profiling/{ds}"          # 기존 경로 = matrix_symmetric 포함본
    sel  = f"{os.environ['AROMA_OUT']}/newpipe/{ds}/sel_aroma"
    os.environ['PROF'], os.environ['SEL'] = prof, sel
    !python $AROMA_SCRIPTS/roi_selection.py \
        --profiling_dir     $PROF \
        --prompts_dir       {os.environ['AROMA_OUT']}/prompts/{ds} \
        --sampling_strategy deficit_aware --score_mode realism \
        --top_k 200 --img_diversity_cap 1 \
        --class_mode multi --class_floor --per_pair_cap_frac 0.05 \
        --output_dir        $SEL
    print(f"✓ selection {ds} → {sel}")
```
> 로그: `stratified_pair_aware` + class별 floor 확인. leather는 소스 희소 → floor 미달 클래스 경고 가능(정직 기록). 산출 `sel_aroma/{roi_selected.json, roi_candidates.json}`이 STEP C의 `--roi_dir` 입력.

---

## 4. STEP C — aroma-CN-sym 합성 (compat symmetric + positive placement, GPU)

필터 설정이 바뀌었으므로 **새 output_dir(`$SYN_CN_SYM`)** 사용(캐시 오귀속 방지 — controlnet 문서 §441–444).

```python
for ds in DATASETS:
    prof = f"{os.environ['AROMA_OUT']}/profiling/{ds}"
    sel  = f"{os.environ['AROMA_OUT']}/newpipe/{ds}/sel_aroma"
    out  = f"{os.environ['SYN_CN_SYM']}/{ds}"
    tau  = TAU[ds]; assert tau and 0 < tau < 0.5
    os.environ.update({'PROF':prof,'SEL':sel,'OUT':out,'TAU':str(tau),
                       'NORMAL':_cfg[ds]['image_dir'],'CNP':f"{os.environ['CN_MODELS']}/{ds}/best_model"})
    !python $AROMA_SCRIPTS/generate_defects.py \
        --roi_dir     $SEL \
        --normal_dir  $NORMAL \
        --output_dir  $OUT \
        --method      controlnet \
        --controlnet_path  $CNP \
        --morphology_csv   $PROF/morphology_features.csv \
        --context_features $PROF/context_features.csv \
        --config           $PROF/recommended_config.yaml \
        --n_per_roi 3 --seed 42 --blend_mode seamless \
        --reject-clean-bg --min-bg-quality 0.7 --bg-blur-threshold 100.0 \
        --compat_mode symmetric --compat_threshold $TAU \
        --compat_matrix_json $PROF/compatibility_matrix.json \
        --cn_ar_threshold 2.5
```
> 로그: `compat gate ON: threshold=… mode=symmetric`, `placement-gate stats: fallback=M%`. **fallback>50%면 τ 과대**(§A 재확인). matrix_symmetric 없으면 hard-fail(설계).

---

## 5. STEP D — parity 재확인 (positive placement가 풀 축소 가능, CPU)

```python
import json, os
RANDOM_N = {"mvtec_leather": 64, "severstal": 2534, "aitex": 246}   # mtd 제외
for ds in DATASETS:
    ann = json.load(open(f"{os.environ['SYN_CN_SYM']}/{ds}/annotations.json"))
    live = [a for a in ann if not a.get('dry_run') and a.get('bbox')]
    n_arfb = sum(1 for a in live if a.get('method')=='copy_paste_arfallback')
    need = RANDOM_N[ds]; ok = len(live) >= need
    print(f"{'✓' if ok else '✗ PARITY FAIL'} {ds:14s} labelable={len(live)} "
          f"(ar_fallback={n_arfb}, {100*n_arfb/max(1,len(live)):.0f}%) needed>={need}")
```
> ✗이면 `--n_per_roi += 1` 후 STEP C 재실행(캐시로 부족분만 추가). `ar_fallback%` 기록 — 논문에 "aroma-CN 중 X%는 elongated이라 copy_paste, delta는 나머지 생성분 기여"로 명시.

---

## 6. STEP E — exp4v2 (baseline/random 이식 + aroma-CN-sym만 학습)

### E-1. baseline/random 이식 (aroma 키 제거) — controlnet STEP 8-A와 동일, mtd만 제외

```python
import pathlib, json
GRAFT_SOURCES = [
    (os.environ['EXP4V2_REF'],       {"mvtec_leather", "severstal"}),  # 20260705
    (os.environ['EXP4V2_AITEX_REF'], {"aitex"}),                       # 20260706_aitex
]
for s in (42, 1, 2):
    dst = pathlib.Path(f"{os.environ['EXP4V2_CN_SYM']}/_seeds/seed{s}"); dst.mkdir(parents=True, exist_ok=True)
    out = {}
    for ref, keep in GRAFT_SOURCES:
        res = json.load(open(f"{ref}/_seeds/seed{s}/exp4v2_results.json"))
        for ds, models in res.items():
            if ds not in keep: continue
            out[ds] = {m: {c:v for c,v in conds.items() if c != "aroma"} for m, conds in models.items()}
    json.dump(out, open(dst / "exp4v2_results.json", "w"), indent=2)
    print(f"seed{s}: 이식 {list(out)}")
```

### E-2. leather / severstal 학습 (imgsz 640, rect, epochs 100, seeds 42 1 2)

```python
!python $AROMA_SCRIPTS/experiments/exp4_v2_supervised_detection.py \
    --model yolov8n --condition all \
    --dataset_keys mvtec_leather severstal \
    --random_synthetic_dir $AROMA_OUT_RANDOM \
    --aroma_synthetic_dir  $SYN_CN_SYM \
    --real_data_dir        $AROMA_DATA \
    --output_dir           $EXP4V2_CN_SYM \
    --yolo_cache_dir       $AROMA_OUT/yolo_cache \
    --imgsz 640 --val_frac 0.3 --synth_ratio 1.0 \
    --baseline_epochs 100 --patience 50 --batch 64 --cache ram --rect \
    --seeds 42 1 2 --resume
```
> `--resume` + 이식 JSON → baseline/random skip, **aroma(CN-sym)만 학습** (2 ds × 3 seed = 6 run). 파라미터는 20260705와 완전 동일(변경 시 이식 무효).

### E-3. aitex 학습 (imgsz 256, no rect, epochs 300, seeds 1 2 42)

```python
!python $AROMA_SCRIPTS/experiments/exp4_v2_supervised_detection.py \
    --model yolov8n --condition all \
    --dataset_keys aitex \
    --random_synthetic_dir $AROMA_OUT_RANDOM \
    --aroma_synthetic_dir  $SYN_CN_SYM \
    --real_data_dir        $AROMA_DATA \
    --output_dir           $EXP4V2_CN_SYM \
    --yolo_cache_dir       $AROMA_OUT/yolo_cache \
    --imgsz 256 --val_frac 0.3 --synth_ratio 1.0 \
    --baseline_epochs 300 --patience 50 --batch 64 --cache ram \
    --seeds 1 2 42 --resume
```
> aitex는 20260706_aitex 규약(imgsz 256, **rect 미사용**, 300/50, seeds 1 2 42). 동일 `$EXP4V2_CN_SYM`에 aitex 키만 추가(exp4v2 --resume 병합).

---

## 7. STEP F — 4종 통합 성능표 (mtd 이식 포함)

```python
import json, os
cn  = json.load(open(f"{os.environ['EXP4V2_CN_SYM']}/exp4v2_results.json"))
ref = json.load(open(f"{os.environ['EXP4V2_REF']}/exp4v2_results.json"))
ref_aitex = json.load(open(f"{os.environ['EXP4V2_AITEX_REF']}/exp4v2_results.json"))
CP_REF = {"mvtec_leather": ref, "severstal": ref, "aitex": ref_aitex}

# mtd: 보류·완료 → 성능표에 aroma-sym만 이식 (baseline/random은 20260705 이식본)
MTD_AROMA_SYM = 0.9322   # exp4v2_mtd_symmetric_execute.md 결과 (100 ep, 640, rect)
mtd_ref = ref["mtd"]["yolov8n"]

print(f"{'dataset':<14}{'baseline':>9}{'random':>9}{'aroma-CP':>9}{'aroma-CNsym':>12}{'Δ(sym-R)':>9}{'Δ(sym-CP)':>10}")
# 3종
for ds in ["mvtec_leather","severstal","aitex"]:
    b=cn[ds]["yolov8n"]["baseline"]["map50"]; r=cn[ds]["yolov8n"]["random"]["map50"]
    a2=cn[ds]["yolov8n"]["aroma"]["map50"]; a1=CP_REF[ds][ds]["yolov8n"]["aroma"]["map50"]
    print(f"{ds:<14}{b:>9.4f}{r:>9.4f}{a1:>9.4f}{a2:>12.4f}{a2-r:>+9.4f}{a2-a1:>+10.4f}")
# mtd (이식): baseline/random=20260705, aroma=symmetric 완료값
b=mtd_ref["baseline"]["map50"]; r=mtd_ref["random"]["map50"]; a1=mtd_ref["aroma"]["map50"]
print(f"{'mtd(이식)':<14}{b:>9.4f}{r:>9.4f}{a1:>9.4f}{MTD_AROMA_SYM:>12.4f}{MTD_AROMA_SYM-r:>+9.4f}{MTD_AROMA_SYM-a1:>+10.4f}")
```
> per-seed paired delta도 3종은 `cn[ds]['yolov8n']['aroma']['per_seed']` vs `['random']['per_seed']`로 확인. **mtd는 완료 결과라 per-seed는 symmetric 파일에서 참조**(Δ는 이식 random 기준이면 20260705 random 사용 — 출처를 표에 병기).
> **aitex 관전점**: Δ(sym-CP) — symmetric 생성이 aroma-CP 0.4847(20260706_aitex 유일 positive)을 넘는지가 개선의 직접 증거. CN-CP 음수라도 CN-R 양수면 "생성 arm도 random은 이긴다"로 분리 해석.

---

## 8. 무결성 / 정직

- **mtd는 완료·보류** — 재학습 금지, aroma-sym 값(0.9322)만 표 이식. mtd near-ceiling이라 개선 성패 headline로 쓰지 말 것. arbiter는 **aitex**.
- **이식 유효 조건**: leather/severstal는 20260705와 (640/rect/100/val0.3/synth1.0/batch64/seeds 42 1 2), aitex는 20260706_aitex와 (256/no-rect/300/seeds 1 2 42) **완전 동일**. 하나라도 다르면 비교 불가.
- **τ 사후 튜닝 금지** — §A prescan 확정값 고정. 결과 보고 후 τ·seed·synth_ratio 변경 금지.
- **ar_fallback% 병기** — elongated는 양 arm copy_paste 처리, delta는 non-elongated 생성분 기여로 명시.
- **Δ의 random 출처 명시** — mtd 행은 baseline/random이 20260705 이식이고 aroma만 symmetric임을 표에 표기(출처 혼재 투명화).
- 시간·처리량 벤치 금지(load-test policy), pytest 금지(CLAUDE.md).
