# exp4v2 downstream — mtd, 개선(clean-grounded symmetric + positive placement) 검증

> **목적**: compat 게이트 개선(commit 6c8658f SGM + 6b027f7 타일링 + 362e7dd positive placement)이 **detector mAP를 실제로 올리는가** = **H1 벽 판정**. mtd 단일 데이터셋.
> **개선 활성 방법**: AROMA arm 합성 시 `--compat_mode symmetric --compat_threshold <τ>` → SGM matrix_symmetric + 64px 타일링 query + positive placement(scan·rank·place)가 한 번에 켜짐. `defect`(기본)면 legacy.
> **비교 설계**: baseline / random(통제) / **aroma-symmetric(개선)** + (격리) **aroma-defect(legacy)**. aroma-symmetric vs aroma-defect = 개선 순효과.
> ⚠️ **정직 경고 (mtd 한계)**: mtd는 **near-ceiling(baseline mAP50~0.92)** + multi-class. placement 이득은 **null일 수 있음(천장)** — 이건 개선 실패가 아니라 mtd headroom 부족 탓. mtd에서 flat이어도 개선 무효 결론 금지 → headroom 있는 aitex로 별도 검증. mtd는 (a) 개선이 회귀 안 일으키는지, (b) class_key/selection 정상인지, (c) 기제 로그(fallback률·positive 위치) 확인용.
> **실행 환경**: selection·random·prescan = CPU | aroma 합성(ControlNet)·exp4v2 = GPU(A100 권장).

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

DS = 'mtd'
os.environ['DS'] = DS
# ⚠️ matrix_symmetric emit된 신 profiling 사용 (profiling_symmetric_rebuild_verify 완료본)
os.environ['PROFILING_DIR'] = f"{os.environ['AROMA_OUT']}/profiling/{DS}"
os.environ['PROMPTS_DIR']   = f"{os.environ['AROMA_OUT']}/prompts/{DS}"
os.environ['COMPAT_JSON']   = f"{os.environ['PROFILING_DIR']}/compatibility_matrix.json"
os.environ['AROMA_CONFIG']  = f"{os.environ['PROFILING_DIR']}/recommended_config.yaml"
with open(os.environ['DATASET_CONFIG']) as f: _cfg = json.load(f)
os.environ['NORMAL_DIR'] = _cfg[DS]['image_dir']

os.environ['SEL_AROMA']         = f"{os.environ['AROMA_OUT']}/newpipe/{DS}/sel_aroma"
os.environ['AROMA_SYM_DIR']     = f"{os.environ['AROMA_OUT']}/newpipe/{DS}/synth_aroma_sym"    # 개선(symmetric)
os.environ['AROMA_DEF_DIR']     = f"{os.environ['AROMA_OUT']}/newpipe/{DS}/synth_aroma_def"    # legacy(defect) ablation
os.environ['RANDOM_SYNTH_DIR']  = f"{os.environ['AROMA_OUT']}/newpipe/{DS}/synth_random"
os.environ['EXP_OUT']           = f"{os.environ['AROMA_OUT']}/newpipe/{DS}/exp4v2_sym"
os.environ['OUT_DIR']           = f"{os.environ['AROMA_OUT']}/diagnostics/{DS}/compat_gate"
os.makedirs(os.environ['OUT_DIR'], exist_ok=True)
print("DS =", DS, "| NORMAL_DIR =", os.environ['NORMAL_DIR'])
```

## 2. 선행 확인 (matrix_symmetric 필수)

```python
import os, json, pathlib
for k in ('PROFILING_DIR','COMPAT_JSON','AROMA_CONFIG','NORMAL_DIR'):
    print(f"{k:<14} {'OK' if pathlib.Path(os.environ[k]).exists() else 'MISSING':<8} {os.environ[k]}")
print(f"CN_MODEL       {'OK' if pathlib.Path(f'{os.environ['CN_MODELS']}/{DS}/best_model').exists() else 'MISSING'}")
c = json.load(open(os.environ['COMPAT_JSON']))
have = [k for k in ('matrix','matrix_symmetric','P_def_patch','clean_dist','symmetric_epsilon') if k in c]
print("compat keys:", have)
assert 'matrix_symmetric' in c, "matrix_symmetric 없음 → profiling_symmetric_rebuild_verify_execute.md 먼저 실행"
```

## 3. selection (aroma 공통 후보 — multi realism)

```python
!python $AROMA_SCRIPTS/roi_selection.py \
    --profiling_dir     $PROFILING_DIR \
    --prompts_dir       $PROMPTS_DIR \
    --sampling_strategy deficit_aware --score_mode realism \
    --top_k 200 --img_diversity_cap 1 \
    --class_mode multi --class_floor --per_pair_cap_frac 0.05 \
    --output_dir        $SEL_AROMA
```
> ⚠️ `--rarity_temp` 미전달(realism 정합). 로그 `stratified_pair_aware` + class별 floor 확인(fray 32<40 과소 주의).

## 4. τ 사전스캔 (symmetric 스케일, 타일링-aware)

`compat_gate_cpu_diagnosis_execute.md §10` 셀 실행(DS=mtd) → `compat_tau_prescan_mtd.json`의 `ds_tau`. **τ=0.5 금지.** 로컬 참고치 mtd≈0.23.

```python
import json, os
p = f"{os.environ['OUT_DIR']}/compat_tau_prescan_{DS}.json"
if os.path.exists(p):
    TAU = json.load(open(p)).get('ds_tau')
else:
    TAU = 0.2348   # 로컬 사전스캔 참고치 (Colab §10 미실행 시 임시)
os.environ['TAU'] = str(TAU)
print("compat_threshold τ =", TAU)
assert TAU and 0.0 < TAU < 0.5, "τ 이상 — §10 재확인"
```

## 5. aroma 합성 — 개선(symmetric) + legacy(defect) 두 arm

### 5.1 aroma-symmetric (개선: SGM + 타일링 + positive placement) — GPU

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
    --compat_matrix_json $COMPAT_JSON
```
> **개선 활성 확인(로그)**: `compat gate ON: threshold=… mode=symmetric`. positive placement가 위치를 scan·rank·place로 결정. `placement-gate stats: fallback=M%` — fallback은 non-void 후보 없거나 best<τ인 normal(재픽). **fallback 과다(>50%)면 τ 과대**(사전스캔 재확인).
> matrix_symmetric 없으면 **hard-fail**(설계).

### 5.2 aroma-defect (legacy 게이트 — 개선효과 격리용) — GPU

```python
import os
out = f"{os.environ['AROMA_DEF_DIR']}/{DS}"
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
    --compat_mode defect --compat_threshold $TAU \
    --compat_matrix_json $COMPAT_JSON
```
> 동일 조건, **게이트 모드만 defect**(legacy image-mean matrix + crop-size query + reject-only). aroma-symmetric과의 차이 = 개선 순효과. (τ=0.5 아님에 주의 — defect 스케일에선 τ 의미 다르나 통제 목적상 동일 τ 유지; 순수 legacy는 별도.)

## 6. random arm (통제군, 무변경 CPU)

```python
import os
out = f"{os.environ['RANDOM_SYNTH_DIR']}/{DS}"
!python $AROMA_SCRIPTS/generate_random.py \
    --candidates_json $SEL_AROMA/roi_candidates.json \
    --normal_dir      $NORMAL_DIR --output_dir {out} \
    --top_k 200 --n_per_roi 3 --seed 42 --local_staging
```
> random은 개선 무관(positive placement·compat 미적용). 통제군 그대로.

## 7. exp4v2 — baseline / random / aroma (symmetric)

```python
!python $AROMA_SCRIPTS/experiments/exp4_v2_supervised_detection.py \
    --model yolov8n --condition all --dataset_keys $DS --class_mode multi \
    --random_synthetic_dir $RANDOM_SYNTH_DIR \
    --aroma_synthetic_dir  $AROMA_SYM_DIR \
    --real_data_dir        $AROMA_DATA \
    --output_dir           $EXP_OUT \
    --yolo_cache_dir       $AROMA_OUT/yolo_cache \
    --imgsz 640 --val_frac 0.3 --synth_ratio 1.0 \
    --baseline_epochs 300 --patience 50 --batch 64 --cache ram --rect \
    --seeds 42 1 2 --resume
```
> **개선효과 격리(선택)**: `--aroma_synthetic_dir $AROMA_DEF_DIR`로 한 번 더 실행(출력 `${EXP_OUT}_def`) → aroma-symmetric vs aroma-defect 대조.
> 로그: `multi mode ... -> class 0` 경고 없어야(class_key 정상).

## 8. 판정

```python
import json, os
def load(exp_out):
    return json.load(open(f"{exp_out}/exp4v2_results.json"))[DS]["yolov8n"]
r = load(os.environ['EXP_OUT'])
for c in ("baseline","random","aroma"):
    cell=r.get(c,{}); m=cell.get("map50"); sd=(cell.get("std") or {}).get("map50")
    ps={k:round(v['map50'],4) for k,v in cell.get("per_seed",{}).items()}
    print(f"  {c:<9} map50 {m if m is None else round(m,4)} ± {sd if sd is None else round(sd,4)}  per_seed={ps}")
a, rd = r.get("aroma",{}).get("per_seed",{}), r.get("random",{}).get("per_seed",{})
seeds=sorted(set(a)&set(rd))
if seeds:
    d=[a[s]["map50"]-rd[s]["map50"] for s in seeds]
    print(f"Δ(aroma_sym - random) per seed={[round(x,4) for x in d]} mean={sum(d)/len(d):+.4f} "
          f"{'✅ 전seed aroma우위' if all(x>0 for x in d) else '❌ 비일관/무이득'}")
```

### 판정 규칙 (사전 등록)

| 관측 | 해석 |
|---|---|
| aroma_sym > random 전 seed **and** > baseline | 개선이 downstream 이득 (H1 벽 넘음) — 단 mtd 천장 고려, aitex 확인 필요 |
| aroma_sym ≈ random ≈ baseline (**flat**) | **mtd near-ceiling으로 placement 이득 측정 불가** — 개선 무효 결론 **금지**. headroom 있는 aitex 재검증. 기제(fallback률·positive 로그)만 확인 |
| aroma_sym < random/baseline | 회귀 — 개선이 해로움(조합다양성 과축소? τ 과대?) → fallback률·H1 조합다양성 점검 |
| aroma_sym ≈ aroma_def (§7 격리) | 개선이 mtd downstream 무차이 (천장 or H1) |

## 무결성 / 정직

- **mtd 결과를 개선 성패 headline으로 쓰지 말 것** — near-ceiling. placement 이득 판정의 arbiter는 headroom 있는 **aitex**(별도 실행).
- **개선 순효과** = aroma-symmetric vs aroma-defect(§5.1 vs §5.2, 동일 selection·seed·CN). random은 무변경 통제.
- τ는 §4 사전스캔 확정값. defect arm(§5.2)의 동일 τ는 통제 목적이지 legacy 최적 아님(정직 표기).
- **H1 벽 유효**: 게이트/배치가 기제로 완벽해도 copy-paste/controlnet 재조합이 무정보면 flat 가능 — 이건 배치개선의 한계가 아니라 파이프라인 상류(생성 novelty) 문제. flat이면 3단계(생성 arm) 검토.
- 사후 튜닝 금지(τ·seed·synth_ratio 결과 보고 변경 금지). 테스트코드·pytest 금지(CLAUDE.md).
