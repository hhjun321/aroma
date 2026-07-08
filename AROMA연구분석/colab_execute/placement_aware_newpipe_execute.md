# AROMA 새 파이프라인 실행 가이드 (realism 선택 + ControlNet 합성 + compat 배치 게이트 + class_key)

> **ablation이 아니다.** placement-aware 전환(커밋 bad8b31) + class_key 라벨 상속(이번 수정)을 **채택할 새 AROMA 구성으로 확정**해 표준 3-way(baseline/random/aroma)로 검증한다.
> **실행 환경**: selection·random = CPU | aroma 합성(ControlNet) = GPU | exp4v2 = GPU (A100 권장)
> **데이터셋 역할** (4종 selection 로그 분석으로 확정):
> - **multi-class = mtd, severstal** — class_key(A)·per-class 밸런스 검증. (둘 다 profiling에 defect_type 채워짐 — 재프로파일 불필요)
> - **single-class = aitex, mvtec_leather** — placement·selection 검증. class_key 무효(단일).

---

## 0. 데이터셋별 구성

| DS | class_mode | compat τ | grayscale | 검증 목표 | 근거/주의 |
| --- | --- | --- | --- | --- | --- |
| **aitex** | single | 0.5 (ON) | ON | **placement(compat)** | coverage 77.8%·헤드룸 0.372 = 유일 실현 |
| **mtd** | multi 5 | 0.5 (ON) | ON | **class_key(A)+선택** | fray floor미달; 천장0.92 placement null 가능 |
| **severstal** | multi 4 | 0 (OFF) | ON | 대규모 multi 선택 | defect_type 4-class 기채움(재프로파일 불필요); full-frame이라 placement 미검증→OFF |
| **mvtec_leather** | single | 0 (OFF) | **OFF** | 참고(소표본) | 92결함·삼중무효; single 병합으로 최소 신호만, 헤드라인 금지 |

```python
DS_CFG = {
  "aitex":         dict(class_mode="single", tau=0.5, grayscale=True,  role="placement"),
  "mtd":           dict(class_mode="multi",  tau=0.5, grayscale=True,  role="class_key"),
  "severstal":     dict(class_mode="multi",  tau=0.0, grayscale=True,  role="multi_scale"),
  "mvtec_leather": dict(class_mode="single", tau=0.0, grayscale=False, role="reference"),
}
```

**구성 = aroma arm에만** realism 선택 + ControlNet 생성 + compat 배치. random/baseline은 통제군.
- **selection**: `--sampling_strategy deficit_aware --score_mode realism`. multi면 `--class_mode multi --class_floor --per_pair_cap_frac 0.05`, single이면 이 3개 **생략**(no-op).
  - ⚠️ `--rarity_temp` 미전달(기본 1.0). ≠1.0이면 레거시 deficit 재주입 → realism 비정합.
- **placement**: `--compat_threshold τ` + `--compat_matrix_json` + `--config` (τ>0일 때만).
- **정직성**: 풀 메서드 vs 베이스라인 비교다. 이득이 선택/배치/생성 중 어디서 오는지 confound 분리는 범위 밖.

---

## 1. 환경변수

```python
import os, json

os.environ['AROMA_SCRIPTS'] = '/content/AROMA/scripts/aroma'
os.environ['AROMA_REF']     = '/content/AROMA'   # repo root — ControlNet 경로 utils/ import 필수 (없으면 ModuleNotFoundError: utils)
os.environ['AROMA_OUT']     = f"{os.environ['DRIVE']}/aroma_output"
os.environ['AROMA_DATA']    = f"{os.environ['DRIVE']}/Aroma"
os.environ['DATASET_CONFIG']= os.environ.get('DATASET_CONFIG', '/content/AROMA/dataset_config.json')
os.environ['CN_MODELS']     = f"{os.environ['AROMA_OUT']}/controlnet_models"   # 4종 기학습 (재학습 불필요)

DS = 'aitex'   # ← aitex / mtd / severstal / mvtec_leather 중 선택
os.environ['DS'] = DS
os.environ['PROFILING_DIR'] = f"{os.environ['AROMA_OUT']}/profiling/{DS}"
os.environ['PROMPTS_DIR']   = f"{os.environ['AROMA_OUT']}/prompts/{DS}"
os.environ['COMPAT_JSON']   = f"{os.environ['PROFILING_DIR']}/compatibility_matrix.json"
os.environ['AROMA_CONFIG']  = f"{os.environ['PROFILING_DIR']}/recommended_config.yaml"
with open(os.environ['DATASET_CONFIG']) as f:
    _cfg = json.load(f)
os.environ['NORMAL_DIR'] = _cfg[DS]['image_dir']
os.environ['SEL_AROMA']        = f"{os.environ['AROMA_OUT']}/newpipe/{DS}/sel_aroma"
os.environ['AROMA_SYNTH_DIR']  = f"{os.environ['AROMA_OUT']}/newpipe/{DS}/synth_aroma"
os.environ['RANDOM_SYNTH_DIR'] = f"{os.environ['AROMA_OUT']}/newpipe/{DS}/synth_random"
os.environ['EXP_OUT']          = f"{os.environ['AROMA_OUT']}/newpipe/{DS}/exp4v2"
print("DS =", DS, "| NORMAL_DIR =", os.environ['NORMAL_DIR'])
```

## 2. 선행 확인

```python
import os, pathlib
for k, v in {
  'PROFILING_DIR': os.environ['PROFILING_DIR'], 'PROMPTS_DIR': os.environ['PROMPTS_DIR'],
  'COMPAT_JSON': os.environ['COMPAT_JSON'], 'AROMA_CONFIG': os.environ['AROMA_CONFIG'],
  'NORMAL_DIR': os.environ['NORMAL_DIR'],
  'CN_MODEL': f"{os.environ['CN_MODELS']}/{os.environ['DS']}/best_model",
}.items():
    print(f"{k:<14} {'OK' if pathlib.Path(v).exists() else 'MISSING':<8} {v}")
```

> **severstal 검증 노트 (재프로파일 불필요)**: severstal profiling은 이미 defect_type=class1~4를 담고 있어 multi-4로 정상 동작한다. selection 로그에 `stratified_pair_aware` 라인이 안 보이는 건 slack=0(floor 50×4=200 정확 충족 → backfill 블록 skip)일 뿐, 단일취급이 아니다. 확인은 산출물로:
> ```python
> import json, os, collections
> cand = json.load(open(f"{os.environ['SEL_AROMA']}/roi_candidates.json"))
> sel  = json.load(open(f"{os.environ['SEL_AROMA']}/roi_selected.json"))
> print("class_key 종류:", sorted({c.get('class_key') for c in cand}))          # {class1..4} 기대
> print("selected per-class:", dict(collections.Counter(s.get('class_key') for s in sel)))  # 50×4 기대
> ```
> severstal 실측: class_key={class1,class2,class3,class4}, selected=50/50/50/50 균등 확인됨. class_key(A)가 severstal에서도 실효(`_resolve_synth_class` 우선순위 0가 `name_to_id['classN']`로 해결).

## 3. selection — aroma (deficit_aware + realism)

```python
import os, json
cfg = DS_CFG[os.environ['DS']]
multi = "--class_mode multi --class_floor --per_pair_cap_frac 0.05" if cfg["class_mode"] == "multi" else ""
print(f"DS={os.environ['DS']} class_mode={cfg['class_mode']}  multi_flags={multi!r}")
!python $AROMA_SCRIPTS/roi_selection.py \
    --profiling_dir     $PROFILING_DIR \
    --prompts_dir       $PROMPTS_DIR \
    --sampling_strategy deficit_aware --score_mode realism \
    --top_k             200 \
    --img_diversity_cap 1 \
    --output_dir        $SEL_AROMA \
    {multi}
```

> **로그 확인**:
> - multi(mtd/severstal): `stratified_pair_aware: N classes` + class별 floor 충족 여부. floor 미달 클래스(예: mtd fray 32<40)는 과소합성 → per-class map 주의.
> - single(aitex/leather): plain `pair_aware_allocation`. leather는 92 distinct라 `selected<200`(bounded repetition) 정상 — 소표본.
> - **selected 수·distinct 크롭 수** 기록(공정성 근거).

**quality/roi_score 분산 확인 (realism 실효 점검):**
```python
import json, os, statistics
c = json.load(open(f"{os.environ['SEL_AROMA']}/roi_candidates.json"))
qs = [x['quality_score'] for x in c]; rs = [x['roi_score'] for x in c]
print(f"n={len(c)}  quality stdev={statistics.pstdev(qs):.4f}  roi_score stdev={statistics.pstdev(rs):.4f}")
```
> quality stdev ~0이면 realism이 legacy와 사실상 동일(quality 항 무효) → 새 선택 실효 낮음을 정직 기록.

## 4. 합성

### 4.1 aroma — ControlNet + compat 게이트 (GPU)

```python
import os
cfg = DS_CFG[os.environ['DS']]
os.environ['PROF'] = os.environ['PROFILING_DIR']
out = f"{os.environ['AROMA_SYNTH_DIR']}/{os.environ['DS']}"
gray = "" if cfg["grayscale"] else "--cn_no_grayscale"
compat = ""
if cfg["tau"] > 0.0:
    compat = (f"--compat_threshold {cfg['tau']} "
              f"--compat_matrix_json {os.environ['COMPAT_JSON']} "
              f"--config {os.environ['AROMA_CONFIG']}")
print(f"DS={os.environ['DS']}  τ={cfg['tau']}  gray={cfg['grayscale']}")
!python $AROMA_SCRIPTS/generate_defects.py \
    --roi_dir     $SEL_AROMA \
    --normal_dir  $NORMAL_DIR \
    --output_dir  {out} \
    --method      controlnet \
    --controlnet_path  $CN_MODELS/$DS/best_model \
    --morphology_csv   $PROF/morphology_features.csv \
    --context_features $PROF/context_features.csv \
    --config           $AROMA_CONFIG \
    --n_per_roi 3 --seed 42 \
    --blend_mode seamless \
    --reject-clean-bg --min-bg-quality 0.7 --bg-blur-threshold 100.0 \
    {compat} {gray}
```

> **AROMA_REF 필수**(§1) — 없으면 `ModuleNotFoundError: utils`(build_train_jsonl의 hint_generator import).
> **모델 재사용**: 기학습 `$CN_MODELS/$DS/best_model` 그대로. selection·placement 축과 무관.
> **`--local_staging` 금지**: ControlNet sidecar 캐시는 Drive 직결이어야 세션 재개 시 GPU 스킵.
> **로그**: `controlnet stats`(blank_rate<0.2), compat ON이면 `placement-gate stats: fallback=M%`. **fallback>50%면** placement no-op(legacy 회귀)로 "placement-aware" 주장 금지. aitex <50% 예상.
> **grayscale**: leather만 OFF(컬러 가죽), 나머지 ON.

### 4.2 random — 기존 copy-paste (무변경 통제군, CPU)

```python
import os
out = f"{os.environ['RANDOM_SYNTH_DIR']}/{os.environ['DS']}"
!python $AROMA_SCRIPTS/generate_random.py \
    --candidates_json $SEL_AROMA/roi_candidates.json \
    --normal_dir      $NORMAL_DIR \
    --output_dir      {out} \
    --top_k 200 --n_per_roi 3 --seed 42 \
    --local_staging
```

> random은 §3의 roi_candidates.json(동일 후보 풀)에서 균일추출 → copy_paste. compat·realism·controlnet 미적용.
> ⚠️ random은 img_diversity_cap 미적용 → 후보 중복배수 높은 DS(severstal 73.6·mtd 43.5)에서 같은 크롭 반복 가능(aroma는 distinct 보장). 다양성 비대칭을 결과 해석 시 감안.

## 5. class_key 상속 검증 (multi 데이터셋만 — mtd/severstal)

```python
import json, os, collections
if DS_CFG[os.environ['DS']]["class_mode"] != "multi":
    print("single-class DS — class_key 검증 N/A (전부 class 0)")
else:
    ann = json.load(open(f"{os.environ['AROMA_SYNTH_DIR']}/{os.environ['DS']}/annotations.json"))
    ck = collections.Counter(a.get("class_key") for a in ann)
    print(f"{os.environ['DS']}: n={len(ann)}  class_key={dict(ck)}")
```

> **mtd**: class_key가 fray 등 5종 문자열로 채워져야 함(우선순위 0 경로 실효). None이면 §3 재실행.
> **severstal**: class_key=class1~4 기채움(실측 확인) → 우선순위 0로 직접 해결. cluster_id/path fallback 불필요.

## 6. exp4v2 — baseline+random+aroma

```python
import os
cm = DS_CFG[os.environ['DS']]["class_mode"]
!python $AROMA_SCRIPTS/experiments/exp4_v2_supervised_detection.py \
    --model yolov8n --condition all \
    --dataset_keys $DS \
    --class_mode {cm} \
    --random_synthetic_dir $RANDOM_SYNTH_DIR \
    --aroma_synthetic_dir  $AROMA_SYNTH_DIR \
    --real_data_dir        $AROMA_DATA \
    --output_dir           $EXP_OUT \
    --yolo_cache_dir       $AROMA_OUT/yolo_cache \
    --imgsz 640 --val_frac 0.3 --synth_ratio 1.0 \
    --baseline_epochs 300 --patience 50 \
    --batch 64 --cache ram --rect \
    --seeds 42 1 2 --resume
```

> **class_mode = cfg**: aitex/leather=single(nc=1), mtd/severstal=multi.
> **multi 라벨 상속 확인**: 로그에 `multi mode: could not resolve defect class ... -> class 0` 경고가 없어야 함(class_key 우선순위 0 교정). mtd/severstal 둘 다 class_key 채워져 우선순위 0로 해결.

## 7. 판정 (paired aroma vs random)

```python
import json, os
r = json.load(open(f"{os.environ['EXP_OUT']}/exp4v2_results.json"))[os.environ['DS']]["yolov8n"]
for c in ("baseline","random","aroma"):
    cell = r.get(c, {}); m = cell.get("map50"); sd = (cell.get("std") or {}).get("map50")
    ps = {k: round(v['map50'],4) for k,v in cell.get("per_seed", {}).items()}
    ms = f"{m:.4f} ± {sd:.4f}" if isinstance(sd,float) else (f"{m:.4f}" if isinstance(m,float) else "N/A")
    print(f"  {c:<9} map50 {ms}   per_seed={ps}")
a, rd = r.get("aroma",{}).get("per_seed",{}), r.get("random",{}).get("per_seed",{})
seeds = sorted(set(a) & set(rd))
if seeds:
    d = [a[s]["map50"] - rd[s]["map50"] for s in seeds]
    print(f"\nΔ(aroma-random) per seed = {[round(x,4) for x in d]}  mean={sum(d)/len(d):+.4f}  "
          f"{'✅ 전 seed aroma 우위' if all(x>0 for x in d) else '❌ 비일관'}")
```

**판정**: aroma > random 전 seed 일관 **and** aroma > baseline → 채택 근거. aroma ≈ random → 이득 없음(천장/저노출/fallback 과다) 정직 보고. fallback률·quality 분산·헤드룸 함께 보고.

---

## 주의사항 / 정직성 (데이터셋별)

- **aitex** (placement 검증): single-class라 class_key(A) 미검증. compat fallback<50% 확인. 헤드룸 있어 placement 이득 관측 가능한 유일 케이스.
- **mtd** (class_key 검증): 천장 0.92 → placement 이득은 null 가능(정상, selection·class_key 신호에 집중). fray floor 미달 → fray per-class 저조 시 합성 편향 탓.
- **severstal** (multi, 재프로파일 불필요): defect_type 4-class 기채움 → class_key로 우선순위 0 해결(실측 50×4 균등). placement 미검증(full-frame)이라 compat OFF. 73.6배 중복 → random 편향 최대.
- **mvtec_leather** (참고): 92결함·삼중무효(천장+coverage4.7%+선택기아). single 병합해도 소표본 → **헤드라인 주장 금지**, 데이터-한계 명시. compat OFF(4.7% no-op).
- **공통**: realism 0.5/0.3/0.2·τ=0.5 사후 튜닝 금지. 풀 메서드 비교(confound 분리 아님).

## TODO / 후속

- **baseline 재고**: 현재 random = "AROMA 후보 풀 내 균일추출"(진짜 random-paste 아님). candidate가 AROMA 산물(실결함×점유cell). 개선 selection 재비교 후 naive-random baseline 도입 여부 결정.
- **moderated_score realism 가드 미구현**: realism+rarity_temp≠1.0이 레거시 deficit 재주입(bad8b31에 가드 없음). 가이드는 rarity_temp 미전달로 회피.
- **오프라인 clean-bg cell inventory 인덱스**: compat 게이트 런타임 재계산 → 사전 인덱스(placement devnote 후속).
- **CCI adaptive-range**(`aroma_step1_cci-adaptive-range.md`): 미구현, 독립 진행.
- **confound 분리 ablation**(placement×selection): 이득 귀속 필요 시 별도 작성.
