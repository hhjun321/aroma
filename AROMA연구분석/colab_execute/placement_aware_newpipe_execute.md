# AROMA 새 파이프라인 실행 가이드 (realism 선택 + ControlNet 합성 + compat 배치 게이트 + class_key)

> **이건 ablation이 아니다.** placement-aware 전환(커밋 bad8b31) + class_key 라벨 상속(이번 수정)을 **채택할 새 AROMA 구성으로 확정**해 돌리는 가이드다. confound 분리(2×2 A/B/C/D)가 아니라, 새 파이프라인을 표준 3-way(baseline/random/aroma)로 검증한다.
> **실행 환경**: selection(step3)·random 합성 = CPU | aroma 합성(ControlNet) = GPU | exp4v2 = GPU (A100 권장)
> **1차 대상 = aitex** (clean-patch coverage 77.8%, 헤드룸 0.372 — compat 배치가 물리적으로 실현되는 데이터셋). leather/severstal 주의는 하단.

---

## 0. 구성 (arm별 = 표준 augmentation 비교)

| arm | selection | synthesis | placement | 비고 |
| --- | --- | --- | --- | --- |
| **baseline** | — | — (real only) | — | 무증강 하한 |
| **random** | 균일 랜덤 | **copy_paste** | 랜덤 | 기존 방식 **무변경** (`generate_random.py`) |
| **aroma** | deficit_aware + **realism** | **controlnet** (기존 모델) | **compat 게이트** (τ>0) | 새 파이프라인 |

- 새 구성 = **aroma arm에만** realism 선택 + ControlNet 생성 + compat 배치. random/baseline은 통제군.
- **정직성**: 이건 "풀 메서드 vs 베이스라인" 비교다. 이득이 선택/배치/생성 중 어디서 오는지의 **confound 분리는 이 가이드 범위 밖**(그건 별도 ablation). 여기선 "새 AROMA가 random·baseline을 이기는가"만 본다.

**선택 축 플래그** (step3 `roi_selection.py`): `--sampling_strategy deficit_aware --score_mode realism`
- realism = `0.5·ctx + 0.3·morph + 0.2·quality` (deficit 항 폐기, quality 승격).
- ⚠️ `deficit_aware`는 이제 **배분 전략** 이름일 뿐 — realism 하에서 순위는 realism roi_score로 매겨짐(deficit 항 없음). AROMA 가치는 deficit 순위가 아니라 **다중클래스 대칭 배분(class_floor)**. `--rarity_temp`는 **건드리지 말 것**(기본 1.0이 아니면 레거시 deficit가 재주입되어 realism과 비정합).

**배치 축 플래그** (step4 `generate_defects.py`): `--compat_threshold <τ>` + `--compat_matrix_json` + `--config` (셋 다 있어야 게이트 활성).

---

## 1. 환경변수

```python
import os, json

os.environ['AROMA_SCRIPTS'] = '/content/AROMA/scripts/aroma'
os.environ['AROMA_OUT']     = f"{os.environ['DRIVE']}/aroma_output"
os.environ['AROMA_DATA']    = f"{os.environ['DRIVE']}/Aroma"
os.environ['DATASET_CONFIG']= os.environ.get('DATASET_CONFIG', '/content/AROMA/dataset_config.json')

DS = 'aitex'
os.environ['DS'] = DS
os.environ['PROFILING_DIR'] = f"{os.environ['AROMA_OUT']}/profiling/{DS}"
os.environ['PROMPTS_DIR']   = f"{os.environ['AROMA_OUT']}/prompts/{DS}"
os.environ['COMPAT_JSON']   = f"{os.environ['PROFILING_DIR']}/compatibility_matrix.json"
os.environ['AROMA_CONFIG']  = f"{os.environ['PROFILING_DIR']}/recommended_config.yaml"
os.environ['CN_MODELS']     = f"{os.environ['AROMA_OUT']}/controlnet_models"   # 4종 기학습 (재학습 불필요)
with open(os.environ['DATASET_CONFIG']) as f:
    _cfg = json.load(f)
os.environ['NORMAL_DIR'] = _cfg[DS]['image_dir']
# selection 산출물 (roi_selected.json + roi_candidates.json)
os.environ['SEL_AROMA']  = f"{os.environ['AROMA_OUT']}/newpipe/sel_aroma"
# 합성 루트 — generate_*는 {루트}/{DS}에 기록, exp4v2는 {루트}를 받음
os.environ['AROMA_SYNTH_DIR']  = f"{os.environ['AROMA_OUT']}/newpipe/synth_aroma"
os.environ['RANDOM_SYNTH_DIR'] = f"{os.environ['AROMA_OUT']}/newpipe/synth_random"
os.environ['EXP_OUT'] = f"{os.environ['AROMA_OUT']}/newpipe/exp4v2"
print("NORMAL_DIR =", os.environ['NORMAL_DIR'])
```

## 2. 선행 확인

```python
import os, pathlib
checks = {
  'PROFILING_DIR': os.environ['PROFILING_DIR'],
  'PROMPTS_DIR':   os.environ['PROMPTS_DIR'],
  'COMPAT_JSON':   os.environ['COMPAT_JSON'],
  'AROMA_CONFIG':  os.environ['AROMA_CONFIG'],
  'NORMAL_DIR':    os.environ['NORMAL_DIR'],
  'CN_MODEL':      f"{os.environ['CN_MODELS']}/{os.environ['DS']}/best_model",
}
for k, v in checks.items():
    print(f"{k:<14} {'OK' if pathlib.Path(v).exists() else 'MISSING':<8} {v}")
```

> 전부 OK여야 진행. `COMPAT_JSON`/`AROMA_CONFIG` 부재 → compat 게이트 자동 OFF(placement-blind로 회귀). `CN_MODEL` 부재 → 4.1을 copy_paste로 대체하거나 `controlnet_aroma_arm_execute.md` STEP 1–4로 해당 DS 학습. (4종 기학습 상태면 OK.)

## 3. selection — aroma (deficit_aware + realism)

```python
# aitex는 dataset_config.json class_mode:multi → stratified allocation 적용
# (step3_execute.md 규약과 동일). roi_candidates.json(전체 후보 풀)도 함께 저장됨 → random arm이 재사용.
!python $AROMA_SCRIPTS/roi_selection.py \
    --profiling_dir     $PROFILING_DIR \
    --prompts_dir       $PROMPTS_DIR \
    --sampling_strategy deficit_aware --score_mode realism \
    --top_k             200 \
    --img_diversity_cap 1 \
    --output_dir        $SEL_AROMA \
    --class_mode multi --class_floor --per_pair_cap_frac 0.05
```

> **quality_score 분산 확인**(로그): ~0이면 realism이 legacy와 거의 동일(quality 항 무효) → 새 선택의 실효가 낮음을 기록. `--rarity_temp` 미전달(기본 1.0) → 순위 = realism roi_score verbatim(정합).

## 4. 합성

### 4.1 aroma — ControlNet + compat 게이트 (GPU)

```python
import os
TAU = 0.5   # compat 게이트 임계, 사전 고정
os.environ['PROF'] = os.environ['PROFILING_DIR']
out = f"{os.environ['AROMA_SYNTH_DIR']}/{os.environ['DS']}"
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
    --compat_threshold {TAU} \
    --compat_matrix_json $COMPAT_JSON \
    --config $AROMA_CONFIG
```

> **모델 재사용 — 재학습 불필요**: ControlNet 모델은 selection·placement 축과 무관(ROI별 hint+프롬프트로 결함 생성만). 기학습 `$CN_MODELS/$DS/best_model` 그대로 사용. `controlnet_aroma_arm_execute.md` STEP 1–4는 skip.
> **`--config` 이중 역할**: controlnet hint bin_edges = compat 게이트 bin_edges 동일 파일. (위 명령에 `--config`가 두 번 나오나 같은 값 — argparse가 마지막 값 사용, 무해. 한 번만 써도 됨.)
> **`--local_staging` 금지**: ControlNet은 이미지별 sidecar 캐시가 Drive 직결이어야 세션 재개 시 GPU 스킵.
> **로그 확인**: `controlnet stats`(blank_rate<0.2), `placement-gate stats: ... fallback=M%`. **compat fallback >50%면** 배치 게이트가 사실상 no-op(대부분 legacy 위치로 회귀) → 그 실행에 "placement-aware" 주장 금지. aitex는 <50% 예상.
> **선택 필터**: 세장형 bbox 왜곡 방지 `--cn_ar_threshold 2.5`(자동 copy_paste 폴백), 구조 반복 배경 오배치 차단 `--texture-dist-threshold`는 필요 시 추가(`controlnet_aroma_arm_execute.md` 5-0/5-0b 참조).

> **(대안) copy_paste aroma**: ControlNet 없이 새 선택+compat만 볼 때 — `--method copy_paste --n_per_roi 3 --feather_px 4 --local_staging` + 동일 `--compat_threshold/--compat_matrix_json/--config`. CPU. controlnet과 비교하려면 별도 `--output_dir`로.

### 4.2 random — 기존 copy-paste (무변경, CPU)

```python
import os
out = f"{os.environ['RANDOM_SYNTH_DIR']}/{os.environ['DS']}"
# candidates_json = 3의 roi_candidates.json (동일 후보 풀에서 균일 랜덤). compat 게이트 없음(통제군).
!python $AROMA_SCRIPTS/generate_random.py \
    --candidates_json $SEL_AROMA/roi_candidates.json \
    --normal_dir      $NORMAL_DIR \
    --output_dir      {out} \
    --top_k 200 --n_per_roi 3 --seed 42 \
    --local_staging
```

> random arm은 **의도적으로 기존 copy-paste 그대로** — 새 파이프라인의 대조군. compat 게이트·realism·controlnet 미적용.

## 5. class_key 상속 검증 (이번 수정)

```python
import json, os, collections
ann = json.load(open(f"{os.environ['AROMA_SYNTH_DIR']}/{os.environ['DS']}/annotations.json"))
ck = collections.Counter(a.get("class_key") for a in ann)
print(f"aroma: n={len(ann)}  class_key={dict(ck)}")
```

> **기대**: class_key가 None이 아니라 aitex 결함 유형 문자열. None만 나오면 구 roi_selected.json → §3 재실행. 이 필드가 §6 multi-mode YOLO의 경로 재파싱 없는 클래스 해결을 가능케 함.

## 6. exp4v2 — baseline+random+aroma (multi 모드)

```python
!python $AROMA_SCRIPTS/experiments/exp4_v2_supervised_detection.py \
    --model yolov8n --condition all \
    --dataset_keys $DS \
    --class_mode multi \
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

> **`--class_mode multi` 필수** — class_key 수정의 실효 범위. single이면 전부 class 0.
> **라벨 상속 확인**: 로그에 `multi mode: could not resolve defect class ... -> defaulting to class 0` **경고가 사라졌는지** 확인. 수정 전 aitex는 source_roi 경로가 `test/{type}` 구조 아니거나 stale하면 이 경고 후 class 0 강등됐다. class_key 우선순위 0가 교정.

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

**판정**:
- aroma > random 전 seed 일관 **and** aroma > baseline → 새 파이프라인 채택 근거.
- aroma ≈ random → 새 구성의 이득 없음(천장/저노출/fallback 과다 가능) — 정직 보고, cherry-pick 금지.
- fallback률·quality 분산·헤드룸을 함께 보고해 "왜 이득/무이득인지" 설명.

---

## 주의사항 / 정직성

- **풀 메서드 비교, confound 분리 아님**: aroma에만 realism+controlnet+compat이 겹쳐 있어, 이득이 어느 요소에서 오는지는 이 가이드로 알 수 없다. 분리하려면 별도 ablation 필요(요청 시 작성).
- **leather 이중 무효**: 천장(0.83) + compat coverage 4.7%(~95% fallback) → placement 이득 주장 금지.
- **severstal 미검증**: full-frame·clean-bg 게이트 의미 불명 → 별도 검증 전 주장 금지.
- **aitex 헤드룸/coverage가 유일 실현 조건**: 조건부 주장만 — "헤드룸 있고 clean-patch coverage 높은 도메인에서만 이득 가능".
- **가중치·τ 사후 튜닝 금지**: realism 0.5/0.3/0.2, τ=0.5는 결과 보기 전 고정.
- **class_key 재현성**: annotations.json에 키 추가(파일 변화)되나, 경로 재파싱 성공 케이스는 라벨 .txt 불변. 실효는 재파싱 실패→class 0 강등 교정. single 모드·구 annotations.json 불변.

## TODO / 후속

- **오프라인 clean-bg cell inventory 인덱스**: compat 게이트 런타임 `_extract_context_features` 재계산 → 사전 인덱스 대체(placement devnote 후속).
- **moderated_score realism 가드 미구현**: realism + rarity_temp≠1.0 조합이 레거시 deficit 재주입(커밋 bad8b31에 가드 없음). 본 가이드는 rarity_temp 미전달로 회피.
- **CCI adaptive-range**(`aroma_step1_cci-adaptive-range.md`): 미구현 — 독립 진행.
- **confound 분리 ablation**(placement×selection 2×2): 이득 귀속이 필요해지면 별도 작성.
