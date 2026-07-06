# MVTec-leather multi/single 비교 실행 가이드 (Workflow)

> **실행 환경**: 합성(step1-4) = CPU | exp4v2 = GPU (A100 권장)
> **목적**: leather를 **single-class(nc=1)와 multi-class(nc=5)** 두 config로 exp4v2 3-seed 비교 → aitex(tiled) 단일클래스 결과의 **config-강건성** 검증. 핵심 질문 = "aroma > random 신호가 class 분해와 무관하게 재현되는가".
> **aitex와의 차이**: leather는 **정사각(1024×1024) → 종횡비 병리 없음 → 타일링/재구성 불필요**. 단일클래스는 exp4v2 `--class_mode single`(기본값)이 네이티브로 처리(5개 test/{type}를 하나의 `defect`로 병합). aitex의 `prepare_aitex.py` 같은 데이터 재구성 **없음**.

---

## 0. 원리 (재구성 0의 근거)

`exp4_v2_supervised_detection.py:418-433`(generic mvtec 분기)이 `test/{color,cut,fold,glue,poke}`를 모두 `test_defect`로 수집. 이후:
- `--class_mode single`(기본) → 라벨 전부 `class_id=0` → **nc=1 단일클래스** (aitex 단일클래스와 동일 산출).
- `--class_mode multi` → 폴더명 열거 → **nc=5 유형별**.

**같은 데이터·같은 합성**을 쓰고 exp4v2 실행 시 `--class_mode`만 토글. 합성 재생성 불필요.

---

## 1. 환경변수

```python
import os
os.environ['AROMA_OUT']     = f"{os.environ['DRIVE']}/aroma_output"
os.environ['AROMA_SCRIPTS'] = "/content/AROMA/scripts/aroma"
os.environ['AROMA_DATA']    = f"{os.environ['DRIVE']}/Aroma"
os.environ['RANDOM_SYNTH_DIR'] = f"{os.environ['AROMA_OUT']}/synthetic_random"
os.environ['AROMA_SYNTH_DIR']  = f"{os.environ['AROMA_OUT']}/synthetic"
os.environ['LEATHER_MULTI']  = f"{os.environ['AROMA_OUT']}/exp4v2/leather_multi"
os.environ['LEATHER_SINGLE'] = f"{os.environ['AROMA_OUT']}/exp4v2/leather_single"
```

## 2. 선행 — leather 합성 존재 확인 (없으면 step1-4 생성)

```python
import os, pathlib, json
for label, root in [("aroma", os.environ['AROMA_SYNTH_DIR']),
                    ("random", os.environ['RANDOM_SYNTH_DIR'])]:
    ann = pathlib.Path(root)/"mvtec_leather"/"annotations.json"
    n = len(json.load(open(ann))) if ann.exists() else 0
    print(f"{label:<7} mvtec_leather annotations: {n}")
```

> 둘 다 > 0이어야 진행. 부재 시(스모크 때 aroma 부재 확인됨) `step1_execute.md`~`step4_execute.md`를 mvtec_leather 대상으로 실행 후 `exp3_execute.md` 0단계(random)로 생성.
> **합성은 1회만** — 아래 multi/single 두 run이 동일 synth를 공유한다(step3의 multi-stratified 선택이 AROMA의 가치이고, downstream 라벨 granularity는 별개 축).

## 3. exp4v2 — multi (nc=5)

```python
!python $AROMA_SCRIPTS/experiments/exp4_v2_supervised_detection.py \
    --model yolov8n --condition all \
    --dataset_keys mvtec_leather \
    --class_mode multi \
    --random_synthetic_dir $RANDOM_SYNTH_DIR \
    --aroma_synthetic_dir  $AROMA_SYNTH_DIR \
    --real_data_dir        $AROMA_DATA \
    --output_dir           $LEATHER_MULTI \
    --yolo_cache_dir       $AROMA_OUT/yolo_cache \
    --imgsz 640 --val_frac 0.3 --synth_ratio 1.0 \
    --baseline_epochs 300 --patience 50 \
    --batch 64 --cache ram --rect \
    --seeds 42 1 2 --resume
```

## 4. exp4v2 — single (nc=1, aitex 단일클래스 병렬)

```python
!python $AROMA_SCRIPTS/experiments/exp4_v2_supervised_detection.py \
    --model yolov8n --condition all \
    --dataset_keys mvtec_leather \
    --class_mode single \
    --random_synthetic_dir $RANDOM_SYNTH_DIR \
    --aroma_synthetic_dir  $AROMA_SYNTH_DIR \
    --real_data_dir        $AROMA_DATA \
    --output_dir           $LEATHER_SINGLE \
    --yolo_cache_dir       $AROMA_OUT/yolo_cache \
    --imgsz 640 --val_frac 0.3 --synth_ratio 1.0 \
    --baseline_epochs 300 --patience 50 \
    --batch 64 --cache ram --rect \
    --seeds 42 1 2 --resume
```

> **별도 output_dir 필수**: 둘 다 `results["mvtec_leather"]` 키라 같은 dir이면 덮어씀.
> **YOLO 캐시 자동 격리**: multi → `real_multi` leaf, single → `real` leaf (`--yolo_cache_dir` 공유해도 충돌 없음, 각 1회 빌드).
> **seed 규약**: 여기서는 aitex(tiled) 단일클래스 결과와의 직접 비교를 위해 **42 1 2**로 맞춤. ⚠️ 세션 표준(1/2/43)과 다름 — 4종 통합표 작성 시 규약 하나로 통일 필요(hygiene). single↔multi는 동일 seed여야 paired 비교 가능.

## 5. 비교 분석 — config 강건성 (paired)

```python
import json, os

def load(path):
    return json.load(open(f"{path}/exp4v2_results.json"))["mvtec_leather"]["yolov8n"]

for tag, path in [("multi(nc=5)", os.environ['LEATHER_MULTI']),
                  ("single(nc=1)", os.environ['LEATHER_SINGLE'])]:
    try:
        arms = load(path)
    except FileNotFoundError:
        print(f"{tag}: 미실행"); continue
    print(f"\n=== leather {tag} ===")
    for c in ("baseline", "random", "aroma"):
        cell = arms.get(c, {})
        m = cell.get("map50"); sd = (cell.get("std") or {}).get("map50")
        ps = cell.get("per_seed", {})
        ms = f"{m:.4f} ± {sd:.4f}" if isinstance(sd, float) else (f"{m:.4f}" if isinstance(m, float) else "N/A")
        print(f"  {c:<9} map50 {ms}   per_seed={ {k: round(v['map50'],4) for k,v in ps.items()} }")
    # paired aroma-random (같은 seed끼리)
    a, r = arms.get("aroma", {}).get("per_seed", {}), arms.get("random", {}).get("per_seed", {})
    seeds = sorted(set(a) & set(r))
    if seeds:
        d = [a[s]["map50"] - r[s]["map50"] for s in seeds]
        mean = sum(d)/len(d)
        allpos = all(x > 0 for x in d)
        print(f"  Δ(A-R) per seed = {[round(x,4) for x in d]}  mean={mean:+.4f}  "
              f"{'✅ 전 seed aroma 우위' if allpos else '❌ 비일관'}")
```

**판정 (config 강건성)**:
- **single·multi 둘 다에서 aroma>random 전 seed 일관** → aitex 결과가 우연이 아니라 **class config에 강건한 ROI 선택 가치** 입증(강력).
- 한쪽만 성립 → 어느 config에서 신호가 나오는지·왜인지 정직하게 서술(예: multi는 유형당 표본 희소로 noise, single은 병합으로 관측 가능 — aitex와 동형 해석).
- 둘 다 비유의 → leather는 signal 없음으로 보고(cherry-pick 금지).

---

## 주의사항 / 정직성

- **single을 "성능 부스터"로 오해 금지**: single은 유형당 표본 희소(leather ~18/type)를 병합해 **관측 가능성**을 높이는 것. 연구 신호는 어느 config든 aroma−random이며, config는 측정 렌즈일 뿐.
- **multi·single 둘 다 보고**(cherry-pick 금지) — single은 aitex-비교 가능한 aggregate 관점, multi는 per-class 관점.
- 동일 합성 pool 공유가 공정 — 두 run의 유일한 차이는 exp4v2 라벨 granularity.
- exp5(PRDC)·exp6(knn/rare)는 class-agnostic(crop/임베딩 기반)이라 leather에 그대로 적용 — 이 downstream 신호와 교차 검증 권장.
