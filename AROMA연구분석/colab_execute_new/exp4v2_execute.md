# exp4v2 — YOLOv8n Supervised Detection 실행 가이드 (sym_final)

> 이 문서는 `colab_execute_new/_SPEC.md`를 **정본**으로 따른다. STEP 0 환경 셀은 `_SPEC §1`을 그대로 복사한 것이며, 명령은 `_SPEC §3 exp4v2`만 사용한다. env·output 루트 규약을 재발명하지 않는다.

**목적**: sym_final 파이프라인(step4 산출 `synth_aroma`/`synth_random`)을 downstream **지도학습 결함 검출기**(YOLOv8n)로 평가한다. `baseline`(real만) / `random`(real + random 합성) / `aroma`(real + AROMA-symmetric 합성) 3조건의 mAP50을 비교해 **AROMA ROI 선택·배치가 검출기 학습에 기여하는가**를 판정한다.

**설계 — fresh 전조건 학습**: legacy exp4(PaDiM 등 unsupervised AD)를 **폐기**하고 exp4v2(YOLOv8n supervised detection)로 대체한다. 세 조건 모두 **COCO pretrained에서 처음부터(scratch) 독립 학습** — baseline/random/aroma가 서로의 weight를 재사용하지 않으며(graft 미사용), 합성 결함을 **지도학습 레이블 데이터**로 사용한다. 평가셋은 **항상 real 결함 이미지만**(합성은 train에만).

**실행 순서 체인**: `phase0 → step1 → step2 → step3 → step5(CN 학습+τ) → step4(생성) → **exp4v2(이 문서)**`.

**환경**: YOLO 학습은 **GPU 필수** (Colab Pro A100 권장). 판정·집계는 CPU.

**전제 (step4 완료)**: 데이터셋별로 아래가 존재해야 한다.
- AROMA arm: `S('synth_aroma', ds)/annotations.json` (+ `images/`) — step4 생성 위치. exp4v2엔 루트 `S('synth_aroma')`만 넘기고 스크립트가 `/{ds}`를 붙여 이 경로를 찾는다.
- Random arm: `S('synth_random', ds)/annotations.json` (+ `images/`) — 동일 규약(루트 `S('synth_random')` + `/{ds}`).

**데이터셋**: v2-1 4종 `severstal · mvtec_leather · mtd · aitex`. aitex = tiled(256×256/stride128, single-class).

---

## 워크플로우 개요

| STEP | 내용 | 런타임 |
|------|------|--------|
| 0 | 공통 환경 셀 (`_SPEC §1` 그대로) + 전제 확인 | CPU |
| 1 | 패키지 설치 & GPU 확인 | CPU |
| 2 | 3종(severstal · mvtec_leather · mtd) 실행 — multi / 640 / rect / 100ep / seeds 42 1 2 | **GPU** |
| 3 | aitex 실행 — single / 256 / no-rect / 300ep / seeds 1 2 42 | **GPU** |
| 4 | 판정 (baseline/random/aroma + Δ + per-seed 부호 일치) | CPU |

> **2 그룹 파라미터 상이** (`_SPEC §4`): 3종과 aitex는 imgsz·epochs·rect·class_mode·seed 순서가 다르다. **하나라도 섞으면 비교 불가** — 반드시 그룹별로 분리 실행한다.

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

### exp4v2 전용 경로 (env로 export)

```python
# 루트 인자 — 스크립트가 /{ds}를 붙인다 (stage-first 규약, _SPEC §2)
os.environ['SYNTH_AROMA']  = S('synth_aroma')     # sym_final/synth_aroma
os.environ['SYNTH_RANDOM'] = S('synth_random')    # sym_final/synth_random
os.environ['EXP4V2_OUT']   = S('exp4v2')          # sym_final/exp4v2
os.environ['YOLO_CACHE']   = f"{os.environ['AROMA_OUT']}/yolo_cache"

for k in ('SYNTH_AROMA', 'SYNTH_RANDOM', 'EXP4V2_OUT', 'YOLO_CACHE', 'AROMA_DATA'):
    print(f"{k:<14} {os.environ[k]}")
```

### 전제 확인 (step4 산출 — 모두 ✓여야 진행)

```python
import pathlib, json

print(f"{'데이터셋':<16} {'aroma_ann':>10} {'aroma_img':>10} {'random_ann':>11} {'random_img':>11}")
print("-" * 62)
need = []
for ds in DATASETS:
    row = []
    for root in (os.environ['SYNTH_AROMA'], os.environ['SYNTH_RANDOM']):
        ann = pathlib.Path(f"{root}/{ds}/annotations.json")
        n_ann = sum(1 for a in json.load(open(ann))
                    if not a.get("dry_run") and a.get("bbox")) if ann.exists() else 0
        img_dir = pathlib.Path(f"{root}/{ds}/images")
        n_img = len(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))) if img_dir.exists() else 0
        if n_ann == 0: need.append(f"{root}/{ds}")
        row += [n_ann, n_img]
    a_ann, a_img, r_ann, r_img = row
    print(f"{ds:<16} {a_ann:>10} {a_img:>10} {r_ann:>11} {r_img:>11}")
print("\nMISSING (labelable=0):", need)
```

> 4종 모두 `aroma_ann > 0` **및** `random_ann > 0`이어야 유효하다. `0`이면 해당 데이터셋의 step4(생성)를 먼저 완료한다.

---

## STEP 1 — 패키지 설치 & GPU 확인

```python
!pip install ultralytics -q
!pip install opencv-python-headless -q   # cv2 (합성 bbox 추출)
```

```python
!nvidia-smi
```

> ultralytics 설치 후 런타임 재시작(Runtime → Restart runtime)이 필요하면 STEP 0 환경 셀을 다시 실행한다.

---

## 운영 노트 — synth_ratio 스윕 & arm parity 범위

**① synth_ratio 스윕은 재생성 불요(한 번 크게 생성 → ratio만 변경)**: exp4v2는 로드된 합성 풀에서 `cap=int(n_real_train*synth_ratio)`만큼 **동일 seed 결정적 subsample**한다(`exp4_v2_supervised_detection.py:2418-2434`). 따라서 step4(`generate_defects`)를 **최대 ratio(≥1.0)를 채울 만큼 크게** 한 번 생성(`--n_per_roi` 상향)해두면, `--synth_ratio`만 바꿔 여러 번 실행하면 된다. 주의: **ratio마다 `--output_dir` 분리**(같은 dir이면 `--resume` skip에 걸려 재학습 안 됨), `--seed`·`--val_frac` 고정.

```python
for R in ["1.0", "0.8", "0.6", "0.4"]:
    !python $AROMA_SCRIPTS/experiments/exp4_v2_supervised_detection.py \
        ... (동일 인자) ... \
        --synth_ratio $R --output_dir ${EXP4V2_OUT}/ratio_$R
```

**② arm parity 범위(현행 = 총량 동수만)**: `synth_ratio` cap은 random/aroma/casda에 **동일 cap·동일 seed**를 적용해 **전체 합성 개수**를 맞춘다. **클래스별 개수·라벨화-후 실개수는 강제 매칭하지 않는다**(uniform subsample). 이는 의도적 — 합성 결함의 클래스 분포·bbox 라벨 수율은 AROMA 선택·배치의 **결과(post-treatment)**라, 강제 동일화하면 AROMA의 정당한 이득 경로를 지우는 **bad control**이 된다. 현행은 `n_synth_per_class`·distinct sources 로그로 **계측·보고**만 한다(분해 해석용). per-class stratified cap + 라벨화-후 동수는 **리뷰어가 명시 요청할 때만** 도입(보존 스펙: `.claude/.dev_note/aroma_exp4v2_perclass-parity-cap.md`).

---

## STEP 2 — 3종 실행 (severstal · mvtec_leather · mtd)

**그룹 A 파라미터** (`_SPEC §4`): `--class_mode multi` · `--imgsz 640` · `--rect` · `--baseline_epochs 100` · `--patience 50` · `--seeds 42 1 2`. 공통: `--condition all` · `--val_frac 0.3` · `--synth_ratio 1.0` · `--batch 64` · `--cache ram` · `--resume`.

세 조건 모두 COCO pretrained에서 fresh 학습(graft 미사용). 학습량 = 3 ds × 3 seed × 3 cond. `--resume`가 완료된 `(seed, ds, cond)`를 skip하고 중단 지점부터 재개한다.

```python
!python $AROMA_SCRIPTS/experiments/exp4_v2_supervised_detection.py \
    --model yolov8n \
    --condition all \
    --dataset_keys severstal mvtec_leather mtd \
    --class_mode multi \
    --aroma_synthetic_dir  $SYNTH_AROMA \
    --random_synthetic_dir $SYNTH_RANDOM \
    --real_data_dir        $AROMA_DATA \
    --output_dir           $EXP4V2_OUT \
    --yolo_cache_dir       $YOLO_CACHE \
    --imgsz 640 \
    --val_frac 0.3 \
    --synth_ratio 1.0 \
    --baseline_epochs 100 \
    --patience 50 \
    --batch 64 \
    --cache ram \
    --rect \
    --seeds 42 1 2 \
    --resume
```

> **소요 시간**: A100 기준 seed당 수 시간, 3 seed 전체 수 시간~하루. 세션 한도를 고려해 `--resume`을 항상 함께 쓴다(seed 단위·`(ds, cond)` 단위 모두 중단 지점부터 재개).
> **정상 신호**: `aroma train: N/… synth imgs labeled`의 `N>0`, `_get_real: … labels > 0`, `multi mode … -> class 0` 경고 **없음**(class_key 정상). leather처럼 test 폴더가 실질 단일이면 자동 single 축퇴(정상).

---

## STEP 3 — aitex 실행 (tiled, single-class)

**그룹 B 파라미터** (`_SPEC §4`): `--imgsz 256` · `--baseline_epochs 300` · `--patience 50` · `--seeds 1 2 42`. **`--class_mode` 미지정**(=single, nc=1 'defect') · **`--rect` 미사용**(타일이 정사각 256). 공통 인자는 그룹 A와 동일.

**동일 `--output_dir`**(`$EXP4V2_OUT`): `--resume`가 이미 완료된 3종을 건드리지 않고 aitex만 추가 학습·집계한다. 데이터셋 키가 다르므로 seed 폴더가 겹쳐도 충돌하지 않는다.

```python
!python $AROMA_SCRIPTS/experiments/exp4_v2_supervised_detection.py \
    --model yolov8n \
    --condition all \
    --dataset_keys aitex \
    --aroma_synthetic_dir  $SYNTH_AROMA \
    --random_synthetic_dir $SYNTH_RANDOM \
    --real_data_dir        $AROMA_DATA \
    --output_dir           $EXP4V2_OUT \
    --yolo_cache_dir       $YOLO_CACHE \
    --imgsz 256 \
    --val_frac 0.3 \
    --synth_ratio 1.0 \
    --baseline_epochs 300 \
    --patience 50 \
    --batch 64 \
    --cache ram \
    --seeds 1 2 42 \
    --resume
```

> **aitex 절대 금지 사항**: `--rect`·`--class_mode multi`를 붙이지 않는다(정본 regime = tiled single-class). imgsz 640·epochs 100으로 실행하면 비타일 폐기 regime이 되어 무효.
> aitex는 **tile-level mAP**(256 타일 단위, 50% overlap 중복 계수) — 절대값을 타 데이터셋과 직접 비교하지 않는다(상대 Δ만 유효).

---

## STEP 4 — 판정 (CPU)

데이터셋별 baseline/random/aroma의 map50(±std) + Δ(aroma−random) + per-seed 부호 일치를 본다.

```python
import json, os

with open(f"{os.environ['EXP4V2_OUT']}/exp4v2_results.json") as f:
    results = json.load(f)

CONDS = ["baseline", "random", "aroma"]
ORDER = ["severstal", "mvtec_leather", "mtd", "aitex"]

def fmt(m, sd):
    if not isinstance(m, float): return "N/A"
    return f"{m:.4f} ± {sd:.4f}" if isinstance(sd, float) else f"{m:.4f}"

print(f"{'dataset':<16} {'baseline':>16} {'random':>16} {'aroma':>16} {'Δ(A-R)':>9}")
print("-" * 78)
for ds in [d for d in ORDER if d in results] + [d for d in results if d not in ORDER]:
    arms = results[ds].get("yolov8n", results[ds])
    cells = {c: arms.get(c, {}) for c in CONDS}
    m  = {c: cells[c].get("map50") for c in CONDS}
    sd = {c: (cells[c].get("std") or {}).get("map50") for c in CONDS}
    d  = (m["aroma"] - m["random"]) if isinstance(m["aroma"], float) and isinstance(m["random"], float) else None
    tl = " (tile-level)" if ds == "aitex" else ""
    print(f"{ds:<16} {fmt(m['baseline'],sd['baseline']):>16} {fmt(m['random'],sd['random']):>16} "
          f"{fmt(m['aroma'],sd['aroma']):>16} {(f'{d:+.4f}' if d is not None else 'N/A'):>9}{tl}")

# per-seed paired Δ (부호 일치 확인) — aroma vs random
print("\nper-seed Δ(aroma − random):")
for ds in [d for d in ORDER if d in results] + [d for d in results if d not in ORDER]:
    arms = results[ds].get("yolov8n", results[ds])
    a  = arms.get("aroma",  {}).get("per_seed", {})
    rd = arms.get("random", {}).get("per_seed", {})
    seeds = sorted(set(a) & set(rd))
    d = [a[s]["map50"] - rd[s]["map50"] for s in seeds]
    if not d:
        print(f"  {ds:<16} seed 교집합 없음"); continue
    tag = "✅ 전seed aroma 우위" if all(x > 0 for x in d) else ("❌ 전seed aroma 열위" if all(x < 0 for x in d) else "⚠️ 부호 비일관")
    print(f"  {ds:<16} {[round(x,4) for x in d]}  mean={sum(d)/len(d):+.4f}  {tag}")
```

### 판정 규칙 (사전 등록)

| 관측 | 해석 |
|---|---|
| aroma > random 전 seed **and** > baseline | 개선이 downstream 이득 (H1 벽 넘음). 단 near-ceiling 데이터셋은 천장 고려 |
| aroma ≈ random ≈ baseline (**flat**) | near-ceiling(예: mtd baseline~0.90)이면 placement 이득 **측정 불가** — 개선 무효 결론 **금지**. 기제(fallback률·positive 로그)만 확인하고 headroom 있는 데이터셋을 arbiter로 삼는다 |
| aroma < random/baseline | **회귀** — 개선이 해로움(조합다양성 과축소? τ 과대?) → step5 τ 사전스캔·step4 fallback률 재점검 |

> **near-ceiling 주의**: mtd는 baseline mAP50이 높아 flat이 흔하다 — mtd 단독으로 개선 성패를 headline 삼지 말 것. headroom 있는 데이터셋이 arbiter.
> **aitex는 tile-level·single-class**: 절대값을 3종(multi)과 직접 비교하지 않는다. Δ(aroma−random)만 유효.
> **정밀 검정**: `per_seed`의 seed별 값으로 paired Wilcoxon/t-test를 별도 수행(rough 판정은 위 부호 일치).

---

## 무결성 / 정직 (`_SPEC §5`)

- **사후 튜닝 금지**: τ·seed·synth_ratio·epochs·imgsz는 위 그룹별 확정값을 그대로 쓰고, 결과 보고 후 변경하지 않는다. 파라미터를 바꾸면 skip 조건에 걸려 재학습되지 않으니 fresh `--output_dir` 또는 해당 항목 삭제로만 재실행한다.
- **그룹 파라미터 불변**: 3종 = multi·640·rect·100ep·seeds 42 1 2 / aitex = single·256·no-rect·300ep·seeds 1 2 42. 두 그룹을 섞으면 비교 불가.
- **fresh 전조건**: baseline/random/aroma 모두 COCO에서 독립 학습. graft(전학습 weight 재사용) 미사용. 합성은 train에만, test/defect는 항상 real.
- **aitex는 tile-level·single-class** → 절대값 타 데이터셋 직접 비교 금지, Δ만 유효.
- **테스트 코드 신규 작성·pytest 금지**(CLAUDE.md). 검증은 Colab 실행으로.
- **시간 실측·처리량 벤치 미수행** (load-test policy).
```
