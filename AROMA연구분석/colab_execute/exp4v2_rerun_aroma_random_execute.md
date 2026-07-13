# exp4v2 재실행 — aroma·random arm만 (baseline 보존) 실행 가이드

> 경로·환경설정 정본 = `colab_execute_new/_SPEC.md`. STEP 0 환경 셀은 `_SPEC §1` 그대로, 명령은 `_SPEC §3 exp4v2` 파라미터를 그대로 쓴다. **env·output 루트 규약 재발명 금지.**

**목적**: 다른 세션에서 개선한 `synth_aroma`/`synth_random`(step5 산출)으로 downstream 검출기(YOLOv8n) 성능을 다시 측정한다. 단, **기존 `baseline`(real-only) 학습 결과는 재학습하지 않고 그대로 보존**하고 `random`·`aroma` 두 arm만 fresh 재학습한다. baseline은 합성과 무관(real만 사용)하므로 재실행이 불필요 — 시간·GPU 절약.

**핵심 메커니즘 (스크립트 실측)**:
- 최종 `exp4v2_results.json`/`exp4v2_summary.md`는 **per-seed 파일**(`_seeds/seed{N}/exp4v2_results.json`)을 seed 집계해 **매 실행 재생성**된다.
- `--resume`는 per-seed JSON에 `map50`이 있는 `(ds, model, cond)`를 **skip**한다.
- `--condition random aroma`로 실행하면 baseline은 학습 루프에 **아예 진입하지 않고**, `results`가 기존 per-seed(baseline 포함)의 복사로 초기화되어 **baseline 항목이 집계까지 그대로 전달**된다 (`exp4_v2_supervised_detection.py:2280, 2510-2534`).
- 따라서 재학습을 강제하려면 per-seed JSON에서 **aroma·random 키만 제거**하면 된다 (baseline 키는 남긴다). 그러면 resume이 baseline은 skip 유지, aroma/random은 캐시 없음 → 재학습.

**전제**: 개선된 step5 산출이 `S('synth_aroma', ds)`·`S('synth_random', ds)`에 이미 갱신 완료되어 있어야 한다(같은 경로에 덮어썼다면 exp4v2가 자동으로 새 합성을 로드).

**실행 순서 체인**: `… → step5(개선 재생성) → **exp4v2 재실행(이 문서)**`.

**환경**: YOLO 학습 GPU 필수(A100 권장). strip·집계는 CPU.

---

## 워크플로우 개요

| STEP | 내용 | 런타임 |
|------|------|--------|
| 0 | 공통 환경 셀(`_SPEC §1`) + exp4v2 경로 + 전제 확인 | CPU |
| 1 | **per-seed JSON에서 aroma·random 제거** (baseline 보존) + 백업 + 구 weight 정리 | CPU |
| 2 | 패키지 설치 & GPU 확인 | CPU |
| 3 | 3종(severstal·mvtec_leather·mtd) 재실행 — `--condition random aroma` | **GPU** |
| 4 | aitex 재실행 — `--condition random aroma` (해당 시) | **GPU** |
| 5 | 판정 (baseline 보존 확인 + 신 aroma/random Δ) | CPU |

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
os.environ['CN_MODELS'] = f"{os.environ['SYM_ROOT']}/controlnet_models"
def S(stage, ds=None):
    p = f"{os.environ['SYM_ROOT']}/{stage}"
    return f"{p}/{ds}" if ds else p

DATASETS = ["severstal", "mvtec_leather", "mtd", "aitex"]   # v2-1 4종
with open(os.environ['DATASET_CONFIG']) as f: CFG = json.load(f)
def normal_dir(ds): return CFG[ds]["image_dir"]
def is_multi(ds):   return CFG[ds].get("class_mode") == "multi"
```

### exp4v2 전용 경로

```python
os.environ['SYNTH_AROMA']  = S('synth_aroma')     # sym_final/synth_aroma
os.environ['SYNTH_RANDOM'] = S('synth_random')    # sym_final/synth_random
os.environ['EXP4V2_OUT']   = S('exp4v2')          # sym_final/exp4v2
os.environ['YOLO_CACHE']   = f"{os.environ['AROMA_OUT']}/yolo_cache"

for k in ('SYNTH_AROMA', 'SYNTH_RANDOM', 'EXP4V2_OUT', 'YOLO_CACHE', 'AROMA_DATA'):
    print(f"{k:<14} {os.environ[k]}")
```

### 전제 확인 (개선된 step5 산출 — 재실행할 데이터셋이 모두 ✓여야 진행)

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
        row += [n_ann, n_img]
    a_ann, a_img, r_ann, r_img = row
    print(f"{ds:<16} {a_ann:>10} {a_img:>10} {r_ann:>11} {r_img:>11}")
```

> 재실행 대상 데이터셋은 `aroma_ann > 0` **및** `random_ann > 0`이어야 한다. 개선 대상이 아닌 데이터셋(예: aitex 미포함)은 STEP 1에서 자동으로 건너뛴다(baseline만 남아있으면 손 안 댐).

---

## STEP 1 — per-seed JSON에서 aroma·random 제거 (baseline 보존) ★핵심

`_seeds/seed{N}/exp4v2_results.json`에서 **`random`·`aroma` 키만 삭제**하고 `baseline`은 남긴다. 삭제된 arm은 resume 캐시가 없어져 재학습되고, baseline은 캐시가 남아 skip → 보존.

**먼저 원본 백업** (사후 튜닝·롤백 대비). 그다음 strip. 삭제 전 반드시 dry-run으로 무엇을 지울지 확인한다.

```python
import json, shutil, pathlib, os

EXP4V2_OUT = os.environ['EXP4V2_OUT']
SEEDS = [42, 1, 2]                       # 3종·aitex 공통 seed 집합 (순서 무관)
STRIP = {"random", "aroma"}              # 재학습할 arm — 이 키만 제거
KEEP  = {"baseline", "casda"}            # 보존 (참고용; 실제로는 STRIP만 지운다)

# --- 1a. dry-run: 무엇을 지우고 무엇을 남기는지 표시 ---
seed_files = []
for s in SEEDS:
    p = pathlib.Path(EXP4V2_OUT) / "_seeds" / f"seed{s}" / "exp4v2_results.json"
    if p.exists():
        seed_files.append(p)

print("=== DRY-RUN (아직 변경 안 함) ===")
if not seed_files:
    print("⚠️  per-seed 파일 없음 — EXP4V2_OUT 경로/기존 실행 여부 확인")
for p in seed_files:
    data = json.load(open(p))
    print(f"\n{p}")
    for ds, models in data.items():
        if not isinstance(models, dict):
            continue
        for m, arms in models.items():
            if not isinstance(arms, dict):
                continue
            present = [c for c in arms if isinstance(arms.get(c), dict)]
            to_strip = [c for c in present if c in STRIP]
            to_keep  = [c for c in present if c not in STRIP]
            print(f"  {ds:<14}/{m:<8}  삭제→ {to_strip or '없음':<24}  보존→ {to_keep}")
```

> 위 출력에서 **삭제→ random, aroma** / **보존→ baseline** 인지 데이터셋마다 확인한다. baseline이 보존 목록에 없으면 그 데이터셋은 baseline이 없는 것이니(신규) 별도 판단 필요.

```python
# --- 1b. 실제 백업 + strip (dry-run 확인 후 실행) ---
BACKUP = pathlib.Path(EXP4V2_OUT) / "_seeds_backup_pre_synth_rerun"
BACKUP.mkdir(parents=True, exist_ok=True)

for p in seed_files:
    # 백업 (seed 구조 유지)
    rel = p.relative_to(pathlib.Path(EXP4V2_OUT) / "_seeds")
    bdst = BACKUP / rel
    bdst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(p, bdst)

    data = json.load(open(p))
    removed = 0
    for ds, models in data.items():
        if not isinstance(models, dict):
            continue
        for m, arms in models.items():
            if not isinstance(arms, dict):
                continue
            for c in list(arms):
                if c in STRIP:
                    del arms[c]; removed += 1
    json.dump(data, open(p, "w"), indent=2, ensure_ascii=False)
    print(f"stripped {removed:>2} arm(s) → {p}   (backup: {bdst})")

print(f"\n✅ 백업 위치: {BACKUP}")
print("   롤백 필요 시 이 폴더의 파일을 _seeds/ 로 되돌린다.")
```

**(선택) 구 aroma/random weight 정리** — fresh 재학습이라 덮어써지지만, 혼동 방지용으로 삭제 권장. baseline weight는 절대 건드리지 않는다.

```python
import pathlib, shutil, os
EXP4V2_OUT = os.environ['EXP4V2_OUT']
for s in [42, 1, 2]:
    for ds in DATASETS:
        for arm in ("random", "aroma"):          # baseline 제외 — 보존
            d = pathlib.Path(EXP4V2_OUT) / "_seeds" / f"seed{s}" / ds / "yolov8n" / arm
            if d.exists():
                shutil.rmtree(d, ignore_errors=True)
                print("removed", d)
```

> ⚠️ 이 셀은 `arm in ("random","aroma")`만 지운다. `baseline` 디렉터리·`best.pt`는 목록에 없으므로 안전하다.

---

## STEP 2 — 패키지 설치 & GPU 확인

```python
!pip install ultralytics -q
!pip install opencv-python-headless -q
```

```python
!nvidia-smi
```

> 설치 후 런타임 재시작이 필요하면 STEP 0 환경 셀을 다시 실행한 뒤 STEP 3으로(STEP 1 strip은 이미 완료됐으니 재실행 금지).

---

## STEP 3 — 3종 재실행 (severstal · mvtec_leather · mtd)

**그룹 A 파라미터** (`_SPEC §4`, 기존 실행과 **동일값 고정**): `--imgsz 640` · `--rect` · `--class_mode multi` · `--baseline_epochs 100` · `--patience 50` · `--seeds 42 1 2` · `--val_frac 0.3` · `--synth_ratio 1.0` · `--batch 64` · `--cache ram`.

**차이는 `--condition random aroma` 하나뿐** — baseline은 학습하지 않고 per-seed에 남은 값이 그대로 집계된다. `--resume`을 반드시 함께 써서 baseline skip·중단재개를 활성화한다.

```python
!python $AROMA_SCRIPTS/experiments/exp4_v2_supervised_detection.py \
    --model yolov8n \
    --condition random aroma \
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

> **정상 신호**: 로그에 `RESUME skip … / baseline (cached map50=…)`가 데이터셋·seed마다 찍혀야 한다(baseline 보존 확인). random/aroma는 `[i/N] … / random`·`… / aroma`로 새로 학습된다.
> **학습량**: 3 ds × 3 seed × 2 arm = 18 run (baseline 9 run 절약). A100 기준 seed당 수 시간.
> **파라미터 불변**: STEP 1에서 지운 것은 random/aroma뿐이므로, imgsz·epochs 등을 바꾸면 baseline(imgsz 640/100ep로 학습됨)과 regime이 어긋나 비교 불가. 위 값 그대로 쓴다.

---

## STEP 4 — aitex 재실행 (해당 시)

aitex 합성도 개선했다면 실행. 기존 exp4v2에 aitex가 없었다면(= per-seed에 aitex baseline 없음) **건너뛴다**(신규 데이터셋은 이 "baseline 보존 재실행" 시나리오 밖 — 정본 `exp4v2_execute.md`로 3조건 전체 실행).

**그룹 B 파라미터** (`_SPEC §4`): `--imgsz 256` · `--baseline_epochs 300` · `--seeds 1 2 42` · **`--class_mode` 미지정**(single) · **`--rect` 미사용**. `--condition random aroma`.

```python
!python $AROMA_SCRIPTS/experiments/exp4_v2_supervised_detection.py \
    --model yolov8n \
    --condition random aroma \
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

> aitex 절대 금지: `--rect`·`--class_mode multi` 미부착(tiled single-class 정본 유지). tile-level mAP → 절대값 타 데이터셋 비교 금지, Δ만 유효.

---

## STEP 5 — 판정 (CPU)

baseline이 **이전 값 그대로 보존**됐는지 + 신 aroma/random Δ를 확인한다.

```python
import json, os

with open(f"{os.environ['EXP4V2_OUT']}/exp4v2_results.json") as f:
    results = json.load(f)

CONDS = ["baseline", "random", "aroma"]
ORDER = ["severstal", "mvtec_leather", "mtd", "aitex"]

def fmt(m, sd):
    if not isinstance(m, float): return "N/A"
    return f"{m:.4f} ± {sd:.4f}" if isinstance(sd, float) else f"{m:.4f}"

print(f"{'dataset':<16} {'baseline':>16} {'random':>16} {'aroma':>16} {'Δ(A-R)':>9} {'seeds':>6}")
print("-" * 86)
for ds in [d for d in ORDER if d in results] + [d for d in results if d not in ORDER]:
    arms = results[ds].get("yolov8n", results[ds])
    cells = {c: arms.get(c, {}) for c in CONDS}
    m  = {c: cells[c].get("map50") for c in CONDS}
    sd = {c: (cells[c].get("std") or {}).get("map50") for c in CONDS}
    d  = (m["aroma"] - m["random"]) if isinstance(m["aroma"], float) and isinstance(m["random"], float) else None
    ns = cells["aroma"].get("n_seeds", cells["baseline"].get("n_seeds"))
    tl = " (tile)" if ds == "aitex" else ""
    print(f"{ds:<16} {fmt(m['baseline'],sd['baseline']):>16} {fmt(m['random'],sd['random']):>16} "
          f"{fmt(m['aroma'],sd['aroma']):>16} {(f'{d:+.4f}' if d is not None else 'N/A'):>9} {str(ns):>6}{tl}")

# per-seed Δ(aroma − random) 부호 일치
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

### baseline 보존 검증 (재실행 전후 동일해야 정상)

```python
# 백업(strip 이전) per-seed의 baseline map50 vs 현재 top-level baseline map50 비교.
# 두 값이 seed별로 일치하면 baseline이 재학습 없이 보존된 것.
import json, pathlib, os
EXP4V2_OUT = os.environ['EXP4V2_OUT']
BACKUP = pathlib.Path(EXP4V2_OUT) / "_seeds_backup_pre_synth_rerun"
for s in [42, 1, 2]:
    bp = BACKUP / f"seed{s}" / "exp4v2_results.json"
    cp = pathlib.Path(EXP4V2_OUT) / "_seeds" / f"seed{s}" / "exp4v2_results.json"
    if not (bp.exists() and cp.exists()): continue
    b, c = json.load(open(bp)), json.load(open(cp))
    for ds in b:
        bb = b.get(ds, {}).get("yolov8n", {}).get("baseline", {}).get("map50")
        cc = c.get(ds, {}).get("yolov8n", {}).get("baseline", {}).get("map50")
        ok = (bb == cc)
        print(f"seed{s} {ds:<16} baseline map50  before={bb}  after={cc}  {'✅' if ok else '❌ 변경됨!'}")
```

> baseline `after != before`이면 STEP 1을 잘못 실행(baseline까지 strip)했거나 `--condition`에 baseline이 섞인 것 — 백업에서 복원 후 재실행.

### 판정 규칙 (`exp4v2_execute.md §STEP4`와 동일, 사전 등록)

| 관측 | 해석 |
|---|---|
| aroma > random 전 seed **and** > baseline | 개선이 downstream 이득. near-ceiling(mtd 등) 천장 고려 |
| aroma ≈ random ≈ baseline (flat) | near-ceiling이면 측정 불가 — 개선 무효 결론 금지. headroom 있는 데이터셋이 arbiter |
| aroma < random/baseline | 회귀 — 개선이 해로움 → step5 fallback률·τ 재점검 |

> **비교 대상**: 이번 재실행의 목적은 **신 aroma/random vs (보존된) baseline** 및 **신 aroma vs 신 random**. 개선 전후 절대비교를 하려면 백업(`_seeds_backup_pre_synth_rerun`)의 구 aroma/random 값을 별도로 읽어 대조한다.

---

## 무결성 / 정직 (`_SPEC §5`)

- **baseline 미변경**: 이 시나리오는 baseline을 재학습하지 않는다. baseline은 real만 사용 → 합성 개선과 독립. STEP 5의 보존 검증으로 before==after 확인 필수.
- **파라미터 불변**: random/aroma 재학습 시 imgsz·epochs·rect·class_mode·seed·val_frac·synth_ratio를 보존된 baseline과 **정확히 동일**하게 쓴다(그룹 A=multi·640·rect·100ep·42 1 2 / aitex=single·256·no-rect·300ep·1 2 42). 어긋나면 baseline과 비교 불가.
- **사후 튜닝 금지**: τ·seed·synth_ratio·epochs·imgsz는 확정값 고정, 결과 보고 후 변경 금지.
- **strip은 1회만**: STEP 1을 재실행할 때는 이미 aroma/random이 없으므로 무해하지만, 백업 폴더를 덮어쓰지 않도록 주의(최초 백업이 개선 전 원본이어야 롤백 가치).
- **fresh 전조건**: random/aroma는 COCO pretrained에서 처음부터 독립 학습(graft 미사용). 합성은 train에만, val/defect는 항상 real.
- **테스트 코드 신규 작성·pytest 금지**(CLAUDE.md). 검증은 Colab 실행으로.
- **시간 실측·처리량 벤치 미수행**(load-test policy).
