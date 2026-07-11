# exp4v2 downstream — aroma-symmetric arm 이식 실행 (severstal / mvtec_leather / mtd / aitex)

**목적**: 이미 생성한 **aroma-symmetric**(SGM + 64px 타일링 + positive placement) 합성물을 downstream detector(exp4v2)로 평가한다. `controlnet_aroma_arm_execute.md` **STEP 8**과 동일한 방식 — baseline/random 결과는 **기존 벤치에서 이식**하고 **aroma arm만 학습**해 GPU를 절약한다.

**전제(사용자 완료)**: `exp4v2_mtd_symmetric_execute.md` §1~§5.1을 실행해 aroma-symmetric 합성물이 `newpipe/{DS}/synth_aroma_sym/{DS}` (= `$AROMA_SYM_DIR/{DS}`)에 존재.
> severstal / mvtec_leather / mtd 완료. **aitex는 tiled 규약**(256×256/stride128, 단일클래스)으로 별도 생성돼 있어야 한다 — STEP 1에서 확인, 없으면 aitex 트랙 skip.

**이식 소스 (2 track — controlnet STEP 8 구조와 동일)**:

| 트랙 | 데이터셋 | 이식 소스 | class_mode | imgsz / rect / epochs | seeds |
|------|---------|----------|-----------|----------------------|-------|
| **A (3종)** | mvtec_leather · mtd · severstal | `exp4v2/20260705_article` | **multi** | 640 / rect / **100** | 42 1 2 |
| **B (aitex)** | aitex | `exp4v2/20260706_aitex` (tiled) | **single** | 256 / **no rect** / **300** | 1 2 42 |

> 트랙 A 검증 완료 — 20260705_article은 **multi-class**(`per_class`: mtd nc=5, severstal nc=4, leather nc=5). baseline/random 이식 + **aroma-sym `--class_mode multi`** 학습.
> ⚠️ **aitex는 20260705_article에서 이식 금지**. 20260705_article의 aitex는 **비타일(non-tiled) multi**(baseline mAP50~0.06, 폐기된 regime)다. aitex의 정본 regime은 **tiled 단일클래스**(`20260706_aitex`, baseline 0.3719) — controlnet STEP 8-C와 동일하게 여기서만 이식한다.
> ⚠️ `controlnet_aroma_arm_execute.md` STEP 8-B는 트랙 A에 `--class_mode`를 생략(single)했는데, 20260705_article이 multi인 이상 이는 그 문서의 불일치다. **본 가이드는 트랙 A를 multi로 정정**한다.

**설계(단일변수)**: 동일 real 데이터 · 동일 split(seed·val_frac·class_mode 동일 → train/val 동일) · 동일 학습 파라미터. **바뀌는 것은 aroma arm의 합성물뿐**(기존 aroma = copy-paste `aroma-CP`, 본 실행 aroma = `aroma-sym`). baseline/random은 그대로 이식 → aroma-sym vs random(통제) / vs baseline / **vs aroma-CP(개선 순효과)** 삼각 비교.

**런타임**: 이식·판정 = CPU | aroma-sym 학습 = **GPU**(A100 권장). 학습량 = 4 ds × 3 seed = **12 run**(baseline/random은 skip; aitex 미생성 시 9 run).

> ⚠️ **정직 경고**: (1) mtd는 near-ceiling(baseline mAP50~0.90)이라 placement 이득이 null일 수 있음 — 개선 실패 아님(천장). headroom 있는 데이터셋이 arbiter. (2) aroma-sym 결과는 **각 트랙의 이식 소스 규약에만 정합** — aroma-CN(controlnet arm)과 직접 비교 금지. 비교는 본 실행 내부(aroma-sym vs 이식된 random/baseline) + 참조 aroma-CP로 한정. (3) **aitex는 tile-level mAP**(256 타일 단위, 50% overlap 중복 계수) — 절대값을 타 데이터셋과 직접 비교 금지, 상대 비교(3조건 동일 적용)만 유효.

---

## 워크플로우 개요

| STEP | 내용 | 런타임 |
|------|------|--------|
| 0 | 환경변수 (2 track: 3종 + aitex) | CPU |
| 1 | 전제 확인 (aroma-sym 산출 + 이식 소스 per-seed + real data) | CPU |
| 2 | baseline/random 이식 (트랙별 소스, aroma 제거) | CPU |
| 3 | exp4v2 실행 — 트랙 A(3종) + 트랙 B(aitex), aroma-sym만 학습 | **GPU** |
| 4 | 판정 (aroma-sym vs random / baseline / aroma-CP) | CPU |

---

## STEP 0 — 환경변수

```python
import os, json

os.environ['DRIVE'] = '/content/drive/MyDrive/data/Aroma'
os.environ['AROMA_SCRIPTS'] = '/content/AROMA/scripts/aroma'
os.environ['AROMA_REF']     = '/content/AROMA'
os.environ['AROMA_OUT']     = f"{os.environ['DRIVE']}/aroma_output"
os.environ['AROMA_DATA']    = f"{os.environ['DRIVE']}"
os.environ['DATASET_CONFIG']= os.environ.get('DATASET_CONFIG', '/content/AROMA/dataset_config.json')

# 이식 소스 (트랙 A = 20260705_article multi / 트랙 B = 20260706_aitex tiled single)
os.environ['EXP4V2_REF']       = f"{os.environ['AROMA_OUT']}/exp4v2/20260705_article"
os.environ['EXP4V2_AITEX_REF'] = f"{os.environ['AROMA_OUT']}/exp4v2/20260706_aitex"
# random arm(통제) = 각 소스와 동일 random을 이식하므로 재생성 불필요. 아래 경로는 skip되는 조건의 로드용.
os.environ['RANDOM_REF_DIR'] = f"{os.environ['AROMA_OUT']}/synthetic_random"

# 트랙별 대상
GRAFT_DATASETS = ["mvtec_leather", "mtd", "severstal"]   # 트랙 A (20260705_article, multi)
AITEX          = ["aitex"]                                # 트랙 B (20260706_aitex, tiled single)
DATASETS       = GRAFT_DATASETS + AITEX

# 데이터셋 → 이식 소스 ref 디렉토리
def ref_for(ds):
    return os.environ['EXP4V2_AITEX_REF'] if ds == "aitex" else os.environ['EXP4V2_REF']

with open(os.environ['DATASET_CONFIG']) as f:
    CFG = json.load(f)

# aroma-symmetric 합성물 루트 (symmetric §1과 동일 규약). exp4v2는 이 루트에 /{ds}를 붙여 조회.
def aroma_sym_dir(ds):
    return f"{os.environ['AROMA_OUT']}/newpipe/{ds}/synth_aroma_sym"

# 본 실행 출력 — per-DS (symmetric §7 규약). baseline/random 이식 + aroma-sym 학습.
def exp_out_dir(ds):
    return f"{os.environ['AROMA_OUT']}/newpipe/{ds}/exp4v2_sym"

for ds in DATASETS:
    print(f"{ds:14s} src={ref_for(ds).split('/')[-1]:16s} aroma_sym={aroma_sym_dir(ds)}/{ds}")
for ds in DATASETS:
    print(ds, "→ EXP_SYM:", exp_out_dir(ds))
```

> **출력 레이아웃 변경**: 단일 공통 출력에서 **per-DS 출력**(`newpipe/{ds}/exp4v2_sym`)으로 전환 — aroma-sym 합성물이 per-DS `newpipe/{ds}/…`에 흩어져 있어 symmetric §7과 동일 규약이 자연스럽다. STEP 2 이식과 STEP 3 실행 모두 데이터셋별 디렉토리에 쓴다.
> `AROMA_DATA = $DRIVE`(= `/content/drive/MyDrive/data/Aroma`) — `real_data_dir`로 사용. 이식 소스: 3종=`20260705_article`, aitex=`20260706_aitex`(`ref_for(ds)`가 분기).

---

## STEP 1 — 전제 확인 (CPU)

```python
import pathlib, json

# (a) aroma-symmetric 합성물 (4종; aitex 미생성이면 자동 skip)
avail = {}
for ds in DATASETS:
    p = f"{aroma_sym_dir(ds)}/{ds}/annotations.json"
    ok = pathlib.Path(p).exists()
    avail[ds] = ok
    n = sum(1 for a in json.load(open(p)) if not a.get("dry_run") and a.get("bbox")) if ok else 0
    print(f"{'✓' if ok else '✗'} {ds:14s} aroma-sym labelable={n}  {p}")

# (b) 이식 소스 per-seed (트랙 A=20260705_article multi / 트랙 B=20260706_aitex tiled single)
for ds in DATASETS:
    if not avail[ds]:
        continue
    for s in (42, 1, 2):
        p = f"{ref_for(ds)}/_seeds/seed{s}/exp4v2_results.json"
        ok = pathlib.Path(p).exists()
        conds = []
        if ok:
            cell = json.load(open(p)).get(ds, {}).get("yolov8n", {})
            conds = sorted(c for c in cell if cell[c].get("map50") is not None)
        print(f"{'✓' if ok else '✗'} {ds:14s} seed{s}: {conds}  {p}")

# (c) real data
print(f"\n{'✓' if pathlib.Path(os.environ['AROMA_DATA']).exists() else '✗'} real_data_dir {os.environ['AROMA_DATA']}")

# 실행 대상 (aroma-sym 있는 것만) — 트랙별로 STEP 2/3에서 사용
RUN_A = [ds for ds in GRAFT_DATASETS if avail[ds]]
RUN_B = [ds for ds in AITEX if avail[ds]]
print("트랙 A(3종):", RUN_A, "| 트랙 B(aitex):", RUN_B)
```

> 각 seed 파일에 `['aroma','baseline','random']`이 모두 떠야 이식이 유효. 트랙 A는 multi(mtd 5 / severstal 4 / leather 5 per_class), 트랙 B(aitex)는 **single**(class 1개).
> **aitex ✗이면**: aroma-sym이 tiled로 아직 생성되지 않은 것 — aitex는 트랙 B에서 자동 제외되고 3종만 진행된다. aitex를 포함하려면 **`aitex_aroma_sym_generate_execute.md`**를 먼저 실행해 `newpipe/aitex/synth_aroma_sym/aitex`를 만든다(tiled single-class + symmetric 게이트, 선결 3종: aitex profiling·CN 모델·τ 사전스캔).
> aroma-sym `annotations.json`의 labelable 수가 이후 `synth_ratio=1.0` cap을 채울 만큼(random arm n_synth_train 이상) 되는지 확인 — 부족하면 symmetric §5.1을 `--n_per_roi` 상향으로 재생성.

---

## STEP 2 — baseline/random 이식 (CPU)

각 트랙의 이식 소스(`ref_for(ds)`)에서 **baseline/random만** 새 출력의 per-seed JSON으로 복사한다. `aroma` 셀은 제거 — 이 run의 `aroma` 키는 aroma-sym이 채운다(기존 aroma = copy-paste, STEP 4 참조 비교용으로만 사용).

```python
import pathlib, json

for ds in (RUN_A + RUN_B):
    for s in (42, 1, 2):
        src = f"{ref_for(ds)}/_seeds/seed{s}/exp4v2_results.json"
        models = json.load(open(src)).get(ds)
        if not models:
            print(f"⚠️ {ds} seed{s}: 이식 소스에 없음 — 확인"); continue
        # aroma(copy-paste) 제거, baseline/random만 이식
        out = {ds: {m: {c: v for c, v in conds.items() if c != "aroma"}
                    for m, conds in models.items()}}
        dst_dir = pathlib.Path(f"{exp_out_dir(ds)}/_seeds/seed{s}")
        dst_dir.mkdir(parents=True, exist_ok=True)
        json.dump(out, open(dst_dir / "exp4v2_results.json", "w"), indent=2)
        print(f"{ds:14s} seed{s}: 이식 → {sorted(out[ds].get('yolov8n', {}))}")
```

> 각 `{ds}/exp4v2_sym/_seeds/seed{s}/`에 `['baseline','random']`만 담기면 정상. aroma 키가 비어 있어야 STEP 3에서 aroma-sym이 새로 학습된다.

---

## STEP 3 — exp4v2 실행 (GPU) — aroma-sym만 학습

aroma-symmetric 합성물이 per-DS `newpipe/{ds}/synth_aroma_sym` 루트에 흩어져 있으므로 **데이터셋별로 1회씩** 실행한다(`--dataset_keys $DS`). `--output_dir`는 데이터셋별(`exp_out_dir(DS)`, STEP 2에서 이식한 그 디렉토리), `--resume`가 이식된 baseline/random을 skip하고 **aroma만 학습**한다.

⚠️ **파라미터는 각 트랙의 이식 소스와 완전 동일해야 이식이 유효**하다(하나라도 다르면 비교 불가):

| 트랙 | class_mode | imgsz | rect | epochs / patience | seeds |
|------|-----------|-------|------|-------------------|-------|
| **A (3종)** | `multi` | 640 | `--rect` | 100 / 50 | 42 1 2 |
| **B (aitex)** | (미지정=single) | 256 | (없음) | 300 / 50 | 1 2 42 |

> **트랙 A epochs=100 주의**: symmetric 원본 §7 표기는 300이나, 20260705 실측 results.csv가 **100 epochs**(controlnet 조사, 2026-07-07)로 확인됨. 이식이므로 **100 고정**(cosine LR schedule이 baseline_epochs에 종속).
> **트랙 B(aitex)**: `20260706_aitex`(tiled 재실행) 규약 — **imgsz 256, `--rect` 미사용**(타일이 정사각 256), epochs **300**/patience 50, **single-class**(`--class_mode` 미지정), seeds 1 2 42. `aitex_tiled_rerun_execute.md` §7-B와 정합.

### 3-A. 트랙 A — 3종 (multi, 640/rect/100)

```python
for DS in RUN_A:
    os.environ['DS'] = DS
    os.environ['AROMA_SYM'] = aroma_sym_dir(DS)   # exp4v2가 /{DS} 붙여 조회
    os.environ['EXP_OUT']   = exp_out_dir(DS)      # STEP 2에서 이식한 per-DS 출력
    print(f"\n========== exp4v2 {DS} (aroma-sym only, track A) ==========")
    !python $AROMA_SCRIPTS/experiments/exp4_v2_supervised_detection.py \
        --model yolov8n \
        --condition all \
        --dataset_keys $DS \
        --class_mode multi \
        --random_synthetic_dir $RANDOM_REF_DIR \
        --aroma_synthetic_dir  $AROMA_SYM \
        --real_data_dir        $AROMA_DATA \
        --output_dir           $EXP_OUT \
        --yolo_cache_dir       $AROMA_OUT/yolo_cache \
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

### 3-B. 트랙 B — aitex (single, 256/no-rect/300). `RUN_B` 비어 있으면 자동 skip

```python
for DS in RUN_B:   # aitex aroma-sym 미생성이면 빈 리스트 → 실행 안 됨
    os.environ['DS'] = DS
    os.environ['AROMA_SYM'] = aroma_sym_dir(DS)
    os.environ['EXP_OUT']   = exp_out_dir(DS)
    print(f"\n========== exp4v2 {DS} (aroma-sym only, track B / tiled) ==========")
    !python $AROMA_SCRIPTS/experiments/exp4_v2_supervised_detection.py \
        --model yolov8n \
        --condition all \
        --dataset_keys $DS \
        --random_synthetic_dir $RANDOM_REF_DIR \
        --aroma_synthetic_dir  $AROMA_SYM \
        --real_data_dir        $AROMA_DATA \
        --output_dir           $EXP_OUT \
        --yolo_cache_dir       $AROMA_OUT/yolo_cache \
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

> 로그에서 각 seed·ds마다 `RESUME skip … baseline (cached map50=…)` / `RESUME skip … random` 2줄이 뜨고, `aroma`만 `[n/N] … synth_ann=…`로 실제 학습되어야 한다(= 이식 정상).
> `--random_synthetic_dir`는 random이 skip되므로 학습에 쓰이지 않는다(로드만; 없으면 WARN 후 무해). random 통제군은 각 소스의 이식 셀을 그대로 사용 → **random arm 재생성(symmetric §6) 불필요**.
> 트랙 A: `multi mode … -> class 0` 경고가 없어야(class_key 정상). leather처럼 test 폴더가 실질 단일이면 자동 single 축퇴(정상). 트랙 B(aitex): `--class_mode` 미지정이므로 single(nc=1 'defect').
> 세션 끊김 → 동일 셀 재실행(`--resume`). 이미 학습된 seed의 aroma도 캐시된 map50으로 skip된다.

---

## STEP 4 — 판정 (CPU)

```python
import json, os

# aroma-CP(copy-paste) 참조 — 트랙별 소스에서 로드 (3종=20260705_article, aitex=20260706_aitex)
ref_cache = {}
def cp_ref(ds):
    r = ref_for(ds)
    if r not in ref_cache:
        ref_cache[r] = json.load(open(f"{r}/exp4v2_results.json"))
    return ref_cache[r]

print(f"{'dataset':<14} {'baseline':>9} {'random':>9} {'aroma-CP':>9} {'aroma-sym':>10} {'Δ(sym-R)':>9} {'Δ(sym-CP)':>10}")
print("-" * 80)
sym_by_ds = {}
for ds in (RUN_A + RUN_B):
    sym = json.load(open(f"{exp_out_dir(ds)}/exp4v2_results.json"))   # per-DS 출력
    sym_by_ds[ds] = sym
    b  = sym[ds]["yolov8n"]["baseline"]["map50"]
    r  = sym[ds]["yolov8n"]["random"]["map50"]
    a2 = sym[ds]["yolov8n"]["aroma"]["map50"]              # aroma-symmetric (본 실행)
    a1 = cp_ref(ds)[ds]["yolov8n"]["aroma"]["map50"]       # aroma-CP (copy-paste, 참조)
    tl = " (tile-level)" if ds == "aitex" else ""
    print(f"{ds:<14} {b:>9.4f} {r:>9.4f} {a1:>9.4f} {a2:>10.4f} {a2-r:>+9.4f} {a2-a1:>+10.4f}{tl}")

# per-seed paired delta (부호 일치 확인) — aroma-sym vs random
print("\nper-seed Δ(aroma-sym − random):")
for ds in (RUN_A + RUN_B):
    sym = sym_by_ds[ds]
    a = sym[ds]["yolov8n"]["aroma"].get("per_seed", {})
    rd = sym[ds]["yolov8n"]["random"].get("per_seed", {})
    seeds = sorted(set(a) & set(rd))
    d = [a[s]["map50"] - rd[s]["map50"] for s in seeds]
    tag = "✅ 전seed 우위" if d and all(x > 0 for x in d) else "❌ 비일관/무이득"
    print(f"  {ds:<14} {[round(x,4) for x in d]}  mean={sum(d)/len(d):+.4f}  {tag}" if d else f"  {ds}: seed 교집합 없음")
```

> **aitex 판독**: 20260706_aitex 실측(tile-level) baseline 0.3719 / random 0.3881 / **aroma-CP 0.4847** — 4종 중 유일하게 copy-paste aroma가 random을 +9.7pp 이긴 케이스. 관전점은 Δ(sym−R)뿐 아니라 **Δ(sym−CP)**: aroma-sym이 이미 강한 aroma-CP(0.4847)를 넘는지가 placement 개선의 직접 증거. tile-level 절대값은 타 데이터셋과 직접 비교 금지.

### 판정 규칙 (사전 등록)

| 관측 | 해석 |
|---|---|
| aroma-sym > random 전 seed **and** > baseline | 개선이 downstream 이득 (H1 벽 넘음). 단 mtd 천장 고려 |
| aroma-sym ≈ random ≈ baseline (flat) | near-ceiling(mtd)이면 측정 불가 — 개선 무효 결론 **금지**. 기제(fallback률·positive 로그)만 확인 |
| aroma-sym < random/baseline | 회귀 — 개선이 해로움(조합다양성 과축소? τ 과대?) → symmetric §5.1 로그 재점검 |
| **aroma-sym vs aroma-CP** | **개선 순효과의 직접 증거**. sym > CP면 placement/게이트 개선이 copy-paste 대비 실이득. sym ≈ CP면 개선이 downstream 무차이(천장 or H1 상류 문제) |
| aitex Δ(sym−CP) > 0 | aitex는 aroma-CP가 이미 random을 이긴 유일 케이스(0.4847) → sym이 CP를 넘으면 placement 개선의 강한 증거. CP−R은 음수라도 sym−R>0이면 "생성/배치 arm도 random은 이긴다"로 분리 해석 |

---

## 무결성 / 정직

- **이식 정합(트랙별)**: aroma-sym은 이식 소스 규약과 완전 동일하게 학습해야 함. 트랙 A(3종) = **multi · 640 · rect · epochs 100 · patience 50 · seeds 42 1 2**. 트랙 B(aitex) = **single · 256 · no-rect · epochs 300 · patience 50 · seeds 1 2 42**. 공통 = val_frac 0.3 · synth_ratio 1.0 · batch 64. STEP 3 파라미터 변경 금지.
- **비교 범위**: 유효 비교는 (a) aroma-sym vs 이식 random/baseline(같은 트랙), (b) aroma-sym vs aroma-CP(같은 소스). **aroma-CN(controlnet arm)과는 규약이 달라 직접 비교 금지**. **트랙 A vs 트랙 B 절대값 직접 비교 금지**(aitex는 tile-level·single, 3종은 multi).
- **개선 순효과** = aroma-sym vs aroma-CP (동일 selection·seed·real, 합성 파이프라인만 상이). random은 무변경 통제.
- **mtd 천장**: mtd 단독으로 개선 성패를 headline 삼지 말 것 — near-ceiling. flat이면 headroom 있는 데이터셋으로 재검증.
- **H1 벽**: 게이트/배치가 기제로 완벽해도 copy-paste 재조합이 무정보면 flat 가능 — 배치개선의 한계가 아니라 생성 novelty(상류) 문제.
- **사후 튜닝 금지**: 결과 보고 후 τ·seed·synth_ratio·epochs 변경 금지.
- **테스트 코드·pytest 금지**(CLAUDE.md). 검증은 Colab 실행으로.
