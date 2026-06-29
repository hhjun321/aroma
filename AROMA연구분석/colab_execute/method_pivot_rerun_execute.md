# AROMA 방법론 전환 재실행 가이드 (compatibility 선택 + seamless 블렌딩)

**대상**: deficit-aware 폐기 → **compatibility ROI 선택(L1) + seamlessClone+Reinhard 블렌딩(L3)** 적용 후 전 파이프라인 재실행.
dev_note: `.claude/.dev_note/aroma_roi-synthesis_compatibility-context-blend.md`.

## 확정 실험 설계

| 실험 | 데이터셋 | 조건 | 지표 |
|------|---------|------|------|
| exp2 | 5셋 | random vs aroma | 선택 ROI 평균 compatibility(ctx_prior)+quality |
| exp3 | 5셋 | baseline/random/aroma | FID + one-class AD |
| exp4v2 | severstal | baseline/random/**casda**/aroma (4-way) | supervised mAP |
| exp4v2 | carpet/leather/macaroni/fryum | baseline/random/aroma (3-way) | supervised mAP |

**5셋** = `severstal carpet leather macaroni fryum`. casda=철강 전용→severstal만.
**조건 격리**: 합성 3 arm(random/casda/aroma) 모두 `--blend-mode seamless` 동일, ROI 선택만 다름.

---

## 0. 전제 / 환경변수

> ⚠️ **최신 코드 클론 필수** — L1/L3(`compatibility` strategy, `seamless` blend, utils import 수정)가 반영된 커밋이어야 함. 구버전이면 전환 효과 안 나옴.

```python
import os, json

os.environ['AROMA_SCRIPTS'] = "/content/AROMA/scripts/aroma"
os.environ['AROMA_OUT']     = f"{os.environ['DRIVE']}/aroma_output"
os.environ['AROMA_DATA']    = f"{os.environ['DRIVE']}/Aroma"
os.environ['DATASET_CONFIG'] = os.environ.get('DATASET_CONFIG', '/content/AROMA/dataset_config.json')
# utils import 복구는 코드(__file__ 부트스트랩)로 처리되나, 보강용으로 AROMA_REF도 클론 경로로:
os.environ['AROMA_REF']     = "/content/AROMA"

# 신규 method-pivot 출력 루트 (기존 산출과 분리)
os.environ['ROI_MP']        = f"{os.environ['AROMA_OUT']}/roi_mp"          # compatibility 선택
os.environ['ROI_MP_RAND']   = f"{os.environ['AROMA_OUT']}/roi_mp_random"  # random 선택
os.environ['SYN_AROMA']     = f"{os.environ['AROMA_OUT']}/synthetic_mp"        # aroma seamless
os.environ['SYN_RAND']      = f"{os.environ['AROMA_OUT']}/synthetic_mp_random" # random seamless
os.environ['SYN_CASDA']     = f"{os.environ['AROMA_OUT']}/synthetic_mp_casda"  # casda seamless (severstal)
os.environ['EXP2_MP']       = f"{os.environ['AROMA_OUT']}/exp2_mp"
os.environ['EXP3_MP']       = f"{os.environ['AROMA_OUT']}/exp3_mp"
os.environ['EXP4_MP']       = f"{os.environ['AROMA_OUT']}/exp4v2_mp"

DATASETS = ["severstal", "mvtec_carpet", "mvtec_leather", "visa_macaroni", "visa_fryum"]
SEED = 42
for k in ['ROI_MP','SYN_AROMA','SYN_RAND','EXP3_MP','EXP4_MP']:
    print(k, os.environ[k])

with open(os.environ['DATASET_CONFIG']) as f:
    CFG = json.load(f)
def is_multi(ds): return CFG.get(ds, {}).get("class_mode") == "multi"   # severstal=True
```

전제: step0(profiling) + step2(prompts)가 5셋에 존재(`$AROMA_OUT/profiling/{ds}`, `.../prompts/{ds}`). 없으면 해당 stage 먼저.

---

## 1. Stage 3 — ROI 선택 (compatibility=aroma, random)

AROMA arm은 신규 `--sampling_strategy compatibility`, random arm은 `--sampling_strategy random`. severstal은 multi-class 플래그.

```python
import sys, subprocess
from pathlib import Path
AROMA_OUT, AROMA_SCRIPTS = os.environ['AROMA_OUT'], os.environ['AROMA_SCRIPTS']
ROI_SEL = f"{AROMA_SCRIPTS}/roi_selection.py"

def multi_flags(ds):
    return ["--class_mode","multi","--class_floor"] if is_multi(ds) else []

def select(ds, strategy, out_root):
    prof, prom = f"{AROMA_OUT}/profiling/{ds}", f"{AROMA_OUT}/prompts/{ds}"
    if not (Path(prof).exists() and Path(prom).exists()):
        return 'skip', f"{ds}: profiling/prompts 없음"
    cmd = [sys.executable, ROI_SEL,
           "--profiling_dir", prof, "--prompts_dir", prom,
           "--output_dir", f"{out_root}/{ds}",
           "--sampling_strategy", strategy,
           "--top_k", "200", "--seed", str(SEED)]
    if strategy == "compatibility":
        cmd += ["--img_diversity_cap","1"] + multi_flags(ds)
        # quality gate: 첫 run서 quality_score 분포 확인 후 임계 조정. 우선 OFF(0.0):
        cmd += ["--min_quality","0.0"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        return 'fail', '\n'.join((r.stderr or '').splitlines()[-3:])
    return 'ok', ''

for ds in DATASETS:
    sa,ma = select(ds, "compatibility", os.environ['ROI_MP'])
    sr,mr = select(ds, "random",        os.environ['ROI_MP_RAND'])
    print(f"{'✓' if sa==sr=='ok' else '✗'} {ds:16s} aroma={sa} random={sr}")
    for m in (ma,mr):
        if m: print("   ", m)
```

---

## 2. exp2 — compatibility/quality 선택 지표 (inline, 스크립트 불요)

선택된 ROI(`roi_selected.json`)의 평균 `ctx_prior`(=compatibility)·`quality_score`를 aroma vs random 비교. coverage/deficit 폐기.

```python
import json, numpy as np, pathlib

def mean_metrics(path):
    sel = json.load(open(path))
    if not sel: return None
    cp = np.mean([c.get("ctx_prior",0.0) for c in sel])
    q  = np.mean([c.get("quality_score",1.0) for c in sel])
    return float(cp), float(q), len(sel)

print(f"{'dataset':16}{'compat A/R':>20}{'quality A/R':>20}")
rows={}
for ds in DATASETS:
    a = mean_metrics(f"{os.environ['ROI_MP']}/{ds}/roi_selected.json")
    r = mean_metrics(f"{os.environ['ROI_MP_RAND']}/{ds}/roi_selected.json")
    if not (a and r): 
        print(f"{ds:16} (missing)"); continue
    rows[ds]={"aroma":a,"random":r}
    print(f"{ds:16}{f'{a[0]:.3f}/{r[0]:.3f}':>20}{f'{a[1]:.3f}/{r[1]:.3f}':>20}")
pathlib.Path(os.environ['EXP2_MP']).mkdir(parents=True, exist_ok=True)
json.dump(rows, open(f"{os.environ['EXP2_MP']}/exp2_compat_quality.json","w"), indent=2)
print("\n기대: aroma compatibility > random (더 적합한 배경 선택). quality도 ≥.")
```

---

## 3. Stage 4 — 합성 (seamless 블렌딩, clean-bg gate ON)

3 arm 모두 `--blend-mode seamless` + 게이트 ON. aroma는 `$ROI_MP`, random은 candidates 풀, casda는 severstal만.

```python
GEN_DEF  = f"{AROMA_SCRIPTS}/generate_defects.py"
GEN_RAND = f"{AROMA_SCRIPTS}/generate_random.py"
GATE = ["--reject-clean-bg","--min-bg-quality","0.7","--bg-blur-threshold","100.0"]

def normal_dir(ds): return CFG[ds]["image_dir"]

def synth_aroma(ds):
    cmd=[sys.executable, GEN_DEF,
         "--roi_dir", f"{os.environ['ROI_MP']}/{ds}",
         "--normal_dir", normal_dir(ds),
         "--output_dir", f"{os.environ['SYN_AROMA']}/{ds}",
         "--method","copy_paste","--blend_mode","seamless",
         "--n_per_roi","3","--seed",str(SEED),"--local_staging"] + GATE
    return subprocess.run(cmd, capture_output=True, text=True)

def synth_random(ds):
    cmd=[sys.executable, GEN_RAND,
         "--candidates_json", f"{os.environ['ROI_MP']}/{ds}/roi_candidates.json",
         "--normal_dir", normal_dir(ds),
         "--output_dir", f"{os.environ['SYN_RAND']}/{ds}",
         "--top_k","200","--n_per_roi","3","--seed",str(SEED),
         "--blend-mode","seamless"] + GATE
    return subprocess.run(cmd, capture_output=True, text=True)

for ds in DATASETS:
    ra = synth_aroma(ds); rr = synth_random(ds)
    print(f"{'✓' if ra.returncode==rr.returncode==0 else '✗'} {ds:16s} "
          f"aroma={'ok' if ra.returncode==0 else 'FAIL'} random={'ok' if rr.returncode==0 else 'FAIL'}")
    for r in (ra,rr):
        if r.returncode!=0: print("   ", '\n'.join((r.stderr or '').splitlines()[-3:]))
```

### casda arm (severstal 전용)

CASDA metadata_csv 경로 필요(기존 casda run 산출). seamless 동일 적용.

```python
GEN_CASDA = f"{AROMA_SCRIPTS}/generate_casda.py"
SEVERSTAL_CASDA_META = f"{os.environ['DRIVE']}/severstal/casda_metadata.csv"  # ← 실제 경로로 수정

cmd=[sys.executable, GEN_CASDA,
     "--metadata_csv", SEVERSTAL_CASDA_META,
     "--normal_dir", normal_dir("severstal"),
     "--output_dir", f"{os.environ['SYN_CASDA']}/severstal",
     "--n_per_roi","3","--seed",str(SEED),"--local_staging",
     "--blend-mode","seamless"] + GATE
r=subprocess.run(cmd, capture_output=True, text=True)
print("casda severstal:", "ok" if r.returncode==0 else "FAIL")
if r.returncode!=0: print('\n'.join((r.stderr or '').splitlines()[-5:]))
```

**합성 검증 (배치 measurable 육안)**: 동일 ROI를 다른 배경에 배치 시 출력 픽셀이 실제 달라지는지 `$SYN_AROMA/{ds}/images/`서 1~2장 확인. seamless+Reinhard가 작동하면 alpha 대비 경계 자연스럽고 색조 정합.

---

## 4. exp3 — FID + one-class AD (5셋, baseline/random/aroma)

```python
!pip install torchmetrics[image] anomalib lpips -q
```

```python
EXP3 = f"{AROMA_SCRIPTS}/experiments/exp3_generation_quality.py"
DS_KEYS = " ".join(DATASETS)
os.environ['DS_KEYS'] = DS_KEYS
# FID (CPU 가능)
!python $EXP3 --mode fid \
    --random_synthetic_dir $SYN_RAND \
    --aroma_synthetic_dir  $SYN_AROMA \
    --real_data_dir        $AROMA_DATA \
    --dataset_keys         $DS_KEYS \
    --output_dir           $EXP3_MP \
    --seed                 42 --device cpu
# AD (GPU)
!python $EXP3 --mode ad \
    --random_synthetic_dir $SYN_RAND \
    --aroma_synthetic_dir  $SYN_AROMA \
    --real_data_dir        $AROMA_DATA \
    --dataset_keys         $DS_KEYS \
    --output_dir           $EXP3_MP \
    --seed                 42 --image_size 256
```

> ⚠️ one-class AD는 nominal-partition confound 캐비엇 있음(메모리). 다운스트림 신뢰지표는 exp4v2(아래). exp3 AD는 보조.

---

## 5. exp4v2 — supervised detection (2층위)

`{synthetic_dir}/{ds}/annotations.json`을 읽음 → §3 합성 산출 사용. multi-seed 권장(`--seeds 42 1 2`).

### 5a. severstal — 4-way (casda 포함, multi-class)

```python
EXP4 = f"{AROMA_SCRIPTS}/experiments/exp4_v2_supervised_detection.py"
!python $EXP4 \
    --model yolov8n \
    --condition baseline random casda aroma \
    --dataset_keys severstal \
    --class_mode multi \
    --random_synthetic_dir $SYN_RAND \
    --casda_synthetic_dir  $SYN_CASDA \
    --aroma_synthetic_dir  $SYN_AROMA \
    --real_data_dir        $AROMA_DATA \
    --output_dir           $EXP4_MP/severstal \
    --seeds 42 1 2 \
    --synth_ratio 0.5 --patience 15
```

### 5b. 나머지 4셋 — 3-way (casda 제외, single-class)

```python
os.environ['DS_OBJ'] = "mvtec_carpet mvtec_leather visa_macaroni visa_fryum"
!python $EXP4 \
    --model yolov8n \
    --condition baseline random aroma \
    --dataset_keys $DS_OBJ \
    --class_mode single \
    --random_synthetic_dir $SYN_RAND \
    --aroma_synthetic_dir  $SYN_AROMA \
    --real_data_dir        $AROMA_DATA \
    --output_dir           $EXP4_MP/objcentric \
    --seeds 42 1 2 \
    --synth_ratio 0.5 --patience 15
```

### 결과 확인

```python
import json
for sub in ["severstal","objcentric"]:
    p=f"{os.environ['EXP4_MP']}/{sub}/exp4v2_summary.md"
    try:
        print(f"\n===== {sub} ====="); print(open(p).read())
    except FileNotFoundError:
        print(f"[MISS] {p}")
```

---

## 6. ablation (선택)

핵심 기여 분리:
- **ROI 선택**: `--sampling_strategy compatibility` vs `random` vs `deficit_aware`(보존됨) — 동일 blend(seamless) 하.
- **blend**: `--blend_mode seamless` vs `alpha` — 동일 선택 하. "배치가 measurable해졌나" 검증.

```python
# 예: deficit_aware ablation arm 추가 생성 후 exp4v2 비교 (필요 시)
# select(ds, "deficit_aware", f"{AROMA_OUT}/roi_mp_deficit")  ... 동일 패턴
```

---

## 출력 파일

| 경로 | 내용 |
|------|------|
| `$ROI_MP/{ds}/roi_selected.json` | compatibility 선택 ROI |
| `$EXP2_MP/exp2_compat_quality.json` | 평균 compatibility/quality (aroma vs random) |
| `$SYN_AROMA|SYN_RAND|SYN_CASDA/{ds}/` | seamless 합성 + annotations.json |
| `$EXP3_MP/exp3_results.json` | FID + AD |
| `$EXP4_MP/{severstal,objcentric}/exp4v2_results.json` | supervised mAP (multi-seed) |

## 주의

- **최신 코드(L1/L3) 클론 확인** — 구버전이면 `compatibility` strategy/`seamless` blend 부재로 실패.
- **clean-bg gate 항상 ON** (3 arm 동일).
- **조건 격리**: random/casda/aroma 모두 seamless — 차이는 ROI 선택뿐.
- **casda metadata_csv 경로** 실제값으로 수정 필요(§3 casda arm).
- **quality threshold**: 첫 compatibility run의 `roi_candidates.json` `quality_score` 분포 확인 후 `--min_quality` 상향 검토(현재 0.0=OFF).
- 신규 .ipynb 금지 — 본 .md 셀을 Colab에 복사 실행. pytest 금지(Colab 직접 검증).
- 부하/성능 실측 자동 금지(load-test 정책) — mAP·FID는 기능 검증 목적이라 허용.
