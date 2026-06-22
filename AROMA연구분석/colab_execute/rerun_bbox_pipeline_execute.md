# 결함 bbox·mask 영속화 — 파이프라인 전체 재실행 가이드 (B안)

> **목적**: copy_paste 합성이 결함을 크롭하지 않고 full 이미지를 통째로 paste하던 버그(`bbox=[0,0,W,H]`) 수정 반영.
> profiling이 결함 mask PNG + `defect_bbox`/`defect_mask_path`를 영속화 → roi_selection 전파 → generate_defects가 타이트 크롭 + 실제 mask 형태로 합성.
> **전 데이터셋 병렬 재실행.** 기존 per-stage 가이드의 skip 가드는 "출력 존재 시 skip"이라 재생성을 막으므로, 본 문서는 **새 산출물(마이그레이션 마커) 기준 skip**으로 교체했다.

---

## 재실행 의존 순서

```
phase0 (distribution_profiling.py)   ← 결함 mask PNG + CSV 신규 컬럼 생성
   ↓
step3  (roi_selection.py)            ← 신규 컬럼 전파
   ↓
step4  (generate_defects.py, AROMA)  +  generate_random.py (RANDOM)   ← 타이트 크롭 합성
   ↓
exp4v2 (exp4_v2_supervised_detection.py)   ← precision 재측정
```

**재실행 불필요**:
- **step1(compute_complexity, MCI/CCI)**: morph 특징 *값*은 불변(컬럼만 additive 추가) → MCI/CCI 동일. 신규 컬럼은 `MORPH_FEATURES` 화이트리스트 밖이라 무시됨.
- **step2(prompt_generation)**: 입력(클러스터/프롬프트)과 무관.
- 클러스터/deficit JSON(`morphology_clusters.json`, `compatibility_matrix.json`, `deficit_analysis.json`)은 phase0가 재생성하므로 step3가 자동으로 최신값 사용.

> ⚠️ **compute_complexity.py 재실행 시 주의 없음** — 신규 CSV 컬럼은 무시되어 안전. 단 굳이 재실행할 필요는 없다.

---

## 0. 공통 환경변수

```python
import os, json

os.environ['AROMA_REF']      = '/content/AROMA'
os.environ['AROMA_SCRIPTS']  = '/content/AROMA/scripts/aroma'
os.environ['AROMA_OUT']      = f"{os.environ['DRIVE']}/aroma_output"
os.environ['AROMA_DATA']     = f"{os.environ['DRIVE']}/Aroma"
os.environ['DATASET_CONFIG'] = os.environ.get('DATASET_CONFIG', '/content/AROMA/dataset_config.json')

# exp4v2 전용
os.environ['RANDOM_SYNTH_DIR'] = f"{os.environ['AROMA_OUT']}/synthetic_random"
os.environ['AROMA_SYNTH_DIR']  = f"{os.environ['AROMA_OUT']}/synthetic"
os.environ['EXP4V2_OUT']       = f"{os.environ['AROMA_OUT']}/exp4v2"

with open(os.environ['DATASET_CONFIG']) as f:
    CFG = json.load(f)
DATASETS = [k for k in CFG if not k.startswith('_')]
print(f"총 {len(DATASETS)}개 데이터셋:", DATASETS)
```

> 최신 코드 pull 필수 (B안 수정 반영):
> ```python
> !cd /content/AROMA && git pull
> ```

---

## 1. phase0 재실행 — distribution_profiling.py (병렬)

결함 mask PNG(`defect_masks/`) + CSV 신규 컬럼(`defect_bbox`, `defect_mask_path`) 생성.

> **skip 마커**: `morphology_features.csv` 헤더에 `defect_bbox` 컬럼이 있으면 이미 마이그레이션됨 → skip.
> **`--num_workers 1`** 고정 (데이터셋 병렬 ×3과 내부 워커 중첩 회피).

```python
import os, json, sys, subprocess, csv
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

AROMA_OUT      = os.environ['AROMA_OUT']
AROMA_REF      = os.environ.get('AROMA_REF', '/content/AROMA')
DATASET_CONFIG = os.environ['DATASET_CONFIG']
SCRIPT         = f"{AROMA_REF}/scripts/distribution_profiling.py"
MAX_WORKERS    = 3

with open(DATASET_CONFIG) as f:
    cfg = json.load(f)
datasets = [k for k in cfg if not k.startswith('_')]

def _csv_has_col(path, col):
    try:
        with open(path, newline='') as f:
            return col in (next(csv.reader(f), []) or [])
    except Exception:
        return False

def run_one(ds):
    csv_path = f"{AROMA_OUT}/profiling/{ds}/morphology_features.csv"
    # 이미 마이그레이션된 데이터셋은 skip (defect_bbox 컬럼 존재)
    if _csv_has_col(csv_path, 'defect_bbox'):
        return 'skip', ds, '이미 마이그레이션됨(defect_bbox 존재)'
    cmd = [sys.executable, SCRIPT,
           '--dataset_config', DATASET_CONFIG,
           '--dataset_key',    ds,
           '--output_dir',     f"{AROMA_OUT}/profiling/{ds}",
           '--num_workers',    '1']
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        tail = '\n'.join(r.stderr.strip().splitlines()[-3:]) if r.stderr else ''
        return 'fail', ds, tail
    return 'ok', ds, ''

results = []
with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
    futs = {ex.submit(run_one, ds): ds for ds in datasets}
    for fut in as_completed(futs):
        status, ds, msg = fut.result()
        results.append((status, ds, msg))
        icon = '✓' if status == 'ok' else ('↷' if status == 'skip' else '✗')
        print(f"{icon} {ds:30s}  {status}")
        if msg:
            print(f"    {msg}")

from collections import Counter
c = Counter(s for s, *_ in results)
print(f"\n완료: {c['ok']}개  skip: {c['skip']}개  실패: {c['fail']}개")
if c['fail']:
    for s, ds, msg in results:
        if s == 'fail':
            print(f"  {ds}: {msg}")
```

> **강제 재실행** (이미 마이그레이션된 것도 다시): `run_one`의 `_csv_has_col` skip 분기를 주석 처리.

**검증**:

```python
import csv, os
from pathlib import Path
for ds in DATASETS:
    pdir = Path(f"{os.environ['AROMA_OUT']}/profiling/{ds}")
    csv_p = pdir / 'morphology_features.csv'
    mask_n = len(list((pdir / 'defect_masks').glob('*.png'))) if (pdir / 'defect_masks').exists() else 0
    if not csv_p.exists():
        print(f"❌ {ds}: CSV 없음"); continue
    with open(csv_p, newline='') as f:
        rd = csv.DictReader(f); rows = list(rd)
        has = 'defect_bbox' in rd.fieldnames
        # bbox가 타이트(full 이미지 아님)인지 샘플
        sample = next((r['defect_bbox'] for r in rows if r.get('defect_bbox')), '없음')
    print(f"{'✅' if has else '❌'} {ds:20s} defect_bbox컬럼={has}  masks={mask_n}  샘플bbox={sample}")
```

---

## 2. step3 재실행 — roi_selection.py (병렬)

`defect_bbox`/`defect_mask_path`를 `roi_candidates.json`/`roi_selected.json`로 전파.

> **skip 마커**: `roi_candidates.json` 첫 항목에 `defect_mask_path` 키가 있으면 마이그레이션됨.
> step2 출력(`prompts/{ds}/prompts.json`) 필요 — 없으면 skip.

```python
import os, json, sys, subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

AROMA_OUT      = os.environ['AROMA_OUT']
AROMA_SCRIPTS  = os.environ['AROMA_SCRIPTS']
SCRIPT         = f"{AROMA_SCRIPTS}/roi_selection.py"
MAX_WORKERS    = 3

with open(os.environ['DATASET_CONFIG']) as f:
    cfg = json.load(f)
datasets = [k for k in cfg if not k.startswith('_')]

def _cand_migrated(path):
    try:
        with open(path) as f:
            d = json.load(f)
        return bool(d) and 'defect_mask_path' in d[0]
    except Exception:
        return False

def run_one(ds):
    if not Path(f"{AROMA_OUT}/prompts/{ds}/prompts.json").exists():
        return 'skip', ds, 'step2 출력 없음'
    if _cand_migrated(f"{AROMA_OUT}/roi/{ds}/roi_candidates.json"):
        return 'skip', ds, '이미 마이그레이션됨(defect_mask_path 존재)'
    cmd = [sys.executable, SCRIPT,
           '--profiling_dir',     f"{AROMA_OUT}/profiling/{ds}",
           '--prompts_dir',       f"{AROMA_OUT}/prompts/{ds}",
           '--sampling_strategy', 'deficit_aware',
           '--top_k',             '200',
           '--output_dir',        f"{AROMA_OUT}/roi/{ds}"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        tail = '\n'.join(r.stderr.strip().splitlines()[-3:]) if r.stderr else ''
        return 'fail', ds, tail
    return 'ok', ds, ''

results = []
with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
    futs = {ex.submit(run_one, ds): ds for ds in datasets}
    for fut in as_completed(futs):
        status, ds, msg = fut.result()
        results.append((status, ds, msg))
        icon = '✓' if status == 'ok' else ('↷' if status == 'skip' else '✗')
        print(f"{icon} {ds:30s}  {status}")
        if msg:
            print(f"    {msg}")

from collections import Counter
c = Counter(s for s, *_ in results)
print(f"\n완료: {c['ok']}개  skip: {c['skip']}개  실패: {c['fail']}개")
```

**검증**:

```python
import json, os
for ds in DATASETS:
    p = f"{os.environ['AROMA_OUT']}/roi/{ds}/roi_selected.json"
    if not os.path.exists(p):
        print(f"❌ {ds}: roi_selected 없음"); continue
    sel = json.load(open(p))
    e = sel[0] if sel else {}
    print(f"{'✅' if 'defect_bbox' in e else '❌'} {ds:20s} "
          f"n={len(sel)}  bbox={e.get('defect_bbox')}  mask={'있음' if e.get('defect_mask_path') else '없음'}")
```

---

## 3. step4 + RANDOM 재합성 (병렬)

마스크 정보가 새로 생겼으므로 AROMA·RANDOM 합성 **모두 재생성**. 이 단계는 **항상 재실행**(기존 합성물 덮어쓰기) — 상류(step3) 마이그레이션 확인 후 진행.

### 3-1. AROMA 합성 — generate_defects.py

```python
import os, json, sys, subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

AROMA_OUT      = os.environ['AROMA_OUT']
AROMA_SCRIPTS  = os.environ['AROMA_SCRIPTS']
SCRIPT         = f"{AROMA_SCRIPTS}/generate_defects.py"
MAX_WORKERS    = 3

with open(os.environ['DATASET_CONFIG']) as f:
    cfg = json.load(f)
datasets = [k for k in cfg if not k.startswith('_')]

def _roi_migrated(ds):
    try:
        d = json.load(open(f"{AROMA_OUT}/roi/{ds}/roi_selected.json"))
        return bool(d) and 'defect_mask_path' in d[0]
    except Exception:
        return False

def run_one(ds):
    if not _roi_migrated(ds):
        return 'skip', ds, 'step3 미마이그레이션 — roi_selection 먼저 재실행'
    normal_dir = cfg[ds].get('image_dir', '')
    if not normal_dir:
        return 'skip', ds, 'image_dir 없음'
    cmd = [sys.executable, SCRIPT,
           '--roi_dir',    f"{AROMA_OUT}/roi/{ds}",
           '--normal_dir', normal_dir,
           '--output_dir', f"{AROMA_OUT}/synthetic/{ds}",
           '--method',     'copy_paste',
           '--n_per_roi',  '3',
           '--feather_px', '4',
           '--seed',       '42',
           '--local_staging']
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        tail = '\n'.join(r.stderr.strip().splitlines()[-3:]) if r.stderr else ''
        return 'fail', ds, tail
    return 'ok', ds, ''

results = []
with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
    futs = {ex.submit(run_one, ds): ds for ds in datasets}
    for fut in as_completed(futs):
        status, ds, msg = fut.result()
        results.append((status, ds, msg))
        icon = '✓' if status == 'ok' else ('↷' if status == 'skip' else '✗')
        print(f"{icon} {ds:30s}  {status}")
        if msg: print(f"    {msg}")

from collections import Counter
c = Counter(s for s, *_ in results)
print(f"\n완료: {c['ok']}개  skip: {c['skip']}개  실패: {c['fail']}개")
```

### 3-2. RANDOM 합성 — generate_random.py

```python
import os, json, sys, subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

AROMA_OUT      = os.environ['AROMA_OUT']
AROMA_SCRIPTS  = os.environ['AROMA_SCRIPTS']
RANDOM_SYNTH   = os.environ['RANDOM_SYNTH_DIR']
SCRIPT         = f"{AROMA_SCRIPTS}/generate_random.py"
MAX_WORKERS    = 3

with open(os.environ['DATASET_CONFIG']) as f:
    cfg = json.load(f)
datasets = [k for k in cfg if not k.startswith('_')]

def _cand_migrated(ds):
    try:
        d = json.load(open(f"{AROMA_OUT}/roi/{ds}/roi_candidates.json"))
        return bool(d) and 'defect_mask_path' in d[0]
    except Exception:
        return False

def run_one(ds):
    if not _cand_migrated(ds):
        return 'skip', ds, 'roi_candidates 미마이그레이션'
    normal_dir = cfg[ds].get('image_dir', '')
    if not normal_dir:
        return 'skip', ds, 'image_dir 없음'
    cmd = [sys.executable, SCRIPT,
           '--candidates_json', f"{AROMA_OUT}/roi/{ds}/roi_candidates.json",
           '--normal_dir',      normal_dir,
           '--output_dir',      f"{RANDOM_SYNTH}/{ds}",
           '--top_k',           '200',
           '--n_per_roi',       '3',
           '--seed',            '42']
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        tail = '\n'.join(r.stderr.strip().splitlines()[-3:]) if r.stderr else ''
        return 'fail', ds, tail
    return 'ok', ds, ''

results = []
with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
    futs = {ex.submit(run_one, ds): ds for ds in datasets}
    for fut in as_completed(futs):
        status, ds, msg = fut.result()
        results.append((status, ds, msg))
        icon = '✓' if status == 'ok' else ('↷' if status == 'skip' else '✗')
        print(f"{icon} {ds:30s}  {status}")
        if msg: print(f"    {msg}")

from collections import Counter
c = Counter(s for s, *_ in results)
print(f"\n완료: {c['ok']}개  skip: {c['skip']}개  실패: {c['fail']}개")
```

**검증 (AROMA + RANDOM 공통)**: bbox가 더 이상 `[0,0,W,H]`가 아닌지, mask가 실제 형태인지.

```python
import json, os
from pathlib import Path
for label, base in [("AROMA", os.environ['AROMA_SYNTH_DIR']), ("RANDOM", os.environ['RANDOM_SYNTH_DIR'])]:
    print(f"\n=== {label} ===")
    for ds in DATASETS:
        ann_p = Path(base) / ds / 'annotations.json'
        mask_dir = Path(base) / ds / 'masks'
        if not ann_p.exists():
            print(f"  ❌ {ds}: annotations 없음"); continue
        ann = json.load(open(ann_p))
        real = [a for a in ann if not a.get('dry_run')]
        e = real[0] if real else {}
        nmask = len(list(mask_dir.glob('*.png'))) if mask_dir.exists() else 0
        bbox = e.get('bbox')
        full = bbox and bbox[0] == 0 and bbox[1] == 0  # full-image 의심
        print(f"  {'⚠️full?' if full else '✅'} {ds:18s} n={len(real)} masks={nmask} bbox샘플={bbox}")
```

> `bbox샘플`이 `[0,0,1024,1024]`처럼 이미지 전체면 여전히 버그 — 상류 단계 마이그레이션/재합성 재확인. 타이트 박스(예: `[312,544,88,120]`)면 정상.
> 합성 이미지 1-2장 + 같은 stem의 `masks/*.png`를 직접 열어 결함이 타이트 영역에만, 실제 형태(타원 아님)로 들어갔는지 육안 확인 권장.

---

## 4. exp4v2 재측정 — GT mask + full-frame 데이터셋만 (ISP·visa_pcb 제외)

> ⚠️ **ISP·visa_pcb 제외 결정**:
> - **ISP**: GT 결함 mask 없음 → Otsu fallback이 이미지 62~85%를 결함으로 잡음 → defect_bbox·val GT 모두 full-image라 supervised detection 부적합.
> - **visa_pcb**: 작은 PCB 객체 + 큰 배경 void → 결함이 배경에 paste되는 배치 문제(별도 연구 주제).
> - 둘 다 **exp3 unsupervised AD(image-level)** 로만 평가.
> exp4v2 대상 = GT mask 보유 + full-frame `mvtec_cable visa_cashew mvtec_carpet mvtec_leather` (carpet/leather 텍스처는 배경 void 없어 배치 자동 정상).
> (§1~3 phase0/step3/step4·random 합성은 exp3용으로 ISP·visa_pcb 포함 전체 실행 유지.)

bbox/mask가 정확해진 합성으로 supervised detection 재실행. 기존 결과와 정합 안 되므로 **기존 `exp4v2_results.json` 삭제 후** 실행(또는 `--resume` 없이).

```python
# 기존 결과 초기화 (파라미터/합성 변경으로 캐시 불일치)
!rm -f $EXP4V2_OUT/exp4v2_results.json

!python $AROMA_SCRIPTS/experiments/exp4_v2_supervised_detection.py \
    --model yolov8n \
    --condition all \
    --dataset_keys mvtec_cable visa_cashew mvtec_carpet mvtec_leather \
    --random_synthetic_dir $RANDOM_SYNTH_DIR \
    --aroma_synthetic_dir  $AROMA_SYNTH_DIR \
    --real_data_dir        $AROMA_DATA \
    --output_dir           $EXP4V2_OUT \
    --seed 42 \
    --imgsz 640 \
    --val_frac 0.3 \
    --max_synth_per_ds 128 \
    --baseline_epochs 50 \
    --finetune_epochs 30
```

> 로그에서 `_write_yolo_labels`가 **mask 우선 경로** 발동(diff fallback 미사용), YOLO 라벨 bbox가 타이트한지 확인.
> 결과 분석은 `exp4v2_execute.md` §결과 확인 셀 사용.

---

## 재실행 체크리스트

- [ ] `git pull`로 B안 코드 반영
- [ ] phase0 → `defect_masks/` + CSV `defect_bbox` 컬럼 (타이트 샘플)
- [ ] step3 → `roi_selected.json`에 `defect_bbox`/`defect_mask_path` 전파
- [ ] step4(AROMA) + generate_random(RANDOM) → annotations bbox ≠ `[0,0,W,H]`, mask 실제 형태 (MVTec/VisA 기준; ISP는 Otsu라 여전히 full — exp3 전용)
- [ ] exp4v2 재측정 → **mvtec_cable visa_cashew mvtec_carpet mvtec_leather** (ISP·visa_pcb 제외), precision 회복 (이전 aroma 0.19/random 0.09 대비)
- [ ] (불필요) step1/step2 재실행 안 함 — morph 값 불변
