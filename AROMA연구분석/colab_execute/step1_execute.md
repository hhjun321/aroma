# Step 1 — `compute_complexity.py` Colab 실행

> **실행 환경**: CPU
> Phase 0 출력을 읽어 MCI / CCI 계산 + 정책 선택.
> Phase 0 완료 후 실행.

## 환경변수

```python
import os

os.environ['AROMA_SCRIPTS']  = '/content/AROMA/scripts/aroma'
os.environ['AROMA_OUT']      = f"{os.environ['DRIVE']}/aroma_output"

DATASET_KEY = 'isp_LSM_1'   # ← 변경 시 여기만 수정

os.environ['DATASET_KEY']    = DATASET_KEY
os.environ['PROFILING_DIR']  = f"{os.environ['AROMA_OUT']}/profiling/{DATASET_KEY}"
os.environ['COMPLEXITY_DIR'] = f"{os.environ['AROMA_OUT']}/complexity/{DATASET_KEY}"
```

## 실행

**Colab 실행 셀**:

```python
# 전체 실행 (~1-2분, --local_staging으로 Drive CSV I/O 단축)
!python $AROMA_SCRIPTS/compute_complexity.py \
    --profiling_dir  $PROFILING_DIR \
    --output_dir     $COMPLEXITY_DIR \
    --weight_mode    equal \
    --local_staging
```

## 결과 확인

```python
import json, os

with open(f"{os.environ['COMPLEXITY_DIR']}/complexity_report.json") as f:
    r = json.load(f)

print(f"MCI : {r['mci']:.4f}  |  CCI : {r['cci']:.4f}")
print(f"Morphology Policy : {r['morphology_policy']}")
print(f"Context Policy    : {r['context_policy']}")
print()
print('--- MCI 구성 요소 ---')
for k, v in r.get('mci_components', {}).get('raw', {}).items():
    norm_v = r['mci_components']['normalized'].get(k)
    norm_str = f"{norm_v:.4f}" if isinstance(norm_v, (int, float)) else "-.----"
    print(f"  {k:<20} raw={v:.4f}  norm={norm_str}")
sil = r.get('mci_components', {}).get('silhouette_score')
if sil is not None:
    print(f"  {'silhouette_score':<20} {sil:.4f}  (diagnostic)")
print()
print('--- CCI 구성 요소 ---')
for k, v in r.get('cci_components', {}).get('raw', {}).items():
    norm_v = r['cci_components']['normalized'].get(k)
    norm_str = f"{norm_v:.4f}" if isinstance(norm_v, (int, float)) else "-.----"
    print(f"  {k:<20} raw={v:.4f}  norm={norm_str}")
print()
print('--- 후보 정책 평가 ---')
for ev in r.get('evaluation_results', []):
    print(f"  [{ev.get('axis','?'):12s}] {ev.get('policy','?'):<14} "
          f"silhouette={ev.get('silhouette',0):.4f}  stability={ev.get('stability','-')}")
```

---

## 전체 데이터셋 일괄 실행 (병렬)

`dataset_config.json`에서 datasets 목록 자동 로드, phase0 출력(`morphology_features.csv`)이 없는 데이터셋은 skip.  
`!python` 매직은 IPython 전용이라 스레드 내에서 동작하지 않으므로 의도적으로 `subprocess.run` 사용.

**셀 1 — 준비 상태 확인:**

```python
import os, json
from pathlib import Path

AROMA_OUT      = os.environ['AROMA_OUT']
DATASET_CONFIG = os.environ.get('DATASET_CONFIG', '/content/AROMA/dataset_config.json')

with open(DATASET_CONFIG) as f:
    cfg = json.load(f)
datasets = [k for k in cfg if not k.startswith('_')]
print(f"총 {len(datasets)}개 데이터셋\n")

for ds in datasets:
    ready = Path(f"{AROMA_OUT}/profiling/{ds}/morphology_features.csv").exists()
    print(f"{'✓ 준비됨' if ready else '↷ skip(phase0 없음)':<22} {ds}")
```

**셀 2 — 병렬 실행:**

```python
import os, json, sys, subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

AROMA_OUT      = os.environ['AROMA_OUT']
AROMA_SCRIPTS  = os.environ['AROMA_SCRIPTS']
DATASET_CONFIG = os.environ.get('DATASET_CONFIG', '/content/AROMA/dataset_config.json')
SCRIPT         = f"{AROMA_SCRIPTS}/compute_complexity.py"
MAX_WORKERS    = 3

with open(DATASET_CONFIG) as f:
    cfg = json.load(f)
datasets = [k for k in cfg if not k.startswith('_')]

def run_one(ds):
    if not Path(f"{AROMA_OUT}/profiling/{ds}/morphology_features.csv").exists():
        return 'skip', ds, 'phase0 출력 없음'
    cmd = [sys.executable, SCRIPT,
           '--profiling_dir', f"{AROMA_OUT}/profiling/{ds}",
           '--output_dir',    f"{AROMA_OUT}/complexity/{ds}",
           '--weight_mode',   'equal',
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
        if msg:
            print(f"    {msg}")

from collections import Counter
c = Counter(s for s, *_ in results)
print(f"\n완료: {c['ok']}개  skip: {c['skip']}개  실패: {c['fail']}개")
if c['fail']:
    print("\n실패 목록:")
    for s, ds, msg in results:
        if s == 'fail':
            print(f"  {ds}: {msg}")
```
