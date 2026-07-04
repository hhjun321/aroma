# Step 2 — `prompt_generation.py` Colab 실행

> **실행 환경**: CPU
> Phase 0 + Step 1 출력을 읽어 형태학 클러스터 × 컨텍스트 빈 조합별 자연어 프롬프트 생성.
> Step 1 완료 후 실행.

> **exp4v2 대상 데이터셋 (v2-1 확정 4종)**: `severstal`·`mvtec_leather`·`aitex`·`mtd`. (aitex/mtd 선행: `multidomain_integration_verify_execute.md`)

## 환경변수

```python
import os

os.environ['AROMA_SCRIPTS']  = '/content/AROMA/scripts/aroma'
os.environ['AROMA_OUT']      = f"{os.environ['DRIVE']}/aroma_output"

DATASET_KEY = 'severstal'   # ← 확정 4종: severstal / mvtec_leather / aitex / mtd

os.environ['DATASET_KEY']    = DATASET_KEY
os.environ['PROFILING_DIR']  = f"{os.environ['AROMA_OUT']}/profiling/{DATASET_KEY}"
os.environ['COMPLEXITY_DIR'] = f"{os.environ['AROMA_OUT']}/complexity/{DATASET_KEY}"
os.environ['PROMPTS_DIR']    = f"{os.environ['AROMA_OUT']}/prompts/{DATASET_KEY}"
```

## 실행

```python
!python $AROMA_SCRIPTS/prompt_generation.py \
    --profiling_dir  $PROFILING_DIR \
    --complexity_dir $COMPLEXITY_DIR \
    --output_dir     $PROMPTS_DIR
```

## 결과 확인

```python
import json, os

with open(f"{os.environ['PROMPTS_DIR']}/prompts.json") as f:
    prompts = json.load(f)

print(f"총 프롬프트 수: {len(prompts)}")

# 클러스터별 집계
from collections import defaultdict
by_cluster = defaultdict(list)
for key, entry in prompts.items():
    by_cluster[entry['cluster_id']].append(entry)

print()
for cid in sorted(by_cluster.keys()):
    entries = by_cluster[cid]
    label = entries[0].get('phase0_label', '')
    print(f"Cluster {cid} [{label}]  조합수={len(entries)}")
    # 상위 deficit 조합 출력
    top = sorted(entries, key=lambda x: x.get('deficit', 0), reverse=True)[:2]
    for e in top:
        print(f"  [{e['cell_key']}] deficit={e['deficit']:.3f}  prob={e['prior_prob']:.3f}")
        print(f"    → {e['prompt']}")
```

## 출력 파일

| 파일 | 내용 |
|------|------|
| `prompts.json` | `"{cluster_id}_{cell_key}"` 키 → prompt, descriptor, deficit, prior_prob |
| `prompts_summary.md` | 마크다운 테이블 (전체 조합 목록) |


---

## 전체 데이터셋 일괄 실행 (병렬)

`dataset_config.json`에서 datasets 목록 자동 로드, step1 출력(`complexity/{ds}/`)이 없는 데이터셋은 skip.  
`!python` 매직은 IPython 전용이라 스레드 내에서 동작하지 않으므로 의도적으로 `subprocess.run` 사용.

**셀 1 — 준비 상태 확인:**

```python
import os, json
from pathlib import Path

AROMA_OUT      = os.environ['AROMA_OUT']
DATASET_CONFIG = os.environ.get('DATASET_CONFIG', '/content/AROMA/dataset_config.json')

with open(DATASET_CONFIG) as f:
    cfg = json.load(f)
# exp4v2 확정 4종만 실행. config 전체(비-exp4v2 포함)를 돌리려면 아래 줄로 교체.
datasets = ["severstal", "mvtec_leather", "aitex", "mtd"]
# datasets = [k for k in cfg if not k.startswith('_')]
print(f"총 {len(datasets)}개 데이터셋\n")

for ds in datasets:
    ready = Path(f"{AROMA_OUT}/complexity/{ds}").exists()
    print(f"{'✓ 준비됨' if ready else '↷ skip(step1 없음)':<22} {ds}")
```

**셀 2 — 병렬 실행:**

```python
import os, json, sys, subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

AROMA_OUT      = os.environ['AROMA_OUT']
AROMA_SCRIPTS  = os.environ['AROMA_SCRIPTS']
DATASET_CONFIG = os.environ.get('DATASET_CONFIG', '/content/AROMA/dataset_config.json')
SCRIPT         = f"{AROMA_SCRIPTS}/prompt_generation.py"
MAX_WORKERS    = 3

with open(DATASET_CONFIG) as f:
    cfg = json.load(f)
# exp4v2 확정 4종만 실행. config 전체(비-exp4v2 포함)를 돌리려면 아래 줄로 교체.
datasets = ["severstal", "mvtec_leather", "aitex", "mtd"]
# datasets = [k for k in cfg if not k.startswith('_')]

def run_one(ds):
    if not Path(f"{AROMA_OUT}/complexity/{ds}").exists():
        return 'skip', ds, 'step1 출력 없음'
    cmd = [sys.executable, SCRIPT,
           '--profiling_dir',  f"{AROMA_OUT}/profiling/{ds}",
           '--complexity_dir', f"{AROMA_OUT}/complexity/{ds}",
           '--output_dir',     f"{AROMA_OUT}/prompts/{ds}"]
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
