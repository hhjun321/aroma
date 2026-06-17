# Phase 0 — `distribution_profiling.py` Colab 실행

> **실행 환경**: CPU
> ISP 도메인: GT mask 없음 → Otsu fallback (SAM 미지정 시 자동)

## 환경변수

```python
import os

os.environ['AROMA_REF']      = '/content/AROMA'
os.environ['DATASET_CONFIG'] = os.environ.get('DATASET_CONFIG', '/content/AROMA/dataset_config.json')
os.environ['AROMA_OUT']      = f"{os.environ['DRIVE']}/aroma_output"

DATASET_KEY = 'isp_LSM_1'   # ← 변경 시 여기만 수정

os.environ['DATASET_KEY']   = DATASET_KEY
os.environ['PROFILING_DIR'] = f"{os.environ['AROMA_OUT']}/profiling/{DATASET_KEY}"
```

## 실행

**Colab 실행 셀**:

```python
# smoke test (train 이미지 5장으로 빠른 경로 확인)
!python $AROMA_REF/scripts/distribution_profiling.py \
    --dataset_config $DATASET_CONFIG \
    --dataset_key    $DATASET_KEY \
    --output_dir     $PROFILING_DIR \
    --num_workers    8 \
    --max_images     5

# 전체 실행 (isp_LSM_1 기준 Otsu: ~5-8분, SAM T4: ~10-15분)
!python $AROMA_REF/scripts/distribution_profiling.py \
    --dataset_config $DATASET_CONFIG \
    --dataset_key    $DATASET_KEY \
    --output_dir     $PROFILING_DIR \
    --num_workers    8
```

> SAM 사용 시 `--sam_checkpoint /content/sam_vit_h.pth` 추가.

## 출력 확인

```python
import os
from pathlib import Path

profiling_dir = Path(os.environ['PROFILING_DIR'])
for fname in [
    'morphology_features.csv', 'context_features.csv',
    'distribution_analysis.json', 'morphology_clusters.json',
    'compatibility_matrix.json', 'deficit_analysis.json',
]:
    status = '✅' if (profiling_dir / fname).exists() else '❌ 누락'
    print(f'  {status}  {fname}')
```

## 소요 시간 참고

| 데이터셋 | Otsu | SAM (T4) | 병목 |
|---------|------|---------|------|
| isp_LSM_1 (3678 train) | ~5–8분 | ~10–15분 | context 3,773 이미지 × Drive I/O |
| mvtec_cable (224 train) | ~3–4분 | — (GT mask) | context 316 이미지 × Drive I/O |

---

## 전체 데이터셋 일괄 실행 (병렬)

`dataset_config.json`에서 datasets 목록 자동 로드, 이미 완료된 데이터셋은 skip(재실행 방지).  
`!python` 매직은 IPython 전용이라 스레드 내에서 동작하지 않으므로 의도적으로 `subprocess.run` 사용.

> **주의**: `--num_workers`(이미지 내부 병렬)와 `max_workers=3`(데이터셋 병렬)가 중첩되므로
> 일괄 실행에서는 `--num_workers 1`로 고정한다.

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
    done = Path(f"{AROMA_OUT}/profiling/{ds}/morphology_features.csv").exists()
    print(f"{'✓ 완료(skip)' if done else '○ 미완료':<14} {ds}")
```

**셀 2 — 병렬 실행:**

```python
import os, json, sys, subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

AROMA_OUT      = os.environ['AROMA_OUT']
AROMA_REF      = os.environ.get('AROMA_REF', '/content/AROMA')
DATASET_CONFIG = os.environ.get('DATASET_CONFIG', '/content/AROMA/dataset_config.json')
SCRIPT         = f"{AROMA_REF}/scripts/distribution_profiling.py"
MAX_WORKERS    = 3

with open(DATASET_CONFIG) as f:
    cfg = json.load(f)
datasets = [k for k in cfg if not k.startswith('_')]

def run_one(ds):
    if Path(f"{AROMA_OUT}/profiling/{ds}/morphology_features.csv").exists():
        return 'skip', ds, ''  # 이미 완료 — 재실행 방지
    cmd = [sys.executable, SCRIPT,
           '--dataset_config', DATASET_CONFIG,
           '--dataset_key',    ds,
           '--output_dir',     f"{AROMA_OUT}/profiling/{ds}",
           '--num_workers',    '1']  # 데이터셋 병렬 중첩으로 내부 workers 1 고정
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
