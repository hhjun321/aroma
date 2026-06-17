# Step 4 — `generate_defects.py` Colab 실행

> **실행 환경**: CPU (`copy_paste` 기본값) / GPU (`controlnet`, `inpainting` — 미구현 stub)
> Step 3 ROI 목록을 읽어 결함 이미지를 정상 배경 위에 합성.
> Step 3 완료 후 실행.

## 환경변수

```python
import json, os

os.environ['AROMA_SCRIPTS']    = '/content/AROMA/scripts/aroma'
os.environ['AROMA_OUT']      = f"{os.environ['DRIVE']}/aroma_output"
os.environ['DATASET_CONFIG']   = os.environ.get('DATASET_CONFIG', '/content/AROMA/dataset_config.json')

DATASET_KEY = 'isp_LSM_1'   # ← 변경 시 여기만 수정

os.environ['DATASET_KEY']      = DATASET_KEY
os.environ['ROI_DIR']          = f"{os.environ['AROMA_OUT']}/roi/{DATASET_KEY}"
os.environ['SYNTHETIC_DIR']    = f"{os.environ['AROMA_OUT']}/synthetic/{DATASET_KEY}"

# NORMAL_DIR: dataset_config.json의 image_dir에서 자동 조회
with open(os.environ['DATASET_CONFIG']) as f:
    _cfg = json.load(f)
os.environ['NORMAL_DIR'] = _cfg[DATASET_KEY]['image_dir']
print(f"NORMAL_DIR = {os.environ['NORMAL_DIR']}")
```

## 실행

```python
# copy_paste: CPU, --local_staging으로 Drive I/O 최소화 (권장)
!python $AROMA_SCRIPTS/generate_defects.py \
    --roi_dir       $ROI_DIR \
    --normal_dir    $NORMAL_DIR \
    --output_dir    $SYNTHETIC_DIR \
    --method        copy_paste \
    --n_per_roi     3 \
    --feather_px    4 \
    --seed          42 \
    --local_staging
```

`--local_staging` 동작:
1. defect 이미지 + normal 이미지 → `/content/tmp/aroma_step4_{dataset}/` 복사
2. 합성 전 과정 로컬 디스크에서 실행 (Drive I/O 없음)
3. 완료 후 이미지만 Drive `$SYNTHETIC_DIR/images/` 로 일괄 push
4. `annotations.json`은 Drive 경로 기준으로 저장

### 합성 방법 옵션

| `--method` | 상태 | 실행 환경 |
|-----------|------|---------|
| `copy_paste` (기본) | 구현 완료 | CPU |
| `controlnet` | stub (미구현) | GPU 필요 |
| `inpainting` | stub (미구현) | GPU 필요 |

## 결과 확인

```python
import json, os
from pathlib import Path

with open(f"{os.environ['SYNTHETIC_DIR']}/annotations.json") as f:
    annotations = json.load(f)

n_total = len(annotations)
n_dry   = sum(1 for a in annotations if a.get('dry_run'))
n_real  = n_total - n_dry

print(f"생성 완료: {n_real}개  (dry_run: {n_dry}개)")

# 클러스터별 분포
from collections import Counter
dist = Counter(a['cluster_id'] for a in annotations if not a.get('dry_run'))
print("\n클러스터별 생성 수:")
for cid, cnt in sorted(dist.items()):
    print(f"  cluster {cid}: {cnt}개")

# 이미지 파일 수 확인
img_dir = Path(os.environ['SYNTHETIC_DIR']) / 'images'
if img_dir.exists():
    n_files = len(list(img_dir.glob('*.jpg')) + list(img_dir.glob('*.png')))
    print(f"\nimages/ 파일 수: {n_files}")
```

## 출력 파일

| 파일 | 내용 |
|------|------|
| `images/` | 합성 결함 이미지 (JPEG) |
| `annotations.json` | 이미지별 메타 (source_roi, cluster_id, prompt, roi_score, deficit, method) |


---

## 전체 데이터셋 일괄 실행 (병렬)

`dataset_config.json`에서 datasets 목록 자동 로드, step3 출력(`roi/{ds}/roi_selected.json`)이 없는 데이터셋은 skip.  
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
    ready = Path(f"{AROMA_OUT}/roi/{ds}/roi_selected.json").exists()
    print(f"{'✓ 준비됨' if ready else '↷ skip(step3 없음)':<22} {ds}")
```

**셀 2 — 병렬 실행:**

```python
import os, json, sys, subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

AROMA_OUT      = os.environ['AROMA_OUT']
AROMA_SCRIPTS  = os.environ['AROMA_SCRIPTS']
DATASET_CONFIG = os.environ.get('DATASET_CONFIG', '/content/AROMA/dataset_config.json')
SCRIPT         = f"{AROMA_SCRIPTS}/generate_defects.py"
MAX_WORKERS    = 3

with open(DATASET_CONFIG) as f:
    cfg = json.load(f)
datasets = [k for k in cfg if not k.startswith('_')]

def run_one(ds):
    if not Path(f"{AROMA_OUT}/roi/{ds}/roi_selected.json").exists():
        return 'skip', ds, 'step3 출력 없음'
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
