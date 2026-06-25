# Step 3 — `roi_selection.py` Colab 실행

> **실행 환경**: CPU
> Phase 0 + Step 2 출력을 읽어 결함 이미지 × 컨텍스트 빈 후보를 스코어링하고 ROI 목록 선별.
> Step 2 완료 후 실행.

## 환경변수

```python
import os

os.environ['AROMA_SCRIPTS']  = '/content/AROMA/scripts/aroma'
os.environ['AROMA_OUT']      = f"{os.environ['DRIVE']}/aroma_output"

DATASET_KEY = 'isp_LSM_1'   # ← 변경 시 여기만 수정

os.environ['DATASET_KEY']    = DATASET_KEY
os.environ['PROFILING_DIR']  = f"{os.environ['AROMA_OUT']}/profiling/{DATASET_KEY}"
os.environ['PROMPTS_DIR']    = f"{os.environ['AROMA_OUT']}/prompts/{DATASET_KEY}"
os.environ['ROI_DIR']        = f"{os.environ['AROMA_OUT']}/roi/{DATASET_KEY}"
```

## 실행

```python
# 멀티클래스 게이트(제너릭): dataset_config.json의 class_mode == 'multi'인
# 데이터셋만 stratified-allocation 플래그를 받는다. single-class 데이터셋은
# 아무것도 전달하지 않아 roi_selected.json이 기존과 byte-identical.
import os, json

DATASET_CONFIG = os.environ.get('DATASET_CONFIG', '/content/AROMA/dataset_config.json')
with open(DATASET_CONFIG) as f:
    _cfg = json.load(f)

if _cfg.get(DATASET_KEY, {}).get('class_mode') == 'multi':
    os.environ['MULTI_FLAGS'] = '--class_mode multi --class_floor --per_pair_cap_frac 0.05'
else:
    os.environ['MULTI_FLAGS'] = ''
print('MULTI_FLAGS:', repr(os.environ['MULTI_FLAGS']))
```

```python
# deficit_aware: 희귀 조합(Deficit ≥ p75) 우선 선택 후 나머지로 채움
# $MULTI_FLAGS 는 multi 데이터셋에서만 채워지고, 그 외에는 빈 문자열이라 무영향.
#
# --img_diversity_cap 1 (Fix4): 동일한 소스 결함 crop((image_path, defect_bbox))을
#   최대 1번만 선택 → AROMA 선택이 소수의 crop을 수십 번 반복하는 다양성 붕괴
#   (distinct (image,bbox) 88 vs CASDA 1692)를 제거한다. distinct source 수가
#   top_k 보다 적은 클래스에 한해서만 bounded repetition 을 허용하고 로그를 남긴다.
#   default=1 (생략해도 cap=1 적용). legacy 무제한 복원은 999 같은 큰 값 전달.
#   deficit_aware allocation 에만 적용(random/weighted/top_k 는 무영향).
!python $AROMA_SCRIPTS/roi_selection.py \
    --profiling_dir     $PROFILING_DIR \
    --prompts_dir       $PROMPTS_DIR \
    --sampling_strategy deficit_aware \
    --top_k             200 \
    --img_diversity_cap 1 \
    --output_dir        $ROI_DIR \
    $MULTI_FLAGS
```

### 전략 옵션

| `--sampling_strategy` | 설명 |
|----------------------|------|
| `deficit_aware` (기본) | Deficit ≥ 75th percentile 우선 선택, 나머지로 충원 |
| `top_k` | roi_score 내림차순 상위 K개 |
| `weighted` | roi_score 비례 확률 가중 추출 |

## 결과 확인

```python
import json, os

with open(f"{os.environ['ROI_DIR']}/roi_selected.json") as f:
    selected = json.load(f)

with open(f"{os.environ['ROI_DIR']}/roi_candidates.json") as f:
    candidates = json.load(f)

print(f"전체 후보: {len(candidates)}  선택된 ROI: {len(selected)}")
print()

# 클러스터별 분포
from collections import Counter
cluster_dist = Counter(r['cluster_id'] for r in selected)
print("클러스터별 선택 수:")
for cid, cnt in sorted(cluster_dist.items()):
    print(f"  cluster {cid}: {cnt}개")

print()
# 상위 deficit ROI
top_deficit = sorted(selected, key=lambda x: x.get('deficit', 0), reverse=True)[:5]
print("Deficit 상위 5개:")
for r in top_deficit:
    print(f"  [{r['cluster_id']}|{r['cell_key']}] score={r['roi_score']:.3f}  deficit={r['deficit']:.3f}")
    print(f"    {r['prompt']}")
```

## 출력 파일

| 파일 | 내용 |
|------|------|
| `roi_candidates.json` | 전체 스코어링 결과 (image_id, cluster_id, cell_key, roi_score, deficit, prompt) |
| `roi_selected.json` | 선택된 top_k개 ROI 목록 |
| `roi_summary.md` | 마크다운 테이블 |


---

## 전체 데이터셋 일괄 실행 (병렬)

`dataset_config.json`에서 datasets 목록 자동 로드, step2 출력(`prompts/{ds}/prompts.json`)이 없는 데이터셋은 skip.  
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
    ready = Path(f"{AROMA_OUT}/prompts/{ds}/prompts.json").exists()
    print(f"{'✓ 준비됨' if ready else '↷ skip(step2 없음)':<22} {ds}")
```

**셀 2 — 병렬 실행:**

```python
import os, json, sys, subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

AROMA_OUT      = os.environ['AROMA_OUT']
AROMA_SCRIPTS  = os.environ['AROMA_SCRIPTS']
DATASET_CONFIG = os.environ.get('DATASET_CONFIG', '/content/AROMA/dataset_config.json')
SCRIPT         = f"{AROMA_SCRIPTS}/roi_selection.py"
MAX_WORKERS    = 3

with open(DATASET_CONFIG) as f:
    cfg = json.load(f)
datasets = [k for k in cfg if not k.startswith('_')]

def run_one(ds):
    if not Path(f"{AROMA_OUT}/prompts/{ds}/prompts.json").exists():
        return 'skip', ds, 'step2 출력 없음'
    cmd = [sys.executable, SCRIPT,
           '--profiling_dir',     f"{AROMA_OUT}/profiling/{ds}",
           '--prompts_dir',       f"{AROMA_OUT}/prompts/{ds}",
           '--sampling_strategy', 'deficit_aware',
           '--top_k',             '200',
           # Fix4: cap each source defect crop at 1 selection so AROMA draws
           # DISTINCT defects (removes the distinct-(image,bbox) diversity
           # collapse confound). deficit_aware-only; None ⇒ byte-identical.
           '--img_diversity_cap', '1',
           '--output_dir',        f"{AROMA_OUT}/roi/{ds}"]
    # Multi-class gate (generic): only datasets that declare class_mode=='multi'
    # in dataset_config.json get the stratified-allocation flags. Single-class
    # datasets pass nothing → roi_selected.json byte-identical to before.
    if cfg.get(ds, {}).get('class_mode') == 'multi':
        cmd += ['--class_mode', 'multi', '--class_floor', '--per_pair_cap_frac', '0.05']
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
