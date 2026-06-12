# Step 2 — `prompt_generation.py` Colab 실행

> **실행 환경**: CPU
> Phase 0 + Step 1 출력을 읽어 형태학 클러스터 × 컨텍스트 빈 조합별 자연어 프롬프트 생성.
> Step 1 완료 후 실행.

## 환경변수

```python
import os

os.environ['AROMA_SCRIPTS']  = '/content/AROMA/scripts/aroma'
os.environ['AROMA_OUT']      = '/content/drive/MyDrive/data/Aroma/aroma_output'

DATASET_KEY = 'isp_LSM_1'   # ← 변경 시 여기만 수정

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
