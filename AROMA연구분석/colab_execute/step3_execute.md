# Step 3 — `roi_selection.py` Colab 실행

> **실행 환경**: CPU
> Phase 0 + Step 2 출력을 읽어 결함 이미지 × 컨텍스트 빈 후보를 스코어링하고 ROI 목록 선별.
> Step 2 완료 후 실행.

## 환경변수

```python
import os

os.environ['AROMA_SCRIPTS']  = '/content/AROMA/scripts/aroma'
os.environ['AROMA_OUT']      = '/content/drive/MyDrive/data/Aroma/aroma_output'

DATASET_KEY = 'isp_LSM_1'   # ← 변경 시 여기만 수정

os.environ['DATASET_KEY']    = DATASET_KEY
os.environ['PROFILING_DIR']  = f"{os.environ['AROMA_OUT']}/profiling/{DATASET_KEY}"
os.environ['PROMPTS_DIR']    = f"{os.environ['AROMA_OUT']}/prompts/{DATASET_KEY}"
os.environ['ROI_DIR']        = f"{os.environ['AROMA_OUT']}/roi/{DATASET_KEY}"
```

## 실행

```python
# deficit_aware: 희귀 조합(Deficit ≥ p75) 우선 선택 후 나머지로 채움
!python $AROMA_SCRIPTS/roi_selection.py \
    --profiling_dir     $PROFILING_DIR \
    --prompts_dir       $PROMPTS_DIR \
    --sampling_strategy deficit_aware \
    --top_k             200 \
    --output_dir        $ROI_DIR
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
