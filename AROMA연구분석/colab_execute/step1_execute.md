# Step 1 — `compute_complexity.py` Colab 실행

> **실행 환경**: CPU
> Phase 0 출력을 읽어 MCI / CCI 계산 + 정책 선택.
> Phase 0 완료 후 실행.

## 환경변수

```python
import os

os.environ['AROMA_SCRIPTS']  = '/content/AROMA/scripts/aroma'
os.environ['AROMA_OUT']      = '/content/drive/MyDrive/data/Aroma/aroma_output'

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
