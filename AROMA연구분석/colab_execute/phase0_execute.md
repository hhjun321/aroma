# Phase 0 — `distribution_profiling.py` Colab 실행

> ISP 도메인: GT mask 없음 → Otsu fallback (SAM 미지정 시 자동)

## 환경변수

```python
import os

os.environ['AROMA_REF']      = '/content/AROMA'
os.environ['DATASET_CONFIG'] = '/content/AROMA/dataset_config.json'
os.environ['AROMA_OUT']      = '/content/drive/MyDrive/data/Aroma/aroma_output'

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
