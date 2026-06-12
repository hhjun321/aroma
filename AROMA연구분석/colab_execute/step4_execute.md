# Step 4 — `generate_defects.py` Colab 실행

> **실행 환경**: CPU (`copy_paste` 기본값) / GPU (`controlnet`, `inpainting` — 미구현 stub)
> Step 3 ROI 목록을 읽어 결함 이미지를 정상 배경 위에 합성.
> Step 3 완료 후 실행.

## 환경변수

```python
import os

os.environ['AROMA_SCRIPTS']   = '/content/AROMA/scripts/aroma'
os.environ['AROMA_OUT']       = '/content/drive/MyDrive/data/Aroma/aroma_output'
os.environ['AROMA_DATA_BASE'] = '/content/drive/MyDrive/data/Aroma'

DATASET_KEY = 'isp_LSM_1'   # ← 변경 시 여기만 수정

os.environ['DATASET_KEY']     = DATASET_KEY
os.environ['ROI_DIR']         = f"{os.environ['AROMA_OUT']}/roi/{DATASET_KEY}"
os.environ['SYNTHETIC_DIR']   = f"{os.environ['AROMA_OUT']}/synthetic/{DATASET_KEY}"

# 정상(good) 이미지 경로 — 데이터셋별 조정
# isp_LSM_1  → ISP/LSM_1/train/good
# mvtec_cable → MVTec/cable/train/good
# visa_cashew → VisA/cashew/train/good
os.environ['NORMAL_DIR'] = f"{os.environ['AROMA_DATA_BASE']}/isp/unsupervised/LSM_1/train/good"
```

## 실행

```python
# copy_paste: CPU, --n_per_roi ROI당 합성 이미지 수
!python $AROMA_SCRIPTS/generate_defects.py \
    --roi_dir    $ROI_DIR \
    --normal_dir $NORMAL_DIR \
    --output_dir $SYNTHETIC_DIR \
    --method     copy_paste \
    --n_per_roi  3 \
    --feather_px 4 \
    --seed       42
```

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
