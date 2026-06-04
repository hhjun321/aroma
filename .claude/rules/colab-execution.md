# Colab 실행 규칙

CASDA 프로젝트의 모든 스크립트는 Google Colab Pro 환경에서 실행된다.
스크립트 실행 명령 작성 시 반드시 아래 규칙을 따른다.

## 환경변수 참조 형식

- **올바름**: `$VAR`
- **틀림**: `${VAR}` (Colab bash 셀에서 일부 오작동)

## 스크립트 실행 형식

Colab 노트북 셀에서는 `!python` 접두사 사용:

```python
# 올바름
!python $SCRIPTS/analyze_dataset.py \
    --image_dir $TRAIN_IMAGES \
    --train_csv $TRAIN_CSV \
    --output_dir $DRIVE/analysis

# 틀림
python ${SCRIPTS}/analyze_dataset.py \
    --image_dir ${TRAIN_IMAGES}
```

## 프로젝트 환경변수 목록

| 변수 | 경로 |
|------|------|
| `SCRIPTS` | `/content/CASDA/scripts` |
| `CONFIG` | `/content/CASDA/configs/benchmark_experiment.yaml` |
| `DRIVE` | `/content/drive/MyDrive/data/Severstal` |
| `TRAIN_IMAGES` | `$DRIVE/train_images` |
| `TRAIN_CSV` | `$DRIVE/train.csv` |
| `AUG_DATASET` | `$DRIVE/augmented_dataset_v5.6` |
| `AUG_IMAGES` | `$DRIVE/augmented_images_v5.5` |
| `ROI_DIR` | `$DRIVE/roi_patches_v5.1` |
| `YOLO_DATASETS` | `$DRIVE/yolo_datasets` |
| `BENCHMARK_RESULTS` | `$DRIVE/benchmark_results` |
| `FID_RESULTS` | `$DRIVE/fid_results` |
| `LOCAL_IMAGES` | `/content/dataset_local/train_images` |

`ANALYSIS_DIR`, `ANALYSIS_CONFIG`는 기본 설정에 없음 → 별도 셀에서 추가:
```python
import os
os.environ['ANALYSIS_DIR']    = f"{os.environ['DRIVE']}/analysis"
os.environ['ANALYSIS_CONFIG'] = f"{os.environ['DRIVE']}/analysis/recommended_config.yaml"
```

`ANALYSIS_CONFIG`는 Stage 0(`analyze_dataset.py`) 완료 후 생성되는 파일. Stage A 이후 스크립트의 `--config` 인자로 전달.

## 저장소 클론

```python
!git clone --single-branch -b approve https://github.com/hhjun321/CASDA.git /content/CASDA
```
