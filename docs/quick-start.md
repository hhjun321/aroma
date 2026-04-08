# AROMA Quick Start Guide

이 가이드는 AROMA 파이프라인을 처음 실행하는 사용자를 위한 단계별 실행 가이드입니다.

---

## 목차

1. [환경 설정](#1-환경-설정)
2. [데이터 준비](#2-데이터-준비)
3. [설정 파일 작성](#3-설정-파일-작성)
4. [Stage별 실행](#4-stage별-실행)
5. [결과 확인](#5-결과-확인)
6. [문제 해결](#6-문제-해결)

---

## 1. 환경 설정

### Google Colab (권장)

```python
# 1. Google Drive 마운트
from google.colab import drive
drive.mount('/content/drive')

# 2. 작업 디렉터리로 이동
%cd /content/drive/MyDrive/aroma

# 3. 필요한 패키지 설치
!pip install -r requirements.txt

# 4. GPU 사용 가능 확인
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
```

### 로컬 환경

```bash
# 1. Python 가상환경 생성
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 2. 패키지 설치
pip install -r requirements.txt

# 3. GPU 확인
python -c "import torch; print(torch.cuda.is_available())"
```

---

## 2. 데이터 준비

### 2.1 디렉터리 구조

AROMA는 다음과 같은 디렉터리 구조를 기대합니다:

```
/content/drive/MyDrive/data/Aroma/
├── mvtec/
│   ├── bottle/
│   │   ├── train/
│   │   │   └── good/              # 정상 이미지 (209장)
│   │   │       ├── 000.png
│   │   │       ├── 001.png
│   │   │       └── ...
│   │   └── test/
│   │       ├── good/              # 테스트용 정상 이미지
│   │       └── broken_large/      # Defect seed 이미지
│   │           ├── 000.png
│   │           └── ...
│   ├── cable/
│   └── ...
├── isp/
│   └── unsupervised/
│       ├── ASM/
│       ├── LSM_1/
│       └── ...
└── visa/
    ├── candle/
    └── ...
```

### 2.2 최소 데이터 요구사항

단일 카테고리 실행을 위한 최소 요구사항:

- **정상 이미지**: 50장 이상 (`train/good/`)
- **Defect seed**: 1장 이상 (`test/{defect_type}/`)
- **이미지 형식**: PNG 또는 JPG
- **권장 크기**: 256×256 이상 (Stage 0에서 자동 리사이즈)

---

## 3. 설정 파일 작성

### 3.1 dataset_config.json

프로젝트 루트에 `dataset_config.json` 생성:

```json
{
  "_comment": "AROMA dataset config",
  "mvtec_bottle": {
    "domain": "mvtec",
    "image_dir": "/content/drive/MyDrive/data/Aroma/mvtec/bottle/train/good",
    "seed_dir": "/content/drive/MyDrive/data/Aroma/mvtec/bottle/test/broken_large",
    "notes": "MVTec bottle category - 209 good images"
  }
}
```

**필드 설명:**
- `domain`: 도메인 타입 (`isp`, `mvtec`, `visa`)
- `image_dir`: 정상 이미지 디렉터리 (절대 경로)
- `seed_dir`: Defect seed 디렉터리 (절대 경로)
- `notes`: 메모 (선택사항)

### 3.2 benchmark_experiment.yaml (선택)

고급 설정이 필요한 경우 `configs/benchmark_experiment.yaml` 수정:

```yaml
# 도메인별 증강 비율 설정
augmentation_ratio_by_domain:
  mvtec:
    full: 2.0      # 정상 이미지 대비 합성 defect 비율
    pruned: 1.5    # 품질 필터링 후 비율

# 카테고리 필터 (선택)
category_filter:
  exclude:
    - "mvtec_zipper"  # 제외할 카테고리
```

---

## 4. Stage별 실행

### Stage 0: 이미지 리사이즈

모든 이미지를 256×256으로 정규화합니다.

```bash
python stage0_resize.py \
  --config dataset_config.json \
  --target_size 256 \
  --workers -1
```

**출력:**
- `{domain}/{category}/resized/good/*.png` - 리사이즈된 정상 이미지
- `{domain}/{category}/resized/test/{defect_type}/*.png` - 리사이즈된 defect seed

**소요 시간:** ~1-2분 (209장 기준)

---

### Stage 1: ROI 추출 및 배경 분석

이미지를 ROI 그리드로 분할하고 배경 질감을 분석합니다.

```bash
python stage1_roi_extraction.py \
  --image_dir /content/drive/MyDrive/data/Aroma/mvtec/bottle/train/good \
  --output_dir /content/drive/MyDrive/data/Aroma/mvtec/bottle/stage1_output \
  --grid_size 64 \
  --workers -1
```

**주요 파라미터:**
- `--grid_size`: ROI 셀 크기 (32, 64, 128 중 선택)
  - 작을수록: 더 세밀한 분석, 더 긴 처리 시간
  - 클수록: 빠른 처리, 덜 세밀한 분석
- `--workers`: 병렬 처리 워커 수 (-1 = 자동)

**출력:**
- `stage1_output/roi_metadata.json` - ROI 위치 및 배경 타입 정보
- `stage1_output/sentinel.json` - 완료 마커

**소요 시간:** ~5-10분 (209장, workers=-1)

**검증:**
```python
import json
with open('/content/drive/MyDrive/data/Aroma/mvtec/bottle/stage1_output/roi_metadata.json') as f:
    metadata = json.load(f)
print(f"Processed images: {len(metadata)}")
print(f"Sample entry: {metadata[0]}")
```

---

### Stage 1b: Defect Seed 특성 분석

Defect seed의 형태적 특성을 분석합니다.

```bash
python stage1b_seed_characterization.py \
  --seed_dir /content/drive/MyDrive/data/Aroma/mvtec/bottle/test/broken_large \
  --output_dir /content/drive/MyDrive/data/Aroma/mvtec/bottle/stage1b_output \
  --workers -1
```

**출력:**
- `stage1b_output/seed_profile.json` - Defect 타입 분류 결과
  ```json
  {
    "000.png": {
      "defect_subtype": "compact_blob",
      "linearity_score": 0.23,
      "aspect_ratio": 1.1
    }
  }
  ```

**소요 시간:** ~30초 (10-20개 seed 기준)

---

### Stage 2: Defect 변이체 생성

단일 seed로부터 다양한 변형을 생성합니다 (Elastic Warping).

```bash
python stage2_defect_seed_generation.py \
  --seed_profile /content/drive/MyDrive/data/Aroma/mvtec/bottle/stage1b_output/seed_profile.json \
  --output_dir /content/drive/MyDrive/data/Aroma/mvtec/bottle/stage2_output \
  --num_variants 100 \
  --workers -1
```

**주요 파라미터:**
- `--num_variants`: 시드당 생성할 변형 개수
  - 권장: 50-200 (많을수록 다양성 증가)

**출력:**
- `stage2_output/{seed_id}_var_{i}.png` - 변이체 이미지
- 예: `000_var_000.png`, `000_var_001.png`, ...

**소요 시간:** ~2-5분 (10 seeds × 100 variants)

**검증:**
```bash
ls /content/drive/MyDrive/data/Aroma/mvtec/bottle/stage2_output/*.png | wc -l
# 출력: 1000 (10 seeds × 100 variants)
```

---

### Stage 3: Context-Aware 배치 계획

배경 질감과 defect 형태의 매칭을 기반으로 최적 배치 위치를 계산합니다.

```bash
python stage3_layout_logic.py \
  --roi_metadata /content/drive/MyDrive/data/Aroma/mvtec/bottle/stage1_output/roi_metadata.json \
  --defect_seeds_dir /content/drive/MyDrive/data/Aroma/mvtec/bottle/stage2_output \
  --output_json /content/drive/MyDrive/data/Aroma/mvtec/bottle/stage3_output/placement_map.json \
  --workers -1
```

**출력:**
- `placement_map.json` - 이미지별 배치 계획
  ```json
  {
    "000": [
      {
        "defect_path": "stage2_output/000_var_005.png",
        "x": 120,
        "y": 80,
        "rotation": 45.0,
        "suitability_score": 0.89
      }
    ]
  }
  ```

**소요 시간:** ~3-5분 (GPU 사용 시 1-2분)

---

### Stage 4: MPB 합성

Modified Poisson Blending을 사용하여 defect를 배경에 합성합니다.

```bash
python stage4_mpb_synthesis.py \
  --image_dir /content/drive/MyDrive/data/Aroma/mvtec/bottle/train/good \
  --placement_map /content/drive/MyDrive/data/Aroma/mvtec/bottle/stage3_output/placement_map.json \
  --output_dir /content/drive/MyDrive/data/Aroma/mvtec/bottle/stage4_output \
  --workers -1 \
  --format cls
```

**주요 파라미터:**
- `--format`: 출력 형식
  - `cls`: Classification용 (이미지만 저장)
  - `seg`: Segmentation용 (이미지 + 마스크)

**출력:**
- `stage4_output/{seed_id}/{image_id}.png` - 합성된 이미지
- `stage4_output/{seed_id}/sentinel.json` - 완료 마커

**소요 시간:** ~10-20분 (209 images × 10 seeds)

**검증:**
```python
from PIL import Image
img = Image.open('/content/drive/MyDrive/data/Aroma/mvtec/bottle/stage4_output/000/000.png')
img.show()  # 합성 결과 확인
```

---

### Stage 5: 품질 점수 계산

합성 이미지의 품질을 평가합니다 (아티팩트, 블러).

```bash
python stage5_quality_scoring.py \
  --stage4_seed_dir /content/drive/MyDrive/data/Aroma/mvtec/bottle/stage4_output \
  --workers -1
```

**출력:**
- `stage4_output/{seed_id}/quality_scores.json` - 품질 점수
  ```json
  {
    "000.png": {
      "artifact_score": 0.12,
      "blur_score": 0.85,
      "final_score": 0.485
    }
  }
  ```

**소요 시간:** ~2-3분 (2000+ 이미지 기준)

**점수 의미:**
- `artifact_score`: 낮을수록 좋음 (경계 아티팩트 적음)
- `blur_score`: 높을수록 좋음 (선명함)
- `final_score`: 두 점수의 가중 평균

---

### Stage 6: 데이터셋 구성

품질 필터링을 적용하여 최종 데이터셋을 구성합니다.

```bash
python stage6_dataset_builder.py \
  --cat_dir /content/drive/MyDrive/data/Aroma/mvtec/bottle \
  --image_dir /content/drive/MyDrive/data/Aroma/mvtec/bottle/train/good \
  --seed_dirs /content/drive/MyDrive/data/Aroma/mvtec/bottle/test/broken_large \
  --config /content/drive/MyDrive/aroma/configs/benchmark_experiment.yaml \
  --pruning_threshold 0.6 \
  --workers -1
```

**주요 파라미터:**
- `--pruning_threshold`: 품질 필터 임계값 (0.0-1.0)
  - 높을수록: 엄격한 필터링 (적은 이미지, 높은 품질)
  - 낮을수록: 느슨한 필터링 (많은 이미지, 낮은 품질)
- `--config`: 도메인별 증강 비율 설정 파일 (선택)

**출력:**
```
mvtec/bottle/augmented_dataset/
├── baseline/
│   └── good/           # 원본 정상 이미지 (209장)
├── aroma_full/
│   ├── good/           # 원본 정상 이미지 (심볼릭 링크)
│   └── broken_large/   # 모든 합성 defect
├── aroma_pruned/
│   ├── good/           # 원본 정상 이미지 (심볼릭 링크)
│   └── broken_large/   # 품질 필터링된 defect
└── build_report.json   # 구성 보고서
```

**소요 시간:** ~1-2분 (복사 작업)

**검증:**
```python
import json
with open('/content/drive/MyDrive/data/Aroma/mvtec/bottle/augmented_dataset/build_report.json') as f:
    report = json.load(f)

print(f"Baseline good: {report['baseline']['good_count']}")
print(f"AROMA full defects: {report['aroma_full']['defect_count']}")
print(f"AROMA pruned defects: {report['aroma_pruned']['defect_count']}")
print(f"Applied ratio (full): {report.get('ratio_full', 'N/A')}")
```

**예상 출력:**
```
Baseline good: 209
AROMA full defects: 418  (ratio=2.0)
AROMA pruned defects: 313  (ratio=1.5)
Applied ratio (full): 2.0
```

---

### Stage 7: 벤치마크 (선택)

YOLO11 또는 DRAEM 모델을 학습하고 평가합니다.

```bash
python stage7_benchmark.py \
  --config /content/drive/MyDrive/aroma/configs/benchmark_experiment.yaml \
  --model yolo11 \
  --tasks mvtec_bottle \
  --device cuda \
  --epochs 100
```

**주요 파라미터:**
- `--model`: 모델 타입 (`yolo11`, `rtdetr`, `draem`)
- `--tasks`: 실행할 카테고리 (공백으로 구분)
- `--epochs`: 학습 에포크 수

**출력:**
- `results/{model}/{timestamp}/` - 학습 결과
- `results/{model}/{timestamp}/metrics.json` - 평가 지표

**소요 시간:** ~30분-2시간 (모델, 데이터셋 크기에 따라)

---

## 5. 결과 확인

### 5.1 품질 검증

생성된 합성 이미지를 시각적으로 확인:

```python
import matplotlib.pyplot as plt
from PIL import Image
import random

# Stage 4 출력 디렉터리
output_dir = '/content/drive/MyDrive/data/Aroma/mvtec/bottle/stage4_output/000'

# 랜덤 샘플 5개 선택
images = list(Path(output_dir).glob('*.png'))
samples = random.sample(images, min(5, len(images)))

# 시각화
fig, axes = plt.subplots(1, 5, figsize=(20, 4))
for ax, img_path in zip(axes, samples):
    img = Image.open(img_path)
    ax.imshow(img)
    ax.set_title(img_path.stem)
    ax.axis('off')
plt.show()
```

### 5.2 데이터셋 통계

```python
import json
from pathlib import Path

report_path = Path('/content/drive/MyDrive/data/Aroma/mvtec/bottle/augmented_dataset/build_report.json')
with open(report_path) as f:
    report = json.load(f)

print("=" * 50)
print("Dataset Build Report")
print("=" * 50)
print(f"Category: {report.get('domain', 'N/A')}")
print(f"Pruning threshold: {report.get('pruning_threshold', 'N/A')}")
print(f"\nBaseline:")
print(f"  Good: {report['baseline']['good_count']}")
print(f"\nAROMA Full:")
print(f"  Good: {report['aroma_full']['good_count']}")
print(f"  Defects: {report['aroma_full']['defect_count']}")
print(f"  Ratio: {report['aroma_full']['defect_count'] / report['aroma_full']['good_count']:.2f}:1")
print(f"\nAROMA Pruned:")
print(f"  Good: {report['aroma_pruned']['good_count']}")
print(f"  Defects: {report['aroma_pruned']['defect_count']}")
print(f"  Ratio: {report['aroma_pruned']['defect_count'] / report['aroma_pruned']['good_count']:.2f}:1")
print(f"  Pruned: {report['aroma_pruned']['defect_pruned']} images")
```

---

## 6. 문제 해결

### 6.1 일반적인 문제

#### ❌ `FileNotFoundError: Cannot read image`

**원인:** 이미지 경로가 잘못되었거나 파일이 없음

**해결:**
```bash
# 경로 확인
ls -la /content/drive/MyDrive/data/Aroma/mvtec/bottle/train/good

# 파일 개수 확인
ls /content/drive/MyDrive/data/Aroma/mvtec/bottle/train/good/*.png | wc -l
```

#### ❌ `ValueError: roi_metadata.json missing required fields`

**원인:** Stage 1 출력이 손상되었거나 불완전함

**해결:**
```bash
# Stage 1 재실행 (sentinel 삭제)
rm /content/drive/MyDrive/data/Aroma/mvtec/bottle/stage1_output/sentinel.json
python stage1_roi_extraction.py ...
```

#### ❌ `CUDA out of memory`

**원인:** GPU 메모리 부족

**해결:**
```python
# Stage 3, 7에서 배치 크기 줄이기
# 또는 CPU 모드로 전환
python stage3_layout_logic.py --device cpu ...
python stage7_benchmark.py --device cpu ...
```

### 6.2 성능 최적화

#### 느린 처리 속도

**진단:**
```bash
# 워커 수 확인
python stage4_mpb_synthesis.py --workers -1  # 자동 감지
python stage4_mpb_synthesis.py --workers 4   # 명시적 지정
```

**최적화:**
- Google Colab: `--workers=2` (무료 티어 제한)
- 로컬 (8코어): `--workers=-1` (자동)
- 메모리 제약 시: `--workers=0` (순차 실행)

#### 디스크 공간 부족

**진단:**
```bash
df -h /content/drive/MyDrive
```

**해결:**
```bash
# 임시 파일 정리
rm -rf /content/drive/MyDrive/data/Aroma/*/stage4_output/*/sentinel.json

# 중간 결과 삭제 (필요 시)
rm -rf /content/drive/MyDrive/data/Aroma/*/stage2_output
```

### 6.3 디버깅 모드

순차 실행으로 에러 추적:

```bash
# workers=0으로 순차 실행
python stage4_mpb_synthesis.py \
  --workers 0 \
  --image_dir ... \
  --placement_map ... \
  --output_dir ...
```

상세 로그 활성화:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# 그 후 stage 스크립트 실행
```

---

## 다음 단계

- **전체 파이프라인 자동화**: `docs/작업일지/smoke_test_3cat.md` 참조
- **설정 튜닝**: [docs/configuration.md](configuration.md) 참조
- **벤치마크 실험**: Stage 7 상세 가이드

---

**참고 자료:**
- [README.md](../README.md) - 프로젝트 개요
- [configuration.md](configuration.md) - 설정 파일 가이드
- `docs/작업일지/` - 개발 로그 및 실험 노트
