# AROMA Configuration Guide

이 문서는 AROMA 프로젝트의 설정 파일을 설명합니다.

---

## 목차

1. [dataset_config.json](#1-dataset_configjson)
2. [benchmark_experiment.yaml](#2-benchmark_experimentyaml)
3. [Stage별 CLI 파라미터](#3-stage별-cli-파라미터)
4. [파라미터 튜닝 가이드](#4-파라미터-튜닝-가이드)

---

## 1. dataset_config.json

### 1.1 개요

각 카테고리의 데이터 경로를 정의하는 JSON 파일입니다. Stage 0-6에서 사용됩니다.

**위치:** `dataset_config.json` (프로젝트 루트)

### 1.2 스키마

```json
{
  "_comment": "설명 (선택사항)",
  "<category_key>": {
    "domain": "<domain_type>",
    "image_dir": "<absolute_path_to_train_good>",
    "seed_dir": "<absolute_path_to_test_defect>",
    "notes": "<메모 (선택사항)>"
  }
}
```

**필드 설명:**

| 필드 | 타입 | 필수 | 설명 |
|------|------|------|------|
| `<category_key>` | string | ✓ | 카테고리 식별자 (예: `mvtec_bottle`) |
| `domain` | string | ✓ | 도메인 타입: `isp`, `mvtec`, `visa` |
| `image_dir` | string | ✓ | 정상 이미지 디렉터리 (절대 경로) |
| `seed_dir` | string | ✓ | Defect seed 디렉터리 (절대 경로) |
| `notes` | string | - | 메모 (무시됨) |

### 1.3 예제

```json
{
  "_comment": "AROMA dataset configuration for Google Colab",
  
  "isp_ASM": {
    "domain": "isp",
    "image_dir": "/content/drive/MyDrive/data/Aroma/isp/unsupervised/ASM/train/good",
    "seed_dir": "/content/drive/MyDrive/data/Aroma/isp/unsupervised/ASM/test/area",
    "notes": "256x256px, ISP screen-printed circuit board"
  },
  
  "mvtec_bottle": {
    "domain": "mvtec",
    "image_dir": "/content/drive/MyDrive/data/Aroma/mvtec/bottle/train/good",
    "seed_dir": "/content/drive/MyDrive/data/Aroma/mvtec/bottle/test/broken_large"
  },
  
  "visa_candle": {
    "domain": "visa",
    "image_dir": "/content/drive/MyDrive/data/Aroma/visa/candle/train/good",
    "seed_dir": "/content/drive/MyDrive/data/Aroma/visa/candle/test/bad"
  }
}
```

### 1.4 경로 규칙

**디렉터리 구조 예상:**

```
{base_path}/
├── {domain}/
│   ├── {category}/
│   │   ├── train/
│   │   │   └── good/          # image_dir이 가리킴
│   │   │       ├── 000.png
│   │   │       ├── 001.png
│   │   │       └── ...
│   │   └── test/
│   │       ├── good/          # (사용 안 함)
│   │       └── {defect_type}/ # seed_dir이 가리킴
│   │           ├── 000.png
│   │           └── ...
```

**주의사항:**
- 모든 경로는 **절대 경로** 사용 (Google Colab 환경)
- `image_dir`은 반드시 `train/good` 디렉터리를 가리켜야 함
- `seed_dir`은 `test/{defect_type}` 디렉터리를 가리킴

### 1.5 새 카테고리 추가

1. 데이터 준비 (최소 50개 good + 1개 defect seed)
2. `dataset_config.json`에 엔트리 추가:
   ```json
   "new_category": {
     "domain": "mvtec",
     "image_dir": "/path/to/new_category/train/good",
     "seed_dir": "/path/to/new_category/test/defect_type"
   }
   ```
3. Stage 0-6 실행:
   ```bash
   python stage0_resize.py --config dataset_config.json
   # ... (각 stage 실행)
   ```

---

## 2. benchmark_experiment.yaml

### 2.1 개요

Stage 6 데이터셋 구성 및 Stage 7 벤치마크 실험 설정을 정의합니다.

**위치:** `configs/benchmark_experiment.yaml`

### 2.2 주요 섹션

#### 2.2.1 `dataset` - 데이터셋 설정

```yaml
dataset:
  eval_batch_size: 32           # 평가 배치 크기
  image_size: 512               # 모델 입력 크기
  num_workers: 4                # DataLoader 워커 수
  pruning_threshold: 0.6        # 품질 필터 임계값
  
  # 도메인별 증강 비율
  augmentation_ratio_by_domain:
    isp:
      full: 1.0                 # good 대비 합성 defect 비율
      pruned: 0.5               # 품질 필터링 후 비율
    mvtec:
      full: 2.0
      pruned: 1.5
    visa:
      full: 2.0
      pruned: 1.5
  
  # Fallback 기본값
  augmentation_ratio_full: null      # null = 모든 defect 사용
  augmentation_ratio_pruned: null
  
  # 도메인별 Task 타입
  task_by_domain:
    isp: classification         # Image-level classification
    mvtec: segmentation         # Pixel-level segmentation
    visa: segmentation
  
  train_batch_size: 16          # 학습 배치 크기
```

**필드 설명:**

| 필드 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `pruning_threshold` | float | 0.6 | 품질 점수 임계값 (0.0-1.0) |
| `augmentation_ratio_by_domain` | dict | - | 도메인별 증강 비율 |
| `augmentation_ratio_full` | float/null | null | 전역 full 비율 (도메인 설정 무시) |
| `augmentation_ratio_pruned` | float/null | null | 전역 pruned 비율 |
| `image_size` | int | 512 | 모델 입력 이미지 크기 |
| `train_batch_size` | int | 16 | 학습 배치 크기 |
| `eval_batch_size` | int | 32 | 평가 배치 크기 |

**증강 비율 동작:**

1. **도메인별 비율 우선**: `augmentation_ratio_by_domain`에서 도메인 검색
2. **Fallback**: 도메인 매칭 실패 시 `augmentation_ratio_full/pruned` 사용
3. **None**: 비율이 None이면 모든 합성 이미지 사용

**예제:**
```yaml
augmentation_ratio_by_domain:
  mvtec:
    full: 2.0      # 209 good → 418 defect
    pruned: 1.5    # 209 good → 313 defect (품질 필터링)
```

---

#### 2.2.2 `dataset_groups` - 데이터셋 그룹

```yaml
dataset_groups:
  baseline:
    description: 원본 train/good만 (pretrained 특징 거리 평가)
  aroma_full:
    description: 원본 + 전체 Stage 4 합성
  aroma_pruned:
    description: 원본 + quality_score 필터링 합성
```

**그룹 설명:**

| 그룹 | Good | Defect | 용도 |
|------|------|--------|------|
| `baseline` | 원본 | 없음 | Zero-shot 베이스라인 |
| `aroma_full` | 원본 | 모든 합성 | 최대 데이터 증강 |
| `aroma_pruned` | 원본 | 필터링된 합성 | 품질 제어 증강 |

---

#### 2.2.3 `category_filter` - 카테고리 필터

```yaml
category_filter:
  exclude:
    isp:
      - LSM_2        # LSM_1과 동일 센서, 중복 제외
    mvtec: []        # 모든 카테고리 포함
    visa: []
```

**용도:** 벤치마크 실험에서 특정 카테고리 제외

---

#### 2.2.4 `models` - 모델 설정

```yaml
models:
  yolo11:
    model: yolo11n-cls.pt       # Pretrained 모델 파일
    epochs: 30                   # 학습 에포크
    lr: 0.01                     # Learning rate
    pretrained: true
  
  efficientdet_d0:
    backbone: efficientdet_d0
    epochs: 30
    lr: 0.0001
    optimizer: adam
    pretrained: true
```

**지원 모델:**
- `yolo11`: YOLO11 (classification/detection)
- `rtdetr`: RT-DETR (transformer-based detection)
- `draem`: DRAEM (reconstruction-based anomaly detection)
- `efficientdet_d0`: EfficientDet-D0

---

#### 2.2.5 `evaluation` - 평가 설정

```yaml
evaluation:
  metrics:
    - image_auroc       # Image-level AUROC
    - image_f1          # Image-level F1-score
    - pixel_auroc       # Pixel-level AUROC (segmentation만)
  
  pixel_auroc_domains:  # Pixel-level 평가 대상 도메인
    - mvtec
    - visa
```

---

### 2.3 전체 예제

```yaml
# 최소 설정 예제 (단일 도메인 실험)
dataset:
  pruning_threshold: 0.6
  augmentation_ratio_by_domain:
    mvtec:
      full: 2.0
      pruned: 1.5

dataset_groups:
  baseline:
    description: Baseline without augmentation
  aroma_pruned:
    description: With quality-filtered augmentation

models:
  yolo11:
    model: yolo11n-cls.pt
    epochs: 50
    lr: 0.01

evaluation:
  metrics:
    - image_auroc
    - image_f1
```

---

## 3. Stage별 CLI 파라미터

### Stage 0: Resize

```bash
python stage0_resize.py --config <path> --target_size <int> --workers <int>
```

| 파라미터 | 타입 | 기본값 | 설명 |
|---------|------|--------|------|
| `--config` | str | 필수 | dataset_config.json 경로 |
| `--target_size` | int | 256 | 출력 이미지 크기 |
| `--workers` | int | 0 | 병렬 워커 수 |

---

### Stage 1: ROI Extraction

```bash
python stage1_roi_extraction.py \
  --image_dir <path> \
  --output_dir <path> \
  --grid_size <int> \
  --workers <int>
```

| 파라미터 | 타입 | 기본값 | 설명 |
|---------|------|--------|------|
| `--image_dir` | str | 필수 | 정상 이미지 디렉터리 |
| `--output_dir` | str | 필수 | 출력 디렉터리 |
| `--grid_size` | int | 64 | ROI 셀 크기 (32, 64, 128) |
| `--workers` | int | 0 | 병렬 워커 수 |

**grid_size 선택 가이드:**
- **32**: 고해상도 분석, 긴 처리 시간 (512×512 이미지에 256개 ROI)
- **64**: 균형 (512×512 이미지에 64개 ROI) ✅ 권장
- **128**: 빠른 처리, 낮은 해상도 (512×512 이미지에 16개 ROI)

---

### Stage 2: Defect Generation

```bash
python stage2_defect_seed_generation.py \
  --seed_profile <path> \
  --output_dir <path> \
  --num_variants <int> \
  --workers <int>
```

| 파라미터 | 타입 | 기본값 | 설명 |
|---------|------|--------|------|
| `--seed_profile` | str | 필수 | Stage 1b 출력 JSON |
| `--output_dir` | str | 필수 | 출력 디렉터리 |
| `--num_variants` | int | 100 | 시드당 변이체 수 |
| `--workers` | int | 0 | 병렬 워커 수 |

**num_variants 가이드:**
- **50-100**: 작은 데이터셋 (good < 100)
- **100-200**: 중간 데이터셋 (good 100-500) ✅ 권장
- **200+**: 큰 데이터셋 (good > 500)

---

### Stage 4: MPB Synthesis

```bash
python stage4_mpb_synthesis.py \
  --image_dir <path> \
  --placement_map <path> \
  --output_dir <path> \
  --format <cls|seg> \
  --workers <int>
```

| 파라미터 | 타입 | 기본값 | 설명 |
|---------|------|--------|------|
| `--format` | str | cls | 출력 형식 (cls/seg) |
| `--workers` | int | 0 | 병렬 워커 수 |

**format 옵션:**
- `cls`: 이미지만 저장 (classification task)
- `seg`: 이미지 + 마스크 저장 (segmentation task)

---

### Stage 6: Dataset Builder

```bash
python stage6_dataset_builder.py \
  --cat_dir <path> \
  --image_dir <path> \
  --seed_dirs <path>... \
  --config <yaml_path> \
  --pruning_threshold <float> \
  --workers <int>
```

| 파라미터 | 타입 | 기본값 | 설명 |
|---------|------|--------|------|
| `--cat_dir` | str | 필수 | 카테고리 루트 디렉터리 |
| `--image_dir` | str | 필수 | 정상 이미지 디렉터리 |
| `--seed_dirs` | str+ | 필수 | Defect seed 디렉터리들 (공백 구분) |
| `--config` | str | - | benchmark_experiment.yaml 경로 |
| `--pruning_threshold` | float | 0.6 | 품질 필터 임계값 |
| `--augmentation_ratio_full` | float | None | Full 비율 (config 무시) |
| `--augmentation_ratio_pruned` | float | None | Pruned 비율 (config 무시) |
| `--workers` | int | 0 | 병렬 워커 수 |

**우선순위:**
1. CLI 인자 (`--augmentation_ratio_*`)
2. Config 파일의 도메인별 설정 (`augmentation_ratio_by_domain`)
3. Config 파일의 전역 설정 (`augmentation_ratio_full/pruned`)
4. 기본 동작 (모든 이미지 사용)

---

### Stage 7: Benchmark

```bash
python stage7_benchmark.py \
  --config <yaml_path> \
  --model <name> \
  --tasks <category>... \
  --device <cuda|cpu> \
  --epochs <int>
```

| 파라미터 | 타입 | 기본값 | 설명 |
|---------|------|--------|------|
| `--config` | str | 필수 | benchmark_experiment.yaml |
| `--model` | str | 필수 | 모델 이름 (yolo11, draem 등) |
| `--tasks` | str+ | 필수 | 실행할 카테고리들 |
| `--device` | str | cuda | 디바이스 (cuda/cpu) |
| `--epochs` | int | config | 학습 에포크 (config 무시) |

---

## 4. 파라미터 튜닝 가이드

### 4.1 증강 비율 튜닝

**목표:** 데이터 불균형 최소화

**현재 문제 (Stage 6 이전):**
```
mvtec_bottle: 209 good vs 4,180 defect (20:1)  ← 심각한 불균형
visa_candle:  900 good vs 9,000 defect (10:1)
```

**해결 (도메인별 비율 설정):**
```yaml
augmentation_ratio_by_domain:
  mvtec:
    full: 2.0      # 209 × 2.0 = 418 defect
    pruned: 1.5    # 209 × 1.5 = 313 defect
```

**튜닝 기준:**

| 비율 | 사용 사례 |
|------|----------|
| `0.5-1.0` | Classification (ISP) - 균형 유지 |
| `1.5-2.0` | Segmentation (MVTec/ViSA) - 픽셀 다양성 |
| `> 2.0` | 매우 적은 데이터 (good < 50) |

---

### 4.2 품질 필터 임계값

**pruning_threshold 효과:**

| 임계값 | 필터링 강도 | 결과 |
|--------|------------|------|
| 0.4-0.5 | 약함 | 많은 이미지, 일부 아티팩트 포함 |
| 0.6 ✅ | 중간 | 균형 (권장) |
| 0.7-0.8 | 강함 | 적은 이미지, 높은 품질 |

**측정 방법:**
```python
import json
with open('augmented_dataset/build_report.json') as f:
    report = json.load(f)

pruned_count = report['aroma_pruned']['defect_pruned']
total_count = report['aroma_full']['defect_count']
rejection_rate = pruned_count / total_count

print(f"Rejection rate: {rejection_rate:.1%}")
# 목표: 10-30% (너무 높으면 임계값 낮추기)
```

---

### 4.3 병렬 처리 워커 수

**workers 설정:**

| 값 | 동작 | 사용 사례 |
|----|------|----------|
| `0` | 순차 실행 | 디버깅, 메모리 제약 |
| `-1` | 자동 감지 | 로컬 환경 (권장) ✅ |
| `N` | N개 프로세스 | 수동 제어 |

**환경별 권장값:**

| 환경 | workers | 이유 |
|------|---------|------|
| Google Colab (무료) | `2` | CPU 코어 제한 |
| Google Colab (Pro) | `-1` | 자동 감지 |
| 로컬 (8코어) | `-1` | 최대 활용 |
| 메모리 부족 시 | `0` | 순차 처리 |

---

### 4.4 ROI 그리드 크기

**grid_size 영향:**

| 크기 | ROI 개수 (512px) | 처리 시간 | 정밀도 |
|------|------------------|-----------|--------|
| 32 | 256 | 느림 | 높음 |
| 64 ✅ | 64 | 중간 | 중간 |
| 128 | 16 | 빠름 | 낮음 |

**선택 기준:**
- 복잡한 질감 (PCB, 직물): 32-64
- 단순 질감 (병, 케이블): 64-128
- 빠른 프로토타이핑: 128

---

### 4.5 배치 크기 (Stage 7)

**train_batch_size 튜닝:**

```yaml
dataset:
  train_batch_size: 16    # GPU 메모리에 따라 조정
  eval_batch_size: 32     # 학습보다 2배 가능
```

**GPU별 권장값:**

| GPU | 메모리 | batch_size | image_size |
|-----|--------|------------|------------|
| T4 (Colab) | 16GB | 16 | 512 |
| V100 | 32GB | 32 | 512 |
| A100 | 40GB | 64 | 512 |

**OOM 발생 시:**
1. `train_batch_size` 줄이기 (16 → 8)
2. `image_size` 줄이기 (512 → 256)
3. `--device cpu` 사용 (매우 느림)

---

## 5. 예제 시나리오

### 시나리오 1: 작은 데이터셋 (good < 100)

```yaml
dataset:
  pruning_threshold: 0.5         # 느슨한 필터링
  augmentation_ratio_by_domain:
    mvtec:
      full: 3.0                  # 3배 증강
      pruned: 2.0
```

---

### 시나리오 2: 균형 잡힌 데이터셋 (good 100-500)

```yaml
dataset:
  pruning_threshold: 0.6         # 기본값 ✅
  augmentation_ratio_by_domain:
    mvtec:
      full: 2.0
      pruned: 1.5
```

---

### 시나리오 3: 큰 데이터셋 (good > 500)

```yaml
dataset:
  pruning_threshold: 0.7         # 엄격한 필터링
  augmentation_ratio_by_domain:
    mvtec:
      full: 1.0                  # 1:1 균형
      pruned: 0.8
```

---

## 6. 문제 해결

### Q: 증강 비율이 적용되지 않음

**확인:**
```bash
# build_report.json 확인
cat augmented_dataset/build_report.json | grep ratio
```

**원인:**
- Config 파일 경로가 잘못됨
- 도메인 이름 오타 (`mvtec` vs `MVTec`)

---

### Q: 너무 많은 이미지가 필터링됨

**증상:**
```json
{
  "aroma_full": {"defect_count": 4180},
  "aroma_pruned": {"defect_count": 50, "defect_pruned": 4130}
}
```

**해결:**
```yaml
dataset:
  pruning_threshold: 0.4  # 임계값 낮추기
```

---

### Q: OOM (Out of Memory)

**해결 순서:**
1. 배치 크기 줄이기
2. 워커 수 줄이기
3. 이미지 크기 줄이기
4. CPU 모드로 전환

```yaml
dataset:
  train_batch_size: 8      # 16 → 8
  num_workers: 2           # 4 → 2
  image_size: 256          # 512 → 256
```

---

**참고:**
- [README.md](../README.md) - 프로젝트 개요
- [quick-start.md](quick-start.md) - 단계별 실행 가이드
