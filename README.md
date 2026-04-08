# AROMA: Adaptive ROI-Aware Augmentation Framework

[한국어](#한국어) | [English](#english)

---

## 한국어

### 개요

**AROMA**(Adaptive ROI-Oriented Multi-domain Augmentation)는 산업용 결함 검사 시스템의 데이터 희소성 문제를 해결하기 위한 맞춤형 결함 합성 및 증강 프레임워크입니다.

**핵심 기술:**
- **Modified Poisson Blending (MPB)**: 그래디언트 최적화를 통한 경계 아티팩트 제거
- **Context-Aware Layout**: 배경 질감과 결함 형태의 물리적 매칭
- **Training-Free Generation**: 단일 결함 샘플로부터 다양한 변이체 생성

**지원 도메인:**
- ISP-AD (스크린 인쇄 결함)
- MVTec AD (산업용 객체 결함)
- ViSA (시각적 이상 탐지)

### 주요 기능

✅ ROI 기반 자동 배경 분석 및 분할  
✅ 물리적으로 타당한 결함 배치 로직  
✅ 고품질 경계 블렌딩 (MPB)  
✅ 도메인별 증강 비율 제어  
✅ 병렬 처리 지원 (모든 Stage)  
✅ YOLO11, RT-DETR, DRAEM 벤치마크 통합  

---

### 빠른 시작

#### 1. 환경 요구사항

- **플랫폼**: Google Colab (권장) 또는 로컬 GPU 환경
- **Python**: 3.10+
- **GPU**: CUDA 지원 (Stage 3, 7 병렬 처리)

#### 2. 설치

```bash
# 저장소 클론
git clone <repository-url>
cd aroma

# 의존성 설치
pip install -r requirements.txt
```

#### 3. 최소 실행 예제 (1개 카테고리)

Google Colab 환경에서 MVTec bottle 카테고리를 처리하는 예제:

```bash
# Google Drive 마운트
from google.colab import drive
drive.mount('/content/drive')

# Stage 0: 이미지 리사이즈 (256x256)
!python stage0_resize.py \
  --config /content/drive/MyDrive/aroma/dataset_config.json

# Stage 1: ROI 추출 및 배경 분석
!python stage1_roi_extraction.py \
  --image_dir /content/drive/MyDrive/data/Aroma/mvtec/bottle/train/good \
  --output_dir /content/drive/MyDrive/data/Aroma/mvtec/bottle/stage1_output \
  --grid_size 64 \
  --workers -1

# Stage 1b: Defect Seed 특성 분석
!python stage1b_seed_characterization.py \
  --seed_dir /content/drive/MyDrive/data/Aroma/mvtec/bottle/test/broken_large \
  --output_dir /content/drive/MyDrive/data/Aroma/mvtec/bottle/stage1b_output

# Stage 2: Defect 변이체 생성
!python stage2_defect_seed_generation.py \
  --seed_profile /content/drive/MyDrive/data/Aroma/mvtec/bottle/stage1b_output/seed_profile.json \
  --output_dir /content/drive/MyDrive/data/Aroma/mvtec/bottle/stage2_output \
  --num_variants 100 \
  --workers -1

# Stage 3: Context-Aware 배치 계획
!python stage3_layout_logic.py \
  --roi_metadata /content/drive/MyDrive/data/Aroma/mvtec/bottle/stage1_output/roi_metadata.json \
  --defect_seeds_dir /content/drive/MyDrive/data/Aroma/mvtec/bottle/stage2_output \
  --output_json /content/drive/MyDrive/data/Aroma/mvtec/bottle/stage3_output/placement_map.json \
  --workers -1

# Stage 4: MPB 합성
!python stage4_mpb_synthesis.py \
  --image_dir /content/drive/MyDrive/data/Aroma/mvtec/bottle/train/good \
  --placement_map /content/drive/MyDrive/data/Aroma/mvtec/bottle/stage3_output/placement_map.json \
  --output_dir /content/drive/MyDrive/data/Aroma/mvtec/bottle/stage4_output \
  --workers -1

# Stage 5: 품질 점수 계산
!python stage5_quality_scoring.py \
  --stage4_seed_dir /content/drive/MyDrive/data/Aroma/mvtec/bottle/stage4_output \
  --workers -1

# Stage 6: 데이터셋 구성
!python stage6_dataset_builder.py \
  --cat_dir /content/drive/MyDrive/data/Aroma/mvtec/bottle \
  --image_dir /content/drive/MyDrive/data/Aroma/mvtec/bottle/train/good \
  --seed_dirs /content/drive/MyDrive/data/Aroma/mvtec/bottle/test/broken_large \
  --config /content/drive/MyDrive/aroma/configs/benchmark_experiment.yaml \
  --workers -1

# Stage 7: 벤치마크 (YOLO11)
!python stage7_benchmark.py \
  --config /content/drive/MyDrive/aroma/configs/benchmark_experiment.yaml \
  --model yolo11 \
  --tasks isp_ASM mvtec_bottle visa_candle
```

상세한 단계별 가이드는 [docs/quick-start.md](docs/quick-start.md)를 참조하세요.

---

### 파이프라인 구조

```
Stage 0: Resize          → 이미지 정규화 (256×256)
    ↓
Stage 1: ROI Extraction  → 배경 분석 + ROI 그리드 추출
    ↓
Stage 1b: Seed Characterization → Defect 유형 분류 (linear/blob/elongated)
    ↓
Stage 2: Variant Generation → 단일 시드로부터 N개 변형 생성
    ↓
Stage 3: Layout Planning → 배경-결함 매칭 + 배치 최적화
    ↓
Stage 4: MPB Synthesis   → Poisson Blending으로 합성
    ↓
Stage 5: Quality Scoring → 아티팩트/블러 점수 계산
    ↓
Stage 6: Dataset Builder → 품질 필터링 + 데이터셋 구성
    ↓
Stage 7: Benchmark       → YOLO/DRAEM 학습 및 평가
```

---

### 프로젝트 구조

```
aroma/
├── stage0_resize.py              # 이미지 리사이즈
├── stage1_roi_extraction.py      # ROI 추출 및 배경 분석
├── stage1b_seed_characterization.py  # Defect 특성 분석
├── stage2_defect_seed_generation.py  # 변이체 생성
├── stage3_layout_logic.py        # 배치 계획
├── stage4_mpb_synthesis.py       # MPB 합성
├── stage5_quality_scoring.py     # 품질 평가
├── stage6_dataset_builder.py     # 데이터셋 구성
├── stage7_benchmark.py           # 모델 학습 및 평가
├── utils/                        # 유틸리티 모듈
│   ├── background_characterization.py
│   ├── defect_characterization.py
│   ├── suitability.py
│   ├── quality_scoring.py
│   ├── dataset_builder.py
│   ├── io.py
│   └── parallel.py
├── configs/
│   └── benchmark_experiment.yaml # 실험 설정
├── dataset_config.json           # 데이터셋 경로 설정
├── tests/                        # 단위 테스트 (157개)
└── docs/                         # 문서
```

---

### 설정 파일

#### `dataset_config.json`
각 카테고리의 경로를 정의:
```json
{
  "mvtec_bottle": {
    "domain": "mvtec",
    "image_dir": "/path/to/train/good",
    "seed_dir": "/path/to/test/defect_type"
  }
}
```

#### `configs/benchmark_experiment.yaml`
도메인별 증강 비율 및 벤치마크 설정:
```yaml
augmentation_ratio_by_domain:
  isp:
    full: 1.0
    pruned: 0.5
  mvtec:
    full: 2.0
    pruned: 1.5
```

설정 파일 가이드: [docs/configuration.md](docs/configuration.md)

---

### 주요 파라미터

| 파라미터 | 설명 | 기본값 | 권장값 |
|---------|------|--------|--------|
| `--workers` | 병렬 워커 수 (-1=자동) | 0 | -1 |
| `--grid_size` | ROI 그리드 크기 | 64 | 32-128 |
| `--num_variants` | 시드당 변이체 수 | 100 | 50-200 |
| `--pruning_threshold` | 품질 필터 임계값 | 0.6 | 0.5-0.7 |
| `--augmentation_ratio_full` | 증강 비율 (전체) | None | 1.0-2.0 |

---

### 성능 최적화

**병렬 처리:**
- `--workers=-1`: CPU 코어 수 자동 감지
- `--workers=N`: N개 프로세스 사용
- `--workers=0`: 순차 실행 (디버깅용)

**GPU 가속:**
- Stage 3: suitability 계산 (CUDA)
- Stage 7: 모델 학습 (CUDA)

**캐싱:**
- 모든 Stage는 sentinel 파일로 재실행 방지
- Stage 6은 품질 점수 기반 캐시 검증

---

### 벤치마크 결과

**Augmentation Ratio 제어 효과** (Stage 6 적용 후):

| 카테고리 | Good | Before (Full) | After (Full) | 비율 감소 |
|---------|------|--------------|--------------|----------|
| isp_ASM | 500 | 4,356 | 500 | 8.7x → 1.0x |
| mvtec_bottle | 209 | 4,180 | 418 | 20x → 2.0x |
| visa_candle | 900 | 9,000 | 1,800 | 10x → 2.0x |

---

### 문서

- **[Quick Start Guide](docs/quick-start.md)**: 단계별 실행 가이드
- **[Configuration Guide](docs/configuration.md)**: 설정 파일 상세 설명
- **Work Logs**: `docs/작업일지/` (한글)

---

### 테스트

```bash
# 전체 테스트 실행 (157개)
pytest tests/

# 특정 Stage 테스트
pytest tests/test_stage6.py -v

# 커버리지 포함
pytest --cov=. --cov-report=html
```

---

### 라이선스

이 프로젝트는 연구 목적으로 작성되었습니다.

---

### 인용

```bibtex
@article{aroma2026,
  title={AROMA: Adaptive ROI-Aware Augmentation for Industrial Defect Synthesis},
  author={Your Name},
  journal={TBD},
  year={2026}
}
```

---

## English

### Overview

**AROMA** (Adaptive ROI-Oriented Multi-domain Augmentation) is a customized defect synthesis and augmentation framework designed to solve the data scarcity problem in industrial visual inspection systems.

**Core Technologies:**
- **Modified Poisson Blending (MPB)**: Eliminates boundary artifacts through gradient optimization
- **Context-Aware Layout**: Physical matching between background texture and defect morphology
- **Training-Free Generation**: Generates diverse variants from a single defect sample

**Supported Domains:**
- ISP-AD (Screen-printed defects)
- MVTec AD (Industrial object defects)
- ViSA (Visual anomaly detection)

### Key Features

✅ Automatic ROI-based background analysis and segmentation  
✅ Physically plausible defect placement logic  
✅ High-quality boundary blending (MPB)  
✅ Domain-specific augmentation ratio control  
✅ Parallel processing support (all stages)  
✅ YOLO11, RT-DETR, DRAEM benchmark integration  

---

### Quick Start

#### 1. Requirements

- **Platform**: Google Colab (recommended) or local GPU environment
- **Python**: 3.10+
- **GPU**: CUDA support (for Stage 3, 7 parallel processing)

#### 2. Installation

```bash
# Clone repository
git clone <repository-url>
cd aroma

# Install dependencies
pip install -r requirements.txt
```

#### 3. Minimal Example (Single Category)

Example processing MVTec bottle category in Google Colab:

```bash
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Stage 0: Image resize (256x256)
!python stage0_resize.py \
  --config /content/drive/MyDrive/aroma/dataset_config.json

# Stage 1: ROI extraction and background analysis
!python stage1_roi_extraction.py \
  --image_dir /content/drive/MyDrive/data/Aroma/mvtec/bottle/train/good \
  --output_dir /content/drive/MyDrive/data/Aroma/mvtec/bottle/stage1_output \
  --grid_size 64 \
  --workers -1

# Stage 1b: Defect seed characterization
!python stage1b_seed_characterization.py \
  --seed_dir /content/drive/MyDrive/data/Aroma/mvtec/bottle/test/broken_large \
  --output_dir /content/drive/MyDrive/data/Aroma/mvtec/bottle/stage1b_output

# Stage 2: Generate defect variants
!python stage2_defect_seed_generation.py \
  --seed_profile /content/drive/MyDrive/data/Aroma/mvtec/bottle/stage1b_output/seed_profile.json \
  --output_dir /content/drive/MyDrive/data/Aroma/mvtec/bottle/stage2_output \
  --num_variants 100 \
  --workers -1

# Stage 3: Context-aware placement planning
!python stage3_layout_logic.py \
  --roi_metadata /content/drive/MyDrive/data/Aroma/mvtec/bottle/stage1_output/roi_metadata.json \
  --defect_seeds_dir /content/drive/MyDrive/data/Aroma/mvtec/bottle/stage2_output \
  --output_json /content/drive/MyDrive/data/Aroma/mvtec/bottle/stage3_output/placement_map.json \
  --workers -1

# Stage 4: MPB synthesis
!python stage4_mpb_synthesis.py \
  --image_dir /content/drive/MyDrive/data/Aroma/mvtec/bottle/train/good \
  --placement_map /content/drive/MyDrive/data/Aroma/mvtec/bottle/stage3_output/placement_map.json \
  --output_dir /content/drive/MyDrive/data/Aroma/mvtec/bottle/stage4_output \
  --workers -1

# Stage 5: Quality scoring
!python stage5_quality_scoring.py \
  --stage4_seed_dir /content/drive/MyDrive/data/Aroma/mvtec/bottle/stage4_output \
  --workers -1

# Stage 6: Dataset construction
!python stage6_dataset_builder.py \
  --cat_dir /content/drive/MyDrive/data/Aroma/mvtec/bottle \
  --image_dir /content/drive/MyDrive/data/Aroma/mvtec/bottle/train/good \
  --seed_dirs /content/drive/MyDrive/data/Aroma/mvtec/bottle/test/broken_large \
  --config /content/drive/MyDrive/aroma/configs/benchmark_experiment.yaml \
  --workers -1

# Stage 7: Benchmark (YOLO11)
!python stage7_benchmark.py \
  --config /content/drive/MyDrive/aroma/configs/benchmark_experiment.yaml \
  --model yolo11 \
  --tasks isp_ASM mvtec_bottle visa_candle
```

For detailed step-by-step guide, see [docs/quick-start.md](docs/quick-start.md).

---

### Pipeline Architecture

```
Stage 0: Resize          → Image normalization (256×256)
    ↓
Stage 1: ROI Extraction  → Background analysis + ROI grid extraction
    ↓
Stage 1b: Seed Characterization → Defect type classification (linear/blob/elongated)
    ↓
Stage 2: Variant Generation → Generate N variants from single seed
    ↓
Stage 3: Layout Planning → Background-defect matching + placement optimization
    ↓
Stage 4: MPB Synthesis   → Synthesis via Poisson Blending
    ↓
Stage 5: Quality Scoring → Artifact/blur score calculation
    ↓
Stage 6: Dataset Builder → Quality filtering + dataset construction
    ↓
Stage 7: Benchmark       → YOLO/DRAEM training and evaluation
```

---

### Project Structure

```
aroma/
├── stage0_resize.py              # Image resize
├── stage1_roi_extraction.py      # ROI extraction and background analysis
├── stage1b_seed_characterization.py  # Defect characterization
├── stage2_defect_seed_generation.py  # Variant generation
├── stage3_layout_logic.py        # Placement planning
├── stage4_mpb_synthesis.py       # MPB synthesis
├── stage5_quality_scoring.py     # Quality scoring
├── stage6_dataset_builder.py     # Dataset construction
├── stage7_benchmark.py           # Model training and evaluation
├── utils/                        # Utility modules
│   ├── background_characterization.py
│   ├── defect_characterization.py
│   ├── suitability.py
│   ├── quality_scoring.py
│   ├── dataset_builder.py
│   ├── io.py
│   └── parallel.py
├── configs/
│   └── benchmark_experiment.yaml # Experiment configuration
├── dataset_config.json           # Dataset path configuration
├── tests/                        # Unit tests (157 tests)
└── docs/                         # Documentation
```

---

### Configuration Files

#### `dataset_config.json`
Define paths for each category:
```json
{
  "mvtec_bottle": {
    "domain": "mvtec",
    "image_dir": "/path/to/train/good",
    "seed_dir": "/path/to/test/defect_type"
  }
}
```

#### `configs/benchmark_experiment.yaml`
Domain-specific augmentation ratios and benchmark settings:
```yaml
augmentation_ratio_by_domain:
  isp:
    full: 1.0
    pruned: 0.5
  mvtec:
    full: 2.0
    pruned: 1.5
```

Configuration guide: [docs/configuration.md](docs/configuration.md)

---

### Key Parameters

| Parameter | Description | Default | Recommended |
|-----------|-------------|---------|-------------|
| `--workers` | Number of parallel workers (-1=auto) | 0 | -1 |
| `--grid_size` | ROI grid size | 64 | 32-128 |
| `--num_variants` | Variants per seed | 100 | 50-200 |
| `--pruning_threshold` | Quality filter threshold | 0.6 | 0.5-0.7 |
| `--augmentation_ratio_full` | Augmentation ratio (full) | None | 1.0-2.0 |

---

### Performance Optimization

**Parallel Processing:**
- `--workers=-1`: Auto-detect CPU cores
- `--workers=N`: Use N processes
- `--workers=0`: Sequential execution (for debugging)

**GPU Acceleration:**
- Stage 3: Suitability computation (CUDA)
- Stage 7: Model training (CUDA)

**Caching:**
- All stages use sentinel files to prevent re-execution
- Stage 6 validates cache based on quality scores

---

### Benchmark Results

**Augmentation Ratio Control Effect** (After Stage 6):

| Category | Good | Before (Full) | After (Full) | Ratio Reduction |
|----------|------|---------------|--------------|-----------------|
| isp_ASM | 500 | 4,356 | 500 | 8.7x → 1.0x |
| mvtec_bottle | 209 | 4,180 | 418 | 20x → 2.0x |
| visa_candle | 900 | 9,000 | 1,800 | 10x → 2.0x |

---

### Documentation

- **[Quick Start Guide](docs/quick-start.md)**: Step-by-step execution guide
- **[Configuration Guide](docs/configuration.md)**: Detailed configuration file explanation
- **Work Logs**: `docs/작업일지/` (Korean)

---

### Testing

```bash
# Run all tests (157 tests)
pytest tests/

# Test specific stage
pytest tests/test_stage6.py -v

# With coverage
pytest --cov=. --cov-report=html
```

---

### License

This project is created for research purposes.

---

### Citation

```bibtex
@article{aroma2026,
  title={AROMA: Adaptive ROI-Aware Augmentation for Industrial Defect Synthesis},
  author={Your Name},
  journal={TBD},
  year={2026}
}
```

---

**Contact**: For questions or issues, please open an issue on GitHub.
