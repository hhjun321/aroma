# AROMA Phase 2 파이프라인 흐름도

## 전체 흐름

```mermaid
flowchart TD
    RAW["📁 원본 데이터\n{domain}/{cat}/train/good\n{domain}/{cat}/test/{defect_type}"]

    S0["Stage 0: Resize\nstage0_resize.py\n─────────────────\n▸ 입력 이미지 → 512×512 리사이즈\n▸ Drive 저장 (1회성)"]

    S1["Stage 1: ROI Extraction\nstage1_roi_extraction.py\n─────────────────\n▸ test 결함 이미지에서 ROI 추출\n▸ placement_map.json 생성"]

    S1B["Stage 1b: Seed Characterization\nstage1b_seed_characterization.py\n─────────────────\n▸ SAM / Otsu로 결함 마스크 추출\n▸ seed 이미지 특성화\n▸ 출력: stage1b_output/{seed_id}/"]

    S2["Stage 2: Defect Seed Generation\nstage2_defect_seed_generation.py\n─────────────────\n▸ Elastic warp로 변형 이미지 생성\n▸ ROI 배치 위치 계산에만 사용\n▸ 출력: stage2_output/"]

    S3["Stage 3: Layout Logic\nstage3_layout_logic.py\n─────────────────\n▸ good 이미지 위에 ROI 배치 맵 생성\n▸ 출력: placement_map.json"]

    RAW --> S0 --> S1 --> S1B --> S2 --> S3

    S3 --> S4A
    S3 --> S4B

    subgraph SYNTH["합성 단계 (Stage 4)"]
        S4A["Stage 4-MPB (Phase 1)\nstage4_mpb_synthesis.py\n─────────────────\n▸ 픽셀 복사·붙여넣기\n▸ Poisson 경계 보정\n⚠ 블러 발생\n─────────────────\n▸ 출력: stage4_output/"]

        S4B["Stage 4-Diffusion (Phase 2)\nstage4_diffusion_synthesis.py\n─────────────────\n▸ SD Inpainting + ControlNet\n▸ ControlNet 입력: 1b seed Canny edge\n▸ 인페인팅 마스크: Stage 3 ROI\n─────────────────\n⚙ strength=0.7\n⚙ guidance_scale=7.5\n⚙ num_inference_steps=30\n⚙ conditioning_scale=0.7\n⚙ seed_ratio=0.25\n⚙ max_images_per_seed=5\n─────────────────\n▸ 출력: stage4_diffusion_output/"]
    end

    S5["Stage 5: Quality Scoring\nstage5_quality_scoring.py\n─────────────────\n▸ 합성 이미지 품질 점수 산출\n▸ 출력: quality_scores.json (seed별)"]

    S4A --> S5
    S4B --> S5

    S6["Stage 6: Dataset Builder\nstage6_dataset_builder.py\n─────────────────\n⚙ pruning_ratio=0.5  (상위 50% 선택)\n⚙ augmentation_ratio_by_domain\n⚙ balance_defect_types=True\n⚙ split_ratio=0.8"]

    S5 --> S6

    subgraph GROUPS["비교 그룹 (augmented_dataset/)"]
        G0["baseline\n원본 good만\n(결함 없음)"]
        G1["aroma_mpb\nPhase 1 MPB 합성\n(stage4_output 기반)"]
        G2["aroma_diffusion\nPhase 2 Diffusion 합성\n(pruning 0.5 + domain ratio)"]
        G3["aroma_ratio_10\nratio=10%\n(pruning 0.5 적용)"]
    end

    S6 --> G0
    S6 --> G1
    S6 --> G2
    S6 --> G3

    S7["Stage 7: Benchmark\nstage7_benchmark.py\n─────────────────\n⚙ epochs=30\n⚙ train_batch_size=16"]

    G0 --> S7
    G1 --> S7
    G2 --> S7
    G3 --> S7

    subgraph MODELS["평가 모델"]
        M1["YOLO11n-cls\n(분류)"]
        M2["EfficientDet-D0\n(분류)"]
    end

    S7 --> M1
    S7 --> M2

    RESULT["📊 벤치마크 결과\nbenchmark_results.json\n─────────────────\n▸ image_auroc\n▸ image_f1\n▸ pixel_auroc (mvtec/visa)"]

    M1 --> RESULT
    M2 --> RESULT
```

---

## 성능 개선 지점 분석

### 합성 품질 (Stage 4)

| 파라미터 | 현재값 | 개선 방향 |
|---|---|---|
| `num_inference_steps` | 30 | 높일수록 품질↑, 속도↓ |
| `strength` | 0.7 | 낮추면 원본 보존↑, 높이면 변형↑ |
| `conditioning_scale` | 0.7 | 높이면 구조 보존↑ (아티팩트 위험) |
| `seed_ratio` | 0.25 | 낮음 → 합성 다양성 제한 |
| `max_images_per_seed` | 5 | 낮음 → 합성 풀 부족 |

### 합성 풀 크기 (Stage 4 → 5)

현재 pcb4 기준: **200개** (20 seed × 5장)  
pruning 후: **100개**  
→ `aroma_ratio_10` 이상 모든 비율이 100개 상한에 걸림

개선: `max_images_per_seed` 또는 `seed_ratio` 상향 → 풀 확대

### 품질 필터링 (Stage 5 → 6)

| 파라미터 | 현재값 | 비고 |
|---|---|---|
| `pruning_ratio` | 0.5 | 하위 50% 제거 |
| `augmentation_ratio` (visa/mvtec) | 2.0 | 원본의 2배 합성 사용 |

pruning_ratio 낮추면 더 엄격하게 필터 → 품질↑ 수량↓  
현재 합성 풀이 작아서 더 낮추면 데이터 부족 심화

### 비교 그룹 설계

```
baseline ──────────────────────────────────────────────▶ 기준선 (random ~0.5)
aroma_mpb ─────────────────────────────────────────────▶ Phase 1 성능
aroma_diffusion ────────────────────────────────────────▶ Phase 2 핵심 지표
aroma_ratio_10 ─────────────────────────────────────────▶ 비율 탐색 (현재 합성 풀 = 100개로 동일)
```

**핵심 비교**: `aroma_diffusion` vs `aroma_mpb` → Diffusion이 MPB 블러 문제를 해소하는지 검증

### ControlNet 입력 설계 (Stage 4)

현재: Stage 1b seed 원본 이미지 → Canny edge → ControlNet  
대안: Stage 3 placement_map의 마스크 윤곽 → ControlNet (구조 보존↑)  
대안2: Fine-tuned ControlNet 사용 (도메인 특화)

### 평가 모델 (Stage 7)

현재: YOLO11n-cls + EfficientDet-D0  
pixel_auroc 없음 (isp 도메인) → 분류 성능만 측정  
개선: segmentation 모델 추가로 결함 위치 정확도 측정
