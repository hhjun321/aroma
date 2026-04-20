# Stage 6 — 증강 데이터셋 구성

## 목적

Stage 4 합성 이미지와 Stage 5 품질 점수를 사용해 3개 그룹 데이터셋 구성.

## 입력 / 출력 / Sentinel

| 항목 | 경로 |
|------|------|
| 입력 | `entry["image_dir"]` (원본 good 이미지) |
| 입력 | `{cat_dir}/stage4_output/{seed_id}/defect/*.png` |
| 입력 | `{cat_dir}/stage4_output/{seed_id}/quality_scores.json` |
| 출력 | `{cat_dir}/augmented_dataset/{baseline,aroma_full,aroma_pruned}/` |
| Sentinel | `{cat_dir}/augmented_dataset/build_report.json` |

## 스크립트

[[stage6_dataset_builder]] → `stage6_dataset_builder.py`
- `run_dataset_builder(cat_dir, image_dir, seed_dirs, pruning_threshold, ...)`

`utils/dataset_builder.py`
- `build_dataset_groups(...)` — 실제 구성 로직

## 핵심 파라미터

```python
PRUNING_THRESHOLD            = 0.6
PRUNING_THRESHOLD_BY_DOMAIN  = {"isp": 0.6, "mvtec": 0.6, "visa": 0.4}
SPLIT_RATIO                  = 0.8    # good 80% train / 20% test
SPLIT_SEED                   = 42     # 결정적 분할
BALANCE_DEFECT_TYPES         = True   # 결함 유형별 균등 샘플링
NUM_IO_THREADS               = 8
CAT_THREADS                  = 2
```

## 계산 로직 / 임계값

**3개 데이터셋 그룹:**

| 그룹 | 구성 | 설명 |
|------|------|------|
| `baseline` | good 이미지만 | pretrained 특징 거리 기반 평가 |
| `aroma_full` | good + 전체 합성 | Stage 4 전체 사용 |
| `aroma_pruned` | good + 품질 필터링 합성 | `quality_score ≥ pruning_threshold` |

**도메인별 증강 비율 (good 대비 defect 비율):**

| 도메인 | aroma_full | aroma_pruned | 태스크 |
|--------|-----------|-------------|--------|
| isp | 1.0 (1:1) | 0.5 | classification |
| mvtec | 2.0 (2:1) | 1.5 | segmentation |
| visa | 2.0 (2:1) | 1.5 | segmentation |

**VisA pruning_threshold=0.4 이유:**
artifact_score의 `hf_ratio` 정규화 상수(5.0)가 candle 왁스 매끄러운 표면(mean_mag 작음)에서 hf_ratio 과대평가 → 낮은 threshold 적용.

**BALANCE_DEFECT_TYPES=True:**
`seed_id` 접두사(`broken_large_`, `broken_small_` 등)로 결함 유형 식별 후 할당량을 유형 수로 균등 분배 → 특정 유형 편향 방지.

**디렉터리 구조:**

```
augmented_dataset/
├── baseline/
│   ├── train/good/
│   └── test/{good,defect}/
├── aroma_full/
│   ├── train/{good,defect}/
│   └── test/ → symlink → baseline/test
└── aroma_pruned/
    ├── train/{good,defect}/
    └── test/ → symlink → baseline/test
```
