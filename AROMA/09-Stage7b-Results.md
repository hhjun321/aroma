# Stage 7b — 결과 해석 가이드

## 메트릭 해석

**Image AUROC (주요 지표):**

| 범위 | 해석 |
|------|------|
| > 0.95 | 우수 — 합성 데이터가 실제 결함 패턴을 잘 반영 |
| 0.80 ~ 0.95 | 양호 |
| 0.60 ~ 0.80 | 미흡 — 합성 품질 또는 파이프라인 설정 재검토 |
| < 0.60 | 불량 |
| ≈ 1.0 (완벽) | 데이터 누수 의심 — test/train 분리 확인 필요 |

## 비교 분석 기준

**그룹 간 AUROC 향상 의미:**

| 비교 | 검증 내용 |
|------|---------|
| `baseline` → `aroma_full` 향상 | AROMA 합성의 전반적 유효성 |
| `aroma_full` → `aroma_pruned` 향상 | quality scoring pruning 효과 |
| `aroma_pruned` < `aroma_full` | pruning_threshold 너무 엄격 → threshold 낮추기 |

**도메인별 특성:**
- `isp` (classification): 낮은 증강비율(1:1), 결함 유형 단순 → 높은 AUROC 기대
- `mvtec`/`visa` (segmentation): 높은 증강비율(2:1), 다양한 표면 텍스처 → 카테고리 간 편차 큼

## 결과 파일 위치

```
outputs/benchmark_results/
└── {cat_name}/
    └── {yolo11|efficientdet_d0}/
        └── {baseline|aroma_full|aroma_pruned}/
            └── experiment_meta.json
```

```json
{
  "image_auroc": 0.0,
  "image_f1": 0.0,
  "pixel_auroc": 0.0
}
```

## 결과 초기화 절차

```python
import shutil
from pathlib import Path
for d in Path("outputs/benchmark_results").glob("*"):
    shutil.rmtree(d, ignore_errors=True)
```
