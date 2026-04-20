# Stage 5 — 합성 품질 점수

## 목적

합성 defect 이미지의 품질을 수치화. Stage 6 pruning의 기준.

## 입력 / 출력 / Sentinel

| 항목 | 경로 |
|------|------|
| 입력 | `{cat_dir}/stage4_output/{seed_id}/defect/*.png` |
| 입력 | `{cat_dir}/stage4_output/{seed_id}/defect/{image_id}_mask.png` (선택) |
| 출력 | `{cat_dir}/stage4_output/{seed_id}/quality_scores.json` |
| Sentinel | `{cat_dir}/stage4_output/{seed_id}/quality_scores.json` |

## 스크립트

[[stage5_quality_scoring]] → `stage5_quality_scoring.py`
- `run_quality_scoring_batch(stage4_seed_dirs, workers, parallel_seeds)`

`utils/quality_scoring.py`
- `score_defect_images(stage4_seed_dir, w_artifact, w_blur, workers)`

## 핵심 파라미터

```python
w_artifact     = 0.5
w_blur         = 0.5
workers        = -1   # 이미지 단위 병렬 (seed 내부)
parallel_seeds = -1   # seed 단위 병렬 (카테고리 내부)
# L4 Colab (8 CPU) 권장: parallel_seeds=-1, workers=0
```

## 계산 로직 / 임계값

**ROI 마스크 전처리:**
- `{stem}_mask.png` 로드 → 31×31 타원 커널 dilate → 결함 인접 픽셀만으로 통계 산출

**artifact_score (높을수록 아티팩트 없음):**

```
Sobel gradient magnitude:
  outlier_ratio = mean(mag > mean_mag + 3σ)
  edge_score    = 1 - min(outlier_ratio × 10, 1)   ← 10% 이상이면 최악

Laplacian energy:
  hf_ratio  = lap_energy / (mean_mag + std_mag + 1e-6)
  hf_score  = 1 - min(hf_ratio / 5.0, 1)           ← ratio 5.0 이상이면 최악

artifact = 0.6 × edge_score + 0.4 × hf_score
```

**blur_score (높을수록 선명):**

```
해상도 보정:
  scale     = num_pixels / (256 × 256)              ← 기준 해상도 256×256
  lap_score = clip(lap_var / (1000.0 × scale), 0, 1)

Gradient contrast:
  edge_sharp = clip((P90/P50 - 1) / 9.0, 0, 1)     ← ratio 1~10 → 0~1

blur = 0.5 × lap_score + 0.5 × edge_sharp
```

**최종 품질 점수:**

```
quality_score = 0.5 × artifact + 0.5 × blur
```

**quality_scores.json 구조:**

```json
{
  "weights": {"artifact": 0.5, "blur": 0.5},
  "scores": [
    {"image_id": "...", "artifact_score": 0.0, "blur_score": 0.0, "quality_score": 0.0}
  ],
  "stats": {
    "count": 0, "mean": 0.0, "std": 0.0,
    "min": 0.0, "max": 0.0,
    "p25": 0.0, "p50": 0.0, "p75": 0.0, "p90": 0.0
  }
}
```

**병렬화 조건:** `utils/parallel.py` — `num_workers > 1 AND len(tasks) >= 2` 일 때만 ProcessPoolExecutor 사용.
