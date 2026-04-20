# Stage 3 — 레이아웃 로직 (적합도 기반 배치)

## 목적

각 배경 이미지의 ROI에 결함 seed를 배치할 최적 위치 결정. Stage 4 합성 입력.

## 입력 / 출력 / Sentinel

| 항목 | 경로 |
|------|------|
| 입력 | `{cat_dir}/stage1_output/roi_metadata.json` |
| 입력 | `{cat_dir}/stage1b_output/{seed_id}/seed_profile.json` |
| 입력 | `{cat_dir}/stage2_output/{seed_id}/*.png` |
| 출력 | `{cat_dir}/stage3_output/{seed_id}/placement_map.json` |
| Sentinel | `{cat_dir}/stage3_output/{seed_id}/placement_map.json` |

## 스크립트

[[stage3_layout_logic]] → `stage3_layout_logic.py`
- `run_layout_logic(roi_metadata, defect_seeds_dir, output_dir, seed_profile, domain, use_gpu)`

`utils/suitability.py`
- `SuitabilityEvaluator.compute_suitability(defect_subtype, background_type, continuity_score, stability_score)`
- `GPUSuitabilityEvaluator.compute_batch(defect_subtype, roi_boxes)` — PyTorch 배치 연산

## 핵심 파라미터

```python
use_gpu = True    # GPUSuitabilityEvaluator 사용 (PyTorch 필요)
workers = -1      # CPU 모드 시 이미지 단위 병렬
```

## 계산 로직 / 임계값

**적합도 점수 (0.0~1.0):**

```
suitability = 0.4 × matching
            + 0.3 × continuity_score
            + 0.2 × stability_score
            + 0.1 × gram_similarity   (기본값 0.0)
```

**MATCHING_RULES (defect_subtype × background_type):**

| subtype | smooth | directional | periodic | organic | complex |
|---------|--------|-------------|---------|---------|---------|
| linear_scratch | 0.5 | **1.0** | 0.7 | 0.3 | 0.3 |
| elongated | 0.6 | **0.9** | 0.7 | 0.4 | 0.4 |
| compact_blob | **0.9** | 0.4 | 0.7 | 0.6 | 0.5 |
| irregular | 0.5 | 0.4 | 0.5 | **0.8** | **0.9** |
| general | 0.7 | 0.7 | 0.7 | 0.7 | 0.7 |

**회전 결정:**
- `directional` 배경 + `linear_scratch` 또는 `elongated` → `rotation = dominant_angle` (결 정렬)
- 그 외 모든 경우 → `rotation = uniform(0, 360)`

**배치 좌표:** `x = roi_box.x + w//4`, `y = roi_box.y + h//4`

**GPU 모드:** 이미지별로 모든 ROI 스코어를 1회 계산 후 전체 seed에 재사용.

**placement_map.json 주요 필드:**

```json
[{
  "image_id": "string",
  "placements": [{
    "defect_path": "절대경로",
    "x": 0, "y": 0,
    "scale": 1.0,
    "rotation": 0.0,
    "suitability_score": 0.0,
    "matched_background_type": "smooth"
  }]
}]
```
