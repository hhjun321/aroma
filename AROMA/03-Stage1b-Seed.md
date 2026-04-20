# Stage 1b — Seed 결함 특성 분석

## 목적

결함 seed 이미지의 기하 지표를 분석해 서브타입 분류. Stage 3 적합도 계산의 입력.

## 입력 / 출력 / Sentinel

| 항목 | 경로 |
|------|------|
| 입력 | `entry["seed_dirs"][i]/*.png` |
| 출력 | `{cat_dir}/stage1b_output/{seed_id}/seed_profile.json` |
| 출력 | `{cat_dir}/stage1b_output/{seed_id}/seed_mask.png` |
| Sentinel | `{cat_dir}/stage1b_output/{seed_id}/seed_profile.json` |

**seed_id 명명:**
- seed_dirs 1개: `{seed.stem}` (예: `crack_001`)
- seed_dirs 복수: `{seed_dir.name}_{seed.stem}` (예: `broken_large_crack_001`)

## 스크립트

[[stage1b_seed_characterization]] → `stage1b_seed_characterization.py`
- `run_seed_characterization(seed_defect, output_dir, model_checkpoint)`
- `run_seed_characterization_batch(tasks, workers, model_checkpoint)`

`utils/defect_characterization.py`
- `DefectCharacterizer.analyze_defect_region(mask)` → 4개 기하 지표
- `DefectCharacterizer.classify_defect_subtype(metrics)` → 서브타입 문자열

## 핵심 파라미터

```python
NUM_WORKERS = 4   # seed 단위 ThreadPoolExecutor
```

## 계산 로직 / 임계값

**마스크 추출:**
- SAM(vit_b) 우선, 실패 시 Otsu(`THRESH_BINARY_INV + THRESH_OTSU`) 폴백
- `segmentation_method` 필드에 `"sam"` 또는 `"otsu"` 기록

**4개 기하 지표:**

| 지표 | 계산 | 의미 |
|------|------|------|
| linearity | `1 - (λ_min / λ_max)` — 픽셀 좌표 공분산 행렬의 고유값 비 | 직선=1.0, 원=0.0 |
| solidity | `region_area / convex_hull_area` | 볼록 채움 정도 |
| extent | `region_area / bounding_box_area` | 바운딩박스 대비 채움 |
| aspect_ratio | `major_axis_length / minor_axis_length` | 장단축 비 |

**결함 서브타입 분류 (우선순위 순):**

| 우선순위 | 서브타입 | 조건 |
|---------|---------|------|
| 1 | `linear_scratch` | linearity > **0.85** AND aspect_ratio > **5.0** |
| 2 | `elongated` | aspect_ratio > **5.0** AND linearity > **0.6** |
| 3 | `compact_blob` | aspect_ratio < **2.0** AND solidity > **0.9** |
| 4 | `irregular` | solidity < **0.7** |
| 5 | `general` | otherwise |

**seed_profile.json 구조:**

```json
{
  "seed_path": "절대경로",
  "subtype": "linear_scratch|elongated|compact_blob|irregular|general",
  "linearity": 0.0,
  "solidity": 0.0,
  "extent": 0.0,
  "aspect_ratio": 0.0,
  "mask_path": "절대경로",
  "segmentation_method": "sam|otsu"
}
```
