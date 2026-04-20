# Stage 1 — ROI 추출

## 목적

배경 이미지에서 ROI 추출 및 배경 텍스처 분류. Stage 3 배치 로직의 핵심 입력.

## 입력 / 출력 / Sentinel

| 항목 | 경로 |
|------|------|
| 입력 | `entry["image_dir"]/*.png` |
| 출력 | `{cat_dir}/stage1_output/roi_metadata.json` |
| 출력 | `{cat_dir}/stage1_output/masks/global\|local/` |
| Sentinel | `{cat_dir}/stage1_output/roi_metadata.json` |

## 스크립트

[[stage1_roi_extraction]] → `stage1_roi_extraction.py`
- `run_extraction(image_dir, output_dir, domain, roi_levels, grid_size, workers)`

## 핵심 파라미터

```python
roi_levels  = "both"   # global | local | both
grid_size   = 64       # 배경 분석 그리드 셀 크기 (px)
CAT_THREADS = 2        # Drive 동시 쓰기 안정성 고려
IMG_WORKERS = -1       # 이미지 단위 병렬 (cpu_count - 1)
```

## 계산 로직 / 임계값

**세그멘테이션:**
- Otsu: `cv2.THRESH_BINARY + cv2.THRESH_OTSU` (기본)
- SAM(vit_b): 체크포인트 있으면 우선, 가장 작은 마스크 선택

**ROI 추출:**
- `connectedComponentsWithStats`, `min_area = 100 px`
- 조건 미달 컴포넌트 없으면 전체 이미지를 단일 ROI로 폴백

**배경 분류 (lazy evaluation, 비용 순):**

| 단계 | 방법 | 임계값 | 결과 |
|------|------|--------|------|
| 1 | 픽셀 분산 | `variance < 100.0` | smooth |
| 2 | 그래디언트 방향 엔트로피 (8-bin, 22.5°/bin) | `entropy < 1.0` | directional |
| 3 | 자기상관 FFT 피크 (off-origin 정규화) | `peak > 0.15` | periodic |
| 4 | LBP 엔트로피 (P=8, R=1, uniform, 10-bin) | `entropy > 2.5` | organic |
| 5 | fallback | — | complex |

**roi_metadata.json 주요 필드:**

```json
{
  "image_id": "string",
  "roi_boxes": [{
    "level": "local",
    "box": [x, y, w, h],
    "background_type": "smooth|directional|periodic|organic|complex",
    "continuity_score": 0.0,
    "stability_score": 0.0,
    "dominant_angle": null
  }]
}
```
