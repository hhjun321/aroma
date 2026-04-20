# Stage 4 — MPB 합성 (Modified Poisson Blending)

## 목적

결함 patch를 배경 이미지에 합성해 증강 defect 이미지 생성. Stage 5 품질 평가의 입력.

## 입력 / 출력 / Sentinel

| 항목 | 경로 |
|------|------|
| 입력 | `entry["image_dir"]/*.png` (배경 이미지) |
| 입력 | `{cat_dir}/stage3_output/{seed_id}/placement_map.json` |
| 출력 | `{cat_dir}/stage4_output/{seed_id}/defect/*.png` |
| 출력 | `{cat_dir}/stage4_output/{seed_id}/defect/{image_id}_mask.png` |
| Sentinel | `{cat_dir}/stage4_output/{seed_id}/defect/` 에 PNG 존재 |

## 스크립트

[[stage4_mpb_synthesis]] → `stage4_mpb_synthesis.py`
- `run_synthesis_batch(image_dir, seed_placement_maps, output_root, format, use_fast_blend, workers, png_compression, max_background_dim)`

## 핵심 파라미터

```python
USE_FAST_BLEND     = True   # False → seamlessClone (느리지만 품질 높음)
IMG_THREADS        = 4      # ThreadPoolExecutor, I/O+CPU 혼합
format             = "cls"  # Stage 6 연동 필수값 — "yolo" 사용 불가
MAX_BACKGROUND_DIM = None   # 절대 변경 금지
PNG_COMPRESSION    = {"isp": 3, "mvtec": 3, "visa": 1}
```

## 계산 로직 / 임계값

**합성 방식 비교:**

| 방식 | 속도 | 품질 | 사용 조건 |
|------|------|------|---------|
| seamlessClone (`cv2.NORMAL_CLONE`) | 느림 | 높음 | `use_fast_blend=False` |
| Gaussian soft-mask alpha compositing | ~10-30× 빠름 | 낮음 | `use_fast_blend=True` |

**Gaussian soft-mask 계산:**

```
ksize   = max(3, (min(patch_h, patch_w) // 6) | 1)   # 홀수 보정
alpha   = GaussianBlur(ones(rh, rw), ksize)
blended = patch × alpha + bg × (1 - alpha)
```

**ROI 마스크 저장:** `{image_id}_mask.png` — Stage 5 품질 평가에서 결함 영역 집중 분석용.

**배치 최적화:** 배경 이미지 1회 로드 후 전 seed 합성 (seed 수만큼 I/O 절감).

**주의사항:**
- `MAX_BACKGROUND_DIM=None` 유지 필수 — 변경 시 good(원본 해상도) vs defect(축소 해상도) 불일치 → 학습 불가
- `format="yolo"` 사용 시 Stage 6 연동 불가 — 반드시 `format="cls"`
- VisA `png_compression=1`: 원본 ~2MB(3032×2016) → 쓰기 속도 최적화
