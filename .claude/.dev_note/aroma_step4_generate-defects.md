# AROMA Step 4 — Defect Generation

## 사용할 skills: feature-dev

## 개요

Step 3 ROI 목록을 읽어 결함 이미지를 정상 배경 위에 합성한다.
Copy-Paste 방식을 기본 구현, ControlNet/Inpainting은 인터페이스 stub만 제공.

---

## 합성 방법 (--method)

| method | 상태 | 비고 |
|--------|------|------|
| `copy_paste` | **구현 완료** | GPU 불필요, Alpha composite 방식 |
| `controlnet` | stub | Colab GPU 환경에서 diffusers로 구현 |
| `inpainting` | stub | Colab GPU 환경에서 diffusers로 구현 |

---

## Copy-Paste 블렌딩 방식

- **Alpha composite** (기본값): PIL RGBA paste + Gaussian edge feather
- Poisson blending: scipy 사용 가능 시 향후 확장 예정 (현재 미사용)
- 결함 이미지 → 타원형 마스크 자동 생성 → 정상 이미지에 랜덤 위치 paste
- Feather sigma: `--feather_px` (기본 4px)

---

## 입력

| 파일 | 위치 |
|------|------|
| `roi_selected.json` | Step 3 output |
| `{normal_dir}/*.jpg` | 정상(good) 이미지 디렉토리 |

---

## 구현 항목

1. `copy_paste_synthesis(roi_entry, normal_image_path, output_path, ...) -> bool`
   - PIL 없으면 False 반환 (graceful)
   - defect/normal 이미지 없으면 False (경고만)
   - 성공 시 output_path에 RGB JPEG 저장

2. `controlnet_synthesis(...)` → `NotImplementedError`
3. `inpainting_synthesis(...)` → `NotImplementedError`

4. `run(roi_dir, normal_dir, output_dir, method, n_per_roi, ...)` → dict
   - normal_dir 없으면 dry_run 모드 (이미지 없이 annotation만 기록)
   - `n_per_roi`: ROI당 생성 개수 (기본 3)

5. CLI: `--roi_dir`, `--normal_dir`, `--output_dir`, `--method`, `--n_per_roi`, `--blend_mode`, `--feather_px`, `--seed`

---

## 출력

```
{output_dir}/
  images/                 synthetic defect images
  annotations.json        [{image_path, source_roi, normal_image, cluster_id,
                            cell_key, prompt, method, roi_score, deficit, dry_run}]
```

---

## TODO (미확정)

- **Poisson blending 완성**: scipy `spsolve` 기반 구현 → alpha composite 대비 품질 비교
- **ControlNet 구현**: diffusers + Colab GPU 셀에서 직접 구현
- Step 4 blending 방식은 Exp 3 (합성 품질 평가) 결과 후 최종 확정 예정

---

## 완료 기준

- `annotations.json` 생성
- dry_run 모드 동작 (normal_dir 없어도 실행)
- Pillow 없는 환경에서 graceful fallback

---

## 구현 완료 (2026-06-09)

- `scripts/aroma/generate_defects.py` 생성
- `tests/aroma/test_generate_defects.py` — 14/14 pass
