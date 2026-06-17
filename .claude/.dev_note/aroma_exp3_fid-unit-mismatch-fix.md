# AROMA Exp 3 — FID 측정 단위 불일치 수정 (patch crop → full image)

## (사용할 skills: micro-fix)

## 개요

Exp 3 FID 측정에서 real side는 GT mask bbox로 crop한 작은 결함 패치를, synth side는 `source_roi`(test set 전체 이미지 512×512)를 비교하여 분포 단위가 일치하지 않았다. `annotations.json`에 `mask` 필드가 없어 patch-level 정렬이 원천 불가하며, 이로 인해 FID가 147–326으로 비정상 높고 AROMA ≈ Random인 무차별 결과가 나왔다. 양측 모두 **full image 비교**로 통일하여, copy-paste synthesis에서 AROMA가 최적화하는 ROI placement context(배경 텍스처·위치)가 metric에 반영되도록 교정한다.

이 수정은 이미 구현 완료된 상태이며, 이전에 산출된 `exp3_results.json` / `exp3_summary.md`는 무효 → 전 dataset 재실행 필요.

---

## 영향도 분석

### 이 기능이 변경하는 상태
- `$AROMA_OUT/exp3/exp3_results.json` — FID/KID/LPIPS 수치 (이전 결과 무효, 재실행 필요)
- `$AROMA_OUT/exp3/exp3_summary.md` — FID Delta 섹션 수치 변경

### 그 상태를 전제로 동작하는 기존 로직
- 없음 (평가 전용 스크립트)

### AD mode 영향 없음 명시
`_get_image_lists()`가 반환하는 `mask_map`은 AD mode(`_prepare_ad_dataset_with_masks` → pixel AUROC 세그멘테이션)에서 계속 사용된다. FID 호출부에서만 `mask_map` 인자를 끊었으므로 AD mode 회귀 없음.

---

## 수정 내용

### 1. `scripts/aroma/experiments/exp3_generation_quality.py` — `_load_real_defect_patches`

**구 시그니처**: `_load_real_defect_patches(test_defect_paths, mask_map, num_workers)`  
**신 시그니처**: `_load_real_defect_patches(test_defect_paths, num_workers)`

- `mask_map` 파라미터 제거
- `_mask_to_bbox` 호출 및 bbox crop 슬라이싱 로직 전체 제거
- full image 로드: `PIL Image.open(img_p).convert("RGB")` → `np.array`, 크롭 없음
- docstring: full-image 비교 근거(mask 미존재 → patch 정렬 불가) 기재

### 2. `scripts/aroma/experiments/exp3_generation_quality.py` — `_load_source_roi_crops`

- `ann.get("source_roi")` → `ann.get("image_path")` 로 변경 (합성된 결과 이미지 전체 로드)
- `is_fallback` 분기 로직 전체 제거
- fallback warning 제거, skip 카운트 로그를 `image_path missing or load fail` 기준으로 변경
- docstring 갱신 (함수명은 `_load_source_roi_crops` 유지 — TODO 참조)

### 3. `scripts/aroma/experiments/exp3_generation_quality.py` — `_run_fid_mode` 호출부

```python
# 변경 전
real_patches = _load_real_defect_patches(
    lists["test_defect"], lists["mask_map"], num_workers=num_workers
)

# 변경 후
real_patches = _load_real_defect_patches(
    lists["test_defect"], num_workers=num_workers
)
```

---

## 수정 대상 파일

- `scripts/aroma/experiments/exp3_generation_quality.py`

---

## 엣지 케이스

| 상황 | 처리 |
|------|------|
| `image_path` 미존재 또는 이미지 로드 실패 | `_load_one`에서 None 반환 후 필터링, skip 카운트 warning 로그 |
| real defect 0건 | `no_real_patches` 분기로 dataset skip (기존 유지) |
| synth 0건 (빈 annotations) | `no_synth_patches` 분기, FID/KID/LPIPS 각 `None` + error 기록 (기존 유지) |
| 이미지 해상도 상이 | `_resize_to_tensor`가 299×299로 통일 → Inception 입력 정합 |

---

## 테스트

CLAUDE.md: pytest 금지. Colab 직접 검증.

```python
!python $AROMA_SCRIPTS/experiments/exp3_generation_quality.py \
    --mode fid \
    --random_synthetic_dir $AROMA_OUT/synthetic_random \
    --aroma_synthetic_dir  $AROMA_OUT/synthetic \
    --real_data_dir        $AROMA_DATA \
    --dataset_keys         isp_LSM_1 mvtec_cable visa_cashew visa_pcb \
    --output_dir           $AROMA_OUT/exp3 \
    --num_workers          4
```

검증 포인트:
1. 로그에 `Loaded N real defect images` / `Loaded N synthetic images` 확인 (이전 "patches" 문구 아님)
2. FID 수치 147–326 → 정상 범위로 하락 (~20–80 예상)
3. AROMA FID < Random FID (음수 Delta) — ROI 품질 차이 반영 여부
4. `--mode ad` 회귀 확인: pixel AUROC 정상 산출 여부

---

## TODO

- `_load_source_roi_crops` 함수명이 실제 동작(full synthetic image 로드)과 불일치 → `_load_synthetic_images`로 리네이밍 검토 (호출부 1곳, 소규모)
- `_mask_to_bbox()`가 FID 경로에서 더 이상 호출되지 않음 → AD mode 외 사용처 없으면 dead code 제거 검토
- 논문 본문에서 "patch-level FID"로 기술된 부분이 있다면 "full-image FID"로 서술 수정 필요
