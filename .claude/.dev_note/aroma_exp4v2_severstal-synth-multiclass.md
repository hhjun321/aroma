# Exp4v2 — Severstal multi 모드 synth 4-class 라벨

## (사용할 skills: micro-fix)

## 개요

Severstal exp4v2 multi 모드(`--class_mode multi`, nc=4)에서 real val/train은 per-class mask로 class 1~4(0-indexed 0~3) 라벨링되나, **synth 라벨은 `_write_yolo_labels`가 `class_id=0` 고정**(L780). 결과: multi + random/aroma 학습 시 synth 결함이 전부 class0으로 들어가 real(4-class)과 충돌 → per-class AP 오염 → DEFICIT Compensation(Class2 per-class 이득) 측정 불가.

수정: synth annotation의 `source_roi`(severstal = `.../test/class{N}/{id}.png`)에서 ClassId를 파싱해 `class_id = N-1`로 라벨. single 모드(기본)는 class0 유지(현행 불변). synth 1장 = 단일 ROI(class N) paste → 그 이미지의 전 bbox 동일 class.

## 영향도 분석

### 변경 상태
- `_write_yolo_labels`가 multi 모드에서 synth bbox에 source 기반 class_id 부여 (기존 class0 → N-1).

### 그 상태를 전제로 동작하는 기존 로직 (하위호환)
- `_bboxes_to_yolo_lines(class_id)` 이미 파라미터화 — class_id만 바꿔 전달.
- call site L1401 `_write_yolo_labels(...)`: `class_mode` 인자 추가 전달 (severstal multi → "multi").
- single/타 데이터셋: class_mode default "single" → class0 (byte-identical).

### 회귀 위험
- class_mode default "single" → 기존 전 데이터셋·single severstal 불변.
- multi는 severstal 전용(ds_class_mode) → 타 데이터셋 multi 미적용.

## 수정 내용

### scripts/aroma/experiments/exp4_v2_supervised_detection.py

1. **`_write_yolo_labels(..., class_mode="single")`** 시그니처에 추가.
2. annotation 루프 내, synth bbox lines 작성 시:
   - `class_mode == "multi"`이면 `cls_id = _parse_severstal_class(ann.get("source_roi"))` (정규식 `class(\d+)` → int-1, 0~3). 파싱 실패 시 0 + 1회 경고(또는 skip).
   - else `cls_id = 0` (현행).
   - `_bboxes_to_yolo_lines(bboxes, img_w, img_h, class_id=cls_id)` (L780 `class_id=0` → `class_id=cls_id`).
3. **신규 `_parse_severstal_class(source_roi: Optional[str]) -> Optional[int]`**: `re.search(r"class(\d+)", path)` → `int(grp)-1`, 범위 0~3 검증, 실패 None.
4. **call site L1401**: `_write_yolo_labels(..., min_area=50, class_mode=ds_class_mode)`.

> 참고: synth 한 이미지는 단일 ClassId source라 전 contour 동일 class. (드물게 한 synth에 다중 defect면 source 기준 일괄 — 현 합성은 1 ROI/이미지라 무관.)

## 수정 대상 파일
- `scripts/aroma/experiments/exp4_v2_supervised_detection.py`

## 암묵적 요구사항 (엣지)
- **source_roi 파싱 실패**(class 경로 아님): None → class0 fallback + 경고(타 데이터셋 multi는 ds_class_mode로 차단되므로 실질 severstal만).
- **single 불변**: class_mode != "multi"면 무조건 class0 (현행 경로).
- **0-indexed 정합**: real(_get_real_test_images_and_labels L1011)이 `cls-1` 사용 → synth도 동일 `N-1`. nc=4, names ['c1'..'c4'] 정합.
- **min_area·mask 우선순위**: 기존 synth bbox 추출(mask_path→diff→template) 그대로, class만 부여.

## 테스트 (Colab, pytest 금지)
1. severstal `--class_mode multi --condition all` → synth 라벨 txt의 class_id가 0~3 분포(전부 0 아님) 확인.
2. multi baseline(real만) 회귀: 기존과 동일.
3. single 모드(타 데이터셋 포함) 회귀: class0 불변.
4. per-class AP 산출 → Class2(희귀) AP에서 random vs aroma 비교 가능.

## 미확정 TODO
- 합성이 향후 다중-ROI/이미지로 바뀌면 per-bbox source 추적 필요(현재 1 ROI/이미지라 무관).
- multi 모드를 MVTec로 확장 시 defect_type subfolder 기반 class 파싱 일반화(현재 severstal class{N} 전용).
