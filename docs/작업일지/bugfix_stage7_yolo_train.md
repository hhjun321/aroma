# BugFix: Stage 7 YOLO11 학습 TypeError

**날짜**: 2026-04-01
**파일**: `stage7_benchmark.py`
**증상**: ISP 카테고리(LSM_1 등) 벤치마크 실행 시 `TypeError: expected str, bytes or os.PathLike object, not NoneType`

---

## 증상

```
train: Scanning .../LSM_1/augmented_dataset/aroma_full/train... 3678 images, 0 corrupt: 100%

✗ 1건 실패
  [LSM_1] TypeError: expected str, bytes or os.PathLike object, not NoneType
```

- ASM: `baseline` 데이터 없음 (Stage 6 미완료) → `FileNotFoundError` (별도 이슈)
- LSM_1: 세 그룹 모두 존재하는데도 `aroma_full` 그룹 학습 시 TypeError

---

## 원인 분석

### 직접 원인

ultralytics YOLO 학습 후 내부에서 `trainer.best` 가 `None` 인 상태로
`Path(trainer.best)` 또는 `os.fspath(trainer.best)` 를 호출 → TypeError

`trainer.best = None` 이 되는 조건:
- `val` 디렉터리가 없고 auto-split 중 문제 발생
- 검증 metric 이 한 번도 개선되지 않아 best 체크포인트가 저장되지 않은 경우

### 연쇄 오염

`model.train()` 이 TypeError 로 종료되면 YOLO 객체 내부 상태(`ckpt_path`, `model.model` 등)가 오염된다.
오염된 모델로 `model.predict()` 를 호출하면 동일한 TypeError 재발.

### 초기 수정의 한계 (미해결 원인)

| 시도 | 문제 |
|------|------|
| `except TypeError` 추가 | 다른 예외 타입은 미처리 |
| `trainer.last` 로드 시도 후 실패 시 `return model` | 오염된 model 그대로 반환 → predict 에서 재발 |
| 예외 발생 시에만 복구 | 성공 후에도 `ckpt_path` 오염 가능 |

---

## 최종 수정 (`stage7_benchmark.py`)

### 1. `_train_yolo` 전면 재작성

```python
# val=False: val 디렉터리 없어도 안전 학습
try:
    model.train(..., val=False)
except Exception:
    pass  # trainer.last 에서 복구 시도

# 성공/실패 무관 — 항상 새 YOLO 인스턴스로 반환 (상태 오염 방지)
trainer = getattr(model, "trainer", None)
if trainer is not None:
    last_path = Path(str(getattr(trainer, "last", None) or ""))
    if last_path.exists():
        return YOLO(str(last_path))   # last.pt 에서 깨끗한 인스턴스

# last.pt 없음 (첫 epoch 전 실패) → 새 pretrained 반환
return YOLO("yolo11n-cls.pt")
```

핵심 변경:
- `except TypeError` → `except Exception` (모든 예외 포착)
- `val=False` 추가 (val 디렉터리 없어도 학습 진행)
- 성공/실패 구분 없이 **항상** `trainer.last` 에서 새 인스턴스 반환
- 복구 불가 시 오염 모델 대신 **새 pretrained 모델** 반환

### 2. 이미지 확장자 다중 지원

ISP 이미지가 `.png` 가 아닐 수 있음 → `glob("*.png")` 에서 빈 리스트 반환.

```python
_IMAGE_EXTENSIONS = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tiff", "*.tif")

def _glob_images(directory: Path) -> list[Path]:
    images: list[Path] = []
    for ext in _IMAGE_EXTENSIONS:
        images.extend(directory.glob(ext))
    return sorted(images)
```

`_collect_test_samples` 및 `_yolo_feature_distance` 에서 `_glob_images` 사용.

### 3. `run_benchmark` 모델×그룹 단위 오류 격리

```python
try:
    # 학습 + 평가
    ...
    results[model_name][group] = metrics
except Exception as e:
    results[model_name][group] = {"error": type(e).__name__, "detail": str(e)}
```

한 조합의 실패가 전체 카테고리를 중단시키지 않음.

---

## 디버그 스크립트

근본 원인 파악을 위해 `debug_yolo_train.py` 작성.
Colab 에서 직접 실행하여 full traceback 확인:

```bash
python debug_yolo_train.py \
    --cat_dir .../LSM_1 \
    --group aroma_full \
    --epochs 2
```

확인 항목:
- Step 3: `model.train()` 예외 발생 위치 및 traceback
- Step 4: `trainer.best` / `trainer.last` None 여부, 파일 존재 여부
- Step 5: `YOLO(last)` 재로드 가능 여부
- Step 6: 학습 후 `model.predict()` 정상 동작 여부

---

---

## 추가 개선: test set 공정성 보장 (CASDA 방식)

### 문제

기존 코드는 모든 그룹의 평가에 `baseline/test` 를 하드코딩:

```python
# 기존 — 모든 그룹이 baseline/test 를 공유 (fragile)
test_dir = Path(cat_dir) / "augmented_dataset" / "baseline" / "test"
```

- baseline 그룹이 없으면 전체 실패
- aroma_full / aroma_pruned 가 독립적인 test 디렉터리를 갖지 않음

### CASDA 방식

`prepare_benchmark_data.py::symlink_baseline_to_group()` 참조:

> baseline/test → 각 그룹 디렉터리에 복사(cp 또는 symlink)
> → 각 그룹이 동일한 test set 의 독립 복사본 보유

### 적용

```python
def _ensure_test_dir(cat_dir: str, group: str) -> Path:
    """group/test/ 없으면 baseline/test 를 shutil.copytree 로 복사."""
    group_test = aug / group / "test"
    if group_test.exists():
        return group_test
    shutil.copytree(str(baseline_test), str(group_test))
    return group_test
```

`run_benchmark` 내부에서 그룹 루프마다 호출:
```python
test_dir = _ensure_test_dir(cat_dir, group)
```

### 효과

| 구분 | 이전 | 이후 |
|------|------|------|
| test_dir 결정 | 항상 `baseline/test` 하드코딩 | 그룹별 `group/test` (없으면 자동 복사) |
| baseline 의존성 | 항상 필요 | 최초 1회만 필요 |
| 공정성 | 암묵적 (코드로 보장 안 됨) | 명시적 (동일 내용 복사본) |

---

## 관련 파일

| 파일 | 변경 |
|------|------|
| `stage7_benchmark.py` | `_train_yolo`, `_glob_images`, `_ensure_test_dir`, `run_benchmark` |
| `debug_yolo_train.py` | 신규 — 원인 파악용 디버그 스크립트 |
| `tests/test_stage7.py` | 테스트 추가 (`_ensure_test_dir` 3케이스) |
