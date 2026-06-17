# AROMA VisA 데이터셋 준비 실행 가이드

> **목적**: `prepare_visa.py`를 실행하여 VisA 데이터셋을 AROMA 파이프라인 호환 구조로 재구성한다.
> **실행 환경**: CPU
> **전제**: `DRIVE` 환경변수가 이미 설정되어 있어야 한다. (00_setup.md 참고)

---

## 개요

VisA 표준 배포본은 `split_csv/1cls.csv` (또는 카테고리별 CSV)를 통해 이미지-레이블을 관리한다.
`prepare_visa.py`는 이 CSV를 읽어 카테고리별로 아래 구조의 심볼릭 링크(또는 파일 복사)를 생성한다:

```
<category>/
  train/good/           ← 정상 이미지 (train split)
  test/good/            ← 정상 이미지 (test split)
  test/anomaly/         ← 이상 이미지 (test split)
  ground_truth/anomaly/ ← 이상 마스크 (test split)
```

**멱등성**: 이미 존재하는 심볼릭 링크/파일은 건너뛴다. 재실행해도 안전하다.

---

## 사전 확인

VisA 데이터셋이 Drive에 다운로드되어 있어야 한다. 디렉터리 구조:

```
$VISA_DIR/
  split_csv/
    1cls.csv          ← 표준 VisA 배포본 (또는 카테고리별 CSV)
  cashew/             ← 카테고리 폴더
  pcb1/
  ...
```

---

## 환경변수 설정

```python
import os

# DRIVE는 00_setup.md 에서 이미 설정되어 있어야 함
# os.environ['DRIVE'] = '/content/drive/MyDrive/data/Aroma'  # 미설정 시 여기서 직접 설정

os.environ['VISA_DIR'] = f"{os.environ['DRIVE']}/VisA"
os.environ['VISA_OUT'] = f"{os.environ['DRIVE']}/VisA"   # 재구성 결과를 원본 위치 내에 생성

print("VISA_DIR:", os.environ['VISA_DIR'])
print("VISA_OUT:", os.environ['VISA_OUT'])
```

> `--visa_dir`은 VisA 루트 경로(split_csv/ 와 카테고리 폴더가 모두 위치하는 곳)를 가리킨다.
> 스크립트는 각 카테고리 폴더 안에 직접 링크를 생성하므로 별도 출력 디렉터리가 필요없다.

---

## 실행

### 기본 실행 (심볼릭 링크, 병렬 처리)

```python
!python /content/AROMA/prepare_visa.py     --visa_dir $VISA_DIR     --workers  -1
```

| 인자 | 기본값 | 설명 |
|------|--------|------|
| `--visa_dir` | (필수) | VisA 루트 경로 (`split_csv/` 포함) |
| `--workers` | `-1` | `-1` = auto (cpu_count-1), `0` = 순차 실행, `N` = N개 worker |
| `--convert_png` | (미지정) | 이미지를 PNG로 변환 (VisA 원본이 JPG인 경우 사용) |

### PNG 변환이 필요한 경우

VisA 원본 이미지가 JPG 형식이고 파이프라인이 PNG를 요구하는 경우:

```python
!python /content/AROMA/prepare_visa.py     --visa_dir   $VISA_DIR     --workers    -1     --convert_png
```

> `--convert_png` 사용 시 OpenCV(`cv2`)가 설치되어 있어야 한다.

---

## 출력 확인

```python
import os
from pathlib import Path

visa_dir = Path(os.environ['VISA_DIR'])

# 카테고리별 구조 확인
categories = [d.name for d in visa_dir.iterdir() if d.is_dir() and d.name != 'split_csv']
categories.sort()

print(f"카테고리 수: {len(categories)}")
print()

for cat in categories:
    cat_dir = visa_dir / cat
    ok = True
    for subdir in ['train/good', 'test/good', 'test/anomaly', 'ground_truth/anomaly']:
        if not (cat_dir / subdir).exists():
            ok = False
            print(f"  MISSING  {cat}/{subdir}")
    if ok:
        n_train = len(list((cat_dir / 'train/good').iterdir()))
        n_test_good = len(list((cat_dir / 'test/good').iterdir()))
        n_test_anom = len(list((cat_dir / 'test/anomaly').iterdir()))
        print(f"  OK  {cat:<20}  train={n_train}  test_good={n_test_good}  test_anom={n_test_anom}")
```

---

## dataset_config.json 등록

`prepare_visa.py` 실행 후 `dataset_config.json`에 VisA 카테고리를 추가해야 파이프라인에서 사용할 수 있다.

예시 (cashew 카테고리):

```json
"visa_cashew": {
    "image_dir": "/content/drive/MyDrive/data/Aroma/VisA/cashew/train/good",
    "defect_dir": "/content/drive/MyDrive/data/Aroma/VisA/cashew/test/anomaly",
    "mask_dir":   "/content/drive/MyDrive/data/Aroma/VisA/cashew/ground_truth/anomaly",
    "domain":     "visa"
}
```

> `image_dir`, `defect_dir`, `mask_dir` 경로는 실제 `DRIVE` 값에 맞게 수정한다.

---

## 주의사항

- **심볼릭 링크**: Colab 환경에서는 심볼릭 링크가 동작한다.
  Drive 마운트 해제 후 재마운트 시 링크가 깨질 수 있으므로, 중요한 실험 전에 재실행하거나 `--convert_png`로 실제 파일을 생성한다.
- **재실행 안전**: 기존 링크/파일이 있으면 건너뛰므로 중복 실행해도 문제없다.
- **1cls.csv 우선**: `split_csv/1cls.csv`가 있으면 이를 사용하고, 없으면 카테고리별 CSV(`split_csv/<category>.csv`)로 fallback한다.
