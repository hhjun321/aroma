# AROMA Colab 공통 환경 설정 가이드

> **목적**: 모든 AROMA Colab 실행 가이드에 공통으로 필요한 환경을 설정한다.
> 이 셀들을 먼저 실행한 후 각 단계별 가이드를 실행한다.

---

## 셀 1 — Google Drive 마운트

```python
from google.colab import drive
drive.mount('/content/drive')
```

---

## 셀 2 — AROMA 저장소 클론

```python
import os

# 이미 클론되어 있으면 건너뜀
if not os.path.exists('/content/AROMA'):
    !git clone --single-branch -b main https://github.com/hhjun321/aroma.git /content/AROMA
else:
    print('AROMA already cloned, skipping.')
    !git -C /content/AROMA pull --ff-only
```

---

## 셀 3 — 의존성 설치

```python
!pip install -r /content/AROMA/requirements.txt -q
```

> 설치 후 런타임 재시작이 필요할 수 있다. Colab 메뉴 → 런타임 → 런타임 재시작.
> 재시작 후 셀 1부터 다시 실행한다.

---

## 셀 4 — 환경변수 설정

```python
import os

# Drive 데이터 루트 경로 (실제 Drive 경로에 맞게 수정)
os.environ['DRIVE'] = '/content/drive/MyDrive/data/Aroma'

# AROMA 스크립트 경로
os.environ['AROMA_SCRIPTS'] = '/content/AROMA/scripts/aroma'

# dataset_config.json 경로
os.environ['DATASET_CONFIG'] = os.environ.get(
    'DATASET_CONFIG', '/content/AROMA/dataset_config.json'
)

# AROMA 출력 루트 (Phase 0 ~ Step 4 결과 저장 위치)
os.environ['AROMA_OUT'] = f"{os.environ['DRIVE']}/aroma_output"

# 출력 디렉터리 생성
import pathlib
pathlib.Path(os.environ['AROMA_OUT']).mkdir(parents=True, exist_ok=True)

print("DRIVE          :", os.environ['DRIVE'])
print("AROMA_SCRIPTS  :", os.environ['AROMA_SCRIPTS'])
print("DATASET_CONFIG :", os.environ['DATASET_CONFIG'])
print("AROMA_OUT      :", os.environ['AROMA_OUT'])
```

> `DRIVE` 경로가 실제 Google Drive 마운트 위치와 다를 경우 위에서 직접 수정한다.

---

## 셀 5 — 설정 확인

```python
import os
from pathlib import Path

# dataset_config.json 존재 확인
dc_path = Path(os.environ['DATASET_CONFIG'])
if dc_path.exists():
    import json
    with open(dc_path) as f:
        cfg = json.load(f)
    datasets = [k for k in cfg if not k.startswith('_')]
    print(f"dataset_config.json OK — {len(datasets)}개 데이터셋 등록됨")
    for ds in datasets:
        print(f"  {ds}")
else:
    print(f"dataset_config.json 없음: {dc_path}")
    print("  AROMA 저장소 클론 또는 DATASET_CONFIG 경로를 확인하세요.")

print()

# AROMA 저장소 확인
aroma_scripts = Path(os.environ['AROMA_SCRIPTS'])
if aroma_scripts.exists():
    scripts = list(aroma_scripts.glob('*.py'))
    print(f"AROMA_SCRIPTS OK — {len(scripts)}개 스크립트 발견")
else:
    print(f"AROMA_SCRIPTS 없음: {aroma_scripts}")

print()

# Drive 마운트 확인
drive_path = Path(os.environ['DRIVE'])
if drive_path.exists():
    print(f"DRIVE OK: {drive_path}")
else:
    print(f"DRIVE 경로 없음: {drive_path}")
    print("  Google Drive가 마운트되어 있는지, 경로가 올바른지 확인하세요.")
```

---

## 참고: 환경변수 요약

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `DRIVE` | `/content/drive/MyDrive/data/Aroma` | Drive 데이터 루트 (수동 설정 필요) |
| `AROMA_SCRIPTS` | `/content/AROMA/scripts/aroma` | aroma 파이프라인 스크립트 디렉터리 |
| `DATASET_CONFIG` | `/content/AROMA/dataset_config.json` | 데이터셋 설정 파일 |
| `AROMA_OUT` | `$DRIVE/aroma_output` | 파이프라인 출력 루트 |

각 단계별 가이드는 이 환경변수들이 설정된 것을 전제로 작성되어 있다.
