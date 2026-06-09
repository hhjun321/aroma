# AROMA 구현 코드를 aroma 저장소로 이전 및 저장소 역할 분리

## 사용할 skills: feature-dev

## 개요

현재 `aroma-plus` 저장소가 구현 코드(Step 1-4 스크립트, 테스트, Colab 노트북)와 설계 문서(dev_note, spec)를 함께 보유하여 역할이 혼재되어 있다.
코드 저장소(`aroma`)와 문서 저장소(`aroma-plus`)를 명확히 분리한다.

- `D:\project\aroma` = **코드 저장소**: Phase 0 + Step 1-4 스크립트, 테스트, Colab 노트북
- `D:\project\aroma-plus` = **문서 전용**: dev_note, spec, Colab 가이드 문서

분리 후 Colab에서는 단일 저장소(`AROMA`)만 클론하면 전체 파이프라인이 실행 가능해진다.

---

## 영향도 분석

### 이 작업이 변경하는 상태
- `aroma` 저장소에 `scripts/aroma/`, `tests/aroma/`, `notebooks/` 디렉토리 신규 생성
- `aroma-plus` 저장소에서 `scripts/`, `tests/`, `notebooks/` 삭제 (문서 전용)

### 기존 로직과의 의존 관계
- 테스트는 `sys.path.insert(0, .../scripts)` + `from aroma.X import` 패턴 → `scripts/aroma/__init__.py` 필수
- `compute_complexity.py`가 `Path(__file__).parent / "config" / "aroma_step1.yaml"` 상대 경로 사용 → 디렉토리 구조 유지 필요
- 기존 `aroma/tests/`의 `conftest.py`(cv2 기반)와 새 `tests/aroma/`는 독립적 → 충돌 없음

---

## 수정 내용

### 1. `aroma/scripts/aroma/` — 신규 디렉토리, 스크립트 추가

aroma-plus에서 복사. 디렉토리 구조 그대로 유지:

```
aroma/scripts/aroma/
  __init__.py                          ← 요청 목록에 없으나 import 필수
  compute_complexity.py
  prompt_generation.py
  roi_selection.py
  generate_defects.py
  config/
    aroma_step1.yaml
```

### 2. `aroma/tests/aroma/` — 신규 디렉토리, 테스트 추가

```
aroma/tests/aroma/
  __init__.py                          ← 요청 목록에 없으나 패키지 인식 필수
  test_compute_complexity.py   (23 tests)
  test_prompt_generation.py    (22 tests)
  test_roi_selection.py        (21 tests)
  test_generate_defects.py     (14 tests)
```

### 3. `aroma/notebooks/colab_step1_guide.ipynb` — 신규, 경로 수정 포함

aroma-plus에서 복사 후 다음 셀 수정:

| 셀 | 변경 전 | 변경 후 |
|----|---------|---------|
| cell-02-clone | `AROMA_PLUS_REPO` 변수 + 클론 라인 존재 | `AROMA_PLUS_REPO` 제거, 단일 `AROMA_REPO` 클론만 유지 |
| cell-02-header | "YOUR_AROMA_PLUS_REPO_URL" 언급 | 제거 |
| cell-04-env | `AROMA_SCRIPTS = '/content/AROMA_PLUS/scripts/aroma'` | `'/content/AROMA/scripts/aroma'` |
| cell-04-env | `DATASET_CONFIG = '/content/AROMA_PLUS/scripts/aroma/config/...'` | `TODO: 경로 결정 필요` (아래 미확정 참고) |
| cell-00-intro | "(aroma-plus 신규)" | "(aroma 저장소)" |

### 4. `aroma-plus/scripts/`, `aroma-plus/tests/`, `aroma-plus/notebooks/` — 삭제

이전 완료 후 aroma-plus에서 코드 디렉토리 제거. 문서 파일만 잔류:
- `.claude/` (dev_note, rules, skills)
- `AROMA-sharpened-spec.md`
- 기타 spec 문서

---

## 수정 대상 파일

### aroma 저장소 — 신규 생성
- `scripts/aroma/__init__.py`
- `scripts/aroma/compute_complexity.py`
- `scripts/aroma/prompt_generation.py`
- `scripts/aroma/roi_selection.py`
- `scripts/aroma/generate_defects.py`
- `scripts/aroma/config/aroma_step1.yaml`
- `tests/aroma/__init__.py`
- `tests/aroma/test_compute_complexity.py`
- `tests/aroma/test_prompt_generation.py`
- `tests/aroma/test_roi_selection.py`
- `tests/aroma/test_generate_defects.py`
- `notebooks/colab_step1_guide.ipynb` (수정본)

### aroma 저장소 — 선택적 수정
- `requirements.txt` — pyyaml / pillow 추가 여부 검토

### aroma-plus 저장소 — 삭제
- `scripts/aroma/` 전체
- `tests/aroma/` 전체
- `notebooks/` 전체

---

## 테스트

```bash
# aroma 저장소 루트에서
cd D:\project\aroma
python -m pytest tests/aroma/ -v
# 기대: 80 tests PASS (23+22+21+14)

# import 정상 여부 확인
python -c "import sys; sys.path.insert(0,'scripts'); from aroma.compute_complexity import load_config; print('ok')"

# 기존 aroma 테스트 회귀 없음 확인
python -m pytest tests/ -v --ignore=tests/aroma/
```

---

## 미확정 사항

- [ ] **`DATASET_CONFIG` 경로 결정**: 노트북 cell-04-env의 `DATASET_CONFIG`가 가리키는 `dataset_config.json`은 `scripts/aroma/config/` 위치에 없음. 선택지:
  - (A) `aroma` 저장소 루트에 이미 있는 `dataset_config.json`을 참조: `'/content/AROMA/dataset_config.json'`
  - (B) `dataset_config.json`을 `scripts/aroma/config/`에 복사하고 노트북 경로 유지
  - 이번 이전 범위에 `dataset_config.json` 포함 여부 결정 필요

- [ ] `aroma/requirements.txt`에 `pyyaml`, `pillow` 추가 여부

- [ ] git 처리 방식: 단순 복사 vs `git mv` (커밋은 사용자 지시 시에만)

- [ ] aroma-plus의 코드 디렉토리 삭제 시점 (이전 검증 완료 후)

---

## 구현 완료 기준

- `python -m pytest tests/aroma/ -v` → 80/80 PASS
- Colab 노트북에 `AROMA_PLUS` 문자열 잔재 없음
- aroma-plus에 `scripts/`, `tests/`, `notebooks/` 디렉토리 없음
