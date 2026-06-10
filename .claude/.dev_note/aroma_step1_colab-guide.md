# AROMA Step 1 — Colab 실행 가이드 스크립트

## 사용할 skills: feature-dev

## 개요

Step 1 (`compute_complexity.py`)을 Google Colab Pro에서 실행하기 위한
Jupyter Notebook 가이드를 작성한다.
Phase 0 → Step 1 전체 파이프라인을 셀 단위로 실행할 수 있어야 함.

---

## 출력 파일

```
notebooks/
  colab_step1_guide.ipynb     Colab에서 바로 열고 실행 가능한 노트북
```

---

## 실행 환경

| 연산 | 자원 | 비고 |
|------|------|------|
| 형태학/컨텍스트 특징 추출 | **CPU** | `--num_workers` 병렬 |
| SAM 세그멘테이션 (fallback) | **GPU** (선택적) | GPU 없으면 Otsu 자동 사용 |
| MCI/CCI 계산, 정책 평가 | **CPU** | 단일 프로세스 |

> **권장 런타임**: T4 GPU (SAM 사용 시) / CPU 런타임 (SAM 미사용 시)

---

## 노트북 셀 구성

| 셀 | 내용 |
|----|------|
| 0. 제목/소개 | Markdown — 파이프라인 개요 |
| 1. Google Drive 마운트 | `from google.colab import drive` |
| 2. 저장소 클론 | AROMA (Phase 0) + AROMA_PLUS (Step 1-4) |
| 3. 의존성 설치 | scikit-learn, numpy, pyyaml, pillow |
| 4. GPU 감지 및 환경변수 설정 | `torch.cuda.is_available()` + AROMA_REF 등 |
| 5. 데이터셋 키 선택 | `DATASET_KEY = 'isp_LSM_1'` |
| 6. Phase 0 실행 | `!python $AROMA_REF/... --num_workers $NUM_WORKERS` |
| 7. Phase 0 출력 확인 | 생성 파일 목록 확인 |
| 8. Step 1 실행 | `!python $AROMA_SCRIPTS/compute_complexity.py ... --num_workers $NUM_WORKERS` |
| 9. 결과 확인 | complexity_report.json 출력 |
| 10. Ablation (선택) | --weight_mode equal/entropy_heavy/cluster_heavy |

---

## 환경변수 목록 (Colab 셀)

```python
import os
import torch

# GPU 감지 및 workers 설정
USE_GPU = torch.cuda.is_available()
NUM_WORKERS = 4 if not USE_GPU else 2  # GPU 런타임은 CPU 코어 적음
DEVICE = 'cuda' if USE_GPU else 'cpu'
os.environ['NUM_WORKERS'] = str(NUM_WORKERS)
os.environ['DEVICE']      = DEVICE
print(f"device={DEVICE}, num_workers={NUM_WORKERS}")

# aroma (Phase 0) 스크립트
os.environ['AROMA_REF']       = '/content/AROMA'

# aroma-plus (Step 1-4) 스크립트
os.environ['AROMA_SCRIPTS']   = '/content/AROMA_PLUS/scripts/aroma'
os.environ['AROMA_OUT']       = '/content/drive/MyDrive/data/Aroma/aroma_output'
os.environ['AROMA_DATA_BASE'] = '/content/drive/MyDrive/data/Aroma'
os.environ['DATASET_CONFIG']  = '/content/AROMA_PLUS/scripts/aroma/config/dataset_config.json'
```

---

## Colab 실행 규칙 준수 사항

- 환경변수 참조: `$VAR` (not `${VAR}`)
- 실행 prefix: `!python`
- 줄 이음: `\` 사용

---

## 참고

- `aroma_project_roadmap.md` — 전체 파이프라인 구조
- `aroma_step1_complexity-analysis.md` — Step 1 상세 명세
- `.claude/rules/colab-execution.md` — Colab 실행 규칙
