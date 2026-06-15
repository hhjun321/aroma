# AROMA Exp 1 (논문 Exp 1) — CASDA Baseline Comparison (Severstal)

---

## (사용할 skills: feature-dev)

## 개요

논문 Experiment 1: AROMA vs CASDA vs Random, Severstal Steel Defect Detection 데이터셋.

**핵심 질문**: CASDA가 원래 설계된 도메인(Severstal 철강)에서도 AROMA가 CASDA를 능가하는가?

모든 방법은 **동일한 copy-paste synthesis**를 사용하고 ROI selection 전략만 다름
(synthesis 개선이 아닌 ROI modeling 기여 분리를 위해).

참조:
- CASDA 원본 파이프라인: `D:\project\CASDA\CASDA\CASDA\`
- Severstal 데이터: `https://www.kaggle.com/c/severstal-steel-defect-detection/data`
- CASDA ROI 추출: `CASDA/scripts/extract_rois.py` (Stage A)

---

## 영향도 분석

### 이 기능이 변경하는 상태
- `$AROMA_OUT_SEVERSTAL/exp1/exp1_results.json`
- `$AROMA_OUT_SEVERSTAL/exp1/exp1_summary.md`

### 그 상태를 전제로 동작하는 기존 로직
- 없음 (평가 전용)

### Severstal 전용 파이프라인 신규 구성 필요

```
Severstal 데이터 준비
        ↓
AROMA Phase 0 (distribution_profiling.py)
        ↓
AROMA Step 1 (complexity_analysis.py)
        ↓
AROMA Step 2 (prompt_generation.py)
        ↓
AROMA Step 3 (roi_selection.py — pair-aware allocation)
        ↓
aroma_roi.json
        ↓
          ┌──────────────────────┐
          ↓                      ↓                       ↓
generate_aroma.py   generate_casda.py       generate_random.py
          ↓                      ↓                       ↓
  synthetic_aroma     synthetic_casda         synthetic_random
```

---

## 수정 내용

### 1. Severstal 데이터 준비

Severstal 구조 (Kaggle 표준):
```
train_images/   ← 1600×256px 철강 이미지
train.csv       ← RLE 마스크 레이블 (class 1~4)
test_images/
```

AROMA 파이프라인 입력 형식으로 변환 필요:
- defect instance별 crop + annotation JSON 생성
- AROMA가 기대하는 `images/`, `annotations/` 구조로 변환

### 2. `scripts/aroma/generate_casda.py` — 신규 작성 (Severstal 전용)

CASDA ROI selection을 AROMA copy-paste 엔진에 적용.

**CASDA ROI selection 로직** (CASDA Stage A 기준):
- `extract_rois.py`의 suitability score 기반 ROI 후보 생성
  - min_suitability=0.5 이상만 사용
  - 결함 마스크 + 배경 패치 조합 scoring
- 결함 클래스별 per_class_cap 적용 (class 1~4)
- 출력: AROMA `roi_selected.json` 호환 형식으로 변환

```python
# CASDA roi_metadata.csv → AROMA roi_selected.json 어댑터 필요
# roi_metadata 컬럼: patch_id, class_id, suitability_score, image_id, ...
```

**TODO**: CASDA `extract_rois.py` 코드 직접 확인 후 suitability score 계산 로직 파악 필요
→ `D:\project\CASDA\CASDA\scripts\extract_rois.py`

### 3. `scripts/aroma/experiments/exp1_casda_comparison.py` — 신규 작성

Exp 3 (`exp3_generation_quality.py`)와 동일 구조, CASDA 조건 추가.

```
비교 조건:
  Random ROI → copy-paste → synthetic_random
  CASDA ROI  → copy-paste → synthetic_casda
  AROMA ROI  → copy-paste → synthetic_aroma

평가:
  1. ROI Modeling Quality (Exp 2와 동일):
     - morphology_coverage, context_coverage, rare_pair_coverage, entropy, gini
  2. FID (Defect Patch Crop 기준):
     - FID(real_patch, random_patch)
     - FID(real_patch, casda_patch)
     - FID(real_patch, aroma_patch)
  3. Anomaly Detection (PaDiM):
     - 조건: baseline / random / casda / aroma
     - 지표: Image-level AUROC, Pixel-level AUROC, PRO Score (optional)
```

**출력 형식**:
```json
{
  "severstal": {
    "roi_quality": {
      "random": {"morphology_coverage": ..., "context_coverage": ..., "rare_pair_coverage": ..., "entropy": ..., "gini": ...},
      "casda":  {...},
      "aroma":  {...}
    },
    "fid": {
      "random": 68.4, "casda": 51.2, "aroma": 43.7,
      "n_real_patches": 500, "fid_unstable": false
    },
    "ad": {
      "baseline": {"image_auroc": 0.70, "pixel_auroc": 0.65, "pro": null},
      "random":   {"image_auroc": 0.74, "pixel_auroc": 0.69, "pro": null},
      "casda":    {"image_auroc": 0.79, "pixel_auroc": 0.73, "pro": null},
      "aroma":    {"image_auroc": 0.83, "pixel_auroc": 0.77, "pro": null}
    }
  }
}
```

### 4. `AROMA연구분석/colab_execute/exp1_execute.md` — Colab 실행 가이드

```
0단계: Severstal 데이터 준비 (Kaggle → Drive)
1단계: AROMA Phase 0 (distribution_profiling.py)
2단계: AROMA Step 1 (complexity_analysis.py)
3단계: AROMA Step 2 (prompt_generation.py)
4단계: AROMA Step 3 (roi_selection.py)
5단계: CASDA ROI 생성 (generate_casda.py)
6단계: Random ROI 생성 (generate_random.py)
7단계: AROMA ROI 생성 (generate_aroma.py)
8단계: Exp 1 평가 (exp1_casda_comparison.py --mode all)
```

---

## 수정 대상 파일

- `scripts/aroma/generate_casda.py` (신규, Severstal 전용)
- `scripts/aroma/experiments/exp1_casda_comparison.py` (신규)
- `AROMA연구분석/colab_execute/exp1_execute.md` (신규)

---

## TODO (구현 전 확인 필요)

1. **CASDA extract_rois.py suitability score 로직**: `D:\project\CASDA\CASDA\scripts\extract_rois.py` 직접 읽어 ROI selection 기준 파악
2. **Severstal → AROMA 변환기**: 1600×256 원본에서 결함 인스턴스 crop + annotation 생성 스크립트 필요
3. **PRO Score**: anomalib PaDiM에서 PRO 지원 여부 확인 (optional metric)
4. **Severstal class 정의**: class 1~4 결함 유형 → AROMA cluster_id 매핑 방식 결정

---

## 엣지 케이스

| 상황 | 처리 |
|------|------|
| Severstal class 4 (희귀 클래스) | CASDA per_class_cap 동일 적용 |
| FID patch n<50 | `fid_unstable: true` + 경고 |
| CASDA roi 어댑터 변환 실패 | 에러 + 종료 코드 1 |
| GPU 없이 AD 실행 | 에러 메시지 + 종료 |

---

## 테스트

CLAUDE.md: pytest 금지. Colab 직접 검증.

```python
# Severstal 데이터 구조 확인
import os, pathlib
severstal = pathlib.Path(os.environ["SEVERSTAL_DATA"])
print("train_images:", len(list((severstal / "train_images").glob("*.jpg"))))

# CASDA ROI 변환 확인
import json
with open(f"{os.environ['AROMA_OUT_SEVERSTAL']}/roi/severstal/casda_roi_selected.json") as f:
    casda_roi = json.load(f)
print(f"CASDA ROI 수: {len(casda_roi)}")

# Exp 1 실행
!python $AROMA_SCRIPTS/experiments/exp1_casda_comparison.py \
    --mode all \
    --random_synthetic_dir $AROMA_OUT_SEVERSTAL/synthetic_random \
    --casda_synthetic_dir  $AROMA_OUT_SEVERSTAL/synthetic_casda \
    --aroma_synthetic_dir  $AROMA_OUT_SEVERSTAL/synthetic_aroma \
    --real_data_dir        $SEVERSTAL_DATA \
    --output_dir           $AROMA_OUT_SEVERSTAL/exp1
```

기대:
- FID: aroma < casda < random
- AUROC (image + pixel): aroma > casda > random > baseline
- Claim 2 입증: CASDA home domain에서도 AROMA > CASDA
