# AROMA Exp 3 (논문 Exp 2) — Cross-Domain 생성 품질 평가 (FID + AD)

---

## (사용할 skills: feature-dev)

## 개요

논문 Experiment 2: AROMA vs Random, Cross-Domain Generalization.

CASDA는 Severstal 전용(domain-specific) → 현재 4개 데이터셋에 미적용.
비교 조건 2가지: **Random ROI / AROMA ROI** (copy-paste synthesis 동일).

> AROMA Exp 1 (CASDA Comparison, Severstal) → 별도 devnote: `aroma_exp1_severstal-casda-comparison.md`

데이터셋: isp_LSM_1, mvtec_cable, visa_cashew, visa_pcb.

> **Copy-paste limitation (논문 명시 필수)**:
> "Copy-paste synthesis preserves the original defect appearance and therefore cannot generate novel defect morphologies.
> The objective of this study is not to improve the synthesis model itself, but to evaluate whether adaptive ROI modeling
> improves the quality of synthesized training data."

---

## 영향도 분석

### 이 기능이 변경하는 상태
- `$AROMA_OUT/exp3/exp3_results.json`
- `$AROMA_OUT/exp3/exp3_summary.md`

### 그 상태를 전제로 동작하는 기존 로직
- 없음 (평가 전용, 읽기만)

### 선행 조건

```
Step 3 ROI Selection (완료 ✅)
        ↓
aroma_roi.json         random_roi.json
        ↓                      ↓
generate_aroma.py     generate_random.py
        ↓                      ↓
  synthetic_aroma        synthetic_random
```

- `$AROMA_OUT/synthetic_aroma/{dataset}/` — Step 4 완료 ✅
- `$AROMA_OUT/synthetic_random/{dataset}/` — **generate_random.py 신규 실행 필요**

---

## 수정 내용

### 1. `scripts/aroma/generate_random.py` — 신규 작성

Random ROI selection (uniform random from roi_candidates.json) → copy-paste synthesis.
기존 `generate_defects.py`(= generate_aroma.py)와 동일 copy-paste 엔진, ROI 입력만 다름.

### 2. `scripts/aroma/generate_aroma.py` — 기존 generate_defects.py 리네이밍

기존 코드 그대로, 파일명만 변경 또는 심볼릭 래핑.

### 3. `scripts/aroma/experiments/exp3_generation_quality.py` — 신규 작성

`--mode {fid, ad, all}` 플래그로 분리.

#### Mode 1: FID — Defect Patch Crop 기반

```
Real Defect Patch Crop  vs  Synthetic Defect Patch Crop
FID(real_patch, random_patch) → FID_random
FID(real_patch, aroma_patch)  → FID_aroma
```

- **단위**: full image가 아닌 결함 영역 crop patch (annotation bbox 기준)
- 라이브러리: `torchmetrics.image.fid.FrechetInceptionDistance` (feature=2048, Inception v3 pool3)
- 소표본 경고: patch n<50이면 `fid_unstable: true` 플래그 기록

#### Mode 2: Anomaly Detection — PaDiM

```
조건 1: baseline — train/good 만 학습
조건 2: random   — train/good + synthetic_random
조건 3: aroma    — train/good + synthetic_aroma
```

- **모델**: PaDiM (anomalib)
- **백본**: ResNet-18, 4개 데이터셋 동일, seed=42
- **Train/Test 분리 원칙 (절대 불변)**:
  - Train = Real Normal + Synthetic (조건별)
  - Test  = Real Defect + Real Normal (Synthetic 절대 포함 금지)
- **지표**: Image-level AUROC, Pixel-level AUROC

#### 출력 형식

```json
{
  "isp_LSM_1": {
    "fid": {
      "random": 62.1, "aroma": 44.3,
      "n_real_patches": 95, "n_random_patches": 200, "n_aroma_patches": 200,
      "fid_unstable": false
    },
    "ad": {
      "baseline": {"image_auroc": 0.72, "pixel_auroc": 0.68},
      "random":   {"image_auroc": 0.75, "pixel_auroc": 0.71},
      "aroma":    {"image_auroc": 0.81, "pixel_auroc": 0.77}
    }
  }
}
```

`exp3_summary.md`: 데이터셋 × (FID 2열) + (AD 3조건 × image/pixel AUROC) 테이블 + delta 섹션.

### 4. `AROMA연구분석/colab_execute/exp3_execute.md` — Colab 실행 가이드

```
0단계: synthetic_random 생성
       !python $AROMA_SCRIPTS/generate_random.py ...
1단계: FID 평가 (CPU 가능)
       !python $AROMA_SCRIPTS/experiments/exp3_generation_quality.py --mode fid ...
2단계: AD 평가 (GPU 필요)
       !python $AROMA_SCRIPTS/experiments/exp3_generation_quality.py --mode ad ...
3단계: 결과 확인
```

---

## 수정 대상 파일

- `scripts/aroma/generate_random.py` (신규)
- `scripts/aroma/generate_aroma.py` (generate_defects.py 리네이밍)
- `scripts/aroma/experiments/exp3_generation_quality.py` (신규)
- `AROMA연구분석/colab_execute/exp3_execute.md` (신규)

---

## 엣지 케이스

| 상황 | 처리 |
|------|------|
| FID patch n<50 | `fid_unstable: true` + 값 기록 |
| annotation 없어 crop 불가 | 해당 이미지 skip + WARNING |
| 데이터셋 디렉터리 없음 | graceful skip + WARNING |
| GPU 없이 AD 모드 | 에러 + 종료 코드 1 |
| 생성 이미지 0개 | FID skip + WARNING |

---

## 테스트

CLAUDE.md: pytest 금지. Colab 직접 검증.

```python
# 0단계: 생성 이미지 수 확인
import os, pathlib
for method in ["synthetic_random", "synthetic_aroma"]:
    for ds in ["isp_LSM_1", "mvtec_cable", "visa_cashew", "visa_pcb"]:
        p = pathlib.Path(f"{os.environ['AROMA_OUT']}/{method}/{ds}/images")
        n = len(list(p.glob("*.jpg"))) if p.exists() else 0
        print(f"{method}/{ds}: {n} images")

# 1단계: FID
!python $AROMA_SCRIPTS/experiments/exp3_generation_quality.py \
    --mode fid \
    --random_synthetic_dir $AROMA_OUT/synthetic_random \
    --aroma_synthetic_dir  $AROMA_OUT/synthetic_aroma \
    --real_data_dir        $AROMA_DATA \
    --dataset_keys         isp_LSM_1 mvtec_cable visa_cashew visa_pcb \
    --output_dir           $AROMA_OUT/exp3

# 2단계: AD
!python $AROMA_SCRIPTS/experiments/exp3_generation_quality.py \
    --mode ad \
    --random_synthetic_dir $AROMA_OUT/synthetic_random \
    --aroma_synthetic_dir  $AROMA_OUT/synthetic_aroma \
    --real_data_dir        $AROMA_DATA \
    --dataset_keys         isp_LSM_1 mvtec_cable visa_cashew visa_pcb \
    --output_dir           $AROMA_OUT/exp3
```

기대:
- FID: aroma < random
- AUROC (image + pixel): aroma > random > baseline
