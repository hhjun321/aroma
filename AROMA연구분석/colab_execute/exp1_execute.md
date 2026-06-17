# AROMA Exp 1 — CASDA Baseline Comparison (Severstal) 실행 가이드

**목적**: Random ROI vs CASDA ROI vs AROMA ROI 3-way 비교 (ROI 품질 + FID + PaDiM)  
**런타임**: ROI/FID 모드 = CPU 가능 | AD 모드 = GPU 필수  
**전제**: Severstal 데이터셋 Drive에 준비, CASDA Stage A 완료 (roi_metadata.csv 생성)

---

## 0단계 — 패키지 설치 (최초 1회)

```python
!pip install torchmetrics[image] anomalib -q
```

---

## 환경변수 설정

```python
import os

# AROMA 기본 환경변수 (기존 셀에서 설정된 경우 생략)
os.environ['AROMA_REF']     = "/content/AROMA"
os.environ['AROMA_SCRIPTS'] = "/content/AROMA/scripts/aroma"

# Severstal 데이터 경로
os.environ['SEVERSTAL_DATA']    = f"{os.environ['DRIVE']}/severstal"
# 디렉터리 구조:
#   $SEVERSTAL_DATA/train.csv
#   $SEVERSTAL_DATA/train_images/*.jpg
#   $SEVERSTAL_DATA/train_normal/*.jpg  ← 정상 이미지 (결함 없는 이미지 별도 추출)

# AROMA Severstal 출력 경로
os.environ['AROMA_OUT_SEVERSTAL'] = f"{os.environ['DRIVE']}/aroma_output_severstal"

# CASDA Stage A 출력 (roi_metadata.csv)
os.environ['CASDA_ROI_CSV'] = f"{os.environ['DRIVE']}/casda_output/roi_metadata.csv"

# Exp 1 전용
os.environ['EXP1_OUT']      = f"{os.environ['AROMA_OUT_SEVERSTAL']}/exp1"
os.environ['CASDA_ROI_DIR'] = f"{os.environ['AROMA_OUT_SEVERSTAL']}/roi_casda/severstal"
os.environ['RANDOM_ROI_DIR'] = f"{os.environ['AROMA_OUT_SEVERSTAL']}/roi_random/severstal"

print("AROMA_OUT_SEVERSTAL:", os.environ['AROMA_OUT_SEVERSTAL'])
print("SEVERSTAL_DATA     :", os.environ['SEVERSTAL_DATA'])
print("CASDA_ROI_CSV      :", os.environ['CASDA_ROI_CSV'])
print("EXP1_OUT           :", os.environ['EXP1_OUT'])
```

---

## 1단계 — Severstal → AROMA Phase 0 (distribution_profiling)

Severstal 파이프라인은 isp/mvtec/visa와 동일한 방식으로 실행.  
`train_images/`를 Phase 0 입력으로 사용.

```python
os.environ['SEVERSTAL_PHASE0_OUT'] = f"{os.environ['AROMA_OUT_SEVERSTAL']}/profiling/severstal"

!python $AROMA_SCRIPTS/../distribution_profiling.py \
    --image_dir   $SEVERSTAL_DATA/train_images \
    --train_csv   $SEVERSTAL_DATA/train.csv \
    --output_dir  $SEVERSTAL_PHASE0_OUT
```

---

## 2단계 — Step 1~3 (complexity → prompts → ROI selection)

```python
os.environ['PROFILING_DIR'] = f"{os.environ['AROMA_OUT_SEVERSTAL']}/profiling/severstal"
os.environ['PROMPTS_DIR']   = f"{os.environ['AROMA_OUT_SEVERSTAL']}/prompts/severstal"
os.environ['AROMA_ROI_DIR'] = f"{os.environ['AROMA_OUT_SEVERSTAL']}/roi/severstal"

# Step 1: MCI/CCI
!python $AROMA_SCRIPTS/compute_complexity.py \
    --profiling_dir $PROFILING_DIR \
    --output_dir    $PROFILING_DIR

# Step 2: Prompts
!python $AROMA_SCRIPTS/prompt_generation.py \
    --profiling_dir $PROFILING_DIR \
    --output_dir    $PROMPTS_DIR

# Step 3: AROMA ROI selection (pair-aware allocation)
!python $AROMA_SCRIPTS/roi_selection.py \
    --profiling_dir    $PROFILING_DIR \
    --prompts_dir      $PROMPTS_DIR \
    --sampling_strategy deficit_aware \
    --top_k            200 \
    --seed             42 \
    --output_dir       $AROMA_ROI_DIR
```

---

## 3단계 — CASDA ROI → AROMA 형식 변환

CASDA Stage A 출력 (`roi_metadata.csv`)을 AROMA `roi_selected.json`으로 변환.

```python
!python $AROMA_SCRIPTS/casda_roi_adapter.py \
    --metadata_csv    $CASDA_ROI_CSV \
    --output_dir      $CASDA_ROI_DIR \
    --min_suitability 0.5
```

**출력 확인:**

```python
import json, os

with open(f"{os.environ['CASDA_ROI_DIR']}/roi_selected.json") as f:
    casda_rois = json.load(f)
print(f"CASDA ROI 수: {len(casda_rois)}")

from collections import Counter
bg_types = Counter(r['cell_key'] for r in casda_rois)
print("background_type 분포:", dict(bg_types))
cluster_dist = Counter(r['cluster_id'] for r in casda_rois)
print("cluster_id (class) 분포:", dict(cluster_dist))
```

---

## 4단계 — Random ROI baseline 생성 (ROI 선택)

AROMA의 `roi_candidates.json`에서 균등 랜덤 샘플링.

```python
!python $AROMA_SCRIPTS/roi_selection.py \
    --profiling_dir     $PROFILING_DIR \
    --prompts_dir       $PROMPTS_DIR \
    --sampling_strategy random \
    --top_k             200 \
    --seed              42 \
    --output_dir        $RANDOM_ROI_DIR
```

---

## 5단계 — AROMA 합성 이미지 생성 (generate_defects.py)

```python
os.environ['AROMA_SYNTH_DIR']  = f"{os.environ['AROMA_OUT_SEVERSTAL']}/synthetic_aroma/severstal"

!python $AROMA_SCRIPTS/generate_defects.py \
    --roi_dir     $AROMA_ROI_DIR \
    --normal_dir  $SEVERSTAL_DATA/train_normal \
    --output_dir  $AROMA_SYNTH_DIR \
    --method      copy_paste \
    --n_per_roi   3 \
    --seed        42
```

---

## 6단계 — CASDA 합성 이미지 생성

```python
os.environ['CASDA_SYNTH_DIR'] = f"{os.environ['AROMA_OUT_SEVERSTAL']}/synthetic_casda/severstal"

!python $AROMA_SCRIPTS/generate_casda.py \
    --metadata_csv  $CASDA_ROI_CSV \
    --normal_dir    $SEVERSTAL_DATA/train_normal \
    --output_dir    $CASDA_SYNTH_DIR \
    --casda_roi_dir $CASDA_ROI_DIR \
    --n_per_roi     3 \
    --seed          42
```

---

## 7단계 — Random 합성 이미지 생성

```python
os.environ['RANDOM_SYNTH_DIR'] = f"{os.environ['AROMA_OUT_SEVERSTAL']}/synthetic_random/severstal"

!python $AROMA_SCRIPTS/generate_random.py \
    --candidates_json  $AROMA_ROI_DIR/roi_candidates.json \
    --normal_dir       $SEVERSTAL_DATA/train_normal \
    --output_dir       $RANDOM_SYNTH_DIR \
    --random_roi_dir   $RANDOM_ROI_DIR \
    --top_k            200 \
    --seed             42 \
    --n_per_roi        3
```

---

## 8단계 — Exp 1: ROI 품질 비교 (CPU 가능)

```python
!python $AROMA_SCRIPTS/experiments/exp1_casda_comparison.py \
    --mode roi \
    --aroma_roi_dir        $AROMA_ROI_DIR \
    --casda_roi_dir        $CASDA_ROI_DIR \
    --random_roi_dir       $RANDOM_ROI_DIR \
    --aroma_synthetic_dir  $AROMA_SYNTH_DIR \
    --casda_synthetic_dir  $CASDA_SYNTH_DIR \
    --random_synthetic_dir $RANDOM_SYNTH_DIR \
    --real_data_dir        $SEVERSTAL_DATA \
    --output_dir           $EXP1_OUT
```

---

## 9단계 — Exp 1: FID 평가 (CPU 가능)

```python
!python $AROMA_SCRIPTS/experiments/exp1_casda_comparison.py \
    --mode fid \
    --aroma_roi_dir        $AROMA_ROI_DIR \
    --casda_roi_dir        $CASDA_ROI_DIR \
    --random_roi_dir       $RANDOM_ROI_DIR \
    --aroma_synthetic_dir  $AROMA_SYNTH_DIR \
    --casda_synthetic_dir  $CASDA_SYNTH_DIR \
    --random_synthetic_dir $RANDOM_SYNTH_DIR \
    --real_data_dir        $SEVERSTAL_DATA \
    --output_dir           $EXP1_OUT
```

---

## 10단계 — Exp 1: AD 평가 (GPU 필수)

```python
!python $AROMA_SCRIPTS/experiments/exp1_casda_comparison.py \
    --mode ad \
    --aroma_roi_dir        $AROMA_ROI_DIR \
    --casda_roi_dir        $CASDA_ROI_DIR \
    --random_roi_dir       $RANDOM_ROI_DIR \
    --aroma_synthetic_dir  $AROMA_SYNTH_DIR \
    --casda_synthetic_dir  $CASDA_SYNTH_DIR \
    --random_synthetic_dir $RANDOM_SYNTH_DIR \
    --real_data_dir        $SEVERSTAL_DATA \
    --output_dir           $EXP1_OUT \
    --seed                 42 \
    --image_size           256
```

---

## 결과 확인

### JSON 원본

```python
import json, os

with open(f"{os.environ['EXP1_OUT']}/exp1_results.json") as f:
    results = json.load(f)

data = results["severstal"]

# ROI 품질
if "roi_quality" in data:
    print("\n=== ROI 품질 ===")
    roi = data["roi_quality"]
    print(f"{'방법':<10} {'morph':>8} {'ctx':>8} {'rare_pair':>10} {'entropy':>8} {'gini':>8}")
    for method in ["aroma", "casda", "random"]:
        if method in roi:
            m = roi[method]
            print(f"{method.upper():<10} {m['morphology_coverage']:>8.4f} "
                  f"{m['context_coverage']:>8.4f} {m['rare_pair_coverage']:>10.4f} "
                  f"{m['entropy']:>8.4f} {m['gini']:>8.4f}")

# FID
if "fid" in data:
    print("\n=== FID ===")
    fid = data["fid"]
    for method in ["aroma", "casda", "random"]:
        m = fid.get(method, {})
        print(f"  {method.upper():<8} FID={m.get('fid')}")

# AD
if "ad" in data:
    print("\n=== Anomaly Detection ===")
    ad = data["ad"]
    print(f"{'조건':<12} {'Image AUROC':>12} {'Pixel AUROC':>12}")
    for cond in ["baseline", "random", "casda", "aroma"]:
        if cond in ad:
            m = ad[cond]
            print(f"{cond.upper():<12} {m.get('image_auroc') or 'N/A':>12} "
                  f"{m.get('pixel_auroc') or 'N/A':>12}")
```

### Markdown 요약

```python
with open(f"{os.environ['EXP1_OUT']}/exp1_summary.md") as f:
    print(f.read())
```

---

## 기대 결과

| 지표 | 기대 방향 |
|------|---------|
| rare_pair_coverage | AROMA > CASDA > Random |
| FID | AROMA < CASDA < Random (낮을수록 좋음) |
| Image AUROC | AROMA > CASDA > Random > Baseline |
| Pixel AUROC | AROMA > CASDA > Random > Baseline |

**논문 Claim 2**: CASDA 원래 도메인(Severstal)에서도 AROMA > CASDA  

---

## 출력 파일

| 파일 | 내용 |
|------|------|
| `$EXP1_OUT/exp1_results.json` | 3-way ROI 품질 + FID + AD 수치 |
| `$EXP1_OUT/exp1_summary.md` | Markdown 비교 테이블 + delta 섹션 |
| `$EXP1_OUT/checkpoints/` | PaDiM 체크포인트 (조건별) |
| `$CASDA_ROI_DIR/roi_selected.json` | CASDA → AROMA 변환 결과 |
| `$RANDOM_ROI_DIR/roi_selected.json` | Random 선택 결과 |

---

## 주의사항

- **train_normal 디렉터리**: `$SEVERSTAL_DATA/train_normal/` = 결함 없는 이미지만 포함해야 함.  
  `train.csv`에서 `EncodedPixels`가 비어 있거나 `-1`인 이미지 ID를 추출하여 별도 디렉터리로 준비.

- **CASDA Stage A 전제**: `roi_metadata.csv`에 `roi_image_path` 컬럼이 존재하고 해당 경로에 ROI 크롭 이미지가 있어야 함 (CASDA 실행 시 `--save_patches` 옵션 필요).

- **AD 모드 GPU**: PaDiM 학습 시 GPU 메모리 8GB 이상 권장. Colab Pro A100 사용 권장.
