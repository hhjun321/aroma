# AROMA Exp 2 — ROI 품질 비교 실행 가이드

**런타임**: CPU (GPU 불필요)  
**전제**: Step 3 및 Step 4가 4개 데이터셋(isp_LSM_1, mvtec_cable, visa_cashew, visa_pcb)에서 완료된 상태

---

## 환경변수 설정

```python
import os

# AROMA 기본 환경변수 (기존 셀에서 이미 설정된 경우 생략)
os.environ['AROMA_OUT']    = f"{os.environ['DRIVE']}/aroma_output"
os.environ['AROMA_SCRIPTS'] = "/content/CASDA/scripts/aroma"

# Exp 2 전용
os.environ['RANDOM_ROI_DIR'] = f"{os.environ['AROMA_OUT']}/roi_random"
os.environ['EXP2_OUT']       = f"{os.environ['AROMA_OUT']}/exp2"

print("AROMA_OUT     :", os.environ['AROMA_OUT'])
print("RANDOM_ROI_DIR:", os.environ['RANDOM_ROI_DIR'])
print("EXP2_OUT      :", os.environ['EXP2_OUT'])
```

---

## STEP 1 — Random ROI 기준선 생성

4개 데이터셋에 대해 `--sampling_strategy random`으로 ROI를 선택한다.  
AROMA ROI(`$AROMA_OUT/roi/`)를 덮어쓰지 않도록 **별도 디렉터리**(`roi_random/`)에 저장한다.

```python
DATASETS = ["isp_LSM_1", "mvtec_cable", "visa_cashew", "visa_pcb"]

for ds in DATASETS:
    print(f"\n=== {ds} ===")
    !python $AROMA_SCRIPTS/roi_selection.py \
        --profiling_dir  $AROMA_OUT/profiling/$ds \
        --prompts_dir    $AROMA_OUT/prompts/$ds \
        --sampling_strategy random \
        --top_k          200 \
        --seed           42 \
        --output_dir     $RANDOM_ROI_DIR/$ds
```

**출력 확인:**

```python
import json, os

for ds in DATASETS:
    path = f"{os.environ['RANDOM_ROI_DIR']}/{ds}/roi_selected.json"
    if os.path.exists(path):
        with open(path) as f:
            sel = json.load(f)
        print(f"{ds}: {len(sel)} ROI selected")
    else:
        print(f"{ds}: MISSING — {path}")
```

---

## STEP 2 — Exp 2 품질 비교 실행

AROMA ROI vs Random ROI의 5개 지표(Coverage × 3, Entropy, Gini)를 비교한다.

```python
!python $AROMA_SCRIPTS/experiments/exp2_roi_quality.py \
    --aroma_roi_dir  $AROMA_OUT/roi \
    --random_roi_dir $RANDOM_ROI_DIR \
    --dataset_keys   isp_LSM_1 mvtec_cable visa_cashew visa_pcb \
    --output_dir     $EXP2_OUT
```

---

## 결과 확인

### JSON 원본

```python
import json

with open(f"{os.environ['EXP2_OUT']}/exp2_results.json") as f:
    results = json.load(f)

for ds, data in results.items():
    print(f"\n{'='*40}")
    print(f"Dataset: {ds}")
    print(f"{'지표':<22} {'AROMA':>8} {'Random':>8} {'Δ':>8}")
    print("-" * 50)
    metrics = ["morphology_coverage", "context_coverage", "rare_pair_coverage", "entropy", "gini"]
    for m in metrics:
        a = data["aroma"][m]
        r = data["random"][m]
        delta = a - r
        sign = "↑" if (m != "gini" and delta > 0) or (m == "gini" and delta < 0) else ("↓" if delta != 0 else "=")
        print(f"  {m:<20} {a:>8.4f} {r:>8.4f} {delta:>+8.4f} {sign}")
```

### Markdown 요약

```python
with open(f"{os.environ['EXP2_OUT']}/exp2_summary.md") as f:
    print(f.read())
```

---

## 기대 결과

| 지표 | 기대 방향 |
|------|---------|
| morphology_coverage | AROMA ≥ Random |
| context_coverage | AROMA ≥ Random |
| rare_pair_coverage | AROMA > Random (deficit-aware 선택의 핵심 효과) |
| entropy | AROMA ≥ Random (더 균등한 cluster 분포) |
| gini | AROMA ≤ Random (낮을수록 균등) |

---

## 출력 파일

| 파일 | 내용 |
|------|------|
| `$EXP2_OUT/exp2_results.json` | 데이터셋 × 전략 × 5개 지표 수치 |
| `$EXP2_OUT/exp2_summary.md` | Markdown 비교 테이블 + delta 요약 |
| `$RANDOM_ROI_DIR/{ds}/roi_selected.json` | Random 전략 ROI 선택 결과 |
| `$RANDOM_ROI_DIR/{ds}/roi_candidates.json` | Random 전략 후보 (AROMA와 동일) |
