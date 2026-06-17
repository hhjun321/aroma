# AROMA Exp 4 — Downstream Anomaly Detection 평가 실행 가이드

**목적**: Random ROI vs AROMA ROI 2-way + Baseline 비교 (4개 AD 모델 × 3조건 × 4데이터셋)
**런타임**: GPU 필수 (Colab Pro A100 권장)
**전제**: Step 3 + Step 4가 4개 데이터셋에서 완료 (`synthetic/`, `synthetic_random/` 존재)

---

## 환경변수 설정

```python
import os

# AROMA 기본 환경변수 (기존 셀에서 설정된 경우 생략)
os.environ['AROMA_OUT']     = f"{os.environ['DRIVE']}/aroma_output"
os.environ['AROMA_SCRIPTS'] = "/content/AROMA/scripts/aroma"
os.environ['AROMA_DATA']    = f"{os.environ['DRIVE']}/Aroma"

# Exp 4 전용
os.environ['RANDOM_SYNTH_DIR'] = f"{os.environ['AROMA_OUT']}/synthetic_random"
os.environ['AROMA_SYNTH_DIR']  = f"{os.environ['AROMA_OUT']}/synthetic"
os.environ['EXP4_OUT']         = f"{os.environ['AROMA_OUT']}/exp4"

print("AROMA_DATA       :", os.environ['AROMA_DATA'])
print("RANDOM_SYNTH_DIR :", os.environ['RANDOM_SYNTH_DIR'])
print("AROMA_SYNTH_DIR  :", os.environ['AROMA_SYNTH_DIR'])
print("EXP4_OUT         :", os.environ['EXP4_OUT'])
```

---

## 패키지 설치 (최초 1회)

```python
# numpy 2.x + anomalib 충돌 방지 — 런타임 재시작 필수
!pip install "numpy<2.0" -q
!pip install anomalib -q
```

> **주의**: numpy 설치 후 런타임 재시작(Runtime → Restart runtime) 필수.

---

## GPU 확인

```python
!nvidia-smi
```

---

## Exp 4 실행

4개 모델 × 3조건 × 4데이터셋 = 48회 AD 학습.

```python
!python $AROMA_SCRIPTS/experiments/exp4_downstream_ad.py \
    --model all \
    --condition all \
    --dataset_keys isp_LSM_1 mvtec_cable visa_cashew visa_pcb \
    --random_synthetic_dir $RANDOM_SYNTH_DIR \
    --aroma_synthetic_dir  $AROMA_SYNTH_DIR \
    --real_data_dir        $AROMA_DATA \
    --output_dir           $EXP4_OUT \
    --seed 42
```

> **소요 시간**: Colab Pro A100 기준 약 3-4시간 (48회 학습).  
> 단일 모델만 실행하려면 `--model patchcore` 등으로 지정.

### 중단 후 재개 (--resume)

조건 완료 시마다 `exp4_results.json`에 incremental 저장. 중단 후 `--resume` 추가로 재개:

```python
!python $AROMA_SCRIPTS/experiments/exp4_downstream_ad.py \
    --model all \
    --condition all \
    --dataset_keys isp_LSM_1 mvtec_cable visa_cashew visa_pcb \
    --random_synthetic_dir $RANDOM_SYNTH_DIR \
    --aroma_synthetic_dir  $AROMA_SYNTH_DIR \
    --real_data_dir        $AROMA_DATA \
    --output_dir           $EXP4_OUT \
    --seed 42 \
    --resume
```

> **resume 단위**: `(dataset, model, condition)` 트리플. 조건 내부 crash는 해당 조건 처음부터 재실행.  
> 특정 조건 재실행 필요 시: `exp4_results.json`에서 해당 키 삭제 후 `--resume`.

---

## 결과 확인

```python
import json
import os

with open(f"{os.environ['EXP4_OUT']}/exp4_results.json") as f:
    results = json.load(f)

MODELS    = ["patchcore", "simplenet", "efficient_ad", "rd_plus_plus"]
CONDITIONS = ["baseline", "random", "aroma"]

for ds in sorted(results):
    print(f"\n=== {ds} ===")
    print(f"{'조건':<12}", end="")
    for m in MODELS:
        print(f"  {m:>14}", end="")
    print()
    print("-" * (12 + 18 * len(MODELS)))
    for cond in CONDITIONS:
        print(f"{cond:<12}", end="")
        for m in MODELS:
            val = results[ds].get(m, {}).get(cond, {}).get("image_auroc")
            cell = f"{val:.4f}" if isinstance(val, float) else "N/A"
            print(f"  {cell:>14}", end="")
        print()

    print("\n  Delta (AROMA - Random):")
    for m in MODELS:
        r_ia = results[ds].get(m, {}).get("random", {}).get("image_auroc")
        a_ia = results[ds].get(m, {}).get("aroma",  {}).get("image_auroc")
        if isinstance(r_ia, float) and isinstance(a_ia, float):
            delta = round(a_ia - r_ia, 4)
            print(f"    {m:<16} image_auroc delta={delta:+.4f}")
        else:
            print(f"    {m:<16} N/A")
```

---

## 출력 파일

| 파일 | 내용 |
|------|------|
| `$EXP4_OUT/exp4_results.json` | 데이터셋 × 모델 × 조건별 image_auroc / pixel_auroc |
| `$EXP4_OUT/exp4_summary.md` | Markdown 비교 테이블 (조건 × 모델) + delta 섹션 |
| `$EXP4_OUT/checkpoints/{ds}/{model}/{cond}/` | 각 모델 체크포인트 |

---

## 주의사항

- **Test set 원칙**: 합성 이미지는 train에만 포함. test/defect는 항상 real 이미지만 사용.
- **GPU 필수**: anomalib 모델(PatchCore, SimpleNet, EfficientAD, RD++) 학습은 CUDA 없이 실행 불가.
- **visa_pcb**: pcb4 단독 사용.
- **anomalib 버전**: 최신 버전 기준 `Supersimplenet` 사용 (구버전 `Simplenet` 제거됨). 설치 후 `import anomalib; print(anomalib.__version__)` 확인 권장.
- **numpy**: anomalib은 numpy 1.x 기준 빌드. `numpy<2.0` 필수 (2.x에서 `_center` ImportError 발생).
