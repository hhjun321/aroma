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

> **resume 단위**: `(dataset, model, condition)` 트리플. `image_auroc`가 유효한 조건만 skip — `None`으로 저장된 실패 조건은 자동 재실행.  
> 특정 조건 강제 재실행: `exp4_results.json`에서 해당 키 삭제 후 `--resume`.

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
| `/content/tmp/exp4_checkpoints/{ds}/{model}/{cond}/` | 각 모델 체크포인트 (세션 내 임시, Drive 저장 안 함) |

---

## 콘솔 로그

anomalib/Lightning 내부 verbose 출력(progress bar, coreset index 로그 등)은 자동 억제.  
출력되는 로그는 아래 항목만:

```
05:06:23 [INFO] === AD dataset: isp_LSM_1 ===
05:06:23 [INFO] isp_LSM_1: train_normal=3678  test_good=1470  test_defect=95  masks_matched=95
05:13:43 [INFO] Local cache ready for isp_LSM_1: 439.7s  (train=3678 ...)
05:13:43 [INFO]   RESUME skip isp_LSM_1 / patchcore / baseline  (cached image_auroc=0.8432)
05:15:00 [INFO] patchcore / baseline done: image_auroc=0.8432  pixel_auroc=0.7211
```

이전 실행에서 coreset 로그가 과다 출력된 경우 → git pull 후 재실행.

---

## 로컬 캐시 (자동)

Colab 실행 시 real 이미지를 `/content/tmp/aroma_exp4_cache/`에 dataset당 **1회** 자동 복사.  
이후 모든 모델/조건 학습은 local 경로에서 읽어 Drive I/O 병목 해소.

정상 동작 로그:
```
Local cache: copying 5243 real images for isp_LSM_1 -> /content/tmp/aroma_exp4_cache/isp_LSM_1
Local cache ready for isp_LSM_1: 42.3s  (train=3678 test_good=1470 test_defect=95)
```

> **/content/tmp 경로 사용**: Ctrl+C 후 재실행해도 캐시 유지. Runtime 재시작 시만 소거.  
> 4개 데이터셋 전체 캐시 시 ~20GB 사용. /content/ 디스크는 ~107GB이므로 통상 문제없음.

---

## 주의사항

- **Test set 원칙**: 합성 이미지는 train에만 포함. test/defect는 항상 real 이미지만 사용.
- **GPU 필수**: anomalib 모델(PatchCore, SimpleNet, EfficientAD, RD++) 학습은 CUDA 없이 실행 불가.
- **visa_pcb**: pcb4 단독 사용.
- **anomalib 버전**: 최신 버전 기준 `Supersimplenet` 사용 (구버전 `Simplenet` 제거됨). 설치 후 `import anomalib; print(anomalib.__version__)` 확인 권장.
- **numpy**: anomalib은 numpy 1.x 기준 빌드. `numpy<2.0` 필수 (2.x에서 `_center` ImportError 발생).
