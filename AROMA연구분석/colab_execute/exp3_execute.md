# AROMA Exp 3 — Cross-Domain 생성 품질 평가 실행 가이드

**목적**: Random ROI vs AROMA ROI 2-way 비교 (FID + PaDiM AUROC)  
**런타임**: FID 모드 = CPU 가능 | AD 모드 = GPU 필수  
**전제**: Step 3 + Step 4가 4개 데이터셋에서 완료 (`synthetic_aroma/` 존재)

---

## 환경변수 설정

```python
import os

# AROMA 기본 환경변수 (기존 셀에서 설정된 경우 생략)
os.environ['AROMA_OUT']     = f"{os.environ['DRIVE']}/aroma_output"
os.environ['AROMA_SCRIPTS'] = "/content/CASDA/scripts/aroma"
os.environ['AROMA_DATA']    = "/content/drive/MyDrive/data/Aroma"

# Exp 3 전용
os.environ['RANDOM_SYNTH_DIR'] = f"{os.environ['AROMA_OUT']}/synthetic_random"
os.environ['AROMA_SYNTH_DIR']  = f"{os.environ['AROMA_OUT']}/synthetic_aroma"
os.environ['EXP3_OUT']         = f"{os.environ['AROMA_OUT']}/exp3"

print("AROMA_DATA       :", os.environ['AROMA_DATA'])
print("RANDOM_SYNTH_DIR :", os.environ['RANDOM_SYNTH_DIR'])
print("AROMA_SYNTH_DIR  :", os.environ['AROMA_SYNTH_DIR'])
print("EXP3_OUT         :", os.environ['EXP3_OUT'])
```

---

## 패키지 설치 (최초 1회)

```python
!pip install torchmetrics[image] anomalib -q
```

---

## 0단계 — Random 합성 이미지 생성

AROMA synthetic (`synthetic_aroma/`)은 Step 4에서 이미 완료.  
Random baseline synthetic만 생성한다.

### dataset_config.json에서 normal_dir 자동 조회

```python
import json

_cfg_path = "/content/CASDA/dataset_config.json"
with open(_cfg_path) as _f:
    DATASET_CONFIG = json.load(_f)

DATASETS = ["isp_LSM_1", "mvtec_cable", "visa_cashew", "visa_pcb"]

# 조회 결과 확인
for ds in DATASETS:
    img_dir = DATASET_CONFIG.get(ds, {}).get("image_dir", "(없음)")
    print(f"{ds:20s} → {img_dir}")
```

### Random 합성 실행

```python
for ds in DATASETS:
    cfg = DATASET_CONFIG.get(ds, {})
    normal_dir = cfg.get("image_dir", "")
    if not normal_dir:
        print(f"[SKIP] {ds}: dataset_config.json에 image_dir 없음")
        continue

    os.environ['NORMAL_DIR_DS']   = normal_dir
    os.environ['RANDOM_SYNTH_DS'] = f"{os.environ['RANDOM_SYNTH_DIR']}/{ds}"
    print(f"\n=== {ds} ===  normal_dir={normal_dir}")
    !python $AROMA_SCRIPTS/generate_random.py \
        --candidates_json $AROMA_OUT/roi/$ds/roi_candidates.json \
        --normal_dir      $NORMAL_DIR_DS \
        --output_dir      $RANDOM_SYNTH_DS \
        --top_k           200 \
        --seed            42 \
        --n_per_roi       3
```

**이미지 수 확인:**

```python
import pathlib

for ds in DATASETS:
    for method in ["synthetic_random", "synthetic_aroma"]:
        p = pathlib.Path(f"{os.environ['AROMA_OUT']}/{method}/{ds}/images")
        n = len(list(p.glob("*.jpg")) + list(p.glob("*.png"))) if p.exists() else 0
        print(f"{method}/{ds}: {n} images")
```

---

## 1단계 — FID 평가 (CPU 가능)

실제 결함 패치 vs 합성 결함 패치의 분포 거리 측정.

```python
!python $AROMA_SCRIPTS/experiments/exp3_generation_quality.py \
    --mode fid \
    --random_synthetic_dir $RANDOM_SYNTH_DIR \
    --aroma_synthetic_dir  $AROMA_SYNTH_DIR \
    --real_data_dir        $AROMA_DATA \
    --dataset_keys         isp_LSM_1 mvtec_cable visa_cashew visa_pcb \
    --output_dir           $EXP3_OUT \
    --seed                 42 \
    --device               cpu
```

**FID 결과 확인:**

```python
import json

with open(f"{os.environ['EXP3_OUT']}/exp3_results.json") as f:
    results = json.load(f)

print(f"{'데이터셋':<16} {'FID_Random':>12} {'FID_AROMA':>12} {'n_real':>8} {'unstable':>10}")
print("-" * 62)
for ds, data in sorted(results.items()):
    fd = data.get("fid", {})
    r  = fd.get("random", {})
    a  = fd.get("aroma", {})
    fid_r = f"{r['fid']:.2f}" if isinstance(r.get("fid"), float) else "N/A"
    fid_a = f"{a['fid']:.2f}" if isinstance(a.get("fid"), float) else "N/A"
    n_real = fd.get("n_real_patches", "?")
    unstable = "⚠" if (r.get("fid_unstable") or a.get("fid_unstable")) else ""
    print(f"{ds:<16} {fid_r:>12} {fid_a:>12} {str(n_real):>8} {unstable:>10}")
```

---

## 2단계 — AD 평가 (GPU 필수)

PaDiM 3조건 학습 및 AUROC 측정. **Colab Pro A100 권장.**

```python
!python $AROMA_SCRIPTS/experiments/exp3_generation_quality.py \
    --mode ad \
    --random_synthetic_dir $RANDOM_SYNTH_DIR \
    --aroma_synthetic_dir  $AROMA_SYNTH_DIR \
    --real_data_dir        $AROMA_DATA \
    --dataset_keys         isp_LSM_1 mvtec_cable visa_cashew visa_pcb \
    --output_dir           $EXP3_OUT \
    --seed                 42 \
    --image_size           256
```

**AD 결과 확인:**

```python
import json

with open(f"{os.environ['EXP3_OUT']}/exp3_results.json") as f:
    results = json.load(f)

print(f"{'데이터셋':<16} {'조건':<10} {'Image AUROC':>12} {'Pixel AUROC':>12}")
print("-" * 54)
for ds, data in sorted(results.items()):
    ad = data.get("ad", {})
    for cond in ["baseline", "random", "aroma"]:
        m  = ad.get(cond, {})
        ia = f"{m['image_auroc']:.4f}" if isinstance(m.get("image_auroc"), float) else "N/A"
        pa = f"{m['pixel_auroc']:.4f}" if isinstance(m.get("pixel_auroc"), float) else "N/A"
        print(f"{ds:<16} {cond:<10} {ia:>12} {pa:>12}")
    print()
```

---

## 3단계 — Markdown 요약 확인

```python
with open(f"{os.environ['EXP3_OUT']}/exp3_summary.md") as f:
    print(f.read())
```

---

## 기대 결과

| 지표 | 기대 방향 |
|------|---------|
| FID | AROMA < Random (낮을수록 좋음) |
| Image AUROC | AROMA > Random > Baseline |
| Pixel AUROC | AROMA > Random > Baseline |

---

## 출력 파일

| 파일 | 내용 |
|------|------|
| `$EXP3_OUT/exp3_results.json` | 데이터셋 × FID + AD 수치 |
| `$EXP3_OUT/exp3_summary.md` | Markdown 비교 테이블 + delta 섹션 |
| `$EXP3_OUT/checkpoints/{ds}/{cond}/` | PaDiM 체크포인트 |
| `$RANDOM_SYNTH_DIR/{ds}/images/` | Random 합성 이미지 |
| `$RANDOM_SYNTH_DIR/{ds}/annotations.json` | Random 합성 어노테이션 |

---

## 주의사항

- **AD 모드**: PaDiM 학습 4조건 × 4데이터셋 = 12회 학습. Colab Pro A100 기준 약 30-60분 소요.
- **FID 소표본 경고**: `fid_unstable: true`이면 real 패치 수가 50개 미만 — 값은 참고용.
- **visa_pcb**: pcb4 단독 사용 (dataset_config.json 및 평가 스크립트 모두 pcb4 기준).
- **Test set 원칙**: 합성 이미지는 train에만 포함. test/defect는 항상 real 이미지만.
