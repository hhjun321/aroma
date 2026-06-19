# AROMA Exp 4v2 — Supervised YOLO Detection 평가 실행 가이드

**목적**: Random ROI vs AROMA ROI 비교 — YOLOv8 지도학습 기반 결함 검출 성능 비교 (mAP)  
**런타임**: GPU 필수 (Colab Pro A100 권장)  
**전제**: Step 3 + Step 4가 4개 데이터셋에서 완료 (`synthetic/`, `synthetic_random/` 존재)

---

## 실험 설계 변경 배경

### 기존 exp4 (one-class AD) 의 문제

exp4 결과: **baseline(0.765) >> random(0.502) ≈ aroma(0.500)**

원인: train/good 폴더에 합성 결함 이미지를 추가하는 방식은 one-class AD 모델의 정상 분포를 오염시킨다.  
PatchCore 등 one-class 모델은 "정상 이미지만" 학습해 정상 분포를 모델링하는데, 합성 결함이 train/good에 섞이면 결함 패턴이 정상 분포에 포함되어 오히려 AUROC가 하락한다.

### exp4v2 (supervised detection) 로의 전환

합성 결함 이미지를 YOLO 지도학습용 레이블 데이터로 사용한다.

---

## 새로운 학습 패러다임: Baseline → Finetune

### 설계 원칙

이전 exp4v2 설계에서는 baseline 조건이 "배경 이미지만 fine-tune"(no defect label)으로 정의되어 있었다.  
이 방식은 세 조건 간 공정한 비교가 어렵다는 문제가 있다.

**새로운 패러다임** (CASDA 접근 방식과 동일):

1. **baseline**: real 결함 이미지만으로 YOLOv8을 학습 → checkpoint 저장
2. **random**: baseline checkpoint에서 출발해 random synthetic 데이터로 추가 fine-tune
3. **aroma**: baseline checkpoint에서 출발해 AROMA synthetic 데이터로 추가 fine-tune

### CASDA 접근 방식과의 정합성

CASDA 논문의 핵심 주장은 "합성 데이터 품질이 다운스트림 성능을 결정한다"는 것이다.  
공정한 비교를 위해 모든 조건이 동일한 출발점(baseline checkpoint)에서 시작해야 한다.

- random / aroma가 각자 독립적으로 pretrained 모델에서 학습을 시작하면, 학습 초기 상태 차이로 인한 노이즈가 포함된다.
- baseline checkpoint를 공유하면 "합성 데이터 품질의 차이"만 순수하게 측정할 수 있다.
- 이 구조는 CASDA의 "same baseline, different augmentation" 설계와 동일하다.

### 조건 요약

| 조건 | 학습 데이터 | 출발점 |
|------|------------|--------|
| baseline | real 결함 이미지만 | pretrained YOLOv8 |
| random | baseline checkpoint + random synthetic | baseline checkpoint |
| aroma | baseline checkpoint + AROMA synthetic | baseline checkpoint |

평가 지표: **mAP50** (mean Average Precision at IoU=0.50)  
평가셋: 항상 real 결함 이미지만 사용 (합성 이미지는 finetune train에만 포함).

---

## CLI 인자 변경사항

| 구 인자 | 신 인자 | 설명 |
|---------|---------|------|
| `--epochs 50` | `--baseline_epochs 50` | baseline (real 이미지) 학습 epoch 수 |
| (없음) | `--finetune_epochs 30` | random / aroma fine-tuning epoch 수 |
| `--val_frac` 기본값 | 0.5 → 0.3 | train≈64/val≈28 (n=92 기준, 이론적 최적) |
| `--max_synth_per_ds` | (신규) | synth:real 비율 제어. None=제한없음, 권장 128~192 (real-train의 2~3배) |

`--epochs`는 더 이상 사용하지 않는다.

---

## 선택적 CLI 인자 — 속도 최적화

### `--yolo_cache_dir` (선택)

YOLO 학습 데이터셋(real 이미지 기반 GT mask bbox 추출 결과)을 Drive에 캐싱한다.

```
--yolo_cache_dir $AROMA_OUT/yolo_cache
```

| 항목 | 내용 |
|------|------|
| 타입 | 선택 인자 (생략 시 캐시 없이 매 실행마다 bbox 추출) |
| 첫 실행 | YOLO dataset 빌드 후 지정 경로에 저장 |
| 이후 실행 | 캐시가 존재하면 GT mask bbox 추출 단계를 건너뛰어 빠르게 시작 |
| 권장 조합 | `--resume`과 함께 사용하면 최대 효율 (완료된 조건 skip + bbox 재추출 skip) |

### `--max_synth_per_ds` (선택)

synth:real 비율을 제어한다. 기본값 None(제한 없음 = 전량 사용).

| 항목 | 내용 |
|------|------|
| 타입 | 선택 인자 (생략 시 기존 동작 = synth 전량 사용) |
| 이론적 근거 | copy-paste augmentation 연구 권장 synth:real = 1:1 ~ 3:1. 현재 600:46=13:1은 과잉 주입으로 precision 붕괴(0.19) 유발 |
| 권장값 | real-train 수의 2~3배. val_frac=0.3 시 real-train≈64 → **128~192** |
| 효과 | 재합성 없이 비율만 조정해 즉시 precision 회복 가능 |

### Local image cache (자동)

Linux / Colab 환경에서는 CLI 인자 없이 자동으로 활성화된다.
Drive 이미지를 `/tmp`로 복사해 I/O 병목을 줄인다.

- **활성 조건**: Windows가 아닌 환경(`os.name != 'nt'`)에서 자동 적용
- **별도 설정 불필요**: 스크립트가 내부적으로 처리
- **효과**: Drive 직접 읽기 대비 I/O 속도 개선 (특히 이미지 수가 많은 데이터셋)

---

## 환경변수 설정

```python
import os

# AROMA 기본 환경변수 (기존 셀에서 설정된 경우 생략)
os.environ['AROMA_OUT']     = f"{os.environ['DRIVE']}/aroma_output"
os.environ['AROMA_SCRIPTS'] = "/content/AROMA/scripts/aroma"
os.environ['AROMA_DATA']    = f"{os.environ['DRIVE']}/Aroma"

# Exp 4v2 전용
os.environ['RANDOM_SYNTH_DIR'] = f"{os.environ['AROMA_OUT']}/synthetic_random"
os.environ['AROMA_SYNTH_DIR']  = f"{os.environ['AROMA_OUT']}/synthetic"
os.environ['EXP4V2_OUT']       = f"{os.environ['AROMA_OUT']}/exp4v2"

print("AROMA_DATA       :", os.environ['AROMA_DATA'])
print("RANDOM_SYNTH_DIR :", os.environ['RANDOM_SYNTH_DIR'])
print("AROMA_SYNTH_DIR  :", os.environ['AROMA_SYNTH_DIR'])
print("EXP4V2_OUT       :", os.environ['EXP4V2_OUT'])
```

---

## 패키지 설치 (최초 1회)

```python
!pip install ultralytics -q
!pip install opencv-python-headless -q  # cv2 for bbox extraction
```

> **주의**: ultralytics 설치 후 런타임 재시작(Runtime → Restart runtime)이 필요한 경우 환경변수를 다시 설정한다.

---

## GPU 확인

```python
!nvidia-smi
```

---

## 빠른 검증 실행 (mvtec_cable 단독)

전체 실행 전 파이프라인 정상 동작을 확인하기 위한 단축 실행.  
`baseline_epochs=30`, `finetune_epochs=20` 으로 소요 시간을 단축한다.

```python
!python $AROMA_SCRIPTS/experiments/exp4_v2_supervised_detection.py \
    --model yolov8n \
    --condition all \
    --dataset_keys mvtec_cable \
    --random_synthetic_dir $RANDOM_SYNTH_DIR \
    --aroma_synthetic_dir  $AROMA_SYNTH_DIR \
    --real_data_dir        $AROMA_DATA \
    --output_dir           $EXP4V2_OUT \
    --seed 42 \
    --baseline_epochs 30 \
    --finetune_epochs 20 \
    --max_synth_per_ds 128
```

> **소요 시간**: Colab Pro A100 기준 약 5-10분 (mvtec_cable 단독, 3조건).

### 권장 실행 명령 (캐시 + resume 조합)

`--yolo_cache_dir`와 `--resume`을 함께 사용하면 최대 효율로 실행된다.  
첫 실행에 YOLO dataset 빌드 후 캐시에 저장하고, 이후 실행부터는 bbox 재추출 없이 바로 학습을 시작한다.

```python
!python $AROMA_SCRIPTS/experiments/exp4_v2_supervised_detection.py \
    --model yolov8n \
    --condition all \
    --dataset_keys mvtec_cable \
    --random_synthetic_dir $RANDOM_SYNTH_DIR \
    --aroma_synthetic_dir  $AROMA_SYNTH_DIR \
    --real_data_dir        $AROMA_DATA \
    --output_dir           $EXP4V2_OUT \
    --yolo_cache_dir       $AROMA_OUT/yolo_cache \
    --baseline_epochs 30 \
    --finetune_epochs 20 \
    --resume
```

> **첫 실행**: YOLO dataset 빌드 + 학습 (캐시 저장). 이후 실행부터 캐시를 재사용해 bbox 추출 단계 생략.

---

## 전체 실행 (4개 데이터셋)

3조건 × 4데이터셋 = 12회 학습. baseline은 데이터셋당 1회만 실행되고 checkpoint가 random / aroma에 공유된다.

```python
!python $AROMA_SCRIPTS/experiments/exp4_v2_supervised_detection.py \
    --model yolov8n \
    --condition all \
    --dataset_keys isp_LSM_1 mvtec_cable visa_cashew visa_pcb \
    --random_synthetic_dir $RANDOM_SYNTH_DIR \
    --aroma_synthetic_dir  $AROMA_SYNTH_DIR \
    --real_data_dir        $AROMA_DATA \
    --output_dir           $EXP4V2_OUT \
    --seed 42 \
    --baseline_epochs 50 \
    --finetune_epochs 30
```

> **소요 시간**: Colab Pro A100 기준 약 1-2시간 (12회 학습).  
> 단일 조건만 실행하려면 `--condition random` 또는 `--condition aroma` 등으로 지정.

### 중단 후 재개 (--resume)

조건 완료 시마다 `exp4v2_results.json`에 incremental 저장. 중단 후 `--resume` 추가로 재개:

```python
!python $AROMA_SCRIPTS/experiments/exp4_v2_supervised_detection.py \
    --model yolov8n \
    --condition all \
    --dataset_keys isp_LSM_1 mvtec_cable visa_cashew visa_pcb \
    --random_synthetic_dir $RANDOM_SYNTH_DIR \
    --aroma_synthetic_dir  $AROMA_SYNTH_DIR \
    --real_data_dir        $AROMA_DATA \
    --output_dir           $EXP4V2_OUT \
    --seed 42 \
    --baseline_epochs 50 \
    --finetune_epochs 30 \
    --resume
```

> **resume 단위**: `(dataset, condition)` 페어. `mAP50`이 유효한 조건만 skip — `None`으로 저장된 실패 조건은 자동 재실행.

---

## 결과 확인

```python
import json
import os

with open(f"{os.environ['EXP4V2_OUT']}/exp4v2_results.json") as f:
    results = json.load(f)

CONDITIONS = ["baseline", "random", "aroma"]

for ds in sorted(results):
    print(f"\n=== {ds} ===")
    print(f"{'조건':<12}  {'mAP50':>10}")
    print("-" * 26)
    baseline_map = None
    random_map   = None
    aroma_map    = None
    for cond in CONDITIONS:
        val = results[ds].get(cond, {}).get("mAP50")
        cell = f"{val:.4f}" if isinstance(val, float) else "N/A"
        print(f"{cond:<12}  {cell:>10}")
        if cond == "baseline": baseline_map = val
        if cond == "random":   random_map   = val
        if cond == "aroma":    aroma_map    = val

    print()
    if isinstance(random_map, float) and isinstance(aroma_map, float):
        delta = round(aroma_map - random_map, 4)
        print(f"  Delta (AROMA - Random): {delta:+.4f}")
    if isinstance(baseline_map, float) and isinstance(aroma_map, float):
        delta_b = round(aroma_map - baseline_map, 4)
        print(f"  Delta (AROMA - Baseline): {delta_b:+.4f}")
```

비교표 형식 (컬럼 순서: baseline mAP50 | random mAP50 | aroma mAP50 | Δ(aroma-random)):

```python
print(f"\n{'데이터셋':<18}  {'baseline':>10}  {'random':>10}  {'aroma':>10}  {'Δ(A-R)':>10}")
print("-" * 66)
for ds in sorted(results):
    b = results[ds].get("baseline", {}).get("mAP50")
    r = results[ds].get("random",   {}).get("mAP50")
    a = results[ds].get("aroma",    {}).get("mAP50")
    delta = round(a - r, 4) if isinstance(a, float) and isinstance(r, float) else None
    print(
        f"{ds:<18}  "
        f"{(f'{b:.4f}' if isinstance(b, float) else 'N/A'):>10}  "
        f"{(f'{r:.4f}' if isinstance(r, float) else 'N/A'):>10}  "
        f"{(f'{a:.4f}' if isinstance(a, float) else 'N/A'):>10}  "
        f"{(f'{delta:+.4f}' if delta is not None else 'N/A'):>10}"
    )
```

---

## 출력 파일

| 파일 | 내용 |
|------|------|
| `$EXP4V2_OUT/exp4v2_results.json` | 데이터셋 × 조건별 mAP50 / mAP50-95 |
| `$EXP4V2_OUT/exp4v2_summary.md` | Markdown 비교표 (조건 × 데이터셋) + delta 섹션 |

---

## 주의사항

- **`--aroma_synthetic_dir`는 AROMA가 합성한 이미지 디렉터리를 가리켜야 함**: 이 경로 아래에는 `{dataset_key}/annotations.json`과 `{dataset_key}/images/`가 존재해야 한다. 경로가 틀리면 `annotations.json NOT FOUND` 에러와 함께 해당 조건의 `n_synth=0`이 되어 random/aroma 조건이 거부된다(`no_synth_annotations`). `--random_synthetic_dir`도 동일하게 random 합성 이미지 디렉터리를 가리켜야 한다.
- **AROMA `n_train=0`(또는 `n_synth=0`)이면 `annotations.json`의 `image_path` 필드를 확인**: 각 항목의 `image_path`가 실제 존재하는 파일을 가리켜야 한다(절대경로 또는 synth 디렉터리 기준 상대경로). 모든 항목이 resolve 실패하면 `had N entries but 0 resolved` 에러가 출력된다. 로그에서 `aroma train: N/600 synth imgs labeled`의 `N>0`을 확인할 것.
- **annotations.json의 `normal_image` 필드 필요**: 합성 결함 생성 시 배경 이미지 경로가 기록된 필드. mask가 없을 때 스크립트가 bbox 추출 시 이 경로를 사용한다.
- **합성 결함 bbox 추출 방식**: `annotations.json`에 `mask`/`roi_mask`/`mask_path` 필드나 `masks/{stem}.png` 파일이 있으면 그 mask에서 직접 bbox를 산출한다. mask가 없으면 composite 이미지 - background 이미지 차분 후 임계처리(threshold=8, 미검출 시 Otsu fallback)로 결함 영역의 bounding box를 자동 산출한다. (Poisson/alpha 블렌딩으로 차분 대비가 낮은 결함도 잡기 위해 임계값을 8로 낮추고 Otsu fallback을 추가함.)
- **real 결함 라벨이 0개이면 진단 로그 확인**: real defect 이미지에서 라벨이 하나도 추출되지 않으면 `_get_real: 0 labels ... (no_mask=.. bad_size=.. no_bbox=.. no_lines=..)` 에러가 출력된다. `no_mask`가 크면 mask 경로 매핑 문제, `no_bbox`가 크면 `min_area`(기본 50)가 해당 데이터셋 mask 해상도에 비해 너무 큰 것이다.
- **YOLO 캐시 schema_version=2**: bbox 추출 로직 변경으로 캐시 스키마가 v2로 상향되었다. `--yolo_cache_dir`로 기존(v1) 캐시가 있으면 자동으로 무효화되어 1회 재빌드된다.
- **baseline checkpoint 공유**: baseline 학습이 완료된 뒤 해당 가중치를 random / aroma가 출발점으로 사용한다. baseline 학습이 실패하면 이후 두 조건도 실행되지 않는다.
- **baseline_epochs=50 수렴 가능**: 데이터셋당 real 결함 약 수백 장 기준으로 50 epoch 내 수렴이 확인됨.
- **finetune_epochs=30**: baseline checkpoint 위에서 synthetic 데이터로 추가 학습. 30 epoch이면 충분히 수렴한다.
- **Test set 원칙**: 합성 이미지는 finetune train에만 포함. test/defect는 항상 real 이미지만 사용.
- **visa_pcb**: pcb4 단독 사용.
- **GPU 필수**: YOLO fine-tuning은 CUDA 없이 실행 불가.
