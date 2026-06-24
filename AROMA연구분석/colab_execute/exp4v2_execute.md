# AROMA Exp 4v2 — Supervised YOLO Detection 평가 실행 가이드

**목적**: Random ROI vs AROMA ROI 비교 — YOLOv8 지도학습 기반 결함 검출 성능 비교 (mAP)  
**런타임**: GPU 필수 (Colab Pro A100 권장)  
**전제**: Step 3 + Step 4가 데이터셋에서 완료 (`synthetic/`, `synthetic_random/` 존재)  
**대상**: GT mask 보유 `mvtec_cable visa_cashew mvtec_carpet mvtec_leather` (4개). ISP·visa_pcb 제외 — ISP는 GT mask 없음, visa_pcb는 작은 객체+큰 배경 void(배치 별도 연구). exp3 unsupervised AD는 전체 데이터셋

---

## 실험 설계 변경 배경

### 기존 exp4 (one-class AD) 의 문제

exp4 결과: **baseline(0.765) >> random(0.502) ≈ aroma(0.500)**

원인: train/good 폴더에 합성 결함 이미지를 추가하는 방식은 one-class AD 모델의 정상 분포를 오염시킨다.  
PatchCore 등 one-class 모델은 "정상 이미지만" 학습해 정상 분포를 모델링하는데, 합성 결함이 train/good에 섞이면 결함 패턴이 정상 분포에 포함되어 오히려 AUROC가 하락한다.

### exp4v2 (supervised detection) 로의 전환

합성 결함 이미지를 YOLO 지도학습용 레이블 데이터로 사용한다.

---

## 학습 패러다임: COCO에서 처음부터 (Scratch)

### 설계 원칙

세 조건 모두 COCO pretrained YOLOv8에서 출발하여 처음부터 학습한다.  
random / aroma는 real 결함 + 합성 데이터를 **처음부터 함께** 학습한다.

| 조건 | 학습 데이터 | 출발점 | epoch |
|------|------------|--------|-------|
| baseline | real 결함 이미지만 | COCO pretrained YOLOv8 | baseline_epochs |
| random | real 결함 + random synthetic | COCO pretrained YOLOv8 | baseline_epochs |
| aroma | real 결함 + AROMA synthetic | COCO pretrained YOLOv8 | baseline_epochs |

평가 지표: **mAP50** (mean Average Precision at IoU=0.50)  
평가셋: 항상 real 결함 이미지만 사용 (합성 이미지는 train에만 포함).

### 이전 설계(2-stage finetune)와의 차이

이전에는 baseline을 먼저 학습(50 epoch)하고, 그 best.pt에서 random/aroma를 추가 fine-tune(30 epoch)하는 2-stage 방식을 사용했다.

**2-stage finetune의 문제**: baseline best.pt는 real 데이터만으로 이미 수렴된 상태이다. 여기에 합성 데이터를 짧게 추가 학습시키면 기존에 굳어진 분포 위에 synth distribution을 억지로 끼워넣는 형태가 된다. 실험 결과 recall은 올라가지만 precision이 크게 하락하는 패턴이 나타났다(AROMA: precision -9.35pp, recall +10.51pp).

**Scratch 방식의 장점**: real과 synth를 처음부터 함께 학습하면 모델이 두 분포를 동시에 학습한다. 이것이 copy-paste augmentation의 표준 방식이며, 합성 데이터의 순수한 기여도를 측정하기에 더 적합하다.

---

## CLI 인자

| 인자 | 기본값 | 설명 |
|------|--------|------|
| `--baseline_epochs` | 300 | 전체 조건(baseline/random/aroma) 학습 epoch 수 |
| `--val_frac` | 0.3 | real defect 중 val 비율. train≈64/val≈28 (n=92 기준 최적) |
| `--max_synth_per_ds` | None | synth:real 비율 제어. 권장 128 (real-train의 2배) |
| `--synth_ratio` | None | synth 비율 지정 (예: 0.3, 0.5, 0.7). 지정 시 `--max_synth_per_ds` 무시 |
| `--patience` | 0 | EarlyStopping patience. 0=비활성(끝까지 훈련). N>0 → best mAP50-95 갱신 없이 N epoch 경과 시 조기 종료 |
| `--imgsz` | 256 | YOLO image size. 640 권장 (high-res defect 검출 정확도 개선) |
| `--seed` | 42 | 재현 가능한 train/val split 및 subsampling (단일 seed) |
| `--seeds` | None | multi-seed 반복 목록 (예: `--seeds 42 1 2`). 지정 시 seed별 독립 학습 후 조건별 mean±std + 95% CI 집계. 미지정 시 `[--seed]` 단일 실행 |
| `--class_mode` | single | **(severstal 전용)** `multi` 지정 시 nc=4(`c1`~`c4`) 클래스별 학습 + 결과에 `per_class` 추가. 타 데이터셋은 자동으로 무시되어 항상 single(`defect` 단일 클래스). 상세: "Severstal multi-class" 섹션 |

> `--finetune_epochs`는 제거됨. 모든 조건이 `--baseline_epochs`를 공유한다.

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
| 타입 | 선택 인자 (생략 시 synth 전량 사용) |
| 이론적 근거 | copy-paste augmentation 연구 권장 synth:real = 1:1 ~ 3:1 |
| 권장값 | real-train 수의 2~3배. val_frac=0.3 시 real-train≈64 → **128~192** |
| 효과 | 재합성 없이 비율만 조정해 즉시 실험 가능 |

### `--synth_ratio` (선택)

데이터셋별로 `n_synth_cap = max(1, int(n_real_train × ratio))` 를 동적 계산해 synth를 제한한다.  
지정 시 `--max_synth_per_ds` 는 무시된다.

| 항목 | 내용 |
|------|------|
| 타입 | 선택 인자 (생략 시 `--max_synth_per_ds` 또는 전량 사용) |
| 비율 기준 | n_real_train = `_split_defects()` 결과 (`val_frac`, `seed` 에 종속) |
| 권장값 | 0.5~2.0. 0.5 → n_real의 절반, 1.0 → n_real과 동수, 2.0 → n_real의 2배 |
| cap > 가용 synth | trim 없이 전량 사용, 로그에 `no trim` 표기 |

**예시**: `val_frac=0.3`, mvtec_cable real_train=64 기준

| `--synth_ratio` | n_synth_cap |
|----------------|------------|
| 0.3 | max(1, int(64 × 0.3)) = **19** |
| 0.5 | max(1, int(64 × 0.5)) = **32** |
| 1.0 | max(1, int(64 × 1.0)) = **64** |
| 2.0 | max(1, int(64 × 2.0)) = **128** |

> **주의**: `synth_ratio` 변경 후 `--resume` 사용 시 기존에 완료된 조건은 자동 재실행되지 않는다.  
> 재실험하려면 `exp4v2_results.json`에서 해당 데이터셋 항목을 삭제 후 재실행할 것.

### `--patience` (선택)

YOLOv8 EarlyStopping을 활성화한다. best mAP50-95 갱신 없이 N epoch이 경과하면 훈련을 조기 종료한다.

| 항목 | 내용 |
|------|------|
| 기본값 | **0 (비활성)** — epochs 전부 실행 (기존 동작 유지) |
| 기준 지표 | `mAP50-95` — mAP50이 아님에 주의 |
| best.pt 보존 | EarlyStopping 발동 전 최고 fitness epoch에서 이미 저장됨 → 조기 종료해도 best.pt 안전 |
| 권장값 | `baseline_epochs × 0.1~0.2`. epochs=300 → **30~60** (권장 50), epochs=50 → **5~10** (권장 15) |
| 최솟값 | ≥10 권장 — 소규모 데이터(42~70장)는 mAP 곡선 epoch-to-epoch 진동이 크므로 너무 작으면 오조기종료 |
| 비활성화 | `--patience 0` (기본) |

| `--baseline_epochs` | 권장 patience |
|---------------------|--------------|
| 50 | 15 |
| 100 | 20 |
| 300 | 50 |

> **주의**: EarlyStopping 기준은 `mAP50-95`이다. 실험 목표가 `mAP50` 최대화라면 두 지표의 peak epoch이 다를 수 있다.  
> 조건 간 공정 비교를 위해 모든 조건(baseline/random/aroma)에 동일한 patience가 적용된다.

### `--batch` / `--cache` / `--rect` (선택) — 학습 속도 인자

`model.train()`에 전달되는 YOLO 학습 속도 인자 3종. **세 인자 모두 기본값=Ultralytics 기본(현행)** 이라 미지정 시 기존 동작과 byte-identical. 전 조건(baseline/random/aroma)에 동일하게 적용되어 비교 공정성·val 셋 불변.

| 인자 | 기본값 | 설명 |
|------|--------|------|
| `--batch` | 16 | YOLO train batch size. A100에서 step 과다로 GPU 저활용 시 **64/128**로 상향 권장. `-1`=auto(GPU 60% VRAM)도 가능하나 method/조건간 실제 batch가 달라져 공정성·재현성을 해치므로 고정값 권장 |
| `--cache` | `""` (False) | 데이터 캐싱. `""`=비캐싱(매 epoch 디스크 재디코드). `ram`=RAM 캐시(최고속, 메모리 충분 시), `disk`=디스크 캐시, `True`=`ram` 별칭. I/O 병목 시 속도 향상 |
| `--rect` | False | rectangular training. 활성화 시 배치 내 종횡비 유사 이미지를 묶어 letterbox 회색 패딩을 최소화 → 속도 향상. 단 batch 내 이미지 크기 변동으로 결과가 미세하게 달라질 수 있고, Ultralytics가 shuffle을 비활성화한다(opt-in, 전 조건 동일 적용이라 공정성 불변) |

> **현행 보존**: 세 인자 미지정 시 batch=16 / cache off / rect off = 기존과 동일. 회귀 없음.
> **cache 파싱**: `""`/None → False, `ram`/`True` → `ram`, `disk` → `disk`.
> **RAM 부족**: `--cache ram` OOM 시 `--cache disk`로 폴백.

```python
!python $AROMA_SCRIPTS/experiments/exp4_v2_supervised_detection.py \
    --model yolov8n \
    --condition all \
    --dataset_keys mvtec_cable visa_cashew mvtec_carpet mvtec_leather \
    --random_synthetic_dir $RANDOM_SYNTH_DIR \
    --aroma_synthetic_dir  $AROMA_SYNTH_DIR \
    --real_data_dir        $AROMA_DATA \
    --output_dir           $EXP4V2_OUT \
    --yolo_cache_dir       $AROMA_OUT/yolo_cache \
    --imgsz 640 \
    --val_frac 0.3 \
    --synth_ratio 0.5 \
    --baseline_epochs 300 \
    --patience 50 \
    --batch 64 \
    --cache ram \
    --rect \
    --resume
```

> `--batch 64 --cache ram --rect` → step 수 감소 + 디스크 재디코드 제거 + 패딩 연산 절감으로 학습 속도 개선.  
> Severstal류 고종횡비(1600×256, 6.25:1) 이미지에서 `--rect` 효과가 특히 크다(정사각 letterbox 시 ~84% 회색 패딩).

### Local image cache (자동)

Linux / Colab 환경에서는 CLI 인자 없이 자동으로 활성화된다.  
Drive 이미지를 `/tmp`로 복사해 I/O 병목을 줄인다.

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
`baseline_epochs=20`으로 소요 시간을 단축한다.

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
    --imgsz 640 \
    --baseline_epochs 20 \
    --max_synth_per_ds 128
```

> **소요 시간**: Colab Pro A100 기준 약 5-10분 (mvtec_cable 단독, 3조건).

### 권장 실행 명령 (캐시 + resume 조합)

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
    --imgsz 640 \
    --baseline_epochs 20 \
    --max_synth_per_ds 128 \
    --resume
```

---

## 전체 실행 (4개 데이터셋 — GT mask 보유만)

3조건 × 4데이터셋 = 12회 학습. 모든 조건이 독립적으로 COCO에서 출발한다.

> ⚠️ **ISP·visa_pcb 제외**: (1) ISP는 GT 결함 mask가 없어 Otsu fallback이 이미지의 62~85%를 결함으로 잡음 → defect_bbox·val GT 모두 full-image라 supervised detection 부적합. (2) visa_pcb는 작은 PCB 객체 + 큰 배경 void 구조라 결함이 배경에 paste되는 배치 문제 → 별도 연구 주제. 둘 다 **exp3 unsupervised AD(image-level)** 로만 평가. exp4v2 대상 = GT mask 보유 + full-frame `mvtec_cable visa_cashew mvtec_carpet mvtec_leather` (carpet/leather는 텍스처 — 배경 void 없어 배치 자동 정상).

```python
!python $AROMA_SCRIPTS/experiments/exp4_v2_supervised_detection.py \
    --model yolov8n \
    --condition all \
    --dataset_keys mvtec_cable visa_cashew mvtec_carpet mvtec_leather \
    --random_synthetic_dir $RANDOM_SYNTH_DIR \
    --aroma_synthetic_dir  $AROMA_SYNTH_DIR \
    --real_data_dir        $AROMA_DATA \
    --output_dir           $EXP4V2_OUT \
    --seed 42 \
    --imgsz 640 \
    --val_frac 0.3 \
    --synth_ratio 0.5 \
    --baseline_epochs 300
```

> **소요 시간**: Colab Pro A100 기준 약 3-5시간 (12회 학습, epochs=300).  
> 단일 조건만 실행하려면 `--condition random` 또는 `--condition aroma` 등으로 지정.

### `--synth_ratio` 사용 예시

`--max_synth_per_ds` 대신 비율 기반으로 synth 수를 제어한다.

```python
!python $AROMA_SCRIPTS/experiments/exp4_v2_supervised_detection.py \
    --model yolov8n \
    --condition all \
    --dataset_keys mvtec_cable visa_cashew mvtec_carpet mvtec_leather \
    --random_synthetic_dir $RANDOM_SYNTH_DIR \
    --aroma_synthetic_dir  $AROMA_SYNTH_DIR \
    --real_data_dir        $AROMA_DATA \
    --output_dir           $EXP4V2_OUT \
    --seed 42 \
    --imgsz 640 \
    --val_frac 0.3 \
    --synth_ratio 1.0 \
    --baseline_epochs 300
```

> `--synth_ratio 1.0` → 데이터셋마다 real_train 수와 동일한 양의 synth 사용.  
> `--synth_ratio 2.0` → real_train의 2배 (mvtec_cable 기준 64 real → synth 최대 128).  
> `--max_synth_per_ds`와 동시에 지정하면 `--synth_ratio`가 우선 적용된다.

### `--patience` 사용 예시

EarlyStopping을 활성화해 불필요한 후반 epoch을 건너뛴다.

```python
!python $AROMA_SCRIPTS/experiments/exp4_v2_supervised_detection.py \
    --model yolov8n \
    --condition all \
    --dataset_keys mvtec_cable visa_cashew mvtec_carpet mvtec_leather \
    --random_synthetic_dir $RANDOM_SYNTH_DIR \
    --aroma_synthetic_dir  $AROMA_SYNTH_DIR \
    --real_data_dir        $AROMA_DATA \
    --output_dir           $EXP4V2_OUT \
    --seed 42 \
    --imgsz 640 \
    --val_frac 0.3 \
    --synth_ratio 0.5 \
    --baseline_epochs 300 \
    --patience 50
```

> `--patience 50` → best mAP50-95 갱신 없이 50 epoch 경과 시 조기 종료 (epochs=300의 약 17%).  
> 소규모 데이터 기준 평균 150~250 epoch에서 종료 예상 (약 20~50% 시간 절약).  
> `--patience 0` (기본) 이면 300 epoch 전부 실행.

### Multi-seed 반복 + mean±std 집계 (--seeds)

단일 seed 결과는 작은 val(18~42장)에서 노이즈가 커 AROMA vs RANDOM delta(평균 +1.7pp)의 통계적 유의성을 판정하기 어렵다. `--seeds`로 여러 seed를 반복하면 seed별 독립 train/val split·synth subsample·YOLO train이 수행되고, 조건별 **mean ± std + 95% CI**가 자동 집계된다.

```python
!python $AROMA_SCRIPTS/experiments/exp4_v2_supervised_detection.py \
    --model yolov8n \
    --condition all \
    --dataset_keys mvtec_cable visa_cashew mvtec_carpet mvtec_leather \
    --random_synthetic_dir $RANDOM_SYNTH_DIR \
    --aroma_synthetic_dir  $AROMA_SYNTH_DIR \
    --real_data_dir        $AROMA_DATA \
    --output_dir           $EXP4V2_OUT \
    --yolo_cache_dir       $AROMA_OUT/yolo_cache \
    --imgsz 640 \
    --val_frac 0.3 \
    --synth_ratio 0.5 \
    --baseline_epochs 300 \
    --patience 50 \
    --seeds 42 1 2 \
    --resume
```

| 항목 | 내용 |
|------|------|
| 동작 | 각 seed로 전체 (datasets × models × conditions) 학습을 독립 수행 후 집계 |
| per-seed 결과 | `$EXP4V2_OUT/_seeds/seed{N}/exp4v2_results.json` (각자 `--resume` 독립 적용) |
| 최종 집계 | `$EXP4V2_OUT/exp4v2_results.json` — top-level은 **평균값**(기존 plot/reader 그대로 동작) + `per_seed{}`·`std{}`·`ci95{}`·`n_seeds` 추가 |
| checkpoint | `$EXP4V2_OUT/_seeds/seed{N}/{ds}/{model}/{cond}/weights/best.pt` (seed별 격리) |
| std/CI | sample std(ddof=1, n_seeds<2면 0), 95% CI는 t-분포(scipy 있으면) 또는 소표본 t 임계표 fallback |
| 일부 seed 실패 | 성공한 seed만으로 집계, `n_seeds`에 실제값 기록, 로그 경고 |

> **소요 시간**: 단일 실행 × seed 수. 3 seed(42,1,2) 전체 → 단일(약 3~5시간)의 약 **3배 ≈ 9~15시간**.  
> A100 세션 한도를 고려해 `--resume`을 항상 함께 사용하면 seed 단위·조건 단위 모두 중단 지점부터 재개된다.  
> `--seeds 42`(또는 미지정)이면 기존 단일-seed 동작과 동일(per_seed 1개, std=0).
>
> ⚠️ **디스크 누적**: checkpoint가 `seed × ds × cond` 만큼 누적된다(예: 3 seed × 4 ds × 3 cond = **36개 best.pt**). 단일 실행 대비 Drive 용량이 seed 수배로 증가. 집계(`exp4v2_results.json`) 완료 후 `_seeds/*/{ds}/{model}/{cond}/weights/`를 정리하면 용량 회수 가능(원본 per-seed 수치는 `_seeds/seed{N}/exp4v2_results.json`에 보존됨).

#### multi-seed 결과 확인 (mean ± std + 95% CI)

```python
import json, os

with open(f"{os.environ['EXP4V2_OUT']}/exp4v2_results.json") as f:
    results = json.load(f)

CONDS = ["baseline", "random", "aroma"]

for ds in sorted(results):
    for model in sorted(results[ds]):
        arms = results[ds][model]
        n_seeds = next((arms[c].get("n_seeds") for c in CONDS if c in arms), "?")
        print(f"\n=== {ds} / {model}  (n_seeds={n_seeds}) ===")
        print(f"{'조건':<10}  {'map50 (mean±std)':>22}  {'95% CI':>20}")
        print("-" * 56)
        for c in CONDS:
            cell = arms.get(c, {})
            m   = cell.get("map50")
            sd  = (cell.get("std") or {}).get("map50")
            ci  = (cell.get("ci95") or {}).get("map50")
            if isinstance(m, float):
                ms = f"{m:.4f} ± {sd:.4f}" if isinstance(sd, float) else f"{m:.4f}"
                cis = f"[{ci[0]:.4f}, {ci[1]:.4f}]" if isinstance(ci, list) and ci[0] is not None else "N/A"
            else:
                ms, cis = "N/A", "N/A"
            print(f"{c:<10}  {ms:>22}  {cis:>20}")

        a, r = arms.get("aroma", {}), arms.get("random", {})
        if isinstance(a.get("map50"), float) and isinstance(r.get("map50"), float):
            d = (a["map50"] - r["map50"]) * 100
            sa = (a.get("std") or {}).get("map50", 0.0)
            sr = (r.get("std") or {}).get("map50", 0.0)
            comb = ((sa ** 2 + sr ** 2) ** 0.5) * 100
            verdict = "유의 가능성" if abs(d) > 2 * comb else "noise 범위"
            print(f"\n  Delta(AROMA-Random) map50: {d:+.2f}pp  (combined std {comb:.2f}pp) -> {verdict}")
```

> **유의성 rough 판정**: `|delta| > 2 × combined_std` 이면 noise floor 밖일 가능성. 정밀 검정은 `per_seed`의 seed별 값으로 paired t-test 등을 별도 수행.

### 중단 후 재개 (--resume)

조건 완료 시마다 `exp4v2_results.json`에 incremental 저장. 중단 후 `--resume` 추가로 재개:

```python
!python $AROMA_SCRIPTS/experiments/exp4_v2_supervised_detection.py \
    --model yolov8n \
    --condition all \
    --dataset_keys mvtec_cable visa_cashew mvtec_carpet mvtec_leather \
    --random_synthetic_dir $RANDOM_SYNTH_DIR \
    --aroma_synthetic_dir  $AROMA_SYNTH_DIR \
    --real_data_dir        $AROMA_DATA \
    --output_dir           $EXP4V2_OUT \
    --seed 42 \
    --imgsz 640 \
    --val_frac 0.3 \
    --synth_ratio 0.5 \
    --baseline_epochs 300 \
    --resume
```

> **resume 단위**: `(dataset, condition)` 페어. `mAP50`이 유효한 조건만 skip — `None`으로 저장된 실패 조건은 자동 재실행.

---

## Severstal multi-class (`--class_mode multi`)

> ⚠️ **본 섹션은 위 4개 GT-mask 데이터셋과 별개**로, Severstal 4-class 결함을 **클래스별로** 학습/평가하는 실험이다. `--class_mode multi`는 **severstal에만** 적용되며(타 데이터셋은 무시), `nc=4`(`c1`~`c4`)로 학습해 결함 클래스별 성능(특히 희귀 클래스의 AROMA vs Random 기여)을 분리 측정한다. 미지정 시 기본 `single`(`defect` 단일 클래스)로 기존 동작과 동일.

```python
!python $AROMA_SCRIPTS/experiments/exp4_v2_supervised_detection.py \
    --model yolov8n \
    --condition all \
    --dataset_keys severstal \
    --class_mode multi \
    --random_synthetic_dir $RANDOM_SYNTH_DIR \
    --aroma_synthetic_dir  $AROMA_SYNTH_DIR \
    --real_data_dir        $AROMA_DATA \
    --output_dir           $EXP4V2_OUT/severstal_multi \
    --seed 42 \
    --val_frac 0.3 \
    --synth_ratio 0.5 \
    --imgsz 640 \
    --baseline_epochs 300 \
    --patience 50 \
    --cache ram \
    --batch 64 \
    --rect \
    --yolo_cache_dir $AROMA_OUT/yolo_cache \
    --resume
```

| 항목 | 내용 |
|------|------|
| 적용 범위 | `severstal`만. 타 데이터셋에 `--class_mode multi`를 줘도 single로 처리 |
| 클래스 | `nc=4`, names `c1`/`c2`/`c3`/`c4` (Severstal defect class 1~4) |
| top-level 지표 | 기존과 동일(`map50` 등)이며 **4개 클래스의 macro-average** |
| **`per_class` 추가** | 각 조건 결과에 `per_class: {c1: {map50, map50_95, precision, recall}, ...}` 기록. val GT에 등장한 클래스만 포함(검출 전무 시 빈 dict) |
| multi-seed | `per_class`도 클래스별 mean으로 집계. per-class std는 미추적(전체 수치는 seed별 JSON 참조) |
| summary.md | 각 model 표 아래에 **`per-class map50` 표**(rows=class, cols=조건 + Δ(A-R)) 자동 출력 |

> **주의**: `--cache ram`은 `rem` 등 오타 시 자동으로 비캐싱(False)으로 폴백된다. `--imgsz`는 32의 배수여야 하며(아니면 Ultralytics가 올림 보정), 640 권장.

### per-class 결과 확인 (multi 모드)

```python
import json, os

with open(f"{os.environ['EXP4V2_OUT']}/severstal_multi/exp4v2_results.json") as f:
    results = json.load(f)

CONDS = ["baseline", "random", "aroma"]
arms = results["severstal"]["yolov8n"]

# 클래스 union (등장 클래스만)
classes = sorted({c for cond in CONDS for c in (arms.get(cond, {}).get("per_class") or {})})

print(f"{'class':<6}  " + "  ".join(f"{c:>10}" for c in CONDS) + f"  {'Δ(A-R)':>10}")
print("-" * 56)
for cls in classes:
    row = []
    vals = {}
    for cond in CONDS:
        v = ((arms.get(cond, {}).get("per_class") or {}).get(cls) or {}).get("map50")
        vals[cond] = v if isinstance(v, float) else None
        row.append(f"{v:.4f}" if isinstance(v, float) else "N/A")
    a, r = vals.get("aroma"), vals.get("random")
    d = f"{a - r:+.4f}" if isinstance(a, float) and isinstance(r, float) else "N/A"
    print(f"{cls:<6}  " + "  ".join(f"{c:>10}" for c in row) + f"  {d:>10}")
```

> 희귀 클래스(예: Severstal Class2)의 `Δ(A-R)`가 양수로 크면 AROMA 합성이 그 클래스 검출에 기여했다는 신호. 단일 seed는 노이즈가 크므로 `--seeds`로 반복해 판정할 것.

---

## CASDA를 4번째 조건으로 추가 (CASDA vs AROMA 공정 비교, Severstal)

> ⚠️ **본 섹션은 Severstal multi-class 전용**이며 `baseline / random / casda / aroma` **4조건**을 동시 비교한다.
> 세 합성 조건(random/casda/aroma)은 **동일 copy-paste 엔진**을 쓰고 **ROI 선택 전략만 다르다** → AROMA의 ROI 모델링 기여를 격리.
> CASDA 합성은 `--casda_synthetic_dir`로 주입하며, 이 경로 아래 `{dataset_key}/annotations.json`이 존재해야 한다(`generate_casda.py`가 생성).
> CASDA Drive 루트는 `/content/drive/MyDrive/data/Severstal`로 **AROMA Drive 루트(`.../data/Aroma`)와 다르다.** 둘 다 Colab에 마운트되어야 한다.

### 공정성 불변식 (load-bearing)

- `--synth_ratio` 동일 적용 → random/casda/aroma 모두 동일한 synth cap(`max(1, int(n_real_train × ratio))`). 결과 JSON의 `n_synth_train`이 세 조건 **동일**해야 공정. 다르면 cap 누락 의심.
- `--n_per_roi` 동일 → 세 생성기 동일값.
- CASDA native(결정 A): `min_suitability=0.5`(CASDA 기본), `per_class_cap` 미지정(=None, 무제한). class balance 통제 안 함. c4/c2 starvation은 **정당한 발견**이며 `n_synth_per_class` 로깅/JSON 필드로 투명화된다.

### 환경변수 (CASDA 전용 추가)

```python
import os

# AROMA 측 (기존 셀에서 설정되었으면 생략)
os.environ['AROMA_OUT']     = f"{os.environ['DRIVE']}/aroma_output"
os.environ['AROMA_SCRIPTS'] = "/content/AROMA/scripts/aroma"
os.environ['AROMA_DATA']    = f"{os.environ['DRIVE']}/Aroma"

# CASDA 측 — Drive 루트가 AROMA와 다름에 주의
os.environ['CASDA_DRIVE']  = "/content/drive/MyDrive/data/Severstal"
os.environ['SCRIPTS']      = "/content/CASDA/scripts"                 # CASDA Stage A 스크립트
os.environ['TRAIN_IMAGES'] = f"{os.environ['CASDA_DRIVE']}/train_images"
os.environ['TRAIN_CSV']    = f"{os.environ['CASDA_DRIVE']}/train.csv"
os.environ['ROI_DIR']      = f"{os.environ['CASDA_DRIVE']}/roi_patches_v5.1"
os.environ['NORMAL_DIR']   = f"{os.environ['AROMA_DATA']}/severstal/train/good"  # 배경(정상) 이미지

# 합성 출력 디렉터리 (3 조건 모두 동일 엔진)
os.environ['RANDOM_SYNTH_DIR'] = f"{os.environ['AROMA_OUT']}/synthetic_random"
os.environ['CASDA_SYNTH_DIR']  = f"{os.environ['AROMA_OUT']}/synthetic_casda"
os.environ['AROMA_SYNTH_DIR']  = f"{os.environ['AROMA_OUT']}/synthetic"
os.environ['EXP4V2_OUT']       = f"{os.environ['AROMA_OUT']}/exp4v2"

# 합성 공통 파라미터 (3 조건 동일)
os.environ['N_PER_ROI'] = "3"

print("CASDA_DRIVE     :", os.environ['CASDA_DRIVE'])
print("ROI_DIR         :", os.environ['ROI_DIR'])
print("CASDA_SYNTH_DIR :", os.environ['CASDA_SYNTH_DIR'])
```

### Cell A — CASDA Stage A (CPU, clone + ROI 추출)

> ROI 추출만 필요(Stage B/C/D의 ControlNet/GPU는 **불필요**). `roi_metadata.csv` + crops를 Drive에 1회 생성한다.
> Stage A와 아래 generate_casda 셀은 **같은 Colab 세션**에서 연속 실행 권장(ROI 절대경로 drift 방지).

```python
!git clone --single-branch -b approve https://github.com/hhjun321/CASDA.git /content/CASDA

!python $SCRIPTS/extract_rois.py \
    --image_dir $TRAIN_IMAGES \
    --train_csv $TRAIN_CSV \
    --output_dir $ROI_DIR \
    --roi_size 256 \
    --min_suitability 0.5
# → $ROI_DIR/roi_metadata.csv + images/*.png  (class{N} 정보 보유)
```

### Cell B — CASDA 합성 생성 (동일 copy-paste 엔진)

> `casda_roi_adapter`가 `min_suitability=0.5`, `per_class_cap` 미지정(=None) 기본으로 동작(결정 A).
> 로그에서 `n_missing_image ≈ 0`과 class별 ROI 수(c3/c4 포함)를 확인할 것.

```python
!python $AROMA_SCRIPTS/generate_casda.py \
    --metadata_csv $ROI_DIR/roi_metadata.csv \
    --normal_dir   $NORMAL_DIR \
    --output_dir   $CASDA_SYNTH_DIR/severstal \
    --n_per_roi    $N_PER_ROI \
    --seed 42
# → $CASDA_SYNTH_DIR/severstal/annotations.json + images/
```

### Cell C — 4조건 exp4v2 실행 (baseline / random / casda / aroma)

```python
!python $AROMA_SCRIPTS/experiments/exp4_v2_supervised_detection.py \
    --model yolov8n \
    --condition all \
    --dataset_keys severstal \
    --class_mode multi \
    --random_synthetic_dir $RANDOM_SYNTH_DIR \
    --casda_synthetic_dir  $CASDA_SYNTH_DIR \
    --aroma_synthetic_dir  $AROMA_SYNTH_DIR \
    --real_data_dir        $AROMA_DATA \
    --output_dir           $EXP4V2_OUT/severstal_casda \
    --synth_ratio 1.0 \
    --seeds 42 1337 2025 \
    --imgsz 640 \
    --val_frac 0.3 \
    --baseline_epochs 100 \
    --patience 20 \
    --batch 64 \
    --cache ram \
    --rect \
    --yolo_cache_dir $AROMA_OUT/yolo_cache \
    --resume
```

| 항목 | 내용 |
|------|------|
| 조건 | `baseline / random / casda / aroma` (4) — `ALL_CONDITION_KEYS` 자동 확장 |
| 공정성 | `--synth_ratio`가 random/casda/aroma에 동일 적용 → `n_synth_train` 3조건 동일 기대 |
| `--n_per_roi` 없음 | exp4v2 스크립트에는 `--n_per_roi` 인자가 없다(주면 argparse가 즉시 종료). n_per_roi parity는 **생성 단계**(각 generate_* 셀, Cell B의 `$N_PER_ROI`)에서 보장하며 학습 단계에서 통제하지 않는다. synth 수 parity는 `--synth_ratio 1.0`이 담당 |
| `--casda_synthetic_dir` 생략 시 | casda 조건은 합성 부재로 `no_synth_annotations` 거부(다른 조건은 정상 진행) |
| per-class synth 투명성 | 각 조건 결과에 `n_synth_per_class` 추가 + 로그 `synth per-class (0-idx, -1=unparsed)` 출력 → CASDA c4/c2 starvation 확인 |

> **소요 시간**: 단일 실행 × seed 수. 4조건 × 3 seed → 단일(약 3~5시간)의 약 3배. `--resume` 항상 함께 사용.

### CASDA vs AROMA 결과 확인 (4조건 + per-class synth parity)

```python
import json, os

with open(f"{os.environ['EXP4V2_OUT']}/severstal_casda/exp4v2_results.json") as f:
    results = json.load(f)

CONDS = ["baseline", "random", "casda", "aroma"]
arms = results["severstal"]["yolov8n"]

# (1) map50 (mean ± std) + 95% CI
print(f"{'조건':<10}  {'map50 (mean±std)':>22}  {'95% CI':>20}  {'n_synth_train':>14}")
print("-" * 72)
for c in CONDS:
    cell = arms.get(c, {})
    m  = cell.get("map50")
    sd = (cell.get("std") or {}).get("map50")
    ci = (cell.get("ci95") or {}).get("map50")
    ns = cell.get("n_synth_train")
    if isinstance(m, float):
        ms  = f"{m:.4f} ± {sd:.4f}" if isinstance(sd, float) else f"{m:.4f}"
        cis = f"[{ci[0]:.4f}, {ci[1]:.4f}]" if isinstance(ci, list) and ci[0] is not None else "N/A"
    else:
        ms, cis = "N/A", "N/A"
    print(f"{c:<10}  {ms:>22}  {cis:>20}  {str(ns):>14}")

# (2) 공정성 parity: random/casda/aroma의 n_synth_train 동일해야 함
ns = {c: arms.get(c, {}).get("n_synth_train") for c in ("random", "casda", "aroma")}
print(f"\nn_synth_train parity {ns} -> {'OK' if len(set(v for v in ns.values() if v is not None)) <= 1 else '⚠️ 불일치(cap 누락 의심)'}")

# (3) per-class synth 분포 (CASDA c4/c2 starvation 투명화)
print(f"\n{'조건':<10}  per-class synth (0-idx, -1=unparsed)")
print("-" * 56)
for c in ("random", "casda", "aroma"):
    print(f"{c:<10}  {arms.get(c, {}).get('n_synth_per_class')}")
```

> per-class AP(c1~c4)는 `arms[cond]['per_class']`에서 확인(위 multi 모드 per-class 확인 셀 재사용). 성공 기준: AROMA map50 > CASDA, 3-seed 95% CI 비겹침, `n_synth_train` 3조건 동일, 소수 클래스(c3·c4) AP에서 AROMA ≥ CASDA.

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
        val = results[ds].get(cond, {}).get("map50")
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
    b = results[ds].get("baseline", {}).get("map50")
    r = results[ds].get("random",   {}).get("map50")
    a = results[ds].get("aroma",    {}).get("map50")
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
| `$EXP4V2_OUT/exp4v2_results.json` | 데이터셋 × 조건별 mAP50 / mAP50-95 / precision / recall. multi-seed 시 top-level=평균 + `per_seed`/`std`/`ci95`/`n_seeds`. **`--class_mode multi` 시 조건별 `per_class` 추가** |
| `$EXP4V2_OUT/exp4v2_summary.md` | Markdown 비교표 (조건 × 데이터셋) + delta 섹션. multi-seed 시 각 셀 `mean ± std`. **multi 모드 시 model별 `per-class map50` 표 추가** |
| `$EXP4V2_OUT/_seeds/seed{N}/exp4v2_results.json` | (multi-seed) seed별 원본 결과. 각자 `--resume` 독립 적용 |

---

## 주의사항

- **`--aroma_synthetic_dir`는 AROMA가 합성한 이미지 디렉터리를 가리켜야 함**: 이 경로 아래에는 `{dataset_key}/annotations.json`과 `{dataset_key}/images/`가 존재해야 한다. 경로가 틀리면 `annotations.json NOT FOUND` 에러와 함께 해당 조건의 `n_synth=0`이 되어 random/aroma 조건이 거부된다(`no_synth_annotations`). `--random_synthetic_dir`도 동일하게 random 합성 이미지 디렉터리를 가리켜야 한다.
- **AROMA `n_synth=0`이면 `annotations.json`의 `image_path` 필드를 확인**: 각 항목의 `image_path`가 실제 존재하는 파일을 가리켜야 한다(절대경로 또는 synth 디렉터리 기준 상대경로). 모든 항목이 resolve 실패하면 `had N entries but 0 resolved` 에러가 출력된다. 로그에서 `aroma train: N/600 synth imgs labeled`의 `N>0`을 확인할 것.
- **annotations.json의 `normal_image` 필드 필요**: 합성 결함 생성 시 배경 이미지 경로가 기록된 필드. mask가 없을 때 스크립트가 bbox 추출 시 이 경로를 사용한다.
- **합성 결함 bbox 추출 방식**: `annotations.json`에 `mask`/`roi_mask`/`mask_path` 필드나 `masks/{stem}.png` 파일이 있으면 그 mask에서 직접 bbox를 산출한다. mask가 없으면 composite 이미지 - background 이미지 차분 후 임계처리(threshold=8, 미검출 시 Otsu fallback)로 결함 영역의 bounding box를 자동 산출한다.
- **real 결함 라벨이 0개이면 진단 로그 확인**: real defect 이미지에서 라벨이 하나도 추출되지 않으면 `_get_real: 0 labels ... (no_mask=.. bad_size=.. no_bbox=.. no_lines=..)` 에러가 출력된다. `no_mask`가 크면 mask 경로 매핑 문제, `no_bbox`가 크면 `min_area`(기본 50)가 해당 데이터셋 mask 해상도에 비해 너무 큰 것이다.
- **YOLO 캐시 schema_version=2**: bbox 추출 로직 변경으로 캐시 스키마가 v2로 상향되었다. `--yolo_cache_dir`로 기존(v1) 캐시가 있으면 자동으로 무효화되어 1회 재빌드된다.
- **YOLO 캐시 자동 재빌드 조건**: 소스 결함 이미지 수가 캐시 빌드 시점과 다를 경우(mask 파일 추가/삭제 등) 다음 실행 시 자동으로 캐시를 무효화하고 재빌드한다. 로그에서 `[YoloCache] built+saved` 메시지로 확인 가능.
- **`n_train`의 의미**: `exp4v2_results.json`의 `n_train` = 레이블 있는 훈련 이미지 수 (real_defect + synth). background negative 이미지(`n_real_train`만큼 추가)는 포함되지 않는다. 실제 YOLO가 보는 이미지 수 ≈ `n_train + n_real_train` (약 2×).
- **세 조건이 독립적으로 실행됨**: baseline best.pt를 random/aroma가 사용하지 않는다. 따라서 baseline 실패 여부와 관계없이 random/aroma를 단독으로 실행할 수 있다(`--condition random` 또는 `--condition aroma`).
- **파라미터 변경 후 재실행 시 결과 초기화 필요**: `val_frac`, `max_synth_per_ds`, `imgsz`, `baseline_epochs` 등 학습 조건이 변경되면 이전 `exp4v2_results.json`의 캐시 결과가 불일치한다. `--resume` 사용 시 기존 결과가 skip 조건으로 판정되어 재학습이 일어나지 않는다. 해당 데이터셋 조건의 결과를 json에서 직접 삭제하거나 파일 전체를 삭제 후 재실행.
- **baseline_epochs=300 수렴 안정**: 소규모 데이터(42~70장)는 300 epoch이 안전한 상한. EarlyStopping(`--patience 50`)을 함께 사용하면 실제 수렴 시점에서 자동 종료된다.
- **imgsz=640 권장**: 기본값 256은 고해상도 이미지에서 small defect 검출 능력 저하. 640 사용 시 약간 느리지만 검출 정확도 개선.
- **학습 속도 인자(`--batch`/`--cache`/`--rect`)**: 세 인자 모두 기본값=현행이라 미지정 시 회귀 없음. 전 조건에 동일 적용되어 공정성 불변. `--batch -1`(auto)은 조건간 batch가 달라질 수 있어 비권장. 상세는 "선택적 CLI 인자 — 속도 최적화" 섹션 참고.
- **Test set 원칙**: 합성 이미지는 train에만 포함. test/defect는 항상 real 이미지만 사용.
- **mvtec_carpet / mvtec_leather**: texture 계열, exp4v2 신규 추가(visa_pcb 대체). full-frame 표면이라 배경 void 없음.
- **`--class_mode multi` 는 severstal 전용**: 타 데이터셋에 줘도 single로 처리되어 결과 byte-identical. multi 시에만 `per_class`가 추가되고 summary에 per-class map50 표가 출력된다. top-level 지표는 4클래스 macro-average라 클래스별 편차가 평균으로 뭉개짐 → 클래스별 비교는 `per_class`를 볼 것.
- **staging 로그 = 정상 동작 신호**: real 이미지는 데이터셋당 1회만 local에 staging되고 조건별로는 hardlink(병렬)된다. 로그의 `[Stage] ... done N in X.Xs (hardlink=H copy=C)`로 진행이 보인다. 데이터셋 1회 staging은 Drive→local이라 `copy`, 조건별은 `hardlink`로 즉시 끝나는 것이 정상. 이 줄들이 보이면 "무응답"이 아니라 staging 진행 중이다.
- **GPU 필수**: YOLO 학습은 CUDA 없이 실행 불가.