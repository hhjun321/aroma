# AROMA Exp 4v2 — Supervised YOLO Detection 실행 가이드 (Workflow)

**목적**: Baseline vs Random ROI vs AROMA ROI — YOLOv8 지도학습 결함 검출 성능(mAP50) 비교
**연구 위치**: **downstream 이차 축**. AROMA 일차 축은 exp2(intrinsic ROI 품질)·exp3(생성 품질). exp4v2는 "AROMA ROI 선택이 실제 검출기 학습에 기여하는가"를 동일 copy-paste 엔진 위에서 검증한다. Severstal은 CASDA와의 참조 비교(4번째 조건, STEP 6)에만 사용한다.
**데이터셋 (v2-1 확정 4종)**: `severstal`(강판) · `mvtec_leather`(가죽) · `aitex`(텍스타일) · `mtd`(자성타일)
**런타임**: GPU 필수 (Colab Pro A100 권장)

---

## 워크플로우 개요

| STEP | 내용 | 런타임 |
|------|------|--------|
| 0 | 환경변수 설정 | CPU |
| 1 | 패키지 설치 & GPU 확인 | CPU |
| 2 | 선행 조건 검증 (synth annotations 존재 확인) | CPU |
| 3 | 스모크 테스트 (단일 데이터셋, epochs=20) | GPU, ~5–10분 |
| 4 | 전체 실행 (4 데이터셋 × 3조건, multi-seed) | GPU, 수 시간 |
| 5 | 결과 확인 (mean±std + delta) | CPU |
| 6 | (선택) Severstal 4조건 참조 비교 (+CASDA) | GPU |

> **선행 전제**: exp2·exp3와 동일한 Stage 1–4 산출물이 4개 데이터셋에서 완료돼 있어야 한다.
> - AROMA 합성: Stage 4 완료 → `$AROMA_SYNTH_DIR/{ds}/annotations.json` (+ `images/`)
> - Random 합성: `exp3_execute.md` 0단계(`generate_random.py`) 완료 → `$RANDOM_SYNTH_DIR/{ds}/annotations.json`
> - aitex/mtd 선행: `multidomain_integration_verify_execute.md`로 download+prepare+Stage1–3 완료
> exp4v2는 합성을 **재생성하지 않는다** — 이미 존재하는 합성본에서 bbox 라벨을 post-hoc 유도한다.

---

## 실험 설계 (요약)

세 조건 모두 **COCO pretrained YOLOv8에서 처음부터(scratch)** 학습한다.

| 조건 | 학습 데이터 | 출발점 | epoch |
|------|------------|--------|-------|
| baseline | real 결함 이미지만 | COCO pretrained | `baseline_epochs` |
| random | real 결함 + random synthetic | COCO pretrained | `baseline_epochs` |
| aroma | real 결함 + AROMA synthetic | COCO pretrained | `baseline_epochs` |

- 평가 지표: **mAP50** (IoU=0.50). 평가셋은 **항상 real 결함 이미지만** (합성은 train에만).
- 세 조건은 독립 실행 — baseline best.pt를 random/aroma가 재사용하지 않는다.
- 합성 label 유도: composite(synth) − background(normal) image diff → threshold → contour bbox. mask가 있으면 mask에서 직접 산출.

> **설계 배경**: 구 exp4(one-class AD)는 합성 결함을 train/good에 섞어 정상분포를 오염시켜 실패(baseline 0.765 ≫ random/aroma ≈ 0.50). exp4v2는 합성 결함을 **지도학습 레이블 데이터**로 사용해 이 문제를 회피한다. 2-stage finetune(baseline 선학습 후 짧은 추가학습)도 폐기 — real+synth를 처음부터 함께 학습하는 것이 copy-paste augmentation의 표준이며 합성 기여도를 공정히 측정한다.

---

## STEP 0 — 환경변수 설정

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

## STEP 1 — 패키지 설치 & GPU 확인

```python
!pip install ultralytics -q
!pip install opencv-python-headless -q  # cv2 (bbox 추출)
```

```python
!nvidia-smi
```

> **주의**: ultralytics 설치 후 런타임 재시작(Runtime → Restart runtime)이 필요하면 STEP 0 환경변수를 다시 설정한다.

---

## STEP 2 — 선행 조건 검증 (synth annotations 존재 확인)

전체 실행 전, 4개 데이터셋의 AROMA·Random 합성본이 실제로 존재하는지 확인한다. (없으면 해당 조건이 `no_synth_annotations`로 거부됨)

```python
import os, json, pathlib

DATASETS = ["severstal", "mvtec_leather", "aitex", "mtd"]

print(f"{'데이터셋':<16} {'aroma_ann':>10} {'aroma_img':>10} {'random_ann':>11} {'random_img':>11}")
print("-" * 62)
for ds in DATASETS:
    row = []
    for root in (os.environ['AROMA_SYNTH_DIR'], os.environ['RANDOM_SYNTH_DIR']):
        ann = pathlib.Path(f"{root}/{ds}/annotations.json")
        n_ann = len(json.load(open(ann))) if ann.exists() else 0
        img_dir = pathlib.Path(f"{root}/{ds}/images")
        n_img = len(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))) if img_dir.exists() else 0
        row += [n_ann, n_img]
    a_ann, a_img, r_ann, r_img = row
    print(f"{ds:<16} {a_ann:>10} {a_img:>10} {r_ann:>11} {r_img:>11}")
```

> 4개 데이터셋 모두 `aroma_ann > 0` **및** `random_ann > 0`이어야 STEP 4 전체 실행이 유효하다.
> `0`인 데이터셋이 있으면 → AROMA는 Stage 4를, Random은 `exp3_execute.md` 0단계를 먼저 재실행한다.

---

## STEP 3 — 스모크 테스트 (단일 데이터셋)

파이프라인 정상 동작을 먼저 확인한다. 소형 데이터셋(`mvtec_leather`) + `baseline_epochs=20`으로 단축.

```python
!python $AROMA_SCRIPTS/experiments/exp4_v2_supervised_detection.py \
    --model yolov8n \
    --condition all \
    --dataset_keys mvtec_leather \
    --random_synthetic_dir $RANDOM_SYNTH_DIR \
    --aroma_synthetic_dir  $AROMA_SYNTH_DIR \
    --real_data_dir        $AROMA_DATA \
    --output_dir           $EXP4V2_OUT/_smoke \
    --yolo_cache_dir       $AROMA_OUT/yolo_cache \
    --imgsz 640 \
    --val_frac 0.3 \
    --synth_ratio 1.0 \
    --baseline_epochs 20 \
    --seed 1
```

> **소요 시간**: A100 기준 약 5–10분 (3조건).
> **통과 기준**: 세 조건 모두 `mAP50 > 0`, 로그에 `aroma train: N/… synth imgs labeled`의 `N>0`, `_get_real: … labels` > 0. 하나라도 `0.0000`이면 STEP 4로 넘어가지 말고 "주의사항"의 진단 항목을 확인.

---

## STEP 4 — 전체 실행 (4 데이터셋 × 3조건)

3조건 × 4데이터셋 = 12회 학습(seed당). 모든 조건이 독립적으로 COCO에서 출발한다.

### 4-A. 단일 seed (빠른 확인)

```python
!python $AROMA_SCRIPTS/experiments/exp4_v2_supervised_detection.py \
    --model yolov8n \
    --condition all \
    --dataset_keys severstal mvtec_leather aitex mtd \
    --random_synthetic_dir $RANDOM_SYNTH_DIR \
    --aroma_synthetic_dir  $AROMA_SYNTH_DIR \
    --real_data_dir        $AROMA_DATA \
    --output_dir           $EXP4V2_OUT \
    --yolo_cache_dir       $AROMA_OUT/yolo_cache \
    --imgsz 640 \
    --val_frac 0.3 \
    --synth_ratio 1.0 \
    --baseline_epochs 300 \
    --patience 50 \
    --batch 64 \
    --cache ram \
    --rect \
    --seed 1 \
    --resume
```

### 4-B. Multi-seed (통계 유의성 — 권장 최종)

단일 seed는 작은 val(데이터셋에 따라 수십 장)에서 노이즈가 커 AROMA vs Random delta의 유의성 판정이 어렵다. **exp4v2 seed 규약 = `1 2 43` 고정**(모든 조건 동일 세트). seed별 독립 split·subsample·학습 후 조건별 **mean ± std + 95% CI** 자동 집계.

```python
!python $AROMA_SCRIPTS/experiments/exp4_v2_supervised_detection.py \
    --model yolov8n \
    --condition all \
    --dataset_keys severstal mvtec_leather aitex mtd \
    --random_synthetic_dir $RANDOM_SYNTH_DIR \
    --aroma_synthetic_dir  $AROMA_SYNTH_DIR \
    --real_data_dir        $AROMA_DATA \
    --output_dir           $EXP4V2_OUT \
    --yolo_cache_dir       $AROMA_OUT/yolo_cache \
    --imgsz 640 \
    --val_frac 0.3 \
    --synth_ratio 1.0 \
    --baseline_epochs 300 \
    --patience 50 \
    --batch 64 \
    --cache ram \
    --rect \
    --seeds 1 2 43 \
    --resume
```

> **소요 시간**: 단일 실행 × seed 수. 3 seed 전체 → 단일(약 3–5시간)의 약 3배 ≈ 9–15시간.
> A100 세션 한도를 고려해 `--resume`을 항상 함께 사용 — seed 단위·(dataset, condition) 단위 모두 중단 지점부터 재개된다.
> ⚠️ **디스크 누적**: checkpoint가 `seed × ds × cond`(3×4×3 = 36개 best.pt) 누적. 집계 완료 후 `_seeds/*/{ds}/{model}/{cond}/weights/`를 정리하면 회수 가능(수치는 `_seeds/seed{N}/exp4v2_results.json`에 보존).
> ⚠️ **파라미터 변경 후 재실행**: `synth_ratio`/`val_frac`/`imgsz`/`baseline_epochs`를 바꾸면 기존 결과가 skip 조건에 걸려 재학습되지 않는다. `exp4v2_results.json`(+`_seeds/seed*/`)에서 해당 데이터셋 항목을 삭제하거나 fresh `--output_dir` 사용.

---

## STEP 5 — 결과 확인

### 5-A. 단일 seed 비교표

```python
import json, os

with open(f"{os.environ['EXP4V2_OUT']}/exp4v2_results.json") as f:
    results = json.load(f)

CONDITIONS = ["baseline", "random", "aroma"]

print(f"{'데이터셋':<16}  {'baseline':>10}  {'random':>10}  {'aroma':>10}  {'Δ(A-R)':>10}")
print("-" * 64)
for ds in sorted(results):
    arms = results[ds].get("yolov8n", results[ds])   # multi-seed/구조 모두 대응
    b = arms.get("baseline", {}).get("map50")
    r = arms.get("random",   {}).get("map50")
    a = arms.get("aroma",    {}).get("map50")
    d = round(a - r, 4) if isinstance(a, float) and isinstance(r, float) else None
    fmt = lambda v: f"{v:.4f}" if isinstance(v, float) else "N/A"
    print(f"{ds:<16}  {fmt(b):>10}  {fmt(r):>10}  {fmt(a):>10}  {(f'{d:+.4f}' if d is not None else 'N/A'):>10}")
```

### 5-B. Multi-seed (mean ± std + 95% CI)

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
            m  = cell.get("map50")
            sd = (cell.get("std") or {}).get("map50")
            ci = (cell.get("ci95") or {}).get("map50")
            if isinstance(m, float):
                ms  = f"{m:.4f} ± {sd:.4f}" if isinstance(sd, float) else f"{m:.4f}"
                cis = f"[{ci[0]:.4f}, {ci[1]:.4f}]" if isinstance(ci, list) and ci[0] is not None else "N/A"
            else:
                ms, cis = "N/A", "N/A"
            print(f"{c:<10}  {ms:>22}  {cis:>20}")

        a, r = arms.get("aroma", {}), arms.get("random", {})
        if isinstance(a.get("map50"), float) and isinstance(r.get("map50"), float):
            d  = (a["map50"] - r["map50"]) * 100
            sa = (a.get("std") or {}).get("map50", 0.0)
            sr = (r.get("std") or {}).get("map50", 0.0)
            comb = ((sa ** 2 + sr ** 2) ** 0.5) * 100
            verdict = "유의 가능성" if abs(d) > 2 * comb else "noise 범위"
            print(f"\n  Delta(AROMA-Random) map50: {d:+.2f}pp  (combined std {comb:.2f}pp) -> {verdict}")
```

> **유의성 rough 판정**: `|delta| > 2 × combined_std`이면 noise floor 밖 가능성. 정밀 검정은 `per_seed`의 seed별 값으로 paired Wilcoxon/t-test를 별도 수행.

---

## STEP 6 — (선택) Severstal 4조건 참조 비교 (+CASDA)

> ⚠️ **Severstal 전용 · 참조용**. `baseline / random / casda / aroma` **4조건**을 동시 비교한다. 세 합성 조건(random/casda/aroma)은 **동일 copy-paste 엔진**을 쓰고 **ROI 선택 전략만 다르다** → AROMA의 ROI 모델링 기여를 CASDA(단일도메인 특화) 대비 격리. 상세 절차·공정성 불변식·per-class 확인은 **부록 E**를 참조한다.

핵심 실행만 요약:

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
    --seeds 1 2 43 \
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

> **공정성 게이트**: 결과 JSON에서 random/casda/aroma의 `n_synth_train`이 **3조건 동일**해야 한다(cap 트리밍 결과). 다르면 cap 누락 의심.
> **quality gate ON 시 추가 확인**: step3/exp3에서 `--min_quality`를 켰다면 (1) aroma·random이 **동일 threshold**로 생성됐는지, (2) 게이트-후 synth pool이 여전히 `synth_ratio` cap 이상인지(부족하면 no-trim → parity 붕괴), (3) `n_synth_train` 동일 여부를 반드시 재확인. 확인 코드·CASDA 합성 생성(Stage A + generate_casda)은 부록 E.

---

## class_mode — 데이터셋별 per-class 측정 (4종 전체 지원)

`--class_mode multi`는 이제 **전 데이터셋에 데이터셋-일반적으로** 적용된다(구 severstal 전용 하드코딩 제거). 각 결함 이미지의 `test/{type}` 폴더명으로 클래스를 자동 열거해 `nc`·`names`를 데이터셋별로 결정한다.

| 데이터셋 | `--class_mode multi` 시 nc / 클래스 | per_class 키 |
|----------|-------------------------------------|--------------|
| severstal | 4 (`class1`~`class4`) | `c1`~`c4` (표시명 유지) |
| mvtec_leather | 5 (`color/cut/fold/glue/poke`) | 폴더명 |
| aitex | 12 (`002/006/…/036`) | 코드명 |
| mtd | 5 (`blowhole/break/crack/fray/uneven`) | 폴더명 |

- **동작**: `test/{type}` 폴더가 1개뿐인 데이터셋(visa `anomaly` 등)은 자동으로 single로 축퇴(byte-identical). `--class_mode`(기본 `single`) 미지정 시 전 데이터셋 기존과 동일.
- **연구 crux**: 소수 클래스(rare-class)별 AP로 "AROMA가 어떤 결함 유형에 기여하는가"를 이제 **4종 모두**에서 관측 가능(aitex 희소 코드·mtd fray·severstal c2 등).
- **rare-class 유의**: aitex는 12종 중 일부(027/029/036 등)가 1장뿐 — 해당 클래스는 학습 사실상 불가, `per_class`에 val 등장분만 정직 보고된다.
- **CASDA(STEP 6)**: severstal 전용. severstal-CASDA 합성은 `cluster_id` 폴백으로 클래스 해소.

---

## 부록 A — CLI 인자 레퍼런스

| 인자 | 기본값 | 설명 |
|------|--------|------|
| `--model` | yolov8n | `yolov8n/s/m` 또는 `all` |
| `--condition` | all | `baseline random casda aroma` 다중 선택(nargs+). `all` 포함 시 전체 확장 |
| `--dataset_keys` | (필수) | 예: `severstal mvtec_leather aitex mtd` |
| `--baseline_epochs` | 50 | 전체 조건 공통 학습 epoch. 소규모 데이터는 300 권장 |
| `--val_frac` | 0.3 | real defect 중 val 비율 |
| `--synth_ratio` | None | synth:real_train 비율. `n_synth_cap = max(1, int(n_real_train × ratio))`. 지정 시 `--max_synth_per_ds` 무시 |
| `--max_synth_per_ds` | None | 데이터셋·조건당 synth 최대 수(절대값). `--synth_ratio` 미지정 시에만 적용 |
| `--patience` | 0 | EarlyStopping patience(0=비활성). 기준 지표 = `mAP50-95`. epochs=300 → 50 권장 |
| `--imgsz` | 256 | YOLO image size. **640 권장**(고해상도 small defect 검출) |
| `--seed` | 42 | 단일 seed. `--seeds` 미지정 시 사용 |
| `--seeds` | None | multi-seed 목록. **exp4v2 규약: `1 2 43`**. 지정 시 mean±std+95%CI 집계 |
| `--class_mode` | single | `multi`=데이터셋별 per-class(`test/{type}` 폴더로 nc/names 자동, 전 데이터셋 적용; 위 "class_mode" 섹션 참조) |
| `--resume` | off | 완료된 `(dataset, condition)` skip 후 재개. `mAP50` 유효 조건만 skip |
| `--yolo_cache_dir` | None | real (image,label) 셋 build-once 영속 캐시. 동일 `(min_area/val_frac/seed)`면 재빌드 없음 |
| `--no_local_cache` | off | Drive→/tmp 로컬 캐시 staging 비활성(Linux/Colab 기본 활성, Windows 항상 비활성) |

> `--casda_synthetic_dir`는 부록 E(Severstal 참조 비교)에서만 사용.

### synth_ratio 데이터셋별 스케일 주의

`n_synth_cap`은 **데이터셋별 real_train 수에 종속**되어 자릿수가 크게 다르다.

| 데이터셋 | real_train(val_frac=0.3) | `--synth_ratio 1.0` → n_synth_cap |
|----------|--------------------------|-----------------------------------|
| severstal | ≈ 2534 | ≈ 2534 |
| mvtec_leather / aitex / mtd | 수십~수백 | real_train과 동수 |

> mvtec류 소규모 데이터에 쓰이던 `--max_synth_per_ds 128` 같은 절대값은 **severstal에 그대로 쓰면 안 된다**(자릿수 불일치). 데이터셋 혼합 실행 시에는 `--synth_ratio`(비율 기반)를 사용해 각 데이터셋에 스케일-불변으로 적용한다.

---

## 부록 B — 속도 최적화 인자 (`--batch` / `--cache` / `--rect`)

세 인자 모두 기본값=Ultralytics 기본 → 미지정 시 기존 동작과 byte-identical. 전 조건 동일 적용이라 비교 공정성·val 셋 불변.

| 인자 | 기본값 | 설명 |
|------|--------|------|
| `--batch` | 16 | A100에서 step 과다 시 **64/128** 상향 권장. `-1`(auto)은 조건간 batch가 달라져 재현성 해침 → 고정값 권장 |
| `--cache` | `""`(False) | `ram`=RAM 캐시(최고속), `disk`=디스크, `True`=`ram` 별칭. OOM 시 `ram`→`disk` 폴백 |
| `--rect` | False | rectangular training. letterbox 회색 패딩 최소화 → 속도↑. Severstal류 고종횡비(1600×256, 6.25:1)·aitex(4096×256, 16:1)에서 효과 큼 |

> **cache 파싱**: `""`/None → False, `ram`/`True` → `ram`, `disk` → `disk`. 오타 시 자동 비캐싱 폴백.
> **Local image cache**: Linux/Colab에서 CLI 없이 자동 활성(Drive→/tmp 복사). `--no_local_cache`로 비활성.

---

## 부록 C — multi-seed 상세

| 항목 | 내용 |
|------|------|
| 동작 | 각 seed로 전체 (datasets × models × conditions) 독립 학습 후 집계 |
| per-seed 결과 | `$EXP4V2_OUT/_seeds/seed{N}/exp4v2_results.json` (각자 `--resume` 독립 적용) |
| 최종 집계 | `$EXP4V2_OUT/exp4v2_results.json` — top-level=**평균값** + `per_seed{}`·`std{}`·`ci95{}`·`n_seeds` 추가 |
| checkpoint | `$EXP4V2_OUT/_seeds/seed{N}/{ds}/{model}/{cond}/weights/best.pt` (seed별 격리) |
| std/CI | sample std(ddof=1, n_seeds<2면 0), 95% CI는 t-분포(scipy 있으면) 또는 소표본 t 임계표 fallback |
| 일부 seed 실패 | 성공 seed만으로 집계, `n_seeds`에 실제값 기록, 로그 경고 |

---

## 부록 D — 출력 파일

| 파일 | 내용 |
|------|------|
| `$EXP4V2_OUT/exp4v2_results.json` | 데이터셋 × 조건별 mAP50 / mAP50-95 / precision / recall. multi-seed 시 top-level=평균 + `per_seed`/`std`/`ci95`/`n_seeds`. `--class_mode multi` 시 조건별 `per_class` 추가(전 데이터셋) |
| `$EXP4V2_OUT/exp4v2_summary.md` | Markdown 비교표(조건 × 데이터셋) + delta. multi-seed 시 각 셀 `mean ± std`. multi 모드 시 per-class map50 표 추가 |
| `$EXP4V2_OUT/_seeds/seed{N}/exp4v2_results.json` | (multi-seed) seed별 원본 결과 |

---

## 부록 E — Severstal CASDA 4조건 상세 (STEP 6 확장)

> Severstal 참조 비교의 전체 절차. AROMA Drive 루트(`.../data/Aroma`)와 CASDA Drive 루트(`.../data/Severstal`)가 **다르며 둘 다 마운트**되어야 한다.

### 공정성 불변식 (load-bearing)

- `--synth_ratio` 동일 적용 → random/casda/aroma 모두 동일 synth cap. 결과 JSON `n_synth_train`이 세 조건 동일해야 공정.
- `--n_per_roi`(=3) 세 생성기 동일 — **생성 단계**에서 보장(exp4v2 스크립트엔 `--n_per_roi` 인자 없음).
- **CASDA는 자기 기준(suitability)으로 top-N SELECT**: `--per_class_cap 525`로 abundant class(c1/c3/c4)는 각 525 컷, c2는 candidate≈117(<525)이라 전량. pool ≈ 5076 ≥ 학습 cap(≤5068, ratio 2.0까지) → 4조건 등가 budget. **honest asymmetry**: c2는 suitability 필터 전 수치라 일부가 <0.5면 CASDA c2가 117보다 약간 작아질 수 있음(AROMA c2는 gate 없음) — `n_synth_per_class`로 드러낼 것.
- **무결성**: 선택 *전략*만 조건마다 다르고 pool 규모·n_per_roi·학습 cap budget은 4조건 동일. AROMA에만 큰 pool 부여·CASDA만 random-trim·AROMA 유리 ratio cherry-pick은 모두 rig — 회피.

### 환경변수 (CASDA 전용 추가)

```python
import os

os.environ['CASDA_DRIVE']  = "/content/drive/MyDrive/data/Severstal"   # AROMA와 다름
os.environ['SCRIPTS']      = "/content/CASDA/scripts"
os.environ['TRAIN_IMAGES'] = f"{os.environ['CASDA_DRIVE']}/train_images"
os.environ['TRAIN_CSV']    = f"{os.environ['CASDA_DRIVE']}/train.csv"
os.environ['ROI_DIR']      = f"{os.environ['CASDA_DRIVE']}/roi_patches_v5.1"
os.environ['NORMAL_DIR']   = f"{os.environ['AROMA_DATA']}/severstal/train/good"
os.environ['CASDA_SYNTH_DIR'] = f"{os.environ['AROMA_OUT']}/synthetic_casda"
os.environ['N_PER_ROI']       = "3"
print("CASDA_SYNTH_DIR :", os.environ['CASDA_SYNTH_DIR'])
```

### Cell E-1 — CASDA Stage A (CPU, clone + ROI 추출)

> ROI 추출만 필요(Stage B/C/D의 ControlNet/GPU 불필요). Stage A와 아래 generate_casda 셀은 **같은 세션**에서 연속 실행 권장(ROI 절대경로 drift 방지).

```python
!git clone --single-branch -b approve https://github.com/hhjun321/CASDA.git /content/CASDA

!python $SCRIPTS/extract_rois.py \
    --image_dir $TRAIN_IMAGES \
    --train_csv $TRAIN_CSV \
    --output_dir $ROI_DIR \
    --roi_size 256 \
    --min_suitability 0.5
```

### Cell E-2 — CASDA 합성 생성 (동일 copy-paste 엔진)

```python
!python $AROMA_SCRIPTS/generate_casda.py \
    --metadata_csv  $ROI_DIR/roi_metadata.csv \
    --normal_dir    $NORMAL_DIR \
    --output_dir    $CASDA_SYNTH_DIR/severstal \
    --n_per_roi     $N_PER_ROI \
    --per_class_cap 525 \
    --min_suitability 0.5 \
    --seed 42 \
    --local_staging
# → $CASDA_SYNTH_DIR/severstal/annotations.json + images/
```

> 로그에서 `n_missing_image ≈ 0`, class별 ROI 수(c3/c4 포함), `n_filtered_cap`(525 cap이 abundant class에 걸린 수)을 확인.
> `--local_staging`: manifest(suitability·per_class_cap 통과분)만 `/content`로 병렬 복사(Drive FUSE 지연 회피). 동시성은 `AROMA_STAGE_WORKERS`(기본 16, 1~64).

### Cell E-3 — 4조건 실행

STEP 6의 실행 커맨드 사용. `--casda_synthetic_dir $CASDA_SYNTH_DIR`가 반드시 포함돼야 하며, 생략 시 casda 조건은 `no_synth_annotations`로 거부(다른 조건은 정상 진행).

### Cell E-4 — 결과 확인 (4조건 + parity + per-class)

```python
import json, os

with open(f"{os.environ['EXP4V2_OUT']}/severstal_casda/exp4v2_results.json") as f:
    results = json.load(f)

CONDS = ["baseline", "random", "casda", "aroma"]
arms = results["severstal"]["yolov8n"]

# (1) map50 (mean±std) + 95% CI + n_synth_train
print(f"{'조건':<10}  {'map50 (mean±std)':>22}  {'95% CI':>20}  {'n_synth_train':>14}")
print("-" * 72)
for c in CONDS:
    cell = arms.get(c, {})
    m  = cell.get("map50"); sd = (cell.get("std") or {}).get("map50")
    ci = (cell.get("ci95") or {}).get("map50"); ns = cell.get("n_synth_train")
    if isinstance(m, float):
        ms  = f"{m:.4f} ± {sd:.4f}" if isinstance(sd, float) else f"{m:.4f}"
        cis = f"[{ci[0]:.4f}, {ci[1]:.4f}]" if isinstance(ci, list) and ci[0] is not None else "N/A"
    else:
        ms, cis = "N/A", "N/A"
    print(f"{c:<10}  {ms:>22}  {cis:>20}  {str(ns):>14}")

# (2) 공정성 parity
ns = {c: arms.get(c, {}).get("n_synth_train") for c in ("random", "casda", "aroma")}
ok = len(set(v for v in ns.values() if v is not None)) <= 1
print(f"\nn_synth_train parity {ns} -> {'OK' if ok else '⚠️ 불일치(cap 누락 의심)'}")

# (3) per-class synth 분포
print(f"\n{'조건':<10}  per-class synth (0-idx, -1=unparsed)")
for c in ("random", "casda", "aroma"):
    print(f"{c:<10}  {arms.get(c, {}).get('n_synth_per_class')}")

# (4) per-class map50 (c1~c4)
classes = sorted({c for cond in CONDS for c in (arms.get(cond, {}).get("per_class") or {})})
print(f"\n{'class':<6}  " + "  ".join(f"{c:>10}" for c in CONDS) + f"  {'Δ(A-R)':>10}")
for cls in classes:
    vals = {cond: ((arms.get(cond, {}).get('per_class') or {}).get(cls) or {}).get('map50') for cond in CONDS}
    row = "  ".join(f"{vals[c]:.4f}" if isinstance(vals[c], float) else f"{'N/A':>10}" for c in CONDS)
    a, r = vals.get("aroma"), vals.get("random")
    d = f"{a - r:+.4f}" if isinstance(a, float) and isinstance(r, float) else "N/A"
    print(f"{cls:<6}  {row}  {d:>10}")
```

> **성공 기준**: AROMA map50 ≥ CASDA, 3-seed CI 비겹침 여부, `n_synth_train` 3조건 동일, 소수 클래스(c2·c3·c4) AP에서 AROMA ≥ CASDA.

---

## 주의사항

- **Test set 원칙**: 합성 이미지는 train에만. test/defect는 항상 real 이미지만.
- **GPU 필수**: YOLO 학습은 CUDA 없이 실행 불가.
- **AROMA `n_synth=0` 진단**: `annotations.json`의 `image_path`가 실제 파일을 가리켜야 함(절대/synth 상대). 전부 resolve 실패 시 `had N entries but 0 resolved`. 로그의 `aroma train: N/… synth imgs labeled`에서 `N>0` 확인.
- **`annotations.json`의 `normal_image` 필드**: mask 없을 때 bbox 추출에 배경 경로로 사용.
- **합성 bbox 추출**: `mask`/`roi_mask`/`mask_path` 또는 `masks/{stem}.png`가 있으면 mask에서 직접. 없으면 composite−background diff → threshold=8(미검출 시 Otsu) → contour bbox.
- **real 라벨 0개**: `_get_real: 0 labels (no_mask=.. bad_size=.. no_bbox=.. no_lines=..)`. `no_mask`↑=mask 경로 매핑 문제, `no_bbox`↑=`min_area`(기본 50)가 해당 mask 해상도 대비 과대.
- **YOLO 캐시 schema_version=2**: bbox 로직 변경으로 기존 v1 캐시는 자동 무효화·1회 재빌드. 소스 결함 이미지 수가 변하면 자동 재빌드(`[YoloCache] built+saved`).
- **`n_train` 의미**: 레이블 있는 훈련 이미지 수(real_defect + synth). background negative(`n_real_train`만큼)는 미포함. 실제 YOLO가 보는 이미지 ≈ `n_train + n_real_train`.
- **세 조건 독립 실행**: baseline best.pt를 random/aroma가 미사용 → baseline 실패와 무관하게 `--condition random`/`aroma` 단독 실행 가능.
- **staging 로그 = 정상 신호**: real 이미지는 데이터셋당 1회 local staging(Drive→local `copy`), 조건별은 `hardlink`. `[Stage] ... done N in X.Xs (hardlink=H copy=C)`가 보이면 진행 중.
- **데이터셋 특성**: severstal(1600×256, 6.25:1) / aitex(4096×256, 16:1 극단 종횡비) → `--rect` 효과 큼. mvtec_leather·mtd는 full-frame 텍스처 → 배경 void 없어 배치 자동 정상.
