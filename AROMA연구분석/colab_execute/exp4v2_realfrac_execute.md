# Exp4v2 — real_frac label-efficiency 커브 실행 가이드 (Workflow)

**목적**: 최소 downstream 증거(L3) — real 25/50/100% × 3조건(baseline/random/aroma) 커브로 "real이 희소할수록 AROMA-선택 합성이 gap을 더 메우는가" 검증 (`low_compute_validation_plan.md` 5순위)
**커브 데이터셋**: `severstal`(대형) + `mtd`(소형) 2종. **도메인 폭**은 4종 × 100% 지점으로 분담
**런타임**: GPU 총 5–8시간 추정 (25/50% 지점은 데이터가 적어 빠름; epoch 파일럿 산정값 적용 시)

---

## 주장 경계 (사전 등록 — load-bearing)

- **주 주장 = Δ(aroma−random) 커브**. 25/50% 지점에서 제외된 real에 합성 ROI 소스가 포함될 수 있으나(leakage), **양 arm에 대칭**이라 A-vs-R 비교는 유효.
- **vs-baseline "few-shot 해결" 서사 금지** — leakage가 비대칭으로 작용할 수 있는 축이라 각주로만. ROI/합성 pool 구성을 논문에 투명 기술.
- 10% 지점 제외(mtd 결함 수 한 자릿수 → split 불안정). AUBC 단일 스칼라 대신 **지점별 Δ + 단조성**.
- seed 정책(분산이 작은 곳에서만 절감): severstal `--seeds 1 2` / mtd(및 100% 지점의 aitex·leather) `--seeds 1 2 43`.

**선행 조건**: ① epoch 파일럿(`exp4v2_epoch_pilot_execute.md`) 완료 → `<EPOCHS>`/`--patience 30` 확정, ② mvtec_leather aroma 합성 생성, ③ aitex는 union-bbox fix(schema v3) 후 캐시 자동 재빌드.

---

## STEP 0 — 환경변수 (exp4v2_execute.md STEP 0 + 추가)

```python
import os
os.environ['RF_OUT'] = f"{os.environ['AROMA_OUT']}/exp4v2"   # 지점별 서브디렉토리 사용
```

> ⚠️ **frac 지점별 fresh output_dir 필수**: results key가 dataset 단위라 같은 dir에 다른 frac을 돌리면 `--resume`이 이전 지점을 skip한다. `frac025`/`frac050`/`frac100`으로 분리.

## STEP 1 — 커브 실행 (severstal + mtd × frac 0.25 / 0.5)

```python
for FRAC, TAG in [("0.25", "frac025"), ("0.5", "frac050")]:
    os.environ['FRAC'] = FRAC
    os.environ['TAG']  = TAG
    # severstal: 2 seeds (분산 작음 — 절감 지점)
    !python $AROMA_SCRIPTS/experiments/exp4_v2_supervised_detection.py \
        --model yolov8n --condition all \
        --dataset_keys severstal \
        --random_synthetic_dir $RANDOM_SYNTH_DIR \
        --aroma_synthetic_dir  $AROMA_SYNTH_DIR \
        --real_data_dir        $AROMA_DATA \
        --output_dir           $RF_OUT/$TAG \
        --yolo_cache_dir       $AROMA_OUT/yolo_cache \
        --imgsz 640 --val_frac 0.3 --synth_ratio 1.0 \
        --real_frac $FRAC \
        --baseline_epochs <EPOCHS> --patience 30 \
        --batch 64 --cache ram --rect \
        --seeds 1 2 --resume
    # mtd: 3 seeds (분산 큼 — 절감 대상 아님, GPU 소요도 작음)
    !python $AROMA_SCRIPTS/experiments/exp4_v2_supervised_detection.py \
        --model yolov8n --condition all \
        --dataset_keys mtd \
        --random_synthetic_dir $RANDOM_SYNTH_DIR \
        --aroma_synthetic_dir  $AROMA_SYNTH_DIR \
        --real_data_dir        $AROMA_DATA \
        --output_dir           $RF_OUT/$TAG \
        --yolo_cache_dir       $AROMA_OUT/yolo_cache \
        --imgsz 640 --val_frac 0.3 --synth_ratio 1.0 \
        --real_frac $FRAC \
        --baseline_epochs <EPOCHS> --patience 30 \
        --batch 64 --cache ram --rect \
        --seeds 1 2 43 --resume
```

> 로그 확인: `[RealFrac] {ds}: real train N -> k (frac=..., val unchanged=...)` + `[SynthRatio]` cap이 축소된 train 기준으로 재계산되는지.
> YOLO real 캐시는 frac별 `real*_frac{f}` leaf로 자동 격리 — full 캐시와 thrash 없음.

## STEP 2 — 100% 지점 + 도메인 폭 (4종)

epoch 파일럿의 severstal·mtd(seed 1) 결과를 재사용하고, 부족분만 보충:

```python
# severstal seed 2 보충 + mtd seed 2,43 보충 + aitex/leather 3 seeds (도메인 폭)
!python $AROMA_SCRIPTS/experiments/exp4_v2_supervised_detection.py \
    --model yolov8n --condition all \
    --dataset_keys severstal mtd aitex mvtec_leather \
    --random_synthetic_dir $RANDOM_SYNTH_DIR \
    --aroma_synthetic_dir  $AROMA_SYNTH_DIR \
    --real_data_dir        $AROMA_DATA \
    --output_dir           $RF_OUT/frac100 \
    --yolo_cache_dir       $AROMA_OUT/yolo_cache \
    --imgsz 640 --val_frac 0.3 --synth_ratio 1.0 \
    --baseline_epochs <EPOCHS> --patience 30 \
    --batch 64 --cache ram --rect \
    --seeds 1 2 43 --resume
```

> severstal은 seed 3개가 필요 없으면(정책상 1 2) `_seeds/seed43` 결과를 집계에서 제외해도 된다 — 집계 셀에서 seed 선택.

## STEP 3 — 커브 집계 (지점별 Δ + 단조성)

```python
import json, os

RF = os.environ['RF_OUT']
FRACS = [("0.25", "frac025"), ("0.5", "frac050"), ("1.0", "frac100")]
SEEDS = {"severstal": [1, 2], "mtd": [1, 2, 43]}

print(f"{'ds':<12} {'frac':>6} {'baseline':>10} {'random':>10} {'aroma':>10} {'Δ(A-R)':>10}")
print("-" * 64)
curve = {}
for ds in ("severstal", "mtd"):
    deltas = []
    for frac, tag in FRACS:
        try:
            res = json.load(open(f"{RF}/{tag}/exp4v2_results.json"))
            arms = res[ds]["yolov8n"]
        except (FileNotFoundError, KeyError):
            print(f"{ds:<12} {frac:>6}  (미실행)"); continue
        b = arms.get("baseline", {}).get("map50")
        r = arms.get("random", {}).get("map50")
        a = arms.get("aroma", {}).get("map50")
        d = (a - r) if isinstance(a, float) and isinstance(r, float) else None
        deltas.append((float(frac), d))
        fmt = lambda v: f"{v:.4f}" if isinstance(v, float) else "N/A"
        print(f"{ds:<12} {frac:>6} {fmt(b):>10} {fmt(r):>10} {fmt(a):>10} "
              f"{(f'{d:+.4f}' if d is not None else 'N/A'):>10}")
    # 단조성: real 이 적을수록 Δ 가 큰가 (frac↑ → Δ↓ 기대)
    ds_d = [d for _, d in sorted(deltas) if d is not None]
    if len(ds_d) >= 2:
        mono = all(ds_d[i] >= ds_d[i+1] for i in range(len(ds_d)-1))
        print(f"{'':<12} 단조성(Δ 비증가): {'✅' if mono else '❌ (지점별 해석)'}\n")
```

**판정**: ① 각 지점 Δ(A−R) > 0(특히 저-frac에서), ② Δ가 frac 증가에 따라 비증가(단조성) — 둘 다 성립하면 "real 희소 구간에서 AROMA 선택 가치가 커진다" 주장 성립. 도메인 폭(100% 4종)은 exp4v2_execute.md의 multi-seed 집계 셀로 별도 확인.

## STEP 4 — 사후 게이트 확인

`exp4v2_epoch_pilot_execute.md` STEP 2 셀을 각 output_dir에 대해 재실행 — best_epoch ≥ 상한×0.9인 셀은 상한 상향 후 재실행(사전 등록 규칙).

---

## 주의사항

- **val 불변 확인**: 전 frac 지점 로그의 `val unchanged=N`이 동일해야 함 — 다르면 split 규약 위반.
- **synth cap**: `--synth_ratio 1.0` 기준 cap = 축소된 train 수 — 25% 지점 severstal cap ≈ 634 → pool(5070) 충분.
- seed별 real 부분집합이 다름(`--seeds` 반복 시) — 의도된 동작(분산에 subsample 변동 포함).
- strict leakage 변형(25% 1지점, ROI 소스를 retained real로 제한)은 본 커브 결과 후 필요성 판단(TODO).
