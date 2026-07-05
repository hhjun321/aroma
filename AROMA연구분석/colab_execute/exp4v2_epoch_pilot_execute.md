# Exp4v2 — epochs·patience 실측 재산정 파일럿 (Workflow)

**목적**: `--baseline_epochs 300`(보수 상한)을 실측 수렴점 기반으로 재산정 — 이후 모든 GPU run(real_frac 커브 포함)에 30–50% 절감 적용 (`low_compute_validation_plan.md` 4순위)
**대상**: severstal(대형) + mtd(소형) 2셀 × 3조건 — 소형은 epoch당 스텝이 적어 늦게 수렴할 수 있어 2종 필수
**런타임**: GPU 1–3시간 (이 run은 real_frac 커브의 **100% 지점과 겸용** — 순증 ≈ 0)
**코드 수정 없음** — 기존 exp4v2 커맨드 + CPU 분석 셀

---

## STEP 0 — 환경변수 (exp4v2_execute.md STEP 0 재사용)

```python
import os
os.environ['PILOT_OUT'] = f"{os.environ['AROMA_OUT']}/exp4v2/pilot"
```

## STEP 1 — 파일럿 실행 (단일 seed, 수렴점 추정 목적)

```python
!python $AROMA_SCRIPTS/experiments/exp4_v2_supervised_detection.py \
    --model yolov8n \
    --condition all \
    --dataset_keys severstal mtd \
    --random_synthetic_dir $RANDOM_SYNTH_DIR \
    --aroma_synthetic_dir  $AROMA_SYNTH_DIR \
    --real_data_dir        $AROMA_DATA \
    --output_dir           $PILOT_OUT \
    --yolo_cache_dir       $AROMA_OUT/yolo_cache \
    --imgsz 640 --val_frac 0.3 --synth_ratio 1.0 \
    --baseline_epochs 300 --patience 50 \
    --batch 64 --cache ram --rect \
    --seeds 1 \
    --resume
```

## STEP 2 — 수렴 분석 (CPU)

각 조건의 ultralytics `results.csv`에서 best epoch(mAP50-95 argmax)을 추출:

```python
import csv, os, math
from pathlib import Path

PILOT = os.environ['PILOT_OUT']
best = {}
for ds in ("severstal", "mtd"):
    for cond in ("baseline", "random", "aroma"):
        # checkpoint_dir 구조: _seeds/seed1/{ds}/yolov8n/{cond}/results.csv
        p = Path(PILOT) / "_seeds" / "seed1" / ds / "yolov8n" / cond / "results.csv"
        if not p.exists():
            print(f"missing: {p}"); continue
        rows = list(csv.DictReader(open(p)))
        col = next(c for c in rows[0] if "mAP50-95" in c)
        vals = [float(r[col]) for r in rows]
        be = max(range(len(vals)), key=lambda i: vals[i]) + 1
        best[(ds, cond)] = be
        print(f"{ds:<12} {cond:<10} best_epoch={be:>4}  (rows={len(rows)})")

max_be = max(best.values())
new_epochs = math.ceil(max_be * 1.3)
print(f"\nmax best_epoch = {max_be}  ->  권장 --baseline_epochs {new_epochs} --patience 30")
```

## STEP 3 — 사후 게이트 규칙 (사전 명시)

이후 본실험(새 상한 적용)에서 **어떤 셀의 best_epoch ≥ new_epochs × 0.9**이면 그 셀은 상한이 binding — **자동 재실행 대상**(상한 상향 후). synth를 받은 arm이 늦게 수렴해 Δ가 조건-비대칭으로 왜곡되는 것을 방지하는 규칙이며, 본실험 후 STEP 2 셀을 재사용해 확인한다.

## 산출 사용처

- 산정된 `--baseline_epochs <new> --patience 30`을 `exp4v2_realfrac_execute.md`의 모든 run에 적용.
- epochs 변경 시 fresh `--output_dir` 사용(resume skip-cache 규칙).
- 이 파일럿의 severstal·mtd 결과(100%, seed 1)는 real_frac 커브의 1.0 지점 데이터로 **재사용**(소형 mtd는 seed 2·43 추가 run만 보충).

## 한계 (정직)

- 절감 장치일 뿐 증거를 생산하지 않음. 수렴점은 seed 종속 분산 있음 — ×1.3 마진 + 사후 게이트가 방어선.
