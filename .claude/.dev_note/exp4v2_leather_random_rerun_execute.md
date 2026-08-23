# exp4v2 Leather — random arm 단독 재수행 가이드 (Colab)

- 작성일: 2026-08-19
- 목적: 20260813 결과에서 mvtec_leather **random arm 붕괴**(seed1 mAP50 0.0492, precision 0.0002) 치유. batch128+patience25 병리 (leather random 128장 = 정확히 1 batch → 1 update/epoch, patience 25에 조기 사살).
- 처치: kolektor 치유와 동일 knob — `--batch 16`, `--patience 0`, `--rect` 제거.
- 관련: `대응계획_SOP.md` E1 / T5 기준표. 절차 원형은 kolektor 재수행 (session-log 2026-08-13 #11).

---

## 사전 조건

- 기존 `$EXP4V2_OUT` (= `.../sym_final/exp4v2_tobe_gate`) 그대로 사용 — 기존 결과에 병합.
- synth 디렉토리 변경 없음 (`$SYNTH_RANDOM` 기존 것 재사용 — 합성은 문제없고 훈련 프로토콜만 교정).
- `--yolo_cache_dir` 캐시 키는 (min_area/val_frac/seed) — batch/patience 변경과 무관하게 재사용되어 **split 동일성 유지됨**.

## STEP 1 — per-seed 파일에서 leather random 항목만 제거

resume 단위는 per-seed JSON (`$EXP4V2_OUT/_seeds/seed{N}/exp4v2_results.json`)이고 skip은 dataset/model/condition 단위. random 키만 pop하면 baseline/aroma는 skip(기존값 유지), random만 재훈련된다.

```python
import json, shutil, os

out = os.environ['EXP4V2_OUT']
for s in [42, 1, 2]:
    p = f"{out}/_seeds/seed{s}/exp4v2_results.json"
    if not os.path.exists(p):
        print(f"seed{s}: 파일 없음 — skip (해당 seed는 처음부터 훈련됨)")
        continue
    shutil.copy(p, p + ".bak_leather_random")   # 백업
    with open(p) as f:
        r = json.load(f)
    removed = r.get("mvtec_leather", {}).get("yolov8n", {}).pop("random", None)
    with open(p, "w") as f:
        json.dump(r, f, indent=2)
    print(f"seed{s}: mvtec_leather/yolov8n/random removed = {removed is not None}")
```

## STEP 2 — random 조건만 재실행 (교정 knob)

```python
!python $AROMA_SCRIPTS/experiments/exp4_v2_supervised_detection.py \
    --model yolov8n \
    --condition random \
    --dataset_keys mvtec_leather \
    --class_mode multi \
    --aroma_synthetic_dir  $SYNTH_AROMA \
    --random_synthetic_dir $SYNTH_RANDOM \
    --real_data_dir        $AROMA_DATA \
    --output_dir           $EXP4V2_OUT \
    --yolo_cache_dir       $YOLO_CACHE \
    --imgsz 640 \
    --val_frac 0.3 \
    --synth_ratio 1.0 \
    --baseline_epochs 100 \
    --patience 0 \
    --batch 16 \
    --cache ram \
    --workers 12 \
    --seeds 42 1 2 \
    --resume
```

kolektor 치유 명령과의 차이: `--condition random` (baseline/aroma 제외), `--dataset_keys mvtec_leather`. knob 3개(batch 16 / patience 0 / rect 없음)는 동일.

## STEP 3 — 확인 포인트

1. **collapse 소멸**: random 3-seed 전부 mAP50 ≥ 0.5 (붕괴 시그니처 = mAP≈0.0x + precision≈0.00x + recall만 높음).
2. **최종 수치는 집계본에서**: `$EXP4V2_OUT/exp4v2_results.json` (3-seed 평균) — 20260813 파일은 seed1 단독 per-seed 파일이므로 판정에 쓰지 않는다.
3. **A−R 판정**: leather random 치유값 vs aroma 0.8674(seed1) — 단 aroma/baseline은 batch128 체제 값이므로 아래 주의 참조.
4. 기존 4개 데이터셋 수치 byte 불변 확인 (random pop 외 무접촉).

## ⚠️ 주의 — arm 간 프로토콜 분기 (T5 표 확정 전 결정 필요)

이 재수행 후 leather 행은 **random = batch16/patience0, baseline·aroma = batch128/patience25** 혼재 상태가 된다.

- 0812 관측: leather collapse는 random 2/3만이 아니라 **baseline 1/3, aroma 1/3**도 발생 — 3-seed 집계 시 baseline/aroma에도 붕괴 seed가 섞여 있을 수 있음.
- 리뷰어 관점: 동일 행 내 arm별 훈련 설정 상이 = 공정성 공격 표면 (R3 성향 리뷰어 취약점).
- 권장: random 치유값 확인 후, **논문 표 확정 전에 baseline·aroma도 동일 knob로 재수행** (kolektor와 동일한 3-arm 동일 처치). 이번 random 단독 실행은 진단·중간 확인 단계로 취급.
- 최종 표 각주: leather·kolektor = batch 16 / patience 0 / rect 없음, severstal·mtd·aitex = batch 128 / patience 25 / rect (n_train ≫ batch라 병리 무해) — 프로토콜 분기 명시.
