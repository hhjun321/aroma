# Severstal 4조건 exp4v2 재실행 가이드 (synth_ratio scale-up)

**목적**: `baseline / random / casda / aroma` 4조건을 **동일 TOTAL synth budget**으로 공정 비교 (mAP50, multi-class).
**핵심 변경**: synth pool을 600 → ~5070 으로 확대 (`top_k 200→1690`, CASDA `per_class_cap=525`), `synth_ratio=1.0` primary.
**런타임**: GPU (Colab Pro A100 권장).

이 문서는 **오케스트레이션 가이드**다. 실제 셀은 기존 두 문서를 순서대로 참조한다:
- **R** = `severstal_exp4v2_multiclass_fix_rerun.md` (AROMA·random 생성)
- **E** = `exp4v2_execute.md` (CASDA 생성 + 4조건 학습 + 검증)

> **프레이밍** (확정): headline = **isolate-full-pipeline** — TOTAL budget만 4조건 동일, per-class 분배는 각 전략 자율(AROMA deficit-aware가 c2에 자연히 더 기움). inverse-frequency 고정쿼터는 **추후 ablation**으로 분리(이번 run 범위 밖).

---

## ⚠️ Pre-flight 체크 (실행 전 반드시)

### P1. 배경 풀(NORMAL_DIR) 정렬 — 공정성 load-bearing
현재 두 문서가 **서로 다른 배경 풀**을 쓴다:
- R(AROMA/random): `$DRIVE/severstal/context_select/context`
- E(CASDA): `$AROMA_DATA/severstal/train/good`

조건마다 배경 분포가 다르면 "ROI 선택 전략"만 격리되지 않는다(배경이 교란). **3 생성기 모두 동일 NORMAL_DIR을 쓰도록 맞춘다.** 하나를 골라 R·E 양쪽 env 셀의 `NORMAL_DIR`을 통일할 것.
- 권장: 학습 때 실제 평가 도메인과 같은 풀. (`context_select/context`는 큐레이션 subset, `train/good`은 전량 5902.) 어느 쪽이든 **3조건 동일**이 핵심.

### P2. 코드 pull
새 `.py`(병렬 staging/push, `--condition` 다중선택, CASDA manifest staging)가 Colab에 있어야 재생성 push가 빠르다. **로컬 미커밋 상태면 먼저 commit+push 후:**
```python
!cd /content/AROMA && git pull
```
> 안 하면 push가 순차 → 조건당 ~5070×2 파일 Drive 쓰기로 매우 느림. pull로 병렬 push 적용.

### P3. seed 확정
E의 Cell C는 `--seeds 42 1337 2025`. 직전 run 로그는 `42 1 2`였음 — **의도한 seed로 통일**할 것(재현성).

### P4. (선택) 복사 동시성
```python
import os; os.environ['AROMA_STAGE_WORKERS'] = '24'   # 기본 16, 1~64
```

---

## Phase 1 — AROMA + random 재생성 (문서 R)

1. **R 상단 환경 셀** 실행. `class_mode: multi` 출력 확인 (아니면 git pull 누락).
   - ⚠️ P1대로 `NORMAL_DIR` 통일 여부 확인.
2. **R §2 — roi_selection (AROMA)**: `--top_k 1690 --class_mode multi --class_floor --per_pair_cap_frac 0.05`
3. **R §2 — random arm**: `--sampling_strategy random --top_k 1690` (이전 top_k=200 캐시 덮어씀)
4. **R §2.1 게이트** (학습 전 필수). 합격 기준:
   - class 1~4 전부 count > 0 (starvation 해소)
   - **c2 ≈ 117** (floor 422 미달이 정상 — availability clamp 작동 = c2-full directive 증거)
   - 최다 (cluster,cell) pair ≤ `ceil(0.05×1690)=85`
5. **R §3 — generate_defects (AROMA)**: `--n_per_roi 3 --local_staging`
   **R §3 — generate_random**: `--top_k 1690 --n_per_roi 3 --local_staging`
6. 확인: AROMA·random `annotations.json` 각 **≈ 5070** (1690×3).

> `--n_per_roi=3`은 AROMA·random·CASDA **세 생성기 동일**(생성단계 parity 불변식).

---

## Phase 2 — CASDA 재생성 (문서 E)

1. **E CASDA 환경 셀(≈L535~)** 실행 (`ROI_DIR`→CASDA roi_patches, `CASDA_SYNTH_DIR`, `N_PER_ROI=3`).
   - ⚠️ P1대로 `NORMAL_DIR`을 Phase 1과 동일하게.
2. `roi_metadata.csv` 없으면 **E Cell A**(extract_rois) 먼저 실행.
3. **E Cell B — generate_casda**: `--per_class_cap 525 --min_suitability 0.5 --n_per_roi $N_PER_ROI --local_staging`
4. 로그 확인:
   - `n_filtered_cap` > 0 (abundant class가 525 cap에 걸림)
   - c2는 candidate ≈117 < 525 → **전량 선택**(AROMA와 동일 c2-full 형태)
   - `annotations.json` ≈ **5076** ((525+117+525+525)×3)
   - push 로그: `push_outputs → ...: X staged, ... 0 failed ... using 24 workers` (병렬 push 적용 확인)

---

## Phase 3 — 옛 결과 purge (synth_ratio 변경)

`synth_ratio`/pool이 바뀌어 `--resume`이 옛 severstal 결과를 skip하면 안 된다. 둘 중 하나:

**(a) 기존 결과에서 severstal 항목 제거**
```python
import json, os, glob
base = f"{os.environ['EXP4V2_OUT']}/severstal_casda"
for p in [f"{base}/exp4v2_results.json"] + glob.glob(f"{base}/_seeds/seed*/exp4v2_results.json"):
    if os.path.exists(p):
        d = json.load(open(p)); d.pop("severstal", None)
        json.dump(d, open(p, "w"))
        print("purged severstal in", p)
```

**(b) (더 안전) fresh output_dir 사용** — Cell C `--output_dir`를 `$EXP4V2_OUT/severstal_casda_r1.0`로.

---

## Phase 4 — 4조건 학습 (문서 E Cell C)

**E Cell C** 그대로 실행 (현재 셀 = scale-up 반영됨):
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
    --imgsz 640 --val_frac 0.3 \
    --baseline_epochs 100 --patience 20 \
    --batch 64 --cache ram --rect \
    --yolo_cache_dir $AROMA_OUT/yolo_cache \
    --resume
```
- cap = max(1, int(2534×1.0)) = **2534**. pool 5070/5070/5076 ≥ 2534 → 3 synth arm 전부 정확히 2534로 trim → 동일 budget.
- 일부 조건만: `--condition baseline random aroma` (공백 나열, 다중선택).
- 소요: 단일 run × seed 수 (4조건 × 3seed).

---

## Phase 5 — 검증 (문서 E 확인 셀)

**E "CASDA vs AROMA 결과 확인" 셀** 실행:
- **(2) parity**: `n_synth_train` random/casda/aroma **동일(=2534)** → `OK`여야 공정. 불일치 시 cap 누락 의심.
- **(3) per-class synth**: `n_synth_per_class` — 조건 간 분포 확인.
- **per-class AP**: `arms[cond]['per_class']` — 희소 class(c3·c4 + c2)에서 AROMA ≥ CASDA 확인.

**성공 기준**: AROMA map50 > CASDA, 3-seed 95% CI 비겹침, `n_synth_train` 3조건 동일.

> **무결성**: 선택 **전략**만 조건별로 다르고(그게 실험), pool 규모·n_per_roi(=3)·학습 cap budget은 4조건 동일. AROMA에만 큰 pool / CASDA만 random-trim / c2 AROMA만 과반복 / 이기는 ratio cherry-pick = rig, 모두 회피. **나온 결과 그대로 보고.**
> 평가는 **real c2 test의 per-class precision**도 확인(synth val은 artifact 공유로 c2 이득 과대).

---

## 스윕 (선택, 추후)

1.5(cap=3801)·2.0(cap=5068)는 **생성 재사용**(pool 5070/5076이 cap 5068까지 커버). 생성 0회, Cell C `--synth_ratio 1.5`(또는 2.0) + **fresh `--output_dir`**(`severstal_casda_r1.5`)로 재실행.

## 추후 ablation (이번 범위 밖)

inverse-frequency / per-class 고정쿼터 (isolate-selection): "동일 class mix서 누구 크롭이 나은가". 라벨 분리해 별도 실행. c2는 117 distinct 천장 → 351(3x)~585(5x) 상한, 그 이상은 precision collapse. 코드 필요 시 `generate_defects`에 `--n_per_roi_map` (~15-30줄) micro-fix.
