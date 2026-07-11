# Step 3.5 — `clean_bg_selection.py` (clean-bg 사전 선정) Colab 실행

> **목적**: 원본 good을 생성 시점에 재스캔하지 않고, **프로파일링 파생 파일**(`context_features.csv`·`compatibility_matrix.json`)과 `roi_selected.json`으로 clean-background를 **사전 선정**한다. `roi_selection`(profiling→`roi_selected.json`)과 대칭인 신규 단계로, `clean_bg_selected.json`을 산출하고 step4(`generate_defects`)가 소비한다.
> **실행 환경**: **CPU**. GPU·픽셀 재스캔 불요(셀 정보는 이미 context_features에 있음).
> **체인 위치**: phase0 → step1 → step2 → **step3(roi_selection)** → **step3.5(clean_bg_selection, 본 문서)** → step4(generate_defects) → exp*.
> **경로 주의**: `clean_bg_selection.py`는 `scripts/aroma/`에 있다.

---

## 정직성 (읽고 시작)

- 히스토그램 매칭의 변별력은 **도메인-조건부**: aitex는 강한 신호(로컬 검증 best hist∩ 0.89), **severstal/mtd는 랜덤 배경과 사실상 구분 불가**(E1, `pivot_local_validation_20260711.md`).
- 본 단계의 **확실한 가치 = 재현성 + 대칭 대조군(random arm에 동일 배경 배정) + per-seed 배치 분산 제거**. **일반적 mAP 향상은 주장하지 않는다**(GPU 검증 별도).
- **data-driven(no-hardcoding)**: void 컷·pool 컷은 관측 분포에서 유도(void_frac_max=P90, pool=P95, void floor=P1). 유도값은 `clean_bg_summary.md`에 기록.
- **leather 제외**: `mvtec_leather`는 `image_id` stem-collision(dev_note `aroma_phase0_image-id-unique-key.md`) 때문에 context_features 조인이 무효 → **phase0 재실행(고유키) 후에만** 유효. Phase 1은 severstal/mtd/aitex.

---

## STEP 0 — 공통 환경 셀 (sym_final 전 문서 동일 — 그대로 복사)

```python
import os, json

os.environ['DRIVE']          = '/content/drive/MyDrive/data/Aroma'
os.environ['AROMA_REF']      = '/content/AROMA'
os.environ['AROMA_SCRIPTS']  = '/content/AROMA/scripts/aroma'
os.environ['AROMA_OUT']      = f"{os.environ['DRIVE']}/aroma_output"
os.environ['DATASET_CONFIG'] = os.environ.get('DATASET_CONFIG', '/content/AROMA/dataset_config.json')
os.environ['SYM_ROOT'] = f"{os.environ['AROMA_OUT']}/sym_final"
def S(stage, ds=None):
    p = f"{os.environ['SYM_ROOT']}/{stage}"
    return f"{p}/{ds}" if ds else p

# Phase 1: leather 제외 (phase0 고유키 재실행 후 추가)
DATASETS = ["severstal", "mtd", "aitex"]
```

---

## STEP 1 — clean_bg_selection 실행 (DATASETS 루프)

`roi_selection`(step3) 산출 `roi/{ds}/roi_selected.json`이 있어야 한다. 출력은 같은 `roi/{ds}`에 쓴다(step4가 자동 로드).

```python
for DS in DATASETS:
    os.environ['DS']   = DS
    os.environ['PROF'] = S('profiling', DS)     # context_features.csv, compatibility_matrix.json
    os.environ['ROI']  = f"{os.environ['AROMA_OUT']}/roi/{DS}"  # roi_selected.json (step3 출력)
    print(f"\n===== clean_bg_selection: {DS} =====")
    !python $AROMA_SCRIPTS/clean_bg_selection.py \
        --profiling_dir  $PROF \
        --roi_dir        $ROI \
        --output_dir     $ROI \
        --emit_random_arm
```

> `--emit_random_arm`: 대칭 대조군용 `clean_bg_random_arm.json`(동일 ROI 집합, random 배경)을 함께 생성. **AROMA arm vs random arm이 배경 정체성만 다르고 배치/블렌딩은 동일**해지는 계측기(step4에서 `--clean_bg_json`으로 선택).
>
> **선택 인자**:
> - `--pool_k <int>` — per-ROI 배경 풀 크기 상한(기본: 데이터-유도 P95). 명시 시 고정.
> - `--void_frac_max <float>` — void 컷 상한(기본: 데이터-유도 P90). 명시 시 고정.
> - `--no_reject_clean_bg` — void/품질 선행 필터 비활성(전 good 유지).

---

## STEP 2 — 산출 확인 + E1 재현 게이트

### 2-1. 파일·요약 확인

```python
from pathlib import Path
for DS in DATASETS:
    roi = f"{os.environ['AROMA_OUT']}/roi/{DS}"
    for f in ['clean_bg_selected.json', 'clean_bg_random_arm.json', 'clean_bg_summary.md']:
        print(f"  {'OK  ' if Path(f'{roi}/{f}').exists() else '누락 '} {DS}/{f}")
    print(Path(f"{roi}/clean_bg_summary.md").read_text(encoding='utf-8').split('## Sample')[0])
```

### 2-2. E1 재현 하드 게이트 (context_features 소싱 ≡ 픽셀 소싱)

배포 전 필수: 오프라인 히스토그램 매칭이 생성-시점 값을 재현하는지. `clean_bg_selected`의 best `hist_intersection`이 E1의 sim_best와 근접해야 한다.

```python
import json, statistics as st
# 로컬 검증 기준값 (AROMA연구분석/pivot_local_validation_20260711.md, scripts_local/e1.py)
E1_SIM_BEST = {"aitex": 0.895, "mtd": 0.502, "severstal": 0.623}
for DS in DATASETS:
    sel = json.load(open(f"{os.environ['AROMA_OUT']}/roi/{DS}/clean_bg_selected.json"))
    hi = [s['hist_intersection'] for s in sel if s.get('assigned_normal_id')]
    m = st.mean(hi) if hi else 0.0
    ref = E1_SIM_BEST.get(DS)
    ok = ref is None or abs(m - ref) < 0.05
    print(f"{DS:10s} best hist∩ mean {m:.3f}  vs E1 {ref}  → {'PASS' if ok else 'DRIFT(조사)'}")
```

> DRIFT 시(±0.05 초과): profiling의 비-overlap truncate 타일링 ↔ generate_defects의 far-edge-inclusive 타일링 **discretization 갭** 조사(설계 §2). 통과 못 하면 배포 보류.

---

## STEP 3 — step4(generate_defects) 소비 (참고 — 상세는 step4 문서)

`generate_defects`가 `roi_dir/clean_bg_selected.json`을 **자동 로드**한다. 별도 인자 불요:

```python
# AROMA arm (clean_bg_selected.json 자동 소비)
!python $AROMA_SCRIPTS/generate_defects.py \
    --roi_dir $ROI --normal_dir $NORMAL_DIR --output_dir $SYNTH_DIR \
    --method copy_paste --n_per_roi 3 --seed 42 --blend_mode seamless \
    --reject-clean-bg --min-bg-quality 0.7 --bg-blur-threshold 100.0

# 대칭 대조군 arm (같은 배경 배정을 random arm에)
!python $AROMA_SCRIPTS/generate_defects.py \
    --roi_dir $ROI --normal_dir $NORMAL_DIR --output_dir ${SYNTH_DIR}_randarm \
    --clean_bg_json $ROI/clean_bg_random_arm.json \
    --method copy_paste --n_per_roi 3 --seed 42 --blend_mode seamless \
    --reject-clean-bg --min-bg-quality 0.7 --bg-blur-threshold 100.0
```

> 로그에 `clean_bg assignment ON: clean_bg_selected.json (N ROIs)`가 떠야 소비 성공. 안 뜨면(파일 없음/로드 실패) **legacy 생성-시점 선정으로 자동 fallback**(무해).
> 배경 정체성은 json에서 rep→pool 결정론적 인덱싱(`pool[rep_idx % K]`)으로 오며, **rng draw 0** → AROMA·random arm이 배치/feather 스트림 동일(배경만 상이) = 대칭 대조군.

---

## 판정 / 다음 단계

- [ ] STEP 1: 3종 `clean_bg_selected.json` + `clean_bg_random_arm.json` 생성
- [ ] STEP 2-2: **E1 재현 PASS**(mtd~0.50 / aitex~0.89 / severstal~0.62, ±0.05) — 배포 하드 게이트
- [ ] STEP 3: step4 로그에 `clean_bg assignment ON` 확인, random arm 별도 생성

통과 시 → **step4**(생성) → exp4v2에서 **AROMA arm vs random-arm(대칭 대조)** 비교. GPU 학습 후에야 mAP 판정(현재 TBD).

---

## 무결성 / 정직

- 원본 good 픽셀 재스캔 금지 — 선정은 profiling 파생 파일에서만. (paste 픽셀은 step4가 배정된 normal만 연다.)
- 임계·pool은 데이터-유도(P90/P95/P1), magic number 금지. 유도값은 summary에 기록.
- **severstal/mtd는 매칭 신호 약함**(E1) → "context 배치가 mAP를 올린다" 무검증 주장 금지. 대칭 대조군은 "통하는지 측정"하는 계측기.
- **leather는 phase0 고유키 재실행 후** 추가(현재 제외). Phase 2(3-기준 class/bg-type)·Phase 3(bg-type per-image + 기하 prior)은 후속 dev_note 범위.
- 사후 튜닝 금지, pytest 금지(CLAUDE.md) — 검증은 본 셀 실행 + E1 재현.
