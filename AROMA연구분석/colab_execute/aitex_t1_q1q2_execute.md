# AITeX T1-Q1 / T1-Q2 실행 가이드 — 프레임워크 우월성 vs ROI 선택 단일효과

> **근거 문서**: `AROMA연구분석/roi_check_improvement_guide_20260708.md`(테스트 매트릭스 T1-Q1/Q2), `AROMA연구분석/newpipe_asis_tobe_reanalysis_20260708.md`(Q1/Q2 분리).
> **대상**: tiled AITeX (유일한 headroom+positive 데이터셋, baseline≈0.372). MTD는 near-ceiling이라 제외.
> **선행 조건**: (a) AITeX tiled ROI 후보/선택 산출(`step2/step3_execute.md`), (b) AITeX ControlNet 학습 완료(`controlnet_aroma_arm_execute.md` STEP 4 → `$CN_MODELS/aitex/best_model`).

---

## 0. 무엇을 답하는 실험인가 (두 질문을 분리한다)

`20260704_1`(MTD)은 **full-stack AROMA vs naive random**을 섞어 비교했고 near-ceiling이라 결론이 안 났다. 여기서는 headroom 있는 AITeX에서 **두 질문을 각각 통제된 arm으로** 답한다.

| 질문 | 비교 | arm 차이 | 답하는 것 |
|------|------|----------|-----------|
| **Q1 (프레임워크 우월성)** | B vs A | 선택+엔진+blend 전부 | AROMA 프레임워크 총동원이 단순 증강을 이기는가 |
| **Q2 (ROI 선택 단일효과)** | C vs D | **선택만** (엔진·blend 동일) | ROI 선택 전략 자체의 기여 (논문 Table 13) |

### 4개 합성 arm 정의

| arm | 선택 | 합성 엔진 | blend | 용도 |
|-----|------|-----------|-------|------|
| **A** random-naive | random | copy_paste | **alpha** | Q1 baseline (naive) |
| **B** aroma-fullstack | **realism** | **controlnet** | **seamless** | Q1 AROMA (총동원) |
| **C** aroma-cp | **realism** | copy_paste | seamless | Q2 AROMA |
| **D** random-cp | random | copy_paste | seamless | Q2 random |

- **Q1 = B vs A**, **Q2 = C vs D**. baseline(real-only)은 두 run 공통.
- ⚠️ **Q1 pair와 Q2 pair를 교차 비교하지 말 것.** B vs D, C vs A 같은 조합은 무의미(요인 혼재). 각 질문은 자기 pair 안에서만 판정.
- B(aroma-fullstack)와 C(aroma-cp)는 **동일 `roi_selected.json`(realism)**을 공유 — 선택은 한 번만, 합성만 두 번.

---

## 1. 환경변수

```python
import os, json

os.environ['AROMA_SCRIPTS']  = '/content/AROMA/scripts/aroma'
os.environ['AROMA_OUT']      = f"{os.environ['DRIVE']}/aroma_output"
os.environ['DATASET_CONFIG'] = os.environ.get('DATASET_CONFIG', '/content/AROMA/dataset_config.json')

DS = 'aitex'                      # tiled AITeX (dataset_config의 tiled 레이아웃 entry)
os.environ['DS']         = DS
os.environ['ROI_DIR']    = f"{os.environ['AROMA_OUT']}/roi/{DS}"                 # roi_candidates.json / roi_selected.json
os.environ['CN_MODELS']  = f"{os.environ['AROMA_OUT']}/controlnet_models"        # $CN_MODELS/aitex/best_model
os.environ['T1_OUT']     = f"{os.environ['AROMA_OUT']}/t1_qexp/{DS}"             # 이 실험 전용 출력 루트

# normal_dir 은 dataset_config 에서 자동 조회
with open(os.environ['DATASET_CONFIG']) as f:
    _cfg = json.load(f)
os.environ['NORMAL_DIR'] = _cfg[DS]['image_dir']
print("NORMAL_DIR =", os.environ['NORMAL_DIR'])

# 4개 arm 출력 디렉토리
for arm in ['A_random_naive', 'B_aroma_fullstack', 'C_aroma_cp', 'D_random_cp']:
    os.environ[f'SYN_{arm[0]}'] = f"{os.environ['T1_OUT']}/{arm}"
print("arms:", {k: os.environ[k] for k in ['SYN_A','SYN_B','SYN_C','SYN_D']})
```

> ⚠️ **재생성 전 기존 출력 삭제 필수** (append 금지). 4개 arm 모두 `$T1_OUT` 하위이므로 재현 시 `!rm -rf $T1_OUT` 후 처음부터.

---

## 2. STEP 1 — ROI 선택 (AROMA=realism, 1회)

B·C가 공유할 realism 선택을 **한 번만** 생성한다. random arm(A·D)은 `generate_random.py`가 자체 random 선택을 하므로 별도 선택 불필요.

```python
# AROMA realism 선택 → $ROI_DIR/roi_selected.json (B·C 공유)
!python $AROMA_SCRIPTS/roi_selection.py \
    --roi_dir     $ROI_DIR \
    --top_k       200 \
    --seed        42 \
    --score_mode  realism
```

> `--score_mode realism` = `0.5·ctx + 0.3·morph + 0.2·quality` (newpipe의 to-be 공식). legacy 대비 실측 효과는 16.5% quality 스왑 수준이나, 여기서는 **AROMA 선택의 표준으로 realism을 고정**하고 Q2에서 그 기여를 측정한다.
> `roi_candidates.json`은 step2 산출물이 `$ROI_DIR`에 있어야 한다. `--min_quality`는 0.0(OFF) 유지 — Q2 대칭을 위해 A/D의 generate_random 에도 동일 값 적용.

---

## 3. STEP 2 — 4개 arm 합성

**parity 원칙**: 네 arm의 합성 수(n_synth)가 동일해야 공정하다. `--n_per_roi`를 동일하게 두고(예 3), 합성 후 개수를 반드시 대조한다(STEP 2.5).

```python
NPR = 3                                  # 모든 arm 공통
os.environ['NPR'] = str(NPR)
os.environ['CN_PATH'] = f"{os.environ['CN_MODELS']}/{DS}/best_model"
```

### arm A — random-naive (random 선택 + copy_paste + alpha)

```python
!python $AROMA_SCRIPTS/generate_random.py \
    --candidates_json $ROI_DIR/roi_candidates.json \
    --normal_dir      $NORMAL_DIR \
    --output_dir      $SYN_A \
    --top_k           200 \
    --n_per_roi       $NPR \
    --seed            42 \
    --blend-mode      alpha \
    --min_quality     0.0
```

### arm D — random-cp (random 선택 + copy_paste + seamless)

```python
# A와 동일하되 blend 만 seamless (Q2 대칭용)
!python $AROMA_SCRIPTS/generate_random.py \
    --candidates_json $ROI_DIR/roi_candidates.json \
    --normal_dir      $NORMAL_DIR \
    --output_dir      $SYN_D \
    --top_k           200 \
    --n_per_roi       $NPR \
    --seed            42 \
    --blend-mode      seamless \
    --min_quality     0.0
```

### arm C — aroma-cp (realism 선택 + copy_paste + seamless)

```python
!python $AROMA_SCRIPTS/generate_defects.py \
    --roi_dir     $ROI_DIR \
    --normal_dir  $NORMAL_DIR \
    --output_dir  $SYN_C \
    --method      copy_paste \
    --blend_mode  seamless \
    --n_per_roi   $NPR \
    --seed        42
```

### arm B — aroma-fullstack (realism 선택 + controlnet + seamless)

```python
# ⚠️ --local_staging 금지 (controlnet sidecar 캐시가 Drive 직결이어야 세션 재개 시 GPU skip)
!python $AROMA_SCRIPTS/generate_defects.py \
    --roi_dir         $ROI_DIR \
    --normal_dir      $NORMAL_DIR \
    --output_dir      $SYN_B \
    --method          controlnet \
    --controlnet_path $CN_PATH \
    --blend_mode      seamless \
    --n_per_roi       $NPR \
    --seed            42 \
    --cn_ar_threshold 2.5
```

> **AITeX 세장형 주의**: `controlnet_aroma_arm_execute.md` STEP 5에서 aitex는 AR 임계 2.5에서 fallback이 매우 높았다(사례: 98%). fallback이 과도하면 arm B가 사실상 copy_paste가 되어 **Q1이 Q2로 붕괴**한다. 아래 STEP 2.5에서 `controlnet stats`의 `ar_fallback` 비율을 반드시 확인하고, 과도하면(>50%) 논문/보고에 "arm B는 대부분 copy_paste 폴백 — Q1 해석 제한"으로 **정직 표기**한다(임계 상향은 아티팩트 위험이라 신중).

### STEP 2.5 — parity & 무결성 확인 (필수)

```python
import json, os
from collections import Counter
def load(a):
    with open(f"{os.environ[f'SYN_{a}']}/annotations.json") as f: return json.load(f)
for a in ['A','B','C','D']:
    ann = load(a)
    real = [x for x in ann if not x.get('dry_run')]
    methods = Counter(x.get('method') for x in real)
    blends  = Counter(x.get('blend_mode') for x in real)
    print(f"arm {a}: n_synth={len(real):4d}  methods={dict(methods)}  blend={dict(blends)}")
# 기대: A/D n_synth 동일, C n_synth 동일 수준. B는 ar_fallback 만큼 copy_paste_arfallback 섞임 → 비율 기록.
```

- **수용 기준**: A·D·C·(B) 의 `n_synth`가 근접(±5% 이내). 크게 어긋나면 `--n_per_roi` 재조정 후 재생성.
- **arm B ar_fallback 비율** = `copy_paste_arfallback / (controlnet + copy_paste_arfallback)` 를 기록해 Q1 해석에 병기.

---

## 4. STEP 3 — exp4v2 실행 (Q1·Q2 각각, 3 seeds)

exp4v2를 **두 번** 돌린다. baseline(real-only)은 두 run 공통이므로 첫 run 결과를 두 번째에 이식(graft)해도 된다.

```python
os.environ['REAL_DATA'] = f"{os.environ['AROMA_OUT']}/exp4v2/{DS}/real_labeled"   # real labeled 데이터 루트
os.environ['Q1_OUT'] = f"{os.environ['T1_OUT']}/exp4v2_Q1"
os.environ['Q2_OUT'] = f"{os.environ['T1_OUT']}/exp4v2_Q2"
```

### Q1 run — B(aroma-fullstack) vs A(random-naive)

```python
!python $AROMA_SCRIPTS/experiments/exp4_v2_supervised_detection.py \
    --dataset_keys          $DS \
    --class_mode            single \
    --condition             all \
    --real_data_dir         $REAL_DATA \
    --random_synthetic_dir  $SYN_A \
    --aroma_synthetic_dir   $SYN_B \
    --model                 yolov8n \
    --seeds                 1 2 42 \
    --synth_ratio           1.0 \
    --output_dir            $Q1_OUT
```

### Q2 run — C(aroma-cp) vs D(random-cp)

```python
!python $AROMA_SCRIPTS/experiments/exp4_v2_supervised_detection.py \
    --dataset_keys          $DS \
    --class_mode            single \
    --condition             all \
    --real_data_dir         $REAL_DATA \
    --random_synthetic_dir  $SYN_D \
    --aroma_synthetic_dir   $SYN_C \
    --model                 yolov8n \
    --seeds                 1 2 42 \
    --synth_ratio           1.0 \
    --output_dir            $Q2_OUT
```

> **인자 확인됨**(스크립트 argparse): `--dataset_keys`, `--model`(single, not `--models`), `--condition`, `--real_data_dir`, `--random_synthetic_dir`, `--aroma_synthetic_dir`, `--synth_ratio`, `--seeds`, `--class_mode`(tiled AITeX=single), `--output_dir`. 기타 기본값: `--imgsz 256`, `--baseline_epochs 50`, `--val_frac 0.3` — 필요 시 명시.
> `--condition all` = baseline/random/aroma 3조건. Q1에서 "aroma"=B·"random"=A, Q2에서 "aroma"=C·"random"=D로 해석. **결과 라벨(aroma/random)을 arm(A~D)으로 재라벨링해 보고**한다(혼동 방지).

---

## 5. STEP 4 — 분석 (사전 등록 기준)

3-seed paired 분석. **point delta 단독 보고 금지 — seed 간 paired t와 CI를 병기.**

```python
import json, os, statistics as st
def per_seed(qout, ds, model='yolov8n'):
    rows = {}
    for s in [1,2,42]:
        p = f"{qout}/exp4v2_results_{s}.json"
        d = json.load(open(p))[ds][model]
        rows[s] = {k: d[k]['map50'] for k in ('baseline','random','aroma')}
    return rows
def paired(qout, ds, label):
    r = per_seed(qout, ds)
    A = [r[s]['aroma']-r[s]['random'] for s in (1,2,42)]
    m = st.mean(A); sd = st.stdev(A); se = sd/len(A)**0.5
    t = m/se if se else 0
    lo, hi = m-2.92*se, m+2.92*se   # t_.975(df=2)=4.303 → 95% CI: m ± 4.303*se
    lo95, hi95 = m-4.303*se, m+4.303*se
    print(f"[{label}] per-seed A-R={[round(x,4) for x in A]}  mean={m:+.4f}  t={t:+.2f}  95%CI=[{lo95:+.4f},{hi95:+.4f}]  pos={sum(x>0 for x in A)}/3")
paired(os.environ['Q1_OUT'], DS, 'Q1  B(fullstack) vs A(naive)')
paired(os.environ['Q2_OUT'], DS, 'Q2  C(aroma-cp)  vs D(random-cp)')
```

### 성공/중단 기준 (사전 등록)

| 질문 | advance | kill |
|------|---------|------|
| **Q1** | mean(B−A) > 0 **AND 95% CI 하한 > 0** | CI가 0 포함 → "프레임워크 우월성 미입증"으로 정직 보고 (near-ceiling 아님에도 null이면 진짜 null) |
| **Q2** | mean(C−D) > 0 **AND 95% CI 하한 > 0** | CI가 0 포함 → "ROI 선택 단일효과 미입증(n=3)" |

- **검정력 한계**: n=3에서 탐지 가능 효과는 대략 ≥Δ0.02. 예상 효과가 이보다 작으면 seed 수를 늘리거나(≥5) "underpowered"로 표기. **underpowered null을 '효과 없음'의 증거로 쓰지 말 것.**
- AITeX 기존 positive(Δ+0.097, n=1)는 **이 재현으로 대체**된다. n=1 수치를 결론으로 인용 금지 — 이 3-seed CI가 정본.

---

## 6. 하지 말 것 (무결성)

1. **Q1 pair와 Q2 pair 교차 비교 금지** (B vs D, C vs A 등) — 요인 혼재.
2. **arm B ar_fallback 비율 은폐 금지** — 과도하면 Q1이 copy_paste 비교로 붕괴함을 병기.
3. **point delta 단독 보고 금지** — 항상 seed 간 paired t + 95% CI.
4. **n_synth parity 미확인 상태로 결론 금지** — STEP 2.5 대조 필수. 어긋나면 재생성.
5. **cherry-pick 금지** — Q1·Q2 결과를 둘 다 보고. 한쪽만 제시 금지.
6. **`--min_quality` 등 필터를 한 arm에만 적용 금지** — 네 arm 동일 필터/seed/top_k.
7. **AR 임계를 결과가 좋아질 때까지 조정 금지** — 아티팩트 유입. 임계는 pilot 품질로만 사전 확정.

---

## 7. 요약

- **선택 1회(realism) + 합성 4 arm(A/B/C/D) + exp4v2 2 run(Q1/Q2) × 3 seeds.**
- Q1 = 프레임워크 총동원 vs 단순 증강(B vs A). Q2 = 선택 단일효과(C vs D, 엔진·blend 동일).
- 판정은 seed 간 **95% CI 하한 > 0**. AITeX n=1 positive를 3-seed로 대체하는 것이 1차 목표.
- arm B의 ar_fallback을 반드시 관측·병기(AITeX 세장형은 폴백이 높을 수 있음 → Q1 해석 제한).
