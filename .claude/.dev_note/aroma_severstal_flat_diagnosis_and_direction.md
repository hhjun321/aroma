# AROMA Severstal 평탄 결과(flat mAP) 진단 및 연구 방향 — 전략 노트

---

## (성격: 연구 전략·진단 노트 — 코드 패치 노트 아님)

이 문서는 구현 패치 노트가 아니다. AROMA의 **검출 mAP가 조건 간 평탄(baseline≈random≈aroma)** 한 현상을 진단하고, 향후 작업이 참조할 **유일하게 방어 가능한 연구 방향**과 **무결성 가드**를 기록한다. 동결된 TBD 원고 테이블은 이 문서로 채우지 않는다.

> **검증 원칙**: 본 문서의 모든 수치는 추적 가능한 근거(세션 로그 / on-disk `.claude/.etc/` 파일 / strategy-workflow 출력 `wjtd7igue.output`)에서 나온 것만 사용한다. 진행 중·미확정 항목은 그렇게 명시한다. 결과를 창작하지 않는다.

---

## 1. 배경 — 이번 세션 인프라 작업 (커밋 완료)

평탄 결과 진단에 앞서, 동일-규모 공정 비교를 위한 인프라 정비가 이번 세션에 커밋되었다.

| 커밋 | 내용 |
|------|------|
| `b439811` | casda_roi_adapter manifest staging 병렬화 (성능 개선) |
| `72fc839` | generate_defects synthetic Drive push 병렬화 |
| `4bcb16e` | `--condition` nargs 다중 선택 지원 (CLI) |
| `472292f` | 스케일업 파라미터: `top_k` 200→1690, CASDA `per_class_cap` None→525, `synth_ratio` 1.0 primary |

**재합성 결과 (matched scale)**: aroma=5070, random=5070, casda=5076 이미지. 세 합성 arm 모두 학습 cap(`real_train`≈2534)을 초과(`synth_ratio=1.0` 기준) → **세 arm의 TOTAL 예산이 동일**해졌다.

---

## 2. 문제 — aggregate 검출 mAP가 평탄하다

### 검증된 평탄 결과

- **3-seed Severstal YOLOv8n multi (구 600-pool)**: baseline=0.3821 > aroma=0.3595 > random=0.3379.
  - baseline이 두 arm을 모두 이김. aroma는 random보다 marginal하게만 위.
  - per-seed delta(aroma−random): +0.057 / +0.010 / −0.002 → **신뢰구간이 0을 포함**.
- **신규 5070-pool multi-seed 실행**: 보고상 여전히 평탄. **(진행 중 / 작성 시점 미확정 — 그대로 명시)**.
- **MVTec multi-seed (선행)**: AROMA>Random이 반증됨(평탄/역전).
- **exp4 one-class AD**: 증강이 **악화**시킴 — baseline 0.765 >> 증강 0.50.

### 평탄성을 둘러싼 두 가지 거짓 전제 (교정 — on-disk 데이터로 검증)

평탄성의 원인으로 의심되던 두 가설은 **모두 거짓**으로 확인되었다. 향후 이 두 길로 되돌아가지 않는다.

#### (a) "synth-ratio cap이 AROMA 균형을 무작위로 깨서 평탄하다" → 거짓

- `exp4_v2_supervised_detection.py` ~L1906-1908, ~L1953-1954의 `rng_sub.sample`(비층화 random subsample) cap.
- **ratio_1.0 + 구 600 pool**: cap = int(2534×1.0) = 2534지만 합성은 arm당 600개뿐 → L1959 "no trim" 분기. **cap이 발동조차 안 함(inert)**. AROMA는 균형 풀(검증: `annotations_1.json`에서 150/150/150/150 = 정확히 클래스당 25%)을 그대로 학습에 투입, random은 자연빈도 600을 투입 → **두 학습셋의 클래스 구성은 실제로 다르다**. 따라서 평탄성은 cap이 균형을 깬 결과가 아니다.
- **신규 5070 pool**: cap이 발동하긴 함. 그러나 이미 균형 잡힌 AROMA 풀을 random-subsample해도 기대값상 클래스 균형이 보존됨 → **AROMA 균형을 파괴하지 않는다.**
- **결론**: 비층화 cap은 클래스 내 점수 순위를 잃는 **잠재적 correctness 버그**로 남아 있어(층화 cap fix 가치 있음), **평탄성의 원인은 아니다.** 그리고 headline ratio(1.0)에서는 fix해도 inert다.

#### (b) "macro-AP로 바꾸면 AROMA 이득이 드러난다" → 거짓 + p-hack (CLOSED dead-end)

- **YOLO map50은 이미 클래스 비가중 평균(unweighted class mean)이다.** 검증: 0.4126 = mean(0.384, 0.280, 0.525, 0.461). 즉 micro == macro.
- aggregate가 이미 모든 클래스를 동등 가중하고 rare 클래스(c2/c4)를 up-weight하고 있다. "벗어날 frequency-weighted dilution"이 존재하지 않는다.
- "macro-AP로 전환"은 **사실관계상 틀렸고 동시에 p-hack**이다. → **막다른 길(CLOSED). 경로 아님.** aggregate를 있는 그대로 보고한다.

> **단일-seed ratio_1.0 결과를 증거로 인용하지 않는다.** (이는 위 §2 3-seed 구 600-pool 실험과는 **별개의 run**이다 — `summary_ratio_1.0` 단일 seed.) n=1, std=0이고, 사실상 한 클래스(c1 +0.164)가 견인. c2는 flat-negative(−0.002), c4 +0.013, c3 +0.053. 멀티-seed 재실행은 평탄. 모든 주장은 **≥5-seed CI**에만 근거한다.

---

## 3. 근본 원인 — 포화(saturation) + 다양성 한계 + 이미 동등가중

세 메커니즘이 복합되어 평탄성을 **예상된 결과**로 만든다.

1. **real_train≈2534가 풍부(abundant)**: ratio_1.0에서조차 synth:real ≈ 0.24:1. copy-paste 문헌상 real이 충분하면 한계 synth 가치 → 0 (포화). 충분한 real 위에 저정보 반복을 얹는 셈.
2. **rare 클래스 c2의 하드 다양성 천장**: c2 = 117 ROIs / 95 distinct source images / 2 subtypes / 3 backgrounds (c1·c4는 수백 장 규모). c2용 합성은 사실상 ~95 crop의 반복 → **어떤 선택 전략도 c2 다양성을 만들어낼 수 없다.**
3. **aggregate가 이미 클래스 동등가중** (2.b): 숨겨진 macro 이득이 없다.

**Net**: 포화 + 동등가중 regime에서는 어떤 선택 신호든 움직일 headroom이 거의 없다.

---

## 4. 작동하는 메커니즘 (검증됨 — 단, downstream win이 아님)

AROMA 선택은 **합성 클래스 분포를 실제로 바꾸고**, headroom 있는 곳(c1/c4)에서 **더 많은 distinct source crop을 커버**한다. CASDA보다 scarce 클래스에 더 많이 배분한다.

- AROMA c2 = 292 ROIs (morphology candidate pool 기준) vs CASDA c2 = 117 (`roi_metadata` 기준).
- `roi_selection` at `top_k=1690`: selected=1690, distinct images=1426, max repetition 4. c2=292 / c4=359 (uniform floor 422 미만 → availability clamp). c2·c4가 floor를 못 채운 잔여 슬롯은 headroom 있는 c1·c3가 surplus redistribution으로 흡수(global backfill)하여 총 1690을 맞춘다.
- **cross-class deficit 분기는 inert**: `roi_selection.py` ~L678-693 — `top_k`가 K의 배수이면 cross-class split이 순수 uniform이고 deficit-mass 분기가 동작하지 않는다. **현재 배송된 AROMA = uniform-floor + within-class deficit**이며, "cross-class deficit-aware" 주장은 **default config에서 문자 그대로 실행되지 않는다.**

> 이 결과는 **검증된 중간 메커니즘(intermediate mechanism)** 이다. **downstream win으로 절대 주장하지 않는다.** 헤드라인으로 단독 제시하면 전형적 post-hoc rescue가 된다.

---

## 5. 권장 방향 — Data-Efficiency Curve (유일하게 방어 가능한 길)

**real_train을 subsample**하여 synth에 headroom을 만들고, AROMA−random 격차가 real 희소 구간에서 벌어지는지를 본다.

- real fraction = {0.05, 0.1, 0.25, 0.5, 1.0}, 조건 = baseline/random/AROMA, **≥5 seeds**.
- 플롯: **AROMA−RANDOM** map50 delta vs n_real (AROMA−baseline 아님).
- **가장 저렴한 정보 슬라이스 먼저**: {0.1, 1.0} × 3 조건 × 5 seeds ≈ **30 runs**.
- 이 길이 방어 가능한 이유:
  - 가장 강한 구조적 사실(synth는 real이 희소할 때 가장 도움)을 직접 검정.
  - 1.0의 null이 thesis를 **반박이 아니라 지지**하는 증거가 됨.
  - **기존 aggregate 지표 그대로 사용** → metric-shopping 비판에 면역.
- **반드시 AROMA−RANDOM을 격리**한다. random도 희소 구간 이득을 잡으면 결론은 "synth가 돕는다"이지 "선택이 돕는다"가 아니다.

---

## 6. 의사결정 기준 / Kill 기준 (사전 등록)

**Decision (AROMA 기여 인정)**: 0.1/1.0 슬라이스에서 AROMA−random map50의 95% CI(또는 ≥5-seed paired Wilcoxon)가 **희소 끝(0.1/0.05)에서 0을 배제**하고, 그 격차가 1.0을 향해 **단조 감소**하면. 승리는 반드시 AROMA-over-random(선택)이어야 하며 synth-over-baseline이 아니다. → **결과를 보기 전에 사전 등록.**

**Kill (정직한 negative 수용)**: ≥5-seed에서 **가장 희소한 fraction(0.05/0.1)에서조차 AROMA−random이 flat-zero(CI가 0 포함)** 이면. 가장 유리한 regime에서도 random과 분리되지 않으면 어디서도 분리되지 않는다.

**Kill 시 산출물 — boundary-conditions 논문**:
- MVTec single-class null (multi-class allocator가 inert) +
- Severstal abundant-data null +
- exp4 AD-poisoning negative +
- deterministic coverage/allocation 결과를 **검증된 중간 메커니즘**으로 (명시적으로 downstream win 아님).

---

## 7. 무결성 가드 (load-bearing — 반드시 준수)

- **검증된 수치만 사용.** 결과 창작 금지. 모든 수치는 추적 가능해야 하고, 진행 중·미확정은 그렇게 표기.
- **동결된 TBD 원고 테이블은 계속 동결.** 이 노트로 채우지 않는다.
- **macro-AP win 주장 금지.** YOLO map50은 이미 클래스 비가중 평균(0.4126 = per-class 평균). p-hack이자 사실관계상 오류 → CLOSED dead-end로 기록(§2.b).
- **단일-seed ratio_1.0 결과를 증거로 인용 금지.** n=1, std=0, c1(+0.164) 단일 클래스 견인, c2 flat-negative(−0.002).
- **AROMA−RANDOM을 격리** (AROMA−baseline 아님).
- **coverage/allocation을 단독 헤드라인으로 제시 금지** — 결정적·bankable하나 중간 메커니즘이다.
- **stratified-cap fix는 correctness 버그 fix로 프레이밍**하고 결과와 무관하게 보고. ratio_1.0에서는 inert이므로 헤드라인을 바꾸지 않음. fix 후 AROMA에 유리한 run만 골라 보고 금지.
- **데이터셋/지표를 승리할 때까지 갈아타지 않는다.**

---

## 8. 즉시 다음 단계 — 무료 진단 (GPU 불필요, 사용자 선택)

detector와 분리된, 즉시 bankable한 진단 두 가지를 먼저 수행한다.

1. **train-label 클래스 히스토그램 (aroma vs random)**: 최종 YOLO train label(`syn_*.txt`) / `annotations_1.json`의 class id를 히스토그램. AROMA가 균형(~150/150/150/150), random이 c3-heavy임을 확인. → 두 학습셋이 이미 다름을 증명, 평탄 멀티-seed 결과가 harness artifact가 아님을 마감.
2. **intrinsic diversity audit**: 클래스별 distinct source crop 수 + reuse multiplicity. c2 ≤95 distinct = 반복, c1/c4 수백 distinct를 정량화. → "AROMA는 headroom 있는 곳에서 effective coverage를 늘리고 없는 곳(c2)에서는 못 늘린다"는 결정적 결과. c2 회귀를 변명이 아닌 **반증 가능한 scope condition**으로 프레이밍.

두 진단 모두 detector-decoupled하며 downstream mAP가 평탄해도 살아남는다.

---

## 9. 추후 ablation (deferred)

- **inverse-frequency / fixed per-class quota**: isolate-selection 프레이밍. c2 reuse 천장 351(3×)~585(5×), >10× 시 precision 붕괴. 추진 시 `--n_per_roi_map` ~15-30 라인 코드 필요.
- **aroma-xdeficit**: symmetric floor를 줄여 cross-class deficit 분기가 발동하도록(`roi_selection.py` ~L734-751). **사전 등록**: xdeficit == uniform-floor가 rare-class AP에서 동일하면 "cross-class deficit-aware" 주장은 falsified → drop. 어느 결과든 publishable.
- **stratified-cap correctness fix**: `exp4_v2:1906-08`, `1953-54`의 비층화 `rng_sub.sample`을 클래스 층화 subsample로 교체. 결과 무관 보고, headline에서 inert. CLAUDE.md 정책: 구현 + Colab용 설명, 로컬 실행 금지.

---

## 참고 출처

- strategy-workflow 전체 출력: `wjtd7igue.output` (상세 추론 + per-class 단일-seed 수치)
- on-disk 검증 데이터: `.claude/.etc/annotations_1.json`, `.claude/.etc/exp4v2/summary_ratio_1.0/exp4v2_results.json`, `roi_metadata.csv`
- 메모리 노트: `project_paper_state_gaps`, `project_severstal_c2_regression`
