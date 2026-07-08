<!-- 출처: roi_check.md(20260704_1) + MTD ControlNet arm(exp4v2_results_{1,2,42}) 분석. workflow roi-check-improvement-guide(wf_d4079e5d-2fc, 8 agents) 산출. 2026-07-08 -->

# roi_check 개선안 구현 가이드

## 0. 정직한 프레이밍 (읽고 시작할 것)

### roi_check.md가 제안하는 것
`roi_check.md`(`D:/project/aroma/.claude/.etc/20260704_1/roi_check.md`)는 MTD에서 aroma≈random인 원인을 두 시나리오로 제시한다.
- **시나리오 A (Scorer is Blind / flat scoring)**: ROI 점수 분포가 평탄해 개별 소스 ROI를 변별하지 못하고, 결국 선택이 사실상 random과 같아진다. → "점수를 날카롭게(polarize) 만들자 / 고득점 좌표를 반복 사용하자."
- **시나리오 B (Dataset Invariance / edge-vs-surface)**: 결함을 어디에 놓든 saturated 검출기의 feature를 흔들지 못한다. crack/break/fray는 edge-bound, blowhole/spot은 surface-anywhere라는 배치-기하 가설.

### 비판에서 나온 두 개의 하드 제약 (이 가이드 전체를 지배함)

**제약 1 — MTD는 잘못된 검증 대상이다 (ceiling effect).**
MTD baseline map50 ≈ 0.92 (`D:/project/aroma/AROMA연구분석/exp4v2_copypaste_consolidated_20260707.md:40`). 소스 가용성 제약을 제거한 ControlNet arm(`D:/project/aroma/.claude/.etc/20260704_1/exp4v2_results_{1,2,42}.json`, yolov8n, 3 seeds)조차 A−R = −0.006 (t=−1.56, 0/3), 클래스 synth 개수를 리밸런싱(crack 22→59, fray 31→44)했음에도 per-class map50는 오히려 하락(crack −0.009, fray −0.028). **더 강한 생성 엔진이 아무것도 못 움직였으므로, 그보다 약한 개입인 점수 재가중(ITEMS 1–3)은 near-ceiling MTD에서 2차적(second-order)이다.** 실제 headroom이 있는 데이터셋에서 검증해야 한다.

**제약 2 — 집중/중복은 metric-rigging에 취약하다.**
소수 고득점 좌표로 배치를 몰아 metric을 부풀리는 것은 메커니즘 없는 지표 조작(INTEGRITY 규칙의 SUSPECT 패턴)이다. **동일 필터를 Random arm에도 대칭 적용한 대조군(random-concentrate) 없이는 어떤 AROMA 우위도 보고 금지.** vanilla `random`(distinct row, without replacement)은 concentrate arm의 대칭 대조가 아니다.

### 코드에서 확인된, MEMORY와 상충하는 사실 (ITEM 0의 가치를 높임)
읽을 수 있는 유일한 후보-인접 아티팩트 `D:/project/aroma/.claude/.etc/exp4v2/casda_aroma/roi_selected.json`(1690 recs) 실측:
- `morph_prior` = **4** distinct (MEMORY의 5 아님)
- `ctx_prior` = **108** distinct (MEMORY의 33 아님)
- `deficit` = **142 distinct, min 0.0 / max 0.071 — dead 아님** (범위는 ctx의 ~1/2.4로 좁으나 변별함)
- `roi_score` = 232 distinct / 1690, 최빈 bin = 85/1690 (5%, monoculture 아님)

⚠️ 즉 ITEM 0이 "확인해줄" 예정이던 사전 가정(morph=5, ctx=33, deficit dead)은 **이미 읽을 수 있는 파일에서 반증됨.** 이 때문에 ITEM 0(측정)을 가장 먼저 실행하는 것이 더욱 정당하다. 단, `roi_selected.json`은 **선택 후(post-selection)** 분포이며 후보 풀(`roi_candidates.json`은 디스크에 존재하지 않음)의 cand→sel shift·정확한 tie 구조는 새 dump 없이는 재현 불가.

---

## Phase 0 — 진단 (반드시 먼저, 동작 변경 없음)

### ITEM 0 — 점수 분포 진단 리포트  *(roi_check idea #1a)*

**목적:** 어떤 항(term)이 실제로 변별력을 갖는지 측정해, 이후 모든 동작 변경(ITEM 1–3)의 정당성 자체를 판정한다. 사전 가정이 이미 반증되었으므로(위 참조) 측정이 선행 조건.

**구체 변경(파일·함수·인자):**
- 신규 함수 `profile_score_distribution(candidates_or_selected, out_path)` — `roi_selection.py`의 `_diversity_stats`(~line 1495) 인근에 배치.
- 출력(JSON, per-dataset · per-class): 각 term(`morph_prior`/`ctx_prior`/`deficit`/`quality_score`/`roi_score`)의 **distinct-value 수, Shannon 엔트로피, Gini, 최빈 bin 비율(tie-mass)**, 그리고 (후보 풀이 있을 때) cand→sel 중앙값 shift.
- **두 갈래로 분리(측정 비용이 다름):**
  - **ITEM 0-a (즉시, 비용 0):** `roi_selected.json`만 읽는 오프라인 스크립트. `np.random` import 금지. GPU 불필요, <5초. 위 실측치(morph=4, ctx=108, deficit 142/nonzero, roi_score=232, 최빈 85/1690) 재현이 수용 기준.
  - **ITEM 0-b (CPU 1-pass, 학습 아님):** `select_rois()`(line 1364)의 `build_candidates()` 직후·strategy dispatch 직전에 `score_report_path: str|None = None`(기본 no-op) 게이트로 호출. 별도로 `--dump_candidates` 경로를 추가해 **pre-selection** 후보 풀 JSON을 덤프해야 cand→sel shift·정확한 tie(image-blind 여부, `roi_selection.py:484–490`) 확인 가능.

**기존 코드 충돌:** 없음 (순수 observer). ⚠️ 단 매핑이 "no new runs"로 과장했던 부분 정정 — ITEM 0-b는 CPU 1-pass 덤프가 필요(학습 실행은 아님).

**측정·수용 기준:** ITEM 0-a는 위 6개 수치 재현. ITEM 0-b는 cand count ≥ selected count, cand→sel 중앙값 shift 산출 가능. **대칭 대조 불필요**(읽기 전용).

**anti-rigging 가드:** 선택 로직 무변경. `score_report_path=None`·덤프 플래그 미전달 시 기존 동작 byte-identical. **RNG 무위험**(rng 생성 미접촉).

---

## Phase 1 — 결함 배치 근본 원인 진단 (binding limiter 대상)

### ITEM 4 — per-class 배치-기하 진단 + 2×2 ablation  *(roi_check idea #2, Scenario A vs B 판정)*

> Phase 0 다음, **동작 변경 ITEM 1–3보다 먼저** 실행. ControlNet null이 지목하는 binding limiter(ceiling/geometry)를 직접 겨냥하는 유일한 항목이기 때문.

**목적:** roi_check §2의 edge-vs-surface 가설을 검증. `D:/project/aroma/AROMA연구분석/placement_aware_score_redesign_20260708.md:78–92`의 2×2 placement×selection ablation을 실행하고, 클래스별로 ROI가 tile edge vs surface 중 어디에 착지하는지 프로파일링.

**구체 변경(파일·함수·인자):**
- `_place_on`(`generate_defects.py:845`)은 **placement-blind** — paste 시점에 배경 타입을 계산하지 않음(`placement_redesign:11,145`). edge-vs-surface 검증에는 placement 시점의 배경-타입 인식을 **신규로** 추가해야 함(단순 tweak 아님).
- 진단 전용 프로파일러: 클래스별 착지 위치(edge/surface) 집계.

**기존 코드 충돌:** 구조적 충돌 없음. 단 ctx_prior는 선택 스칼라에만 들어가고 픽셀 배치에는 무관(`placement_redesign:11`)하므로, edge-vs-surface 구분은 현재 파이프라인이 **실제로 행사하지 않음** — 그래서 새 메커니즘이 필요.

**측정·수용 기준:**
- 4 arms × 3 seeds = 12 runs. A−R은 **각 placement cell 내부에서** 계산해 ControlNet arm을 흐렸던 engine-confound 회피.
- per-seed paired mAP (아래 매트릭스). ⚠️ **검정력 게이트:** n=3에서 near-ceiling map50의 탐지 가능 효과는 대략 ≥Δ0.02. 예상 placement 효과가 이보다 작으면 실행 전 중단 — **underpowered null을 증거로 보고 금지.**

**anti-rigging 가드:** cell 내부 A−R로 confound 차단. 실행 전 최소 탐지 가능 Δ를 명시. null이 나오면 "효과 없음"이 아니라 "검정력 부족"으로 표기.

---

## Phase 2 — Opt-in 동작 변경 (Phase 0/1 결과에 게이트됨)

### ITEM 1 — per-class Top-K / per-class cutoff  *(roi_check idea #3.1)*

**목적:** 현재 존재하지 않는 **per-class 점수/품질 cutoff** 추가 (기존 per-class *quota floor*와 구별).

**구체 변경(파일·함수·인자):**
- `apply_quality_gate()`(`roi_selection.py:401`): `min_quality: float | dict[str,float]`로 시그니처 변경. dict면 `class_key`(후보에 이미 존재, line 383)로 조회. **scalar 경로는 byte-identical 유지**(기본 0.0 = OFF).
- `_stratified_pair_aware()`(line 870): 자체 docstring(896–911)이 cross-class 분할이 starvation-first/uniform(`top_k//K`)임을 경고. `class_floor` and K>1일 때 uniform remainder를 PairDeficit-가중 분할로 대체.

**기존 코드 충돌:** 낮음. img_cap과 무충돌. **단일-클래스 데이터셋(aitex, leather, MTD-crack-only 경로)은 그대로 fall-through** — 그들의 null을 설명·해결하지 못함. genuinely multi-class MTD에만 영향.

**측정·수용 기준:**
- scalar path: `min_quality`를 float로 주면 현재 선택 **정확히 재현**(dict path만 신규 동작).
- 대칭 대조: 사용한 per-class cutoff/top-k를 **Random arm에도 byte-identical 적용**. 한 arm에만 필터 적용해 생긴 우위는 artifact.
- ⚠️ 후보 count를 바꿔 `top_k` clamp(line 1414)·`rng.choice` draw 차원을 이동시키므로 A−R이 confounded됨(ControlNet arm과 동일 confound class). A-vs-A 재현성만 보장.

**anti-rigging 가드:** ITEM 0-b가 **per-class discriminative term의 존재를 보이기 전에는 실행 금지** — ctx가 유일 신호이고 class-agnostic이면 per-class cutoff는 굶기기(starve)만 함. 필터를 양 arm에 대칭 적용.

---

### ITEM 2 — concentrate_mode (좌표 독점 + 중복)  *(roi_check idea #3.2 — img_cap과 직접 충돌, SUSPECT)*

**목적:** `img_diversity_cap`을 완화해 소수 최고득점 소스 crop을 N회 재사용하는 **opt-in** 모드.

**구체 변경(파일·함수·인자):**
- `select_rois()` 시그니처(line 1372, `img_diversity_cap=1`) + `_pair_aware_allocation`의 `_src_ok`/`_src_take`(lines 613–621). 플래그 `concentrate_mode: bool = False` 추가. True일 때 effective cap 상향.
- ⚠️ **대칭 대조는 반드시 별도 strategy 문자열 `random_concentrate`로 구현하고 자체 `np.random.default_rng(seed)`를 가질 것. vanilla `random` 분기(line 1419–1422)는 byte-identical 유지.** `random` 분기를 조건부로 편집하면 vanilla baseline의 draw stream이 바뀌어 **기존 모든 3-seed baseline이 조용히 무효화됨** — 이 프로젝트 최고 우선순위의 parity hazard.

**기존 코드 충돌 (명시적 해소):** shipped Fix4 monoculture 제어(`img_diversity_cap=1`, `roi_selection.py:497–621`)와 **정면 충돌**. Fix4는 "같은 고득점 좌표 oversample"을 기본적으로 불가능하게 만들려고 설계됨. **기본값 절대 변경 금지.** 오직 `concentrate_mode=False` 기본 opt-in으로만 구현.

**측정·수용 기준 (대칭 대조 필수):**
- AROMA-concentrate는 **오직 random-concentrate와만** 비교(동일 cap 완화, 동일 N, 동일 top-k/quality 필터, uniform 좌표 pick). vanilla random과의 비교 결과는 **보고 금지.**
- mAP delta 옆에 **distinct-coordinate 수 · placement 엔트로피를 병기** — 다양성을 붕괴시켜 산 "우위"가 보이도록.

**anti-rigging 가드 (두 게이트 모두 통과해야 실행):**
- **Gate 1:** ITEM 0-b가 **non-tied, per-image top score**를 보여야 함. 현재 image-blind(bin 내 정확히 tie)이므로 "고득점 좌표"를 고를 근거 자체가 없음 — 통과 전 차단.
- **Gate 2:** ITEM 4가 ceiling이 limiter가 아님을 보여야 함. MTD에선 ControlNet null이 이미 기대효과 ≈0을 예측.
- 중복은 near-ceiling에서 다양성↔중복을 맞바꾸는 **downside-only** 거래(gain headroom 없음).

---

### ITEM 3 — 중복 시 다양성 주입 (per-rep seed/prompt)  *(roi_check idea #3.3 / #4)*

**목적:** 재사용된 좌표에 per-rep 생성 다양성 부여.

**구체 변경(파일·함수·인자):** `generate_defects.py` fan-out `for rep_idx in range(n_per_roi)`(line 2341).
- **Seed 변형: 이미 구현됨** (controlnet) — `_cn_content_seed(..., _cn_rep_idx(output_path))`(line 1678)가 rep_idx를 해시(line 1432)해 N개의 서로 다른 latent 생성. **신규 작업 없음.**
- **Prompt 변형: 미구현, 의도적으로 억제됨** — prompt는 training-parity를 위해 재구성되고(lines 1749–1758) `roi_entry["prompt"]`는 고의로 미사용. per-rep prompt 변형이 **유일한 신규 작업**이며 touch-point는 prompt 조립 블록(1752–1758). trade-off: `build_train_jsonl`과의 train-distribution parity 붕괴.
- **copy_paste: 해당 없음** — 생성 단계 없음. per-rep 다양성은 shared `rng`(line 2375)의 background/position뿐. idea #3.3은 controlnet 전용.

**기존 코드 충돌:** img_cap과 orthogonal. ⚠️ per-rep seed 다양성은 **이미 선택된** ROI를 N장으로 늘릴 뿐 — 상류의 과다-재사용을 해결하지 못하고, 어느 좌표가 뽑히는지도 바꾸지 못함. ITEM 0/1의 선택-blindness를 고치는 것으로 **팔아선 안 됨.**

**측정·수용 기준:** ⚠️ 3-seed 하베스에서 prompt-변형의 한계효과는 이미 존재하는 seed-driven latent 다양성(`_cn_content_seed`)과 얽힘. seed-frozen A/B가 필요하나 현 하베스는 미지원 — **명세대로는 사실상 측정 불가.**

**anti-rigging 가드:** **ITEM 2 다음에만 유의미**(cap=1이면 좌표당 1회라 중복 rep 없음). 선택-blindness 해결로 오도 금지.

---

## 테스트 대상·성공 기준 (headroom 있는 데이터셋 명시)

### headroom 랭킹 (flat-scoring 가설의 검증 가능성 순)

| Dataset | Baseline / 상태 | Headroom | Multi-class | flat-scoring fix 검증 가능? |
|---|---|---|---|---|
| **MTD** | ~0.92, near-ceiling; ControlNet A−R −0.006 (0/3) | ~0.08 | Yes (crack/fray) | **No** — saturated; per-class 리밸런싱 이미 실패 |
| **tiled AITeX** | 유일한 positive: Δ+0.097, **n=1** | 큼 (낮은 base) | 사실상 tile당 단일 결함타입 | **Maybe** — 단 n=1은 신호 아닌 잡음 |
| **Leather** | copy-paste aroma≈random | TBD | Multi-class | **Plausibly** — baseline pull 필요 |
| **AITeX single-class / 기타** | aroma≈random | TBD | Single | No (단일-클래스 → ITEM 1/2 fall-through) |

출처: MTD baseline/ControlNet `exp4v2_copypaste_consolidated_20260707.md:40` + `exp4v2_results_{1,2,42}.json`; AITeX positive Δ+0.097(n=1) `D:/project/aroma/AROMA연구분석/aitex_positive_reverification_20260708.md`; leather/severstal headroom **TBD**(commit된 profiling 없음, `placement_aware_score_redesign_20260708.md:132`).

**정직한 결론: (a) 입증된 headroom과 (b) non-null AROMA 효과를 모두 가진 유일한 데이터셋은 tiled AITeX뿐 — 단 n=1.** 그러므로 정답 타깃은 **MTD가 아니라 tiled AITeX**이고, 첫 수는 다섯 개 레버를 쌓기 전에 **n=1 → n≥3 복제**다.

### 사전 등록 성공 기준 (대칭)
- tiled AITeX를 **≥3 seeds**로 복제. point delta가 아닌 seed 간 **paired t로 mean A−R** 보고.
- **Gate:** AROMA-copy-paste vs random-copy-paste, mean A−R > 0 이면서 **lower CI bound > 0** (≥3 seeds). 통과해야만 sharpening할 신호가 존재.
- 어떤 concentrate_mode(ITEM 2) 결과도 **random-concentrate와만** 비교, vanilla random 금지.

### 테스트 매트릭스 (순서대로, 각 단계가 다음을 게이트)

| # | Dataset | Arms | Seeds | 목적 | Kill/advance |
|---|---|---|---|---|---|
| **T0** | MTD + AITeX | ITEM 0 프로파일러 (진단만) | n/a | term별 distinct/entropy/tie-mass per class; ControlNet run의 blank_rate/ar_fallback 복구 | ctx만 live·morph/deficit dead면 → ITEM 1 moot |
| **T1** | **tiled AITeX** | AROMA-cp vs Random-cp | **≥3** | 유일 positive 복제 (n=1→n≥3) | A−R CI가 0 포함 → 신호는 잡음, **중단** |
| **T2** | leather | AROMA-cp vs Random-cp | ≥3 | 두 번째 headroom 데이터셋의 AROMA 효과 유무 | AITeX 넘어선 일반성 확인/부정 |
| **T3** | tiled AITeX (T1 positive일 때만) | AROMA-sharpened vs AROMA-baseline vs Random | ≥3 | live term(T0 기준) sharpening이 vanilla AROMA 대비 추가되는가 | roi_check thesis의 실제 검정 |
| **T4** | tiled AITeX (T1 positive일 때만) | AROMA-concentrate vs **Random-concentrate** | ≥3 | ITEM 2, 대칭 대조 필수 | vanilla-random 대비 우위는 inadmissible |
| **T5** | MTD | 2×2 placement×selection (ITEM 4) | ≥3 | Scenario A(scorer) vs B(ceiling/geometry) 판정 | B 확인 예상; 그러면 MTD scorer 작업 공식 종료 |

**메트릭:** 모든 arm에서 **per-seed paired map50**, seed 간 paired t 및 CI. point delta 단독 보고 금지.

### 현재 측정 불가 (명시 TBD)
1. per-class 배치 기하(roi_check §2 핵심) — commit된 아티팩트에 없음; 2×2 harness 미실행.
2. MTD ControlNet arm의 blank_rate / ar_fallback — `exp4v2_results_{1,2,42}.json`에 없음(`_log_cn_stats`가 stderr로만 출력, `generate_defects.py:1825`). **TBD.**
3. AITeX positive의 대칭 대조 확인 — `aitex_positive_reverification_20260708.md:44`가 요구하는 Random arm 동일 필터 미실행. 현재 A−R은 selection-strategy-vs-candidate-pool confounded.
4. leather/severstal placement coverage — profiling 미commit(`placement_aware_score_redesign_20260708.md:132`).

---

## 하지 말 것 (rigging 패턴 금지 목록)

1. **`img_diversity_cap=1` 기본값 변경 금지.** shipped Fix4 monoculture 제어(`roi_selection.py:497–621`). concentrate는 opt-in `concentrate_mode=False`로만.
2. **vanilla `random` 분기(`roi_selection.py:1419–1422`) 편집 금지.** concentrate 대조는 별도 strategy `random_concentrate`+자체 `default_rng(seed)`로. 편집 시 기존 3-seed baseline 전부 조용히 무효화.
3. **concentrate 결과를 vanilla random과 비교 금지.** 오직 대칭 random-concentrate와만. 대칭 대조 없는 AROMA 우위는 보고 금지.
4. **필터를 한 arm에만 적용 금지.** per-class cutoff·top-k·quality gate는 AROMA와 Random에 byte-identical 적용.
5. **소수 고득점 좌표로 배치 집중 후 metric 상승을 성과로 보고 금지** — 메커니즘 없는 SUSPECT 조작. distinct-coordinate 수·placement 엔트로피를 mAP 옆에 항상 병기.
6. **ITEM 0(측정) 전에 ITEM 1/2 실행 금지.** 특히 non-tied per-image top score 미입증 상태에서 concentrate 실행 금지 — 고를 근거 자체가 없음(deterministic jitter head 증폭에 불과).
7. **near-ceiling MTD에서 scorer/selection 레버(ITEM 1–3)의 결과를 thesis 증거로 삼기 금지** — ControlNet null이 이미 2차성 입증.
8. **underpowered null을 "효과 없음" 증거로 보고 금지** (ITEM 4, n=3 < ~Δ0.02 탐지 한계). 검정력 부족으로 명시.
9. **per-rep seed/prompt 다양성(ITEM 3)을 선택-blindness의 해결로 오도 금지** — 상류 과다-재사용·좌표 선택을 바꾸지 못함.
10. **커밋된 파일 경로 없는 정량 주장 금지.** 없으면 TBD로 표기. 수행하지 않은 실험을 수행한 것처럼 창작 금지.
