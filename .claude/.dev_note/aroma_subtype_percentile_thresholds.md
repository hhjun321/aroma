# Table 4 subtype 임계 percentile 전환 (데이터셋별 data-driven 수치)

## (사용할 skills: feature-dev)

## 개요

Table 4의 subtype 분류 임계가 **전 데이터셋 공통 고정 상수**(`AR>5.0`, `AR<2.0`, `Sol>0.9`, `Sol<0.7`)로 하드코딩돼 있다. 이는 §3.2.2의 "per-dataset, data-driven partitions that require no hand-set constants"와 §3.2의 "CASDA의 handcrafted morphological rules 비판"에 정면으로 배치된다. 사용자 확인 결과 **원 설계 의도가 percentile 방식**이었고 구현이 고정 상수로 남았다.

본 작업은 임계를 **데이터셋 자체 분위점(P33/P66)** 으로 전환하고, 산출된 수치를 데이터셋별 표로 논문에 명시한다. 규칙(절차)은 데이터셋 불변, 수치는 데이터가 결정하는 구조.

선행 분석·1단계 정리: [[aroma_table3_background_descriptor_definitions]]

---

## 1. 근거 실측 (2026-07-29)

### 1-1. 현행 고정 임계가 데이터셋마다 전혀 다른 분위에 놓인다

| 임계 | aitex | severstal | mtd | kolektor | leather | 편차 |
|---|---|---|---|---|---|---|
| AspectRatio > 5.0 | 49.7% | 52.0% | 77.3% | 40.4% | 78.3% | 37.9%p |
| AspectRatio > 2.0 | 26.7% | 19.8% | 38.9% | **1.9%** | 45.7% | 43.7%p |
| Solidity > 0.9 | 66.8% | 32.9% | 49.7% | **9.6%** | 42.4% | **57.1%p** |
| Solidity < 0.7 | 37.2% | 7.4% | 14.4% | 5.8% | 7.6% | 31.4%p |

결과 subtype 비율: linear_scratch가 kolektor 59.6% ↔ leather 21.7% (**2.7배**), compact_blob이 kolektor 1.9% ↔ leather 37.0% (**19배**).

### 1-2. percentile 전환 시 일관성 (실측 비교)

| 설계 | linear_scratch 5종 편차 | 현행 라벨 일치율 |
|---|---|---|
| FIXED (현행) | **37.9%p** | 100% |
| **P33/P66 tertile** | **0.9%p** | 50~81% |
| 평균분위 매칭(P27/P60/P15/P40) | 0.4%p | 64~83% |

**P33/P66 채택.** 평균분위 매칭이 현행에 더 가깝고 편차도 약간 작지만, 앵커 숫자(P27/P60/P15/P40)가 자의적이라 "왜 P27인가"를 별도 방어해야 한다. P33/P66은 §3.2.2가 context cell 이산화에 **이미 사용하는 앵커**이므로 "AROMA는 모든 이산화를 데이터셋 자체 tertile로 한다"는 단일 원칙에 흡수된다 — 정당화 부담 0.

### 1-3. linearity 중복 (1단계에서 이미 반영)

`linearity ≡ 1 − AR⁻²`, `eccentricity ≡ √linearity` — 동일 2차 모멘트 유래, 4,504건 전량 기계 정밀도 성립(max err 4.4e-15), Spearman(lin, AR)=1.000. 따라서 독립축은 **AR·Solidity 둘뿐**이고 subtype 4종이 자연 상한이다.

제3 독립축 후보 부재 확인 (AR과의 Spearman):
- `circularity` −0.76~−0.92 → 거의 같은 축, 사용 불가
- `extent` −0.79(leather)~+0.13(severstal) → 부호 반전, 불안정
- `solidity` −0.03(severstal)~−0.66(leather) → 유일한 실효 제2축

---

## 2. 설계

### 2-1. 규칙 (데이터셋 불변)

```
1. AspectRatio > P66(AR)                            → linear_scratch
2. AspectRatio < P33(AR) AND Solidity > P66(Sol)    → compact_blob
3. Solidity < P33(Sol)                              → irregular
4. otherwise                                        → general
```

캐스케이드 순서 유지 필수 — 규칙1 ∩ 규칙3(길쭉하면서 경계 거친 결함)이 aitex 34.7%·severstal 3.9%로 비어 있지 않다.

### 2-2. 산출 수치 (P33/P66, 실측)

| Dataset | n | AR P33 | AR P66 | Sol P33 | Sol P66 |
|---|---|---|---|---|---|
| AITeX | 352 | 3.13 | **16.43** | 0.648 | 0.900 |
| Severstal | 3620 | 2.80 | **7.80** | 0.900 | 0.965 |
| Kolektor | 52 | 4.39 | **5.73** | 0.981 | 0.984 |
| MVTec Leather | 92 | 1.58 | **4.03** | 0.874 | 0.954 |
| MTD | 388 | 1.69 | **3.62** | 0.846 | 0.934 |

`AR P66`이 3.62~16.43(**4.5배**), `Sol P33`이 0.648~0.981. 고정 임계에서 subtype 비율에 함의로만 남던 형태학 차이가 **수치로 명시**된다 — percentile 균등화로 잃는 것을 표가 회수한다.

### 2-3. ⚠ Kolektor solidity 분할 퇴화

`Sol P66 − P33 = 0.0029` (전체 sd 0.0861의 3.4%). n=52·단일 결함유형으로 solidity에 실질 구조가 없어 소수 셋째 자리에서 3분할한다. AR도 폭 1.343(sd 1.446)으로 좁다.

**은폐되지 않고 표에 P33≈P66으로 노출되는 것이 오히려 장점**이다("kolektor 결함은 형태가 균질해 solidity 분할이 무의미"를 표가 스스로 말함). 논문에 각주 1문장 필요. 고정 임계에서는 kolektor가 linear_scratch 59.6%로 뭉치는 것이 신호인지 임계 위치의 우연인지 구분 불가였다.

---

## 3. 수정 내용

### 3-1. `scripts/aroma/roi_selection.py` (구현 완료)

- **신규 상수** `_TERTILE_DEGENERATE_RATIO = 0.15`, `_FIXED_TERTILE_EQUIV` (§5 참조).
- **신규 헬퍼 `_subtype_percentiles(morph_rows)`** — `aspect_ratio`/`solidity`의 P33/P66을 morphology 행 전량에서 산출해 dict 반환. 유효값 3건 미만이면 `None`(→ 전체 fixed 폴백). 퇴화 특징은 해당 특징만 fixed 값으로 치환하고 `_fixed_fallback` 목록에 기록.
- **신규 헬퍼 `_percentile_subtype(ar, sol, th)`** — §2-1 캐스케이드.
- **`quality_proxy(...)`에 `thresholds=None` 인자 추가** — `None`이면 기존 `classify_defect_subtype`(fixed) 경로. **기본값이 None이므로 미지정 시 byte-identical.**
- **`build_candidates(...)`에 `subtype_mode="fixed"` 추가** — `percentile`일 때만 `_subtype_percentiles`를 1회 계산해 per-row 루프에 전달.
- **`run(...)`·`_parse_args()`에 `--subtype_mode {fixed,percentile}` 전파** (기본 `fixed`).
- 산출물 provenance: `roi_candidates.json` 각 항목에 `subtype_mode` 기록 + 로그에 산출 분위값 출력.

`utils/defect_characterization.py`는 **수정하지 않는다** — CASDA 계승 유틸이므로 AROMA 측 로직은 `roi_selection.py`에 둔다.

### 3-2. 논문 (반영 완료)

`AROMA연구분석/Article/text/section3_2.txt` §3.2.3:

- **Table 4** → 규칙을 percentile 형태로 기술(`AspectRatio > P66(AspectRatio)` 등). 데이터셋 불변 **절차**.
- **Table 4b 신설** → 데이터셋별 산출 수치(AR P33/P66, Sol P33/P66) + 결함 수. Table 2(데이터셋별 CCI 분해)와 동일 양식. Kolektor solidity에 `*` 폴백 각주.
  - **표 번호**: `Table 5`(§3.3 dataset splits)·`Table 6`/`7`(§4.2 detection)이 이미 사용 중이라 renumber를 피해 `4b`로 부여. 레퍼런스 임시번호와 동일 방침 — 일괄 재번호 시 함께 처리.
- Table 4 직후 산문 3단락: (a) tertile 앵커가 §3.2.2 context cell과 동일 규약임 + 고정 상수가 데이터셋마다 다른 분위에 놓이는 실측 예시(AR 5.0 = kolektor P40 vs leather P78), (b) linearity/eccentricity 중복 항등식, (c) cascade 순서 + 동질성 가드(15%) + kolektor 폴백 공개.
- Table 4b 직후 1문장: 임계가 AR 4.5배(3.62~16.43)·Sol 0.648~0.981로 갈리므로 같은 규칙이 도메인마다 다른 형태 경계를 표현함을 명시.
- §3.2.3 본문: "on aspect ratio and solidity" → "whose numeric values are derived from each dataset's own defect population (Table 4b)" 추가.
- **§3.2.2 "no hand-set constants" 주장이 회복된다** (가드 상수 0.15는 별도 공개).

### 3-3. Figure 3.2.3-1 (반영 완료)

`figure/script/[figure 3.2.3 1] morphology_distribution.py` — 고정 상수를 그리던 것을 **데이터셋별 tertile + 동질성 폴백**으로 전환(`subtype_thresholds()`가 `_subtype_percentiles` 로직 미러). 5장 재생성. note에 해당 데이터셋의 실제 임계값·폴백 여부 표기. 긴 note가 레이아웃을 6505px로 늘려 `textwrap.wrap(150)` 적용 → 종횡비 1.84:1 정상화. `.md` 스펙도 동기화.

---

## 4. 영향도 분석

### 이 변경이 바꾸는 상태

- `roi_candidates.json` / `roi_selected.json`의 `defect_subtype`, `quality_score`, `roi_score`, 선택 집합.
- 하류: `generate_defects` 합성 결과 → `annotations.json` → exp4v2 downstream.

### 그 상태를 전제로 동작하는 기존 로직

- **`quality_score` → `ROI_score`의 0.2 항 (load-bearing)**: 실행된 공식은 `score_mode=realism` = `0.5·ctx + 0.3·morph + 0.2·quality`(5종 100% 일치 확인). 이 항의 실효 변별력은 **4종 중 3종에서 최대**(가중 기여 sd: mtd 0.0419 vs ctx 0.0113). quality 항 제거 시 top-200 겹침이 mtd **0/200**. → **subtype 라벨 변경은 선택 결과를 크게 바꾼다. 결과 이동을 전제해야 한다.**
- `apply_quality_gate(min_quality)`: 현행 실행은 `--min_quality` 미전달(0.0=OFF)이라 게이트 자체는 no-op. 단 `quality_score` 값은 항상 기록·사용됨.
- `MATCHING_RULES[subtype][background_type]`: `background_type`이 CLI 기본값 `"directional"`로 5종 전부 동일(실행 가이드 어디서도 미전달). subtype만 바뀌므로 매핑은 그대로 동작.
- single/multi class 경로, `class_floor`, `img_diversity_cap`, `per_pair_cap_frac`: subtype과 직교 — 무영향.

### "없음(0개)" 상태 가능성

- 유효 morphology 값 < 3건이면 percentile 산출 불가 → `None` 반환 → fixed 폴백(경고 로그). silent 0-output 없음.
- percentile 임계가 퇴화(P33==P66)해도 분류는 성립(경계값 비교) — 다만 §2-3 경고 대상.

---

## 5. 퇴화 특징 처리 — 특징 단위 fixed 폴백 (구현 확정)

kolektor 1차 실행에서 문제가 드러났다. `solidity` 중간 tertile 폭이 **0.0029 = sd의 3.4%**로, 52건 중 **17건이 그 0.003 폭 안**에 들어가 compact_blob / irregular / general로 갈렸다. 그 결과 `quality_score=0.4`(구조적 부적합) 버킷이 **7 → 165건(23배)** 폭증 — 측정 노이즈를 ROI_score의 0.2 항으로 승격시킨 셈이다.

특징별 중간 tertile 폭 / sd:

| 데이터셋 | 폭 | sd | 비율 |
|---|---|---|---|
| aitex | 0.2518 | 0.2137 | 117.9% |
| leather | 0.0798 | 0.1146 | 69.6% |
| mtd | 0.0877 | 0.1313 | 66.8% |
| severstal | 0.0646 | 0.1112 | 58.1% |
| **kolektor** | **0.0029** | 0.0861 | **3.4%** |

kolektor만 17~35배 좁아 분리가 깨끗하다. **처리 = 해당 특징만 fixed 임계로 폴백**(사용자 결정, 선택지 2). 다른 특징은 데이터 유도 tertile 유지.

- `_TERTILE_DEGENERATE_RATIO = 0.15` — 중간 tertile 폭이 sd의 15% 미만이면 퇴화 판정. 절대 폭은 특징 간 비교 불가라 sd 대비 비율 사용.
- `_FIXED_TERTILE_EQUIV = {"aspect_ratio": (2.0, 5.0), "solidity": (0.7, 0.9)}` — `classify_defect_subtype`과 동일 상수를 (P33역할, P66역할) 형태로.
- 폴백 발생 시 WARNING + `subtype_mode` 필드에 `percentile+fixed(solidity)` 형태로 provenance 기록.

⚠ 이 가드 자체가 하드코딩 상수(0.15)다. 다만 **분류 임계가 아닌 안전장치**이며, 실측 분리폭(3.4% vs 58~118%)이 넓어 값에 민감하지 않다. 논문에 명시할 것.

첫 구현의 `p66 - p33 <= 1e-9` 판정은 kolektor를 못 잡아 무용했다 — 절대 폭 기준의 실패 사례로 기록.

---

## 6. 실측 검증 결과 (2026-07-29, selection 단계, CPU only)

로컬 미러 `D:/project/aroma_dataset` 사용. `prompts/`가 미러에 없어 기존 `roi_candidates.json`의 `prompt`·`ctx_label`을 `(cluster_id, cell_key)`별로 역복원해 입력으로 사용.

### 6-1. Regression — fixed 모드가 커밋본을 재현

| 데이터셋 | candidates | selected | 불일치 |
|---|---|---|---|
| aitex | 5,887 | 200 | **0** |
| kolektor | 511 | 200 | **0** |
| severstal | 266,624 | 1,000 | **0** |
| mtd | 16,877 | 200 | **0** |
| mvtec_leather | 968 | 200 | **0** |

신규 `subtype_mode` 필드 제외 전 필드 일치. **기본값 byte-identical 확인** + prompts.json 역복원의 충실성도 동시 입증.

실행 파라미터: `--sampling_strategy deficit_aware --score_mode realism --img_diversity_cap 1`, multi 4종은 `--class_mode multi --class_floor --per_pair_cap_frac 0.05`. `top_k`는 severstal 1000, 나머지 200 (커밋본 선택 수에서 역추정).

### 6-2. fixed → percentile 변화량

| 데이터셋 | 라벨 일치 | 선택쌍 겹침 | distinct src | roi_score | quality |
|---|---|---|---|---|---|
| aitex | 78.1% | 73.0% | 200 → 200 | 0.2590 → 0.2418 | 0.775 → 0.688 |
| severstal | 57.6% | **42.2%** | 890 → **899** | 0.2261 → 0.2059 | 0.769 → 0.668 |
| mtd | 66.2% | 49.5% | 200 → 200 | 0.2225 → **0.2290** | 0.675 → **0.707** |
| kolektor | 42.1% | 79.8% | 52 → 52 | 0.3079 → 0.2726 | 0.867 → 0.690 |
| mvtec_leather | 64.3% | 51.2% | 90 → **92** | 0.3121 → **0.3219** | 0.698 → **0.746** |

### 6-3. subtype 분포 (candidates, %)

| 데이터셋 | mode | linear_scratch | compact_blob | irregular | general |
|---|---|---|---|---|---|
| aitex | fixed | 47.8 | 19.7 | 3.2 | 29.3 |
| | percentile | 28.6 | 22.1 | 10.4 | 38.9 |
| severstal | fixed | 42.9 | 16.0 | 3.9 | 37.2 |
| | percentile | 29.4 | 12.8 | **27.1** | 30.7 |
| mtd | fixed | 20.6 | 21.9 | 7.0 | 50.5 |
| | percentile | 32.6 | 12.7 | 17.5 | 37.2 |
| kolektor | fixed | 56.9 | 0.4 | 1.0 | 41.7 |
| | percentile | 32.1 | **33.5** | 2.0 | 32.5 |
| leather | fixed | 28.9 | 25.4 | 4.3 | 41.3 |
| | percentile | **44.8** | 16.7 | 12.6 | 25.8 |

### 6-4. 판정

**긍정 신호**
- `irregular`이 fixed에서 1.0~7.0%로 사실상 사문화돼 있었으나 percentile에서 2.0~27.1%로 실질 범주가 된다. 4종 분류가 실제로 4종으로 작동.
- 소스 다양성 개선: severstal 890 → **899**, leather 90 → **92**(전량). fixed에서 한 번도 선택되지 못한 결함이 percentile에서 사용됨 — `img_diversity_cap=1` 취지에 부합.
- leather에서 rare class 공급 개선(`poke` 32 → 36, class_floor 미달 로그 18<40 → 23<40).

**주의 신호**
- **선택집합이 크게 바뀐다** — 겹침 42.2%(severstal) ~ 79.8%(kolektor). severstal은 절반 이상 교체. downstream 결과 이동은 확정적.
- **roi_score 이동 방향이 데이터셋마다 반대다.** aitex·severstal·kolektor는 하락(quality 하락), mtd·leather는 상승. 일률적 예측 불가 → downstream을 미리 낙관/비관할 수 없다.
- kolektor는 **이중 비대표**: (a) solidity 퇴화 → 폴백 적용, (b) `n=52 < top_k=200`이라 `img_diversity_cap` 하에서 52개 소스 전부가 어차피 선택돼 소스 선택 변화를 원리적으로 관측 불가. 겹침 79.8%는 (image, cell_key) 짝짓기·반복 횟수 변화만 반영. **판정 근거로 쓰지 말 것.**

**결론**: selection 단계 관점에서 percentile 전환은 방법론적으로 정당하고(임계가 데이터 유도) 부수 효과도 긍정적(범주 실질화·소스 다양성)이다. 단 downstream 결과가 움직이므로 §7 절차를 따른다.

---

## 7. 남은 검증 절차

### 7-1. 2차 — §4.1 지표 재산출 (GPU 0)

`clean_bg_selection.py`를 percentile 산출물로 재실행 → ROI 커버리지 / 배경 호환(`[figure 4.1 3]`) 재산출. 배경 호환 지표는 `clean_bg_selected.json`을 요구하므로 이 단계가 선행돼야 한다.

### 7-2. 3차 — downstream (조건부)

합성 + YOLO 재실행. §6-4 실측으로 결과 이동이 확정적이므로 2차에서 개선이 확인된 뒤에만 착수. arm 간 공정성(`synth_ratio` cap) 유지 필수.

테스트 코드 작성·pytest 금지(CLAUDE.md). 검증은 실제 스크립트 실행 산출물 비교로 수행.

---

## 8. ⚠ 미해소 정합성 문제 (제출 전 필수)

**§3.2.3(Table 4/4b)은 percentile을 기술하지만 §4의 결과는 fixed 임계로 산출된 것이다.** 현재 원고는 방법과 결과가 불일치한다. 해소 경로 택1:

1. **downstream 재실행** — §7-2. percentile 산출물로 합성·YOLO를 다시 돌려 §4를 갱신. §6-4 실측상 결과 이동 확정적(선택쌍 겹침 42~80%).
2. **fixed로 되돌리고 percentile은 §5 한계/향후과제로 서술** — 결과는 그대로 두고 Table 4를 fixed로 복원. 단 §3.2.2 "no hand-set constants" 모순이 남는다.

1번이 정공법이며 사용자가 selection 재실행까지 진행한 방향과 일치한다. **2차(§7-1) → 3차(§7-2) 완료 전에는 원고를 제출본으로 확정하지 말 것.**

같은 성질의 기존 문제(§9 첫 항목, ROI_score 공식)와 함께 해소해야 한다.

## 9. Colab 가이드 반영 (완료)

Colab에서 재수행하기로 결정(사용자). **step3_5 단독 재실행은 불가** — percentile 변경이 `roi_selection.py`(step3)에 있고 clean_bg는 `roi_selected.json`을 소비만 한다. step3_5_execute.md의 STEP 2-2 assert(`src_match_frac >= 0.999`)가 구 roi × 신 profiling 혼용을 이미 차단한다.

**재수행 범위 = step3 → step3.5 → step5 → exp4v2(GPU).** 앞단은 불요 — `defect_subtype` 소비처 전수 확인 결과 prepare_datasets·phase0·step1·step2(`prompt_generation` grep 0건)는 무관.

수정 파일:

- **`_SPEC.md`** — §0 정본 selection 명령에 `--subtype_mode percentile` + 기본값 `fixed`가 구 산출물과 byte-identical이라 반드시 명시해야 한다는 경고. §3 step3 셀에도 플래그 추가 + 로그 확인 문구.
- **`step3_execute.md`** — STEP 2 공통 인자 줄 + 2개 분기(multi 3종 / single aitex·kolektor) 셀 전부에 플래그 추가. 플래그 누락 시 "조용히 fixed로 돌아 재실행 의미가 사라진다"는 ⚠ 경고. 로그 확인 포인트에 `subtype_mode=percentile — thresholds in use:` 줄 부재 = 누락 신호, kolektor만 `Fixed fallback: solidity` 정상 명시.
- **`step3_execute.md` STEP 2-1 신설** — percentile 임계를 논문 Table 4b와 자동 대조하는 셀. `TABLE4B` 기준값 + 동질성 가드 재현(`DEGEN_RATIO=0.15`, `FIXED_EQUIV`) + `assert` 하드 게이트(±0.02). DRIFT 시 임계가 아니라 **profiling이 논문 기준과 다른 것**임을 안내(`31ee0aa` image_id 고유키 / `b1bb497` image_w·h 확인).

대조 셀은 로컬에서 실행 검증했다 — 5종 전부 PASS, kolektor만 `[fixed fallback: solidity]`:

```
severstal      AR   2.799/  7.795  Sol 0.9004/0.9650  PASS
mvtec_leather  AR   1.585/  4.027  Sol 0.8741/0.9538  PASS
mtd            AR   1.695/  3.624  Sol 0.8460/0.9337  PASS
aitex          AR   3.132/ 16.433  Sol 0.6482/0.9000  PASS
kolektor       AR   4.391/  5.734  Sol 0.7000/0.9000  PASS  [fixed fallback: solidity]
```

### step3.5 실행 시 추가 확인 (percentile 전환 고유)

- STEP 2-2 `src_fit_ceiling_mean`이 E1 기준(aitex 0.895 / mtd 0.502 / severstal 0.623, ±0.05) 내인지. **DRIFT면 percentile 전환이 배경 매칭 신호를 훼손했다는 뜻**이므로 반드시 조사.
- `clean_bg_summary.md`의 `w_src`/`w_class`/`w_size` 데이터 유도 가중이 fixed 때와 달라질 수 있다 — subtype이 `w_class`(클래스-조건부 hist) 산출에 들어가므로 정상 변화. 값 변화를 기록해 둘 것.

## 10. TODO / 후속
- [ ] **§3.2.4 ROI_score 공식 정정 (선행 필수·별건)** — 논문은 `0.6·ctx + 0.4·morph`, 실제는 realism 3항. 현재 보고된 공식으로 결과 재현 불가.
- [ ] `background_type`이 5종 전부 CLI 기본값 `"directional"` 고정 — 데이터 유도 아님. §3.2.3 배경 분류기(`BackgroundAnalyzer`)는 현행 파이프라인에서 실행되지 않고 `roi_metadata` 산출물도 없음. 별건 검토.
- [ ] §3.2.2 GMM이 6특징 중 elongation 축을 3중 가중(linearity·aspect_ratio·eccentricity) — `morph_prior`에 영향. 별건.
