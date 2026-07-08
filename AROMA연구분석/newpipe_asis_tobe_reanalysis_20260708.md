<!-- 출처: .claude/.etc/newpipe/{roi_selected_asis,roi_selected_tobe,annotations_synth_aroma,annotations_synth_random}.json + .claude/.etc/20260704_1/exp4v2_results_{1,2,42}.json. provenance chain 재분석. 2026-07-08 -->

# newpipe as-is/to-be 재분석 — 성능비교의 진짜 입력을 열어보고 보정

## 분석 순서 (사용자 제시 provenance)
`newpipe`(입력 산출물) → `20260704_1`(성능비교 결과) → `roi_check.md`(개선 문서).
이전 분석은 결과 JSON과 committed 리포트만 봤고 **newpipe(실제 선택·합성 입력)를 보지 않아** 세 가지를 놓쳤다. 이 문서가 그것을 보정한다.

---

## 1. newpipe가 실제로 담은 것 (MTD, 4 파일)

| 파일 | 정체 | n |
|------|------|---|
| `roi_selected_asis.json` | **legacy scoring** 선택 | 200 |
| `roi_selected_tobe.json` | **realism scoring** 선택 | 200 |
| `annotations_synth_aroma.json` | AROMA arm 합성 | 400 (200×2) |
| `annotations_synth_random.json` | Random arm 합성 | 400 (200×2) |

### as-is = legacy, to-be = realism (수치 검산으로 확정)
- **ASIS** roi_score 0.107 = `0.4·morph(0.237) + 0.4·ctx(0.026) + 0.2·deficit(0.008)` → **legacy 공식**
- **TOBE** roi_score 0.229 = `0.5·ctx(0.027) + 0.3·morph(0.230) + 0.2·quality(0.733)` → **realism 공식**

즉 "as-is → to-be"는 **이미 시도된 개선 = legacy→realism scoring 전환**이다. roi_check가 제안하는 "점수를 날카롭게"의 한 형태가 **이미 적용된 상태**.

---

## 2. ★ 보정 1 — 분석된 aroma arm은 이미 to-be(realism)였다

`annotations_synth_aroma.json`의 roi_score 평균 **0.2292 = tobe(realism)**. 
→ 20260704_1의 aroma 결과(mean 0.925)는 **realism scoring을 이미 쓴 결과**이고, 그럼에도 random(0.931)에 졌다.
→ roi_check idea #1("scorer sharpen")의 실현형(realism)이 **motivating 결과에 이미 반영**돼 있었다. 추가 sharpening의 여지를 논하려면 이 사실을 전제해야 한다.

### to-be가 as-is에서 바꾼 것은 33/200(16.5%)뿐, 그리고 순수 품질 스왑
- 공유 167/200, 교체 33개.
- **버려진 33개**(legacy): quality **0.500**, morph 0.236, ctx 0.041, deficit 0.005
- **추가된 33개**(realism): quality **0.836**, morph 0.194, ctx 0.051, deficit 0.005

→ realism 전환의 실질은 **저품질(0.50) ROI 33개를 고품질(0.84) ROI로 교체**한 것뿐. morph/ctx/deficit은 사실상 불변. 즉 "sharpen"의 현재 형태는 **16.5% ROI의 quality-gate 재셔플**이라는 작은 섭동이며, downstream을 움직이지 못한 것이 놀랍지 않다.

---

## 3. ★ 보정 2 — A−R은 삼중 교란 (이전엔 이중으로만 봤음)

| | 선택 전략 | 합성 엔진 | blend mode | 소스 다양성 | 클래스 분포 |
|--|-----------|-----------|-----------|------------|-------------|
| **aroma** | realism(tobe) | controlnet 212 + **copy_paste_arfallback 188** | **seamless(Poisson)** | 200 distinct, max reuse 2 | 균등(~80/class) |
| **random** | random | copy_paste 400 | **alpha** | 143 distinct, **max reuse 10** | 가용도순(crack 38·fray 46 vs 100+) |

→ aroma vs random은 **선택·엔진·블렌드 3요소가 동시에 다름**. "aroma < random"을 ROI 선택 탓으로 돌릴 수 없다. 이전 가이드는 "엔진+선택 이중 교란"이라 했으나, 실제로는 **blend(seamless vs alpha)까지 3중**이다. 이것이 측정의 가장 큰 결함.

---

## 4. ★ 보정 3 — ar_fallback 47% 확정 (이전엔 "TBD")

aroma_synth method: `controlnet 212` + `copy_paste_arfallback 188` → **47%가 실제로는 copy_paste 폴백**.
→ "ControlNet arm"의 절반은 ControlNet이 아니다. AR 게이트(세장형 bbox)에 걸린 결함이 copy_paste로 떨어졌다.
→ roi_check §2의 **edge/elongated 가설과 정합**: crack/break/fray(모서리·세장형)가 AR 게이트로 다수 폴백됐을 개연성. 즉 MTD에서 "ControlNet 생성"은 non-elongated 결함에만 실제로 작동했다.

---

## 5. 클래스 리밸런싱은 됐으나 무효 (기존 결론 재확인)
aroma(realism+quota)는 클래스 균등(~80), random은 가용도순(crack 38 starved). aroma가 rare-class(crack/fray)를 실제로 더 생성했지만 per-class map50는 하락(20260704_1: crack −0.009, fray −0.028).
→ **near-ceiling(base 0.92)에서는 rare-class 보강이 이득으로 전환되지 않는다.** random이 easy-class에 개수를 더 써도 천장이라 손해가 없다.

---

## 6. 보정된 결론 — roi_check 개선안 착수 전 반드시 할 일

이전 가이드(`roi_check_improvement_guide_20260708.md`)의 결론(MTD는 잘못된 타깃, 집중/중복은 rigging 취약)은 유지된다. 여기에 newpipe가 더한 **가장 중요한 시정 사항**:

1. **측정 하베스를 단일변수로 고치는 것이 scoring 개선보다 선행이다.** 현재 random(alpha+copy_paste)과 aroma(seamless+controlnet/cp)는 blend·engine이 달라 어떤 선택 효과도 분리 불가.
   - random arm도 **seamless blend + 동일 엔진(또는 동일 copy_paste)**으로 맞춰 재생성해야 한다.
   - 이 confound를 제거하기 전 산출된 A−R(20260704_1 포함)은 **선택 효과의 증거로 쓸 수 없다.**
2. **realism(tobe)은 이미 aroma에 적용돼 있었고 효과는 16.5% quality 스왑에 그쳤다.** roi_check가 원하는 "더 날카로운 점수"를 시험하려면, (a) legacy vs realism이 아니라 (b) 훨씬 큰 대비의 새 scoring을 (c) **headroom 있는 데이터셋(tiled AITeX)** 에서 (d) 삼중 교란을 제거한 하베스로 측정해야 한다.
3. **ar_fallback 47%를 보고에 명시.** "ControlNet arm"을 순수 생성 arm으로 서술하면 안 됨. 세장형(crack/break/fray) 결함은 양 arm 모두 copy_paste로 통제됐음을 병기(guide ITEM 3, controlnet execute 문서의 AR 게이트 설계와 일치).

### 측정 불가/미확정 (TBD)
- asis(legacy)의 **downstream 결과 없음** — 20260704_1 aroma는 tobe만. legacy vs realism의 downstream 직접 비교 불가(선택 스왑 33개의 효과 미측정).
- newpipe 파일은 **MTD 단일**. 이전 가이드 ITEM 0에서 인용한 severstal `casda_aroma/roi_selected.json`(1690, deficit max 0.071) 수치는 **다른 데이터셋**이므로 MTD에 적용 금지 — MTD 실측은 위 §1 표(deficit max 0.051, deficit>0 73%)를 쓸 것.

---

## 7. 한 줄 요약
20260704_1의 "aroma"는 **이미 realism(tobe) 선택**이었고, random과 **선택·엔진·blend 3중으로 달라** 비교 자체가 오염됐으며, ControlNet arm의 47%는 copy_paste 폴백이었다. roi_check의 scoring 개선을 논하기 전에 **단일변수 하베스 복구가 선행**이고, realism 전환은 이미 16.5% quality 스왑으로 시도됐으나 near-ceiling MTD에서 무효였다.
