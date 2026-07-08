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

## 3. ★ 보정 2 — 이것은 "교란"이 아니라 의도된 **풀 프레임워크 vs 단순 baseline** 비교다

> **설계 의도(저자 확인):** AROMA arm은 프레임워크가 동원할 수 있는 **모든 기법을 총동원**(realism 선택 + ControlNet 합성 + seamless 블렌드)한 것이고, random arm은 **단순 copy_paste(+alpha)** 단순 증강이다. 연구 질문은 "ROI 선택 단일 효과"가 아니라 **"AROMA 프레임워크 전체가 단순 증강보다 우월한가"**이다. 따라서 세 요소가 함께 다른 것은 결함이 아니라 **의도된 대비(full-stack vs naive)**다.

| | 선택 전략 | 합성 엔진 | blend mode | 소스 다양성 | 클래스 분포 |
|--|-----------|-----------|-----------|------------|-------------|
| **AROMA (full stack)** | realism(tobe) | controlnet 212 + **copy_paste_arfallback 188** | **seamless(Poisson)** | 200 distinct, max reuse 2 | 균등(~80/class) |
| **random (naive)** | random | copy_paste 400 | **alpha** | 143 distinct, **max reuse 10** | 가용도순(crack 38·fray 46 vs 100+) |

### 이 프레이밍에서의 정직한 결과 (MTD)
- 이 "총동원 AROMA"(0.925)조차 **단순 random copy_paste(0.931)에 졌다**(A−R −0.006, t=−1.56, 0/3). 즉 **프레임워크 우월성이 MTD에서는 나타나지 않았다** — 오히려 근소 열세. 단 MTD는 near-ceiling(base 0.92)이라 정보량이 낮다.
- **주의 — 두 연구 질문을 섞지 말 것:** 이 full-stack 비교는 논문 Table 13이 표방하는 *"오직 ROI 선택 전략만 다르다(single-factor swap)"* 비교와 **다른 질문**이다. 이 run을 논문에 넣는다면 **"프레임워크 전체 vs 단순 증강"** 결과로만 서술해야 하고, "ROI 선택의 기여"로 서술하면 안 된다(그건 Table 13의 별도 통제 run이 답한다). 두 프레이밍의 혼동이 유일한 리스크.

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

이전 가이드(`roi_check_improvement_guide_20260708.md`)의 결론(MTD는 잘못된 타깃, 집중/중복은 rigging 취약)은 유지된다. 설계 의도(full-stack vs naive)를 반영한 **정정된 권고**:

1. **두 연구 질문을 분리해 각각 정직하게 보고한다.**
   - **Q1 (프레임워크 우월성, 이 run의 의도):** "총동원 AROMA vs 단순 copy_paste". 이 질문에는 세 요소 동시 차이가 **정당**하다. 단 결과를 "ROI 선택 기여"로 오독하면 안 됨.
   - **Q2 (ROI 선택 단일 효과, 논문 Table 13):** 이때만 single-factor(선택만 다름)가 필요. Q1 run을 Q2 증거로 쓰지 말 것.
2. **Q1이 진짜 관심이면, headroom 있는 데이터셋에서 반복해야 결론이 산다.** MTD(near-ceiling)에서 full-stack이 naive에 근소 열세인 것은 "프레임워크가 나쁘다"가 아니라 **"천장이라 어떤 증강도 우열이 안 드러난다"**로 읽어야 한다. tiled AITeX(유일 headroom+positive)에서 full-stack vs naive를 ≥3 seed로 재현하는 것이 프레임워크 우월성 주장의 핵심 증거.
3. **realism(tobe)은 이미 full-stack에 포함**돼 있었고, legacy→realism 전환의 실질은 16.5% quality 스왑이었다. roi_check가 원하는 "더 날카로운 점수"의 추가 이득을 보려면 headroom 데이터셋에서 별도 측정이 필요.
4. **ar_fallback 47%를 보고에 명시.** MTD full-stack AROMA의 절반은 실제로 copy_paste(세장형 AR 폴백)였음 — "ControlNet 총동원"이라 해도 non-elongated에만 생성이 적용됐다. 프레임워크 우월성 주장 시 이 한계를 병기.

### 측정 불가/미확정 (TBD)
- asis(legacy)의 **downstream 결과 없음** — 20260704_1 aroma는 tobe(full-stack)만. legacy vs realism의 downstream 직접 비교 불가.
- newpipe 파일은 **MTD 단일**. 이전 가이드 ITEM 0에서 인용한 severstal `casda_aroma/roi_selected.json`(1690, deficit max 0.071) 수치는 **다른 데이터셋**이므로 MTD에 적용 금지 — MTD 실측은 §1 표(deficit max 0.051, deficit>0 73%)를 쓸 것.

---

## 7. 한 줄 요약
20260704_1은 **의도된 "총동원 AROMA(realism+ControlNet+seamless) vs 단순 copy_paste" 프레임워크 비교**다. 이 프레이밍에서 MTD 결과는 full-stack AROMA(0.925)가 naive random(0.931)에 **근소 열세**였고(near-ceiling이라 정보량 낮음), ControlNet arm의 47%는 copy_paste 폴백이었다. 프레임워크 우월성을 입증하려면 **headroom 있는 tiled AITeX에서 full-stack vs naive를 ≥3 seed로 재현**해야 하며, 이 run을 논문 Table 13(ROI 선택 단일 효과)의 증거로 전용하지 말아야 한다.
