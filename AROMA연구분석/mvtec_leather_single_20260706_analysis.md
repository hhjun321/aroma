# MVTec-leather single-class 결과 분석 — 2026-07-06 (3-seed)

원본: `.claude/.etc/exp4v2/mvtec_leather_single/exp4v2_results.json`
(key=`mvtec_leather`, weights=`20260706_leather_single` — 검증 완료)
설정: single-class(nc=1, 5개 결함유형 병합), seeds 42/1/2, synth_ratio 1.0(cap 64 = n_real_train), imgsz 640 rect.

---

## 결과 (mAP50, mean±std, n=3)

| 조건 | mAP50 | map50_95 | precision | recall | n_train |
|------|-------|----------|-----------|--------|---------|
| baseline | 0.836 ± 0.054 | 0.492 | 0.868 | 0.798 | 64 |
| random | 0.888 ± 0.052 | 0.559 | 0.869 | 0.865 | 64+64 |
| aroma | 0.895 ± 0.078 | 0.573 | 0.862 | 0.839 | 64+64 |

### Paired seed 분석 (같은 seed끼리)

| 비교 | seed별 Δ (42/1/2) | mean | paired t(df=2) | 전 seed 일관 |
|------|-------------------|------|----------------|--------------|
| aroma − random | +0.108 / **−0.104** / +0.018 | +0.008 | +0.12 | ❌ **비일관** |
| aroma − baseline | +0.119 / +0.012 / +0.046 | +0.059 | 1.88 | ✅ (유의는 아님) |
| random − baseline | +0.010 / +0.116 / +0.028 | +0.051 | 1.57 | ✅ (유의는 아님) |

---

## 판정: leather single에서는 **aroma > random 신호 없음** (config-강건성 미성립)

1. **aroma ≈ random (구분 불가)**: mean Δ +0.008, seed1에서 오히려 aroma가 −0.104로 뒤집힘 → **방향 비일관, t=0.12**. aitex single의 강한 신호(전 seed +, t=4.51)와 **정반대**. leather single은 config-강건성 검증에서 **부정 결과**.
2. **증강 자체는 baseline보다 나음(random·aroma 모두 +0.05~0.06, 전 seed 일관)** — 단 유의는 아니고, **random과 aroma가 동급**. 즉 leather에서는 "합성을 더하면 조금 낫다"까지이고 **"AROMA의 선택이 random보다 낫다"는 성립하지 않음**.
3. **천장 효과(ceiling)가 유력 원인**: leather baseline이 이미 **0.836**로 매우 높음(질감이 단순·결함이 뚜렷). 헤드룸이 0.16뿐이라 random 합성만으로도 0.888까지 차고, ROI 선택의 추가 여지가 거의 없음. aitex(baseline 0.372, 헤드룸 큼)와 대조적 — **선택 가치는 baseline이 낮아 개선 여지가 큰 데이터셋에서 드러난다**는 해석.
4. **소표본·고분산**: n=3, aroma std 0.078(seed1 0.805가 견인). CI 겹침 심각. 어느 쪽도 강한 주장 불가.

## aitex single과의 대조 (config-강건성 질문의 답)

| | baseline | Δ(aroma−random) | 전 seed 일관 | paired t |
|--|----------|-----------------|--------------|----------|
| **aitex** (single, tiled) | 0.372 | **+0.097** | ✅ 3/3 | **4.51** |
| **leather** (single) | 0.836 | +0.008 | ❌ 1/3 역전 | 0.12 |

→ **config-강건성은 "부분적"**: 단일클래스라는 config 자체가 신호를 만드는 게 아니라, **데이터셋의 헤드룸(baseline 낮음)이 있을 때만** AROMA 선택 우위가 드러남. leather는 single이든 multi든 천장이라 신호가 안 나올 가능성 — **multi 결과와 대조 필수**(아직 미확인).

## 정직한 결론

- leather single: **aroma > random 불성립**(noise 내, seed1 역전). 증강(random·aroma)이 baseline보다 소폭 낫지만 서로 동급.
- 이는 실패가 아니라 **경계 조건 데이터**: baseline 0.836 천장 효과. AROMA의 가치는 aitex처럼 **개선 여지가 큰(어려운) 데이터셋**에서 관측되며, 쉬운 데이터셋에서는 random과 수렴한다 — thesis를 약화시키는 게 아니라 **적용 범위를 규정**(정직 보고 대상).
- cherry-pick 금지: leather를 "aroma 우위" 근거로 쓸 수 없음. 4종 통합표에서 aitex(강), severstal(crux, 별도), leather(천장·중립), mtd(?)를 **있는 그대로** 제시.

## 다음

1. **leather multi 결과 확보 후 대조** — single/multi 둘 다 aroma≈random이면 "leather=천장 효과"로 확정, single만 다르면 config 효과 분리. (multi 파일 필요)
2. leather의 낮은 헤드룸을 **exp5 PRDC/exp6 coverage로 교차 확인** — 커버리지 지표에서도 aroma≈random이면 downstream 중립과 정합(기제 일치).
3. 4종 중 leather는 "천장 데이터셋" 사례로 논문에 포지셔닝(적용 범위 논의).
