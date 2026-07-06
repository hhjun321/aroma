# MVTec-leather multi vs single 대조 분석 — 2026-07-06 (3-seed, config 강건성)

원본: `mvtec_leather_multi/exp4v2_results.json`(nc=5, 검증됨) + `mvtec_leather_single/exp4v2_results.json`(nc=1). seeds 42/1/2, synth_ratio 1.0(cap 64), imgsz 640 rect.

---

## multi (nc=5) 결과 — mAP50

| 조건 | mAP50 | precision | recall |
|------|-------|-----------|--------|
| baseline | 0.814 ± 0.059 | 0.822 | 0.750 |
| random | 0.926 ± 0.033 | 0.870 | 0.876 |
| aroma | 0.933 ± 0.047 | 0.918 | 0.894 |

### paired (같은 seed)

| 비교 | Δ(42/1/2) | mean | t(df=2) | 일관 |
|------|-----------|------|---------|------|
| aroma − random | +0.078 / **−0.072** / +0.014 | +0.007 | 0.16 | ❌ 비일관 |
| aroma − baseline | +0.218 / +0.053 / +0.087 | +0.119 | 2.38 | ✅ |
| random − baseline | +0.139 / +0.125 / +0.073 | +0.112 | **5.60** | ✅ |

---

## ★ single vs multi 대조 — 결론이 일치 (config-강건)

| config | baseline | Δ(aroma−random) mean | seed 일관 | t |
|--------|----------|----------------------|-----------|---|
| **single (nc=1)** | 0.836 | +0.008 | 1/3 (seed1 역전) | 0.12 |
| **multi (nc=5)** | 0.814 | +0.007 | 1/3 (seed1 역전) | 0.16 |

→ **leather는 single·multi 둘 다에서 aroma ≈ random** (Δ≈+0.007, 동일하게 seed1에서 역전, t≈0.1). **config에 무관하게 결과 동일** → 앞선 "천장 효과" 가설 **확정**.

## 해석

1. **aroma > random 불성립 (config 무관)**: single·multi 양쪽 모두 Δ mean +0.007, 방향 비일관(seed1 역전) — 단일클래스 병합 여부가 원인이 아님. leather 자체가 AROMA 선택 우위를 드러내지 못하는 데이터셋.
2. **증강 자체는 강하게 유효**: random−baseline이 multi에서 **t=5.60**(전 seed +0.07~0.14) — 합성 추가가 baseline을 확실히 끌어올림. 다만 **aroma가 그 위에 더 얹지 못함**(random과 동급).
3. **천장 효과 확정**: baseline 0.81~0.84로 이미 높음. random 합성만으로 0.93까지 포화 → ROI 선택의 추가 여지 소진. aitex(baseline 0.372, Δ +0.097 t=4.51)와 정반대 — **AROMA 가치는 baseline이 낮아 헤드룸이 큰 데이터셋에서 발현**.
4. **per-class 단서**: multi에서 aroma가 `fold`(가장 어려운 클래스, baseline 0.52)를 random 대비 낮춤(0.945→0.797)—희소·난해 클래스에서 오히려 불리. 반면 color/glue/poke(이미 高)에선 소폭 우위. 즉 leather에선 aroma 선택이 쉬운 클래스에 쏠려 어려운 fold를 못 살림 → 순효과 상쇄. (n=3 소표본이라 단정 말 것.)

## 4종 종합 위치 (cherry-pick 없이)

| 데이터셋 | baseline | Δ(A−R) | AROMA 우위 | 성격 |
|----------|----------|--------|-----------|------|
| **aitex** (single, tiled) | 0.372 | +0.097 (t=4.51, 3/3) | ✅ 입증 | 헤드룸 큰 난이도 → 선택 가치 발현 |
| **mvtec_leather** (single+multi) | 0.81~0.84 | +0.007 (t≈0.1) | ❌ 중립 | 천장 효과 → random과 수렴 |
| severstal | (multi, crux) | 미확인 | — | c2 rare-class 별도 |
| mtd | 미확인 | — | — | — |

## 정직한 결론

- leather는 **"AROMA가 안 통하는 경계 조건"의 정직한 사례** — thesis를 깨는 게 아니라 **적용 범위를 규정**: *baseline이 낮아(=어려워) 개선 여지가 큰 데이터셋에서 AROMA 선택이 가치를 낸다.* leather는 이미 쉬워 random으로 포화.
- **논문 포지셔닝**: leather를 "aroma 우위" 근거로 쓰지 않음. 대신 aitex(강한 positive) + leather(천장 null)를 **함께 제시**해 "언제 통하고 언제 안 통하는가"의 조건부 주장을 강화. 이게 4종 모두 positive보다 오히려 방어적으로 강함(과대주장 회피).
- **config 강건성 검증 자체는 성공**: 단일클래스 전환이 신호를 만들지도 없애지도 않음(leather에선 애초에 신호 없음) — single/multi 방법론이 일관됨을 확인.

## 다음

1. **exp5 PRDC / exp6 coverage를 leather에** — 커버리지 지표에서도 aroma≈random이면 "천장=커버리지 포화"로 기제-결과 완전 정합(강력한 교차검증).
2. mtd 결과 확보 → 4종 완성. mtd baseline이 낮으면(어려우면) aitex형 positive 기대, 높으면 leather형 null 기대 — baseline 높이로 사전 예측 가능.
3. severstal은 crux(c2)라 별도 multi 분석 유지.
