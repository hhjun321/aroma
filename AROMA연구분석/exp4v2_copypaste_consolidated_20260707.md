# exp4v2 copy-paste 방식 — 전체 결과 통합 (2026-07-07 기준)

> 본 문서의 모든 수치는 **copy-paste 합성(AROMA/random 동일 엔진, ROI 선택 전략만 차이)** downstream(YOLOv8n mAP50) 결과다. ControlNet 생성 arm(`controlnet_aroma_arm_execute.md`)은 **별도·진행중**이며 여기 포함하지 않는다.
> 핵심 질문: **aroma(선택) > random(선택)?** — 동일 합성 엔진·동일 budget에서 ROI 선택 전략의 가치.

---

## 통합 표 (3-seed, paired Δ)

| 데이터셋 | run / config | baseline | random | aroma | **Δ(A−R)** | t(df=2) | aroma>random | Δ(synth−base) |
|----------|--------------|----------|--------|-------|-----------|---------|--------------|----------------|
| **aitex (tiled, single)** | **aitex_single** (20260706) | 0.372 | 0.388 | **0.485** | **+0.097** | **4.51** | ✅ **입증** | +0.113 (t 2.64) |
| severstal (multi-4) | 20260705 | 0.498 | 0.497 | 0.497 | +0.000 | 0.14 | ❌ | **−0.001 (무효)** |
| mtd (multi-5) | 20260705 | 0.920 | 0.931 | 0.935 | +0.004 | 0.67 | ❌ | +0.014 (t 3.24) |
| mvtec_leather (multi-5) | 20260706_multi | 0.814 | 0.926 | 0.933 | +0.007 | 0.16 | ❌ | +0.065 (t 2.55) |
| mvtec_leather (single) | 20260706_single | 0.836 | 0.888 | 0.895 | +0.008 | 0.12 | ❌ | +0.059 |

> **aitex 유효 결과 = `aitex_single`(tiled 단일클래스) 단 하나.** 비타일 aitex(20260705 파일 내 multi-11, baseline 0.066)와 20260704 스모크 aitex는 **무효로 제외**(종횡비 붕괴·epochs=1). 비타일 aitex는 20260705 파일에 severstal/mtd/leather와 함께 들어있어 파일 삭제는 불가 — 본 분석에서 aitex 행으로 취급하지 않는다.

부가 확인:
- leather는 20260705(multi, 0.834)·20260706 multi(0.814)·single(0.836) **세 run 모두 aroma≈random**(config·run 무관) → 결과 견고.
- 20260704 스모크(severstal/aitex/mtd, epochs=1)는 노이즈 — 전량 폐기.

---

## 두 층위로 분리한 결론

### (1) copy-paste 증강 자체가 baseline을 올리는가? (synth vs baseline)
- **올림**: aitex(+0.113)·leather(+0.06)·mtd(+0.014, t=3.24) — 3종에서 유효.
- **무효**: **severstal(−0.001)** — synth=real(ratio 1.0) 규모에서 random·aroma 모두 baseline을 못 넘음.

### (2) AROMA 선택이 random 선택보다 나은가? (aroma vs random) — 연구 핵심
- **positive 1건**: **aitex_single(tiled)만** (Δ+0.097, t=4.51, 3/3 seed 일관).
- **null 3건**: severstal(평탄)·mtd(천장)·leather(천장, single·multi). (비타일 aitex는 무효로 제외 — 카운트 안 함.)

---

## 패턴 해석 (과대주장 없이)

1. **천장 효과가 지배적**: mtd(base 0.92)·leather(0.83)는 이미 쉬워 random 합성만으로 포화 → 선택 여지 소진. → aroma≈random. **AROMA 가치는 baseline이 낮아(어려워) 헤드룸이 큰 데이터셋에서만 발현.**
2. **측정이 정상이어야 신호가 보인다**: aitex 비타일(종횡비 16:1 붕괴, base 0.066)은 **측정 무효로 제외**. tiling으로 측정 복구한 `aitex_single`(base 0.372)에서만 **aroma>random이 드러남** → 측정 정상 + 헤드룸 큰 유일 케이스가 유일 positive.
3. **severstal(crux)의 이중 null**: baseline 0.497로 낮은데도 synth 자체가 무효(random=baseline)이고 aroma도 평탄. per_class c2 0.327→0.323(회복 없음). ratio 0.4(과거 메모리 낙관)와 달리 **ratio 1.0에서 신호 소멸** — 별도 원인(희석/결함특성/측정) 규명 필요.

## 정직한 현 위치

- **"다중도메인 copy-paste ROI 선택 우위"의 downstream 근거는 현재 tiled-aitex 1건.** 4종 breadth 주장을 copy-paste 데이터로 뒷받침할 수 없음. tiled-aitex만 제시하는 cherry-pick은 금지.
- 이는 thesis 폐기가 아니라 **적용 범위 규정**: *측정이 정상이고 baseline 헤드룸이 큰(어려운) 데이터셋에서 AROMA 선택이 downstream을 개선*. 쉬운(천장) 데이터셋은 random과 수렴.

## 후속 (우선순위)

1. **severstal 평탄 규명 (최우선, crux)**: ratio 0.4/0.5 재확인 + tiling 적용(6.25:1 종횡비 병리 가능성 — aitex 선례) 시 신호 부활 여부. per-class(c2) 유지.
2. **mtd/leather = 천장 데이터셋 확정** — 적용범위 논의용(positive 대상 아님).
3. **exp5 PRDC / exp6 coverage를 4종에** — downstream null이 커버리지(기제)와 정합하는지 교차검증. 정합 시 "언제·왜 통하는가"를 정량 규정(천장=커버리지 포화, tiled-aitex=커버리지 우위).
4. **ControlNet arm**(별도)이 copy-paste null 구간(특히 severstal)에서 다른 결과를 내는지 — 진행중.

## 참조 (개별 보고서)
- `aitex_tiled_20260706_analysis.md` (tiled-aitex positive 상세)
- `mvtec_leather_single_20260706_analysis.md` / `mvtec_leather_multi_single_compare_20260706.md` (천장)
- `exp4v2_20260705_multiclass_4ds_analysis.md` (4종 정식 run + severstal 평탄)
- `exp4v2_20260704_smoke_analysis.md` (스모크·폐기, 파이프라인 검증만)
