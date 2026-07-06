# exp4v2 20260705 정식 run 분석 — 4종 multi-class copy-paste (3-seed 1/2/42)

원본: `.claude/.etc/exp4v2/20260705/exp4v2_results.json`. class_mode=multi, synth_ratio 1.0(cap=n_real_train), imgsz 640 rect, epochs 300.
(이 run의 aitex는 **비타일 11-class(baseline 0.066)** — tiled 단일클래스 aitex(0.372)와 별개·구버전.)

---

## mtd (요청) — 천장 효과, aroma ≈ random

| 조건 | map50 | 
|------|-------|
| baseline | 0.9204 |
| random | 0.9310 |
| aroma | 0.9348 |

paired(seed 1/2/42):
- **aroma − random**: +0.0124 / +0.0057 / **−0.0068** → mean **+0.0038, t=0.67, 2/3** → **비유의, aroma≈random**
- aroma − baseline: +0.0144, **t=3.24, 3/3** → synth가 baseline은 확실히 상회
- random − baseline: +0.0107, t=2.08, 3/3 → random도 상회

per_class(aroma): break 0.951→0.974, crack 0.952→0.972, uneven 0.804→0.833 — 소폭 개선이나 전부 이미 高.

→ **mtd = leather형 확정**: baseline 0.92로 이미 높음(쉬움) → 증강은 유효하나 **random에서 포화, aroma가 추가 이득 없음**. baseline 높이가 결과를 예측한다는 프레임 재확인(mtd 0.92·leather 0.83 = null).

---

## ★ 4종 종합 (20260705 정식 run) — 정직한 전체 그림

| 데이터셋 | baseline | Δ(A−R) mean | t | aroma>random | 성격 |
|----------|----------|-------------|---|--------------|------|
| **severstal** (multi, crux) | 0.497 | +0.0002 | 0.14 | ❌ | **완전 평탄 (synth 자체 무효)** |
| **mtd** (multi) | 0.920 | +0.0038 | 0.67 | ❌ | 천장 |
| **mvtec_leather** (multi) | 0.834 | −0.0091 | −0.27 | ❌ | 천장 |
| aitex (비타일, multi 11-cls) | 0.066 | +0.0075 | 0.42 | ❌ | 측정 붕괴(구버전) |
| **aitex (tiled, single)** ※별도 run | 0.372 | **+0.097** | **4.51** | ✅ | **유일 positive** |

**중대 관찰 — 정직 보고 필수**:

1. **20260705 정식 run에서 4종 모두 aroma ≈ random** (전부 t<0.7, 방향 비일관). 즉 **copy-paste + multi-class 설정에서 AROMA 선택 우위가 나타나지 않음**.
2. **severstal(crux)이 완전 평탄**: baseline≈random≈aroma≈0.497이고 **random−baseline도 ≈0(synth 자체가 무효)**. per_class c2도 0.327→0.323로 **rare-class 회복 없음**. 메모리의 ratio 0.4 낙관(aroma>random)이 **ratio 1.0에서는 소멸** — synth=real 규모에서 오히려 신호 희석.
3. **유일한 positive는 tiled 단일클래스 aitex**(별도 run, baseline 0.372, Δ+0.097 t=4.51). 이건 측정 병리(종횡비)를 고쳐 헤드룸을 복구한 경우.

## 해석 (과대주장 금지)

- **"AROMA 선택 우위"는 강건하지 않다**: 4종 정식 run 중 3종(severstal/mtd/leather)이 null, 비타일 aitex도 null. **positive는 tiled-aitex 단 하나.**
- 패턴: **baseline이 높은(쉬운) 데이터셋**(mtd 0.92·leather 0.83)은 random 포화로 null. **낮은(어려운)** 데이터셋 중 유일하게 측정이 정상인 tiled-aitex만 positive. severstal은 낮은데도(0.497) synth 자체가 무효 → 별도 원인(ratio 1.0 희석? severstal 결함 특성?) 규명 필요.
- **thesis 재조정 불가피**: "다중도메인 ROI 선택이 downstream을 개선"은 **현재 근거로는 tiled-aitex 1건**뿐. 4종 breadth 주장을 이 데이터로 뒷받침할 수 없음. cherry-pick(tiled-aitex만 제시)은 금지.

## 후속 (우선순위)

1. **severstal 평탄 원인 규명 (최우선)**: crux가 죽으면 논문 축이 흔들림. (a) synth_ratio 0.4/0.5 재확인(메모리 낙관 구간), (b) ratio 1.0에서 synth 품질/노이즈 점검, (c) exp5 PRDC로 severstal aroma-synth가 real manifold를 실제로 커버하는지 — 커버리지도 aroma≈random이면 "선택이 downstream에 안 통함"이 기제 수준에서 확정.
2. **tiled 프로토콜을 severstal에도?** severstal(1600×256, 6.25:1)도 종횡비 병리 가능 — tiling으로 측정 복구 시 신호가 살아나는지(aitex 선례). 단 crux per-class는 유지.
3. mtd/leather는 "천장 데이터셋"으로 확정 — 적용범위 논의용(positive 기대 대상 아님).
4. exp5/exp6(커버리지)를 4종에 돌려 downstream null이 기제(커버리지)와 정합하는지 교차검증 — 정합하면 "언제 통하는가"의 조건을 정량적으로 규정.

## 결론

mtd는 천장 효과로 aroma≈random(leather형). 그러나 이 파일의 더 큰 함의는 **20260705 정식 run 전반이 null이고 severstal crux까지 평탄**이라는 점 — AROMA downstream 우위의 현재 유일 근거는 tiled-aitex 1건입니다. 과대주장 없이, severstal 평탄 원인 규명이 다음 최우선입니다.
