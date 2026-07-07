# roi_selected vs roi_candidates 분석 — mtd / mvtec_leather (2026-07-07)

원본: `D:\project\AROMA_DATASET\roi\{mtd,mvtec_leather}\roi_{candidates,selected}.json`
목적: aroma≈random(천장 null)의 **선택-수준 기제** 규명 + 개선안(6a980a6) 검증.

---

## 핵심 발견 — AROMA 차별화 신호(deficit)가 죽어 있음

| 신호 | mtd (cand 16877→sel 200) | mvtec_leather (304→200) |
|------|--------------------------|--------------------------|
| deficit>0 | 21.6% (max 0.051, mean 0.002) | **0% (전부 정확히 0)** |
| morph_prior 고유값 | 5 (=클러스터 수) | **2** |
| ctx_prior 고유값 | 33 | 5 |
| distinct 결함 수 | 388 (→200 선택) | **92 (→200, 2.2× 반복 강제)** |
| roi_score sel−cand(중앙값) | **−0.006** (diversity-cap이 점수 낮춤) | +0.114 (ctx 주도) |
| ctx_prior sel−cand | +0.001 | **+0.286** |

## 기제 — `roi_score = 0.4·morph + 0.4·ctx + 0.2·deficit`의 붕괴

- **deficit ≈ 0** (leather 완전 0, mtd 거의 0) → 0.2 항 무효.
- **morph_prior ≈ 상수**(2~5개 값) → 0.4 항 변별력 미미.
- → 실질 선택이 **ctx_prior(문맥 빈도 prior) 단독**으로 붕괴 = "전형적 문맥의 결함 선택" → random 평균 추출과 구분 불가.
- **개선안이 지적한 "빈도 편향 + image-blind"가 실측 확증**되고, 더 근본적으로 **deficit 신호 자체가 생성되지 않음**이 드러남.

## 데이터셋별 (둘 다 null, 원인 상이)

- **leather = 데이터 희소성 천장**: distinct 결함 92개뿐인데 200 선택 → 선택 자유도 원천 부족. aroma·random 둘 다 같은 92개 소진 → **선택으로 해결 불가**(구조적 한계 확증).
- **mtd = 신호 부재**: 388 distinct(여지 있음)이나 deficit 죽음 + morph 5값 → ctx 빈도만 남음 → 사실상 빈도 기반 선택.

## 개선안(6a980a6)에 대한 함의

1. **P1 "deficit-dominant"는 leather/mtd에서 무용** — deficit=0을 지배항으로 못 올림. 단 **P1의 "morph rarity 반전"은 유효**(2~5값이라도 신호).
2. **근본 선행 = deficit 생성 진단**: deficit이 죽으면 AROMA 핵심 축이 3/4에서 비작동 → **aroma≈random의 진짜 뿌리 후보**.
3. **leather는 선택으로 불가 확정**(92 천장). mtd가 그나마 여지(388) — deficit 생성 선행 조건.
4. **P2(coverage-greedy)** 가 신호 부재 상황에서 가장 현실적(형태공간 균등 커버로 ctx-빈도 대체).

## 다음
- **deficit 생성 진단**: `profiling/{ds}/deficit_analysis.json`에서 deficit이 왜 0/near-0인지(데이터 특성 vs 계산 붕괴). aitex(positive)와 대조.
