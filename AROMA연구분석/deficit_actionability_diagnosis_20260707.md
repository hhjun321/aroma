# deficit 생성/작동 진단 — deficit은 "채울 수 없는 조합"에만 높다 (2026-07-07)

원본: `AROMA_DATASET/profiling/{profiling_mtd, profilng_leather}/deficit_analysis.json` + `roi/{ds}/roi_candidates.json`.
목적: roi_candidates의 deficit≈0(leather 완전 0)이 버그인지 데이터 특성인지 규명.

---

## 결론: 코드 버그 아님 — 구조적 개념 한계

- 로딩 정상: roi_selection.py:311-313이 `info["deficit"]` 중첩 올바르게 추출, :350이 `[cid]["deficit"][cell_key]` 조회. mtd cell_key **89.5% 매칭**(190 중 170).
- deficit 계산 정상: deficit_analysis에 풍부한 nonzero (leather max **0.208**, mtd max 0.060).
- **문제 = deficit이 후보가 점유하지 않은 조합에만 높음:**

| | deficit_analysis nonzero / max | 후보 점유 조합 중 deficit>0 / max |
|--|-------------------------------|-----------------------------------|
| leather | 134 / **0.208** | **0 / 0.000** |
| mtd | 430 / 0.060 | 40 / 0.051 |

leather 최고 deficit 빈(0.208·0.151·0.082…) **전부 "채울 결함 없음"**. mtd top 빈 대부분 동일.

## 기제 — deficit ⟂ 소스 가용성 (반비례)

```
deficit(조합) 高 ⟺ real 데이터에 희소 ⟺ copy할 crop 없음 ⟺ copy-paste 생성 불가
deficit(조합) 低 ⟺ 잘 대표됨    ⟺ crop 많음   ⟺ 타겟 가치 없음
```

AROMA가 채우려는 부족 조합엔 복제 소스가 없고, 소스 있는 조합은 부족하지 않음. → **deficit_aware 선택이 copy-paste에서 구조적으로 무용** → roi_score가 ctx 빈도 단독으로 붕괴 → random과 동일.

**이것이 세션 전체 aroma≈random의 최종 뿌리다** (measurement/ceiling은 상위 증상, 이건 신호-수준 근본 원인).

## 데이터셋별
- **leather**: 2 morph 클러스터 + distinct 결함 92 + actionable deficit 0 → **어떤 copy-paste 선택도 random 불가 확정**.
- **mtd**: actionable deficit 40개(max 0.051, 미미) → 여지 극소.

## ★ 함의 1 — copy-paste 선택 개선의 한계
- **P1(deficit-dominant) 완전 사망 확정** — unactionable을 가중치로 못 살림.
- copy-paste에서 남는 actionable 신호 = **morph-rarity 반전 / P2 coverage-greedy**(가진 결함의 형태공간 커버) — random 대비 소폭 우위 가능성뿐. leather는 그마저 2클러스터라 room 없음.

## ★ 함의 2 — ControlNet arm 재정립 (중요)
- copy-paste는 가진 결함만 붙여 **부족 조합 원천 불가**. → deficit-aware 선택은 copy-paste에서 무용.
- ControlNet **생성**은 소스 crop 없이 부족 조합 결함 생성 가능. → **deficit-aware 선택의 가치는 생성과 결합해야 실현**.
- 논문 서사 후보: "copy-paste에서 aroma≈random인 이유 = 부족 조합에 소스 부재 → 생성(ControlNet)이 그 gap을 메워야 선택이 가치를 낸다." ControlNet arm이 **원리적 정당성** 획득.
- ⚠️ 단 ControlNet이 유용한 novel-combo 결함을 실제 생성하는지 미검증(aitex 세장형 AR 난점 등). 가설로만.

## 검증 제안
1. **aitex(유일 positive)의 actionable deficit 확인** — aitex_single이 positive였던 게 actionable deficit 덕인지, 아니면 diversity/coverage 덕인지. (roi/aitex 미업로드 — complexity/aitex는 있음. aitex roi 필요.)
2. ControlNet arm이 부족 조합을 실제로 채우는지 — 생성된 합성의 (morph×context) 분포가 deficit 높은 빈을 커버하는지 측정.
