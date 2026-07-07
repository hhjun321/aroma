# aitex positive 재검증 — deficit·coverage 둘 다 반증 (2026-07-08)

원본: `AROMA_DATASET/{profiling/profiling_aitex, roi/aitex}`. 목적: aitex(유일 positive)의 aroma>random이 actionable deficit 덕인지 coverage 덕인지.

> ⚠️ **이 보고서는 직전 `deficit_actionability_diagnosis_20260707.md`(16c569d)의 결론을 정정한다.** deficit-unactionable이 근본이라던 결론은 부정확 — deficit은 aitex positive를 만든 적이 없다.

---

## 선택된 ROI의 deficit — 3종 비교

| 데이터셋 | baseline | 선택 ROI deficit>0 | downstream |
|----------|----------|---------------------|-----------|
| **aitex** | 0.372 (헤드룸) | **0/200 (0%)** | ✅ positive |
| mtd | 0.920 (천장) | 50/200 (25%) | ❌ null |
| leather | 0.834 (천장) | 0/200 (0%) | ❌ null |

→ **deficit ↔ 승패 무관**: aitex는 deficit 0으로 이겼고, mtd는 deficit 최다로 졌다. **deficit 가설 반증.**

## aitex 선택이 실제로 최적화한 것

| 신호 | cand med → sel med (Δ) |
|------|------------------------|
| ctx_prior | 0.029 → **0.456 (+0.427)** ← 지배 |
| quality_score | 0.70 → **1.00 (+0.30)** |
| roi_score | 0.113 → 0.315 (+0.202) |
| morph_prior | 0.257 → 0.331 (+0.074) |
| cluster | 5개 → **전부 cluster 1** (집중) |
| distinct src | 311 → **103/200** (2× 반복, 저다양성) |

→ **coverage/diversity 가설도 반증**: aitex는 단일 클러스터 집중·저다양성으로 이겼다. P2 coverage-greedy는 오히려 역효과 우려.

## 실제 driver (데이터 근거)

aitex aroma>random = **ctx_prior(문맥 호환) + quality_score(사실적 매칭) 집중** + **헤드룸**. "전형적·잘 맞는·사실적 결함을 골라 깨끗이 blend" → random이 포함하는 부적합 결함 대비 우위. 단 baseline 헤드룸이 있어야 발현(천장 mtd/leather는 무효).

## 함의 (정직)

1. **AROMA 표방 메커니즘(deficit-aware 희소조합 타겟팅)이 유일 positive를 만들지 않았다.** 승리는 compatibility+realism 선택에서 나옴 — 논문 핵심 주장과 어긋남.
2. **취약성**: n=1(aitex), tiling 특수성 가능. "AROMA 작동 증거"로 쓰면 "deficit 아니라 compatibility + n=1"로 반박당함.
3. **ControlNet 재정립(16c569d) 근거 약화**: "생성이 deficit 채운다"는 서사는 deficit이 애초에 결과와 무관하므로 재검토 필요.
4. **개선안 갱신**: P1(deficit) 사망 재확인. **P2(coverage-greedy) 반증**. 실제 correlated 신호 = ctx_prior + quality_score → P3(suitability soft) 계열에 가장 가까움(단 이는 compatibility 선택이지 AROMA deficit 서사 아님).

## 남은 검증
- aitex 승리가 **compatibility 선택 때문인지 tiling 때문인지** 분리 — random arm에 동일 ctx/quality 상위 필터를 적용(대칭)해도 aroma가 이기는지(선택 전략 vs 후보 풀 효과 분리). 확정 전 "AROMA가 통한다" 주장 보류.
- 근본 재고: AROMA의 downstream 가치 주장은 현 증거로 **deficit 메커니즘 미입증 + n=1 compatibility 효과**뿐. 논문 프레이밍 전면 재검토 필요.
