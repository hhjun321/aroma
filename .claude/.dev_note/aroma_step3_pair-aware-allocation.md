# Step 3 ROI 선택 전략 재설계 — Pair-aware 2-Stage Allocation

> WARNING: Exp 2 re-run results are pending. rare_pair_cov AROMA > Random is expected but not yet verified with actual measurements.

---

## (사용할 skills: feature-dev)

## 개요

Exp 2 결과: AROMA가 Random보다 morph_cov/ctx_cov/rare_pair_cov 모두 낮음
(isp_LSM_1: AROMA morph=0.50, ctx=0.14 vs Random morph=1.00, ctx=0.91).

근본 원인: 기존 `_rare_pair_deficit_quantile()`이 rare 21개 선택 후
나머지 179개를 roi_score 상위로 패딩 → ctx_prior가 지배적인 cell_key가
roi_score 상위를 독점 → 특정 (cluster, cell_key) pair에 depth-first 집중.

AROMA 논문 스토리는 "deficit-aware ROI modeling → Random 대비 높은 pair coverage"인데
현재 구현은 반대 결과를 냄. 전략을 **Coverage-first + Quality-second 2-stage 구조**로 재설계.

AROMA의 Multi-objective Selection 목표:
- **Coverage**: 모든 (cluster, cell_key) pair 최소 1개 보장
- **Deficit-awareness**: under-represented context에 heavy quota 배분
- **Quality**: 남은 슬롯은 global roi_score 상위로 채움

---

## 1차 구현 이후 발견된 추가 문제

Phase 1만으로 실행한 결과 (n_sel << top_k):

| 데이터셋 | n_sel | top_k | slack |
|---------|-------|-------|-------|
| isp_LSM_1 | 126 | 200 | -74 |
| mvtec_cable | 47 | 200 | -153 |
| visa_cashew | 63 | 200 | -137 |
| visa_pcb | 88 | 200 | -112 |

원인: Hamilton이 deficit 높은 소수 pair에 quota 집중 배분 → 그 pair의
candidates 수가 quota보다 적어 slack 발생. Phase 2 backfill로 해결.

---

## 영향도 분석

### 이 기능이 변경하는 상태
- `roi_selected.json` 선택 결과 (pair 분포 변화)
- `exp2_roi_quality.py` 실행 결과 (Exp 2 지표 값 변화)

### 그 상태를 전제로 동작하는 기존 로직
- `generate_defects.py` Step 4: `roi_selected.json`을 입력으로 결함 생성
  → 선택된 이미지/cluster/cell_key 변경되므로 Step 4 재실행 필요
- Exp 2 (`exp2_roi_quality.py`): 재실행 시 새 지표 확인

---

## 수정 내용

### 1. `scripts/aroma/roi_selection.py` — `_rare_pair_deficit_quantile()` 알고리즘 교체

기존 함수를 `_pair_aware_allocation()`으로 대체. `select_rois()`의
`deficit_aware` 분기 호출부 1줄만 수정, 외부 API 변화 없음.

**신규 알고리즘 — 2-stage**:

```
Phase 1 (Coverage + Deficit):

  Step 1: (cluster_id, cell_key) pair별 PairDeficit 계산
          PairDeficit(M, C) = mean(deficit of all candidates in that pair)

  Step 2: 모든 pair에 quota=1 기본 부여 (coverage 보장)

  Step 3: 남은 quota (top_k - n_pairs)를 PairDeficit 비례 분배
          Hamilton Method (Largest Remainder)
          deficit=0인 pair: 추가 quota=0 (Step 2의 최소 1개만 유지)

  Step 4: 각 pair 내부에서 roi_score 내림차순 선택 quota개
          → Selected_A

Phase 2 (Quality backfill):

  slack = top_k - len(Selected_A)
  남은 candidates 중 global roi_score 상위 slack개 선택 → Selected_B

Final: Selected_A ∪ Selected_B  (n_sel = top_k 보장)
```

**엣지 케이스**:

| 상황 | 처리 |
|------|------|
| `top_k <= n_pairs` | roi_score 전체 상위 top_k (fallback) |
| `candidates` 빈 리스트 | 빈 리스트 반환 |
| 모든 pair deficit=0 | remaining을 pair 수로 균등 분배 |
| pair 내 candidates < quota | slack 발생 → Phase 2 backfill |
| `top_k == n_pairs` | remaining=0, Phase 2 slack=0 |
| Phase 2 backfill 후 n_sel = top_k | 항상 보장 (candidates >= top_k 전제) |

**구현된 최종 코드** (`scripts/aroma/roi_selection.py` 257-335행):

```python
def _pair_aware_allocation(
    candidates: List[Dict[str, Any]],
    top_k: int,
) -> List[Dict[str, Any]]:
    if not candidates:
        return []

    # Phase 1: pair-aware coverage + deficit allocation
    pair_groups: Dict[Any, List[Dict[str, Any]]] = defaultdict(list)
    for c in candidates:
        pair_groups[(c["cluster_id"], c["cell_key"])].append(c)

    pairs = list(pair_groups.keys())
    n_pairs = len(pairs)

    if top_k <= n_pairs:
        return sorted(candidates, key=lambda c: c["roi_score"], reverse=True)[:top_k]

    pair_deficit = {
        p: float(np.mean([c["deficit"] for c in pair_groups[p]]))
        for p in pairs
    }

    quotas: Dict[Any, int] = {p: 1 for p in pairs}
    remaining = top_k - n_pairs

    deficit_pairs = {p: d for p, d in pair_deficit.items() if d > 0}
    total_deficit = sum(deficit_pairs.values())

    if total_deficit == 0:
        weight: Dict[Any, float] = {p: 1.0 for p in pairs}
        total_w = float(n_pairs)
    else:
        weight = dict(deficit_pairs)
        total_w = total_deficit

    ideal = {p: remaining * w / total_w for p, w in weight.items()}
    floor_q = {p: math.floor(v) for p, v in ideal.items()}
    shortfall = remaining - sum(floor_q.values())
    remainders = {p: ideal[p] - floor_q[p] for p in weight}
    for p in sorted(remainders, key=remainders.__getitem__, reverse=True)[:shortfall]:
        floor_q[p] += 1

    for p, extra in floor_q.items():
        quotas[p] += extra

    selected: List[Dict[str, Any]] = []
    for p, quota in quotas.items():
        top_in_pair = sorted(pair_groups[p], key=lambda c: c["roi_score"], reverse=True)
        selected.extend(top_in_pair[:quota])

    # Phase 2: quality backfill for slack slots
    slack = top_k - len(selected)
    if slack > 0:
        selected_ids = {id(c) for c in selected}
        rest = sorted(
            (c for c in candidates if id(c) not in selected_ids),
            key=lambda c: c["roi_score"],
            reverse=True,
        )
        selected += rest[:slack]
        logger.info(
            "pair_aware_allocation: phase2 backfill %d / %d slack slots",
            len(rest[:slack]), slack,
        )

    return selected
```

**변경 import** (파일 상단):
- `import math` 추가
- `from collections import defaultdict` 추가

---

## 수정 대상 파일

- `scripts/aroma/roi_selection.py` — `_pair_aware_allocation()` 구현 (완료)
- `scripts/aroma/experiments/exp2_roi_quality.py` — 주석 갱신 (완료)

---

## Exp 2 결과 (Phase 1 only → 2-stage 이전)

| 데이터셋 | 전략 | morph | ctx | rare_pair | n_sel |
|---------|------|-------|-----|-----------|-------|
| isp_LSM_1 | AROMA (Phase 1 only) | 1.0 | 1.0 | 1.0 | 126 |
| isp_LSM_1 | RANDOM | 1.0 | 0.91 | 0.67 | 200 |

Phase 2 추가 후 n_sel=200 보장되고 커버리지 1.0 유지 예상.

---

## 테스트

CLAUDE.md: pytest 금지. Colab에서 직접 검증.

```python
import json, numpy as np, os
from collections import Counter

for ds in ["isp_LSM_1", "mvtec_cable", "visa_cashew", "visa_pcb"]:
    with open(f"{os.environ['AROMA_OUT']}/roi/{ds}/roi_selected.json") as f:
        sel = json.load(f)
    with open(f"{os.environ['AROMA_OUT']}/roi/{ds}/roi_candidates.json") as f:
        cands = json.load(f)

    n_clusters = len({c["cluster_id"] for c in cands})
    n_cells = len({c["cell_key"] for c in cands})
    print(f"\n{ds}: n_sel={len(sel)}, n_cand={len(cands)}")
    print(f"  morph_cov={len({s['cluster_id'] for s in sel})}/{n_clusters}")
    print(f"  ctx_cov={len({s['cell_key'] for s in sel})}/{n_cells}")
```

기대:
- n_sel = 200 (top_k, Phase 2 backfill 보장)
- morph_cov = 1.0, ctx_cov = 1.0 (Phase 1 coverage 보장)
- rare_pair_cov: AROMA > Random
