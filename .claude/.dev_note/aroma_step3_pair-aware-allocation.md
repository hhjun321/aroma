# Step 3 ROI 선택 전략 재설계 — Pair-aware Allocation

---

## (사용할 skills: feature-dev)

## 개요

Exp 2 결과: AROMA가 Random보다 morph_cov/ctx_cov/rare_pair_cov 모두 낮음
(isp_LSM_1: AROMA morph=0.50, ctx=0.14 vs Random morph=1.00, ctx=0.91).

근본 원인: 기존 `_rare_pair_deficit_quantile()`이 rare 21개 선택 후
나머지 179개를 roi_score 상위로 패딩 → ctx_prior가 지배적인 cell_key가
roi_score 상위를 독점 → 특정 (cluster, cell_key) pair에 depth-first 집중.

AROMA 논문 스토리는 "deficit-aware ROI modeling → Random 대비 높은 pair coverage"인데
현재 구현은 반대 결과를 냄. 전략을 Pair-aware Allocation으로 재설계한다.

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

기존 함수를 `_pair_aware_allocation()`으로 대체. 함수명은 변경하되
`select_rois()`에서 `deficit_aware` 전략 진입 시 호출하는 구조 유지.

**신규 알고리즘 (4단계)**:

```
Step 1: (cluster_id, cell_key) pair별 PairDeficit 계산
        PairDeficit(M, C) = mean(deficit of all candidates in that pair)

Step 2: 모든 pair에 quota=1 기본 부여 (coverage 보장)
        base_total = n_pairs

Step 3: 남은 quota (top_k - n_pairs)를 PairDeficit 비례 분배
        Hamilton Method (Largest Remainder) 사용 → 합이 정확히 remaining
        deficit=0인 pair: 추가 quota=0 (Step 2의 최소 1개만 유지)

Step 4: 각 pair 내부에서 roi_score 내림차순 선택 quota개
```

**Hamilton Method (Largest Remainder)**:

```python
total_deficit = sum(pair_deficits[p] for p in deficit_pairs)
# 이상적 quota (실수)
ideal = {p: remaining * d / total_deficit for p, d in deficit_pairs.items()}
# 정수 내림차순 부여 후 나머지를 소수점 기준 큰 순서로 1씩 추가
floor_quotas = {p: int(q) for p, q in ideal.items()}
remainders = {p: ideal[p] - floor_quotas[p] for p in deficit_pairs}
shortfall = remaining - sum(floor_quotas.values())
# shortfall개만큼 remainder 큰 pair 순서로 +1
for p in sorted(remainders, key=remainders.get, reverse=True)[:shortfall]:
    floor_quotas[p] += 1
```

**엣지 케이스 처리**:

| 상황 | 처리 |
|------|------|
| `top_k < n_pairs` | roi_score 전체 상위 top_k (fallback) |
| `candidates` 빈 리스트 | 빈 리스트 반환 |
| 모든 pair deficit=0 | remaining을 pair 수로 균등 분배 (Hamilton, weight=1) |
| pair 내 candidates < quota | 가용한 만큼만 선택 (slack 허용, 보충 없음) |
| `top_k == n_pairs` | Step 3 skip (remaining=0), 각 pair에서 roi_score 1위만 선택 |

**코드 스케치**:

```python
def _pair_aware_allocation(
    candidates: List[Dict[str, Any]],
    top_k: int,
) -> List[Dict[str, Any]]:
    if not candidates:
        return []

    from collections import defaultdict

    # Step 1: group by pair, compute PairDeficit = mean
    pair_groups: Dict[tuple, List] = defaultdict(list)
    for c in candidates:
        pair_groups[(c["cluster_id"], c["cell_key"])].append(c)

    pairs = list(pair_groups.keys())
    n_pairs = len(pairs)

    # Fallback: top_k < n_pairs
    if top_k <= n_pairs:
        return sorted(candidates, key=lambda c: c["roi_score"], reverse=True)[:top_k]

    pair_deficit = {
        p: float(np.mean([c["deficit"] for c in pair_groups[p]]))
        for p in pairs
    }

    # Step 2: base quota=1 per pair
    quotas = {p: 1 for p in pairs}
    remaining = top_k - n_pairs

    # Step 3: Hamilton allocation of remaining
    deficit_pairs = {p: d for p, d in pair_deficit.items() if d > 0}
    total_deficit = sum(deficit_pairs.values())

    if total_deficit == 0:
        # All deficit=0 → uniform remaining allocation
        weight = {p: 1.0 for p in pairs}
        total_w = float(len(pairs))
    else:
        weight = deficit_pairs
        total_w = total_deficit

    ideal = {p: remaining * w / total_w for p, w in weight.items()}
    floor_q = {p: int(v) for p, v in ideal.items()}
    shortfall = remaining - sum(floor_q.values())
    remainders = {p: ideal[p] - floor_q[p] for p in weight}
    for p in sorted(remainders, key=remainders.__getitem__, reverse=True)[:shortfall]:
        floor_q[p] += 1

    for p, extra in floor_q.items():
        quotas[p] += extra

    # Step 4: select roi_score top-quota within each pair
    selected = []
    for p, quota in quotas.items():
        sorted_cands = sorted(pair_groups[p], key=lambda c: c["roi_score"], reverse=True)
        selected.extend(sorted_cands[:quota])

    return selected
```

`select_rois()`의 `deficit_aware` 분기에서 `_rare_pair_deficit_quantile` 대신
`_pair_aware_allocation` 호출로 교체.

---

## 수정 대상 파일

- `scripts/aroma/roi_selection.py` — `_rare_pair_deficit_quantile()` → `_pair_aware_allocation()` 교체

---

## 테스트

CLAUDE.md: pytest 금지. Colab에서 직접 검증.

1. Step 3 재실행 (isp_LSM_1, deficit_aware 전략)

```python
import json, numpy as np, os
from collections import Counter

with open(f"{os.environ['AROMA_OUT']}/roi/isp_LSM_1/roi_selected.json") as f:
    sel = json.load(f)

print("cluster dist:", Counter(s["cluster_id"] for s in sel))
print("cell_key dist:", Counter(s["cell_key"] for s in sel))
print("n_pairs covered:", len({(s["cluster_id"], s["cell_key"]) for s in sel}))
print("morph_cov:", len({s["cluster_id"] for s in sel}), "/ 4 clusters")
```

기대:
- morph_cov = 1.0 (4/4 clusters)
- ctx_cov = 1.0 (7/7 cell_keys)
- deficit>0 pair → 더 많은 quota, deficit=0 pair → quota=1만

2. Exp 2 재실행 → AROMA > Random on rare_pair_cov 확인

3. slack 발생 케이스 확인: pair 내 candidates < quota이면 n_selected < top_k 가능
   → 실제 발생 시 Step 3 후 `len(selected) < top_k` 로그 확인
