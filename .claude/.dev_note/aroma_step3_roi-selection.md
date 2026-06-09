# AROMA Step 3 — ROI Selection

## 사용할 skills: feature-dev

## 개요

Phase 0 + Step 2 출력을 읽어 모든 (결함 이미지 × 컨텍스트 빈) 후보를 스코어링하고,
샘플링 전략에 따라 합성 생성에 사용할 ROI 목록을 선별한다.

---

## ROI Score 수식 (확정)

```
ROI_score = 0.4 × P(M_i) + 0.4 × P(C_j) + 0.2 × Deficit(M_i, C_j)
```

- `P(M_i)`: cluster prior (deficit_analysis.json에서 추출)
- `P(C_j)`: P(ctx_bin | cluster) (compatibility_matrix.json에서 추출)
- `Deficit(M_i, C_j) = 1 - Observed/Expected` (deficit_analysis.json에서 추출)
- 가중치 합계: 0.4 + 0.4 + 0.2 = 1.0

---

## 입력

| 파일 | 위치 |
|------|------|
| `morphology_features.csv` | Phase 0 |
| `morphology_clusters.json` | Phase 0 |
| `compatibility_matrix.json` | Phase 0 |
| `deficit_analysis.json` | Phase 0 |
| `prompts.json` | Step 2 output |

---

## 구현 항목

1. `score_roi(morph_prior, ctx_prior, deficit) -> float`
   - 수식 적용, 입력 clamped to [0, 1]

2. `build_candidates(data) -> list`
   - morphology_features.csv의 각 행 × context_cell_key 계산
   - cluster assignment join → missing image_id 제외
   - deficit, prior, prompt lookup

3. `select_rois(candidates, strategy, top_k) -> list`
   - **deficit_aware** (기본): Top-K Deficit Quantile 오버샘플링
     - deficit >= 75th percentile → 우선 선택
     - 나머지로 채움
   - **top_k**: roi_score 내림차순 상위 K
   - **weighted**: roi_score 비례 확률 weighted random draw

4. CLI: `--profiling_dir`, `--prompts_dir`, `--sampling_strategy`, `--top_k`, `--output_dir`

---

## Rare Pair 정의

```
Deficit(M_i, C_j) = 1 - Observed(M_i, C_j) / Expected(M_i, C_j)
```
deficit_analysis.json에 이미 계산되어 저장됨.
Top-K Deficit Quantile: 75th percentile 이상을 rare pair로 간주.

---

## 출력

```
{output_dir}/
  roi_candidates.json   전체 scored candidates
  roi_selected.json     선택된 ROI 목록 (top_k개)
  roi_summary.md        마크다운 테이블
```

각 candidate schema:
```json
{
  "image_id", "image_path", "cluster_id", "cell_key",
  "roi_score", "morph_prior", "ctx_prior", "deficit",
  "prompt", "morph_label", "ctx_label"
}
```

---

## 완료 기준

- `roi_selected.json` 생성, top_k개 포함
- 3개 전략 모두 동작
- roi_score ∈ [0, 1]

---

## 구현 완료 (2026-06-09)

- `scripts/aroma/roi_selection.py` 생성
- `tests/aroma/test_roi_selection.py` — 21/21 pass
