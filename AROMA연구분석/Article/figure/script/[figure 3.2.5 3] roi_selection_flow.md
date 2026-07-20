# Figure 6 — ROI Selection & Compatibility-Aware Placement Flow (spec)

## 목적
§3.2.5의 ROI 선택→배치 흐름을 하나의 흐름도로 시각화한다. 정책선택·deficit·quality(Table 5)가 제거된 **compat-스파인(quality-free) 실제 실행 흐름**을 반영한다. 실데이터(severstal linear-scratch 예시)를 박스에 병기해 수치를 검증 가능하게 한다.

## 데이터 출처 (실측)
- `aroma_dataset/profiling/severstal/`
  - morphology_features.csv: `class1_00ac8372f` → linearity 0.961 / solidity 0.882 / aspect_ratio 5.09
  - morphology_clusters.json: cluster_id=1 (n_samples=859)
  - deficit_analysis.json: cluster 1 prior(morph_prior) = 0.2373  (859/3620)
  - compatibility_matrix.json: matrix_symmetric["1"] best cell = `0_0_0_1_0` → ctx_prior 1.000
    (구성: P_def_patch=0.159, clean_dist=0.052, ε=1e-3)
- ROI_score = 0.6·1.000 + 0.4·0.2373 = 0.695 ≈ 0.70

## 구성 (세로 흐름도)
1. Defect crop (morphology: lin 0.961, sol 0.882, AR 5.09)
   - 좌분기: GMM → morphology cluster k=1, morph_prior P(k)=0.24
   - 측면(회색): rule subtype = linear_scratch → Stage 2 elastic-warp only (스코어링 미개입)
2. Candidate background patch → context cell 0_0_0_1_0, ctx_prior = compat_sym = 1.00
3. ROI_score = 0.6·ctx + 0.4·morph = 0.70  (ctx=파랑 강조, morph=회색 약)
4. Rank all candidates, take Top-K (K=200) — no deficit; per-class uniform floor (multi)
5. Clean-bg assignment (histogram ∩; void P15 floor)
6. Compat gate: 64px tiling scan-rank-place, best_mean ≥ τ
7. ★ Final pixel-level ROI (bbox) ★

색상: ctx/compat 경로 = 파랑 계열(핵심), morph/prior = 회색, warp side-branch = 옅은 회색 점선.

## 축·해상도
- flowchart(축 없음), figure 크기 ~ 6.5 × 10 in, dpi=300 → >1900px 세로. 콤마 규칙: 5자리↑만.

## Caption (초안)
**Figure 6.** ROI selection and compatibility-aware placement flow, traced on a Severstal linear-scratch example: a defect's morphology cluster and a candidate background's context cell yield morph_prior and ctx_prior, whose weighted sum ranks candidates before the symmetric compatibility gate fixes the pixel-level ROI.
