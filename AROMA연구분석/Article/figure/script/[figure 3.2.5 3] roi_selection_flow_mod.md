# Figure 6 (mod) — ROI Selection & Compatibility-Aware Placement Flow (theoretical spec)

## 목적
§3.2.5의 ROI 선택→배치 흐름을 하나의 흐름도로 시각화한다. 원본(`[figure 3.2.5 3] roi_selection_flow.md`)은
Severstal linear-scratch 실측 수치를 박스에 병기했으나, 본 `_mod` 버전은 **특정 데이터셋·표본에 종속되지 않는
이론적 흐름도**로 작성한다. 모든 박스는 기호·수식만 표기하고, 실측 데이터 출처 섹션은 두지 않는다.

## 구성 (세로 흐름도, 기호만)
1. Defect crop — morphology features (linearity, solidity, aspect ratio)
   - 좌분기: GMM → morphology cluster k, morph_prior P(k)
2. Candidate background patch → context cell c, ctx_prior(k, c) = matrix_symmetric(k, c)
3. ROI_score(k, c) = 0.6·ctx_prior(k, c) + 0.4·morph_prior(k)
4. Rank all candidates, take Top-K
5. Clean-bg assignment
6. Compat gate: 64px tiling scan–rank–place, accept if best_mean ≥ τ
7. Final pixel-level ROI (bbox)

## 축·해상도
- flowchart(축 없음), figure 크기 ~ 6.5 × 10 in, dpi=300 → >1900px 세로.

## Caption (초안)
**Figure 6 (mod).** ROI selection and compatibility-aware placement flow (theoretical). A defect crop's
morphology cluster k and a candidate background's context cell c yield morph_prior(k) and ctx_prior(k, c),
whose weighted sum ROI_score(k, c) ranks candidates before the symmetric compatibility gate fixes the
pixel-level ROI. No dataset-specific values are shown; see `[figure 3.2.5 3] roi_selection_flow.md` for a
concrete Severstal-traced instance of this same flow.
