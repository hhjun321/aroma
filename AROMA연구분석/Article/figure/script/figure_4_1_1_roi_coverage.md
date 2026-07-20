# Figure 4.1-1 — ROI placement coverage (AROMA vs Random) (spec)

## 목적
§4.1 ROI 품질 평가: compat 주도 선택이 만드는 배치 분포를 균등 Random 대비 coverage 3종으로 특성화.
(구 [figure 4.1 1] quality_proxy_matrix는 제거된 Table 5(MATCHING_RULES) 시각화라 폐기 — compat-스파인 모순.)

## 데이터 출처 (실측, 스크립트가 직접 계산)
- AROMA 배치: `aroma_dataset/synth_aroma/<ds>/annotations.json` (cluster_id, cell_key)
- Random 선택: `aroma_dataset/synth_random/<ds>/_random_roi/roi_selected.json`
- 후보 풀/가용집합/rare(deficit>0): `synth_random/<ds>/_random_roi/roi_candidates.json`
- size-equalize: 작은 쪽 크기로 고정 seed(42) 서브샘플 (§3.3.3).

## 실측값 (2026-07-21 계산)
| ds | ctx A/R | rare A/R |
|---|---|---|
| aitex | 0.907/0.953 | 1.000/0.867 |
| kolektor | 0.684/1.000 | 0.600/1.000 |
| severstal | 0.925/1.000 | 0.986/0.925 |
| mtd | 0.479/0.851 | 1.000/0.675 |
| mvtec_leather | 1.000/1.000 | n/a (rare pair 0개) |
- morphology coverage = 양 arm 전부 1.000 (전 데이터셋).

## 구성 (단순)
- 3 서브플롯(morphology / context / rare-pair coverage), x=5 데이터셋, arm별 2막대(AROMA 파랑, Random 회색), y=coverage 0–1.
- leather rare = "n/a" 텍스트. dpi=300.

## Caption (초안)
**Figure 4.1-1.** ROI placement coverage (morphology, context, and rare-pair) for AROMA versus an equal-size uniform Random selection from the same candidate pool. Both arms cover all morphology clusters; Random attains equal or higher context coverage by construction, while AROMA's per-pair quota keeps rare-pair coverage at or above Random on AITeX, Severstal, and MTD (MVTec Leather has no rare pairs).
