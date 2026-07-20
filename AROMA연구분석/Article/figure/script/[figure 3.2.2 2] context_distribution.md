# Figure 13 — Background Context-Feature Distributions per dataset (spec)

## 목적
defect_type의 morphology_histograms(형태 특징 분포 → 결함 군집 근거)와 **대칭**되는 background_type figure.
배경 셀/타입(background_type)의 근거인 **context 특징 5종의 데이터셋별 분포**를 보여주고, 실제 compat 셀 경계(P33/P66 tertile = `bin_edges`)를 오버레이한다.

## 데이터 출처 (실측)
- `aroma_dataset/profiling/<ds>/context_features.csv` — 5 context 특징(local_variance, edge_density, texture_entropy, frequency_energy, orientation_consistency), 전 patch 모집단(= bin_edges 산출 모집단과 동일).
- `aroma_dataset/profiling/<ds>/compatibility_matrix.json` → `bin_edges[feature]` = per-feature [P33, P66] 2개 경계(= 3 context bin). 이 값을 그대로 dashed line으로 표시.
- 데이터셋: aitex, kolektor, severstal, mtd, mvtec_leather (각 1장).

## 구성 (데이터셋당 1장, morphology_hist 스타일)
- 2×3 그리드: 5개 특징 서브플롯 + 6번째는 범례/주석.
- 각 서브플롯: 파랑 히스토그램(전 patch), 빨강 dashed 2줄 = bin_edges(P33/P66) → 3 context bin 경계.
- x축은 [P1, P99]로 클립(outlier 방지). 서브플롯 제목 = 특징명(orientation_consistency는 "orientation_consistency (entropy; low=coherent)"로 부기).
- 전체 제목: "Background Context Distributions — {dataset}".

## 축·해상도
- figsize ~ 13×7 in, dpi=300 → >3000px. 콤마: 5자리↑만.

## Caption (초안)
**Figure 13.** Per-dataset distributions of the five profiled background context features, with the P33/P66 tertile boundaries (dashed) that discretize each feature into the three-level context cells used by the compatibility model; the roster's structure-sensitive features (frequency energy, orientation entropy) separate directional/periodic surfaces (AITeX) from near-uniform ones (Kolektor, MVTec Leather).
