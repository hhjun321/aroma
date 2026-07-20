# Figure 15 — Morphology clusters (defines k) (spec)

## 목적
§3.2.2의 GMM/BIC morphology 군집을 시각적으로 정의. Figure 14의 y축 `k`가 "형태 공간에서 데이터로 유도된 군집"임을 정박한다.

## 대표셋
- severstal, aitex (2행).

## 데이터 출처 (실측)
- `profiling/<ds>/morphology_features.csv` — image_id, linearity, aspect_ratio (+6특징)
- `profiling/<ds>/morphology_clusters.json` — cluster_assignments(image_id→k), clusters[{cluster_id, n_samples, label, centroid}]
- P(k) = n_k / Σn = morph_prior.

## 구성 (scatter + 군집크기 막대)
- 2행(dataset) × 2열: [scatter (넓게) | cluster-size 막대 (좁게)], gridspec width_ratios=[3,1].
- **Scatter**: x = log10(aspect_ratio), y = linearity, 점 = 결함 인스턴스, **색 = GMM 군집 k**(tab10 이산), 각 군집 centroid = ★(검은 테두리).
- **막대**: 가로 barh, per-cluster P(k)=n_k/N, 색은 scatter와 동일 매핑, y라벨 = `k{n}·label`.
- 범례 불필요(막대 y라벨이 k·label 담당). dpi=300.

## Caption (초안)
**Figure 15.** Data-driven morphology clusters for Severstal and AITeX. Left: each defect instance in the linearity–log(aspect ratio) plane, colored by its Gaussian-mixture cluster k (BIC-selected count; cluster centroids marked ★). Right: cluster prior P(k)=n_k/N (the morph_prior of §3.2.5). k is a shape-based cluster, not the defect class.
