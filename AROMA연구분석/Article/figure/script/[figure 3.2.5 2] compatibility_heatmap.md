# Figure 16 — Compatibility heatmap (ctx_prior), all 5 datasets (spec)

## 목적
§3.2.5의 ctx_prior(= matrix_symmetric)를 정의·설명. 각 morphology 클러스터 k가 어떤 context cell과 호환되는지(compat_sym)를 클러스터 × 셀 히트맵으로 보여준다. 클러스터마다 선호 배경이 다름(또는 severstal처럼 단일 지배 셀로 수렴)을 드러내 데이터-유도 대칭 호환 모델을 시각화.

## 데이터셋
- **5개 전부**: aitex, kolektor, severstal, mtd, mvtec_leather (각 1장).

## 데이터 출처 (실측)
- `profiling/<ds>/compatibility_matrix.json` → `matrix_symmetric` (행-정규화, peak=1).
- `morphology_clusters.json` → cluster label.

## 구성 (단순)
- 행 = morphology 클러스터 (label 병기), 열 = compat 상위 20 cell(max-across-clusters, mean-compat 내림차순).
- 색 = ctx_prior(compat_sym) 0→1 (Blues). 셀당 수치 없음. 각 행 peak 셀만 crimson 박스.
- x축 = c1…c20(인덱스), y축 = k{k}·label. colorbar = ctx_prior. 제목 = dataset.
- dpi=300.

## Caption (초안)
**Figure 16.** Symmetric compatibility (ctx_prior) across the five datasets, shown as morphology-cluster × context-cell heatmaps (top-20 most-compatible cells, row-normalized so each cluster peaks at 1; each cluster's peak cell boxed). ctx_prior = compat_sym(k,c) = √((P_def(k,c)+ε)·(P_clean(c)+ε)); high only where a cell is both a typical background for the defect cluster and a common clean background. Clusters prefer different cells on heterogeneous surfaces (e.g., AITeX) but share a dominant cell on near-uniform ones (e.g., Severstal).
