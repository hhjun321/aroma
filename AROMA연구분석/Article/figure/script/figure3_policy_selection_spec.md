# Figure 3 — Policy Selection Results (Per-Dataset Evaluation Scores)

## 목적 (연결 section)

§3.2.2: "Figure~\ref{fig:policy_selection}, which reports the empirical evaluation scores of the surviving candidate policies and the policy ultimately chosen for each dataset."

5개 dataset별로 **morphology & context axis에서 candidate policy들의 silhouette 평가 점수**를 보이고, **선택된 정책을 강조**하여, policy selection의 투명성과 per-dataset data-driven 선택을 가시화.

## 데이터 출처 (실측)

`D:\project\aroma_dataset\complexity\<dataset>\complexity_report.json`

각 JSON의 `evaluation_results[]` array:
- `policy`: 정책명 ("otsu", "gmm", "percentile")
- `silhouette`: 평가 점수 (scalar)
- `axis`: "morphology" 또는 "context"

selected policies:
- `morphology_policy`: 선택된 morphology 정책
- `context_policy`: 선택된 context 정책

| Dataset | Morph Candidates | Morph Selected | Ctx Candidates | Ctx Selected |
|---|---|---|---|---|
| AITeX | Otsu (0.424), GMM (0.268) | **Otsu** | GMM (0.327), Percentile (−0.015) | **GMM** |
| Kolektor | Otsu (0.288), GMM (0.368) | **GMM** | GMM (0.090), Percentile (0.065) | **GMM** |
| Severstal | Otsu (0.312), GMM (0.146) | **Otsu** | GMM (0.102), Percentile (0.167) | **Percentile** |
| MTD | Otsu (0.337), GMM (0.271) | **Otsu** | GMM (0.078), Percentile (0.111) | **Percentile** |
| MVTec Leather | Otsu (0.477), GMM (0.297) | **Otsu** | GMM (0.165), Percentile (0.075) | **GMM** |

## 플롯 구성

### 전체 레이아웃
- **형식**: 병렬 bar chart (grouped bars)
- **행 (y축)**: 5개 dataset (AITeX, Kolektor, Severstal, MTD, MVTec Leather) — 위에서 아래
- **x축**: silhouette score [−0.05, 0.50] (음수도 포함하므로 0을 중심축으로)
- **막대 그룹**:
  - dataset당 2개 그룹: Morphology | Context
  - 각 그룹 내 2-3개 bar (후보 정책별)
  
### 막대 색상 및 강조
- **Morphology 후보**: 파란색 계열 (Otsu, GMM 구분 가능)
- **Context 후보**: 초록색 계열 (GMM, Percentile 구분 가능)
- **선택된 정책 강조**: 각 dataset의 선택 정책 bar 위에 **★ 마크** 또는 **굵은 테두리(linewidth=2.5)**
- 비선택 후보: alpha=0.6 (투명도 낮춤)

### 라벨 및 범례
- **y축 라벨**: dataset명
- **x축 라벨**: "Silhouette Score"
- **축 격자**: x축만 (0.05 간격) — 가독성
- **범례** (우상단): 
  - "Morphology: Otsu | GMM"
  - "Context: GMM | Percentile"
  - "★ = selected"

### 데이터 정합성
- 모든 점수는 JSON에서 직접 로드 (하드코딩 금지)
- 음수 점수(Percentile on AITeX, −0.015) 포함 → x축 음수 영역 표시

## 스타일

- figure_patterns.md 규약 준수
- 크기: 약 10×6 in (넓고 얕은 비율, dataset 라벨 가독성)
- 300 dpi, serif 폰트, 흰 배경
- 격자 옅게(linestyle=":", alpha=0.4)

## 저장

- 스크립트: `figure/script/figure3_policy_selection.py`
- 출력: `figure/image/[figure3] policy_selection.png`

## Caption (초안)

**Figure 3.** Empirical evaluation scores for candidate morphology and context policies across the five evaluation datasets. For each dataset, the two-member context-axis pool (Gaussian mixture and percentile partitioning) and the morphology-axis pool (shaped by per-feature valley counts and gated by MCI thresholds) were evaluated by silhouette separability. The selected policy for each dataset (marked with ★) was determined entirely by measured complexity statistics rather than manual tuning, making the policy decision auditable and per-dataset. The Otsu morphology policy was selected on four datasets (AITeX, Severstal, MTD, MVTec Leather) and Gaussian mixture on Kolektor; the context axis selected either Gaussian mixture (AITeX, Kolektor, MVTec Leather) or percentile partitioning (Severstal, MTD) depending on the measured context complexity regime.
