# Figure 파일명 규칙 (섹션 기반)

논문 개정 때마다 저널의 순차 그림 번호(Figure 1, 2, …)가 계속 바뀌므로, **이미지 파일명은 그림이 참조되는 섹션 번호로 고정**한다. 순차 번호는 본문 prose(`Figure N`)에서만 관리하고, 파일명과 분리한다.

## 규칙

```
[figure <섹션> <인덱스> <데이터셋?>] <slug>.png
```

- `<섹션>`: 그림이 처음 콜아웃되는 절 번호 (예: `3.1`, `3.2`, `3.2.2`, `4.1`)
- `<인덱스>`: 같은 절 안에서 콜아웃 순서 (1부터). 절에 그림이 하나뿐이면 인덱스 생략 가능(예: `[figure 3.1]`, `[figure 3.2]`)
- `<데이터셋>`: 데이터셋별 다패널 그림일 때만 부착 (`severstal`/`aitex`/`kolektor`/`mtd`/`mvtec_leather`)

## 현재 매핑 (파일 ↔ 섹션 ↔ 본문 Figure 번호)

| 파일명 | 섹션 | 본문 표기 | 내용 |
|--------|------|-----------|------|
| `[figure 3.1] complexity_landscape.png` | §3.1 | Figure 3.1 | MCI vs. CCI 복잡도 지형 |
| `[figure 3.2] aroma_pipeline.png` | §3.2 | Figure 3.2 | AROMA 파이프라인 아키텍처 |
| `[figure 3.2.2 1] morphology_clusters.png` | §3.2.2 | Figure 3.2.2-1 | data-driven morphology clusters |
| `[figure 3.2.2 2 <ds>] context_distribution.png` | §3.2.2 | Figure 3.2.2-2 | 배경 context feature 분포 (5종) |
| `[figure 3.2.4 1 <ds>] morphology_distribution.png` | §3.2.4 | Figure 3.2.4-1 | defect morphology feature 분포 (5종) |
| `[figure 3.2.5 1] roi_score_composition.png` | §3.2.5 | Figure 3.2.5-1 | ROI_score 구성 |
| `[figure 3.2.5 2 <ds>] compatibility_heatmap.png` | §3.2.5 | Figure 3.2.5-2 | symmetric compatibility 히트맵 (5종) |
| `[figure 3.2.5 3] roi_selection_flow.png` | §3.2.5 | Figure 3.2.5-3 | ROI 선택·배치 흐름 |
| `[figure 4.1 1] quality_proxy_matrix.png` | §4.1 | Figure 4.1-1 | ROI placement quality metrics |
| `[figure 4.1 2] roi_bbox_qualitative.png` | §4.1 | Figure 4.1-2 | qualitative ROI placement |
| `[figure 4.1 3] bg_similarity_datasets.png` | §4.1 | Figure 4.1-3 | background-selection compatibility |
| `[figure 4.2 1] aitex_roi_comparison.png` | §4.2 | Figure 4.2-1 | AITeX ROI 비교 |
| `[figure 4.2 2] kolektor_roi_comparison.png` | §4.2 | Figure 4.2-2 | Kolektor ROI 비교 |
| `[figure 4.3 1] severstal_roi_comparison.png` | §4.3 | Figure 4.3-1 | Severstal ROI 비교 |
| `[figure 4.3 2] mtd_roi_comparison.png` | §4.3 | Figure 4.3-2 | MTD ROI 비교 |
| `[figure 4.3 3] mvtec_leather_roi_comparison.png` | §4.3 | Figure 4.3-3 | MVTec Leather ROI 비교 |

> 본문 표기(prose)도 파일명 규칙에 맞춰 통일 완료: prose는 `Figure 3.2.5-1`(인덱스는 하이픈), 단일 그림 절은 `Figure 3.1`. 다패널(5종) 그림은 prose에서 하나의 라벨로 참조하고 데이터셋 접미사는 이미지 파일에만 붙인다.
>
> §3.2.2: morphology_clusters(`3.2.2-1`, 콜아웃 먼저) → context distributions(`3.2.2-2`) 순으로 본문에 모두 참조됨.

## Deprecated

- `[figure3] policy_selection.png` — policy-selection 서사가 compat-gate 재편으로 제거되어 본문에서 더 이상 참조되지 않음. 규칙 미적용(옛 이름 유지), 사용 시 재검토.

## 스크립트

- **단일 그림 스크립트/스펙**(`script/*.py`·`*.md`)도 같은 규칙으로 리네임 완료 (예: `[figure 3.2] aroma_pipeline.py`, `[figure 3.2] pipeline_spec.md`, `[figure 3.2.5 2] compatibility_heatmap.py`). 각 스크립트의 출력 경로·상호 참조도 함께 갱신.
- **다중 그림 배치 생성기**는 여러 절/데이터셋을 한 번에 생성하므로 단일 섹션명을 붙일 수 없어 서술형 이름 유지: `generate_all_roi_comparison_figures.py`, `generate_figures8_9_roi_comparison(.py/_v2.py)`, `generate_figure8_aitex_roi_comparison.py`.
- **비그림·deprecated 유지**: `table_background_categories_spec.md`(§3.2.3 Table 3), `figure3_policy_selection.py`/`_spec.md`(deprecated).
- 실행 시 파일명에 공백·대괄호가 있으므로 따옴표로 감쌀 것: `python "[figure 3.2] aroma_pipeline.py"`.
