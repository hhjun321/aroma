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
| `[figure 3.2.2 3] morph_features.png` | §3.2.2 | Figure 3.2.2-3 | 형태 특징 6종의 검사 영역 (k의 입력 벡터) |
| `[figure 3.2.2 4] context_cell.png` | §3.2.2 | Figure 3.2.2-4 | 64px 패치 → context cell 환산 4단계 |
| `[figure 3.2.3 1 <ds>] morphology_distribution.png` | §3.2.3 | Figure 3.2.3-1 | defect morphology feature 분포 (5종) |
| `[figure 3.2.4 1] roi_selection_flow.png` | §3.2.4 | Figure 3.2.4-1 | ROI 선택→배경 배정→자리 확정 3단 흐름도 (§3.2.4 도입, 구 `3.2.5 3`에서 이동, 신 3단 흐름으로 재생성 완료 2026-08-14) |
| `[figure 3.2.4 2 <ds>] compatibility_heatmap.png` | §3.2.4 | Figure 3.2.4-2 | symmetric compatibility 히트맵 (5종) (구 `3.2.4 1`) |
| `[figure 3.2.4 3] roi_score_composition.png` | §3.2.4 | Figure 3.2.4-3 | ROI_score 가중항 분해 (severstal·aitex 2패널, 구 휴면 `3.2.5 1` 부활 2026-08-14) |
| `[figure 3.2.4 4 <ds>] bg_score_composition.png` | §3.2.4 | Figure 3.2.4-4 | 배경 배정 top-3 대조 (5종) — 원본+crop(≈30%) / src_fit top-3 / class_fit top-3 / bg_score top-3(★=배정) (2026-08-14 최종안: size 패널 제거, 대표 ROI = 면적비 0.30 근접 규칙) |
| `[figure 3.2.4 5 <ds>] placement_ring.png` | §3.2.4 | Figure 3.2.4-5 | 자리 확정 (5종) — -4와 표본 연속(동일 crop·배정 배경, 변 지배 60% 금지 규칙 공유): A 원본 결함의 실제 ring 타일 tgt[k] tint / B 배정 배경 best·mid·worst 자리(best ring tint) / C 실측 h_ring·h_s* vs tgt[k] 분포 대조. rank-1 폴백 시 제목 명기 (2026-08-14 2차 개정 단순 대조판, 구 severstal 단일 4패널→3패널판 대체) |
| ~~`_retired/[figure 3.2.4 2] placement_footprint.png`~~ | — | — | **대체됨 (2026-08-03) → `_retired/` 이동 (2026-08-14).** footprint mean-compat / top-K 샘플 / τ 게이트 — 채택안(`ring_sgm`)에서 제거된 요소들 |
| ~~`_retired/[figure 3.2.4 3] bg_cue_decomposition.png`~~ | — | — | **은퇴 (2026-08-14) → `_retired/`.** §3.2.4 본문이 3-cue 요약판으로 개편되며 4 cue·lift 가중 삽화 제거 (사유: `_retired/README.md`) |
| `[figure 4.1 1] roi_coverage.png` | §4.1 | Figure 4.1-1 | ROI 커버리지 지표 |
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

## 정정 이력 (2026-07-31)

매핑표가 한 리비전 stale 상태였다 — 실제 파일명·섹션과 대조해 정정:

| 이전 표기 | 실제 | 비고 |
|---|---|---|
| `[figure 3.2.4 1 <ds>] morphology_distribution` / §3.2.4 | `[figure 3.2.3 1 <ds>]` / §3.2.3 | 섹션 재번호 미반영 |
| `[figure 3.2.5 2 <ds>] compatibility_heatmap` / §3.2.5 | `[figure 3.2.4 1 <ds>]` / §3.2.4 | 동일 |
| `[figure 4.1 1] quality_proxy_matrix` | `[figure 4.1 1] roi_coverage` | slug 변경 미반영 |
| `[figure 3.2.5 1] roi_score_composition` | **이미지 없음** | script/에 `.py`·`.md`만 존재, PNG 미생성 |

또한 `[figure 3.2.5 3] roi_selection_flow_mod.py`·`.md`가 script/에 있으나 대응 이미지가 없다(`_mod` 변형안). `[figure3] policy_selection.png`는 구 명명 규칙 잔존물이다.

## 스크립트 규약

`script/` 하위에 그림당 `.md`(스펙) + `.py`(생성) 한 쌍을 둔다.

- 출력 경로는 `../image/`, 파일명은 위 표의 고정명. 샘플 stem 등 가변 요소를 파일명에 넣지 않는다
- 운영 함수는 **import해서 호출**하고 재구현하지 않는다(값 불일치 방지). 중간량을 자체 계산해야 하면 최종값이 운영 함수 결과와 일치하는지 `assert`로 확인한다
- 데이터 루트는 `AROMA_DATASET_ROOT`, 리포 루트는 `AROMA_REPO` 환경변수로 재지정 가능하게 한다
- 라벨 언어는 영문이 기본. `[figure 3.2.2 3]`·`[figure 3.2.2 4]` 2건은 현재 한글이며 영문 교체 대기(2026-07-31). `[figure 3.2.4 2]`는 `placement_ring` 재제작으로 영문화 완료(2026-08-03)

## 정정 이력 (2026-08-14) — §3.2.4 재편에 따른 재번호

| 이전 | 이후 | 비고 |
|---|---|---|
| `[figure 3.2.5 3] roi_selection_flow` | `[figure 3.2.4 1]` | §3.2.4 도입 흐름도로 이동 (본문 §3.2.5에는 콜아웃이 원래 없었음). script `_mod` 쌍도 함께 리네임. PNG·스펙 신판 재생성 완료 (`_mod` 쌍은 구식 — 스펙 비고 참조) |
| `[figure 3.2.4 1 <ds>] compatibility_heatmap` | `[figure 3.2.4 2 <ds>]` | 콜아웃 순서 캐스케이드 |
| `[figure 3.2.4 2] placement_ring` | `[figure 3.2.4 3]` | 〃 (은퇴한 구 -3 번호 재사용) |
| `[figure 3.2.4 2] placement_footprint` | `_retired/` | 2026-08-03 대체본의 물리 이동 |
| `[figure 3.2.4 3] bg_cue_decomposition` | `_retired/` | §3.2.4 3-cue 요약 개편으로 은퇴 |
| `[figure 3.2.5 1] roi_score_composition` (PNG 미생성 휴면) | `[figure 3.2.4 3]` **부활·PNG 생성** | Defect Crop Selection 삽화. 경로 수정(AROMA_DATASET_ROOT)·주석 추가 후 생성 |
| `[figure 3.2.4 3] placement_ring` (당일 재번호분) | `[figure 3.2.4 4]` | 신규 -3 삽입에 따른 2차 캐스케이드 |
| `[figure 3.2.4 4] placement_ring` (당일 재번호분) | `[figure 3.2.4 5]` | 신규 -4(bg_score 과정도) 삽입에 따른 3차 캐스케이드 |
