# Figure 7 — ROI Placement Qualitative Comparison (Baseline vs AROMA vs Random)

## 목적
데이터셋별 대표 이미지 1장 위에서 ROI 배치 정책의 거동을 육안으로 확인한다. 세 가지를 한 행에 나란히 놓는다 — 실제 결함이 발생한 위치(Baseline), AROMA의 compatibility 스캔 배치(`clean_bg_selection` + `roi_selection`의 scan-rank-place 로직)가 후보 결함을 놓을 위치, 동일 후보를 균등 무작위로 놓을 위치(Random). **정성 확인 전용**이며 정량 지표는 아니다(정량은 Figure 6).

## 구성
**5행(데이터셋) × 3열**
- **열**: Baseline | AROMA | Random (열 제목은 첫 행에만 표기)
- **행**: AITeX, Kolektor, Severstal, MTD, MVTec Leather (좌측 여백에 데이터셋명만 표기)
- 한 행의 3개 열은 **모두 같은 원본 이미지**(데이터셋별 대표 실 결함 이미지 1장)를 쓰고 bbox 오버레이만 다르다:
  - **Baseline**: 박스 1개 — 실제 ground-truth 결함 bbox (대표 `roi_selected.json` 엔트리의 `defect_bbox` 필드).
  - **AROMA**: 박스 5개 — 실제 운영 scan-rank-place 함수(`scripts/aroma/generate_defects.py`의 `_positive_place`)를 대표 이미지에 직접 호출해 산출한 후보 배치. 박스 **크기**는 동일 데이터셋 `roi_selected.json` 풀에서 뽑은 실제 결함 bbox 5개(subtype·종횡비가 다양한 것)를 쓰고, cluster별 compatibility 행은 `compatibility_matrix.json`의 `matrix_symmetric`을 쓴다. void 타일은 운영 게이트와 동일하게 `_is_clean_background`로 배제한다.
  - **Random**: 동일한 bbox 크기 5개를, 운영 코드의 `_random_paste_position`(균등 무작위 유효 top-left)으로 배치. compatibility·void 고려 없음.
- 박스 크기는 전부 실측값이며 창작하지 않는다. AROMA/Random의 **위치**만 이 대표 이미지에서 실제 배치 함수를 재실행해 계산한다 — 저장된 ROI JSON은 이미지당 다중 후보 위치를 보존하지 않기 때문이다(전 데이터셋 모든 엔트리의 `position` 필드가 `null`, 2026-07-16 확인).
- **후보 크기 기준 (2026-07-16 개정)**: AITeX는 5개 후보 크기를 여전히 **이미지 면적의 ~40%** 로 맞춘다(작은 타일 위에 크고 뚜렷한 후보). Kolektor·Severstal·MTD·MVTec Leather는 40% 대신 **해당 대표 이미지 자신의 baseline 결함 bbox 면적**으로 변경했다 — 40%에서는 이 4종의 AROMA/Random 후보가 모두 비슷하게 과대해져 배치 차이가 잘 보이지 않았고, 실제 baseline 결함 스케일에 맞추면 AROMA-vs-Random의 위치 분산이 드러난다.
- **표시 종횡비 (2026-07-16 개정)**: 원본 이미지의 실제 종횡비와 무관하게(예: Severstal 1600×256, Kolektor 500×1255) 모든 행의 표시 셀을 MVTec Leather의 1024×1024와 같은 **1:1**로 통일한다. 격자를 균일하게 보기 위한 **표시 전용** 처리(`imshow` + `aspect="auto"`)이며, 픽셀 내용과 bbox 좌표는 건드리지 않는다.

## 데이터 출처
- 대표 이미지 + baseline bbox + 후보 bbox 크기: `D:\project\aroma_dataset\roi\<dataset>\roi_selected.json` (`image_path`, `defect_bbox`, `cluster_id`, `defect_subtype`)
- Compatibility 행: `D:\project\aroma_dataset\profiling\profiling\<dataset>\compatibility_matrix.json` (`matrix_symmetric[str(cluster_id)]`, `bin_edges`)
- 배치 함수 (**재구현 아님, import해서 호출**):
  - `scripts/aroma/generate_defects.py` — `_positive_place`, `_random_paste_position`, `_is_clean_background`, `_load_context_cell_helpers`
  - `scripts/aroma/clean_bg_selection.py` — `_effective_wh` (실제 생성 시점의 fit-rescale. 배치 전에 적용해, 과대한 실 bbox가 조용히 잘리는 대신 운영과 동일하게 재스케일되도록 함)
- 이미지 파일: `D:\project\aroma_dataset\{aitex_tiled,kolektor,severstal,mtd,mvtec_leather}\test\...` 로컬 미러 (데이터셋 루트별로 Colab 경로 → 로컬 경로 매핑)

## 이미지 규격
- **해상도**: ≥300 dpi, 한 변 ≥1000 px (FIGURE_TABLE_WORKFLOW.md §6.4 준수)
- **레이아웃**: GridSpec(5, 3), 행 = 데이터셋, 열 = arm, 셀 종횡비 1:1 통일(위 참조)
- **박스 색**: Baseline = 빨강, AROMA = 초록, Random = 파랑 — 외곽선만, 두께 2px. 한 열 내 5개 박스는 같은 색(단순화 지시 반영)
- **라벨**: 열 제목("Baseline"/"AROMA"/"Random")은 첫 행에만, 데이터셋명은 좌측 여백 행 라벨로만. 그 외 주석(점수·크기 등) 없음 — 단순 유지 지시 반영
- **RNG seed**: 데이터셋별 고정(`42 + dataset_index`). AROMA top-K 샘플과 Random 추출 양쪽의 재현성 확보

## 캡션
초안 (영문, 34 단어 — 삽입 시 ≤25 단어로 축약):
> **Figure 7.** Qualitative ROI placement comparison across the five datasets. Each row shows one representative image with the real defect region (Baseline), five AROMA-scored candidate placements, and five uniform-random placements of identical box sizes.

최종 (영문, 24 단어 ✓):
> Qualitative ROI placement comparison. Baseline shows the real defect region; AROMA and Random overlay five identical-size candidate boxes scored by compatibility versus placed uniformly at random.

## 논문 내 위치
- **섹션**: 4.1 (ROI Quality Evaluation)
- **배치**: Figure 6 논의 뒤. 정량 지표(coverage/entropy/Gini)에 대한 정성적 보완.
- **본문 호출**: "Figure 7 illustrates this behavior qualitatively on one representative image per dataset: ..."
