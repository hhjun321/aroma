# Figure 3.2.4-2 — ring 분포 매칭으로 붙일 좌표를 정하는 과정 (spec)

**스크립트**: `[figure 3.2.4 2] placement_ring.py`
**출력**: `../image/[figure 3.2.4 2] placement_ring.png`
**데이터 루트**: `D:/project/aroma_dataset` (환경변수 `AROMA_DATASET_ROOT`)
**리포 루트**: `D:/project/aroma` (환경변수 `AROMA_REPO`)

**라벨은 영문.** (기존 `placement_footprint` 도식의 한글 라벨 문제를 승계하지 않는다.)

## 선행 도식 대체

이 그림은 `[figure 3.2.4 2] placement_footprint.png` 를 **대체**한다. 구 도식은 `_positive_place` 의 footprint mean-compat · top-K(8) 무작위 샘플 · τ 게이트를 설명했는데, 채택안(`ring_sgm`)에서 그 세 요소가 모두 사라진다. 구 파일은 이력 보존용으로 남기되 본문에서는 인용하지 않는다.

## 목적

§3.2.4 의 `q_k` 가 **실제 붙일 좌표로 바뀌는 과정**을 보인다. 차단해야 할 오해 두 가지:

1. **판정 대상은 footprint 가 아니라 ring 이다.** footprint 는 결함이 덮어써 사라지는 영역이고, 합성 후 남아 결함과 접하는 것은 ring 이다.
2. **점수는 셀별 값의 평균이 아니라 분포 간 교집합이다.** 평균은 "흔한 셀이 많은 자리"를 고르므로 데이터셋에서 가장 평범한 표면으로 수렴한다.

## 구성 — 4 패널 (2×2)

| 패널 | 내용 |
|---|---|
| **A** | 선택된 정상 이미지의 64px 비겹침 격자. 타일마다 `q_k(cell(t))` 를 색으로, cell key 를 하단 소자로. void 타일은 해칭 + 라벨. 컬러바는 target probability |
| **B** | 동일 crop 을 서로 다른 위치에 둔 후보 3개 (best / median / worst by score). 실선 = footprint `F(s)`, 점선 음영 = ring `R(s)`. 각 후보에 score 와 ring 타일 수 표기. footprint 에 void 가 걸린 후보는 `rejected` 로 표시 |
| **C** | best 후보와 worst 후보의 `h_s` 를 `q_k` 와 겹쳐 그린 막대쌍 (상위 셀 12개). `min(h_s, q_k)` 를 음영으로 채워 교집합을 시각화하고 각 패널 제목에 score 값 |
| **D** | 모든 admissible 후보의 score 를 내림차순 정렬한 곡선. argmax 마커, B 의 세 후보 위치 마커, `valid(s)` 로 탈락한 후보 수를 제목에 |

## 대응하는 코드

`scripts/aroma/clean_bg_selection.py`

| 단계 | 함수 | 그림 |
|---|---|---|
| 목표 분포 `q_k` | `_target_by_cluster` — `matrix_symmetric[k]` L1 정규화 | A 색상 · C 회색 막대 |
| 정상 이미지 타일 격자 (void 제외) | `_tile_grid` — `_patch_void` 로 void 판정 | A |
| ring 좌표 | `_ring_keys(si, sj, bw, bh)` — 8이웃 링 | B 점선 |
| footprint void 배제 + 점수 argmax | `_best_ring_site` | B `rejected` · D |

**재구현 금지** — 위 함수를 import 해서 쓴다. 점수 곡선은 `_best_ring_site` 와 동일한 순회로 재계산한다(그 함수는 argmax 만 반환하므로 곡선용으로 내부 로직을 한 번 더 돈다).

## 데이터

- 데이터셋: **severstal** (구 도식과 동일 대상 유지)
- cluster: **k=1** (elongated, 구 도식과 동일)
- ROI / 배경: `roi/severstal/clean_bg_selected_ring.json` 에서 `cluster_id == 1` 이고 `position` 이 있는 항목 중, 후보 자리 수가 충분하고 void 탈락이 발생하는 사례를 고정 시드로 선택
- 호환성 행: `profiling/profiling/severstal/compatibility_matrix.json` 의 `matrix_symmetric["1"]`
- 문맥 셀: `profiling/profiling/severstal/context_features.csv`
- void 바닥값: `_derive_void_floors` 로 산출 (p15, 데이터 유도)

전부 실측. 합성이나 추정값 없음.

## 축·범례·색상

- A: `viridis`, 값은 `q_k(cell)` ∈ [0, max q_k]. void 는 해칭(`///`) + 무채색
- B: footprint 실선 `#ff2828`, ring 점선 + 반투명 `#28c828`, rejected 후보는 실선 회색 + `×`
- C: `q_k` 회색 막대, `h_s` 초록 막대, 교집합 `min(·,·)` 은 진한 초록 음영
- D: 곡선 `#2878c8`, argmax `★`, B 후보 마커 `o`
- 해상도 `dpi=300`, 최소 가로 1000px 이상

## 참조할 section 문장

- *"the footprint F(s), the tiles the defect would cover, and the ring R(s), the eight-neighbour tiles of that rectangle"*
- *"valid(s) ⟺ ∀ t ∈ F(s) : t is observed and not void"*
- *"score(k, s) = ∩( h_s, q_k ), s* = argmax_{ s : valid(s) } score(k, s)"*
- *"the ring is read rather than the footprint, because the footprint is precisely the region the pasted defect overwrites"*
- *"Averaging a compatibility row over tiles rewards positions holding many individually high-scoring cells ... that objective drives placement toward the most ordinary surface of the dataset"*

## Caption (초안)

**Figure 3.2.4-2.** Ring-matched placement on a Severstal background for morphology cluster k=1. (A) 64-pixel tile grid coloured by the cluster's target probability q_k of each tile's context cell, void tiles hatched. (B) Three candidate positions with their footprint F(s) (solid) and ring R(s) (dashed); a candidate whose footprint straddles a void tile is rejected before scoring. (C) Ring histogram h_s against q_k for the best and worst admissible candidate, with the intersection shaded. (D) Score over all admissible candidates, sorted, with the selected position marked.
