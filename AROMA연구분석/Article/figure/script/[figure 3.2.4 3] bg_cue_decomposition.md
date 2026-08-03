# Figure 3.2.4-3 — clean 배경 랭킹의 4 cue 분해 (spec)

**스크립트**: `[figure 3.2.4 3] bg_cue_decomposition.py`
**출력**: `../image/[figure 3.2.4 3] bg_cue_decomposition.png`
**데이터 루트**: `D:/project/aroma_dataset` (환경변수 `AROMA_DATASET_ROOT`)
**리포 루트**: `D:/project/aroma` (환경변수 `AROMA_REPO`)

라벨은 영문.

## 목적

§3.2.4 의 배경 할당식

```
U(g) = Σ_j w_j · u_j(g),   j ∈ { src, cls, mor, siz }
λ_j = E_ROI[ max_g u_j(g) − median_g u_j(g) ],   w_j = λ_j / Σ λ
```

이 실제로 무엇을 하는지 보인다. 차단해야 할 오해 두 가지:

1. **4 cue 가 서로 대체 가능한 게 아니다.** `u_src`·`u_cls` 는 표면 전체에 반응하고, `u_mor` 는 그 둘이 비슷하게 매긴 후보들의 순서를 바꾼다. 형태 군집 축은 도메인 클래스 축과 다른 분할이다.
2. **가중치는 손으로 정한 값이 아니다.** lift 가 0 인 cue 는 가중치도 0 이 되어 랭킹에 영향을 주지 못한다 — `u_siz` 가 4/5 데이터셋에서 실제로 그렇다.

## 구성 — 3 패널

| 패널 | 내용 |
|---|---|
| **A** | 한 ROI 의 valid 배경 풀 전체에 대해 4 cue 를 각각 곡선으로. x축은 **결합 점수 U(g) 내림차순 순위**, y축은 cue 값(각 cue 를 자기 최대값으로 정규화해 한 축에 겹침). 상위 K 개 pool 밴드 음영. 실제 배정된 배경에 마커 |
| **B** | 같은 ROI 의 상위 12 후보에 대한 **누적 막대** — `w_src·u_src`, `w_cls·u_cls`, `w_mor·u_mor`, `w_siz·u_siz` 를 쌓아 U(g) 를 구성. 어느 cue 가 순위를 만들었는지 한눈에 |
| **C** | 데이터셋 5종의 lift 유도 가중치 `w_src / w_cls / w_mor / w_siz` 를 수평 누적 막대로. 각 막대 오른쪽에 `lift_k` 값 병기. `w_siz` 가 대부분 0 인 것이 보이도록 |

## 대응하는 코드

`scripts/aroma/clean_bg_selection.py::build_and_rank`

| 요소 | 코드 |
|---|---|
| `u_src` | `_hist_intersection(good_hist[iid], src_dv)` — `src_dv` 는 소스 결함 이미지 배경 히스토그램 |
| `u_cls` | `_hist_intersection(good_hist[iid], cls_dv)` — `_class_bg_hist` 의 class 집계 |
| `u_mor` | `_hist_intersection(good_hist[iid], tgt_by_k[k])` — `_target_by_cluster` (`--k_fit`) |
| `u_siz` | `_scale_to_fit(wh, good_dim[iid])` |
| `λ_j`, `w_j` | `lifts_*` 수집 → `w_* = λ_* / Σλ` |
| valid pool | `valid_bg_pool` (void_frac ≤ 0.5) |

**재구현 금지** — 위 함수를 import 한다. 패널 C 의 가중치는 5종 각각의 `clean_bg_summary_ring.md` 에 기록된 실측값을 읽는다(재계산 아님).

## 데이터

- 패널 A·B: **severstal**, `roi/severstal/clean_bg_selected_ring.json` 에서 `k_fit` 이 순위를 실제로 바꾼 ROI 를 고정 시드로 선택 (`u_mor` 없이 매긴 순위와 `U(g)` 순위의 상위 1 후보가 다른 사례)
- 패널 C: 5종의 `roi/<ds>/clean_bg_summary_ring.md` 에서 `w_src / w_class / w_size / w_k` 파싱

전부 실측. 추정·창작 없음.

## 축·범례·색상

- cue 색: `u_src` `#2878c8`, `u_cls` `#c85a28`, `u_mor` `#28a03c`, `u_siz` `#9060c0`
- A: y축 "cue value (normalised to its own max)". 배정 배경은 `★`
- B: 누적 막대, x축은 배경 id (상위 12), y축 `U(g)`
- C: 수평 누적, x축 weight (합 1.0)
- `dpi=300`, 가로 1000px 이상

## 참조할 section 문장

- *"u_src(g) = ∩( h_g, p_src(v) ) ... u_siz(g) = min( 1, 0.95 · W_g / w, 0.95 · H_g / h )"*
- *"λ_j = E_ROI [ max u_j − median u_j ], w_j = λ_j / Σ λ_j′"*
- *"a cue that is flat across the pool contributes nothing and cannot distort the ranking"*
- *"it is not redundant with u_cls, because the class axis and the morphology-cluster axis are distinct partitions"*

## Caption (초안)

**Figure 3.2.4-3.** Decomposition of the clean-background ranking for one ROI, with the four cues scored over the valid background pool and combined by their lift-derived weights. Candidates are ordered by the combined score U(g); the panel on the right reports the per-dataset weights the lift derivation yields. A cue's weight reflects how far it separates candidates, not how large its values are — u_src is small in absolute terms among the leading candidates yet carries the largest weight on every dataset, whereas u_siz is flat across this pool and is therefore weighted zero.

> 캡션 마지막 문장은 패널 A/C 의 외견상 모순(파랑 곡선이 상위 후보에서 낮은데 `w_src` 는 최대)을 선제 차단한다. lift 는 값의 크기가 아니라 후보 간 격차를 잰다.
