# Figure 3.2.4-2 — 위치 결정 5단계와 footprint mean-compat (spec)

**스크립트**: `[figure 3.2.4 2] placement_footprint.py`
**출력**: `../image/[figure 3.2.4 2] placement_footprint.png`
**데이터 루트**: `D:/project/aroma_dataset` (환경변수 `AROMA_DATASET_ROOT`)
**리포 루트**: `D:/project/aroma` (환경변수 `AROMA_REPO`)

> ⚠ **라벨이 한글이다.** figure 내용을 숙지한 뒤 영문으로 교체 예정(사용자, 2026-07-31).

## 목적

§3.2.4의 `ctx_prior(k, c)`가 **실제 붙일 좌표로 바뀌는 과정**을 보인다. 핵심 오해 차단 지점: **판정 단위는 cell이 아니라 crop footprint**다. 같은 cell 위에 놓인 두 후보라도 crop이 함께 덮는 이웃 타일 조합이 달라 점수가 갈린다.

## 구성 — 3 패널

| 패널 | 내용 |
|---|---|
| **A** | normal 이미지의 64px 비겹침 격자. 타일별 compat(색 + 상단 숫자)과 cell key(하단), void 해칭 |
| **B** | 동일 crop을 서로 다른 위치에 둔 후보 3개(best/median/worst). 실선=footprint, 점선 음영=`_tile_anchors`가 반환한 덮는 타일. 각 후보의 mean-compat + 타일 수 표기 |
| **C** | 비-void 후보를 mean-compat 내림차순 정렬한 곡선 + top-K(8) 밴드 + best non-void mean(τ 판정 기준) 파선 + A/B의 세 후보 순위 마커 |

## 대응하는 코드 5단계

`scripts/aroma/generate_defects.py::_positive_place`(`:888-1030`)

| 단계 | 코드 | 그림 |
|---|---|---|
| 1 후보 격자 열거 | `stride=_POS_STRIDE(32)`, `_POS_MAX_CAND` 초과 시 stride 2배씩 조대화, 우·하단 끝 좌표 항상 포함 | C 제목의 후보 수 |
| 2 타일별 (compat, void) 캐시 | `compat_row.get(cell, 0.5)` — **미관측 cell은 거부가 아니라 중립 0.5**. cluster 무관이라 캐시 성립 → O(서로 다른 타일 수) | A |
| 3 footprint mean-compat | `_tile_anchors`로 덮는 타일을 모아 compat 평균. `_COMPAT_TILE_AGG="mean"` 기본(`'min'`은 상수에만 존재, 미사용) | **B** |
| 4 void 배제 | footprint에 void 타일이 1개라도 있으면 탈락(compat 무관). 비-void 0개면 `None` 반환 → 호출부가 다른 normal로 escape | C 제목 |
| 5 top-K 샘플링 | 안정 정렬(동점=스캔 순서) 후 상위 `_POS_TOPK(8)`에서 `rng.choice`. τ 판정은 `nonvoid[0]`(최선)으로, 배치는 샘플 위치로 | C |

## 데이터 출처 (실측)

- `severstal/train/good/00031f466.jpg` — `[figure 3.2.2 4] context_cell`과 **동일 이미지**(두 그림 연결)
- `profiling/profiling/severstal/compatibility_matrix.json` → `matrix_symmetric["1"]`(cluster 1 행), `bin_edges`

**운영 함수 직접 import** (재구현 금지): `distribution_profiling._extract_context_features` / `_context_cell_key`, `generate_defects._is_clean_background` / `_tile_anchors`, 상수 `_COMPAT_TILE`(64) / `_POS_STRIDE`(32) / `_POS_TOPK`(8).

## 파라미터

| 항목 | 값 | 근거 |
|---|---|---|
| 대상 cluster | `1` | severstal elongated 군집, `P(k)=0.24`, n=859 |
| crop 크기 | 160×96 | x 3타일 × y 2타일 = footprint 6타일. 덮는 타일이 복수임을 보이는 최소 크기 |
| `MIN_BG_QUALITY` | **0.5** | 운영 설정. 0.7은 5종 good 타일 98~100%를 void로 만들어 사용 불가 |
| `BLUR_THRESHOLD` | 100.0 | `_background_quality_score` 기본값 |

## 실측 결과

후보 총 **276개**, 비-void 276개, void 탈락 **0개**.

| 후보 | 위치 | footprint 타일 | mean-compat |
|---|---|---|---|
| best | (1376, 128) | 6 | **0.3565** |
| median | (416, 160) | 6 | 0.1354 |
| worst | (512, 64) | 6 | **0.0276** |

**타일 수가 같은데 평균이 13배 차이난다** — 이것이 이 그림의 핵심 메시지다. 패널 A에서 타일 compat이 0.01~0.36으로 갈리고 cell도 `0_0_1_0_1`·`1_1_1_0_1`·`0_0_2_0_1` 등으로 흩어지므로, 어떤 조합을 덮는지가 점수를 정한다.

## ⚠ void 탈락 0개의 진짜 이유

데이터에 void가 없어서가 아니다. **`_is_clean_background`가 현 환경에서 fail-open이다.**

`_background_quality_score`(`:457-460`)가 입력을 `float32`로 변환한 뒤 `cv2.Laplacian(gray, cv2.CV_64F)`를 호출하는데, **OpenCV 4.13.0은 `CV_32F → CV_64F` 조합을 미지원**해 예외를 던진다. `_is_clean_background`(`:517-521`)가 이를 `except: return True`로 삼켜 순수 검은 패치도 clean으로 판정한다. **로컬·Colab 모두 재현.**

⇒ 스크립트는 이 사실을 패널 C 제목에 자동 표기한다(`void 탈락 0`일 때). 그림을 "void 필터가 잘 작동한다"의 근거로 인용하면 안 된다.

상세·수정 시 결정 사항: `.claude/.dev_note/aroma_cleanbg_gate_cv2_dtype_failopen.md`

## 이미지 규격

- `figsize=(16, 11)`, `dpi=200`, `bbox_inches="tight"` → 약 3873×2107, 종횡비 1.84:1
- GridSpec 3×1, `height_ratios=[1.25, 1.25, 1.0]`
- 하단 note는 `textwrap.wrap(150)` — 한 줄로 두면 레이아웃이 6505px로 늘어난다

## Caption (초안)

**Figure 3.2.4-2.** 호환성 점수가 배치 좌표로 바뀌는 과정. (A) 64픽셀 타일별 compat과 context cell, (B) 동일 crop을 다른 위치에 둔 세 후보와 각각의 footprint 평균 compat, (C) 후보 순위와 top-K 추첨 구간.

## 논문 내 위치

- **섹션**: 3.2.4 (ROI Selection and Compatibility-Aware Placement)
- **배치**: Figure 3.2.4-1(cluster × cell 히트맵) 직후 — 행렬을 본 뒤 그 값이 좌표로 소비되는 과정을 보이는 순서
- **근거 문서**: `../../../aroma_core_compatibility_model_20260729.md` §6
