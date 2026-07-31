# Figure 3.2.2-4 — 64px 패치가 context cell `c`로 환산되는 흐름 (spec)

**스크립트**: `[figure 3.2.2 4] context_cell.py`
**출력**: `../image/[figure 3.2.2 4] context_cell.png`
**데이터 루트**: `D:/project/aroma_dataset` (환경변수 `AROMA_DATASET_ROOT`)
**리포 루트**: `D:/project/aroma` (환경변수 `AROMA_REPO`)

> ⚠ **라벨이 한글이다.** figure 내용을 숙지한 뒤 영문으로 교체 예정(사용자, 2026-07-31).

## 목적

§3.2.2의 context cell `c`가 **어떻게 만들어지는지**를 4단계 흐름으로 보인다. 독해에서 반복 확인된 두 오해를 차단하는 것이 이 그림의 일차 목적이다:

1. `c`를 개별 64px 패치로 오해 — 실제로는 다수 패치를 묶는 **범주**
2. `c`의 판정 단위가 이미지라고 오해 — 실제 판정 단위는 **패치**

CASDA `ex_casda/figure/[figure3] background_type.png` 형식 준용.

## 구성 — 4단계

| 단계 | 위치 | 내용 |
|---|---|---|
| **1** | 상단 전폭 패널 | 64px 비겹침 격자에서 대상 패치 1개 선택 |
| **2** | 열1 Computation Map / 열2 Reduced Distribution | context feature 5종 각각 산출 |
| **3** | 열3 Bin Assignment | 데이터셋 분포 + P33/P66 tertile → bin 0/1/2 |
| **4** | 하단 전폭 패널 | 다섯 자리 조립 → `c = d0_d1_d2_d3_d4` |

**행 = `CONTEXT_FEATURES` 5종** — 순서가 cell key 자릿수 순서와 동일하다: `local_variance`(자리0) · `edge_density`(1) · `texture_entropy`(2) · `frequency_energy`(3) · `orientation_consistency`(4). 좌여백에 자릿수·특징명·수식·설명·값·bin 배지.

특징별 열1·열2 구성:

| 특징 | 열1 Computation Map | 열2 Reduced Distribution |
|---|---|---|
| local_variance | 원 패치 | 픽셀 명도 히스토그램 + 평균선 |
| edge_density | Sobel \|grad\| 맵 | \|grad\| 히스토그램 + 평균선 |
| texture_entropy | LBP 코드 맵 (P=8, R=1, uniform) | LBP 10-bin (엔트로피 대상) |
| frequency_energy | fftshift \|F\| (log) + 저주파 반경 원 | 저주파 vs 고주파 에너지 막대 |
| orientation_consistency | 기울기 방향 맵 (−180~180°) | 방향 18-bin (엔트로피 대상) |

열3은 `[figure 3.2.2 3] morph_features`의 `Dataset Position` 열과 동일 스타일 — 데이터셋 good 패치 분포 + P33/P66 파선 + 이 패치 위치(행 색 굵은 선) + 선택된 tertile 음영 + bin 숫자.

## 데이터 출처 (실측)

- `severstal/train/good/<stem>.jpg` — 대상 normal 이미지
- `profiling/profiling/severstal/compatibility_matrix.json` → `bin_edges` (데이터셋별 P33/P66)
- `profiling/profiling/severstal/context_features.csv` — 열3 분포. **115MB이므로 필요한 5개 컬럼만 스트리밍**(`load_good_dist`). 전체 dict 적재 시 `MemoryError`

**bin 환산·cell key는 운영 함수 직접 호출**: `distribution_profiling._extract_context_features` / `_context_cell_key`.

**중간량 검증**: 열1·열2의 중간 계산은 `_extract_context_features`(`:213-254`) 내부를 미러한 뒤, 산출된 5개 최종값이 import한 운영 함수 결과와 일치하는지 **`assert` (오차 < 1e-9)** 로 확인한다.

## 대상 패치 선정

`STEM = "00031f466"` — `[figure 3.2.4 2] placement_footprint`와 **동일 이미지**를 써서 두 그림이 연결된다.

패치는 그 이미지 **최빈 cell의 인스턴스 중 가로 중앙에 가장 가까운 타일**로 자동 선정한다(`TARGET_CELL = ""`이면 자동). 하드코딩 없음.

실측 결과 — 타일 `(i=0, j=16)`, `c = 0_0_1_0_1`:

| 자리 | 특징 | 값 | P33 | P66 | bin |
|---|---|---|---|---|---|
| 0 | local_variance | 21.2024 | 50.34 | 169 | **0** |
| 1 | edge_density | 16.5227 | 24.66 | 43.7 | **0** |
| 2 | texture_entropy | 3.2473 | 3.194 | 3.251 | **1** |
| 3 | frequency_energy | 0.4245 | 0.443 | 0.5094 | **0** |
| 4 | orientation_consistency | 4.1248 | 4.121 | 4.143 | **1** |

이 이미지 100패치가 **29개 서로 다른 cell**로 갈리며, 최빈 cell이 24패치다.

## 이미지 규격

- `figsize=(14.5, 20.5)`, `dpi=170`, `bbox_inches="tight"`
- GridSpec 8×4 — 행0 상단 전폭(1단계), 행1 열 제목, 행2~6 특징 5종, 행7 하단 전폭(4단계). 열0은 좌여백
- 행 색: local_variance `#2563eb` / edge_density `#16a34a` / texture_entropy `#ea580c` / frequency_energy `#7c3aed` / orientation_consistency `#dc2626`
- 격자선 `#22d3ee`, 대상 패치 테두리 `#f59e0b`

> 현재 크기는 분석·설명 가독성 우선. 논문 삽입 시 별도 축소 필요.

## 이 그림이 드러내는 것

1. **판정 단위가 패치다.** 이미지 1장(100패치)이 29개 cell로 갈린다. 이미지 단위 판정이라면 1이어야 한다. 이것이 `ctx_prior`가 "이미지 안 어디에 붙일지"를 판단할 수 있는 근거다.
2. **`texture_entropy`·`orientation_consistency`의 tertile 폭이 극단적으로 좁다.** 각각 `3.194~3.251`(폭 0.057), `4.121~4.143`(폭 **0.022**)다. 분포가 상한에 몰려 세 tertile 중 두 개가 사실상 한 점에 붙어 있어 이 두 자리는 변별력이 매우 낮다. Table 2에서 texture entropy가 데이터셋 간 2.58~3.22로만 변하는 것과 같은 현상이다.
3. **`c`는 5차원 공간의 직육면체 한 칸이며 매우 넓다.** 전체 공간 3⁵=243, severstal good 관측 208 cell, 590,200 패치 → 평균 약 **2,838 patches/cell**.

## Caption (초안)

**Figure 3.2.2-4.** 64픽셀 패치가 context cell로 환산되는 과정. 각 행은 하나의 context feature에 대해 산출 맵, 식이 축약하는 중간 분포, 데이터셋 tertile 기준 bin 배정을 보이고, 하단에서 다섯 자리를 이어 cell key를 만든다.

## 논문 내 위치

- **섹션**: 3.2.2 (Morphology and Context Modeling)
- **배치**: Figure 3.2.2-2(context feature 분포 + tertile 경계) 직후 — 분포·경계를 본 뒤 단일 패치의 환산 과정을 보이는 순서
- **근거 문서**: `../../../aroma_core_compatibility_model_20260729.md` §3
