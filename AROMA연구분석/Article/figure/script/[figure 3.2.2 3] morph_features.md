# Figure 3.2.2-3 — 형태 특징 6종의 검사 영역 (spec)

**스크립트**: `[figure 3.2.2 3] morph_features.py`
**출력**: `../image/[figure 3.2.2 3] morph_features.png`
**데이터 루트**: `D:/project/aroma_dataset` (환경변수 `AROMA_DATASET_ROOT`로 재지정)
**리포 루트**: `D:/project/aroma` (환경변수 `AROMA_REPO`)

> ⚠ **라벨이 한글이다.** figure 내용을 숙지한 뒤 영문으로 교체 예정(사용자, 2026-07-31). 교체 시 `Malgun Gothic`/`Gulim` 폰트 의존도 함께 제거되어 Colab 재현성이 확보된다.

## 목적

§3.2.2의 morphology 군집 `k`가 **어떤 입력 벡터로 만들어지는지**를 보인다. `MORPH_FEATURES` 6종이 각각 결함 영역의 **어떤 기하 구성물**을 재는지 패널로 표시하고, 그 값이 데이터셋 분포에서 어디에 놓이며 GMM에 어떤 min-max 값으로 들어가는지까지 추적한다.

CASDA `ex_casda/figure/[figure2] defect_type.png` 형식 준용 — 좌여백=범주명·수식·검사대상·값, 열 제목 박스, 하단 legend + 각주.

## 구성

**행 = `MORPH_FEATURES` 6종** (linearity, solidity, extent, aspect_ratio, eccentricity, circularity). 각 행 좌여백에 색 바 + 특징명 + 수식 + 검사 대상 + 산출값.

**상단 전폭 패널** — 측정 대상 불일치. 노란 실선 = 최대 blob(6종이 재는 유일한 대상), 회색 점선 = 나머지 성분(측정 제외), 초록 파선 = `defect_bbox`(전 성분 = 합성 crop 범위).

**열 4개**

| 열 | 내용 |
|---|---|
| Inspected Region | 특징이 쓰는 기하 구성물을 crop 위에 표시. 마스크는 반투명 적색 |
| Isolated Construct | 측정 대상만 분리 — 회색=최대 blob, 초록=hull이 덮되 결함 아닌 영역, 점선=측정 제외 성분 |
| Computation | 식에 들어가는 두 양의 대비 막대 |
| Dataset Position | severstal 전체 분포 + 이 샘플 위치(분위) + GMM 입력 min-max 값 + 분산 점유율 |

특징별 열1 구성물:

| 특징 | 표시 |
|---|---|
| linearity | 모멘트 등가 타원 + major/minor 축선 |
| solidity | convex hull 윤곽 + hull이 덮되 결함 아닌 영역(회색) |
| extent | 최대 blob **자신의** bbox (`defect_bbox` 아님) |
| aspect_ratio | 동일 등가 타원 |
| eccentricity | 동일 등가 타원 + 초점(×) |
| circularity | 경계 윤곽선 |

## 데이터 출처 (실측)

- `profiling/profiling/severstal/morphology_features.csv` — 3,620행. medoid 선정 + 열4 분포
- `severstal/masks/<class>/<stem>.png` — 결함 마스크(per-class, `_find_mask_path`가 우선 조회하는 경로)
- `severstal/test/<class>/<stem>.jpg` — 원본. severstal은 이미지 jpg / 마스크 png

**값 산출은 운영 함수 import** (재구현 금지): `utils.defect_characterization.DefectCharacterizer.analyze_defect_region` + `skimage.regionprops`(eccentricity) + `4*pi*area/perimeter^2`(circularity, `distribution_profiling.py:301-309`과 동일 식).

## 대표 샘플

**medoid** — min-max 표준화 6차원에서 전체 centroid에 가장 가까운 샘플. 자동 선정이며 하드코딩 없음.

실측 결과: `class3 / 5377dbac1`, dist=0.0592, 연결성분 5개, 최대 blob area=103,133.

| 특징 | 값 | min-max |
|---|---|---|
| linearity | 0.8483 | 0.844 |
| solidity | 0.9103 | 0.891 |
| extent | 0.7246 | 0.716 |
| aspect_ratio | 2.5671 | **0.021** |
| eccentricity | 0.9210 | 0.905 |
| circularity | 0.3929 | 0.417 |

`defect_bbox=(4,0,1563,256)` vs 최대blob bbox=`(0,299,238,897)` — 범위가 크게 다르다.

## 이미지 규격

- `figsize=(15.5, 21.5)`, `dpi=170`, `bbox_inches="tight"`
- GridSpec 8×5 — 행0 상단 전폭, 행1 열 제목, 행2~7 특징 6종. 열0은 좌여백 라벨
- 행 색: linearity `#2563eb` / solidity `#16a34a` / extent `#ea580c` / aspect_ratio `#7c3aed` / eccentricity `#0d9488` / circularity `#dc2626`

> 현재 크기는 **분석·설명 가독성 우선**이다. 논문 삽입 시 별도 축소 필요(사용자 결정, 2026-07-31).

## 이 그림이 드러내는 것

1. **linearity·aspect_ratio·eccentricity 세 행의 타원이 동일하다.** `Computation` 열도 같은 `major/minor`를 쓴다 → 독립 축이 아니다(`linearity = 1 − AR⁻²`, `eccentricity = √linearity`, Spearman = 1.000).
2. **`aspect_ratio`의 min-max 붕괴가 수치로 노출된다.** 이 샘플 AR=2.567은 분위 30%인데 GMM 입력은 **0.021**, 분산 점유는 **3.5%**로 6종 최저다. severstal AR은 `p99=26.04` / `max=74.74`로 극단값 1개가 축을 압축한다. 반면 `circularity`가 **31.4%**로 실제 지배 축이다.
3. **6종 전부 최대 blob 1개만 기술하는데 합성 crop은 전 성분을 담는다** — 군집이 기술하는 대상과 붙여지는 대상의 괴리. severstal class3은 다중 성분이 73.5%다.

## Caption (초안)

**Figure 3.2.2-3.** 형태 특징 6종이 각각 재는 기하 구성물. 상단은 측정 대상 불일치(최대 blob vs `defect_bbox`), 각 행은 한 특징의 검사 영역·중간량·데이터셋 내 위치. 대표 샘플은 severstal의 min-max 6D medoid.

## 논문 내 위치

- **섹션**: 3.2.2 (Morphology and Context Modeling)
- **배치**: Figure 3.2.2-1(군집 산점도) 앞 또는 뒤 — 군집 입력 벡터의 정의를 먼저 보이는 편이 읽기 순서에 맞다
- **근거 문서**: `../../../aroma_core_compatibility_model_20260729.md` §2
