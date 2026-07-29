# Figure 3.2.3-1 — Defect morphology-feature distributions (spec)

**스크립트**: `[figure 3.2.3 1] morphology_distribution.py`
**출력**: `../image/[figure 3.2.3 1 <ds>] morphology_distribution.png` (5종)
**데이터 루트**: `D:/project/aroma_dataset` (환경변수 `AROMA_DATASET_ROOT`로 재지정)

## 목적

§3.2.3의 Table 4 subtype 임계가 형태학 특징 분포의 어디에 놓이는지 데이터로 보인다. 배경 쪽 `[figure 3.2.2 2] context_distribution`(context feature + P33/P66 경계)의 결함 쪽 대응물이다.

## 대표셋

aitex / kolektor / severstal / mtd / mvtec_leather — 데이터셋당 PNG 1장.

## 데이터 출처 (실측)

- `profiling/profiling/<ds>/morphology_features.csv` — 6개 형태학 특징 (linearity, solidity, extent, aspect_ratio, eccentricity, circularity). 결함 인스턴스 전량.
- 임계값은 **데이터셋별 P33/P66 tertile을 스크립트가 직접 산출**(Table 4b). `roi_selection.py::_subtype_percentiles`와 동일 로직 — 동질성 폴백 포함.

| 특징 | Table 4 임계 |
|---|---|
| aspect_ratio | P33 / P66 (데이터셋별) |
| solidity | P33 / P66 (데이터셋별, 퇴화 시 0.7 / 0.9 폴백) |
| linearity, eccentricity | 없음 — `aspect_ratio`의 결정론적 재매개화 |
| extent, circularity | 없음 (Table 4가 사용하지 않음) |

산출값 (실측):

| Dataset | AR P33 | AR P66 | Sol P33 | Sol P66 |
|---|---|---|---|---|
| AITeX | 3.13 | 16.43 | 0.648 | 0.900 |
| Kolektor | 4.39 | 5.73 | 0.700 * | 0.900 * |
| Severstal | 2.80 | 7.80 | 0.900 | 0.965 |
| MTD | 1.70 | 3.62 | 0.846 | 0.934 |
| MVTec Leather | 1.58 | 4.03 | 0.874 | 0.954 |

\* Kolektor solidity는 중간 tertile 폭이 sd의 3.4%(다른 4종 58~118%)로 동질성 가드에 걸려 fixed 폴백. aspect_ratio는 데이터 유도 유지.

**linearity·eccentricity 중복 근거** (실측 2026-07-29, 4,504건 전량): `linearity ≡ 1 − AR⁻²`(max err 4.4e-15), `eccentricity ≡ √linearity`(max err 3.4e-15), Spearman(lin, AR)=1.000. 동일 2차 중심모멘트 유래이므로 독립축이 아니다 → Table 4 임계에서 제외.

**임계 출처 확정 이력**: 구버전 논문 Table 4 값(lin 0.9, AR∈[2,5] 등)은 실제 산출물과 67.8~96.2%만 일치해 기각. `classify_defect_subtype`(fixed)가 `roi_selected.json`의 `defect_subtype`와 **5종 100% 일치**로 확정되었고(aitex 200 / severstal 890 / mtd 200 / kolektor 52 / leather 90), 그 fixed 규칙을 percentile로 전환한 것이 현재 설계다. 경합 후보 `distribution_profiling.py::_auto_label`은 76.9~92.2%(클러스터 centroid용, 대상 아님).

## 구성

- 2행 × 3열, `figsize=(13, 7)`, `dpi=300`, `bbox_inches="tight"`.
- 패널 순서 = `MORPH_FEATURES` 순서 (linearity, solidity, extent / aspect_ratio, eccentricity, circularity).
- 히스토그램 40 bins, 색 `#4c78a8`. 표시 구간은 **1–99 percentile 클리핑**(형제 스크립트와 동일 규약) — 극단 이상치가 축을 지배하는 것을 막는다.
- **`aspect_ratio`만 log-x + logspace bins.** 우편향이 심해 선형 bins에서는 최저 구간 1개 막대가 모드를 독점하고 임계 2.0이 그 막대 안에 묻힌다. log 축에서만 임계 위치가 판독 가능하다. (Figure 3.2.2-1이 `log10(aspect ratio)`를 쓰는 것과 동일한 이유.)
- Table 4 임계는 빨간 파선 + 값 라벨. 1–99 percentile 표시 구간을 벗어나는 임계는 **그리지 않는다**.
- Table 4가 쓰지 않는 3개 특징은 패널 제목에 `no Table 4 threshold` 명시.
- 그림 하단 `fig.text`에 파선의 의미·출처·표시 규약 주석.

## 이력 — 기존 커밋본과의 차이 (중요)

교체 전 `[figure 3.2.3 1 <ds>] morphology_distribution.png`(3508×2277)은 **생성 스크립트가 리포에 없었다**. 추적 결과:

- `scripts/distribution_profiling.py::_plot_figures`(:1291-1320)가 유사한 그림을 만들지만 산출물은 `figures/morphology_histograms.png`, `figsize=(15,8) dpi=100`(=1500×800)이라 커밋본과 크기·파일명이 다르다.
- 그리고 그 코드의 빨간 파선은 **Table 4 임계가 아니라** `distribution_analysis.json`의 `boundaries`(valley/percentile/GMM 이산화 경계)다(`:1311-1313`).
- 커밋본 PNG의 선 위치를 픽셀 측정해 현재 프로파일링 데이터로 재현한 것과 비교하면 불일치(정규화 x: 커밋본 `[0.222, 0.248, 0.299]` vs 재현 `[0.268, 0.306, 0.315]`).

즉 커밋본은 **캡션이 주장하는 Table 4 경계가 아닌 다른 경계를 그리고 있었다.** 본 스크립트는 캡션과 정합하도록 Table 4 임계를 그린다.

## ⚠ 실측 경고 — 본문 주장이 aspect_ratio에서 성립하지 않는다

§3.2.3 본문은 "the Table 4 decision boundaries fall between populated regions of the feature space rather than cutting through dense modes"라고 쓴다. 5종 실측 결과:

- **linearity (0.6 / 0.85)** — 성립. 두 임계 모두 희소한 좌측 꼬리·상승 구간에 놓이고, 지배적 모드(≈1.0)를 가르지 않는다.
- **solidity (0.7 / 0.9)** — 대체로 성립. 단 0.9는 피크(≈0.96) 직전 상승 구간이라 여유가 크지 않다(mtd 0.66·leather 0.40 상대밀도).
- **aspect_ratio (2.0 / 5.0)** — **성립하지 않는다.** log 축에서 두 임계가 넓은 populated plateau **내부**에 놓인다(severstal 해당 구간 count ≈ 90–120). 선형 bins 측정에서는 상대밀도 1.00(aitex·severstal AR=2.0, kolektor AR=5.0)으로 최빈 bin을 정통으로 가른다.

따라서 이 그림을 본문 문장의 근거로 인용하면 **aspect_ratio 패널이 반례가 된다.** 본문을 (a) linearity·solidity에 한정하거나 (b) "aspect_ratio는 연속 스펙트럼이라 임계가 밀집 구간을 지날 수밖에 없다"는 취지로 완화하는 정정이 필요하다. 본 스펙은 사실만 기록하고 본문은 손대지 않았다.

## Caption (초안)

**Figure 3.2.3-1.** Per-dataset distributions of the six profiled defect-morphology features, with the Table 4 subtype thresholds overlaid (dashed) on the three features Table 4 constrains. Aspect ratio is shown on a logarithmic axis.

## 논문 내 위치

- **섹션**: 3.2.3 (Background Categories and Defect Subtypes)
- **배치**: Table 4 직후.
