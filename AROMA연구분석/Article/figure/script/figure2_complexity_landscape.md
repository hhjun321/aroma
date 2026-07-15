# Figure 2 — Complexity Landscape (MCI × CCI)

## 목적 (연결 section)

§3.1 핵심 주장을 시각화한다:
> "The complexity indices span a broad range (MCI 0.33–0.60, CCI 0.20–0.44), confirming that the roster exercises the policy-selection mechanism across distinct complexity regimes."

5개 데이터셋이 형태(MCI)·맥락(CCI) 복잡도 평면에서 **서로 다른 영역에 분포**함을 보이고, 각 데이터셋에서 프레임워크가 **자동 선택한 policy**를 함께 표시한다.

## 데이터 출처 (실측)

`D:\project\aroma_dataset\complexity\<dataset>\complexity_report.json` 의 `mci`, `cci`, `morphology_policy`, `context_policy`.

| Dataset | MCI | CCI | Granularity | Morph policy | Ctx policy |
|---|---|---|---|---|---|
| AITeX | 0.392 | 0.440 | single | otsu | gmm |
| Kolektor | 0.328 | 0.224 | single | gmm | gmm |
| Severstal | 0.455 | 0.306 | multi | otsu | percentile |
| MTD | 0.561 | 0.246 | multi | otsu | percentile |
| MVTec Leather | 0.595 | 0.200 | multi | otsu | gmm |

수치는 스크립트가 JSON에서 직접 로드한다 (하드코딩 금지, 위 표는 참고용).

## 플롯 구성

- **형식**: 단일 2D scatter. x축 = MCI (형태 복잡도), y축 = CCI (맥락 복잡도).
- **축 범위**: x ∈ [0.30, 0.62], y ∈ [0.18, 0.46] (여유 padding). 눈금 0.05 간격.
- **점 표현**:
  - marker 모양으로 **class granularity** 구분: single-class = 사각형(square), multi-class = 원(circle).
  - marker 크기: 고정(적당히 큰 값, ~180).
  - 색상: 데이터셋별 구분되는 정성 팔레트(색맹 안전). 범례에서 데이터셋명 표기.
- **라벨**: 각 점 옆에 데이터셋명 + 선택된 policy를 부제로 표기.
  - 예: `AITeX\n(otsu / gmm)` — 위=morph, 아래=ctx policy. 겹치면 offset·annotate로 회피.
- **가이드**: single vs multi 구분을 범례로. policy 조합은 각 점 라벨로 전달(별도 색 인코딩 불필요).
- **강조**: §3.1이 언급하는 극단값 표시
  - AITeX = 최고 CCI (0.44), MVTec Leather = 최저 CCI(0.20)·최고 MCI(0.59), Kolektor = 최저 MCI(0.33).
  - 축 라벨 아래 얇은 화살표/텍스트로 "higher morphological complexity →", "higher contextual complexity ↑" 방향 주석.

## 스타일

- figure_patterns.md 규약 준수(존재 시). 기본: 300 dpi, 흰 배경, serif 라벨, 격자 옅게.
- 크기: 약 6×5 in (단일 컬럼~1.5컬럼 폭).

## 저장

- 스크립트: `figure/script/figure2_complexity_landscape.py`
- 출력: `figure/image/[figure2] complexity_landscape.png`

## 캡션 (초안)

**Figure 2.** Morphological (MCI) versus contextual (CCI) complexity of the five evaluation datasets. The roster spans distinct complexity regimes (MCI 0.33–0.60, CCI 0.20–0.44); AITeX occupies the high-CCI corner while MVTec Leather sits at high MCI / low CCI. Marker shape encodes class granularity (square = single-class, circle = multi-class); the policy pair below each label (morphology / context) is the one AROMA selected automatically for that dataset, illustrating per-dataset data-driven policy selection without manual tuning.
