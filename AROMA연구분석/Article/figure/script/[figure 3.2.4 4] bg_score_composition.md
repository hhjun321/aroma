# Figure 3.2.4-4 — Background assignment: cue top-3s and the final top-3 (spec)

> 2026-08-14 신규 → 당일 2차 수렴(최종안): 데이터셋별 파일, src/class top-3 + bg_score top-3,
> size_fit 패널 제거, 대표 crop = 원본의 ≈30%.

## 목적

배경 배정의 두 히스토그램 적합도가 **각각 어떤 후보를 올리는지**, 그리고 **최종 bg_score가
이를 어떻게 종합하는지**를 실물 이미지 top-3 대조로 보여준다. cue 간 겹침(예: bg 1위 =
src 2위)이 그림 안에서 자연히 드러나는 것이 정보값.

## 구성 (데이터셋당 4행 3열)

```
행0: [원본 이미지(bbox 빨강)] [defect crop (크기·면적비 라벨)] [공백]
행1: src_fit   top-3  (파랑)   — 라벨: src_fit 값 · bg_score
행2: class_fit top-3  (주황)   — 라벨: cls_fit 값 · bg_score
행3: bg_score  top-3  (초록)   — rank1 = 배정(★), 3항 분해 병기
```

- **size_fit 패널 없음** (사용자 결정) — 단 bg_score 합산은 본문 수식(src+cls+siz) 그대로,
  행3 분해 라벨에 siz 값 병기로 수식 정합 유지.

## 대표 ROI 선정 (결정적)

**crop/원본 면적비가 0.30에 가장 가까운 ROI** (최소변 48px, 동률 시 roi_idx 최소).
근거: crop이 원본 대부분을 차지하면 원본-후보 육안 대조가 무의미해짐 (사용자 요구 "약 30% 내외").

**4차 수렴 (2026-08-14, -5와 규칙 공유)**: 어느 변도 원본의 **60%(SIDE_MAX) 초과 지배 금지** —
초과분은 `|ratio−0.30| + 4·side_over` 페널티로 최소화. severstal 전폭 스트립형 crop(폭 지배)이
표본으로 뽑히는 것을 배제. 실측 비율(개정 후): severstal 0.31 · aitex 0.24 · mtd 0.13 ·
leather 0.14 · kolektor 0.07. **-5(placement_ring)와 동일 규칙 — 표본 연속의 전제.**

## 데이터 (실측, 운영 함수 import — 재구현 없음)

`clean_bg_selection`: load_inputs / _derive_void_floors / valid_bg_pool / _image_hist /
_class_bg_hist / _hist_intersection / _scale_to_fit / _image_dim.
순위는 §3.2.4 간소판 수식 — 본문과 문자 단위 일치 (실 파이프라인 lift 가중·k_fit 미표기).

## 출력·캡션

`../image/[figure 3.2.4 4 <ds>] bg_score_composition.png` (5파일, dpi=300, 세로 4행).
prose 단일 라벨 Figure 3.2.4-4 (다패널 규칙). 캡션 정본 = section3_2.txt.

## 3차 수렴 (2026-08-14) — 격자 색 투영 추가

은퇴 그림 `_retired/[figure 3.2.4 2] placement_footprint.png` A패널 형식 차용 (사용자 제안):
원본·후보 이미지의 64px 타일을 **행의 참조 히스토그램에서의 그 셀 질량**으로 반투명(viridis, α=0.45)
채색 — 행1 = p_src, 행2 = p_cls, vmax = 해당 분포 최대 질량(행 그룹 내 색 스케일 공유).

- **효과**: 히스토그램 교집합이 "원본과 후보가 같은 색 혼합으로 물드는가"로 육안 번역됨
- 무채색 타일 = void·미관측·결함겹침 (CSV에 행 없음 → NaN → 투명) — 원본에서 결함 자리가
  뚫려 보이는 것 자체가 "P_def는 둘레 통계"의 시각 증거
- **행3(bg_score)은 무채색 유지** — "렌즈 뷰(1·2행) vs 실제 결과물(3행)" 대비 의도
- 격자 데이터 = `CBS._tile_grid` (운영 함수, void 제외 격자 그대로)
