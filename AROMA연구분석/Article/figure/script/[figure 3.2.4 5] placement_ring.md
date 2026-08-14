# Figure 3.2.4-5 — Site resolution: ring context, source vs placement (spec)

> 2026-08-14 2차 개정: 3패널(A tgt 전면 투영 / B best·worst / C h_s 막대)판을
> **단순 대조판**으로 대체 — 사용자 요구: `_retired/[figure 3.2.4 2] placement_footprint.png`
> 수준의 단순함 + **원본 결함 이미지의 ring 특징 vs clean 배정 배경의 ring 특징 육안 대조**.
> 당일 3차 수렴: ① 대표 표본 규칙에 변 지배 금지(SIDE_MAX 60%) 추가(-4와 공유),
> ② B에 best/mid/worst 3자리 표시, ③ C 분포 대조 패널 추가(실측 c 값 vs tgt[k]).
> 표본 연속(-4의 대표 crop·rank-1 배경 승계) 유지.

## 구성 (데이터셋당 3패널 세로 적층)

```
A: 원본 결함 이미지 — 실제 결함 bbox(빨강 점선)를 덮는 타일 사각형의 8이웃 ring 타일만
   tgt[k] 질량으로 tint(viridis α=.5, 값 인쇄). footprint 실선·ring 점선(빨강).
   제목에 ∩(h_ring, tgt[k]) 수치.
B: 배정 배경 — best s*(초록, ring tint + footprint 실선·ring 점선) / mid(주황) / worst(빨강)
   박스와 score 라벨. 라벨 앵커 분리(best 좌상 / mid 우상 / worst 좌하) — 겹침 자리 충돌 회피.
   중복 자리(n<3 등)는 생략. 제목에 자리 수·site_score(best).
C: 분포 대조 막대 (상위 12셀) — target tgt[k](회색) vs 원본 ring 실측 h_ring(적색) vs
   확정 자리 h_s*(녹색). 제목에 두 ∩ 수치 병기 — "실측 c 분포와 tgt[k]의 유사성"의 정량 패널.
```

핵심 독법: A·B의 ring tint(동일 vmax=max tgt[k])와 C의 막대가 같은 사실을 두 방식으로 —
∩ 목적함수가 보상하는 "실결함 주변 문맥 재현"이 육안(색)과 분포(막대)로 확인됨.
무채색 ring 타일 = 프로파일링 제외분(결함 겹침·void·미관측) — A에서 자연 발생.

## 대표 표본 규칙 (-4와 공유 — 표본 연속의 전제)

면적비 0.30 근접 + **어느 변도 원본의 60% 초과 지배 금지**(`|ratio−0.30| + 4·side_over` 최소화)
+ 최소변 48px. 배정 배경 = bg_score 단순합 rank 1.

## 폴백 케이스 처리 (정직성)

rank-1 배정 배경이 admissible 자리 0개면 자리가 존재하는 최상위 배경으로 내려가되 제목에
`[rank-1 background admits no valid position (runtime fallback) — shown: rank N]` 명기.
(구 표본에서 mtd가 해당했으나, 변 지배 금지 규칙의 신표본에서는 5셋 모두 rank-1에서 해소 —
로직은 유지.)

## 데이터·정합 (운영 함수 import — 재구현 없음)

- `clean_bg_selection`: load_inputs / _derive_void_floors / valid_bg_pool / _image_hist /
  _class_bg_hist / _hist_intersection / _scale_to_fit / _image_dim / _tile_grid / _ring_keys /
  _target_by_cluster / _effective_wh / _best_ring_site / _parse_bbox
- 원본 ring: bbox 픽셀 좌표 → 타일 footprint(si0=x//64, bwS=⌈(x+w)/64⌉−si0) → `_ring_keys`,
  결함 이미지 타일 격자(defect_rows)에서 셀 조회 — 결함 겹침 타일은 격자에 없어 자동 무채색
- **argmax 정합 assert**: 그림의 열거 결과 최고점이 `CBS._best_ring_site` 반환 score와 일치
- 실측 (2026-08-14 신표본): sev 11자리 site .213/src .088 · aitex 6 .038/.022 ·
  kolektor 51 .194/.129 · mtd 10 .164/.035 · leather 108 .301/.279

## 출력·캡션

`../image/[figure 3.2.4 5 <ds>] placement_ring.png` (5파일, dpi=300, 12.8×11.4in 3×1).
prose 단일 라벨 Figure 3.2.4-5. 캡션 정본 = section3_2.txt.
