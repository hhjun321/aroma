# Figure 3.2.4-1 — ROI Selection & Compatibility-Aware Placement Flow (spec)

> 2026-08-14 개정판. 구 `[figure 3.2.5 3]`에서 §3.2.4 도입 흐름도로 이동·재생성.
> 구판(compat gate scan–rank–place · best_mean ≥ τ · subtype warp 곁가지)은 ring 개정(2026-08-04)
> 및 §3.2.4 본문 간소화(2026-08-14, 3-cue 요약판)로 폐기 — git 이력 참조.

## 목적

§3.2.4의 세 순차 결정 — (1) defect crop selection, (2) background assignment, (3) site resolution —
을 하나의 세로 흐름도로 제시한다. 본문 서술 **앞**에 배치되는 도입 그림이므로, 본문이 정의하는
기호(ROI_score, bg_score, h_s, tgt[k])만 사용하고 본문에 없는 요소(τ, lift 가중, subtype warp)는
그리지 않는다.

## 데이터 출처

- 없음 — **전 박스 기호(symbolic)만 표기** (2026-08-22 개정: stage-1 Severstal 실측 예시값 제거).
  구판의 실측값 출처(참고): `aroma_dataset/profiling/profiling/severstal/` — linearity 0.961 /
  solidity 0.882 / AR 5.09, k=1, P(k)=0.24, peak cell `0_0_0_1_0`=1.00.

## 구성 (세로 흐름도, 기호만)

```
Defect crop (morphology features) ─┬→ GMM cluster k, morph_prior P(k)
                                   └→ source context cell c, ctx_prior = matrix_symmetric(k, c)
→ ROI_score = ctx_prior + morph_prior
→ Rank all → keep Top-K sources
→ bg_score = src_fit + class_fit + size_fit → Assign highest-scoring normal image
→ site_score = ∩(h_s, tgt[k])
→ argmax → final paste position (bbox)
```

스타일(2026-08-21~22 개정): **흑백 단일** — 색 구분·범례(blue/grey/stage-1 설명 3줄)·★ 글리프 제거,
좌상단 스테이지 라벨(1./2./3. 이탤릭) 제거, 박스별 예시 수치 제거(기호 전용).

## 본문 정합 계약

- 박스 라벨의 수식·기호는 §3.2.4 본문과 문자 단위 일치 (bg_score 3항 합산, tgt[k], ∩)
- "valid position"은 본문의 void/unobserved 사전 배제 문장에 대응 — 그림에선 한 단어로만
- 캡션은 section3_2.txt의 **Figure 3.2.4-1.** 항목이 정본

## 축·해상도

flowchart(축 없음), 6.6 × 9.6 in, dpi=300. 실행: `python "[figure 3.2.4 1] roi_selection_flow.py"`

## 비고

- `_mod` 쌍(`roi_selection_flow_mod.md/.py`)은 구판 기준의 이론판 변형안 — **구식**(τ 게이트 포함).
  사용 금지, 재작성 시 본 스펙 기준.
