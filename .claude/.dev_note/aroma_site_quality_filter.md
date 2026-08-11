# 자리(site) 단위 quality filter — ring 배치 2차 게이트

## (사용할 skills: feature-dev)

## (성격: 기능 추가 — **구현·로컬 검증 완료 (2026-08-11).** 기본 OFF, 승격은 §미확정 3)

### 검증 결과 (2026-08-11 로컬)

- [x] **OFF byte-identical**: kolektor 재실행, 기존 `_ring` baseline과 좌표·score 0/200 diff. 신규 필드는 전부 None
- [x] **ON — kolektor**: floor=0.4556, 자리 20,454개 배제, 폴백 84→88 (2.6→2.8%, 시뮬 예측 +0.1p 부합). 선택 자리 quality min 0.4557 ≥ floor. 정렬 위반 0
- [x] **ON — leather**: floor=0.4616, 자리 79,789개 배제, 폴백 2→2 (0.1% 불변, 시뮬 +0.0p 부합). min 0.4622 ≥ floor. 미해석 배경 0
- [ ] Colab step5 전파 확인 (annotation `site_quality`) — 재합성 시

## 개요

ring_sgm이 추천하는 admissible 자리들에 **자리 영역(crop-sized 패치) 전체의 quality score 분위 필터**를 2차 게이트로 추가한다. 순서는 사용자 확정: **① void 게이트(타일, admissibility — 현행 불변) → ② 자리 quality 분위 필터 → ③ 생존 자리 중 ring argmax.**

근거 실측 (2026-08-11 로컬 시뮬레이션, [[project-quality-filter-site-unit]] 메모리):

| 판정 | 근거 |
|---|---|
| 절대 임계 0.7 기각 | leather 자리 quality max 0.67 — 100% 포화. 분위 컷은 스케일 불변 |
| 타일 단위 배제 기각 | severstal 폴백 +20.2p 폭발 (세로 관통 footprint의 비선형 전멸) |
| **자리 단위 P15 채택** | 5종 폴백 델타 +0.0~+3.5p — 전부 한계 내 (severstal +0.1p). 자리생존 med 89~100%로 필터는 나쁜 자리에 집중 |
| tie 문제 없음 | 고유 점수값 충분 (leather 27,442/54,991), 분위 경계 동점 0~1타일 |

quality score의 고유 기여 = **밝기 축** — void 게이트(분산·엣지)가 원리적으로 못 보는 밝기 극단(leather 실측: P15 거부 타일의 100%가 밝기 극단 vs 통과 34.6%)을 잡는다. [[aroma_adjacent_context_bg_selection]] 위험 5(밝은 침전물 사각)의 부분 보완.

## 범위

- **포함**: `clean_bg_selection.py` 오프라인 자리 필터 + provenance 기록 + `generate_defects.py` annotation 전파 + 가이드 갱신
- **제외 ①**: 런타임 `_is_clean_background` 부활 — dtype만 고치면 절대 임계(0.5/0.7)가 미캘리브레이션 상태로 활성화돼 leather 99.6% 거부 등 역효과. 임계 재설계와 함께 별건 ([[aroma_cleanbg_gate_cv2_dtype_failopen]] §3 결정 4개)
- **제외 ②**: exp4v2 수정 — 자리 필터는 폴백률을 사실상 안 바꾸므로(≤+3.5p) parity·ring-first 로직 무변경
- **제외 ③**: 학습셋 레벨 quality 필터 — 수량 parity 파괴 + 큐레이션 오염. 금지 확정

## 수정 내용

### 1. `scripts/aroma/clean_bg_selection.py`

**(a) CLI 3종 신규**
- `--site_quality_filter` (flag, 기본 OFF)
- `--site_quality_pct` (float, 기본 15.0 — `void_floor_pct`와 통일)
- `--image_dir` (str) — good 이미지 디렉터리. 필터 ON일 때 필수(픽셀 필요). 스크립트는 현재 프로파일 CSV만 읽으므로 신규
- 필터 ON + `--site_selection ring` 아니면 에러 종료 (자리 개념이 ring 전용)

**(b) `_site_quality(gray_img, x, y, ew, eh)` 신규** — CASDA 4-성분 수식 + `cv2.Laplacian(g, cv2.CV_32F)` (dtype 지뢰 회피). generate_defects 미import (독립 구현)

**(c) 필터 ON 경로 = 2-phase (OFF 경로 무접촉)**
- OFF: 기존 인라인 `_positions_for` 그대로 — **byte-identical 보장이 최우선**
- ON: 메인 루프는 (record, pool_ids, bbox) 수집만 → 루프 후:
  1. `(nid, bw, bh)` 캐시로 admissible 자리 + 자리 quality 전수 산출
  2. `q_floor = percentile(전체 자리 quality, site_quality_pct)`
  3. 레코드별: quality ≥ floor 생존 자리 중 ring argmax → `position`/`topk_positions`/`topk_site_scores` + **`topk_site_quality`**(정렬)/`site_quality`(top-1) 기록
- 이미지 미해석 배경(stem 매핑 실패): 해당 배경만 필터 미적용 + 카운터 로그 (죽지 않음)
- `derived`에 `site_quality_floor` · `site_quality_pct` · `site_quality_filtered`(배제 자리 수) · `site_quality_unresolved_bg`

**(d) 생존 자리 0개** → `position=None` → 기존 폴백 (`position_source="fallback"` 자동)

### 2. `scripts/aroma/generate_defects.py` — annotation 전파

`_cbg_pairs` triple → quad (`topk_site_quality` 추가, 결측 → None). annotation에 `"site_quality"` 필드 — `site_score` 전파와 동일 패턴. 하위호환: 구 json 필드 없음 → None.

### 3. 가이드

- `step3_5_execute.md` — 옵션 3종 + "필터 ON 시 §2-3 검증 셀에 floor·filtered 확인" 추가
- `_SPEC.md` — 한 줄 (기본 OFF, 발효 시 step3.5→step5 재실행)

## 수정 대상 파일

- `scripts/aroma/clean_bg_selection.py`
- `scripts/aroma/generate_defects.py`
- `AROMA연구분석/colab_execute_new/step3_5_execute.md`, `_SPEC.md`

## 테스트 (CLAUDE.md: 새 테스트 코드·pytest 금지 — 실측)

로컬 (`D:\project\aroma_dataset`):

- [ ] **OFF byte-identical**: kolektor `--site_selection ring` (필터 미지정) 재실행 → 기존 `_ring` baseline과 `position`·`topk_positions`·`score` 완전 일치
- [ ] **ON 동작**: kolektor + leather 필터 ON 실행 →
  - `derived.site_quality_floor` 기록, `site_quality_filtered > 0`
  - 선택된 모든 `position`의 `site_quality ≥ floor`
  - 폴백률 델타가 시뮬레이션 예측(kolektor +0.1p, leather +0.0p) 수준
  - `topk_site_quality` 길이 == `topk_positions` 길이
- [ ] **전파**: (Colab step5 시) annotation `site_quality` 기록, fallback 항목은 None

## 미확정

| # | 항목 | 상태 |
|---|---|---|
| 1 | 런타임 게이트 부활 (shadow-mode 계측 포함) | 별건 — 임계 재설계와 함께 |
| 2 | 논문 §3.2.6 문안 | 수정안 7-B와 병합, frozen table 정책 하에서 별도 결정 |
| 3 | 필터 ON을 기본 운용으로 승격할지 | 5종 실코드 재측정 + (원하면) exp4v2 ablation arm(`aroma_qf`) 후 결정 |

## 부록 — score **상위 선별**(synth_final / aroma_sel)은 기각 (2026-08-11)

본 노트의 필터(하위 P15 배제)와 혼동 금지. exp4v2 직전 score 상위로 학습셋을 큐레이션하는 방안은 전수 분포 분석(소비 샘플 5,336개)까지 수행한 뒤 **사용자 결정으로 기각**했다:

- **전형성 함정**: P1↔다운스트림 역상관 실측(Spearman −1) — 배치 점수 상위 = 가장 전형적 표본 집중 위험
- **클래스 skew 실측**: site_score 클래스 격차 mtd crack 1.8배 · aitex linear_scratch 2.7배 — 전역 top이면 배분 파괴 (mtd는 crack 증량이 map50 하락과 동행한 실측 있음)
- **다양성 붕괴 선례**: leather 배경 16장 (top-1 소비 구조)
- **severstal 선별 여지 1.16×** — 재합성 없이 무의미
- **정상관 근거 부재**: 어느 score 축도 mAP와의 정상관이 입증된 바 없음
- 부수: score 순위는 결정적이라 seed별 표본 분산이 소멸 — n=3 해석 변질

분포 분석 자체의 산출(축 직교성 Spearman ≈ 0, quality 축의 kolektor/leather 평탄성 CV 3~4%, `_np_quality` 극소영역 가드 1.0 아티팩트)은 유효한 발견으로 남긴다. 가드 아티팩트는 quality 값을 **순위**에 쓰는 어떤 후속 작업에서도 선결 수정 대상.
