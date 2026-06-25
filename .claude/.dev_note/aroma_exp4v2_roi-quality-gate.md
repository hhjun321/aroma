# exp4v2 ROI 선택 품질 게이트 (subtype-matching proxy)

origin/main이 이미 다양성·균형(Fix1–4)을 해결한 위에, **품질(부적합 ROI 배제)** 축만
추가하는 작업. `scripts/aroma/roi_selection.py` 단일 파일, 기본 OFF.

---

## 배경 — exp4v2(severstal, yolov8n) 진단 (옛 코드 결과 기준)

| 전략 | mAP50 | c1 | c2 | c3 | c4 |
|------|-------|----|----|----|----|
| baseline | 0.4163 | 0.3669 | 0.2257 | 0.5269 | 0.5458 |
| casda | 0.4330 | 0.3570 | 0.2987 | 0.5288 | 0.5476 |
| aroma | 0.3132 | 0.1862 | **0.0675** | 0.4803 | 0.5188 |

진단(`.claude/.etc/exp4v2/casda_aroma/` 산출물 분석):
- AROMA가 c2에 합성 429장(CASDA 175)으로 과다 투입했으나 c2 붕괴.
- CASDA suitability≥0.5 통과 c2 ROI는 전체에서 **117개뿐** — AROMA는 품질 컷오프가 없어
  부적합 ROI까지 다 사용.
- AROMA defect_bbox가 near-full-frame(전 클래스). ※ bbox 정밀화는 Phase 2 별도.

> ⚠️ 위 수치는 **옛 roi_selection.py 결과**. 그 사이 origin/main이 Fix1–4를 추가해
> AROMA 동작이 이미 바뀌었을 수 있음(특히 img_diversity_cap=1). 게이트 추가 효과는
> **새 코드 기준 재측정** 필요.

---

## origin/main이 이미 가진 것 (재화해 결과)

새 `roi_selection.py`는 다음을 이미 구현 — 본 작업과 **겹치는 부분은 재구현하지 않음**:
- **Fix4 `img_diversity_cap`(기본 1)**: 동일 source crop `(image_path, defect_bbox)` 1회만 선택
  → c2 중복 과다할당 직접 차단.
- **Fix2 `_stratified_pair_aware`/`class_floor`**: 클래스별 대칭 floor(굶김·독식 방지).
- **Fix1 `per_pair_cap_frac`**: pair monoculture 방지.
- **Fix3 `rarity_temp`**: deficit 항 온도.
- `class_key`는 `defect_type` 컬럼에서 생성.

→ 당초 계획의 **③ 공급량 연동 per-class cap은 위에 흡수되어 폐기**.
→ **② 품질 게이트만** 새 코드 위에 얹음(품질 축은 origin이 안 다룸 — 보완재).

---

## 구현 (② 품질 게이트, roi_selection.py)

- **subtype-matching proxy**: 기존 `utils/suitability.py SuitabilityEvaluator.matching_score` +
  `utils/defect_characterization.py classify_defect_subtype`를 배선. full suitability는 입력
  (continuity/stability/gram/background_type) 부재로 불가 → matching 항만 사용.
  severstal=directional: linear_scratch=1.0 / elongated=0.9 / general=0.7 / compact_blob·irregular=0.4.
- `build_candidates`: 후보에 `defect_subtype`·`quality_score` 추가(전량 기록).
- `apply_quality_gate(candidates, min_quality)`: `quality_score < min_quality` 후보 제거(pre-filter).
  selection/Fix1–4 기계는 통과분 위에서 그대로 동작.
- **기본값 `--min_quality 0.0` = 게이트 OFF = 레거시 동일.** `>0` 지정 시 활성.
  `--background_type` 기본 directional(min_quality>0일 때만 사용).
- 결측 morph 지표 → `('general', 1.0)` 통과(오분류·silent drop 방지). NaN quality_score →
  worst-case(0.0)로 드롭. 클래스 전량 탈락 시 warning(silent drop 금지).

제약 준수: `compute_complexity.py:97-99` MORPH_FEATURES 화이트리스트 무수정, CSV 신규 컬럼 없음
(in-memory/JSON 추가만). 다운스트림 `generate_defects.py`는 `.get()`만 써 추가 필드 무해.

## 검증
- py_compile(3.10) OK. 순수함수 스모크: gate OFF=identity / 결측→통과 / 필터링 / NaN 드롭 /
  0-클래스 warning 모두 OK. pytest 미실행(CLAUDE.md). 실측은 Colab.
- 코드리뷰(이전 라운드) 반영: 결측 morph 오분류 버그 수정, import except 한정(ImportError),
  NaN 가드.

## Colab 검증 순서
[[aroma_exp4v2_roi-quality-gate_colab-guide]] 참조. 요약:
1. `--min_quality 0`로 분포 확인 → 컷오프 결정
2. `--min_quality <값> --background_type directional` → 클래스별 passing 로그 확인
3. Step 4 재생성 → exp4v2 AROMA 재학습 → c2·전체 mAP50 회복 확인 (Fix1–4 적용된 새 baseline 대비)

## git 정리 이력
- 이전에 옛 베이스(f8a5e27) 위에 작업이 커밋(5124301)됐고 origin/main(13커밋 앞섬) 머지 충돌 발생.
- `merge --abort` → `reset --hard origin/main`(47b1239)으로 정리, 작업은 patch 백업 후 새 코드 위
  ② only로 재적용. 백업: scratchpad `phase1_5124301.patch` / `0001-roi-selection-quality.patch`.

## 진단 산출물 위치
`.claude/.etc/exp4v2/casda_aroma/` — roi_selected_{aroma,casda}.json, annotations_*, roi_metadata.csv
