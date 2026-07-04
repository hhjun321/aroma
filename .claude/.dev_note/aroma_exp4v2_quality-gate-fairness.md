# Exp4v2 — ROI quality gate 공정성 보정 (random 대칭 게이트)

## (사용할 skills: feature-dev)

> 계열: [[aroma_exp4v2_roi-quality-gate]](게이트 자체 구현), [[exp4v2-roi-selection-vs-background]], [[aroma_exp4v2_multiclass-all-datasets]].
> 목적: `--min_quality`를 켤 때 aroma만 게이트되고 random이 우회하는 **budget-parity confound**를 제거한다.

## 개요

quality gate(subtype↔background matching proxy)는 현재 **aroma 선택 경로에만** 적용된다:
- aroma: `roi_selection.py` → `apply_quality_gate()` → `roi_selected.json` (gated)
- random: `generate_random.py` → `roi_candidates.json` **균일 샘플 (un-gated)** ← confound
- casda: severstal 전용, 자체 `min_suitability=0.5` (전략 정체성 — 유지)

이대로 게이트를 켜면 aroma 풀만 축소 → exp4v2 학습 cap에서 aroma만 no-trim → `n_synth_train` 불일치 → 공정성 붕괴. **보정 = random도 동일 게이트 통과 풀에서 샘플** (메모리 선택지1). "selection 전략만 다르다" 불변식 유지.

### 핵심 사실 (코드 확인)
- `quality_score`는 gate ON/OFF와 무관하게 **모든 후보에 항상 계산·저장**됨(`roi_selection.py:362`, `score_candidates`에서 `quality_proxy()` 호출). → random 쪽에서 **재계산 없이 동일 값으로 필터 가능** (동일 background_type로 계산된 값 공유 = 대칭 자동 보장).
- deps(`utils/suitability.py`, `utils/defect_characterization.py`)는 AROMA 레포에 존재 — 순수 AROMA 환경에서 작동.
- deps/metric 결측 시 `('general', 1.0)` → 통과(안전) — random 필터도 동일 semantics(저장된 1.0 사용)로 자동 일치.

---

## 수정 내용

### 1. `scripts/aroma/generate_random.py` — `--min_quality` 추가 (핵심)

```python
p.add_argument("--min_quality", type=float, default=0.0,
               help="roi_candidates.json의 quality_score 필터 (0=OFF, 기존 동작). "
                    "aroma --min_quality와 동일 값 → 동일 풀에서 균일 샘플(공정성).")
```
- `select_random()` 이전에 `candidates = [c for c in candidates if _q(c) >= min_quality]` (NaN→0.0, `quality_score` 결측→1.0 통과 — roi_selection `_q`와 동일 semantics).
- 필터 후 pool < top_k면 경고 로그(silent 금지) + 전량 사용.
- 로그: `Quality gate (random parity): min_quality=X → M/N candidates pass`.

### 2. 공정성 불변식 (load-bearing)

- **동일 threshold**: aroma `roi_selection --min_quality X` == random `generate_random --min_quality X`. 문서(step3·exp3 0단계)에 같은 env var(`MIN_QUALITY`)로 강제.
- **동일 스코어 원천**: 둘 다 `roi_candidates.json`의 저장된 `quality_score` (roi_selection이 계산·기록, random은 읽기만) → background_type 불일치 원천 차단.
- **casda**: severstal 전용 + `min_suitability`가 전략 정체성이므로 별도 게이트 주입 안 함. honest asymmetry로 문서 명시(기존 규약 유지).
- **pool ≥ cap 재확인**: 게이트가 풀을 줄이므로, exp4v2 `synth_ratio` cap 대비 게이트-후 pool이 여전히 ≥ cap인지 (특히 severstal) 결과 JSON `n_synth_train` parity로 검증.

### 3. 데이터셋별 `--background_type` (게이트 켤 때 필수 지정)

| 데이터셋 | 제안 | 근거 |
|----------|------|------|
| severstal | `directional` | 강판 스트립 (검증됨, 기본값) |
| aitex | `periodic` | 직조 텍스타일 반복 패턴 |
| mvtec_leather | `organic` | 유기 표면 |
| mtd | `smooth` | 자성 타일 평활 표면 |

> ⚠️ severstal 외 3종의 matching_score 의미는 미검증 — 켜기 전 per-dataset로 `quality_score` 분포(pass율)를 1회 확인하고 threshold를 정할 것(전량 거부/전량 통과면 무의미).

### 4. 문서 갱신 (구현 후)

- `step3_execute.md`: `--min_quality $MIN_QUALITY --background_type <표>` 옵션 셀(기본 OFF 유지) + "random과 동일 값 필수" 경고.
- `exp3_execute.md` 0단계: generate_random에 `--min_quality $MIN_QUALITY` 추가.
- `exp4v2_execute.md`: parity 확인 항목에 "게이트 ON 시 n_synth_train 동일 + pool≥cap" 추가.

---

## 수정 대상 파일

- `scripts/aroma/generate_random.py` — `--min_quality` 필터 (+`_q` semantics 일치)
- (문서) `step3_execute.md` / `exp3_execute.md` / `exp4v2_execute.md`

## 테스트 (Colab, pytest 금지)

1. `py_compile`.
2. random `--min_quality 0`(기본) → 기존과 byte-identical(선택 목록 동일 seed 동일).
3. `--min_quality 0.5` → 로그의 pass 수가 aroma 게이트 로그의 pass 수(동일 threshold)와 동일 풀 크기인지.
4. exp4v2 스모크: 게이트 ON 재합성 후 `n_synth_train` random==aroma 확인.

## 미확정 (TODO)

1. **threshold 값 — ★data-driven 규칙 확정 (2026-07-04, 사용자 지시)**: quality_score는 연속값이 아니라 `MATCHING_RULES`(subtype×background lookup, utils/suitability.py:11) **이산 수준**(데이터셋당 ≤5개). 따라서 percentile이 아닌 **최대-갭 컷**으로 결정:
   - 데이터셋별 `roi_candidates.json`의 저장 `defect_subtype` 분포 → 해당 background_type 기준 score 수준별 질량 재계산 (⚠️ 저장 `quality_score`는 step3 실행 당시 `--background_type`(기본 directional) 기준이라 비-severstal엔 재계산 필수 — subtype이 저장돼 있어 재프로파일링 불요).
   - threshold = 인접 수준 간 **최대 갭의 중점**. 가드: pass율 ≥ 50%(cap parity 보호) + class_key별 passing>0. 위반 시 차선 갭, 없으면 OFF.
   - Colab 실측 셀은 대화 로그/step3 문서 참조. 확정값을 데이터셋별 `MIN_QUALITY`로 기록 후 aroma·random 동일 적용.
2. **background_type 3종 검증**: aitex/leather/mtd에서 subtype 분류·매칭이 유의미한지 pass율 분포로 1회 확인.
3. **게이트 ON을 실험 기본으로 할지**: 우선 OFF 유지(현행), severstal ratio 1.0 재실험 결과 본 뒤 ON 여부 결정(메모리 결정분기 순서 유지).
