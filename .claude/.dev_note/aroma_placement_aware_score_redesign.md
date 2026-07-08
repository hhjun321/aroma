# AROMA placement-aware 전환 — score_mode(realism) + compat 배치 게이트

---

## (사용할 skills: feature-dev)

> **설계 출처**: `AROMA연구분석/placement_aware_score_redesign_20260708.md` (이전 다중-에이전트 워크플로우의 채택/기각 최종 판정서 — planner/architect/adversarial-verify 단계 완료됨). 본 devnote는 그 채택 항목의 **1차 증분** 구현 스펙.
> **진단 근거**: [[project_method_pivot_compat_blend]](deficit 폐기), [[aroma_step4_h1-recombination-no-info]](재조합 무정보). 배경type↔결함type 호환(ctx_prior)은 선택 스칼라(0.4wt)에만 쓰이고 픽셀 배치에는 미반영(placement-blind)임을 코드로 확인 (`generate_defects.py:845 _place_on`).

## 개요

ROI 점수에서 반증된 deficit 항을 제거하고(realism 모드), 선택이 약속한 배경 호환을 배치가 물리적으로 이행하도록 `_place_on`에 위치-조건화 compat 게이트를 추가한다. 두 변경 모두 **opt-in 기본 OFF** — 기존 실험 byte-identical 보존. 목적은 §3의 **Placement×Selection 2×2 ablation**(A=baseline, B=선택만, C=배치만, D=풀)을 CLI 플래그 2개로 돌릴 수 있게 하는 것.

**1차 증분 범위 (본 devnote)**:
1. `roi_selection.py` — `--score_mode {legacy,realism}`. realism = `0.5·ctx + 0.3·morph + 0.2·quality`. legacy(기본) = 현행 `0.4·morph+0.4·ctx+0.2·deficit` byte-identical.
2. `generate_defects.py` — `--compat_threshold τ` (기본 0=OFF). `_place_on` 재샘플 술어에 `compat_ok(pos)` 추가 (texture 게이트와 동형 rng-discipline).
3. placement telemetry — 기존 `gate_stats`에 compat fallback률 집계 → `run()` 로그.

**후속(별도 증분, 본 devnote 범위 밖)**: 오프라인 clean-bg cell inventory 인덱스(런타임 재계산 대체), Colab 2×2 harness 가이드 + `placement_report.json` 사전등록 telemetry.

## 영향도 분석

### 이 기능이 변경하는 상태
- realism 모드: `roi_candidates.json`/`roi_selected.json`의 `roi_score` 값·정렬 순서 변경 (선택되는 ROI 집합이 달라짐). deficit는 provenance로 JSON에 남기되 weight 0.
- compat 게이트 ON: 합성 이미지의 결함 paste **위치** 변경 (배경 이미지 선택 `rng.choice`는 불변 — confound 최소). 개수·클래스·bbox는 parity 유지(기존 clean-bg/texture 게이트 계약 동일: 소진 시 마지막 후보 paste, silent drop 없음).

### 그 상태를 전제로 동작하는 기존 로직
- **`moderated_score`(L438) 상호작용 — 주의**: `rarity_temp==1.0`(기본)이면 저장된 `roi_score` verbatim 반환 → realism 점수 그대로 사용, 정합. 단 `rarity_temp≠1.0`이면 L452-455가 **legacy 가중치로 재계산**(realism 무시) → realism+rarity_temp≠1.0 조합은 불일치. **가드**: realism 모드에서 rarity_temp≠1.0이면 경고 로그 + realism 재계산식 사용하거나, 최소 문서화. (1차: 경고 후 realism식 재계산으로 정합.)
- **rng 결정론**: compat 게이트는 texture_on과 동일 패턴 — `τ=0`이면 신규 rng 소비 0, legacy byte-identical. τ>0일 때만 위치 재샘플 rng 소비 (deterministic-but-stream-shift, [[aroma_controlnet-arm_quality-filters]] 전례와 동일).
- **matrix/bin_edges 로드**: controlnet 경로가 이미 `_btj._load_bin_edges(config_yaml)`(L1237) 사용 — compat 게이트도 이 경로 재사용. copy_paste 경로는 `--config` 미전달이 기본이므로, config 없으면 compat 게이트 자동 OFF(게이트 무해 통과).

### delete/remove/bulk 계열
- 해당 없음 (선택·배치 로직 변경).

## 수정 내용

### 1. `roi_selection.py` — score_mode

- `score_roi(morph_prior, ctx_prior, deficit, quality_score=0.0, score_mode="legacy")` (L236):
  - `legacy`: `_W_MORPH·mp + _W_CONTEXT·cp + _W_DEFICIT·d` (현행 유지, quality 무시)
  - `realism`: `0.5·cp + 0.3·mp + 0.2·qs` (상수 `_W_R_CTX/_W_R_MORPH/_W_R_QUALITY`, 합=1 assert)
- 호출부 `build_candidates`(L349-351): `quality_score`(L326에서 이미 계산됨) + `score_mode` 전달. `build_candidates` 시그니처에 `score_mode` 추가.
- `moderated_score`(L438): realism 모드 provenance를 위해 `score_mode` 인지 — rarity_temp≠1.0 재계산 분기에 realism식 추가 or 경고.
- CLI `--score_mode {legacy,realism}` 기본 legacy → `run()` → `build_candidates` 배선.

### 2. `generate_defects.py` — compat 배치 게이트

- `_paste_and_finalize`(L775) 시그니처에 `compat_row: Optional[dict]=None`, `bin_edges=None`, `compat_threshold: float=0.0` 추가.
- `_place_on`(L845) 내부 `_tex_ok`(L859) 옆에 `_compat_ok(pos)` 헬퍼:
  - patch = `nrgb[py:py+ch, px:px+cw]` → `_extract_context_features`(GRID_SIZE=64 재사용) → `_context_cell_key(feat, bin_edges)` → `compat_row.get(cell, 0.5) >= compat_threshold`
  - **soft 매칭 필수**(leather 4.7% coverage 교훈): exact cell match 아니라 row의 여러 cell 수용. `compat_row.get(cell,0.5)` 기본 0.5는 미관측 cell 중립 통과.
- `compat_on` 가드 = `compat_threshold > 0 and compat_row and HAS_CV2` (texture_on 패턴 모방). foreground 루프(L884-894)·random-fallback 루프(L903-917)의 `clean and close` 조건에 `and compat_ok` 결합.
- `distribution_profiling`의 `_extract_context_features`/`_context_cell_key` import (sibling 모듈, controlnet ctx가 이미 유사 import).
- `run()`: `roi_entry['cluster_id']`로 `matrix[cid]` row 추출 → synthesis kwargs 전달. matrix/bin_edges는 `--config`에서 로드. `--compat_threshold` CLI 추가.

### 3. Telemetry

- `_place_on` → `meta["gate_stats"]`에 `compat_fallback` bool 추가 (compat_on인데 소진).
- `run()` gate_agg에 compat fallback 집계 → 로그 `placement-gate stats: compat_active=N fallback=M`. **fallback률 >50%면 경고**("placement-aware no-op 위험 — leather류").

## 수정 대상 파일

- `scripts/aroma/roi_selection.py` — score_mode
- `scripts/aroma/generate_defects.py` — compat 게이트 + telemetry
- (후속) `AROMA연구분석/colab_execute/` 2×2 harness 가이드

## 테스트 (Colab 검증 — 테스트 코드 작성·pytest 금지)

1. **legacy byte-identical**: `--score_mode legacy` roi_selection → 기존 roi_selected.json과 동일. `--compat_threshold 0` generate → 기존 산출물과 동일(로컬 fixture 회귀, [[aroma_controlnet-arm_quality-filters]] 방식).
2. **realism 정렬 변경**: `--score_mode realism` → roi_score 재계산, 선택 집합 변화 확인. quality_score 분산 0이면 no-op임을 로그로 확인(구조적 한계 §5.5).
3. **compat 게이트 발동**: `--compat_threshold` 후보값으로 generate → `placement-gate stats` fallback률 데이터셋별 확인. leather 고fallback(무효), aitex 저fallback(유효) 예상.
4. **2×2 harness**: A/B/C/D 4 arm을 `--score_mode`×`--compat_threshold`로 구성, exp4v2 비교.

## TODO / 후속

- 오프라인 clean-bg cell inventory 인덱스(런타임 `_extract_context_features` 재계산 → 사전 계산 lookup).
- Colab 2×2 harness 가이드 + `placement_report.json` 사전등록 telemetry(baseline 헤드룸/fallback률/mean texture-dist).
- realism 가중치(0.5/0.3/0.2)는 **test 수치 보기 전 고정, 사후 튜닝 금지**(보고서 §1.1 정직성 경고).
- severstal coverage 미측정 → severstal placement 주장 금지(보고서 §5.4).
