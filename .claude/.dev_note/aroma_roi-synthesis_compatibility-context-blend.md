# AROMA 방법론 전환 — compatibility+quality ROI 선택(L1) + context-aware 블렌딩(seamlessClone+Reinhard, L3)

## (사용할 skills: feature-dev)

> 대규모 다중 파일 방법론 변경(신규 알고리즘 2종). micro-fix 아님 — L1(선택 목적함수 교체 + import 버그 수정)·L3(블렌딩 엔진 신규)는 별개 파일·별개 알고리즘, 핵심 주장(AROMA>Random)을 바꾸는 전환이라 설계 단계 필요.

## 개요

exp2 실증 분석으로 두 사실 확인: (1) **deficit-aware는 object-centric서 무의미** — 실제 `roi_candidates.json` 점유에서 carpet/leather/wood/metal_nut의 rare_pair(deficit>0)=0, severstal만 신호(rare 146, cells 107, pairs 364). object-centric은 점유 pair ≤30 → random이 전부 커버. (2) **copy-paste 합성 하에선 ROI 배치가 downstream(FID/detection)에 거의 안 보임** → ROI 선택 천장이 합성품질에 막힘, 선택·합성 분리 불가.

→ deficit-aware를 헤드라인 원리에서 폐기. **선택(L1)과 합성(L3)을 함께 개선.** 합성이 배경 context에 의존하게 만들어야 "어디 두나"가 픽셀에 반영되고, ROI 선택이 measurable + random 추월 가능. training-free 유지.

근거: [[AROMA_worklist]] §8, exp2 점유 분석(`.claude/.etc/roi`), [[aroma_exp4v2_clean-background-gate]], [[aroma_exp4v2_aroma-underperformance-diagnosis]].

---

## 영향도 분석

### 이 기능이 변경하는 상태
- `roi_candidates.json`/`roi_selected.json`: quality gate가 실제 동작하면 selected 구성 변화. 신규 strategy 시 `roi_score` 의미 변화 가능 → 원점수 provenance 보존 검토.
- `annotations.json`: `blend_mode` 필드에 신규 값(`seamless`) 기록. `bbox`·`mask_path` 의미 유지 — seamlessClone은 mask 영역 *내부 픽셀만* 변경, GT mask/bbox 좌표 불변.
- 합성 출력 픽셀: 배경 context 반영(Reinhard) + 경계 seamless화.

### 그 상태를 전제로 동작하는 기존 로직 (깨지면 안 됨)
- **clean-bg gate (항상 ON 정책)**: `_is_clean_background`(`generate_defects.py:331`), pool gate(`:860-928`), random fallback gate(`:572-581`). 신규 블렌더는 position 결정 *이후* 작동 → 직교, 안 깨짐. 단 Reinhard 로컬 배경 통계가 void 패치 안 읽도록 게이트와 순서 보장.
- **foreground/void-rejection**: `_foreground_mask`(`:180`, `_FG_VOID_*` `:173-177`), `_foreground_paste_position`(`:369`). position 산출 그대로 두고 결과만 사용 → 영향 없음.
- **wrapper 위임**: `generate_random.run()`(`generate_random.py:150-161`)·`generate_casda.run()`(`generate_casda.py:113-124`) 모두 `generate_defects.run()` 호출, **`method="copy_paste"` 하드코딩 + `blend_mode`/`feather_px` 미전달**. → 신규 `blend_mode='seamless'`를 세 arm(aroma/random/casda) 동일 적용하려면 두 wrapper의 `run()` 시그니처+위임+CLI에 `blend_mode` 추가 필요 (공정비교 = 동일 블렌더).
- **exp4_v2 mask→bbox**: `annotations.json`의 `mask_path`에서 bbox 복원 → mask 저장 로직(`copy_paste_synthesis:596-608`) 신규 블렌더서도 유지. seamlessClone이 mask 내부만 바꾸므로 mask 불변.

### ablation 보존
- `deficit_aware` strategy + `_pair_aware_allocation`/`_stratified_pair_aware` 전부 **삭제 금지** — ablation 옵션으로 보존.

---

## 수정 내용

### 1. `scripts/aroma/roi_selection.py` — L1: compatibility+quality 선택

**(a) 품질 게이트 import 수정 (현재 BROKEN)**
- 현상: exp2 로그 `No module named 'utils'` → `:99-112`의 `from utils.suitability import ...` / `from utils.defect_characterization import ...` 실패 → `_HAS_QUALITY=False` → quality_score 항상 1.0, `apply_quality_gate`(`:362-415`) no-op.
- 원인 후보: `_bootstrap_aroma_ref()`(`:56-66`)가 넣는 `AROMA_REF`(기본 `D:\project\aroma`)와 Colab 클론 경로(`/content/...`) 불일치 또는 `utils/` 패키지 미존재.
- 수정: utils 모듈 실제 위치 확인 → sys.path bootstrap 보강(택1) **또는** AROMA-native 경량 재구현(택2, §TODO). `_HAS_QUALITY` graceful no-op 안전망은 보존.

**(b) 신규 strategy: compatibility (+ quality gate)**
- 현 점수식 `score_roi()`(`:220-234`) = `0.4·morph_prior + 0.4·ctx_prior + 0.2·deficit` (가중치 `_W_*` `:155-159`, 합=1).
- **사실 확인**: candidate에 `compatibility`/`compat` 필드는 **없음**. compatibility는 `compatibility_matrix.json`의 `matrix[cluster_id][cell_key]` 값이고 이게 곧 현재 `ctx_prior`로 저장됨(`build_candidates:333-335`). → **"compatibility = ctx_prior" 가 이미 코드의 사실.**
- candidate 실제 필드(`build_candidates:339-356`): `quality_score, roi_score, morph_prior, ctx_prior, deficit, defect_subtype, class_key, cluster_id, cell_key, defect_bbox, defect_mask_path, image_path, image_id, prompt, morph_label, ctx_label`.
- 수정:
  - `select_rois()`(`:1170`) 분기(`:1218 top_k / :1221 random / :1226 weighted / :1236 deficit_aware`)에 **신규 `compatibility` 분기 신설** — deficit 미사용, `ctx_prior`(compatibility)로 WHERE 점수화 + `quality_score` gate/필터.
  - 점수 재가중: deficit 비중 폐기 또는 ablation 전용. (정확 공식 §TODO)
  - `deficit_aware`(`:1236-1247`) + 머신리 전부 보존. CLI `--sampling_strategy` choices(`:1411-1413`)에 신규 값 추가.
  - quality gate 헤드라인 기본 ON 검토(현재 `min_quality=0.0` 기본 OFF, `run():1319`, CLI `:1442`). 빈-pool fallback(`run():1347-1352`) 유지.

### 2. `scripts/aroma/generate_defects.py` — L3: context-aware 블렌딩

**현 구조**: `_alpha_composite()`(`:112-151`)가 유일 블렌딩(alpha+scipy feather). `copy_paste_synthesis()`(`:422-614`)가 본체, position 결정 후 `_alpha_composite` 호출(`:582`). `blend_mode` choices=`["alpha"]`(`:1138`). **Poisson은 docstring만 있고 구현 없음.**

**수정**:
- 신규 `_context_aware_composite()` (또는 `blend_mode='seamless'` 분기):
  1. **Reinhard 색/조도 전이**: defect crop의 Lab 평균/표준편차를 **로컬 배경 패치**(paste position 주변 crop-sized)의 Lab 통계에 매칭 (`cv2.COLOR_RGB2LAB`). → 배치가 픽셀에 반영.
  2. **cv2.seamlessClone**(`NORMAL_CLONE`, gradient-domain): 경계 seamless화.
- **주입 지점**: `copy_paste_synthesis()`의 `_alpha_composite(...)` 호출부(`:582`). 직전에 `normal_rgb`(`:559`)+`position`(`:561-581`)로 로컬 배경 패치 추출 → Reinhard → seamlessClone. position 결정 로직(foreground/void/clean-bg) **무변경**, 출력만 새 블렌더로.
- **선택**: `blend_mode` choices에 `seamless` 추가(`:1137-1139`), `copy_paste_synthesis(blend_mode=...)`(`:426-434`)·`run()` blend_mode(`:941`)·CLI 연동.
- **training-free**: seamlessClone(Poisson solver)·Reinhard(통계매칭) = 고전 CV, DL 없음.
- **HAS_CV2 가드**: cv2 없으면 alpha로 graceful fallback.

### 3. wrapper 포워딩 (random/casda 동일 블렌더)
- `generate_random.py` / `generate_casda.py`: `run()` 시그니처+위임+CLI에 `blend_mode` 추가 → 세 arm 동일 블렌더(공정 비교). 미전달 시 기본값.

### 4. `scripts/aroma/experiments/exp2_roi_quality.py` — 평가 격하 (선택)
- coverage 지표를 secondary로. 코드 변경은 리포팅 수준(선택).

---

## 수정 대상 파일

- `scripts/aroma/roi_selection.py` (L1: import 수정, compatibility strategy, 라우팅, deficit_aware 보존)
- `scripts/aroma/generate_defects.py` (L3: seamlessClone+Reinhard 블렌더, blend_mode 확장)
- `scripts/aroma/generate_random.py` (blend_mode 포워딩)
- `scripts/aroma/generate_casda.py` (blend_mode 포워딩)
- `scripts/aroma/experiments/exp2_roi_quality.py` (평가 격하, 선택)

---

## 암묵적 요구사항 / 엣지 케이스

1. **seamlessClone 실패**: mask 극소/center 경계 → cv2 예외·검은 출력. try/except → alpha+feather fallback. center 내부·mask bbox 경계 clamp.
2. **Reinhard on grayscale (severstal)**: 강판 사실상 grayscale → Lab a/b 통계 무의미. L 채널만 매칭 or grayscale 전용 경로 (§TODO).
3. **quality gate empty-after-filter**: 전부 탈락 시 `run()` 빈 selected 경고(`:1347-1352`). 게이트 ON 시 threshold 완화/skip fallback.
4. **training-free 제약**: 신규 의존성(diffusers/torch) 금지.
5. **결정성/seed**: Reinhard·seamlessClone RNG 없음(결정적). position `rng`(`:475-476`) 유지. 동일 seed→동일 출력.
6. **compatibility_matrix 누락 bin**: `matrix.get(cid_str,{})`(`:324`) 빈 fallback(`:332`). 신규 strategy 빈 matrix서 graceful(동점→random).
7. **로컬 배경 패치 경계**: 이미지 경계서 잘림 → 빈 패치면 전이 skip.
8. **cv2 부재**: `HAS_CV2=False`(`:77`) → alpha degrade + 경고.

---

## 테스트 (Colab, pytest 금지)

신규 테스트 코드/pytest 금지, Colab 직접 검증 + .md 가이드. 부하 실측 자동 금지.

1. **before/after FID@defect-patch**: severstal + diversity set, copy_paste(alpha) vs seamless+Reinhard → 합성품질 개선.
2. **before/after detection (exp4_v2)**: mAP — placement-sensitive 지표로 AROMA>random 확인.
3. **visual check**: 동일 ROI를 다른 배경에 배치 시 출력 픽셀 실제 변화(=measurable) 육안.
4. **ablation**: deficit_aware vs compatibility, quality gate ON/OFF, alpha vs seamless.
5. **dataset 분리**: exp2=severstal-centric, exp3/exp4=diversity set.

---

## 확정 사항 (TODO 해소, 2026-06-29)

| # | 결정 |
|---|------|
| compat 공식 | `0.6·ctx_prior + 0.4·morph_prior` (deficit 無), quality_score=상류 gate 필터. **유지(튜닝 가능, ablation서 검증)** |
| Reinhard 색공간 | ref(로컬배경) a/b std<1.0이면 L채널만, 아니면 Lab 전체. **구현 완료** |
| exp2 | **재목적화** — coverage/deficit 폐기 → 선택 ROI의 **평균 compatibility(ctx_prior) + 평균 quality_score**(AROMA vs random vs casda). 포화 없음, 전 셋 측정 |
| realism(합성성능) 지표 | **결과 표에서 제외** — AROMA 기여=ROI 선택. 블렌드(L3)는 배치가 측정되게 하는 enabler일 뿐, 합성품질은 헤드라인 아님 |
| quality threshold | 코드 default 0.0 유지. gate 복구 후 첫 run의 quality_score 분포 확인 후 임계 결정 (**defer**) |
| 블렌더 random/casda | **세 arm 모두 seamless**(공정 — 조건은 ROI 선택만 다름). 포워딩 구현 완료 |
| utils 해결 | `__file__` 루트 부트스트랩으로 import 복구. **재구현 불필요** |

## 확정 실험 설계

| 실험 | 데이터셋 | 조건 | 지표 |
|------|---------|------|------|
| **exp2** | 5셋 (severstal+carpet+leather+macaroni+fryum) | random vs aroma | 선택 ROI 평균 compatibility+quality |
| **exp3** | 동 5셋 | baseline/random/aroma | FID + one-class AD |
| **exp4v2 (2층위)** | severstal | baseline/random/**casda**/aroma (4-way) | supervised detection mAP |
|  | carpet/leather/macaroni/fryum | baseline/random/aroma (3-way) | supervised detection mAP |

- casda는 철강 전용(handcrafted compat) → severstal 4-way에만. AROMA는 전 셋 → 일반화 입증.
- 전 5셋 supervised detection(신뢰 지표) 확보 → 다운스트림 일반화 성립.
- 합성 조건은 **ROI 선택만 다르고 blend=seamless 동일**(조건 격리).
