# AROMA 자체 ControlNet 학습데이터 빌더 신규 구현 (`build_train_jsonl.py`) — CASDA `prepare_controlnet_data.py` 의존 제거

## (사용할 skills: feature-dev)

> 신규 파일 + 신규 추상화(빌더 함수/CLI) + 다중 유틸 배선(hint/prompt/crop/morphology join) + 도메인별 출력 설계. micro-fix 조건(단일 관심사·새 추상화 없음·50줄 미만) 전부 위반 → feature-dev. 설계 spec [[aroma_research-core_self-contained-multidomain-design]] 섹션5 P0, 170행과 정합.

## 개요

설계 spec [[aroma_research-core_self-contained-multidomain-design]] 발견① — 현재 AROMA가 ControlNet 학습데이터를 만드는 **유일 경로가 CASDA 경유**다: AROMA `roi_selected.json` → `aroma_to_casda_roi.py`(FORWARD adapter) → CASDA `prepare_controlnet_data.py`/`ControlNetDatasetPackager`가 hint+prompt 재생성. 이 상태로 "self-contained, CASDA 병합 아님"을 논문에 쓰면 `aroma_to_casda_roi.py` 헤더(3–24행)만 읽어도 사실오류로 반박됨.

P0 = real 결함 → target crop + 3채널 hint PNG + prompt를 **AROMA 안에서 직접 생산**하는 CPU-side glue 스크립트(`scripts/aroma/build_train_jsonl.py`) 신규 작성. `scripts/train_controlnet.py`가 CASDA 없이 곧바로 소비할 도메인별 `train.jsonl` 생성. self-contained 서사의 gating. **GPU 무관(순수 CPU).**

### ⚠️ spec 대비 코드 정정 (Opus grounding, file:line 근거)

1. **hint 생성기는 AROMA에 이미 존재** — `utils/hint_generator.py`의 `HintImageGenerator`가 R/G/B 3채널 알고리즘 완비. spec 23행 "grep 0건, CASDA에만"은 stale. **신규 hint 알고리즘 작성 불요 → 재사용.**
2. **`PromptGenerator.generate_prompt(...)`는 prompt 문자열만 반환** (튜플 아님). negative는 `generate_negative_prompt()`(무인자) 별도. 위치 = `utils/prompt_generator.py` (scripts/aroma/prompt_generation.py 아님 — 그건 cluster×cell 배치 템플릿 생성기).
3. **pool = `morphology_features.csv` 행** (1행=1 real 결함). `roi_candidates.json`/`roi_selected.json`은 `roi_selection.py:349`가 각 morphology 행을 occupied (cluster,cell_key) bin마다 **cross-join** → 결함 1개가 여러 candidate로 중복 등장 → pool 부적합. 확정 결정 "전체 real 결함 pool"과 정합.

### 확정 결정 (사용자) — 전부 data-driven, 하드코딩 금지

실데이터(`D:\project\aroma\.claude\.etc\mvtec_cable`: complexity/profiling/roi/prompts) 분석으로 확정. mvtec_cable = 92 real 결함, defect_type 8종.

- **train 입력 범위** = 전체 real 결함 pool (`morphology_features.csv` 행, 1행=1결함). roi_selected subset 아님. (roi_candidates=225행은 결함×배치cell cross-join이라 pool 아님; 단 결함당 cluster_id/ctx_prior 조인 소스로 사용 — 92 결함 100% 커버 확인.)
- **hint morphology 소스** = `morphology_features.csv` join 전용. csv 없는(=`defect_mask_path` 빈) 행 skip+count. DefectCharacterizer 즉석계산 미사용(스케일 불일치 회피).
- **hint 채널 표현** = CASDA와 동일하게 R/G/B를 **정보 채널(결함/배경구조/텍스처)**로 쓰고 **가중 그레이스케일 병합 후 3채널 복제**. literal RGB 아님. → `HintImageGenerator.generate_hint_image`가 `gray=clip(0.5·R+0.3·G+0.2·B)` → `np.stack([gray]*3)` (`utils/hint_generator.py:113-120`)로 **이미 정확히 구현** → 재사용만으로 충족, 수정 불요.
- **prompt 소스** = `utils/prompt_generator.py::PromptGenerator` (per-defect technical). roi의 `prompt` 필드는 전부 빈값이라 재사용 불가.
- **defect_subtype** = **morph_label** (data-driven, `morphology_clusters.json` cluster label: elongated/general/compact_blob). 결함당 cluster_id/morph_label은 `roi_candidates.json`에 100% 존재 → 조인. (하드코딩·'general' 고정 아님.)
- **background_type** = **결함별 own 이미지 context에서 data-driven 유도.** categorical이 profiling에 없음 → `context_features.csv`(결함이미지 patch, image_type=='defect')에서 해당 image_id patch 집계 → `recommended_config.yaml`/`compatibility_matrix.json`의 **bin_edges(데이터 산출 임계)**로 5-feature binning → 카테고리 매핑. (도메인 단일 상수·하드코딩 아님.)
- **stability_score** = **ctx_prior** 유도 (data-driven). 결함당 `roi_candidates.json` ctx_prior(=compatibility_matrix 값, 결함-배경 공존확률). 결함이 여러 cell이면 **max**. hint G채널 스케일(`hint_generator.py:73`)+prompt에 사용.

---

## 영향도 분석

### 이 기능이 변경/신설하는 상태
- **신규 파일** `scripts/aroma/build_train_jsonl.py`.
- **신규 데이터**: 도메인별 `train.jsonl` + target/hint PNG 하위 디렉토리(도메인별 output_dir 분리).
- 기존 상태 변경 **없음** — 순수 신규 경로 추가.

### 그 상태를 전제로 동작하는 기존 로직 (깨지면 안 됨)
- **소비자 `scripts/train_controlnet.py` 무수정**: 데이터셋 로더가 읽는 키 = `sample["target"]`(:261), `sample["hint"]`(:273), `sample.get("prompt","")`(:287). `negative_prompt`/`source`는 로드는 되나 `__getitem__`서 안 읽힘(학습 무효과, 스키마 완결성용). 빌더 출력이 이 키를 정확히 맞춰야 함.
- **재사용 유틸 무수정**: `utils/hint_generator.py`, `utils/prompt_generator.py`, `aroma_to_casda_roi.py` 헬퍼, `distribution_profiling.py`(입력 csv 생산자).
- **Step 1–3 회귀 0**: compute_complexity/prompt_generation/roi_selection 무관.

### 신규성 (delete/bulk 아님)
- 신규 경로 추가만 → 기존 copy_paste/Tier0 재현성 보존. "0개" 상태 생성 없음.

---

## 수정 내용

### 1. `scripts/aroma/build_train_jsonl.py` (신규) — 도메인별 train.jsonl + hint/target 빌더

per-real-defect 루프: morphology 행 → crop(target) → hint → prompt → jsonl 1줄.

**train.jsonl 라인 스키마** (근거 `scripts/train_controlnet.py`):
```json
{"target": "<crop image PNG 경로>", "hint": "<3ch hint PNG 경로>", "prompt": "<str>", "negative_prompt": "<str>", "source": "<target과 동일 경로>"}
```
- JSONL(줄당 1 JSON object), `ensure_ascii=False`.
- 로더 실제 사용: target/hint/prompt만. negative_prompt/source는 레거시(기록만, 학습 무효과 — dev_note 명시).
- 경로: **절대경로 기록**(morphology_features.csv의 image_path가 이미 절대). 로더 `_resolve_image_path`(:133-177)/`_resolve_hint_path`(:179-221)가 절대 우선 + `--image_root` 상대 fallback 처리 → 절대 권장(가장 견고). (상대 옵션은 TODO)

**재사용 헬퍼 (정확 시그니처)**:
- `utils/hint_generator.py::HintImageGenerator`
  - `__init__(enhance_linearity=True, enhance_background=True)`
  - `generate_hint_image(roi_image, roi_mask, defect_metrics: Dict, background_type: str, stability_score: float) -> np.ndarray` — **입력은 bbox-crop된 배열**. R(:22 linearity>0.7 skeletonize / solidity>0.8 fill / else Canny)·G(:49 Sobel 방향엣지×stability)·B(:77 국소분산) → 가중 그레이스케일 병합 3채널(:113-120).
  - `save_hint_image(hint_image, output_path)` (:122, RGB→BGR 저장)
- `utils/prompt_generator.py::PromptGenerator`
  - `__init__(style='technical')` (도메인중립)
  - `generate_prompt(defect_subtype: str, background_type: str, stability_score=0.5, defect_metrics: Dict=None, suitability_score=0.5) -> str` (:170-188, 문자열만)
  - `generate_negative_prompt() -> str` (:190-194, 무인자 고정)
- `scripts/aroma/aroma_to_casda_roi.py` (import 재사용)
  - `_make_crop_pair(image_path, mask_path, bbox, crops_dir, base) -> Optional[Tuple[str,str]]` (:407) — image+mask 동일 bbox crop, 크기정렬 강제, PNG 2개 저장, 실패 None+WARN. → **target crop 생산.**
  - `_crop_xywh(arr, b) -> Optional[np.ndarray]` (:394) — in-memory crop(hint 입력 배열용).
  - `_read_gray_or_color(path, gray) -> Optional[np.ndarray]` (:358)
  - `_load_morphology(csv) -> (by_mask, by_img_bbox)` (:219), `_join_metrics(...)` (:264), `_as_xywh(bbox)` (:159), `_remap_root(p, root)` (:197)

**입력 pool = `morphology_features.csv`** (writer `scripts/distribution_profiling.py:288-303`, header `_MORPH_CSV_FIELDS`:92-97). 컬럼:
```
image_id, image_path, defect_type, domain, mask_source,
linearity, solidity, extent, aspect_ratio, eccentricity, circularity, area,
defect_bbox("x,y,w,h"), defect_mask_path(full-frame 0/255 PNG)
```
- crop 3요소(image_path/defect_bbox/defect_mask_path) 전부 존재.
- linearity/solidity/extent/aspect_ratio/area → hint(R) + prompt 입력.

**data-driven 보조 입력 + 조인 (하드코딩 대체)**:
- `roi_candidates.json` (조인 by `image_id`+`defect_mask_path`, 92 결함 100% 커버 확인) → 결함당:
  - `cluster_id` + `morph_label` → **defect_subtype** = morph_label.
  - `ctx_prior` (결함이 여러 candidate cell이면 **max**) → **stability_score**.
- `context_features.csv` (`image_type=='defect'`, image_id 일치 patch 집계) → 결함 own 이미지 배경 context.
- `recommended_config.yaml`(또는 `compatibility_matrix.json`) `bin_edges` → 5-feature(local_variance/edge_density/texture_entropy/frequency_energy/orientation_consistency) binning(0/1/2).
- (`morphology_clusters.json`은 cluster_id→label 참조 — morph_label이 candidates에 이미 있으면 조인 불요.)

**background_type data-driven 유도 (신규 헬퍼 `_derive_background_type`)**:
1. 결함 image_id의 `context_features.csv` defect patch들 → feature별 mean.
2. `bin_edges`로 각 feature bin(0/1/2) 산출 (임계=데이터 산출치, 하드코딩 아님).
3. bin→카테고리 정책 (ordinal-bin 규칙; hint_generator.py:63-94 카테고리에 매핑):
   - `orientation_consistency` bin==2 (directional) → `'directional'`
   - elif `texture_entropy` bin==2 OR `local_variance` bin==2 → `'complex_pattern'`
   - elif `edge_density` bin==0 AND `local_variance` bin==0 → `'smooth'`
   - else → `'complex_pattern'` (textured 기본; hint_generator가 textured/complex_pattern 동일 처리)
   - ⚠️ 정책의 **임계는 전부 bin_edges(데이터)**, 규칙은 context 의미라벨(prompt_generation `_CTX_LABELS`)에 근거. 특정 도메인 문자열 하드코딩 없음.

**빌더 함수 (제안)**:
- `build(morphology_csv, roi_candidates, context_features_csv, config_yaml, output_dir, image_root=None, style='technical', ...) -> int` — 도메인 1개 처리, 작성 라인 수 반환. background_type/stability_score/defect_subtype는 인자 아님 → data-driven 유도(위 조인/매퍼).
- 루프(결함당): `morphology` 행 → candidates 조인(cluster_id/morph_label/ctx_prior) → `_derive_background_type(image_id)` → `_as_xywh(defect_bbox)` → target `_make_crop_pair` → hint 입력 crop(`_read_gray_or_color`+`_crop_xywh` image & mask) → `HintImageGenerator.generate_hint_image(metrics, background_type, stability_score=ctx_prior)` → `save_hint_image` → `generate_prompt(defect_subtype=morph_label, background_type, stability_score=ctx_prior, defect_metrics, suitability_score=ctx_prior)`+`generate_negative_prompt()` → jsonl append.
- 출력 하위구조: `{output_dir}/targets/`, `{output_dir}/hints/`, `{output_dir}/train.jsonl`.
- CLI: `--morphology_csv --roi_candidates --context_features --config --output_dir [--image_root --style]`. (background_type/stability는 data-driven이라 CLI 상수 없음; 조인 miss 시 fallback 정책은 TODO.)

---

## 수정 대상 파일

- **신규**: `scripts/aroma/build_train_jsonl.py`
- **재사용(무수정)**: `utils/hint_generator.py`, `utils/prompt_generator.py`, `scripts/aroma/aroma_to_casda_roi.py`, `scripts/train_controlnet.py`, `scripts/distribution_profiling.py`
- **입력(무수정, data-driven)**: `morphology_features.csv`(pool), `roi_candidates.json`(cluster_id/morph_label/ctx_prior 조인), `context_features.csv`(결함 배경 context), `recommended_config.yaml`/`compatibility_matrix.json`(bin_edges)
- **참조(무수정)**: `dataset_config.json`, `morphology_clusters.json`(cluster label), `scripts/aroma/prompt_generation.py`(`_CTX_LABELS` bin 의미 근거)

---

## 암묵적 요구사항 (엣지 케이스)

- **crop 실패**(소스 없음/판독불가/degenerate bbox): 해당 결함 skip+WARN, no-crash. `_make_crop_pair`/`_crop_xywh`가 None 반환으로 계약 구현 → `n_skipped` 증가.
- **mask 없음**(`defect_mask_path` 빈 문자열, distribution_profiling.py:277 저장실패 행): mask crop 불가 → skip+count(`n_no_mask`).
- **빈 결과**(pool 0행 / 전부 skip): 빈 train.jsonl 방지 → 빌더가 **명확 에러+비정상 종료**(로더 :652 "No valid training samples" 전에 원인 보고).
- **도메인별 출력 분리**: 도메인마다 별 output_dir → 별 train.jsonl + 별 hints/targets.
- **결정론적 순서**: csv 행 순서(또는 image_id,bbox 정렬) 고정 → 재실행 동일 jsonl. RNG 금지. ⚠️ `PromptGenerator._surface_quality`(:90-96)가 `random.choice` 사용 — style='technical'은 미사용(:126-168 확인)이라 무영향; 'detailed' 사용 시 seed 고정 필요.
- **hint/target 크기 정합**: hint를 동일 crop 배열에서 생성 → target crop과 동일 영역·크기. `_make_crop_pair`가 image/mask 정렬 보장.
- **skimage 지연 import**: R채널 linearity>0.7 분기가 `skimage.morphology.skeletonize`(hint_generator.py:34) 런타임 import. Colab 존재, 로컬 py_compile 무관.

---

## 테스트 (Colab 전용, 새 테스트코드·pytest 금지 — CLAUDE.md)

- **로컬 정적**: `python -m py_compile scripts/aroma/build_train_jsonl.py`. import 헬퍼 해석 확인.
- **경로 assertion 셀**: morphology_features.csv 존재 + output_dir 쓰기가능.
- **소규모 smoke(Colab)**: 한 도메인(예 `mvtec_cable`/`isp_LSM_1`) morphology 앞 몇 행만 → train.jsonl 라인수·키 확인 + target/hint PNG 육안(hint가 R=결함/G=구조/B=텍스처 반영 그레이스케일, target crop이 결함 포함).
- **로더 정합(Colab)**: `train_controlnet.py::run_sanity_check`(:323)를 생성 데이터에 실행 → shape/NaN/range 통과(기존 스크립트 활용, 신규 테스트 아님).
- 환경변수 `$VAR`, `!python` 접두사 (colab-execution.md). self-contained 자체 env 매핑(CASDA env표 재사용 금지).

---

## 확정된 결정 (실데이터 분석 후 — 구 TODO 해소)

1. ✅ **defect_subtype** = morph_label (candidates 조인, data-driven). 'general' 고정/quality_proxy 파생 아님.
2. ✅ **background_type** = 결함별 own 이미지 context에서 `_derive_background_type` 유도 (context_features + bin_edges). 도메인 단일 상수 아님.
3. ✅ **stability_score/suitability_score** = ctx_prior 유도 (candidates, 결함당 max). 고정상수 아님.
4. ✅ hint 생성기 = `HintImageGenerator` 재사용 (신규 작성 불요). skimage Colab 확인만.
5. ✅ linearity>0.7 skeletonize 분기 재현 불요 (클래스 내부 구현, hint_generator.py:32-37).
6. ✅ hint 채널 = 가중 그레이스케일 병합 3채널(정보채널, literal RGB 아님) — 재사용으로 충족.

## 미확정 사항 (TODO — 착수 중 결정)

1. **candidate 조인 miss fallback**: 확인상 92/92 커버라 miss 없음. 그래도 방어적으로 — candidates에 없는 결함(향후 도메인) 시 cluster_id/ctx_prior 부재 → (a) skip+count / (b) morph GMM 재할당·ctx_prior=중립값. **skip+count 권장(정직).**
2. **context patch 집계 범위**: 결함 image_id의 전체 defect patch mean vs bbox 겹치는 patch만(patch_xy 그리드 매핑). 전체 mean 단순·견고 권장, bbox-국소는 정밀하나 patch 그리드 크기 필요. **전체 mean 우선.**
3. **경로 형식**: jsonl 절대(Drive) vs output_dir 상대. 절대 권장(로더 최우선). 재루팅 시 `--image_root` 상대.
4. **negative_prompt/source**: 로더 미사용. negative=`generate_negative_prompt()` 채우되 학습 무효과 명시. source=target 복제(레거시). 이견 없으면 이대로.
5. **bin_edges 소스 파일**: `recommended_config.yaml`(context.bin_edges) vs `compatibility_matrix.json`(bin_edges) — 동일값 확인됨. 하나 택(config.yaml 권장, 사람이 읽기 쉬움).
