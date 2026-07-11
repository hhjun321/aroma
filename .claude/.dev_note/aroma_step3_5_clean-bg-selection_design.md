I have all the ground findings and the dev_note. Here is the concrete implementation design.

---

# clean_bg_selection.py — 구현 설계 (Step 3.5)

> **정직성 앵커** (E1/E2/V2, dev_note §정직성): 본 모듈의 **확실한 가치 = 재현성 + 대칭 대조군(random arm에 동일 배경 배정) + per-seed placement variance 제거**. 히스토그램 교차 랭킹의 변별력은 **도메인 조건부**(aitex lift +0.78 강, severstal/mtd ≈0 랜덤과 구분 불가). placement는 geometry-blind(E2). context 신호 전반 약함(V2, I/H 1~4%). **일반적 mAP 향상은 주장하지 않는다** — docstring·로그·summary.md 전반에 이 문구를 심는다.

---

## 1. 모듈 스켈레톤 (roi_selection.py 미러)

### 1.1 파일 상단 (verbatim 복사 대상)

`roi_selection.py`의 top-to-bottom 레이아웃을 그대로 미러한다.

- **module docstring** (1–26 미러): 제목·공식·전략·`Usage (Colab)` `!python $VAR` 블록·`Outputs` 목록. **정직성 문구 포함**.
- `from __future__ import annotations` (27) — `int | None` 유니온 사용하므로 필수.
- imports (29–42): `argparse, csv, hashlib, json, logging, math, os, sys` + `from collections import Counter, defaultdict` + `from pathlib import Path` + `from typing import ...` + `import numpy as np`.
- logging bootstrap (44–49): `logger = logging.getLogger("aroma.clean_bg")`.
- **`_bootstrap_aroma_ref()`** (56–79) + `_REF_SOURCE = _bootstrap_aroma_ref()` (82) — **verbatim 복사** (같은 depth `scripts/aroma/`, `parents[2]`=repo root).
- **`load_json`/`save_json` import-or-fallback** (84–94) — verbatim (`json.dump(..., indent=2, ensure_ascii=False)`).
- **optional-dependency gate** (115–128): `try: from utils.suitability import SuitabilityEvaluator` → `_HAS_QUALITY`, no-op on ImportError. 기준 3(bg-type) 재사용.

### 1.2 함수 목록 + 시그니처

```python
# --- CSV/bbox 원시 헬퍼 (verbatim 복사) ---
def _read_csv_rows(path: Path) -> List[Dict[str, str]]                    # copy 192–196
def _parse_bbox(bbox_str: str) -> list | None                            # copy 273–288 (size-fit §1-a.2)

# --- 이산화 primitive (compatibility_matrix 재사용, 재유도 금지) ---
def _load_discretizer(compat: Dict) -> Tuple[List[str], List[List[float]]]
    # compat["context_features"](5개) + compat["bin_edges"] 반환. n_context_bins=3.
def _context_cell_key(feats: Dict[str,float], names, bin_edges) -> str
    # distribution_profiling._context_cell_key(472) 로직 미러 → "0_0_0_0_0" 형태 5자리 셀키.

# --- 입력 로드 (status dict, 절대 raise 안 함; 199–239 미러) ---
def load_inputs(profiling_dir: str, roi_dir: str) -> Dict[str, Any]
    # required = [context_features.csv, compatibility_matrix.json, roi_dir/roi_selected.json]
    # 누락 시 {"status":"missing_inputs","missing":[...]}, else {"status":"ok", "context_rows":..,
    #   "compat":.., "roi":.., "good_rows":.., "defect_rows":..}
    # good_rows/defect_rows = image_type 컬럼으로 분리(context_features.csv에 image_type 존재).

# --- 히스토그램 코어 (generate_defects에서 추출, 소스=context_features NOT 픽셀) ---
def _cell_hist(cells: List[Tuple[str,bool]]) -> Dict[str,float]           # extract gen:1152 (non-void 셀 정규화 분포)
def _good_image_cells(good_rows_of_img, names, bin_edges,
                      void_fn) -> List[Tuple[str,bool]]
    # good 이미지 1장의 patch 행들 → [(cell_key, void)] (CSV 직공급; _normal_tile_cells 픽셀 경로 대체)
def _dv_bg_hist(defect_rows, names, bin_edges, void_fn,
                class_key: str | None = None) -> Dict[str,float]          # extract gen:1174
    # image_type=='defect' 행을 (선택) class로 필터 → 각 배경 패치 셀화 → 정규화 히스토그램.
    # NOTE: 프로파일링 defect 행은 이미 defect-tile 제외됨(profiling _context_worker mean>0.5, 375) → 재-제외 불필요.
def _hist_intersection(h1: Dict, h2: Dict) -> float                      # extract gen:1284 (verbatim, [0,1])

# --- §1-0 유효 풀 prefilter (void/quality 오프라인) ---
def _image_void_quality(good_rows_of_img) -> Dict[str,float]
    # per-image 집계: void_frac(local_variance≈0 & edge_density≈0 패치 비율),
    #   mean_local_variance, mean_edge_density. (데이터-유도 지표, dev_note no-hardcoding)
def valid_bg_pool(good_by_img: Dict[str,list], reject_clean_bg: bool,
                  min_quality: float | None, blur_threshold: float | None
                  ) -> Tuple[List[str], Dict[str,str], Dict[str,float]]
    # 데이터-유도 컷: 임계 None이면 관측 void_frac 분포의 percentile로 컷 유도(§설계원칙).
    # 반환: (valid_image_ids, reason_by_img, derived_thresholds)
    # ALL-reject → 전체 풀 반환(silent 0 금지, gen:load_normal_images 2685 fallback 준수) + logger.warning.

# --- §1-a 3-기준 후보 게이팅 ---
def _good_image_dim(good_rows_of_img) -> Tuple[int,int]                   # patch_xy 최대범위+64 → (W,H)
def _size_ok(defect_wh, bg_dim, clean_extent=None) -> bool               # §1-a.2 HARD gate
def _bgtype_of_image(good_cells, names, bin_edges) -> str                 # §1-a.3 (TODO 7; Phase 3, else dataset-single)
def build_candidates(data, valid_ids, background_type="directional",
                     granularity="roi") -> List[Dict[str, Any]]
    # roi_selected.json의 각 ROI × 각 valid good 이미지:
    #   size_ok       = _size_ok(_parse_bbox(defect_bbox)[2:], bg_dim)        (HARD, §1-a.2)
    #   class_fit     = _hist_intersection(good_cell_hist, _dv_bg_hist(defect_rows, class=class_axis))  (SOFT, §1-a.1)
    #   bgtype_score  = SuitabilityEvaluator.matching_score(subtype, bgtype)  (§1-a.3, TODO 7)
    #   hist_intersection = _hist_intersection(good_cell_hist, dv_hist)       (랭킹 점수 = E1 지표)
    # class_axis: class_key 사용, 단 aitex는 class_key='defect' 균일 → defect_subtype/morph_label 사용.
    # size_ok==False는 후보에서 제외(하드). 나머지는 dict로 방출(모든 3점수 기록, auditable TODO 10).
    # 모든 수치 round(...,6) (gen:385–393 규약).

# --- 랭킹·배정 (determinism tie-break; roi_selection 미러) ---
def _source_key(c) -> Tuple[str,str]                                     # copy 497–506
def _img_jitter(c) -> float                                             # copy 509–528 (blake2b, salt 없음)
def rank_and_assign(candidates, top_k, per_roi=1) -> List[Dict[str, Any]]
    # ROI별로 candidates를 hist_intersection desc + _img_jitter tie-break 정렬 → top_k pool,
    #   best per_roi개 배정. size_ok==False는 이미 build에서 제외됨. rng 없음(결정적, 대칭 arm).
    # 각 배정에 assigned_normal_id + topk_pool(재현용) 기록.

# --- 전략 dispatcher (select_rois 1364–1475 미러) ---
def select_clean_bg(candidates, strategy="histogram", top_k=..., seed=42,
                    emit_random_arm=False) -> Tuple[list, list | None]
    # "histogram": E1 default (rank_and_assign).
    # "random": np.random.default_rng(seed)로 valid pool에서 균일 배정 (대칭 대조군).
    # emit_random_arm=True면 histogram 배정과 함께 SAME candidate pool에서 random arm도 산출.
    # 반환 (symmetric_selected, random_arm | None).

# --- 요약/출력 ---
def _diversity_stats(selected) -> Dict[str,int]                          # copy 1482–1496
def _build_summary(candidates, selected, strategy, derived) -> str       # 1499–1529 미러 (:.4f), 정직성 문구 포함

def run(profiling_dir, roi_dir, output_dir, strategy="histogram", top_k=...,
        seed=42, emit_random_arm=False, granularity="roi",
        reject_clean_bg=True, min_quality=None, bg_blur_threshold=None,
        background_type="directional") -> Dict[str, Any]
    # load_inputs → (status!=ok → early return) → valid_bg_pool → build_candidates
    #   → select_clean_bg → save.
    # save_json(candidates, out/"clean_bg_candidates.json")
    # save_json(symmetric,  out/"clean_bg_selected.json")
    # if emit_random_arm: save_json(random_arm, out/"clean_bg_random_arm.json")
    # (out/"clean_bg_summary.md").write_text(_build_summary(...))
    # return {"status":"ok", "n_candidates":.., "n_selected":.., "derived_thresholds":..}

def _parse_args(argv=None) -> argparse.Namespace                        # 1638–1698 미러
def main(argv=None) -> None                                            # 1701–1719 미러; status!=ok → sys.exit(1)
if __name__ == "__main__": main()
```

### 1.3 CLI (`_parse_args`, 1638–1698 미러)

| arg | 기본값 | 근거 |
|---|---|---|
| `--profiling_dir` (required) | — | context_features.csv + compatibility_matrix.json |
| `--roi_dir` (required, NEW) | — | roi_selected.json 소재 (template `--prompts_dir` 대체) |
| `--output_dir` (required) | — | |
| `--seed` | 42 | random arm 결정성 |
| `--top_k` | **데이터-유도**(valid pool 크기 percentile) 아니면 200 | dev_note no-hardcoding — `_POS_TOPK=8` 복제 금지 |
| `--sampling_strategy` | `histogram` (choices histogram/random) | 1648 미러 |
| `--emit_random_arm` | `store_true` | TODO 4 대칭 대조군 |
| `--granularity` | `roi` (choices roi/cell) | TODO 1 |
| **`--reject_clean_bg`** | `True` | §1-0 승계 — gen `load_normal_images` 동일 시맨틱 |
| **`--min_bg_quality`** (→ min_quality) | **None → 데이터-유도** | 승계, 임계 일관 |
| **`--bg_blur_threshold`** | **None → 데이터-유도** | 승계 |
| `--background_type` | `directional` (choices smooth/directional/periodic/organic/complex) | 1687–1691 복사; per-image 유도는 Phase 3, 아니면 dataset-single fallback |

> 승계 임계 3종은 **데이터-유도 기본값** 지향(dev_note §설계원칙): CLI 미지정 시 관측 void_frac/variance 분포의 percentile로 유도하고, 유도값을 `derived_thresholds`로 summary·json 메타에 기록(auditable).

### 1.4 `clean_bg_selected.json` 스키마 (bare array, per-assignment dict, `round(...,6)`)

```jsonc
{
  "roi_ref": "<image_id | roi_idx | (cluster,cell) key>",   // granularity(TODO 1) 따라 결정
  "roi_idx": 0,                       // gen 소비 키 (syn_{roi_idx:05d}_{rep_idx:02d}와 정합)
  "class_key": "class1",              // aitex는 defect_subtype/morph_label 대입
  "class_axis": "class_key|defect_subtype",  // 어떤 축을 class로 썼는지 기록
  "cluster_id": 3, "cell_key": "0_1_0_2_0",
  "defect_bbox": [212,6,44,174],

  "assigned_normal_id": "good_0157",  // WHICH good image (필수)
  "assigned_normal_path": "<rglob-resolvable path>",  // image_id≠path (gen:381) → 매핑 carry (leather 충돌 TODO5)
  "position": null,                   // WHERE — Phase1/2 null(placement 생성시점 유지, E2 geometry-blind), Phase3 optional {"patch_xy":[x,y]}
  "topk_pool": ["good_0157","good_0093", ...],  // stage2 재픽·재현용

  // --- 3-기준 근거 (auditable, TODO 10) ---
  "hist_intersection": 0.7321,        // 랭킹 점수 = E1 재현 지표
  "class_fit": 0.6810,                // §1-a.1 SOFT
  "size_ok": true,                    // §1-a.2 HARD gate (false면 애초 미방출)
  "bgtype_score": 0.5,                // §1-a.3 (Phase1/2 dataset-single fallback)
  "bg_type": "directional",           // 후보 이미지 bg-type (Phase3 per-image, else dataset값)
  "valid_pool_reason": "kept|void_frac=0.02|fallback_all_reject"  // §1-0 provenance
}
```

`clean_bg_random_arm.json`(옵션): **동일 스키마**, `assigned_normal_id`만 random arm 배정. symmetric·random 두 arm이 **동일 ROI 집합**에 각각 배경을 배정 → generate_defects가 양 arm에서 json만 소비하면 배경 정체성만 다르고 나머지 동일 → **대칭 대조군 성립**.

---

## 2. 파이프라인 (모듈 내부) — 프로파일링 파생 소스 명시

### §1-0 void/quality 필터 (오프라인, context_features good 행)

- **소스**: `context_features.csv` `image_type=='good'` 행. 데이터-가용성 확인: aitex good 행에 `local_variance,edge_density = 0.0,0.0` 인 void 타일이 **직접 관측됨**.
- **유도**: per-image로 patch 집계 → `void_frac` = (local_variance≈0 & edge_density≈0 패치 비율), `mean_local_variance`, `mean_edge_density`.
- **컷**: `--min_bg_quality`/`--bg_blur_threshold` 미지정 시 관측 void_frac 분포의 percentile로 컷 유도(no-hardcoding). void_frac 높음/저-variance = flat/void → 배제.
- **fallback**: 전부 배제 → 전체 풀 반환 + warning (gen `load_normal_images` 2685 정책, `apply_quality_gate` all-fail 448–453 미러). **silent 0 금지**.
- ⚠️ **정직성/TODO 11**: `_is_clean_background`(486)는 픽셀 기반이고 `_background_quality_score`(480)는 CSV에 없는 brightness/noise/**blur** 항을 가중 → 오프라인 유도는 **재현이 아닌 재-유도**. blur는 특히 under-captured(흐린 non-flat 이미지도 edge_density 중간 가능). **byte-identical 아님** — E1-style 검증 전엔 픽셀 게이트를 fallback으로 유지.

### §1-a 3-기준 (size-fit HARD → bg-type + class-fit SOFT)

**(2) size-fit — HARD gate.**
- 소스: `roi_selected.json` `defect_bbox`(w,h) + 배경 dim = good 이미지 `patch_xy` 최대범위 +64 (per-ds: aitex 256×256 / mtd 576×384 / severstal 1600×256 / leather 1024×1024, 데이터-가용성 확인). `_parse_bbox`(273) 재사용.
- 판정: crop (w,h)가 배경 dim에 안 들어가면 후보 제외. ⚠️ patch_xy-max는 **per-dataset max**(per-image dim 아님) — 이 4개 셋은 dataset-uniform이라 근사 안전하나, **non-void 청정영역 범위**는 patch_xy-max로 못 잡음 → §1-0 void map으로 보완(TODO 8: profiling이 per-image dim emit하면 더 정확).

**(1) class-fit — SOFT score.**
- 소스: `context_features.csv` `image_type=='defect'` 행을 **class별 집계** → class-conditioned `_dv_bg_hist`. class 축: `class_key`(mtd/severstal/leather 다중), **aitex는 `defect_subtype`/`morph_label`** (class_key='defect' 균일). image_id→class 조인은 `roi_selected.json`.
- 판정: 후보 good 셀 히스토그램 ↔ 클래스 실제 배경 분포의 `_hist_intersection`. E1 소스-단위 매칭을 class-조건부로 일반화(E2 정합).
- ⚠️ V2: 변별력 도메인-조건부(aitex 강) → **소프트 스코어, 하드 컷 금지**. ⚠️ **leather 무효**(image_id stem 충돌, TODO 5) — phase0 fix 전엔 leather class-집계 제외.

**(3) bg-type match — score (Phase 3).**
- 소스: **미보존** — per-good-image bg-type 라벨 없음(`--background_type` dataset-single, 1687 확인). `ctx_label`은 **defect-side** NL 문자열이라 good bg-type 소스 아님(사용 금지).
- 유도(TODO 7): good 이미지 patch 집계 → 이산화(compat.bin_edges) → cell → {smooth/directional/periodic/organic/complex} 매핑 규칙 신설. `SuitabilityEvaluator.matching_score`(:164)를 dataset-single → per-image 확장.
- Phase 1/2 fallback: dataset-single `--background_type` (현행), `bgtype_score` 상수. V2 약신호 경고.

**랭킹·배정**: size_ok 통과 후보 → `hist_intersection` desc + `_img_jitter`(blake2b tie-break) 정렬 → top_k → best per_roi 배정. **rng 없음**(대칭 arm 결정적). random arm만 `np.random.default_rng(seed)`.

**이산화 공유**: (1)(3)(4) 모두 `compatibility_matrix.bin_edges`(n_context_bins=3, 5 feature→5자리 cell_key) 재사용 — bin 재유도 금지.

> ⚠️ **discretization 갭 (검증 항목 2)**: profiling은 `i*gs`/`range(h//gs)` **non-overlapping, 원경계 truncate**(368–371); gen `_dv_bg_hist`/`_normal_tile_cells`는 `dim-tile` 원경계 **항상 포함**(1235/1071). 64로 안 나뉘는 이미지에서 타일 집합 불일치 → 히스토그램 미세 차. GRID_SIZE=64==_COMPAT_TILE=64 정렬은 되나 anchor 규칙 다름. aitex +0.78 재현 검증이 이 갭의 guard.

---

## 3. generate_defects.py 통합

### 3.1 로드 splice (~2779–2789)

`roi_selected.json` 로드(2779) + `load_normal_images`(2784) 직후:
```python
clean_bg_path = Path(roi_dir) / "clean_bg_selected.json"
clean_bg_map = None
if clean_bg_path.exists():
    entries = load_json(clean_bg_path)          # list of assignment dicts
    clean_bg_map = {e["roi_idx"]: e for e in entries}   # roi_idx 키 (syn_{roi_idx:05d} 정합)
# missing → None → 전체 fallback (2775 missing-roi 패턴 미러, non-fatal)
```
`assigned_normal_id`→path 매핑: `load_normal_images` rglob 풀에 존재해야 함. image_id=`{defect_type}_{stem}`(381)는 path 아님 → json의 `assigned_normal_path` 사용(leather 충돌 TODO 5).

### 3.2 선정 splice (3062–3100)

- 3062–3063 `compat_row` 조회 **유지** (within-image `_positive_place`에 여전히 필요, 3073 주석대로).
- **3076–3098** (`if image_rank_on:` 브랜치: `_dv_hist_for`→`_scored`→cache→`_rank_normals` @3091) = **json이 대체**:
  ```python
  entry = clean_bg_map.get(roi_idx) if clean_bg_map else None
  if entry is not None:
      normal_path = resolve(entry["assigned_normal_path"])
      stage2_pool = [resolve(p) for p in entry.get("topk_pool", [])] or normal_images
      # _dv_hist_for / _dv_bg_hist / pool loop / _normal_hist_for / _hist_intersection / _rank_normals 전부 skip
  else:
      # 기존 3076–3100 로직 그대로 (image_rank_on 경로, else rng.choice)
  ```
- `dv_hist_cache`/`dv_scored_cache`/`dv_pool_cache`(2948–2950), `_dv_hist_for`/`_normal_hist_for`/`_normal_cells_for`(2952–3022)는 json 경로에서 **dead지만 fallback 위해 잔존**.
- 3101–3120 `synthesis_fn(...)` **불변**: `normal_path` 픽셀 열어 paste, `stage2_pool` stage-2 재픽. dev_note §2 "어느 normal·어디만 json에서" — 블렌딩/feather/mask는 생성시점.

### 3.3 rng 결정성 계약 (load-bearing 리스크)

**현 계약**: 단일 `rng=random.Random(seed)`(2910)를 루프 전체에 스레드. 각 (roi_idx,rep_idx)에서 **선정 슬롯 1 draw**(`_rank_normals`의 `rng.choice` @3091/1149, 또는 bare `rng.choice` @3096/3098/3100 — 브랜치 무관 정확히 1 draw, 주석 3072 "byte-identical stream") → 이후 `synthesis_fn`이 `_positive_place` 위치 샘플·feather·texture 재픽 등 추가 draw 소비.

**문제**: 선정을 offline 이관 시 선정 슬롯의 1 draw가 사라지면 **공유 스트림의 모든 후속 draw가 시프트** → placement·feather·texture 결과가 전부 바뀜. "동일 출력, precompute만" 주장 불가.

**권장 완화 (A) — placeholder draw 유지**:
```python
if entry is not None:
    normal_path = resolve(entry["assigned_normal_path"])
    _ = rng.random()      # 선정 슬롯 draw 소비 유지 → 다운스트림 스트림 정렬 보존
```
→ 배경 정체성만 offline이 바꾸고 위치/feather는 same-seed 대비 결정적 불변. **대칭 대조군이 정직해지는 핵심**(양 arm이 배경만 다르고 placement 노이즈 동일).

**대안 (B)** — per-ROI child rng `random.Random(hash((seed,roi_idx,rep_idx)))`: 장기적으로 깨끗하나 **기존 baseline 스트림 1회 이동** → re-baseline 필요할 때만.

**금지**: draw 무언 삭제 = uncontrolled placement drift → E1 재측정 오염.

offline 모듈 자체 rng(random arm의 `default_rng(seed)`)는 생성-시점 스트림과 **완전 분리** — 대칭 대조군은 symmetric·random arm에 동일 배경 배정을 emit하며 (A)/(B) 선택과 무관.

---

## 4. 단계별 계획

### Phase 1 — 추출-우선 MVP (buildable, low-risk core) ✅ 권장 시작
1. `_cell_hist`/`_dv_bg_hist`/`_hist_intersection`/`_rank_normals` 추출(context_features 소싱) + `_context_cell_key`/이산화 재사용.
2. §1-0 void 필터 오프라인(context_features good 유도, all-reject fallback).
3. `clean_bg_selected.json` emit (histogram 전략, size_ok 하드 게이트만; class_fit/bgtype는 기록만·미사용 또는 Phase 2).
4. generate_defects 소비 + **placeholder draw (완화 A)**.
5. `--emit_random_arm` 대칭 대조군.
- **의존**: 없음(aitex/mtd/severstal 즉시). **leather 제외**(TODO 5).
- **가치**: 재현성 + 대칭 계측기 — dev_note §정직성의 확실한 가치.

### Phase 2 — 3-기준 후보 결정
- §1-a (1) class-fit SOFT + (2) size-fit HARD 결합(데이터-유도 가중, TODO 9). class_axis: aitex=subtype.
- **의존**: leather class-집계는 TODO 5 fix + phase0 재실행 후.

### Phase 3 — bg-type per-image + class-conditioned geometry (E2 레버)
- (3) per-image bg-type 유도(TODO 7) + `SuitabilityEvaluator` per-image 확장.
- `position` precompute(TODO 3) + class_key edge/surface prior 배치(E2 objective 상향) — **검증 후에만 개선 주장**(V2).
- **의존**: E1/E2 재측정 통과.

---

## 5. TODO 11개 해소

| # | 권장 | 판정 |
|---|---|---|
| 1 Granularity | **`roi`(per selected-ROI) 기본**, `--granularity` cell 옵션. roi_idx 키가 gen 소비(`syn_{roi_idx:05d}`)와 직결. per-rep vs per-ROI(1이미지 공유)는 사용자 확인 필요. | **(b) 부분 FORK** — per-rep 여부 |
| 2 Objective | **추출-우선**(dev_note TODO2 명시 권장). Phase1=D_v 매칭 재현+계측기, geometry prior는 Phase3. | (a) 근거 해소 |
| 3 Position precompute | **이미지만 배정, 위치는 생성시점 유지**(Phase1/2). placement geometry-blind(E2) + discretization 갭 + rng 이중 disruption → precompute는 검증 후 Phase3. `position:null`. | (a) 해소 (Phase3 재론) |
| 4 Random-arm emit | **YES, `--emit_random_arm`**. 대칭 대조군 = 모듈의 확실한 가치. | (a) 해소 |
| 5 leather 선행의존 | leather는 phase0 image-id fix + 재실행 **전까지 제외**. Phase1은 3개 셋으로 진행. | (a) 해소(차단 명시) |
| 6 copy-paste pivot | 전제 수용, ControlNet future work. | (a) 해소 |
| 7 bg-type per-image | **Phase 3로 연기**. Phase1/2는 dataset-single `--background_type` fallback. 유도=good patch 집계→이산화→cell→type 매핑 규칙(신설, ground-truth 없음). | **(b) FORK** — 매핑 규칙 승인(무검증 유도) |
| 8 배경 크기 소스 | **patch_xy-max 근사(Phase1)** — 4셋 dataset-uniform이라 안전. 정확 per-image dim은 profiling 확장(TODO, 청정영역 fit엔 §1-0 void map 필요). | (a) 해소 |
| 9 3-기준 결합 | **(2) HARD + (1)(3) SOFT 가중 스코어**, 가중치 **데이터-유도**(관측 분리도, no-hardcoding). class-fit 소프트(V2). | **(b) FORK** — 가중 유도식 승인 |
| 10 근거 기록 | **YES** — class_fit/size_ok/bgtype_score per-candidate 기록(스키마 §1.4). auditable+재현. | (a) 해소 |
| 11 void 오프라인화 | **context_features 유도 + 픽셀 게이트 fallback 유지**. blur under-capture → E1-style 검증 전 재현 가정 금지. profiling per-image void/quality emit은 후속. | **(b) FORK** — 유도치≈픽셀게이트 검증 통과 여부 |

### 사용자에게 되물어야 할 GENUINE FORK (짧게)
1. **TODO 1**: 배경 배정 granularity — **ROI당 1이미지(모든 rep 공유)** vs **rep당 1이미지**? (다양성 vs 재현 단순성)
2. **TODO 7 (Phase 3)**: per-image bg-type 매핑 규칙을 **ground-truth 없이 유도**해 채택할지, Phase 3까지 dataset-single로 둘지.
3. **TODO 9**: 3-기준 결합 가중치를 **데이터-유도식**으로 자동 산출할지, 사용자가 명시 지정할지.
4. **TODO 11**: 오프라인 void 유도가 픽셀 게이트를 **충분히 재현**한다고 볼 임계(검증 통과 기준)를 사용자가 정할지.

---

## 6. 검증 (pytest 금지 — Colab + 로컬 재측정)

1. **Colab step3.5**: `clean_bg_selection.py`가 `clean_bg_selected.json` 4종 생성(leather 제외 시 3종).
2. **E1 재현 (aitex +0.78)**: `clean_bg_selected`의 `hist_intersection` 수치를 `AROMA연구분석/scripts_local/e1.py` 방식으로 재측정 → 생성-시점 `_dv_bg_hist`/`_hist_intersection`과 **일치 여부**. 불일치 시 §2 **discretization 갭**(non-overlap truncate vs far-edge-inclusive) 원인 조사. **이것이 context_features 소싱↔픽셀 소싱 등가성의 유일한 하드 게이트** — 통과 못 하면 Phase1 배포 보류.
3. **대칭 대조군 실현성**: `--emit_random_arm` 산출 → symmetric·random arm이 동일 ROI 집합에 각각 배경 배정, generate_defects가 양 arm에서 json 소비로 배경만 다르고 placement/feather 동일(**완화 A placeholder draw**로 same-seed 스트림 정렬)임을 syn 파일 diff로 확인.
4. **§1-0 유도 vs 픽셀 게이트(TODO 11)**: valid_bg_pool 판정을 `_is_clean_background` 픽셀 결과와 per-image 대조 — 일치율·blur 오분류 케이스 리포트. 재현 미달 시 픽셀 fallback 유지, 유도치는 기록만.
5. (E2 후속, 범위 밖) class_key edge/surface prior 배치 시 mtd/leather placement 개선 — Phase 3 검증 항목으로만 기록, **무검증 개선 주장 금지**.