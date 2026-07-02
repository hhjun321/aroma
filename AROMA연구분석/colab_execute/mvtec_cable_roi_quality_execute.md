# AROMA — mvtec_cable ROI 선택 품질 분석 실행 가이드

> **목적**: mvtec_cable에서 AROMA의 ROI 선택이 **올바르게 검출/선택**되는지, 그리고 Random 선택과 **어떻게 다른지**를 분석한다.
> 대상 지표: `morphology_coverage` / `context_coverage` / `rare_pair_coverage` / `entropy` / `gini` + ROI 검출 sanity.
>
> **핵심 실험 설계 (spec 준수)**:
> - AROMA(`deficit_aware`)는 **결정론적** → seed를 바꿔도 결과 불변. **1회만** 실행한다. (blake2b는 정렬 tie-break jitter일 뿐 RNG가 아님)
> - Random arm만 **stochastic** → seed `42~46` (≥5개)로 sweep, per-metric **mean±std**를 고정된 AROMA 값과 비교한다.
> - `_equalize_budget`(기본 ON)로 두 arm의 n_selected를 맞춰 **선택 품질**(volume 아님)을 비교한다.
> - **applicability(파이프라인이 돈다) ≠ significance(random보다 유의하게 낫다)** — 이 둘을 명시적으로 분리 보고. seed<5이면 significance는 **예비/미검증(PRELIMINARY)** 라벨.

**런타임**: CPU (GPU 불필요)
**전제**: mvtec_cable에 대해 profiling / prompts / roi(AROMA `deficit_aware`) 스테이지가 완료되어 Drive에 존재.

---

## §0. 환경 매핑 + 경로 존재 검증

> ⚠️ colab-execution.md의 env 테이블은 **CASDA/Severstal 전용**이므로 여기서 재사용하지 않는다. 본 가이드는 **AROMA 자체 env 매핑**을 정의한다.
> Drive base는 `/content/drive/MyDrive/data/Aroma`, 스테이지 레이아웃은 `aroma_output/<stage>/mvtec_cable/`.

```python
import os

# --- Drive 마운트 (미마운트 시) ---
# from google.colab import drive
# drive.mount('/content/drive')

# --- AROMA 저장소 클론 후 scripts 경로 (실제 repo 이름/브랜치에 맞게 조정) ---
# !git clone --single-branch -b <branch> <AROMA_REPO_URL> /content/AROMA
os.environ['AROMA_SCRIPTS'] = "/content/AROMA/scripts/aroma"   # roi_selection.py, experiments/exp2_roi_quality.py 위치

# --- 데이터셋 키 (일반화를 위해 변수화) ---
CATEGORY   = "cable"
DATASET    = f"mvtec_{CATEGORY}"   # = mvtec_cable
os.environ['DATASET'] = DATASET

# --- Drive base + 스테이지 디렉터리 ---
os.environ['DRIVE']    = "/content/drive/MyDrive/data/Aroma"
os.environ['AROMA_OUT'] = f"{os.environ['DRIVE']}/aroma_output"
os.environ['PROFILING'] = f"{os.environ['AROMA_OUT']}/profiling/{DATASET}"
os.environ['PROMPTS']   = f"{os.environ['AROMA_OUT']}/prompts/{DATASET}"

# AROMA deficit_aware ROI (기존 산출물, exp2의 aroma_roi_dir/<dataset_key> 형태로 참조)
os.environ['ROI_AROMA_ROOT'] = f"{os.environ['AROMA_OUT']}/roi"          # {root}/{dataset_key}/roi_selected.json
os.environ['ROI_AROMA_DS']   = f"{os.environ['ROI_AROMA_ROOT']}/{DATASET}"

# Random baseline (이번에 생성). seed별로 하위 디렉터리를 둔다.
os.environ['ROI_RANDOM_ROOT'] = f"{os.environ['AROMA_OUT']}/roi_random"  # seed별: {root}_seed{S}/{dataset_key}/...

# 분석 산출물 루트
os.environ['ANALYSIS_OUT'] = f"{os.environ['AROMA_OUT']}/roi_quality_{DATASET}"

for k in ['AROMA_SCRIPTS','DRIVE','AROMA_OUT','PROFILING','PROMPTS','ROI_AROMA_DS','ROI_RANDOM_ROOT','ANALYSIS_OUT']:
    print(f"{k:16}: {os.environ[k]}")
```

### 경로 존재 검증 (fail-fast)

> colab-execution.md 테이블을 신뢰할 수 없으므로, 샘플에서 확인된 **구체 경로**가 실제 존재하는지 먼저 assert 한다.
> (Drive 미마운트 / 카테고리 오타 / 스테이지 미완료를 즉시 잡는다.)

```python
import os

checks = {
    "Drive base (dir)"            : (os.path.isdir,  os.environ['DRIVE']),
    "입력 이미지 (dir)"           : (os.path.isdir,  f"{os.environ['DRIVE']}/mvtec/{CATEGORY}/test"),
    "profiling morphology csv"    : (os.path.isfile, f"{os.environ['PROFILING']}/morphology_features.csv"),
    "profiling clusters json"     : (os.path.isfile, f"{os.environ['PROFILING']}/morphology_clusters.json"),
    "profiling compat matrix"     : (os.path.isfile, f"{os.environ['PROFILING']}/compatibility_matrix.json"),
    "profiling deficit json"      : (os.path.isfile, f"{os.environ['PROFILING']}/deficit_analysis.json"),
    "profiling defect_masks (dir)": (os.path.isdir,  f"{os.environ['PROFILING']}/defect_masks"),
    "prompts json"                : (os.path.isfile, f"{os.environ['PROMPTS']}/prompts.json"),
    "AROMA roi_selected"          : (os.path.isfile, f"{os.environ['ROI_AROMA_DS']}/roi_selected.json"),
    "AROMA roi_candidates"        : (os.path.isfile, f"{os.environ['ROI_AROMA_DS']}/roi_candidates.json"),
}

missing = []
for name, (fn, path) in checks.items():
    ok = fn(path)
    print(f"[{'OK ' if ok else 'MISS'}] {name:28}: {path}")
    if not ok:
        missing.append((name, path))

assert not missing, (
    "누락된 입력이 있습니다. Drive 마운트 여부 / CATEGORY 값 / 스테이지 완료 여부를 확인하세요.\n"
    + "\n".join(f"  - {n}: {p}" for n, p in missing)
)
print("\n모든 필수 입력 확인 완료.")
```

> 참고: `roi_selected.json`의 `defect_mask_path`는 **profiling 스테이지**(`.../profiling/mvtec_cable/defect_masks/`)를 가리킨다 — roi 디렉터리 아래에 마스크가 있다고 가정하지 말 것 (cross-stage 참조).

---

## §1. Random ROI 기준선 생성 — ≥5개 seed sweep

`roi_selection.py`를 `--sampling_strategy random`으로 실행한다. **AROMA와 동일한 `--top_k`** 를 써야 예산이 맞는다.
샘플에서 AROMA `roi_selected.json` 길이 = **200** (기본 `--top_k 200`)이므로 여기서도 `TOP_K=200`.

> - Random branch(roi_selection.py L1396-1399)는 `rng=np.random.default_rng(seed); rng.choice(len(candidates), size=top_k, replace=False)` — seed만 결과를 바꾼다.
> - random branch가 우회하는 것은 **allocation knob**(`--img_diversity_cap`, `--class_floor`, `--per_pair_cap_frac`, `--rarity_temp`)뿐이다 — 이들은 deficit_aware/compatibility allocator 내부에서만 소비된다(select_rois의 random branch는 무시, L1384-1386). 반면 **`--min_quality`는 pre-selection 필터**로, `run()`에서 `select_rois` 호출 **이전에** `apply_quality_gate`가 적용되므로(roi_selection.py L1551 → L1558) random을 포함한 **모든 전략**의 후보 풀에 영향을 준다. 다만 본 가이드는 `--min_quality`를 전달하지 않아 기본 0.0(게이트 OFF)이므로 여기선 실질 효과가 없다.
> - `--sampling_strategy`를 생략하면 기본값 `deficit_aware`(=AROMA arm)로 떨어지므로 **`random`을 반드시 명시**.
> - 필수 입력: profiling에 `morphology_features.csv`, `morphology_clusters.json`, `compatibility_matrix.json`, `deficit_analysis.json`; prompts에 `prompts.json`. (플래그가 불확실하면 `!python $AROMA_SCRIPTS/roi_selection.py --help`로 확인)

```python
SEEDS = [42, 43, 44, 45, 46]   # ≥5개 distinct seed
TOP_K = 200                    # AROMA roi_selected.json 길이와 동일해야 함

os.environ['PROFILING_DIR'] = os.environ['PROFILING']
os.environ['PROMPTS_DIR']   = os.environ['PROMPTS']

for s in SEEDS:
    # exp2는 {random_roi_dir}/{dataset_key}/roi_selected.json 를 읽으므로
    # random_roi_dir = {ROI_RANDOM_ROOT}_seed{s}, 그 아래 {DATASET}/ 서브디렉터리를 만든다.
    out_ds = f"{os.environ['ROI_RANDOM_ROOT']}_seed{s}/{os.environ['DATASET']}"
    os.environ['RANDOM_OUT_DS'] = out_ds
    os.environ['SEED'] = str(s)
    print(f"\n=== random seed={s} → {out_ds} ===")
    !python $AROMA_SCRIPTS/roi_selection.py \
        --profiling_dir     $PROFILING_DIR \
        --prompts_dir       $PROMPTS_DIR \
        --output_dir        $RANDOM_OUT_DS \
        --sampling_strategy random \
        --top_k             200 \
        --seed              $SEED
```

### 출력 확인 + top_k 클램프 점검

```python
import json

TOP_K = 200
for s in SEEDS:
    path = f"{os.environ['ROI_RANDOM_ROOT']}_seed{s}/{os.environ['DATASET']}/roi_selected.json"
    if os.path.exists(path):
        with open(path) as f:
            sel = json.load(f)
        note = "" if len(sel) == TOP_K else f"  ⚠️ top_k={TOP_K}보다 적음 → 후보 풀이 작아 클램프됨"
        print(f"seed {s}: {len(sel):>4} ROI selected{note}")
    else:
        print(f"seed {s}: MISSING — {path}")

# AROMA arm 길이 (참조)
with open(f"{os.environ['ROI_AROMA_DS']}/roi_selected.json") as f:
    aroma_sel = json.load(f)
print(f"\nAROMA (deficit_aware): {len(aroma_sel)} ROI selected")
```

> `roi_selection.py`는 `top_k = max(1, min(top_k, len(candidates)))`로 클램프한다. 후보 풀이 200보다 작으면 200 미만이 나올 수 있고, 이는 §4의 n_selected parity 점검에서 다시 확인한다.

---

## §2. exp2_roi_quality.py 실행 — 각 random seed × AROMA

각 random seed에 대해 exp2를 1회씩 돌려 per-seed 지표 JSON을 만든다.
AROMA는 매번 동일한 `$ROI_AROMA_ROOT`를 참조하므로 **AROMA 값은 seed 간 불변**(결정론적)이다.

> - exp2 디렉터리 규약: `{aroma_roi_dir}/{dataset_key}/roi_selected.json` + `roi_candidates.json`, `{random_roi_dir}/{dataset_key}/roi_selected.json`. **flat 레이아웃 아님** — `{dir}/{dataset_key}/` 서브디렉터리 필수.
> - candidates(모든 coverage 지표의 분모 baseline)는 **aroma_roi_dir에서만** 읽는다. random_roi_dir은 `roi_selected.json`만 쓰인다.
> - `_equalize_budget`은 기본 ON(예산 동등화). `--no_equalize_budget`은 진단용으로만.
> - `--seed`는 **budget-equalization 서브샘플링에만** 영향(random ROI 재선택 아님). 여기서는 exp2 내부 seed를 42로 고정하고, 변동은 §1에서 만든 random 선택 자체에서 온다.

```python
os.environ['AROMA_ROI_ROOT'] = os.environ['ROI_AROMA_ROOT']   # {root}/{dataset_key}/...

for s in SEEDS:
    os.environ['RANDOM_ROI_ROOT_S'] = f"{os.environ['ROI_RANDOM_ROOT']}_seed{s}"
    os.environ['EXP2_OUT_S']        = f"{os.environ['ANALYSIS_OUT']}/exp2_seed{s}"
    print(f"\n=== exp2  AROMA vs random(seed={s}) ===")
    !python $AROMA_SCRIPTS/experiments/exp2_roi_quality.py \
        --aroma_roi_dir  $AROMA_ROI_ROOT \
        --random_roi_dir $RANDOM_ROI_ROOT_S \
        --dataset_keys   $DATASET \
        --output_dir     $EXP2_OUT_S
```

> 각 실행은 `$EXP2_OUT_S/exp2_results.json` (per-dataset `{aroma, random, n_equalized}`) 와 `exp2_summary.md`를 쓴다.
> `--dataset_keys`는 space-separated list이며 여기서는 `mvtec_cable` 하나만 전달한다. (`--seeds` 복수 플래그는 존재하지 않음 — 외부 driver=이 가이드가 반복 호출.)

---

## §3. 집계 + 통계 (순수 python, 새 스크립트 파일 없음)

각 seed의 `exp2_results.json`을 읽어:
1. AROMA 단일 결정론적 값
2. Random per-metric **mean±std** (≥5 seed)
3. per-metric delta = AROMA − random_mean (gini는 방향 반대)
4. **within-domain one-sample signed-rank test** (scipy.stats.wilcoxon) — 결정론적 AROMA 스칼라 대비 random-seed 분포. ⚠️ 이것은 spec의 **cross-domain paired Wilcoxon이 아니다**(아래 셀 주석 참조).
5. **applicability vs significance 분리** — 단일 도메인이므로 significance는 항상 PRELIMINARY

```python
import json, os
import numpy as np

METRICS = ["morphology_coverage", "context_coverage", "rare_pair_coverage", "entropy", "gini"]
# 방향: gini만 낮을수록 좋음
LOWER_BETTER = {"gini"}

ds = os.environ['DATASET']

# --- 각 seed 결과 로드 ---
aroma_vals   = {}                       # metric -> AROMA 값 (seed 간 동일해야 함)
random_byseed = {m: [] for m in METRICS} # metric -> [seed별 random 값]
aroma_check  = {m: set() for m in METRICS}
n_eq_byseed  = []
loaded_seeds = []

for s in SEEDS:
    p = f"{os.environ['ANALYSIS_OUT']}/exp2_seed{s}/exp2_results.json"
    if not os.path.exists(p):
        print(f"⚠️ seed {s} 결과 없음, skip: {p}")
        continue
    with open(p) as f:
        res = json.load(f)
    if ds not in res:
        print(f"⚠️ seed {s}: {ds} 키 없음, skip")
        continue
    loaded_seeds.append(s)
    n_eq_byseed.append(res[ds].get("n_equalized"))
    for m in METRICS:
        a = res[ds]["aroma"][m]
        r = res[ds]["random"][m]
        aroma_vals[m] = a
        aroma_check[m].add(round(a, 8))
        random_byseed[m].append(r)

n_seeds = len(loaded_seeds)
print(f"로드된 seed: {loaded_seeds}  (n={n_seeds})")
print(f"n_equalized per seed: {n_eq_byseed}")

# AROMA 결정론성 확인 (seed 간 값이 흔들리면 경고)
for m in METRICS:
    if len(aroma_check[m]) > 1:
        print(f"⚠️ AROMA {m} 값이 seed 간 변동함(예상: 불변) → 확인 필요: {aroma_check[m]}")
```

```python
# --- per-metric 요약 테이블 ---
REQUIRED_SEEDS = 5
print(f"\n{'지표':<22}{'AROMA':>9}{'Rand mean':>11}{'Rand std':>10}{'Δ(A-Rmean)':>12}  방향")
print("-" * 68)
summary = {}
for m in METRICS:
    a = aroma_vals[m]
    rs = np.array(random_byseed[m], dtype=float)
    rmean, rstd = float(rs.mean()), float(rs.std(ddof=1)) if len(rs) > 1 else (float(rs.mean()), 0.0)
    if len(rs) > 1:
        rmean, rstd = float(rs.mean()), float(rs.std(ddof=1))
    else:
        rmean, rstd = float(rs.mean()), 0.0
    delta = a - rmean
    better = (delta < 0) if m in LOWER_BETTER else (delta > 0)
    arrow = "AROMA 우세" if better and abs(delta) > 1e-9 else ("동률" if abs(delta) < 1e-9 else "Random 우세")
    dir_note = "(낮을수록↑)" if m in LOWER_BETTER else "(높을수록↑)"
    summary[m] = dict(aroma=a, rand_mean=rmean, rand_std=rstd, delta=delta)
    print(f"{m:<22}{a:>9.4f}{rmean:>11.4f}{rstd:>10.4f}{delta:>+12.4f}  {arrow} {dir_note}")
```

```python
# --- within-domain ONE-SAMPLE signed-rank test (spec의 paired Wilcoxon 아님) ---
# 주의: AROMA는 결정론적 단일 스칼라 a. 아래 검정은 random-seed 분포의 중앙값이
#   그 고정 스칼라와 다른지를 보는 one-sample signed-rank test다.
#   한쪽이 N개 seed 값이고 다른 쪽이 broadcast된 단일 상수이므로 "진짜 pairing"이 아니며,
#   spec §3(line 86)/Claim-Scoping이 요구하는 cross-domain PAIRED Wilcoxon과는 다르다.
#   spec의 검정: 도메인마다 1 pair(aroma_domain vs random_mean_domain)를 만들고
#   도메인 집합(family≥2) 위에서 diff 벡터 [aroma_j - random_mean_j]_j 로 검정.
#   → 단일 도메인(cable)만으로는 그 검정을 수행할 수 없다. 아래는 예비 within-domain 진단.
from scipy.stats import wilcoxon

print(f"\n=== within-domain one-sample signed-rank test (per-metric, n_seed={n_seeds}) ===")
print("  ⚠️ spec의 cross-domain paired Wilcoxon 아님 — 단일 도메인 예비 진단")
print("H0: random-seed 분포 중앙값 == 결정론적 AROMA 값.  차이 벡터 = [aroma - random_seed_i]_i")
print(f"{'지표':<22}{'W':>10}{'p-value':>12}   비고")
print("-" * 58)
for m in METRICS:
    a = aroma_vals[m]
    diffs = np.array([a - r for r in random_byseed[m]], dtype=float)
    # 방향 통일: 모든 지표에서 "양수 diff = AROMA 우세"가 되도록 gini(낮을수록 좋음)는 부호 반전.
    # 이렇게 해야 유의 결과가 나왔을 때 어느 쪽이 우세인지 diff 중앙값 부호로 일관되게 읽을 수 있다.
    if m in LOWER_BETTER:
        diffs = -diffs
    if np.allclose(diffs, 0):
        print(f"{m:<22}{'-':>10}{'n/a':>12}   모든 차이=0 (검정 불가)")
        continue
    try:
        w, p = wilcoxon(diffs)
        favors = "AROMA" if np.median(diffs) > 0 else "Random"
        flag = f"유의(p<.05, {favors} 우세)" if p < 0.05 else "미유의"
        print(f"{m:<22}{w:>10.3f}{p:>12.4g}   {flag}")
    except ValueError as e:
        print(f"{m:<22}{'-':>10}{'n/a':>12}   {e}")

print("\n※ 이것은 within-domain one-sample signed-rank 검정이다(단일 상수 vs N seed).")
print("  spec의 cross-domain PAIRED Wilcoxon은 도메인마다 1 pair를 만들어")
print("  family≥2 도메인 집합 위에서 [aroma_j - random_mean_j]_j 벡터로 검정한다.")
print("  단일 도메인(cable)에서는 그 검정을 만들 수 없으므로, 이 p-value가")
print("  무엇이든 cable 단독 significance 판정은 PRELIMINARY로 유지한다.")
```

```python
# --- applicability vs significance 분리 보고 (spec REQ-4, redteam hole) ---
print("=" * 64)
print(" 보고 요약 — 두 주장을 반드시 분리한다")
print("=" * 64)
print(" (A) APPLICABILITY : AROMA 파이프라인이 mvtec_cable에서 정상 동작")
print(f"     → exp2가 {n_seeds}개 seed 실행 모두 산출물을 생성 = 구조적으로 참")
print()
print(" (B) SIGNIFICANCE  : AROMA가 Random보다 '유의하게' 낫다")
print(f"     → cable은 **단일 도메인**이다. 위 검정은 within-domain one-sample signed-rank일 뿐,")
print("        spec의 cross-domain paired Wilcoxon(family≥2)이 아니다.")
print("     ⇒ within-domain p-value가 무엇이든 cable 단독 significance = **예비/미검증 (PRELIMINARY)**")
if n_seeds < REQUIRED_SEEDS:
    print(f"        (게다가 seed {n_seeds}개 < 요구 {REQUIRED_SEEDS}개로 within-domain 진단조차 미달.)")
else:
    print(f"        (seed {n_seeds}개 ≥ {REQUIRED_SEEDS}개로 within-domain 진단은 참고 가능하나,")
    print("         확정 significance는 cross-domain paired 검정을 갖춘 뒤에만.)")
print()
print(" ※ 하류 detection(exp4v2)은 corroboration이 아니라 secondary/contextual.")
print("   applicability를 significance/breadth로 확대 해석 금지.")
```

---

## §4. ROI 검출 sanity — "올바르게 검출되는가"의 4가지

spec REQ-5의 4가지를 점검한다:
(i) **defect-type coverage** — 선택 ROI가 기대 defect type/cluster를 고루 포괄하는가 (한 종류로 붕괴 아님)
(ii) **quality gate** — 저장되는 유일한 게이트 신호 `quality_score` 분포만 확인한다. clean-bg/flat 배경 거부는 이 JSON에서 **관측 불가**(per-ROI `background_type` 필드 부재 + 기본 `--min_quality=0.0`으로 게이트 OFF). 이 게이트가 AROMA를 random 위로 올리고 c2 붕괴를 해소하는 load-bearing 요소라는 점(project memory)은 하류 detection에서 간접 확인할 사안.
(iii) **n_selected parity** — arm 간 n_selected가 동일한가 (다르면 budget 외 사유일 수 있고 `_equalize_budget`이 이를 가릴 수 있음 → 신뢰 전 확인)
(iv) **degenerate collapse** — entropy≈0 / gini≈1 (한 cluster로 붕괴) red flag 탐지

### (i) defect-type coverage — cable 8종 defect type이 다 나오는가

> mvtec cable의 defect type은 image_path의 `.../test/<defect_type>/NNN.png`에서 파싱한다.
> (예: bent_wire, cable_swap, combined, cut_inner_insulation, cut_outer_insulation, missing_cable, missing_wire, poke_insulation — 실제 목록은 아래 코드가 데이터에서 뽑는다.)

```python
import json, os
from collections import Counter

def defect_type_of(entry):
    # image_path: .../mvtec/cable/test/<defect_type>/NNN.png
    ip = entry.get("image_path", "")
    parts = ip.replace("\\", "/").split("/")
    if "test" in parts:
        i = parts.index("test")
        if i + 1 < len(parts):
            return parts[i + 1]
    return "UNKNOWN"

def load_sel(path):
    with open(path) as f:
        return json.load(f)

# AROMA 선택
aroma_sel = load_sel(f"{os.environ['ROI_AROMA_DS']}/roi_selected.json")
aroma_types = Counter(defect_type_of(e) for e in aroma_sel)

# 후보 전체에서 관찰되는 defect type = "기대 목록"
cands = load_sel(f"{os.environ['ROI_AROMA_DS']}/roi_candidates.json")
expected_types = sorted(set(defect_type_of(e) for e in cands))
print(f"후보에서 관찰된 defect type ({len(expected_types)}종): {expected_types}\n")

# 대표 random seed 하나(첫 seed)로 비교
s0 = SEEDS[0]
rand_sel = load_sel(f"{os.environ['ROI_RANDOM_ROOT']}_seed{s0}/{os.environ['DATASET']}/roi_selected.json")
rand_types = Counter(defect_type_of(e) for e in rand_sel)

print(f"{'defect_type':<28}{'AROMA':>8}{'Random(s%d)'%s0:>12}" )
print("-" * 48)
for t in expected_types:
    print(f"{t:<28}{aroma_types.get(t,0):>8}{rand_types.get(t,0):>12}")
print("-" * 48)
print(f"{'커버된 type 수':<28}{len(set(aroma_types)&set(expected_types)):>8}"
      f"{len(set(rand_types)&set(expected_types)):>12}  / 기대 {len(expected_types)}")

missing_aroma = set(expected_types) - set(aroma_types)
if missing_aroma:
    print(f"⚠️ AROMA 선택에서 누락된 defect type: {sorted(missing_aroma)}")
else:
    print("AROMA: 모든 defect type이 최소 1개 이상 선택됨.")
```

> **해석**: "올바르게 검출"의 1차 의미 = 선택 ROI가 특정 defect type/cluster로 **붕괴하지 않고** 기대 type을 고르게 포괄. AROMA가 (deficit-aware로) 희소 type까지 잡으면 random보다 커버리지가 넓어야 한다. 단, cable은 object-centric 셋이라 deficit>0 신호가 약하면 rare_pair 이점이 작을 수 있음(§5 참조).

### (ii) quality gate — 저장 신호(quality_score)만 확인 (clean-bg는 관측 불가)

> ⚠️ **이 셀에서 clean-bg 게이트 동작은 검증할 수 없다.** 이유:
> - `roi_selected.json`/`roi_candidates.json` 엔트리 스키마(roi_selection.build_candidates, L355-372)에는 **per-ROI `background_type` 필드가 없다.** `background_type`은 quality proxy에 넘기는 **run 레벨 입력**일 뿐, 엔트리마다 저장되지 않는다(roi_selection.py L1541-1542).
> - 유일하게 저장되는 게이트 신호는 `quality_score`이며, 게이트(`apply_quality_gate`)는 `--min_quality`가 **0.0(기본, OFF)** 일 때 후보를 그대로 통과시킨다(roi_selection.py L393-398). 본 가이드는 `--min_quality`를 전달하지 않으므로 게이트는 항상 OFF다.
> - 따라서 `quality_score`는 build_candidates가 계산한 원값(대체로 균일)이며, 'black'/'flat' 배경 필터링은 이 JSON에서 **관측 불가**하다. clean-bg 게이트의 load-bearing 효과는 하류 detection(exp4v2)에서 간접 확인할 사안이지, 여기서 per-ROI로 확인할 수 없다.

```python
# 저장되는 유일한 게이트 신호 = quality_score 분포만 확인한다.
# (background_type은 엔트리에 없음 — run 레벨 입력이라 per-ROI로 관측 불가)
# 또한 --min_quality 기본 0.0 = 게이트 OFF이므로 이 분포는 필터링 결과가 아니라 원 스코어.
import numpy as np

def get(entry, *keys, default=None):
    for k in keys:
        if k in entry:
            return entry[k]
    return default

qs = [get(e, "quality_score", "quality") for e in aroma_sel]
qs = [q for q in qs if isinstance(q, (int, float))]
if qs:
    qs = np.array(qs, float)
    print(f"AROMA 선택 quality_score: n={len(qs)}  min={qs.min():.3f}  "
          f"mean={qs.mean():.3f}  max={qs.max():.3f}")
    if np.allclose(qs, qs[0]):
        print("  → 값이 균일: --min_quality가 기본 0.0(게이트 OFF)이라 필터링이 없었음(정상).")
else:
    print("quality_score 필드 없음 — roi_selected.json 스키마 확인 필요.")

print("\n※ clean-bg/flat 배경 거부는 이 JSON에서 검증 불가 "
      "(per-ROI background_type 필드 부재 + 기본 게이트 OFF).")
print("  게이트의 load-bearing 효과는 project memory상 하류 detection 결과로만 관측된다.")
```

> 필드명이 스키마와 다르면 `aroma_sel[0].keys()`를 출력해 실제 키를 확인한다.
> 실제 엔트리 키: image_id, image_path, cluster_id, cell_key, class_key, defect_subtype,
> quality_score, roi_score, morph_prior, ctx_prior, deficit, prompt, morph_label,
> ctx_label, defect_bbox, defect_mask_path (roi_selection.py L355-372).

### (iii) n_selected parity + (iv) degenerate collapse

```python
# n_selected parity: AROMA vs 각 random seed (raw 크기) + exp2가 실제 쓴 n_equalized 대조.
# n_equalized = min(len(aroma), len(random)) 로 큰 쪽을 서브샘플한 값(exp2 _equalize_budget).
# raw min보다 크게 낮으면 equalization이 실제 diversity 격차를 가리고 있다는 구체 신호.
print("=== n_selected parity (raw) + n_equalized 대조 ===")
print(f"AROMA raw: {len(aroma_sel)}")
n_eq_map = dict(zip(loaded_seeds, n_eq_byseed))   # §3에서 캡처한 seed별 n_equalized
for s in SEEDS:
    rp = f"{os.environ['ROI_RANDOM_ROOT']}_seed{s}/{os.environ['DATASET']}/roi_selected.json"
    if not os.path.exists(rp):
        print(f"  random seed {s}: MISSING")
        continue
    n_rand = len(load_sel(rp))
    raw_min = min(len(aroma_sel), n_rand)
    n_eq = n_eq_map.get(s)   # exp2 미실행 seed면 None
    parity = "" if n_rand == len(aroma_sel) else "  ⚠️ raw 불일치 → budget 외 사유(img_diversity_cap shortfall 등) 의심"
    eq_str = f"{n_eq}" if n_eq is not None else "n/a(exp2 미실행)"
    eq_flag = ""
    if isinstance(n_eq, (int, float)) and n_eq < 0.8 * raw_min:
        eq_flag = f"  ⚠️ n_equalized({n_eq}) < 0.8·raw_min({raw_min}) → equalization이 diversity 격차를 가리는 중"
    print(f"  random seed {s}: raw={n_rand}  raw_min={raw_min}  n_equalized={eq_str}{parity}{eq_flag}")
print("※ n_equalized가 raw_min보다 크게 낮으면 _equalize_budget이 유효 diversity 차이를 가릴 수 있으므로,")
print("  equalized 지표를 신뢰하기 전에 원인을 규명할 것 (spec open question #5).\n")

# degenerate collapse: §3 summary의 entropy / gini 로 판정
print("=== degenerate collapse guard (AROMA arm) ===")
ent = summary["entropy"]["aroma"]
gin = summary["gini"]["aroma"]
print(f"AROMA entropy (normalized H/log(n), nats): {ent:.4f}")
print(f"AROMA gini                               : {gin:.4f}")
if ent < 0.1:
    print("  ⚠️ entropy≈0 → 선택이 한 cluster로 붕괴했을 가능성 (red flag)")
if gin > 0.9:
    print("  ⚠️ gini≈1 → 극단적으로 불균등한 cluster 분포 (red flag)")
if ent >= 0.1 and gin <= 0.9:
    print("  OK — 명백한 붕괴 징후 없음.")
```

### 빠른 plausibility 체크 — 선택 ROI가 유효한 bbox/mask를 가지는가

```python
# 선택 ROI의 bbox/mask 경로가 유효한지 표본 점검 (필드명은 스키마에 맞게 조정)
sample = aroma_sel[:5]
print("샘플 AROMA 선택 ROI 필드:", list(aroma_sel[0].keys()))
for e in sample:
    bbox = e.get("defect_bbox")   # 스키마 실제 필드명 (roi_selection.build_candidates). [x,y,w,h] 또는 None
    mp   = e.get("defect_mask_path")
    ok_mask = (mp is not None) and os.path.exists(mp)
    print(f"  cluster={e.get('cluster_id')}  cell={e.get('cell_key')}  "
          f"bbox={bbox}  mask_exists={ok_mask}")
print("※ defect_mask_path는 profiling 스테이지(.../profiling/mvtec_cable/defect_masks/)를 가리킴.")
```

---

## §5. 해석 노트 — delta를 정직하게 읽는 법

**지표 5종과 좋은 방향**
| 지표 | 방향 | 의미 |
|------|------|------|
| morphology_coverage | 높을수록 ↑ | 선택된 고유 cluster_id / 후보 고유 cluster_id |
| context_coverage | 높을수록 ↑ | 선택된 고유 cell_key / 후보 고유 cell_key |
| rare_pair_coverage | 높을수록 ↑ (**load-bearing**) | deficit>0 인 (cluster_id, cell_key) pair 커버율 |
| entropy | 높을수록 ↑ | cluster_id 분포의 정규화 Shannon 엔트로피 (더 균등) |
| gini | **낮을수록** ↑ | cluster_id 빈도의 Gini (낮을수록 균등) |

**정직한 해석 원칙**

1. **품질이지 volume이 아니다.** coverage/entropy/gini는 n_selected에 대해 monotone이다 — 더 많이 고르면 자동으로 더 넓게 덮인다. 그래서 `_equalize_budget`(기본 ON)으로 예산을 맞춘 상태에서만 "AROMA > Random"을 **품질 주장**으로 읽는다. `--no_equalize_budget` 결과는 진단용일 뿐 헤드라인 금지.

2. **rare_pair_coverage는 deficit>0 기준(measurement)이다.** (과거 p75 기준의 collapse 버그로 1.0에 pin된 캐시 값과 비교 금지 — 재생성 필요.)
   - object-centric 셋(cable 포함)은 defect-image cell이 좁게 분포하고 good-image cell이 넓게 퍼져 compat cell의 deficit이 ~0에 수렴할 수 있다. 이 경우 후보에 **deficit>0 pair가 없어** exp2는 `rare_pair_coverage=1.0`을 (구조적 부재 의미로) 반환하고, AROMA와 Random의 delta가 0이 된다.
   - 즉 **cable에서 rare_pair delta≈0이면 "AROMA가 못 했다"가 아니라 "이 도메인엔 rare 신호가 없어 지표가 비적용"** 일 수 있다. summary에서 rare_pair가 양쪽 1.0이면 이 케이스로 주석 처리.
   - selection 쪽(roi_selection.py)의 rare 기준은 **nonzero deficit의 p75 quantile**로, measurement 기준(deficit>0)과 **의도적으로 다르다** — 통합하지 말 것.

3. **applicability ≠ significance.** (§3 재확인) mvtec_cable은 **단일 도메인**이므로 within-domain seed 수와 무관하게 significance는 항상 PRELIMINARY다. §3의 검정은 within-domain one-sample signed-rank(단일 상수 vs N seed)일 뿐 spec의 cross-domain paired Wilcoxon이 아니다. 확정 주장은 ≥5 seed within-domain **그리고** cross-domain paired Wilcoxon(family≥2, 도메인마다 1 pair)까지 갖춘 뒤에만.

4. **하류 결과를 corroboration으로 쓰지 말 것.** exp4v2 detection은 secondary/contextual이며, 알려진 정직 사실(Severstal aug≤baseline, leather는 역전, rare-oversample이 c2를 최악 클래스로 만든 backfire)을 숨기지 않고 병기.

5. **gini delta 부호 주의.** exp2_summary.md의 gini delta는 `random.gini − aroma.gini`로 계산되어(다른 4개 지표는 `aroma − random`) **양수면 AROMA가 더 낮음(우세)** 을 뜻한다. 위 §3 요약 테이블은 `aroma − random_mean`으로 계산하므로 gini는 "낮을수록 좋음"(LOWER_BETTER) 규칙으로 별도 판정했다 — summary.md와 부호 규약이 다름에 유의. §3 signed-rank 검정 셀에서는 gini diff를 부호 반전하여 5개 지표 모두 "양수 diff = AROMA 우세"로 통일했고, 유의 결과에는 어느 쪽이 우세인지(diff 중앙값 부호)를 함께 출력한다.

---

## 출력 파일

| 파일 | 내용 |
|------|------|
| `$AROMA_OUT/roi_random_seed{S}/mvtec_cable/roi_selected.json` | random 전략 선택 (seed별) |
| `$AROMA_OUT/roi_random_seed{S}/mvtec_cable/roi_candidates.json` | random 후보 (전체 스코어) |
| `$ANALYSIS_OUT/exp2_seed{S}/exp2_results.json` | seed별 AROMA vs random 5지표 |
| `$ANALYSIS_OUT/exp2_seed{S}/exp2_summary.md` | seed별 markdown 비교표 |

> 집계/통계(§3)와 sanity(§4)는 셀 출력으로만 확인한다(별도 스크립트 파일 생성 안 함).
