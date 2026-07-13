# clean_bg void-floor 수정 재실행 + 검증 (48% black 합성 제거) Colab 실행

> **목적**: `scripts/aroma/clean_bg_selection.py`의 **void-floor 재유도(p1→p15) + 절대 void_frac 컷(상대 p90→0.5)** 수정을 반영해 Step 3.5 → Step 5(AROMA arm + random arm)를 재실행하고, **합성 composite의 black-fraction이 급감**(class2 현행 평균 48% → 대폭 하락)했는지 검증한다.
> **정본 근거**: 수정 설계·측정 증거는 dev_note `.claude/.dev_note/aroma_cleanbg_void-floor-fix.md`. 본 가이드는 그 §"재실행할 Colab 스테이지"의 실행판이다.
> **수정 파일**: `scripts/aroma/clean_bg_selection.py` **단독**. `generate_defects.py`·profiling generator는 **미변경**(profiling-derived, no raw pixels 계약 유지).
> **실행 환경**: prescan·Step 3.5·copy_paste 생성 = **CPU**. (ControlNet arm을 쓸 때만 GPU — 본 검증은 copy_paste 무학습 경로로 충분.)
> **체인 위치**: phase0 → step1 → step2 → step3(roi_selection) → **step3.5(본 수정 발효)** → step5(generate_defects) → 검증.

---

## 정직성 / 사전 이해 (읽고 시작)

- **문제**: exp4v2 AROMA class2 composite가 평균 48%(최대 88%) BLACK. 로컬 실측 `corr(composite black, normal black)=1.000` → black은 paste/blend가 아니라 **배정된 normal(good) 배경**(부분/가장자리 강판)에서 100% 유래.
- **원인 2건** (모두 `clean_bg_selection.py`):
  1. `_derive_void_floors()`가 void floor를 관측 분포의 **p1**로 유도(var=0.10/edge=0.65 실측) → dark-void 클러스터(var median 0.21) 아래라 black 패치의 ~1%만 잡힘. **수정: p15**(`--void_floor_pct 15`).
  2. `valid_bg_pool()`이 void_frac 컷을 관측 void_frac의 **상대 p90**으로 유도 → 구조상 항상 ~90% 유지, partial plate 완전 제거 불가. **수정: 절대 majority 컷 0.5**.
- **결정성**: 변경 지점은 percentile+비교 연산뿐, **rng 미사용**. 단 floor 상향은 **유지 normal 집합을 바꿈** → selection 변경 → 이전 실행과 **byte-identical 아님** → Step 3.5 + generate_defects **재실행 필수**(본 가이드).
- **크로스-데이터셋 워시아웃 주의**(프로젝트 메모리 교훈): severstal은 제거 의도대로. leather/mtd(textured, high-var)는 p15가 절대값이 커도 **각 데이터셋 percentile**이라 과제거 안 됨(예상). aitex(bright-but-flat, low-var)는 flag되어 제거 — 의도. **leather/mtd/aitex good 풀은 로컬에 없음** → 전역 활성화 **전** STEP 1 CPU prescan으로 데이터셋별 발동률을 반드시 확인.

> ⚠️ **게이트 필수 조건**: 수정된 void/품질 필터는 `clean_bg_selection.py`의 `reject_clean_bg=True`(기본 ON)일 때만 동작한다. **Step 3.5(STEP 2)에서 `--no_reject_clean_bg`를 절대 붙이지 말 것** — 붙이면 게이트가 꺼져 수정이 무력화된다. (`generate_defects.py`의 `--reject-clean-bg`는 별개의 생성-시점 fallback이며 JSON 경로에선 moot — 혼동 금지.)

---

## STEP 0 — 공통 환경 셀 (`step3_5_execute.md`·`step5_execute.md`와 동일 — 그대로 복사)

```python
import os, json

os.environ['DRIVE']          = '/content/drive/MyDrive/data/Aroma'
os.environ['AROMA_REF']      = '/content/AROMA'
os.environ['AROMA_SCRIPTS']  = '/content/AROMA/scripts/aroma'
os.environ['AROMA_OUT']      = f"{os.environ['DRIVE']}/aroma_output"
os.environ['DATASET_CONFIG'] = os.environ.get('DATASET_CONFIG', '/content/AROMA/dataset_config.json')
os.environ['SYM_ROOT'] = f"{os.environ['AROMA_OUT']}/sym_final"
def S(stage, ds=None):
    p = f"{os.environ['SYM_ROOT']}/{stage}"
    return f"{p}/{ds}" if ds else p

# phase0를 image_id 고유키(31ee0aa) 이상으로 재실행했으면 leather 포함(4종).
# 구 profiling만 있으면 leather 제외(3종) — STEP 2의 src_match_frac assert가 혼용을 잡는다.
DATASETS = ["severstal", "mvtec_leather", "mtd", "aitex"]

with open(os.environ['DATASET_CONFIG']) as f: CFG = json.load(f)
def normal_dir(ds): return CFG[ds]["image_dir"]     # step5와 동일 규약
```

---

## STEP 1 — CPU PRESCAN (필수 선행: 데이터셋별 void 발동률 확인)

**GPU 의존 임계는 데이터셋별 발동률 CPU 사전 스캔 후 확정**(프로젝트 규칙). Step 3.5를 전량 재실행하기 **전에**, 각 데이터셋 `context_features.csv`의 good 행으로 **수정된 p15 floor**를 유도하고 **per-image void_frac 분포(p50/p90/max)** + **절대 컷 0.5가 drop하는 이미지 수**를 출력한다. 여기서 severstal은 partial plate가 제거되는지, textured(leather/mtd)는 과제거되지 않는지, aitex 제거가 의도대로인지 확인하고 문제 있으면 데이터셋별 override를 확정한다.

> 재구현이 아니라 **실제 수정 코드의 함수**(`_derive_void_floors`, `_patch_void`)를 import해 Step 3.5와 동일한 계산을 재현한다.

```python
import csv, sys
from collections import defaultdict
import numpy as np

sys.path.insert(0, os.environ['AROMA_SCRIPTS'])   # /content/AROMA/scripts/aroma
from clean_bg_selection import _derive_void_floors, _patch_void

FLOOR_PCT      = 15.0          # 수정된 기본 percentile (Step 3.5 --void_floor_pct 기본값과 동일)
VOID_FRAC_MAX  = 0.5           # 수정된 절대 majority 컷 (Step 3.5 --void_frac_max 기본값과 동일)
# 절대 override 후보(참고 병기) — dev_note 측정치. p15가 undershoot일 때만 --var_floor/--edge_floor로 핀.
ABS_VAR, ABS_EDGE = 2.0, 3.0

def good_rows_by_img(prof_dir):
    """context_features.csv에서 image_type==good 행만 image_id로 묶는다 (load_inputs와 동일 규약)."""
    p = f"{prof_dir}/context_features.csv"
    if not os.path.exists(p):
        return None
    g = defaultdict(list)
    with open(p, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("image_type", "good") == "good":
                g[r.get("image_id", "")].append(r)
    return dict(g)

def void_frac_dist(good_by_img, var_floor, edge_floor):
    fracs = []
    for rows in good_by_img.values():
        n = len(rows) or 1
        v = sum(1 for r in rows if _patch_void(r, var_floor, edge_floor))
        fracs.append(v / n)
    return np.array(fracs, dtype=float)

print(f"{'dataset':<14} {'floor(p15) var/edge':>22} | {'void_frac p50/p90/max':>24} | {'drop@0.5':>10} | {'drop@abs(2/3)':>14}")
print("-" * 96)
for DS in DATASETS:
    gbi = good_rows_by_img(S('profiling', DS))
    if gbi is None:
        print(f"{DS:<14} context_features.csv 없음 → phase0 확인"); continue
    # (1) 수정된 p15 유도 floor
    vf, ef = _derive_void_floors(gbi, FLOOR_PCT)
    d_pct = void_frac_dist(gbi, vf, ef)
    n_drop_pct = int((d_pct > VOID_FRAC_MAX).sum())
    # (2) 절대 override 후보(참고) — 같은 데이터로 var<=2.0/edge<=3.0일 때 drop 수
    d_abs = void_frac_dist(gbi, ABS_VAR, ABS_EDGE)
    n_drop_abs = int((d_abs > VOID_FRAC_MAX).sum())
    ng = len(gbi)
    print(f"{DS:<14} {vf:>10.4g}/{ef:<10.4g} | "
          f"{np.percentile(d_pct,50):>6.2f}/{np.percentile(d_pct,90):>5.2f}/{d_pct.max():>5.2f} | "
          f"{n_drop_pct:>4}/{ng:<4} ({100*n_drop_pct/ng:>4.1f}%) | "
          f"{n_drop_abs:>4}/{ng:<4} ({100*n_drop_abs/ng:>4.1f}%)")
```

**판정 기준 (prescan)**:
- **severstal**: p15 floor로 partial plate가 `void_frac>0.5`로 잡혀 **drop@0.5 > 0**이어야 한다(로컬 참조: 절대 컷@0.5는 pool의 ~6.5%만 drop = 최악 partial plate). drop이 0에 가까우면 p15가 undershoot → `--var_floor 2.0 --edge_floor 3.0`(위 drop@abs 컬럼이 84% 분리를 보이는 값)로 severstal만 핀.
- **leather / mtd** (textured): **drop@0.5가 매우 낮아야**(과제거 아님) — p15는 각 데이터셋 하위 15% 패치만 flag하고 이미지 전반에 분산되므로 per-image void_frac « 0.5. drop 비율이 높으면(예: >20%) 워시아웃 신호 → 해당 셋 floor/컷 재검토(그 셋만 `--void_floor_pct` 하향 또는 `--void_frac_max` 상향).
- **aitex** (bright-but-flat): 저-variance라 drop 발생 가능 — **의도**. good 전부가 flat이라 all-reject 되면 Step 3.5가 full-pool로 fallback(안전, silent 0 아님).
- 어떤 데이터셋도 **severstal-튜닝 상수를 전역 하드코딩 금지**. 조정이 필요하면 그 데이터셋에만 CLI override.

---

## STEP 2 — Step 3.5 재실행 (`clean_bg_selection.py`, DATASETS 루프)

수정이 여기서 발효한다. `--emit_random_arm`으로 AROMA arm(`clean_bg_selected.json`)과 random arm(`clean_bg_random_arm.json`)에 게이트를 **대칭 적용**한다. **`--no_reject_clean_bg`를 붙이지 않는다**(기본 ON = 게이트 활성).

```python
for DS in DATASETS:
    os.environ['DS']   = DS
    os.environ['PROF'] = S('profiling', DS)   # context_features.csv, compatibility_matrix.json
    os.environ['ROI']  = S('roi', DS)         # roi_selected.json (step3 출력) — 입출력 동일 경로
    # STEP 1에서 severstal만 override가 필요하다고 나왔을 때만 아래처럼 EXTRA를 세팅(그 외 빈 문자열):
    #   os.environ['EXTRA'] = "--var_floor 2.0 --edge_floor 3.0" if DS == "severstal" else ""
    os.environ['EXTRA'] = ""
    print(f"\n===== clean_bg_selection (void-floor fix): {DS} =====")
    !python $AROMA_SCRIPTS/clean_bg_selection.py \
        --profiling_dir  $PROF \
        --roi_dir        $ROI \
        --output_dir     $ROI \
        --emit_random_arm \
        $EXTRA
```

> 기본 실행이면 `--void_floor_pct 15`(기본)·`--void_frac_max 0.5`(기본)가 자동 적용된다. STEP 1에서 특정 데이터셋만 절대 floor가 필요하면 그 데이터셋에 한해 `--var_floor`/`--edge_floor`를 `EXTRA`로 전달(percentile 유도값을 대체). `--void_floor_pct`/`--void_frac_max`로도 조정 가능.

### 2-1. 산출 + 유도 floor 확인 (수정 발효 검증)

```python
from pathlib import Path
import re
for DS in DATASETS:
    roi = S('roi', DS)
    for f in ['clean_bg_selected.json', 'clean_bg_random_arm.json', 'clean_bg_summary.md']:
        ok = Path(f'{roi}/{f}').exists()
        print(f"  {'OK  ' if ok else '누락 '} {DS}/{f}")
    sm = Path(f"{roi}/clean_bg_summary.md").read_text(encoding='utf-8')
    # summary에 기록된 유도값 — p15 floor + majority 컷 + valid/good 유지 수
    m = re.search(r'var_floor=([\d.eg+-]+)\s+edge_floor=([\d.eg+-]+).*?void_frac_max=([\d.]+).*?valid (\d+)/(\d+)', sm, re.S)
    if m:
        print(f"    → {DS}: floor var={m.group(1)} edge={m.group(2)}  "
              f"void_frac_max={m.group(3)}  valid {m.group(4)}/{m.group(5)} good")
```

> `clean_bg_summary.md`의 "Derived thresholds" 라인에 `floor_pct=15.0`·`void_frac_max=0.5000 (majority)`·`floor_source=percentile`(또는 override 시 `override`)가 찍혀야 수정 발효. `valid N/M`이 STEP 1 prescan의 "유지 이미지 수"와 일치해야 한다.

### 2-2. E1 재현 하드 게이트 (배포 전 필수 — `step3_5_execute.md` STEP 2-2와 동일)

floor 상향은 histogram 매칭 자체를 바꾸지 않아야 한다. E1 재현이 여전히 통과해야 배포 가능.

```python
# 로컬 검증 기준값 (pivot_local_validation_20260711.md)
E1_SIM_BEST = {"aitex": 0.895, "mtd": 0.502, "severstal": 0.623}
for DS in DATASETS:
    sm = open(f"{S('roi', DS)}/clean_bg_summary.md", encoding='utf-8').read()
    m = re.search(r'src_fit_ceiling_mean=([0-9.]+)', sm)
    ceil = float(m.group(1)) if m else 0.0
    ref = E1_SIM_BEST.get(DS)
    ok = ref is None or abs(ceil - ref) < 0.05
    mf = re.search(r'src_match_frac=([0-9.]+)', sm)
    frac = float(mf.group(1)) if mf else 1.0
    flag = "" if frac >= 0.999 else f"  ⚠️ image_id 매칭율 {frac:.0%} — step3 재실행 필요"
    print(f"{DS:10s} src_fit_ceiling {ceil:.3f} vs E1 {ref} → {'PASS' if ok else 'DRIFT(조사)'}{flag}")
    assert frac >= 0.999, f"[{DS}] src_match_frac={frac} — roi_selected.json이 이 profiling 산출이 아님(step3 재실행)."
```

> DRIFT(±0.05 초과)면 floor 변경이 histogram을 흔든 게 아니라 profiling/roi staleness 문제일 가능성이 높다 → `step3_5_execute.md` STEP 2-2 주석(discretization 갭·image_id 고유키) 참조.

---

## STEP 3 — generate_defects 재실행 (AROMA arm + random arm)

새 `clean_bg_selected.json`으로 composite를 재생성한다. **before/after 직접 비교**를 위해 현행(48%-black) 산출을 덮지 않고 **별도 출력 디렉토리**(`synth_aroma_voidfix`)로 생성한다. copy_paste(무학습·CPU) 경로로 검증하며, ControlNet 학습·GPU 불요.

> **주의**: `generate_defects.py`는 타 세션 소유 — 본 작업은 **수정하지 않는다**. 여기서는 실행만 한다. 명령·플래그는 `step5_execute.md` STEP 3B(copy_paste)를 그대로 따른다.

```python
DATASETS_GEN = DATASETS   # 세션 분리 시 ["severstal"] 등으로 좁힘

for DS in DATASETS_GEN:
    os.environ['DS']     = DS
    os.environ['ROI']    = S('roi', DS)
    os.environ['NORMAL'] = normal_dir(DS)
    os.environ['OUT']    = S('synth_aroma_voidfix', DS)   # ← 신규(after). 현행 synth_aroma는 보존(before)
    print(f"\n===== AROMA copy_paste 재생성 (void-fix): {DS} =====")
    !python $AROMA_SCRIPTS/generate_defects.py \
        --roi_dir     $ROI \
        --normal_dir  $NORMAL \
        --output_dir  $OUT \
        --method      copy_paste \
        --n_per_roi 3 --seed 42 --blend_mode seamless \
        --reject-clean-bg --min-bg-quality 0.7 --bg-blur-threshold 100.0 \
        --local_staging
```

```python
# random arm (동일 clean-bg 게이트, 배경만 random) — 대칭 대조군 재생성
for DS in DATASETS_GEN:
    os.environ['DS']     = DS
    os.environ['ROI']    = S('roi', DS)
    os.environ['NORMAL'] = normal_dir(DS)
    os.environ['OUT_R']  = S('synth_random_voidfix', DS)
    print(f"\n===== random arm 재생성 (void-fix): {DS} =====")
    !python $AROMA_SCRIPTS/generate_defects.py \
        --roi_dir     $ROI \
        --normal_dir  $NORMAL \
        --output_dir  $OUT_R \
        --method      copy_paste \
        --clean_bg_json $ROI/clean_bg_random_arm.json \
        --n_per_roi 3 --seed 42 --blend_mode seamless \
        --reject-clean-bg --min-bg-quality 0.7 --bg-blur-threshold 100.0 \
        --local_staging
```

> **활성 확인(로그)**: `clean_bg assignment ON: clean_bg_selected.json (N ROIs)` + `clean_bg resolve: used=U fallback=F mismatch=M / T`(U≈T, F·M≈0)가 떠야 새 배경 배정 소비 성공. 안 뜨면 legacy 생성-시점 선정으로 fallback → `clean_bg_selected.json` 경로·존재를 먼저 확인(fallback 상태로 검증 금지).
> **compat 게이트 병용**(선택): placement까지 통제하려면 `step5_execute.md` STEP 3B처럼 `--compat_mode symmetric --compat_threshold $TAU --compat_matrix_json …`를 추가(step4c τ 필요). 본 void-floor 검증만이면 위 순수 clean_bg 경로로 충분.

---

## STEP 4 — 검증: composite black-fraction 재측정 + per-class bbox 몽타주

현행(before = `synth_aroma`)과 수정후(after = `synth_aroma_voidfix`) composite의 **클래스별 black-fraction**을 재측정한다. class2(index 1)가 현행 평균 48%에서 대폭 하락해야 한다. 이어서 클래스별 bbox 몽타주를 재출력해 육안 확인한다.

```python
import json, re
from pathlib import Path
import numpy as np, cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

DS = "severstal"            # 문제 데이터셋. 필요시 다른 셋으로 교체
BLACK_T = 15               # near-black 픽셀 임계 (0..255) — severstal void는 ~0
BEFORE = S('synth_aroma', DS)          # 현행(48%-black)
AFTER  = S('synth_aroma_voidfix', DS)  # 수정후

def resolve_path(base, val):
    if not val: return None
    p = Path(val)
    if p.is_absolute() and p.exists(): return str(p)
    for cand in (base/val, base/"images"/Path(val).name):
        if cand.exists(): return str(cand)
    return str(p) if p.exists() else None

def load_anns(root):
    base = Path(root)
    ann = base / "annotations.json"
    if not ann.exists(): return base, []
    raw = json.loads(ann.read_text())
    items = (raw.get("annotations") or raw.get("items") or []) if isinstance(raw, dict) else (raw if isinstance(raw, list) else [])
    return base, [e for e in items if isinstance(e, dict) and e.get("dry_run") is not True]

def parse_severstal_class(source_roi):
    if not source_roi: return None
    m = re.search(r"[\\/]class(\d+)[\\/]", str(source_roi)) or re.search(r"class(\d+)", str(source_roi))
    if not m: return None
    c = int(m.group(1)) - 1
    return c if 0 <= c <= 3 else None

def resolve_synth_class(e):
    c = parse_severstal_class(e.get("source_roi"))
    if c is not None: return c
    try: cid = int(e.get("cluster_id")) - 1
    except (TypeError, ValueError): return None
    return cid if 0 <= cid <= 3 else None

def black_frac_by_class(root):
    """composite 전체 픽셀 중 near-black 비율을 클래스별로 집계."""
    base, items = load_anns(root)
    by_cls = {i: [] for i in range(4)}
    for e in items:
        img_p = resolve_path(base, e.get("image_path"))
        if not img_p: continue
        img = cv2.imread(img_p, cv2.IMREAD_GRAYSCALE)
        if img is None: continue
        bf = float((img <= BLACK_T).mean())
        c = resolve_synth_class(e)
        if c is not None: by_cls[c].append(bf)
    return by_cls

CLASS_NAMES = {0: "c1", 1: "c2(rare)", 2: "c3", 3: "c4"}
bf_before = black_frac_by_class(BEFORE)
bf_after  = black_frac_by_class(AFTER)

print(f"=== {DS}: composite black-fraction (pixels<= {BLACK_T}) 클래스별 평균 ===")
print(f"{'class':10s} | {'BEFORE avg/max (n)':>26} | {'AFTER avg/max (n)':>26}")
print("-" * 70)
for c in range(4):
    b, a = bf_before[c], bf_after[c]
    bs = f"{100*np.mean(b):5.1f}% / {100*np.max(b):5.1f}% ({len(b)})" if b else "N/A"
    as_ = f"{100*np.mean(a):5.1f}% / {100*np.max(a):5.1f}% ({len(a)})" if a else "N/A"
    print(f"{CLASS_NAMES[c]:10s} | {bs:>26} | {as_:>26}")
```

> **합격 기준**: **class2(c2) AFTER 평균 black-fraction이 BEFORE(≈48%) 대비 대폭 하락**(예: <10%대). max도 88% 근처에서 크게 떨어져야 한다. 다른 클래스도 하락 또는 유지(악화 없음). AFTER가 여전히 높으면 STEP 1 prescan으로 돌아가 severstal에 `--var_floor 2.0 --edge_floor 3.0` 핀이 필요한지 재확인.

```python
# class2 몽타주 재출력 (before vs after) — bbox 오버레이 + 이미지별 black% 표기
def show_class_grid(root, cls, title, n=12, cols=4):
    base, items = load_anns(root)
    items = [e for e in items if resolve_synth_class(e) == cls]
    items = items[:n]
    if not items:
        print(f"{title}: class{cls+1} 항목 없음"); return
    rows = (len(items) + cols - 1) // cols
    fig, axs = plt.subplots(rows, cols, figsize=(cols*3, rows*3)); axs = np.array(axs).reshape(-1)
    for ax in axs: ax.axis("off")
    for ax, e in zip(axs, items):
        img_p = resolve_path(base, e.get("image_path"))
        if not img_p: continue
        gray = cv2.imread(img_p, cv2.IMREAD_GRAYSCALE)
        bf = float((gray <= BLACK_T).mean()) if gray is not None else float('nan')
        ax.imshow(cv2.cvtColor(cv2.imread(img_p), cv2.COLOR_BGR2RGB))
        bb = e.get("bbox") or e.get("defect_bbox")
        if bb and len(bb) == 4:
            x, y, w, h = [int(round(v)) for v in bb]
            ax.add_patch(mpatches.Rectangle((x, y), w, h, fill=False, edgecolor="lime", lw=1.5))
        ax.set_title(f"black={100*bf:.0f}%", fontsize=8)
    fig.suptitle(f"{title} — {CLASS_NAMES[cls]} (n={len(items)})", fontsize=12)
    plt.tight_layout(); plt.show()

show_class_grid(BEFORE, 1, "[BEFORE] AROMA")   # 검은 배경 위 결함이 다수 보여야(문제 재현)
show_class_grid(AFTER,  1, "[AFTER] AROMA")    # 검은 배경 급감 확인
```

> **육안 체크**: BEFORE 몽타주에는 검은/평탄 배경 위에 붙은 결함이 다수(black% 높음), AFTER에는 강판 표면 위 결함으로 바뀌어 black%가 낮아야 한다. bbox(초록)는 두 경우 모두 결함을 감싸야 한다(라벨 정합은 배경 교체와 무관하게 유지).

---

## 판정 / 다음 단계

- [ ] STEP 1 prescan: severstal drop@0.5 > 0(partial plate 제거), leather/mtd drop 낮음(과제거 없음), aitex 의도대로. override 필요 데이터셋 확정.
- [ ] STEP 2: 4종 `clean_bg_selected.json` + `clean_bg_random_arm.json` 재생성, summary에 `floor_pct=15.0`·`void_frac_max=0.5 (majority)` 기록. **E1 재현 PASS**(±0.05).
- [ ] STEP 3: generate_defects 로그에 `clean_bg assignment ON` + `used≈T`. AROMA·random arm 모두 재생성.
- [ ] STEP 4: **class2 composite black-fraction이 48% → 대폭 하락**. 몽타주에서 검은 배경 급감 육안 확인.

통과 시 → (요청 시) exp4v2 YOLO 학습/평가로 black-composite 제거의 **mAP 영향** 측정. 단 학습·성능 측정은 **사용자 명시 요청 시에만**(부하/성능 자동 실행 금지 정책). 검증이 통과하면 `synth_aroma_voidfix`를 정본 `synth_aroma` 경로로 승격(또는 exp*의 `--aroma_synthetic_dir`를 voidfix 경로로 지정)한다.

---

## 무결성 / 정직

- 수정은 `clean_bg_selection.py` 단독. `generate_defects.py`·profiling generator 미변경(profiling-derived, no raw pixels 계약).
- 임계·컷은 데이터-유도(p15 percentile)/시맨틱 상수(majority 0.5) + CLI override. severstal-튜닝 상수 전역 하드코딩 금지 — 조정은 데이터셋별 CLI로만.
- floor 상향은 selection을 바꿔 이전 실행과 byte-identical 아님 → Step 3.5 + generate_defects 재실행 필수(위 STEP 2·3).
- Step 3.5 실행 시 `--no_reject_clean_bg` 금지(게이트 유지). `--emit_random_arm`으로 게이트를 두 arm에 대칭 적용.
- pytest·신규 테스트 코드 금지(CLAUDE.md) — 검증은 본 셀 실행 + E1 재현. 시간 실측·처리량 벤치 미수행(load-test policy).
```
