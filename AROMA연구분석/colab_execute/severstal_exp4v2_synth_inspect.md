# Severstal exp4v2 합성 이미지 육안 검사 가이드

목적: exp4v2 결과에서 **모든 증강 조건(random/casda/aroma)이 baseline보다 낮고, 특히 AROMA의 c2(최희소 클래스)가 붕괴**하는 원인을 합성 이미지를 직접 보며 찾는다.

## 결과 요약 (왜 보는가)

mAP50 평균 (seed1/seed2 → avg, vs baseline):

| 조건 | seed1 | seed2 | avg | Δ baseline |
|------|-------|-------|-----|------------|
| baseline | 0.4163 | 0.3767 | **0.3965** | — |
| casda | 0.4330 | 0.3048 | **0.3689** | −0.028 |
| random | 0.3867 | 0.3369 | **0.3618** | −0.035 |
| aroma | 0.3132 | 0.3297 | **0.3215** | −0.075 |

**c2(index 1) map50** — AROMA만 붕괴:

| c2 | baseline | random | casda | aroma |
|----|----------|--------|-------|-------|
| seed1 | 0.2257 | 0.1867 | 0.2987 | **0.0675** |
| seed2 | 0.1761 | 0.1145 | 0.1019 | **0.0960** |

**역설**: c2 합성 투입량은 AROMA가 최대(seed1 429, seed2 424)인데 c2 성능 최악 → AROMA c2 합성이 학습을 **해친다**. 양 적음 아니라 **질 나쁨** 의심. 이 가이드로 그 질을 눈으로 확인.

검사 우선순위: **c2(index 1) 먼저**, 그다음 c1(index 0, AROMA seed1 0.1862 약함).

---

## Cell 0 — 환경 설정

```python
# 저장소 + Drive 마운트는 00_setup.md 따름. 여기선 env만.
import os
os.environ['DRIVE']        = "/content/drive/MyDrive/data/Severstal"  # 또는 기존 세팅 그대로
os.environ['AROMA_OUT']    = f"{os.environ['DRIVE']}/aroma_output"
# ↑ exp4v2_execute.md 와 동일 경로 체계. 본인 세션 값과 맞는지 print로 확인.

os.environ['RANDOM_SYNTH_DIR'] = f"{os.environ['AROMA_OUT']}/synthetic_random"
os.environ['AROMA_SYNTH_DIR']  = f"{os.environ['AROMA_OUT']}/synthetic"
os.environ['CASDA_SYNTH_DIR']  = f"{os.environ['AROMA_OUT']}/synthetic_casda"

# 각 조건의 합성셋: $X_SYNTH_DIR/severstal/annotations.json + (이미지: root 또는 images/) + masks/
DATASET_KEY = "severstal"
SYNTH_DIRS = {
    "random": os.environ['RANDOM_SYNTH_DIR'],
    "casda":  os.environ['CASDA_SYNTH_DIR'],
    "aroma":  os.environ['AROMA_SYNTH_DIR'],
}
for k, v in SYNTH_DIRS.items():
    p = f"{v}/{DATASET_KEY}/annotations.json"
    print(f"{k:7s}: {p}  exists={os.path.exists(p)}")
```

> ⚠️ `exists=False`면 경로 틀림. exp4v2 학습 때 쓴 `--*_synthetic_dir` 인자값과 정확히 일치해야 한다. weights_path는 `_seeds/seedN/...`이지만 **합성셋은 거기 없다** — 위 SYNTH_DIR 들에 있다.

---

## Cell 1 — 헬퍼 (클래스 분류 + 그리드 출력)

exp4v2 학습 스크립트(`exp4_v2_supervised_detection.py`)의 클래스 결정 로직을 그대로 복제한다. 그래야 "학습에 c2로 들어간 바로 그 이미지"를 본다.

```python
import json, re
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import numpy as np

def parse_severstal_class(source_roi):
    """source_roi .../class{N}/ → N-1 (0-indexed). 없으면 None."""
    if not source_roi:
        return None
    m = re.search(r"[\\/]class(\d+)[\\/]", str(source_roi)) or re.search(r"class(\d+)", str(source_roi))
    if not m:
        return None
    c = int(m.group(1)) - 1
    return c if 0 <= c <= 3 else None

def resolve_synth_class(ann):
    """학습 스크립트 _resolve_synth_class 복제: source_roi 우선, cluster_id fallback."""
    c = parse_severstal_class(ann.get("source_roi"))
    if c is not None:
        return c
    raw = ann.get("cluster_id")
    try:
        cid = int(raw) - 1
    except (TypeError, ValueError):
        return None
    return cid if 0 <= cid <= 3 else None

def resolve_path(synth_base, val):
    """학습 스크립트 _resolve_path 복제: 절대경로 → base/val → base/images/name."""
    if not val:
        return None
    p = Path(val)
    if p.is_absolute() and p.exists():
        return str(p)
    for cand in (synth_base / val, synth_base / "images" / Path(val).name):
        if cand.exists():
            return str(cand)
    return str(p) if p.exists() else None

def load_synth(cond):
    """조건별 valid annotation 로드. 각 항목에 resolved class/이미지/마스크 경로 부착."""
    base = Path(SYNTH_DIRS[cond]) / DATASET_KEY
    raw = json.loads((base / "annotations.json").read_text())
    entries = raw.get("annotations") or raw.get("items") or raw.get("data") or raw if isinstance(raw, dict) else raw
    out = []
    for e in entries:
        if not isinstance(e, dict) or e.get("dry_run") is True:
            continue
        img = resolve_path(base, e.get("image_path"))
        if img is None:
            continue
        mask_raw = e.get("mask") or e.get("roi_mask") or e.get("mask_path")
        mask = resolve_path(base, mask_raw)
        if mask is None:
            stem = Path(img).stem
            for cand in (base/"masks"/f"{stem}.png", base/"masks"/f"{stem}_mask.png", Path(img).with_name(f"{stem}_mask.png")):
                if cand.exists():
                    mask = str(cand); break
        out.append({
            "cond": cond, "cls": resolve_synth_class(e), "image": img,
            "mask": mask, "normal": resolve_path(base, e.get("normal_image")),
            "source_roi": e.get("source_roi"), "cluster_id": e.get("cluster_id"),
        })
    return out

SYNTH = {c: load_synth(c) for c in SYNTH_DIRS}
for c, lst in SYNTH.items():
    print(f"{c:7s}: {len(lst)} valid")
```

---

## Cell 2 — 클래스별 개수 sanity check

결과 JSON의 `n_synth_per_class`(seed1 기준: random `{0:369,1:85,2:1860,3:220}`, casda `{0:793,1:175,2:794,3:772}`, aroma `{0:655,1:429,2:916,3:534}`)와 대조. 학습 시점 cap/seed가 달라 정확히 같진 않으나 **분포 경향**이 맞아야 한다.

```python
CLASS_NAMES = {0: "c1", 1: "c2(rare)", 2: "c3", 3: "c4"}
print(f"{'cond':8s} | " + " | ".join(f"{CLASS_NAMES[i]:9s}" for i in range(4)) + " | None")
for c, lst in SYNTH.items():
    cnt = {i: 0 for i in range(4)}; none = 0
    for a in lst:
        if a["cls"] is None: none += 1
        else: cnt[a["cls"]] += 1
    print(f"{c:8s} | " + " | ".join(f"{cnt[i]:9d}" for i in range(4)) + f" | {none}")
```

> 확인: AROMA c2 개수가 random/casda보다 크게 많은가? (결과 JSON상 429 vs 85/175). `None`이 많으면 class 분류 실패 → 라벨 오염 가능성, 별도 조사.

---

## Cell 3 — c2 합성 나란히 비교 (핵심)

```python
def show_grid(cond, cls, n=12, cols=4, show_mask=True):
    items = [a for a in SYNTH[cond] if a["cls"] == cls][:n]
    if not items:
        print(f"{cond} class{cls+1}: 없음"); return
    rows = (len(items) + cols - 1) // cols
    fig, axs = plt.subplots(rows, cols, figsize=(cols*3, rows*3))
    axs = np.array(axs).reshape(-1)
    for ax in axs: ax.axis("off")
    for ax, a in zip(axs, items):
        img = cv2.cvtColor(cv2.imread(a["image"]), cv2.COLOR_BGR2RGB)
        ax.imshow(img)
        # 마스크 → bbox 오버레이 (학습에 들어간 라벨 근사)
        if show_mask and a["mask"] and Path(a["mask"]).exists():
            m = cv2.imread(a["mask"], cv2.IMREAD_GRAYSCALE)
            cnts, _ = cv2.findContours((m > 127).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in cnts:
                x, y, w, h = cv2.boundingRect(cnt)
                ax.add_patch(patches.Rectangle((x, y), w, h, fill=False, edgecolor="lime", linewidth=1.5))
        ax.set_title(Path(a["image"]).name[:18], fontsize=7)
    fig.suptitle(f"{cond} — {CLASS_NAMES[cls]}  (n={len(items)})", fontsize=12)
    plt.tight_layout(); plt.show()

# c2(index 1) 세 조건 연속 — AROMA가 왜 망가지는지 여기서 본다
for cond in ("random", "casda", "aroma"):
    show_grid(cond, cls=1, n=12)
```

**육안 체크리스트 (c2)**:
- [ ] **현실성**: 결함 텍스처가 실제 강판 결함처럼 보이는가, 아니면 흐릿/반복/붙여넣기 티가 나는가
- [ ] **mode collapse**: AROMA c2 12장이 거의 같은 패턴 반복인가 (다양성 부족)
- [ ] **라벨 정합 (초록 bbox)**: bbox가 실제 결함을 감싸는가, 빈 배경/전체 프레임을 감싸는가
- [ ] **합성 아티팩트**: copy-paste 경계선(seam), 색조 불일치, 잘린 결함
- [ ] **normal 영역 오염**: 결함 아닌 곳에 결함 라벨이 붙었는가

---

## Cell 4 — c1 및 나머지 클래스

```python
for cond in ("random", "casda", "aroma"):
    show_grid(cond, cls=0, n=8)   # c1 (AROMA seed1 약함)
# 필요시 cls=2(c3), cls=3(c4)도 동일하게
```

---

## Cell 5 — 라벨 추출 디버그 (AROMA c2 정밀 확인)

학습 스크립트는 마스크가 있으면 마스크에서, 없으면 composite vs normal **이미지 차분**으로 bbox를 뽑는다. 마스크 부재 시 차분이 엉뚱한 영역을 잡으면 라벨이 오염된다. AROMA c2 항목들의 마스크 유무를 본다.

```python
aroma_c2 = [a for a in SYNTH["aroma"] if a["cls"] == 1]
has_mask = sum(1 for a in aroma_c2 if a["mask"] and Path(a["mask"]).exists())
print(f"AROMA c2: {len(aroma_c2)}장, mask 보유 {has_mask}장, mask 없음 {len(aroma_c2)-has_mask}장")
# mask 없는 항목은 image-diff fallback으로 라벨 생성 → 오염 위험 높음

# composite vs normal 차분 시각화 (mask 없는 케이스)
def show_diff(a):
    comp = cv2.cvtColor(cv2.imread(a["image"]), cv2.COLOR_BGR2RGB)
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(comp); axs[0].set_title("composite (synth)")
    if a["normal"] and Path(a["normal"]).exists():
        norm = cv2.cvtColor(cv2.imread(a["normal"]), cv2.COLOR_BGR2RGB)
        axs[1].imshow(norm); axs[1].set_title("normal (bg)")
        d = cv2.absdiff(comp, norm).sum(axis=2)
        axs[2].imshow(d, cmap="hot"); axs[2].set_title("abs diff → bbox 근거")
    for ax in axs: ax.axis("off")
    plt.tight_layout(); plt.show()

for a in aroma_c2[:5]:
    show_diff(a)
```

> diff가 결함 영역에 집중되면 정상, 전면에 퍼지거나 결함과 무관하면 라벨 오염 확정.

---

## 판정 가이드

| 관찰 | 결론 | 다음 액션 |
|------|------|-----------|
| AROMA c2 패턴 반복(거의 동일) | ROI 선택/생성 mode collapse | roi_selection 다양성, c2 후보 부족 점검 |
| bbox가 결함 안 감쌈 / 빈 영역 | 라벨 오염 (mask 부재 + diff fallback 실패) | mask 영속화 강제, fallback 임계 조정 |
| 합성 자체 비현실(seam/색조) | copy-paste 엔진 품질 | MPB blending, 색보정 점검 |
| random/casda는 정상, AROMA만 나쁨 | ROI 모델링이 c2에서 역효과 | AROMA ROI 전략을 c2에 한해 random fallback |

검사 결과를 보고 원인이 (1)생성 품질 (2)라벨 오염 (3)다양성 붕괴 중 무엇인지 좁힌 뒤, 해당 단계 코드(stage5_quality_scoring / roi_selection / generate_defects)로 들어간다.

---

# c1 관찰: "random bbox 거대 + ROI 오목" 원인 분석

관찰(c1 그리드): **random의 bbox가 aroma/casda보다 훨씬 크고, 붙은 결함 영역이 오목(concave)**.

## 코드 추적 — bbox 크기는 어디서 결정되나

`generate_defects.copy_paste_synthesis` (모든 조건 공통 엔진):

- `use_real_mask=True`면 후보가 들고 온 **`defect_bbox`(소스 결함 크기) 그대로** 크롭→paste. 반환 bbox = 그 영역 (`generate_defects.py:382-386, 318`).
- 즉 **붙은 bbox 크기 = 선택된 소스 결함의 크기**. 합성이 키우는 게 아니라 **어떤 소스 ROI를 골랐냐**가 전부.
- `defect_mask`는 실제 GT 마스크 → 결함 형태 그대로 보존 → **오목(저 solidity) 결함이면 오목하게** 붙음. 라벨 자체는 정확.

## 왜 random만 크고 오목한가 — 선택 차이

세 조건은 **동일 paste 엔진, ROI 선택 전략만 다름**:

| 조건 | 선택 방식 | 대면적·오목 결함 운명 |
|------|-----------|----------------------|
| **random** | `roi_candidates.json` **균등 샘플** (`generate_random.py`) | 면적·solidity 무관 무차별 포함 → 큰/오목 결함 다수 통과 |
| **aroma** | `roi_score = 0.4·morph_prior + 0.4·ctx_prior + 0.2·deficit` top_k (`roi_selection.py:220-234`) | 큰 비정형 결함 = 희소 morph cluster → **낮은 morph_prior → top_k 탈락** → 작고 전형적 위주 |
| **casda** | `casda_roi_adapter` suitability≥0.5 + per_class_cap 게이트 | 저suitability(부적합) ROI 제외 |

→ **random의 큰+오목 bbox는 버그 아니라 selection bias**. roi_score/suitability 둘 다 **면적을 직접 벌점하지 않지만**, morph_prior(형태 전형성)·matching_score가 간접적으로 대형 비정형을 걸러 aroma/casda는 작고 둥근(고 solidity) 결함으로 쏠린다.

`solidity = area / convex_hull_area`. **낮을수록 오목**. `morphology_features.csv`에 `solidity`, `area`, `defect_bbox` 컬럼 존재 → 정량 확인 가능 (아래 Cell 7).

## 함의

- 라벨(bbox/mask)은 정확 → "라벨 오염"은 c1 random의 원인 아님.
- 진짜 이슈: **ROI 선택이 morphology 분포(면적·solidity)를 조건마다 크게 바꾼다**. "ROI 선택만 다르다"는 설계 의도대로지만, 분포 shift 자체가 학습 결과 교란 요인. random이 큰/오목(=어려운/다양한) 결함을 더 학습 → c1에선 오히려 도움될 수도, c2에선 노이즈일 수도. Cell 7 분포 비교로 판정.

---

## Cell 6 — ROI 후보/선택 영역 직접 검사 (한 단계 위)

합성 결과 말고 **paste 직전의 소스 ROI 영역**을 본다. 공유 후보 풀 `roi_candidates.json` + 조건별 `roi_selected.json`.

```python
# 경로: 공유 후보 풀 + 조건별 선택 결과
ROI_CANDIDATES = f"{os.environ['AROMA_OUT']}/roi/{DATASET_KEY}/roi_candidates.json"
ROI_SELECTED = {
    "random": f"{SYNTH_DIRS['random']}/{DATASET_KEY}/_random_roi/roi_selected.json",
    "aroma":  f"{SYNTH_DIRS['aroma']}/{DATASET_KEY}/roi_selected.json",
    # casda는 casda_roi_adapter 출력(아래 fallback이 annotations에서 복원)
}
for k, p in [("candidates", ROI_CANDIDATES), *ROI_SELECTED.items()]:
    print(f"{k:11s}: {p}  exists={os.path.exists(p)}")

def load_rois(path):
    if not path or not os.path.exists(path):
        return None
    raw = json.loads(Path(path).read_text())
    return raw.get("rois") or raw.get("selected") or raw.get("candidates") or raw if isinstance(raw, dict) else raw

def roi_class(r):
    """후보 ROI의 0-indexed class. source_roi 우선, cluster_id fallback."""
    c = parse_severstal_class(r.get("image_path") or r.get("source_roi"))
    if c is not None:
        return c
    try:
        cid = int(r.get("cluster_id")) - 1
        return cid if 0 <= cid <= 3 else None
    except (TypeError, ValueError):
        return None

# 조건별 선택 ROI: roi_selected.json 우선, 없으면 annotations.json의 source_roi로 복원
def selected_rois(cond):
    sel = load_rois(ROI_SELECTED.get(cond))
    if sel:
        return sel
    # fallback: annotations source_roi + 공유 후보에서 defect_bbox/mask 조인
    cands = load_rois(ROI_CANDIDATES) or []
    by_img = {c.get("image_path"): c for c in cands}
    out = []
    for a in SYNTH[cond]:
        c = by_img.get(a["source_roi"])
        out.append(c if c else {"image_path": a["source_roi"], "cluster_id": a["cluster_id"]})
    return out

SEL = {c: selected_rois(c) for c in SYNTH_DIRS}
for c, lst in SEL.items():
    print(f"{c:7s}: {len(lst) if lst else 0} selected ROIs")
```

## Cell 7 — ROI 영역 그리드 + 면적·solidity 분포 비교

소스 이미지에서 `defect_bbox` 크롭, GT 마스크 외곽선 오버레이, **area / solidity** 타이틀 표시. random이 정말 크고 오목한지 눈+숫자로 확인.

```python
def show_roi_grid(cond, cls, n=12, cols=4):
    rois = [r for r in (SEL[cond] or []) if roi_class(r) == cls and r.get("defect_bbox")][:n]
    if not rois:
        print(f"{cond} class{cls+1}: defect_bbox 가진 ROI 없음"); return
    rows = (len(rois) + cols - 1) // cols
    fig, axs = plt.subplots(rows, cols, figsize=(cols*3.2, rows*3.2)); axs = np.array(axs).reshape(-1)
    for ax in axs: ax.axis("off")
    for ax, r in zip(axs, rois):
        bb = r["defect_bbox"]
        x, y, w, h = (bb if isinstance(bb, (list, tuple)) else [int(v) for v in str(bb).split(",")])
        img = cv2.cvtColor(cv2.imread(r["image_path"]), cv2.COLOR_BGR2RGB)
        crop = img[max(0,y):y+h, max(0,x):x+w]
        ax.imshow(crop)
        sol = r.get("solidity"); area = r.get("area", w*h)
        # GT 마스크 외곽선 (오목 형태 확인)
        mp = r.get("defect_mask_path")
        if mp and os.path.exists(mp):
            m = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)[max(0,y):y+h, max(0,x):x+w]
            if m.size:
                cnts,_ = cv2.findContours((m>127).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for cnt in cnts:
                    pts = cnt.reshape(-1,2)
                    ax.plot(np.append(pts[:,0],pts[0,0]), np.append(pts[:,1],pts[0,1]), c="lime", lw=1.2)
        t = f"{w}x{h} a={area}"
        if sol is not None: t += f"\nsolidity={float(sol):.2f}"
        ax.set_title(t, fontsize=7)
    fig.suptitle(f"{cond} ROI — c{cls+1}  (n={len(rois)})", fontsize=12)
    plt.tight_layout(); plt.show()

# c1 세 조건 — random이 크고 오목한지 직접 비교
for cond in ("random", "casda", "aroma"):
    show_roi_grid(cond, cls=0, n=12)
```

```python
# 정량: 조건별 선택 ROI의 면적·solidity 분포 (관찰을 숫자로 확정)
def roi_stats(cond):
    A, S = [], []
    for r in (SEL[cond] or []):
        bb = r.get("defect_bbox")
        if bb:
            x,y,w,h = (bb if isinstance(bb,(list,tuple)) else [int(v) for v in str(bb).split(",")])
            A.append(r.get("area", w*h))
        if r.get("solidity") is not None:
            S.append(float(r["solidity"]))
    return np.array(A,dtype=float), np.array(S,dtype=float)

fig, axs = plt.subplots(1, 2, figsize=(13,4))
for cond in ("random","casda","aroma"):
    A, S = roi_stats(cond)
    if len(A): axs[0].hist(A, bins=30, alpha=0.5, label=f"{cond} (med={np.median(A):.0f})")
    if len(S): axs[1].hist(S, bins=30, alpha=0.5, label=f"{cond} (med={np.median(S):.2f})")
axs[0].set_title("defect area 분포"); axs[0].set_xlabel("area(px)"); axs[0].legend()
axs[1].set_title("solidity 분포 (낮을수록 오목)"); axs[1].set_xlabel("solidity"); axs[1].legend()
plt.tight_layout(); plt.show()
# 기대: random area median ↑, solidity median ↓ → 관찰 정량 확정
```

## Cell 8 — (선택) roi_patches 원본 패치 직접 보기

`roi_metadata.csv` + `roi_patches_v5.1/` 패치(=ROI 후보 생성 원천)를 본다. exp4v2 paste에 직접 안 쓰여도 ROI 추출 품질의 ground.

```python
import pandas as pd
ROI_META = f"{os.environ['DRIVE']}/roi_patches_v5.1"  # 또는 본인 ROI_DIR
df = pd.read_csv(f"{os.environ['AROMA_OUT']}/.../roi_metadata.csv")  # 실제 경로로 교체
print(df.columns.tolist())
sub = df[df.class_id == 1].sort_values("area", ascending=False)  # c1, 큰 것부터
fig, axs = plt.subplots(3, 4, figsize=(14,10)); axs = axs.reshape(-1)
for ax, (_, row) in zip(axs, sub.head(12).iterrows()):
    ax.axis("off")
    p = row["roi_image_path"]
    if os.path.exists(p):
        ax.imshow(cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB))
    ax.set_title(f"a={row['area']} sol={row['solidity']:.2f}\n{row['defect_subtype']}", fontsize=7)
plt.tight_layout(); plt.show()
```

> `solidity` 낮은 패치가 random에 다수 선택됐고 aroma/casda엔 적다면, "random=오목" 관찰이 선택 단계에서 발생함이 확정. 개선 방향: roi_score/suitability에 면적·solidity 정규화 항 추가하거나, random에도 동일 morphology 분포 매칭 적용(공정 비교 위해).

---

# ⚠️ 중대 보정: 학습은 5070장 전부가 아니라 cap=1013장만 썼다

exp4v2 로그:

```
Synth annotations severstal: 5070 valid ... synthetic_random
[SynthRatio] severstal/random: 0.40 x 2534 real_train -> cap=1013  (5070 -> 1013 synth)
random synth per-class (0-idx): {0: 151, 1: 34, 2: 733, 3: 95}
```

**Cell 1~7은 annotations.json 전체(5070장)를 본다 — 실제 학습 분포가 아니다.** 실제로는 `--synth_ratio 0.40` → `cap = int(2534 × 0.40) = 1013`장만 `random.Random(seed).sample()`로 뽑혀 학습됨. 클래스 붕괴의 진짜 표본은:

| 실제 학습된 synth (random, seed1) | c1 | **c2** | c3 | c4 |
|---|---|---|---|---|
| 이미지 수 | 151 | **34** | 733 | 95 |

**c2는 random에서 단 34장.** c3(733)에 4분의 3이 쏠림. 즉 cap 적용 후 **희소 클래스가 더 희소해진다**(uniform sample이 원래 분포 5070→1013 비례 축소). c2 성능 붕괴의 직접 후보. AROMA c2도 같은 방식으로 줄었는지 확인 필요.

staged YOLO 학습셋은 `tempfile.TemporaryDirectory`(`exp4_v2_supervised_detection.py:1614`)에 만들어져 **학습 후 삭제** → 직접 못 봄. 하지만 선택이 **시드 결정적**(`random.Random(seed).sample`, line 2050)이라 **정확히 재구성** 가능.

---

## Cell 9 — 학습에 실제 쓰인 정확한 1013장 재구성 (시드 동일)

학습 스크립트 함수를 그대로 import → 동일 순서·동일 시드 → byte-identical 표본.

```python
import sys
sys.path.insert(0, "/content/AROMA/scripts/aroma")   # AROMA_SCRIPTS 와 일치
import random as _random
from experiments.exp4_v2_supervised_detection import _load_synth_annotations, _resolve_synth_class

# 로그에서 읽은 실제 값 — 본인 run 로그와 일치시킬 것
SEED          = 1       # === SEED 1 === / SEED 2 면 2
SYNTH_RATIO   = 0.40
N_REAL_TRAIN  = 2534    # "real defect split — train=2534"
CAP = max(1, int(N_REAL_TRAIN * SYNTH_RATIO))   # = 1013
print("cap =", CAP)

def reconstruct_trained(cond, synth_root):
    # _load_synth_annotations: annotations.json valid 항목을 파일 순서대로 (학습과 동일)
    anns = _load_synth_annotations(synth_root, DATASET_KEY)
    if len(anns) > CAP:
        anns = _random.Random(SEED).sample(anns, CAP)   # line 2050 와 동일
    return anns

TRAINED = {
    "random": reconstruct_trained("random", SYNTH_DIRS["random"]),
    "casda":  reconstruct_trained("casda",  SYNTH_DIRS["casda"]),
    "aroma":  reconstruct_trained("aroma",  SYNTH_DIRS["aroma"]),
}

# 검증: 로그의 per-class 수와 일치해야 한다
print(f"{'cond':8s} | c1 | c2 | c3 | c4 | tot")
for c, lst in TRAINED.items():
    cnt = {i:0 for i in range(4)}
    for a in lst:
        k = _resolve_synth_class(a)
        if k is not None: cnt[k]+=1
    print(f"{c:8s} | {cnt[0]:3d}|{cnt[1]:3d}|{cnt[2]:3d}|{cnt[3]:3d}| {len(lst)}")
# random 줄이 {151, 34, 733, 95} 와 맞으면 재구성 정확 (안 맞으면 SEED/RATIO/N_REAL_TRAIN 재확인)
```

> 일치 확인이 이 셀의 목적. 일치하면 아래 그리드는 **YOLO가 실제로 본 바로 그 이미지들**이다.

## Cell 10 — 실제 학습 표본 per-class 그리드 (c2 집중)

```python
def show_trained_grid(cond, cls, n=16, cols=4):
    items = [a for a in TRAINED[cond] if _resolve_synth_class(a) == cls]
    print(f"{cond} c{cls+1}: 학습에 실제 {len(items)}장")
    items = items[:n]
    if not items: return
    rows = (len(items)+cols-1)//cols
    fig, axs = plt.subplots(rows, cols, figsize=(cols*3, rows*3)); axs=np.array(axs).reshape(-1)
    for ax in axs: ax.axis("off")
    for ax, a in zip(axs, items):
        ax.imshow(cv2.cvtColor(cv2.imread(a["image_path"]), cv2.COLOR_BGR2RGB))
        mp = a.get("mask_path")
        if mp and Path(mp).exists():
            m = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
            cnts,_ = cv2.findContours((m>127).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in cnts:
                x,y,w,h = cv2.boundingRect(cnt)
                ax.add_patch(patches.Rectangle((x,y),w,h,fill=False,edgecolor="lime",lw=1.5))
        ax.set_title(Path(a["image_path"]).name[:16], fontsize=7)
    fig.suptitle(f"[TRAINED] {cond} — c{cls+1} (n={len(items)})", fontsize=12)
    plt.tight_layout(); plt.show()

# c2 = 실제 학습 표본. random 34장 전부 vs casda vs aroma
for cond in ("random","casda","aroma"):
    show_trained_grid(cond, cls=1, n=16)

# c1도 (Cell 3의 full-pool 관찰을 실제 표본에서 재확인)
for cond in ("random","casda","aroma"):
    show_trained_grid(cond, cls=0, n=12)
```

**판정 추가** (실제 표본 기준):

| 관찰 | 결론 |
|------|------|
| 학습된 c2 표본 수 자체가 너무 적다(random 34) | cap+uniform이 희소 클래스 학습신호 소멸 → ratio/per-class 균형 샘플링 필요 |
| 적은 c2 표본마저 품질 나쁨/반복 | 양·질 동시 문제 |
| AROMA c2 학습수가 random보다 많은데도 성능 낮다 | AROMA c2 표본 품질 문제 (양 아님) — Cell 7 ROI 분포로 교차 확인 |
| 세 조건 c2 표본이 거의 동일 | 합성 차이 아닌 cap 샘플링이 지배 → 실험 설계상 synth_ratio 재고 |

> 이 셀이 가장 충실한 분석: 추정 풀(5070)이 아니라 **YOLO가 실제 학습한 표본**을 본다. seed2도 `SEED=2`로 바꿔 반복.

