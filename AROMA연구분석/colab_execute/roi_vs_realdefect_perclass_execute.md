# 실제 결함 위치 vs 추천/배치 ROI — 클래스별 대조 진단 (roi_check §2 / 가이드 ITEM 4)

> **목적**: roi_check §2의 "실제 결함 이미지와 추출기의 추천 영역을 클래스별로 대조"를 시각·정량으로 수행해 **Scenario A(scorer-blind) vs Scenario B(dataset-invariance / placement-geometry 미모델링)**를 판정한다.
> **성격**: **진단 전용. 코드·파이프라인 변경 없음.** 개선 가이드(`roi_check_improvement_guide_20260708.md`) Phase 1(ITEM 4)의 시각 파트이며, 동작 변경(ITEM 1–3)보다 **먼저** 실행되는 게이트.
> **환경**: Colab (정상 이미지·결함 원본 픽셀이 Drive에만 있으므로 로컬 불가).

---

## 0. 좌표계 (로컬 검증 완료 — 그대로 사용)

`.claude/.etc/newpipe/`의 실측으로 확정된 사실:

| 필드 | 소재(파일) | 좌표계 | 형식 |
|------|-----------|--------|------|
| `defect_bbox` | `roi_selected_{asis,tobe}.json` | **원본 test 결함 이미지**(`image_path`) 기준 | **xywh 픽셀** (정규화 아님) |
| `bbox` | `annotations_synth_{aroma,random}.json` | **정상 이미지**(`normal_image`) 기준 배치 위치 | **xywh 픽셀** |

- 검산: blowhole `[128,211,9,27]`(9×27 세로), crack `[313,138,11,104]`(11×104 세로선=모서리형), fray `[0,26,605,214]`·uneven `[0,149,604,111]`(x=0, 폭 604 전면=표면형). → roi_check §2 edge/surface 직관과 일치.
- 배치는 crop 크기(w,h)를 보존하고 **위치(x,y)만 합성 시점에 결정**(placement-blind). 예: blowhole aroma 배치 `[189,140,9,27]` (source w,h=9,27 동일, 위치만 이동).
- ⚠️ **xyxy 아님**: 200개 중 col2>col0 & col3>col1이 69개뿐 → x2/y2가 아니라 w/h. 오버레이 시 `Rectangle((x,y), w, h)` 사용.

---

## 1. 환경변수 · 입력 경로

```python
import os, json
os.environ.setdefault('DRIVE', '/content/drive/MyDrive/data/Aroma')  # 실제 마운트 경로로

DS = 'mtd'
AO = f"{os.environ['DRIVE']}/aroma_output"

# 입력(둘 중 존재하는 것 사용): Drive 라이브 산출 또는 커밋된 newpipe 스냅샷
PATHS = {
    # 추천(선택)된 소스 결함 + 실제 defect_bbox — realism 선택본
    'roi_selected': f"{AO}/newpipe/{DS}/roi_selected_tobe.json",
    # 배치 결과 (arm별)
    'synth_aroma':  f"{AO}/newpipe/{DS}/annotations_synth_aroma.json",
    'synth_random': f"{AO}/newpipe/{DS}/annotations_synth_random.json",
}
# ↑ 라이브 경로가 다르면 여기만 수정. 커밋 스냅샷을 쓰려면:
#   /content/AROMA/.claude/.etc/newpipe/{roi_selected_tobe,annotations_synth_aroma,annotations_synth_random}.json
CLASSES = ['blowhole', 'break', 'crack', 'fray', 'uneven']
OUT_DIR = f"{AO}/diagnostics/{DS}/roi_vs_real"
os.makedirs(OUT_DIR, exist_ok=True)
print(PATHS); print("OUT:", OUT_DIR)
```

---

## 2. Cell 1 — edge/surface 분류기 (경계 근접 기준)

정상 이미지·결함 이미지의 **실제 dim을 읽어** bbox의 경계 근접도로 edge vs surface를 판정한다.

```python
import cv2, numpy as np, json
from collections import defaultdict

# margin: bbox가 이미지 경계로부터 이 비율 이내면 'edge'로 본다. 사전 고정값 — 결과 보고 튜닝 금지.
EDGE_MARGIN = 0.08          # min(H,W)의 8%
SPAN_FRAC   = 0.80          # bbox가 폭/높이의 80% 이상을 덮으면 'span'(전면) → edge 포함 취급

def classify_bbox(bbox, img_hw):
    """return 'edge' | 'surface' | 'span'. bbox=xywh(pixel), img_hw=(H,W)."""
    x, y, w, h = bbox
    H, W = img_hw
    if W <= 0 or H <= 0:
        return 'unknown'
    if w >= SPAN_FRAC * W or h >= SPAN_FRAC * H:
        return 'span'
    m = EDGE_MARGIN * min(H, W)
    near = (x <= m) or (y <= m) or ((W - (x + w)) <= m) or ((H - (y + h)) <= m)
    return 'edge' if near else 'surface'

def imread_hw(path):
    im = cv2.imread(path)
    return None if im is None else im.shape[:2]

def load(p):
    with open(p) as f: return json.load(f)
```

> **분류 규칙 근거**: MTD 타일에서 crack/break/fray는 모서리(경계) 근접, blowhole은 표면 내부라는 roi_check §2 가설을 그대로 정량화. `span`(fray/uneven류 전면)은 별도 범주로 두어 "표면 vs 모서리" 이분법 왜곡을 막는다. `EDGE_MARGIN`·`SPAN_FRAC`는 **사전 고정** — 원하는 결론이 나오도록 조정 금지(무결성).

---

## 3. Cell 2 — 실제 결함 위치 프로파일 (roi_selected 기준)

```python
sel = load(PATHS['roi_selected'])
real_geom = defaultdict(lambda: defaultdict(int))
missing = 0
for r in sel:
    c = r.get('class_key'); bb = r.get('defect_bbox'); ip = r.get('image_path')
    hw = imread_hw(ip)
    if hw is None: missing += 1; continue
    real_geom[c][classify_bbox(bb, hw)] += 1

print(f"[실제 결함 위치] (이미지 로드 실패 {missing}건 제외)")
for c in CLASSES:
    d = real_geom[c]; tot = sum(d.values()) or 1
    print(f"  {c:9s}  edge {d['edge']:3d} | surface {d['surface']:3d} | span {d['span']:3d}  "
          f"→ edge+span {100*(d['edge']+d['span'])/tot:.0f}%")
```

> 기대(roi_check §2 성립 시): crack/break/fray는 edge+span 비율 高, blowhole/uneven는 상대적으로 surface. 실측이 다르면 §2 가설 자체를 데이터로 반증 → 보고.

---

## 4. Cell 3 — 배치 위치 프로파일 (arm별: aroma vs random)

```python
def placement_profile(ann_path):
    ann = load(ann_path)
    geom = defaultdict(lambda: defaultdict(int))
    miss = 0
    for a in ann:
        if a.get('dry_run'): continue
        c = a.get('class_key'); bb = a.get('bbox'); nrm = a.get('normal_image')
        hw = imread_hw(nrm)
        if hw is None: miss += 1; continue
        geom[c][classify_bbox(bb, hw)] += 1
    return geom, miss

for arm, key in [('AROMA','synth_aroma'), ('RANDOM','synth_random')]:
    g, miss = placement_profile(PATHS[key])
    print(f"\n[배치 위치 — {arm}] (로드 실패 {miss}건 제외)")
    for c in CLASSES:
        d = g[c]; tot = sum(d.values()) or 1
        print(f"  {c:9s}  edge {d['edge']:3d} | surface {d['surface']:3d} | span {d['span']:3d}  "
              f"→ edge+span {100*(d['edge']+d['span'])/tot:.0f}%")
    globals()[f'place_{arm}'] = g
```

---

## 5. Cell 4 — 시각 그리드 (클래스별 오버레이, 육안 대조)

```python
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw(ax, img_path, bbox, title):
    im = cv2.imread(img_path)
    if im is None:
        ax.set_title(f"{title}\n(로드 실패)"); ax.axis('off'); return
    ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    x, y, w, h = bbox
    ax.add_patch(patches.Rectangle((x, y), w, h, fill=False, edgecolor='red', linewidth=2))
    ax.set_title(title, fontsize=8); ax.axis('off')

# 클래스별 대표 1건: 실제 결함(왼) | AROMA 배치(중) | RANDOM 배치(우)
sel_by_c = defaultdict(list)
for r in sel: sel_by_c[r['class_key']].append(r)
aroma_by_c = defaultdict(list); random_by_c = defaultdict(list)
for a in load(PATHS['synth_aroma']):
    if not a.get('dry_run'): aroma_by_c[a['class_key']].append(a)
for a in load(PATHS['synth_random']):
    if not a.get('dry_run'): random_by_c[a['class_key']].append(a)

fig, axes = plt.subplots(len(CLASSES), 3, figsize=(11, 3.2*len(CLASSES)))
for i, c in enumerate(CLASSES):
    if sel_by_c[c]:
        r = sel_by_c[c][0]; draw(axes[i,0], r['image_path'], r['defect_bbox'], f"{c} · REAL defect (source)")
    if aroma_by_c[c]:
        a = aroma_by_c[c][0]; draw(axes[i,1], a['normal_image'], a['bbox'], f"{c} · AROMA placement")
    if random_by_c[c]:
        a = random_by_c[c][0]; draw(axes[i,2], a['normal_image'], a['bbox'], f"{c} · RANDOM placement")
plt.tight_layout(); plt.savefig(f"{OUT_DIR}/perclass_overlay_grid.png", dpi=110)
plt.show()
print("saved:", f"{OUT_DIR}/perclass_overlay_grid.png")
```

> 대표 1건이 아니라 클래스별 N건(예 6건) 그리드가 필요하면 위 루프에서 `[:6]` 슬라이스로 확장. 대표만 보고 결론짓지 말 것 — Cell 3의 **분포 수치**가 정본, 그리드는 보조.

---

## 6. Cell 5 — 판정: Scenario A vs B

```python
def edge_span_pct(g, c):
    d = g[c]; t = sum(d.values()) or 1; return 100*(d['edge']+d['span'])/t

print(f"{'class':9s} {'REAL':>6s} {'AROMA':>7s} {'RANDOM':>7s}   해석")
for c in CLASSES:
    real = edge_span_pct(real_geom, c)
    aro  = edge_span_pct(place_AROMA, c)
    ran  = edge_span_pct(place_RANDOM, c)
    # AROMA 배치가 REAL 기하를 따르는가? AROMA vs RANDOM 배치 차이가 있는가?
    follows = abs(aro - real) < 15          # 실제 결함 위치를 15%p 이내로 재현하면 '따름'
    differs = abs(aro - ran) >= 10          # AROMA가 random과 10%p 이상 다르면 '선택이 배치기하에 영향'
    print(f"{c:9s} {real:5.0f}% {aro:6.0f}% {ran:6.0f}%   "
          f"{'배치=실제' if follows else '배치≠실제'} / {'AROMA≠RANDOM' if differs else 'AROMA≈RANDOM'}")
```

### 판정 규칙 (사전 등록)

| 관측 | 진단 | 함의 |
|------|------|------|
| REAL은 클래스별 edge/surface가 뚜렷한데 **AROMA 배치 분포가 이를 안 따르고**, 게다가 **AROMA≈RANDOM** | **Scenario B (placement-geometry 미모델링)** | 현 파이프라인은 배치 기하를 행사하지 않음(placement-blind, 가이드 §ITEM 4). ROI 선택 스칼라(ctx_prior)는 배치 위치에 영향 없음 → 점수 개선(ITEM 1–3)으로 downstream 못 살림. **placement-aware 재설계**가 필요. |
| AROMA 배치가 REAL 기하를 따르고 **AROMA≠RANDOM** (선택이 배치 위치를 실제로 바꿈) | placement은 작동 중 | 남은 null 원인은 배치가 아니라 ceiling/측정 → 다른 레버로. |
| REAL 자체가 클래스별 edge/surface 구분이 약함 | roi_check §2 가설(edge-vs-surface) 자체 반증 | §2 기반 개선 아이디어 폐기, 다른 신호 탐색. |

### 한계 (정직)
- 이 진단은 **배치 사실성(placement realism)** 만 본다. edge 배치가 downstream mAP를 실제로 올리는지는 **증명하지 않는다** — 그 판정은 가이드 **T5(2×2 placement×selection ablation, ≥3 seed)** 가 arbiter다. 본 문서는 T5를 돌릴 가치가 있는지(배치가 기하를 무시하는지)를 먼저 싸게 확인하는 게이트.
- MTD는 near-ceiling → 여기서 Scenario B가 확인돼도 "MTD에서 배치개선이 mAP를 올린다"는 결론은 별개(headroom 있는 AITeX에서 검증). MTD 결과는 **기제 진단**용.
- `defect_mask_path`(profiling)와 `image_path`(test 원본)의 좌표계가 어긋나면 mask 오버레이는 생략하고 bbox만 신뢰(마스크는 원본 test 이미지 기준으로 렌더된 것 확인 후 사용).

---

## 7. 무결성 (하지 말 것)
1. **5개 클래스 전부 보고** — edge 가설이 맞는 클래스만 제시(cherry-pick) 금지.
2. **`EDGE_MARGIN`/`SPAN_FRAC`를 원하는 결론이 나오도록 조정 금지** — 사전 고정, 민감도가 걱정되면 0.05/0.10 두 값 병기.
3. **대표 이미지 1건으로 결론 금지** — Cell 3/6의 분포 수치가 정본.
4. **이 진단으로 "배치개선이 성능을 올린다" 주장 금지** — arbiter는 T5. 본 문서는 기하 무시 여부만 판정.
5. **MTD 결과를 성능 결론으로 확대 금지** — near-ceiling. 기제 진단용.

---

## 8. 산출물
- `{OUT_DIR}/perclass_overlay_grid.png` — 클래스별 REAL/AROMA/RANDOM 오버레이 그리드.
- Cell 3/6 콘솔 표 — REAL vs 배치(arm별) edge+span% + Scenario A/B 판정.
- 이 표를 `AROMA연구분석/` 리포트로 옮겨 T5 실행 여부(가이드 매트릭스)를 결정.
