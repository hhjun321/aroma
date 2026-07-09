# ROI 검증·진단 통합 가이드 — localization 무결성 + placement-geometry 판정

> **실행 환경**: Colab (정상 이미지·결함 원본 픽셀이 Drive에만 있으므로 로컬 불가. Part 1 렌더는 CPU만)
> **목적 (2파트)**:
> - **Part 1 — 검증(localization/selection 육안)**: `roi_selection`이 고른 ROI가 (1) 실제 결함 위치와 정확히 일치하는지(localization 무결성), (2) 실결함 중 무엇이 선택됐는지를 원본에 bbox·mask 투영해 **육안** 확인.
> - **Part 2 — 진단(placement-geometry)**: roi_check §2의 "실제 결함 위치 vs 배치 ROI를 클래스별 대조"를 시각·정량으로 수행해 **Scenario A(scorer-blind) vs Scenario B(placement-geometry 미모델링)**를 판정.
> **성격**: **진단·검증 전용. 코드·파이프라인 변경 없음.** 개선 가이드(`roi_check_improvement_guide_20260708.md`) Phase 1(ITEM 4) 시각 파트이며, 동작 변경(ITEM 1–3)보다 **먼저** 실행되는 게이트.
> **선행**: 대상 DS의 roi_selection 완료 → `roi_selected_tobe.json`(+옵션 `roi_candidates.json`) + `annotations_synth_{aroma,random}.json` 존재 (`placement_aware_newpipe_execute.md §3`).

---

## 0. 좌표계 (로컬 검증 완료 — 그대로 사용)

`.claude/.etc/newpipe/`의 실측으로 확정된 사실:

| 필드 | 소재(파일) | 좌표계 | 형식 |
|------|-----------|--------|------|
| `defect_bbox` | `roi_selected_tobe.json` | **원본 test 결함 이미지**(`image_path`) 기준 | **xywh 픽셀** (정규화 아님) |
| `bbox` | `annotations_synth_{aroma,random}.json` | **정상 이미지**(`normal_image`) 기준 배치 위치 | **xywh 픽셀** |

- 검산: blowhole `[128,211,9,27]`(9×27 세로), crack `[313,138,11,104]`(11×104 세로선=모서리형), fray `[0,26,605,214]`·uneven `[0,149,604,111]`(x=0, 폭 604 전면=표면형). → roi_check §2 edge/surface 직관과 일치.
- 배치는 crop 크기(w,h)를 보존하고 **위치(x,y)만 합성 시점에 결정**(placement-blind). 예: blowhole aroma 배치 `[189,140,9,27]` (source w,h=9,27 동일, 위치만 이동).
- ⚠️ **xyxy 아님**: 200개 중 col2>col0 & col3>col1이 69개뿐 → x2/y2가 아니라 w/h. 오버레이 시 `Rectangle((x,y), w, h)` / `cv2.rectangle` w,h 사용.

---

## 1. 환경변수 · 입력 경로 (Part 1·2 공통)

```python
import os, json
os.environ.setdefault('DRIVE', '/content/drive/MyDrive/data/Aroma')  # 실제 마운트 경로로

DS = 'mtd'   # ← aitex / mtd / severstal / mvtec_leather
os.environ['DS'] = DS
AO = f"{os.environ['DRIVE']}/aroma_output"

# 입력(둘 중 존재하는 것 사용): Drive 라이브 산출 또는 커밋된 newpipe 스냅샷
PATHS = {
    # 추천(선택)된 소스 결함 + 실제 defect_bbox — realism 선택본
    'roi_selected':  f"{AO}/newpipe/{DS}/roi_selected_tobe.json",
    # (옵션) 후보 전체 — candidate-vs-selected 대조용. 없으면 Part 1의 노란 층 생략.
    'roi_candidates':f"{AO}/newpipe/{DS}/roi_candidates.json",
    # 배치 결과 (arm별)
    'synth_aroma':   f"{AO}/newpipe/{DS}/annotations_synth_aroma.json",
    'synth_random':  f"{AO}/newpipe/{DS}/annotations_synth_random.json",
}
# ↑ 라이브 경로가 다르면 여기만 수정. 커밋 스냅샷을 쓰려면:
#   /content/AROMA/.claude/.etc/newpipe/{roi_selected_tobe,annotations_synth_aroma,annotations_synth_random}.json
CLASSES = ['blowhole', 'break', 'crack', 'fray', 'uneven']   # DS에 맞게 조정
OUT_DIR = f"{AO}/diagnostics/{DS}/roi_verify"
os.makedirs(OUT_DIR, exist_ok=True)
print(PATHS); print("OUT:", OUT_DIR)
```

```python
# 공통 헬퍼
import cv2

def load(p):
    with open(p) as f: return json.load(f)

def imread_hw(path):
    im = cv2.imread(path)
    return None if im is None else im.shape[:2]

def has(key):
    return os.path.exists(PATHS[key])
```

---

# PART 1 — 검증: localization 무결성 + selection 육안

## 1-0. 개념 (무엇을 보는가)

roi_selection은 ROI 위치를 **생성하지 않는다.** 각 후보 = Phase0(`distribution_profiling`)가 실결함 이미지에서 추출한 **실제 결함 영역**(`image_path` + `defect_bbox` + `defect_mask_path`). selection은 그 실결함 풀에서 top_k를 **고를** 뿐.

따라서 3층을 원본에 투영:

| 층 | 출처 | 색 | 의미 |
| --- | --- | --- | --- |
| **GT mask 윤곽** | `defect_mask_path` | 빨강 | Phase0가 잡은 실제 결함 픽셀 (localization 정답) |
| **selected bbox** | `roi_selected_tobe.json` | 녹색 실선 | 선택돼 합성에 쓰일 ROI |
| **candidate-only bbox** | `roi_candidates.json` − selected | 노랑 점선 | 실결함이나 미선택 (candidates 파일 있을 때만) |

검증 질문:
1. **Localization**: bbox·mask가 원본의 실제 결함을 정확히 덮나? (Phase0 추출 오류·오프셋 없나)
2. **Selection**: 한 이미지의 여러 실결함 중 어느 것이 뽑혔나? 뽑힌 것이 더 그럴듯한 결함인가?

> candidate는 (실결함 × 점유 context cell)로 **같은 bbox가 cell마다 중복** → 아래 스크립트는 `(image_path, defect_bbox)`로 **dedup**해 "실결함 단위"로 본다.

## 1-1. 로드 + 실결함 단위 그룹핑

```python
import collections

sel = load(PATHS['roi_selected'])

def bkey(e):
    b = e.get('defect_bbox')
    return (e.get('image_path',''), tuple(b)) if b else None

sel_keys = {bkey(e) for e in sel if bkey(e)}

# candidates 있으면 후보 전체, 없으면 selected를 후보 풀로 사용(노란 층 없음)
cand = load(PATHS['roi_candidates']) if has('roi_candidates') else sel
have_cand = has('roi_candidates')

# image_path → { bbox_tuple: (entry, is_selected) }  — cell 중복 제거
by_img = collections.defaultdict(dict)
for e in cand:
    k = bkey(e)
    if not k: continue
    ip, bb = k
    if bb not in by_img[ip]:
        by_img[ip][bb] = (e, k in sel_keys)

n_imgs = len(by_img)
n_defects = sum(len(v) for v in by_img.values())
n_sel_defects = len(sel_keys)
print(f"이미지 {n_imgs}개 / 실결함(distinct bbox) {n_defects}개 / 선택 {n_sel_defects}개"
      f"{'' if have_cand else '  (candidates 없음 → 후보=선택)'}")
print("이미지당 실결함 분포:", collections.Counter(len(v) for v in by_img.values()))
```

## 1-2. 샘플 선택 (선택 ROI 포함 이미지 우선)

```python
cfg_class = True   # 클래스 균등 샘플 원하면 True (multi DS)
N = 12             # 몽타주에 넣을 이미지 수

sel_imgs = [ip for ip, d in by_img.items() if any(is_s for _, is_s in d.values())]
print(f"선택 ROI 포함 이미지: {len(sel_imgs)}")

if cfg_class:
    # 선택된 ROI의 class_key 기준 균등 샘플
    by_cls = collections.defaultdict(list)
    for e in sel:
        if bkey(e): by_cls[e.get('class_key','_')].append(e['image_path'])
    picks, per = [], max(1, N // max(1, len(by_cls)))
    for c, ips in by_cls.items():
        seen=set()
        for ip in ips:
            if ip not in seen: picks.append(ip); seen.add(ip)
            if len([p for p in picks if p in set(ips)])>=per: break
    sample = list(dict.fromkeys(picks))[:N]
else:
    sample = sel_imgs[:N]
print(f"샘플 {len(sample)}개")
```

## 1-3. 전체이미지 오버레이 몽타주

```python
import numpy as np, matplotlib.pyplot as plt

def render(image_path, defects):
    bgr = cv2.imread(image_path)
    if bgr is None:
        return None
    img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    H, W = img.shape[:2]
    for bb, (e, is_sel) in defects.items():
        x, y, w, h = [int(v) for v in bb]
        # GT mask 윤곽 (원본과 동일 해상도일 때만)
        mp = e.get('defect_mask_path')
        if mp and os.path.exists(mp):
            m = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
            if m is not None and m.shape[:2] == (H, W):
                cnts, _ = cv2.findContours((m > 127).astype('uint8'),
                                           cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(img, cnts, -1, (255, 0, 0), 1)   # 빨강 = GT mask
        color = (0, 220, 0) if is_sel else (255, 210, 0)          # 녹=선택, 노=미선택
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2 if is_sel else 1)
        if is_sel:
            lbl = f"{e.get('class_key','')} q{float(e.get('quality_score',0)):.2f} s{float(e.get('roi_score',0)):.2f}"
            cv2.putText(img, lbl, (x, max(10, y - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 220, 0), 1, cv2.LINE_AA)
    return img

cols = 3
rows = (len(sample) + cols - 1) // cols
fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5))
axes = np.array(axes).reshape(-1)
for ax in axes: ax.axis('off')
for ax, ip in zip(axes, sample):
    im = render(ip, by_img[ip])
    if im is None:
        ax.set_title("LOAD FAIL", fontsize=8); continue
    ax.imshow(im)
    n_sel = sum(1 for _, s in by_img[ip].values() if s)
    ax.set_title(f"{os.path.basename(ip)}  실결함{len(by_img[ip])} 선택{n_sel}", fontsize=8)
fig.suptitle(f"{os.environ['DS']} ROI overlay — 빨강:GTmask 녹:선택 노:미선택", fontsize=12)
plt.tight_layout()
out = f"{OUT_DIR}/overlay_{os.environ['DS']}.png"
plt.savefig(out, dpi=120, bbox_inches='tight'); plt.show()
print("저장:", out)
```

## 1-4. 선택 ROI 크롭 갤러리 (결함 실체 확대 확인)

```python
PAD = 0.4   # bbox 주변 여백 비율
gal = [e for e in sel if bkey(e)][:24]
cols = 6
rows = (len(gal) + cols - 1) // cols
fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.6, rows * 2.8))
axes = np.array(axes).reshape(-1)
for ax in axes: ax.axis('off')
for ax, e in zip(axes, gal):
    bgr = cv2.imread(e['image_path'])
    if bgr is None: ax.set_title("fail", fontsize=6); continue
    img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB); H, W = img.shape[:2]
    x, y, w, h = [int(v) for v in e['defect_bbox']]
    px, py = int(w * PAD), int(h * PAD)
    x0, y0, x1, y1 = max(0, x-px), max(0, y-py), min(W, x+w+px), min(H, y+h+py)
    crop = img[y0:y1, x0:x1].copy()
    cv2.rectangle(crop, (x-x0, y-y0), (x-x0+w, y-y0+h), (0, 220, 0), 2)
    mp = e.get('defect_mask_path')
    if mp and os.path.exists(mp):
        m = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
        if m is not None and m.shape[:2] == (H, W):
            mc = m[y0:y1, x0:x1]
            cnts, _ = cv2.findContours((mc > 127).astype('uint8'),
                                       cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(crop, cnts, -1, (255, 0, 0), 1)
    ax.imshow(crop)
    ax.set_title(f"{e.get('class_key','')}\nq{float(e.get('quality_score',0)):.2f} s{float(e.get('roi_score',0)):.2f}", fontsize=6)
fig.suptitle(f"{os.environ['DS']} 선택 ROI 크롭 (녹:bbox 빨강:mask)", fontsize=11)
plt.tight_layout()
out = f"{OUT_DIR}/crops_{os.environ['DS']}.png"
plt.savefig(out, dpi=130, bbox_inches='tight'); plt.show()
print("저장:", out)
```

## 1-5. Localization 무결성 정량 체크 (선택)

bbox가 mask를 제대로 감싸는지 수치로 보조 확인 (mask가 원본 해상도일 때):

```python
ious, covers, mismatch = [], [], 0
for e in sel:
    if not bkey(e): continue
    mp = e.get('defect_mask_path')
    if not (mp and os.path.exists(mp)): continue
    m = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
    if m is None: continue
    x, y, w, h = [int(v) for v in e['defect_bbox']]
    box = np.zeros_like(m); box[y:y+h, x:x+w] = 255
    mb = (m > 127); bb = (box > 127)
    inter = (mb & bb).sum(); union = (mb | bb).sum()
    if union == 0: continue
    ious.append(inter / union)
    covers.append(inter / max(1, mb.sum()))   # mask 픽셀 중 bbox 안 비율
print(f"bbox↔mask IoU  mean={np.mean(ious):.3f}  min={np.min(ious):.3f}")
print(f"mask coverage(=bbox가 mask 포함률) mean={np.mean(covers):.3f}  <0.95 개수={sum(c<0.95 for c in covers)}")
```

> **coverage≈1.0**이면 bbox가 결함 mask를 온전히 감쌈(정상). 낮으면 bbox가 결함을 잘림 → Phase0 추출/파싱 점검. IoU는 결함이 bbox보다 작으면 낮게 나오는 게 정상(mask는 결함, bbox는 사각) — coverage가 더 신뢰 지표.

## 1-6. 육안 체크리스트

- [ ] **녹색 bbox가 실제 결함 위에** 있나? (배경·정상 영역에 찍힌 게 없나 = 오탐)
- [ ] **빨강 mask 윤곽이 결함 경계와 일치**하나? (오프셋·엉뚱한 영역 = Phase0 추출 오류)
- [ ] bbox가 결함을 **잘리지 않고** 감싸나? (§1-5 coverage와 교차)
- [ ] 선택(녹)이 미선택(노)보다 **더 뚜렷/전형적 결함**인가? (selection이 품질 좋은 결함을 골랐나 — realism=quality 승격 가설의 육안 확인. candidates 파일 있을 때만)
- [ ] (multi DS) 클래스별로 **대표성 있는 결함**이 뽑혔나? (class_key 라벨이 실제 결함 유형과 맞나)
- [ ] mask 해상도 불일치로 윤곽 미표시된 케이스 수 (크롭-마스크 데이터셋이면 §1-3/§1-4 윤곽 생략됨 — bbox만으로 판단)

---

# PART 2 — 진단: placement-geometry (Scenario A vs B)

## 2-1. edge/surface 분류기 (경계 근접 기준)

정상 이미지·결함 이미지의 **실제 dim을 읽어** bbox의 경계 근접도로 edge vs surface를 판정한다.

```python
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
```

> **분류 규칙 근거**: MTD 타일에서 crack/break/fray는 모서리(경계) 근접, blowhole은 표면 내부라는 roi_check §2 가설을 그대로 정량화. `span`(fray/uneven류 전면)은 별도 범주로 두어 "표면 vs 모서리" 이분법 왜곡을 막는다. `EDGE_MARGIN`·`SPAN_FRAC`는 **사전 고정** — 원하는 결론이 나오도록 조정 금지(무결성).

## 2-2. 실제 결함 위치 프로파일 (roi_selected 기준)

```python
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

## 2-3. 배치 위치 프로파일 (arm별: aroma vs random)

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

## 2-4. 시각 그리드 (클래스별 오버레이, 육안 대조)

```python
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

> 대표 1건이 아니라 클래스별 N건(예 6건) 그리드가 필요하면 위 루프에서 `[:6]` 슬라이스로 확장. 대표만 보고 결론짓지 말 것 — §2-2/§2-3의 **분포 수치**가 정본, 그리드는 보조.

## 2-5. 판정: Scenario A vs B

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

# PART 3 — 배치 육안 검증: clean image의 paste 위치 (AROMA)

## 3-0. 개념 (무엇을 보는가)

Part 1은 **소스 결함**(어떤 실결함을 고르나)을, Part 2는 배치 **분포**(클래스별 edge/surface 통계)를 본다. **Part 3은 한 장 한 장의 배치 결정** — 결함이 없는 clean 이미지의 **어디에** ROI가 붙었는지를 육안 확인한다.

배치 로직(`generate_defects._paste_and_finalize`):

| 단계 | 동작 |
| --- | --- |
| 1. foreground 착지 | `_foreground_paste_position` — 결함 mask 중심이 object 전경 픽셀에 떨어지도록 위치 샘플 (object-centric DS) |
| 2. random 폴백 | 전경 추정 실패 시 `_random_paste_position` 무작위 위치 |
| 3. 게이트 재샘플 | clean-bg(void 거부, 폴백 경로만) / texture / compat(배경↔결함 호환) 게이트가 위치를 재샘플 → 통과 or `max_bg_tries` 소진 |
| 4. normal 재픽 | 게이트 소진 시 `normal_pool`에서 다른 clean 이미지로 교체 (`compat`/`texture` on일 때) |

`annotations_synth_aroma.json` 샘플이 담는 것:

| 필드 | 의미 |
| --- | --- |
| `normal_image` | paste **전** clean 배경 (재픽 시 실제 사용된 경로) |
| `image_path` | paste **후** 합성 출력 (결함 붙은 결과) |
| `bbox` | clean 이미지 상 배치 위치 (xywh 픽셀) |
| `class_key` / `cell_key` | 결함 유형 / 배치 patch context cell |

검증 질문:

1. 배치가 **유효 배경**(전경·표면)에 있나? 검은/평탄 void나 이미지 경계 밖 클리핑은 없나?
2. before(위치)와 after(합성 결과)가 **좌표 일치**하나? bbox가 실제 붙은 자리와 맞나?
3. (compat/texture on) 배치 patch가 결함과 **어울리는 배경**인가? (게이트가 일했나 — 육안 근사)

> ⚠️ **per-sample 게이트 통과여부는 annotations에 없다** (run 집계 `gate_stats`로만 pop). 따라서 개별 샘플이 "게이트 통과본"인지 "소진 후 last-candidate"인지는 여기서 못 가른다 — 배치 위치의 **결과**만 육안 판정. 집계율은 generate_defects 실행 로그의 `placement-gate stats`로 확인.

## 3-1. 로드 + 클래스 균등 샘플 (AROMA)

```python
import collections

aroma = [a for a in load(PATHS['synth_aroma']) if not a.get('dry_run') and a.get('bbox')]
print(f"AROMA 합성 샘플 {len(aroma)}개")

by_cls = collections.defaultdict(list)
for a in aroma:
    by_cls[a.get('class_key','_')].append(a)
print("클래스별:", {c: len(v) for c, v in by_cls.items()})

N = 12                                  # 그리드에 넣을 샘플 수
per = max(1, N // max(1, len(by_cls)))
psample = []
for c, items in by_cls.items():
    psample.extend(items[:per])
psample = psample[:N]
print(f"샘플 {len(psample)}개 (클래스 균등)")
```

## 3-2. before/after 배치 그리드

각 샘플 2열: **왼=clean `normal_image` + 배치 bbox**, **오=합성 `image_path` + 동일 bbox**. 좌우 bbox 좌표가 같으므로, 오른쪽에서 결함이 그 사각 안에 실제로 나타나야 정상.

```python
import numpy as np, matplotlib.pyplot as plt
import matplotlib.patches as patches

def _show(ax, path, bbox, title, edge):
    im = cv2.imread(path)
    if im is None:
        ax.set_title(f"{title}\n(LOAD FAIL)", fontsize=7); ax.axis('off'); return
    ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    x, y, w, h = [int(v) for v in bbox]
    ax.add_patch(patches.Rectangle((x, y), w, h, fill=False, edgecolor=edge, linewidth=2))
    ax.set_title(title, fontsize=7); ax.axis('off')

rows = len(psample)
fig, axes = plt.subplots(rows, 2, figsize=(8, 3.4 * rows))
axes = np.array(axes).reshape(rows, 2)
for i, a in enumerate(psample):
    c = a.get('class_key', '')
    bb = a['bbox']
    _show(axes[i, 0], a['normal_image'], bb, f"{c} · CLEAN (배치 위치)", 'yellow')
    _show(axes[i, 1], a['image_path'],   bb, f"{c} · SYNTH (paste 결과)", 'lime')
fig.suptitle(f"{os.environ['DS']} AROMA 배치 육안 — 노랑:clean 배치 bbox  녹:합성 결과", fontsize=12)
plt.tight_layout()
out = f"{OUT_DIR}/placement_{os.environ['DS']}.png"
plt.savefig(out, dpi=115, bbox_inches='tight'); plt.show()
print("저장:", out)
```

## 3-3. 배치 patch void/경계 정량 보조 (선택)

per-sample 게이트 통과여부가 없으므로, 배치된 `bbox` 지점의 clean patch가 **void(검정/평탄)** 거나 **경계 클리핑**인지를 사후 스캔해 게이트가 놓쳤을 후보를 센다.

```python
void_flat, clipped, ok = 0, 0, 0
FLAT_STD = 6.0          # patch 밝기 표준편차 < 이 값이면 평탄(void 의심)
for a in aroma:
    im = cv2.imread(a['normal_image'])
    if im is None: continue
    H, W = im.shape[:2]
    x, y, w, h = [int(v) for v in a['bbox']]
    if x < 0 or y < 0 or x + w > W or y + h > H:
        clipped += 1; continue
    patch = cv2.cvtColor(im[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
    if patch.size == 0: continue
    if float(patch.std()) < FLAT_STD or float(patch.mean()) < 10:
        void_flat += 1
    else:
        ok += 1
tot = void_flat + clipped + ok or 1
print(f"배치 patch 진단 (n={tot}):")
print(f"  유효 배경        {ok:4d} ({100*ok/tot:.0f}%)")
print(f"  void/평탄 의심   {void_flat:4d} ({100*void_flat/tot:.0f}%)  ← clean-bg 게이트가 놓친 후보")
print(f"  경계 클리핑      {clipped:4d} ({100*clipped/tot:.0f}%)")
```

> `void/평탄`·`클리핑` 비율이 높으면 배치가 유효 배경을 못 고른 것 → clean-bg/foreground 게이트 점검. **주의**: `FLAT_STD`는 근사 proxy(진짜 게이트 판정 아님). texture/compat 게이트를 실제로 쓴 실행이면 이 비율은 낮아야 정상.

## 3-4. 배치 육안 체크리스트

- [ ] **노랑 bbox(clean)가 유효 배경 위**인가? 검은/평탄 void·이미지 밖이 아닌가? (§3-3 수치와 교차)
- [ ] **녹 bbox(synth)** 안에 결함이 실제로 나타나나? (before/after 좌표 정합 = bbox가 진짜 paste 위치)
- [ ] 배치 위치가 결함 유형에 **자연스러운가**? (예: crack이 표면 한복판보다 모서리 근처 — Part 2 분포와 일관?)
- [ ] paste 경계(feather/seamless)가 **부자연스럽게 튀지** 않나? (합성 품질 — 배치와 별개지만 육안에 같이 잡힘)
- [ ] compat/texture on 실행이면, 배치 patch가 결함과 **어울리는 배경**인가? (게이트 효과 육안)

---

## 3. 무결성 (하지 말 것)

**Part 1 (검증)**
- **candidate 중복**: 반드시 `(image_path, defect_bbox)` dedup 후 봐야 실결함 수가 맞다(cell별 중복 제거).
- 이 검증은 **selection의 localization·대표성** 확인용. 합성 결과(paste 품질)는 별도(`generate_defects` 산출 육안).

**Part 2 (진단)**
1. **CLASSES 전부 보고** — edge 가설이 맞는 클래스만 제시(cherry-pick) 금지.
2. **`EDGE_MARGIN`/`SPAN_FRAC`를 원하는 결론이 나오도록 조정 금지** — 사전 고정, 민감도가 걱정되면 0.05/0.10 두 값 병기.
3. **대표 이미지 1건으로 결론 금지** — §2-2/§2-5의 분포 수치가 정본.
4. **이 진단으로 "배치개선이 성능을 올린다" 주장 금지** — arbiter는 T5. 본 문서는 기하 무시 여부만 판정.
5. **MTD 결과를 성능 결론으로 확대 금지** — near-ceiling. 기제 진단용.

**Part 3 (배치 육안)**
- **per-sample 게이트 통과여부는 annotations에 없다** — §3-2 그리드·§3-3 proxy로 개별 샘플을 "게이트 통과본"이라 단정 금지. 통과율은 generate_defects 실행 로그 `placement-gate stats` 집계로만.
- **§3-3 `FLAT_STD`는 void 근사 proxy** — 실제 게이트 판정 아님. 이 수치로 "게이트 실패" 확정 금지, 육안(§3-2)과 교차만.
- 대표 12장으로 배치 전체를 결론짓지 말 것 — 이상 의심 시 `N` 늘려 재확인.

---

## 4. 주의 / 산출물

**주의**
- **mask 해상도**: `defect_mask_path`가 원본과 다른 해상도(크롭 마스크)면 윤곽 생략(스크립트가 shape 일치만 그림). bbox는 항상 원본 좌표계라 신뢰 가능.
- **image_path 접근**: Drive 원본 경로. 없으면 "LOAD FAIL" — 데이터 마운트/경로 확인.

**산출물**
- `{OUT_DIR}/overlay_{DS}.png` — Part 1 전체이미지 오버레이 몽타주(빨강 GTmask / 녹 선택 / 노 미선택).
- `{OUT_DIR}/crops_{DS}.png` — Part 1 선택 ROI 크롭 갤러리.
- `{OUT_DIR}/perclass_overlay_grid.png` — Part 2 클래스별 REAL/AROMA/RANDOM 오버레이 그리드.
- `{OUT_DIR}/placement_{DS}.png` — Part 3 AROMA 배치 before/after 그리드(노랑:clean 배치 / 녹:합성 결과).
- §2-2/§2-5 콘솔 표 — REAL vs 배치(arm별) edge+span% + Scenario A/B 판정. 이 표를 `AROMA연구분석/` 리포트로 옮겨 T5 실행 여부(가이드 매트릭스)를 결정.
- §3-3 콘솔 표 — 배치 patch 유효배경/void/클리핑 비율(clean-bg 게이트 사후 근사).
