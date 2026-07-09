# compat 게이트 clean-grounded 착수 진단 (CPU, Colab) — no-op 여부 실측 게이트

> **실행 환경**: Colab, **CPU only** (GPU·재프로파일 불필요)
> **목적**: `aroma_compat_gate_clean-grounded_redesign` devnote §5의 **착수 게이트**. compat 기준을 clean-grounded로 바꾸는 구현이 **실제로 게이트 동작을 바꾸는가(=no-op 아닌가)**를 재프로파일 예산 쓰기 전에 데이터로 판정.
> **선행 산출물** (이미 존재, 재생성 불필요): `$PROFILING_DIR/compatibility_matrix.json`, `$PROFILING_DIR/context_features.csv`, `$SEL_AROMA/roi_selected.json`.
> **판정 요약**: `TV(good,defect)<0.10` → 확정 no-op → 해당 DS 착수 중단. fallback률이 image-mean 기준 高인데 patch 기준 급락 → granularity가 lever → 진행.

---

## 0. 배경 (무엇을 측정하나)

devnote 진단 확정 사실:
- matrix `P(cell|cluster)` = 결함 이미지 **image-mean context**로 계수(이미지당 1 cell) → 관측 cell 극소.
- 게이트는 **local crop-size patch** 1개의 cell을 질의 → 대부분 미관측 → `get(cell,0.5)` 중립통과. ("leather 4.7% coverage/~95% 중립"의 실제 기제 = **granularity 불일치**, distribution 자체가 아님).
- 제안 수정: clean_dist를 good 이미지 **전 patch** granularity로 재정의 + SGM 대칭결합.

이 진단은 4개 질문에 답한다:

| # | 질문 | 지표 |
|---|---|---|
| H1 | matrix 관측 cell이 image-mean이라 극소인가 | 관측 cell 수 vs patch-gran 고유 cell 수 |
| H2 | clean local patch가 matrix 관측 cell을 얼마나 덮나 (4.7% 재현) | coverage(local patch cell ∈ matrix 관측) |
| TV | good vs defect **분포 자체**가 다른가 (no-op gate) | TV·cosine(P_good_patch, P_defect_patch) |
| H3 | soft+τ에서 accept/neutral/reject 분해 | τ∈{0.3,0.5,0.7}별 |
| flip | SGM 재정렬이 관측 cell 판정을 바꾸나 | τ 넘나드는 cell 수 |

---

## 1. 환경변수

```python
import os, json
DS = 'mvtec_leather'   # ← mvtec_leather / aitex / mtd / severstal
os.environ['DS'] = DS
os.environ['AROMA_OUT']     = f"{os.environ['DRIVE']}/aroma_output"
os.environ['AROMA_REF']     = '/content/AROMA'
os.environ['PROFILING_DIR'] = f"{os.environ['AROMA_OUT']}/profiling/{DS}"
os.environ['COMPAT_JSON']   = f"{os.environ['PROFILING_DIR']}/compatibility_matrix.json"
os.environ['CONTEXT_CSV']   = f"{os.environ['PROFILING_DIR']}/context_features.csv"
os.environ['SEL_AROMA']     = f"{os.environ['AROMA_OUT']}/newpipe/{DS}/sel_aroma"
os.environ['OUT_DIR']       = f"{os.environ['AROMA_OUT']}/diagnostics/{DS}/compat_gate"
os.makedirs(os.environ['OUT_DIR'], exist_ok=True)
for k in ('COMPAT_JSON','CONTEXT_CSV'):
    print(k, '=', os.environ[k], '  exists:', os.path.exists(os.environ[k]))
```

## 2. 로드 (matrix, bin_edges, cell 헬퍼, context_features.csv)

```python
import sys, json, csv, numpy as np, collections
sys.path.insert(0, f"{os.environ['AROMA_REF']}/scripts")
sys.path.insert(0, f"{os.environ['AROMA_REF']}/scripts/aroma")
import distribution_profiling as dp

compat = json.load(open(os.environ['COMPAT_JSON']))
matrix    = compat['matrix']                 # {cluster: {cell: P(cell|cluster)}}
bin_edges = compat['bin_edges']              # matrix 구축·게이트 공용 (config 로드 불필요)
FEATS = dp.CONTEXT_FEATURES                   # 5 context feature 이름

def cell_of_featrow(row):
    """context_features.csv 한 행(patch) → cell_key."""
    feat = {}
    for f in FEATS:
        v = row.get(f, '')
        feat[f] = float(v) if v not in ('', None) else 0.0
    return dp._context_cell_key(feat, bin_edges)

# matrix 관측 cell (image-mean 기준 — 현행 게이트 참조집합)
observed = set()
for cid, r in matrix.items():
    observed |= set(r.keys())
print(f"clusters={list(matrix.keys())}  matrix 관측 cell 수(합집합)={len(observed)}")
for cid, r in matrix.items():
    if r:
        print(f"  cluster {cid}: {len(r)} cells  compat[{min(r.values()):.2f},{max(r.values()):.2f}]")
    else:
        print(f"  cluster {cid}: EMPTY")

# context_features.csv patch 행 로드 → good / defect 분리
rows = list(csv.DictReader(open(os.environ['CONTEXT_CSV'])))
good_cells   = [cell_of_featrow(r) for r in rows if r.get('image_type') == 'good']
defect_cells = [cell_of_featrow(r) for r in rows if r.get('image_type') == 'defect']
print(f"\npatch 행: good={len(good_cells)}  defect={len(defect_cells)}")

# 결과 누적기 — 이후 셀이 여기에 append, §8 이전 셀에서 OUT_DIR에 저장
RESULT = {
    'DS': os.environ['DS'],
    'n_clusters': len(matrix),
    'matrix_observed_cells': len(observed),
    'cluster_stats': {cid: {'n_cells': len(r),
                            'compat_min': (min(r.values()) if r else None),
                            'compat_max': (max(r.values()) if r else None)}
                      for cid, r in matrix.items()},
    'n_good_patch': len(good_cells),
    'n_defect_patch': len(defect_cells),
}
```

## 3. H1·H2 — granularity 불일치 & coverage

```python
# H1: patch-gran 고유 cell 수 vs matrix 관측 cell 수 (image-mean이 얼마나 정보를 압축했나)
uniq_good_patch   = set(good_cells)
uniq_defect_patch = set(defect_cells)
print(f"[H1] 고유 cell 수")
print(f"  matrix 관측(image-mean)     : {len(observed)}")
print(f"  good  patch-gran 고유       : {len(uniq_good_patch)}")
print(f"  defect patch-gran 고유      : {len(uniq_defect_patch)}")

# H2: clean local patch가 (a) matrix 관측 cell / (b) defect patch-gran support 를 덮는 비율
def cov(cells, ref):
    n = len(cells) or 1
    return sum(c in ref for c in cells) / n

cov_matrix = cov(good_cells, observed)              # 현행 게이트 유효 coverage (리포트 4.7% 대응)
cov_defpat = cov(good_cells, uniq_defect_patch)     # granularity 정합 시 도달 가능 coverage
print(f"\n[H2] good local patch coverage")
print(f"  ∈ matrix 관측(image-mean)   : {cov_matrix*100:.1f}%   ← 현행 게이트 (낮으면 무력)")
print(f"  ∈ defect patch-gran support : {cov_defpat*100:.1f}%   ← patch 정합 시 상한")
print(f"  → 격차 크면 granularity(image-mean)가 무력화 주범 = 수정 lever 유효")

RESULT['H1H2'] = {
    'uniq_good_patch': len(uniq_good_patch),
    'uniq_defect_patch': len(uniq_defect_patch),
    'cov_good_in_matrix': round(cov_matrix, 4),
    'cov_good_in_defect_patch': round(cov_defpat, 4),
}
```

## 4. TV — 분포 발산 (no-op 착수 게이트)

```python
def marginal(cells):
    n = len(cells) or 1
    c = collections.Counter(cells)
    return {k: v / n for k, v in c.items()}

P_good = marginal(good_cells)
P_def  = marginal(defect_cells)
allc = set(P_good) | set(P_def)
tv = 0.5 * sum(abs(P_good.get(c, 0.0) - P_def.get(c, 0.0)) for c in allc)
vg = np.array([P_good.get(c, 0.0) for c in allc]); vd = np.array([P_def.get(c, 0.0) for c in allc])
cos = float(vg @ vd / ((np.linalg.norm(vg) * np.linalg.norm(vd)) or 1))
print(f"[TV] P(cell|good) vs P(cell|defect)  (patch-gran)")
print(f"  Total Variation = {tv:.3f}   cosine = {cos:.3f}")
print(f"  판정: TV<0.10 → clean/defect 분포 사실상 동일 → clean-grounding 확정 no-op → 이 DS 착수 중단")
print(f"        TV≥0.10 → domain-shift 존재 → granularity(H1/H2)와 함께 착수 판단")

RESULT['TV'] = {'total_variation': round(tv, 4), 'cosine': round(cos, 4),
                'noop_verdict': bool(tv < 0.10)}
```

## 5. H3 — τ별 게이트 판정 분해 (soft 매칭)

```python
def decompose(cells, row, tau):
    n = len(cells) or 1
    accept  = sum(row.get(c, 0.5) >= tau for c in cells)   # soft: 미관측=0.5
    neutral = sum(c not in row for c in cells)             # 미관측(중립 통과 후보)
    rej_obs = sum((c in row) and (row[c] < tau) for c in cells)
    return accept/n*100, neutral/n*100, rej_obs/n*100

print("[H3] good local patch 기준 τ별 판정 (클러스터별)")
h3 = {}
for cid, row in matrix.items():
    if not row: continue
    print(f" cluster {cid}:")
    h3[cid] = {}
    for tau in (0.3, 0.5, 0.7):
        a, ne, ro = decompose(good_cells, row, tau)
        h3[cid][str(tau)] = {'accept': round(a, 1), 'neutral': round(ne, 1), 'reject_obs': round(ro, 1)}
        print(f"   τ={tau}: accept={a:5.1f}%  (미관측중립={ne:5.1f}%)  관측거부={ro:5.1f}%")
print("> accept 높고 미관측중립 대부분 → over-accept 무력(저τ). accept 낮고 거부 큼 → reject-heavy(고τ).")
RESULT['H3'] = h3
```

## 6. flip — SGM 재정렬 preview (관측 cell 판정 변화)

```python
# clean_dist(patch-gran) = P_good. SGM: sqrt((P_def_row+e)(P_clean+e)) per-cluster max 정규화.
EPS = 1e-3
def sgm_row(def_row, P_clean):
    raw = {c: (def_row[c] + EPS) ** 0.5 * (P_clean.get(c, 0.0) + EPS) ** 0.5 for c in def_row}
    m = max(raw.values()) if raw else 1.0
    return {c: v / (m or 1.0) for c, v in raw.items()}

TAU = 0.5   # 현행 예시값 (실착수 시 사전스캔으로 재보정)
print(f"[flip] 관측 cell 중 raw→SGM 로 τ={TAU} 판정이 바뀌는 수 (per-cluster max 정규화)")
tot_flip = 0
flip_by_cluster = {}
for cid, row in matrix.items():
    if not row: continue
    sr = sgm_row(row, P_good)
    flips = [c for c in row if (row[c] >= TAU) != (sr[c] >= TAU)]
    tot_flip += len(flips)
    flip_by_cluster[cid] = {'n_observed': len(row), 'n_flip': len(flips)}
    print(f"  cluster {cid}: 관측 {len(row)} cell 중 flip {len(flips)}")
print(f"  총 flip={tot_flip}  → 0이면 SGM이 관측 cell 판정을 안 바꿈(값 재정렬만). "
      f"단 게이트 실효는 §3 coverage·§4 TV가 지배.")
RESULT['flip'] = {'tau': TAU, 'eps': EPS, 'total_flip': tot_flip, 'by_cluster': flip_by_cluster}
```

## 6.5 결과 저장 (OUT_DIR)

§2~6에서 누적한 `RESULT`를 JSON + 요약 markdown으로 `OUT_DIR`에 저장. 4종(leather/aitex/mtd/severstal) 각각 실행 후 파일이 DS별로 남아 §8 판정표·devnote append에 재사용.

```python
import json, os
os.makedirs(os.environ['OUT_DIR'], exist_ok=True)
DS = os.environ['DS']

# 1) 원본 지표 JSON
json_path = f"{os.environ['OUT_DIR']}/compat_gate_diag_{DS}.json"
with open(json_path, 'w') as f:
    json.dump(RESULT, f, indent=2, ensure_ascii=False)

# 2) 사람이 읽는 요약 markdown (§8 판정표 대응)
tv = RESULT['TV']; h = RESULT['H1H2']
lines = [
    f"# compat 게이트 진단 — {DS}",
    "",
    f"- matrix 관측 cell(image-mean): **{RESULT['matrix_observed_cells']}**  "
    f"(good patch 고유 {h['uniq_good_patch']} / defect patch 고유 {h['uniq_defect_patch']})",
    f"- good coverage ∈ matrix: **{h['cov_good_in_matrix']*100:.1f}%**  "
    f"| ∈ defect patch-gran: {h['cov_good_in_defect_patch']*100:.1f}%",
    f"- TV(good,defect) = **{tv['total_variation']}**  cosine={tv['cosine']}  "
    f"→ {'no-op(착수중단)' if tv['noop_verdict'] else 'domain-shift 존재'}",
    f"- SGM flip(τ={RESULT['flip']['tau']}) 총 {RESULT['flip']['total_flip']}",
    f"- patch: good={RESULT['n_good_patch']} defect={RESULT['n_defect_patch']}",
]
md_path = f"{os.environ['OUT_DIR']}/compat_gate_diag_{DS}.md"
with open(md_path, 'w') as f:
    f.write("\n".join(lines) + "\n")

print("저장:", json_path)
print("저장:", md_path)
```

## 7. (선택) H4 — steering power 확증

정적 분석(§3~6)으로 착수 판단은 충분. 확증이 필요하면 소규모 생성으로 실제 배치 변화 측정:

```
generate_defects.py --compat_threshold <τ>  vs  --compat_threshold 0  두 번 실행
→ 로그의 placement-gate stats: fallback률 + paste 위치(bbox) 차이 비교.
```
(부하/대량 측정 아님 — 소규모 fixture. 재프로파일 불필요.)

---

## 8. 판정표 (§3~6 → 착수 결정)

| 관측 | 진단 | 결정 |
|---|---|---|
| **TV<0.10** | clean≈defect 분포 → 제거할 domain-shift 없음 | 해당 DS **착수 중단** (올바른 no-op) |
| cov(∈matrix) 低 + cov(∈defect-patch) 高 + TV≥0.10 | granularity(image-mean)가 무력화 주범, 분포는 다름 | **patch-gran clean_dist 착수** (devnote §3-1) |
| accept 높고 미관측중립 지배(저τ) | over-accept 무력 | patch-gran + **τ 사전스캔 재보정** 필수 |
| accept 낮고 관측/미관측거부 큼(고τ) | reject-heavy fallback | soft 유지 + patch-gran, hard gate 금지 |
| leather coverage 극저 재현(≈4.7%) | 리포트 재현 | leather **placement 게이트 제외**(τ=0, newpipe 가이드) |
| flip=0 & coverage 개선 없음 | SGM 값재정렬만, 게이트 실효 없음 | 구현 가치 없음 — 착수 보류 |

---

## 9. 무결성

- **재프로파일·GPU 금지**: 기존 `compatibility_matrix.json`·`context_features.csv`만 읽는 정적 분석.
- **bin_edges는 compat json 것 사용** — 게이트/matrix와 동일 택소노미 보장(별도 config 로드 시 불일치 위험).
- **τ=0.5는 예시** — 실착수 threshold는 사전스캔으로 데이터셋별 확정([[feedback_prescan_thresholds]]). 이 진단 수치 보고 τ를 원하는 결론에 맞추지 말 것.
- **4종 전부 실행**(leather/aitex/mtd/severstal) 후 §8 표로 데이터셋별 판정 — cherry-pick 금지.
- 결과 표를 devnote `aroma_compat_gate_clean-grounded_redesign` §5에 append하고 착수 여부 확정.
- 부하 측정 항목(H4 대량화)은 자동 실행 안 함(load-test 정책).

---

## §10. τ 사전스캔 — symmetric 스케일 + 타일링-aware (compat_mode=symmetric 전용)

> **목적**: `compat_mode=symmetric`에서 데이터셋별 `compat_threshold` τ를 **데이터로 확정**([[feedback_prescan_thresholds]]). `matrix_symmetric`은 per-cluster max-norm으로 [0,1]이나 **τ=0.5 금지**(0.5=legacy 확률 스케일 중립값). τ는 good 배경의 실제 tiled mean-compat 분포에서 목표 reject율 percentile로.
> **타일링-aware**: 게이트 symmetric 경로(`_compat_ok`, generate_defects.py)와 **동일** — footprint를 64px window(`_tile_anchors`)로 덮어 cell compat mean 집계. 사전스캔이 이를 그대로 재현.
> **무결성**: 목표 reject율 **R=0.25를 수치 보기 전 고정**. R20/R30은 민감도 참고. **신 profiling(`matrix_symmetric`) 필수** — 구 profile hard-fail. severstal/aitex 大 → Colab만.

### 10.1 로드 (§1·§2 이어서 — matrix_symmetric + 타일링 헬퍼)

```python
import os, sys, json, csv, numpy as np, collections, random
os.environ['DATASET_CONFIG'] = os.environ.get('DATASET_CONFIG', f"{os.environ['AROMA_REF']}/dataset_config.json")
with open(os.environ['DATASET_CONFIG']) as f: _dscfg = json.load(f)
os.environ['NORMAL_DIR'] = _dscfg[os.environ['DS']]['image_dir']
os.environ['ROI_SELECTED'] = f"{os.environ['SEL_AROMA']}/roi_selected.json"   # 실 crop 크기 출처
for k in ('NORMAL_DIR','ROI_SELECTED','COMPAT_JSON'):
    print(k, '=', os.environ[k], ' exists:', os.path.exists(os.environ[k]))

sys.path.insert(0, f"{os.environ['AROMA_REF']}/scripts")
sys.path.insert(0, f"{os.environ['AROMA_REF']}/scripts/aroma")
import distribution_profiling as dp
import generate_defects as gd
_tile_anchors = gd._tile_anchors; TILE = gd._COMPAT_TILE; AGG = gd._COMPAT_TILE_AGG
print(f"tiling: TILE={TILE} AGG={AGG}")

compat = json.load(open(os.environ['COMPAT_JSON']))
msym = compat.get('matrix_symmetric')
if msym is None:
    raise SystemExit("matrix_symmetric 없음 — 구 profile. distribution_profiling 재실행 필요.")
bin_edges = compat['bin_edges']; FEATS = dp.CONTEXT_FEATURES
clusters = [c for c, r in msym.items() if r]
print(f"clusters(비어있지않음)={clusters}")
for c in clusters:
    r = msym[c]; print(f"  cluster {c}: {len(r)} cells compat[{min(r.values()):.3f},{max(r.values()):.3f}]")
```

### 10.2 실 crop 크기 + good 이미지

```python
import cv2
from pathlib import Path
IMG_EXTS = {'.png','.jpg','.jpeg','.bmp','.tif','.tiff'}
sel = json.load(open(os.environ['ROI_SELECTED']))
crop_sizes = [(int(e['defect_bbox'][2]), int(e['defect_bbox'][3])) for e in sel
              if isinstance(e.get('defect_bbox'), (list,tuple)) and len(e['defect_bbox'])==4
              and int(e['defect_bbox'][2])>0 and int(e['defect_bbox'][3])>0]
assert crop_sizes, "roi_selected defect_bbox 크기 없음"
_cw=np.array([c[0] for c in crop_sizes]); _ch=np.array([c[1] for c in crop_sizes])
print(f"실 crop n={len(crop_sizes)} w[{_cw.min()}/{int(np.median(_cw))}/{_cw.max()}] h[{_ch.min()}/{int(np.median(_ch))}/{_ch.max()}]")

N_GOOD, M_POS, SEED = 60, 40, 42
rng = random.Random(SEED)
good_paths = sorted(str(p) for p in Path(os.environ['NORMAL_DIR']).rglob('*') if p.suffix.lower() in IMG_EXTS)
if len(good_paths) > N_GOOD: good_paths = rng.sample(good_paths, N_GOOD)
print(f"good 이미지: {len(good_paths)}")
```

### 10.3 타일링 mean-compat 분포 (게이트 재현)

```python
_cell_cache = {}
def _cell_at(gray, gi, ax, ay):
    key=(gi,ax,ay); v=_cell_cache.get(key)
    if v is None:
        v = dp._context_cell_key(dp._extract_context_features(gray[ay:ay+TILE, ax:ax+TILE]), bin_edges)
        _cell_cache[key]=v
    return v

footprints=[]
for gi, gp in enumerate(good_paths):
    g = cv2.imread(gp, cv2.IMREAD_GRAYSCALE)
    if g is None: continue
    H,W = g.shape[:2]
    for _ in range(M_POS):
        cw,ch = crop_sizes[rng.randrange(len(crop_sizes))]
        if cw>W or ch>H:                       # rescale-to-fit 재현
            s=min(W/cw,H/ch)*0.95; cw,ch=max(1,int(cw*s)),max(1,int(ch*s))
        px,py = rng.randint(0,max(0,W-cw)), rng.randint(0,max(0,H-ch))
        cells=[_cell_at(g,gi,ax,ay) for ay in _tile_anchors(py,ch,H,TILE) for ax in _tile_anchors(px,cw,W,TILE)]
        if cells: footprints.append(cells)
print(f"footprint 샘플={len(footprints)} (cell 캐시={len(_cell_cache)})")

def _agg(cells,row):
    vals=[float(row.get(c,0.5)) for c in cells]
    return min(vals) if AGG=='min' else sum(vals)/len(vals)
agg_by_cluster={c: np.array([_agg(cells,msym[c]) for cells in footprints]) for c in clusters}
```

### 10.4 목표 reject율 → τ (percentile)

```python
R_PRIMARY=0.25                       # 채택값 (수치 보기 전 고정)
def _tau_at(a,R): return float(np.percentile(a, R*100.0))
print(f"[τ 사전스캔] R={R_PRIMARY} AGG={AGG}\n")
TAU={}
for c in clusters:
    a=agg_by_cluster[c]; amin,amed,amax=float(a.min()),float(np.median(a)),float(a.max())
    neutral=float(np.mean(a==0.5)); tau=_tau_at(a,R_PRIMARY)
    rej=float(np.mean(a<tau)); acc=1-rej
    q25,q50=_tau_at(a,0.25),_tau_at(a,0.50); degen=abs(q25-q50)<1e-6
    ok=(amin<=tau<amed) and (not degen) and (abs(tau-0.5)>1e-6)
    verdict="OK" if ok else ("DEGENERATE" if degen else ("τ≈0.5" if abs(tau-0.5)<=1e-6 else "범위이탈"))
    TAU[c]={'tau':round(tau,4),'accept':round(acc,3),'reject':round(rej,3),
            'agg_min':round(amin,4),'agg_med':round(amed,4),'agg_max':round(amax,4),
            'neutral':round(neutral,3),'verdict':verdict,
            'tau_R20':round(_tau_at(a,0.20),4),'tau_R30':round(_tau_at(a,0.30),4)}
    print(f" cluster {c}: τ={tau:.4f} acc={acc*100:4.1f}% rej={rej*100:4.1f}% "
          f"agg[{amin:.3f}/{amed:.3f}/{amax:.3f}] neutral={neutral*100:4.1f}% → {verdict} "
          f"(R20={TAU[c]['tau_R20']:.4f} R30={TAU[c]['tau_R30']:.4f})")
ok_taus=[TAU[c]['tau'] for c in clusters if TAU[c]['verdict']=='OK']
ds_tau=round(float(np.median(ok_taus)),4) if ok_taus else None
print(f"\n데이터셋 대표 τ (OK median)={ds_tau} ({len(ok_taus)}/{len(clusters)} OK)")
```

### 10.5 저장 · 판정

```python
res={'DS':os.environ['DS'],'mode':'symmetric','agg':AGG,'tile':TILE,'R_primary':R_PRIMARY,
     'seed':SEED,'n_footprints':len(footprints),'n_good':len(good_paths),'crop_n':len(crop_sizes),
     'per_cluster':TAU,'ds_tau':ds_tau,'n_ok':len(ok_taus),'n_cluster':len(clusters)}
out=f"{os.environ['OUT_DIR']}/compat_tau_prescan_{os.environ['DS']}.json"
json.dump(res, open(out,'w'), indent=2, ensure_ascii=False)
print("저장:",out,"→ 채택 τ=",ds_tau,"(--compat_mode symmetric --compat_threshold)")
```

| 관측 | 판정 | 조치 |
|---|---|---|
| 다수 cluster **OK**, ds_tau ∈ (agg_min, median) | τ 확정 성공 | `--compat_mode symmetric --compat_threshold <ds_tau>` |
| 다수 **DEGENERATE** | tiled-compat 한 점 집중 → 무판별 | 해당 DS 게이트 무의미 — 착수 보류 |
| 다수 **τ≈0.5** | neutral 질량 지배 | §3-1 patch-gran support·profiling 재점검(τ 교정 불가) |
| R20/R30 폭 큼 | 분포 평탄 | R=0.25 고정, 사후 변경 금지 |

> **재현 스케일**: `gd._tile_anchors`·`TILE=64`·`AGG=mean`·`dp._context_cell_key/_extract_context_features` = 게이트 `_compat_ok` symmetric 경로와 동일. crop 크기는 roi_selected `defect_bbox` 전역 분포 + rescale-to-fit 재현. 위치는 uniform(random+reject 프록시). cell은 cluster-무관이라 crop↔cluster 결합은 2차 효과(정직 표기).
