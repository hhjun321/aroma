# step4 — ControlNet 학습 + τ 확정 실행 가이드 (sym_final)

> 이 문서는 `colab_execute_new/_SPEC.md`를 **정본**으로 따른다. STEP 0 환경 셀은 `_SPEC §1`을 그대로 복사한 것이며, 명령은 `_SPEC §3 step4`(4a/4b/4c)만 사용한다. env·output 루트 규약을 재발명하지 않는다.

**목적**: step5(AROMA arm 생성)가 소비하는 **모든 생성 선결물을 이 단계에서 확정**한다.
1. **ControlNet 학습본** — step0(profiling)·step3(roi_candidates)로 학습 데이터를 빌드(4a)하고 SD1.5 + canny ControlNet을 fine-tune(4b)하여 `$CN_MODELS/{ds}/best_model` 산출.
2. **τ(compat_threshold) 사전스캔**(4c, 본 문서에 절차 인라인) — symmetric 스케일 + 64px 타일링-aware로 데이터셋별 `ds_tau` 확정(`S('compat_gate',ds)/compat_tau_prescan_{ds}.json`). **aitex 전용**으로 AR/텍스처 게이트 임계(`AR_T`/`TEX_T`)도 함께 사전스캔.

> τ·AR·TEX는 step5의 `--compat_threshold`·`--cn_ar_threshold`·`--texture-dist-threshold`가 그대로 소비한다. **생성 직전 재발명 금지** — step4 말미에서 모두 확정한다(`_SPEC §3 step4 4c`).

**실행 순서 체인**: `phase0 → step1 → step2 → step3 → **step4(CN 학습 + τ 확정, 이 문서)** → step5(생성)`.
- step4는 **step3 뒤·step5 앞**이다. step0(profiling)·step3(roi)의 산출을 입력으로 쓰고, step5의 `generate_defects --method controlnet --compat_mode symmetric`이 이 CN 모델(`$CN_MODELS/{ds}/best_model`)과 τ/AR/TEX 임계를 소비한다.

**환경**: 학습(4b)은 **GPU 필수** (Colab Pro A100 권장). 4a(빌드)·4c(τ 사전스캔)는 **CPU**.

**전제**: phase0(profiling) · step1(complexity) · step2(prompts) · step3(roi) 완료. 즉 데이터셋별로 아래가 존재해야 한다.
- `S('profiling',ds)/morphology_features.csv` · `context_features.csv` · `recommended_config.yaml`
- `S('profiling',ds)/compatibility_matrix.json` — **`matrix_symmetric` 키 필수**(신 profiling). 없으면 4c hard-fail (구 profile은 `--compat_mode symmetric` 불가).
- `S('roi',ds)/roi_candidates.json` · `roi_selected.json`

**데이터셋**: v2-1 4종 `severstal · mvtec_leather · mtd · aitex`. aitex = tiled(256×256/stride128, single-class).

---

## STEP 0 — 공통 환경 셀 (`_SPEC §1` 그대로 — 수정 금지)

```python
import os, json

# ===== 공통 환경 (sym_final 전 문서 동일 — 수정 금지) =====
os.environ['DRIVE']          = '/content/drive/MyDrive/data/Aroma'
os.environ['AROMA_REF']      = '/content/AROMA'
os.environ['AROMA_SCRIPTS']  = '/content/AROMA/scripts/aroma'
os.environ['AROMA_OUT']      = f"{os.environ['DRIVE']}/aroma_output"
os.environ['AROMA_DATA']     = f"{os.environ['DRIVE']}"
os.environ['DATASET_CONFIG'] = os.environ.get('DATASET_CONFIG', '/content/AROMA/dataset_config.json')
# ===== 단일 버전 루트 (stage-first: {stage}/{ds}) =====
os.environ['SYM_ROOT'] = f"{os.environ['AROMA_OUT']}/sym_final"
os.environ['CN_MODELS'] = f"{os.environ['SYM_ROOT']}/controlnet_models"   # ControlNet 학습본(step4 산출, step5 소비)
def S(stage, ds=None):
    p = f"{os.environ['SYM_ROOT']}/{stage}"
    return f"{p}/{ds}" if ds else p

DATASETS = ["severstal", "mvtec_leather", "mtd", "aitex"]   # v2-1 4종
with open(os.environ['DATASET_CONFIG']) as f: CFG = json.load(f)
def normal_dir(ds): return CFG[ds]["image_dir"]                 # aitex → aitex_tiled/train/good
def is_multi(ds):   return CFG[ds].get("class_mode") == "multi" # aitex=single (자동)
```

### 전제 확인 (모두 ✓여야 진행)

```python
import pathlib

need = []
for ds in DATASETS:
    prof, roi = S('profiling', ds), S('roi', ds)
    checks = {
        "morphology_features.csv": f"{prof}/morphology_features.csv",
        "context_features.csv":    f"{prof}/context_features.csv",
        "recommended_config.yaml": f"{prof}/recommended_config.yaml",
        "compatibility_matrix.json": f"{prof}/compatibility_matrix.json",  # 4c τ 사전스캔 (matrix_symmetric)
        "roi_candidates.json":     f"{roi}/roi_candidates.json",
        "roi_selected.json":       f"{roi}/roi_selected.json",             # 4c crop 크기 출처
    }
    for name, p in checks.items():
        ok = pathlib.Path(p).exists()
        if not ok: need.append(p)
        print(f"{'✓' if ok else '✗'} {ds:14s} {name}: {p}")
print("\nMISSING:", len(need))

# matrix_symmetric 키 존재 확인 (4c hard-fail 예방 — 신 profiling 필수)
import json as _json
for ds in DATASETS:
    cj = f"{S('profiling', ds)}/compatibility_matrix.json"
    if not pathlib.Path(cj).exists():
        print(f"✗ {ds:14s} compat json 없음"); continue
    has = "matrix_symmetric" in _json.load(open(cj))
    print(f"{'✓' if has else '✗ 구 profile — phase0 재실행 필요'} {ds:14s} matrix_symmetric")
```

---

## STEP 1 — 패키지 설치 (CPU)

```python
!pip install diffusers transformers accelerate safetensors -q
!pip install scikit-image opencv-python-headless -q
# xformers는 선택 (없으면 SDPA로 자동 fallback)
```

```python
!nvidia-smi
```

---

## STEP 4a — ControlNet 학습 데이터 빌드 (`build_train_jsonl.py`, CPU, 4종)

학습(4b)과 생성(step5)이 **같은 hint/prompt 생성기**를 쓰도록 `build_train_jsonl.py`로 빌드한다 (`--style technical` 고정). 입력은 step0(profiling)·step3(roi) 산출, 출력은 `S('cn_data', ds)`.

```python
for ds in DATASETS:
    os.environ['DS']      = ds
    os.environ['PROF']    = S('profiling', ds)
    os.environ['ROI']     = S('roi', ds)
    os.environ['CN_DATA'] = S('cn_data', ds)
    print(f"\n=== build {ds} ===")
    !python $AROMA_SCRIPTS/build_train_jsonl.py \
        --morphology_csv   $PROF/morphology_features.csv \
        --roi_candidates   $ROI/roi_candidates.json \
        --context_features $PROF/context_features.csv \
        --config           $PROF/recommended_config.yaml \
        --output_dir       $CN_DATA \
        --style            technical
```

**확인**: 각 `S('cn_data',ds)/train.jsonl` 라인 수 출력. `background_type collapsed` WARN이 뜨면 그대로 진행하되 기록한다(학습·생성 동일 분포이므로 parity 유지).

```python
for ds in DATASETS:
    p = f"{S('cn_data', ds)}/train.jsonl"
    n = sum(1 for _ in open(p)) if pathlib.Path(p).exists() else 0
    print(f"{ds}: train.jsonl lines = {n}")
```

---

## STEP 4b — ControlNet 학습 (`train_controlnet.py`, GPU, 데이터셋당 1세션 권장)

공통: SD1.5 base + canny ControlNet init, `--mixed_precision fp16`, `--gradient_checkpointing`, early stopping, `--checkpointing_steps 500`, `--save_optimizer_state`, `--save_fp16`, `--resume_from_checkpoint latest`.

산출: `$CN_MODELS/$DS/best_model/` (= `S('controlnet_models', ds)/best_model` = `sym_final/controlnet_models/{ds}/best_model`). `ControlNetModel.from_pretrained`로 로드 가능하며 step5가 이를 소비한다.

**데이터셋별 차등** (`_SPEC §3 step4` / `controlnet_aroma_arm_execute.md` STEP 4 규약):

| 데이터셋 | epochs | `--augment` | grayscale target | early_stopping_patience |
|---------|--------|-------------|------------------|-------------------------|
| severstal | 100 | OFF | 기본 ON (강판 grayscale) | 20 |
| mtd | 150 | **ON** | 기본 ON (자성타일 grayscale) | 20 |
| aitex (tiled) | 150 | **ON** | 기본 ON (직물 grayscale — mtd 준용) | 20 |
| mvtec_leather | 300 | **ON** | **OFF** (`--no_force_grayscale_target`, 컬러 가죽) | 30 |

> **권장**: 데이터셋당 1세션. severstal는 최장(수천 crop)이므로 단독 세션 권장. 세션이 끊기면 **동일 셀 재실행** → `--resume_from_checkpoint latest`가 마지막 checkpoint에서 이어서 학습한다.

```python
# ── severstal (100ep, no-aug, grayscale ON) — 최장, 단독 세션 권장 ──
os.environ['DS'] = "severstal"; os.environ['CN_DATA'] = S('cn_data', "severstal")
!python /content/AROMA/scripts/train_controlnet.py \
    --data_dir $CN_DATA \
    --output_dir $CN_MODELS/$DS \
    --mixed_precision fp16 \
    --gradient_checkpointing \
    --num_train_epochs 100 \
    --early_stopping_patience 20 \
    --checkpointing_steps 500 \
    --save_optimizer_state \
    --save_fp16 \
    --resume_from_checkpoint latest
```

```python
# ── mtd (150ep, augment ON, grayscale ON) ──
os.environ['DS'] = "mtd"; os.environ['CN_DATA'] = S('cn_data', "mtd")
!python /content/AROMA/scripts/train_controlnet.py \
    --data_dir $CN_DATA \
    --output_dir $CN_MODELS/$DS \
    --mixed_precision fp16 \
    --gradient_checkpointing \
    --num_train_epochs 150 \
    --augment \
    --early_stopping_patience 20 \
    --checkpointing_steps 500 \
    --save_optimizer_state \
    --save_fp16 \
    --resume_from_checkpoint latest
```

```python
# ── aitex (tiled/single: 150ep, augment ON, grayscale 기본 ON — mtd 준용) ──
os.environ['DS'] = "aitex"; os.environ['CN_DATA'] = S('cn_data', "aitex")
!python /content/AROMA/scripts/train_controlnet.py \
    --data_dir $CN_DATA \
    --output_dir $CN_MODELS/$DS \
    --mixed_precision fp16 \
    --gradient_checkpointing \
    --num_train_epochs 150 \
    --augment \
    --early_stopping_patience 20 \
    --checkpointing_steps 500 \
    --save_optimizer_state \
    --save_fp16 \
    --resume_from_checkpoint latest
```

```python
# ── mvtec_leather (300ep, augment ON, grayscale OFF — 컬러 가죽) ──
os.environ['DS'] = "mvtec_leather"; os.environ['CN_DATA'] = S('cn_data', "mvtec_leather")
!python /content/AROMA/scripts/train_controlnet.py \
    --data_dir $CN_DATA \
    --output_dir $CN_MODELS/$DS \
    --mixed_precision fp16 \
    --gradient_checkpointing \
    --num_train_epochs 300 \
    --augment \
    --no_force_grayscale_target \
    --early_stopping_patience 30 \
    --checkpointing_steps 500 \
    --save_optimizer_state \
    --save_fp16 \
    --resume_from_checkpoint latest
```

> 학습 중 validation 이미지가 `$CN_MODELS/{ds}/validation/step_*/`에 주기 저장된다.

### validation 육안 게이트 (간단)

전량 생성(step5) 전에 각 데이터셋 validation 이미지를 **육안으로 한 번** 본다. 결함 텍스처가 배경과 이질적이지 않고 그럴듯하면 통과. leather가 끝내 비현실적이면 학습 하이퍼(augment/epoch) 재조정 후 재학습이 원칙이며, 그래도 안 되면 step5에서 "controlnet arm 부적합"으로 정직 보고. (본격 pilot 육안·`blank_rate`·AR/텍스처 게이트는 step5의 소관.)

```python
from PIL import Image
import matplotlib.pyplot as plt
for ds in DATASETS:
    vd = sorted(pathlib.Path(f"{os.environ['CN_MODELS']}/{ds}/validation").glob("step_*"))
    if not vd: 
        print(f"{ds}: validation 없음"); continue
    imgs = sorted(vd[-1].glob("*.png")) + sorted(vd[-1].glob("*.jpg"))
    imgs = imgs[:4]
    if not imgs: continue
    fig, axes = plt.subplots(1, len(imgs), figsize=(16, 4)); fig.suptitle(f"{ds} — {vd[-1].name}")
    for ax, p in zip(axes if len(imgs) > 1 else [axes], imgs):
        ax.imshow(Image.open(p)); ax.axis('off')
    plt.show()
```

---

## STEP 4c — τ (+aitex AR/TEX) 사전스캔 (CPU) — step5 생성 선결

step5의 `generate_defects --compat_mode symmetric --compat_threshold <τ>`가 소비하는 **τ·AR·TEX 임계를 여기서 모두 확정**한다(생성 직전 재사전스캔 금지). 신 profiling(`matrix_symmetric`, phase0) 필수.

### τ 사전스캔 (symmetric 스케일, 타일링-aware)

**목표 reject율 R=0.25를 수치 보기 전 고정.** τ=0.5 금지(0.5=legacy 확률 스케일 중립값). **신 profiling(`matrix_symmetric`, phase0) 필수** — 구 profile hard-fail. severstal/aitex 大 → Colab만.

절차: `matrix_symmetric`은 per-cluster max-norm으로 [0,1]. good 배경에 실 crop을 uniform 배치(rescale-to-fit 재현)하고 **generate_defects의 게이트와 동일한 64px 타일링 + mean-compat**를 재현해, cluster별 tiled mean-compat 분포에서 목표 reject율 R=0.25 percentile을 τ로 잡는다. 재현 스케일이 게이트 `_compat_ok` symmetric 경로와 동일해야 유효(`gd._tile_anchors`·`TILE=64`·`AGG=mean`·`dp._context_cell_key/_extract_context_features`).

**4c-i. per-DS 실행** (아래 5개 셀을 DS마다 순서대로 실행, severstal/mvtec_leather/mtd/aitex 4종 반복):

```python
# [4c-0] per-DS env 세팅 (DS만 교체하며 아래 4c-1~4c-5 재실행)
import os
DS = "severstal"                       # ← 데이터셋마다 교체
os.environ['DS']          = DS
os.environ['COMPAT_JSON'] = f"{S('profiling', DS)}/compatibility_matrix.json"
os.environ['SEL_AROMA']   = S('roi', DS)                 # roi_selected.json 출처 (실 crop 크기)
os.environ['OUT_DIR']     = S('compat_gate', DS)         # 산출: compat_tau_prescan_{ds}.json
os.makedirs(os.environ['OUT_DIR'], exist_ok=True)
# AROMA_REF·DATASET_CONFIG는 STEP 0에서 세팅됨
for k in ('COMPAT_JSON','SEL_AROMA','OUT_DIR'):
    print(k, '=', os.environ[k], ' exists:', os.path.exists(os.environ[k]))
```

```python
# [4c-1] 로드 — matrix_symmetric + 타일링 헬퍼 (generate_defects 게이트와 동일 경로)
import os, sys, json, csv, numpy as np, collections, random
os.environ['DATASET_CONFIG'] = os.environ.get('DATASET_CONFIG', f"{os.environ['AROMA_REF']}/dataset_config.json")
with open(os.environ['DATASET_CONFIG']) as f: _dscfg = json.load(f)
os.environ['NORMAL_DIR']   = _dscfg[os.environ['DS']]['image_dir']
os.environ['ROI_SELECTED'] = f"{os.environ['SEL_AROMA']}/roi_selected.json"
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

```python
# [4c-2] 실 crop 크기(roi_selected defect_bbox) + good 이미지 샘플
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

```python
# [4c-3] 타일링 mean-compat 분포 (게이트 재현: rescale-to-fit + 64px 타일)
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
        if cw>W or ch>H:                       # rescale-to-fit 재현 (generate_defects 0.95 margin)
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

```python
# [4c-4] 목표 reject율 R=0.25 → τ (percentile). R20/R30은 민감도 참고만.
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

```python
# [4c-5] 저장 (step5가 compat_tau_prescan_{ds}.json 의 ds_tau 키를 파일로 소비)
res={'DS':os.environ['DS'],'mode':'symmetric','agg':AGG,'tile':TILE,'R_primary':R_PRIMARY,
     'seed':SEED,'n_footprints':len(footprints),'n_good':len(good_paths),'crop_n':len(crop_sizes),
     'per_cluster':TAU,'ds_tau':ds_tau,'n_ok':len(ok_taus),'n_cluster':len(clusters)}
out=f"{os.environ['OUT_DIR']}/compat_tau_prescan_{os.environ['DS']}.json"
json.dump(res, open(out,'w'), indent=2, ensure_ascii=False)
print("저장:",out,"→ 채택 τ=",ds_tau,"(--compat_mode symmetric --compat_threshold)")
```

**판정표** (per-cluster verdict 종합):

| 관측 | 판정 | 조치 |
|---|---|---|
| 다수 cluster **OK**, ds_tau ∈ (agg_min, median) | τ 확정 성공 | step5에서 `--compat_mode symmetric --compat_threshold <ds_tau>` |
| 다수 **DEGENERATE** | tiled-compat 한 점 집중 → 무판별 | 해당 DS 게이트 무의미 — 착수 보류 |
| 다수 **τ≈0.5** | neutral 질량 지배 | patch-gran support·profiling 재점검(τ 교정 불가) |
| R20/R30 폭 큼 | 분포 평탄 | R=0.25 고정, 사후 변경 금지 |

> **정직 표기**: 위치는 uniform(random+reject 프록시), cell은 cluster-무관이라 crop↔cluster 결합은 2차 효과. crop 크기는 roi_selected `defect_bbox` 전역 분포 + rescale-to-fit 재현.

**4c-ii. τ 확정값 확인** (4종 §10 완료 후):

```python
import json, os, pathlib
TAU = {}
for ds in DATASETS:
    p = f"{S('compat_gate', ds)}/compat_tau_prescan_{ds}.json"
    if pathlib.Path(p).exists():
        TAU[ds] = json.load(open(p)).get('ds_tau')
    print(f"{'✓' if TAU.get(ds) else '✗'} {ds:14s} ds_tau={TAU.get(ds)}  {p}")
    assert TAU.get(ds) is None or 0.0 < TAU[ds] < 0.5, f"{ds} τ 이상(0.5 금지) — §10 재확인"
```

> `verdict`가 다수 DEGENERATE/`τ≈0.5`면 §10.5 판정표대로 조치(게이트 무의미 → 착수 보류, 또는 profiling 재점검). τ는 반드시 (agg_min, median) 구간·τ≠0.5여야 유효.

### aitex 전용 — AR / 텍스처 게이트 임계 사전스캔

aitex는 elongated 결함이 많아 ControlNet 512² squash 왜곡 → AR 초과 ROI는 **copy_paste 폴백**(개수·bbox parity 유지). `controlnet_aroma_arm_execute.md` STEP 5-0/5-0b로 발동률을 CPU 예측 후 임계 확정(발동률 ≤ 50% 권장, 텍스처 거부율 10~50% 밴드). pilot 육안으로 왜곡 최악 케이스 확인.

```python
# 5-0. AR 분포 스캔 → fallback% 예측 (roi_selected.json defect_bbox만으로 완전 결정)
import json, numpy as np, os, pathlib
sel = json.load(open(f"{S('roi','aitex')}/roi_selected.json"))
ars = [max(e['defect_bbox'][2], e['defect_bbox'][3]) / min(e['defect_bbox'][2], e['defect_bbox'][3])
       for e in sel if e.get('defect_bbox') and min(e['defect_bbox'][2], e['defect_bbox'][3]) > 0]
ars = np.array(ars)
print(f"aitex n={len(ars)} AR med={np.median(ars):.2f} p90={np.percentile(ars,90):.2f} max={ars.max():.1f}")
for t in (2.0, 2.5, 3.0, 4.0, 6.0, 8.0):
    print(f"  AR>{t}: fallback {100*(ars>t).mean():.0f}%")
# 판정: fallback ≤ 50%(이상 ≤ 30%) 되는 최소 임계 선택. 2.5 초과 채택 시 step5 pilot에서 AR 상위 ROI 육안 재검.
```

```python
# 5-0b. 텍스처 거리 분포 스캔 → 거부율 예측 (적정 밴드 10~50%)
import sys, random
from PIL import Image
sys.path.insert(0, os.environ['AROMA_SCRIPTS'])
import generate_defects as gd
TEX_CAND = [0.15, 0.20, 0.25, 0.30, 0.35]
prng = random.Random(0)
normals = gd.load_normal_images(normal_dir('aitex'))
descs = []
for e in prng.sample(sel, min(40, len(sel))):
    bb, mp, ip = e.get("defect_bbox"), e.get("defect_mask_path"), e.get("image_path")
    if not (bb and mp and ip and pathlib.Path(mp).exists() and pathlib.Path(ip).exists()): continue
    d = gd._source_bg_descriptor(np.asarray(Image.open(ip).convert("RGB")),
                                 np.asarray(Image.open(mp).convert("L")), tuple(int(v) for v in bb))
    if d is not None: descs.append((d, bb))
dists = []
for d, bb in descs:
    for _ in range(5):
        nim = np.asarray(Image.open(prng.choice(normals)).convert("RGB"))
        h, w = nim.shape[:2]; bw, bh = min(int(bb[2]), w), min(int(bb[3]), h)
        x, y = prng.randint(0, max(0, w - bw)), prng.randint(0, max(0, h - bh))
        dists.append(gd._texture_distance(d, gd._texture_descriptor(nim[y:y+bh, x:x+bw])))
a = np.array(dists)
print(f"aitex n_desc={len(descs)} pairs={len(a)} dist med={np.median(a):.3f} p90={np.percentile(a,90):.3f}")
print("거부율: " + "  ".join(f"@{t}={100*(a>t).mean():.0f}%" for t in TEX_CAND))
# 판정: 거부율 10~50% 밴드의 임계 선택(0%=무의미, 80%+=fallback 폭주). n_desc 급감 시 게이트 자동통과(0=OFF 동일).
```

```python
# 5-0c. 사전스캔 + pilot 육안으로 확정 → JSON 저장 (step5가 파일로 읽어 소비)
AR_T, TEX_T = 2.5, 0.25   # ← 위 두 스캔/pilot 결과로 교체. 0=OFF
pathlib.Path(S('compat_gate','aitex')).mkdir(parents=True, exist_ok=True)
json.dump({'ar_threshold': AR_T, 'tex_threshold': TEX_T},
          open(f"{S('compat_gate','aitex')}/ar_tex_prescan_aitex.json", 'w'), indent=2)
print(f"aitex AR_T={AR_T} TEX_T={TEX_T} → {S('compat_gate','aitex')}/ar_tex_prescan_aitex.json 저장")
```

> **확정값은 step5가 파일로 소비**: τ=`compat_tau_prescan_{ds}.json`(키 `ds_tau`), aitex AR/TEX=`ar_tex_prescan_aitex.json`(키 `ar_threshold`·`tex_threshold`). 문서(세션) 분리 → 파일 인터페이스 필수. 3종(severstal/mvtec_leather/mtd)은 AR/텍스처 게이트 미사용(`_SPEC §3 step5`: aitex 전용).
> aitex fallback이 매우 높으면(과거 임계 2.5에서 98% 실측) aroma-sym 대부분이 copy_paste가 되어 "ControlNet 생성 기여 제한"으로 정직 보고 — 폴백 자체는 parity를 해치지 않는다. 임계를 무리하게 올려(왜곡 허용) fallback을 낮추는 것은 pilot 육안 통과 없이 금지.

---

## 확인 — best_model 산출

step5 진입 전, 4종 모두 `$CN_MODELS/{ds}/best_model/`이 존재해야 한다(step5 선결 assert).

```python
for ds in DATASETS:
    bm = pathlib.Path(f"{os.environ['CN_MODELS']}/{ds}/best_model")
    ok = bm.exists() and any(bm.iterdir())
    print(f"{'✓' if ok else '✗'} {ds:14s} best_model: {bm}")
```

> ✓가 아니면 step5(생성)로 넘어가지 말 것 — `generate_defects --controlnet_path $CN_MODELS/$DS/best_model`가 선결 assert에서 실패한다.

---

## 공통 무결성 / 정직 (`_SPEC §5`)

- **사후 튜닝 금지**: epochs·augment·grayscale·patience(학습)와 τ·AR·TEX(사전스캔)는 확정값을 그대로 쓰고, 결과 보고 후 변경하지 않는다.
- **prescan 필수**: τ·AR·텍스처 임계는 CPU 사전스캔 확정값(τ는 목표 R=0.25 percentile, **τ=0.5 금지**). mtd 값을 aitex에 무검증 일괄 적용 금지 — aitex는 tiled/single 준용이 규약으로 확정된 것이며, 임의 전용이 아니다.
- **aitex는 tile-level·single-class** → 절대값을 타 데이터셋과 직접 비교 금지, Δ만 유효.
- **테스트 코드 신규 작성·pytest 금지**(CLAUDE.md). 검증은 Colab 실행으로.
- **`--local_staging` 미사용**: ControlNet 경로는 sidecar 캐시가 Drive 직결이어야 세션 재개 시 살아남는다. 학습(4b)도 Drive 직결 output_dir을 유지한다.
- **시간 실측·처리량 벤치 미수행** (load-test policy).
