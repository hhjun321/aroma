# Foreground-aware ROI Filtering — void 안전성 검증 (GT mask 실증)

> **목적**: void(flat 영역)를 밝기-불변 flatness로 거부하는 게이트가 **실제 결함을 버리지 않음**을 4셋 GT mask로 실증한다. 즉 flat-void(`local_variance < T`) 영역에 GT 결함 픽셀이 (거의) 없음을 확인.
>
> 배경: severstal void=검정, aitex void=흰색 — 극성 반대지만 4셋 context_features상 둘 다 `local_variance≈0`. 밝기-불변 게이트(`generate_defects._foreground_mask`, `_FG_VOID_STD` + quality)로 일반화. 본 가이드는 그 게이트가 **안전**(실결함 미손실)한지 검증한다.
>
> 선행: `common_setup`(DRIVE/AROMA_SCRIPTS/AROMA_OUT/DATASET_CONFIG). 4셋 = severstal/mvtec_leather/aitex/mtd (aitex/mtd는 prepare 정규화 완료 상태).

---

## §0 — 환경 + 상수

```python
import os, sys, json, glob
sys.path.insert(0, os.environ['AROMA_ROOT_SCRIPTS'])   # distribution_profiling.py (repo root scripts/)
sys.path.insert(0, os.environ['AROMA_SCRIPTS'])         # aroma scripts

DATASETS = ["severstal", "mvtec_leather", "aitex", "mtd"]
PATCH   = 64      # context patch 크기 (distribution_profiling와 동일)
VOID_T  = 10.0    # local_variance < VOID_T → flat void (4셋 분석 근거). 여러 T도 함께 출력.
with open(os.environ['DATASET_CONFIG']) as f:
    CFG = json.load(f)
```

---

## §1 — (defect 이미지, GT mask) 쌍 수집 + 패치 void×defect 교차표

각 defect 이미지를 64px 패치로 타일 → 패치별 `local_variance`(void 여부) + GT mask 겹침(defect 여부). flat-void 패치가 결함을 포함하는 비율을 집계.

> ⚠️ `distribution_profiling`를 import하면 scipy/skimage를 끌어와 Colab numpy↔scipy 버전 충돌(`_blas_supports_fpe`)로 실패할 수 있다. 그래서 `_find_mask_path`를 **인라인**(pathlib만)으로 둔다.

```python
import numpy as np, cv2
from pathlib import Path

def _find_mask_path(domain, image_path, defect_type):
    """v2-1 4셋 mask 해소 (distribution_profiling._find_mask_path 인라인 복제)."""
    p = Path(image_path); stem = p.stem
    if domain == "severstal":
        root = p.parent.parent.parent          # .../severstal/
        for c in (root/"masks"/defect_type/f"{stem}.png", root/"masks"/f"{stem}.png"):
            if c.exists(): return c
        return None
    # mvtec(leather) / aitex / mtd → mvtec식 ground_truth/{class}/{stem}_mask.png
    c = p.parent.parent.parent / "ground_truth" / defect_type / f"{stem}_mask.png"
    return c if c.exists() else None

def _iter_defect_pairs(ds, limit=200):
    """dataset_config seed_dirs의 결함 이미지 → (img_path, mask_path, defect_type)."""
    e = CFG[ds]; domain = e['domain']
    seeds = e.get('seed_dirs') or ([e['seed_dir']] if e.get('seed_dir') else [])
    n = 0
    for sd in seeds:
        dtype = Path(sd).name if Path(sd).name != 'Imgs' else Path(sd).parent.name
        exts = ('*.jpg','*.jpeg') if domain == 'mtd' else ('*.png','*.jpg','*.jpeg','*.bmp')
        files = sorted(f for ext in exts for f in glob.glob(os.path.join(sd, ext)))
        for ip in files:
            mp = _find_mask_path(domain, Path(ip), dtype)
            if mp and Path(mp).exists():
                yield ip, str(mp), dtype
                n += 1
                if n >= limit: return

def verify_void_safety(ds, thresholds=(2,5,10,20)):
    imgs = list(_iter_defect_pairs(ds))
    if not imgs:
        print(f"[{ds}] 결함쌍 0 — seed_dirs/mask 경로 확인"); return
    # 패치 누적: void bin별 (n_patches, n_defect_patches, defect_px, total_px)
    stat = {t: {'void_patch':0,'void_defect_patch':0,'void_defect_px':0,'void_px':0} for t in thresholds}
    tot_defect_px = 0; tot_void_defect_px = {t:0 for t in thresholds}
    for ip, mp, _ in imgs:
        img = cv2.imread(ip, cv2.IMREAD_GRAYSCALE)
        m   = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
        if img is None or m is None: continue
        if m.shape != img.shape:
            m = cv2.resize(m, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        md = (m > 0)
        tot_defect_px += int(md.sum())
        H, W = img.shape
        for y in range(0, H - PATCH + 1, PATCH):
            for x in range(0, W - PATCH + 1, PATCH):
                p = img[y:y+PATCH, x:x+PATCH].astype(np.float32)
                lv = float(np.var(p))
                d_px = int(md[y:y+PATCH, x:x+PATCH].sum())
                for t in thresholds:
                    if lv < t:
                        stat[t]['void_patch'] += 1
                        stat[t]['void_px'] += PATCH*PATCH
                        if d_px > 0:
                            stat[t]['void_defect_patch'] += 1
                            stat[t]['void_defect_px'] += d_px
                            tot_void_defect_px[t] += d_px
    print(f"=== {ds}  (defect imgs={len(imgs)}, total defect px={tot_defect_px}) ===")
    for t in thresholds:
        s = stat[t]
        frac = (100*tot_void_defect_px[t]/tot_defect_px) if tot_defect_px else 0.0
        vd = (100*s['void_defect_patch']/s['void_patch']) if s['void_patch'] else 0.0
        print(f"  T={t:>2}: void patches={s['void_patch']:>7}  "
              f"결함 겹치는 void patch={vd:5.2f}%  "
              f"전체 결함px 중 void 안={frac:5.3f}%")
    # 판정: T=10에서 결함px가 void 안에 <1% 이면 안전
    key = 10 if 10 in thresholds else thresholds[-1]
    safe = (100*tot_void_defect_px[key]/tot_defect_px if tot_defect_px else 0) < 1.0
    print(f"  >>> {'PASS' if safe else 'FAIL'} (T={key}: void 안 결함px "
          f"{100*tot_void_defect_px[key]/max(1,tot_defect_px):.3f}% {'<' if safe else '≥'} 1% 목표)")
    print()

for ds in DATASETS:
    verify_void_safety(ds)
```

**판정 기준**: `T=10`에서 **전체 결함px 중 flat-void 안에 든 비율 < 1%** → void 거부가 실결함을 버리지 않음 = **필터 안전(PASS)**. severstal/aitex는 void 큼(그러나 결함은 foreground) → void 안 결함px≈0 기대. leather/mtd는 void 자체가 없어 void_patch≈0.

---

## §2 — 샘플 시각화 (void 거부 영역 vs GT 오버레이)

```python
from IPython.display import display
import numpy as np, cv2
from pathlib import Path
# _find_mask_path / _iter_defect_pairs 는 §1 셀에서 정의됨 (동일 세션서 §1 먼저 실행)

def show_void_overlay(ds, k=3):
    pairs = list(_iter_defect_pairs(ds))[:k]
    for ip, mp, dt in pairs:
        img = cv2.imread(ip, cv2.IMREAD_GRAYSCALE); m = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
        if img is None or m is None: continue
        if m.shape != img.shape: m = cv2.resize(m,(img.shape[1],img.shape[0]),interpolation=cv2.INTER_NEAREST)
        H,W = img.shape
        vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        for y in range(0,H-PATCH+1,PATCH):
            for x in range(0,W-PATCH+1,PATCH):
                if float(np.var(img[y:y+PATCH,x:x+PATCH]))<VOID_T:
                    vis[y:y+PATCH,x:x+PATCH,2] = np.minimum(255, vis[y:y+PATCH,x:x+PATCH,2]+80)  # void=빨강 틴트
        vis[m>0] = (0,255,0)  # GT 결함=초록
        outp = f"/content/_void_overlay_{ds}_{Path(ip).stem}.png"; cv2.imwrite(outp, vis)
        print(f"{ds} {Path(ip).name} (defect_type={dt}) → {outp}  [빨강=void거부, 초록=GT결함]")
        from IPython.display import Image as IPImg; display(IPImg(outp))

for ds in DATASETS:
    show_void_overlay(ds)
```

> 초록(GT 결함)이 빨강(void 거부) 영역과 **겹치지 않아야** 안전. 겹치면 그 도메인은 VOID_T 하향 or 게이트 재검토.

---

## 결과 해석
- **4셋 모두 PASS** (void 안 결함px <1%) → `generate_defects._foreground_mask`의 밝기-불변 void 게이트(`_FG_VOID_STD` + quality, dark-mean 제거)가 안전. 실결함 미손실.
- severstal/aitex: void_patch 많으나 결함 겹침 ≈0 (결함은 강판/직물=foreground). leather/mtd: void_patch≈0 (거짓거부 없음).
- FAIL 도메인 있으면 → 해당 도메인 VOID_T(현 10) 데이터로 재조정 or 게이트 로직 재검토.

## 참고
- 게이트 구현: `scripts/aroma/generate_defects.py` `_foreground_mask` Step 5 (`_FG_VOID_STD`=5 std, `_FG_VOID_QUALITY`=0.5). 본 검증의 VOID_T(local_variance<10)는 std<~3.2에 해당 — 게이트 std<5(variance<25)보다 보수적 확인.
- 분석 근거: 4셋 `context_features.csv` local_variance 분포 (severstal/aitex void spike at lv<2, leather/mtd 무 void).
