# AROMA 정성 비교 그리드 (real | random-synth | aroma-synth) 실행 가이드

**목적**: 배치 차이를 **시각**으로 대조 — real 결함 vs random 배치 합성 vs compatibility(aroma) 배치 합성. FID(평탄)·exp2(동어반복) 한계를 보완하는 정성 증거(논문 §4.5).
**런타임**: **CPU**. **전제**: §3 합성 완료(`$SYN_AROMA`, `$SYN_RAND`에 5셋 images/ 존재), dataset_config 로드.

> 배치 효과: random은 부적합 배경(예 평탄/무관 텍스처)에 결함이 떠 보일 수 있고, aroma(compatibility)는 결함이 자연스러운 배경 위에 놓임 — 이를 나란히 보여줌.

---

## 0. 환경변수 (method_pivot §0 이어서)

```python
import os, json, glob, random
from pathlib import Path

# method_pivot 가이드 §0에서 이미 설정된 것 가정. 없으면 아래:
os.environ.setdefault('AROMA_OUT', f"{os.environ['DRIVE']}/aroma_output")
os.environ.setdefault('DATASET_CONFIG', '/content/AROMA/dataset_config.json')
SYN_AROMA = os.environ.get('SYN_AROMA', f"{os.environ['AROMA_OUT']}/synthetic_mp")
SYN_RAND  = os.environ.get('SYN_RAND',  f"{os.environ['AROMA_OUT']}/synthetic_mp_random")
DATASETS  = ["severstal", "mvtec_carpet", "mvtec_leather", "visa_macaroni", "visa_fryum"]
OUT_PNG   = f"{os.environ['AROMA_OUT']}/qualitative_grid.png"
N_SYNTH   = 2   # 조건당 표시 합성 샘플 수
SEED      = 42

with open(os.environ['DATASET_CONFIG']) as f:
    CFG = json.load(f)
print("SYN_AROMA:", SYN_AROMA, "\nSYN_RAND :", SYN_RAND)
```

---

## 1. 샘플 수집 (real 결함 + 두 조건 합성)

```python
IMG_EXT = ("*.png","*.jpg","*.jpeg","*.bmp")
def globs(d):
    out=[]
    for e in IMG_EXT: out += glob.glob(os.path.join(d, e))
    return sorted(out)

def real_defect_samples(ds, k=1):
    """dataset_config seed_dirs(test/<type>)에서 결함 이미지 k장."""
    rng = random.Random(SEED)
    paths=[]
    for d in CFG.get(ds,{}).get("seed_dirs",[]):
        paths += globs(d)
    if not paths:  # fallback: test/* (good 제외)
        root = Path(CFG[ds]["image_dir"]).parent.parent / "test"
        if root.exists():
            for sub in sorted(root.iterdir()):
                if sub.is_dir() and sub.name!="good": paths += globs(str(sub))
    return rng.sample(paths, min(k,len(paths))) if paths else []

def synth_samples(root, ds, k=N_SYNTH):
    rng = random.Random(SEED)
    paths = globs(os.path.join(root, ds, "images"))
    return rng.sample(paths, min(k,len(paths))) if paths else []

# 수집 확인
for ds in DATASETS:
    r  = real_defect_samples(ds,1)
    sa = synth_samples(SYN_AROMA, ds)
    sr = synth_samples(SYN_RAND, ds)
    print(f"{ds:16} real={len(r)} random={len(sr)} aroma={len(sa)}")
```

---

## 2. 그리드 렌더 (행=데이터셋, 열=[real | random×N | aroma×N])

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

ncols = 1 + 2*N_SYNTH
fig, axes = plt.subplots(len(DATASETS), ncols,
                         figsize=(2.6*ncols, 2.6*len(DATASETS)), squeeze=False)

col_titles = ["real defect"] + [f"random #{i+1}" for i in range(N_SYNTH)] \
                              + [f"aroma #{i+1}" for i in range(N_SYNTH)]

def show(ax, path, title=None, border=None):
    ax.set_xticks([]); ax.set_yticks([])
    if path and os.path.exists(path):
        ax.imshow(Image.open(path).convert("RGB"))
    else:
        ax.text(0.5,0.5,"(none)",ha="center",va="center",fontsize=8,color="#999")
    if title: ax.set_title(title, fontsize=9)
    if border:
        for s in ax.spines.values():
            s.set_color(border); s.set_linewidth(2.5)

for i, ds in enumerate(DATASETS):
    r  = real_defect_samples(ds,1)
    sr = synth_samples(SYN_RAND, ds)
    sa = synth_samples(SYN_AROMA, ds)
    cells = [(r[0] if r else None, "#000")] \
          + [((sr[j] if j<len(sr) else None), "#1f77b4") for j in range(N_SYNTH)] \
          + [((sa[j] if j<len(sa) else None), "#d62728") for j in range(N_SYNTH)]
    for j,(p,bc) in enumerate(cells):
        show(axes[i][j], p, title=(col_titles[j] if i==0 else None), border=bc)
    axes[i][0].set_ylabel(ds.replace("mvtec_","").replace("visa_",""),
                          fontsize=10, fontweight="bold", rotation=90, labelpad=8)

fig.suptitle("Qualitative comparison: real defect (black) | random placement (blue) | "
             "AROMA compatibility placement (red)", fontsize=12, fontweight="bold")
plt.tight_layout(rect=[0,0,1,0.97])
plt.savefig(OUT_PNG, bbox_inches="tight", dpi=140)
print("saved:", OUT_PNG)
```

```python
# 미리보기
from IPython.display import Image as IPImage
IPImage(OUT_PNG)
```

---

## 3. 논문 figure로 내보내기 (선택)

```python
# Drive → 로컬/리포 figure 디렉토리로 복사 후 논문에 삽입
import shutil
DST = "/content/AROMA/AROMA연구분석/Article/figure/fig_qualitative_grid.png"
shutil.copy(OUT_PNG, DST)
print("copied →", DST)
```

논문 §4.5 삽입용 caption 예:
```latex
\begin{figure}[t]
\centering
\includegraphics[width=\textwidth]{figure/fig_qualitative_grid.png}
\caption{Qualitative comparison across the five datasets. Each row shows a real
defect (black border), two random-placement synthetic samples (blue), and two
AROMA compatibility-placement synthetic samples (red). AROMA composites defects
onto contextually appropriate background regions, whereas random placement may
position defects on incompatible or implausible surfaces. This visual evidence
complements the distribution-level metrics (FID/KID/LPIPS), which—measuring the
global image distribution—are insensitive to the localized placement differences
that AROMA is designed to exploit.}
\label{fig:qualitative_grid}
\end{figure}
```

---

## 주의 / 팁

- **severstal**: full-frame strip(1600×256)이라 가로로 길게 표시됨 — 정상. 셀 비율 자동 처리.
- **샘플 수 조정**: `N_SYNTH` 늘리면 조건당 더 많은 예시(열 증가). 데이터셋별 별도 figure 원하면 루프 분리.
- **대표 샘플 선별**: 현재 seeded random. 배치 차이가 뚜렷한 샘플을 수동 지정하려면 `synth_samples`가 반환한 경로 중 골라 리스트 직접 구성.
- **carpet 주의**: 1-cell 퇴화셋이라 random/aroma 배치 차 미미할 수 있음(시각으로도 확인됨).
- 신규 .ipynb 금지 — 본 .md 셀 Colab 복사 실행.
