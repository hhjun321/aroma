# Step 4 — 합성 결과 시각화

> **실행 환경**: CPU
> 3종 데이터셋 합성 결과(`annotations.json`) 로드 후 품질 확인.

## 환경변수

```python
import json, os

os.environ['AROMA_OUT'] = f"{os.environ['DRIVE']}/aroma_output"
```

---

## 1. 3종 데이터셋 annotations 집계

```python
import json, os
from pathlib import Path
from collections import Counter, defaultdict

AROMA_OUT       = os.environ['AROMA_OUT']
DATASET_CONFIG  = os.environ.get('DATASET_CONFIG', '/content/AROMA/dataset_config.json')

with open(DATASET_CONFIG) as f:
    _cfg = json.load(f)
DATASET_KEYS = [k for k in _cfg if not k.startswith('_')]

all_ann = {}
for dk in DATASET_KEYS:
    p = Path(AROMA_OUT) / 'synthetic' / dk / 'annotations.json'
    if p.exists():
        with open(p) as f:
            all_ann[dk] = json.load(f)
        n = len(all_ann[dk])
        n_dry = sum(1 for a in all_ann[dk] if a.get('dry_run'))
        print(f"{dk:20s}  총={n}  dry={n_dry}  실={n - n_dry}")
    else:
        print(f"{dk:20s}  ✗ annotations.json 없음")
```

---

## 2. 클러스터별 생성 분포

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, len(all_ann), figsize=(5 * len(all_ann), 4))
if len(all_ann) == 1:
    axes = [axes]

for ax, (dk, anns) in zip(axes, all_ann.items()):
    real = [a for a in anns if not a.get('dry_run')]
    dist = Counter(str(a['cluster_id']) for a in real)
    labels, counts = zip(*sorted(dist.items())) if dist else ([], [])
    ax.bar(labels, counts)
    ax.set_title(dk)
    ax.set_xlabel('cluster_id')
    ax.set_ylabel('생성 수')

plt.tight_layout()
plt.show()
```

---

## 3. 데이터셋별 샘플 이미지 그리드

```python
from PIL import Image
import matplotlib.pyplot as plt
import random

def show_grid(anns, title, n_cols=5, n_rows=3, seed=0):
    rng = random.Random(seed)
    real = [a for a in anns if not a.get('dry_run') and Path(a['image_path']).exists()]
    samples = rng.sample(real, min(n_cols * n_rows, len(real)))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.5, n_rows * 2.5))
    axes = axes.flatten()
    fig.suptitle(title, fontsize=13)

    for i, ax in enumerate(axes):
        if i < len(samples):
            img = Image.open(samples[i]['image_path']).convert('RGB')
            ax.imshow(img)
            cid = samples[i].get('cluster_id', '?')
            def_ = samples[i].get('deficit', 0)
            ax.set_title(f"c{cid} | d={def_:.2f}", fontsize=8)
        ax.axis('off')

    plt.tight_layout()
    plt.show()

for dk, anns in all_ann.items():
    show_grid(anns, dk)
```

---

## 4. Deficit 상위 샘플 (희귀 조합)

```python
def show_top_deficit(anns, title, top_n=10):
    real = [a for a in anns if not a.get('dry_run') and Path(a['image_path']).exists()]
    top = sorted(real, key=lambda x: x.get('deficit', 0), reverse=True)[:top_n]

    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()
    fig.suptitle(f"{title} — Deficit 상위 {top_n}", fontsize=13)

    for i, ax in enumerate(axes):
        if i < len(top):
            img = Image.open(top[i]['image_path']).convert('RGB')
            ax.imshow(img)
            info = f"c{top[i].get('cluster_id','?')} | d={top[i].get('deficit',0):.3f}\n{top[i].get('cell_key','')}"
            ax.set_title(info, fontsize=7)
        ax.axis('off')

    plt.tight_layout()
    plt.show()

for dk, anns in all_ann.items():
    show_top_deficit(anns, dk)
```

---

## 5. 클러스터별 프롬프트 확인

```python
for dk, anns in all_ann.items():
    print(f"\n{'='*60}")
    print(f"  {dk}")
    print(f"{'='*60}")
    by_cluster = defaultdict(list)
    for a in anns:
        by_cluster[a.get('cluster_id', '?')].append(a)
    for cid in sorted(by_cluster.keys()):
        sample = by_cluster[cid][0]
        print(f"  cluster {cid} ({len(by_cluster[cid])}개)")
        print(f"    prompt : {sample.get('prompt', '-')}")
        print(f"    deficit: {sample.get('deficit', 0):.3f}  score: {sample.get('roi_score', 0):.3f}")
```

---

## 6. 원본 결함 vs 합성 이미지 비교 (단일 ROI)

```python
def show_original_vs_synthetic(anns, dataset_key, n=5, seed=0):
    """원본 결함 이미지(source_roi)와 합성 결과 나란히 비교.
    source_roi 경로가 없으면 image_id 텍스트로 대체 표시."""
    rng = random.Random(seed)
    real = [a for a in anns
            if not a.get('dry_run')
            and Path(a.get('image_path', '')).exists()]
    samples = rng.sample(real, min(n, len(real)))

    if not samples:
        print(f"{dataset_key}: 합성 이미지를 찾을 수 없음")
        return

    fig, axes = plt.subplots(len(samples), 2, figsize=(6, len(samples) * 2.8))
    if len(samples) == 1:
        axes = [axes]
    fig.suptitle(f"{dataset_key} — 원본 vs 합성", fontsize=12)

    for i, s in enumerate(samples):
        src_path = s.get('source_roi', '')
        if src_path and Path(src_path).exists():
            orig = Image.open(src_path).convert('RGB')
            axes[i][0].imshow(orig)
            axes[i][0].set_title('원본 결함', fontsize=8)
        else:
            image_id = s.get('image_id') or Path(src_path).name or '알 수 없음'
            axes[i][0].text(0.5, 0.5, f"image_id:\n{image_id}",
                            ha='center', va='center', fontsize=7, wrap=True,
                            transform=axes[i][0].transAxes)
            axes[i][0].set_facecolor('#e0e0e0')
            axes[i][0].set_title('원본 없음', fontsize=8)
        axes[i][0].axis('off')

        syn = Image.open(s['image_path']).convert('RGB')
        axes[i][1].imshow(syn)
        axes[i][1].set_title('합성', fontsize=8)
        axes[i][1].axis('off')

    plt.tight_layout()
    plt.show()

for dk, anns in all_ann.items():
    show_original_vs_synthetic(anns, dk)
```
