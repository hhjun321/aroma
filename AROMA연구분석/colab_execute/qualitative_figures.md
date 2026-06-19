# Qualitative Figures — Colab 실행 가이드

**목적**: 논문용 정성적 비교 그림 생성 — Normal / Real Defect / Random Synth / AROMA Synth 4-열 그리드  
**전제**: Step 4 (AROMA synthesis) + generate_random.py 완료, synthetic/annotations.json 존재  
**런타임**: CPU 가능 (GPU 불필요)

---

## Figure 목록

| ID | 내용 | 파일명 |
|----|------|--------|
| A | Normal / Real / Random / AROMA 4-열 비교 (3개 데이터셋 행) | `fig_qualitative_comparison.png` |
| B | AROMA synthesis pipeline 단계별 시각화 (normal → roi → composite) | `fig_qualitative_pipeline.png` |
| C | AROMA 다양성 그리드 (동일 배경, 다른 ROI placement) | `fig_qualitative_diversity.png` |

---

## 셀 1 — 환경변수

```python
import os
from pathlib import Path

# 00_setup.md 실행 후 이 셀 실행
AROMA_OUT = os.environ.get('AROMA_OUT', f"{os.environ['DRIVE']}/aroma_output")
FIGURE_OUT = f"{os.environ['DRIVE']}/figures/qualitative"
os.makedirs(FIGURE_OUT, exist_ok=True)

# 복잡도 스펙트럼 커버하는 3개 데이터셋
TARGET_DATASETS = ['mvtec_cable', 'isp_LSM_1', 'visa_cashew']
print(f"AROMA_OUT: {AROMA_OUT}")
print(f"FIGURE_OUT: {FIGURE_OUT}")
```

---

## 셀 2 — 이미지 경로 수집 함수

```python
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
import random

def load_dataset_config(config_path='/content/AROMA/dataset_config.json'):
    with open(config_path) as f:
        return json.load(f)

def get_normal_images(ds, cfg, n=4):
    """정상(배경) 이미지 경로 샘플링"""
    image_dir = cfg[ds]['image_dir']
    paths = sorted(Path(image_dir).glob('**/*.png')) + sorted(Path(image_dir).glob('**/*.jpg'))
    random.seed(42)
    return random.sample(paths, min(n, len(paths)))

def get_real_defect_images(ds, cfg, n=4):
    """실제 결함 이미지 경로 샘플링 (test set)"""
    test_dir = cfg[ds].get('test_dir') or cfg[ds].get('image_dir', '').replace('/train/', '/test/')
    # MVTec 구조: test/{defect_type}/*.png (good/ 제외)
    all_paths = []
    for subdir in Path(test_dir).iterdir():
        if subdir.is_dir() and subdir.name != 'good':
            all_paths.extend(sorted(subdir.glob('*.png')))
            all_paths.extend(sorted(subdir.glob('*.jpg')))
    if not all_paths:
        # fallback: test_defect_dir 키
        test_dir_defect = cfg[ds].get('test_defect_dir', test_dir)
        all_paths = sorted(Path(test_dir_defect).glob('**/*.png'))
    random.seed(42)
    return random.sample(all_paths, min(n, len(all_paths)))

def get_synth_images(ds, synth_base, n=4):
    """합성 이미지 경로 샘플링 (annotations.json의 image_path 사용)"""
    ann_path = Path(synth_base) / ds / 'annotations.json'
    if not ann_path.exists():
        print(f"  WARNING: annotations.json 없음: {ann_path}")
        return []
    with open(ann_path) as f:
        anns = json.load(f)
    # annotations.json 구조: {"annotations": [...]} 또는 flat list
    entries = anns if isinstance(anns, list) else anns.get('annotations', [])
    paths = [e['image_path'] for e in entries if Path(e['image_path']).exists()]
    random.seed(42)
    return [Path(p) for p in random.sample(paths, min(n, len(paths)))]

def load_img_rgb(p, size=256):
    """PIL Image → numpy RGB (정사각 크롭 후 리사이즈)"""
    img = Image.open(str(p)).convert('RGB')
    w, h = img.size
    s = min(w, h)
    img = img.crop(((w-s)//2, (h-s)//2, (w+s)//2, (h+s)//2))
    return np.array(img.resize((size, size)))

print("함수 정의 완료")
```

---

## 셀 3 — Figure A: 4열 비교 그리드

```python
cfg = load_dataset_config()

DATASET_LABELS = {
    'mvtec_cable': 'MVTec Cable\n(MCI=0.66)',
    'isp_LSM_1':   'ISP LSM-1\n(MCI=0.44)',
    'visa_cashew': 'VisA Cashew\n(MCI=0.38)',
}
COL_LABELS = ['Normal', 'Real Defect', 'Random Synth', 'AROMA Synth']
N_COLS = 4
N_DS   = len(TARGET_DATASETS)

fig, axes = plt.subplots(N_DS, N_COLS, figsize=(N_COLS * 2.8, N_DS * 2.8 + 0.5))
fig.suptitle('Synthesis Quality: Normal / Real / Random / AROMA', fontsize=13, y=1.01)

for col_i, label in enumerate(COL_LABELS):
    axes[0, col_i].set_title(label, fontsize=10, fontweight='bold')

for row_i, ds in enumerate(TARGET_DATASETS):
    print(f"Processing {ds}...")
    # 이미지 경로 수집
    normals  = get_normal_images(ds, cfg, n=1)
    reals    = get_real_defect_images(ds, cfg, n=1)
    randoms  = get_synth_images(ds, f"{AROMA_OUT}/synthetic_random", n=1)
    aromas   = get_synth_images(ds, f"{AROMA_OUT}/synthetic", n=1)

    imgs = [
        load_img_rgb(normals[0])  if normals  else np.zeros((256,256,3), dtype=np.uint8),
        load_img_rgb(reals[0])    if reals    else np.zeros((256,256,3), dtype=np.uint8),
        load_img_rgb(randoms[0])  if randoms  else np.zeros((256,256,3), dtype=np.uint8),
        load_img_rgb(aromas[0])   if aromas   else np.zeros((256,256,3), dtype=np.uint8),
    ]

    for col_i, img in enumerate(imgs):
        ax = axes[row_i, col_i]
        ax.imshow(img)
        ax.axis('off')
        if col_i == 0:
            ax.set_ylabel(DATASET_LABELS.get(ds, ds), fontsize=9, rotation=90,
                          labelpad=4, va='center', ha='right')
            ax.yaxis.set_label_position('left')

plt.tight_layout()
out_path = f"{FIGURE_OUT}/fig_qualitative_comparison.png"
plt.savefig(out_path, dpi=150, bbox_inches='tight')
plt.show()
print(f"저장: {out_path}")
```

---

## 셀 4 — Figure B: AROMA Pipeline 단계 시각화

```python
# 하나의 예시 이미지로 pipeline 단계 시각화
# Normal → (ROI 선택 영역 하이라이트) → Synthesis Result

TARGET_DS = 'mvtec_cable'  # ← 변경 가능
STAGE_LABELS = ['Normal Background', 'ROI Region', 'AROMA Composite']

ann_path = Path(AROMA_OUT) / 'synthetic' / TARGET_DS / 'annotations.json'
with open(ann_path) as f:
    anns = json.load(f)
entries = anns if isinstance(anns, list) else anns.get('annotations', [])

# 유효한 첫 번째 항목 선택
entry = None
for e in entries:
    if Path(e['image_path']).exists() and Path(e.get('normal_image', '')).exists():
        entry = e
        break

if entry is None:
    print("유효한 항목 없음. annotations.json의 image_path 경로를 확인하세요.")
else:
    normal_img  = load_img_rgb(entry['normal_image'])
    composite   = load_img_rgb(entry['image_path'])

    # ROI 영역: source_roi bbox 시각화 (없으면 차이 이미지로 대체)
    diff = np.abs(composite.astype(float) - normal_img.astype(float)).astype(np.uint8)
    diff_vis = np.clip(diff * 5, 0, 255).astype(np.uint8)

    stages = [normal_img, diff_vis, composite]

    fig, axes = plt.subplots(1, 3, figsize=(9, 3.2))
    fig.suptitle(f'AROMA Synthesis Pipeline ({TARGET_DS})', fontsize=11)
    for i, (ax, img, lbl) in enumerate(zip(axes, stages, STAGE_LABELS)):
        ax.imshow(img)
        ax.set_title(lbl, fontsize=9)
        ax.axis('off')
    plt.tight_layout()
    out_path = f"{FIGURE_OUT}/fig_qualitative_pipeline.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"저장: {out_path}")
```

---

## 셀 5 — Figure C: AROMA 다양성 그리드

```python
# 동일 데이터셋에서 여러 AROMA 합성 이미지 비교 (다양성 시각화)
TARGET_DS = 'mvtec_cable'
N_SAMPLES = 8  # 2행 × 4열

ann_path = Path(AROMA_OUT) / 'synthetic' / TARGET_DS / 'annotations.json'
with open(ann_path) as f:
    anns = json.load(f)
entries = anns if isinstance(anns, list) else anns.get('annotations', [])

valid_paths = [e['image_path'] for e in entries if Path(e['image_path']).exists()]
random.seed(42)
sample_paths = random.sample(valid_paths, min(N_SAMPLES, len(valid_paths)))

n_cols = 4
n_rows = (len(sample_paths) + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.5, n_rows * 2.5))
fig.suptitle(f'AROMA Synthesis Diversity ({TARGET_DS})', fontsize=11)
axes = np.array(axes).reshape(-1)

for i, p in enumerate(sample_paths):
    axes[i].imshow(load_img_rgb(p))
    axes[i].axis('off')
for ax in axes[len(sample_paths):]:
    ax.axis('off')

plt.tight_layout()
out_path = f"{FIGURE_OUT}/fig_qualitative_diversity.png"
plt.savefig(out_path, dpi=150, bbox_inches='tight')
plt.show()
print(f"저장: {out_path}")
```

---

## 셀 6 — Drive → 로컬 다운로드 (논문 제출용)

```python
# 생성된 그림을 /content/로 복사 후 Colab 파일 다운로드
import shutil

for fname in ['fig_qualitative_comparison.png', 'fig_qualitative_pipeline.png', 'fig_qualitative_diversity.png']:
    src = Path(FIGURE_OUT) / fname
    dst = Path('/content') / fname
    if src.exists():
        shutil.copy2(src, dst)
        print(f"다운로드 준비: {dst}")

from google.colab import files
for fname in ['fig_qualitative_comparison.png', 'fig_qualitative_pipeline.png', 'fig_qualitative_diversity.png']:
    p = Path('/content') / fname
    if p.exists():
        files.download(str(p))
```

---

## 주의사항

### dataset_config.json의 경로 확인
각 데이터셋의 `image_dir`, `test_dir` 키가 Drive 마운트 기준 절대경로로 설정되어 있어야 한다.

```python
# 경로 검증 셀
cfg = load_dataset_config()
for ds in TARGET_DATASETS:
    if ds not in cfg:
        print(f"WARNING: {ds} not in dataset_config.json")
        continue
    img_dir = cfg[ds].get('image_dir', 'N/A')
    exists  = Path(img_dir).exists()
    print(f"{ds}: image_dir={img_dir} [{'OK' if exists else 'MISSING'}]")
```

### AROMA synthetic annotations.json 경로
`$AROMA_OUT/synthetic/{ds}/annotations.json`의 `image_path` 필드가 Drive 절대경로로 기록되어 있어야 한다.  
`generate_defects.py --local_staging` 옵션 사용 시 자동으로 Drive 경로로 기록된다.

### exp4v2 결과와 연계
`exp4v2_results.json`에서 `aroma n_train=0`인 데이터셋은 AROMA 합성 이미지가 없거나 `annotations.json` 경로가 잘못된 것이다.  
Figure A에서 AROMA 열이 검은색으로 나오면 해당 데이터셋의 Step 4를 재실행한다.
