# MCI / CCI / DEFICIT 핵심 지표 Figure 생성 — Colab 실행 가이드

**목적**: AROMA 논문 핵심 지표 시각화  
**데이터**: 24개 데이터셋 complexity_report.json (Drive) + deficit_analysis.json (Drive)  
**출력**: `$DRIVE/figures/complexity/` 하위 PNG 파일 4종  
**런타임**: CPU 가능 (GPU 불필요)

---

## Figure 목록

| ID | 제목 | 데이터 소스 | 파일명 |
|----|------|------------|--------|
| A | MCI/CCI Computation Flow Diagram | 없음 (정적) | `fig_mci_cci_flow.png` |
| B | Cross-Dataset MCI × CCI Scatter (24 DS) | complexity_report.json | `fig_mci_cci_scatter.png` |
| C | MCI Component Breakdown (상위/하위 DS) | complexity_report.json | `fig_mci_components.png` |
| D | DEFICIT Coverage Heatmap (단일 DS) | deficit_analysis.json | `fig_deficit_heatmap.png` |

---

## 셀 0 — 환경변수 설정

```python
import os
from pathlib import Path

DRIVE = os.environ.get('DRIVE', '/content/drive/MyDrive/data/Aroma')
AROMA_OUT = f"{DRIVE}/aroma_output"
FIGURE_OUT = f"{DRIVE}/figures/complexity"
os.makedirs(FIGURE_OUT, exist_ok=True)
print(f"FIGURE_OUT: {FIGURE_OUT}")
```

---

## 셀 1 — 공통 설정 및 데이터 임베딩

24개 데이터셋 complexity_report 값을 직접 임베딩. Drive에서도 로드 가능 (셀 2 참고).

```python
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch

# ── 공통 색상 팔레트 ──────────────────────────────────────────
C_MVTEC = '#2196F3'   # blue
C_VISA  = '#4CAF50'   # green
C_ISP   = '#F44336'   # red
C_ENTROPY  = '#5C6BC0'
C_VALLEY   = '#26A69A'
C_DIVERSITY= '#FFA726'
C_SILHOUETTE= '#EF5350'

# ── 임베딩 데이터 (complexity_report.json 집계) ────────────────
COMPLEXITY_DATA = {
    "mvtec_pill":        {"mci":0.6779,"cci":0.461, "e":0.4055,"v":0.3902,"d":0.9158,"s":1.0,   "te":0.2294,"cc":0.5714,"fc":0.0432,"ov":1.0   },
    "mvtec_carpet":      {"mci":0.6705,"cci":0.2334,"e":0.3724,"v":0.5366,"d":0.7733,"s":1.0,   "te":0.3613,"cc":0.5714,"fc":0.0005,"ov":0.0003},
    "mvtec_tile":        {"mci":0.6703,"cci":0.2293,"e":0.3232,"v":0.5854,"d":0.7728,"s":1.0,   "te":0.3452,"cc":0.5714,"fc":0.0002,"ov":0.0002},
    "mvtec_cable":       {"mci":0.6636,"cci":0.2438,"e":0.2679,"v":0.3902,"d":0.9964,"s":1.0,   "te":0.3531,"cc":0.5714,"fc":0.0013,"ov":0.0491},
    "mvtec_leather":     {"mci":0.6451,"cci":0.2361,"e":0.2216,"v":0.5854,"d":0.7735,"s":1.0,   "te":0.3726,"cc":0.5714,"fc":0.0002,"ov":0.0003},
    "mvtec_bottle":      {"mci":0.6139,"cci":0.4652,"e":0.464, "v":0.4634,"d":0.528, "s":1.0,   "te":0.2775,"cc":0.5714,"fc":0.0119,"ov":1.0   },
    "mvtec_zipper":      {"mci":0.6071,"cci":0.472, "e":0.3717,"v":0.122, "d":0.9348,"s":1.0,   "te":0.3092,"cc":0.5714,"fc":0.0076,"ov":1.0   },
    "mvtec_grid":        {"mci":0.6046,"cci":0.2267,"e":0.4011,"v":0.2439,"d":0.7735,"s":1.0,   "te":0.3294,"cc":0.5714,"fc":0.0006,"ov":0.0052},
    "mvtec_metal_nut":   {"mci":0.5994,"cci":0.429, "e":0.3412,"v":0.3902,"d":0.6661,"s":1.0,   "te":0.2705,"cc":0.5714,"fc":0.0186,"ov":0.8553},
    "mvtec_wood":        {"mci":0.5887,"cci":0.2425,"e":0.295, "v":0.3171,"d":0.7427,"s":1.0,   "te":0.3938,"cc":0.5714,"fc":0.0005,"ov":0.0044},
    "mvtec_screw":       {"mci":0.585, "cci":0.2444,"e":0.4688,"v":0.0976,"d":0.7737,"s":1.0,   "te":0.4008,"cc":0.5714,"fc":0.0005,"ov":0.0047},
    "mvtec_hazelnut":    {"mci":0.562, "cci":0.3155,"e":0.289, "v":0.2927,"d":0.6665,"s":1.0,   "te":0.2642,"cc":0.5714,"fc":0.0132,"ov":0.4134},
    "isp_LSM_2":         {"mci":0.5248,"cci":0.4507,"e":0.5572,"v":0.7073,"d":0.3095,"s":0.5254,"te":0.2178,"cc":0.5714,"fc":0.0137,"ov":1.0   },
    "mvtec_transistor":  {"mci":0.4776,"cci":0.3172,"e":0.0,   "v":0.2439,"d":0.6667,"s":1.0,   "te":0.334, "cc":0.5714,"fc":0.0104,"ov":0.3531},
    "isp_LSM_1":         {"mci":0.4432,"cci":0.4304,"e":0.5506,"v":0.3171,"d":0.3331,"s":0.5721,"te":0.3004,"cc":0.5714,"fc":0.0145,"ov":0.8353},
    "visa_pipe_fryum":   {"mci":0.432, "cci":0.255, "e":0.5345,"v":0.6098,"d":0.0,   "s":0.5836,"te":0.3591,"cc":0.5714,"fc":0.0053,"ov":0.0842},
    "visa_candle":       {"mci":0.4277,"cci":0.247, "e":0.4996,"v":0.3902,"d":0.0,   "s":0.821, "te":0.388, "cc":0.5714,"fc":0.0108,"ov":0.0176},
    "visa_chewinggum":   {"mci":0.4272,"cci":0.2813,"e":0.5469,"v":0.4878,"d":0.0,   "s":0.6742,"te":0.3356,"cc":0.5714,"fc":0.0048,"ov":0.2134},
    "isp_ASM":           {"mci":0.4074,"cci":0.4584,"e":0.3905,"v":0.2439,"d":0.3305,"s":0.6649,"te":0.2491,"cc":0.5714,"fc":0.0132,"ov":1.0   },
    "visa_macaroni":     {"mci":0.4047,"cci":0.2392,"e":0.5204,"v":0.4634,"d":0.0,   "s":0.635, "te":0.3728,"cc":0.5714,"fc":0.0032,"ov":0.0094},
    "visa_fryum":        {"mci":0.4024,"cci":0.2389,"e":0.4881,"v":0.3171,"d":0.0,   "s":0.8046,"te":0.3829,"cc":0.5714,"fc":0.0007,"ov":0.0006},
    "visa_cashew":       {"mci":0.3834,"cci":0.2483,"e":0.4808,"v":0.3659,"d":0.0,   "s":0.6871,"te":0.4066,"cc":0.5714,"fc":0.0082,"ov":0.007 },
    "visa_pcb":          {"mci":0.354, "cci":0.2491,"e":0.5298,"v":0.3171,"d":0.0,   "s":0.5689,"te":0.4025,"cc":0.5714,"fc":0.0195,"ov":0.0028},
    "mvtec_toothbrush":  {"mci":0.2927,"cci":0.4509,"e":0.5357,"v":0.0732,"d":0.0,   "s":0.5618,"te":0.1887,"cc":0.5714,"fc":0.0434,"ov":1.0   },
}

def ds_color(name):
    if name.startswith('mvtec'): return C_MVTEC
    if name.startswith('visa'):  return C_VISA
    return C_ISP

print(f"로드된 데이터셋: {len(COMPLEXITY_DATA)}개")
```

---

## 셀 2 — (선택) Drive에서 실시간 로드

```python
# Drive에서 최신 complexity_report.json 로드 (셀 1 데이터 덮어씀)
import json
from pathlib import Path

LOADED = {}
for ds_dir in sorted(Path(f"{AROMA_OUT}/complexity").iterdir()):
    rpt = ds_dir / 'complexity_report.json'
    if not rpt.exists(): continue
    with open(rpt) as f:
        d = json.load(f)
    mc = d.get('mci_components', {}).get('normalized', {})
    cc = d.get('cci_components', {}).get('normalized', {})
    LOADED[ds_dir.name] = {
        'mci': d['mci'], 'cci': d['cci'],
        'e':  mc.get('entropy', 0),
        'v':  mc.get('valley_count', 0),
        'd':  mc.get('class_diversity', 0),
        's':  mc.get('inv_silhouette', 0),
        'te': cc.get('texture_entropy', 0),
        'cc': cc.get('cluster_count_ctx', 0),
        'fc': cc.get('freq_complexity', 0),
        'ov': cc.get('orient_variance', 0),
    }
if LOADED:
    COMPLEXITY_DATA = LOADED
    print(f"Drive에서 {len(COMPLEXITY_DATA)}개 로드 완료")
else:
    print("Drive 데이터 없음 — 셀 1 임베딩 데이터 사용")
```

---

## 셀 3 — Figure A: Computation Flow Diagram

```python
fig, axes = plt.subplots(1, 2, figsize=(14, 7))
fig.patch.set_facecolor('#FAFAFA')

def box(ax, x, y, w, h, text, fc='#E3F2FD', ec='#1565C0', fs=9, bold=False):
    rect = mpatches.FancyBboxPatch(
        (x - w/2, y - h/2), w, h,
        boxstyle="round,pad=0.05", facecolor=fc, edgecolor=ec, linewidth=1.5
    )
    ax.add_patch(rect)
    ax.text(x, y, text, ha='center', va='center', fontsize=fs,
            fontweight='bold' if bold else 'normal', wrap=True)

def arrow(ax, x1, y1, x2, y2):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color='#546E7A', lw=1.5))

# ─── Panel 1: MCI Flow ────────────────────────────────────────
ax = axes[0]
ax.set_xlim(0, 10); ax.set_ylim(0, 11)
ax.axis('off')
ax.set_title('MCI — Morphological Complexity Index', fontsize=11, fontweight='bold', pad=10)

# Input features
box(ax, 2, 9.5, 3.5, 0.7, 'Defect morphology features\n(linearity, solidity, aspect_ratio …)', '#FFF8E1', '#F57F17', 8)
box(ax, 8, 9.5, 3.5, 0.7, 'Defect class labels\n(defect_type column)', '#FFF8E1', '#F57F17', 8)

# Components
comp_y = [7.5, 6.2, 4.9, 3.6]
comp_labels = [
    ('Shannon Entropy\nH = –Σ p_k log₂ p_k', C_ENTROPY),
    ('Valley Count\nΣ n_valleys across 6 features', C_VALLEY),
    ('Class Diversity\nN_eff = exp(H_class)', C_DIVERSITY),
    ('1 – Silhouette\n(morphology cluster quality)', C_SILHOUETTE),
]
for cy, (lbl, clr) in zip(comp_y, comp_labels):
    box(ax, 5, cy, 7, 0.9, lbl, clr + '33', clr, 8.5)
    arrow(ax, 5, cy - 0.45, 5, cy - 0.85)

# Normalization
box(ax, 5, 2.5, 4, 0.7, 'Min-Max Normalization → [0, 1]', '#E8F5E9', '#2E7D32', 9)
arrow(ax, 5, 3.15, 5, 2.85)

# Weighted mean
box(ax, 5, 1.4, 4.5, 0.75,
    'MCI = 0.25 × Entropy_norm\n+ 0.25 × Valley_norm\n+ 0.25 × Diversity_norm\n+ 0.25 × (1–Silhouette)_norm',
    '#EDE7F6', '#4527A0', 8.5)
arrow(ax, 5, 2.12, 5, 1.77)

# Output
box(ax, 5, 0.45, 3.5, 0.6, 'MCI ∈ [0, 1]', '#F3E5F5', '#6A1B9A', 11, bold=True)
arrow(ax, 5, 1.03, 5, 0.75)

arrow(ax, 2, 9.12, 5, 7.95)
arrow(ax, 2, 9.12, 5, 6.65)
arrow(ax, 2, 9.12, 5, 5.35)
arrow(ax, 8, 9.12, 5, 5.35)
arrow(ax, 2, 9.12, 5, 4.05)

# ─── Panel 2: CCI + DEFICIT Flow ──────────────────────────────
ax = axes[1]
ax.set_xlim(0, 10); ax.set_ylim(0, 11)
ax.axis('off')
ax.set_title('CCI — Context Complexity Index  +  DEFICIT', fontsize=11, fontweight='bold', pad=10)

# Input
box(ax, 5, 9.5, 7, 0.7, 'Context patches from normal images\n(texture_entropy, freq_energy, orient_consistency …)', '#FFF8E1', '#F57F17', 8)

# CCI components
cci_comp = [
    ('LBP Texture Entropy\nmean(H_texture) per patch', '#5C6BC0'),
    ('Context Clusters (GMM+BIC)\nk = argmin BIC over [1…8]', '#26A69A'),
    ('Frequency Complexity\nvar(freq_energy)', '#FFA726'),
    ('Orientation Variance\nvar(orient_consistency)', '#EF5350'),
]
cci_y = [7.8, 6.5, 5.2, 3.9]
for cy, (lbl, clr) in zip(cci_y, cci_comp):
    box(ax, 5, cy, 7, 0.9, lbl, clr + '33', clr, 8.5)

box(ax, 5, 2.7, 4, 0.65, 'Min-Max Normalization → [0, 1]', '#E8F5E9', '#2E7D32', 9)
box(ax, 5, 1.75, 4.5, 0.7,
    'CCI = 0.25 × (Texture + Cluster + Freq + Orient)_norm',
    '#EDE7F6', '#4527A0', 8.5)
box(ax, 5, 0.85, 3.2, 0.6, 'CCI ∈ [0, 1]', '#F3E5F5', '#6A1B9A', 11, bold=True)

for cy in cci_y:
    arrow(ax, 5, cy - 0.45, 5, cy - 0.75)
arrow(ax, 5, 3.45, 5, 3.03)
arrow(ax, 5, 2.38, 5, 2.1)
arrow(ax, 5, 1.4, 5, 1.15)

# DEFICIT annotation in corner
deficit_text = (
    "DEFICIT\n"
    "P_normal(cell) – P_cluster(cell)\n"
    "→ Coverage gap per cluster\n"
    "→ Guides ROI synthesis priority"
)
deficit_box = mpatches.FancyBboxPatch((6.8, 4.5), 3.0, 2.2,
    boxstyle="round,pad=0.1", facecolor='#FFF3E0', edgecolor='#E65100', linewidth=2, linestyle='--')
ax.add_patch(deficit_box)
ax.text(8.3, 5.6, deficit_text, ha='center', va='center', fontsize=7.5, color='#BF360C')
ax.text(8.3, 7.1, 'DEFICIT', ha='center', va='center',
        fontsize=10, fontweight='bold', color='#E65100')

plt.tight_layout(pad=1.5)
out = f"{FIGURE_OUT}/fig_mci_cci_flow.png"
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.show()
print(f"저장: {out}")
```

---

## 셀 4 — Figure B: Cross-Dataset MCI × CCI Scatter

```python
ds_names = list(COMPLEXITY_DATA.keys())
mci_vals  = [COMPLEXITY_DATA[d]['mci'] for d in ds_names]
cci_vals  = [COMPLEXITY_DATA[d]['cci'] for d in ds_names]
colors    = [ds_color(d) for d in ds_names]

fig, ax = plt.subplots(figsize=(9, 7))

scatter = ax.scatter(mci_vals, cci_vals, c=colors, s=80, edgecolors='white', linewidths=0.8, zorder=3)

# Dataset labels (select non-overlapping)
LABEL_SET = {'mvtec_cable', 'isp_LSM_1', 'visa_cashew', 'mvtec_pill',
             'mvtec_toothbrush', 'isp_ASM', 'visa_pcb', 'isp_LSM_2'}
for ds, mx, cy in zip(ds_names, mci_vals, cci_vals):
    if ds in LABEL_SET:
        short = ds.replace('mvtec_', '').replace('visa_', '').replace('isp_', '')
        ax.annotate(short, (mx, cy), textcoords='offset points', xytext=(5, 4),
                    fontsize=7.5, color='#212121')

# Complexity region annotations
ax.axvline(0.5, color='#B0BEC5', lw=1, ls='--', alpha=0.7)
ax.axhline(0.35, color='#B0BEC5', lw=1, ls='--', alpha=0.7)
ax.text(0.52, 0.48, 'High Morph.\nComplexity', fontsize=7.5, color='#607D8B', va='top')
ax.text(0.27, 0.48, 'Low Morph.\nComplexity', fontsize=7.5, color='#607D8B', va='top')
ax.text(0.27, 0.36, 'Low Context\nComplexity', fontsize=7.5, color='#607D8B', va='bottom')

# Legend
legend_patches = [
    mpatches.Patch(color=C_MVTEC, label='MVTec (14 DS)'),
    mpatches.Patch(color=C_VISA,  label='VisA (7 DS)'),
    mpatches.Patch(color=C_ISP,   label='ISP (3 DS)'),
]
ax.legend(handles=legend_patches, loc='lower right', fontsize=9, framealpha=0.9)

ax.set_xlabel('MCI — Morphological Complexity Index', fontsize=11)
ax.set_ylabel('CCI — Context Complexity Index', fontsize=11)
ax.set_title('AROMA Complexity Landscape: 24 Anomaly Detection Datasets\n'
             'MCI measures defect shape diversity; CCI measures background texture complexity',
             fontsize=11)
ax.set_xlim(0.2, 0.75)
ax.set_ylim(0.18, 0.53)
ax.grid(True, alpha=0.2)

plt.tight_layout()
out = f"{FIGURE_OUT}/fig_mci_cci_scatter.png"
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.show()
print(f"저장: {out}")
print(f"MCI range: {min(mci_vals):.3f} ~ {max(mci_vals):.3f}")
print(f"CCI range: {min(cci_vals):.3f} ~ {max(cci_vals):.3f}")
```

---

## 셀 5 — Figure C: MCI Component Breakdown

```python
# 전체 24개 데이터셋을 MCI 내림차순으로 정렬, stacked bar
ds_sorted = sorted(COMPLEXITY_DATA.keys(), key=lambda d: COMPLEXITY_DATA[d]['mci'], reverse=True)

ent_vals  = [COMPLEXITY_DATA[d]['e'] for d in ds_sorted]
val_vals  = [COMPLEXITY_DATA[d]['v'] for d in ds_sorted]
div_vals  = [COMPLEXITY_DATA[d]['d'] for d in ds_sorted]
sil_vals  = [COMPLEXITY_DATA[d]['s'] for d in ds_sorted]
mci_line  = [COMPLEXITY_DATA[d]['mci'] for d in ds_sorted]

x = np.arange(len(ds_sorted))
bar_w = 0.55

fig, (ax_bar, ax_mci) = plt.subplots(2, 1, figsize=(16, 8),
                                      gridspec_kw={'height_ratios': [3, 1]},
                                      sharex=True)

# Stacked bar
b1 = ax_bar.bar(x, ent_vals, bar_w, label='Entropy (H)', color=C_ENTROPY, alpha=0.9)
b2 = ax_bar.bar(x, val_vals, bar_w, bottom=ent_vals, label='Valley Count', color=C_VALLEY, alpha=0.9)
b3 = ax_bar.bar(x, div_vals, bar_w,
                bottom=[e+v for e,v in zip(ent_vals, val_vals)],
                label='Class Diversity', color=C_DIVERSITY, alpha=0.9)
b4 = ax_bar.bar(x, sil_vals, bar_w,
                bottom=[e+v+d for e,v,d in zip(ent_vals, val_vals, div_vals)],
                label='1 − Silhouette', color=C_SILHOUETTE, alpha=0.9)

ax_bar.set_ylabel('Normalized Component Value\n(each ∈ [0, 1])', fontsize=9)
ax_bar.set_title('MCI Component Breakdown — 24 Datasets (sorted by MCI ↓)', fontsize=11)
ax_bar.set_ylim(0, 4.2)
ax_bar.axhline(2.0, color='#B0BEC5', lw=0.8, ls='--', alpha=0.6)
ax_bar.legend(loc='upper right', fontsize=9, ncol=2, framealpha=0.9)
ax_bar.grid(axis='y', alpha=0.2)
ax_bar.set_axisbelow(True)

# MCI line below
ax_mci.bar(x, mci_line, bar_w * 0.8,
           color=[ds_color(d) for d in ds_sorted], alpha=0.85)
ax_mci.set_ylabel('MCI', fontsize=9)
ax_mci.set_ylim(0, 0.85)
ax_mci.grid(axis='y', alpha=0.2)
ax_mci.set_axisbelow(True)

# x-axis labels
short_labels = [d.replace('mvtec_', 'm.').replace('visa_', 'v.')
                 .replace('isp_', 'i.') for d in ds_sorted]
ax_mci.set_xticks(x)
ax_mci.set_xticklabels(short_labels, rotation=45, ha='right', fontsize=8)

# Highlight our 3 key datasets
HIGHLIGHT = {'mvtec_cable', 'isp_LSM_1', 'visa_cashew'}
for i, ds in enumerate(ds_sorted):
    if ds in HIGHLIGHT:
        short = ds.replace('mvtec_', '').replace('visa_', '').replace('isp_', '')
        ax_bar.annotate(f'★ {short}', xy=(i, 0.08), ha='center', fontsize=7,
                        color='#D32F2F', fontweight='bold')

# Dataset type color legend
patch_m = mpatches.Patch(color=C_MVTEC, label='MVTec')
patch_v = mpatches.Patch(color=C_VISA,  label='VisA')
patch_i = mpatches.Patch(color=C_ISP,   label='ISP')
ax_mci.legend(handles=[patch_m, patch_v, patch_i], loc='upper right', fontsize=8)

plt.tight_layout(h_pad=0.3)
out = f"{FIGURE_OUT}/fig_mci_components.png"
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.show()
print(f"저장: {out}")
```

---

## 셀 6 — Figure D: DEFICIT Coverage Heatmap

```python
# deficit_analysis.json 로드 (Drive 필요)
TARGET_DS = 'mvtec_cable'   # ← 변경 가능

deficit_path = Path(AROMA_OUT) / 'complexity' / TARGET_DS / 'deficit_analysis.json'

if not deficit_path.exists():
    print(f"파일 없음: {deficit_path}")
    print("Step 0 (distribution_profiling.py) 완료 후 재실행하세요.")
else:
    with open(deficit_path) as f:
        deficit_data = json.load(f)

    n_clusters = len(deficit_data)
    # 모든 컨텍스트 셀 키 수집
    all_cells = sorted(set(
        k for cl in deficit_data.values()
        for k in cl.get('deficit', {}).keys()
    ))

    # 행렬 생성: clusters × cells
    cluster_ids = sorted(deficit_data.keys(), key=lambda x: int(x))
    mat = np.zeros((len(cluster_ids), len(all_cells)))
    for ci, cid in enumerate(cluster_ids):
        def_dict = deficit_data[cid].get('deficit', {})
        for j, cell in enumerate(all_cells):
            mat[ci, j] = def_dict.get(cell, 0.0)

    # 셀 순서: 총 deficit 내림차순으로 정렬 (중요 셀 왼쪽)
    col_order = np.argsort(-mat.sum(axis=0))
    mat = mat[:, col_order]
    sorted_cells = [all_cells[i] for i in col_order]

    # 상위 30개 셀만 표시 (가독성)
    MAX_CELLS = min(30, len(sorted_cells))
    mat_vis = mat[:, :MAX_CELLS]
    cell_labels = sorted_cells[:MAX_CELLS]

    fig, ax = plt.subplots(figsize=(max(10, MAX_CELLS * 0.38), max(4, n_clusters * 0.9)))
    im = ax.imshow(mat_vis, aspect='auto', cmap='YlOrRd', vmin=0)

    ax.set_xticks(range(MAX_CELLS))
    ax.set_xticklabels(cell_labels, rotation=60, ha='right', fontsize=7)
    ax.set_yticks(range(n_clusters))
    ax.set_yticklabels([f'Cluster {cid}\n(prior={deficit_data[cid].get("prior",0):.2f})'
                         for cid in cluster_ids], fontsize=9)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    cbar.set_label('DEFICIT\nmax(0, P_normal − P_cluster)', fontsize=8)

    # Cell-level annotation (small values)
    for ci in range(mat_vis.shape[0]):
        for cj in range(mat_vis.shape[1]):
            v = mat_vis[ci, cj]
            if v > 0.01:
                ax.text(cj, ci, f'{v:.2f}', ha='center', va='center',
                        fontsize=6.5, color='black' if v < 0.08 else 'white')

    ax.set_xlabel('Context Cell (local_var_bin × edge_density_bin × … [3³ grid])', fontsize=9)
    ax.set_title(
        f'DEFICIT Heatmap: {TARGET_DS}\n'
        f'Rows = morphology clusters, Cols = context cells, '
        f'Color = coverage gap (higher = more synthesis needed)',
        fontsize=10
    )

    plt.tight_layout()
    out = f"{FIGURE_OUT}/fig_deficit_heatmap.png"
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"저장: {out}")
    print(f"  Clusters: {n_clusters}, Context cells (total): {len(all_cells)}, 표시: {MAX_CELLS}")
    max_deficit_cell = sorted_cells[0]
    print(f"  최대 DEFICIT 셀: {max_deficit_cell} ({mat[:,0].max():.4f})")
```

---

## 셀 7 — 로컬 다운로드

```python
import shutil
FNAMES = [
    'fig_mci_cci_flow.png',
    'fig_mci_cci_scatter.png',
    'fig_mci_components.png',
    'fig_deficit_heatmap.png',
]
from google.colab import files
for fn in FNAMES:
    src = Path(FIGURE_OUT) / fn
    if src.exists():
        dst = Path('/content') / fn
        shutil.copy2(src, dst)
        files.download(str(dst))
        print(f"다운로드: {fn}")
    else:
        print(f"파일 없음: {fn}")
```

---

## 주의사항

### DEFICIT 데이터 경로
`$DRIVE/aroma_output/complexity/{ds}/deficit_analysis.json` 존재 여부 확인 필요.  
없으면 `distribution_profiling.py --mode deficit --dataset {ds}` 재실행.

### Figure B — MCI/CCI 좌표 범위
임베딩 데이터 기반 고정 범위 (`xlim 0.2~0.75`). Drive 로드 후 다른 데이터셋 추가 시 범위 조정 필요.

### 색상 일관성
논문 내 모든 Figure에서:
- MVTec = Blue `#2196F3`
- VisA = Green `#4CAF50`  
- ISP = Red `#F44336`
- AROMA = Red `#d62728` (기존 palette와 통일)
