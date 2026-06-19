# -*- coding: utf-8 -*-
"""AROMA논문 핵심 지표 Figure 생성 — MCI / CCI / DEFICIT

Figure A: MCI + CCI Computation Flow Diagram (static)
Figure B: Cross-Dataset MCI x CCI Scatter (24 datasets)
Figure C: MCI Component Breakdown — stacked bar (all 24 DS sorted by MCI)

Data source: .claude/.etc/complexity/<ds>/complexity_report.json (local)
Figure D (DEFICIT heatmap): Colab-only, see colab_execute/mci_cci_deficit_figures.md

Output: AROMA연구분析/Article/figure/
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── paths ──────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / '.claude' / '.etc' / 'complexity'

try:
    from fig_common import setup_font, save, PROJECT_ROOT as PR
    OUTPUT_DIR = None  # save() handles it
except ImportError:
    def setup_font():
        pass
    OUTPUT_DIR = PROJECT_ROOT / 'AROMA연구분析' / 'Article' / 'figure'
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    def save(fig, fname):
        out = OUTPUT_DIR / fname
        fig.savefig(out, dpi=150, bbox_inches='tight')
        print(f"저장: {out}")
        plt.close(fig)

# ── colors ──────────────────────────────────────────────────────
C_MVTEC    = '#2196F3'
C_VISA     = '#4CAF50'
C_ISP      = '#F44336'
C_ENTROPY  = '#5C6BC0'
C_VALLEY   = '#26A69A'
C_DIVERSITY= '#FFA726'
C_SILHOUETTE='#EF5350'


def ds_color(name):
    if name.startswith('mvtec'): return C_MVTEC
    if name.startswith('visa'):  return C_VISA
    return C_ISP


def load_all():
    data = {}
    for rpt in sorted(DATA_ROOT.glob('*/complexity_report.json')):
        ds = rpt.parent.name
        with open(rpt, encoding='utf-8') as f:
            d = json.load(f)
        mc = d.get('mci_components', {}).get('normalized', {})
        cc = d.get('cci_components', {}).get('normalized', {})
        data[ds] = {
            'mci': d['mci'], 'cci': d['cci'],
            'e': mc.get('entropy', 0),
            'v': mc.get('valley_count', 0),
            'd': mc.get('class_diversity', 0),
            's': mc.get('inv_silhouette', 0),
            'te': cc.get('texture_entropy', 0),
            'cc': cc.get('cluster_count_ctx', 0),
            'fc': cc.get('freq_complexity', 0),
            'ov': cc.get('orient_variance', 0),
        }
    return data


# ── Figure A: Flow Diagram ─────────────────────────────────────
def fig_flow():
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    fig.patch.set_facecolor('#FAFAFA')

    def box(ax, x, y, w, h, text, fc='#E3F2FD', ec='#1565C0', fs=9, bold=False):
        rect = mpatches.FancyBboxPatch(
            (x - w/2, y - h/2), w, h,
            boxstyle='round,pad=0.05', facecolor=fc, edgecolor=ec, linewidth=1.5
        )
        ax.add_patch(rect)
        ax.text(x, y, text, ha='center', va='center', fontsize=fs,
                fontweight='bold' if bold else 'normal')

    def arrow(ax, x1, y1, x2, y2):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color='#546E7A', lw=1.5))

    # MCI panel
    ax = axes[0]
    ax.set_xlim(0, 10); ax.set_ylim(0, 11); ax.axis('off')
    ax.set_title('MCI — Morphological Complexity Index', fontsize=11, fontweight='bold', pad=10)

    box(ax, 2, 9.5, 3.5, 0.7,
        'Defect morphology features\n(linearity, solidity, aspect_ratio …)', '#FFF8E1', '#F57F17', 8)
    box(ax, 8, 9.5, 3.5, 0.7,
        'Defect class labels\n(defect_type column)', '#FFF8E1', '#F57F17', 8)

    comps = [
        (7.5, 'Shannon Entropy\nH = –Σ p_k log₂ p_k', C_ENTROPY),
        (6.2, 'Valley Count\nΣ n_valleys across 6 features', C_VALLEY),
        (4.9, 'Class Diversity\nN_eff = exp(H_class)', C_DIVERSITY),
        (3.6, '1 – Silhouette\n(morphology cluster quality)', C_SILHOUETTE),
    ]
    for cy, lbl, clr in comps:
        box(ax, 5, cy, 7, 0.9, lbl, clr + '33', clr, 8.5)
        arrow(ax, 5, cy - 0.45, 5, cy - 0.8)

    box(ax, 5, 2.5, 4, 0.7, 'Min-Max Normalization → [0, 1]', '#E8F5E9', '#2E7D32', 9)
    arrow(ax, 5, 3.15, 5, 2.85)
    box(ax, 5, 1.4, 4.5, 0.75,
        'MCI = 0.25 × Entropy\n+ 0.25 × Valley\n+ 0.25 × Diversity\n+ 0.25 × (1–Silhouette)',
        '#EDE7F6', '#4527A0', 8.5)
    arrow(ax, 5, 2.12, 5, 1.77)
    box(ax, 5, 0.45, 3.5, 0.6, 'MCI ∈ [0, 1]', '#F3E5F5', '#6A1B9A', 11, True)
    arrow(ax, 5, 1.03, 5, 0.75)

    for y_in, x_in in [(9.12, 2), (9.12, 2), (9.12, 2), (9.12, 8), (9.12, 2)]:
        pass
    arrow(ax, 2, 9.12, 5, 7.95)
    arrow(ax, 2, 9.12, 5, 6.65)
    arrow(ax, 2, 9.12, 5, 5.35)
    arrow(ax, 8, 9.12, 5, 5.35)
    arrow(ax, 2, 9.12, 5, 4.05)

    # CCI + DEFICIT panel
    ax = axes[1]
    ax.set_xlim(0, 10); ax.set_ylim(0, 11); ax.axis('off')
    ax.set_title('CCI — Context Complexity Index  +  DEFICIT', fontsize=11, fontweight='bold', pad=10)

    box(ax, 5, 9.5, 7, 0.7,
        'Context patches from normal images\n(texture_entropy, freq_energy, orient_consistency …)',
        '#FFF8E1', '#F57F17', 8)

    cci_comps = [
        (7.8, 'LBP Texture Entropy\nmean(H_texture)', C_ENTROPY),
        (6.5, 'Context Clusters (GMM+BIC)\nk = argmin BIC over [1…8]', C_VALLEY),
        (5.2, 'Frequency Complexity\nvar(freq_energy)', C_DIVERSITY),
        (3.9, 'Orientation Variance\nvar(orient_consistency)', C_SILHOUETTE),
    ]
    for cy, lbl, clr in cci_comps:
        box(ax, 5, cy, 7, 0.9, lbl, clr + '33', clr, 8.5)
        arrow(ax, 5, cy - 0.45, 5, cy - 0.75)

    box(ax, 5, 2.7, 4, 0.65, 'Min-Max Normalization → [0, 1]', '#E8F5E9', '#2E7D32', 9)
    box(ax, 5, 1.75, 5, 0.7,
        'CCI = 0.25 × (Texture + Cluster + Freq + Orient)_norm', '#EDE7F6', '#4527A0', 8.5)
    box(ax, 5, 0.85, 3.2, 0.6, 'CCI ∈ [0, 1]', '#F3E5F5', '#6A1B9A', 11, True)
    arrow(ax, 5, 3.45, 5, 3.03)
    arrow(ax, 5, 2.38, 5, 2.1)
    arrow(ax, 5, 1.4, 5, 1.15)

    deficit_box = mpatches.FancyBboxPatch(
        (6.8, 4.5), 3.0, 2.2,
        boxstyle='round,pad=0.1', facecolor='#FFF3E0', edgecolor='#E65100',
        linewidth=2, linestyle='--'
    )
    ax.add_patch(deficit_box)
    ax.text(8.3, 7.1, 'DEFICIT', ha='center', fontsize=10,
            fontweight='bold', color='#E65100')
    ax.text(8.3, 5.6,
            'P_normal(cell) – P_cluster(cell)\n→ Coverage gap per cluster\n→ Guides ROI synthesis priority',
            ha='center', va='center', fontsize=7.5, color='#BF360C')

    plt.tight_layout(pad=1.5)
    save(fig, 'fig_mci_cci_flow.png')


# ── Figure B: Scatter ──────────────────────────────────────────
def fig_scatter(data):
    names  = list(data.keys())
    mcis   = [data[d]['mci'] for d in names]
    ccis   = [data[d]['cci'] for d in names]
    colors = [ds_color(d) for d in names]

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.scatter(mcis, ccis, c=colors, s=80, edgecolors='white', linewidths=0.8, zorder=3)

    LABEL_SET = {'mvtec_cable', 'isp_LSM_1', 'visa_cashew', 'mvtec_pill',
                 'mvtec_toothbrush', 'isp_ASM', 'visa_pcb', 'isp_LSM_2'}
    for ds, mx, cy in zip(names, mcis, ccis):
        if ds in LABEL_SET:
            short = ds.replace('mvtec_', '').replace('visa_', '').replace('isp_', '')
            ax.annotate(short, (mx, cy), textcoords='offset points', xytext=(5, 4), fontsize=7.5)

    ax.axvline(0.5, color='#B0BEC5', lw=1, ls='--', alpha=0.7)
    ax.axhline(0.35, color='#B0BEC5', lw=1, ls='--', alpha=0.7)
    ax.text(0.52, 0.48, 'High Morph.\nComplexity', fontsize=7.5, color='#607D8B', va='top')
    ax.text(0.27, 0.48, 'Low Morph.\nComplexity', fontsize=7.5, color='#607D8B', va='top')
    ax.text(0.27, 0.36, 'Low Context\nComplexity', fontsize=7.5, color='#607D8B', va='bottom')

    ax.legend(handles=[
        mpatches.Patch(color=C_MVTEC, label='MVTec (14 DS)'),
        mpatches.Patch(color=C_VISA,  label='VisA (7 DS)'),
        mpatches.Patch(color=C_ISP,   label='ISP (3 DS)'),
    ], loc='lower right', fontsize=9, framealpha=0.9)

    ax.set_xlabel('MCI — Morphological Complexity Index', fontsize=11)
    ax.set_ylabel('CCI — Context Complexity Index', fontsize=11)
    ax.set_title('AROMA Complexity Landscape: 24 Anomaly Detection Datasets\n'
                 'MCI = defect shape diversity; CCI = background texture complexity',
                 fontsize=11)
    ax.set_xlim(0.2, 0.75)
    ax.set_ylim(0.18, 0.53)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    save(fig, 'fig_mci_cci_scatter.png')


# ── Figure C: MCI Component Breakdown ─────────────────────────
def fig_mci_components(data):
    ds_sorted = sorted(data.keys(), key=lambda d: data[d]['mci'], reverse=True)

    ent = np.array([data[d]['e'] for d in ds_sorted])
    val = np.array([data[d]['v'] for d in ds_sorted])
    div = np.array([data[d]['d'] for d in ds_sorted])
    sil = np.array([data[d]['s'] for d in ds_sorted])
    mci = np.array([data[d]['mci'] for d in ds_sorted])

    x = np.arange(len(ds_sorted))
    w = 0.55

    fig, (ax_bar, ax_mci) = plt.subplots(
        2, 1, figsize=(16, 8),
        gridspec_kw={'height_ratios': [3, 1]},
        sharex=True
    )

    ax_bar.bar(x, ent, w, label='Entropy (H)', color=C_ENTROPY, alpha=0.9)
    ax_bar.bar(x, val, w, bottom=ent, label='Valley Count', color=C_VALLEY, alpha=0.9)
    ax_bar.bar(x, div, w, bottom=ent+val, label='Class Diversity', color=C_DIVERSITY, alpha=0.9)
    ax_bar.bar(x, sil, w, bottom=ent+val+div, label='1-Silhouette', color=C_SILHOUETTE, alpha=0.9)
    ax_bar.set_ylabel('Normalized Component Value (each ∈ [0, 1])', fontsize=9)
    ax_bar.set_title('MCI Component Breakdown — 24 Datasets (sorted by MCI ↓)', fontsize=11)
    ax_bar.set_ylim(0, 4.2)
    ax_bar.axhline(2.0, color='#B0BEC5', lw=0.8, ls='--', alpha=0.6)
    ax_bar.legend(loc='upper right', fontsize=9, ncol=2, framealpha=0.9)
    ax_bar.grid(axis='y', alpha=0.2); ax_bar.set_axisbelow(True)

    HIGHLIGHT = {'mvtec_cable', 'isp_LSM_1', 'visa_cashew'}
    for i, ds in enumerate(ds_sorted):
        if ds in HIGHLIGHT:
            short = ds.replace('mvtec_', '').replace('visa_', '').replace('isp_', '')
            ax_bar.annotate(f'★ {short}', xy=(i, 0.1), ha='center', fontsize=7,
                            color='#D32F2F', fontweight='bold')

    ax_mci.bar(x, mci, w * 0.8, color=[ds_color(d) for d in ds_sorted], alpha=0.85)
    ax_mci.set_ylabel('MCI', fontsize=9)
    ax_mci.set_ylim(0, 0.85)
    ax_mci.grid(axis='y', alpha=0.2); ax_mci.set_axisbelow(True)
    ax_mci.legend(handles=[
        mpatches.Patch(color=C_MVTEC, label='MVTec'),
        mpatches.Patch(color=C_VISA,  label='VisA'),
        mpatches.Patch(color=C_ISP,   label='ISP'),
    ], loc='upper right', fontsize=8)

    short_labels = [d.replace('mvtec_', 'm.').replace('visa_', 'v.').replace('isp_', 'i.')
                    for d in ds_sorted]
    ax_mci.set_xticks(x)
    ax_mci.set_xticklabels(short_labels, rotation=45, ha='right', fontsize=8)

    plt.tight_layout(h_pad=0.3)
    save(fig, 'fig_mci_components.png')


if __name__ == '__main__':
    setup_font()
    data = load_all()
    print(f"로드: {len(data)}개 데이터셋")

    print("Figure A: flow diagram...")
    fig_flow()

    print("Figure B: MCI x CCI scatter...")
    fig_scatter(data)

    print("Figure C: MCI components...")
    fig_mci_components(data)

    print("완료. Figure D (DEFICIT heatmap)는 Colab에서 실행하세요.")
