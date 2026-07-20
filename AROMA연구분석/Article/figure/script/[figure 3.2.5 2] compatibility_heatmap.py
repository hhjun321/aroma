# -*- coding: utf-8 -*-
"""Figure 16 — Compatibility heatmap (ctx_prior) for all 5 datasets.
morphology cluster x context cell, top-20 cells, row-normalized, peak boxed.
See [figure 3.2.5 2] compatibility_heatmap.md.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import json, io

PROF = "D:/aroma_dataset/profiling"
IMG  = "D:/project/aroma/AROMA연구분석/Article/figure/image"
DATASETS = ["aitex", "kolektor", "severstal", "mtd", "mvtec_leather"]
TOPN = 20

def build(ds):
    cm = json.load(io.open(f"{PROF}/{ds}/compatibility_matrix.json", encoding="utf-8"))["matrix_symmetric"]
    mc = json.load(io.open(f"{PROF}/{ds}/morphology_clusters.json", encoding="utf-8"))
    labels = {str(c["cluster_id"]): c.get("label", "") for c in mc["clusters"]}
    clusters = sorted(cm.keys(), key=lambda k: int(k))
    cells = set()
    for row in cm.values():
        cells |= set(row.keys())
    def maxc(c): return max(cm[k].get(c, 0.0) for k in clusters)
    top = sorted(cells, key=maxc, reverse=True)[:TOPN]
    top = sorted(top, key=lambda c: -np.mean([cm[k].get(c, 0.0) for k in clusters]))
    M = np.array([[cm[k].get(c, 0.0) for c in top] for k in clusters])
    ylabels = [f"k{k} · {labels.get(k,'')}" for k in clusters]
    return M, ylabels

for ds in DATASETS:
    M, ylabels = build(ds)
    nrow, ncol = M.shape
    fig, ax = plt.subplots(figsize=(10, 0.55 * nrow + 1.7))
    im = ax.imshow(M, aspect="auto", cmap="Blues", vmin=0.0, vmax=1.0)
    for r in range(nrow):
        c = int(np.argmax(M[r]))
        ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False,
                     edgecolor="crimson", linewidth=1.8))
    ax.set_xticks(range(ncol))
    ax.set_xticklabels([f"c{i+1}" for i in range(ncol)], fontsize=7)
    ax.set_yticks(range(nrow))
    ax.set_yticklabels(ylabels, fontsize=9)
    ax.set_xlabel(f"context cell (top {ncol} by compatibility)", fontsize=9)
    ax.set_title(f"Compatibility (ctx_prior) — {ds}", fontsize=13)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cb.set_label("ctx_prior (compat_sym)", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    plt.tight_layout()
    out = f"{IMG}/[figure 3.2.5 2 {ds}] compatibility_heatmap.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("saved:", out, "shape", M.shape)
