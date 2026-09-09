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
import json, io, os

PROF = os.environ.get("AROMA_DATASET_ROOT", "D:/project/aroma_dataset") + "/profiling/profiling"
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
    fig, ax = plt.subplots(figsize=(11, 0.85 * nrow + 2.2))
    im = ax.imshow(M, aspect="auto", cmap="Blues", vmin=0.0, vmax=1.0)
    for r in range(nrow):
        c = int(np.argmax(M[r]))
        ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False,
                     edgecolor="crimson", linewidth=2.2))
    ax.set_xticks(range(ncol))
    ax.set_xticklabels([f"c{i+1}" for i in range(ncol)], fontsize=12)
    ax.set_yticks(range(nrow))
    ax.set_yticklabels(ylabels, fontsize=13)
    ax.set_xlabel(f"context cell (top {ncol} by compatibility)", fontsize=14)
    ax.set_title(f"Compatibility (ctx_prior) — {ds}", fontsize=16)
    cb = fig.colorbar(im, ax=ax, fraction=0.030, pad=0.012)
    cb.set_label("ctx_prior (matrix_symmetric)", fontsize=13)
    # explicit decimal-dot tick labels (no locale-dependent comma separator)
    cb.set_ticks(np.linspace(0.0, 1.0, 6))
    cb.set_ticklabels([f"{v:.1f}" for v in np.linspace(0.0, 1.0, 6)])
    cb.ax.tick_params(labelsize=12)
    plt.tight_layout()
    out = f"{IMG}/[figure 3.2.4 2 {ds}] compatibility_heatmap.png"
    fig.savefig(out, dpi=400, bbox_inches="tight")
    plt.close(fig)
    print("saved:", out, "shape", M.shape)
