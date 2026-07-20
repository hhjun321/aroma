# -*- coding: utf-8 -*-
"""Figure 14 — ROI_score composition: 0.6*ctx_prior + 0.4*morph_prior.
Visualizes the PROCESS of producing ROI_score (not placement results).
Panels: severstal, aitex. See [figure 3.2.5 1] roi_score_composition.md.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import json, io

PROF = "D:/aroma_dataset/profiling"
OUT  = "D:/project/aroma/AROMA연구분석/Article/figure/image/[figure 3.2.5 1] roi_score_composition.png"
TOPN = 12
BLUE = "#2c6fbb"   # 0.6 * ctx_prior
GREY = "#b0b0b0"   # 0.4 * morph_prior

def top_candidates(ds):
    cm = json.load(io.open(f"{PROF}/{ds}/compatibility_matrix.json", encoding="utf-8"))["matrix_symmetric"]
    da = json.load(io.open(f"{PROF}/{ds}/deficit_analysis.json", encoding="utf-8"))
    cand = []
    for k, row in cm.items():
        mp = da[k]["prior"]
        for c, ctx in row.items():
            cand.append((0.6 * ctx + 0.4 * mp, ctx, mp, k))
    cand.sort(reverse=True)
    return cand[:TOPN]

fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2), sharex=True)
fig.suptitle("ROI_score = 0.6 · ctx_prior + 0.4 · morph_prior", fontsize=14, y=1.00)

for ax, ds in zip(axes, ["severstal", "aitex"]):
    cand = top_candidates(ds)
    y = np.arange(len(cand))                # index 0 = best; invert_yaxis puts it on top
    ctx_term = [0.6 * c[1] for c in cand]
    mor_term = [0.4 * c[2] for c in cand]
    ax.barh(y, ctx_term, color=BLUE, label="0.6 · ctx_prior")
    ax.barh(y, mor_term, left=ctx_term, color=GREY, label="0.4 · morph_prior")
    # total score annotation
    for yi, c in zip(y, cand):
        ax.text(c[0] + 0.008, yi, f"{c[0]:.3f}", va="center", fontsize=7.5)
    ax.set_yticks(y)
    ax.set_yticklabels([f"k{c[3]}" for c in cand], fontsize=8)
    ax.set_xlim(0, 0.85)
    ax.set_xlabel("score contribution", fontsize=9)
    ax.set_title(ds, fontsize=12)
    ax.invert_yaxis()                       # best (index 0) at top

axes[0].legend(loc="lower right", fontsize=8, framealpha=0.95)
plt.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(OUT, dpi=300, bbox_inches="tight")
print("saved:", OUT)
