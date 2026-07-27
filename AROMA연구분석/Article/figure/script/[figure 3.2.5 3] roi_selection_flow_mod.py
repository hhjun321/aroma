# -*- coding: utf-8 -*-
"""Figure 6 (mod) — ROI selection & compatibility-aware placement flow (theoretical, symbol-only).
No dataset-specific measured values (contrast with the Severstal-traced original). See
[figure 3.2.5 3] roi_selection_flow_mod.md.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = ("D:/project/aroma/AROMA연구분석/Article/figure/image/"
       "[figure 3.2.5 3] roi_selection_flow_mod.png")

BLUE   = "#2c6fbb"   # ctx / compat (core)
BLUEBG = "#dceaf7"
GREY   = "#6b6b6b"   # morph prior
GREYBG = "#ececec"
STAR   = "#b8860b"
STARBG = "#fdf1cf"

fig, ax = plt.subplots(figsize=(6.6, 8.6))
ax.set_xlim(0, 10); ax.set_ylim(4.6, 26); ax.axis("off")

def box(x, y, w, h, text, ec, fc, lw=1.6, fs=9.5, style="round,pad=0.1", ls="-", tc="black"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=style,
                 linewidth=lw, edgecolor=ec, facecolor=fc, linestyle=ls))
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fs, color=tc, wrap=True)

def arrow(x0, y0, x1, y1, color="black", lw=1.6):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                 mutation_scale=15, linewidth=lw, color=color))

cx = 5.0
# 1. Defect crop
box(2.6, 24.0, 4.8, 1.6,
    "Defect crop\nmorphology features (linearity, solidity, AR)",
    "black", "white", fs=9.5)

# 2a. morphology cluster (morph_prior) — left
box(0.4, 21.2, 4.2, 1.7,
    "GMM morphology cluster  k\nmorph_prior  P(k)",
    GREY, GREYBG, fs=9.0)
# 2b. context cell (ctx_prior) — right
box(5.4, 21.2, 4.2, 1.7,
    "candidate background\ncontext cell c\nctx_prior(k, c) = matrix_symmetric(k, c)",
    BLUE, BLUEBG, fs=8.8, tc=BLUE)
arrow(4.0, 24.0, 2.5, 22.9)                     # defect -> cluster
arrow(6.0, 24.0, 7.5, 22.9, color=BLUE)         # defect -> bg cell

# 3. ROI_score
box(2.4, 18.4, 5.2, 1.7,
    "ROI_score(k, c) =\n0.6·ctx_prior(k, c) + 0.4·morph_prior(k)",
    BLUE, "white", fs=9.2)
arrow(2.5, 21.2, 4.3, 20.1, color=GREY)         # cluster -> score
arrow(7.5, 21.2, 5.7, 20.1, color=BLUE)         # bg cell -> score

# 4. Top-K
box(2.2, 15.6, 5.6, 1.7,
    "Rank all candidates → Top-K",
    "black", "white", fs=8.9)
arrow(cx, 18.4, cx, 17.3)

# 5. Clean-bg
box(2.4, 12.8, 5.2, 1.6,
    "Clean-background assignment",
    "black", "white", fs=9.0)
arrow(cx, 15.6, cx, 14.4)

# 6. Compat gate
box(2.0, 9.9, 6.0, 1.8,
    "Compatibility gate\n64px tiling · scan–rank–place\naccept if best_mean ≥ τ",
    BLUE, BLUEBG, fs=9.0, tc=BLUE)
arrow(cx, 12.8, cx, 11.7)

# 7. Final ROI
box(2.6, 7.4, 4.8, 1.6,
    "Final pixel-level ROI (bbox)",
    STAR, STARBG, lw=2.0, fs=10.0, tc="black")
arrow(cx, 9.9, cx, 9.0, color=STAR)

plt.tight_layout()
fig.savefig(OUT, dpi=300, bbox_inches="tight")
print("saved:", OUT)
