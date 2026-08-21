# -*- coding: utf-8 -*-
"""Figure 3.2.4-1 — ROI selection & compatibility-aware placement flow.

Three sequential decisions of §3.2.4 (2026-08-14 simplified body):
  (1) defect crop selection (ROI_score top-K)
  (2) background assignment (three fitness measures)
  (3) site resolution (ring histogram vs target profile).
Stage-1 numbers are real Severstal linear-scratch profiling values
(matrix_symmetric["1"] peak cell 0_0_0_1_0 = 1.00, P(k=1) = 0.24);
stages 2-3 are symbolic. See [figure 3.2.4 1] roi_selection_flow.md.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = ("D:/project/aroma/AROMA연구분석/Article/figure/image/"
       "[figure 3.2.4 1] roi_selection_flow.png")

BLUE   = "#2c6fbb"   # compatibility signal (core)
BLUEBG = "#dceaf7"
GREY   = "#6b6b6b"   # morphology prior
GREYBG = "#ececec"
STAR   = "#b8860b"
STARBG = "#fdf1cf"

fig, ax = plt.subplots(figsize=(6.6, 9.6))
ax.set_xlim(0, 10); ax.set_ylim(3.4, 27.4); ax.axis("off")

def box(x, y, w, h, text, ec, fc, lw=1.6, fs=9.5, style="round,pad=0.1", ls="-", tc="black"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=style,
                 linewidth=lw, edgecolor=ec, facecolor=fc, linestyle=ls))
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fs, color=tc, wrap=True)

def arrow(x0, y0, x1, y1, color="black", lw=1.6):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                 mutation_scale=15, linewidth=lw, color=color))

def stage(y, label):
    ax.text(0.02, y, label, fontsize=8.2, color="#444444", style="italic")

cx = 5.0

# --- Stage 1: defect crop selection -----------------------------------------
stage(26.9, "1. defect crop selection")
box(2.6, 25.2, 4.8, 1.6,
    "Defect crop\nlinearity 0.961 · solidity 0.882 · AR 5.09",
    "black", "white", fs=9.5)

box(0.4, 22.4, 4.2, 1.7,
    "GMM morphology cluster  k = 1\nmorph_prior  P(k) = 0.24",
    GREY, GREYBG, fs=9.0)
box(5.4, 22.4, 4.2, 1.7,
    "source context cell 0_0_0_1_0\nctx_prior = matrix_symmetric = 1.00",
    BLUE, BLUEBG, fs=8.8, tc=BLUE)
arrow(4.0, 25.2, 2.5, 24.1)                     # defect -> cluster
arrow(6.0, 25.2, 7.5, 24.1, color=BLUE)         # defect -> context cell

box(2.4, 19.6, 5.2, 1.7,
    "ROI_score = ctx_prior + morph_prior\n= 1.00 + 0.24 = 1.24",
    BLUE, "white", fs=9.2)
arrow(2.5, 22.4, 4.3, 21.3, color=GREY)
arrow(7.5, 22.4, 5.7, 21.3, color=BLUE)

box(2.4, 17.0, 5.2, 1.5,
    "Rank all candidates → keep Top-K sources",
    "black", "white", fs=9.2)
arrow(cx, 19.6, cx, 18.5)

# --- Stage 2: background assignment ------------------------------------------
stage(16.4, "2. background assignment")
box(2.0, 13.9, 6.0, 1.9,
    "bg_score = src_fit + class_fit + size_fit\n(histogram ∩ with source / class context;\nfit-rescale factor)",
    BLUE, BLUEBG, fs=8.8, tc=BLUE)
arrow(cx, 17.0, cx, 15.8)

box(2.4, 11.4, 5.2, 1.5,
    "Assign highest-scoring normal image",
    "black", "white", fs=9.2)
arrow(cx, 13.9, cx, 12.9)

# --- Stage 3: site resolution -------------------------------------------------
stage(10.8, "3. site resolution")
box(2.0, 8.3, 6.0, 1.9,
    "For each valid position s:\nring histogram h_s vs target tgt[k]\nsite_score = ∩( h_s, tgt[k] )",
    BLUE, BLUEBG, fs=8.8, tc=BLUE)
arrow(cx, 11.4, cx, 10.2)

box(2.6, 5.8, 4.8, 1.6,
    "★ argmax site_score →\nfinal paste position (bbox) ★",
    STAR, STARBG, lw=2.0, fs=9.6, tc="black")
arrow(cx, 8.3, cx, 7.4, color=STAR)

# legend
ax.text(0.1, 4.6, "blue = compatibility signal read from matrix_symmetric (core placement signal)",
        fontsize=7.8, color=BLUE)
ax.text(0.1, 4.1, "grey = morphology-cluster prevalence prior (population regularizer)",
        fontsize=7.8, color=GREY)
ax.text(0.1, 3.6, "stage-1 values: real Severstal linear-scratch example; stages 2–3 symbolic",
        fontsize=7.8, color="#444444")

plt.tight_layout()
fig.savefig(OUT, dpi=300, bbox_inches="tight")
print("saved:", OUT)
