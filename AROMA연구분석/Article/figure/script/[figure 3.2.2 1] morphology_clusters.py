# -*- coding: utf-8 -*-
"""Figure 15 — Morphology clusters (defines k), simplified single-panel.
severstal, aitex. Scatter (linearity x log aspect_ratio, colored by cluster k,
centroids); legend gives k, label, and P(k)=morph_prior. See spec .md.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import csv, io, json

PROF = "D:/aroma_dataset/profiling"
OUT  = "D:/project/aroma/AROMA연구분석/Article/figure/image/[figure 3.2.2 1] morphology_clusters.png"
PALETTE = ["#4c78a8", "#f58518", "#54a24b", "#e45756", "#72b7b2", "#b279a2"]

def load(ds):
    rows = list(csv.DictReader(io.open(f"{PROF}/{ds}/morphology_features.csv", encoding="utf-8")))
    mc = json.load(io.open(f"{PROF}/{ds}/morphology_clusters.json", encoding="utf-8"))
    ca = mc["cluster_assignments"]
    clusters = sorted(mc["clusters"], key=lambda c: c["cluster_id"])
    N = sum(c["n_samples"] for c in clusters)
    lin, lar, kk = [], [], []
    for r in rows:
        iid = r.get("image_id")
        if iid not in ca:
            continue
        try:
            lv = float(r["linearity"]); ar = float(r["aspect_ratio"])
        except (KeyError, ValueError):
            continue
        if ar <= 0:
            continue
        lin.append(lv); lar.append(np.log10(ar)); kk.append(int(ca[iid]))
    return np.array(lin), np.array(lar), np.array(kk), clusters, N

fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
fig.suptitle("Data-driven morphology clusters (k) — grouped along the elongation spectrum",
             fontsize=13, y=1.00)

for ax, ds in zip(axes, ["severstal", "aitex"]):
    lin, lar, kk, clusters, N = load(ds)
    for c in clusters:
        k = c["cluster_id"]; col = PALETTE[k % len(PALETTE)]
        m = kk == k
        pk = c["n_samples"] / N
        ax.scatter(lar[m], lin[m], s=8, alpha=0.30, color=col, edgecolors="none",
                   label=f"k{k} · {c.get('label','')} (P={pk:.2f})")
        cen = c["centroid"]
        ax.scatter(np.log10(max(cen["aspect_ratio"], 1e-6)), cen["linearity"],
                   marker="*", s=260, color=col, edgecolors="black", linewidths=1.1, zorder=5)
    ax.set_xlabel("log10(aspect ratio)", fontsize=10)
    ax.set_ylabel("linearity", fontsize=10)
    ax.set_title(ds, fontsize=12)
    ax.tick_params(labelsize=8)
    leg = ax.legend(loc="lower right", fontsize=8, framealpha=0.95, markerscale=2)
    for lh in leg.legend_handles:
        lh.set_alpha(1.0)

plt.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(OUT, dpi=300, bbox_inches="tight")
print("saved:", OUT)
