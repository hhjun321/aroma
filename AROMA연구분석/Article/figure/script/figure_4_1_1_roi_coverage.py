# -*- coding: utf-8 -*-
"""Figure 4.1-1 — ROI placement coverage (AROMA vs Random), computed from source JSONs.
See figure_4_1_1_roi_coverage.md."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import json, io, random

BASE = "D:/aroma_dataset"
OUT  = "D:/project/aroma/AROMA연구분석/Article/figure/image/[figure 4.1 1] roi_coverage.png"
DS   = ["aitex", "kolektor", "severstal", "mtd", "mvtec_leather"]
LBL  = ["AITeX", "Kolektor", "Severstal", "MTD", "Leather"]
BLUE, GREY = "#2c6fbb", "#b0b0b0"

def pairs(entries):
    return [(e.get("cluster_id"), e.get("cell_key")) for e in entries
            if e.get("cluster_id") is not None and e.get("cell_key")]

def coverage(ds):
    aroma = pairs(json.load(io.open(f"{BASE}/synth_aroma/{ds}/annotations.json", encoding="utf-8")))
    rand  = pairs(json.load(io.open(f"{BASE}/synth_random/{ds}/_random_roi/roi_selected.json", encoding="utf-8")))
    cand  = json.load(io.open(f"{BASE}/synth_random/{ds}/_random_roi/roi_candidates.json", encoding="utf-8"))
    cp = pairs(cand)
    avail_m = {k for k, _ in cp}; avail_c = {c for _, c in cp}
    rare = {(e.get("cluster_id"), e.get("cell_key")) for e in cand
            if float(e.get("deficit") or 0) > 0}
    n = min(len(aroma), len(rand))
    a = random.Random(42).sample(aroma, n) if len(aroma) > n else aroma
    r = random.Random(42).sample(rand, n) if len(rand) > n else rand
    def cov(sel):
        sm = {k for k, _ in sel}; sc = {c for _, c in sel}; sp = set(sel)
        return (len(sm & avail_m) / max(1, len(avail_m)),
                len(sc & avail_c) / max(1, len(avail_c)),
                (len(sp & rare) / len(rare)) if rare else None)
    return cov(a), cov(r)

A, R = [], []
for ds in DS:
    a, r = coverage(ds); A.append(a); R.append(r)

titles = ["Morphology coverage", "Context coverage", "Rare-pair coverage"]
fig, axes = plt.subplots(1, 3, figsize=(13, 4.0), sharey=True)
x = np.arange(len(DS)); w = 0.36
for m, ax in enumerate(axes):
    av = [A[i][m] for i in range(len(DS))]
    rv = [R[i][m] for i in range(len(DS))]
    ax.bar(x - w/2, [v if v is not None else 0 for v in av], w, color=BLUE, label="AROMA")
    ax.bar(x + w/2, [v if v is not None else 0 for v in rv], w, color=GREY, label="Random")
    for i, (va, vr) in enumerate(zip(av, rv)):
        if va is None and vr is None:
            ax.text(i, 0.05, "n/a", ha="center", fontsize=9, color="#666666")
        else:
            ax.text(i - w/2, va + 0.015, f"{va:.2f}", ha="center", fontsize=6.5)
            ax.text(i + w/2, vr + 0.015, f"{vr:.2f}", ha="center", fontsize=6.5)
    ax.set_xticks(x); ax.set_xticklabels(LBL, fontsize=8, rotation=15)
    ax.set_ylim(0, 1.12); ax.set_title(titles[m], fontsize=11)
    ax.tick_params(labelsize=8)
axes[0].set_ylabel("coverage", fontsize=9)
axes[0].legend(fontsize=8, loc="lower left")
plt.tight_layout()
fig.savefig(OUT, dpi=300, bbox_inches="tight")
print("saved:", OUT)
