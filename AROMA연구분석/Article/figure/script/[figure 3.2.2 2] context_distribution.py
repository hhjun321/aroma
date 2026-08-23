# -*- coding: utf-8 -*-
"""Figure 13 — Background context-feature distributions per dataset (background_type).
Mirror of morphology_histograms (defect_type). Overlays real compat cell boundaries
(bin_edges = P33/P66 tertiles). See [figure 3.2.2 2] context_distribution.md.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import csv, io, json, os

PROF = os.environ.get("AROMA_DATASET_ROOT", "D:/project/aroma_dataset") + "/profiling/profiling"
IMG  = "D:/project/aroma/AROMA연구분석/Article/figure/image"
DATASETS = ["aitex", "kolektor", "severstal", "mtd", "mvtec_leather"]
FEATS = ["local_variance", "edge_density", "texture_entropy",
         "frequency_energy", "orientation_consistency"]
LABELS = {
    "local_variance": "local_variance",
    "edge_density": "edge_density",
    "texture_entropy": "texture_entropy (LBP)",
    "frequency_energy": "frequency_energy (HF ratio)",
    "orientation_consistency": "orientation_consistency\n(entropy; low = coherent)",
}
BLUE = "#4c78a8"

def load_feats(ds):
    cols = {f: [] for f in FEATS}
    with io.open(f"{PROF}/{ds}/context_features.csv", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            for f in FEATS:
                v = r.get(f)
                if v not in (None, "", "nan"):
                    try: cols[f].append(float(v))
                    except ValueError: pass
    return {f: np.asarray(v, dtype=np.float64) for f, v in cols.items()}

for ds in DATASETS:
    data = load_feats(ds)
    edges = json.load(io.open(f"{PROF}/{ds}/compatibility_matrix.json", encoding="utf-8"))["bin_edges"]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    fig.suptitle(f"Background Context Distributions — {ds}", fontsize=15, y=0.98)
    axes = axes.ravel()
    for i, f in enumerate(FEATS):
        ax = axes[i]
        x = data[f]
        if len(x) == 0:
            ax.set_visible(False); continue
        lo, hi = np.percentile(x, [1, 99])
        if hi <= lo: hi = lo + 1e-6
        xc = x[(x >= lo) & (x <= hi)]
        ax.hist(xc, bins=40, color=BLUE, edgecolor="white", linewidth=0.3)
        for e in edges.get(f, []):
            if lo <= e <= hi:
                ax.axvline(e, color="red", linestyle="--", linewidth=1.4)
        ax.set_title(LABELS[f], fontsize=12)
        ax.set_xlabel("value", fontsize=12)
        ax.set_ylabel("count", fontsize=12)
        ax.tick_params(labelsize=11)
    # 6th cell: unused
    axes[5].axis("off")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out = f"{IMG}/[figure 3.2.2 2 {ds}] context_distribution.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("saved:", out)
