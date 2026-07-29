# -*- coding: utf-8 -*-
"""Figure 3.2.3-1 — Defect morphology-feature distributions per dataset.

Mirror of [figure 3.2.2 2] context_distribution.py (background side). Overlays the
Table 4 subtype thresholds (dashed) on the two features Table 4 constrains —
aspect_ratio and solidity. The other four profiled features are shown for
completeness: extent and circularity carry no criterion, while linearity and
eccentricity are deterministic reparameterisations of aspect_ratio. See
[figure 3.2.3 1] morphology_distribution.md.

Thresholds are the dataset's own P33/P66 tertiles (Table 4b) with the homogeneity
fallback, mirroring `roi_selection.py::_subtype_percentiles` — the same values the
pipeline uses under `--subtype_mode percentile`.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import csv, io, os, textwrap

ROOT = os.environ.get("AROMA_DATASET_ROOT", "D:/project/aroma_dataset")
PROF = f"{ROOT}/profiling/profiling"
IMG  = "D:/project/aroma/AROMA연구분석/Article/figure/image"
DATASETS = ["aitex", "kolektor", "severstal", "mtd", "mvtec_leather"]

FEATS = ["linearity", "solidity", "extent",
         "aspect_ratio", "eccentricity", "circularity"]

# Features carrying a Table 4 criterion. Thresholds are the dataset's own P33/P66
# tertiles (Table 4b), matching roi_selection.py::_subtype_percentiles — not fixed
# constants. linearity = 1 - aspect_ratio^-2 and eccentricity = sqrt(linearity) are
# deterministic functions of aspect_ratio (verified to machine precision on all 4504
# profiled defects), so they carry no criterion.
TABLE4_FEATS = ["aspect_ratio", "solidity"]

# Homogeneity safeguard, mirroring roi_selection.py: a middle tertile narrower than
# this fraction of the feature's standard deviation reverts to the fixed threshold.
TERTILE_DEGENERATE_RATIO = 0.15
FIXED_TERTILE_EQUIV = {"aspect_ratio": (2.0, 5.0), "solidity": (0.7, 0.9)}
LABELS = {
    "linearity":    "linearity = 1 - AR^-2\n(no criterion; reparam. of aspect_ratio)",
    "solidity":     "solidity\n(area / convex-hull area)",
    "extent":       "extent\n(area / bbox area) - no criterion",
    "aspect_ratio": "aspect_ratio (log x)\n(major / minor axis)",
    "eccentricity": "eccentricity = sqrt(linearity)\n(no criterion; reparam. of aspect_ratio)",
    "circularity":  "circularity - no criterion",
}
BLUE = "#4c78a8"
LOG_X = {"aspect_ratio"}   # heavy right skew — linear bins put the mode in one bar


def load_feats(ds):
    cols = {f: [] for f in FEATS}
    with io.open(f"{PROF}/{ds}/morphology_features.csv", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            for f in FEATS:
                v = r.get(f)
                if v not in (None, "", "nan"):
                    try:
                        cols[f].append(float(v))
                    except ValueError:
                        pass
    return {f: np.asarray(v, dtype=np.float64) for f, v in cols.items()}


def subtype_thresholds(data):
    """Per-dataset P33/P66 with the homogeneity fallback (mirrors roi_selection)."""
    th, fallback = {}, []
    for f in TABLE4_FEATS:
        x = data[f]
        p33, p66 = float(np.percentile(x, 33)), float(np.percentile(x, 66))
        sd = float(x.std())
        if sd > 0 and (p66 - p33) / sd < TERTILE_DEGENERATE_RATIO:
            th[f] = FIXED_TERTILE_EQUIV[f]
            fallback.append(f)
        else:
            th[f] = (p33, p66)
    return th, fallback


for ds in DATASETS:
    data = load_feats(ds)
    TABLE4, fellback = subtype_thresholds(data)
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    fig.suptitle(f"Defect Morphology Distributions — {ds}", fontsize=15, y=0.98)
    axes = axes.ravel()
    for i, f in enumerate(FEATS):
        ax = axes[i]
        x = data[f]
        if len(x) == 0:
            ax.set_visible(False)
            continue
        lo, hi = np.percentile(x, [1, 99])
        if hi <= lo:
            hi = lo + 1e-6
        xc = x[(x >= lo) & (x <= hi)]
        if f in LOG_X:
            xc = xc[xc > 0]
            bins = np.logspace(np.log10(max(xc.min(), 1e-3)), np.log10(xc.max()), 40)
            ax.set_xscale("log")
        else:
            bins = 40
        ax.hist(xc, bins=bins, color=BLUE, edgecolor="white", linewidth=0.3)
        for t in TABLE4.get(f, ()):
            if lo <= t <= hi:
                ax.axvline(t, color="red", linestyle="--", linewidth=1.4)
                ax.text(t, ax.get_ylim()[1] * 0.96, f" {t:.3g}", color="red",
                        fontsize=7, va="top", ha="left")
        ax.set_title(LABELS[f], fontsize=9)
        ax.set_xlabel("value", fontsize=8)
        ax.set_ylabel("count", fontsize=8)
        ax.tick_params(labelsize=7)
    note = (
        "Red dashed = Table 4 subtype thresholds for this dataset: aspect_ratio "
        f"{TABLE4['aspect_ratio'][0]:.3g} / {TABLE4['aspect_ratio'][1]:.3g}, "
        f"solidity {TABLE4['solidity'][0]:.3g} / {TABLE4['solidity'][1]:.3g} "
        "(its own P33 / P66 tertiles; Table 4b)."
    )
    if fellback:
        note += (" Fixed fallback on " + ", ".join(fellback)
                 + " - middle tertile below 15% of its standard deviation.")
    note += (" linearity and eccentricity are deterministic reparameterisations of "
             "aspect_ratio and carry no criterion. Thresholds outside the 1-99 "
             "percentile display range are not drawn.")
    fig.text(0.5, 0.005, "\n".join(textwrap.wrap(note, 150)),
             fontsize=8, ha="center", va="bottom")
    plt.tight_layout(rect=[0, 0.06, 1, 0.96])
    out = f"{IMG}/[figure 3.2.3 1 {ds}] morphology_distribution.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("saved:", out)
