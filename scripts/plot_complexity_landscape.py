# -*- coding: utf-8 -*-
"""fig:complexity_landscape — MCI vs CCI over 24 datasets.

Source: 24 complexity_report.json files (REAL JSON).
Scatter: x=MCI, y=CCI; color = morphology_policy, marker = context_policy.
The four downstream-evaluation datasets are highlighted with a ring + label.
"""
import os
import json
import glob

import matplotlib.pyplot as plt

from fig_common import COMPLEXITY_DIR, setup_font, save

# datasets used in downstream evaluation (Table 11). manuscript "visa_pcb4"
# corresponds to the visa_pcb complexity report on disk.
EVAL_DATASETS = {"isp_LSM_1", "mvtec_cable", "visa_cashew", "visa_pcb"}

# color per morphology policy
MORPH_COLORS = {
    "gmm": "#d62728",
    "otsu": "#1f77b4",
    "hierarchical": "#2ca02c",
    "percentile": "#9467bd",
}
# marker per context policy
CTX_MARKERS = {
    "gmm": "o",
    "percentile": "^",
    "otsu": "s",
    "hierarchical": "D",
}


def load():
    rows = []
    for path in sorted(glob.glob(os.path.join(COMPLEXITY_DIR, "*", "complexity_report.json"))):
        ds = os.path.basename(os.path.dirname(path))
        with open(path, "r", encoding="utf-8") as fh:
            r = json.load(fh)
        rows.append({
            "dataset": ds,
            "mci": r["mci"],
            "cci": r["cci"],
            "morph": r["morphology_policy"],
            "ctx": r["context_policy"],
        })
    return rows


def plot(rows):
    fig, ax = plt.subplots(figsize=(9, 7))

    for r in rows:
        color = MORPH_COLORS.get(r["morph"], "#7f7f7f")
        marker = CTX_MARKERS.get(r["ctx"], "o")
        is_eval = r["dataset"] in EVAL_DATASETS
        ax.scatter(
            r["mci"], r["cci"],
            c=color, marker=marker,
            s=170 if is_eval else 90,
            edgecolors="black" if is_eval else "white",
            linewidths=1.6 if is_eval else 0.5,
            alpha=0.95 if is_eval else 0.8,
            zorder=4 if is_eval else 2,
        )
        if is_eval:
            ax.annotate(
                r["dataset"],
                (r["mci"], r["cci"]),
                xytext=(7, 7), textcoords="offset points",
                fontsize=9, fontweight="bold",
            )

    ax.set_xlabel("Morphology Complexity Index (MCI)", fontsize=11)
    ax.set_ylabel("Context Complexity Index (CCI)", fontsize=11)
    ax.set_title(
        "Complexity landscape of 24 industrial datasets\n"
        "color = morphology policy, marker = context policy "
        "(auto-selected by the Meta Policy Generator)",
        fontsize=12,
    )
    ax.grid(alpha=0.3, linewidth=0.5)
    ax.set_axisbelow(True)

    # legends: two separate
    from matplotlib.lines import Line2D
    morph_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=c,
               markersize=10, label=k)
        for k, c in MORPH_COLORS.items()
        if any(r["morph"] == k for r in rows)
    ]
    ctx_handles = [
        Line2D([0], [0], marker=m, color="w", markerfacecolor="#555555",
               markersize=10, label=k)
        for k, m in CTX_MARKERS.items()
        if any(r["ctx"] == k for r in rows)
    ]
    eval_handle = [Line2D([0], [0], marker="o", color="w",
                          markerfacecolor="none", markeredgecolor="black",
                          markeredgewidth=1.6, markersize=12,
                          label="eval dataset")]

    leg1 = ax.legend(handles=morph_handles, title="Morphology policy",
                     loc="upper right", fontsize=9, framealpha=0.9)
    ax.add_artist(leg1)
    ax.legend(handles=ctx_handles + eval_handle, title="Context policy",
              loc="lower right", fontsize=9, framealpha=0.9)

    fig.tight_layout()
    save(fig, "fig_complexity_landscape.png")


if __name__ == "__main__":
    setup_font()
    rows = load()
    plot(rows)
