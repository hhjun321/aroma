# -*- coding: utf-8 -*-
"""fig:roi_coverage — ROI placement quality (AROMA vs Random).

Source: Table 8 in AROMA.txt.
Chart: 5 faceted grouped-bar panels (Morphology Cov., Context Cov.,
Rare-Pair Cov., Entropy, Gini), AROMA vs Random per dataset.
Gini panel annotated "lower is better".
"""
import numpy as np
import matplotlib.pyplot as plt

from fig_common import (
    AROMA_TXT, parse_markdown_table, setup_font, save,
    COLOR_RANDOM, COLOR_AROMA,
)

METRICS = ["Morphology Cov.", "Context Cov.", "Rare-Pair Cov.", "Entropy", "Gini"]
LOWER_BETTER = {"Gini"}


def load():
    _, rows = parse_markdown_table(
        AROMA_TXT, "Table 8. ROI placement quality")
    datasets = []
    # data[dataset][method] = {metric: value}
    data = {}
    for r in rows:
        ds, method = r["Dataset"], r["Method"]
        if ds not in data:
            data[ds] = {}
            datasets.append(ds)
        data[ds][method] = {m: r[m] for m in METRICS}
    return datasets, data


def plot(datasets, data):
    fig, axes = plt.subplots(1, len(METRICS), figsize=(3.1 * len(METRICS), 4.4))
    width = 0.38
    x = np.arange(len(datasets))

    for ax, metric in zip(axes, METRICS):
        rand = [data[ds]["Random"][metric] for ds in datasets]
        arom = [data[ds]["AROMA"][metric] for ds in datasets]
        b1 = ax.bar(x - width / 2, rand, width, label="Random",
                    color=COLOR_RANDOM, edgecolor="white", linewidth=0.4)
        b2 = ax.bar(x + width / 2, arom, width, label="AROMA",
                    color=COLOR_AROMA, edgecolor="white", linewidth=0.4)
        for bars in (b1, b2):
            for b in bars:
                ax.annotate(f"{b.get_height():.2f}",
                            (b.get_x() + b.get_width() / 2, b.get_height()),
                            ha="center", va="bottom", fontsize=6,
                            xytext=(0, 1), textcoords="offset points")
        title = metric + ("\n(lower is better)" if metric in LOWER_BETTER else "")
        ax.set_title(title, fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(datasets, rotation=35, ha="right", fontsize=7)
        ax.grid(axis="y", alpha=0.3, linewidth=0.5)
        ax.set_axisbelow(True)
        top = max(max(rand), max(arom))
        ax.set_ylim(0, top * 1.18)

    axes[0].set_ylabel("Value", fontsize=10)
    axes[0].legend(loc="upper left", fontsize=8, framealpha=0.9)
    fig.suptitle(
        "ROI placement quality (AROMA vs. Random) across four datasets",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save(fig, "fig_roi_coverage.png")


if __name__ == "__main__":
    setup_font()
    datasets, data = load()
    plot(datasets, data)
