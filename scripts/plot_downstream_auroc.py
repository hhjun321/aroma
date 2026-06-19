# -*- coding: utf-8 -*-
"""fig:downstream_auroc — Image-level AUROC across datasets x architectures.

Source: Table 11 in AROMA.txt (parsed from the Markdown pipe-table).
Produces TWO figures:
  1. fig_downstream_auroc.png  — 1x4 grouped bars (Baseline/Random/AROMA)
  2. fig_downstream_slope.png  — dumbbell/slope chart of Baseline->Random->AROMA
"""
import numpy as np
import matplotlib.pyplot as plt

from fig_common import (
    AROMA_TXT, parse_markdown_table, setup_font, save,
    COLOR_BASELINE, COLOR_RANDOM, COLOR_AROMA,
)


def load():
    _, rows = parse_markdown_table(
        AROMA_TXT,
        "Table 11. Image-level AUROC across four datasets",
    )
    # group by dataset, preserving order
    datasets = []
    data = {}
    for r in rows:
        ds = r["Dataset"]
        if ds not in data:
            data[ds] = []
            datasets.append(ds)
        data[ds].append((r["Model"], r["Baseline"], r["Random"], r["AROMA"]))
    return datasets, data


def grouped_bars(datasets, data):
    n = len(datasets)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.6), sharey=True)
    if n == 1:
        axes = [axes]

    conds = ["Baseline", "Random", "AROMA"]
    colors = [COLOR_BASELINE, COLOR_RANDOM, COLOR_AROMA]
    width = 0.26

    for ax, ds in zip(axes, datasets):
        rows = data[ds]
        models = [m for (m, _, _, _) in rows]
        vals = {
            "Baseline": [b for (_, b, _, _) in rows],
            "Random": [rr for (_, _, rr, _) in rows],
            "AROMA": [a for (_, _, _, a) in rows],
        }
        x = np.arange(len(models))
        for j, cond in enumerate(conds):
            offs = (j - 1) * width
            bars = ax.bar(x + offs, vals[cond], width, label=cond,
                          color=colors[j], edgecolor="white", linewidth=0.4)
            for b in bars:
                ax.annotate(f"{b.get_height():.3f}",
                            (b.get_x() + b.get_width() / 2, b.get_height()),
                            ha="center", va="bottom", fontsize=6, rotation=90,
                            xytext=(0, 1), textcoords="offset points")
        ax.set_title(ds, fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=30, ha="right", fontsize=8)
        ax.set_ylim(0.75, 1.0)
        ax.grid(axis="y", alpha=0.3, linewidth=0.5)
        ax.set_axisbelow(True)

    axes[0].set_ylabel("Image-level AUROC", fontsize=10)
    axes[0].legend(loc="lower left", fontsize=8, framealpha=0.9)
    fig.suptitle(
        "Image-level AUROC across four datasets and four architectures\n"
        "(Baseline / Random / AROMA), y-axis clamped to 0.75–1.0",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save(fig, "fig_downstream_auroc.png")


def slope_chart(datasets, data):
    # one row per (dataset, model); dumbbell of Baseline -> Random -> AROMA
    labels, base, rand, arom = [], [], [], []
    for ds in datasets:
        for (m, b, rr, a) in data[ds]:
            labels.append(f"{ds} / {m}")
            base.append(b)
            rand.append(rr)
            arom.append(a)

    y = np.arange(len(labels))[::-1]  # top-to-bottom
    fig, ax = plt.subplots(figsize=(8.5, 0.42 * len(labels) + 1.5))

    for yi, b, rr, a in zip(y, base, rand, arom):
        ax.plot([b, a], [yi, yi], color="#cccccc", linewidth=2, zorder=1)
    ax.scatter(base, y, color=COLOR_BASELINE, s=42, label="Baseline", zorder=3)
    ax.scatter(rand, y, color=COLOR_RANDOM, s=42, label="Random", zorder=3)
    ax.scatter(arom, y, color=COLOR_AROMA, s=42, label="AROMA", zorder=3)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Image-level AUROC", fontsize=10)
    ax.set_xlim(0.78, 0.99)
    ax.grid(axis="x", alpha=0.3, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
    ax.set_title(
        "Baseline -> Random -> AROMA AUROC deltas per (dataset, architecture)\n"
        "margins are small and model-dependent",
        fontsize=11,
    )
    fig.tight_layout()
    save(fig, "fig_downstream_slope.png")


if __name__ == "__main__":
    setup_font()
    datasets, data = load()
    grouped_bars(datasets, data)
    slope_chart(datasets, data)
