# -*- coding: utf-8 -*-
"""fig:auroc_gain_by_complexity — AROMA-over-Baseline gain vs dataset complexity.

Joins Table 11 deltas (AROMA - Baseline) with MCI+CCI from the complexity
JSON for the four evaluation datasets. One series per architecture.
"""
import os
import json

import numpy as np
import matplotlib.pyplot as plt

from fig_common import (
    AROMA_TXT, COMPLEXITY_DIR, parse_markdown_table, setup_font, save,
)

# manuscript dataset name -> complexity-report dir name
DS_MAP = {
    "isp_LSM_1": "isp_LSM_1",
    "mvtec_cable": "mvtec_cable",
    "visa_cashew": "visa_cashew",
    "visa_pcb4": "visa_pcb",
}


def load_complexity(ds_dir):
    path = os.path.join(COMPLEXITY_DIR, ds_dir, "complexity_report.json")
    with open(path, "r", encoding="utf-8") as fh:
        r = json.load(fh)
    return r["mci"] + r["cci"]


def load():
    _, rows = parse_markdown_table(
        AROMA_TXT, "Table 11. Image-level AUROC across four datasets")
    # arch -> list of (combined_complexity, gain, dataset)
    series = {}
    for r in rows:
        ds = r["Dataset"]
        if ds not in DS_MAP:
            continue
        comp = load_complexity(DS_MAP[ds])
        gain = r["AROMA"] - r["Baseline"]
        series.setdefault(r["Model"], []).append((comp, gain, ds))
    return series


def plot(series):
    fig, ax = plt.subplots(figsize=(8.5, 6))
    cmap = plt.get_cmap("tab10")

    for i, (arch, pts) in enumerate(series.items()):
        pts = sorted(pts, key=lambda p: p[0])
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(xs, ys, marker="o", color=cmap(i), label=arch, linewidth=1.6)

    # label datasets on the x axis using the first arch's points
    any_pts = sorted(next(iter(series.values())), key=lambda p: p[0])
    for comp, _, ds in any_pts:
        ax.axvline(comp, color="#dddddd", linewidth=0.7, zorder=0)
        ax.annotate(ds, (comp, ax.get_ylim()[0]), rotation=90,
                    fontsize=7, va="bottom", ha="right", color="#666666")

    ax.set_xlabel("Dataset complexity (MCI + CCI)", fontsize=11)
    ax.set_ylabel("AUROC gain (AROMA - Baseline)", fontsize=11)
    ax.set_title(
        "AROMA-over-Baseline AUROC gain vs. dataset complexity, per architecture",
        fontsize=12,
    )
    ax.grid(alpha=0.3, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="best", fontsize=9, framealpha=0.9)
    fig.tight_layout()
    save(fig, "fig_gain_vs_complexity.png")


if __name__ == "__main__":
    setup_font()
    series = load()
    plot(series)
