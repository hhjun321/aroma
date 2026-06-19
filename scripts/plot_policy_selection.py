# -*- coding: utf-8 -*-
"""fig:policy_selection_evidence — silhouette/stability per candidate policy.

Source: evaluation_results array inside complexity_report.json (REAL JSON).
Grouped bar of silhouette score per candidate policy, faceted by axis
(morphology / context), for the four evaluation datasets. The selected
(winning) policy per axis is highlighted.
"""
import os
import json

import numpy as np
import matplotlib.pyplot as plt

from fig_common import COMPLEXITY_DIR, setup_font, save, COLOR_AROMA

EVAL_DATASETS = ["isp_LSM_1", "mvtec_cable", "visa_cashew", "visa_pcb"]
AXES = ["morphology", "context"]


def load(ds):
    path = os.path.join(COMPLEXITY_DIR, ds, "complexity_report.json")
    with open(path, "r", encoding="utf-8") as fh:
        r = json.load(fh)
    selected = {"morphology": r["morphology_policy"],
                "context": r["context_policy"]}
    # axis -> list of (policy, silhouette)
    by_axis = {a: [] for a in AXES}
    for e in r["evaluation_results"]:
        by_axis[e["axis"]].append((e["policy"], e.get("silhouette")))
    return by_axis, selected


def plot():
    n = len(EVAL_DATASETS)
    fig, axes = plt.subplots(len(AXES), n, figsize=(3.4 * n, 6.4),
                             sharey="row")

    for col, ds in enumerate(EVAL_DATASETS):
        by_axis, selected = load(ds)
        for row, axis in enumerate(AXES):
            ax = axes[row, col]
            items = by_axis[axis]
            policies = [p for (p, _) in items]
            scores = [s if s is not None else 0.0 for (_, s) in items]
            colors = [COLOR_AROMA if p == selected[axis] else "#bbbbbb"
                      for p in policies]
            x = np.arange(len(policies))
            bars = ax.bar(x, scores, 0.55, color=colors,
                          edgecolor="black", linewidth=0.5)
            for b, p, s in zip(bars, policies, scores):
                tag = "  (selected)" if p == selected[axis] else ""
                ax.annotate(f"{s:.3f}",
                            (b.get_x() + b.get_width() / 2, b.get_height()),
                            ha="center", va="bottom", fontsize=7,
                            xytext=(0, 1), textcoords="offset points")
            ax.set_xticks(x)
            ax.set_xticklabels([p + ("*" if p == selected[axis] else "")
                                for p in policies], fontsize=8)
            ax.grid(axis="y", alpha=0.3, linewidth=0.5)
            ax.set_axisbelow(True)
            top = max(scores) if scores else 1.0
            ax.set_ylim(0, top * 1.25)
            if row == 0:
                ax.set_title(ds, fontsize=10)
            if col == 0:
                ax.set_ylabel(f"{axis}\nsilhouette", fontsize=10)

    fig.suptitle(
        "Policy-selection evidence: candidate silhouette scores per axis\n"
        "red = policy chosen by the Meta Policy Generator (marked *)",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save(fig, "fig_policy_selection.png")


if __name__ == "__main__":
    setup_font()
    plot()
