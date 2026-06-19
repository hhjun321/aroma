# -*- coding: utf-8 -*-
"""fig:exp4v2_map50 — YOLOv8 supervised detection mAP@0.5 across conditions.

Bar chart analogous to fig_downstream_auroc.png, but for the Exp4v2 supervised
detection results. Reads the real result JSON (single source of truth):

    .claude/.etc/exp4v2/exp4v2_results.json

Conditions and colors (shared with the rest of the paper figures):
    baseline = gray   (#7f7f7f)
    random   = blue   (#1f77b4)
    aroma    = red    (#d62728)

NOTE: baseline/random results are valid (pipeline smoke test passed).
AROMA n_synth=0 due to stale normal_image path; bar is marked "synth=0 (invalid)".

Output: AROMA연구분석/Article/figure/fig_exp4v2_map50.png
"""
import json
import os

import numpy as np
import matplotlib.pyplot as plt

from fig_common import (
    PROJECT_ROOT, setup_font, save,
    COLOR_BASELINE, COLOR_RANDOM, COLOR_AROMA,
)

RESULTS_JSON = os.path.join(
    PROJECT_ROOT, ".claude", ".etc", "exp4v2", "exp4v2_results.json"
)

CONDS = ["baseline", "random", "aroma"]
COND_LABELS = ["Baseline", "Random", "AROMA"]
COND_COLORS = [COLOR_BASELINE, COLOR_RANDOM, COLOR_AROMA]
METRIC = "map50"


def load():
    """Return ordered list of (cell_label, {cond: map50}, {cond: n_train}, {cond: n_synth})."""
    with open(RESULTS_JSON, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    cells = []
    for ds, models in data.items():
        for model, arms in models.items():
            label = f"{ds}\n{model}"
            vals = {c: float(arms.get(c, {}).get(METRIC, 0.0)) for c in CONDS}
            ntr = {c: int(arms.get(c, {}).get("n_train", 0)) for c in CONDS}
            nsynth = {c: int(arms.get(c, {}).get("n_synth_train", -1)) for c in CONDS}
            cells.append((label, vals, ntr, nsynth))
    return cells


def grouped_bars(cells):
    n = len(cells)
    fig, ax = plt.subplots(figsize=(max(5.0, 3.0 * n), 5.0))

    x = np.arange(n)
    width = 0.26
    all_vals = []

    for j, cond in enumerate(CONDS):
        offs = (j - 1) * width
        heights = [vals[cond] for (_, vals, _ntr, _ns) in cells]
        all_vals.extend(heights)
        bars = ax.bar(
            x + offs, heights, width, label=COND_LABELS[j],
            color=COND_COLORS[j], edgecolor="white", linewidth=0.5,
        )
        for b, (_, _vals, ntr, nsynth) in zip(bars, cells):
            ann_label = f"{b.get_height():.3f}\n(n={ntr[cond]})"
            if cond == "aroma" and nsynth.get("aroma", -1) == 0:
                ann_label = f"{b.get_height():.3f}\n(synth=0\ninvalid)"
                b.set_hatch("///")
                b.set_alpha(0.6)
            ax.annotate(
                ann_label,
                (b.get_x() + b.get_width() / 2, b.get_height()),
                ha="center", va="bottom", fontsize=7,
                xytext=(0, 2), textcoords="offset points",
            )

    ax.set_xticks(x)
    ax.set_xticklabels([lbl for (lbl, _, _ntr, _ns) in cells], fontsize=9)
    ax.set_ylabel("mAP@0.5", fontsize=10)

    vmax = max(all_vals) if all_vals else 0.0
    # When the run is degenerate (all zeros), show a small symbolic axis so the
    # bars/annotations are readable and the failure is unmistakable.
    if vmax <= 0.0:
        ax.set_ylim(0.0, 1.0)
        ax.text(
            0.5, 0.5,
            "INVALID RUN — all metrics 0.0\n"
            "eval/label-pipeline failure; AROMA n_train=0",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=12, color="#b00020", fontweight="bold",
            bbox=dict(boxstyle="round", facecolor="#fdecea",
                      edgecolor="#b00020", alpha=0.9),
        )
    else:
        ax.set_ylim(0.0, min(1.0, vmax * 1.25))

    ax.grid(axis="y", alpha=0.3, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax.set_title(
        "Exp4v2 — YOLOv8 supervised detection mAP@0.5\n"
        "(Baseline / Random / AROMA)",
        fontsize=12,
    )
    fig.tight_layout()
    save(fig, "fig_exp4v2_map50.png")


if __name__ == "__main__":
    setup_font()
    cells = load()
    grouped_bars(cells)
