# -*- coding: utf-8 -*-
"""fig:synthesis_quality — FID / KID / LPIPS, AROMA vs Random.

Source: Table 9 (FID + Delta) and Table 10 (KID, LPIPS) in AROMA.txt.
Top row: three grouped-bar panels (FID / KID / LPIPS), AROMA vs Random.
Bottom row: Delta (AROMA - Random) bar chart per metric (lower = better,
so negative delta = AROMA wins).
"""
import numpy as np
import matplotlib.pyplot as plt

from fig_common import (
    AROMA_TXT, parse_markdown_table, setup_font, save,
    COLOR_RANDOM, COLOR_AROMA,
)


def load():
    _, fid_rows = parse_markdown_table(AROMA_TXT, "Table 9. Full-image FID")
    _, kl_rows = parse_markdown_table(AROMA_TXT, "Table 10. KID and LPIPS")

    datasets = [r["Dataset"] for r in fid_rows]
    fid_a = {r["Dataset"]: r["AROMA FID"] for r in fid_rows}
    fid_r = {r["Dataset"]: r["Random FID"] for r in fid_rows}
    kid_a = {r["Dataset"]: r["AROMA KID"] for r in kl_rows}
    kid_r = {r["Dataset"]: r["Random KID"] for r in kl_rows}
    lp_a = {r["Dataset"]: r["AROMA LPIPS"] for r in kl_rows}
    lp_r = {r["Dataset"]: r["Random LPIPS"] for r in kl_rows}

    metrics = {
        "FID": (fid_a, fid_r),
        "KID": (kid_a, kid_r),
        "LPIPS": (lp_a, lp_r),
    }
    return datasets, metrics


def plot(datasets, metrics):
    names = list(metrics.keys())
    fig, axes = plt.subplots(2, len(names), figsize=(3.5 * len(names), 7.2))
    width = 0.38
    x = np.arange(len(datasets))

    # ---- top row: grouped bars (each metric own axis/scale) ----
    for col, name in enumerate(names):
        a_map, r_map = metrics[name]
        rand = [r_map[ds] for ds in datasets]
        arom = [a_map[ds] for ds in datasets]
        ax = axes[0, col]
        ax.bar(x - width / 2, rand, width, label="Random",
               color=COLOR_RANDOM, edgecolor="white", linewidth=0.4)
        ax.bar(x + width / 2, arom, width, label="AROMA",
               color=COLOR_AROMA, edgecolor="white", linewidth=0.4)
        ax.set_title(f"{name} (lower is better)", fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(datasets, rotation=35, ha="right", fontsize=7)
        ax.grid(axis="y", alpha=0.3, linewidth=0.5)
        ax.set_axisbelow(True)
        if col == 0:
            ax.legend(loc="upper right", fontsize=8, framealpha=0.9)

    # ---- bottom row: delta (AROMA - Random) ----
    for col, name in enumerate(names):
        a_map, r_map = metrics[name]
        delta = [a_map[ds] - r_map[ds] for ds in datasets]
        ax = axes[1, col]
        colors = [COLOR_AROMA if d <= 0 else COLOR_RANDOM for d in delta]
        bars = ax.bar(x, delta, 0.55, color=colors,
                      edgecolor="white", linewidth=0.4)
        for b, d in zip(bars, delta):
            ax.annotate(f"{d:+.4f}".rstrip("0").rstrip(".") if name != "FID"
                        else f"{d:+.2f}",
                        (b.get_x() + b.get_width() / 2, b.get_height()),
                        ha="center",
                        va="bottom" if d >= 0 else "top",
                        fontsize=6,
                        xytext=(0, 2 if d >= 0 else -2),
                        textcoords="offset points")
        ax.axhline(0, color="black", linewidth=0.7)
        ax.set_title(f"Delta {name} (AROMA - Random)\nnegative = AROMA better",
                     fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(datasets, rotation=35, ha="right", fontsize=7)
        ax.grid(axis="y", alpha=0.3, linewidth=0.5)
        ax.set_axisbelow(True)

    fig.suptitle(
        "Synthesis quality: FID / KID / LPIPS (AROMA vs. Random)\n"
        "margins are tiny (identical copy-paste defect pixels)",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save(fig, "fig_synthesis_quality.png")


if __name__ == "__main__":
    setup_font()
    datasets, metrics = load()
    plot(datasets, metrics)
