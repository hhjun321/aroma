#!/usr/bin/env python
"""
Figure 2 — Complexity Landscape (MCI x CCI)

§3.1을 시각화: 5개 평가 데이터셋이 형태(MCI)·맥락(CCI) 복잡도 평면에서
서로 다른 영역에 분포하며, 각 데이터셋에서 프레임워크가 자동 선택한
morphology / context policy를 함께 표시.

모든 수치는 aroma_dataset/complexity/<dataset>/complexity_report.json 에서 직접 로드
(하드코딩 금지). spec: figure2_complexity_landscape.md
"""
import json
import os

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# --- paths ---
DATA_ROOT = r"D:\project\aroma_dataset\complexity"
OUT_DIR = r"D:\project\aroma\AROMA연구분석\Article\figure\image"
OUT_PATH = os.path.join(OUT_DIR, "[figure2] complexity_landscape.png")

# dataset_key: (display name, granularity)  granularity: 'single' | 'multi'
DATASETS = [
    ("aitex",         "AITeX",         "single"),
    ("kolektor",      "Kolektor",      "single"),
    ("severstal",     "Severstal",     "multi"),
    ("mtd",           "MTD",           "multi"),
    ("mvtec_leather", "MVTec Leather", "multi"),
]

# color-blind-safe qualitative palette (Okabe-Ito subset)
PALETTE = {
    "AITeX":         "#0072B2",  # blue
    "Kolektor":      "#E69F00",  # orange
    "Severstal":     "#009E73",  # green
    "MTD":           "#CC79A7",  # magenta
    "MVTec Leather": "#D55E00",  # vermillion
}

# manual label offsets (dx, dy) in data units to avoid overlap
LABEL_OFFSET = {
    "AITeX":         (0.006, 0.006),
    "Kolektor":      (0.008, -0.004),
    "Severstal":     (0.008, 0.004),
    "MTD":           (0.008, 0.004),
    "MVTec Leather": (0.007, -0.006),
}

# vertical alignment per label ('bottom' places label above the point)
LABEL_VA = {
    "AITeX": "bottom",
    "Kolektor": "bottom",
    "Severstal": "bottom",
    "MTD": "bottom",
    "MVTec Leather": "top",
}


def load_report(key):
    path = os.path.join(DATA_ROOT, key, "complexity_report.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.linewidth": 0.8,
    })

    fig, ax = plt.subplots(figsize=(6.2, 5.2), dpi=300)

    for key, name, gran in DATASETS:
        rep = load_report(key)
        mci = rep["mci"]
        cci = rep["cci"]
        morph = rep["morphology_policy"]
        ctx = rep["context_policy"]
        marker = "s" if gran == "single" else "o"
        color = PALETTE[name]

        ax.scatter(
            mci, cci,
            s=190, marker=marker,
            facecolor=color, edgecolor="black", linewidth=0.9,
            zorder=3,
        )
        dx, dy = LABEL_OFFSET[name]
        ax.annotate(
            f"{name}\n({morph} / {ctx})",
            xy=(mci, cci), xytext=(mci + dx, cci + dy),
            fontsize=9, ha="left", va=LABEL_VA[name],
            color="black", zorder=4,
            linespacing=1.15,
        )

    # axes
    ax.set_xlim(0.30, 0.64)
    ax.set_ylim(0.15, 0.47)
    ax.set_xticks([0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60])
    ax.set_yticks([0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45])
    ax.set_xlabel("Morphological complexity (MCI)  →", fontsize=11)
    ax.set_ylabel("Contextual complexity (CCI)  →", fontsize=11)
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.5, zorder=0)

    # granularity legend
    legend_handles = [
        Line2D([0], [0], marker="s", color="w", markerfacecolor="grey",
               markeredgecolor="black", markersize=11, label="Single-class"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="grey",
               markeredgecolor="black", markersize=11, label="Multi-class"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", frameon=True,
              framealpha=0.9, fontsize=9, title="Class granularity",
              title_fontsize=9)

    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=300, bbox_inches="tight")
    print("saved:", OUT_PATH)


if __name__ == "__main__":
    main()
