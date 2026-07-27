#!/usr/bin/env python
"""
Figure 2 -- AROMA Pipeline Architecture (regenerated)

Static architecture/data-flow diagram (no data file dependency).
Stage structure aligned 1:1 to the §3.2 subsections of the current
section3_2.txt (3.2.1 .. 3.2.6); dataset roster from dataset_config.json
(ground truth): 5 datasets (severstal, mvtec_leather, mtd, aitex,
kolektor).

Labeling kept minimal per user direction: no stage-number badges, no
side artifact/JSON boxes, no "v2-1" or "exp4v2" text in the diagram
itself, no per-box detail line (those stay in the spec doc / caption
only) -- title-only boxes for a simple framework-pipeline look.

Alignment update (2026-07-27, user-confirmed): re-derived to match the
actual 6 subsections of the current §3.2 (section3_2.txt has no 3.2.7;
the former "ROI Extraction" and "Seed Defect Classification" stages
are merged into the single §3.2.3, shifting ROI Selection/Blending/
Quality Gate down by one section number). Quality Gate description
corrected to match the actual background-patch quality gate (blur/
contrast/brightness/noise, accept iff quality >= 0.7) -- not a
post-composite artifact+blur ranking, which does not exist in code.
compat_sym references unified to matrix_symmetric. Per-box detail
subtitle removed; boxes now show titles only.

spec: [figure 3.2] pipeline_spec.md
"""
import os

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT_DIR = r"D:\project\aroma\AROMA연구분석\Article\figure\image"
OUT_PATH = os.path.join(OUT_DIR, "[figure 3.2] aroma_pipeline.png")

# title-only -- aligned to §3.2.1-3.2.6 (see pipeline_spec.md for the
# one-line description of each stage; kept out of the diagram itself)
STAGES = [
    "Dataset Complexity Analysis",
    "Morphology & Context Modeling",
    "ROI Extraction & Defect Subtype Classification",
    "ROI Selection & Compatibility-Aware Placement",
    "Blending Synthesis",
    "Quality Gate",
]

INPUT_TITLE = "Industrial Datasets"
OUTPUT_TITLE = "Downstream Detection"

BOX_COLOR = "#c9b8e8"
END_COLOR = "#e5e5e5"


def add_box(ax, x, y, w, h, facecolor, edgecolor="#333333"):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        facecolor=facecolor, edgecolor=edgecolor, linewidth=1.2,
    )
    ax.add_patch(box)
    return box


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})

    n = len(STAGES)
    row_h = 0.8
    gap = 0.32
    total_h = (n + 2) * row_h + (n + 1) * gap
    fig, ax = plt.subplots(figsize=(7.2, total_h * 0.85), dpi=300)

    main_w, main_x = 6.4, 0.3

    y = total_h - row_h

    # input box (title only)
    add_box(ax, main_x, y, main_w, row_h, END_COLOR)
    ax.text(main_x + main_w / 2, y + row_h / 2, INPUT_TITLE,
            ha="center", va="center", fontsize=12, fontweight="bold")
    ax.annotate("", xy=(main_x + main_w / 2, y - gap * 0.15),
                xytext=(main_x + main_w / 2, y),
                arrowprops=dict(arrowstyle="-|>", color="#555555", linewidth=1.2))
    y -= (row_h + gap)

    for title in STAGES:
        add_box(ax, main_x, y, main_w, row_h, BOX_COLOR)
        ax.text(main_x + main_w / 2, y + row_h / 2, title,
                ha="center", va="center", fontsize=11, fontweight="bold")

        # arrow to next stage
        ax.annotate("", xy=(main_x + main_w / 2, y - gap * 0.15),
                    xytext=(main_x + main_w / 2, y),
                    arrowprops=dict(arrowstyle="-|>", color="#555555", linewidth=1.2))
        y -= (row_h + gap)

    # output box (title only)
    add_box(ax, main_x, y, main_w, row_h, END_COLOR)
    ax.text(main_x + main_w / 2, y + row_h / 2, OUTPUT_TITLE,
            ha="center", va="center", fontsize=12, fontweight="bold")

    ax.set_xlim(0, main_x * 2 + main_w)
    ax.set_ylim(y - 0.2, total_h + 0.1)
    ax.axis("off")

    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=300, bbox_inches="tight")
    print("saved:", OUT_PATH)


if __name__ == "__main__":
    main()
