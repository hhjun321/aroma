#!/usr/bin/env python
"""
Figure 2 -- AROMA Pipeline Architecture (regenerated)

Static architecture/data-flow diagram (no data file dependency).
Stage structure aligned 1:1 to the §3.2 subsections of the current
section3_2.txt (3.2.1 .. 3.2.7); dataset roster from dataset_config.json
(ground truth): 5 datasets (severstal, mvtec_leather, mtd, aitex,
kolektor).

Labeling kept minimal per user direction: no stage-number badges, no
side artifact/JSON boxes, no "v2-1" or "exp4v2" text in the diagram
itself (those stay in the spec doc / caption only).

Alignment update (2026-07-20, user-confirmed): boxes re-derived to match
the 7 subsections of the current §3.2. The stale "Meta Policy Generator /
auto-select modeling policy" narrative is removed -- the symmetric
compatibility gate is the single spine/novelty (memory:
aroma-compat-gate-spine-reframe). ROI score corrected to the current
ROI_score = 0.6*ctx_prior + 0.4*morph_prior (no quality term).

spec: [figure 3.2] pipeline_spec.md
"""
import os

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT_DIR = r"D:\project\aroma\AROMA연구분석\Article\figure\image"
OUT_PATH = os.path.join(OUT_DIR, "[figure 3.2] aroma_pipeline.png")

# (title, detail) -- kept short by design, one line each; aligned to §3.2.1-3.2.7
STAGES = [
    ("Dataset Complexity Analysis",
     "MCI / CCI from patch profiling"),
    ("Morphology & Context Modeling",
     "Data-driven clusters (GMM+BIC) + tertile context cells"),
    ("ROI Extraction",
     "Otsu + connected-components; texture categories"),
    ("Seed Defect Classification",
     "SAM masks -> 5 morphology subtypes"),
    ("ROI Selection & Placement",
     "ROI_score = 0.6*ctx_prior + 0.4*morph_prior; symmetric compat gate"),
    ("Blending Synthesis",
     "Seamless copy-paste; AROMA vs Random arm"),
    ("Quality Gate",
     "Composite Q >= 0.7 filter"),
]

INPUT_TITLE = "Industrial Datasets"
INPUT_DETAIL = "severstal . mvtec_leather . mtd . aitex . kolektor"
OUTPUT_TITLE = "Downstream Detection"
OUTPUT_DETAIL = "YOLOv8n mAP50: baseline / random / aroma"

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
    row_h = 1.0
    gap = 0.35
    total_h = (n + 2) * row_h + (n + 1) * gap
    fig, ax = plt.subplots(figsize=(7.2, total_h * 0.85), dpi=300)

    main_w, main_x = 6.4, 0.3

    y = total_h - row_h

    # input box
    add_box(ax, main_x, y, main_w, row_h, END_COLOR)
    ax.text(main_x + main_w / 2, y + row_h * 0.62, INPUT_TITLE,
            ha="center", va="center", fontsize=12, fontweight="bold")
    ax.text(main_x + main_w / 2, y + row_h * 0.28, INPUT_DETAIL,
            ha="center", va="center", fontsize=9)
    ax.annotate("", xy=(main_x + main_w / 2, y - gap * 0.15),
                xytext=(main_x + main_w / 2, y),
                arrowprops=dict(arrowstyle="-|>", color="#555555", linewidth=1.2))
    y -= (row_h + gap)

    for title, detail in STAGES:
        add_box(ax, main_x, y, main_w, row_h, BOX_COLOR)
        ax.text(main_x + main_w / 2, y + row_h * 0.64, title,
                ha="center", va="center", fontsize=11, fontweight="bold")
        ax.text(main_x + main_w / 2, y + row_h * 0.28, detail,
                ha="center", va="center", fontsize=8.3)

        # arrow to next stage
        ax.annotate("", xy=(main_x + main_w / 2, y - gap * 0.15),
                    xytext=(main_x + main_w / 2, y),
                    arrowprops=dict(arrowstyle="-|>", color="#555555", linewidth=1.2))
        y -= (row_h + gap)

    # output box
    add_box(ax, main_x, y, main_w, row_h, END_COLOR)
    ax.text(main_x + main_w / 2, y + row_h * 0.62, OUTPUT_TITLE,
            ha="center", va="center", fontsize=12, fontweight="bold")
    ax.text(main_x + main_w / 2, y + row_h * 0.28, OUTPUT_DETAIL,
            ha="center", va="center", fontsize=9)

    ax.set_xlim(0, main_x * 2 + main_w)
    ax.set_ylim(y - 0.2, total_h + 0.1)
    ax.axis("off")

    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=300, bbox_inches="tight")
    print("saved:", OUT_PATH)


if __name__ == "__main__":
    main()
