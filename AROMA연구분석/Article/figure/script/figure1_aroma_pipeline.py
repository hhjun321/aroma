#!/usr/bin/env python
"""
Figure 1 -- AROMA Pipeline Architecture (regenerated)

Static architecture/data-flow diagram (no data file dependency).
Content sourced from AROMA_FRAMEWORK/{00-INDEX,01-Overview,02..06}.md
for stage structure, and dataset_config.json (ground truth) for the
dataset roster: 5 datasets (severstal, mvtec_leather, mtd, aitex,
kolektor) -- the framework notes say 4 and are stale.

Labeling kept minimal per user direction (2026-07-16): no stage-number
badges, no side artifact/JSON boxes, no "v2-1" or "exp4v2" text in the
diagram itself (those stay in the spec doc / caption only).

Engine correction (2026-07-16, user-confirmed): official AROMA flow
generates via copy-paste (real defect crop + elastic-warp variants),
NOT ControlNet. "Prompt Generation" stage removed (was ControlNet
conditioning-only, has no role in copy-paste), and the ControlNet
train/fine-tune stage replaced with the copy-paste variant-generation
+ compat-gate-tau-prescan stage.

spec: figure1_pipeline_spec.md
"""
import os

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT_DIR = r"D:\project\aroma\AROMA연구분석\Article\figure\image"
OUT_PATH = os.path.join(OUT_DIR, "[figure1] aroma_pipeline.png")

# (title, detail) -- kept short by design, no stage-number badge
STAGES = [
    ("Prepare Datasets",
     "Normalize 5 datasets to AROMA layout"),
    ("Distribution Profiling",
     "Morphology/context distributions -> compatibility model"),
    ("Complexity + Meta Policy",
     "MCI/CCI scalars -> auto-select modeling policy"),
    ("ROI Selection",
     "Realism score, top-k = 200"),
    ("Clean-BG Selection",
     "Offline background ranking (hist-match + void filter)"),
    ("Elastic-Warp Variants",
     "Copy-paste seed variants (cv2.remap)"),
    ("Generate: AROMA / Random arm",
     "AROMA = copy-paste + sym gate + clean-bg + seamless; Random = naive"),
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
