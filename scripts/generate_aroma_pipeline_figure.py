#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AROMA pipeline figure generator.

Produces a vertical pipeline diagram (CASDA-style) describing the AROMA
synthetic-defect augmentation pipeline, from multi-domain datasets through
the seven processing stages to the downstream anomaly-detection models.

Output:
    D:\\project\\aroma\\AROMA연구분석\\Article\\figure\\[figure2] aroma_pipeline.png
"""

import os

import matplotlib

matplotlib.use("Agg")  # headless backend; no display required
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch


# --------------------------------------------------------------------------- #
# Global style
# --------------------------------------------------------------------------- #
def _pick_font():
    """Pick a clean sans-serif that exists on this machine (Korean-safe)."""
    preferred = [
        "Malgun Gothic",   # Windows Korean
        "Noto Sans CJK KR",
        "NanumGothic",
        "Arial",
        "DejaVu Sans",
    ]
    available = {f.name for f in fm.fontManager.ttflist}
    for name in preferred:
        if name in available:
            return name
    return "DejaVu Sans"


_FONT = _pick_font()
plt.rcParams.update(
    {
        "font.family": _FONT,
        "font.size": 10,
        "axes.unicode_minus": False,
        "figure.dpi": 150,
        "savefig.dpi": 150,
    }
)


# --------------------------------------------------------------------------- #
# Color palette
# --------------------------------------------------------------------------- #
COL_TERMINAL_FILL = "#E8E8E8"   # top/bottom light-gray boxes
COL_TERMINAL_EDGE = "#888888"
COL_STAGE_FILL = "#C8B9E8"      # lavender stage boxes
COL_STAGE_EDGE = "#9B86C9"
COL_ANNOT_FILL = "#B8E8C8"      # light-green right-side annotation boxes
COL_ANNOT_EDGE = "#7FC79A"
COL_SUBTEXT = "#5A5A5A"         # gray italic sub-annotation
COL_ARROW = "#9A9A9A"

# Per-stage label-circle colors
LBL_PHASE0 = "#A8C8E8"   # light blue
LBL_ROI = "#C8B9E8"      # light purple
LBL_TFIDG = "#A8D8C8"    # light teal
LBL_QGATE = "#F0C878"    # light orange


# --------------------------------------------------------------------------- #
# Pipeline content
# --------------------------------------------------------------------------- #
# Each stage:
#   label      -> short text inside the left circle ("0", "1", "1b", ...)
#   title      -> bold stage title
#   sub        -> gray italic sub-annotation (may contain newlines)
#   annot      -> right-side green annotation box text (or None)
#   color      -> label-circle fill color
STAGES = [
    {
        "label": "0",
        "title": "Phase 0:  Dataset Complexity Analysis",
        "sub": (
            "MCI = Mean(Entropy, ValleyCount, ClusterCount, 1-Silhouette)\n"
            "CCI = Mean(TextureEntropy, ClusterDiversity, FreqComplexity, OrientVariance)"
        ),
        "annot": "Meta Policy\nGenerator\nauto-selects\nclustering policy",
        "color": LBL_PHASE0,
    },
    {
        "label": "1",
        "title": "Stage 1:  ROI Extraction",
        "sub": (
            "Otsu thresholding → connected components → background type classification\n"
            "5 types: smooth · directional · periodic · organic · complex"
        ),
        "annot": "roi_metadata\n.json",
        "color": LBL_ROI,
    },
    {
        "label": "1b",
        "title": "Stage 1b:  Seed Defect Classification",
        "sub": (
            "SAM segmentation (Otsu fallback) → 5 subtypes\n"
            "linear_scratch · elongated · compact_blob · irregular · general"
        ),
        "annot": "seed_profile\n.json",
        "color": LBL_ROI,
    },
    {
        "label": "2",
        "title": "Stage 2:  TF-IDG Elastic Warping",
        "sub": (
            "Subtype-specific cv2.remap() displacement · "
            "Brightness/contrast jitter [0.85, 1.15]"
        ),
        "annot": "Training-Free\nNo GAN\nNo Diffusion",
        "color": LBL_TFIDG,
    },
    {
        "label": "3",
        "title": "Stage 3:  Deficit-Aware Placement Optimization",
        "sub": (
            "ROI candidates × seed variants scored for fitness\n"
            "Underrepresented (morphology, context) pairs prioritized"
        ),
        "annot": "placement_map\n.json",
        "color": LBL_TFIDG,
    },
    {
        "label": "4",
        "title": "Stage 4:  Poisson Blending Synthesis",
        "sub": (
            "Defect composited onto normal background → seamless boundary\n"
            "ROI mask co-saved for pixel-level supervision"
        ),
        "annot": "512×512\nsynthetic image\n+ ROI mask",
        "color": LBL_TFIDG,
    },
    {
        "label": "5",
        "title": "Stage 5:  Quality Gate",
        "sub": (
            "artifact_score + blur_score (31×31 dilated ROI mask)\n"
            "pruned: Q < threshold (rejected samples discarded)"
        ),
        "annot": "threshold Q ≥ 0.7\nN accepted",
        "color": LBL_QGATE,
    },
    {
        "label": "6",
        "title": "Stage 6:  Augmented Training Dataset",
        "sub": (
            "Baseline (real only)  +  Random ROI synthetic  +  AROMA ROI synthetic\n"
            "600 synthetic images per condition per dataset"
        ),
        "annot": "600 synthetic\nimages per\ncondition",
        "color": LBL_TFIDG,
    },
]

TOP_TITLE = "Multi-domain Industrial Datasets"
TOP_SUB = "isp_LSM_1 · mvtec_cable · visa_cashew · visa_pcb4"

BOTTOM_TITLE = "Downstream Anomaly Detection Model"
BOTTOM_SUB = "PatchCore · SuperSimpleNet · EfficientAD · ReverseDistillation++"


# --------------------------------------------------------------------------- #
# Layout constants (axis units, 0..100 horizontal / 0..100 vertical)
# --------------------------------------------------------------------------- #
FIG_W, FIG_H = 10, 16

# Horizontal geometry
STAGE_X = 6.0          # left edge of stage box
STAGE_W = 62.0         # stage box width
STAGE_CX = STAGE_X + STAGE_W / 2.0
ANNOT_X = 72.0         # left edge of right annotation box
ANNOT_W = 24.0
CIRCLE_X = STAGE_X + 4.5   # center-x of the left label circle

# Vertical geometry
TOP_Y = 95.0
BOTTOM_Y = 3.5
TERMINAL_H = 6.0
STAGE_H = 7.6
GAP = 2.4              # vertical gap used for arrows between boxes


# --------------------------------------------------------------------------- #
# Drawing helpers
# --------------------------------------------------------------------------- #
def _rounded_box(ax, x, y, w, h, facecolor, edgecolor, lw=1.4, pad=0.018, radius=0.6):
    """Draw a FancyBboxPatch rounded rectangle in axis-fraction-like data units."""
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad={pad},rounding_size={radius}",
        mutation_aspect=1.0,
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=lw,
        zorder=2,
    )
    ax.add_patch(box)
    return box


def _down_arrow(ax, x, y_top, y_bottom):
    """Thin gray downward arrow from y_top to y_bottom at horizontal x."""
    arrow = FancyArrowPatch(
        (x, y_top),
        (x, y_bottom),
        arrowstyle="-|>",
        mutation_scale=14,
        linewidth=1.6,
        color=COL_ARROW,
        shrinkA=0,
        shrinkB=0,
        zorder=1,
    )
    ax.add_patch(arrow)


def _label_circle(ax, cx, cy, text, color):
    """Colored circle on the left with white stage label text."""
    # circle radius in data units; aspect handled by equal-ish framing
    r = 1.9
    circ = Circle(
        (cx, cy),
        radius=r,
        facecolor=color,
        edgecolor="white",
        linewidth=1.6,
        zorder=4,
    )
    ax.add_patch(circ)
    ax.text(
        cx,
        cy,
        text,
        ha="center",
        va="center",
        fontsize=9.5,
        fontweight="bold",
        color="white",
        zorder=5,
    )


# --------------------------------------------------------------------------- #
# Main figure builder
# --------------------------------------------------------------------------- #
def build_figure():
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # ---- Top terminal box -------------------------------------------------- #
    _rounded_box(
        ax,
        STAGE_X,
        TOP_Y - TERMINAL_H,
        STAGE_W,
        TERMINAL_H,
        COL_TERMINAL_FILL,
        COL_TERMINAL_EDGE,
        lw=1.3,
    )
    ax.text(
        STAGE_CX,
        TOP_Y - TERMINAL_H / 2.0 + 1.0,
        TOP_TITLE,
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
        color="#333333",
        zorder=3,
    )
    ax.text(
        STAGE_CX,
        TOP_Y - TERMINAL_H / 2.0 - 1.6,
        TOP_SUB,
        ha="center",
        va="center",
        fontsize=8.5,
        color=COL_SUBTEXT,
        zorder=3,
    )

    # ---- Compute vertical positions for stages ----------------------------- #
    n = len(STAGES)
    # Available vertical span between top box bottom and bottom box top.
    span_top = TOP_Y - TERMINAL_H - GAP
    span_bottom = BOTTOM_Y + TERMINAL_H + GAP
    total_span = span_top - span_bottom
    # Each stage occupies a slot containing the box + sub-text + arrow gap.
    slot_h = total_span / n

    stage_centers = []
    for i in range(n):
        slot_top = span_top - i * slot_h
        box_top = slot_top
        box_bottom = box_top - STAGE_H
        box_cy = (box_top + box_bottom) / 2.0
        stage_centers.append((box_top, box_bottom, box_cy))

    # ---- Arrow from top box into Stage 0 ----------------------------------- #
    _down_arrow(ax, STAGE_CX, TOP_Y - TERMINAL_H, stage_centers[0][0])

    # ---- Draw each stage --------------------------------------------------- #
    for i, st in enumerate(STAGES):
        box_top, box_bottom, box_cy = stage_centers[i]

        # Stage box
        _rounded_box(
            ax,
            STAGE_X,
            box_bottom,
            STAGE_W,
            STAGE_H,
            COL_STAGE_FILL,
            COL_STAGE_EDGE,
            lw=1.4,
        )

        # Left label circle
        # _label_circle(ax, CIRCLE_X, box_cy, st["label"], st["color"])

        # Stage title (offset right of circle so it doesn't overlap)
        title_x = CIRCLE_X + 3.6
        ax.text(
            title_x,
            box_cy + 1.3,
            st["title"],
            ha="left",
            va="center",
            fontsize=11,
            fontweight="bold",
            color="#2A2A2A",
            zorder=3,
        )

        # Sub-annotation (gray italic) below the title, inside the box
        ax.text(
            title_x,
            box_cy - 1.9,
            st["sub"],
            ha="left",
            va="center",
            fontsize=8,
            style="italic",
            color=COL_SUBTEXT,
            linespacing=1.35,
            zorder=3,
        )

        # Right-side annotation box
        if st["annot"]:
            annot_h = STAGE_H + 0.4
            _rounded_box(
                ax,
                ANNOT_X,
                box_cy - annot_h / 2.0,
                ANNOT_W,
                annot_h,
                COL_ANNOT_FILL,
                COL_ANNOT_EDGE,
                lw=1.2,
                radius=0.5,
            )
            ax.text(
                ANNOT_X + ANNOT_W / 2.0,
                box_cy,
                st["annot"],
                ha="center",
                va="center",
                fontsize=8,
                color="#23613D",
                linespacing=1.3,
                zorder=3,
            )

        # Arrow to next stage (or to bottom box after last stage)
        if i < n - 1:
            next_top = stage_centers[i + 1][0]
            _down_arrow(ax, STAGE_CX, box_bottom, next_top)

    # ---- Arrow from last stage into bottom box ----------------------------- #
    last_bottom = stage_centers[-1][1]
    _down_arrow(ax, STAGE_CX, last_bottom, BOTTOM_Y + TERMINAL_H)

    # ---- Bottom terminal box ----------------------------------------------- #
    _rounded_box(
        ax,
        STAGE_X,
        BOTTOM_Y,
        STAGE_W,
        TERMINAL_H,
        COL_TERMINAL_FILL,
        COL_TERMINAL_EDGE,
        lw=1.3,
    )
    ax.text(
        STAGE_CX,
        BOTTOM_Y + TERMINAL_H / 2.0 + 1.0,
        BOTTOM_TITLE,
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
        color="#333333",
        zorder=3,
    )
    ax.text(
        STAGE_CX,
        BOTTOM_Y + TERMINAL_H / 2.0 - 1.6,
        BOTTOM_SUB,
        ha="center",
        va="center",
        fontsize=8.5,
        color=COL_SUBTEXT,
        zorder=3,
    )

    return fig


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main():
    out_dir = os.path.join(
        "D:\\", "project", "aroma", "AROMA연구분석", "Article", "figure"
    )
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "[figure2] aroma_pipeline.png")

    fig = build_figure()
    fig.savefig(
        out_path,
        dpi=150,
        bbox_inches="tight",
        transparent=False,
        facecolor="white",
    )
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
