#!/usr/bin/env python3
"""
Figure 7 — ROI placement qualitative comparison (Baseline vs AROMA vs Random).

One representative image per dataset, shown 3x with different bbox overlays:
  - Baseline: real ground-truth defect bbox (1 box)
  - AROMA:    5 candidate boxes placed by the real production scan-rank-place
              function (_positive_place), using real defect bbox sizes and the
              real per-cluster compatibility row.
  - Random:   the same 5 box sizes, placed by the real production uniform
              random placement function (_random_paste_position).

All box SIZES are real (drawn from roi_selected.json). Only the AROMA/Random
POSITIONS on this specific image are computed here, because the stored ROI
JSONs never persisted multiple candidate positions per image (position=null
for every entry, every dataset).
"""

import json
import random
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

SCRIPTS_DIR = Path("D:/project/aroma/scripts/aroma")
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(SCRIPTS_DIR.parent))  # for sibling distribution_profiling

import generate_defects as gd  # noqa: E402
import clean_bg_selection as cbs  # noqa: E402 -- for the real fit-rescale (_effective_wh)

BASE = Path("D:/project/aroma_dataset")

DATASETS = ["aitex", "kolektor", "severstal", "mtd", "mvtec_leather"]
N_CANDIDATES = 5
TARGET_AREA_FRAC = 0.40


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def colab_to_local_path(colab_path, dataset):
    mapping = {
        "aitex": ("/content/drive/MyDrive/data/Aroma/aitex_tiled", str(BASE / "aitex_tiled")),
        "kolektor": ("/content/drive/MyDrive/data/Aroma/kolektor", str(BASE / "kolektor")),
        "severstal": ("/content/drive/MyDrive/data/Aroma/severstal", str(BASE / "severstal")),
        "mtd": ("/content/drive/MyDrive/data/Aroma/mtd", str(BASE / "mtd")),
        "mvtec_leather": ("/content/drive/MyDrive/data/Aroma/mvtec/leather", str(BASE / "mvtec_leather")),
    }
    remote, local = mapping[dataset]
    return colab_path.replace(remote, local)


def load_image_rgb(path):
    img = cv2.imread(str(path))
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def pick_representative(roi_entries, img_dir_root, dataset, rng):
    """Pick one entry whose image loads successfully, area closest to the
    dataset median (a typical, not extreme, defect instance)."""
    areas = sorted(e["defect_bbox"][2] * e["defect_bbox"][3] for e in roi_entries)
    median_area = areas[len(areas) // 2]
    ranked = sorted(roi_entries, key=lambda e: abs(e["defect_bbox"][2] * e["defect_bbox"][3] - median_area))
    for e in ranked:
        local_path = colab_to_local_path(e["image_path"], dataset)
        if Path(local_path).exists():
            img = load_image_rgb(local_path)
            if img is not None:
                return e, img, local_path
    raise RuntimeError(f"[{dataset}] no loadable representative image found")


def pick_candidate_sizes(roi_entries, target_area, n=N_CANDIDATES):
    """Pick n real (width, height, cluster_id, subtype) tuples whose area is
    closest to target_area, preferring distinct subtypes."""
    scored = []
    for e in roi_entries:
        w, h = e["defect_bbox"][2], e["defect_bbox"][3]
        if w <= 0 or h <= 0:
            continue
        area = w * h
        scored.append((abs(area - target_area), w, h, e["cluster_id"], e.get("defect_subtype", "")))
    scored.sort(key=lambda t: t[0])

    chosen = []
    seen_subtypes = set()
    # Pass 1: best match per distinct subtype.
    for item in scored:
        _, w, h, cluster_id, subtype = item
        if subtype not in seen_subtypes:
            chosen.append((w, h, cluster_id, subtype))
            seen_subtypes.add(subtype)
        if len(chosen) == n:
            break
    # Pass 2: fill remaining slots with next-best matches regardless of subtype.
    if len(chosen) < n:
        for item in scored:
            _, w, h, cluster_id, subtype = item
            cand = (w, h, cluster_id, subtype)
            if cand in chosen:
                continue
            chosen.append(cand)
            if len(chosen) == n:
                break
    return chosen[:n]


def compat_row_for_cluster(compat_matrix, cluster_id):
    return compat_matrix["matrix_symmetric"].get(str(cluster_id), {})


def compute_aroma_boxes(nrgb, sizes, compat_matrix, rng):
    bin_edges = compat_matrix["bin_edges"]
    boxes = []
    h_img, w_img = nrgb.shape[:2]
    for (w, h, cluster_id, subtype) in sizes:
        # Real generation-time fit-rescale (mirrors generate_defects's own
        # 0.95-margin shrink at the crop>bg branch) so an oversized real bbox
        # never gets silently clipped -- it is rescaled, exactly like production.
        cw, ch = cbs._effective_wh((w, h), (w_img, h_img))
        compat_row = compat_row_for_cluster(compat_matrix, cluster_id)
        pos, mean_compat, n_nonvoid = gd._positive_place(
            nrgb, (cw, ch), compat_row, bin_edges,
            min_bg_quality=0.7, blur_threshold=100.0, rng=rng,
        )
        if pos is None:
            # Documented fallback (matches production behavior): no non-void
            # candidate exists for this size on this image -> uniform random.
            pos = gd._random_paste_position((w_img, h_img), (cw, ch), rng)
        x, y = pos
        boxes.append((x, y, cw, ch))
    return boxes


def compute_random_boxes(nrgb, sizes, rng):
    h_img, w_img = nrgb.shape[:2]
    boxes = []
    for (w, h, _cluster_id, _subtype) in sizes:
        cw, ch = cbs._effective_wh((w, h), (w_img, h_img))
        x, y = gd._random_paste_position((w_img, h_img), (cw, ch), rng)
        boxes.append((x, y, cw, ch))
    return boxes


def draw_boxes(ax, img, boxes, color, baseline_box=None):
    ax.imshow(img)
    for (x, y, w, h) in boxes:
        rect = plt.Rectangle((x, y), w, h, linewidth=2, edgecolor=color, facecolor="none")
        ax.add_patch(rect)
    if baseline_box is not None:
        x, y, w, h = baseline_box
        rect = plt.Rectangle((x, y), w, h, linewidth=2, edgecolor=color, facecolor="none")
        ax.add_patch(rect)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def main():
    col_titles = ["Baseline", "AROMA", "Random"]
    col_colors = ["red", "green", "blue"]

    rows_data = []
    for row, dataset in enumerate(DATASETS):
        print(f"\n=== {dataset} ===")
        roi_entries = load_json(BASE / "roi" / dataset / "roi_selected.json")
        compat_matrix = load_json(BASE / "profiling" / "profiling" / dataset / "compatibility_matrix.json")

        rng_pick = random.Random(42 + row)
        entry, img, local_path = pick_representative(roi_entries, BASE, dataset, rng_pick)
        h_img, w_img = img.shape[:2]
        image_area = w_img * h_img
        print(f"  representative: {local_path} ({w_img}x{h_img})")

        baseline_box = tuple(entry["defect_bbox"])  # (x, y, w, h)
        print(f"  baseline bbox: {baseline_box} "
              f"({100*baseline_box[2]*baseline_box[3]/image_area:.1f}% area)")

        # aitex keeps the original ~40%-of-image target (large, clearly visible
        # candidates on a small tile). The other four datasets replace that
        # target with the baseline defect's own area, per 2026-07-16 review --
        # at 40% the AROMA/Random candidate boxes were all similarly oversized
        # and their placement difference was hard to see.
        if dataset == "aitex":
            target_area = TARGET_AREA_FRAC * image_area
        else:
            target_area = baseline_box[2] * baseline_box[3]
        sizes = pick_candidate_sizes(roi_entries, target_area)
        for (w, h, cid, subtype) in sizes:
            print(f"    candidate size: {w}x{h} ({100*w*h/image_area:.1f}% area) "
                  f"cluster={cid} subtype={subtype}")

        rng_aroma = random.Random(1000 + row)
        rng_random = random.Random(2000 + row)
        aroma_boxes = compute_aroma_boxes(img, sizes, compat_matrix, rng_aroma)
        random_boxes = compute_random_boxes(img, sizes, rng_random)

        rows_data.append((dataset, img, baseline_box, aroma_boxes, random_boxes, w_img, h_img))

    # Every row's display cell uses the SAME width:height ratio (1:1, matching
    # mvtec_leather's own 1024x1024 aspect) regardless of the source image's
    # true aspect ratio -- a display-only convenience so the grid reads
    # uniformly; the underlying bbox data and pixel content are untouched
    # (imshow + aspect="auto" stretches the panel, box coordinates stay in the
    # image's own pixel space).
    col_width = 4.0
    fig_w = col_width * 3
    fig_h = col_width * len(rows_data) + len(rows_data) * 0.3
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = GridSpec(len(rows_data), 3, figure=fig, hspace=0.08, wspace=0.05)

    for row, (dataset, img, baseline_box, aroma_boxes, random_boxes, w_img, h_img) in enumerate(rows_data):
        panels = [
            (img, [], baseline_box),
            (img, aroma_boxes, None),
            (img, random_boxes, None),
        ]
        for col, (panel_img, boxes, bbox_single) in enumerate(panels):
            ax = fig.add_subplot(gs[row, col])
            draw_boxes(ax, panel_img, boxes, col_colors[col], baseline_box=bbox_single)
            ax.set_aspect("auto")
            if row == 0:
                ax.set_title(col_titles[col], fontsize=13, fontweight="bold")
            if col == 0:
                ax.set_ylabel(dataset, fontsize=11, fontweight="bold")
                ax.yaxis.set_visible(True)
                ax.set_yticks([])

    output_path = Path("D:/project/aroma/AROMA연구분석/Article/figure/image/[figure7] roi_bbox_qualitative.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\nSaved: {output_path}")
    plt.close()


if __name__ == "__main__":
    main()
