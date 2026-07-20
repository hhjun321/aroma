#!/usr/bin/env python3
"""
Figures 8-12 — ROI Selection Comparison (improved / v2)

Datasets -> figure numbers:
    aitex -> 8, kolektor -> 9, severstal -> 10, mtd -> 11, mvtec_leather -> 12

For each dataset, select 2 representative defects that MAXIMIZE the AROMA-vs-Random
contrast, i.e. defects where AROMA's best placement scores high while Random's worst
placement scores low. For each selected defect, show 3 columns sharing the SAME
defect (image_id):

    - Baseline : original defect image, defect bbox from ground-truth mask
    - AROMA    : AROMA synthetic with the HIGHEST roi_score + its placement bbox
    - Random   : Random synthetic with the LOWEST  roi_score + its placement bbox

Column labels are just "Baseline"/"AROMA"/"Random" (no score text).
The defect bbox is expanded around its center to an eye-visible size.

Run locally:  python generate_figures8_9_roi_comparison_v2.py
"""

import json
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle

SYNTH_ROOT = Path("D:/aroma_dataset")          # synth_aroma / synth_random live here
OUT_DIR = Path("D:/project/aroma/AROMA연구분석/Article/figure/image")

BBOX_COLOR = "#ff2d2d"    # red — visible on grayscale textile/metal
BBOX_LW = 3.5
BBOX_MIN_FRAC = 0.10      # min displayed box side as fraction of min(H, W)
BBOX_MIN_ABS = 56         # absolute floor (px)

# datasets that keep true image aspect (no cell-fill stretch); others fill uniformly
ASPECT_PRESERVE = {"severstal"}

# dataset -> (fig_num, base_root, test_subdir, mask_subdir, mask_suffix, mask_ext)
DATASETS = {
    "aitex":         (8,  "D:/aroma_dataset",         "aitex_tiled/test", "aitex_tiled/ground_truth", "_mask", ".png"),
    "kolektor":      (9,  "D:/aroma_dataset",         "kolektor/test",    "kolektor/ground_truth",    "_mask", ".png"),
    "severstal":     (10, "D:/aroma_dataset",         "severstal/test",   "severstal/masks",          "",      ".png"),
    "mtd":           (11, "D:/project/aroma_dataset", "mtd/test",         "mtd/ground_truth",         "_mask", ".png"),
    "mvtec_leather": (12, "D:/project/aroma_dataset", "leather/test",     "leather/ground_truth",     "_mask", ".png"),
}


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_rgb(path):
    img = cv2.imread(str(path))
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def mask_bbox(mask_path):
    m = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        return None
    ys, xs = np.where(m > 0)
    if xs.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)


def visible_bbox(bbox, img_shape):
    """Expand bbox around its center to an eye-visible size, clamped to bounds."""
    x, y, w, h = bbox
    H, W = img_shape[:2]
    side = max(BBOX_MIN_ABS, int(BBOX_MIN_FRAC * min(H, W)))
    cx, cy = x + w / 2.0, y + h / 2.0
    nw, nh = max(w, side), max(h, side)
    nw, nh = min(nw, W), min(nh, H)
    nx = min(max(cx - nw / 2.0, 0), W - nw)
    ny = min(max(cy - nh / 2.0, 0), H - nh)
    return nx, ny, nw, nh


def draw_panel(ax, img, bbox, title, fill=True):
    if img is not None:
        # fill=True: stretch to fill the cell uniformly; fill=False: preserve true aspect
        ax.imshow(img, aspect=("auto" if fill else "equal"))
        if bbox is not None:
            x, y, w, h = visible_bbox(bbox, img.shape)
            ax.add_patch(Rectangle((x, y), w, h, fill=False,
                                   edgecolor=BBOX_COLOR, linewidth=BBOX_LW))
    else:
        ax.text(0.5, 0.5, "N/A", ha="center", va="center", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.axis("off")


def group_by_id(annot):
    d = {}
    for e in annot:
        d.setdefault(e["image_id"], []).append(e)
    return d


def select_contrast_pairs(aroma_by_id, random_by_id, n=2):
    common = set(aroma_by_id) & set(random_by_id)
    ranked = []
    for iid in common:
        a_best = max(aroma_by_id[iid], key=lambda e: e["roi_score"])
        r_worst = min(random_by_id[iid], key=lambda e: e["roi_score"])
        ranked.append((a_best["roi_score"] - r_worst["roi_score"], iid, a_best, r_worst))
    ranked.sort(key=lambda t: t[0], reverse=True)
    return ranked[:n]


def baseline_from_source_roi(source_roi, cfg):
    """Resolve (rgb, bbox) for the original defect image using its source_roi path."""
    _, base, test_sub, mask_sub, suffix, ext = cfg
    if "/test/" not in source_roi:
        return None, None
    rel = source_roi.split("/test/", 1)[1]          # e.g. "class1/27f44281b.jpg"
    base = Path(base)
    tile = base / test_sub / rel
    cls, stem = Path(rel).parent, Path(rel).stem
    mask = base / mask_sub / cls / f"{stem}{suffix}{ext}"
    img = load_rgb(tile) if tile.exists() else None
    bbox = mask_bbox(mask) if mask.exists() else None
    return img, bbox


def synth_panel(entry, synth_dir):
    img = load_rgb(synth_dir / Path(entry["image_path"]).name)
    bbox = tuple(entry["bbox"]) if entry.get("bbox") else None
    return img, bbox


def build_figure(dataset, cfg):
    fig_num = cfg[0]
    print(f"\n=== {dataset} (Figure {fig_num}) ===")
    aroma_dir = SYNTH_ROOT / "synth_aroma" / dataset / "images"
    random_dir = SYNTH_ROOT / "synth_random" / dataset / "images"
    aroma_annot = load_json(aroma_dir.parent / "annotations.json")
    random_annot = load_json(random_dir.parent / "annotations.json")

    aroma_by_id = group_by_id(aroma_annot)
    random_by_id = group_by_id(random_annot)

    pairs = select_contrast_pairs(aroma_by_id, random_by_id, n=2)
    if not pairs:
        print("  No common defects — skipped.")
        return False

    # Pass 1: gather panels so figure height can match the image aspect ratio
    # (steel strips are very wide; square cells would leave large whitespace).
    rows_data = []
    aspects = []
    for contrast, iid, a_best, r_worst in pairs:
        base_img, base_bbox = baseline_from_source_roi(a_best["source_roi"], cfg)
        a_img, a_bbox = synth_panel(a_best, aroma_dir)
        r_img, r_bbox = synth_panel(r_worst, random_dir)
        print(f"  Row {len(rows_data)+1}: {iid} | contrast={contrast:.3f} "
              f"(AROMA {a_best['roi_score']:.3f} vs Random {r_worst['roi_score']:.3f})")
        if base_img is None:
            print(f"    WARN: baseline unavailable ({a_best['source_roi']})")
        for im in (base_img, a_img, r_img):
            if im is not None:
                aspects.append(im.shape[0] / im.shape[1])
        rows_data.append([(base_img, base_bbox, "Baseline"),
                          (a_img, a_bbox, "AROMA"),
                          (r_img, r_bbox, "Random")])

    ok = any(c[0] is not None for r in rows_data for c in r)
    aspect = float(np.clip(np.median(aspects) if aspects else 1.0, 0.15, 2.0))
    col_w_in = 15.0 / 3.0
    fig_h = col_w_in * aspect * len(pairs) + 1.3   # + title / label headroom

    fill = dataset not in ASPECT_PRESERVE
    fig = plt.figure(figsize=(15, fig_h))
    gs = GridSpec(len(pairs), 3, figure=fig, hspace=0.3, wspace=0.05)
    for row, cols in enumerate(rows_data):
        for col, (img, bbox, title) in enumerate(cols):
            draw_panel(fig.add_subplot(gs[row, col]), img, bbox, title, fill=fill)

    out = OUT_DIR / f"[figure{fig_num}] {dataset}_roi_comparison.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out}")
    return ok


def main():
    for dataset, cfg in DATASETS.items():
        base = Path(cfg[1]) / cfg[2]
        if not base.exists():
            print(f"\n=== {dataset} (Figure {cfg[0]}) SKIPPED: baseline originals not found at {base} ===")
            continue
        build_figure(dataset, cfg)


if __name__ == "__main__":
    main()
