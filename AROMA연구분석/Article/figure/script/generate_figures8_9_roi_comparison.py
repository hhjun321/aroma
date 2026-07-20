#!/usr/bin/env python3
"""
Visualize ROI selection comparison: Baseline vs AROMA vs Random

For each dataset (AITeX, Kolektor):
  - Select 2 representative defect images:
    1. Image with most similar roi_score (confidence) across AROMA vs Random
    2. Image with largest roi_score difference between AROMA and Random
  - For each image, display 3 columns:
    - Baseline: original defect image
    - AROMA: synthetic with AROMA-selected background context
    - Random: synthetic with Random-selected background context
"""

import json
import os
from pathlib import Path
from collections import defaultdict
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def colab_to_local_path(colab_path, dataset):
    """Convert Colab path to local aroma_dataset path."""
    if "aitex_tiled" in colab_path:
        return colab_path.replace(
            "/content/drive/MyDrive/data/Aroma/aitex_tiled",
            str(Path("D:/project/aroma_dataset/aitex"))
        )
    elif "synth_aroma" in colab_path:
        return colab_path.replace(
            "/content/drive/MyDrive/data/Aroma/aroma_output/sym_final/synth_aroma",
            str(Path(f"D:/project/aroma_dataset/synth_aroma"))
        )
    elif "synth_random" in colab_path:
        return colab_path.replace(
            "/content/drive/MyDrive/data/Aroma/aroma_output/sym_final/synth_random",
            str(Path(f"D:/project/aroma_dataset/synth_random"))
        )
    elif "kolektor" in colab_path and "aroma_output" not in colab_path:
        # Original Kolektor path
        return colab_path.replace(
            "/content/drive/MyDrive/data/Aroma/kolektor",
            str(Path("D:/project/aroma_dataset/kolektor"))
        )
    return colab_path


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_image(path):
    """Load image and return RGB."""
    img = cv2.imread(str(path))
    if img is None:
        print(f"Warning: Failed to load {path}")
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def select_representative_images(roi_aroma, roi_random, synth_aroma_annot, synth_random_annot, dataset):
    """
    Select 2 representative defects based on AROMA roi_score:
    Only from defects present in BOTH AROMA and Random annotations.

    Returns: [defect_id_1, defect_id_2]
    """
    # Get IDs that have both synth_aroma and synth_random
    aroma_ids = {e["image_id"] for e in synth_aroma_annot}
    random_ids = {e["image_id"] for e in synth_random_annot}
    common_ids = aroma_ids & random_ids
    print(f"  Common defects: {len(common_ids)} / {len(aroma_ids)} (AROMA)")

    # Filter roi_aroma to common IDs
    common_aroma = [e for e in roi_aroma if e["image_id"] in common_ids]
    # Sort by roi_score
    sorted_aroma = sorted(common_aroma, key=lambda x: x["roi_score"])

    selected = []

    # Lowest roi_score (most similar to baseline quality)
    if sorted_aroma:
        entry = sorted_aroma[0]
        selected.append(entry["image_id"])
        print(f"  [Low score] {entry['image_id']}: "
              f"roi_score={entry['roi_score']:.4f} (most similar to baseline)")

    # Highest roi_score (most selective/confident)
    if sorted_aroma:
        entry = sorted_aroma[-1]
        selected.append(entry["image_id"])
        print(f"  [High score] {entry['image_id']}: "
              f"roi_score={entry['roi_score']:.4f} (most selective AROMA choice)")

    return selected[:2]


def get_synth_image_for_defect(defect_image_id, synth_annot, synth_dir, roi_score_to_match=None):
    """
    Find a synthetic image for the given defect_image_id.
    If roi_score_to_match is provided, find the closest match.
    """
    matching = [e for e in synth_annot if e["image_id"] == defect_image_id]
    if not matching:
        return None

    if roi_score_to_match is None:
        # Just return first one
        entry = matching[0]
    else:
        # Find closest roi_score
        entry = min(matching, key=lambda e: abs(e["roi_score"] - roi_score_to_match))

    img_filename = Path(entry["image_path"]).name
    img_path = synth_dir / img_filename

    img = load_image(img_path)
    return img, entry


def create_comparison_figure(dataset, roi_aroma_path, roi_random_path,
                           test_image_dir, synth_aroma_dir, synth_random_dir,
                           output_path):
    """Create comparison figure for one dataset."""
    print(f"\nProcessing {dataset}...")

    # Load ROI selections
    roi_aroma = load_json(roi_aroma_path)
    roi_random = load_json(roi_random_path)

    print(f"  Loaded {len(roi_aroma)} AROMA selections, {len(roi_random)} Random selections")

    # Load synth annotations
    synth_aroma_annot = load_json(synth_aroma_dir.parent / "annotations.json")
    synth_random_annot = load_json(synth_random_dir.parent / "annotations.json")

    # Select representative images
    selected_ids = select_representative_images(roi_aroma, roi_random, synth_aroma_annot, synth_random_annot, dataset)

    if not selected_ids:
        print(f"  No representative images found for {dataset}")
        return

    if not selected_ids:
        print(f"  No valid defects found")
        return

    # Create figure with N rows (selected defects) × 3 columns (Baseline, AROMA, Random)
    fig = plt.figure(figsize=(15, 6 * len(selected_ids)))
    gs = GridSpec(len(selected_ids), 3, figure=fig, hspace=0.3, wspace=0.05)

    for row, defect_id in enumerate(selected_ids):
        print(f"\n  Row {row+1}: {defect_id}")

        # Find baseline (original defect) image
        # image_id format:
        #   AITeX: "defect_XXXX_YYY_ZZ__tile_rX_cY" → file: XXXX_YYY_ZZ.png in test/YYY/
        #   Kolektor: "defect_kosXX_PartY" → file: kosXX_PartY.jpg in test/defect/

        if dataset == "aitex":
            parts = defect_id.replace("defect_", "").split("__")[0]  # Remove "defect_" and tile suffix
        else:  # kolektor
            parts = defect_id.replace("defect_", "")  # Direct filename (minus "defect_" prefix)

        # Try to find in test directory recursively
        baseline_candidates = list(test_image_dir.rglob(f"{parts}.png")) + list(test_image_dir.rglob(f"{parts}.jpg"))
        if baseline_candidates:
            baseline_img = load_image(baseline_candidates[0])
            print(f"    Found baseline: {baseline_candidates[0].name}")
        else:
            print(f"    Warning: baseline image not found for {defect_id} (searched for {parts}.*)")
            baseline_img = None

        # Find AROMA synthetic
        aroma_roi_entry = next((e for e in roi_aroma if e["image_id"] == defect_id), None)
        if aroma_roi_entry:
            aroma_result = get_synth_image_for_defect(
                defect_id, synth_aroma_annot, synth_aroma_dir,
                roi_score_to_match=aroma_roi_entry["roi_score"]
            )
            aroma_img = aroma_result[0] if aroma_result else None
        else:
            aroma_img = None

        # Find Random synthetic (no roi_score in clean_bg_random_arm.json, just use first)
        random_result = get_synth_image_for_defect(
            defect_id, synth_random_annot, synth_random_dir
        )
        random_img = random_result[0] if random_result else None

        # Plot images
        images = [baseline_img, aroma_img, random_img]
        titles = ["Baseline", "AROMA", "Random"]

        for col, (img, title) in enumerate(zip(images, titles)):
            ax = fig.add_subplot(gs[row, col])
            if img is not None:
                ax.imshow(img)
            else:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center", fontsize=12)

            ax.set_title(title, fontsize=12, fontweight="bold")
            ax.axis("off")

    plt.suptitle(f"{dataset.upper()} — ROI Selection Comparison (Baseline vs AROMA vs Random)",
                 fontsize=14, fontweight="bold")

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\n  Saved to {output_path}")
    plt.close()


def main():
    base_dir = Path("D:/project/aroma_dataset")

    for dataset in ["aitex", "kolektor"]:
        roi_dir = base_dir / "roi" / dataset

        # Different test directory structure per dataset
        if dataset == "aitex":
            test_dir = base_dir / dataset / "test"  # aitex: test/{YYY}/ structure
        else:
            test_dir = base_dir / dataset / "test" / "defect"  # kolektor: test/defect/ flat

        synth_aroma_dir = base_dir / "synth_aroma" / dataset / "images"
        synth_random_dir = base_dir / "synth_random" / dataset / "images"

        output_path = Path(f"D:/project/aroma/AROMA연구분석/Article/figure/image/[figure {'4.2 2' if dataset == 'kolektor' else '4.2 1'}] {dataset}_roi_comparison.png")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        create_comparison_figure(
            dataset,
            roi_aroma_path=roi_dir / "roi_selected.json",
            roi_random_path=roi_dir / "clean_bg_random_arm.json",
            test_image_dir=test_dir,
            synth_aroma_dir=synth_aroma_dir,
            synth_random_dir=synth_random_dir,
            output_path=output_path
        )


if __name__ == "__main__":
    main()
