#!/usr/bin/env python
"""
Figure 3 — Policy Selection Results

§3.2.2를 시각화: 5개 dataset별 morphology/context axis의 candidate policy
평가 점수(silhouette) + 선택된 정책(★ 강조).

모든 수치는 aroma_dataset/complexity/<dataset>/complexity_report.json에서 직접 로드.
spec: figure3_policy_selection_spec.md
"""
import json
import os

import matplotlib.pyplot as plt
import numpy as np

# --- paths ---
DATA_ROOT = r"D:\project\aroma_dataset\complexity"
OUT_DIR = r"D:\project\aroma\AROMA연구분석\Article\figure\image"
OUT_PATH = os.path.join(OUT_DIR, "[figure3] policy_selection.png")

# datasets in order
DATASETS = [
    ("aitex", "AITeX"),
    ("kolektor", "Kolektor"),
    ("severstal", "Severstal"),
    ("mtd", "MTD"),
    ("mvtec_leather", "MVTec Leather"),
]

# colors: morphology (blue), context (green)
COLOR_MORPH = {"otsu": "#1f77b4", "gmm": "#0055aa"}
COLOR_CTX = {"gmm": "#2ca02c", "percentile": "#17aa34"}


def load_report(key):
    path = os.path.join(DATA_ROOT, key, "complexity_report.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.linewidth": 0.8,
    })

    fig, ax = plt.subplots(figsize=(10, 6.5), dpi=300)

    y_pos = 0
    y_labels = []
    y_ticks = []
    bar_height = 0.35

    for key, name in DATASETS:
        rep = load_report(key)
        morph_selected = rep["morphology_policy"]
        ctx_selected = rep["context_policy"]

        # extract evaluation results
        morph_results = {}  # policy -> score
        ctx_results = {}
        for result in rep["evaluation_results"]:
            pol = result["policy"]
            score = result["silhouette"]
            axis = result["axis"]
            if axis == "morphology":
                morph_results[pol] = score
            elif axis == "context":
                ctx_results[pol] = score

        # plot morphology bars (left group)
        morph_x = y_pos - 0.35
        morph_policies = sorted(morph_results.keys(), reverse=True)  # gmm, otsu
        for i, pol in enumerate(morph_policies):
            score = morph_results[pol]
            x_offset = morph_x + i * bar_height * 0.6
            color = COLOR_MORPH.get(pol, "#666")
            alpha = 1.0 if pol == morph_selected else 0.6
            bar = ax.barh(x_offset, score, height=0.25, color=color, alpha=alpha,
                         edgecolor="black", linewidth=1.0 if pol == morph_selected else 0.5)
            # add ★ if selected
            if pol == morph_selected:
                ax.text(score + 0.01, x_offset, "★", fontsize=14, va="center", color="gold")

        # plot context bars (right group)
        ctx_x = y_pos + 0.15
        ctx_policies = ["gmm", "percentile"]  # consistent order
        for i, pol in enumerate(ctx_policies):
            if pol not in ctx_results:
                continue
            score = ctx_results[pol]
            x_offset = ctx_x + i * bar_height * 0.6
            color = COLOR_CTX.get(pol, "#666")
            alpha = 1.0 if pol == ctx_selected else 0.6
            bar = ax.barh(x_offset, score, height=0.25, color=color, alpha=alpha,
                         edgecolor="black", linewidth=1.0 if pol == ctx_selected else 0.5)
            # add ★ if selected
            if pol == ctx_selected:
                ax.text(score + 0.01, x_offset, "★", fontsize=14, va="center", color="gold")

        y_labels.append(name)
        y_ticks.append(y_pos)
        y_pos += 1

    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels, fontsize=11)
    ax.set_xlabel("Silhouette Score", fontsize=11)
    ax.set_xlim(-0.08, 0.52)
    ax.axvline(x=0, color="black", linestyle="-", linewidth=0.8, alpha=0.3)
    ax.grid(axis="x", linestyle=":", linewidth=0.6, alpha=0.4)

    # legend
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor="#1f77b4", edgecolor="black", label="Morphology: Otsu"),
        Patch(facecolor="#0055aa", edgecolor="black", label="Morphology: GMM"),
        Patch(facecolor="#2ca02c", edgecolor="black", label="Context: GMM"),
        Patch(facecolor="#17aa34", edgecolor="black", label="Context: Percentile"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", frameon=True,
             framealpha=0.9, fontsize=9, title="★ = selected policy",
             title_fontsize=9)

    # layout
    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=300, bbox_inches="tight")
    print("saved:", OUT_PATH)


if __name__ == "__main__":
    main()
