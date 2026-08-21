# -*- coding: utf-8 -*-
"""Figure 4.5-1 — Top-K selection retention across the ROI-score weight simplex.

데이터: roi_weight_sensitivity 실측 (66-point simplex sweep, 5 datasets).
표현: 데이터셋별 패널, (w_ctx, w_morph) 평면 산점 히트맵 (w_quality = 1 − ctx − morph),
      색 = adopted (0.5/0.3/0.2) 대비 top-K retention. adopted 지점 ★ 마킹.
메시지: interior 전역 ~100%, 성분 소거 경계(축·빗변)에서만 하락 — §4.5 Table 12 보완.
R3-4 legibility: tick ≥8pt, label ≥10pt, colorbar 라벨 명시.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import json, io

SRC = "D:/project/AROMA_DATASET/roi_weight_sensitivity/severstal/sensitivity_results.json"
OUT = "D:/project/aroma/AROMA연구분석/Article/figure/image/[figure 4.5 1] weight_simplex_retention.png"
DS_ORDER = ["severstal", "mtd", "mvtec_leather", "aitex", "kolektor"]
DS_LABEL = {"severstal": "Severstal", "mtd": "MTD", "mvtec_leather": "MVTec Leather",
            "aitex": "AITeX", "kolektor": "Kolektor"}
ADOPTED = (0.5, 0.3)

data = json.load(io.open(SRC, encoding="utf-8"))["datasets"]

fig, axes = plt.subplots(1, 5, figsize=(19, 4.4), sharey=True)
sc = None
for ax, ds in zip(axes, DS_ORDER):
    d = data[ds]
    top_k = d["top_k"]
    # 0.1 grid 점만 사용 (추가 기준점 0.6/0.4/0.0, 1/3 등은 grid 밖 → 제외해 균일 격자 유지)
    pts = [(r["w_ctx"], r["w_morph"], r["overlap"] / top_k) for r in d["sweep"]
           if abs(r["w_ctx"] * 10 - round(r["w_ctx"] * 10)) < 1e-9
           and abs(r["w_morph"] * 10 - round(r["w_morph"] * 10)) < 1e-9]
    x, y, v = zip(*pts)
    sc = ax.scatter(x, y, c=[vi * 100 for vi in v], cmap="viridis", vmin=30, vmax=100,
                    s=150, marker="s", edgecolors="none")
    ax.plot(*ADOPTED, marker="*", markersize=17, color="red", markeredgecolor="white",
            markeredgewidth=0.8, linestyle="none")
    ax.set_title(DS_LABEL[ds], fontsize=13)
    ax.set_xlabel("w_ctx", fontsize=11)
    ax.set_xlim(-0.06, 1.06)
    ax.set_ylim(-0.06, 1.06)
    ax.tick_params(labelsize=9)
    ax.set_aspect("equal")
axes[0].set_ylabel("w_morph", fontsize=11)

cbar = fig.colorbar(sc, ax=axes, fraction=0.015, pad=0.01)
cbar.set_label("top-K retention vs. adopted weights (%)", fontsize=11)
cbar.ax.tick_params(labelsize=9)

fig.suptitle("ROI-score weight simplex — selection retention "
             "(w_quality = 1 − w_ctx − w_morph;  ★ = adopted 0.5/0.3/0.2)",
             fontsize=14, y=1.02)
fig.savefig(OUT, dpi=300, bbox_inches="tight")
print("saved:", OUT)
