# -*- coding: utf-8 -*-
"""Figure 3.2.4-3 — ROI_score composition: 0.5*ctx_prior + 0.3*morph_prior + 0.2*quality.

Eq.2 3-성분 개정(2026-08-21, report1 대응)에 맞춰 재작성:
  - 후보 = roi_candidates.json 실제 풀 (구판은 compatibility_matrix에서 (cluster,cell)
    쌍을 합성 — quality가 per-crop이라 실제 후보 필요)
  - quality = percentile subtype 재구성 (로컬 미러가 fixed-era라 저장값 사용 금지 —
    dev_note aroma_report1_responses_20260821.md §2 참조; 재구성 경로는
    subtype_threshold_perturbation.py에서 S0/S1 검증 통과)
  - R3-4 legibility: tick/annotation ≥8pt, label ≥10pt

Panels: severstal, aitex. See [figure 3.2.4 3] roi_score_composition.md.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import json, io, os, sys, csv

ROOT = os.environ.get("AROMA_DATASET_ROOT", "D:/project/AROMA_DATASET")
OUT  = "D:/project/aroma/AROMA연구분석/Article/figure/image/[figure 3.2.4 3] roi_score_composition.png"
TOPN = 12
W_CTX, W_MORPH, W_Q = 0.5, 0.3, 0.2
BLUE  = "#2c6fbb"   # 0.5 * ctx_prior
GREY  = "#b0b0b0"   # 0.3 * morph_prior
GREEN = "#4a9d5f"   # 0.2 * quality

sys.path.insert(0, "D:/project/aroma/scripts")
from aroma.roi_selection import _subtype_percentiles, quality_proxy  # noqa: E402


def top_candidates(ds):
    cand = json.load(io.open(f"{ROOT}/roi/{ds}/roi_candidates.json", encoding="utf-8"))
    with io.open(f"{ROOT}/profiling/profiling/{ds}/morphology_features.csv",
                 encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    th = _subtype_percentiles(rows)
    lut = {r["image_id"]: (r.get("linearity"), r.get("solidity"), r.get("aspect_ratio"))
           for r in rows}
    q_img = {iid: quality_proxy(*v, "directional", thresholds=th)[1]
             for iid, v in lut.items()}
    scored = []
    for c in cand:
        ctx = max(0.0, min(1.0, float(c["ctx_prior"])))
        mp  = max(0.0, min(1.0, float(c["morph_prior"])))
        q   = max(0.0, min(1.0, q_img[str(c["image_id"])]))
        s   = W_CTX * ctx + W_MORPH * mp + W_Q * q
        scored.append((s, ctx, mp, q, c["cluster_id"]))
    scored.sort(reverse=True)
    return scored[:TOPN]


fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.4), sharex=True)
fig.suptitle("ROI_score = 0.5 · ctx_prior + 0.3 · morph_prior + 0.2 · quality",
             fontsize=15, y=1.00)

for ax, ds in zip(axes, ["severstal", "aitex"]):
    cand = top_candidates(ds)
    y = np.arange(len(cand))                # index 0 = best; invert_yaxis puts it on top
    ctx_term = [W_CTX * c[1] for c in cand]
    mor_term = [W_MORPH * c[2] for c in cand]
    q_term   = [W_Q * c[3] for c in cand]
    left2 = [a + b for a, b in zip(ctx_term, mor_term)]
    ax.barh(y, ctx_term, color=BLUE, label="0.5 · ctx_prior")
    ax.barh(y, mor_term, left=ctx_term, color=GREY, label="0.3 · morph_prior")
    ax.barh(y, q_term, left=left2, color=GREEN, label="0.2 · quality")
    for yi, c in zip(y, cand):
        ax.text(c[0] + 0.010, yi, f"{c[0]:.3f}", va="center", fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels([f"k{c[4]}" for c in cand], fontsize=9)
    ax.tick_params(axis="x", labelsize=9)
    ax.set_xlim(0, 1.02)
    ax.set_xlabel("score contribution", fontsize=11)
    ax.set_title(ds, fontsize=13)
    ax.invert_yaxis()                       # best (index 0) at top
    n_peak = sum(1 for c in cand if c[1] >= 0.9999)
    if n_peak:
        ax.text(0.02, n_peak - 0.55,
                "cluster peaks (ctx_prior = 1.0): ordered by the remaining terms",
                fontsize=8, color="#444444", style="italic",
                bbox=dict(facecolor="white", alpha=0.85, edgecolor="none", pad=1.5))

axes[0].legend(loc="lower right", fontsize=9, framealpha=0.95)
plt.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(OUT, dpi=300, bbox_inches="tight")
print("saved:", OUT)
