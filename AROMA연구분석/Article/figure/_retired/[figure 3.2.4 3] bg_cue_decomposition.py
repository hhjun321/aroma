# -*- coding: utf-8 -*-
"""Figure 3.2.4-3 — decomposition of the clean-background ranking into four cues.

Operational functions are imported rather than reimplemented:
  scripts/aroma/clean_bg_selection._hist_intersection / _image_hist /
                                   _class_bg_hist / _target_by_cluster /
                                   _scale_to_fit / _image_dim /
                                   _derive_void_floors / valid_bg_pool /
                                   load_inputs

Panels
  A  four cues over the valid background pool, ordered by the combined score U(g)
  B  stacked contribution w_j * u_j for the top candidates
  C  lift-derived weights per dataset, read from clean_bg_summary_ring.md
"""
import os
import re
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams["font.family"] = "DejaVu Sans"
rcParams["axes.unicode_minus"] = False

REPO = os.environ.get("AROMA_REPO", "D:/project/aroma")
ROOT = os.environ.get("AROMA_DATASET_ROOT", "D:/project/aroma_dataset")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "image")

sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.path.insert(0, os.path.join(REPO, "scripts", "aroma"))

import clean_bg_selection as CBS  # noqa: E402

DS = "severstal"
DATASETS = ["aitex", "kolektor", "severstal", "mtd", "mvtec_leather"]
CUES = [("src", "$u_{src}$  source overlap", "#2878c8"),
        ("cls", "$u_{cls}$  class-conditional", "#c85a28"),
        ("mor", "$u_{mor}$  morphology cluster", "#28a03c"),
        ("siz", "$u_{siz}$  size fit", "#9060c0")]
TOP_BARS = 12


def read_weights(ds):
    """clean_bg_summary_ring.md 의 실측 가중치. 재계산하지 않는다."""
    t = (Path(ROOT) / "roi" / ds / "clean_bg_summary_ring.md").read_text(encoding="utf-8")
    m = re.search(r"w_src=([\d.]+)\s+w_class=([\d.]+)\s+w_size=([\d.]+)\s+w_k=([\d.]+)", t)
    lk = re.search(r"lift_k=([\d.]+)", t)
    return (dict(src=float(m.group(1)), cls=float(m.group(2)),
                 siz=float(m.group(3)), mor=float(m.group(4))),
            float(lk.group(1)) if lk else float("nan"))


def score_pool(ds):
    """한 ROI 를 골라 valid 배경 풀 전체에 4 cue 를 매긴다."""
    prof = os.path.join(ROOT, "profiling", "profiling", ds)
    roi_dir = os.path.join(ROOT, "roi", ds)
    data = CBS.load_inputs(prof, roi_dir)
    if data["status"] != "ok":
        raise SystemExit("load_inputs failed: %s" % data)
    names, bin_edges = data["names"], data["bin_edges"]
    vf, ef = CBS._derive_void_floors(data["good_by_img"], 15.0)
    valid_ids, _reasons, _dv = CBS.valid_bg_pool(
        data["good_by_img"], True, None, vf, ef, floor_pct=15.0)
    tgt_by_k = CBS._target_by_cluster(data["compat"])
    w, lift_k = read_weights(ds)

    good_hist, good_dim = {}, {}
    for iid in valid_ids:
        rows = data["good_by_img"].get(iid, [])
        good_hist[iid] = CBS._image_hist(rows, names, bin_edges, vf, ef)
        good_dim[iid] = CBS._image_dim(rows)

    defect_by_img = {}
    for r in data["defect_rows"]:
        defect_by_img.setdefault(r.get("image_id", ""), []).append(r)
    src_hist = {i: CBS._image_hist(rs, names, bin_edges, vf, ef)
                for i, rs in defect_by_img.items()}
    class_hist = CBS._class_bg_hist(data["defect_rows"], names, bin_edges, vf, ef,
                                    data.get("iid_to_class", {}))

    sel = CBS.load_json(os.path.join(roi_dir, "clean_bg_selected_ring.json"))
    best_case = None
    for s in sel:
        k = str(s.get("cluster_id"))
        tgt = tgt_by_k.get(k)
        bbox = CBS._parse_bbox(s.get("defect_bbox"))
        sdv = src_hist.get(str(s.get("image_id", "")))
        if not tgt or not bbox or not sdv:
            continue
        cdv = class_hist.get(str(s.get("class_value") or ""), {})
        wh = (bbox[2], bbox[3])
        rows = []
        for g in valid_ids:
            u = dict(src=CBS._hist_intersection(good_hist[g], sdv),
                     cls=CBS._hist_intersection(good_hist[g], cdv),
                     mor=CBS._hist_intersection(good_hist[g], tgt),
                     siz=CBS._scale_to_fit(wh, good_dim[g]))
            rows.append((g, u, sum(w[j] * u[j] for j in w)))
        rows.sort(key=lambda t: -t[2])
        # u_mor 를 뺀 순위의 1위와 U(g) 1위가 다른 ROI = k_fit 이 순서를 바꾼 사례
        wo = max(rows, key=lambda t: sum(w[j] * t[1][j] for j in w if j != "mor"))
        if wo[0] == rows[0][0]:
            continue
        gap = rows[0][2] - wo[2]
        if best_case is None or gap > best_case[0]:
            best_case = (gap, s, rows)
    if best_case is None:
        raise SystemExit("no ROI where u_mor changed the top-1")
    return best_case[1], best_case[2], w, lift_k


def main():
    s, rows, w, _lift_k = score_pool(DS)
    assigned = s.get("assigned_normal_id")

    fig = plt.figure(figsize=(17, 6.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1.15, 1.0], wspace=0.26)

    # ---- A: four cues over the pool, ordered by U(g) ----
    axA = fig.add_subplot(gs[0, 0])
    xs = np.arange(1, len(rows) + 1)
    for key, lab, col in CUES:
        v = np.array([r[1][key] for r in rows], dtype=float)
        mx = v.max() if v.max() > 0 else 1.0
        axA.plot(xs, v / mx, lw=1.5, color=col, label=lab)
    star = [i for i, r in enumerate(rows) if r[0] == assigned]
    if star:
        axA.axvline(star[0] + 1, color="#c81e1e", ls=":", lw=1.4)
        axA.plot(star[0] + 1, 1.0, marker="*", ms=15, color="#c81e1e",
                 label="assigned background")
    axA.set_xlabel("valid background pool, ordered by combined $U(g)$", fontsize=10)
    axA.set_ylabel("cue value, normalised to its own maximum", fontsize=10)
    axA.set_title("A   Four cues over the pool of %d valid backgrounds" % len(rows),
                  fontsize=11)
    axA.legend(fontsize=8, frameon=False, loc="upper right")
    axA.grid(alpha=0.25)

    # ---- B: stacked w_j * u_j for the top candidates ----
    axB = fig.add_subplot(gs[0, 1])
    top = rows[:TOP_BARS]
    bx = np.arange(len(top))
    bottom = np.zeros(len(top))
    for key, lab, col in CUES:
        vals = np.array([w[key] * r[1][key] for r in top], dtype=float)
        axB.bar(bx, vals, bottom=bottom, color=col, width=0.76,
                label="$w_{%s}\\,u_{%s}$" % (key, key))
        bottom += vals
    axB.set_xticks(bx)
    axB.set_xticklabels([r[0] for r in top], rotation=64, ha="right", fontsize=7)
    axB.set_ylabel("$U(g)$  contribution", fontsize=10)
    axB.set_xlabel("candidate background", fontsize=10)
    axB.set_title("B   Weighted contributions for the top %d candidates" % len(top),
                  fontsize=11)
    axB.legend(fontsize=8, frameon=False, ncol=2)
    axB.grid(axis="y", alpha=0.25)

    # ---- C: lift-derived weights per dataset ----
    axC = fig.add_subplot(gs[0, 2])
    ys = np.arange(len(DATASETS))
    left = np.zeros(len(DATASETS))
    wt = [read_weights(d)[0] for d in DATASETS]
    for key, lab, col in CUES:
        vals = np.array([t[key] for t in wt], dtype=float)
        axC.barh(ys, vals, left=left, color=col, height=0.62)
        for y, (v, l) in enumerate(zip(vals, left)):
            if v >= 0.09:
                axC.text(l + v / 2.0, y, "%.2f" % v, ha="center", va="center",
                         fontsize=8, color="#ffffff")
        left += vals
    axC.set_yticks(ys)
    axC.set_yticklabels(DATASETS, fontsize=9)
    axC.set_xlim(0, 1.0)
    axC.set_xlabel("lift-derived weight $w_j$   (sums to 1)", fontsize=10)
    axC.set_title("C   Weights the lift derivation yields per dataset", fontsize=11)
    axC.grid(axis="x", alpha=0.25)
    axC.invert_yaxis()

    fig.suptitle("Clean-background ranking decomposed — %s, ROI %s (cluster k=%s), "
                 "assigned %s" % (DS, s.get("roi_idx"), s.get("cluster_id"), assigned),
                 fontsize=12, y=1.0)
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, "[figure 3.2.4 3] bg_cue_decomposition.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print("saved:", out)
    print("  case: roi_idx=%s image_id=%s assigned=%s pool=%d"
          % (s.get("roi_idx"), s.get("image_id"), assigned, len(rows)))
    print("  weights:", {k: round(v, 4) for k, v in w.items()})


if __name__ == "__main__":
    main()
