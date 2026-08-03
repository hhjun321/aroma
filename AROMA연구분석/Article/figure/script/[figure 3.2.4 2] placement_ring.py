# -*- coding: utf-8 -*-
"""Figure 3.2.4-2 — ring distribution matching turns q_k into a paste coordinate.

Replaces `[figure 3.2.4 2] placement_footprint.py`, whose subject (footprint
mean-compat, top-K sampling, tau gate) no longer exists in the adopted rule.

Operational functions are imported rather than reimplemented:
  scripts/aroma/clean_bg_selection._target_by_cluster / _tile_grid /
                                   _ring_keys / _derive_void_floors /
                                   _load_discretizer / load_json

Panels
  A  64px tile grid coloured by q_k(cell(t)); void tiles hatched
  B  three candidates: footprint F(s) solid vs ring R(s) dashed; rejected marked
  C  ring histogram h_s vs target q_k for best / worst admissible candidate
  D  score over all admissible candidates, sorted, with argmax
"""
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.patches import Rectangle
from PIL import Image

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
CLUSTER = 1                # elongated; same cluster the superseded figure used
TILE = CBS.__dict__.get("_COMPAT_TILE", 64) if False else 64
TOP_CELLS = 12             # panel C bar count


def load_all():
    prof = os.path.join(ROOT, "profiling", "profiling", DS)
    roi = os.path.join(ROOT, "roi", DS)
    compat = CBS.load_json(os.path.join(prof, "compatibility_matrix.json"))
    names, bin_edges = CBS._load_discretizer(compat)
    tgt = CBS._target_by_cluster(compat)[str(CLUSTER)]

    good_by_img = {}
    for r in CBS._read_csv_rows(Path(prof) / "context_features.csv"):
        if r.get("image_type") == "good":
            good_by_img.setdefault(r.get("image_id", ""), []).append(r)
    var_floor, edge_floor = CBS._derive_void_floors(good_by_img, 15.0)

    sel = CBS.load_json(os.path.join(roi, "clean_bg_selected_ring.json"))
    return names, bin_edges, tgt, good_by_img, var_floor, edge_floor, sel


def all_scores(grid, gw, gh, bw, bh, tgt):
    """_best_ring_site 와 동일한 순회로 (score, xy, rejected) 전부 수집."""
    adm, rej = [], 0
    for sj in range(gh - bh + 1):
        for si in range(gw - bw + 1):
            if any((si + a, sj + b) not in grid
                   for a in range(bw) for b in range(bh)):
                rej += 1
                continue
            ring = [grid[t] for t in CBS._ring_keys(si, sj, bw, bh) if t in grid]
            if not ring:
                rej += 1
                continue
            inv = 1.0 / len(ring)
            h = {}
            for c in ring:
                h[c] = h.get(c, 0.0) + inv
            sc = sum(min(v, tgt[c]) for c, v in h.items() if c in tgt)
            adm.append((sc, (si, sj), h, len(ring)))
    adm.sort(key=lambda t: -t[0])
    return adm, rej


def pick_case(sel, good_by_img, names, bin_edges, var_floor, edge_floor, tgt):
    """cluster 일치 + 후보가 충분하고 void 탈락이 실제로 발생하는 사례."""
    best = None
    for s in sel:
        if s.get("cluster_id") != CLUSTER or not s.get("position"):
            continue
        gid = s.get("assigned_normal_id")
        rows = good_by_img.get(gid)
        if not rows:
            continue
        grid, gw, gh = CBS._tile_grid(rows, names, bin_edges, var_floor, edge_floor)
        bbox = CBS._parse_bbox(s.get("defect_bbox"))
        if not bbox:
            continue
        dim = CBS._image_dim(rows)
        ew, eh = CBS._effective_wh((bbox[2], bbox[3]), dim)
        bw, bh = max(1, -(-ew // TILE)), max(1, -(-eh // TILE))
        if gw < bw or gh < bh:
            continue
        adm, rej = all_scores(grid, gw, gh, bw, bh, tgt)
        if len(adm) < 6 or rej == 0:
            continue
        spread = adm[0][0] - adm[-1][0]
        cand = (spread, s, gid, grid, gw, gh, bw, bh, ew, eh, adm, rej, dim)
        if best is None or spread > best[0]:
            best = cand
    return best


def gray_of(gid, dim):
    stem = str(gid).lstrip("_")
    p = os.path.join(ROOT, DS, "train", "good", stem + ".jpg")
    im = Image.open(p).convert("L")
    if im.size != tuple(dim):
        im = im.resize(tuple(dim))
    return np.asarray(im)


def main():
    names, bin_edges, tgt, good_by_img, vf, ef, sel = load_all()
    case = pick_case(sel, good_by_img, names, bin_edges, vf, ef, tgt)
    if case is None:
        raise SystemExit("no suitable case found")
    _, s, gid, grid, gw, gh, bw, bh, ew, eh, adm, rej, dim = case
    img = gray_of(gid, dim)
    vmax = max(tgt.values())

    # severstal 은 1600x256 로 극단적 와이드 → A/B 는 전폭 행에 둔다.
    fig = plt.figure(figsize=(17, 11))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.0, 2.5],
                          hspace=0.42, wspace=0.18)

    # ---- A: tile grid coloured by q_k(cell) ----
    axA = fig.add_subplot(gs[0, :])
    axA.imshow(img, cmap="gray", vmin=0, vmax=255)
    for j in range(gh):
        for i in range(gw):
            x, y = i * TILE, j * TILE
            if (i, j) not in grid:
                axA.add_patch(Rectangle((x, y), TILE, TILE, facecolor="#202020",
                                        edgecolor="#b0b0b0", hatch="///", lw=0.6,
                                        alpha=0.85))
                continue
            p = tgt.get(grid[(i, j)], 0.0)
            axA.add_patch(Rectangle((x, y), TILE, TILE,
                                    facecolor=plt.cm.viridis(p / vmax if vmax else 0),
                                    alpha=0.66, edgecolor="#404040", lw=0.4))
    axA.set_title(f"A   Target probability $q_k(c)$ of each 64-px tile's context "
                  f"cell (k={CLUSTER});  hatched = void, excluded everywhere",
                  fontsize=11)
    axA.set_xticks([]); axA.set_yticks([])
    sm = plt.cm.ScalarMappable(cmap="viridis",
                               norm=plt.Normalize(vmin=0, vmax=vmax))
    fig.colorbar(sm, ax=axA, fraction=0.012, pad=0.006).set_label("$q_k(c)$",
                                                                 fontsize=9)

    # ---- B: three candidates, footprint vs ring ----
    axB = fig.add_subplot(gs[1, :])
    axB.imshow(img, cmap="gray", vmin=0, vmax=255)
    picks = [(adm[0], "best", "#28c828"), (adm[len(adm) // 2], "median", "#e0a020"),
             (adm[-1], "worst", "#d04020")]
    for (sc, (si, sj), _h, nring), tag, col in picks:
        for t in CBS._ring_keys(si, sj, bw, bh):
            if t in grid:
                axB.add_patch(Rectangle((t[0] * TILE, t[1] * TILE), TILE, TILE,
                                        facecolor=col, alpha=0.30,
                                        edgecolor=col, ls="--", lw=1.1))
        axB.add_patch(Rectangle((si * TILE, sj * TILE), bw * TILE, bh * TILE,
                                facecolor="none", edgecolor="#ffffff", lw=2.2))
        cy = sj * TILE + bh * TILE / 2.0
        axB.text(si * TILE + bw * TILE / 2.0, cy, tag + "\n" + ("%.3f" % sc),
                 color="#ffffff", fontsize=9, ha="center", va="center",
                 bbox=dict(boxstyle="round,pad=0.22", fc="#000000", ec=col,
                           alpha=0.72))
    axB.set_title("B   Footprint $F(s)$ (white outline, overwritten by the paste) "
                  "vs ring $R(s)$ (dashed, the region that survives);  "
                  f"{rej} of {len(adm) + rej} positions rejected by valid(s)",
                  fontsize=11)
    axB.set_xticks([]); axB.set_yticks([])

    # ---- C: ring histogram vs target, best and median ----
    hb, hm = adm[0][2], adm[len(adm) // 2][2]
    order = sorted(tgt, key=lambda c: -tgt[c])[:8]
    for c in sorted(set(hb) | set(hm), key=lambda c: -(hb.get(c, 0) + hm.get(c, 0))):
        if c not in order and len(order) < TOP_CELLS:
            order.append(c)
    axC = fig.add_subplot(gs[2, 0])
    xs = np.arange(len(order))
    axC.bar(xs, [tgt.get(c, 0.0) for c in order], width=0.78, color="#c0c0c0",
            label="target $q_k(c)$", zorder=1)
    for h, col, lab, sc in ((hb, "#1f9d1f", "best $h_s$", adm[0][0]),
                            (hm, "#c98010", "median $h_s$", adm[len(adm) // 2][0])):
        hv = [h.get(c, 0.0) for c in order]
        axC.plot(xs, hv, marker="o", ms=5, lw=1.6, color=col, zorder=3,
                 label=f"{lab}   score={sc:.3f}")
        axC.bar(xs, [min(a, tgt.get(c, 0.0)) for a, c in zip(hv, order)],
                width=0.78, color=col, alpha=0.42, zorder=2)
    axC.set_xticks(xs)
    axC.set_xticklabels(order, rotation=62, ha="right", fontsize=7)
    axC.set_ylabel("probability", fontsize=10)
    axC.set_xlabel("context cell $c$", fontsize=10)
    axC.set_title("C   Ring histogram $h_s$ against target $q_k$;   filled bars = "
                  "elementwise minimum, whose sum is the score", fontsize=11)
    axC.legend(fontsize=9, frameon=False)
    axC.grid(axis="y", alpha=0.25)

    # ---- D: score over admissible candidates ----
    axD = fig.add_subplot(gs[2, 1])
    sc_all = [t[0] for t in adm]
    axD.plot(range(1, len(sc_all) + 1), sc_all, lw=1.8, color="#2878c8")
    axD.plot(1, sc_all[0], marker="*", ms=18, color="#c81e1e", zorder=4,
             label=f"selected $s^*$   score={sc_all[0]:.3f}")
    for (sc, _xy, _h, _n), tag, col in picks[1:]:
        r = sc_all.index(sc) + 1
        axD.plot(r, sc, marker="o", ms=8, mfc="none", mew=1.8, color=col, zorder=4)
        axD.annotate(tag, (r, sc), textcoords="offset points", xytext=(8, 7),
                     fontsize=9, color=col)
    axD.set_xlabel("admissible candidate positions, sorted by score", fontsize=10)
    axD.set_ylabel("score$(k,s)$", fontsize=10)
    axD.set_title(f"D   Score across the {len(sc_all)} admissible positions;  "
                  f"best/worst ratio {sc_all[0] / max(sc_all[-1], 1e-9):.1f}x",
                  fontsize=11)
    axD.legend(fontsize=9, frameon=False)
    axD.grid(alpha=0.25)

    fig.suptitle(f"Ring-matched placement — {DS}, cluster k={CLUSTER}, "
                 f"background {gid}, crop {ew}x{eh}px ({bw}x{bh} tiles)",
                 fontsize=12, y=0.975)
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, "[figure 3.2.4 2] placement_ring.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print("saved:", out)
    print(f"  case: roi_idx={s.get('roi_idx')} image_id={s.get('image_id')} "
          f"bg={gid} admissible={len(sc_all)} rejected={rej} "
          f"score range {sc_all[-1]:.3f}..{sc_all[0]:.3f}")


if __name__ == "__main__":
    main()
