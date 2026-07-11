# -*- coding: utf-8 -*-
"""Local smoke + viz for D_v-specific background-distribution image ranking.

Compares the NEW ranking (histogram-intersection of D_v background hist vs
clean bg hist) against the OLD cluster-aggregate mean-compat ranking
(_image_compat_score over matrix_symmetric[cluster]) on real leather/mtd data.
No pytest. Reuses the exact module-level helpers from generate_defects.py.
"""
import os, sys, csv, json, statistics, random
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ROOT = Path(r"D:/project/aroma")
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "aroma"))

import generate_defects as gd
import distribution_profiling as dp

ETC = ROOT / ".claude" / ".etc"
VIZ_DIR = ETC / "positive_place_viz"
VIZ_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = {
    "leather": {
        "prof": ETC / "profiling_tobe" / "mvtec_leather",
        "base": ETC / "leather",
        "colab_root": "/content/drive/MyDrive/data/Aroma/mvtec/leather",
    },
    "mtd": {
        "prof": ETC / "profiling_tobe" / "mtd",
        "base": ETC / "mtd",
        "colab_root": "/content/drive/MyDrive/data/Aroma/mtd",
    },
}

TOPK_REPORT = 10


def load_gray(path):
    arr = np.asarray(Image.open(path).convert("RGB"))
    return gd.cv2.cvtColor(arr, gd.cv2.COLOR_RGB2GRAY)


def clean_features(gray, bin_edges):
    """Return (p_clean hist, mean local_variance over non-void tiles, n_nonvoid)."""
    cells = gd._normal_tile_cells(gray, bin_edges,
                                  stride=gd._NORMAL_STRIDE, cap=gd._NORMAL_SAMPLE_CAP)
    hist = gd._cell_hist(cells)
    # raw local_variance over the SAME non-void 64px tiles
    h, w = gray.shape[:2]
    tile = gd._COMPAT_TILE
    st = gd._NORMAL_STRIDE

    def anchors(dim, s):
        if dim <= tile:
            return [0]
        a = list(range(0, dim - tile + 1, s))
        if a[-1] != dim - tile:
            a.append(dim - tile)
        return a
    ya, xa = anchors(h, st), anchors(w, st)
    while len(ya) * len(xa) > gd._NORMAL_SAMPLE_CAP and st < max(h, w, 1):
        st *= 2
        ya, xa = anchors(h, st), anchors(w, st)
    lv = []
    idx = 0
    for ay in ya:
        for ax in xa:
            if idx >= len(cells):
                break
            _, void = cells[idx]
            idx += 1
            if void:
                continue
            win = gray[ay:ay + tile, ax:ax + tile]
            lv.append(float(np.var(win.astype(np.float64))))
    mean_lv = sum(lv) / len(lv) if lv else float("nan")
    return hist, mean_lv


def cv(vals):
    vals = [v for v in vals if v == v]  # drop nan
    if len(vals) < 2:
        return float("nan")
    m = statistics.mean(vals)
    if m == 0:
        return float("nan")
    return statistics.pstdev(vals) / abs(m)


def local_path(colab_path, ds):
    """Map a Colab image_path to the local .claude/.etc path via test/<type>/<file>."""
    p = colab_path.replace("\\", "/")
    if "/test/" in p:
        suffix = p.split("/test/", 1)[1]  # <type>/<file>
        return DATASETS[ds]["base"] / "test" / suffix
    return None


def gt_mask_path(ds, dtype, image_id):
    mp = DATASETS[ds]["base"] / "ground_truth" / dtype / f"{image_id}_mask.png"
    return mp if mp.exists() else None


def run_dataset(ds):
    cfg = DATASETS[ds]
    matrix = json.load(open(cfg["prof"] / "compatibility_matrix.json"))
    bin_edges = matrix["bin_edges"]
    msym = matrix["matrix_symmetric"]
    clusters_present = sorted(int(k) for k, v in msym.items() if len(v) > 0)

    ca = json.load(open(cfg["prof"] / "morphology_clusters.json"))["cluster_assignments"]

    # --- morphology_features: image_id -> row ---
    rows = {}
    with open(cfg["prof"] / "morphology_features.csv", newline="") as f:
        for r in csv.DictReader(f):
            rows[r["image_id"]] = r

    # --- build clean pool features (cached once) ---
    good_dir = cfg["base"] / "train" / "good"
    good_files = sorted(good_dir.iterdir())
    clean = {}  # path -> (hist, mean_lv)
    clean_cells = {}  # path -> [(cell, void)]  (cached for OLD ranking)
    for gf in good_files:
        try:
            g = load_gray(gf)
        except Exception:
            continue
        clean[str(gf)] = clean_features(g, bin_edges)
        clean_cells[str(gf)] = gd._normal_tile_cells(
            g, bin_edges, stride=gd._NORMAL_STRIDE, cap=gd._NORMAL_SAMPLE_CAP)
    print(f"[{ds}] clean pool: {len(clean)} good imgs, clusters w/ matrix_symmetric: {clusters_present}")

    # --- pick a representative D_v per present cluster ---
    reps = {}
    for cid in clusters_present:
        # candidate image_ids in this cluster with a resolvable local source
        cand = [iid for iid, c in ca.items() if int(c) == cid and iid in rows]
        chosen = None
        for iid in sorted(cand):
            row = rows[iid]
            lp = local_path(row["image_path"], ds)
            if lp is None or not lp.exists():
                continue
            chosen = (iid, row, lp)
            break
        if chosen:
            reps[cid] = chosen

    results = {"dataset": ds, "clusters": {}}
    seed_div = {}

    for cid, (iid, row, lp) in reps.items():
        dtype = row["defect_type"]
        g_dv = load_gray(lp)
        mp = gt_mask_path(ds, dtype, iid)
        m = np.asarray(Image.open(mp).convert("L")) if mp else None
        bbox = None
        if row.get("defect_bbox"):
            try:
                bbox = tuple(int(x) for x in row["defect_bbox"].split(","))
            except Exception:
                bbox = None
        bb_arg = bbox if (m is None and bbox and len(bbox) == 4) else None
        p_dv = gd._dv_bg_hist(g_dv, mask=m, bbox=bb_arg, bin_edges=bin_edges,
                              stride=gd._NORMAL_STRIDE, cap=gd._NORMAL_SAMPLE_CAP)
        if not p_dv:
            print(f"[{ds}] cluster {cid}: EMPTY p_dv (skip)")
            continue

        compat_row = msym.get(str(cid), {})

        # NEW ranking: histogram-intersection D_v-bg vs clean-bg
        new_scored = []
        for path, (hist, _lv) in clean.items():
            sim = gd._hist_intersection(p_dv, hist) if hist else -1.0
            new_scored.append((sim, path))
        new_ranked = sorted((t for t in new_scored if t[0] >= 0.0),
                            key=lambda t: t[0], reverse=True)

        # OLD ranking: cluster-aggregate mean-compat (_image_compat_score)
        old_scored = []
        for path in clean:
            sc = gd._image_compat_score(clean_cells[path], compat_row)
            old_scored.append((sc, path))
        old_ranked = sorted((t for t in old_scored if t[0] >= 0.0),
                            key=lambda t: t[0], reverse=True)

        new_top = [p for _, p in new_ranked[:TOPK_REPORT]]
        old_top = [p for _, p in old_ranked[:TOPK_REPORT]]

        # (a) intersection with p_dv over top-10 (the "이질감" metric)
        new_inter = [gd._hist_intersection(p_dv, clean[p][0]) for p in new_top]
        old_inter = [gd._hist_intersection(p_dv, clean[p][0]) for p in old_top]
        # (b) local_variance cv over top-10
        new_lv = [clean[p][1] for p in new_top]
        old_lv = [clean[p][1] for p in old_top]

        results["clusters"][cid] = {
            "rep_image_id": iid, "defect_type": dtype,
            "used_mask": mp is not None, "bbox": bbox,
            "p_dv_support": len(p_dv),
            "new_inter_mean": round(statistics.mean(new_inter), 4),
            "old_inter_mean": round(statistics.mean(old_inter), 4),
            "new_lv_cv": round(cv(new_lv), 4),
            "old_lv_cv": round(cv(old_lv), 4),
            "new_lv_mean": round(statistics.mean([v for v in new_lv if v == v]), 1) if any(v==v for v in new_lv) else None,
            "old_lv_mean": round(statistics.mean([v for v in old_lv if v == v]), 1) if any(v==v for v in old_lv) else None,
        }

        # (c) diversity: top-K rng sampling across seeds
        picks = {}
        for seed in (0, 1, 2, 42):
            rng = random.Random(seed)
            picks[seed] = Path(gd._rank_normals(new_scored, rng, gd._NORMAL_TOPK)).name
        seed_div[cid] = picks

        # (d) viz: left defect source (+bbox/mask), right top-10 NEW clean
        _render_viz(ds, cid, dtype, iid, lp, bbox, mp, new_top, new_inter)

    results["seed_diversity"] = seed_div
    return results


def _render_viz(ds, cid, dtype, iid, lp, bbox, mp, top_paths, inters):
    n = len(top_paths)
    ncol = 5
    nrow = 1 + (n + ncol - 1) // ncol
    fig = plt.figure(figsize=(ncol * 2.4, nrow * 2.4))
    # left: defect source spanning top row (use first cell big)
    ax0 = fig.add_subplot(nrow, ncol, 1)
    src = np.asarray(Image.open(lp).convert("RGB"))
    ax0.imshow(src)
    if bbox and len(bbox) == 4:
        x, y, w, h = bbox
        ax0.add_patch(mpatches.Rectangle((x, y), w, h, fill=False,
                                         edgecolor="red", linewidth=2))
    ax0.set_title(f"D_v src c{cid} {dtype}\n{iid}", fontsize=7)
    ax0.axis("off")
    # mask overlay thumbnail in cell 2
    if mp:
        ax1 = fig.add_subplot(nrow, ncol, 2)
        ax1.imshow(np.asarray(Image.open(mp).convert("L")), cmap="gray")
        ax1.set_title("GT mask", fontsize=7); ax1.axis("off")
    # top-10 clean on subsequent cells
    for i, (p, s) in enumerate(zip(top_paths, inters)):
        ax = fig.add_subplot(nrow, ncol, ncol + 1 + i)
        ax.imshow(np.asarray(Image.open(p).convert("RGB")))
        ax.set_title(f"#{i+1} sim={s:.3f}", fontsize=7)
        ax.axis("off")
    fig.suptitle(f"{ds} cluster {cid}: D_v-similarity top-{n} clean bg", fontsize=9)
    fig.tight_layout()
    out = VIZ_DIR / f"defect_vs_cleanbg_dv_{ds}_c{cid}.png"
    fig.savefig(out, dpi=90, bbox_inches="tight")
    plt.close(fig)
    print(f"[{ds}] saved viz {out}")


if __name__ == "__main__":
    only = sys.argv[1] if len(sys.argv) > 1 else None
    allres = {}
    for ds in DATASETS:
        if only and ds != only:
            continue
        allres[ds] = run_dataset(ds)
    print("\n===== RESULTS JSON =====")
    print(json.dumps(allres, indent=2, ensure_ascii=False))
