# -*- coding: utf-8 -*-
"""Figure 3.2.4-5 (per-dataset) — site resolution: ring context, source vs placement.

Sample continuity with Figure 3.2.4-4: the SAME representative defect crop
(crop ~30% of its source image) and the SAME assigned background (bg_score
rank 1) are carried forward. Two stacked panels compare, under one colour
scale, the ring context the defect actually has and the ring context the
resolved site provides:
  A  the source defect image — the tiles ringing the real defect bbox are
     tinted by the target mass tgt[k] of their context cell (value printed);
     untinted ring tiles were excluded from profiling (defect-overlapping,
     void, or unobserved),
  B  the assigned clean background — the resolved position s* (footprint
     solid, ring dashed) with its ring tiles tinted the same way.
Both panels report the ring/target intersection so the visual match is
also a number: ∩(h_ring, tgt[k]) in A, site_score = ∩(h_s, tgt[k]) in B.

Operational functions are imported from scripts/aroma/clean_bg_selection;
the panel-B argmax is asserted equal to CBS._best_ring_site's own choice.
Outputs one PNG per dataset: [figure 3.2.4 5 <ds>] placement_ring.png
"""
import glob
import os
import sys

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

sys.path.insert(0, os.path.join(REPO, "scripts", "aroma"))
import clean_bg_selection as CBS  # noqa: E402

DATASETS = ["severstal", "aitex", "kolektor", "mtd", "mvtec_leather"]
GOOD_DIR = {"severstal": "severstal/train/good", "aitex": "aitex_tiled/train/good",
            "kolektor": "kolektor/train/good", "mtd": "mtd/train/good",
            "mvtec_leather": "mvtec_leather/train/good"}
DEFECT_GLOB = {"severstal": "severstal/test/*/", "aitex": "aitex_tiled/test/*/",
               "kolektor": "kolektor/test/*/", "mtd": "mtd/test/*/",
               "mvtec_leather": "mvtec_leather/test/*/"}
AREA_TARGET = 0.30
SIDE_MAX = 0.60          # crop 이 원본의 한 변을 60% 넘게 지배하면 육안 대조 불리
GREEN, AMBER, RED = "#1d8a3c", "#e08a3c", "#c81e1e"


def stem_map(pattern):
    out = {}
    for p in glob.glob(os.path.join(ROOT, pattern, "*")):
        out[os.path.splitext(os.path.basename(p))[0]] = p
    return out


def resolve(m, iid):
    return m.get(iid) or m.get(iid.lstrip("_")) or m.get("_" + iid) \
        or m.get(iid.split("_", 1)[-1])


def ring_hist(cells):
    """cell 리스트 → 정규화 히스토그램 (관측된 ring 타일 위에서)."""
    if not cells:
        return {}
    inv = 1.0 / len(cells)
    hs = {}
    for c in cells:
        hs[c] = hs.get(c, 0.0) + inv
    return hs


def build(ds):
    """-4 와 동일한 대표 crop·배정 배경을 재현하고 자리 후보 전수를 채점한다."""
    prof = os.path.join(ROOT, "profiling", "profiling", ds)
    roi_dir = os.path.join(ROOT, "roi", ds)
    data = CBS.load_inputs(prof, roi_dir)
    if data["status"] != "ok":
        raise SystemExit("load_inputs failed (%s): %s" % (ds, data))
    names, bin_edges = data["names"], data["bin_edges"]
    vf, ef = CBS._derive_void_floors(data["good_by_img"], 15.0)
    valid_ids, _r, _d = CBS.valid_bg_pool(
        data["good_by_img"], True, None, vf, ef, floor_pct=15.0)

    good_hist, good_dim = {}, {}
    for iid in valid_ids:
        rws = data["good_by_img"].get(iid, [])
        good_hist[iid] = CBS._image_hist(rws, names, bin_edges, vf, ef)
        good_dim[iid] = CBS._image_dim(rws)

    defect_by_img = {}
    for r in data["defect_rows"]:
        defect_by_img.setdefault(r.get("image_id", ""), []).append(r)
    src_hist = {i: CBS._image_hist(rs, names, bin_edges, vf, ef)
                for i, rs in defect_by_img.items()}
    class_hist = CBS._class_bg_hist(data["defect_rows"], names, bin_edges, vf, ef,
                                    data.get("iid_to_class", {}))

    # ---- 대표 ROI: -4 와 동일 규칙 ----
    # 면적비 0.30 근접 + 어느 변도 원본의 60% 초과 지배 금지(육안 대조) + 최소변 48px.
    # 변 지배 조건을 만족하는 후보가 없으면 초과분을 페널티로 더해 최소화.
    dmap = stem_map(DEFECT_GLOB[ds])
    sel = CBS.load_json(os.path.join(roi_dir, "roi_selected.json"))
    sel = sel if isinstance(sel, list) else sel.get("selected", sel)
    best = None
    for idx, s in enumerate(sel):
        bbox = CBS._parse_bbox(s.get("defect_bbox"))
        iid = str(s.get("image_id", ""))
        dpath = resolve(dmap, iid)
        if not bbox or iid not in src_hist or not dpath:
            continue
        x, y, w, h = bbox
        if w < 48 or h < 48:
            continue
        with Image.open(dpath) as im:
            W, H = im.size
        ratio = (w * h) / float(W * H)
        side_over = max(0.0, w / W - SIDE_MAX) + max(0.0, h / H - SIDE_MAX)
        key = (abs(ratio - AREA_TARGET) + 4.0 * side_over, idx)
        if best is None or key < best[0]:
            best = (key, s, dpath, bbox)
    if best is None:
        raise SystemExit("no usable ROI in %s" % ds)
    _, s, dpath, bbox = best
    x, y, w, h = bbox
    src_iid = str(s.get("image_id", ""))
    sdv = src_hist[src_iid]
    cdv = class_hist.get(str(s.get("class_key") or s.get("class_value") or ""), {})

    # ---- 배정 배경: -4 와 동일 규칙 (bg_score 단순합 rank 1) ----
    def bg_score(g):
        return (CBS._hist_intersection(good_hist[g], sdv)
                + CBS._hist_intersection(good_hist[g], cdv)
                + min(1.0, CBS._scale_to_fit((w, h), good_dim[g])))
    ranked_bgs = sorted(valid_ids, key=bg_score, reverse=True)

    k = str(s.get("cluster_id"))
    tgt = CBS._target_by_cluster(data["compat"]).get(k) or {}

    # ---- 원본 결함 이미지의 실제 ring (bbox 를 덮는 타일 사각형의 8이웃) ----
    grid_s, gwS, ghS = CBS._tile_grid(defect_by_img.get(src_iid, []),
                                      names, bin_edges, vf, ef)
    si0, sj0 = x // 64, y // 64
    bwS = max(1, -(-(x + w) // 64) - si0)
    bhS = max(1, -(-(y + h) // 64) - sj0)
    src_ring_keys = CBS._ring_keys(si0, sj0, bwS, bhS)
    src_ring = {t: grid_s[t] for t in src_ring_keys if t in grid_s}
    src_score = sum(min(v, tgt.get(c, 0.0))
                    for c, v in ring_hist(list(src_ring.values())).items())

    def enumerate_sites(g):
        """배경 g 위 admissible 자리 전수 채점. (grid, gw, gh, bw, bh, cands)"""
        grid, gw, gh = CBS._tile_grid(data["good_by_img"].get(g, []),
                                      names, bin_edges, vf, ef)
        ew, eh = CBS._effective_wh((w, h), good_dim[g])
        bw = max(1, -(-int(ew) // 64))
        bh = max(1, -(-int(eh) // 64))
        cands = []                  # (score, (si,sj), ring cell dict)
        for sj in range(gh - bh + 1):
            for si in range(gw - bw + 1):
                if any((si + a, sj + b) not in grid
                       for a in range(bw) for b in range(bh)):
                    continue
                rng = {t: grid[t] for t in CBS._ring_keys(si, sj, bw, bh)
                       if t in grid}
                if not rng:
                    continue
                hs = ring_hist(list(rng.values()))
                sc = sum(min(v, tgt[c]) for c, v in hs.items() if c in tgt)
                cands.append((sc, (si, sj), rng))
        return grid, gw, gh, bw, bh, cands

    # rank 1 배경부터 시도 — admissible 자리가 없으면(실전 폴백 케이스) 다음 순위로.
    # 실제 파이프라인은 그 배경에서 런타임 배치로 폴백하지만, 본 그림은 ring 역학의
    # 삽화이므로 자리가 존재하는 최상위 배경을 쓰고 그 사실을 제목에 명시한다.
    assigned = None
    fallback_note = ""
    for rank, g in enumerate(ranked_bgs, start=1):
        grid, gw, gh, bw, bh, cands = enumerate_sites(g)
        if cands:
            assigned = g
            if rank > 1:
                fallback_note = ("  [rank-1 background admits no valid position "
                                 "(runtime fallback) — shown: rank %d]" % rank)
            break
    if assigned is None:
        raise SystemExit("no background with admissible positions in %s" % ds)
    cands.sort(key=lambda t: (-t[0], t[1][1], t[1][0]))
    # 운영 argmax 와 일치 확인 (score 동률 시 래스터 선착순 차이는 score 로만 대조)
    op_xy, op_sc = CBS._best_ring_site(grid, gw, gh, bw, bh, tgt)
    assert op_xy is not None and abs(op_sc - cands[0][0]) < 1e-9, \
        "figure argmax != operational argmax (%s)" % ds

    gmap = stem_map(GOOD_DIR[ds])
    return dict(s=s, k=k, bw=bw, bh=bh, assigned=assigned,
                src_img=Image.open(dpath).convert("L"),
                bg_img=Image.open(resolve(gmap, assigned)).convert("L"),
                bbox=bbox, src_fp=(si0, sj0, bwS, bhS), src_ring=src_ring,
                src_score=src_score, tgt=tgt, n=len(cands),
                best=cands[0], mid=cands[len(cands) // 2], worst=cands[-1],
                fallback_note=fallback_note)


def pos_box(ax, fp, color, label=None, score=None, lw=2.2, anchor="tl"):
    """footprint 실선 + ring 점선. label 은 footprint 안쪽, 흰 배경.
    anchor: tl/tr/bl — 겹치는 자리들의 라벨 충돌 회피용 코너 분리."""
    si, sj, bw, bh = fp
    ax.add_patch(Rectangle((si * 64, sj * 64), bw * 64, bh * 64,
                           fill=False, edgecolor=color, linewidth=lw))
    ax.add_patch(Rectangle(((si - 1) * 64, (sj - 1) * 64),
                           (bw + 2) * 64, (bh + 2) * 64,
                           fill=False, edgecolor=color, linewidth=1.2,
                           linestyle=(0, (4, 3))))
    if label:
        if anchor == "tr":
            xy, ha, va = ((si + bw) * 64 - 8, sj * 64 + 12), "right", "top"
        elif anchor == "bl":
            xy, ha, va = (si * 64 + 8, (sj + bh) * 64 - 12), "left", "bottom"
        else:
            xy, ha, va = (si * 64 + 8, sj * 64 + 12), "left", "top"
        ax.text(xy[0], xy[1], "%s  %.3f" % (label, score),
                fontsize=8.5, color=color, weight="bold", ha=ha, va=va,
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="none",
                          pad=1.5))


def tint_ring(ax, img, ring, tgt, vmax, fp, color, fp_lw=2.2):
    """실사 위에 ring 타일만 tgt 질량으로 tint + 값 인쇄, footprint 실선·ring 점선."""
    arr = np.asarray(img)
    ax.imshow(arr, cmap="gray", aspect="auto")
    hi = max(max(t[0] for t in ring) + 1, 1) if ring else 1
    hj = max(max(t[1] for t in ring) + 1, 1) if ring else 1
    m = np.full((hj, hi), np.nan)
    for (i, j), c in ring.items():
        m[j, i] = tgt.get(c, 0.0)
    ax.imshow(m, cmap="viridis", alpha=0.5, aspect="auto",
              interpolation="nearest", vmin=0.0, vmax=vmax,
              extent=(0, hi * 64, hj * 64, 0))
    for (i, j), c in ring.items():
        v = tgt.get(c, 0.0)
        ax.text(i * 64 + 32, j * 64 + 32, "%.2f" % v, ha="center", va="center",
                fontsize=6.2, color="white" if v < 0.6 * vmax else "black")
    pos_box(ax, fp, color, lw=fp_lw)
    ax.set_xlim(0, arr.shape[1]); ax.set_ylim(arr.shape[0], 0)
    ax.set_xticks([]); ax.set_yticks([])


for ds in DATASETS:
    d = build(ds)
    best_sc, (bsi, bsj), best_ring = d["best"]
    mid_sc, (msi, msj), _ = d["mid"]
    worst_sc, (wsi, wsj), _ = d["worst"]
    vmax = max(d["tgt"].values()) if d["tgt"] else 1.0
    x, y, w, h = d["bbox"]

    fig, (axA, axB, axC) = plt.subplots(
        3, 1, figsize=(12.8, 11.4),
        gridspec_kw=dict(hspace=0.38, height_ratios=[1, 1, 0.72]))
    fig.suptitle("Site resolution — %s   (cluster k=%s; ring tiles tinted by "
                 "target mass tgt[k], value printed; shared colour scale)%s"
                 % (ds, d["k"], d["fallback_note"]), fontsize=11.5, y=0.99)

    # A: 원본 결함 이미지의 실제 ring
    tint_ring(axA, d["src_img"], d["src_ring"], d["tgt"], vmax,
              d["src_fp"], RED)
    axA.add_patch(Rectangle((x, y), w, h, fill=False,
                            edgecolor=RED, linewidth=1.4, linestyle=":"))
    axA.set_title("A   source defect image — ring around the real defect   "
                  "\u2229(h_ring, tgt[k]) = %.3f" % d["src_score"], fontsize=10)

    # B: 배정 배경 — best 는 ring tint, mid/worst 는 박스만 (중복 자리는 생략)
    tint_ring(axB, d["bg_img"], best_ring, d["tgt"], vmax,
              (bsi, bsj, d["bw"], d["bh"]), GREEN)
    if (wsi, wsj) != (bsi, bsj):
        pos_box(axB, (wsi, wsj, d["bw"], d["bh"]), RED, "worst", worst_sc,
                anchor="bl")
    if (msi, msj) not in ((bsi, bsj), (wsi, wsj)):
        pos_box(axB, (msi, msj, d["bw"], d["bh"]), AMBER, "mid", mid_sc,
                anchor="tr")
    pos_box(axB, (bsi, bsj, d["bw"], d["bh"]), GREEN, "best s*", best_sc)
    axB.set_title("B   assigned background %s — best / mid / worst of %d "
                  "admissible positions   site_score(best) = %.3f"
                  % (d["assigned"], d["n"], best_sc), fontsize=10)

    # C: 분포 대조 — source ring 실측 h_ring vs tgt[k] vs best 자리 h_s
    h_src = ring_hist(list(d["src_ring"].values()))
    h_bst = ring_hist(list(best_ring.values()))
    cells = sorted(set(h_src) | set(h_bst) | set(d["tgt"]),
                   key=lambda c: -(d["tgt"].get(c, 0.0) + h_src.get(c, 0.0)
                                   + h_bst.get(c, 0.0)))[:12]
    xs = np.arange(len(cells))
    axC.bar(xs - 0.27, [d["tgt"].get(c, 0.0) for c in cells], width=0.27,
            color="#bcbcbc", label="target tgt[k]")
    axC.bar(xs, [h_src.get(c, 0.0) for c in cells], width=0.27,
            color="#d98080", label="measured ring of real defect (A)")
    axC.bar(xs + 0.27, [h_bst.get(c, 0.0) for c in cells], width=0.27,
            color="#7cbf8e", label="ring of resolved site s* (B)")
    axC.set_xticks(xs)
    axC.set_xticklabels(cells, rotation=60, fontsize=6.5)
    axC.set_ylabel("probability", fontsize=9, labelpad=2)
    axC.set_title("C   measured ring cells vs target — "
                  "∩(h_ring, tgt) = %.3f,  ∩(h_s*, tgt) = %.3f"
                  % (d["src_score"], best_sc), fontsize=10)
    axC.legend(fontsize=8, frameon=False)
    axC.grid(axis="y", alpha=0.25)

    fig.text(0.01, 0.005,
             "footprint solid, ring dashed; untinted ring tiles were excluded "
             "from profiling (defect-overlapping, void, or unobserved); "
             "tile value = tgt[k] mass of that tile's context cell",
             fontsize=7.6, color="#444444")
    out = os.path.join(OUT_DIR, "[figure 3.2.4 5 %s] placement_ring.png" % ds)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("saved:", out, " positions=%d site=%.3f src_ring=%.3f ring_tiles=%d"
          % (d["n"], best_sc, d["src_score"], len(d["src_ring"])))
