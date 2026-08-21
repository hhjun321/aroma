# -*- coding: utf-8 -*-
"""Figure 3.2.4-4 (per-dataset) — background assignment: cue top-3s and the final top-3.

Per dataset, one representative defect crop is fixed (crop occupies roughly
30% of its source image, for visual legibility) and every background in the
valid clean pool is scored by the fitness measures of the simplified §3.2.4
body. The figure shows, on real images:
  row 0  the source image (bbox in red) and the defect crop,
  row 1  the top-3 candidates by src_fit,
  row 2  the top-3 candidates by class_fit,
  row 3  the top-3 candidates by the final bg_score (rank 1 = assigned, ★).

bg_score follows the body formula (src_fit + class_fit + size_fit); the size
term still enters the total but has no panel of its own.

Operational functions are imported from scripts/aroma/clean_bg_selection
(no reimplementation).
Outputs one PNG per dataset: [figure 3.2.4 4 <ds>] bg_score_composition.png
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
AREA_TARGET = 0.30      # crop 이 원본에서 차지하는 목표 비율 (육안 가독)
SIDE_MAX = 0.60         # crop 이 원본의 한 변을 60% 넘게 지배하면 육안 대조 불리
BLUE, ORANGE, GREEN = "#2c6fbb", "#e08a3c", "#1d8a3c"


def stem_map(pattern):
    out = {}
    for p in glob.glob(os.path.join(ROOT, pattern, "*")):
        out[os.path.splitext(os.path.basename(p))[0]] = p
    return out


def resolve(m, iid):
    return m.get(iid) or m.get(iid.lstrip("_")) or m.get("_" + iid) \
        or m.get(iid.split("_", 1)[-1])


def build(ds):
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

    dmap = stem_map(DEFECT_GLOB[ds])
    sel = CBS.load_json(os.path.join(roi_dir, "roi_selected.json"))
    sel = sel if isinstance(sel, list) else sel.get("selected", sel)

    # 대표 ROI 선정 (결정적): crop/원본 면적비가 AREA_TARGET(0.30) 에 가장 가까운 것.
    # 어느 변도 원본의 SIDE_MAX(60%) 초과 지배 금지 — 초과분은 페널티로 최소화.
    # crop 이 원본 대부분을 차지하지 않아 육안 대조가 쉬운 표본 (최소변 48px).
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
            best = (key, s, dpath, bbox, ratio)
    if best is None:
        raise SystemExit("no usable ROI in %s" % ds)
    _, s, dpath, bbox, ratio = best
    x, y, w, h = bbox
    full = Image.open(dpath).convert("L")
    crop = full.crop((x, y, x + w, y + h))
    sdv = src_hist[str(s.get("image_id", ""))]
    cdv = class_hist.get(str(s.get("class_key") or s.get("class_value") or ""), {})

    rows = []
    for g in valid_ids:
        u = dict(src=CBS._hist_intersection(good_hist[g], sdv),
                 cls=CBS._hist_intersection(good_hist[g], cdv),
                 siz=min(1.0, CBS._scale_to_fit((w, h), good_dim[g])))
        rows.append((g, u, u["src"] + u["cls"] + u["siz"]))

    src_iid = str(s.get("image_id", ""))

    def tile_vals(iid, ref):
        """이미지의 64px 타일 격자를 참조 분포 질량으로 값매김한 (gh,gw) 배열.
        void·미관측·결함겹침 타일(=CSV에 행 없음)은 NaN → 투명 렌더."""
        rws = data["good_by_img"].get(iid) or defect_by_img.get(iid) or []
        grid, gw, gh = CBS._tile_grid(rws, names, bin_edges, vf, ef)
        m = np.full((gh, gw), np.nan)
        for (i, j), c in grid.items():
            m[j, i] = ref.get(c, 0.0)
        return m
    return s, full, bbox, crop, ratio, rows, dict(sdv=sdv, cdv=cdv,
                                                  src_iid=src_iid,
                                                  tile_vals=tile_vals)


def show(ax, im, title, sub, border=None, color="black", overlay=None, vmax=None):
    arr = np.asarray(im)
    ax.imshow(arr, cmap="gray", aspect="auto")
    if overlay is not None:
        gh, gw = overlay.shape
        ax.imshow(overlay, cmap="viridis", alpha=0.45, aspect="auto",
                  interpolation="nearest", vmin=0.0, vmax=vmax,
                  extent=(0, gw * 64, gh * 64, 0))
        ax.set_xlim(0, arr.shape[1]); ax.set_ylim(arr.shape[0], 0)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=12, pad=4, color=color)
    for sp in ax.spines.values():
        sp.set_linewidth(2.2 if border else 0.6)
        sp.set_edgecolor(border or "#666666")


for ds in DATASETS:
    s, full, bbox, crop, ratio, rows, ov = build(ds)
    gmap = stem_map(GOOD_DIR[ds])
    x, y, w, h = bbox
    # 참조 분포별 색 스케일 (해당 분포의 최대 질량 = vmax — 그룹 내 비교 가능)
    vmax_src = max(ov["sdv"].values()) if ov["sdv"] else 1.0
    vmax_cls = max(ov["cdv"].values()) if ov["cdv"] else 1.0

    fig, axes = plt.subplots(4, 3, figsize=(10.8, 13.2),
                             gridspec_kw=dict(wspace=0.10, hspace=0.42))
    fig.suptitle("Background assignment — %s   "
                 "(bg_score = src_fit + class_fit + size_fit)" % ds,
                 fontsize=13, y=0.995)

    # row 0: 원본(2칸 병합 효과 대신 좌 2칸 사용) + crop
    show(axes[0, 0], full, "source image (defect bbox in red)",
         "%dx%d px  ·  tiles tinted by p_src mass" % full.size,
         overlay=ov["tile_vals"](ov["src_iid"], ov["sdv"]), vmax=vmax_src)
    axes[0, 0].add_patch(Rectangle((x, y), w, h, fill=False,
                                   edgecolor="#d62728", linewidth=2.0))
    show(axes[0, 1], crop,
         "defect crop  (cluster k=%s)" % s.get("cluster_id"),
         "%dx%d px  ·  %.0f%% of source" % (w, h, 100 * ratio))
    axes[0, 2].axis("off")

    groups = [
        (1, "src", "top-3 by src_fit", BLUE,
         sorted(rows, key=lambda r: (-r[1]["src"], -r[2]))[:3]),
        (2, "cls", "top-3 by class_fit", ORANGE,
         sorted(rows, key=lambda r: (-r[1]["cls"], -r[2]))[:3]),
        (3, None, "top-3 by bg_score  (rank 1 = assigned)", GREEN,
         sorted(rows, key=lambda r: -r[2])[:3]),
    ]
    for rix, key, gtitle, color, top3 in groups:
        for cix, (g, u, tot) in enumerate(top3):
            gp = resolve(gmap, g)
            bg = Image.open(gp).convert("L") if gp else Image.new("L", (256, 256), 0)
            if key == "src":
                olay, vmx = ov["tile_vals"](g, ov["sdv"]), vmax_src
            elif key == "cls":
                olay, vmx = ov["tile_vals"](g, ov["cdv"]), vmax_cls
            else:
                olay, vmx = None, None      # 최종 행 = 무채색 (실제 결과물 모습)
            if key:                          # cue 행: 해당 cue 값이 주 라벨
                sub = "%s_fit %.2f   ·   bg_score %.3f" % (key, u[key], tot)
            else:                            # 최종 행: 합 + 3항 분해
                sub = "bg_score %.3f\n(src %.2f · cls %.2f · siz %.2f)" % (
                    tot, u["src"], u["cls"], u["siz"])
            title = gtitle if cix == 0 else " "
            star = " ★" if (key is None and cix == 0) else ""
            show(axes[rix, cix], bg, title + star, sub,
                 border=color, color=color, overlay=olay, vmax=vmx)

    out = os.path.join(OUT_DIR, "[figure 3.2.4 4 %s] bg_score_composition.png" % ds)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("saved:", out, " pool=%d ratio=%.2f" % (len(rows), ratio))
