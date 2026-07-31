# -*- coding: utf-8 -*-
"""Figure — 형태 특징 6종이 각각 무엇을 재는가 (k의 입력 벡터)

`aroma_core_compatibility_model_20260729.md` §2 보조 자료.
CASDA `[figure2] defect_type.png` 형식 준용 — 행=범주, 열=단계, 좌여백=범주명·기준,
하단 legend, 지표 박스.

행 = MORPH_FEATURES 6종. 열 4개:
  1) Inspected Region      — 그 특징이 사용하는 기하 구성물을 crop 위에 표시
  2) Isolated Construct    — 구성물만 분리 (무엇이 측정되는지 무모호)
  3) Computation           — 식에 들어가는 두 양을 막대로 대비
  4) Dataset Position      — severstal 전체 분포 + 이 샘플 위치 + GMM이 보는 min-max 값

상단 전폭 패널 = 측정 대상 불일치(본 문서에서 가장 중요한 상이점):
  6종 전부 최대 blob 1개만 기술 / defect_bbox와 저장 마스크는 전 성분 포함.

대표 샘플 = severstal 전체의 medoid (min-max 6D centroid 최근접). 값은 운영 함수를
import해 산출(재구현 금지): DefectCharacterizer.analyze_defect_region + regionprops.

출력: fig_morph_features_<stem>.png
"""
import os
import sys
import csv
import io

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.patches import Rectangle, Ellipse
import cv2
from skimage.measure import label, regionprops, find_contours
from skimage.morphology import convex_hull_image

for _f in ("Malgun Gothic", "Gulim", "Batang", "HCR Dotum"):
    try:
        rcParams["font.family"] = _f
        break
    except Exception:
        continue
rcParams["axes.unicode_minus"] = False

REPO = os.environ.get("AROMA_REPO", "D:/project/aroma")
ROOT = os.environ.get("AROMA_DATASET_ROOT", "D:/project/aroma_dataset")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "image")
sys.path.insert(0, REPO)
from utils.defect_characterization import DefectCharacterizer  # noqa: E402

DS = "severstal"
MF = ["linearity", "solidity", "extent", "aspect_ratio", "eccentricity", "circularity"]
PAD = 14

# 행 색 (CASDA 스타일 좌여백 바 + 열 테두리)
ROW_COLOR = {
    "linearity":    "#2563eb",
    "solidity":     "#16a34a",
    "extent":       "#ea580c",
    "aspect_ratio": "#7c3aed",
    "eccentricity": "#0d9488",
    "circularity":  "#dc2626",
}
ROW_DESC = {
    "linearity":    ("linearity", "1 - lambda_min / lambda_max",
                     "픽셀좌표 공분산\n고윳값 비"),
    "solidity":     ("solidity", "area / convex-hull area",
                     "결함 면적 대\nconvex hull 면적"),
    "extent":       ("extent", "area / own-bbox area",
                     "결함 면적 대\n자기 bbox 면적"),
    "aspect_ratio": ("aspect_ratio", "major / minor",
                     "2차 중심모멘트\n등가 타원 축비"),
    "eccentricity": ("eccentricity", "sqrt(1 - (minor/major)^2)",
                     "동일 타원의 이심률\n= sqrt(linearity)"),
    "circularity":  ("circularity", "4*pi*area / perimeter^2",
                     "경계 윤곽선 길이와\n면적의 관계"),
}
COL_TITLE = [
    ("Inspected Region", "특징이 사용하는 기하 구성물", "#374151"),
    ("Isolated Construct", "측정 대상만 분리", "#1d4ed8"),
    ("Computation", "식에 들어가는 두 양", "#15803d"),
    ("Dataset Position", "severstal 분포 내 위치 / GMM 입력값", "#c2410c"),
]


def pick_medoid():
    p = f"{ROOT}/profiling/profiling/{DS}/morphology_features.csv"
    rows = list(csv.DictReader(io.open(p, encoding="utf-8")))
    X = np.array([[float(r[f]) for f in MF] for r in rows])
    Xn = (X - X.min(0)) / (X.max(0) - X.min(0) + 1e-6)
    d = np.linalg.norm(Xn - Xn.mean(0), axis=1)
    i = int(np.argmin(d))
    return rows[i], X, Xn, i, float(d[i])


def main():
    row, X, Xn, idx, dist = pick_medoid()
    image_id = row["image_id"]
    cls, stem = image_id.split("_", 1)

    mask = cv2.imdecode(np.fromfile(f"{ROOT}/{DS}/masks/{cls}/{stem}.png",
                                    dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    bgr = cv2.imdecode(np.fromfile(f"{ROOT}/{DS}/test/{cls}/{stem}.jpg",
                                   dtype=np.uint8), cv2.IMREAD_COLOR)
    if mask is None or bgr is None:
        raise SystemExit(f"로드 실패: {cls}/{stem}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    binary = (mask > 0).astype(np.uint8)
    lab = label(binary, connectivity=2)
    props = regionprops(lab)
    region = max(props, key=lambda r: r.area)

    met = dc_met = DefectCharacterizer().analyze_defect_region(mask)
    perim = float(region.perimeter) + 1e-6
    vals = {
        "linearity": dc_met["linearity"],
        "solidity": dc_met["solidity"],
        "extent": dc_met["extent"],
        "aspect_ratio": dc_met["aspect_ratio"],
        "eccentricity": float(region.eccentricity),
        "circularity": float(4 * np.pi * region.area / perim ** 2),
    }

    bx, by, bw, bh = cv2.boundingRect(binary)      # defect_bbox = 전 성분
    r0, c0, r1, c1 = region.bbox
    h, w = binary.shape
    y0, x0 = max(0, r0 - PAD), max(0, c0 - PAD)
    y1, x1 = min(h, r1 + PAD), min(w, c1 + PAD)

    sub_rgb = rgb[y0:y1, x0:x1]
    sub_mask = (lab[y0:y1, x0:x1] == region.label)
    sub_other = binary[y0:y1, x0:x1].astype(bool) & ~sub_mask
    hull = convex_hull_image(sub_mask)
    rr, cc = np.where(sub_mask)
    obr0, obr1, obc0, obc1 = rr.min(), rr.max() + 1, cc.min(), cc.max() + 1

    cy, cx = region.centroid[0] - y0, region.centroid[1] - x0
    L, S = float(region.axis_major_length), float(region.axis_minor_length)
    ang = 90.0 - np.degrees(region.orientation)
    t = np.radians(ang)

    cov = np.cov(np.column_stack([rr, cc]).astype(float), rowvar=False)
    ev = np.linalg.eigvalsh(cov)
    lam_min, lam_max = float(ev[0]), float(ev[1])
    hull_area = int(hull.sum())
    own_bbox_area = int((obr1 - obr0) * (obc1 - obc0))

    fig = plt.figure(figsize=(15.5, 21.5))
    outer = fig.add_gridspec(8, 5, height_ratios=[0.62, 0.30] + [1.0] * 6,
                             width_ratios=[0.46, 1.0, 1.0, 1.0, 1.0],
                             hspace=0.34, wspace=0.20,
                             left=0.02, right=0.985, top=0.975, bottom=0.085)

    # ===== 상단 전폭 — 측정 대상 불일치 =====
    axT = fig.add_subplot(outer[0, :])
    axT.imshow(rgb)
    for p in props:
        pr0, pc0, pr1, pc1 = p.bbox
        is_max = (p.label == region.label)
        axT.add_patch(Rectangle((pc0, pr0), pc1 - pc0, pr1 - pr0, facecolor="none",
                                edgecolor="#facc15" if is_max else "#94a3b8",
                                linewidth=2.2 if is_max else 1.0,
                                linestyle="-" if is_max else ":"))
    axT.add_patch(Rectangle((bx, by), bw, bh, facecolor="none",
                            edgecolor="#22c55e", linewidth=2.0, linestyle="--"))
    axT.set_title(
        f"{DS} / {cls} / {stem}   (medoid: min-max 6D centroid 최근접, dist={dist:.4f})\n"
        f"노란 실선 = 최대 blob(area {int(region.area)}) — 아래 6종 특징이 재는 유일한 대상   |   "
        f"회색 점선 = 나머지 성분 {len(props)-1}개 (측정 제외)   |   "
        f"초록 파선 = defect_bbox({bx},{by},{bw},{bh}) = 전 성분 포함 → 합성 crop 범위",
        fontsize=10.5, pad=8)
    axT.set_xticks([]); axT.set_yticks([])
    for sp in axT.spines.values():
        sp.set_edgecolor("#111827"); sp.set_linewidth(1.4)

    # ===== 열 제목 =====
    for j, (t1, t2, col) in enumerate(COL_TITLE):
        axh = fig.add_subplot(outer[1, j + 1])
        axh.axis("off")
        axh.add_patch(Rectangle((0.02, 0.12), 0.96, 0.76, transform=axh.transAxes,
                                facecolor="#f9fafb", edgecolor=col, linewidth=2.0))
        axh.text(0.5, 0.66, t1, transform=axh.transAxes, ha="center", va="center",
                 fontsize=12, weight="bold", color=col)
        axh.text(0.5, 0.32, t2, transform=axh.transAxes, ha="center", va="center",
                 fontsize=8.5, color="#4b5563")

    def strip(ax):
        ax.set_xlim(0, sub_rgb.shape[1]); ax.set_ylim(sub_rgb.shape[0], 0)
        ax.set_xticks([]); ax.set_yticks([])

    def frame(ax, col):
        for sp in ax.spines.values():
            sp.set_edgecolor(col); sp.set_linewidth(1.8)

    for i, feat in enumerate(MF):
        gr = i + 2
        col = ROW_COLOR[feat]
        name, formula, target = ROW_DESC[feat]

        # ---- 좌여백 라벨 ----
        axL = fig.add_subplot(outer[gr, 0])
        axL.axis("off")
        axL.add_patch(Rectangle((0.0, 0.06), 0.055, 0.88, transform=axL.transAxes,
                                facecolor=col, edgecolor="none"))
        axL.text(0.13, 0.74, name, transform=axL.transAxes, fontsize=12.5,
                 weight="bold", color=col, va="center")
        axL.text(0.13, 0.55, formula, transform=axL.transAxes, fontsize=8.2,
                 color="#111827", va="center")
        axL.text(0.13, 0.34, target, transform=axL.transAxes, fontsize=8.2,
                 color="#4b5563", va="center")
        axL.text(0.13, 0.12, f"= {vals[feat]:.4f}", transform=axL.transAxes,
                 fontsize=11, weight="bold", color=col, va="center")

        # ---- 열1: Inspected Region ----
        ax = fig.add_subplot(outer[gr, 1])
        ax.imshow(sub_rgb)
        red = np.zeros(sub_mask.shape + (4,), float)
        red[sub_mask] = (0.86, 0.15, 0.15, 0.34)
        ax.imshow(red)
        if feat in ("linearity", "aspect_ratio", "eccentricity"):
            ax.add_patch(Ellipse((cx, cy), L, S, angle=ang, facecolor="none",
                                 edgecolor=col, linewidth=2.0))
            mdx, mdy = (L / 2) * np.cos(t), (L / 2) * np.sin(t)
            sdx, sdy = -(S / 2) * np.sin(t), (S / 2) * np.cos(t)
            ax.plot([cx - mdx, cx + mdx], [cy - mdy, cy + mdy], color=col, lw=2.0)
            ax.plot([cx - sdx, cx + sdx], [cy - sdy, cy + sdy], color=col, lw=1.3,
                    linestyle="--")
            if feat == "eccentricity":
                c_ = np.sqrt(max((L / 2) ** 2 - (S / 2) ** 2, 0.0))
                ax.plot([cx - c_ * np.cos(t), cx + c_ * np.cos(t)],
                        [cy - c_ * np.sin(t), cy + c_ * np.sin(t)], "x",
                        color=col, ms=8, mew=2)
        elif feat == "solidity":
            gap = hull & ~sub_mask
            ax.imshow(np.ma.masked_where(~gap, gap), cmap="Greys", alpha=0.6,
                      vmin=0, vmax=1)
            for ct in find_contours(hull.astype(float), 0.5):
                ax.plot(ct[:, 1], ct[:, 0], color=col, lw=2.0)
        elif feat == "extent":
            ax.add_patch(Rectangle((obc0, obr0), obc1 - obc0, obr1 - obr0,
                                   facecolor="none", edgecolor=col, lw=2.2))
        else:  # circularity
            for ct in find_contours(sub_mask.astype(float), 0.5):
                ax.plot(ct[:, 1], ct[:, 0], color=col, lw=1.8)
        strip(ax); frame(ax, col)

        # ---- 열2: Isolated Construct ----
        ax = fig.add_subplot(outer[gr, 2])
        canvas = np.zeros(sub_mask.shape + (3,), float)
        canvas[sub_mask] = (0.55, 0.55, 0.55)
        if feat == "solidity":
            canvas[hull & ~sub_mask] = (0.10, 0.62, 0.28)
        ax.imshow(canvas)
        if feat in ("linearity", "aspect_ratio", "eccentricity"):
            ax.add_patch(Ellipse((cx, cy), L, S, angle=ang, facecolor="none",
                                 edgecolor=col, lw=2.0))
        elif feat == "extent":
            ax.add_patch(Rectangle((obc0, obr0), obc1 - obc0, obr1 - obr0,
                                   facecolor="none", edgecolor=col, lw=2.2))
        elif feat == "circularity":
            for ct in find_contours(sub_mask.astype(float), 0.5):
                ax.plot(ct[:, 1], ct[:, 0], color=col, lw=2.0)
        if sub_other.any():
            for ct in find_contours(sub_other.astype(float), 0.5):
                ax.plot(ct[:, 1], ct[:, 0], color="#64748b", lw=0.9, ls=":")
        strip(ax); frame(ax, col)

        # ---- 열3: Computation ----
        ax = fig.add_subplot(outer[gr, 3])
        if feat == "linearity":
            bars, labs = [lam_max, lam_min], ["lambda_max", "lambda_min"]
        elif feat == "solidity":
            bars, labs = [int(region.area), hull_area], ["area", "hull area"]
        elif feat == "extent":
            bars, labs = [int(region.area), own_bbox_area], ["area", "own bbox"]
        elif feat in ("aspect_ratio", "eccentricity"):
            bars, labs = [L, S], ["major", "minor"]
        else:
            bars, labs = [float(region.area), perim ** 2 / (4 * np.pi)], \
                         ["area", "perim^2/4pi"]
        ax.barh(labs[::-1], bars[::-1], color=[col, "#cbd5e1"][::-1],
                edgecolor="#111827", linewidth=0.7)
        for yy, vv in enumerate(bars[::-1]):
            ax.text(vv, yy, f" {vv:,.0f}" if vv >= 10 else f" {vv:.3f}",
                    va="center", fontsize=9)
        ax.set_xlim(0, max(bars) * 1.32)
        ax.tick_params(labelsize=9)
        ax.grid(axis="x", alpha=0.25)
        frame(ax, col)

        # ---- 열4: Dataset Position ----
        ax = fig.add_subplot(outer[gr, 4])
        k = MF.index(feat)
        v = X[:, k]
        lo, hi = np.percentile(v, [0.5, 99.5])
        ax.hist(v[(v >= lo) & (v <= hi)], bins=45, color="#94a3b8",
                edgecolor="white", linewidth=0.3)
        ax.axvline(vals[feat], color=col, lw=2.4)
        ax.set_xlabel(
            f"{feat}   이 샘플 {vals[feat]:.3f}  |  분위 {100*np.mean(v <= vals[feat]):.0f}%\n"
            f"GMM 입력(min-max) = {Xn[idx, k]:.3f}   [분산 점유 {100*Xn[:, k].var()/Xn.var(0).sum():.1f}%]",
            fontsize=7.8, labelpad=2)
        ax.tick_params(labelsize=8)
        ax.grid(axis="y", alpha=0.22)
        frame(ax, col)

    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    handles = [
        Patch(facecolor="#facc15", edgecolor="#facc15", label="최대 blob bbox (6종 측정 대상)"),
        Line2D([0], [0], color="#22c55e", lw=2, ls="--", label="defect_bbox = 전 성분 (합성 crop)"),
        Line2D([0], [0], color="#94a3b8", lw=1, ls=":", label="측정 제외 성분"),
        Patch(facecolor=(0.86, 0.15, 0.15, 0.34), edgecolor="none", label="열1 결함 마스크"),
        Patch(facecolor="#8c8c8c", edgecolor="none", label="열2 측정 대상(최대 blob)"),
        Patch(facecolor="#1a9e47", edgecolor="none", label="열2 hull이 덮되 결함 아닌 영역"),
        Line2D([0], [0], color="#111827", lw=2, label="열1·2 각 행의 기하 구성물(행 색)"),
        Patch(facecolor="#cbd5e1", edgecolor="#111827", label="열3 분모 / 비교량"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8.6,
               frameon=True, bbox_to_anchor=(0.5, 0.038))

    fig.text(
        0.5, 0.012,
        "6종 전부 analyze_defect_region의 max(props, key=area) — 최대 blob 1개만 기술한다(상단 패널의 노란 실선). "
        "linearity·aspect_ratio·eccentricity는 같은 등가 타원의 재매개화라 독립 축이 아니다: "
        "linearity = 1 - AR^-2, eccentricity = sqrt(linearity), Spearman(lin, AR) = 1.000. "
        "GMM은 열4의 min-max 값을 6차원 벡터로 받아 BIC로 군집 수를 정한다.",
        fontsize=9.2, ha="center", color="#111827")

    out = os.path.join(OUT_DIR, "[figure 3.2.2 3] morph_features.png")
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print("saved:", out)
    print(f"  medoid={image_id}  dist={dist:.4f}  성분 {len(props)}개  최대 blob area={int(region.area)}")
    print(f"  defect_bbox=({bx},{by},{bw},{bh})   최대blob bbox={region.bbox}")
    for f in MF:
        print(f"  {f:14s} = {vals[f]:.4f}   min-max={Xn[idx, MF.index(f)]:.3f}")


if __name__ == "__main__":
    main()
