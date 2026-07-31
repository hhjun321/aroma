# -*- coding: utf-8 -*-
"""Figure — 64px 패치가 context cell c로 환산되는 흐름

`aroma_core_compatibility_model_20260729.md` §3 보조 자료.
CASDA `[figure3] background_type.png` 형식 준용 — 좌여백=범주명·기준, 열 제목 박스,
하단 legend.

흐름 4단계:
  1) 상단 전폭 패널  — 64px 비겹침 격자에서 대상 패치 1개 선택
  2) 열1 Computation Map / 열2 Reduced Distribution — context feature 5종 각각 산출
  3) 열3 Bin Assignment — 데이터셋 분포 + P33/P66 tertile → bin 0/1/2
  4) 하단 전폭 패널  — 다섯 자리 조립 → c = d0_d1_d2_d3_d4

행 = CONTEXT_FEATURES 5종 (순서가 cell key 자릿수 순서와 동일).

값 검증: 열별 중간량은 `_extract_context_features`의 내부 계산을 그대로 미러하고,
산출된 5개 최종값이 import한 운영 함수 결과와 일치하는지 assert로 확인한다.
bin 환산·cell key는 운영 함수(`_context_cell_key`)를 직접 호출한다.

출력: fig_context_cell_<stem>_<cell>.png
"""
import os
import sys
import csv
import io
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.patches import Rectangle, Circle
import cv2
from skimage.feature import local_binary_pattern

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
sys.path.insert(0, os.path.join(REPO, "scripts"))
from distribution_profiling import (  # noqa: E402
    _extract_context_features, _context_cell_key, CONTEXT_FEATURES,
    GRID_SIZE, N_CONTEXT_BINS,
)

DS = "severstal"
STEM = "00031f466"        # placement figure(fig_placement_footprint)와 동일 이미지
TARGET_CELL = ""           # 비우면 이미지의 최빈 cell을 자동 선택

ROW_COLOR = {
    "local_variance":          "#2563eb",
    "edge_density":            "#16a34a",
    "texture_entropy":         "#ea580c",
    "frequency_energy":        "#7c3aed",
    "orientation_consistency": "#dc2626",
}
ROW_DESC = {
    "local_variance":          ("local_variance", "Var(p)", "패치 명도 분산"),
    "edge_density":            ("edge_density", "mean |grad|",
                                "Sobel 기울기 크기의\n평균"),
    "texture_entropy":         ("texture_entropy", "H(LBP hist)",
                                "LBP(P=8,R=1,uniform)\n10-bin 히스토그램 엔트로피"),
    "frequency_energy":        ("frequency_energy", "HF / total",
                                "FFT 고주파 에너지 비\n(저주파 반경 min(h,w)/4 밖)"),
    "orientation_consistency": ("orientation_consistency", "H(angle hist)",
                                "기울기 방향 18-bin\n히스토그램 엔트로피\n(낮을수록 일관)"),
}
COL_TITLE = [
    ("Computation Map", "패치에서 산출되는 공간 맵", "#374151"),
    ("Reduced Distribution", "식이 축약하는 중간 분포", "#1d4ed8"),
    ("Bin Assignment", "데이터셋 분포 + P33/P66 → bin", "#c2410c"),
]


def internals(patch):
    """_extract_context_features(:213-254) 내부 계산 미러. 중간량까지 반환."""
    p = patch.astype(np.float64)
    sx = cv2.Sobel(p, cv2.CV_64F, 1, 0, ksize=3)
    sy = cv2.Sobel(p, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(sx ** 2 + sy ** 2)

    lbp = local_binary_pattern(patch, P=8, R=1, method="uniform")
    counts_lbp, _ = np.histogram(lbp.ravel(), bins=10)
    p_lbp = counts_lbp / (counts_lbp.sum() + 1e-6) + 1e-12

    fft_mag = np.fft.fftshift(np.abs(np.fft.fft2(p)))
    hf, wf = fft_mag.shape
    cy, cx = hf // 2, wf // 2
    r = min(hf, wf) // 4
    Y, X = np.ogrid[:hf, :wf]
    lf_mask = (Y - cy) ** 2 + (X - cx) ** 2 <= r ** 2

    angles = np.arctan2(sy, sx)
    hist_a, edges_a = np.histogram(angles, bins=18, range=(-np.pi, np.pi))
    p_a = hist_a / (hist_a.sum() + 1e-6) + 1e-12

    vals = {
        "local_variance": float(np.var(p)),
        "edge_density": float(mag.mean()),
        "texture_entropy": float(-np.sum(p_lbp * np.log2(p_lbp))),
        "frequency_energy": float(fft_mag[~lf_mask].sum() / (fft_mag.sum() + 1e-10)),
        "orientation_consistency": float(-np.sum(p_a * np.log2(p_a))),
    }
    return vals, dict(p=p, mag=mag, lbp=lbp, counts_lbp=counts_lbp,
                      fft_mag=fft_mag, lf_mask=lf_mask, r=r, cy=cy, cx=cx,
                      hist_a=hist_a, edges_a=edges_a)


def load_good_dist(path):
    """good 패치의 5개 context feature만 스트리밍 수집 (전체 dict 적재 시 MemoryError)."""
    acc = {f: [] for f in CONTEXT_FEATURES}
    with io.open(path, encoding="utf-8", newline="") as fh:
        rd = csv.reader(fh)
        hdr = next(rd)
        it = hdr.index("image_type")
        ix = [hdr.index(f) for f in CONTEXT_FEATURES]
        for r in rd:
            if r[it] != "good":
                continue
            for f, k in zip(CONTEXT_FEATURES, ix):
                acc[f].append(r[k])
    return {f: np.asarray(v, dtype=np.float64) for f, v in acc.items()}


def main():
    compat = json.load(io.open(
        f"{ROOT}/profiling/profiling/{DS}/compatibility_matrix.json", encoding="utf-8"))
    bin_edges = compat["bin_edges"]

    gray = cv2.cvtColor(cv2.imdecode(np.fromfile(
        f"{ROOT}/{DS}/train/good/{STEM}.jpg", dtype=np.uint8),
        cv2.IMREAD_COLOR), cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    T = GRID_SIZE
    gh, gw = h // T, w // T

    # 이미지 전 패치의 cell을 구해 최빈 cell을 대상으로 삼는다(TARGET_CELL은 그 확인용).
    import collections
    cells_all = {}
    for i in range(gh):
        for j in range(gw):
            t = gray[i * T:(i + 1) * T, j * T:(j + 1) * T]
            cells_all[(i, j)] = _context_cell_key(
                _extract_context_features(t), bin_edges)
    cnt = collections.Counter(cells_all.values())
    n_cells = len(cnt)
    modal_cell, n_target = cnt.most_common(1)[0]
    if TARGET_CELL and modal_cell != TARGET_CELL:
        print(f"[주의] 최빈 cell {modal_cell} != TARGET_CELL {TARGET_CELL} — 최빈을 사용")
    # 최빈 cell 인스턴스 중 이미지 가로 중앙에 가장 가까운 타일
    hits = sorted((abs(j - gw / 2), i, j)
                  for (i, j), c in cells_all.items() if c == modal_cell)
    _, ti, tj = hits[0]
    patch = gray[ti * T:(ti + 1) * T, tj * T:(tj + 1) * T]

    vals, mid = internals(patch)
    ref = _extract_context_features(patch)
    for f in CONTEXT_FEATURES:                       # 미러 검증
        assert abs(vals[f] - ref[f]) < 1e-9, (f, vals[f], ref[f])
    cell = _context_cell_key(ref, bin_edges)
    assert cell == modal_cell, (cell, modal_cell)
    digits = cell.split("_")

    # 데이터셋 전체 context feature 분포 — csv가 100MB+ 이므로 필요한 컬럼만 스트리밍
    dist = load_good_dist(f"{ROOT}/profiling/profiling/{DS}/context_features.csv")

    fig = plt.figure(figsize=(14.5, 20.5))
    gs = fig.add_gridspec(8, 4, height_ratios=[0.55, 0.26] + [1.0] * 5 + [0.42],
                          width_ratios=[0.50, 1.0, 1.0, 1.0],
                          hspace=0.42, wspace=0.22,
                          left=0.02, right=0.985, top=0.972, bottom=0.055)

    # ===== 1단계: 상단 전폭 — 격자 + 대상 패치 =====
    axT = fig.add_subplot(gs[0, :])
    axT.imshow(gray, cmap="gray")
    for i in range(gh + 1):
        axT.axhline(i * T, color="#22d3ee", lw=0.5, alpha=0.55)
    for j in range(gw + 1):
        axT.axvline(j * T, color="#22d3ee", lw=0.5, alpha=0.55)
    axT.add_patch(Rectangle((tj * T, ti * T), T, T, facecolor="none",
                            edgecolor="#f59e0b", linewidth=3.0))
    axT.set_title(
        f"[1단계] {DS} normal {STEM} ({h}x{w}) — {T}px 비겹침 격자 {gh}x{gw}={gh*gw} 패치. "
        f"주황 = 대상 패치 (타일 i={ti}, j={tj})\n"
        f"이 이미지의 100패치는 {n_cells}개 서로 다른 cell로 갈린다. "
        f"대상 패치는 그중 최빈 cell({cell}, {n_target}/{gh*gw} 패치)에 속한다.",
        fontsize=9.5, pad=7)
    axT.set_xticks([]); axT.set_yticks([])
    for sp in axT.spines.values():
        sp.set_edgecolor("#111827"); sp.set_linewidth(1.3)

    # ===== 열 제목 =====
    for j, (t1, t2, col) in enumerate(COL_TITLE):
        axh = fig.add_subplot(gs[1, j + 1])
        axh.axis("off")
        axh.add_patch(Rectangle((0.02, 0.10), 0.96, 0.80, transform=axh.transAxes,
                                facecolor="#f9fafb", edgecolor=col, linewidth=2.0))
        axh.text(0.5, 0.64, t1, transform=axh.transAxes, ha="center", va="center",
                 fontsize=11.5, weight="bold", color=col)
        axh.text(0.5, 0.30, t2, transform=axh.transAxes, ha="center", va="center",
                 fontsize=8.2, color="#4b5563")

    def frame(ax, col):
        for sp in ax.spines.values():
            sp.set_edgecolor(col); sp.set_linewidth(1.7)

    for i, feat in enumerate(CONTEXT_FEATURES):
        gr = i + 2
        col = ROW_COLOR[feat]
        name, formula, target = ROW_DESC[feat]
        d = int(digits[i])
        e = bin_edges[feat]

        # ---- 좌여백 ----
        axL = fig.add_subplot(gs[gr, 0])
        axL.axis("off")
        axL.add_patch(Rectangle((0.0, 0.06), 0.05, 0.88, transform=axL.transAxes,
                                facecolor=col, edgecolor="none"))
        axL.text(0.12, 0.86, f"자리 {i}", transform=axL.transAxes, fontsize=9,
                 color="#6b7280", va="center")
        axL.text(0.12, 0.71, name, transform=axL.transAxes, fontsize=10.5,
                 weight="bold", color=col, va="center")
        axL.text(0.12, 0.56, formula, transform=axL.transAxes, fontsize=8.4,
                 color="#111827", va="center")
        axL.text(0.12, 0.38, target, transform=axL.transAxes, fontsize=8.0,
                 color="#4b5563", va="center")
        axL.text(0.12, 0.16, f"{vals[feat]:.4f}", transform=axL.transAxes,
                 fontsize=10.5, weight="bold", color=col, va="center")
        axL.add_patch(Rectangle((0.62, 0.08), 0.30, 0.17, transform=axL.transAxes,
                                facecolor=col, edgecolor="none"))
        axL.text(0.77, 0.165, f"bin {d}", transform=axL.transAxes, ha="center",
                 va="center", fontsize=11, weight="bold", color="white")

        # ---- 열1: Computation Map ----
        ax = fig.add_subplot(gs[gr, 1])
        if feat == "local_variance":
            ax.imshow(patch, cmap="gray")
            ax.set_title(f"원 패치 {T}x{T}  (mean={patch.mean():.1f})", fontsize=8.5)
        elif feat == "edge_density":
            ax.imshow(mid["mag"], cmap="magma")
            ax.set_title("Sobel |grad| 맵", fontsize=8.5)
        elif feat == "texture_entropy":
            ax.imshow(mid["lbp"], cmap="tab10")
            ax.set_title("LBP 코드 맵 (P=8, R=1, uniform)", fontsize=8.5)
        elif feat == "frequency_energy":
            ax.imshow(np.log1p(mid["fft_mag"]), cmap="viridis")
            ax.add_patch(Circle((mid["cx"], mid["cy"]), mid["r"], facecolor="none",
                                edgecolor=col, linewidth=2.0, linestyle="--"))
            ax.set_title(f"fftshift |F| (log). 원 안=저주파(r={mid['r']}), 밖=HF",
                         fontsize=8.5)
        else:
            ang = np.degrees(np.arctan2(
                cv2.Sobel(mid["p"], cv2.CV_64F, 0, 1, ksize=3),
                cv2.Sobel(mid["p"], cv2.CV_64F, 1, 0, ksize=3)))
            ax.imshow(ang, cmap="twilight", vmin=-180, vmax=180)
            ax.set_title("기울기 방향 맵 (-180~180도)", fontsize=8.5)
        ax.set_xticks([]); ax.set_yticks([])
        frame(ax, col)

        # ---- 열2: Reduced Distribution ----
        ax = fig.add_subplot(gs[gr, 2])
        if feat == "local_variance":
            ax.hist(patch.ravel(), bins=32, color=col, edgecolor="white", lw=0.3)
            ax.axvline(patch.mean(), color="#111827", lw=1.6, ls="--")
            ax.set_xlabel(f"픽셀 명도 (분산 = {vals[feat]:.2f})", fontsize=8.2)
        elif feat == "edge_density":
            ax.hist(mid["mag"].ravel(), bins=32, color=col, edgecolor="white", lw=0.3)
            ax.axvline(vals[feat], color="#111827", lw=1.8)
            ax.set_xlabel(f"|grad| (평균 = {vals[feat]:.3f})", fontsize=8.2)
        elif feat == "texture_entropy":
            ax.bar(range(10), mid["counts_lbp"], color=col, edgecolor="white", lw=0.4)
            ax.set_xticks(range(10))
            ax.set_xlabel(f"LBP 10-bin (엔트로피 = {vals[feat]:.3f} bits)", fontsize=8.2)
        elif feat == "frequency_energy":
            lf = float(mid["fft_mag"][mid["lf_mask"]].sum())
            hfe = float(mid["fft_mag"][~mid["lf_mask"]].sum())
            ax.bar(["저주파(원 안)", "고주파(원 밖)"], [lf, hfe],
                   color=["#cbd5e1", col], edgecolor="#111827", lw=0.7)
            ax.set_xlabel(f"FFT 에너지 (HF 비 = {vals[feat]:.4f})", fontsize=8.2)
        else:
            ctr = (mid["edges_a"][:-1] + mid["edges_a"][1:]) / 2
            ax.bar(np.degrees(ctr), mid["hist_a"], width=18,
                   color=col, edgecolor="white", lw=0.4)
            ax.set_xlabel(f"방향 18-bin (엔트로피 = {vals[feat]:.3f} bits)", fontsize=8.2)
        ax.tick_params(labelsize=7.5)
        ax.grid(axis="y", alpha=0.22)
        frame(ax, col)

        # ---- 열3: Bin Assignment ----
        ax = fig.add_subplot(gs[gr, 3])
        v = dist[feat]
        lo, hi = np.percentile(v, [0.5, 99.5])
        ax.hist(v[(v >= lo) & (v <= hi)], bins=45, color="#94a3b8",
                edgecolor="white", lw=0.3)
        yl = ax.get_ylim()[1]
        for k_, ee in enumerate(e):
            ax.axvline(ee, color="#111827", lw=1.5, ls="--")
            ax.text(ee, yl * 0.97, f" P{[33, 66][k_]}={ee:.3g}", fontsize=7.2,
                    rotation=90, va="top", color="#111827")
        ax.axvline(vals[feat], color=col, lw=2.6)
        for k_, (a, b) in enumerate([(lo, e[0]), (e[0], e[1]), (e[1], hi)]):
            ax.axvspan(a, b, color=col if k_ == d else "#e5e7eb",
                       alpha=0.20 if k_ == d else 0.10)
            ax.text((max(a, lo) + min(b, hi)) / 2, yl * 0.06, str(k_),
                    ha="center", fontsize=10,
                    weight="bold" if k_ == d else "normal",
                    color=col if k_ == d else "#9ca3af")
        ax.set_xlim(lo, hi)
        ax.set_xlabel(
            f"이 패치 {vals[feat]:.4f}  →  분위 {100*np.mean(v <= vals[feat]):.0f}%  →  bin {d}",
            fontsize=8.2)
        ax.tick_params(labelsize=7.5)
        frame(ax, col)

    # ===== 4단계: 하단 전폭 — cell key 조립 =====
    axB = fig.add_subplot(gs[7, :])
    axB.axis("off")
    axB.text(0.5, 0.86, "[4단계]  다섯 자리를 순서대로 이어 cell key를 만든다",
             transform=axB.transAxes, ha="center", fontsize=11, weight="bold")
    n = len(CONTEXT_FEATURES)
    for i, feat in enumerate(CONTEXT_FEATURES):
        xc = 0.30 + i * 0.09
        axB.add_patch(Rectangle((xc - 0.030, 0.30), 0.060, 0.34,
                                transform=axB.transAxes,
                                facecolor=ROW_COLOR[feat], edgecolor="none"))
        axB.text(xc, 0.47, digits[i], transform=axB.transAxes, ha="center",
                 va="center", fontsize=19, weight="bold", color="white")
        axB.text(xc, 0.21, feat.replace("_", "\n"), transform=axB.transAxes,
                 ha="center", va="top", fontsize=7.0, color="#4b5563")
        if i < n - 1:
            axB.text(xc + 0.045, 0.47, "_", transform=axB.transAxes, ha="center",
                     va="center", fontsize=17, color="#6b7280")
    axB.text(0.80, 0.47, f"c = {cell}", transform=axB.transAxes, ha="center",
             va="center", fontsize=17, weight="bold", color="#111827")
    axB.text(0.80, 0.18, f"전체 공간 = {N_CONTEXT_BINS}^{n} = {N_CONTEXT_BINS**n} cells\n"
                         f"severstal good 관측 = 208 cells",
             transform=axB.transAxes, ha="center", va="top", fontsize=8.2,
             color="#4b5563")

    fig.text(0.5, 0.016,
             f"판정 단위는 64px 패치다 — 이미지가 아니다. 이 이미지 {gh*gw}패치가 {n_cells}개 cell로 갈린다. "
             f"cell은 패치가 아니라 범주이며, severstal good 590,200 패치가 208 cell에 담긴다"
             f"(평균 약 2,838 patches/cell). tertile 경계는 데이터셋 자체 분포에서 유도된다.",
             fontsize=9.0, ha="center", color="#111827")

    out = os.path.join(OUT_DIR, "[figure 3.2.2 4] context_cell.png")
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print("saved:", out)
    print(f"  패치 타일 (i,j)=({ti},{tj})  cell={cell}")
    for i, f in enumerate(CONTEXT_FEATURES):
        print(f"  자리{i} {f:26s} = {vals[f]:10.4f}  P33={bin_edges[f][0]:.4g} "
              f"P66={bin_edges[f][1]:.4g}  → bin {digits[i]}")


if __name__ == "__main__":
    main()
