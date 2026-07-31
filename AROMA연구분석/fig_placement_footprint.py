# -*- coding: utf-8 -*-
"""위치 결정 5단계 시각화 — footprint를 덮는 타일들의 compat 평균.

`aroma_core_compatibility_model_20260729.md` §6 보조 자료. 실제 운영 함수를 import해
재현성을 보장한다(재구현 금지):

  - scripts/distribution_profiling._extract_context_features / _context_cell_key
  - scripts/aroma/generate_defects._is_clean_background / _tile_anchors
  - 상수 _COMPAT_TILE(64) / _POS_STRIDE(32) / _POS_TOPK(8)

출력: fig_placement_footprint_<ds>.png (3 패널)
  A) normal 이미지 64px 타일 격자 + 타일별 compat(색) + cell key + void 표시
  B) 동일 crop을 서로 다른 위치에 둔 후보 3개 — footprint가 덮는 타일과 평균 compat
  C) 전체 후보를 mean-compat 내림차순 정렬 + top-K(8) 밴드 + void 탈락 표시
"""
import os
import sys
import json
import io

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.patches import Rectangle
import cv2

# 한글 라벨용 폰트. 없으면 DejaVu로 떨어지며 한글이 박스로 표시된다.
for _f in ("Malgun Gothic", "Gulim", "Batang", "HCR Dotum"):
    try:
        rcParams["font.family"] = _f
        break
    except Exception:
        continue
rcParams["axes.unicode_minus"] = False

REPO = os.environ.get("AROMA_REPO", "D:/project/aroma")
ROOT = os.environ.get("AROMA_DATASET_ROOT", "D:/project/aroma_dataset")
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))

from distribution_profiling import _extract_context_features, _context_cell_key  # noqa: E402
from scripts.aroma.generate_defects import (  # noqa: E402
    _is_clean_background,
    _tile_anchors,
    _COMPAT_TILE,
    _POS_STRIDE,
    _POS_TOPK,
)

# 시연 대상: severstal. cluster 1(elongated, P(k)=0.24)의 compat 행을 사용.
DS = "severstal"
NORMAL_STEM = "00031f466"      # §3에서 패치 100개가 27 cell에 흩어진 것으로 인용한 이미지
CLUSTER = "1"
CROP_WH = (160, 96)            # 예시 crop — x 3타일 x y 2타일 = footprint 6타일
MIN_BG_QUALITY = 0.5           # 운영 설정(0.7은 전 데이터셋 98%+ void → 하향)
BLUR_THRESHOLD = 100.0


def load_inputs():
    compat_p = f"{ROOT}/profiling/profiling/{DS}/compatibility_matrix.json"
    compat = json.load(io.open(compat_p, encoding="utf-8"))
    img_p = f"{ROOT}/{DS}/train/good/{NORMAL_STEM}.jpg"
    bgr = cv2.imdecode(np.fromfile(img_p, dtype=np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise SystemExit(f"이미지 로드 실패: {img_p}")
    return compat, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def tile_map(nrgb, compat_row, bin_edges):
    """비겹침 64px 격자의 (compat, cell, void). _positive_place._tile과 동일 계산."""
    h, w = nrgb.shape[:2]
    gh, gw = h // _COMPAT_TILE, w // _COMPAT_TILE
    compat = np.full((gh, gw), np.nan)
    void = np.zeros((gh, gw), bool)
    cells = np.empty((gh, gw), dtype=object)
    for i in range(gh):
        for j in range(gw):
            y, x = i * _COMPAT_TILE, j * _COMPAT_TILE
            win = nrgb[y:y + _COMPAT_TILE, x:x + _COMPAT_TILE]
            gray = cv2.cvtColor(win, cv2.COLOR_RGB2GRAY)
            void[i, j] = not _is_clean_background(
                gray, min_quality=MIN_BG_QUALITY, blur_threshold=BLUR_THRESHOLD
            )
            cell = _context_cell_key(_extract_context_features(gray), bin_edges)
            cells[i, j] = cell
            compat[i, j] = float(compat_row.get(cell, 0.5))
    return compat, cells, void


def score_candidate(nrgb, x, y, cw, ch, compat_row, bin_edges, cache):
    """_positive_place의 후보 채점과 동일: footprint를 덮는 타일 compat의 평균."""
    h, w = nrgb.shape[:2]
    anchors = [(ax, ay)
               for ay in _tile_anchors(y, ch, h, _COMPAT_TILE)
               for ax in _tile_anchors(x, cw, w, _COMPAT_TILE)]
    vals, has_void = [], False
    for ax, ay in anchors:
        hit = cache.get((ax, ay))
        if hit is None:
            win = nrgb[ay:ay + _COMPAT_TILE, ax:ax + _COMPAT_TILE]
            gray = cv2.cvtColor(win, cv2.COLOR_RGB2GRAY)
            v = not _is_clean_background(gray, min_quality=MIN_BG_QUALITY,
                                         blur_threshold=BLUR_THRESHOLD)
            c = float(compat_row.get(
                _context_cell_key(_extract_context_features(gray), bin_edges), 0.5))
            hit = (c, v)
            cache[(ax, ay)] = hit
        vals.append(hit[0])
        if hit[1]:
            has_void = True
    return (sum(vals) / len(vals) if vals else 0.5), has_void, anchors


def enumerate_candidates(nrgb, cw, ch, compat_row, bin_edges):
    h, w = nrgb.shape[:2]

    def axis(span, st):
        if span <= 0:
            return [0]
        pts = list(range(0, span + 1, st))
        if pts[-1] != span:
            pts.append(span)
        return pts

    xs, ys = axis(w - cw, _POS_STRIDE), axis(h - ch, _POS_STRIDE)
    cache, out = {}, []
    for y in ys:
        for x in xs:
            mean, has_void, anchors = score_candidate(
                nrgb, x, y, cw, ch, compat_row, bin_edges, cache)
            out.append({"x": x, "y": y, "mean": mean, "void": has_void,
                        "anchors": anchors})
    return out


def main():
    compat, nrgb = load_inputs()
    bin_edges = compat["bin_edges"]
    compat_row = compat["matrix_symmetric"][CLUSTER]
    cw, ch = CROP_WH

    cmap_grid, cells, void = tile_map(nrgb, compat_row, bin_edges)
    cands = enumerate_candidates(nrgb, cw, ch, compat_row, bin_edges)
    nonvoid = sorted([c for c in cands if not c["void"]],
                     key=lambda t: -t["mean"])
    dropped = [c for c in cands if c["void"]]

    # 패널 B에 보일 후보 3개: 최상위 / 중위 / 최하위 (비-void 중)
    picks = ([nonvoid[0], nonvoid[len(nonvoid) // 2], nonvoid[-1]]
             if len(nonvoid) >= 3 else nonvoid)
    tags = ["best", "median", "worst"]

    h, w = nrgb.shape[:2]
    fig = plt.figure(figsize=(16, 11))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.25, 1.25, 1.0], hspace=0.42)

    # ---------- A: 타일 격자 + compat ----------
    axA = fig.add_subplot(gs[0])
    axA.imshow(nrgb)
    gh, gw = cmap_grid.shape
    vmin, vmax = np.nanmin(cmap_grid), np.nanmax(cmap_grid)
    for i in range(gh):
        for j in range(gw):
            y, x = i * _COMPAT_TILE, j * _COMPAT_TILE
            c = cmap_grid[i, j]
            shade = 0.0 if vmax <= vmin else (c - vmin) / (vmax - vmin)
            axA.add_patch(Rectangle((x, y), _COMPAT_TILE, _COMPAT_TILE,
                                    facecolor=plt.cm.viridis(shade), alpha=0.45,
                                    edgecolor="white", linewidth=0.4))
            if void[i, j]:
                axA.add_patch(Rectangle((x, y), _COMPAT_TILE, _COMPAT_TILE,
                                        facecolor="none", edgecolor="red",
                                        hatch="///", linewidth=1.2))
            axA.text(x + 2, y + 12, f"{c:.2f}", color="white", fontsize=5.5)
            axA.text(x + 2, y + _COMPAT_TILE - 4, cells[i, j], color="#ffe08a",
                     fontsize=4.2)
    axA.set_title(
        f"A) {DS} normal {NORMAL_STEM} — 64px 비겹침 격자 ({gh}x{gw}={gh*gw} tiles). "
        f"셀 내 위=compat(cluster {CLUSTER}), 아래=cell key. "
        f"빨간 해칭=void. compat 범위 {vmin:.2f}~{vmax:.2f}", fontsize=10)
    axA.set_xlim(0, w); axA.set_ylim(h, 0); axA.axis("off")

    # ---------- B: 후보 3개의 footprint ----------
    axB = fig.add_subplot(gs[1])
    axB.imshow(nrgb)
    colors = ["#22c55e", "#eab308", "#ef4444"]
    for (cd, tag, col) in zip(picks, tags, colors):
        axB.add_patch(Rectangle((cd["x"], cd["y"]), cw, ch, facecolor="none",
                                edgecolor=col, linewidth=2.4))
        for (ax_, ay_) in cd["anchors"]:
            axB.add_patch(Rectangle((ax_, ay_), _COMPAT_TILE, _COMPAT_TILE,
                                    facecolor=col, alpha=0.16,
                                    edgecolor=col, linewidth=0.8, linestyle=":"))
        axB.text(cd["x"], max(cd["y"] - 6, 8),
                 f"{tag}  mean={cd['mean']:.3f}  tiles={len(cd['anchors'])}",
                 color=col, fontsize=9, weight="bold")
    axB.set_title(
        f"B) 같은 crop {cw}x{ch}를 서로 다른 위치에 둔 후보 — 실선=footprint, "
        f"점선 음영=_tile_anchors가 반환한 덮는 타일. "
        f"점수 = 그 타일들의 compat 평균 (동일 cell 타일이어도 이웃 조합이 달라 평균이 갈린다)",
        fontsize=10)
    axB.set_xlim(0, w); axB.set_ylim(h, 0); axB.axis("off")

    # ---------- C: 후보 순위 + top-K ----------
    axC = fig.add_subplot(gs[2])
    means = [c["mean"] for c in nonvoid]
    axC.plot(range(1, len(means) + 1), means, color="#2c7fb8", linewidth=1.6,
             label=f"비-void 후보 {len(nonvoid)}개 (mean-compat 내림차순)")
    k = max(1, min(_POS_TOPK, len(means)))
    axC.axvspan(0.5, k + 0.5, color="#22c55e", alpha=0.22,
                label=f"top-K = {_POS_TOPK} (여기서 rng.choice)")
    axC.axhline(means[0], color="#16a34a", linestyle="--", linewidth=1.0,
                label=f"best non-void mean = {means[0]:.3f} (τ 게이트 판정 기준)")
    for cd, tag, col in zip(picks, tags, colors):
        r = nonvoid.index(cd) + 1
        axC.plot([r], [cd["mean"]], "o", color=col, markersize=7)
        axC.annotate(f"{tag} (#{r})", (r, cd["mean"]), textcoords="offset points",
                     xytext=(6, 6), color=col, fontsize=9)
    axC.set_xlabel("후보 순위", fontsize=9)
    axC.set_ylabel("footprint mean-compat", fontsize=9)
    axC.set_title(
        f"C) 순위와 top-K 추첨. 전체 후보 {len(cands)}개, void 포함 탈락 {len(dropped)}개 "
        f"(footprint에 void 타일이 1개라도 있으면 compat과 무관하게 제외). "
        f"stride={_POS_STRIDE}"
        + ("   [주의] 현 환경에서 _is_clean_background가 cv2 dtype 예외로 fail-open → void 0건"
           if len(dropped) == 0 else ""), fontsize=10)
    axC.grid(alpha=0.25)
    axC.legend(fontsize=8, loc="upper right")

    out = os.path.join(OUT_DIR, f"fig_placement_footprint_{DS}.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("saved:", out)
    print(f"  후보 총 {len(cands)} / 비-void {len(nonvoid)} / void 탈락 {len(dropped)}")
    print(f"  mean-compat  best={means[0]:.4f}  median={means[len(means)//2]:.4f}  worst={means[-1]:.4f}")
    for cd, tag in zip(picks, tags):
        print(f"  {tag:7s} pos=({cd['x']},{cd['y']}) mean={cd['mean']:.4f} tiles={len(cd['anchors'])}")


if __name__ == "__main__":
    main()
