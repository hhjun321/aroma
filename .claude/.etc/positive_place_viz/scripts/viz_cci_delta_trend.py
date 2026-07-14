"""
Headline figure: Δ(AROMA-Random) vs CCI — 이득의 조건부성(thesis).

thesis(정정본): AROMA 다운스트림 이득은 baseline headroom AND 표면 이질성(CCI)
둘 다 충족 시에만. 5셋 exp4v2(copy-paste) 실측:
  x=CCI, y=Δ(A−R) mAP50. 색=headroom 유무(near-ceiling 은 CCI 무관 flat).
  → Δ 가 CCI 따라 상승하되, 양(+)은 headroom 있는 셋(aitex/severstal)만.

출력: .claude/.etc/positive_place_viz/cci_delta_trend.png
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "D:/project/aroma/.claude/.etc/positive_place_viz"

# (name, CCI, Δ(A−R), baseline_map50, mode)  — CONTRACT canonical
DATA = [
    ("AITeX",         0.440, +0.050, 0.204, "single"),
    ("Severstal",     0.306, +0.011, 0.496, "multi"),
    ("MTD",           0.246, -0.003, 0.909, "multi"),
    ("Kolektor",      0.224, -0.007, 0.974, "single"),
    ("MVTec Leather", 0.200, -0.049, 0.832, "multi"),
]
CEIL = 0.85   # baseline ≥ CEIL → near-ceiling(headroom 없음)


def run():
    names = [d[0] for d in DATA]
    cci = np.array([d[1] for d in DATA])
    delta = np.array([d[2] for d in DATA])
    base = np.array([d[3] for d in DATA])
    modes = [d[4] for d in DATA]
    headroom = base < CEIL   # True=headroom 있음

    fig, ax = plt.subplots(figsize=(8.2, 5.6))
    ax.axhline(0, color="#888", lw=1, ls="--", zorder=1)

    for i, nm in enumerate(names):
        has_hr = headroom[i]
        color = "#2c7fb8" if has_hr else "#d95f0e"
        marker = "o" if modes[i] == "single" else "s"
        ax.scatter(cci[i], delta[i], s=260, c=color, marker=marker,
                   edgecolor="k", linewidth=1.2, zorder=3)
        dy = 0.006 if delta[i] >= 0 else -0.010
        ax.annotate(f"{nm}\n(base {base[i]:.2f}, {modes[i]})",
                    (cci[i], delta[i]), textcoords="offset points",
                    xytext=(8, 10 if delta[i] >= 0 else -22), fontsize=8.5)

    # 추세선 (headroom 있는 셋만 — 정직: near-ceiling 은 CCI-Δ 관계 교란)
    hr_idx = np.where(headroom)[0]
    if len(hr_idx) >= 2:
        z = np.polyfit(cci[hr_idx], delta[hr_idx], 1)
        xs = np.linspace(cci.min() - 0.01, cci.max() + 0.01, 50)
        ax.plot(xs, np.polyval(z, xs), color="#2c7fb8", lw=1.5, alpha=0.5, zorder=2,
                label="추세(headroom 있는 셋)")

    ax.set_xlabel("Context Complexity Index (CCI) — 표면/문맥 이질성")
    ax.set_ylabel("Δ(AROMA - Random)  mAP@0.5")
    ax.set_title("이득의 조건부성: Δ(AROMA-Random) vs CCI (5셋, copy-paste)\n"
                 "양(+)은 headroom AND 이질성 둘 다 충족 시만 — 파랑=headroom, 주황=near-ceiling")
    from matplotlib.lines import Line2D
    leg = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#2c7fb8",
               markeredgecolor="k", markersize=11, label="headroom 있음 (base<0.85)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#d95f0e",
               markeredgecolor="k", markersize=11, label="near-ceiling (base≥0.85)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="gray",
               markeredgecolor="k", markersize=11, label="○ single-class"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="gray",
               markeredgecolor="k", markersize=10, label="□ multi-class"),
    ]
    ax.legend(handles=leg, loc="upper left", fontsize=8.5, framealpha=0.9)
    ax.grid(alpha=0.25)
    plt.tight_layout()
    outp = f"{OUT}/cci_delta_trend.png"
    plt.savefig(outp, dpi=150)
    print("saved:", outp)


if __name__ == "__main__":
    for f in ["Malgun Gothic", "AppleGothic", "NanumGothic"]:
        try:
            matplotlib.rcParams["font.family"] = f; break
        except Exception:
            pass
    matplotlib.rcParams["axes.unicode_minus"] = False
    run()
