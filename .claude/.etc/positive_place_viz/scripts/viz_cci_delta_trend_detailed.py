"""
상세판 Headline figure: Δ(AROMA-Random) vs CCI + 부가설명.

기본판(viz_cci_delta_trend.py)에 추가:
  - headroom/near-ceiling 존 배경 음영 + 라벨
  - per-point 상세 주석(CCI·Δ·t·baseline·mode·부호)
  - 추세선 slope·R²(headroom 셋)
  - thesis 설명 박스 + 콜아웃 화살표
출력: .claude/.etc/positive_place_viz/1차/cci_delta_trend_detailed.png
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "D:/project/aroma/.claude/.etc/positive_place_viz/1차"

# (name, CCI, Δ(A-R), baseline, mode, t(df=2), signs)
DATA = [
    ("AITeX",         0.440, +0.050, 0.204, "single", 1.03, "2/3"),
    ("Severstal",     0.306, +0.011, 0.496, "multi",  1.15, "2/3"),
    ("MTD",           0.246, -0.003, 0.909, "multi", -0.16, "2/3"),
    ("Kolektor",      0.224, -0.007, 0.974, "single", -0.79, "1/3"),
    ("MVTec Leather", 0.200, -0.049, 0.832, "multi", -2.28, "0/3"),
]
CEIL = 0.85


def run():
    os.makedirs(OUT, exist_ok=True)
    nm = [d[0] for d in DATA]
    cci = np.array([d[1] for d in DATA]); delta = np.array([d[2] for d in DATA])
    base = np.array([d[3] for d in DATA]); mode = [d[4] for d in DATA]
    t = [d[5] for d in DATA]; sg = [d[6] for d in DATA]
    hr = base < CEIL

    fig, ax = plt.subplots(figsize=(11, 7))

    # 존 음영: Δ>0(이득) / Δ<0
    ax.axhspan(0, 0.07, color="#e8f4ea", alpha=0.5, zorder=0)
    ax.axhspan(-0.07, 0, color="#fbeae5", alpha=0.5, zorder=0)
    ax.axhline(0, color="#555", lw=1.2, ls="--", zorder=1)
    ax.text(0.305, 0.064, "AROMA 이득 영역 (Δ>0)", fontsize=9.5, color="#2a7", va="top", ha="center")
    ax.text(0.305, -0.064, "AROMA 열위/수렴 (Δ≤0)", fontsize=9.5, color="#c53", va="bottom", ha="center")

    # 겹침 방지 per-point 주석 오프셋 (offset points)
    OFFS = {"AITeX": (-6, 14, "right", "bottom"),
            "Severstal": (-8, 16, "right", "bottom"),
            "MTD": (16, 26, "left", "bottom"),
            "Kolektor": (-14, -30, "right", "top"),
            "MVTec Leather": (14, -8, "left", "top")}
    for i in range(len(DATA)):
        color = "#2c7fb8" if hr[i] else "#d95f0e"
        marker = "o" if mode[i] == "single" else "s"
        ax.scatter(cci[i], delta[i], s=320, c=color, marker=marker,
                   edgecolor="k", linewidth=1.4, zorder=4)
        sign = "양(+)" if delta[i] > 0.002 else ("음(-)" if delta[i] < -0.002 else "~0")
        txt = (f"{nm[i]}  [{mode[i]}]\n"
               f"CCI={cci[i]:.3f}  Δ={delta[i]:+.3f}\n"
               f"base={base[i]:.2f} {'(headroom)' if hr[i] else '(near-ceiling)'}\n"
               f"t={t[i]:+.2f}  seed {sg[i]}  {sign}")
        ox, oy, ha, va = OFFS.get(nm[i], (10, 12, "left", "bottom"))
        ax.annotate(txt, (cci[i], delta[i]), textcoords="offset points",
                    xytext=(ox, oy), fontsize=8, va=va, ha=ha,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=color, alpha=0.92),
                    arrowprops=dict(arrowstyle="-", color=color, lw=0.8))

    # 추세선 (headroom 셋) + R²
    idx = np.where(hr)[0]
    z = np.polyfit(cci[idx], delta[idx], 1)
    yhat = np.polyval(z, cci[idx])
    ss_res = np.sum((delta[idx] - yhat) ** 2)
    ss_tot = np.sum((delta[idx] - delta[idx].mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    xs = np.linspace(cci.min() - 0.015, cci.max() + 0.02, 50)
    ax.plot(xs, np.polyval(z, xs), color="#2c7fb8", lw=2, alpha=0.55, zorder=2)
    ax.text(0.40, np.polyval(z, 0.40) - 0.008,
            f"추세(headroom 셋)\nslope={z[0]:+.3f}, R²={r2:.2f}",
            fontsize=8.5, color="#2c7fb8")

    # thesis 박스
    ax.text(0.198, 0.045,
            "① 이득엔 두 조건이 함께 필요:\n"
            "   headroom(base<0.85) AND 높은 CCI\n"
            "② headroom 셋(파랑): Δ가 CCI 따라 상승\n"
            "③ near-ceiling(주황): CCI 무관 flat~0",
            fontsize=9, va="top",
            bbox=dict(boxstyle="round,pad=0.5", fc="#fffbe6", ec="#caa", alpha=0.95))

    ax.set_xlabel("Context Complexity Index (CCI) — 표면/문맥 이질성 (높을수록 배경 다양)", fontsize=11)
    ax.set_ylabel("Δ(AROMA - Random)  mAP@0.5", fontsize=11)
    ax.set_title("이득의 조건부성: AROMA ROI-선택의 downstream Δ vs 표면 이질성(CCI)\n"
                 "5 industrial datasets · training-free copy-paste engine · YOLOv8n · 3 seeds",
                 fontsize=12)
    from matplotlib.lines import Line2D
    leg = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#2c7fb8", markeredgecolor="k",
               markersize=12, label="headroom 있음 (base<0.85)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#d95f0e", markeredgecolor="k",
               markersize=12, label="near-ceiling (base≥0.85)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="gray", markeredgecolor="k",
               markersize=11, label="○ single-class"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="gray", markeredgecolor="k",
               markersize=10, label="□ multi-class"),
    ]
    ax.legend(handles=leg, loc="lower right", fontsize=9, framealpha=0.95)
    ax.set_ylim(-0.07, 0.07)
    ax.grid(alpha=0.2)
    plt.tight_layout()
    outp = f"{OUT}/cci_delta_trend_detailed.png"
    plt.savefig(outp, dpi=150); print("saved:", outp)


if __name__ == "__main__":
    for f in ["Malgun Gothic", "AppleGothic", "NanumGothic"]:
        try: matplotlib.rcParams["font.family"] = f; break
        except Exception: pass
    matplotlib.rcParams["axes.unicode_minus"] = False
    run()
