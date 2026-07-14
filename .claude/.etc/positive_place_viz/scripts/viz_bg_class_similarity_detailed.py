"""
상세판: 클래스별 배경정합 AROMA vs Random + 부가설명 (severstal).

기본판(viz_bg_class_similarity.py) 함수 재사용 + 추가:
  - violin + box 오버레이(분포 형태)
  - 클래스별 Mann-Whitney U 검정 p값(scipy 없으면 permutation)
  - metric 설명 박스 + 전체 요약(평균 Δ, 우위 클래스 수)
  - 표본수 주석
출력: .claude/.etc/positive_place_viz/1차/bg_class_similarity_severstal_detailed.png
"""
import os, sys, json, io
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import viz_bg_class_similarity as base   # 동일 metric 재사용

DS_ROOT = base.DS_ROOT
OUT = "D:/project/aroma/.claude/.etc/positive_place_viz/1차"


def _mannwhitney_p(a, b):
    try:
        from scipy.stats import mannwhitneyu
        return float(mannwhitneyu(a, b, alternative="greater").pvalue)  # H1: aroma>random
    except Exception:
        # permutation 폴백 (statistic = mean diff, one-sided aroma>random)
        rng = np.random.default_rng(0)
        obs = a.mean() - b.mean()
        pool = np.concatenate([a, b]); n = len(a)
        cnt = 0; R = 2000
        for _ in range(R):
            rng.shuffle(pool)
            if (pool[:n].mean() - pool[n:].mean()) >= obs:
                cnt += 1
        return (cnt + 1) / (R + 1)


def run():
    os.makedirs(OUT, exist_ok=True)
    sel = json.load(io.open(f"{DS_ROOT}/roi/severstal/clean_bg_selected.json", encoding="utf-8"))
    rnd = json.load(io.open(f"{DS_ROOT}/roi/severstal/clean_bg_random_arm.json", encoding="utf-8"))
    classes = sorted({s.get("class_value") for s in sel})
    refs = {c: base.build_class_ref(c) for c in classes}
    a_sim = base.collect(sel, refs)
    r_sim = base.collect(rnd, refs)

    fig, ax = plt.subplots(figsize=(2.1 * len(classes) + 3, 6.5))
    pos = np.arange(len(classes)); w = 0.38
    stats = []
    for i, c in enumerate(classes):
        a = np.array(a_sim.get(c, []), float); a = a[~np.isnan(a)]
        r = np.array(r_sim.get(c, []), float); r = r[~np.isnan(r)]
        # violin
        for data, off, col in ((a, -w/2, "#2c7fb8"), (r, +w/2, "#d95f0e")):
            vp = ax.violinplot([data], positions=[i + off], widths=w*0.95,
                               showmeans=True, showextrema=False)
            for b in vp["bodies"]:
                b.set_facecolor(col); b.set_alpha(0.55); b.set_edgecolor("k")
            vp["cmeans"].set_color("k")
        # box (얇게 오버레이)
        ax.boxplot([a], positions=[i - w/2], widths=w*0.4, showfliers=False,
                   patch_artist=True, boxprops=dict(facecolor="none", edgecolor="#14496b"),
                   medianprops=dict(color="#14496b"))
        ax.boxplot([r], positions=[i + w/2], widths=w*0.4, showfliers=False,
                   patch_artist=True, boxprops=dict(facecolor="none", edgecolor="#7a3407"),
                   medianprops=dict(color="#7a3407"))
        p = _mannwhitney_p(a, r)
        stats.append((c, a.mean(), r.mean(), a.mean()-r.mean(), p, len(a)))

    ax.set_xticks(pos)
    ax.set_xticklabels([c.replace("class", "c") for c in classes], fontsize=11)
    ax.set_ylabel("배경 정합도\n(real 클래스-c 배경과 텍스처 히스토그램 교집합, 0–1)", fontsize=10)
    ax.set_xlabel("defect class", fontsize=11)

    # per-class 통계 주석
    ymax = ax.get_ylim()[1]
    for i, (c, ma, mr, d, p, n) in enumerate(stats):
        star = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "n.s."))
        ax.text(i, ymax * 0.995,
                f"Δ={d:+.3f} {star}\np={p:.3g}\n(a{ma:.2f}/r{mr:.2f}, n={n})",
                ha="center", va="top", fontsize=8.2)

    mean_d = np.mean([s[3] for s in stats])
    n_win = sum(1 for s in stats if s[3] > 0)
    ax.set_title("severstal — 클래스별 배정배경의 'real 클래스 배경' 정합도: AROMA vs Random\n"
                 f"AROMA 가 real 클래스별 배경과 더 유사한 배경 선택 · "
                 f"{n_win}/{len(classes)} 클래스 우위, 평균 Δ={mean_d:+.3f}", fontsize=11)

    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor="#2c7fb8", alpha=0.6, label="AROMA (compat 선택)"),
                       Patch(facecolor="#d95f0e", alpha=0.6, label="Random (무작위 선택)")],
              loc="lower left", fontsize=9)

    # metric 설명 박스
    ax.text(1.012, 0.5,
            "◆ metric (독립, by-construction 회피)\n"
            "  · real 클래스-c 참조 = 실제 클래스-c\n"
            "    결함이미지 배경(mask 제외)의\n"
            "    텍스처 히스토그램 평균\n"
            "  · 각 ROI 배정배경(good 이미지)의\n"
            "    텍스처 히스토그램 → 참조와 교집합\n"
            "  · 텍스처 = intensity+gradient+\n"
            "    local-variance (AROMA compat 반영)\n\n"
            "◆ 해석\n"
            "  높을수록 AROMA/Random 이 그 클래스\n"
            "  실제 배경과 유사한 곳에 배치.\n"
            "  Δ>0 = AROMA 가 더 클래스-정합적\n"
            "  배경 선택 (placement 메커니즘 유효,\n"
            "  엔진독립 → copy-paste 무관).\n"
            "  *p<.05 **p<.01 ***p<.001 (MWU, 1-sided)",
            transform=ax.transAxes, fontsize=8, va="center", ha="left",
            bbox=dict(boxstyle="round,pad=0.5", fc="#f5f9ff", ec="#89a", alpha=0.95))

    ax.grid(axis="y", alpha=0.2)
    plt.subplots_adjust(right=0.72)
    outp = f"{OUT}/bg_class_similarity_severstal_detailed.png"
    plt.savefig(outp, dpi=150, bbox_inches="tight"); print("saved:", outp)
    for c, ma, mr, d, p, n in stats:
        print(f"  {c}: aroma={ma:.4f} random={mr:.4f} Δ={d:+.4f} p={p:.4g} n={n}")


if __name__ == "__main__":
    for f in ["Malgun Gothic", "AppleGothic", "NanumGothic"]:
        try: matplotlib.rcParams["font.family"] = f; break
        except Exception: pass
    matplotlib.rcParams["axes.unicode_minus"] = False
    run()
