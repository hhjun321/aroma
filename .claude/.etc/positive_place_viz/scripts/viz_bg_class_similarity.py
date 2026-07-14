"""
클래스별 배경 정합도 figure: AROMA vs Random (독립 metric, severstal).

주장: AROMA 는 각 클래스 결함이 실제 놓이는 배경(real 클래스-c 배경 분포)과
유사한 배경을 골라 배치, random 은 무작위 → aroma 배정배경이 real 클래스-c 배경에
더 유사.

metric(독립 — pipeline class_fit 아님, by-construction 회피):
  real 클래스-c 참조 = real 클래스-c 결함이미지의 배경(mask==0) 회색 히스토그램 평균.
  각 ROI 배정배경(good 이미지) 회색 히스토그램 → real 클래스-c 참조와 히스토그램 교집합.
  clean_bg_selected(aroma) vs clean_bg_random_arm(random), 클래스별 box/violin.

출력: .claude/.etc/positive_place_viz/bg_class_similarity_<ds>.png
"""
import os, sys, json, glob, io
import numpy as np, cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DS_ROOT = "D:/project/aroma_dataset"
OUT = "D:/project/aroma/.claude/.etc/positive_place_viz"
BINS = 32
REF_SAMPLE = 120          # 클래스별 참조 산정 표본(속도)
EXTS = ("png", "jpg", "jpeg", "bmp")


def _bg_descriptor(gray, mask=None):
    """배경 descriptor = [회색 intensity | gradient-mag | local-variance] 히스토그램 concat.
    AROMA compat 게이트가 텍스처(분산/엣지) 기반이므로 intensity 만으론 부족 —
    gradient-magnitude + local-variance 를 더해 텍스처를 포착(cv2, GPU 불요, 독립 metric)."""
    g = gray.astype(np.float32)
    # gradient magnitude
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    gm = np.hypot(gx, gy)
    # local variance (5x5)
    k = np.ones((5, 5), np.float32) / 25.0
    lm = cv2.filter2D(g, -1, k)
    lv = cv2.filter2D((g - lm) ** 2, -1, k)
    sel = mask if (mask is not None) else np.ones(gray.shape, bool)
    if sel.sum() == 0:
        return None
    parts = []
    for arr, rng in ((g, (0, 256)), (gm, (0, 400)), (lv, (0, 2000))):
        h, _ = np.histogram(arr[sel], bins=BINS, range=rng)
        s = h.sum()
        parts.append(h / s if s > 0 else h)
    return np.concatenate(parts) / len(parts)   # 합=1 → intersection 0-1


def _gray_hist(gray, mask=None):
    return _bg_descriptor(gray, mask)


def _hist_inter(a, b):
    return float(np.minimum(a, b).sum()) if a is not None and b is not None else np.nan


def _find(sub, stem):
    for e in EXTS:
        p = f"{DS_ROOT}/severstal/{sub}/{stem}.{e}"
        if os.path.exists(p):
            return p
    hits = glob.glob(f"{DS_ROOT}/severstal/{sub}/{stem}.*")
    return hits[0] if hits else None


def build_class_ref(cls):
    """real 클래스-c 결함이미지 배경(mask==0) 평균 회색 히스토그램."""
    imgs = sorted(glob.glob(f"{DS_ROOT}/severstal/test/{cls}/*"))
    rng = np.random.default_rng(42)
    if len(imgs) > REF_SAMPLE:
        imgs = [imgs[i] for i in rng.permutation(len(imgs))[:REF_SAMPLE]]
    hists = []
    for p in imgs:
        g = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if g is None:
            continue
        stem = os.path.splitext(os.path.basename(p))[0]
        mp = _find("masks", stem) or _find(f"masks/{cls}", stem)
        m = cv2.imread(mp, cv2.IMREAD_GRAYSCALE) if mp else None
        bg = (m == 0) if (m is not None and m.shape == g.shape) else None  # 결함 제외 배경
        h = _gray_hist(g, bg)
        if h is not None:
            hists.append(h)
    return np.mean(hists, axis=0) if hists else None


_GOOD_CACHE = {}
def good_hist(normal_id):
    stem = normal_id[1:] if normal_id.startswith("_") else normal_id
    if stem in _GOOD_CACHE:
        return _GOOD_CACHE[stem]
    p = _find("train/good", stem)
    g = cv2.imread(p, cv2.IMREAD_GRAYSCALE) if p else None
    h = _gray_hist(g) if g is not None else None
    _GOOD_CACHE[stem] = h
    return h


def collect(anns, refs):
    """ROI 별 (class, 배정배경 vs real-class 참조 유사도)."""
    out = {}
    for a in anns:
        cls = a.get("class_value")
        if cls not in refs or refs[cls] is None:
            continue
        gh = good_hist(a.get("assigned_normal_id", ""))
        if gh is None:
            continue
        out.setdefault(cls, []).append(_hist_inter(gh, refs[cls]))
    return out


def run():
    sel = json.load(io.open(f"{DS_ROOT}/roi/severstal/clean_bg_selected.json", encoding="utf-8"))
    rnd = json.load(io.open(f"{DS_ROOT}/roi/severstal/clean_bg_random_arm.json", encoding="utf-8"))
    classes = sorted({s.get("class_value") for s in sel})
    print("classes:", classes)
    refs = {c: build_class_ref(c) for c in classes}
    a_sim = collect(sel, refs)
    r_sim = collect(rnd, refs)

    # figure — 클래스별 box(aroma/random) 병치
    fig, ax = plt.subplots(figsize=(1.6 * len(classes) + 2, 5))
    pos = np.arange(len(classes))
    w = 0.34
    a_data = [np.array(a_sim.get(c, []), float) for c in classes]
    r_data = [np.array(r_sim.get(c, []), float) for c in classes]
    a_data = [d[~np.isnan(d)] for d in a_data]
    r_data = [d[~np.isnan(d)] for d in r_data]
    bpa = ax.boxplot(a_data, positions=pos - w/2, widths=w, patch_artist=True,
                     showmeans=True, meanline=True)
    bpr = ax.boxplot(r_data, positions=pos + w/2, widths=w, patch_artist=True,
                     showmeans=True, meanline=True)
    for b in bpa["boxes"]: b.set(facecolor="#2c7fb8", alpha=0.7)
    for b in bpr["boxes"]: b.set(facecolor="#d95f0e", alpha=0.7)
    ax.set_xticks(pos)
    ax.set_xticklabels([c.replace("class", "c") for c in classes])
    ax.set_ylabel("배경 유사도 (real 클래스-c 배경과 histogram intersection)")
    ax.set_xlabel("defect class")
    ax.set_title("severstal — 클래스별 배정배경의 real-클래스 배경 정합도\n"
                 "AROMA(파랑) vs Random(주황)  · 높을수록 real 클래스 배경에 유사")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor="#2c7fb8", label="AROMA"),
                       Patch(facecolor="#d95f0e", label="Random")], loc="lower right")
    # 클래스별 mean Δ 주석
    for i, c in enumerate(classes):
        ma = np.mean(a_data[i]) if len(a_data[i]) else np.nan
        mr = np.mean(r_data[i]) if len(r_data[i]) else np.nan
        ax.text(i, ax.get_ylim()[1] * 0.99,
                f"Δ={ma-mr:+.3f}\n(a{ma:.2f}/r{mr:.2f})",
                ha="center", va="top", fontsize=8)
    plt.tight_layout()
    outp = f"{OUT}/bg_class_similarity_severstal.png"
    plt.savefig(outp, dpi=140)
    print("saved:", outp)
    # 수치 요약
    for c in classes:
        a = np.array(a_sim.get(c, []), float); a = a[~np.isnan(a)]
        r = np.array(r_sim.get(c, []), float); r = r[~np.isnan(r)]
        print(f"  {c}: aroma mean={a.mean():.4f} (n={len(a)})  random mean={r.mean():.4f} (n={len(r)})  Δ={a.mean()-r.mean():+.4f}")


if __name__ == "__main__":
    # 한글 폰트
    for f in ["Malgun Gothic", "AppleGothic", "NanumGothic"]:
        try:
            matplotlib.rcParams["font.family"] = f; break
        except Exception:
            pass
    matplotlib.rcParams["axes.unicode_minus"] = False
    run()
