"""
Figure: Background-selection compatibility (AROMA vs Random) across five datasets.

Claim
-----
Under the training-free copy-paste engine, AROMA and Random share the SAME defect
ROIs and differ only in the normal-background image each ROI is assigned to. This
figure quantifies whether AROMA's assigned backgrounds resemble the real defect
backgrounds of each dataset more than Random's do — an ENGINE-INDEPENDENT test of
the placement/selection mechanism (valid under copy-paste, unlike distribution
fidelity metrics).

Metric (independent, dataset-pooled reference)
----------------------------------------------
  * Reference (per dataset): the mean texture histogram of the background region
    (mask-excluded pixels) pooled over the real defect images — class-agnostic.
  * For each ROI: the assigned normal (good) image's texture histogram, intersected
    with the dataset reference (histogram intersection, 0..1).
  * Texture histogram = [intensity | gradient-magnitude | local-variance] (captures
    the texture cues AROMA's compatibility gate uses; not just intensity).
  * AROMA assignments from clean_bg_selected.json, Random from clean_bg_random_arm.json.

One grouped box/violin pair per dataset (ordered by descending CCI). Per-dataset
Δ = mean(AROMA) - mean(Random) and a one-sided Mann-Whitney U p-value (H1: AROMA>Random)
are annotated. Statistical significance here is robust (n = hundreds/thousands of ROIs),
in contrast to the n=3 seed downstream comparison.

Data root: D:/project/aroma_dataset (local mirror of the committed pipeline outputs).
Output: ../image/[figure 4.1 3] bg_similarity_datasets.png
"""
import os, glob, json, io
import numpy as np, cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DS_ROOT = os.environ.get("AROMA_DATASET_ROOT", "D:/project/aroma_dataset")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "image")
BINS = 32
REF_SAMPLE = 150
EXTS = ("png", "jpg", "jpeg", "bmp")

# name -> (roi_key, data_root, good_dir, mask_scheme, CCI)  ordered by CCI desc
DATASETS = [
    ("AITeX",         "aitex",         "aitex_tiled",   "aitex_tiled/train/good",   "gt_type", 0.440),
    ("Severstal",     "severstal",     "severstal",     "severstal/train/good",     "severstal", 0.306),
    ("MTD",           "mtd",           "mtd",           "mtd/train/good",           "gt_type", 0.246),
    ("Kolektor",      "kolektor",      "kolektor",      "kolektor/train/good",      "gt_type", 0.224),
    ("MVTec Leather", "mvtec_leather", "mvtec_leather", "mvtec_leather/train/good", "gt_type", 0.200),
]


def _texture_desc(gray, mask=None):
    """[intensity | gradient-magnitude | local-variance] histogram, sum=1."""
    g = gray.astype(np.float32)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3); gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    gm = np.hypot(gx, gy)
    k = np.ones((5, 5), np.float32) / 25.0
    lm = cv2.filter2D(g, -1, k); lv = cv2.filter2D((g - lm) ** 2, -1, k)
    sel = mask if mask is not None else np.ones(gray.shape, bool)
    if sel.sum() == 0:
        return None
    parts = []
    for arr, rng in ((g, (0, 256)), (gm, (0, 400)), (lv, (0, 2000))):
        h, _ = np.histogram(arr[sel], bins=BINS, range=rng); s = h.sum()
        parts.append(h / s if s > 0 else h)
    return np.concatenate(parts) / len(parts)


def _inter(a, b):
    return float(np.minimum(a, b).sum()) if a is not None and b is not None else np.nan


def _glob_defects(data_root):
    troot = os.path.join(DS_ROOT, data_root, "test")
    out = []
    if os.path.isdir(troot):
        for t in sorted(os.listdir(troot)):
            d = os.path.join(troot, t)
            if os.path.isdir(d) and t != "good":
                out += [(t, p) for p in sorted(glob.glob(f"{d}/*")) if p.lower().endswith(EXTS)]
    return out


def _mask_of(data_root, scheme, dtype, defect_path):
    stem = os.path.splitext(os.path.basename(defect_path))[0]
    root = os.path.join(DS_ROOT, data_root)
    cands = []
    if scheme == "severstal":
        cands = [f"{root}/masks/{stem}.png", f"{root}/masks/{dtype}/{stem}.png"]
    else:  # gt_type: ground_truth/{type}/{stem}_mask.png
        cands = [f"{root}/ground_truth/{dtype}/{stem}_mask.png",
                 f"{root}/ground_truth/{dtype}/{stem}.png"]
    for c in cands:
        if os.path.exists(c):
            return c
    return None


def build_ref(data_root, scheme):
    defs = _glob_defects(data_root)
    rng = np.random.default_rng(42)
    if len(defs) > REF_SAMPLE:
        defs = [defs[i] for i in rng.permutation(len(defs))[:REF_SAMPLE]]
    hists = []
    for dtype, p in defs:
        g = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if g is None:
            continue
        mp = _mask_of(data_root, scheme, dtype, p)
        m = cv2.imread(mp, cv2.IMREAD_GRAYSCALE) if mp else None
        bg = (m == 0) if (m is not None and m.shape == g.shape) else None
        h = _texture_desc(g, bg)
        if h is not None:
            hists.append(h)
    return np.mean(hists, axis=0) if hists else None


def _good_hist(good_dir, nid, cache):
    stem = nid[1:] if nid.startswith("_") else nid
    if stem in cache:
        return cache[stem]
    hit = None
    for e in EXTS:
        p = os.path.join(DS_ROOT, good_dir, f"{stem}.{e}")
        if os.path.exists(p):
            hit = p; break
    if hit is None:
        g = glob.glob(os.path.join(DS_ROOT, good_dir, f"{stem}.*"))
        hit = g[0] if g else None
    im = cv2.imread(hit, cv2.IMREAD_GRAYSCALE) if hit else None
    h = _texture_desc(im) if im is not None else None
    cache[stem] = h
    return h


def collect(roi_key, good_dir, ref, fname):
    p = os.path.join(DS_ROOT, "roi", roi_key, fname)
    if not os.path.exists(p):
        return np.array([])
    anns = json.load(io.open(p, encoding="utf-8"))
    cache = {}
    vals = [_inter(_good_hist(good_dir, a.get("assigned_normal_id", ""), cache), ref) for a in anns]
    v = np.array([x for x in vals if not np.isnan(x)], float)
    return v


def _mwu_p(a, b):
    try:
        from scipy.stats import mannwhitneyu
        return float(mannwhitneyu(a, b, alternative="greater").pvalue)
    except Exception:
        rng = np.random.default_rng(0); obs = a.mean() - b.mean()
        pool = np.concatenate([a, b]); n = len(a); c = 0; R = 1000
        for _ in range(R):
            rng.shuffle(pool)
            if pool[:n].mean() - pool[n:].mean() >= obs:
                c += 1
        return (c + 1) / (R + 1)


def run():
    os.makedirs(OUT, exist_ok=True)
    names, a_all, r_all, stats = [], [], [], []
    for name, key, root, good, scheme, cci in DATASETS:
        ref = build_ref(root, scheme)
        a = collect(key, good, ref, "clean_bg_selected.json")
        r = collect(key, good, ref, "clean_bg_random_arm.json")
        if len(a) == 0 or len(r) == 0:
            print(f"skip {name} (a={len(a)} r={len(r)})"); continue
        names.append(name); a_all.append(a); r_all.append(r)
        p = _mwu_p(a, r)
        stats.append((name, cci, a.mean(), r.mean(), a.mean() - r.mean(), p, len(a)))
        print(f"{name:15} CCI={cci} AROMA={a.mean():.3f} Random={r.mean():.3f} Δ={a.mean()-r.mean():+.3f} p={p:.3g} n={len(a)}")

    fig, ax = plt.subplots(figsize=(2.0 * len(names) + 2, 6.2))
    pos = np.arange(len(names)); w = 0.34
    for i in range(len(names)):
        for data, off, col in ((a_all[i], -w/2, "#2c7fb8"), (r_all[i], +w/2, "#d95f0e")):
            vp = ax.violinplot([data], positions=[i + off], widths=w*0.95, showmeans=True, showextrema=False)
            for b in vp["bodies"]:
                b.set_facecolor(col); b.set_alpha(0.55); b.set_edgecolor("k")
            vp["cmeans"].set_color("k")
        ax.boxplot([a_all[i]], positions=[i - w/2], widths=w*0.4, showfliers=False,
                   patch_artist=True, boxprops=dict(facecolor="none", edgecolor="#14496b"),
                   medianprops=dict(color="#14496b"))
        ax.boxplot([r_all[i]], positions=[i + w/2], widths=w*0.4, showfliers=False,
                   patch_artist=True, boxprops=dict(facecolor="none", edgecolor="#7a3407"),
                   medianprops=dict(color="#7a3407"))

    ymax = ax.get_ylim()[1]
    for i, (name, cci, ma, mr, d, p, n) in enumerate(stats):
        star = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "n.s."))
        ax.text(i, ymax * 0.995, f"$\\Delta$={d:+.3f} {star}\np={p:.2g}\nn={n}",
                ha="center", va="top", fontsize=8.5)

    ax.set_xticks(pos)
    ax.set_xticklabels([f"{n}\n(CCI={c:.2f})" for n, c, *_ in stats], fontsize=10)
    ax.set_ylabel("Background compatibility", fontsize=10)
    ax.set_title("Background selection: AROMA assigns backgrounds more similar to the real defect background than Random",
                 fontsize=11)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor="#2c7fb8", alpha=0.6, label="AROMA (compatibility-selected)"),
                       Patch(facecolor="#d95f0e", alpha=0.6, label="Random (uniform)")],
              loc="lower left", fontsize=9)
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    outp = os.path.join(OUT, "[figure 4.1 3] bg_similarity_datasets.png")
    plt.savefig(outp, dpi=150)
    print("saved:", os.path.abspath(outp))


if __name__ == "__main__":
    run()
