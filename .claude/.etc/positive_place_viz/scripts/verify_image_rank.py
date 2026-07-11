"""
이미지-레벨 compat 랭킹 검증 (leather/mtd 실데이터) — 커밋 23bd373 함수 직접 사용.
(1) 수치: cluster별 ranked top-K compat vs uniform 무작위 표본.
(2) 시각 PNG: cluster별 TOP-compat 6장 vs BOTTOM-compat 6장 clean 이미지 몽타주.
"""
import sys, os, glob, json, random
import numpy as np, cv2
sys.path.insert(0, "D:/project/aroma/scripts"); sys.path.insert(0, "D:/project/aroma/scripts/aroma")
import distribution_profiling as dp
import generate_defects as gd

ETC = "D:/project/aroma/.claude/.etc"
OUT = f"{ETC}/positive_place_viz"; os.makedirs(OUT, exist_ok=True)
THUMB = 150
NTOP = 20
GCOLS = 5

def score_all(ds, good_dir):
    prof = f"{ETC}/profiling_tobe/{ds}"
    compat = json.load(open(f"{prof}/compatibility_matrix.json"))
    msym = compat['matrix_symmetric']; be = compat['bin_edges']
    clusters = [c for c,r in msym.items() if r]
    goods = sorted(glob.glob(f"{good_dir}/*"))
    # 이미지별 tile cells (cluster-무관) 캐시
    cells = {}
    for gp in goods:
        g = cv2.imread(gp, cv2.IMREAD_GRAYSCALE)
        if g is None: continue
        cells[gp] = gd._normal_tile_cells(g, be)
    # cluster별 스코어
    scored = {}
    for c in clusters:
        row = msym[c]
        s = [(gd._image_compat_score(cells[gp], row), gp) for gp in cells]
        s = [(sc,gp) for sc,gp in s if sc >= 0]
        s.sort(reverse=True)
        scored[c] = s
    return scored, clusters

def numeric(ds, scored, clusters):
    print(f"\n=== {ds} 수치 (ranked top-16 vs uniform 200표본) ===")
    rng = random.Random(0)
    for c in clusters:
        s = scored[c]; vals = [x[0] for x in s]
        topk = np.mean(vals[:16])
        uni = np.mean([np.mean(rng.sample(vals, min(16,len(vals)))) for _ in range(200)])
        print(f"  cluster {c}: n={len(s)} topK={topk:.3f} uniform={uni:.3f} "
              f"lift={(topk-uni)/max(uni,1e-6)*100:+.0f}%  range[{min(vals):.3f},{max(vals):.3f}]")

def grid(paths_scores, tag, color, cols=GCOLS):
    thumbs=[]
    for i,(sc, gp) in enumerate(paths_scores):
        im = cv2.imread(gp)
        if im is None: continue
        im = cv2.resize(im, (THUMB, THUMB))
        cv2.rectangle(im, (0,0), (THUMB-1,THUMB-1), color, 2)
        cv2.putText(im, f"#{i+1} {sc:.3f}", (4, THUMB-7), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)
        thumbs.append(im)
    rows=[]
    for r in range(0, len(thumbs), cols):
        chunk = thumbs[r:r+cols]
        while len(chunk) < cols: chunk.append(np.full((THUMB,THUMB,3), 20, np.uint8))
        rows.append(np.hstack(chunk))
    body = np.vstack(rows) if rows else np.zeros((THUMB, THUMB*cols, 3), np.uint8)
    bar = np.full((26, body.shape[1], 3), 30, np.uint8)
    cv2.putText(bar, tag, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
    return np.vstack([bar, body])

def viz(ds, scored, cluster):
    s = scored[cluster]
    top = s[:NTOP]; bottom = s[-NTOP:][::-1]
    top_g = grid(top, f"{ds} cluster{cluster} TOP-{NTOP} compat (green=fit bg)  n={len(s)}", (0,200,0))
    bot_g = grid(bottom, f"{ds} cluster{cluster} BOTTOM-{NTOP} compat (red=unfit)", (0,0,255))
    montage = np.vstack([top_g, bot_g])
    out = f"{OUT}/image_rank_{ds}_top{NTOP}.png"
    cv2.imwrite(out, montage); print(f"  saved: {out} ({montage.shape[1]}x{montage.shape[0]})")

if __name__ == "__main__":
    for ds, gd_dir, cl in [('mvtec_leather', f"{ETC}/leather/train/good", None),
                           ('mtd', f"{ETC}/mtd/train/good", None)]:
        scored, clusters = score_all(ds, gd_dir)
        numeric(ds, scored, clusters)
        viz(ds, scored, clusters[0] if cl is None else cl)
    print("\nDONE ->", OUT)
