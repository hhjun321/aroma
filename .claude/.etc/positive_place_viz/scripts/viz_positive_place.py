"""
positive placement 시각 검증 PNG.
샘플당 나란히: 좌=결함 source 이미지+원본 defect_bbox(빨강), 우=clean-bg+positive place_on footprint(녹색).
대표표본: crop 소/중/대 × leather/mtd. 저장 .claude/.etc/positive_place_viz/.
"""
import sys, os, csv, ast, json, glob, random
import numpy as np, cv2
sys.path.insert(0, "D:/project/aroma/scripts"); sys.path.insert(0, "D:/project/aroma/scripts/aroma")
import distribution_profiling as dp
import generate_defects as gd

ETC = "D:/project/aroma/.claude/.etc"
OUT = f"{ETC}/positive_place_viz"; os.makedirs(OUT, exist_ok=True)
PANEL_H = 384

def parse_bbox(s):
    try: return tuple(int(v) for v in ast.literal_eval(s))
    except Exception: return None

def find_src(ds, dtype, iid):
    for ext in ('.png','.jpg','.jpeg','.bmp'):
        p = f"{ETC}/{ds_dir(ds)}/test/{dtype}/{iid}{ext}"
        if os.path.exists(p): return p
    return None

def ds_dir(ds): return 'leather' if ds=='mvtec_leather' else ds

def fit_h(img, h=PANEL_H):
    s = h/img.shape[0]; return cv2.resize(img, (max(1,int(img.shape[1]*s)), h))

def label_bar(w, txt, h=28):
    bar = np.full((h, w, 3), 30, np.uint8)
    cv2.putText(bar, txt, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
    return bar

def run(ds, good_pick=0):
    prof = f"{ETC}/profiling_tobe/{ds}"
    compat = json.load(open(f"{prof}/compatibility_matrix.json"))
    msym = compat['matrix_symmetric']; be = compat['bin_edges']
    ca = {str(k): int(v) for k,v in json.load(open(f"{prof}/morphology_clusters.json"))['cluster_assignments'].items()}
    rows = list(csv.DictReader(open(f"{prof}/morphology_features.csv")))
    # defect 후보: bbox 유효 + cluster row 비어있지 않음 + source 이미지 존재
    cand = []
    for r in rows:
        bb = parse_bbox(r.get('defect_bbox','')); iid = r['image_id']; dt = r.get('defect_type','')
        cl = ca.get(iid)
        if bb and len(bb)==4 and bb[2]>0 and bb[3]>0 and cl is not None and msym.get(str(cl)):
            src = find_src(ds, dt, iid)
            if src: cand.append((bb[2]*bb[3], bb, dt, iid, str(cl), src))
    cand.sort()
    # 소/중/대
    picks = [cand[len(cand)//10], cand[len(cand)//2], cand[min(len(cand)-1, len(cand)*9//10)]]

    goods = sorted(glob.glob(f"{ETC}/{ds_dir(ds)}/train/good/*"))
    gpath = goods[good_pick % len(goods)]
    good_bgr = cv2.imread(gpath); good_rgb = cv2.cvtColor(good_bgr, cv2.COLOR_BGR2RGB)
    H, W = good_rgb.shape[:2]

    rng = random.Random(42)
    sample_rows = []
    for area, bb, dt, iid, cl, src in picks:
        x,y,cw,ch = bb
        # 좌: source 결함 + 원본 bbox(빨강)
        sb = cv2.imread(src)
        cv2.rectangle(sb, (x,y), (x+cw, y+ch), (0,0,255), max(2, sb.shape[1]//300))
        L = fit_h(sb)
        # fit-rescale (crop>good)
        cw2, ch2 = cw, ch
        if cw2>W or ch2>H:
            sfac = min(W/cw2, H/ch2)*0.95; cw2, ch2 = max(1,int(cw2*sfac)), max(1,int(ch2*sfac))
        pos, best_mean, n = gd._positive_place(good_rgb, (cw2, ch2), msym[cl], be, 0.7, 100.0, rng)
        gb = good_bgr.copy()
        if pos is not None:
            px,py = pos
            cv2.rectangle(gb, (px,py), (px+cw2, py+ch2), (0,220,0), max(2, W//300))
            ptxt = f"place ({px},{py}) mean_compat={best_mean:.3f} nonvoid={n}"
        else:
            ptxt = f"ALL-VOID -> repick (best={best_mean:.3f})"
        R = fit_h(gb)
        # 라벨 + hconcat
        div = np.full((PANEL_H, 6, 3), 80, np.uint8)
        row_img = np.hstack([L, div, R])
        bar = label_bar(row_img.shape[1],
                        f"{ds} {dt} #{iid} cl{cl} crop {cw}x{ch}(area{area}) rescaled {cw2}x{ch2} | {ptxt}")
        sample_rows.append(np.vstack([bar, row_img]))
    montage = np.vstack([r if r.shape[1]==sample_rows[0].shape[1]
                         else cv2.resize(r,(sample_rows[0].shape[1], r.shape[0])) for r in sample_rows])
    # 상단 헤더
    hdr = label_bar(montage.shape[1], f"[{ds}] LEFT=defect source (red=orig bbox)  RIGHT=clean-bg (green=positive place_on)  good={os.path.basename(gpath)}", 32)
    montage = np.vstack([hdr, montage])
    out = f"{OUT}/positive_place_{ds}.png"
    cv2.imwrite(out, montage)
    print(f"saved: {out}  ({montage.shape[1]}x{montage.shape[0]})  picks area={[p[0] for p in picks]}")

if __name__ == "__main__":
    run('mvtec_leather'); run('mtd')
    print("\nDONE ->", OUT)
