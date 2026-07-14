"""
이미지-기준 배경집합 선택 비교: AROMA top-K 풀 vs RANDOM (severstal, 클래스별).

한 baseline 결함이미지 기준으로:
  baseline   = real 결함 patch + 실제 bbox(빨강)
  AROMA 풀   = clean_bg_selected.topk_pool 상위 K 배경(compat 랭킹) patch + bbox(초록)
  RANDOM     = uniform-random K 배경 patch + bbox(초록)
→ AROMA 풀은 baseline 강철 텍스처와 일관, random 은 임의. "배경 선택 집합" 육안 대비.

patch = 각 이미지 중앙 정사각 crop(텍스처 비교용), baseline 은 결함 bbox 중심 crop.
대표 = 클래스별 (AROMA풀 평균유사도 − RANDOM 평균유사도) 최대 ROI (예시, 대표 아님).
출력: .claude/.etc/positive_place_viz/1차/bg_pool_compare_severstal.png
"""
import os, sys, json, io, glob
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import viz_bg_class_similarity as base

DS_ROOT = base.DS_ROOT
OUT = "D:/project/aroma/.claude/.etc/positive_place_viz/1차"
K = 5            # 풀 크기
P = 150          # patch 픽셀
FONT = ImageFont.truetype("C:/Windows/Fonts/malgun.ttf", 14)


def _find(sub, stem):
    for e in ("jpg", "png", "jpeg", "bmp"):
        p = f"{DS_ROOT}/severstal/{sub}/{stem}.{e}"
        if os.path.exists(p):
            return p
    h = glob.glob(f"{DS_ROOT}/severstal/{sub}/{stem}.*")
    return h[0] if h else None


def good_stem(nid):
    return nid[1:] if nid.startswith("_") else nid


def _sq_patch(img, cx, cy):
    """(cx,cy) 중심 P-정사각 crop(경계 클램프) → PxP resize."""
    H, W = img.shape[:2]
    half = min(H, W) // 2
    half = min(half, 128)
    x0 = int(np.clip(cx - half, 0, W - 2 * half)); y0 = int(np.clip(cy - half, 0, H - 2 * half))
    patch = img[y0:y0 + 2 * half, x0:x0 + 2 * half]
    if patch.size == 0:
        patch = img
    return cv2.resize(patch, (P, P)), (x0, y0, 2 * half)


def baseline_patch(image_id, bbox):
    p = _find(f"test/{image_id.split('_')[0]}", image_id.partition('_')[2])
    img = cv2.imread(p, cv2.IMREAD_COLOR) if p else None
    if img is None:
        return np.full((P, P, 3), 60, np.uint8)
    x, y, w, h = (int(v) for v in bbox)
    patch, (x0, y0, side) = _sq_patch(img, x + w // 2, y + h // 2)
    # bbox 를 patch 좌표로
    s = P / side
    bx, by = int((x - x0) * s), int((y - y0) * s)
    cv2.rectangle(patch, (max(0, bx), max(0, by)),
                  (min(P, int((x - x0 + w) * s)), min(P, int((y - y0 + h) * s))), (0, 0, 255), 2)
    return patch


def bg_patch(nid, bbox):
    p = _find("train/good", good_stem(nid))
    img = cv2.imread(p, cv2.IMREAD_COLOR) if p else None
    if img is None:
        return np.full((P, P, 3), 60, np.uint8)
    H, W = img.shape[:2]
    patch, _ = _sq_patch(img, W // 2, H // 2)
    # 배치 bbox(초록) — patch 중앙에 결함 크기 근사
    x, y, w, h = (int(v) for v in bbox)
    side = min(H, W); s = P / min(side, 256)
    bw, bh = max(4, int(w * s)), max(4, int(h * s))
    cx, cy = P // 2, P // 2
    cv2.rectangle(patch, (cx - bw // 2, cy - bh // 2), (cx + bw // 2, cy + bh // 2), (0, 200, 0), 2)
    return patch


def label_bar(w, txt, color=(255, 255, 255), h=26):
    im = Image.new("RGB", (w, h), (30, 30, 30)); d = ImageDraw.Draw(im)
    d.text((6, 4), txt, font=FONT, fill=(color[2], color[1], color[0]))
    return cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)


def pool_sim(nids, ref):
    v = [base._hist_inter(base.good_hist(n), ref) for n in nids]
    v = [x for x in v if not np.isnan(x)]
    return float(np.mean(v)) if v else np.nan


def run():
    os.makedirs(OUT, exist_ok=True)
    sel = json.load(io.open(f"{DS_ROOT}/roi/severstal/clean_bg_selected.json", encoding="utf-8"))
    goods = [os.path.splitext(os.path.basename(p))[0] for p in sorted(glob.glob(f"{DS_ROOT}/severstal/train/good/*"))]
    classes = sorted({s["class_value"] for s in sel})
    refs = {c: base.build_class_ref(c) for c in classes}

    rows = []
    for c in classes:
        items = [s for s in sel if s["class_value"] == c and s.get("topk_pool")]
        # 예시 = AROMA풀 vs RANDOM 유사도차 최대 ROI
        best = None
        for s in items:
            apool = s["topk_pool"][:K]
            rng = np.random.default_rng(s["roi_idx"])
            rset = ["_" + goods[i] for i in rng.permutation(len(goods))[:K]]
            a = pool_sim(apool, refs[c]); r = pool_sim(rset, refs[c])
            if np.isnan(a) or np.isnan(r):
                continue
            if best is None or (a - r) > best[0]:
                best = (a - r, s, apool, rset, a, r)
        if best is None:
            continue
        _, s, apool, rset, a_sim, r_sim = best
        bbox = s["defect_bbox"]
        div = np.full((P, 6, 3), 80, np.uint8)
        bl = baseline_patch(s["image_id"], bbox)
        ar = np.hstack(sum([[bg_patch(n, bbox), div] for n in apool], [])[:-1])
        rd = np.hstack(sum([[bg_patch(n, bbox), div] for n in rset], [])[:-1])
        bigdiv = np.full((P, 14, 3), 110, np.uint8)
        strip = np.hstack([bl, bigdiv, ar, bigdiv, rd])
        w = strip.shape[1]
        top = label_bar(w, f"[{c}] baseline(real+빨강)   |   AROMA top-{K} 배경 풀(compat 선택)   |   RANDOM {K} 배경(uniform)   "
                        f"— 배경 유사도(real {c}): AROMA풀={a_sim:.3f} vs RANDOM={r_sim:.3f} (Δ{a_sim-r_sim:+.3f})",
                        color=(120, 255, 120))
        rows.append(np.vstack([top, strip]))

    maxw = max(r.shape[1] for r in rows)
    rows = [r if r.shape[1] == maxw else cv2.copyMakeBorder(r, 0, 0, 0, maxw - r.shape[1], cv2.BORDER_CONSTANT, value=(80, 80, 80)) for r in rows]
    hdr = label_bar(maxw, "severstal — 이미지당 배경집합 선택: AROMA 는 baseline 클래스 배경과 유사한 배경 풀을 고름, RANDOM 은 임의   "
                          "(예시=클래스별 대비 명확 ROI, 대표 아님 · 집계는 bg_class_similarity)", h=30)
    montage = np.vstack([hdr] + rows)
    outp = f"{OUT}/bg_pool_compare_severstal.png"
    ok, buf = cv2.imencode(".png", montage)
    if ok:
        buf.tofile(outp)
    print("saved:", outp, montage.shape, "ok" if ok else "FAIL")


if __name__ == "__main__":
    run()
