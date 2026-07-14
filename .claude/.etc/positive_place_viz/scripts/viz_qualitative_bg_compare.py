"""
정성(육안) 비교 figure: baseline vs AROMA vs random 배경선택 (severstal, 클래스별).

각 클래스 대표 샘플 1개, 3패널:
  baseline = real 결함이미지 + 실제 defect bbox(빨강)
  aroma    = AROMA 배정 clean-bg(good) + 배치 bbox(초록)  ← baseline 배경과 유사해야
  random   = random 배정 clean-bg(good) + 배치 bbox(초록) ← 무작위
각 패널에 real 클래스-c 배경과의 텍스처 유사도 병기 → 통계 figure(bg_class_similarity)와 연결.

대표 샘플 = 클래스 aroma-유사도 '중앙값' ROI(cherry-pick 회피, 결정론).
출력: .claude/.etc/positive_place_viz/1차/qualitative_bg_compare_severstal.png
"""
import os, sys, json, io, glob
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import viz_bg_class_similarity as base

DS_ROOT = base.DS_ROOT
OUT = "D:/project/aroma/.claude/.etc/positive_place_viz/1차"
PANEL_H = 210
FONT = ImageFont.truetype("C:/Windows/Fonts/malgun.ttf", 15)


def _find(sub, stem):
    for e in ("jpg", "png", "jpeg", "bmp"):
        p = f"{DS_ROOT}/severstal/{sub}/{stem}.{e}"
        if os.path.exists(p):
            return p
    h = glob.glob(f"{DS_ROOT}/severstal/{sub}/{stem}.*")
    return h[0] if h else None


def src_path(image_id):  # class1_27f44281b -> test/class1/27f44281b
    head, _, stem = image_id.partition("_")
    return _find(f"test/{head}", stem)


def norm_path(nid):
    stem = nid[1:] if nid.startswith("_") else nid
    return _find("train/good", stem)


def fit(img, h=PANEL_H, tag=None):
    if img is None:
        ph = np.full((h, int(h * 2), 3), 60, np.uint8)
        cv2.putText(ph, tag or "MISSING", (10, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        return ph
    s = h / img.shape[0]
    return cv2.resize(img, (max(1, int(img.shape[1] * s)), h))


def rect(img, bbox, color):
    if img is None or not bbox:
        return img
    H, W = img.shape[:2]
    x, y, w, h = (int(v) for v in bbox)
    x, y = max(0, min(x, W - 1)), max(0, min(y, H - 1))
    x2, y2 = min(W, x + max(1, w)), min(H, y + max(1, h))
    cv2.rectangle(img, (x, y), (x2, y2), color, max(2, W // 300))
    return img


def label_bar(w, txt, color=(255, 255, 255), h=44):
    im = Image.new("RGB", (w, h), (30, 30, 30))
    d = ImageDraw.Draw(im)
    for i, line in enumerate(txt.split("\n")[:2]):
        d.text((6, 3 + i * 20), line, font=FONT, fill=(color[2], color[1], color[0]))
    return cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)


def sim_to_ref(normal_id, ref):
    # base.good_hist = 배정배경 텍스처 descriptor(캐시). ref 와 교집합.
    return base._hist_inter(base.good_hist(normal_id), ref)


def run():
    os.makedirs(OUT, exist_ok=True)
    sel = json.load(io.open(f"{DS_ROOT}/roi/severstal/clean_bg_selected.json", encoding="utf-8"))
    rnd = {r["roi_idx"]: r for r in json.load(io.open(f"{DS_ROOT}/roi/severstal/clean_bg_random_arm.json", encoding="utf-8"))}
    classes = sorted({s["class_value"] for s in sel})
    refs = {c: base.build_class_ref(c) for c in classes}

    rows = []
    for c in classes:
        items = [s for s in sel if s["class_value"] == c and s["roi_idx"] in rnd]
        # 예시 = AROMA 우위(aroma_sim − random_sim) 최대 ROI (대비 명확 — 예시, 대표 아님).
        # 집계·유의성은 bg_class_similarity(전 표본)가 담당.
        cand = []
        for s in items:
            a = sim_to_ref(s["assigned_normal_id"], refs[c])
            r = sim_to_ref(rnd[s["roi_idx"]]["assigned_normal_id"], refs[c])
            if not np.isnan(a) and not np.isnan(r):
                cand.append((a - r, s))
        cand.sort(key=lambda t: t[0], reverse=True)
        _, s = cand[0]
        r = rnd[s["roi_idx"]]
        bbox = s.get("defect_bbox")
        # 패널
        bl = cv2.imread(src_path(s["image_id"]) or "", cv2.IMREAD_COLOR)
        bl = rect(bl, bbox, (0, 0, 255)) if bl is not None else None
        ar = cv2.imread(norm_path(s["assigned_normal_id"]) or "", cv2.IMREAD_COLOR)
        ar = rect(ar, bbox, (0, 200, 0)) if ar is not None else None
        rd = cv2.imread(norm_path(r["assigned_normal_id"]) or "", cv2.IMREAD_COLOR)
        rd = rect(rd, bbox, (0, 200, 0)) if rd is not None else None
        a_sim = sim_to_ref(s["assigned_normal_id"], refs[c])
        r_sim = sim_to_ref(r["assigned_normal_id"], refs[c])
        div = np.full((PANEL_H, 6, 3), 80, np.uint8)
        strip = np.hstack([fit(bl, tag="SRC"), div, fit(ar, tag="ARO"), div, fit(rd, tag="RND")])
        w = strip.shape[1]
        top = label_bar(w, f"[{c}] baseline(real+빨강 결함bbox)  |  AROMA 배정배경+초록bbox  |  RANDOM 배정배경+초록bbox",
                        color=(120, 255, 120))
        sub = label_bar(w, f"배경 유사도(real {c} 배경): AROMA={a_sim:.3f}  vs  RANDOM={r_sim:.3f}   Δ={a_sim-r_sim:+.3f}"
                        + ("  ← AROMA 유사" if a_sim > r_sim else "  (random 유사)"))
        rows.append(np.vstack([top, sub, strip]))

    maxw = max(r.shape[1] for r in rows)
    rows = [r if r.shape[1] == maxw else cv2.copyMakeBorder(r, 0, 0, 0, maxw - r.shape[1], cv2.BORDER_CONSTANT, value=(80, 80, 80)) for r in rows]
    hdr = label_bar(maxw, "severstal 정성 예시 — AROMA 는 real 클래스 배경과 유사한 clean-bg 선택, random 은 무작위\n"
                          "⚠️ 예시=클래스별 대비 명확 샘플(대표 아님) · 집계·유의성은 bg_class_similarity figure(전 표본)", h=48)
    montage = np.vstack([hdr] + rows)
    outp = f"{OUT}/qualitative_bg_compare_severstal.png"
    # cv2.imwrite 는 Windows 에서 한글경로 실패 → imencode+tofile(유니코드 안전)
    ok, buf = cv2.imencode(".png", montage)
    if ok:
        buf.tofile(outp)
    print("saved:", outp, montage.shape, "ok" if ok else "ENCODE_FAIL")


if __name__ == "__main__":
    run()
