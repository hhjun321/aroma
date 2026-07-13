#!/usr/bin/env python3
"""AROMA vs random 증강셋 육안 비교 몽타주.

class별 10개씩 (roi_score 상위 5 + 하위 5), 좌 AROMA / 우 RANDOM.
각 셀 = 합성 composite + mask 파생 GT bbox(초록) + roi_score 라벨.
합성 이미지가 없으면 bbox 영역(마스크 크롭 or 좌표 placeholder)으로 폴백.

사용:
    python build_quality_compare.py \
        --aroma D:/project/aroma_dataset/synth_aroma/severstal \
        --random D:/project/aroma_dataset/synth_random/severstal \
        --out    D:/project/aroma/.claude/.etc/positive_place_viz \
        --tag    prefix        # 출력 파일 접두사 (before/after 구분)

개선(void-floor + naive-placement) Colab 재생성 후 --aroma/--random 을 새 경로로,
--tag after 로 재실행하면 before/after 비교 가능.
"""
import argparse, json, os
from collections import defaultdict

import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def mask_to_bboxes(mask_path, min_area=50):
    """generate_defects._mask_to_bboxes 복제 (GT bbox = 학습에 실제 쓰는 값)."""
    if not mask_path or not os.path.exists(mask_path):
        return [], 0, 0
    m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        return [], 0, 0
    mh, mw = m.shape[:2]
    _, b = cv2.threshold(m, 0, 255, cv2.THRESH_BINARY)
    cnts, _ = cv2.findContours(b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bx = [cv2.boundingRect(c) for c in cnts
          if cv2.boundingRect(c)[2] * cv2.boundingRect(c)[3] >= min_area]
    if not bx and len(cnts):
        nz = cv2.findNonZero(b)
        if nz is not None:
            bx = [cv2.boundingRect(nz)]
    return bx, mw, mh


def load_arm(root):
    """annotations.json → {class: [(roi_score, img_path, mask_path, bbox), ...]}."""
    ann = os.path.join(root, "annotations.json")
    d = json.load(open(ann, encoding="utf-8"))
    imgdir = os.path.join(root, "images")
    mdir = os.path.join(root, "masks")
    by = defaultdict(list)
    for a in d:
        bn = os.path.basename(a["image_path"])
        fp = os.path.join(imgdir, bn)
        mp = os.path.join(mdir, os.path.splitext(bn)[0] + ".png")
        by[a.get("class_key")].append({
            "roi_score": float(a.get("roi_score", 0.0)),
            "img": fp, "mask": mp, "bbox": a.get("bbox"),
        })
    return by


def pick_top_low(items, n=5):
    """roi_score 상위 n + 하위 n. 중복 방지."""
    s = sorted(items, key=lambda e: e["roi_score"], reverse=True)
    top = s[:n]
    low = s[-n:] if len(s) >= 2 * n else s[n:]
    return top, low


def draw_cell(ax, e, arm_color):
    ax.axis("off")
    black_pct = None
    if os.path.exists(e["img"]):
        img = cv2.cvtColor(cv2.imread(e["img"]), cv2.COLOR_BGR2RGB)
        H, W = img.shape[:2]
        black_pct = float((cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) < 25).mean()) * 100
        ax.imshow(img)
        bx, mw, mh = mask_to_bboxes(e["mask"])
        sx = W / mw if mw else 1
        sy = H / mh if mh else 1
        for (x, y, w, h) in bx:
            ax.add_patch(Rectangle((x * sx, y * sy), w * sx, h * sy,
                                   fill=False, edgecolor="lime", linewidth=1.4))
    else:
        # 폴백: 합성 이미지 없음 → 마스크 bbox 크롭, 없으면 좌표 placeholder
        bx, mw, mh = mask_to_bboxes(e["mask"])
        if os.path.exists(e["mask"]):
            m = cv2.imread(e["mask"], cv2.IMREAD_GRAYSCALE)
            ax.imshow(m, cmap="gray")
            for (x, y, w, h) in bx:
                ax.add_patch(Rectangle((x, y), w, h, fill=False,
                                       edgecolor="red", linewidth=1.4))
            ax.text(0.02, 0.95, "no synth img → mask/bbox", color="red",
                    fontsize=7, transform=ax.transAxes, va="top")
        else:
            bb = e.get("bbox")
            ax.text(0.5, 0.5, f"no synth img / mask\nbbox={bb}", ha="center",
                    va="center", fontsize=8, transform=ax.transAxes)
            ax.set_facecolor("0.9")
    lbl = f"q={e['roi_score']:.3f}"
    if black_pct is not None:
        lbl += f"  blk={black_pct:.0f}%"
    ax.set_title(lbl, fontsize=8, color=arm_color, loc="left")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aroma", required=True)
    ap.add_argument("--random", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="cmp")
    ap.add_argument("--classes", default="class1,class2,class3,class4")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    aroma = load_arm(args.aroma)
    rand = load_arm(args.random)
    classes = args.classes.split(",")

    for cls in classes:
        a_top, a_low = pick_top_low(aroma.get(cls, []))
        r_top, r_low = pick_top_low(rand.get(cls, []))
        a_rows = a_top + a_low          # 0-4 top, 5-9 low
        r_rows = r_top + r_low
        nrow = max(len(a_rows), len(r_rows), 1)
        fig, axes = plt.subplots(nrow, 2, figsize=(20, nrow * 1.9),
                                 squeeze=False)
        for col, (rows, color, name) in enumerate(
                [(a_rows, "deepskyblue", "AROMA"),
                 (r_rows, "red", "RANDOM")]):
            for r in range(nrow):
                ax = axes[r][col]
                if r < len(rows):
                    draw_cell(ax, rows[r], color)
                else:
                    ax.axis("off")
                if r == 0:
                    ax.set_title(f"{name}  ↑TOP-quality (rows 1-5)",
                                 fontsize=12, color=color, fontweight="bold")
                elif r == 5:
                    ax.set_title(f"{name}  ↓LOW-quality (rows 6-10)",
                                 fontsize=11, color=color, fontweight="bold")
        n_a = len(aroma.get(cls, []))
        n_r = len(rand.get(cls, []))
        fig.suptitle(f"severstal {cls} [{args.tag}]  |  AROMA n={n_a} vs RANDOM n={n_r}"
                     f"   (roi_score top5+low5, GT bbox green, blk=검정%)",
                     fontsize=14, y=0.997)
        fig.tight_layout(rect=[0, 0, 1, 0.985])
        outp = os.path.join(args.out, f"{cls}_{args.tag}_qualcmp.png")
        fig.savefig(outp, dpi=78, bbox_inches="tight")
        plt.close(fig)
        print("saved", outp)


if __name__ == "__main__":
    main()
