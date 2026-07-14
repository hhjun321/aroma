#!/usr/bin/env python3
"""
KolektorSDD (v1) → AROMA/mvtec 레이아웃 정규화 (단일클래스 'defect').

입력 (원본 KolektorSDD):
    <root>/kos01 .. kosNN/          # 아이템별 폴더(50개)
        Part0.jpg .. Part7.jpg      # 표면 이미지 (~1263x500, tall)
        Part0_label.bmp ..          # 대응 라벨 마스크(0=정상, >0=결함)
      - label 결함픽셀 0   → good  (결함 없는 표면)
      - label 결함픽셀 >0  → defect (표면 결함)

출력 (mvtec 레이아웃 — distribution_profiling / exp4v2 mvtec·mtd식 분기가 해소):
    <out>/train/good/<item>_<part>.jpg
    <out>/test/defect/<item>_<part>.jpg           # 단일클래스: 폴더 'defect' 하나
    <out>/ground_truth/defect/<item>_<part>_mask.png   # full-frame 0/255 이진
    <out>/kolektor_manifest.json

정책 (prepare_mtd 계약과 동일):
    - 단일 결함타입 → class='defect'(nc=1). exp4v2 --class_mode single(기본)이 네이티브 처리.
    - 빈/결함0 마스크 = good(train/good). 마스크 있으나 이진화 후 0px면 skip+count(날조 방지).
    - 결정론: 파일명 정렬, RNG 없음. idempotent(_link_or_copy: 존재 시 skip).
    - 파일명 flatten: kosXX/PartN → 'kosXX_PartN' (전역 유일 stem).

사용:
    python prepare_kolektor.py --kolektor_root $KSDD_RAW --output_dir $DRIVE/kolektor
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional

try:
    import numpy as np  # type: ignore[import]
    import cv2  # type: ignore[import]
    _HAS_DEPS = True
except Exception:
    _HAS_DEPS = False


def _link_or_copy(src: str, dst: str) -> None:
    """Symlink src→dst; 실패 시 copy. 존재하면 skip (idempotent)."""
    dst_p = Path(dst)
    dst_p.parent.mkdir(parents=True, exist_ok=True)
    if dst_p.exists() or dst_p.is_symlink():
        return
    try:
        os.symlink(os.path.abspath(src), dst)
    except Exception:  # noqa: BLE001
        shutil.copy2(src, dst)


def _find_mask(img_path: Path) -> Optional[Path]:
    """PartN.jpg → 대응 라벨 마스크(PartN_label.bmp / PartN.bmp / PartN_label.png)."""
    stem = img_path.stem  # PartN
    for cand in (
        img_path.with_name(f"{stem}_label.bmp"),
        img_path.with_name(f"{stem}_label.png"),
        img_path.with_name(f"{stem}.bmp"),
        img_path.with_name(f"{stem}_GT.bmp"),
    ):
        if cand.exists():
            return cand
    return None


def prepare(kolektor_root: str, output_dir: str) -> Dict:
    if not _HAS_DEPS:
        raise RuntimeError("prepare_kolektor requires numpy + opencv-python (cv2) for mask decode.")

    root = Path(kolektor_root)
    item_dirs = sorted(d for d in root.iterdir() if d.is_dir() and d.name.lower().startswith("kos"))
    if not item_dirs:
        raise FileNotFoundError(f"no kosNN item folders under {root}")

    out = Path(output_dir)
    print(f"[prepare_kolektor] items={len(item_dirs)}  root={root}")

    n_good = 0
    n_defect = 0
    n_skipped_no_mask = 0
    n_skipped_empty_mask = 0
    skipped_examples: List[str] = []
    manifest_defects: List[Dict] = []
    sizes: List[tuple] = []

    for item in item_dirs:
        for img in sorted(item.glob("*.jpg")) + sorted(item.glob("*.png")):
            if img.stem.endswith("_label") or img.stem.endswith("_GT"):
                continue  # 마스크가 png/jpg인 변형 방어
            name = f"{item.name}_{img.stem}"          # kos01_Part5 (전역 유일)
            mpath = _find_mask(img)
            if mpath is None:
                n_skipped_no_mask += 1
                if len(skipped_examples) < 10:
                    skipped_examples.append(f"no_mask:{name}")
                continue
            m = cv2.imread(str(mpath), cv2.IMREAD_GRAYSCALE)
            if m is None:
                n_skipped_no_mask += 1
                if len(skipped_examples) < 10:
                    skipped_examples.append(f"unreadable_mask:{name}")
                continue

            if int((m > 0).sum()) == 0:
                # 결함 0 → 정상 표면
                _link_or_copy(str(img), str(out / "train" / "good" / f"{name}.jpg"))
                n_good += 1
                continue

            # 결함 존재 → test/defect + ground_truth
            bin_mask = ((m > 0).astype(np.uint8)) * 255
            _link_or_copy(str(img), str(out / "test" / "defect" / f"{name}.jpg"))
            dst_mask = out / "ground_truth" / "defect" / f"{name}_mask.png"
            dst_mask.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(dst_mask), bin_mask)
            n_defect += 1
            sizes.append((int(m.shape[1]), int(m.shape[0])))
            manifest_defects.append({
                "image_id": name,
                "defect_type": "defect",
                "image": f"test/defect/{name}.jpg",
                "mask": f"ground_truth/defect/{name}_mask.png",
                "source": f"{item.name}/{img.name}",
                "defect_px": int((m > 0).sum()),
            })

    manifest = {
        "dataset": "kolektor",
        "source_format": "KolektorSDD v1: kosNN/PartK.jpg + PartK_label.bmp",
        "class_mode": "single",
        "counts": {
            "good": n_good,
            "defect": n_defect,
            "skipped_no_mask": n_skipped_no_mask,
            "skipped_empty_mask": n_skipped_empty_mask,
        },
        "classes": ["defect"],
        "layout": {
            "train_good": "train/good",
            "test_classes": ["test/defect"],
            "ground_truth": "ground_truth/defect/{stem}_mask.png",
        },
        "defects": manifest_defects,
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "kolektor_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"[prepare_kolektor] good={n_good}  defect={n_defect}  "
          f"skipped_no_mask={n_skipped_no_mask}  skipped_empty_mask={n_skipped_empty_mask}")
    if skipped_examples:
        print(f"[prepare_kolektor] skipped examples: {skipped_examples}")
    if sizes:
        ws = sorted(w for w, _ in sizes); hs = sorted(h for _, h in sizes)
        print(f"[prepare_kolektor] defect img size(px) w[min/med/max]={ws[0]}/{ws[len(ws)//2]}/{ws[-1]} "
              f"h[min/med/max]={hs[0]}/{hs[len(hs)//2]}/{hs[-1]}")
    print(f"[prepare_kolektor] wrote layout + masks + manifest → {out}")
    return manifest


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="KolektorSDD v1 → AROMA/mvtec layout normalizer (single-class)")
    p.add_argument("--kolektor_root", required=True,
                   help="KolektorSDD raw root (contains kos01..kosNN item folders)")
    p.add_argument("--output_dir", required=True,
                   help="normalized output root (train/good, test/defect, ground_truth/defect)")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = _parse_args(argv)
    prepare(args.kolektor_root, args.output_dir)


if __name__ == "__main__":
    main()
