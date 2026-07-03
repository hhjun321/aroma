#!/usr/bin/env python3
"""
MTD (Magnetic-Tile-Defect, Supervisely 배포판) → AROMA/mvtec 레이아웃 정규화.

입력 (Dataset Ninja Supervisely 포맷):
    <root>/meta.json                     # 5 클래스 (blowhole/break/crack/fray/uneven), 전부 bitmap
    <root>/ds/img/<name>.jpg             # 원본 이미지 (1344장)
    <root>/ds/ann/<name>.jpg.json        # 주석: {size, objects:[{classTitle, bitmap:{data,origin}}], tags}
      - objects == []  → 결함 없음 (good).  (956장)
      - objects != []  → 결함. bitmap.data = base64(zlib(PNG RGBA)), alpha=mask, bitmap.origin=[x,y].

출력 (mvtec 레이아웃 → distribution_profiling `_find_mask_path` mtd/mvtec식 분기가 해소):
    <out>/train/good/<name>.jpg
    <out>/test/<class>/<name>.jpg
    <out>/ground_truth/<class>/<stem>_mask.png     # full-frame 0/255 이진, class별 union
    <out>/mtd_manifest.json

정책:
    - 다중결함 이미지: 존재하는 각 classTitle마다 (이미지, class-union mask) 엔트리 생성
      (mvtec per-image-per-class 관례). 소수(459 obj / 388 결함이미지).
    - 빈/디코드 실패 mask(전 object가 0px) → skip+count (Otsu 날조 방지, prepare_aitex와 동일 계약).
    - 결정론: 파일명 정렬. RNG 없음. idempotent(_link_or_copy).

사용:
    python prepare_mtd.py --supervisely_root $MTD_RAW --output_dir $DRIVE/mtd
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import shutil
import zlib
from pathlib import Path
from typing import Dict, List, Optional

# cv2 + numpy 필수 (bitmap PNG 디코드 + full-frame 래스터화). 없으면 즉시 안내 후 종료.
try:
    import numpy as np  # type: ignore[import]
    import cv2  # type: ignore[import]
    _HAS_DEPS = True
except Exception:
    _HAS_DEPS = False


def _decode_bitmap(data_b64: str, origin, size: Dict[str, int]) -> Optional["np.ndarray"]:
    """Supervisely bitmap → full-frame 0/255 uint8 mask. 실패/빈 patch면 None."""
    try:
        raw = zlib.decompress(base64.b64decode(data_b64))
        patch = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_UNCHANGED)
    except Exception:  # noqa: BLE001
        return None
    if patch is None:
        return None
    # RGBA면 alpha, 아니면 첫 채널/그레이. 값>0 = 결함.
    if patch.ndim == 3:
        m = patch[..., 3] if patch.shape[2] == 4 else patch[..., 0]
    else:
        m = patch
    ph, pw = m.shape[:2]
    ox, oy = int(origin[0]), int(origin[1])
    H, W = int(size["height"]), int(size["width"])
    full = np.zeros((H, W), np.uint8)
    # 경계 클램프(방어적)
    y0, x0 = max(0, oy), max(0, ox)
    y1, x1 = min(H, oy + ph), min(W, ox + pw)
    if y1 <= y0 or x1 <= x0:
        return None
    sub = (m[y0 - oy:y1 - oy, x0 - ox:x1 - ox] > 0).astype(np.uint8) * 255
    full[y0:y1, x0:x1] = sub
    return full


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


def prepare(supervisely_root: str, output_dir: str) -> Dict:
    if not _HAS_DEPS:
        raise RuntimeError("prepare_mtd requires numpy + opencv-python (cv2) for bitmap decode.")

    root = Path(supervisely_root)
    ann_dir = root / "ds" / "ann"
    img_dir = root / "ds" / "img"
    if not ann_dir.is_dir() or not img_dir.is_dir():
        raise FileNotFoundError(f"expected {ann_dir} and {img_dir} (Supervisely ds/ann + ds/img)")

    out = Path(output_dir)
    ann_files = sorted(ann_dir.glob("*.json"))
    print(f"[prepare_mtd] ann={len(ann_files)}  img_dir={img_dir}")

    n_good = 0
    n_defect_entries = 0        # (image, class) 엔트리 수
    n_skipped_no_img = 0
    n_skipped_empty_mask = 0
    per_class: Dict[str, int] = {}
    skipped_examples: List[str] = []
    manifest_defects: List[Dict] = []

    for af in ann_files:
        name = af.name[:-5] if af.name.endswith(".json") else af.stem  # "<name>.jpg.json" → "<name>.jpg"
        src_img = img_dir / name
        if not src_img.exists():
            n_skipped_no_img += 1
            if len(skipped_examples) < 10:
                skipped_examples.append(f"no_img:{name}")
            continue

        ann = json.loads(af.read_text(encoding="utf-8"))
        objs = ann.get("objects", []) or []
        size = ann.get("size", {})

        if not objs:
            _link_or_copy(str(src_img), str(out / "train" / "good" / name))
            n_good += 1
            continue

        # class별 union mask 빌드
        stem = Path(name).stem  # "<name>.jpg" → "<name>"
        by_class: Dict[str, "np.ndarray"] = {}
        for o in objs:
            cls = str(o.get("classTitle") or "unknown")
            bmp = o.get("bitmap") or {}
            m = _decode_bitmap(bmp.get("data", ""), bmp.get("origin", [0, 0]), size)
            if m is None:
                continue
            if cls in by_class:
                by_class[cls] = np.maximum(by_class[cls], m)
            else:
                by_class[cls] = m

        wrote_any = False
        for cls, mask in by_class.items():
            if int(mask.max()) == 0:  # 빈 mask → skip+count (Otsu 날조 방지)
                n_skipped_empty_mask += 1
                if len(skipped_examples) < 10:
                    skipped_examples.append(f"empty_mask:{name}:{cls}")
                continue
            _link_or_copy(str(src_img), str(out / "test" / cls / name))
            dst_mask = out / "ground_truth" / cls / f"{stem}_mask.png"
            dst_mask.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(dst_mask), mask)
            per_class[cls] = per_class.get(cls, 0) + 1
            n_defect_entries += 1
            wrote_any = True
            manifest_defects.append({
                "image_id": name, "defect_type": cls,
                "image": f"test/{cls}/{name}",
                "mask": f"ground_truth/{cls}/{stem}_mask.png",
            })
        if not wrote_any and by_class:
            pass  # 전부 빈 mask였음(이미 카운트)

    classes = sorted(per_class.keys())
    manifest = {
        "dataset": "mtd",
        "source_format": "supervisely (Dataset Ninja): ds/ann/*.jpg.json + ds/img/*.jpg",
        "counts": {
            "good": n_good,
            "defect_entries": n_defect_entries,
            "per_class": per_class,
            "skipped_no_img": n_skipped_no_img,
            "skipped_empty_mask": n_skipped_empty_mask,
        },
        "classes": classes,
        "layout": {
            "train_good": "train/good",
            "test_classes": [f"test/{c}" for c in classes],
            "ground_truth": "ground_truth/{class}/{stem}_mask.png",
        },
        "defects": manifest_defects,
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "mtd_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"[prepare_mtd] good={n_good}  defect_entries={n_defect_entries}  per_class={per_class}")
    print(f"[prepare_mtd] skipped_no_img={n_skipped_no_img}  skipped_empty_mask={n_skipped_empty_mask}")
    if skipped_examples:
        print(f"[prepare_mtd] skipped examples: {skipped_examples}")
    print(f"[prepare_mtd] classes={classes}")
    print(f"[prepare_mtd] wrote layout + masks + manifest → {out}")
    return manifest


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MTD Supervisely → AROMA/mvtec layout normalizer")
    p.add_argument("--supervisely_root", required=True,
                   help="MTD Supervisely root (contains ds/ann, ds/img, meta.json)")
    p.add_argument("--output_dir", required=True,
                   help="normalized output root (train/good, test/{class}, ground_truth/{class})")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = _parse_args(argv)
    prepare(args.supervisely_root, args.output_dir)


if __name__ == "__main__":
    main()
