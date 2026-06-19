#!/usr/bin/env python3
"""
AROMA Exp 4 v2 — Supervised YOLOv8 Defect Detection 평가 (COCO pretrained start)

CASDA-style 재설계: 세 조건 모두 "real labeled defect"를 공통 토대로 학습한다.

| 조건     | 시작 weights         | 학습 데이터                                   |
|----------|----------------------|-----------------------------------------------|
| baseline | COCO (yolov8n.pt)    | real labeled defect (train split) → best.pt 저장 |
| random   | COCO (yolov8n.pt)    | real labeled defect + random synth (처음부터) |
| aroma    | COCO (yolov8n.pt)    | real labeled defect + aroma synth  (처음부터) |

핵심 변경:
- real defect 이미지를 train/val 로 seeded split (이전엔 전부 val) → baseline 이
  real positive 를 학습하고, val 은 disjoint real-only 로 유지.
- synthetic 은 random/aroma 에서만 real 위에 "추가"되는 데이터.
- baseline 은 COCO 에서 시작 + save=True 로 best.pt 영속화.
  random/aroma 도 COCO 에서 시작, real+synth 처음부터 학습.

Synthetic label 유도:
- composite(synth) 와 background(normal) 의 image diff → threshold → contour bbox.
- synthesis 재실행 불필요 — label 을 post-hoc 로 유도.

Datasets: isp_LSM_1, mvtec_cable, visa_cashew, visa_pcb (pcb4)

I/O 가속 (두 단계 캐시):
- Local cache (`--no_local_cache` 로 끔): Linux/Colab 에서 Drive 이미지를 /tmp 로
  복사해 두고 그 경로로 학습 (Drive 직접 read 의 느린 random-access 회피).
  Windows 에서는 항상 비활성.
- YOLO real-dataset cache (`--yolo_cache_dir`): real (image,label) 셋을 데이터셋당
  한 번만 빌드하고 지정 경로(예: Drive)에 영속화. 동일 (min_area/val_frac/seed)
  조합이면 재빌드 없이 로드. 미지정 시 매 데이터셋마다 새로 빌드 (캐시 없음 —
  기존 동작과 동일).

Usage (Colab):
    !python $AROMA_SCRIPTS/experiments/exp4_v2_supervised_detection.py \\
        --model yolov8n \\
        --condition all \\
        --dataset_keys isp_LSM_1 mvtec_cable visa_cashew visa_pcb \\
        --random_synthetic_dir $RANDOM_SYNTH_DIR \\
        --aroma_synthetic_dir  $AROMA_SYNTH_DIR \\
        --real_data_dir        $AROMA_DATA \\
        --output_dir           $EXP4_OUT \\
        --yolo_cache_dir       $EXP4_OUT/yolo_cache \\
        --seed 42 \\
        --baseline_epochs 50 \\
        --val_frac 0.5 \\
        --resume
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import random
import shutil
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("aroma.exp4v2")

# Suppress ultralytics/torch/tqdm verbose output
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("YOLO_VERBOSE", "False")
import warnings
warnings.filterwarnings("ignore")
logging.getLogger("ultralytics").setLevel(logging.ERROR)
logging.getLogger("timm").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("PIL").setLevel(logging.ERROR)
logging.getLogger("matplotlib").setLevel(logging.ERROR)

# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

_THIS_DIR = Path(__file__).resolve().parent
_AROMA_SCRIPTS = _THIS_DIR.parent


def _bootstrap() -> None:
    aroma_ref = os.environ.get("AROMA_REF") or r"D:\project\aroma"
    for p in (str(_THIS_DIR), str(_AROMA_SCRIPTS), str(Path(aroma_ref))):
        if p not in sys.path:
            sys.path.insert(0, p)


_bootstrap()

# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

try:
    from utils.io import load_json, save_json  # type: ignore[import]
except Exception:
    def load_json(p: str) -> Any:  # type: ignore[misc]
        with open(p, encoding="utf-8") as f:
            return json.load(f)

    def save_json(data: Any, p: str) -> None:  # type: ignore[misc]
        Path(p).parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# OpenCV import (required for bbox extraction)
# ---------------------------------------------------------------------------

try:
    import cv2  # type: ignore[import]
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False
    cv2 = None  # type: ignore[assignment]


ALL_MODEL_KEYS = ["yolov8n", "yolov8s", "yolov8m"]
ALL_CONDITION_KEYS = ["baseline", "random", "aroma"]

_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


# ---------------------------------------------------------------------------
# Dataset path resolution (mirrors exp4)
# ---------------------------------------------------------------------------

def _glob_images(directory: str) -> List[str]:
    d = Path(directory)
    if not d.exists():
        return []
    return sorted(str(p) for p in d.iterdir() if p.suffix.lower() in _IMG_EXTS)


def _split_normal(
    normal_paths: List[str], test_split: float = 0.2, seed: int = 42
) -> Tuple[List[str], List[str]]:
    """For VisA (no disk split): shuffle then split 80/20."""
    rng = np.random.default_rng(seed)
    shuffled = [normal_paths[i] for i in rng.permutation(len(normal_paths))]
    n_test = max(1, int(len(normal_paths) * test_split))
    return shuffled[n_test:], shuffled[:n_test]


def _split_defects(
    test_defect_paths: List[str],
    mask_map: Dict[str, str],
    val_frac: float = 0.5,
    seed: int = 42,
) -> Tuple[List[str], List[str]]:
    """Split real defect images (that HAVE masks) into disjoint train/val pools.

    Returns (train_defect_paths, val_defect_paths). Only mask-bearing images are
    eligible (need GT bbox for both train labels and val metrics).
    """
    eligible = sorted(p for p in test_defect_paths if mask_map.get(p))
    if not eligible:
        # No mask-bearing defect images: cannot form train or val labels.
        return [], []
    rng = random.Random(seed)
    rng.shuffle(eligible)
    # Reserve at least one image for val, but never consume the whole pool so
    # train is also non-empty when >=2 images exist.
    n_val = max(1, int(round(len(eligible) * val_frac)))
    if len(eligible) >= 2:
        n_val = min(n_val, len(eligible) - 1)
    val = eligible[:n_val]
    train = eligible[n_val:]
    return train, val


def _resolve_isp_masks(defect_paths: List[str], gt_dir: str) -> Dict[str, str]:
    """ISP: ground_truth/area/<same_filename>."""
    mapping: Dict[str, str] = {}
    gt = Path(gt_dir)
    for img_p in defect_paths:
        mask = gt / Path(img_p).name
        if mask.exists():
            mapping[img_p] = str(mask)
    return mapping


def _resolve_mvtec_masks(defect_paths: List[str], gt_dir: str) -> Dict[str, str]:
    """MVTec: ground_truth/<defect_type>/<stem>_mask.png."""
    mapping: Dict[str, str] = {}
    for img_p in defect_paths:
        p = Path(img_p)
        mask = Path(gt_dir) / p.parent.name / (p.stem + "_mask.png")
        if mask.exists():
            mapping[img_p] = str(mask)
    return mapping


def _resolve_visa_masks(defect_paths: List[str], masks_dir: str) -> Dict[str, str]:
    """VisA: Masks/Anomaly/<stem>.png (fallback: exact filename)."""
    mapping: Dict[str, str] = {}
    m_dir = Path(masks_dir)
    for img_p in defect_paths:
        stem = Path(img_p).stem
        mask = m_dir / (stem + ".png")
        if mask.exists():
            mapping[img_p] = str(mask)
            continue
        exact = m_dir / Path(img_p).name
        if exact.exists():
            mapping[img_p] = str(exact)
    return mapping


def _get_image_lists(
    dataset_key: str,
    real_data_dir: str,
    seed: int = 42,
    test_split: float = 0.2,
) -> Optional[Dict[str, Any]]:
    """
    Returns {train_normal, test_good, test_defect, mask_map} or None if missing.
    mask_map: {defect_img_path: mask_path}
    """
    base = Path(real_data_dir)

    if dataset_key == "isp_LSM_1":
        ds = base / "isp" / "unsupervised" / "LSM_1"
        if not ds.exists():
            logger.warning("isp_LSM_1 not found: %s", ds)
            return None
        train_normal = _glob_images(str(ds / "train" / "good"))
        test_good    = _glob_images(str(ds / "test" / "good"))
        area_defects   = _glob_images(str(ds / "test" / "area"))
        points_defects = _glob_images(str(ds / "test" / "points"))
        test_defect    = area_defects + points_defects
        mask_map       = {
            **_resolve_isp_masks(area_defects,   str(ds / "ground_truth" / "area")),
            **_resolve_isp_masks(points_defects, str(ds / "ground_truth" / "points")),
        }

    elif dataset_key == "mvtec_cable":
        ds = base / "mvtec" / "cable"
        if not ds.exists():
            logger.warning("mvtec_cable not found: %s", ds)
            return None
        train_normal = _glob_images(str(ds / "train" / "good"))
        test_good    = _glob_images(str(ds / "test" / "good"))
        test_defect: List[str] = []
        for sub in sorted((ds / "test").iterdir()):
            if sub.is_dir() and sub.name != "good":
                test_defect.extend(_glob_images(str(sub)))
        mask_map = _resolve_mvtec_masks(test_defect, str(ds / "ground_truth"))

    elif dataset_key == "visa_cashew":
        ds = base / "visa" / "cashew" / "Data"
        if not ds.exists():
            logger.warning("visa_cashew not found: %s", ds)
            return None
        all_normal   = _glob_images(str(ds / "Images" / "Normal"))
        train_normal, test_good = _split_normal(all_normal, test_split=test_split, seed=seed)
        test_defect  = _glob_images(str(ds / "Images" / "Anomaly"))
        mask_map     = _resolve_visa_masks(test_defect, str(ds / "Masks" / "Anomaly"))

    elif dataset_key == "visa_pcb":
        ds = base / "visa" / "pcb4" / "Data"
        if not ds.exists():
            logger.warning("visa_pcb (pcb4) not found: %s", ds)
            return None
        all_normal   = _glob_images(str(ds / "Images" / "Normal"))
        train_normal, test_good = _split_normal(all_normal, test_split=test_split, seed=seed)
        test_defect  = _glob_images(str(ds / "Images" / "Anomaly"))
        mask_map     = _resolve_visa_masks(test_defect, str(ds / "Masks" / "Anomaly"))

    else:
        logger.warning("Unknown dataset_key: %s", dataset_key)
        return None

    logger.info(
        "%s: train_normal=%d  test_good=%d  test_defect=%d  masks_matched=%d",
        dataset_key, len(train_normal), len(test_good), len(test_defect), len(mask_map),
    )
    return dict(
        train_normal=train_normal,
        test_good=test_good,
        test_defect=test_defect,
        mask_map=mask_map,
    )


# ---------------------------------------------------------------------------
# GPU check
# ---------------------------------------------------------------------------

def _check_gpu() -> int:
    """Return CUDA device index (0). Exit if unavailable."""
    try:
        import torch
        if not torch.cuda.is_available():
            logger.error("GPU not available. Detection mode requires CUDA.")
            sys.exit(1)
        return 0
    except ImportError:
        logger.error("PyTorch not installed. Detection mode requires torch + CUDA.")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Bounding-box extraction from composite vs background diff
# ---------------------------------------------------------------------------

def _extract_defect_bboxes(
    synth_path: str,
    normal_path: str,
    min_area: int = 200,
) -> List[Tuple[int, int, int, int]]:
    """
    composite(synth) 와 background(normal) 의 차이를 이용해 defect bbox 추출.

    Returns list of (x, y, w, h) in pixel coords (composite 좌표계 기준).
    실패/매칭 불가 시 빈 리스트 반환.
    """
    if not _CV2_AVAILABLE:
        return []

    composite = cv2.imread(synth_path, cv2.IMREAD_COLOR)
    background = cv2.imread(normal_path, cv2.IMREAD_COLOR)
    if composite is None or background is None:
        return []

    h, w = composite.shape[:2]
    if background.shape[:2] != (h, w):
        background = cv2.resize(background, (w, h), interpolation=cv2.INTER_LINEAR)

    diff = cv2.absdiff(composite, background)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Blended (Poisson/alpha) defects produce a low-contrast diff; a fixed
    # threshold of 20 misses them entirely. Use a low fixed floor (8), and if
    # that yields nothing, fall back to Otsu on the diff (adapts to globally
    # faint blends). min_area still filters JPEG-compression speckle.
    kernel = np.ones((3, 3), np.uint8)

    _, thresh = cv2.threshold(blur, 8, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, kernel, iterations=3)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        dilated = cv2.dilate(thresh, kernel, iterations=3)
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bboxes: List[Tuple[int, int, int, int]] = []
    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
        if bw * bh >= min_area:
            bboxes.append((int(x), int(y), int(bw), int(bh)))
    return bboxes


def _mask_to_bboxes(
    mask_path: str,
    min_area: int = 50,
) -> Tuple[List[Tuple[int, int, int, int]], int, int]:
    """
    binary mask PNG → bounding boxes.

    Returns (bboxes [(x,y,w,h)], mask_w, mask_h). 실패 시 ([], 0, 0).
    """
    if not _CV2_AVAILABLE:
        return [], 0, 0

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return [], 0, 0

    mh, mw = mask.shape[:2]
    _, binm = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bboxes: List[Tuple[int, int, int, int]] = []
    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
        if bw * bh >= min_area:
            bboxes.append((int(x), int(y), int(bw), int(bh)))
    return bboxes, mw, mh


def _bboxes_to_yolo_lines(
    bboxes: List[Tuple[int, int, int, int]],
    img_w: int,
    img_h: int,
    class_id: int = 0,
) -> List[str]:
    """Convert pixel (x,y,w,h) bboxes to normalized YOLO lines: 'cls cx cy w h'."""
    lines: List[str] = []
    if img_w <= 0 or img_h <= 0:
        return lines
    for (x, y, bw, bh) in bboxes:
        cx = (x + bw / 2.0) / img_w
        cy = (y + bh / 2.0) / img_h
        nw = bw / img_w
        nh = bh / img_h
        # clamp to [0, 1]
        cx = min(max(cx, 0.0), 1.0)
        cy = min(max(cy, 0.0), 1.0)
        nw = min(max(nw, 0.0), 1.0)
        nh = min(max(nh, 0.0), 1.0)
        if nw <= 0.0 or nh <= 0.0:
            continue
        lines.append(f"{class_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
    return lines


# ---------------------------------------------------------------------------
# Synthetic annotations loader
# ---------------------------------------------------------------------------

def _load_synth_annotations(synth_root: str, dataset_key: str) -> List[Dict[str, Any]]:
    """
    Load {synth_root}/{dataset_key}/annotations.json.

    Returns list of dicts with image_path, normal_image, source_roi fields.
    Filter: dry_run=False (또는 미지정) 이고 image_path 파일이 실제 존재하는 항목만.
    """
    ann_path = Path(synth_root) / dataset_key / "annotations.json"
    if not ann_path.exists():
        # Surface the fully-resolved path so a wrong --*_synthetic_dir arg or a
        # missing per-dataset annotations.json is immediately diagnosable.
        logger.error(
            "annotations.json NOT FOUND for '%s' -> synth_annotations will be EMPTY "
            "(n_synth=0). Looked at: %s  (synth_root=%s)",
            dataset_key, ann_path.resolve(), synth_root,
        )
        return []

    try:
        raw = load_json(str(ann_path))
    except Exception as exc:
        logger.warning("Failed to load annotations.json (%s): %s", ann_path, exc)
        return []

    # annotations.json may be a list or a dict with an "annotations"/"items" key
    if isinstance(raw, dict):
        entries = raw.get("annotations") or raw.get("items") or raw.get("data") or []
    elif isinstance(raw, list):
        entries = raw
    else:
        entries = []

    synth_base = Path(synth_root) / dataset_key

    def _resolve_path(val: Optional[str]) -> Optional[str]:
        if not val:
            return None
        p = Path(val)
        if p.is_absolute() and p.exists():
            return str(p)
        # try relative to synth dataset dir, then images/ subdir
        cand1 = synth_base / val
        if cand1.exists():
            return str(cand1)
        cand2 = synth_base / "images" / Path(val).name
        if cand2.exists():
            return str(cand2)
        if p.exists():
            return str(p)
        return None

    valid: List[Dict[str, Any]] = []
    skipped_dry = 0
    skipped_missing = 0
    for e in entries:
        if not isinstance(e, dict):
            continue
        if e.get("dry_run") is True:
            skipped_dry += 1
            continue
        img_p = _resolve_path(e.get("image_path"))
        if img_p is None:
            skipped_missing += 1
            continue
        norm_p = _resolve_path(e.get("normal_image"))
        # Prefer an explicit GT mask when the synthesis pipeline persists one.
        # AROMA's generate_defects.py currently writes no mask, but other
        # pipelines (e.g. stage4 fmt='cls') write '{stem}_mask.png'. Accept
        # any of these keys, plus a sibling masks/ file as a last resort.
        mask_raw = (
            e.get("mask")
            or e.get("roi_mask")
            or e.get("mask_path")
        )
        mask_p = _resolve_path(mask_raw)
        if mask_p is None and img_p:
            stem = Path(img_p).stem
            for cand in (
                synth_base / "masks" / f"{stem}.png",
                synth_base / "masks" / f"{stem}_mask.png",
                Path(img_p).with_name(f"{stem}_mask.png"),
            ):
                if cand.exists():
                    mask_p = str(cand)
                    break
        valid.append({
            "image_path": img_p,
            "normal_image": norm_p,
            "source_roi": e.get("source_roi"),
            "mask_path": mask_p,
        })

    log_fn = logger.info if valid else logger.error
    log_fn(
        "Synth annotations %s: %d valid  (dry_run skipped=%d, missing skipped=%d, "
        "total entries=%d) from %s",
        dataset_key, len(valid), skipped_dry, skipped_missing, len(entries), ann_path,
    )
    if not valid and entries:
        # File present and non-empty but every entry was filtered out — almost
        # always image_path fields pointing at paths that don't resolve under
        # synth_root. Flag explicitly so it isn't silently treated as n_synth=0.
        logger.error(
            "  -> %s annotations.json had %d entries but 0 resolved; check that "
            "image_path values exist under %s (or are valid absolute paths)",
            dataset_key, len(entries), Path(synth_root) / dataset_key,
        )
    return valid


# ---------------------------------------------------------------------------
# Image staging helper (symlink on posix, copy on windows)
# ---------------------------------------------------------------------------

def _link_or_copy(src: str, dst: str, force_copy: bool = False) -> None:
    # force_copy=True: always copy bytes (e.g. building a persistent YOLO cache
    # that must survive the tmpdir / live on a different filesystem). Symlinks
    # into a tmpdir would dangle once the tmpdir is cleaned up.
    use_symlink = os.name != "nt" and not force_copy
    if use_symlink:
        if not Path(dst).exists():
            try:
                os.symlink(src, dst)
                return
            except OSError:
                pass
        else:
            return
    shutil.copy2(src, dst)


def _image_size(path: str) -> Tuple[int, int]:
    """Return (w, h) of an image. (0,0) on failure."""
    if _CV2_AVAILABLE:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is not None:
            h, w = img.shape[:2]
            return w, h
    try:
        from PIL import Image  # type: ignore[import]
        with Image.open(path) as im:
            return im.size  # (w, h)
    except Exception:
        return 0, 0


# ---------------------------------------------------------------------------
# YOLO label writers
# ---------------------------------------------------------------------------

def _template_match_bbox(
    synth_path: str,
    template_path: str,
    min_area: int = 200,
) -> List[Tuple[int, int, int, int]]:
    """source_roi template matching fallback for bbox extraction.

    Used when normal_image is unavailable (stale path from local_staging).
    """
    if not _CV2_AVAILABLE:
        return []
    composite = cv2.imread(synth_path, cv2.IMREAD_COLOR)
    template = cv2.imread(template_path, cv2.IMREAD_COLOR)
    if composite is None or template is None:
        return []
    th, tw = template.shape[:2]
    ch, cw = composite.shape[:2]
    if tw > cw or th > ch:
        return []
    # resize template proportionally to at most 40% of composite
    max_ratio = 0.4
    if tw / cw > max_ratio or th / ch > max_ratio:
        scale = min(max_ratio * cw / tw, max_ratio * ch / th)
        template = cv2.resize(template, (int(tw * scale), int(th * scale)))
        th, tw = template.shape[:2]
    result = cv2.matchTemplate(composite, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)
    if max_val < 0.3:  # low confidence — skip
        return []
    x, y = max_loc
    bw, bh = tw, th
    if bw * bh < min_area:
        return []
    return [(int(x), int(y), int(bw), int(bh))]


def _write_yolo_labels(
    synth_annotations: List[Dict[str, Any]],
    label_dir: str,
    image_dir_out: str,
    min_area: int = 200,
) -> int:
    """
    synthetic annotation 마다 defect bbox 추출 후 YOLO label 작성.

    bbox 가 없으면 (label 파일 없음 → background-only, YOLO가 negative로 취급) skip.
    image 는 image_dir_out 로 stage.

    Returns count of images with valid labels.
    """
    Path(label_dir).mkdir(parents=True, exist_ok=True)
    Path(image_dir_out).mkdir(parents=True, exist_ok=True)

    n_with_labels = 0
    for i, ann in enumerate(synth_annotations):
        synth_path = ann.get("image_path")
        normal_path = ann.get("normal_image")
        mask_path = ann.get("mask_path")
        if not synth_path:
            continue

        bboxes: List[Tuple[int, int, int, int]] = []
        img_w, img_h = _image_size(synth_path)
        if img_w <= 0 or img_h <= 0:
            continue

        # Prefer an explicit GT mask when available (exact bbox); only fall back
        # to composite-vs-normal image differencing when no mask is persisted.
        if mask_path and Path(mask_path).exists():
            m_bboxes, mw, mh = _mask_to_bboxes(mask_path, min_area=min_area)
            if m_bboxes and (mw, mh) != (img_w, img_h) and mw > 0 and mh > 0:
                sx = img_w / mw
                sy = img_h / mh
                m_bboxes = [
                    (int(x * sx), int(y * sy), int(bw * sx), int(bh * sy))
                    for (x, y, bw, bh) in m_bboxes
                ]
            bboxes = m_bboxes
        if not bboxes and normal_path and Path(normal_path).exists():
            bboxes = _extract_defect_bboxes(synth_path, normal_path, min_area=min_area)

        # Final fallback: source_roi template matching (handles stale
        # normal_image paths left over from local_staging runs).
        source_roi_path = ann.get("source_roi")
        if not bboxes and source_roi_path and Path(source_roi_path).exists() and _CV2_AVAILABLE:
            bboxes = _template_match_bbox(synth_path, source_roi_path, min_area=min_area)

        stem = f"syn_{i:06d}"
        img_dst = str(Path(image_dir_out) / f"{stem}{Path(synth_path).suffix}")
        _link_or_copy(synth_path, img_dst)

        if not bboxes:
            # no label file → treated as background negative
            continue

        lines = _bboxes_to_yolo_lines(bboxes, img_w, img_h, class_id=0)
        if not lines:
            continue

        lbl_dst = Path(label_dir) / f"{stem}.txt"
        lbl_dst.write_text("\n".join(lines) + "\n", encoding="utf-8")
        n_with_labels += 1

    return n_with_labels


def _stage_background_images(
    normal_paths: List[str],
    image_dir_out: str,
    prefix: str = "bg",
) -> int:
    """Stage normal/background images into train images (no label = negative)."""
    Path(image_dir_out).mkdir(parents=True, exist_ok=True)
    n = 0
    for i, p in enumerate(normal_paths):
        if not Path(p).exists():
            continue
        dst = str(Path(image_dir_out) / f"{prefix}_{i:06d}{Path(p).suffix}")
        _link_or_copy(p, dst)
        n += 1
    return n


# ---------------------------------------------------------------------------
# Local cache (Drive -> /tmp staging for I/O acceleration; Linux/Colab only)
# ---------------------------------------------------------------------------

def _local_cache_for_yolo(
    ds: str,
    lists: Dict[str, Any],
    synth_by_cond: Dict[str, List[Dict[str, Any]]],
    random_synth_dir: str,
    aroma_synth_dir: str,
    cache_base: str = "/tmp/aroma_exp4v2_cache",
    num_workers: Optional[int] = None,
) -> Tuple[Dict[str, Any], Dict[str, List[Dict[str, Any]]]]:
    """Stage Drive images to local /tmp for I/O acceleration. Linux/Colab only.

    Returns (new_lists, new_synth_by_cond) with paths rewritten to /tmp.
    Idempotent per-file. mask_map keys rewritten in lockstep with test_defect.
    """
    if num_workers is None:
        num_workers = min(32, (os.cpu_count() or 4) * 2)

    cache_ds = Path(cache_base) / ds
    real_dir = cache_ds / "real"
    real_imgs = real_dir / "images"
    real_masks = real_dir / "masks"
    for d in (real_imgs, real_masks):
        d.mkdir(parents=True, exist_ok=True)

    def _copy_if_missing(src: str, dst: Path) -> str:
        if not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, str(dst))
        return str(dst)

    n_real = len(lists["train_normal"]) + len(lists["test_defect"])
    n_synth = sum(len(synth_by_cond.get(c, [])) for c in ("random", "aroma"))
    logger.info("[LocalCache] %s: copying %d images -> %s", ds, n_real + n_synth, cache_base)
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=num_workers) as ex:
        # real train_normal
        tn_futs = [
            ex.submit(_copy_if_missing, p, real_imgs / f"tn_{i:05d}{Path(p).suffix}")
            for i, p in enumerate(lists["train_normal"])
        ]
        # real test_defect
        td_futs = [
            ex.submit(_copy_if_missing, p, real_imgs / f"td_{i:05d}{Path(p).suffix}")
            for i, p in enumerate(lists["test_defect"])
        ]
        local_tn = [f.result() for f in tn_futs]
        local_td = [f.result() for f in td_futs]

        # mask_map: rewrite keys lockstep with test_defect
        local_mask_map: Dict[str, str] = {}
        mask_futs = {}
        for i, (orig_p, local_p) in enumerate(zip(lists["test_defect"], local_td)):
            if orig_p in lists["mask_map"]:
                mdst = real_masks / f"mask_{i:05d}.png"
                mask_futs[local_p] = ex.submit(
                    _copy_if_missing, lists["mask_map"][orig_p], mdst
                )
        for local_p, fut in mask_futs.items():
            local_mask_map[local_p] = fut.result()

        # synth per condition
        synth_roots = {"random": random_synth_dir, "aroma": aroma_synth_dir}
        new_synth_by_cond: Dict[str, List[Dict[str, Any]]] = {
            "baseline": list(synth_by_cond.get("baseline", []))
        }
        for cond in ("random", "aroma"):
            cond_imgs = cache_ds / cond / "images"
            anns = synth_by_cond.get(cond, [])
            futs_list = []
            for j, ann in enumerate(anns):
                img_dst = cond_imgs / f"syn_{j:05d}{Path(ann['image_path']).suffix}"
                f_img = ex.submit(_copy_if_missing, ann["image_path"], img_dst)
                f_norm = None
                if ann.get("normal_image"):
                    norm_dst = cond_imgs / f"nrm_{j:05d}{Path(ann['normal_image']).suffix}"
                    f_norm = ex.submit(_copy_if_missing, ann["normal_image"], norm_dst)
                futs_list.append((ann, f_img, f_norm))
            new_anns = []
            for ann, f_img, f_norm in futs_list:
                new_ann = dict(ann)
                new_ann["image_path"] = f_img.result()
                if f_norm is not None:
                    new_ann["normal_image"] = f_norm.result()
                new_anns.append(new_ann)
            new_synth_by_cond[cond] = new_anns
            # copy annotations.json for fidelity
            src_json = Path(synth_roots[cond]) / ds / "annotations.json"
            if src_json.exists():
                _copy_if_missing(str(src_json), cache_ds / cond / "annotations.json")

    elapsed = time.time() - t0
    logger.info(
        "[LocalCache] %s ready: %.1fs (train_normal=%d defect=%d masks=%d random=%d aroma=%d)",
        ds, elapsed, len(local_tn), len(local_td), len(local_mask_map),
        len(new_synth_by_cond.get("random", [])), len(new_synth_by_cond.get("aroma", [])),
    )

    new_lists = {
        **lists,
        "train_normal": local_tn,
        "test_defect": local_td,
        "mask_map": local_mask_map,
    }
    return new_lists, new_synth_by_cond


def _get_real_test_images_and_labels(
    test_defect_paths: List[str],
    mask_map: Dict[str, str],
    label_dir: str,
    image_dir_out: str,
    min_area: int = 50,
    force_copy: bool = False,
) -> int:
    """
    real defect 이미지 + GT mask → YOLO label set 구성 (condition-agnostic).

    train pool 을 넘기면 train label, val pool 을 넘기면 val label 이 된다.

    각 defect 이미지(mask 보유) 마다:
      - binary mask PNG → bounding boxes
      - YOLO label 작성
      - 이미지 stage
    mask 가 없는 defect 이미지는 제외 (label 없으면 학습/평가 모두 불가).

    Returns count of images with labels.
    """
    Path(label_dir).mkdir(parents=True, exist_ok=True)
    Path(image_dir_out).mkdir(parents=True, exist_ok=True)

    n = 0
    n_no_mask = n_bad_size = n_no_bbox = n_no_lines = 0
    for i, img_p in enumerate(test_defect_paths):
        mask_p = mask_map.get(img_p)
        if not mask_p:
            n_no_mask += 1
            continue

        img_w, img_h = _image_size(img_p)
        if img_w <= 0 or img_h <= 0:
            n_bad_size += 1
            continue

        m_bboxes, mw, mh = _mask_to_bboxes(mask_p, min_area=min_area)
        if not m_bboxes:
            n_no_bbox += 1
            continue

        # mask 해상도가 이미지와 다르면 bbox 를 이미지 좌표로 스케일
        if (mw, mh) != (img_w, img_h) and mw > 0 and mh > 0:
            sx = img_w / mw
            sy = img_h / mh
            m_bboxes = [
                (int(x * sx), int(y * sy), int(bw * sx), int(bh * sy))
                for (x, y, bw, bh) in m_bboxes
            ]

        lines = _bboxes_to_yolo_lines(m_bboxes, img_w, img_h, class_id=0)
        if not lines:
            n_no_lines += 1
            continue

        stem = f"real_{i:06d}"
        img_dst = str(Path(image_dir_out) / f"{stem}{Path(img_p).suffix}")
        _link_or_copy(img_p, img_dst, force_copy=force_copy)
        (Path(label_dir) / f"{stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        n += 1

    if n == 0 and test_defect_paths:
        # Convert the silent "real=0 -> train on backgrounds" symptom into an
        # actionable line that pinpoints WHERE extraction dropped every image.
        logger.error(
            "    _get_real: 0 labels from %d defect imgs "
            "(no_mask=%d bad_size=%d no_bbox=%d no_lines=%d, min_area=%d)",
            len(test_defect_paths), n_no_mask, n_bad_size, n_no_bbox,
            n_no_lines, min_area,
        )

    return n


def _write_yolo_yaml(
    root_dir: str,
    train_img_subdir: str,
    val_img_subdir: str,
    yaml_path: str,
) -> None:
    """Write ultralytics dataset YAML."""
    content = (
        f"path: {root_dir}\n"
        f"train: {train_img_subdir}\n"
        f"val: {val_img_subdir}\n"
        f"nc: 1\n"
        f"names: ['defect']\n"
    )
    Path(yaml_path).write_text(content, encoding="utf-8")


# ---------------------------------------------------------------------------
# Train heartbeat
# ---------------------------------------------------------------------------

def _train_with_heartbeat(train_fn, label: str, interval_s: int = 60):
    """Run train_fn() with periodic heartbeat logs so silent training is visible."""
    stop = threading.Event()
    t0 = time.time()

    def _beat():
        while not stop.wait(interval_s):
            logger.info("  ⏳ %s training... %.0fs elapsed", label, time.time() - t0)

    beat = threading.Thread(target=_beat, daemon=True)
    beat.start()
    try:
        result = train_fn()
    finally:
        stop.set()
    return result, time.time() - t0


# ---------------------------------------------------------------------------
# Real YOLO dataset cache (build-once-per-dataset, optional Drive persistence)
# ---------------------------------------------------------------------------

def _validate_real_cache(
    cache_root: Path,
    min_area: int,
    val_frac: float,
    split_seed: int,
) -> Optional[Dict[str, Any]]:
    """Return parsed meta dict if cache valid + params match, else None."""
    meta_path = cache_root / "meta.json"
    if not meta_path.is_file():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None

    if meta.get("schema_version") != 2:
        return None
    if meta.get("min_area") != min_area:
        return None
    if meta.get("val_frac") != val_frac:
        return None
    if meta.get("split_seed") != split_seed:
        return None
    if int(meta.get("n_labeled", 0)) <= 0:
        return None

    # structural check: train/val non-empty, every image has sibling label
    for split in ("train", "val"):
        img_dir = cache_root / "images" / split
        lbl_dir = cache_root / "labels" / split
        if not img_dir.is_dir() or not lbl_dir.is_dir():
            return None
        imgs = [p for p in img_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
        if not imgs:
            return None
        for img in imgs:
            if not (lbl_dir / (img.stem + ".txt")).is_file():
                return None
    return meta


def _stage_pairs_into(
    triple: Tuple[List[str], List[str], int],
    img_dst_dir: str,
    lbl_dst_dir: str,
) -> None:
    """Copy cached (image, label) pairs into a condition tmpdir.

    Always copies bytes (not symlinks): cache may live on a different
    filesystem (Drive) than the tmpdir.
    """
    img_paths, lbl_paths, _ = triple
    for ip in img_paths:
        shutil.copy2(ip, Path(img_dst_dir) / Path(ip).name)
    for lp in lbl_paths:
        dst = Path(lbl_dst_dir) / Path(lp).name
        if Path(lp).is_file():
            shutil.copy2(lp, dst)


def _build_or_load_real_yolo_dataset(
    ds_key: str,
    mask_map: Dict[str, str],
    defect_splits: Dict[str, List[str]],
    cache_dir: Optional[str],
    min_area: int = 50,
    val_frac: float = 0.5,
    split_seed: int = 42,
) -> Dict[str, Tuple[List[str], List[str], int]]:
    """
    Build or load cached per-split real YOLO dataset.

    Returns {split: (img_paths, lbl_paths, n_labeled)} for train/val/test.
    When cache_dir is None: builds fresh, no persistence (behavior identical
    to the previous per-condition build).
    When cache_dir provided: checks Drive cache, builds+saves on miss.
    """
    use_cache = cache_dir is not None
    cache_root = Path(cache_dir) / ds_key / "real" if use_cache else None

    # --- CACHE HIT ---
    if use_cache and cache_root is not None:
        meta = _validate_real_cache(cache_root, min_area, val_frac, split_seed)
        if meta is not None:
            result: Dict[str, Tuple[List[str], List[str], int]] = {}
            for split in ("train", "val", "test"):
                img_dir = cache_root / "images" / split
                lbl_dir = cache_root / "labels" / split
                imgs = (
                    sorted(
                        str(p)
                        for p in img_dir.glob("*")
                        if p.suffix.lower() in (".jpg", ".jpeg", ".png")
                    )
                    if img_dir.is_dir()
                    else []
                )
                lbls = [str(lbl_dir / (Path(p).stem + ".txt")) for p in imgs]
                lbls = [l for l in lbls if Path(l).is_file()]
                result[split] = (imgs, lbls, len(lbls))
            total = sum(v[2] for v in result.values())
            logger.info(
                "[YoloCache] %s: loaded %d real labels from cache (train=%d val=%d)",
                ds_key, total, result["train"][2], result["val"][2],
            )
            return result

    # --- CACHE MISS: build ---
    if use_cache and cache_root is not None:
        tmp_root = cache_root.parent / "real.building"
        if tmp_root.exists():
            shutil.rmtree(tmp_root, ignore_errors=True)
        target = tmp_root
    else:
        target = Path(tempfile.mkdtemp(prefix=f"realbuild_{ds_key}_"))

    result = {}
    per_split_meta: Dict[str, Dict[str, int]] = {}
    for split in ("train", "val", "test"):
        img_out = target / "images" / split
        lbl_out = target / "labels" / split
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)
        paths = defect_splits.get(split, [])
        # reuse existing _get_real_test_images_and_labels() unchanged in logic;
        # force_copy so cached images survive tmpdir cleanup / different FS.
        n_labeled = _get_real_test_images_and_labels(
            test_defect_paths=paths,
            mask_map=mask_map,
            label_dir=str(lbl_out),
            image_dir_out=str(img_out),
            min_area=min_area,
            force_copy=use_cache,
        )
        imgs = sorted(
            str(p)
            for p in img_out.glob("*")
            if p.suffix.lower() in (".jpg", ".jpeg", ".png")
        )
        lbls = [str(lbl_out / (Path(p).stem + ".txt")) for p in imgs]
        lbls = [l for l in lbls if Path(l).is_file()]
        result[split] = (imgs, lbls, len(lbls))
        per_split_meta[split] = {"n_images": len(imgs), "n_labeled": len(lbls)}

    total = sum(v[2] for v in result.values())

    if use_cache and cache_root is not None:
        meta = {
            "dataset_key": ds_key,
            "schema_version": 2,
            "splits": per_split_meta,
            "n_images": sum(v["n_images"] for v in per_split_meta.values()),
            "n_labeled": total,
            "min_area": min_area,
            "val_frac": val_frac,
            "split_seed": split_seed,
            "build_timestamp": datetime.now(timezone.utc).isoformat(),
        }
        (target / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        # atomic publish: tmp -> real
        if cache_root.exists():
            shutil.rmtree(cache_root, ignore_errors=True)
        cache_root.parent.mkdir(parents=True, exist_ok=True)
        os.replace(str(target), str(cache_root))

        # repoint result paths from tmp_root to cache_root
        def _repoint(paths: List[str], old_root: Path, new_root: Path) -> List[str]:
            return [str(new_root / Path(p).relative_to(old_root)) for p in paths]

        result2: Dict[str, Tuple[List[str], List[str], int]] = {}
        for split in ("train", "val", "test"):
            imgs, lbls, _n = result[split]
            new_imgs = _repoint(imgs, target, cache_root)
            new_lbls = _repoint(lbls, target, cache_root)
            new_lbls = [l for l in new_lbls if Path(l).is_file()]
            result2[split] = (new_imgs, new_lbls, len(new_lbls))
        result = result2
        logger.info(
            "[YoloCache] %s: built+saved %d real labels to cache (train=%d val=%d)",
            ds_key, total, result["train"][2], result["val"][2],
        )
    else:
        logger.info("[YoloCache] %s: built %d real labels (no cache)", ds_key, total)

    return result


# ---------------------------------------------------------------------------
# Per-condition YOLO runner (COCO pretrained start)
# ---------------------------------------------------------------------------

def _run_yolo_condition(
    model_name: str,
    condition: str,
    synth_annotations: List[Dict[str, Any]],
    train_defect_paths: List[str],
    train_normal_paths: List[str],
    val_defect_paths: List[str],
    mask_map: Dict[str, str],
    checkpoint_dir: str,
    real_sets: Dict[str, Tuple[List[str], List[str], int]],
    baseline_epochs: int = 50,
    imgsz: int = 256,
    seed: int = 42,
    device: int = 0,
) -> Dict[str, Any]:
    """
    하나의 (model, condition) 조합을 학습/평가.

    모든 조건이 real labeled defect (train split)를 공통으로 학습한다.
      baseline:        real labeled defect 만 (+ 약간의 real normal background) →
                       COCO 에서 시작, save=True 로 best.pt 영속화.
      random / aroma:  real labeled defect + synth → COCO 에서 처음부터 학습,
                       save=False.

    Returns {map50, map50_95, precision, recall, n_train, n_real_train,
             n_synth_train, weights_path(baseline only)}.
    """
    try:
        from ultralytics import YOLO  # type: ignore[import]
    except ImportError:
        logger.error("ultralytics not installed. Run: pip install ultralytics")
        return {"error": "ultralytics_missing"}

    # local fast scratch on Colab/Linux; default temp on Windows
    tmp_dir_root = "/tmp" if os.name != "nt" else None

    try:
        with tempfile.TemporaryDirectory(dir=tmp_dir_root) as tmpdir:
            train_img = str(Path(tmpdir) / "train" / "images")
            train_lbl = str(Path(tmpdir) / "train" / "labels")
            val_img   = str(Path(tmpdir) / "val" / "images")
            val_lbl   = str(Path(tmpdir) / "val" / "labels")
            for d in (train_img, train_lbl, val_img, val_lbl):
                Path(d).mkdir(parents=True, exist_ok=True)

            # --- TRAIN: real labeled defects (ALL conditions) ---
            # Pre-built/cached real (image,label) pairs are staged in;
            # the build itself happened once per dataset in the caller.
            _stage_pairs_into(real_sets["train"], train_img, train_lbl)
            n_real = real_sets["train"][2]
            logger.info("    train real labeled defects: %d", n_real)
            if n_real == 0:
                logger.error(
                    "    %s/%s: no real train labels — cannot train",
                    model_name, condition,
                )
                return {"error": "no_real_train_labels"}

            # --- TRAIN: a few real normals as background negatives (ALL conditions) ---
            n_bg = _stage_background_images(
                train_normal_paths[:n_real], train_img, prefix="bg"
            )
            logger.info("    train background negatives: %d", n_bg)

            # --- TRAIN: add synthetic (random/aroma only) ---
            n_synth = 0
            if condition != "baseline":
                # An empty synth set means this run would be identical to
                # baseline (real-only) yet get reported as "random"/"aroma" — a
                # misleading row. Refuse instead of silently producing it.
                if not synth_annotations:
                    logger.error(
                        "    %s/%s: no synthetic annotations — refusing to report "
                        "real-only run as '%s'. Check %s annotations.json.",
                        model_name, condition, condition, condition,
                    )
                    return {"error": "no_synth_annotations"}
                n_synth = _write_yolo_labels(
                    synth_annotations=synth_annotations,
                    label_dir=train_lbl,
                    image_dir_out=train_img,
                    min_area=200,
                )
                logger.info(
                    "    %s train: %d/%d synth imgs labeled (additive)",
                    condition, n_synth, len(synth_annotations),
                )
            n_train = n_real + n_synth

            # --- VAL: real labeled defects (disjoint from train) ---
            _stage_pairs_into(real_sets["val"], val_img, val_lbl)
            n_val = real_sets["val"][2]
            logger.info("    val: %d real defect imgs with labels", n_val)
            if n_val == 0:
                logger.error(
                    "    %s/%s: no val images with labels — cannot evaluate",
                    model_name, condition,
                )
                return {"error": "no_val_labels"}

            # --- Dataset YAML ---
            yaml_path = str(Path(tmpdir) / "data.yaml")
            _write_yolo_yaml(
                root_dir=tmpdir,
                train_img_subdir="train/images",
                val_img_subdir="val/images",
                yaml_path=yaml_path,
            )

            # --- Weights selection: baseline=COCO, others=baseline best.pt ---
            Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
            try:
                import torch
                torch.set_float32_matmul_precision("medium")
            except Exception:
                pass

            if condition == "baseline":
                weights = f"{model_name}.pt"      # COCO pretrained (downloaded)
                epochs = baseline_epochs
                do_save = True                    # MUST persist best.pt
            else:
                # scratch: random/aroma도 COCO에서 시작, real+synth 처음부터 학습
                weights = f"{model_name}.pt"
                epochs = baseline_epochs
                do_save = False

            # YOLO("{model}.pt") loads COCO pretrained weights; .train() = transfer learning.
            # NOTE: do NOT pass resume=True (would continue the same run's epoch counter).
            model = YOLO(weights)

            def _do_train():
                return model.train(
                    data=yaml_path,
                    epochs=epochs,
                    imgsz=imgsz,
                    project=checkpoint_dir,
                    name=condition,
                    seed=seed,
                    verbose=False,
                    plots=False,
                    save=do_save,    # baseline: True → best.pt written
                    device=device,
                    exist_ok=True,
                )

            logger.info(
                "    fit start: %s / %s  weights=%s  train_imgs=%d (real=%d synth=%d)  "
                "epochs=%d imgsz=%d save=%s",
                model_name, condition, weights, n_train, n_real, n_synth,
                epochs, imgsz, do_save,
            )
            _, fit_elapsed = _train_with_heartbeat(
                _do_train, label=f"{model_name}/{condition}", interval_s=60,
            )
            logger.info("    fit done: %.1fs", fit_elapsed)

            # --- Validate ---
            logger.info("    val start: %s / %s", model_name, condition)
            t_val = time.time()
            # verbose=True so ultralytics surfaces its "images/instances" summary;
            # a silent 0.0 (no GT loaded / geometry-collapsed labels) would
            # otherwise be indistinguishable from a genuine 0.0 mAP.
            val_results = model.val(data=yaml_path, verbose=True, plots=False)
            logger.info("    val done: %.1fs", time.time() - t_val)

            box = val_results.box
            # Diagnostic: how many GT instances did ultralytics actually match?
            # If 0, the recorded 0.0 is a label/geometry problem, not model perf.
            try:
                n_gt = int(getattr(box, "nc", 0)) and int(
                    getattr(val_results, "seen", 0)
                )
                tp = getattr(box, "tp", None)
                n_tp = int(tp.sum()) if tp is not None and hasattr(tp, "sum") else None
                logger.info(
                    "    val GT check: staged_val_labels=%d  seen=%s  tp=%s  map50=%.4f",
                    n_val, getattr(val_results, "seen", "?"), n_tp,
                    float(box.map50),
                )
                if float(box.map50) == 0.0:
                    logger.warning(
                        "    %s/%s: val map50 == 0.0 — verify staged val labels land "
                        "on defects and imgsz=%d is large enough for the defect scale.",
                        model_name, condition, imgsz,
                    )
            except Exception as _diag_exc:
                logger.warning("    val GT check failed: %s", _diag_exc)
            out = {
                "map50":        round(float(box.map50), 4),
                "map50_95":     round(float(box.map), 4),
                "precision":    round(float(box.mp), 4),
                "recall":       round(float(box.mr), 4),
                "n_train":      int(n_train),
                "n_real_train": int(n_real),
                "n_synth_train": int(n_synth),
            }
            # baseline best.pt 위치를 caller 가 재사용할 수 있도록 보고
            if condition == "baseline":
                out["weights_path"] = str(
                    Path(checkpoint_dir) / condition / "weights" / "best.pt"
                )

    except Exception as exc:
        logger.error("Condition %s/%s failed: %s", model_name, condition, exc)
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        gc.collect()
        return {"error": str(exc)}

    return out


# ---------------------------------------------------------------------------
# Detection mode — datasets x conditions (single model)
# ---------------------------------------------------------------------------

def _run_detection_mode(
    model_keys: List[str],
    condition_keys: List[str],
    dataset_keys: List[str],
    random_synthetic_dir: str,
    aroma_synthetic_dir: str,
    real_data_dir: str,
    output_dir: str,
    baseline_epochs: int = 50,
    val_frac: float = 0.5,
    max_synth_per_ds: Optional[int] = None,
    synth_ratio: Optional[float] = None,
    imgsz: int = 256,
    seed: int = 42,
    existing_results: Optional[Dict[str, Any]] = None,
    output_path: Optional[str] = None,
    yolo_cache_dir: Optional[str] = None,
    local_cache: bool = True,
) -> Dict[str, Any]:
    """
    Iterate datasets x models x conditions (COCO pretrained start).
    Output: {dataset: {model: {condition: {map50, map50_95, precision, recall,
             n_train, n_real_train, n_synth_train, weights_path?}}}}

    CRITICAL:
    - real defect 는 seeded split 으로 train/val 분리 (disjoint). Val = real only.
    - baseline 을 항상 먼저 학습. random/aroma 도 COCO 에서 독립적으로 처음부터 학습.
    - checkpoint 는 {output_dir}/{ds}/{model}/baseline/weights/best.pt 에 영속.
    synth_ratio: real_train 수 대비 synth 비율. 지정 시 max_synth_per_ds 무시.
    """
    device = _check_gpu()
    existing = existing_results or {}
    results: Dict[str, Any] = {k: v for k, v in existing.items()}

    total_planned = len(dataset_keys) * len(model_keys) * len(condition_keys)
    n_done = sum(
        1
        for ds in dataset_keys
        for m in model_keys
        for c in condition_keys
        if existing.get(ds, {}).get(m, {}).get(c, {}).get("map50") is not None
    )
    n_remaining = total_planned - n_done
    logger.info(
        "Exp4v2 plan: %d total | %d done | %d remaining  (datasets=%s models=%s conditions=%s)",
        total_planned, n_done, n_remaining,
        dataset_keys, model_keys, condition_keys,
    )
    run_idx = 0

    for ds in dataset_keys:
        logger.info("=== Detection dataset: %s ===", ds)
        lists = _get_image_lists(ds, real_data_dir, seed=seed)
        if lists is None:
            logger.warning("Detection skip %s — dataset not found", ds)
            continue

        # synth annotations per condition (loaded once per dataset).
        # MOVED UP: must exist before the local cache so its paths get rewritten.
        synth_by_cond: Dict[str, List[Dict[str, Any]]] = {
            "baseline": [],
            "random":   _load_synth_annotations(random_synthetic_dir, ds),
            "aroma":    _load_synth_annotations(aroma_synthetic_dir, ds),
        }

        # Absolute cap (synth_ratio 미지정 시에만 적용).
        if max_synth_per_ds is not None and synth_ratio is None:
            for cond in ("random", "aroma"):
                anns = synth_by_cond.get(cond, [])
                if len(anns) > max_synth_per_ds:
                    rng_sub = random.Random(seed)
                    synth_by_cond[cond] = rng_sub.sample(anns, max_synth_per_ds)
                    logger.info(
                        "  [SubSample] %s/%s: %d → %d",
                        ds, cond, len(anns), max_synth_per_ds,
                    )

        # Local cache: stage Drive images to /tmp for I/O acceleration.
        # Linux/Colab only (os.name != "nt"). Rewrites lists + synth paths to /tmp.
        if local_cache and os.name != "nt":
            lists, synth_by_cond = _local_cache_for_yolo(
                ds, lists, synth_by_cond, random_synthetic_dir, aroma_synthetic_dir,
            )

        train_normal = lists["train_normal"]
        all_defect   = lists["test_defect"]
        mask_map     = lists["mask_map"]

        # real defect train/val split (seeded, disjoint, mask-bearing only)
        train_defect, val_defect = _split_defects(all_defect, mask_map, val_frac, seed)
        logger.info(
            "%s: real defect split — train=%d  val=%d  (val_frac=%.2f)",
            ds, len(train_defect), len(val_defect), val_frac,
        )
        if not train_defect or not val_defect:
            logger.warning(
                "%s: too few labeled defects to split "
                "(train=%d val=%d, need >=1 each) — skipping",
                ds, len(train_defect), len(val_defect),
            )
            continue

        # Ratio-based synth cap (synth_ratio 지정 시 max_synth_per_ds 보다 우선).
        if synth_ratio is not None:
            n_real_train = len(train_defect)
            for cond in ("random", "aroma"):
                anns = synth_by_cond.get(cond, [])
                cap = max(1, int(n_real_train * synth_ratio))
                if len(anns) > cap:
                    rng_sub = random.Random(seed)
                    synth_by_cond[cond] = rng_sub.sample(anns, cap)
                    logger.info(
                        "  [SynthRatio] %s/%s: %.2f x %d real_train -> cap=%d  (%d -> %d synth)",
                        ds, cond, synth_ratio, n_real_train, cap, len(anns), cap,
                    )
                elif anns:
                    logger.info(
                        "  [SynthRatio] %s/%s: %.2f x %d real_train -> cap=%d  (available=%d, no trim)",
                        ds, cond, synth_ratio, n_real_train, cap, len(anns),
                    )

        # Build (or load from Drive cache) the real YOLO (image,label) set ONCE
        # per dataset, hoisted out of the per-condition loop. baseline/random/aroma
        # all stage the SAME pre-built pairs into their tmpdirs.
        real_sets = _build_or_load_real_yolo_dataset(
            ds_key=ds,
            mask_map=mask_map,
            defect_splits={"train": train_defect, "val": val_defect, "test": []},
            cache_dir=yolo_cache_dir,
            min_area=50,
            val_frac=val_frac,
            split_seed=seed,
        )

        # checkpoint base persists under output_dir so best.pt survives the run
        ckpt_base = str(Path(output_dir) / ds)

        ds_results: Dict[str, Any] = dict(existing.get(ds, {}))

        for model_name in model_keys:
            model_results: Dict[str, Any] = dict(ds_results.get(model_name, {}))

            # Run baseline first for result ordering; random/aroma are independent
            ordered = (
                [c for c in condition_keys if c == "baseline"]
                + [c for c in condition_keys if c != "baseline"]
            )

            def _save_incremental() -> None:
                if not output_path:
                    return
                ds_results[model_name] = dict(model_results)
                results[ds] = dict(ds_results)
                try:
                    save_json(results, output_path)
                except Exception as _e:
                    logger.warning("Incremental save failed: %s", _e)

            for cond in ordered:
                if cond not in synth_by_cond:
                    logger.warning("Unknown condition %s — skipping", cond)
                    continue

                # Resume: skip only if already completed with valid map50
                _cached = existing.get(ds, {}).get(model_name, {}).get(cond)
                if _cached is not None and _cached.get("map50") is not None:
                    model_results[cond] = _cached
                    logger.info(
                        "  RESUME skip %s / %s / %s  (cached map50=%s)",
                        ds, model_name, cond, _cached.get("map50"),
                    )
                    continue

                run_idx += 1
                synth_ann = synth_by_cond[cond]
                logger.info(
                    "  [%d/%d] %s / %s / %s  synth_ann=%d",
                    run_idx, n_remaining, ds, model_name, cond, len(synth_ann),
                )

                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass
                gc.collect()

                res = _run_yolo_condition(
                    model_name=model_name,
                    condition=cond,
                    synth_annotations=synth_ann,
                    train_defect_paths=train_defect,
                    train_normal_paths=train_normal,
                    val_defect_paths=val_defect,
                    mask_map=mask_map,
                    checkpoint_dir=str(Path(ckpt_base) / model_name),
                    real_sets=real_sets,
                    baseline_epochs=baseline_epochs,
                    imgsz=imgsz,
                    seed=seed,
                    device=device,
                )
                model_results[cond] = res
                logger.info(
                    "    map50=%s  map50_95=%s  P=%s  R=%s",
                    res.get("map50"), res.get("map50_95"),
                    res.get("precision"), res.get("recall"),
                )

                _save_incremental()

            ds_results[model_name] = model_results

        results[ds] = ds_results

    return results


# ---------------------------------------------------------------------------
# Summary builder
# ---------------------------------------------------------------------------

def _fmt_delta(x: Dict[str, Any], y: Dict[str, Any]) -> str:
    """Format (x[c] - y[c]) in percentage points for the core metrics."""
    parts = []
    for col in ("map50", "map50_95", "precision", "recall"):
        xv, yv = x.get(col), y.get(col)
        if isinstance(xv, (int, float)) and isinstance(yv, (int, float)):
            parts.append(f"{col} {(xv - yv) * 100:+.2f}pp")
        else:
            parts.append(f"{col} N/A")
    return ", ".join(parts)


def _build_summary(results: Dict[str, Any]) -> str:
    """
    Build Markdown summary (COCO pretrained start).
    Per dataset/model: rows=conditions, cols=metrics.
    Delta section: AROMA − Baseline and AROMA − Random (CASDA-style, pp).
    """
    lines = [
        "# AROMA Exp 4 v2 — Supervised YOLOv8 Defect Detection 평가",
        "",
        "비교: Baseline (real-only) vs Random ROI (real+synth) vs AROMA ROI (real+synth)",
        "모든 조건이 real labeled defect (train split) 위에서 학습.",
        "baseline: COCO→real-only 학습. random/aroma: COCO에서 real+synth 처음부터 학습.",
        "Val = real defect (GT mask → bbox), train 과 disjoint.",
        "",
    ]

    all_conds = ALL_CONDITION_KEYS
    metric_cols = ["map50", "map50_95", "precision", "recall", "n_real_train", "n_synth_train"]

    for ds in sorted(results):
        ds_data = results[ds]
        lines += [f"## {ds}", ""]

        # which models present
        models_present = [m for m in ALL_MODEL_KEYS if m in ds_data]
        if not models_present:
            models_present = sorted(ds_data.keys())

        for model_name in models_present:
            m_data = ds_data.get(model_name, {})
            lines += [f"### {model_name}", ""]

            header = " | ".join(f"{c:>12}" for c in metric_cols)
            lines.append(f"| {'조건':<10} | {header} |")
            sep = " | ".join("-" * 12 for _ in metric_cols)
            lines.append(f"|{'-' * 12}|{sep}|")

            for cond in all_conds:
                cm = m_data.get(cond, {})
                cells = []
                for col in metric_cols:
                    val = cm.get(col)
                    if isinstance(val, bool):
                        cells.append(str(val))
                    elif isinstance(val, float):
                        cells.append(f"{val:.4f}")
                    elif isinstance(val, int):
                        cells.append(str(val))
                    else:
                        cells.append("N/A")
                row = " | ".join(f"{c:>12}" for c in cells)
                lines.append(f"| {cond:<10} | {row} |")
            lines.append("")

            # CASDA-style delta panel
            b = m_data.get("baseline", {})
            r = m_data.get("random", {})
            a = m_data.get("aroma", {})
            lines.append("**Delta (AROMA − Baseline)**: " + _fmt_delta(a, b))
            lines.append("**Delta (AROMA − Random)**:   " + _fmt_delta(a, r))
            lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run(
    model_keys: List[str],
    condition_keys: List[str],
    dataset_keys: List[str],
    random_synthetic_dir: str,
    aroma_synthetic_dir: str,
    real_data_dir: str,
    output_dir: str,
    baseline_epochs: int = 50,
    val_frac: float = 0.5,
    max_synth_per_ds: Optional[int] = None,
    synth_ratio: Optional[float] = None,
    imgsz: int = 256,
    seed: int = 42,
    resume: bool = False,
    yolo_cache_dir: Optional[str] = None,
    local_cache: bool = True,
) -> Dict[str, Any]:
    if not _CV2_AVAILABLE:
        logger.error("OpenCV (cv2) not installed. Run: pip install opencv-python-headless")
        return {"status": "error", "results": {}}

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    output_path = str(out / "exp4v2_results.json")

    existing_results: Dict[str, Any] = {}
    if resume and Path(output_path).exists():
        try:
            existing_results = load_json(output_path)
            n = sum(
                1
                for ds in existing_results
                for m in existing_results[ds]
                for c in existing_results[ds][m]
            )
            logger.info("Resume: loaded %d existing results from %s", n, output_path)
        except Exception as e:
            logger.warning("Resume: failed to load existing results: %s — starting fresh", e)

    results = _run_detection_mode(
        model_keys=model_keys,
        condition_keys=condition_keys,
        dataset_keys=dataset_keys,
        random_synthetic_dir=random_synthetic_dir,
        aroma_synthetic_dir=aroma_synthetic_dir,
        real_data_dir=real_data_dir,
        output_dir=output_dir,
        baseline_epochs=baseline_epochs,
        val_frac=val_frac,
        max_synth_per_ds=max_synth_per_ds,
        synth_ratio=synth_ratio,
        imgsz=imgsz,
        seed=seed,
        existing_results=existing_results,
        output_path=output_path,
        yolo_cache_dir=yolo_cache_dir,
        local_cache=local_cache,
    )

    save_json(results, output_path)
    (out / "exp4v2_summary.md").write_text(_build_summary(results), encoding="utf-8")

    logger.info(
        "Saved exp4v2_results.json + exp4v2_summary.md -> %s  (%d datasets)",
        out, len(results),
    )
    return {"status": "ok", "results": results}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="AROMA Exp 4 v2 — Supervised YOLOv8 Defect Detection 평가 (COCO pretrained start)"
    )
    p.add_argument(
        "--model",
        choices=["yolov8n", "yolov8s", "yolov8m", "all"],
        default="yolov8n",
        help="평가할 YOLOv8 모델 (default: yolov8n)",
    )
    p.add_argument(
        "--condition",
        choices=["baseline", "random", "aroma", "all"],
        default="all",
        help="평가 조건 (default: all)",
    )
    p.add_argument(
        "--dataset_keys",
        required=True,
        nargs="+",
        help="평가할 데이터셋 키 목록 (e.g. isp_LSM_1 mvtec_cable visa_cashew visa_pcb)",
    )
    p.add_argument(
        "--random_synthetic_dir",
        required=True,
        help="Random synthetic 출력 루트 ({dir}/{dataset_key}/annotations.json)",
    )
    p.add_argument(
        "--aroma_synthetic_dir",
        required=True,
        help="AROMA synthetic 출력 루트 ({dir}/{dataset_key}/annotations.json)",
    )
    p.add_argument(
        "--real_data_dir",
        required=True,
        help="실제 데이터셋 루트 (isp/, mvtec/, visa/ 포함)",
    )
    p.add_argument(
        "--output_dir",
        required=True,
        help="exp4v2_results.json + exp4v2_summary.md + baseline best.pt 저장 경로",
    )
    p.add_argument("--baseline_epochs", type=int, default=50,
                   help="baseline(real-only) 학습 epoch (default: 50)")
    p.add_argument("--val_frac", type=float, default=0.3,
                   help="real defect 중 val 비율 (나머지는 train, default: 0.3)")
    p.add_argument(
        "--max_synth_per_ds", type=int, default=None,
        help="데이터셋·조건당 synth annotation 최대 사용 수. None=제한 없음(기본). "
             "지정 시 random.sample로 subsampling (seed 적용).",
    )
    p.add_argument(
        "--synth_ratio", type=float, default=None,
        help=(
            "synth:real_train 비율 지정 (예: 0.3, 0.5, 0.7). "
            "n_synth_cap = max(1, int(n_real_train * ratio)). "
            "지정 시 --max_synth_per_ds 무시. None=비활성(기본)."
        ),
    )
    p.add_argument("--imgsz",  type=int, default=256,
                   help="YOLO image size (default: 256)")
    p.add_argument("--seed",   type=int, default=42)
    p.add_argument("--resume", action="store_true",
                   help="기존 exp4v2_results.json에서 완료된 run을 skip하고 재개")
    p.add_argument(
        "--yolo_cache_dir",
        default=None,
        help="real YOLO (image,label) 데이터셋을 build-once 후 영속 캐시할 경로 "
             "(예: Drive). 지정 시 동일 (min_area/val_frac/seed) 조합은 재빌드 없이 "
             "재사용. 미지정 시 매 데이터셋마다 새로 빌드 (캐시 없음).",
    )
    p.add_argument(
        "--no_local_cache",
        action="store_true",
        help="Drive→/tmp 로컬 캐시 staging 비활성화 (Linux/Colab 기본 활성). "
             "Windows 에서는 항상 비활성.",
    )
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = _parse_args(argv)

    model_keys = ALL_MODEL_KEYS if args.model == "all" else [args.model]
    condition_keys = ALL_CONDITION_KEYS if args.condition == "all" else [args.condition]

    result = run(
        model_keys=model_keys,
        condition_keys=condition_keys,
        dataset_keys=args.dataset_keys,
        random_synthetic_dir=args.random_synthetic_dir,
        aroma_synthetic_dir=args.aroma_synthetic_dir,
        real_data_dir=args.real_data_dir,
        output_dir=args.output_dir,
        baseline_epochs=args.baseline_epochs,
        val_frac=args.val_frac,
        max_synth_per_ds=args.max_synth_per_ds,
        synth_ratio=args.synth_ratio,
        imgsz=args.imgsz,
        seed=args.seed,
        resume=args.resume,
        yolo_cache_dir=args.yolo_cache_dir,
        local_cache=not args.no_local_cache,
    )
    status = result.get("status", "unknown")
    n_ds   = len(result.get("results", {}))
    print(f"[exp4v2] status={status}  n_datasets={n_ds}")
    if status != "ok":
        sys.exit(1)


if __name__ == "__main__":
    main()
