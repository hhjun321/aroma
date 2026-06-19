#!/usr/bin/env python3
"""
AROMA Exp 4 v2 — Supervised YOLOv8 Defect Detection 평가

Exp4(one-class AD)를 supervised detection으로 대체한 버전.
Synthetic defect 이미지를 LABELED TRAINING DATA로 사용하고,
real defect test 이미지(+GT mask)를 평가셋으로 사용한다.

핵심 아이디어:
- Synthetic 이미지는 copy-paste composite (normal 배경 위에 defect를 붙임)
- 각 synthetic dir의 annotations.json에 "image_path", "normal_image" 필드 존재
- defect bbox = composite 와 background 의 image difference → threshold → contour bbox
- synthesis 재실행 불필요 — label을 post-hoc 로 유도

조건:
- baseline: synthetic defect 학습 없음 (배경 이미지만, defect label 없음)
- random:   random synthetic 으로 학습
- aroma:    aroma synthetic 으로 학습

Datasets: isp_LSM_1, mvtec_cable, visa_cashew, visa_pcb (pcb4)

Usage (Colab):
    !python $AROMA_SCRIPTS/experiments/exp4_v2_supervised_detection.py \\
        --model yolov8n \\
        --condition all \\
        --dataset_keys isp_LSM_1 mvtec_cable visa_cashew visa_pcb \\
        --random_synthetic_dir $RANDOM_SYNTH_DIR \\
        --aroma_synthetic_dir  $AROMA_SYNTH_DIR \\
        --real_data_dir        $AROMA_DATA \\
        --output_dir           $EXP4_OUT \\
        --seed 42 \\
        --epochs 50 \\
        --resume
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import shutil
import sys
import tempfile
import threading
import time
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
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)

    kernel = np.ones((3, 3), np.uint8)
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
        logger.warning("annotations.json not found: %s", ann_path)
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
        valid.append({
            "image_path": img_p,
            "normal_image": norm_p,
            "source_roi": e.get("source_roi"),
        })

    logger.info(
        "Synth annotations %s: %d valid  (dry_run skipped=%d, missing skipped=%d) from %s",
        dataset_key, len(valid), skipped_dry, skipped_missing, ann_path,
    )
    return valid


# ---------------------------------------------------------------------------
# Image staging helper (symlink on posix, copy on windows)
# ---------------------------------------------------------------------------

def _link_or_copy(src: str, dst: str) -> None:
    use_symlink = os.name != "nt"
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
        if not synth_path:
            continue

        bboxes: List[Tuple[int, int, int, int]] = []
        img_w, img_h = _image_size(synth_path)
        if img_w <= 0 or img_h <= 0:
            continue

        if normal_path and Path(normal_path).exists():
            bboxes = _extract_defect_bboxes(synth_path, normal_path, min_area=min_area)

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


def _get_real_test_images_and_labels(
    test_defect_paths: List[str],
    mask_map: Dict[str, str],
    label_dir: str,
    image_dir_out: str,
    min_area: int = 50,
) -> int:
    """
    real defect test 이미지 + GT mask → YOLO val set 구성.

    각 defect 이미지(mask 보유) 마다:
      - binary mask PNG → bounding boxes
      - YOLO label 작성
      - 이미지 stage
    mask 가 없는 defect 이미지는 평가에서 제외 (label 없으면 mAP 계산 불가).

    Returns count of val images with labels.
    """
    Path(label_dir).mkdir(parents=True, exist_ok=True)
    Path(image_dir_out).mkdir(parents=True, exist_ok=True)

    n = 0
    for i, img_p in enumerate(test_defect_paths):
        mask_p = mask_map.get(img_p)
        if not mask_p:
            continue

        img_w, img_h = _image_size(img_p)
        if img_w <= 0 or img_h <= 0:
            continue

        m_bboxes, mw, mh = _mask_to_bboxes(mask_p, min_area=min_area)
        if not m_bboxes:
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
            continue

        stem = f"real_{i:06d}"
        img_dst = str(Path(image_dir_out) / f"{stem}{Path(img_p).suffix}")
        _link_or_copy(img_p, img_dst)
        (Path(label_dir) / f"{stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        n += 1

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
# Per-condition YOLO runner
# ---------------------------------------------------------------------------

def _run_yolo_condition(
    model_name: str,
    condition: str,
    synth_annotations: List[Dict[str, Any]],
    train_normal_paths: List[str],
    test_defect_paths: List[str],
    mask_map: Dict[str, str],
    checkpoint_dir: str,
    epochs: int = 50,
    imgsz: int = 256,
    seed: int = 42,
    device: int = 0,
) -> Dict[str, Any]:
    """
    하나의 (model, condition) 조합을 학습/평가.

    baseline:        synth_annotations=[] → defect label 없음 (배경만 fine-tune)
    random / aroma:  synth annotation 으로 label 작성

    Returns {map50, map50_95, precision, recall, n_train}.
    """
    try:
        from ultralytics import YOLO  # type: ignore[import]
    except ImportError:
        logger.error("ultralytics not installed. Run: pip install ultralytics")
        return {"error": "ultralytics_missing"}

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            train_img = str(Path(tmpdir) / "train" / "images")
            train_lbl = str(Path(tmpdir) / "train" / "labels")
            val_img   = str(Path(tmpdir) / "val" / "images")
            val_lbl   = str(Path(tmpdir) / "val" / "labels")
            for d in (train_img, train_lbl, val_img, val_lbl):
                Path(d).mkdir(parents=True, exist_ok=True)

            # --- Train set ---
            n_labeled = 0
            if condition == "baseline":
                # background-only fine-tune: stage normal images, no defect labels
                n_bg = _stage_background_images(train_normal_paths, train_img, prefix="bg")
                logger.info("    baseline train: %d background imgs (no defect labels)", n_bg)
                n_train = n_bg
            else:
                n_labeled = _write_yolo_labels(
                    synth_annotations=synth_annotations,
                    label_dir=train_lbl,
                    image_dir_out=train_img,
                    min_area=200,
                )
                n_total_synth = len(synth_annotations)
                logger.info(
                    "    %s train: %d/%d synth imgs labeled",
                    condition, n_labeled, n_total_synth,
                )
                n_train = n_labeled
                if n_labeled == 0:
                    logger.warning(
                        "    %s/%s: no labeled synth images — training will have no positives",
                        model_name, condition,
                    )

            # --- Val set (real defect + GT mask) ---
            n_val = _get_real_test_images_and_labels(
                test_defect_paths=test_defect_paths,
                mask_map=mask_map,
                label_dir=val_lbl,
                image_dir_out=val_img,
                min_area=50,
            )
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

            # --- Train ---
            Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
            try:
                import torch
                torch.set_float32_matmul_precision("medium")
            except Exception:
                pass

            model = YOLO(f"{model_name}.pt")  # downloads pretrained weights

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
                    save=False,   # no checkpoint to save disk
                    device=device,
                    exist_ok=True,
                )

            logger.info(
                "    fit start: %s / %s  train_imgs=%d  epochs=%d imgsz=%d",
                model_name, condition, n_train, epochs, imgsz,
            )
            _, fit_elapsed = _train_with_heartbeat(
                _do_train, label=f"{model_name}/{condition}", interval_s=60,
            )
            logger.info("    fit done: %.1fs", fit_elapsed)

            # --- Validate ---
            logger.info("    val start: %s / %s", model_name, condition)
            t_val = time.time()
            val_results = model.val(data=yaml_path, verbose=False, plots=False)
            logger.info("    val done: %.1fs", time.time() - t_val)

            box = val_results.box
            out = {
                "map50":     round(float(box.map50), 4),
                "map50_95":  round(float(box.map), 4),
                "precision": round(float(box.mp), 4),
                "recall":    round(float(box.mr), 4),
                "n_train":   int(n_train),
            }

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
    epochs: int = 50,
    imgsz: int = 256,
    seed: int = 42,
    existing_results: Optional[Dict[str, Any]] = None,
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Iterate datasets x models x conditions.
    Output structure: {dataset: {model: {condition: {map50, map50_95, precision, recall, n_train}}}}

    CRITICAL: Val set = real defect images only. Synthetic NEVER in val.
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

        train_normal = lists["train_normal"]
        test_defect  = lists["test_defect"]
        mask_map     = lists["mask_map"]
        ckpt_base    = str(Path("/content/tmp/exp4v2_checkpoints") / ds)

        # synth annotations per condition (loaded once per dataset)
        synth_by_cond: Dict[str, List[Dict[str, Any]]] = {
            "baseline": [],
            "random":   _load_synth_annotations(random_synthetic_dir, ds),
            "aroma":    _load_synth_annotations(aroma_synthetic_dir, ds),
        }

        ds_results: Dict[str, Any] = dict(existing.get(ds, {}))

        for model_name in model_keys:
            model_results: Dict[str, Any] = dict(ds_results.get(model_name, {}))

            for cond in condition_keys:
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
                    train_normal_paths=train_normal,
                    test_defect_paths=test_defect,
                    mask_map=mask_map,
                    checkpoint_dir=str(Path(ckpt_base) / model_name / cond),
                    epochs=epochs,
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

                # Incremental save after each condition
                if output_path:
                    ds_results[model_name] = dict(model_results)
                    results[ds] = dict(ds_results)
                    try:
                        save_json(results, output_path)
                    except Exception as _e:
                        logger.warning("Incremental save failed: %s", _e)

            ds_results[model_name] = model_results

        results[ds] = ds_results

    return results


# ---------------------------------------------------------------------------
# Summary builder
# ---------------------------------------------------------------------------

def _build_summary(results: Dict[str, Any]) -> str:
    """
    Build Markdown summary.
    Per dataset/model: rows=conditions, cols=metrics.
    Delta section: AROMA minus Random for map50.
    """
    lines = [
        "# AROMA Exp 4 v2 — Supervised YOLOv8 Defect Detection 평가",
        "",
        "비교: Baseline (background-only) vs Random ROI vs AROMA ROI",
        "Train = synthetic defect (labeled), Val = real defect (GT mask → bbox)",
        "",
    ]

    all_conds = ALL_CONDITION_KEYS
    metric_cols = ["map50", "map50_95", "precision", "recall", "n_train"]

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

            header = " | ".join(f"{c:>10}" for c in metric_cols)
            lines.append(f"| {'조건':<10} | {header} |")
            sep = " | ".join("-" * 10 for _ in metric_cols)
            lines.append(f"|{'-' * 12}|{sep}|")

            for cond in all_conds:
                cm = m_data.get(cond, {})
                cells = []
                for col in metric_cols:
                    val = cm.get(col)
                    if isinstance(val, float):
                        cells.append(f"{val:.4f}")
                    elif isinstance(val, int):
                        cells.append(str(val))
                    else:
                        cells.append("N/A")
                row = " | ".join(f"{c:>10}" for c in cells)
                lines.append(f"| {cond:<10} | {row} |")
            lines.append("")

            # Delta: AROMA - Random
            r = m_data.get("random", {})
            a = m_data.get("aroma", {})
            parts = []
            for col in ("map50", "map50_95", "precision", "recall"):
                rv, av = r.get(col), a.get(col)
                if isinstance(rv, float) and isinstance(av, float):
                    parts.append(f"{col} {av - rv:+.4f}")
                else:
                    parts.append(f"{col} N/A")
            lines.append(f"**Delta (AROMA − Random)**: " + ", ".join(parts))
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
    epochs: int = 50,
    imgsz: int = 256,
    seed: int = 42,
    resume: bool = False,
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
        epochs=epochs,
        imgsz=imgsz,
        seed=seed,
        existing_results=existing_results,
        output_path=output_path,
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
        description="AROMA Exp 4 v2 — Supervised YOLOv8 Defect Detection 평가"
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
        help="exp4v2_results.json + exp4v2_summary.md 저장 경로",
    )
    p.add_argument("--epochs", type=int, default=50,
                   help="YOLO 학습 epoch 수 (default: 50)")
    p.add_argument("--imgsz",  type=int, default=256,
                   help="YOLO image size (default: 256)")
    p.add_argument("--seed",   type=int, default=42)
    p.add_argument("--resume", action="store_true",
                   help="기존 exp4v2_results.json에서 완료된 run을 skip하고 재개")
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
        epochs=args.epochs,
        imgsz=args.imgsz,
        seed=args.seed,
        resume=args.resume,
    )
    status = result.get("status", "unknown")
    n_ds   = len(result.get("results", {}))
    print(f"[exp4v2] status={status}  n_datasets={n_ds}")
    if status != "ok":
        sys.exit(1)


if __name__ == "__main__":
    main()
