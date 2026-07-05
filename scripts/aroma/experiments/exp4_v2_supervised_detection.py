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

Datasets: isp_LSM_1, mvtec_*, visa_* (cashew/pcb4/config-driven), severstal,
aitex, mtd (v2-1 primary set: severstal mvtec_leather aitex mtd)

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
        --dataset_keys severstal mvtec_leather aitex mtd \\
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
import re
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
ALL_CONDITION_KEYS = ["baseline", "random", "casda", "aroma"]

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


def _resolve_severstal_masks(
    defect_paths: List[str], masks_dir: str, class_mode: str = "single",
) -> Tuple[Dict[str, str], Dict[str, Dict[int, str]]]:
    """Severstal: masks/{stem}.png (merged) + masks/class{c}/{stem}.png (per-class).

    Returns (mask_map, class_mask_map):
      mask_map        — {defect_img: merged_mask_png}  (always; single-mode + val GT)
      class_mask_map  — {defect_img: {class_id: per_class_png}}  (multi mode only;
                        empty dict when class_mode='single').
    """
    m_dir = Path(masks_dir)
    mask_map: Dict[str, str] = {}
    class_mask_map: Dict[str, Dict[int, str]] = {}
    for img_p in defect_paths:
        stem = Path(img_p).stem
        merged = m_dir / f"{stem}.png"
        if merged.exists():
            mask_map[img_p] = str(merged)
        if class_mode == "multi":
            per_class: Dict[int, str] = {}
            for c in range(1, 5):
                pc = m_dir / f"class{c}" / f"{stem}.png"
                if pc.exists():
                    per_class[c] = str(pc)
            if per_class:
                class_mask_map[img_p] = per_class
    return mask_map, class_mask_map


# ---------------------------------------------------------------------------
# dataset_config.json — config-driven dataset resolution (ported from exp3).
# Lets generic handlers (e.g. visa_*) resolve non-trivial key→folder mappings
# (visa_macaroni → macaroni1, visa_pcb → pcb4) from a single source of truth.
# ---------------------------------------------------------------------------

_DS_CFG_CACHE: Optional[Dict[str, Any]] = None


def _load_dataset_config() -> Dict[str, Any]:
    """Load dataset_config.json once (env DATASET_CONFIG / AROMA_REF / clone default)."""
    global _DS_CFG_CACHE
    if _DS_CFG_CACHE is not None:
        return _DS_CFG_CACHE
    cands = [
        os.environ.get("DATASET_CONFIG"),
        os.path.join(os.environ.get("AROMA_REF", ""), "dataset_config.json"),
        "/content/AROMA/dataset_config.json",
    ]
    for c in cands:
        if c and Path(c).exists():
            try:
                with open(c, encoding="utf-8") as f:
                    _DS_CFG_CACHE = json.load(f)
                logger.info("dataset_config loaded: %s", c)
                return _DS_CFG_CACHE
            except Exception as e:
                logger.warning("dataset_config load failed %s: %s", c, e)
    logger.warning("dataset_config.json not found (set DATASET_CONFIG or AROMA_REF)")
    _DS_CFG_CACHE = {}
    return _DS_CFG_CACHE


def _resolve_masks_generic(defect_paths: List[str], root: Path) -> Dict[str, str]:
    """Resolve GT masks across mvtec/severstal/visa conventions. First hit wins.

    Per defect image at <root>/test/<type>/<stem>.<ext>, tries:
      ground_truth/<type>/<stem>_mask.png  (MVTec)
      ground_truth/<type>/<stem>.png        (VisA prepared)
      masks/<type>/<stem>.png
      masks/<type>/<stem>_mask.png
      ground_truth/<type>/<filename>
    Missing masks are skipped (that defect carries no bbox label).
    """
    root = Path(root)
    mapping: Dict[str, str] = {}
    for img_p in defect_paths:
        p = Path(img_p)
        typ, stem = p.parent.name, p.stem
        for m in (
            root / "ground_truth" / typ / f"{stem}_mask.png",
            root / "ground_truth" / typ / f"{stem}.png",
            root / "masks" / typ / f"{stem}.png",
            root / "masks" / typ / f"{stem}_mask.png",
            root / "ground_truth" / typ / p.name,
        ):
            if m.exists():
                mapping[img_p] = str(m)
                break
    return mapping


def _get_image_lists(
    dataset_key: str,
    real_data_dir: str,
    seed: int = 42,
    test_split: float = 0.2,
    class_mode: str = "single",
) -> Optional[Dict[str, Any]]:
    """
    Returns {train_normal, test_good, test_defect, mask_map} or None if missing.
    mask_map: {defect_img_path: mask_path}

    For dataset_key == 'severstal' with class_mode == 'multi', an extra
    'class_mask_map' key is included ({defect_img: {class_id: per_class_png}}).
    All other datasets / single mode are unaffected (no extra key).
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

    elif dataset_key.startswith("mvtec_"):
        # Generic MVTec handler — all categories share the same layout:
        #   mvtec/{category}/{train/good, test/{defect_types}, ground_truth}
        # category may contain underscores (e.g. mvtec_metal_nut → metal_nut).
        category = dataset_key.split("_", 1)[1]
        ds = base / "mvtec" / category
        if not ds.exists():
            logger.warning("%s not found: %s", dataset_key, ds)
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

    elif dataset_key.startswith("visa_"):
        # Generic config-driven VisA handler. dataset_config.json maps the key
        # to the PREPARED layout (<cat>/train/good + test/{good,anomaly} +
        # ground_truth), including non-trivial folders (visa_macaroni → macaroni1).
        # visa_cashew / visa_pcb keep their explicit Data/-layout branches ABOVE
        # (elif order: specific keys win before this generic fallback), so their
        # behavior is unchanged.
        entry = _load_dataset_config().get(dataset_key)
        if not entry or not entry.get("image_dir"):
            logger.warning("visa dataset_key not in dataset_config: %s", dataset_key)
            return None
        image_dir = Path(entry["image_dir"])           # .../<cat>/train/good
        if not image_dir.exists():
            logger.warning("%s image_dir missing: %s", dataset_key, image_dir)
            return None
        root = image_dir.parent.parent                 # .../<cat>
        train_normal = _glob_images(str(image_dir))
        test_good    = _glob_images(str(root / "test" / "good"))
        # dataset_config visa entries use singular "seed_dir"; mvtec/isp use
        # plural "seed_dirs". Honor either (config = source of truth); else scan
        # test/<non-good> (robust to multi-defect-dir layouts).
        seed_dirs = entry.get("seed_dirs") or (
            [entry["seed_dir"]] if entry.get("seed_dir") else None
        )
        if seed_dirs:
            defect_dirs = [Path(d) for d in seed_dirs]
        else:
            troot = root / "test"
            defect_dirs = (
                [p for p in sorted(troot.iterdir()) if p.is_dir() and p.name != "good"]
                if troot.exists() else []
            )
        test_defect = []
        for d in defect_dirs:
            test_defect.extend(_glob_images(str(d)))
        mask_map = _resolve_masks_generic(test_defect, root)

    elif dataset_key in ("aitex", "mtd"):
        # Config-driven MVTec-style prepared layouts (prepare_aitex.py /
        # prepare_mtd.py): <root>/train/good + test/{defect_type}/ +
        # ground_truth/{defect_type}/{stem}_mask.png. dataset_config.json's
        # image_dir points at .../<root>/train/good (source of truth for root).
        entry = _load_dataset_config().get(dataset_key)
        if not entry or not entry.get("image_dir"):
            logger.warning("%s not in dataset_config: run prepare first", dataset_key)
            return None
        image_dir = Path(entry["image_dir"])           # .../<root>/train/good
        if not image_dir.exists():
            logger.warning("%s image_dir missing: %s", dataset_key, image_dir)
            return None
        root = image_dir.parent.parent                 # .../<root>
        train_normal = _glob_images(str(image_dir))
        test_good    = _glob_images(str(root / "test" / "good"))
        # Scan test/<non-good> on disk rather than trusting seed_dirs — the
        # aitex config's seed_dirs is a placeholder superset of defect codes;
        # only folders prepare actually produced should be used.
        troot = root / "test"
        defect_dirs = (
            [p for p in sorted(troot.iterdir()) if p.is_dir() and p.name != "good"]
            if troot.exists() else []
        )
        test_defect = []
        for d in defect_dirs:
            test_defect.extend(_glob_images(str(d)))
        mask_map = _resolve_masks_generic(test_defect, root)

    elif dataset_key == "severstal":
        # prepare_severstal.py layout:
        #   train/good/  test/class{1..4}/  masks/{stem}.png + masks/class{c}/{stem}.png
        ds = base / "severstal"
        if not ds.exists():
            logger.warning("severstal not found: %s", ds)
            return None
        all_normal = _glob_images(str(ds / "train" / "good"))
        # Context prototypes (if selected) restrict the normal background pool to
        # the K distribution-representative images.
        proto_json = ds / "context_select" / "context_prototypes.json"
        if proto_json.exists():
            try:
                payload = load_json(str(proto_json))
                names = set(payload.get("prototypes") or [])
                if names:
                    filtered = [p for p in all_normal if Path(p).name in names]
                    if filtered:
                        logger.info(
                            "severstal: context prototypes applied (%d/%d normals)",
                            len(filtered), len(all_normal),
                        )
                        all_normal = filtered
                    else:
                        logger.warning(
                            "severstal: context_prototypes.json had %d names but 0 "
                            "matched train/good — using ALL %d normals (check name basis)",
                            len(names), len(all_normal),
                        )
            except Exception as exc:
                logger.warning("severstal: failed to read context prototypes: %s", exc)
        else:
            # Silent fallback to all normals would change the experiment (no
            # 1000-prototype pool). Surface it explicitly so it isn't mistaken
            # for the intended config.
            logger.warning(
                "severstal: context_prototypes.json NOT found at %s — using ALL %d "
                "normals (run select_context_prototypes.py with --output %s)",
                proto_json, len(all_normal), str(ds / "context_select"),
            )
        train_normal, test_good = _split_normal(all_normal, test_split=test_split, seed=seed)
        test_defect = []
        for c in range(1, 5):
            test_defect.extend(_glob_images(str(ds / "test" / f"class{c}")))
        mask_map, class_mask_map = _resolve_severstal_masks(
            test_defect, str(ds / "masks"), class_mode=class_mode,
        )
        logger.info(
            "severstal: train_normal=%d  test_good=%d  test_defect=%d  masks_matched=%d "
            "(class_mode=%s)",
            len(train_normal), len(test_good), len(test_defect), len(mask_map), class_mode,
        )
        result = dict(
            train_normal=train_normal,
            test_good=test_good,
            test_defect=test_defect,
            mask_map=mask_map,
        )
        if class_mode == "multi":
            result["class_mask_map"] = class_mask_map
            _cn, _n2i = _enumerate_defect_classes(test_defect)
            if _cn:
                result["class_names"] = _cn
                result["name_to_id"] = _n2i
        return result

    else:
        logger.warning("Unknown dataset_key: %s", dataset_key)
        return None

    logger.info(
        "%s: train_normal=%d  test_good=%d  test_defect=%d  masks_matched=%d",
        dataset_key, len(train_normal), len(test_good), len(test_defect), len(mask_map),
    )
    result = dict(
        train_normal=train_normal,
        test_good=test_good,
        test_defect=test_defect,
        mask_map=mask_map,
    )
    if class_mode == "multi":
        # Generic per-class enumeration (mvtec/visa/isp/aitex/mtd): class = the
        # test/{type} parent folder of each resolved defect path. nc<=1 → single.
        _cn, _n2i = _enumerate_defect_classes(test_defect)
        if _cn:
            result["class_names"] = _cn
            result["name_to_id"] = _n2i
            # Route generic multi real labels through the SAME per-image
            # class_mask_map branch severstal uses. That path is staging-safe:
            # _local_cache_for_yolo rewrites class_mask_map KEYS lockstep with
            # test_defect, so a /tmp-staged image still resolves its class —
            # unlike a Path(img).parent.name lookup, which would read the staged
            # ".../images/" folder and collapse every real label to class 0.
            # Keys are 1-based ordinals (YOLO id + 1) so the branch's `cls-1`
            # yields the 0-based id; each generic image has exactly one class
            # (its test/{type} folder), so the merged GT mask is that class's mask.
            result["class_mask_map"] = {
                p: {_n2i[Path(p).parent.name] + 1: mask_map[p]}
                for p in test_defect if p in mask_map
            }
    return result


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

    # Union-bbox fallback: 점상/미세 mask(예: aitex thread 결함 — 전 성분이
    # min_area 미달, 총 유효픽셀 수십 px)는 성분별 필터에서 전멸한다. 그대로
    # 0-box를 반환하면 (1) real 라벨이 통째로 탈락하고 (2) synth 라벨이 diff
    # 폴백으로 빠져 텍스처 노이즈 파편 bbox(이미지당 수십 개)를 양산한다.
    # → 유효픽셀이 존재하면 그 합집합 1-bbox를 반환해 GT를 보존한다.
    # blob mask(severstal 등)는 성분 bbox가 항상 나와 이 분기에 도달하지 않음.
    if not bboxes and contours:
        nz = cv2.findNonZero(binm)
        if nz is not None:
            x, y, bw, bh = cv2.boundingRect(nz)
            bboxes = [(int(x), int(y), int(bw), int(bh))]
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
        # generate_defects.py now writes a full-frame mask PNG + 'mask_path'
        # (and 'bbox') per synthesized image, so the exact-bbox path below is
        # the primary route. Other pipelines (e.g. stage4 fmt='cls') write
        # '{stem}_mask.png'. Accept any of these keys, plus a sibling masks/
        # file as a last resort. Image-diff fallback only when no mask resolves.
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
            # Robust class signal for severstal multi mode. CASDA's source_roi is
            # an ROI-crop path with no .../class{N}/ segment, so it can't be
            # parsed; cluster_id (Severstal class 1-4, set by casda_roi_adapter)
            # is the fallback. For random/aroma source_roi parses fine so this is
            # only consulted when source_roi is unparseable.
            "cluster_id": e.get("cluster_id"),
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
# Image staging helper (hardlink first, copy fallback)
# ---------------------------------------------------------------------------

def _link_or_copy(src: str, dst: str, force_copy: bool = False) -> str:
    """Stage one file, hardlink-first. Returns "link" or "copy" (which path ran).

    Hardlink shares the same inode (zero extra bytes) and, unlike a symlink,
    keeps dst valid even after the original tmpdir is cleaned up. YOLO reads
    staged images read-only, so a shared inode is safe.

    Callers must ensure dst is unique within a staging batch (real images are
    named ``real_*``; background ``bg_*``; synth ``syn_*`` — disjoint), so two
    threads never target the same dst.

    force_copy=True: always copy bytes (e.g. building a persistent YOLO cache
    that must survive on a different filesystem than the source). Hardlinks
    cannot span filesystems (EXDEV) anyway, so the copy fallback below also
    covers the cross-device case.
    """
    if not force_copy:
        # Stale dst (e.g. a prior partial run) would make os.link raise EEXIST.
        # Remove it first so re-staging is deterministic; identical content makes
        # this harmless. A genuinely undeletable dst is surfaced at debug so the
        # confusing downstream shutil error has context.
        try:
            if os.path.lexists(dst):
                os.remove(dst)
        except OSError as _rm_exc:
            logger.debug("stage: could not clear dst %s: %s", dst, _rm_exc)
        try:
            os.link(src, dst)
            return "link"
        except OSError:
            # cross-device (EXDEV), unsupported FS, or a race on dst — fall back.
            pass
    shutil.copy2(src, dst)
    return "copy"


def _bulk_link_or_copy(
    pairs: List[Tuple[str, str]],
    label: str,
    max_workers: Optional[int] = None,
    force_copy: bool = False,
) -> int:
    """Stage (src, dst) pairs in parallel via _link_or_copy.

    Drive I/O is latency-bound, so a thread pool overlaps the per-file syscalls.
    Emits exactly two log lines (start / done) — no per-file logging. The done
    line reports how many were hardlinked vs byte-copied, so a silent fallback
    to copy (e.g. ds_stage on a different mount than the tmpdir) is visible
    instead of just looking slow. The first failing file re-raises (fail-fast);
    other in-flight ops finish as the pool shuts down.
    Returns the number of pairs processed (missing sources are skipped).
    """
    todo = [(s, d) for (s, d) in pairs if Path(s).exists()]
    n = len(todo)
    if n == 0:
        logger.info("[Stage] %s: 0 files", label)
        return 0
    if max_workers is None:
        max_workers = min(16, (os.cpu_count() or 4) * 4)
    logger.info("[Stage] %s: %d files", label, n)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(_link_or_copy, s, d, force_copy) for (s, d) in todo]
        modes = [f.result() for f in futs]
    n_link = modes.count("link")
    n_copy = modes.count("copy")
    logger.info(
        "[Stage] %s: done %d in %.1fs (hardlink=%d copy=%d)",
        label, n, time.time() - t0, n_link, n_copy,
    )
    return n


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


def _enumerate_defect_classes(
    test_defect_paths: List[str],
) -> Tuple[List[str], Dict[str, int]]:
    """Enumerate per-dataset defect classes from resolved real defect paths.

    Class = parent folder name of each defect image (``test/{type}/{stem}``).
    Dataset-agnostic: works for any dataset whose ``_get_image_lists`` resolves
    ``test_defect`` under ``test/{type}/`` (severstal ``class1..class4``,
    mvtec/aitex/mtd defect-type folders). Uses the already-resolved ABSOLUTE
    paths (not a guessed ``{ds}/test`` scan), so nested layouts like
    ``mvtec/leather`` are handled correctly.

    Returns ``(class_names, name_to_id)`` with 0-based ids = sorted index.
    ``nc <= 1`` degenerates to single: returns ``([], {})`` so callers fall back
    to the historical single-class ('defect') path (byte-identical).
    """
    names = sorted({Path(p).parent.name for p in test_defect_paths})
    if len(names) <= 1:
        return [], {}
    return names, {n: i for i, n in enumerate(names)}


def _display_class_names(dataset_key: str, class_names: List[str]) -> List[str]:
    """Display names for the YAML ``names:`` line (→ val_results.names → per_class keys).

    severstal keeps the historical ``c1..c4`` labels (backward-compat with
    existing results + aroma_exp4v2_per-class-metrics), in the SAME id order as
    the raw ``class1..class4`` folders. All other datasets use raw folder names.
    """
    if dataset_key == "severstal":
        return [f"c{i + 1}" for i in range(len(class_names))]
    return list(class_names)


def _resolve_synth_class(
    ann: Dict[str, Any],
    name_to_id: Optional[Dict[str, int]] = None,
    dataset_key: Optional[str] = None,
) -> Optional[int]:
    """0-indexed defect class for a synth annotation (multi mode) — dataset-generic.

    Priority:
      1. source_roi (fallback normal_image) ``.../test/{type}/`` segment →
         name_to_id. Works for every dataset whose synth source_roi points at a
         real defect under ``test/{type}/`` (severstal ``test/class{N}`` →
         name_to_id['classN']=N-1; mvtec/aitex/mtd ``test/{type}``). Subsumes the
         old ``class{N}`` parse.
      2. severstal-only cluster_id fallback — CASDA's source_roi is an ROI-crop
         path with no ``test/{type}`` segment; casda_roi_adapter writes
         cluster_id = Severstal class 1-4, so cluster_id-1 gives the 0-indexed
         class. Guarded by ``dataset_key == 'severstal'`` (aroma's cluster_id is
         a morphology cluster id, NOT a class, so must not be used elsewhere).

    Returns None when neither signal yields a valid class (caller defaults to 0).
    """
    src = str(ann.get("source_roi") or ann.get("normal_image") or "")
    if name_to_id:
        m = re.search(r"[\\/]test[\\/]([^\\/]+)[\\/]", src)
        if m and m.group(1) in name_to_id:
            return name_to_id[m.group(1)]
    if dataset_key == "severstal":
        try:
            cid = int(ann.get("cluster_id")) - 1
        except (TypeError, ValueError):
            return None
        if 0 <= cid <= 3:
            return cid
    return None


def _write_yolo_labels(
    synth_annotations: List[Dict[str, Any]],
    label_dir: str,
    image_dir_out: str,
    min_area: int = 200,
    class_mode: str = "single",
    name_to_id: Optional[Dict[str, int]] = None,
    dataset_key: Optional[str] = None,
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

        # Prefer an explicit GT mask when available (exact bbox). A readable
        # mask is AUTHORITATIVE: diff/template fallbacks are reserved for
        # mask-less annotations only. (aitex 실측: 미세 mask가 0-box일 때 diff로
        # 빠지면 직물 텍스처 노이즈가 이미지당 수십~수백 개 파편 bbox를 양산 —
        # mean 40.4, max 394. _mask_to_bboxes의 union 폴백이 유효픽셀 mask를
        # 1-bbox로 회수하므로, 여기 도달해 0-box면 mask가 진짜 빈 것 → negative.)
        mask_present = bool(mask_path and Path(mask_path).exists())
        if mask_present:
            m_bboxes, mw, mh = _mask_to_bboxes(mask_path, min_area=min_area)
            if m_bboxes and (mw, mh) != (img_w, img_h) and mw > 0 and mh > 0:
                sx = img_w / mw
                sy = img_h / mh
                m_bboxes = [
                    (int(x * sx), int(y * sy), int(bw * sx), int(bh * sy))
                    for (x, y, bw, bh) in m_bboxes
                ]
            bboxes = m_bboxes
        if not bboxes and not mask_present and normal_path and Path(normal_path).exists():
            bboxes = _extract_defect_bboxes(synth_path, normal_path, min_area=min_area)

        # Final fallback: source_roi template matching (handles stale
        # normal_image paths left over from local_staging runs). Mask-less only.
        source_roi_path = ann.get("source_roi")
        if (not bboxes and not mask_present and source_roi_path
                and Path(source_roi_path).exists() and _CV2_AVAILABLE):
            bboxes = _template_match_bbox(synth_path, source_roi_path, min_area=min_area)

        stem = f"syn_{i:06d}"
        img_dst = str(Path(image_dir_out) / f"{stem}{Path(synth_path).suffix}")
        _link_or_copy(synth_path, img_dst)

        if not bboxes:
            # no label file → treated as background negative
            continue

        if class_mode == "multi":
            cls_id = _resolve_synth_class(ann, name_to_id, dataset_key)
            if cls_id is None:
                logger.warning(
                    "multi mode: could not resolve defect class "
                    "(source_roi=%s cluster_id=%s, synth idx %d) -> defaulting to "
                    "class 0",
                    ann.get("source_roi"), ann.get("cluster_id"), i,
                )
                cls_id = 0
        else:
            cls_id = 0

        lines = _bboxes_to_yolo_lines(bboxes, img_w, img_h, class_id=cls_id)
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
    """Stage normal/background images into train images (no label = negative).

    Preserves the prefixed rename (``{prefix}_{i:06d}``) so background filenames
    never collide with defect/synth images; staging itself is parallel +
    hardlink-first via _bulk_link_or_copy.
    """
    Path(image_dir_out).mkdir(parents=True, exist_ok=True)
    pairs: List[Tuple[str, str]] = []
    for i, p in enumerate(normal_paths):
        if not Path(p).exists():
            continue
        dst = str(Path(image_dir_out) / f"{prefix}_{i:06d}{Path(p).suffix}")
        pairs.append((p, dst))
    return _bulk_link_or_copy(pairs, label=f"bg:{prefix}")


# ---------------------------------------------------------------------------
# Local cache (Drive -> /tmp staging for I/O acceleration; Linux/Colab only)
# ---------------------------------------------------------------------------

# Stage-IN copy concurrency. Same workload shape as generate_defects' push-out
# (latency-bound FUSE round-trips, NOT CPU-bound), so we reuse the SAME
# AROMA_STAGE_WORKERS knob users already set for staging/push, and deliberately
# do NOT key off os.cpu_count() (Colab = 2 vCPU would cap concurrency at ~4 and
# leave 19k+ FUSE reads serialized 4-wide ≈ 22 min). These constants/resolver
# are duplicated (not imported) from generate_defects._push_workers ON PURPOSE:
# generate_defects is imported by casda_roi_adapter, and importing it here would
# risk a circular import. The ~15 duplicated lines are preferred over a new
# shared module (same rationale as generate_defects' own duplication).
_DEFAULT_STAGE_WORKERS = 16
_STAGE_WORKERS_MIN = 1
_STAGE_WORKERS_MAX = 64
_COPY_RETRIES = 3                 # total attempts per file
_COPY_BACKOFF = (0.5, 1.0, 2.0)   # sleep before retry attempts 2,3,...


def _stage_workers() -> int:
    """Resolve stage-in concurrency from AROMA_STAGE_WORKERS, clamped to a sane range."""
    raw = os.environ.get("AROMA_STAGE_WORKERS")
    if not raw:
        return _DEFAULT_STAGE_WORKERS
    try:
        n = int(raw)
    except (TypeError, ValueError):
        logger.warning("AROMA_STAGE_WORKERS=%r not an int — using default %d",
                       raw, _DEFAULT_STAGE_WORKERS)
        return _DEFAULT_STAGE_WORKERS
    clamped = max(_STAGE_WORKERS_MIN, min(_STAGE_WORKERS_MAX, n))
    if clamped != n:
        logger.warning("AROMA_STAGE_WORKERS=%d clamped to %d (range %d..%d)",
                       n, clamped, _STAGE_WORKERS_MIN, _STAGE_WORKERS_MAX)
    return clamped


def _stage_one(src: str, dst_str: str) -> str:
    """Copy one file src → dst with retry/backoff. Returns 'copied'|'skipped'.

    - The dst parent dir is pre-created in the main thread before any worker
      runs, so this performs NO mkdir (no shared-state race on dir creation).
    - Same-size skip (cheap stat): if dst already exists with the same byte size
      as src, treat as cached and skip. This makes multi-seed re-runs and
      --resume idempotent AND heals a truncated/partial file from an interrupted
      seed (the old existence-only check could never re-copy a partial file).
    - On transient FUSE errors, retry up to _COPY_RETRIES with exponential backoff.
    - TERMINAL FAILURE = RAISE (diverges from generate_defects._push_one, which
      WARNs+continues). Here a dropped cache file is SILENTLY excluded from the
      trained/eval dataset downstream (_image_size -> (0,0) -> n_bad_size; missing
      mask -> _mask_to_bboxes -> ([],0,0) -> n_no_bbox; a rewritten synth
      image_path points at a nonexistent /tmp file). That would change WHICH
      files are trained on without any error. So after exhausting retries we
      raise so the seed aborts loudly rather than silently shrink the dataset.
    """
    dst = Path(dst_str)

    # Same-size skip: identical file already staged locally.
    try:
        if dst.exists() and dst.stat().st_size == os.path.getsize(src):
            return "skipped"
    except OSError:
        pass  # fall through to (re)copy

    last_err: Optional[Exception] = None
    for attempt in range(_COPY_RETRIES):
        try:
            shutil.copy2(src, dst_str)
            return "copied"
        except Exception as e:  # noqa: BLE001 — transient FUSE EIO/5xx
            last_err = e
            if attempt < _COPY_RETRIES - 1:
                sleep_s = _COPY_BACKOFF[min(attempt, len(_COPY_BACKOFF) - 1)]
                time.sleep(sleep_s)
    logger.error("[LocalCache] failed to stage %s → %s after %d attempts: %s",
                 src, dst_str, _COPY_RETRIES, last_err)
    raise last_err  # type: ignore[misc]


def _local_cache_for_yolo(
    ds: str,
    lists: Dict[str, Any],
    synth_by_cond: Dict[str, List[Dict[str, Any]]],
    random_synth_dir: str,
    aroma_synth_dir: str,
    casda_synth_dir: Optional[str] = None,
    cache_base: str = "/tmp/aroma_exp4v2_cache",
    num_workers: Optional[int] = None,
) -> Tuple[Dict[str, Any], Dict[str, List[Dict[str, Any]]]]:
    """Stage Drive images to local /tmp for I/O acceleration. Linux/Colab only.

    Returns (new_lists, new_synth_by_cond) with paths rewritten to /tmp.
    Idempotent per-file. mask_map keys rewritten in lockstep with test_defect.
    class_mask_map (severstal multi mode), when present in `lists`, has its keys
    rewritten in lockstep too (values/per-class PNGs stay on Drive).
    """
    if num_workers is None:
        num_workers = _stage_workers()

    cache_ds = Path(cache_base) / ds
    real_dir = cache_ds / "real"
    real_imgs = real_dir / "images"
    real_masks = real_dir / "masks"
    for d in (real_imgs, real_masks):
        d.mkdir(parents=True, exist_ok=True)

    # counts mutated ONLY in the main thread (no shared-state race). _stage_one
    # returns a status string; the rewritten path is derived from the dst the
    # main thread already holds, never from the worker return.
    counts = {"copied": 0, "skipped": 0, "failed": 0}

    n_real = len(lists["train_normal"]) + len(lists["test_defect"])
    n_synth = sum(len(synth_by_cond.get(c, [])) for c in ("random", "casda", "aroma"))
    logger.info("[LocalCache] %s: copying %d images -> %s", ds, n_real + n_synth, cache_base)
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=num_workers) as ex:
        # real train_normal — keep dst Path alongside the future so the local
        # list is built from the KNOWN dst (deterministic per index), not the
        # helper return (which is now a status string).
        tn_dsts = [real_imgs / f"tn_{i:05d}{Path(p).suffix}"
                   for i, p in enumerate(lists["train_normal"])]
        tn_futs = [ex.submit(_stage_one, p, str(d))
                   for p, d in zip(lists["train_normal"], tn_dsts)]
        # real test_defect
        td_dsts = [real_imgs / f"td_{i:05d}{Path(p).suffix}"
                   for i, p in enumerate(lists["test_defect"])]
        td_futs = [ex.submit(_stage_one, p, str(d))
                   for p, d in zip(lists["test_defect"], td_dsts)]
        for f in tn_futs:
            counts[f.result()] += 1
        for f in td_futs:
            counts[f.result()] += 1
        local_tn = [str(d) for d in tn_dsts]
        local_td = [str(d) for d in td_dsts]

        # mask_map: rewrite keys lockstep with test_defect
        local_mask_map: Dict[str, str] = {}
        mask_futs = {}  # local_p -> (future, mask_dst)
        for i, (orig_p, local_p) in enumerate(zip(lists["test_defect"], local_td)):
            if orig_p in lists["mask_map"]:
                mdst = real_masks / f"mask_{i:05d}.png"
                mask_futs[local_p] = (
                    ex.submit(_stage_one, lists["mask_map"][orig_p], str(mdst)),
                    mdst,
                )
        for local_p, (fut, mdst) in mask_futs.items():
            counts[fut.result()] += 1
            local_mask_map[local_p] = str(mdst)

        # class_mask_map (severstal multi mode): rewrite keys lockstep with
        # test_defect so per-class lookups by /tmp image path resolve. Without
        # this, _get_real_test_images_and_labels does class_mask_map.get(img_p)
        # with img_p=/tmp/td_* and misses every Drive-keyed entry -> 0 labels.
        # Per-class PNG values stay on Drive (only the key is remapped); they are
        # read once at bbox-extraction time, so /tmp acceleration of the merged
        # mask + image is preserved without staging every per-class mask.
        local_class_mask_map: Optional[Dict[str, Dict[int, str]]] = None
        src_cmm = lists.get("class_mask_map")
        if isinstance(src_cmm, dict):
            local_class_mask_map = {}
            for orig_p, local_p in zip(lists["test_defect"], local_td):
                if orig_p in src_cmm:
                    local_class_mask_map[local_p] = src_cmm[orig_p]

        # synth per condition
        synth_roots = {"random": random_synth_dir, "aroma": aroma_synth_dir}
        if casda_synth_dir:
            synth_roots["casda"] = casda_synth_dir
        new_synth_by_cond: Dict[str, List[Dict[str, Any]]] = {
            "baseline": list(synth_by_cond.get("baseline", []))
        }
        # Iterate only conditions present in synth_roots so casda is staged when
        # its dir was provided; absent → casda annotations carried through below.
        for cond in [c for c in ("random", "casda", "aroma") if c in synth_roots]:
            cond_imgs = cache_ds / cond / "images"
            cond_imgs.mkdir(parents=True, exist_ok=True)  # pre-create in main thread
            anns = synth_by_cond.get(cond, [])
            # Keep each dst Path alongside its future; rewritten ann paths are
            # built from the KNOWN dst (deterministic per index), never from the
            # helper return (now a status string).
            futs_list = []
            for j, ann in enumerate(anns):
                img_dst = cond_imgs / f"syn_{j:05d}{Path(ann['image_path']).suffix}"
                f_img = ex.submit(_stage_one, ann["image_path"], str(img_dst))
                f_norm = None
                norm_dst = None
                if ann.get("normal_image"):
                    norm_dst = cond_imgs / f"nrm_{j:05d}{Path(ann['normal_image']).suffix}"
                    f_norm = ex.submit(_stage_one, ann["normal_image"], str(norm_dst))
                # GT mask (exact-bbox path) — stage to local cache too, else the
                # rewritten image_path points at /tmp while mask stays on Drive
                # (loses /tmp acceleration and breaks if Drive unmounts mid-run).
                f_mask = None
                mask_dst = None
                mask_src = ann.get("mask_path")
                if mask_src and Path(mask_src).exists():
                    mask_dst = cond_imgs / f"msk_{j:05d}{Path(mask_src).suffix}"
                    f_mask = ex.submit(_stage_one, mask_src, str(mask_dst))
                futs_list.append((ann, f_img, img_dst, f_norm, norm_dst, f_mask, mask_dst))
            new_anns = []
            for ann, f_img, img_dst, f_norm, norm_dst, f_mask, mask_dst in futs_list:
                new_ann = dict(ann)
                counts[f_img.result()] += 1
                new_ann["image_path"] = str(img_dst)
                if f_norm is not None:
                    counts[f_norm.result()] += 1
                    new_ann["normal_image"] = str(norm_dst)
                if f_mask is not None:
                    counts[f_mask.result()] += 1
                    new_ann["mask_path"] = str(mask_dst)
                new_anns.append(new_ann)
            new_synth_by_cond[cond] = new_anns
            # copy annotations.json for fidelity
            src_json = Path(synth_roots[cond]) / ds / "annotations.json"
            if src_json.exists():
                counts[_stage_one(str(src_json), str(cache_ds / cond / "annotations.json"))] += 1

    # Carry through any non-baseline condition that wasn't staged above (e.g.
    # casda when --casda_synthetic_dir was omitted): keep its (possibly empty)
    # annotations on their original paths so the cap/refusal logic still sees it.
    for cond, anns in synth_by_cond.items():
        if cond not in new_synth_by_cond:
            new_synth_by_cond[cond] = list(anns)

    elapsed = time.time() - t0
    logger.info(
        "[LocalCache] %s staged: %d copied, %d skipped (cached), %d failed of %d using %d workers",
        ds, counts["copied"], counts["skipped"], counts["failed"],
        counts["copied"] + counts["skipped"] + counts["failed"], num_workers,
    )
    logger.info(
        "[LocalCache] %s ready: %.1fs (train_normal=%d defect=%d masks=%d random=%d casda=%d aroma=%d)",
        ds, elapsed, len(local_tn), len(local_td), len(local_mask_map),
        len(new_synth_by_cond.get("random", [])),
        len(new_synth_by_cond.get("casda", [])),
        len(new_synth_by_cond.get("aroma", [])),
    )

    new_lists = {
        **lists,
        "train_normal": local_tn,
        "test_defect": local_td,
        "mask_map": local_mask_map,
    }
    if local_class_mask_map is not None:
        new_lists["class_mask_map"] = local_class_mask_map
    return new_lists, new_synth_by_cond


def _get_real_test_images_and_labels(
    test_defect_paths: List[str],
    mask_map: Dict[str, str],
    label_dir: str,
    image_dir_out: str,
    min_area: int = 50,
    force_copy: bool = False,
    class_mask_map: Optional[Dict[str, Dict[int, str]]] = None,
) -> int:
    """
    real defect 이미지 + GT mask → YOLO label set 구성 (condition-agnostic).

    train pool 을 넘기면 train label, val pool 을 넘기면 val label 이 된다.

    각 defect 이미지(mask 보유) 마다:
      - binary mask PNG → bounding boxes
      - YOLO label 작성
      - 이미지 stage
    mask 가 없는 defect 이미지는 제외 (label 없으면 학습/평가 모두 불가).

    class_mask_map (severstal multi mode only): {img: {class_id(1..4): per_class_png}}.
    When provided, each per-class mask contributes bboxes with class_id-1 (0-indexed
    for YOLO nc=4). When None (default, every existing dataset + single mode), the
    merged mask_map drives a single class_id=0 — byte-identical to prior behavior.

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

        # --- MULTI MODE (severstal): per-class masks → multi-class YOLO lines ---
        if class_mask_map is not None:
            per_class = class_mask_map.get(img_p)
            if not per_class:
                n_no_bbox += 1
                continue
            lines: List[str] = []
            for cls in sorted(per_class):
                pc_path = per_class[cls]
                c_bboxes, mw, mh = _mask_to_bboxes(pc_path, min_area=min_area)
                if not c_bboxes:
                    continue
                if (mw, mh) != (img_w, img_h) and mw > 0 and mh > 0:
                    sx = img_w / mw
                    sy = img_h / mh
                    c_bboxes = [
                        (int(x * sx), int(y * sy), int(bw * sx), int(bh * sy))
                        for (x, y, bw, bh) in c_bboxes
                    ]
                lines.extend(
                    _bboxes_to_yolo_lines(c_bboxes, img_w, img_h, class_id=cls - 1)
                )
            if not lines:
                n_no_lines += 1
                continue
            stem = f"real_{i:06d}"
            img_dst = str(Path(image_dir_out) / f"{stem}{Path(img_p).suffix}")
            _link_or_copy(img_p, img_dst, force_copy=force_copy)
            (Path(label_dir) / f"{stem}.txt").write_text(
                "\n".join(lines) + "\n", encoding="utf-8"
            )
            n += 1
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

        # single/merged path: class_id=0. multi(generic + severstal)은 위
        # `class_mask_map is not None` 분기에서 처리되어 여기 도달하지 않는다.
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
    class_mode: str = "single",
    class_names: Optional[List[str]] = None,
    dataset_key: Optional[str] = None,
) -> None:
    """Write ultralytics dataset YAML.

    class_mode='single' (default): nc=1, names=['defect'] — byte-identical to the
    historical output for all existing datasets. class_mode='multi' with
    class_names: nc/names are derived per-dataset (severstal→c1..c4 display,
    others→raw defect-type folder names). Multi without class_names degrades to
    single (defensive).
    """
    if class_mode == "multi" and class_names:
        disp = _display_class_names(dataset_key or "", class_names)
        nc_line = f"nc: {len(disp)}\n"
        # Quote every name so numeric-looking codes (e.g. aitex '002') are NOT
        # parsed by YOLO's YAML loader as integers.
        names_line = "names: [" + ", ".join(f"'{n}'" for n in disp) + "]\n"
    else:
        nc_line = "nc: 1\n"
        names_line = "names: ['defect']\n"
    content = (
        f"path: {root_dir}\n"
        f"train: {train_img_subdir}\n"
        f"val: {val_img_subdir}\n"
        f"{nc_line}"
        f"{names_line}"
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
    n_source_defects: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    """Return parsed meta dict if cache valid + params match, else None."""
    meta_path = cache_root / "meta.json"
    if not meta_path.is_file():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None

    # v3: _mask_to_bboxes union-bbox 폴백 추가 (점상/미세 mask 라벨 회수) —
    # real 라벨 결과가 달라지므로 v2 캐시는 자동 무효화·재빌드.
    if meta.get("schema_version") != 3:
        return None
    if meta.get("min_area") != min_area:
        return None
    if meta.get("val_frac") != val_frac:
        return None
    if meta.get("split_seed") != split_seed:
        return None
    if int(meta.get("n_labeled", 0)) <= 0:
        return None
    # Invalidate if source defect image count changed since cache was built.
    # Old caches without this field are treated as stale (force rebuild once).
    cached_n = meta.get("n_source_defects")
    if cached_n is None or (n_source_defects is not None and int(cached_n) != n_source_defects):
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
    label: str = "stage",
) -> None:
    """Stage cached (image, label) pairs into a condition tmpdir.

    Uses _bulk_link_or_copy: when the source already lives on the same local
    filesystem (e.g. the per-dataset staging dir) this is an instant hardlink;
    cross-device sources fall back to a parallel byte copy. Labels are tiny text
    files but are staged the same way for consistency.
    """
    img_paths, lbl_paths, _ = triple
    img_pairs = [(ip, str(Path(img_dst_dir) / Path(ip).name)) for ip in img_paths]
    _bulk_link_or_copy(img_pairs, label=f"{label}/images")
    lbl_pairs = [
        (lp, str(Path(lbl_dst_dir) / Path(lp).name))
        for lp in lbl_paths
        if Path(lp).is_file()
    ]
    _bulk_link_or_copy(lbl_pairs, label=f"{label}/labels")


def _build_or_load_real_yolo_dataset(
    ds_key: str,
    mask_map: Dict[str, str],
    defect_splits: Dict[str, List[str]],
    cache_dir: Optional[str],
    min_area: int = 50,
    val_frac: float = 0.5,
    split_seed: int = 42,
    class_mask_map: Optional[Dict[str, Dict[int, str]]] = None,
    real_frac: float = 1.0,
) -> Dict[str, Tuple[List[str], List[str], int]]:
    """
    Build or load cached per-split real YOLO dataset.

    Returns {split: (img_paths, lbl_paths, n_labeled)} for train/val/test.
    When cache_dir is None: builds fresh, no persistence (behavior identical
    to the previous per-condition build).
    When cache_dir provided: checks Drive cache, builds+saves on miss.
    """
    use_cache = cache_dir is not None
    # Isolate multi-class caches from single so the two label sets never collide.
    # BOTH severstal and generic multi now supply class_mask_map (generic builds a
    # {img: {id+1: merged_mask}} in _get_image_lists), so this single check routes
    # every multi dataset to real_multi and never reuses a single-class 'real' cache.
    cache_leaf = "real_multi" if class_mask_map is not None else "real"
    # real_frac<1.0 은 train 부분집합이라 full 캐시와 n_source_defects 가 달라
    # 상호 무효화 thrash 발생 → frac 별 별도 leaf 로 격리 (1.0 은 기존 경로 불변).
    if real_frac < 1.0:
        cache_leaf = f"{cache_leaf}_frac{real_frac:g}"
    cache_root = Path(cache_dir) / ds_key / cache_leaf if use_cache else None

    # --- CACHE HIT ---
    n_source_defects = sum(len(v) for v in defect_splits.values())
    if use_cache and cache_root is not None:
        meta = _validate_real_cache(cache_root, min_area, val_frac, split_seed,
                                    n_source_defects=n_source_defects)
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
            class_mask_map=class_mask_map,
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
            "schema_version": 3,
            "splits": per_split_meta,
            "n_images": sum(v["n_images"] for v in per_split_meta.values()),
            "n_labeled": total,
            "n_source_defects": n_source_defects,
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
    batch: int = 16,
    cache: str = "",
    rect: bool = False,
    seed: int = 42,
    device: int = 0,
    patience: int = 0,
    class_mode: str = "single",
    class_names: Optional[List[str]] = None,
    name_to_id: Optional[Dict[str, int]] = None,
    dataset_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    하나의 (model, condition) 조합을 학습/평가.

    모든 조건이 real labeled defect (train split)를 공통으로 학습한다.
      baseline:        real labeled defect 만 (+ 약간의 real normal background) →
                       COCO 에서 시작, save=True 로 best.pt 영속화.
      random / aroma:  real labeled defect + synth → COCO 에서 처음부터 학습,
                       save=False.

    Returns {map50, map50_95, precision, recall, n_train, n_real_train,
             n_synth_train, weights_path(baseline only),
             per_class(class_mode=='multi' only): {name: {map50, map50_95,
             precision, recall}}}.
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
            _stage_pairs_into(
                real_sets["train"], train_img, train_lbl,
                label=f"{model_name}/{condition} train",
            )
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

            # --- TRAIN: add synthetic (random/casda/aroma only) ---
            n_synth = 0
            # Per-class synth count (multi mode transparency): how many synth
            # annotations each defect class contributed. Parsed from source_roi
            # via _parse_severstal_class (0-indexed). Exposes CASDA c4/c2
            # starvation. Keyed by 0-indexed class id (matches YOLO nc=4 idx).
            n_synth_per_class: Dict[int, int] = {}
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
                    min_area=50,  # unify with real val GT (_build_real min_area=50)
                    class_mode=class_mode,
                    name_to_id=name_to_id,
                    dataset_key=dataset_key,
                )
                logger.info(
                    "    %s train: %d/%d synth imgs labeled (additive)",
                    condition, n_synth, len(synth_annotations),
                )
                # Per-class synth distribution (multi mode only — single mode has
                # no class signal). Count over the *input* annotations by parsed
                # source_roi class so c4/c2 starvation is visible per condition.
                if class_mode == "multi":
                    # Distinct real source defects per class (synth diversity):
                    # how many UNIQUE source_roi each class draws from. High synth
                    # count + few distinct sources = low-diversity oversampling
                    # (the severstal c2 coverage question). Keyed like n_synth_per_class.
                    n_synth_src_per_class: Dict[int, set] = {}
                    for _ann in synth_annotations:
                        # Same resolver _write_yolo_labels uses (source_roi parse,
                        # then cluster_id fallback) so the diagnostic matches what
                        # actually gets labeled — CASDA c4/c2 starvation surfaces
                        # here instead of collapsing into the -1 bucket.
                        _cls = _resolve_synth_class(_ann, name_to_id, dataset_key)
                        _key = _cls if _cls is not None else -1  # -1 = unresolved
                        n_synth_per_class[_key] = n_synth_per_class.get(_key, 0) + 1
                        _sr = _ann.get("source_roi")
                        if _sr is not None:
                            n_synth_src_per_class.setdefault(_key, set()).add(_sr)
                    logger.info(
                        "    %s synth per-class (0-idx, -1=unparsed): %s",
                        condition,
                        {k: n_synth_per_class[k] for k in sorted(n_synth_per_class)},
                    )
                    logger.info(
                        "    %s synth distinct sources per-class (0-idx): %s",
                        condition,
                        {k: len(n_synth_src_per_class.get(k, ()))
                         for k in sorted(n_synth_per_class)},
                    )
            n_train = n_real + n_synth

            # Per-class instance histogram over staged train labels (real defect
            # + synth). Surfaces rare-class starvation (e.g. a class with ~0 train
            # boxes) right in the log instead of needing an external diagnostic.
            # Background negatives carry no label file, so they're excluded.
            try:
                _cls_hist: Dict[int, int] = {}
                for _lf in Path(train_lbl).glob("*.txt"):
                    for _ln in _lf.read_text().splitlines():
                        _tok = _ln.split()
                        if _tok:
                            _c = int(_tok[0])
                            _cls_hist[_c] = _cls_hist.get(_c, 0) + 1
                logger.info(
                    "    train label instances per class (yolo idx): %s",
                    {k: _cls_hist[k] for k in sorted(_cls_hist)},
                )
            except Exception as _h_exc:
                logger.warning("    train class histogram failed: %s", _h_exc)

            # --- VAL: real labeled defects (disjoint from train) ---
            _stage_pairs_into(
                real_sets["val"], val_img, val_lbl,
                label=f"{model_name}/{condition} val",
            )
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
                class_mode=class_mode,
                class_names=class_names,
                dataset_key=dataset_key,
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
                # scratch: random/aroma도 COCO에서 시작, real+synth 처음부터 학습.
                # do_save=True 필수: save=False면 train 종료 후 model에 best 가중치가
                # 복원되지 않아 model.val()이 학습 전(COCO) 가중치를 측정 → map50 붕괴
                # (P~0.004, epoch 무관 동일값). 조건별 checkpoint_dir(name=condition)로
                # 분리되어 baseline best.pt를 덮어쓰지 않음.
                weights = f"{model_name}.pt"
                epochs = baseline_epochs
                do_save = True

            # YOLO("{model}.pt") loads COCO pretrained weights; .train() = transfer learning.
            # NOTE: do NOT pass resume=True (would continue the same run's epoch counter).
            model = YOLO(weights)

            # Normalize --cache: ""/None→False (no cache), "ram"/"true"→"ram",
            # "disk"→"disk". Avoids relying on Ultralytics string coercion.
            _c = (cache or "").strip().lower()
            train_cache = "ram" if _c in ("ram", "true") else ("disk" if _c == "disk" else False)

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
                    patience=patience,  # 0=disabled (default), N>0=EarlyStopping
                    batch=batch,      # 16=Ultralytics default (기존 동작 불변)
                    cache=train_cache,  # ""→False=비캐싱(기존), ram/disk opt-in
                    rect=rect,        # False=정사각 letterbox (기존 동작 불변)
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

            # multi 모드: 클래스별 metric 기록 (single/타 데이터셋은 미추가 →
            # 기존 JSON byte-identical). out["per_class"] = {name: {map50, ...}}.
            if class_mode == "multi":
                per_class: Dict[str, Dict[str, float]] = {}
                try:
                    # val_results.names: {cls_id: name}; fallback ['c1'..'c4'].
                    # is None 체크: 빈 dict를 유효 매핑으로 오인하지 않음.
                    names = getattr(val_results, "names", None)
                    if names is None:
                        # Mirror the YAML display names so per_class keys stay
                        # consistent (severstal→c1..c4, others→folder names).
                        _disp = _display_class_names(dataset_key or "", class_names or [])
                        names = {i: n for i, n in enumerate(_disp)}
                    # ap_class_index 는 numpy 배열 → `or []` 같은 truthiness 평가
                    # 금지(원소>1 이면 "ambiguous truth value" 예외). None 만 분기.
                    _idx_raw = getattr(box, "ap_class_index", None)
                    idxs = [] if _idx_raw is None else list(_idx_raw)
                    for i, cls_id in enumerate(idxs):
                        cls_id = int(cls_id)
                        # class_result(i) → (P, R, AP@0.5, AP@0.5:0.95)
                        p, r, ap50, ap = box.class_result(i)
                        if isinstance(names, dict):
                            name = names.get(cls_id, f"class{cls_id}")
                        elif 0 <= cls_id < len(names):
                            name = names[cls_id]
                        else:
                            name = f"class{cls_id}"
                        per_class[name] = {
                            "map50":     round(float(ap50), 4),
                            "map50_95":  round(float(ap), 4),
                            "precision": round(float(p), 4),
                            "recall":    round(float(r), 4),
                        }
                except Exception as _pc_exc:
                    logger.warning(
                        "    %s/%s: per-class extract failed: %s",
                        model_name, condition, _pc_exc,
                    )
                # 빈 dict 가능(검출 전무) — multi 모드면 키는 항상 존재
                out["per_class"] = per_class
                # 조건별 synth의 class별 카운트(0-idx 문자열 키로 JSON 직렬화).
                # CASDA c4/c2 starvation 투명화 — n_synth_train과 parity 비교용.
                out["n_synth_per_class"] = {
                    str(k): v for k, v in sorted(n_synth_per_class.items())
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
    casda_synthetic_dir: Optional[str] = None,
    baseline_epochs: int = 50,
    val_frac: float = 0.5,
    max_synth_per_ds: Optional[int] = None,
    synth_ratio: Optional[float] = None,
    imgsz: int = 256,
    batch: int = 16,
    cache: str = "",
    rect: bool = False,
    seed: int = 42,
    existing_results: Optional[Dict[str, Any]] = None,
    output_path: Optional[str] = None,
    yolo_cache_dir: Optional[str] = None,
    local_cache: bool = True,
    patience: int = 0,
    class_mode: str = "single",
    real_frac: float = 1.0,
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
        ds_stage: Optional[str] = None  # per-dataset real staging dir (cleaned in finally)
        try:
            logger.info("=== Detection dataset: %s ===", ds)
            # multi-class는 데이터셋-일반 기능: --class_mode 를 전 데이터셋에 그대로
            # 적용한다. 실제 nc/names는 test/{type} 열거로 데이터셋별 결정되고, test
            # 폴더가 1개뿐이면 _enumerate_defect_classes가 single로 축퇴시킨다.
            # (기본값 single이라 플래그 미지정 시 전 데이터셋 byte-identical.)
            ds_class_mode = class_mode
            lists = _get_image_lists(
                ds, real_data_dir, seed=seed, class_mode=ds_class_mode,
            )
            if lists is None:
                logger.warning("Detection skip %s — dataset not found", ds)
                continue

            # synth annotations per condition (loaded once per dataset).
            # MOVED UP: must exist before the local cache so its paths get rewritten.
            synth_by_cond: Dict[str, List[Dict[str, Any]]] = {
                "baseline": [],
                "random":   _load_synth_annotations(random_synthetic_dir, ds),
                "casda":    (
                    _load_synth_annotations(casda_synthetic_dir, ds)
                    if casda_synthetic_dir else []
                ),
                "aroma":    _load_synth_annotations(aroma_synthetic_dir, ds),
            }

            # Absolute cap (synth_ratio 미지정 시에만 적용).
            if max_synth_per_ds is not None and synth_ratio is None:
                for cond in ("random", "casda", "aroma"):
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
                    casda_synth_dir=casda_synthetic_dir,
                )

            # Read class_mask_map AFTER local caching: _local_cache_for_yolo
            # rewrites its keys to /tmp paths in lockstep with test_defect.
            # Reading it before would retain Drive-path keys that never match the
            # /tmp test_defect paths used downstream -> 0 multi-mode labels.
            class_mask_map = lists.get("class_mask_map") if ds_class_mode == "multi" else None
            # class_names/name_to_id are path-independent (folder-name strings +
            # name→int), so _local_cache_for_yolo's {**lists} spread carries them
            # through unchanged (no key rewrite needed, unlike class_mask_map).
            name_to_id = lists.get("name_to_id") if ds_class_mode == "multi" else None
            class_names = lists.get("class_names") if ds_class_mode == "multi" else None

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

            # --real_frac: label-efficiency 커브용 real train 축소 (val 은 불변 —
            # 전 frac 지점에서 동일 평가셋이어야 커브 비교 가능). seed 종속
            # subsample 이라 --seeds 반복 시 seed 별 다른 부분집합(의도).
            # 하위 synth cap([SynthRatio])은 len(train_defect) 기반이라 자동 축소.
            if real_frac < 1.0 and train_defect:
                _n_before = len(train_defect)
                _k = max(1, int(round(_n_before * real_frac)))
                _rng_rf = random.Random(seed)
                train_defect = sorted(_rng_rf.sample(train_defect, _k))
                logger.info(
                    "  [RealFrac] %s: real train %d -> %d (frac=%.2f, val unchanged=%d)",
                    ds, _n_before, _k, real_frac, len(val_defect),
                )

            # Ratio-based synth cap (synth_ratio 지정 시 max_synth_per_ds 보다 우선).
            if synth_ratio is not None:
                n_real_train = len(train_defect)
                for cond in ("random", "casda", "aroma"):
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
                class_mask_map=class_mask_map,
                real_frac=real_frac,
            )

            # --- Per-dataset real staging (ONCE) -------------------------------
            # The real (image,label) set is byte-identical across baseline/random/
            # aroma, so stage it into one dataset-scoped local dir here and let each
            # condition hardlink from it (zero extra bytes) instead of copying the
            # whole real set 3x from Drive. real_sets paths may point at Drive
            # (yolo_cache) or already-local /tmp (cache-miss build); the first
            # staging copies Drive->local once, then per-condition staging is an
            # instant local->local hardlink.
            #
            # Resume guard: if every (model,cond) for this dataset already has a
            # valid cached map50, no condition will actually run -> skip staging.
            any_to_run = any(
                existing.get(ds, {}).get(m, {}).get(c, {}).get("map50") is None
                for m in model_keys
                for c in condition_keys
            )
            real_sets_local = real_sets
            if any_to_run:
                ds_stage = tempfile.mkdtemp(
                    prefix=f"exp4v2_real_{ds}_",
                    dir="/tmp" if os.name != "nt" else None,
                )
                _stage_dirs = {}
                for split in ("train", "val"):
                    img_d = Path(ds_stage) / split / "images"
                    lbl_d = Path(ds_stage) / split / "labels"
                    img_d.mkdir(parents=True, exist_ok=True)
                    lbl_d.mkdir(parents=True, exist_ok=True)
                    _stage_dirs[split] = (img_d, lbl_d)
                real_sets_local = {}
                for split in ("train", "val"):
                    img_paths, lbl_paths, n_lab = real_sets.get(split, ([], [], 0))
                    img_d, lbl_d = _stage_dirs[split]
                    img_pairs = [
                        (ip, str(img_d / Path(ip).name)) for ip in img_paths
                    ]
                    lbl_pairs = [
                        (lp, str(lbl_d / Path(lp).name))
                        for lp in lbl_paths
                        if Path(lp).is_file()
                    ]
                    _bulk_link_or_copy(img_pairs, label=f"{ds} real {split} images")
                    _bulk_link_or_copy(lbl_pairs, label=f"{ds} real {split} labels")
                    local_imgs = [p for _, p in img_pairs]
                    local_lbls = [p for _, p in lbl_pairs]
                    real_sets_local[split] = (local_imgs, local_lbls, n_lab)
                real_sets_local["test"] = real_sets.get("test", ([], [], 0))

            # checkpoint base persists under output_dir so best.pt survives the run.
            # Multi-seed: isolate per seed so seeds don't overwrite each other's
            # best.pt. Single-seed runs still get a _seeds/seed{N}/ subtree (the
            # weights are scratch-rebuilt anyway, so the path change is harmless).
            ckpt_base = str(Path(output_dir) / "_seeds" / f"seed{seed}" / ds)

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
                        real_sets=real_sets_local,
                        baseline_epochs=baseline_epochs,
                        imgsz=imgsz,
                        batch=batch,
                        cache=cache,
                        rect=rect,
                        seed=seed,
                        device=device,
                        patience=patience,
                        class_mode=ds_class_mode,
                        class_names=class_names,
                        name_to_id=name_to_id,
                        dataset_key=ds,
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

        except Exception as _ds_exc:
            logger.error(
                "Dataset %s failed — skipping and saving progress: %s",
                ds, _ds_exc,
            )
            if ds not in results:
                results[ds] = {"error": str(_ds_exc)}
            if output_path:
                try:
                    save_json(results, output_path)
                except Exception as _save_exc:
                    logger.warning("Error save failed: %s", _save_exc)
            continue
        finally:
            # Drop the per-dataset real staging dir (normal or error path). Only
            # hardlinks/copies live here; the persistent yolo_cache is untouched.
            if ds_stage is not None:
                shutil.rmtree(ds_stage, ignore_errors=True)

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


def _per_class_block(m_data: Dict[str, Any], conds: List[str]) -> List[str]:
    """multi 모드 per-class map50 표 (rows=class, cols=조건 + Δ(A-R)).

    조건 셀의 per_class(단일=실측, 멀티시드=클래스별 mean)에서 map50을 읽는다.
    per_class가 어떤 조건에도 없으면(=single 모드/타 데이터셋) [] 반환 → 표 미출력.
    값 전체(map50_95/precision/recall)는 JSON 참조.
    """
    names: List[str] = []
    seen = set()
    for cond in conds:
        pc = (m_data.get(cond, {}) or {}).get("per_class") or {}
        for cname in pc:
            if cname not in seen:
                seen.add(cname)
                names.append(cname)
    if not names:
        return []
    names = sorted(names)

    out_cols = list(conds) + ["Δ(A-R)"]
    lines = ["#### per-class map50 (multi)", ""]
    header = " | ".join(f"{c:>10}" for c in out_cols)
    lines.append(f"| {'class':<8} | {header} |")
    sep = " | ".join("-" * 10 for _ in out_cols)
    lines.append(f"|{'-' * 10}|{sep}|")
    for cname in names:
        vals: Dict[str, Optional[float]] = {}
        cells = []
        for cond in conds:
            pc = (m_data.get(cond, {}) or {}).get("per_class") or {}
            v = (pc.get(cname) or {}).get("map50")
            v = v if isinstance(v, (int, float)) else None
            vals[cond] = v
            cells.append(f"{v:.4f}" if v is not None else "N/A")
        a, r = vals.get("aroma"), vals.get("random")
        if isinstance(a, (int, float)) and isinstance(r, (int, float)):
            cells.append(f"{(a - r) * 1.0:+.4f}")
        else:
            cells.append("N/A")
        row = " | ".join(f"{c:>10}" for c in cells)
        lines.append(f"| {cname:<8} | {row} |")
    lines.append("")
    return lines


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

            # per-class map50 표 (multi 모드에서만; single은 [] 반환)
            lines += _per_class_block(m_data, list(all_conds))

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Multi-seed aggregation
# ---------------------------------------------------------------------------

_AGG_METRICS = ("map50", "map50_95", "precision", "recall")
_INT_FIELDS = ("n_train", "n_real_train", "n_synth_train")


def _aggregate_seeds(
    per_seed_results: Dict[int, Dict[str, Any]],
    seed_list: List[int],
) -> Dict[str, Any]:
    """
    Aggregate per-seed results into mean ± std + 95% CI per (ds, model, cond).

    per_seed_results: {seed: results_dict} where results_dict is the standard
    {ds: {model: {cond: {map50, map50_95, precision, recall, n_*, ...}}}}.

    Returns the SAME nested shape with top-level metric keys = mean (backward
    compatible: plot_exp4v2_results.py reads results[ds][model][cond]["map50"]).
    Adds per_seed{}, std{}, ci95{}, n_seeds. Failed seeds (missing cell or
    map50 is None) are dropped from a cell's aggregation; k = #successful seeds.
    """
    # Enumerate every (ds, model, cond) that appears in any seed.
    cells: Dict[Tuple[str, str, str], None] = {}
    for s in seed_list:
        rs = per_seed_results.get(s) or {}
        for ds, m_data in rs.items():
            if not isinstance(m_data, dict):
                continue
            for model, c_data in m_data.items():
                if not isinstance(c_data, dict):
                    continue
                for cond in c_data:
                    cells[(ds, model, cond)] = None

    out: Dict[str, Any] = {}
    for (ds, model, cond) in cells:
        # Collect per-seed metric vectors (only seeds with a valid map50 count).
        per_seed_cell: Dict[str, Dict[str, Any]] = {}
        metric_samples: Dict[str, List[float]] = {k: [] for k in _AGG_METRICS}
        int_fields: Dict[str, Optional[int]] = {k: None for k in _INT_FIELDS}
        int_conflict = False
        weights_path: Optional[str] = None
        # per_class(멀티시드): {class_name: {metric: [seed별 값,...]}} — 클래스 union
        pc_samples: Dict[str, Dict[str, List[float]]] = {}
        # n_synth_per_class: synth set은 seed 간 동일(같은 annotations + 같은 cap)이라
        # 결정적 → 처음 만난 seed 값을 그대로 채택해 top-level 집계에 전달한다.
        # 빠지면 Colab per-class synth parity 셀이 None을 읽는다.
        n_synth_per_class: Optional[Dict[str, int]] = None

        for s in seed_list:
            cell = (
                (per_seed_results.get(s) or {})
                .get(ds, {})
                .get(model, {})
                .get(cond)
            )
            if not isinstance(cell, dict):
                continue
            if cell.get("map50") is None:
                # failed/incomplete seed for this cell — skip from aggregation
                continue
            ps_metrics: Dict[str, Any] = {}
            for k in _AGG_METRICS:
                v = cell.get(k)
                if isinstance(v, (int, float)):
                    metric_samples[k].append(float(v))
                    ps_metrics[k] = float(v)
                else:
                    ps_metrics[k] = None
            per_seed_cell[str(s)] = ps_metrics
            # per_class 수집: seed마다 등장 클래스 집합이 다를 수 있어 union 누적
            pc = cell.get("per_class")
            if isinstance(pc, dict):
                for cname, cmetrics in pc.items():
                    if not isinstance(cmetrics, dict):
                        continue
                    slot = pc_samples.setdefault(cname, {m: [] for m in _AGG_METRICS})
                    for m in _AGG_METRICS:
                        cv = cmetrics.get(m)
                        if isinstance(cv, (int, float)):
                            slot[m].append(float(cv))
            # integer fields: expect identical across seeds; flag if not
            for k in _INT_FIELDS:
                v = cell.get(k)
                if isinstance(v, int):
                    if int_fields[k] is None:
                        int_fields[k] = v
                    elif int_fields[k] != v:
                        int_conflict = True
            if cond == "baseline" and weights_path is None and cell.get("weights_path"):
                weights_path = cell.get("weights_path")
            # 결정적 필드: 처음 보이는 seed의 n_synth_per_class를 채택.
            if n_synth_per_class is None:
                _nspc = cell.get("n_synth_per_class")
                if isinstance(_nspc, dict):
                    n_synth_per_class = _nspc

        k = len(per_seed_cell)
        if k == 0:
            # Every seed failed for this cell — DROP it (don't write a None
            # placeholder). Readers use .get(cond, {}) → defaults; resume sees
            # the cell absent → re-runs. A None-filled cell would instead crash
            # plot_exp4v2_results.py (int(None)) and persist a dead entry.
            logger.warning(
                "  [Aggregate] %s/%s/%s: all %d seeds failed — cell dropped",
                ds, model, cond, len(seed_list),
            )
            continue

        if int_conflict:
            logger.warning(
                "  [Aggregate] %s/%s/%s: integer fields differ across seeds "
                "(n_train/n_real_train/n_synth_train) — using a representative value",
                ds, model, cond,
            )

        mean: Dict[str, float] = {}
        std: Dict[str, float] = {}
        ci95: Dict[str, List[float]] = {}
        for col in _AGG_METRICS:
            samples = metric_samples[col]
            if samples:
                m = float(np.mean(samples))
                # sample std (ddof=1); 0.0 when fewer than 2 samples
                s_std = float(np.std(samples, ddof=1)) if len(samples) >= 2 else 0.0
            else:
                m, s_std = 0.0, 0.0
            mean[col] = round(m, 4)
            std[col] = round(s_std, 4)
            ci95[col] = _ci95(samples, m, s_std)

        agg_cell = {
            "map50":        mean["map50"],
            "map50_95":     mean["map50_95"],
            "precision":    mean["precision"],
            "recall":       mean["recall"],
            "n_train":      int_fields["n_train"],
            "n_real_train": int_fields["n_real_train"],
            "n_synth_train": int_fields["n_synth_train"],
            "n_seeds":      k,
            "per_seed":     per_seed_cell,
            "std":          std,
            "ci95":         ci95,
        }
        if weights_path is not None:
            agg_cell["weights_path"] = weights_path
        # n_synth_per_class 부착 (multi 모드만 존재). 없으면 키 미추가 →
        # single/비-멀티 셀은 byte-identical.
        if n_synth_per_class is not None:
            agg_cell["n_synth_per_class"] = n_synth_per_class

        # per_class mean 부착 (멀티시드): 클래스별·metric별로 값이 있는 seed만 평균.
        # 하나도 없으면 키 미추가 → single/multi 비-멀티클래스 셀은 byte-identical.
        if pc_samples:
            per_class_mean: Dict[str, Dict[str, Optional[float]]] = {}
            for cname, slot in pc_samples.items():
                per_class_mean[cname] = {
                    m: (round(float(np.mean(slot[m])), 4) if slot[m] else None)
                    for m in _AGG_METRICS
                }
            agg_cell["per_class"] = per_class_mean

        out.setdefault(ds, {}).setdefault(model, {})[cond] = agg_cell

    return out


def _ci95(samples: List[float], mean: float, std: float) -> List[float]:
    """95% CI half-width via t-distribution (small samples); fallback 1.96·SE.

    Returns [lo, hi]. k<2 → [mean, mean] (no spread estimable).
    """
    k = len(samples)
    if k < 2:
        return [round(mean, 4), round(mean, 4)]
    se = std / (k ** 0.5)
    tcrit: Optional[float] = None
    try:
        from scipy import stats  # type: ignore[import]
        tcrit = float(stats.t.ppf(0.975, df=k - 1))
    except Exception:
        tcrit = None
    if tcrit is None:
        # small-sample t critical values (two-sided 95%); fall back to 1.96
        _T = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571,
              7: 2.447, 8: 2.365, 9: 2.306, 10: 2.262}
        tcrit = _T.get(k, 1.96)
    half = tcrit * se
    return [round(mean - half, 4), round(mean + half, 4)]


# ---------------------------------------------------------------------------
# Multi-seed summary builder (mean ± std)
# ---------------------------------------------------------------------------

def _fmt_mean_std_delta(x: Dict[str, Any], y: Dict[str, Any]) -> str:
    """Format (mean(x) - mean(y)) in pp + combined std for the core metrics."""
    parts = []
    for col in _AGG_METRICS:
        xv, yv = x.get(col), y.get(col)
        if isinstance(xv, (int, float)) and isinstance(yv, (int, float)):
            d = (xv - yv) * 100
            xs = (x.get("std") or {}).get(col, 0.0)
            ys = (y.get("std") or {}).get(col, 0.0)
            comb = ((float(xs) ** 2 + float(ys) ** 2) ** 0.5) * 100
            parts.append(f"{col} {d:+.2f}±{comb:.2f}pp")
        else:
            parts.append(f"{col} N/A")
    return ", ".join(parts)


def _build_summary_multiseed(results: Dict[str, Any], seed_list: List[int]) -> str:
    """
    Build Markdown summary with mean ± std per condition (multi-seed).

    Falls back to the single-seed table layout but renders each metric cell as
    'mean ± std' and prints n_seeds. Delta panel reports mean delta ± combined std.
    """
    lines = [
        "# AROMA Exp 4 v2 — Supervised YOLOv8 Defect Detection 평가 (multi-seed)",
        "",
        f"seeds = {seed_list}  (n_seeds={len(seed_list)})",
        "각 셀 = mean ± std (sample std, ddof=1; n_seeds<2면 std=0). 95% CI는 JSON ci95 참조.",
        "비교: Baseline (real-only) vs Random ROI (real+synth) vs AROMA ROI (real+synth)",
        "Val = real defect (GT mask → bbox), train 과 disjoint. seed별 독립 split.",
        "",
    ]

    all_conds = ALL_CONDITION_KEYS
    metric_cols = list(_AGG_METRICS)
    int_cols = ["n_real_train", "n_synth_train", "n_seeds"]

    for ds in sorted(results):
        ds_data = results[ds]
        lines += [f"## {ds}", ""]

        models_present = [m for m in ALL_MODEL_KEYS if m in ds_data]
        if not models_present:
            models_present = sorted(ds_data.keys())

        for model_name in models_present:
            m_data = ds_data.get(model_name, {})
            lines += [f"### {model_name}", ""]

            cols = metric_cols + int_cols
            header = " | ".join(f"{c:>16}" for c in cols)
            lines.append(f"| {'조건':<10} | {header} |")
            sep = " | ".join("-" * 16 for _ in cols)
            lines.append(f"|{'-' * 12}|{sep}|")

            for cond in all_conds:
                cm = m_data.get(cond, {})
                std = cm.get("std") or {}
                cells = []
                for col in metric_cols:
                    val = cm.get(col)
                    if isinstance(val, (int, float)):
                        sv = std.get(col, 0.0)
                        cells.append(f"{val:.4f}±{float(sv):.4f}")
                    else:
                        cells.append("N/A")
                for col in int_cols:
                    val = cm.get(col)
                    cells.append(str(val) if isinstance(val, int) else "N/A")
                row = " | ".join(f"{c:>16}" for c in cells)
                lines.append(f"| {cond:<10} | {row} |")
            lines.append("")

            a = m_data.get("aroma", {})
            b = m_data.get("baseline", {})
            r = m_data.get("random", {})
            lines.append("**Delta (AROMA − Baseline)**: " + _fmt_mean_std_delta(a, b))
            lines.append("**Delta (AROMA − Random)**:   " + _fmt_mean_std_delta(a, r))
            lines.append("")

            # per-class map50 표 (multi 모드; 멀티시드는 클래스별 mean). single은 [] 반환
            lines += _per_class_block(m_data, list(all_conds))

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
    casda_synthetic_dir: Optional[str] = None,
    baseline_epochs: int = 50,
    val_frac: float = 0.5,
    max_synth_per_ds: Optional[int] = None,
    synth_ratio: Optional[float] = None,
    imgsz: int = 256,
    batch: int = 16,
    cache: str = "",
    rect: bool = False,
    seed: int = 42,
    seeds: Optional[List[int]] = None,
    resume: bool = False,
    yolo_cache_dir: Optional[str] = None,
    local_cache: bool = True,
    patience: int = 0,
    class_mode: str = "single",
    real_frac: float = 1.0,
) -> Dict[str, Any]:
    if not _CV2_AVAILABLE:
        logger.error("OpenCV (cv2) not installed. Run: pip install opencv-python-headless")
        return {"status": "error", "results": {}}

    # --seeds 우선; 미지정 시 단일 [seed] (기존 단일-seed 동작과 동일하게 흡수)
    seed_list = seeds or [seed]

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    output_path = str(out / "exp4v2_results.json")

    per_seed_results: Dict[int, Dict[str, Any]] = {}
    for s in seed_list:
        # Per-seed results live under _seeds/seed{N}/ so each seed resumes
        # independently and never overwrites another seed's JSON.
        seed_dir = out / "_seeds" / f"seed{s}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        seed_output_path = str(seed_dir / "exp4v2_results.json")

        existing_results: Dict[str, Any] = {}
        if resume and Path(seed_output_path).exists():
            try:
                existing_results = load_json(seed_output_path)
                n = sum(
                    1
                    for ds in existing_results
                    for m in existing_results[ds]
                    for c in existing_results[ds][m]
                )
                logger.info(
                    "Resume[seed%s]: loaded %d existing results from %s",
                    s, n, seed_output_path,
                )
            except Exception as e:
                logger.warning(
                    "Resume[seed%s]: failed to load existing results: %s — starting fresh",
                    s, e,
                )

        logger.info("=== SEED %s (%d of %d) ===", s, seed_list.index(s) + 1, len(seed_list))
        try:
            res_s = _run_detection_mode(
                model_keys=model_keys,
                condition_keys=condition_keys,
                dataset_keys=dataset_keys,
                random_synthetic_dir=random_synthetic_dir,
                aroma_synthetic_dir=aroma_synthetic_dir,
                real_data_dir=real_data_dir,
                output_dir=output_dir,
                casda_synthetic_dir=casda_synthetic_dir,
                baseline_epochs=baseline_epochs,
                val_frac=val_frac,
                max_synth_per_ds=max_synth_per_ds,
                synth_ratio=synth_ratio,
                imgsz=imgsz,
                batch=batch,
                cache=cache,
                rect=rect,
                seed=s,
                existing_results=existing_results,
                output_path=seed_output_path,
                yolo_cache_dir=yolo_cache_dir,
                local_cache=local_cache,
                patience=patience,
                class_mode=class_mode,
                real_frac=real_frac,
            )
            per_seed_results[s] = res_s
            save_json(res_s, seed_output_path)
        except Exception as exc:
            logger.error("Seed %s failed entirely: %s — continuing with other seeds", s, exc)
            # keep whatever partial JSON exists so this seed still contributes
            if Path(seed_output_path).exists():
                try:
                    per_seed_results[s] = load_json(seed_output_path)
                except Exception:
                    per_seed_results[s] = {}
            else:
                per_seed_results[s] = {}

    # Aggregate across seeds → top-level metrics = mean (backward compatible),
    # plus per_seed/std/ci95/n_seeds.
    results = _aggregate_seeds(per_seed_results, seed_list)

    save_json(results, output_path)
    if len(seed_list) > 1:
        summary = _build_summary_multiseed(results, seed_list)
    else:
        # single seed → keep the original summary layout (no ± noise)
        summary = _build_summary(results)
    (out / "exp4v2_summary.md").write_text(summary, encoding="utf-8")

    logger.info(
        "Saved exp4v2_results.json + exp4v2_summary.md -> %s  (%d datasets, %d seeds)",
        out, len(results), len(seed_list),
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
        nargs="+",
        choices=["baseline", "random", "casda", "aroma", "all"],
        default=["all"],
        help="평가 조건 (1개 이상, default: all). 예: --condition baseline random aroma",
    )
    p.add_argument(
        "--dataset_keys",
        required=True,
        nargs="+",
        help="평가할 데이터셋 키 목록 (e.g. severstal mvtec_leather aitex mtd)",
    )
    p.add_argument(
        "--random_synthetic_dir",
        required=True,
        help="Random synthetic 출력 루트 ({dir}/{dataset_key}/annotations.json)",
    )
    p.add_argument(
        "--casda_synthetic_dir",
        default=None,
        help="CASDA synthetic 출력 루트 ({dir}/{dataset_key}/annotations.json). "
             "생략 시 casda 조건은 합성 부재로 거부됨(no_synth_annotations).",
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
        "--real_frac", type=float, default=1.0,
        help=(
            "label-efficiency 커브용 real train 축소 비율 (default: 1.0=전량, "
            "기존 동작 불변). val 은 불변(전 frac 지점 동일 평가셋). "
            "synth cap 은 축소된 train 기준 자동 재계산. 주 주장은 Δ(aroma-random) "
            "커브로 한정. frac 지점별 fresh --output_dir 사용(resume 충돌 방지)."
        ),
    )
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
    p.add_argument(
        "--patience", type=int, default=0,
        help=(
            "YOLOv8 EarlyStopping patience (default: 0=비활성). "
            "best mAP50-95 갱신 없이 N epoch 경과 시 훈련 조기 종료. "
            "권장: 10~20 (epochs=50 기준). 0=끝까지 훈련."
        ),
    )
    p.add_argument("--imgsz",  type=int, default=256,
                   help="YOLO image size (default: 256)")
    p.add_argument(
        "--batch", type=int, default=16,
        help=(
            "YOLO train batch size (default: 16 = Ultralytics 기본). "
            "-1=auto(GPU 60%% VRAM) 가능하나 method/조건간 실제 batch가 달라져 "
            "공정성·재현성을 해칠 수 있어 고정값 권장."
        ),
    )
    p.add_argument(
        "--cache", type=str, default="",
        help=(
            "YOLO 데이터 캐싱 (default: '' → False=비캐싱, 기존 동작). "
            "'ram'=RAM 캐시(최고속, 메모리 충분 시), 'disk'=디스크 캐시, "
            "'True'=ram 별칭. I/O 병목 시 속도 향상."
        ),
    )
    p.add_argument(
        "--rect", action="store_true",
        help=(
            "rectangular training 활성화 (default: False=정사각 letterbox, 기존 동작). "
            "배치 내 종횡비 유사 이미지 묶어 패딩 최소화 → 속도 향상. "
            "단 batch 내 이미지 크기 변동으로 결과가 미세하게 달라질 수 있음."
        ),
    )
    p.add_argument(
        "--class_mode",
        choices=["single", "multi"],
        default="single",
        help=(
            "단일(single)=nc1 'defect' (기존 전 데이터셋 동작 불변, default). "
            "multi=데이터셋별 per-class (test/{type} 폴더로 nc/names 자동 결정: "
            "severstal 4·mvtec_leather 5·aitex 12·mtd 5 등). test 폴더가 1개뿐이면 "
            "single로 축퇴. 전 데이터셋 적용."
        ),
    )
    p.add_argument("--seed",   type=int, default=42,
                   help="단일 seed (default: 42). --seeds 미지정 시 사용.")
    p.add_argument(
        "--seeds", type=int, nargs="+", default=None,
        help=(
            "multi-seed 반복 목록 (예: --seeds 42 1 2). 지정 시 각 seed로 "
            "독립 학습(train/val split·synth subsample·YOLO train 모두 seed 종속) 후 "
            "조건별 mean±std + 95% CI 집계. 미지정 시 [--seed] 단일 실행(기존 동작)."
        ),
    )
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
    # --condition은 다중 선택(nargs="+"). 'all'이 포함되면 전체.
    # 그 외에는 ALL_CONDITION_KEYS의 canonical 순서로 정렬 + 중복 제거.
    if "all" in args.condition:
        condition_keys = ALL_CONDITION_KEYS
    else:
        _selected = set(args.condition)
        condition_keys = [c for c in ALL_CONDITION_KEYS if c in _selected]

    # --seeds 우선; 미지정 시 단일 [--seed] (기존 동작과 동일)
    seed_list = args.seeds if args.seeds else [args.seed]
    # Dedup while preserving order — duplicate seeds would share one _seeds/seedN
    # dir (resume-skip the 2nd) and collapse per_seed dict, falsely shrinking std.
    _seen: set = set()
    _deduped = [s for s in seed_list if not (s in _seen or _seen.add(s))]
    if len(_deduped) != len(seed_list):
        logger.warning("Duplicate seeds removed: %s → %s", seed_list, _deduped)
    seed_list = _deduped

    result = run(
        model_keys=model_keys,
        condition_keys=condition_keys,
        dataset_keys=args.dataset_keys,
        random_synthetic_dir=args.random_synthetic_dir,
        aroma_synthetic_dir=args.aroma_synthetic_dir,
        real_data_dir=args.real_data_dir,
        output_dir=args.output_dir,
        casda_synthetic_dir=args.casda_synthetic_dir,
        baseline_epochs=args.baseline_epochs,
        val_frac=args.val_frac,
        max_synth_per_ds=args.max_synth_per_ds,
        synth_ratio=args.synth_ratio,
        imgsz=args.imgsz,
        batch=args.batch,
        cache=args.cache,
        rect=args.rect,
        seed=args.seed,
        seeds=seed_list,
        resume=args.resume,
        yolo_cache_dir=args.yolo_cache_dir,
        local_cache=not args.no_local_cache,
        patience=args.patience,
        class_mode=args.class_mode,
        real_frac=args.real_frac,
    )
    status = result.get("status", "unknown")
    n_ds   = len(result.get("results", {}))
    print(f"[exp4v2] status={status}  n_datasets={n_ds}")
    if status != "ok":
        sys.exit(1)


if __name__ == "__main__":
    main()
