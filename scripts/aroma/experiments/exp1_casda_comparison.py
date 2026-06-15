#!/usr/bin/env python3
"""
AROMA Exp 1 — CASDA Baseline Comparison (Severstal)

3-way comparison: Random ROI / CASDA ROI / AROMA ROI, same copy-paste synthesis.
Evaluates ROI modeling quality + synthesis quality (FID) + downstream AD (PaDiM).

Modes (--mode):
    roi    ROI modeling quality (5 metrics × 3 methods, CPU-only)
    fid    FID(real_patch, method_patch) × 3 methods (CPU or GPU)
    ad     PaDiM anomaly detection × 4 conditions (GPU required, anomalib>=1.0)
    all    roi + fid + ad sequentially

Shared candidate pool:
    All three methods use AROMA's roi_candidates.json as the reference pool
    for ROI quality metrics (fair comparison — same universe of possible ROIs).

Copy-paste limitation (논문 명시):
    Copy-paste preserves original defect appearance and cannot generate novel
    defect morphologies. The objective is to evaluate ROI modeling quality,
    not synthesis model improvement.

Usage (Colab):
    !python $AROMA_SCRIPTS/experiments/exp1_casda_comparison.py \
        --mode all \
        --aroma_roi_dir        $AROMA_OUT_SEVERSTAL/roi/severstal \
        --casda_roi_dir        $AROMA_OUT_SEVERSTAL/roi_casda/severstal \
        --random_roi_dir       $AROMA_OUT_SEVERSTAL/roi_random/severstal \
        --aroma_synthetic_dir  $AROMA_OUT_SEVERSTAL/synthetic_aroma/severstal \
        --casda_synthetic_dir  $AROMA_OUT_SEVERSTAL/synthetic_casda/severstal \
        --random_synthetic_dir $AROMA_OUT_SEVERSTAL/synthetic_random/severstal \
        --real_data_dir        $SEVERSTAL_DATA \
        --output_dir           $AROMA_OUT_SEVERSTAL/exp1

Outputs (written to --output_dir):
    exp1_results.json    per-mode × method × metrics
    exp1_summary.md      comparison tables + delta sections
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import shutil
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("aroma.exp1")

# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

_THIS_DIR = Path(__file__).resolve().parent
_AROMA_SCRIPTS = _THIS_DIR.parent  # scripts/aroma/


def _bootstrap() -> None:
    aroma_ref = os.environ.get("AROMA_REF") or r"D:\project\aroma"
    for p in (str(_THIS_DIR), str(_AROMA_SCRIPTS), str(Path(aroma_ref))):
        if p not in sys.path:
            sys.path.insert(0, p)


_bootstrap()

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

from exp2_roi_quality import compute_metrics  # type: ignore[import]  # noqa: E402

# ---------------------------------------------------------------------------
# Severstal utilities
# ---------------------------------------------------------------------------

_SEVERSTAL_HEIGHT = 256
_SEVERSTAL_WIDTH  = 1600


def _decode_rle(rle_string: str, height: int = _SEVERSTAL_HEIGHT,
                width: int = _SEVERSTAL_WIDTH) -> np.ndarray:
    """Decode Severstal run-length encoding to binary mask (column-major)."""
    tokens = rle_string.strip().split()
    mask = np.zeros(height * width, dtype=np.uint8)
    for i in range(0, len(tokens), 2):
        start = int(tokens[i]) - 1   # 1-indexed → 0-indexed
        length = int(tokens[i + 1])
        mask[start:start + length] = 1
    return mask.reshape(height, width, order="F")


def _bbox_from_mask(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """Return (x1, y1, x2, y2) bounding box of non-zero region, or None."""
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any():
        return None
    y1, y2 = int(np.where(rows)[0][0]), int(np.where(rows)[0][-1])
    x1, x2 = int(np.where(cols)[0][0]), int(np.where(cols)[0][-1])
    return x1, y1, x2 + 1, y2 + 1


def load_severstal_defect_patches(
    real_data_dir: str,
    max_patches: int = 1000,
) -> List[Tuple[np.ndarray, str]]:
    """Load real defect crops from Severstal train.csv + train_images/.

    Returns list of (crop_array_uint8, image_id) tuples.
    """
    try:
        from PIL import Image
    except ImportError:
        logger.error("Pillow required for defect patch loading: pip install Pillow")
        return []

    data_dir = Path(real_data_dir)
    train_csv = data_dir / "train.csv"
    img_dir = data_dir / "train_images"

    if not train_csv.exists():
        logger.warning("train.csv not found: %s", train_csv)
        return []

    patches: List[Tuple[np.ndarray, str]] = []

    with open(train_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if len(patches) >= max_patches:
                break
            encoded = row.get("EncodedPixels", "").strip()
            if not encoded or encoded == "-1":
                continue

            image_file = img_dir / row["ImageId"]
            if not image_file.exists():
                continue

            try:
                img = np.array(Image.open(str(image_file)).convert("RGB"))
                mask = _decode_rle(encoded)
                bbox = _bbox_from_mask(mask)
                if bbox is None:
                    continue
                x1, y1, x2, y2 = bbox
                crop = img[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                patches.append((crop, row["ImageId"]))
            except Exception as e:
                logger.warning("Failed to crop %s: %s", row["ImageId"], e)

    logger.info("Loaded %d real defect patches from Severstal", len(patches))
    return patches


def load_severstal_image_lists(
    real_data_dir: str,
    test_split: float = 0.2,
    seed: int = 42,
) -> Tuple[List[str], List[str], Dict[str, List[str]]]:
    """Split Severstal images into train-normal, test-good, test-defect-by-class.

    Returns:
        train_normal_paths: paths to normal images for training
        test_good_paths:    paths to normal images for test
        test_defect:        {class_id_str: [image_path, ...]}
    """
    data_dir = Path(real_data_dir)
    train_csv = data_dir / "train.csv"
    img_dir = data_dir / "train_images"

    if not train_csv.exists():
        logger.warning("train.csv not found: %s", train_csv)
        return [], [], {}

    defect_images: Dict[str, str] = {}   # image_id → class_id
    with open(train_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            encoded = row.get("EncodedPixels", "").strip()
            if encoded and encoded != "-1":
                defect_images[row["ImageId"]] = row.get("ClassId", "1")

    all_images = sorted(str(p) for p in img_dir.glob("*.jpg"))
    normal_images = [p for p in all_images if Path(p).name not in defect_images]
    defect_image_paths = [p for p in all_images if Path(p).name in defect_images]

    rng = np.random.default_rng(seed)
    rng.shuffle(normal_images)

    n_test = max(1, int(len(normal_images) * test_split))
    test_good = sorted(normal_images[:n_test])
    train_normal = sorted(normal_images[n_test:])

    test_defect: Dict[str, List[str]] = defaultdict(list)
    for p in defect_image_paths:
        cls = defect_images.get(Path(p).name, "1")
        test_defect[cls].append(p)

    logger.info(
        "Severstal split: %d train-normal, %d test-good, %d test-defect",
        len(train_normal), len(test_good), len(defect_image_paths),
    )
    return train_normal, test_good, dict(test_defect)


# ---------------------------------------------------------------------------
# Mode: ROI quality
# ---------------------------------------------------------------------------

def _run_roi_mode(
    aroma_roi_dir: str,
    casda_roi_dir: str,
    random_roi_dir: str,
) -> Dict[str, Any]:
    """Compute 5-metric ROI quality for all 3 methods vs shared candidate pool."""
    cand_path = Path(aroma_roi_dir) / "roi_candidates.json"
    if not cand_path.exists():
        logger.error("roi_candidates.json not found: %s", cand_path)
        return {}

    candidates: List[Dict[str, Any]] = load_json(str(cand_path))

    method_dirs = {
        "aroma":  Path(aroma_roi_dir),
        "casda":  Path(casda_roi_dir),
        "random": Path(random_roi_dir),
    }

    result: Dict[str, Any] = {}
    for method, d in method_dirs.items():
        sel_path = d / "roi_selected.json"
        if not sel_path.exists():
            logger.warning("roi_selected.json missing for %s: %s", method, sel_path)
            continue
        selected: List[Dict[str, Any]] = load_json(str(sel_path))
        result[method] = compute_metrics(selected, candidates)
        logger.info(
            "  %s  morph=%.4f ctx=%.4f rare=%.4f entropy=%.4f gini=%.4f  n=%d",
            method.upper(),
            result[method]["morphology_coverage"],
            result[method]["context_coverage"],
            result[method]["rare_pair_coverage"],
            result[method]["entropy"],
            result[method]["gini"],
            result[method]["n_selected"],
        )

    return result


# ---------------------------------------------------------------------------
# Mode: FID
# ---------------------------------------------------------------------------

def _resize_to_tensor(crop: np.ndarray, size: int = 299):
    """Resize HxWxC uint8 array to CxSxS uint8 torch tensor."""
    try:
        import torch
        from PIL import Image
    except ImportError:
        return None

    img = Image.fromarray(crop).resize((size, size))
    arr = np.array(img, dtype=np.uint8)
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # 1xCxHxW


def _compute_fid_score(
    real_patches: List[np.ndarray],
    synth_patches: List[np.ndarray],
    device: str = "cpu",
) -> Dict[str, Any]:
    """Compute FID between real and synthetic defect patches."""
    try:
        from torchmetrics.image.fid import FrechetInceptionDistance
    except ImportError:
        logger.error("torchmetrics required: pip install torchmetrics[image]")
        return {"fid": None, "error": "torchmetrics_missing"}

    if not real_patches or not synth_patches:
        return {"fid": None, "error": "empty_patches"}

    fid_metric = FrechetInceptionDistance(feature=2048).to(device)

    for crop in real_patches:
        t = _resize_to_tensor(crop)
        if t is not None:
            fid_metric.update(t.to(device), real=True)

    for crop in synth_patches:
        t = _resize_to_tensor(crop)
        if t is not None:
            fid_metric.update(t.to(device), real=False)

    fid_val = float(fid_metric.compute())
    return {
        "fid": round(fid_val, 4),
        "n_real_patches": len(real_patches),
        "n_synth_patches": len(synth_patches),
        "fid_unstable": len(real_patches) < 50,
    }


def _load_source_roi_crops(annotations_path: str) -> List[np.ndarray]:
    """Load source_roi crops listed in annotations.json."""
    try:
        from PIL import Image
    except ImportError:
        return []

    if not Path(annotations_path).exists():
        logger.warning("annotations.json not found: %s", annotations_path)
        return []

    annotations: List[Dict[str, Any]] = load_json(annotations_path)
    crops: List[np.ndarray] = []

    for ann in annotations:
        src = ann.get("source_roi", "")
        if not src or not Path(src).exists():
            continue
        try:
            crops.append(np.array(Image.open(src).convert("RGB")))
        except Exception as e:
            logger.warning("Failed to load source_roi %s: %s", src, e)

    logger.info("Loaded %d source_roi crops from %s", len(crops), annotations_path)
    return crops


def _run_fid_mode(
    real_data_dir: str,
    aroma_synthetic_dir: str,
    casda_synthetic_dir: str,
    random_synthetic_dir: str,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Compute FID for all 3 methods vs real defect patches."""
    real_patch_data = load_severstal_defect_patches(real_data_dir)
    real_patches = [crop for crop, _ in real_patch_data]

    if not real_patches:
        logger.warning("No real patches loaded — FID mode skipped")
        return {"error": "no_real_patches"}

    result: Dict[str, Any] = {
        "n_real_patches": len(real_patches),
        "fid_unstable": len(real_patches) < 50,
    }

    methods = {
        "aroma":  aroma_synthetic_dir,
        "casda":  casda_synthetic_dir,
        "random": random_synthetic_dir,
    }

    for method, synth_dir in methods.items():
        ann_path = str(Path(synth_dir) / "annotations.json")
        synth_patches = _load_source_roi_crops(ann_path)
        fid_result = _compute_fid_score(real_patches, synth_patches, device=device)
        result[method] = fid_result
        logger.info("  FID %s: %.4f (n_synth=%d)",
                    method, fid_result.get("fid") or float("nan"),
                    fid_result.get("n_synth_patches", 0))

    return result


# ---------------------------------------------------------------------------
# Mode: AD (PaDiM, anomalib >= 1.0)
# ---------------------------------------------------------------------------

def _check_gpu() -> str:
    """Return 'cuda' if GPU available, else exit with error."""
    try:
        import torch
        if not torch.cuda.is_available():
            logger.error("GPU not available. AD mode requires CUDA.")
            sys.exit(1)
        return "cuda"
    except ImportError:
        logger.error("PyTorch not installed. AD mode requires torch + CUDA.")
        sys.exit(1)


def _prepare_ad_dataset(
    train_normal_paths: List[str],
    synth_image_paths: List[str],
    test_good_paths: List[str],
    test_defect_paths: List[str],
    tmpdir: str,
) -> str:
    """Create anomalib Folder-compatible dataset directory in tmpdir.

    Structure:
        tmpdir/
          train/good/   (symlinks: train_normal + synthetic images)
          test/good/    (symlinks: test_good)
          test/defect/  (symlinks: test_defect)

    Returns path to tmpdir.
    """
    use_symlink = os.name != "nt"  # symlinks on Linux/Colab, copies on Windows

    def _link_or_copy(src: str, dst: str) -> None:
        Path(dst).parent.mkdir(parents=True, exist_ok=True)
        if use_symlink:
            if not Path(dst).exists():
                os.symlink(src, dst)
        else:
            shutil.copy2(src, dst)

    train_good_dir = Path(tmpdir) / "train" / "good"
    test_good_dir  = Path(tmpdir) / "test"  / "good"
    test_def_dir   = Path(tmpdir) / "test"  / "defect"

    for d in (train_good_dir, test_good_dir, test_def_dir):
        d.mkdir(parents=True, exist_ok=True)

    for p in train_normal_paths + synth_image_paths:
        name = Path(p).name
        _link_or_copy(p, str(train_good_dir / name))

    for p in test_good_paths:
        name = Path(p).name
        _link_or_copy(p, str(test_good_dir / name))

    for p in test_defect_paths:
        name = Path(p).name
        _link_or_copy(p, str(test_def_dir / name))

    return tmpdir


def _run_padim_condition(
    train_normal_paths: List[str],
    synth_image_paths: List[str],
    test_good_paths: List[str],
    test_defect_paths: List[str],
    checkpoint_dir: str,
    seed: int = 42,
    image_size: int = 256,
) -> Dict[str, Any]:
    """Train PaDiM on normal+synthetic, evaluate on real test set.

    Returns image_auroc and pixel_auroc (pixel from real defect masks only).
    """
    try:
        from anomalib.models import Padim
        from anomalib.data import Folder
        from anomalib.engine import Engine
        from anomalib import TaskType
    except ImportError as e:
        logger.error("anomalib not installed: %s  Run: pip install anomalib", e)
        return {"error": "anomalib_missing"}

    try:
        import pytorch_lightning as pl
        pl.seed_everything(seed, workers=True)
    except Exception:
        pass

    with tempfile.TemporaryDirectory() as tmpdir:
        _prepare_ad_dataset(
            train_normal_paths=train_normal_paths,
            synth_image_paths=synth_image_paths,
            test_good_paths=test_good_paths,
            test_defect_paths=test_defect_paths,
            tmpdir=tmpdir,
        )

        datamodule = Folder(
            name="severstal",
            root=tmpdir,
            normal_dir="train/good",
            abnormal_dir="test/defect",
            normal_test_dir="test/good",
            task=TaskType.SEGMENTATION,
            image_size=(image_size, image_size),
            train_batch_size=32,
            eval_batch_size=32,
            num_workers=4,
        )
        model = Padim(backbone="resnet18", layers=["layer1", "layer2", "layer3"])

        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        engine = Engine(default_root_dir=checkpoint_dir, max_epochs=1)

        engine.fit(model=model, datamodule=datamodule)
        test_results = engine.test(model=model, datamodule=datamodule)

    if not test_results:
        return {"error": "no_test_results"}

    metrics = test_results[0] if isinstance(test_results, list) else test_results

    # anomalib 1.x returns keys like 'image_AUROC', 'pixel_AUROC'
    image_auroc = (
        metrics.get("image_AUROC")
        or metrics.get("test_image_AUROC")
        or metrics.get("AUROC")
    )
    pixel_auroc = (
        metrics.get("pixel_AUROC")
        or metrics.get("test_pixel_AUROC")
    )

    return {
        "image_auroc": round(float(image_auroc), 4) if image_auroc is not None else None,
        "pixel_auroc": round(float(pixel_auroc), 4) if pixel_auroc is not None else None,
    }


def _load_synthetic_image_paths(synthetic_dir: str) -> List[str]:
    """Return sorted list of image paths from synthetic_dir/images/."""
    img_dir = Path(synthetic_dir) / "images"
    if not img_dir.exists():
        logger.warning("Synthetic images dir not found: %s", img_dir)
        return []
    return sorted(str(p) for p in img_dir.glob("*.jpg"))


def _run_ad_mode(
    real_data_dir: str,
    aroma_synthetic_dir: str,
    casda_synthetic_dir: str,
    random_synthetic_dir: str,
    output_dir: str,
    seed: int = 42,
    image_size: int = 256,
) -> Dict[str, Any]:
    """Run PaDiM for 4 conditions: baseline / random / casda / aroma."""
    _check_gpu()

    train_normal, test_good, test_defect = load_severstal_image_lists(
        real_data_dir, test_split=0.2, seed=seed,
    )
    test_defect_all = [p for paths in test_defect.values() for p in paths]

    if not train_normal:
        logger.error("No normal training images found in %s", real_data_dir)
        return {"error": "no_train_normal"}

    conditions: Dict[str, List[str]] = {
        "baseline": [],
        "random":   _load_synthetic_image_paths(random_synthetic_dir),
        "casda":    _load_synthetic_image_paths(casda_synthetic_dir),
        "aroma":    _load_synthetic_image_paths(aroma_synthetic_dir),
    }

    result: Dict[str, Any] = {}
    for condition, synth_paths in conditions.items():
        logger.info(
            "AD condition '%s': %d train-normal + %d synthetic | "
            "%d test-good + %d test-defect",
            condition, len(train_normal), len(synth_paths),
            len(test_good), len(test_defect_all),
        )
        ckpt_dir = str(Path(output_dir) / "checkpoints" / condition)
        metrics = _run_padim_condition(
            train_normal_paths=train_normal,
            synth_image_paths=synth_paths,
            test_good_paths=test_good,
            test_defect_paths=test_defect_all,
            checkpoint_dir=ckpt_dir,
            seed=seed,
            image_size=image_size,
        )
        result[condition] = metrics
        logger.info("  %s  image_auroc=%s  pixel_auroc=%s",
                    condition,
                    metrics.get("image_auroc"),
                    metrics.get("pixel_auroc"))

    return result


# ---------------------------------------------------------------------------
# Summary markdown
# ---------------------------------------------------------------------------

_ROI_METRICS = [
    "morphology_coverage", "context_coverage", "rare_pair_coverage", "entropy", "gini",
]
_METHODS = ("aroma", "casda", "random")


def _build_summary(results: Dict[str, Any]) -> str:
    lines: List[str] = ["# AROMA Exp 1 — CASDA Baseline Comparison (Severstal)", ""]

    # ROI quality table
    if "roi_quality" in results:
        roi = results["roi_quality"]
        lines += [
            "## ROI 모델링 품질 (공통 roi_candidates.json 기준)",
            "",
            "| 방법 | morph_cov | ctx_cov | rare_pair_cov | entropy | gini | n_sel |",
            "|-----|-----------|---------|---------------|---------|------|-------|",
        ]
        for method in _METHODS:
            if method not in roi:
                continue
            m = roi[method]
            lines.append(
                f"| {method.upper()} "
                f"| {m['morphology_coverage']:.4f} "
                f"| {m['context_coverage']:.4f} "
                f"| {m['rare_pair_coverage']:.4f} "
                f"| {m['entropy']:.4f} "
                f"| {m['gini']:.4f} "
                f"| {m['n_selected']} |"
            )
        # Delta vs random
        if "aroma" in roi and "random" in roi:
            lines += ["", "**AROMA Δ over Random:**"]
            for k in _ROI_METRICS:
                delta = roi["aroma"][k] - roi["random"][k]
                arrow = "↑" if (k != "gini" and delta > 0) or (k == "gini" and delta < 0) else "↓"
                lines.append(f"  {k}: {delta:+.4f} {arrow}")
        lines.append("")

    # FID table
    if "fid" in results:
        fid = results["fid"]
        lines += [
            "## FID (실제 결함 패치 vs 합성 결함 패치)",
            "",
            f"n_real_patches: {fid.get('n_real_patches', '?')}  "
            f"{'⚠ fid_unstable (n<50)' if fid.get('fid_unstable') else ''}",
            "",
            "| 방법 | FID | n_synth_patches |",
            "|-----|-----|----------------|",
        ]
        for method in _METHODS:
            m = fid.get(method, {})
            fid_val = m.get("fid")
            n_synth = m.get("n_synth_patches", "?")
            lines.append(f"| {method.upper()} | {fid_val if fid_val is not None else 'N/A'} | {n_synth} |")
        lines.append("")

    # AD table
    if "ad" in results:
        ad = results["ad"]
        lines += [
            "## Anomaly Detection (PaDiM, ResNet-18)",
            "Train = Normal + Synthetic | Test = Real Only | Pixel AUROC = Real defects only",
            "",
            "| 조건 | Image AUROC | Pixel AUROC |",
            "|-----|-------------|-------------|",
        ]
        for cond in ("baseline", "random", "casda", "aroma"):
            m = ad.get(cond, {})
            img_a = m.get("image_auroc")
            pix_a = m.get("pixel_auroc")
            lines.append(
                f"| {cond.upper()} "
                f"| {img_a if img_a is not None else 'N/A'} "
                f"| {pix_a if pix_a is not None else 'N/A'} |"
            )
        if "aroma" in ad and "casda" in ad:
            a_img = ad["aroma"].get("image_auroc")
            c_img = ad["casda"].get("image_auroc")
            if a_img is not None and c_img is not None:
                lines += [
                    "",
                    f"**AROMA vs CASDA Image AUROC Δ: {a_img - c_img:+.4f}**",
                ]
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run(
    mode: str,
    aroma_roi_dir: str,
    casda_roi_dir: str,
    random_roi_dir: str,
    aroma_synthetic_dir: str,
    casda_synthetic_dir: str,
    random_synthetic_dir: str,
    real_data_dir: str,
    output_dir: str,
    seed: int = 42,
    device: str = "cpu",
    image_size: int = 256,
) -> Dict[str, Any]:
    results: Dict[str, Any] = {}

    if mode in ("roi", "all"):
        logger.info("=== Mode: ROI quality ===")
        results["roi_quality"] = _run_roi_mode(
            aroma_roi_dir=aroma_roi_dir,
            casda_roi_dir=casda_roi_dir,
            random_roi_dir=random_roi_dir,
        )

    if mode in ("fid", "all"):
        logger.info("=== Mode: FID ===")
        results["fid"] = _run_fid_mode(
            real_data_dir=real_data_dir,
            aroma_synthetic_dir=aroma_synthetic_dir,
            casda_synthetic_dir=casda_synthetic_dir,
            random_synthetic_dir=random_synthetic_dir,
            device=device,
        )

    if mode in ("ad", "all"):
        logger.info("=== Mode: AD (PaDiM) ===")
        results["ad"] = _run_ad_mode(
            real_data_dir=real_data_dir,
            aroma_synthetic_dir=aroma_synthetic_dir,
            casda_synthetic_dir=casda_synthetic_dir,
            random_synthetic_dir=random_synthetic_dir,
            output_dir=output_dir,
            seed=seed,
            image_size=image_size,
        )

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    save_json({"severstal": results}, str(out / "exp1_results.json"))
    summary_md = _build_summary(results)
    (out / "exp1_summary.md").write_text(summary_md, encoding="utf-8")

    logger.info("Saved exp1_results.json + exp1_summary.md → %s", out)
    return {"status": "ok", "results": results}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="AROMA Exp 1 — CASDA Baseline Comparison (Severstal)"
    )
    p.add_argument("--mode", choices=["roi", "fid", "ad", "all"], default="all",
                   help="Evaluation mode (default: all)")
    p.add_argument("--aroma_roi_dir",        required=True,
                   help="AROMA roi 출력 디렉터리 (roi_candidates.json + roi_selected.json)")
    p.add_argument("--casda_roi_dir",        required=True,
                   help="CASDA roi_selected.json 디렉터리 (casda_roi_adapter 출력)")
    p.add_argument("--random_roi_dir",       required=True,
                   help="Random roi_selected.json 디렉터리 (generate_random 출력)")
    p.add_argument("--aroma_synthetic_dir",  required=True,
                   help="AROMA synthetic 이미지 디렉터리")
    p.add_argument("--casda_synthetic_dir",  required=True,
                   help="CASDA synthetic 이미지 디렉터리")
    p.add_argument("--random_synthetic_dir", required=True,
                   help="Random synthetic 이미지 디렉터리")
    p.add_argument("--real_data_dir",        required=True,
                   help="Severstal 데이터셋 루트 (train.csv + train_images/)")
    p.add_argument("--output_dir",           required=True,
                   help="exp1_results.json + exp1_summary.md 저장 디렉터리")
    p.add_argument("--seed",      type=int, default=42)
    p.add_argument("--device",    default="cpu", help="FID device (cpu or cuda)")
    p.add_argument("--image_size", type=int, default=256,
                   help="Image resize for PaDiM (default 256)")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = _parse_args(argv)
    result = run(
        mode=args.mode,
        aroma_roi_dir=args.aroma_roi_dir,
        casda_roi_dir=args.casda_roi_dir,
        random_roi_dir=args.random_roi_dir,
        aroma_synthetic_dir=args.aroma_synthetic_dir,
        casda_synthetic_dir=args.casda_synthetic_dir,
        random_synthetic_dir=args.random_synthetic_dir,
        real_data_dir=args.real_data_dir,
        output_dir=args.output_dir,
        seed=args.seed,
        device=args.device,
        image_size=args.image_size,
    )
    if result.get("status") != "ok":
        sys.exit(1)


if __name__ == "__main__":
    main()
