#!/usr/bin/env python3
"""
AROMA Exp 3 (논문 Exp 2) — Cross-Domain 생성 품질 평가 (FID + PaDiM)

Random ROI vs AROMA ROI, copy-paste synthesis 동일.
Datasets: isp_LSM_1, mvtec_cable, visa_cashew, visa_pcb (pcb4)

Copy-paste limitation:
  Copy-paste synthesis preserves the original defect appearance and therefore
  cannot generate novel defect morphologies. The objective of this study is not
  to improve the synthesis model itself, but to evaluate whether adaptive ROI
  modeling improves the quality of synthesized training data.

Usage (Colab):
    !python $AROMA_SCRIPTS/experiments/exp3_generation_quality.py \\
        --mode fid \\
        --random_synthetic_dir $AROMA_OUT/synthetic_random \\
        --aroma_synthetic_dir  $AROMA_OUT/synthetic \\
        --real_data_dir        $AROMA_DATA \\
        --dataset_keys         isp_LSM_1 mvtec_cable visa_cashew visa_pcb \\
        --output_dir           $AROMA_OUT/exp3 \\
        --num_workers          4
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("aroma.exp3")

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
# Dataset registry — per-dataset path resolution
# ---------------------------------------------------------------------------
# Real data directory layout (from docs/data_dir/):
#   isp_LSM_1   : {root}/isp/unsupervised/LSM_1/
#   mvtec_cable : {root}/mvtec/cable/
#   visa_cashew : {root}/visa/cashew/Data/
#   visa_pcb    : {root}/visa/pcb4/Data/      (pcb4 only, by design)

def _glob_images(directory: str) -> List[str]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
    d = Path(directory)
    if not d.exists():
        return []
    return sorted(str(p) for p in d.iterdir() if p.suffix.lower() in exts)


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
# Real defect patch loader (GT mask → bbox crop)
# ---------------------------------------------------------------------------

def _mask_to_bbox(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    rows = np.any(mask > 0, axis=1)
    cols = np.any(mask > 0, axis=0)
    if not rows.any():
        return None
    y_idx = np.where(rows)[0]; y1, y2 = int(y_idx[0]), int(y_idx[-1])
    x_idx = np.where(cols)[0]; x1, x2 = int(x_idx[0]), int(x_idx[-1])
    if y2 <= y1 or x2 <= x1:
        return None
    return x1, y1, x2 + 1, y2 + 1


def _load_real_defect_patches(
    test_defect_paths: List[str],
    mask_map: Dict[str, str],
    num_workers: int = 4,
) -> List[np.ndarray]:
    """Crop bbox patch from each defect image using its GT mask. Parallel I/O."""
    try:
        from PIL import Image
    except ImportError:
        logger.error("PIL not available")
        return []

    def _load_one(img_p: str) -> Tuple[Optional[np.ndarray], bool]:
        mask_p = mask_map.get(img_p)
        if not mask_p:
            return None, True
        try:
            img  = np.array(Image.open(img_p).convert("RGB"))
            mask = np.array(Image.open(mask_p).convert("L"))
            bbox = _mask_to_bbox(mask)
            if bbox is None:
                return None, True
            x1, y1, x2, y2 = bbox
            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                return None, True
            return crop, False
        except Exception as exc:
            logger.warning("Patch load failed %s: %s", img_p, exc)
            return None, True

    slot: List[Optional[Tuple[Optional[np.ndarray], bool]]] = [None] * len(test_defect_paths)
    future_to_idx: Dict[Any, int] = {}

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        for idx, img_p in enumerate(test_defect_paths):
            future_to_idx[executor.submit(_load_one, img_p)] = idx
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                slot[idx] = future.result()
            except Exception as exc:
                logger.warning("Patch future failed idx=%d: %s", idx, exc)
                slot[idx] = (None, True)

    patches: List[np.ndarray] = []
    n_skip = 0
    for item in slot:
        if item is None:
            n_skip += 1
            continue
        crop, is_skip = item
        if is_skip or crop is None:
            n_skip += 1
        else:
            patches.append(crop)

    if n_skip:
        logger.warning("Skipped %d/%d defect images (no mask or crop fail)",
                       n_skip, len(test_defect_paths))
    logger.info("Loaded %d real defect patches", len(patches))
    return patches


# ---------------------------------------------------------------------------
# FID helpers
# ---------------------------------------------------------------------------

def _resize_to_tensor(crop: np.ndarray, size: int = 299):
    try:
        import torch
        from PIL import Image
    except ImportError:
        return None
    img = Image.fromarray(crop).resize((size, size))
    arr = np.array(img, dtype=np.uint8)
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


def _compute_fid_score(
    real_patches: List[np.ndarray],
    synth_patches: List[np.ndarray],
    device: str = "cpu",
    num_workers: int = 4,
) -> Dict[str, Any]:
    try:
        from torchmetrics.image.fid import FrechetInceptionDistance
    except ImportError:
        logger.error("torchmetrics required: pip install torchmetrics[image]")
        return {"fid": None, "error": "torchmetrics_missing"}

    if not real_patches or not synth_patches:
        logger.warning("Empty patches — FID skipped")
        return {"fid": None, "error": "empty_patches"}

    fid_metric = FrechetInceptionDistance(feature=2048).to(device)

    # resize 병렬 처리; FID metric update는 메인 스레드 순차 (thread-unsafe)
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        real_tensors = list(executor.map(_resize_to_tensor, real_patches))
    for t in real_tensors:
        if t is not None:
            fid_metric.update(t.to(device), real=True)

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        synth_tensors = list(executor.map(_resize_to_tensor, synth_patches))
    for t in synth_tensors:
        if t is not None:
            fid_metric.update(t.to(device), real=False)

    fid_val = float(fid_metric.compute())
    return {
        "fid": round(fid_val, 4),
        "n_real_patches": len(real_patches),
        "n_synth_patches": len(synth_patches),
        "fid_unstable": len(real_patches) < 50,
    }


def _load_source_roi_crops(
    annotations_path: str,
    num_workers: int = 4,
) -> List[np.ndarray]:
    try:
        from PIL import Image
    except ImportError:
        return []
    if not Path(annotations_path).exists():
        logger.warning("annotations.json not found: %s", annotations_path)
        return []
    annotations: List[Dict[str, Any]] = load_json(annotations_path)

    def _load_one(ann: Dict[str, Any]) -> Tuple[Optional[np.ndarray], bool]:
        src = ann.get("source_roi", "")
        is_fallback = False
        if not src or not Path(src).exists():
            src = ann.get("image_path", "")
            if not src or not Path(src).exists():
                return None, False
            is_fallback = True
        try:
            return np.array(Image.open(src).convert("RGB")), is_fallback
        except Exception as exc:
            logger.warning("source_roi load failed %s: %s", src, exc)
            return None, False

    slot: List[Optional[Tuple[Optional[np.ndarray], bool]]] = [None] * len(annotations)
    future_to_idx: Dict[Any, int] = {}

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        for idx, ann in enumerate(annotations):
            future_to_idx[executor.submit(_load_one, ann)] = idx
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                slot[idx] = future.result()
            except Exception as exc:
                logger.warning("source_roi future failed idx=%d: %s", idx, exc)
                slot[idx] = (None, False)

    crops: List[np.ndarray] = []
    n_fallback = 0
    for item in slot:
        if item is None:
            continue
        arr, is_fallback = item
        if arr is not None:
            crops.append(arr)
            if is_fallback:
                n_fallback += 1

    if n_fallback:
        logger.warning(
            "source_roi paths missing (%d/%d) — used synthetic image_path as fallback. "
            "Re-run generate_defects.py to fix (staging path bug corrected).",
            n_fallback, len(annotations),
        )
    logger.info("Loaded %d source_roi crops from %s", len(crops), annotations_path)
    return crops


# ---------------------------------------------------------------------------
# GPU check
# ---------------------------------------------------------------------------

def _check_gpu() -> str:
    try:
        import torch
        if not torch.cuda.is_available():
            logger.error("GPU not available. AD mode requires CUDA.")
            sys.exit(1)
        return "cuda"
    except ImportError:
        logger.error("PyTorch not installed. AD mode requires torch + CUDA.")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Synthetic image loader
# ---------------------------------------------------------------------------

def _load_synthetic_image_paths(synth_root: str, dataset_key: str) -> List[str]:
    """Load synthetic image paths from {synth_root}/{dataset_key}/images/."""
    img_dir = Path(synth_root) / dataset_key / "images"
    if not img_dir.exists():
        logger.warning("Synthetic images not found: %s", img_dir)
        return []
    exts = {".jpg", ".jpeg", ".png"}
    paths = sorted(str(p) for p in img_dir.iterdir() if p.suffix.lower() in exts)
    logger.info("Synthetic images: %d from %s", len(paths), img_dir)
    return paths


# ---------------------------------------------------------------------------
# AD helpers
# ---------------------------------------------------------------------------

def _prepare_ad_dataset_with_masks(
    train_normal_paths: List[str],
    synth_image_paths: List[str],
    test_good_paths: List[str],
    test_defect_paths: List[str],
    mask_map: Dict[str, str],
    tmpdir: str,
    num_workers: int = 4,
) -> Tuple[str, Optional[str]]:
    """
    Build anomalib Folder-compatible layout in tmpdir.

        train/good/   (real normal + synthetic)
        test/good/    (real normal held-out)
        test/defect/  (real defect — synthetic NEVER included)
        test/masks/   (GT masks filename-matched to test/defect; only if available)

    Returns (tmpdir, mask_dir_abs_or_None).
    """
    use_symlink = os.name != "nt"

    def _link_or_copy(src: str, dst: str) -> None:
        if use_symlink:
            if not Path(dst).exists():
                os.symlink(src, dst)
        else:
            shutil.copy2(src, dst)

    train_dir = Path(tmpdir) / "train" / "good"
    tgood_dir = Path(tmpdir) / "test"  / "good"
    tdef_dir  = Path(tmpdir) / "test"  / "defect"
    tmask_dir = Path(tmpdir) / "test"  / "masks"

    # 디렉터리 생성은 병렬화 전 완료 (race condition 방지)
    for d in (train_dir, tgood_dir, tdef_dir):
        d.mkdir(parents=True, exist_ok=True)

    has_masks = any(p in mask_map for p in test_defect_paths)
    if has_masks:
        tmask_dir.mkdir(parents=True, exist_ok=True)

    # train_normal + synth + test_good 일괄 병렬 처리
    bulk_tasks: List[Tuple[str, str]] = []
    for i, p in enumerate(train_normal_paths):
        bulk_tasks.append((p, str(train_dir / f"real_{i:05d}{Path(p).suffix}")))
    for i, p in enumerate(synth_image_paths):
        bulk_tasks.append((p, str(train_dir / f"syn_{i:05d}{Path(p).suffix}")))
    for i, p in enumerate(test_good_paths):
        bulk_tasks.append((p, str(tgood_dir / f"good_{i:05d}{Path(p).suffix}")))

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        bulk_futures = [executor.submit(_link_or_copy, src, dst) for src, dst in bulk_tasks]
        for f in as_completed(bulk_futures):
            f.result()

    # test_defect: 이미지+마스크 쌍 병렬 처리
    def _stage_defect(i: int, p: str) -> int:
        dst_name = f"def_{i:05d}{Path(p).suffix}"
        _link_or_copy(p, str(tdef_dir / dst_name))
        if p in mask_map:
            _link_or_copy(mask_map[p], str(tmask_dir / f"def_{i:05d}.png"))
            return 1
        return 0

    mask_results: List[int] = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        defect_futures = {
            executor.submit(_stage_defect, i, p): i
            for i, p in enumerate(test_defect_paths)
        }
        for f in as_completed(defect_futures):
            mask_results.append(f.result())

    staged_mask_count = sum(mask_results)

    mask_dir: Optional[str] = None
    if staged_mask_count > 0:
        n_defect = len(test_defect_paths)
        if staged_mask_count < n_defect:
            logger.warning(
                "Mask coverage %d/%d — pixel AUROC may be incomplete",
                staged_mask_count, n_defect,
            )
        mask_dir = str(tmask_dir)

    return tmpdir, mask_dir


def _run_padim_condition(
    train_normal_paths: List[str],
    synth_image_paths: List[str],
    test_good_paths: List[str],
    test_defect_paths: List[str],
    mask_map: Dict[str, str],
    checkpoint_dir: str,
    seed: int = 42,
    image_size: int = 256,
    num_workers: int = 4,
) -> Dict[str, Any]:
    """Train PaDiM on normal+synthetic, evaluate on real test set."""
    try:
        from anomalib.models import Padim
        from anomalib.data import Folder
        from anomalib.engine import Engine
        from anomalib import TaskType
    except ImportError as exc:
        logger.error("anomalib not installed: %s  Run: pip install anomalib", exc)
        return {"error": "anomalib_missing"}

    try:
        import pytorch_lightning as pl
        pl.seed_everything(seed, workers=True)
    except Exception:
        try:
            from anomalib.utils.seed import seed_everything
            seed_everything(seed)
        except Exception:
            pass

    with tempfile.TemporaryDirectory() as tmpdir:
        _, mask_dir = _prepare_ad_dataset_with_masks(
            train_normal_paths=train_normal_paths,
            synth_image_paths=synth_image_paths,
            test_good_paths=test_good_paths,
            test_defect_paths=test_defect_paths,
            mask_map=mask_map,
            tmpdir=tmpdir,
            num_workers=num_workers,
        )

        task = TaskType.SEGMENTATION if mask_dir else TaskType.CLASSIFICATION

        folder_kwargs: Dict[str, Any] = dict(
            name="exp3",
            root=tmpdir,
            normal_dir="train/good",
            abnormal_dir="test/defect",
            normal_test_dir="test/good",
            task=task,
            image_size=(image_size, image_size),
            train_batch_size=32,
            eval_batch_size=32,
            num_workers=4,
        )
        if mask_dir:
            folder_kwargs["mask_dir"] = "test/masks"

        datamodule = Folder(**folder_kwargs)
        model = Padim(backbone="resnet18", layers=["layer1", "layer2", "layer3"])

        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        engine = Engine(default_root_dir=checkpoint_dir, max_epochs=1)

        engine.fit(model=model, datamodule=datamodule)
        test_results = engine.test(model=model, datamodule=datamodule)

    if not test_results:
        return {"error": "no_test_results"}

    metrics = test_results[0] if isinstance(test_results, list) else test_results

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


# ---------------------------------------------------------------------------
# FID mode
# ---------------------------------------------------------------------------

def _run_fid_mode(
    dataset_keys: List[str],
    random_synthetic_dir: str,
    aroma_synthetic_dir: str,
    real_data_dir: str,
    device: str = "cpu",
    seed: int = 42,
    num_workers: int = 4,
) -> Dict[str, Any]:

    def _fid_one_dataset(ds: str) -> Tuple[str, Optional[Dict[str, Any]]]:
        logger.info("=== FID: %s ===", ds)
        lists = _get_image_lists(ds, real_data_dir, seed=seed)
        if lists is None:
            logger.warning("FID skip %s — dataset not found", ds)
            return ds, None

        real_patches = _load_real_defect_patches(
            lists["test_defect"], lists["mask_map"], num_workers=num_workers
        )
        if not real_patches:
            logger.warning("FID skip %s — 0 real patches (check masks)", ds)
            return ds, {"fid": {
                "random": {"fid": None, "error": "no_real_patches"},
                "aroma":  {"fid": None, "error": "no_real_patches"},
            }}

        fid_ds: Dict[str, Any] = {"n_real_patches": len(real_patches)}

        for method in ("random", "aroma"):
            synth_base = random_synthetic_dir if method == "random" else aroma_synthetic_dir
            ann_path = str(Path(synth_base) / ds / "annotations.json")
            synth_patches = _load_source_roi_crops(ann_path, num_workers=num_workers)
            if not synth_patches:
                logger.warning("FID %s %s — 0 synthetic patches", ds, method)
                fid_ds[method] = {"fid": None, "error": "no_synth_patches"}
                continue
            fid_ds[method] = _compute_fid_score(
                real_patches, synth_patches, device=device, num_workers=num_workers
            )
            logger.info("  %s FID=%s  unstable=%s",
                        method, fid_ds[method].get("fid"), fid_ds[method].get("fid_unstable"))

        logger.info("=== FID done: %s ===", ds)
        return ds, {"fid": fid_ds}

    results: Dict[str, Any] = {}

    if num_workers > 1:
        # 데이터셋 수준 병렬: 각 thread가 독립적인 FID metric 인스턴스 생성
        with ThreadPoolExecutor(max_workers=min(num_workers, len(dataset_keys))) as executor:
            futures = {executor.submit(_fid_one_dataset, ds): ds for ds in dataset_keys}
            for future in as_completed(futures):
                ds, value = future.result()
                if value is not None:
                    results[ds] = value
    else:
        for ds in dataset_keys:
            ds_key, value = _fid_one_dataset(ds)
            if value is not None:
                results[ds_key] = value

    return results


# ---------------------------------------------------------------------------
# AD mode
# ---------------------------------------------------------------------------

def _run_ad_mode(
    dataset_keys: List[str],
    random_synthetic_dir: str,
    aroma_synthetic_dir: str,
    real_data_dir: str,
    output_dir: str,
    seed: int = 42,
    image_size: int = 256,
    num_workers: int = 4,
) -> Dict[str, Any]:
    _check_gpu()
    results: Dict[str, Any] = {}

    for ds in dataset_keys:
        logger.info("=== AD: %s ===", ds)
        lists = _get_image_lists(ds, real_data_dir, seed=seed)
        if lists is None:
            logger.warning("AD skip %s — dataset not found", ds)
            continue

        train_normal = lists["train_normal"]
        test_good    = lists["test_good"]
        test_defect  = lists["test_defect"]
        mask_map     = lists["mask_map"]
        ckpt_base    = str(Path(output_dir) / "checkpoints" / ds)

        conditions: Dict[str, List[str]] = {
            "baseline": [],
            "random":   _load_synthetic_image_paths(random_synthetic_dir, ds),
            "aroma":    _load_synthetic_image_paths(aroma_synthetic_dir, ds),
        }

        ad_ds: Dict[str, Any] = {}
        for cond, synth_paths in conditions.items():
            logger.info("  %s / %s  synth=%d", ds, cond, len(synth_paths))
            res = _run_padim_condition(
                train_normal_paths=train_normal,
                synth_image_paths=synth_paths,
                test_good_paths=test_good,
                test_defect_paths=test_defect,
                mask_map=mask_map,
                checkpoint_dir=str(Path(ckpt_base) / cond),
                seed=seed,
                image_size=image_size,
                num_workers=num_workers,
            )
            ad_ds[cond] = res
            logger.info("    image_auroc=%s  pixel_auroc=%s",
                        res.get("image_auroc"), res.get("pixel_auroc"))

        results[ds] = {"ad": ad_ds}

    return results


# ---------------------------------------------------------------------------
# Summary builder
# ---------------------------------------------------------------------------

def _build_summary(results: Dict[str, Any]) -> str:
    lines = [
        "# AROMA Exp 3 — Cross-Domain 생성 품질 평가 (FID + AD)",
        "",
        "비교: Random ROI vs AROMA ROI | copy-paste synthesis 동일",
        "",
    ]

    # FID table
    if any("fid" in v for v in results.values()):
        lines += [
            "## FID (낮을수록 좋음)",
            "",
            "| 데이터셋 | FID Random | FID AROMA | n_real_patches | unstable |",
            "|---------|------------|-----------|----------------|---------|",
        ]
        for ds in sorted(results):
            fd = results[ds].get("fid", {})
            r  = fd.get("random", {})
            a  = fd.get("aroma",  {})
            fid_r = f"{r['fid']:.2f}" if isinstance(r.get("fid"), float) else "N/A"
            fid_a = f"{a['fid']:.2f}" if isinstance(a.get("fid"), float) else "N/A"
            n_real = fd.get("n_real_patches", "?")
            unstable = r.get("fid_unstable") or a.get("fid_unstable")
            lines.append(f"| {ds} | {fid_r} | {fid_a} | {n_real} | {'⚠' if unstable else ''} |")
        lines.append("")

        lines += ["## FID Delta (AROMA − Random)", ""]
        for ds in sorted(results):
            fd = results[ds].get("fid", {})
            r_fid = fd.get("random", {}).get("fid")
            a_fid = fd.get("aroma",  {}).get("fid")
            if isinstance(r_fid, float) and isinstance(a_fid, float):
                delta = round(a_fid - r_fid, 4)
                lines.append(f"- **{ds}**: FID Δ{delta:+.2f}  (음수 = AROMA 우위)")
        lines.append("")

    # AD table
    if any("ad" in v for v in results.values()):
        lines += [
            "## Anomaly Detection AUROC (PaDiM, ResNet-18)",
            "",
            "| 데이터셋 | 조건 | Image AUROC | Pixel AUROC |",
            "|---------|------|-------------|-------------|",
        ]
        for ds in sorted(results):
            ad = results[ds].get("ad", {})
            for cond in ("baseline", "random", "aroma"):
                m  = ad.get(cond, {})
                ia = f"{m['image_auroc']:.4f}" if isinstance(m.get("image_auroc"), float) else "N/A"
                pa = f"{m['pixel_auroc']:.4f}" if isinstance(m.get("pixel_auroc"), float) else "N/A"
                lines.append(f"| {ds} | {cond.upper()} | {ia} | {pa} |")
        lines.append("")

        lines += ["## Delta (AROMA − Random)", ""]
        for ds in sorted(results):
            ad = results[ds].get("ad", {})
            r_ia = ad.get("random", {}).get("image_auroc")
            a_ia = ad.get("aroma",  {}).get("image_auroc")
            r_pa = ad.get("random", {}).get("pixel_auroc")
            a_pa = ad.get("aroma",  {}).get("pixel_auroc")
            if isinstance(r_ia, float) and isinstance(a_ia, float):
                d_ia = round(a_ia - r_ia, 4)
                d_pa = (f"{round(a_pa - r_pa, 4):+.4f}"
                        if isinstance(r_pa, float) and isinstance(a_pa, float) else "N/A")
                lines.append(f"- **{ds}**: image_auroc Δ{d_ia:+.4f}, pixel_auroc Δ{d_pa}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run(
    mode: str,
    random_synthetic_dir: str,
    aroma_synthetic_dir: str,
    real_data_dir: str,
    dataset_keys: List[str],
    output_dir: str,
    seed: int = 42,
    device: str = "cpu",
    image_size: int = 256,
    num_workers: int = 4,
) -> Dict[str, Any]:
    results: Dict[str, Any] = {}

    if mode in ("fid", "all"):
        for ds, val in _run_fid_mode(
            dataset_keys=dataset_keys,
            random_synthetic_dir=random_synthetic_dir,
            aroma_synthetic_dir=aroma_synthetic_dir,
            real_data_dir=real_data_dir,
            device=device,
            seed=seed,
            num_workers=num_workers,
        ).items():
            results.setdefault(ds, {}).update(val)

    if mode in ("ad", "all"):
        for ds, val in _run_ad_mode(
            dataset_keys=dataset_keys,
            random_synthetic_dir=random_synthetic_dir,
            aroma_synthetic_dir=aroma_synthetic_dir,
            real_data_dir=real_data_dir,
            output_dir=output_dir,
            seed=seed,
            image_size=image_size,
            num_workers=num_workers,
        ).items():
            results.setdefault(ds, {}).update(val)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    save_json(results, str(out / "exp3_results.json"))
    (out / "exp3_summary.md").write_text(_build_summary(results), encoding="utf-8")

    logger.info("Saved exp3_results.json + exp3_summary.md → %s  (%d datasets)",
                out, len(results))
    return {"status": "ok", "results": results}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="AROMA Exp 3 — Cross-Domain 생성 품질 평가 (FID + PaDiM)"
    )
    p.add_argument("--mode", choices=["fid", "ad", "all"], default="all",
                   help="Evaluation mode (default: all)")
    p.add_argument("--random_synthetic_dir", required=True,
                   help="Random synthetic 이미지 루트 ({dir}/{dataset_key}/images/)")
    p.add_argument("--aroma_synthetic_dir",  required=True,
                   help="AROMA synthetic 이미지 루트 ({dir}/{dataset_key}/images/)")
    p.add_argument("--real_data_dir",        required=True,
                   help="실제 데이터셋 루트 (isp/, mvtec/, visa/ 포함)")
    p.add_argument("--dataset_keys",         required=True, nargs="+",
                   help="평가할 데이터셋 키 목록")
    p.add_argument("--output_dir",           required=True,
                   help="exp3_results.json + exp3_summary.md 저장 경로")
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--device",      default="cpu",
                   help="FID device (cpu or cuda, default cpu)")
    p.add_argument("--image_size",  type=int, default=256,
                   help="PaDiM image resize (default 256)")
    p.add_argument("--num_workers", type=int, default=4,
                   help="병렬 처리 스레드 수 (이미지 로딩·파일 스테이징·FID 데이터셋, default=4)")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = _parse_args(argv)
    result = run(
        mode=args.mode,
        random_synthetic_dir=args.random_synthetic_dir,
        aroma_synthetic_dir=args.aroma_synthetic_dir,
        real_data_dir=args.real_data_dir,
        dataset_keys=args.dataset_keys,
        output_dir=args.output_dir,
        seed=args.seed,
        device=args.device,
        image_size=args.image_size,
        num_workers=args.num_workers,
    )
    status = result.get("status", "unknown")
    n_ds   = len(result.get("results", {}))
    print(f"[exp3] status={status}  n_datasets={n_ds}")
    if status != "ok":
        sys.exit(1)


if __name__ == "__main__":
    main()
