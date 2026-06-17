#!/usr/bin/env python3
"""
AROMA Exp 4 — Downstream Anomaly Detection 평가

Random ROI vs AROMA ROI, copy-paste synthesis 동일.
4개 AD 모델(PatchCore, SimpleNet, EfficientAD, ReverseDistillation++) × 3조건 × 4데이터셋.
Datasets: isp_LSM_1, mvtec_cable, visa_cashew, visa_pcb (pcb4)

Usage (Colab):
    !python $AROMA_SCRIPTS/experiments/exp4_downstream_ad.py \\
        --model all \\
        --condition all \\
        --dataset_keys isp_LSM_1 mvtec_cable visa_cashew visa_pcb \\
        --random_synthetic_dir $RANDOM_SYNTH_DIR \\
        --aroma_synthetic_dir  $AROMA_SYNTH_DIR \\
        --real_data_dir        $AROMA_DATA \\
        --output_dir           $EXP4_OUT \\
        --seed 42 \\
        --image_size 256 \\
        --num_workers 4
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
logger = logging.getLogger("aroma.exp4")

# Suppress anomalib/Lightning/tqdm verbose output
os.environ.setdefault("TQDM_DISABLE", "1")
logging.getLogger("lightning.pytorch").setLevel(logging.WARNING)
logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
logging.getLogger("anomalib").setLevel(logging.WARNING)

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
# anomalib model imports
# ---------------------------------------------------------------------------

try:
    from anomalib.models import Patchcore, Supersimplenet, EfficientAd, ReverseDistillation  # type: ignore[import]
    _ANOMALIB_AVAILABLE = True
except ImportError:
    _ANOMALIB_AVAILABLE = False
    Patchcore = None  # type: ignore[assignment,misc]
    Supersimplenet = None  # type: ignore[assignment,misc]
    EfficientAd = None  # type: ignore[assignment,misc]
    ReverseDistillation = None  # type: ignore[assignment,misc]

MODEL_REGISTRY: Dict[str, Any] = {}

if _ANOMALIB_AVAILABLE:
    MODEL_REGISTRY = {
        "patchcore":   lambda: Patchcore(backbone="wide_resnet50_2", layers=["layer2", "layer3"]),
        "simplenet":   lambda: Supersimplenet(),
        "efficient_ad": lambda: EfficientAd(model_size="small"),
        "rd_plus_plus": lambda: ReverseDistillation(backbone="resnet18"),
    }

ALL_MODEL_KEYS = ["patchcore", "simplenet", "efficient_ad", "rd_plus_plus"]

# ---------------------------------------------------------------------------
# Dataset registry — per-dataset path resolution
# ---------------------------------------------------------------------------

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
# AD dataset staging helper
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


# ---------------------------------------------------------------------------
# Per-model condition runner
# ---------------------------------------------------------------------------

def _run_model_condition(
    model_name: str,
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
    """Train AD model on normal+synthetic, evaluate on real test set."""
    if model_name not in MODEL_REGISTRY:
        logger.error("Unknown model: %s. Available: %s", model_name, list(MODEL_REGISTRY.keys()))
        return {"error": "unknown_model"}

    if not _ANOMALIB_AVAILABLE:
        logger.error("anomalib not installed. Run: pip install anomalib")
        return {"error": "anomalib_missing"}

    try:
        from anomalib.data import Folder
        from anomalib.engine import Engine
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

    try:
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

            folder_kwargs: Dict[str, Any] = dict(
                name="exp4",
                root=tmpdir,
                normal_dir="train/good",
                abnormal_dir="test/defect",
                normal_test_dir="test/good",
                train_batch_size=32,
                eval_batch_size=32,
                num_workers=4,
            )
            if mask_dir:
                folder_kwargs["mask_dir"] = "test/masks"

            datamodule = Folder(**folder_kwargs)
            model = MODEL_REGISTRY[model_name]()

            Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
            engine = Engine(
                default_root_dir=checkpoint_dir,
                max_epochs=1,
                enable_progress_bar=False,
                enable_model_summary=False,
            )

            engine.fit(model=model, datamodule=datamodule)
            test_results = engine.test(model=model, datamodule=datamodule)

    except Exception as exc:
        logger.error("Model %s failed: %s", model_name, exc)
        return {"error": str(exc)}

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
# Local /tmp image cache — pre-copy real images once per dataset
# ---------------------------------------------------------------------------

def _local_cache_real_images(
    ds: str,
    lists: Dict[str, Any],
    cache_base: str = "/content/tmp/aroma_exp4_cache",
    num_workers: int = 4,
) -> Dict[str, Any]:
    """Copy real images (train_normal, test_good, test_defect, masks) to local /tmp once.

    Returns updated lists with local paths so DataLoader reads from fast local storage
    instead of Google Drive symlinks during training.
    Idempotent: skips files that already exist in cache (resume-safe).
    """
    import time
    cache_ds = Path(cache_base) / ds
    dirs = {
        "train_normal": cache_ds / "train_normal",
        "test_good":    cache_ds / "test_good",
        "test_defect":  cache_ds / "test_defect",
        "masks":        cache_ds / "masks",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    def _copy_if_missing(src: str, dst_dir: Path, prefix: str, idx: int) -> str:
        dst = dst_dir / f"{prefix}_{idx:05d}{Path(src).suffix}"
        if not dst.exists():
            shutil.copy2(src, str(dst))
        return str(dst)

    t0 = time.time()
    total = len(lists["train_normal"]) + len(lists["test_good"]) + len(lists["test_defect"])
    logger.info("Local cache: copying %d real images for %s -> %s", total, ds, cache_ds)

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        tn_futures = [
            executor.submit(_copy_if_missing, p, dirs["train_normal"], "tn", i)
            for i, p in enumerate(lists["train_normal"])
        ]
        tg_futures = [
            executor.submit(_copy_if_missing, p, dirs["test_good"], "tg", i)
            for i, p in enumerate(lists["test_good"])
        ]
        td_futures = [
            executor.submit(_copy_if_missing, p, dirs["test_defect"], "td", i)
            for i, p in enumerate(lists["test_defect"])
        ]
        local_tn = [f.result() for f in tn_futures]
        local_tg = [f.result() for f in tg_futures]
        local_td = [f.result() for f in td_futures]

    # Rebuild mask_map with local test_defect paths as keys
    orig_td = lists["test_defect"]
    local_mask_map: Dict[str, str] = {}
    for i, (orig_p, local_p) in enumerate(zip(orig_td, local_td)):
        if orig_p in lists["mask_map"]:
            mask_src = lists["mask_map"][orig_p]
            mask_dst = dirs["masks"] / f"mask_{i:05d}.png"
            if not mask_dst.exists():
                shutil.copy2(mask_src, str(mask_dst))
            local_mask_map[local_p] = str(mask_dst)

    elapsed = time.time() - t0
    logger.info("Local cache ready for %s: %.1fs  (train=%d test_good=%d test_defect=%d)",
                ds, elapsed, len(local_tn), len(local_tg), len(local_td))

    return {
        **lists,
        "train_normal": local_tn,
        "test_good":    local_tg,
        "test_defect":  local_td,
        "mask_map":     local_mask_map,
    }


# ---------------------------------------------------------------------------
# AD mode — datasets x models x conditions
# ---------------------------------------------------------------------------

def _run_ad_mode(
    model_keys: List[str],
    condition_keys: List[str],
    dataset_keys: List[str],
    random_synthetic_dir: str,
    aroma_synthetic_dir: str,
    real_data_dir: str,
    output_dir: str,
    seed: int = 42,
    image_size: int = 256,
    num_workers: int = 4,
    existing_results: Optional[Dict[str, Any]] = None,
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Iterate datasets x models x conditions.
    Output structure: {dataset: {model: {condition: {image_auroc, pixel_auroc}}}}

    CRITICAL: Test set NEVER includes synthetic images. Only train set augmented.
    """
    _check_gpu()
    existing = existing_results or {}
    results: Dict[str, Any] = {k: v for k, v in existing.items()}

    for ds in dataset_keys:
        logger.info("=== AD dataset: %s ===", ds)
        lists = _get_image_lists(ds, real_data_dir, seed=seed)
        if lists is None:
            logger.warning("AD skip %s — dataset not found", ds)
            continue

        # Pre-stage real images to local /tmp — skipped if already cached (resume-safe)
        if os.name != "nt":
            lists = _local_cache_real_images(ds, lists, num_workers=num_workers)

        train_normal = lists["train_normal"]
        test_good    = lists["test_good"]
        test_defect  = lists["test_defect"]
        mask_map     = lists["mask_map"]
        ckpt_base    = str(Path(output_dir) / "checkpoints" / ds)

        # Define conditions — synth_paths is None means "to be loaded"; [] means empty (baseline)
        CONDITIONS: Dict[str, List[str]] = {
            "baseline": [],
            "random":   _load_synthetic_image_paths(random_synthetic_dir, ds),
            "aroma":    _load_synthetic_image_paths(aroma_synthetic_dir, ds),
        }

        ds_results: Dict[str, Any] = {}

        for model_name in model_keys:
            model_results: Dict[str, Any] = {}

            for cond in condition_keys:
                if cond not in CONDITIONS:
                    logger.warning("Unknown condition %s — skipping", cond)
                    continue

                # Resume: skip only if already completed with valid image_auroc
                _cached = existing.get(ds, {}).get(model_name, {}).get(cond)
                if _cached is not None and _cached.get("image_auroc") is not None:
                    model_results[cond] = _cached
                    logger.info(
                        "  RESUME skip %s / %s / %s  (cached image_auroc=%s)",
                        ds, model_name, cond, _cached.get("image_auroc"),
                    )
                    continue

                synth_paths = CONDITIONS[cond]
                logger.info(
                    "  %s / %s / %s  synth=%d",
                    ds, model_name, cond, len(synth_paths),
                )

                res = _run_model_condition(
                    model_name=model_name,
                    train_normal_paths=train_normal,
                    synth_image_paths=synth_paths,
                    test_good_paths=test_good,
                    test_defect_paths=test_defect,
                    mask_map=mask_map,
                    checkpoint_dir=str(Path(ckpt_base) / model_name / cond),
                    seed=seed,
                    image_size=image_size,
                    num_workers=num_workers,
                )
                model_results[cond] = res
                logger.info(
                    "    image_auroc=%s  pixel_auroc=%s",
                    res.get("image_auroc"), res.get("pixel_auroc"),
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
    Table rows=conditions, cols=models, cell=image_auroc.
    Delta section: AROMA minus Random for each model.
    """
    lines = [
        "# AROMA Exp 4 — Downstream Anomaly Detection 평가",
        "",
        "비교: Baseline (real only) vs Random ROI vs AROMA ROI | copy-paste synthesis 동일",
        "",
    ]

    all_models = ALL_MODEL_KEYS
    all_conds  = ["baseline", "random", "aroma"]

    for ds in sorted(results):
        ds_data = results[ds]
        lines += [
            f"## {ds}",
            "",
            "### Image AUROC",
            "",
        ]

        # Header
        model_header = " | ".join(f"{m:>14}" for m in all_models)
        lines.append(f"| {'조건':<10} | {model_header} |")
        sep_models = " | ".join("-" * 14 for _ in all_models)
        lines.append(f"|{'-' * 12}|{sep_models}|")

        for cond in all_conds:
            cells = []
            for model_name in all_models:
                m = ds_data.get(model_name, {}).get(cond, {})
                val = m.get("image_auroc")
                cells.append(f"{val:.4f}" if isinstance(val, float) else "N/A")
            row = " | ".join(f"{c:>14}" for c in cells)
            lines.append(f"| {cond:<10} | {row} |")
        lines.append("")

        # Delta section: AROMA - Random
        lines.append("### Delta (AROMA − Random) per model")
        lines.append("")
        for model_name in all_models:
            r_ia = ds_data.get(model_name, {}).get("random", {}).get("image_auroc")
            a_ia = ds_data.get(model_name, {}).get("aroma",  {}).get("image_auroc")
            r_pa = ds_data.get(model_name, {}).get("random", {}).get("pixel_auroc")
            a_pa = ds_data.get(model_name, {}).get("aroma",  {}).get("pixel_auroc")
            if isinstance(r_ia, float) and isinstance(a_ia, float):
                d_ia = round(a_ia - r_ia, 4)
                if isinstance(r_pa, float) and isinstance(a_pa, float):
                    d_pa = f"{round(a_pa - r_pa, 4):+.4f}"
                else:
                    d_pa = "N/A"
                lines.append(
                    f"- **{model_name}**: image_auroc {d_ia:+.4f}, pixel_auroc {d_pa}"
                )
            else:
                lines.append(f"- **{model_name}**: N/A")
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
    seed: int = 42,
    image_size: int = 256,
    num_workers: int = 4,
    resume: bool = False,
) -> Dict[str, Any]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    output_path = str(out / "exp4_results.json")

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

    results = _run_ad_mode(
        model_keys=model_keys,
        condition_keys=condition_keys,
        dataset_keys=dataset_keys,
        random_synthetic_dir=random_synthetic_dir,
        aroma_synthetic_dir=aroma_synthetic_dir,
        real_data_dir=real_data_dir,
        output_dir=output_dir,
        seed=seed,
        image_size=image_size,
        num_workers=num_workers,
        existing_results=existing_results,
        output_path=output_path,
    )

    save_json(results, output_path)
    (out / "exp4_summary.md").write_text(_build_summary(results), encoding="utf-8")

    logger.info(
        "Saved exp4_results.json + exp4_summary.md -> %s  (%d datasets)",
        out, len(results),
    )
    return {"status": "ok", "results": results}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="AROMA Exp 4 — Downstream Anomaly Detection 평가 (multi-model)"
    )
    p.add_argument(
        "--model",
        choices=["patchcore", "simplenet", "efficient_ad", "rd_plus_plus", "all"],
        default="all",
        help="평가할 AD 모델 (default: all)",
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
        help="Random synthetic 이미지 루트 ({dir}/{dataset_key}/images/)",
    )
    p.add_argument(
        "--aroma_synthetic_dir",
        required=True,
        help="AROMA synthetic 이미지 루트 ({dir}/{dataset_key}/images/)",
    )
    p.add_argument(
        "--real_data_dir",
        required=True,
        help="실제 데이터셋 루트 (isp/, mvtec/, visa/ 포함)",
    )
    p.add_argument(
        "--output_dir",
        required=True,
        help="exp4_results.json + exp4_summary.md 저장 경로",
    )
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--image_size",  type=int, default=256,
                   help="AD 모델 image resize (default 256)")
    p.add_argument("--num_workers", type=int, default=4,
                   help="병렬 처리 스레드 수 (이미지 로딩·파일 스테이징, default=4)")
    p.add_argument("--resume", action="store_true",
                   help="기존 exp4_results.json에서 완료된 run을 skip하고 재개")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = _parse_args(argv)

    model_keys = ALL_MODEL_KEYS if args.model == "all" else [args.model]
    condition_keys = ["baseline", "random", "aroma"] if args.condition == "all" else [args.condition]

    result = run(
        model_keys=model_keys,
        condition_keys=condition_keys,
        dataset_keys=args.dataset_keys,
        random_synthetic_dir=args.random_synthetic_dir,
        aroma_synthetic_dir=args.aroma_synthetic_dir,
        real_data_dir=args.real_data_dir,
        output_dir=args.output_dir,
        seed=args.seed,
        image_size=args.image_size,
        num_workers=args.num_workers,
        resume=args.resume,
    )
    status = result.get("status", "unknown")
    n_ds   = len(result.get("results", {}))
    print(f"[exp4] status={status}  n_datasets={n_ds}")
    if status != "ok":
        sys.exit(1)


if __name__ == "__main__":
    main()
