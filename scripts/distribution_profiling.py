#!/usr/bin/env python3
"""
AROMA Stage 0 — Distribution Profiling

Analyzes defect morphology and background context distributions from a
dataset_config.json entry to derive data-driven thresholds, compatibility
scores, and synthesis priorities.

Usage (Colab):
    !python $SCRIPTS/distribution_profiling.py \\
        --dataset_config $DATASET_CONFIG \\
        --dataset_key    visa_cashew \\
        --output_dir     $PROFILING_DIR \\
        --num_workers    8

Outputs (written to --output_dir):
    morphology_features.csv    defect instance morphology features
    context_features.csv       64px patch context features
    distribution_analysis.json per-feature distribution shape + policy
    morphology_clusters.json   GMM/percentile cluster definitions
    compatibility_matrix.json  P(context_cell | cluster)
    deficit_analysis.json      deficit + target_synthetic per cluster
    threshold_policies.json    threshold derivation policies
    recommended_config.yaml    ready-to-use config for Stage 1b/3/6
    analysis_report.md         human-readable summary
    figures/*.png              histograms, compatibility heatmap, deficit bars
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
import yaml
from skimage.feature import local_binary_pattern
from skimage.measure import label, regionprops
from tqdm.auto import tqdm

# Project root on sys.path so utils/ and stage*.py are importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.defect_characterization import DefectCharacterizer
from utils.io import load_json, safe_imread, save_json
from utils.parallel import resolve_workers, run_parallel
from stage1b_seed_characterization import extract_seed_mask

try:
    from sklearn.mixture import GaussianMixture
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    warnings.warn("scikit-learn not found. GMM policy falls back to Percentile.")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MORPH_FEATURES = [
    "linearity", "solidity", "extent", "aspect_ratio", "eccentricity", "circularity",
]
CONTEXT_FEATURES = [
    "local_variance", "edge_density", "texture_entropy",
    "frequency_energy", "orientation_consistency",
]
GRID_SIZE = 64
N_CONTEXT_BINS = 3          # P33 / P66 → bins 0, 1, 2
MAX_HISTOGRAM_BINS = 50
VALLEY_PROMINENCE_RATIO = 0.1

_MORPH_CSV_FIELDS = [
    "image_id", "image_path", "defect_type", "domain", "mask_source",
    "linearity", "solidity", "extent", "aspect_ratio",
    "eccentricity", "circularity", "area",
]
_CONTEXT_CSV_FIELDS = [
    "image_id", "image_type", "patch_xy",
    "local_variance", "edge_density", "texture_entropy",
    "frequency_energy", "orientation_consistency",
]

# ---------------------------------------------------------------------------
# Helpers: Mask discovery
# ---------------------------------------------------------------------------


def _resolve_seed_entries(entry: dict) -> List[Tuple[str, Path]]:
    """Normalize seed_dir (str) / seed_dirs (list) → [(defect_type, Path)]."""
    if "seed_dirs" in entry:
        return [(Path(d).name, Path(d)) for d in entry["seed_dirs"]]
    if "seed_dir" in entry:
        d = entry["seed_dir"]
        return [(Path(d).name, Path(d))]
    return []


def _find_mask_path(domain: str, image_path: Path, defect_type: str) -> Optional[Path]:
    """Return ground-truth mask path for the image, or None → SAM/Otsu fallback."""
    stem = image_path.stem

    if domain == "mvtec":
        # .../category/test/{defect_type}/img.png
        # → .../category/ground_truth/{defect_type}/img_mask.png
        candidate = (
            image_path.parent.parent.parent
            / "ground_truth"
            / defect_type
            / f"{stem}_mask.png"
        )
        if candidate.exists():
            return candidate

    elif domain == "visa":
        # prepare_visa.py creates masks at: {category}/ground_truth/anomaly/{stem}.png
        # image_path: .../visa/{category}/test/anomaly/{stem}.JPG
        category_dir = image_path.parent.parent.parent  # .../visa/{category}/
        for ext in (".png", ".PNG", ".jpg", ".JPG"):
            candidate = category_dir / "ground_truth" / "anomaly" / f"{stem}{ext}"
            if candidate.exists():
                return candidate

    # isp: no ground-truth masks; other domains also return None → fallback
    return None


# ---------------------------------------------------------------------------
# Feature extraction — standalone functions (pickle-safe for workers)
# ---------------------------------------------------------------------------


def _extract_context_features(patch: np.ndarray) -> Dict[str, float]:
    """Compute 5 physical context descriptors from a uint8 grayscale patch."""
    p = patch.astype(np.float64)

    # 1. Local variance
    local_variance = float(np.var(p))

    # 2. Edge density: mean Sobel magnitude
    sx = cv2.Sobel(p, cv2.CV_64F, 1, 0, ksize=3)
    sy = cv2.Sobel(p, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(sx ** 2 + sy ** 2)
    edge_density = float(mag.mean())

    # 3. Texture entropy: LBP P=8 R=1 uniform, 10-bin histogram
    lbp = local_binary_pattern(patch, P=8, R=1, method="uniform")
    counts_lbp, _ = np.histogram(lbp.ravel(), bins=10)
    p_lbp = counts_lbp / (counts_lbp.sum() + 1e-6) + 1e-12
    texture_entropy = float(-np.sum(p_lbp * np.log2(p_lbp)))

    # 4. Frequency energy: ratio of high-frequency (outer 75% radius) FFT energy
    fft_mag = np.abs(np.fft.fft2(p))
    fft_mag = np.fft.fftshift(fft_mag)
    hf, wf = fft_mag.shape
    cy, cx = hf // 2, wf // 2
    r = min(hf, wf) // 4                   # low-frequency radius threshold
    Y, X = np.ogrid[:hf, :wf]
    lf_mask = (Y - cy) ** 2 + (X - cx) ** 2 <= r ** 2
    frequency_energy = float(fft_mag[~lf_mask].sum() / (fft_mag.sum() + 1e-10))

    # 5. Orientation consistency: gradient direction entropy (low = more consistent)
    angles = np.arctan2(sy, sx)
    hist_a, _ = np.histogram(angles, bins=18, range=(-np.pi, np.pi))
    p_a = hist_a / (hist_a.sum() + 1e-6) + 1e-12
    orientation_consistency = float(-np.sum(p_a * np.log2(p_a)))

    return {
        "local_variance": local_variance,
        "edge_density": edge_density,
        "texture_entropy": texture_entropy,
        "frequency_energy": frequency_energy,
        "orientation_consistency": orientation_consistency,
    }


# ---------------------------------------------------------------------------
# Module-level workers — must be at module level for ProcessPoolExecutor
# ---------------------------------------------------------------------------


def _morph_worker(task: dict) -> Optional[dict]:
    """Extract morphology features from one defect image (with mask)."""
    try:
        image_path = Path(task["image_path"])
        mask_path_str = task.get("mask_path")
        domain = task["domain"]
        defect_type = task["defect_type"]
        checkpoint = task.get("checkpoint")

        img_color = safe_imread(str(image_path))

        mask: Optional[np.ndarray] = None
        mask_source = "ground_truth"

        if mask_path_str and Path(mask_path_str).exists():
            raw = safe_imread(str(mask_path_str), cv2.IMREAD_GRAYSCALE)
            if raw.max() > 0:
                mask = raw

        if mask is None:
            mask_raw, method = extract_seed_mask(img_color, checkpoint)
            mask = mask_raw
            mask_source = f"fallback_{method}"

        if mask is None or mask.max() == 0:
            return None

        char = DefectCharacterizer()
        metrics = char.analyze_defect_region(mask)
        if metrics is None:
            return None

        # eccentricity + circularity not in DefectCharacterizer — add via regionprops
        binary = (mask > 0).astype(np.uint8)
        labeled_arr = label(binary, connectivity=2)
        props = regionprops(labeled_arr)
        if props:
            region = max(props, key=lambda r: r.area)
            eccentricity = float(region.eccentricity)
            perimeter = float(region.perimeter) + 1e-6
            circularity = float(4 * np.pi * region.area / perimeter ** 2)
        else:
            eccentricity = 0.0
            circularity = 0.0

        return {
            "image_id": image_path.stem,
            "image_path": str(image_path),
            "defect_type": defect_type,
            "domain": domain,
            "mask_source": mask_source,
            "linearity": float(metrics["linearity"]),
            "solidity": float(metrics["solidity"]),
            "extent": float(metrics["extent"]),
            "aspect_ratio": float(metrics["aspect_ratio"]),
            "eccentricity": eccentricity,
            "circularity": circularity,
            "area": int(metrics["area"]),
        }
    except Exception as exc:
        logger.warning(f"[morph_worker] {task.get('image_path', '?')}: {exc}")
        return None


def _context_worker(task: dict) -> List[dict]:
    """Extract context features from all non-mask 64px patches of one image."""
    rows: List[dict] = []
    try:
        image_path = Path(task["image_path"])
        mask_path_str = task.get("mask_path")
        image_type = task.get("image_type", "good")
        gs = task.get("grid_size", GRID_SIZE)

        img = safe_imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        h, w = img.shape

        defect_mask: Optional[np.ndarray] = None
        if image_type == "defect" and mask_path_str and Path(mask_path_str).exists():
            raw = safe_imread(str(mask_path_str), cv2.IMREAD_GRAYSCALE)
            defect_mask = raw > 0

        for i in range(h // gs):
            for j in range(w // gs):
                y1, x1 = i * gs, j * gs
                y2, x2 = y1 + gs, x1 + gs

                # Skip patches that are majority defect area
                if defect_mask is not None:
                    if defect_mask[y1:y2, x1:x2].mean() > 0.5:
                        continue

                patch = img[y1:y2, x1:x2]
                feats = _extract_context_features(patch)
                feats.update({
                    "image_id": image_path.stem,
                    "image_type": image_type,
                    "patch_xy": f"{x1}_{y1}",
                })
                rows.append(feats)
    except Exception as exc:
        logger.warning(f"[context_worker] {task.get('image_path', '?')}: {exc}")
    return rows


# ---------------------------------------------------------------------------
# Distribution analysis helpers
# ---------------------------------------------------------------------------


def _detect_valleys(values: np.ndarray) -> Tuple[int, List[float]]:
    """Return (n_valleys, valley_x_positions) via inverted-histogram peak detection.

    Bin count is chosen via Sturges' rule: ceil(log2(n)) + 1, capped at
    MAX_HISTOGRAM_BINS. For typical defect datasets (92-100 samples) this gives
    ~8 bins instead of the former fixed 50, preventing sparse-bin noise peaks.

    Prominence = max(2 * noise_floor, counts.max() * VALLEY_PROMINENCE_RATIO)
    noise_floor = sqrt(n_samples / n_bins) — sampling noise estimate (1σ per bin).
    The 2× factor requires a valley to exceed 2σ noise (95% confidence).

    Right-skewed distributions (p90/p10 > 4) are log1p-transformed before histogram
    so that long-tail features (e.g. aspect_ratio) are linearised for valley detection.
    Valley positions are then back-transformed to original scale via expm1.
    """
    p10, p50, p90 = np.percentile(values, [10, 50, 90])
    upper_range = p90 - p50
    lower_range = p50 - p10 + 1e-9
    # Log-transform only when upper tail is > 2× longer than lower tail (right-skewed scale)
    # p90/p10 ratio alone misclassifies uniform [0,1] features (e.g. linearity, circularity)
    log_transform = (upper_range / lower_range > 2.0) and (p10 > 1e-9)

    work = np.log1p(values) if log_transform else values
    bins = max(int(np.ceil(np.log2(len(work)))) + 1, 5)
    bins = min(bins, min(MAX_HISTOGRAM_BINS, len(work) - 1))
    counts, bin_edges = np.histogram(work, bins=bins)
    inverted = counts.max() - counts
    noise_floor = np.sqrt(len(work) / bins)
    prominence = max(noise_floor * 2, counts.max() * VALLEY_PROMINENCE_RATIO)
    peaks, _ = scipy.signal.find_peaks(inverted, prominence=prominence)

    if log_transform:
        valley_positions = [
            float(np.expm1((bin_edges[int(p)] + bin_edges[int(p) + 1]) / 2))
            for p in peaks
        ]
    else:
        valley_positions = [
            float((bin_edges[int(p)] + bin_edges[int(p) + 1]) / 2)
            for p in peaks
        ]
    return len(peaks), valley_positions


def _fit_gmm_bic(X: np.ndarray, max_k: int):
    """Fit GaussianMixture; return (gmm, best_k) by minimising BIC."""
    best_bic = np.inf
    best_gmm = None
    best_k = 1
    for k in range(1, max_k + 1):
        try:
            gmm = GaussianMixture(n_components=k, random_state=42, n_init=3)
            gmm.fit(X)
            bic = gmm.bic(X)
            if bic < best_bic:
                best_bic, best_gmm, best_k = bic, gmm, k
        except Exception:
            break
    return best_gmm, best_k


def _context_cell_key(feat_dict: dict, bin_edges: Dict[str, List[float]]) -> str:
    """Discretize 5 context features to a cell key like '0_1_2_0_1'."""
    bins = []
    for feat in CONTEXT_FEATURES:
        try:
            val = float(feat_dict.get(feat, 0.0))
        except (TypeError, ValueError):
            val = 0.0
        edges = bin_edges.get(feat, [0.0, 1.0])
        if len(edges) < 2 or edges[1] <= edges[0]:
            b = 0
        else:
            b = min(int(np.searchsorted(edges, val, side="right")), N_CONTEXT_BINS - 1)
        bins.append(str(b))
    return "_".join(bins)


def _compute_bin_edges(rows: List[dict]) -> Dict[str, List[float]]:
    """Compute P33/P66 bin edges for each context feature from a list of rows."""
    edges: Dict[str, List[float]] = {}
    for feat in CONTEXT_FEATURES:
        vals = []
        for r in rows:
            v = r.get(feat)
            if v not in ("", None):
                try:
                    vals.append(float(v))
                except ValueError:
                    pass
        if len(vals) < 3:
            edges[feat] = [0.0, 1.0]
        else:
            p33 = float(np.percentile(vals, 33.33))
            p66 = float(np.percentile(vals, 66.67))
            edges[feat] = [p33, p66]
    return edges


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------


def _save_csv(rows: List[dict], fields: List[str], path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})
    logger.info(f"  Saved {len(rows)} rows → {path.name}")


def _load_csv(path: Path) -> List[dict]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _float_col(rows: List[dict], key: str) -> List[float]:
    out = []
    for r in rows:
        v = r.get(key)
        if v not in ("", None):
            try:
                out.append(float(v))
            except ValueError:
                pass
    return out


# ---------------------------------------------------------------------------
# Main profiler
# ---------------------------------------------------------------------------


class DistributionProfiler:
    """Orchestrates the 9-step AROMA Stage 0 distribution profiling pipeline."""

    def __init__(
        self,
        entry: dict,
        dataset_key: str,
        output_dir: Path,
        num_workers: int,
        max_images: Optional[int],
        n_gmm_max: int,
        checkpoint: Optional[str] = None,
    ):
        self.entry = entry
        self.dataset_key = dataset_key
        self.output_dir = output_dir
        self.num_workers = resolve_workers(num_workers)
        self.max_images = max_images
        self.n_gmm_max = n_gmm_max
        self.checkpoint = checkpoint
        self.domain = entry.get("domain", "unknown")

        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "figures").mkdir(exist_ok=True)

        # Populated in step1
        self.morph_tasks: List[dict] = []
        self.context_tasks: List[dict] = []

        # Populated in step5
        self.cluster_assignments: Dict[str, int] = {}  # image_id → cluster_id

        # Populated in step6
        self.context_bin_edges: Dict[str, List[float]] = {}

    # ------------------------------------------------------------------
    # Step 1 — Load config and discover paths
    # ------------------------------------------------------------------

    def step1_load_and_discover(self) -> None:
        logger.info("[Step 1] Discovering image and mask paths...")
        domain = self.domain
        image_dir = Path(self.entry["image_dir"])
        seed_entries = _resolve_seed_entries(self.entry)

        if not image_dir.exists():
            logger.warning(f"  image_dir not found: {image_dir}")

        good_images = sorted(p for p in image_dir.glob("*.*")
                             if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp"))
        if self.max_images:
            good_images = good_images[: self.max_images]

        for p in good_images:
            self.context_tasks.append({
                "image_path": str(p),
                "image_type": "good",
                "grid_size": GRID_SIZE,
            })

        fallback_count = 0
        for defect_type, seed_dir in seed_entries:
            if not seed_dir.exists():
                logger.warning(f"  seed_dir not found: {seed_dir}")
                continue
            images = sorted(p for p in seed_dir.glob("*.*")
                            if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp"))
            if self.max_images:
                images = images[: self.max_images]

            for p in images:
                mask_p = _find_mask_path(domain, p, defect_type)
                if mask_p is None:
                    fallback_count += 1
                task_base = {
                    "image_path": str(p),
                    "mask_path": str(mask_p) if mask_p else None,
                    "defect_type": defect_type,
                    "domain": domain,
                    "checkpoint": self.checkpoint,
                }
                self.morph_tasks.append(task_base)
                self.context_tasks.append({
                    **task_base,
                    "image_type": "defect",
                    "grid_size": GRID_SIZE,
                })

        logger.info(
            f"  good images: {len(good_images)}, "
            f"defect images: {len(self.morph_tasks)}, "
            f"fallback masks: {fallback_count}"
        )
        if fallback_count > 0:
            logger.warning(
                f"  {fallback_count} images will use SAM/Otsu fallback (mask_source=fallback_*)."
            )
        if domain == "isp":
            logger.warning(
                "  ISP domain: no ground-truth masks — all instances via SAM/Otsu fallback. "
                "Morphology statistics may be less reliable."
            )

    # ------------------------------------------------------------------
    # Step 2 — Morphology feature extraction
    # ------------------------------------------------------------------

    def step2_morphology_features(self) -> None:
        logger.info(
            f"[Step 2] Extracting morphology features ({len(self.morph_tasks)} defect images)..."
        )
        out_path = self.output_dir / "morphology_features.csv"
        if not self.morph_tasks:
            logger.warning("  No defect images — writing empty CSV.")
            _save_csv([], _MORPH_CSV_FIELDS, out_path)
            return

        raw = run_parallel(_morph_worker, self.morph_tasks, self.num_workers, desc="Morphology")
        rows = [r for r in raw if r is not None]
        logger.info(f"  Extracted {len(rows)} instances from {len(self.morph_tasks)} images")
        _save_csv(rows, _MORPH_CSV_FIELDS, out_path)

    # ------------------------------------------------------------------
    # Step 3 — Context feature extraction
    # ------------------------------------------------------------------

    def step3_context_features(self) -> None:
        logger.info(
            f"[Step 3] Extracting context features ({len(self.context_tasks)} images, 64px patches)..."
        )
        out_path = self.output_dir / "context_features.csv"
        if not self.context_tasks:
            logger.warning("  No images — writing empty CSV.")
            _save_csv([], _CONTEXT_CSV_FIELDS, out_path)
            return

        results = run_parallel(
            _context_worker, self.context_tasks, self.num_workers, desc="Context"
        )
        # Stream directly to CSV to avoid double-accumulating list-of-lists in memory
        total_rows = 0
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=_CONTEXT_CSV_FIELDS, extrasaction="ignore")
            writer.writeheader()
            for r in results:
                if isinstance(r, list):
                    for row in r:
                        writer.writerow({k: row.get(k, "") for k in _CONTEXT_CSV_FIELDS})
                        total_rows += 1
        logger.info(f"  Saved {total_rows} patch rows → {out_path.name}")

    # ------------------------------------------------------------------
    # Step 4 — Distribution analysis per morphology feature
    # ------------------------------------------------------------------

    def step4_distribution_analysis(self) -> None:
        logger.info("[Step 4] Analyzing morphology feature distributions...")
        rows = _load_csv(self.output_dir / "morphology_features.csv")
        if not rows:
            logger.warning("  Empty morphology CSV — skipping.")
            save_json({}, self.output_dir / "distribution_analysis.json")
            return

        analysis: Dict[str, dict] = {}
        for feat in MORPH_FEATURES:
            values = np.array(_float_col(rows, feat))
            if len(values) < 3:
                analysis[feat] = {
                    "policy": "percentile",
                    "distribution": "unimodal",
                    "n_valleys": 0,
                    "valley_positions": [],
                    "boundaries": [],
                    "percentiles": {},
                }
                continue

            percentiles = {
                f"p{p}": float(np.percentile(values, p)) for p in [10, 25, 50, 75, 90]
            }
            n_valleys, valley_pos = _detect_valleys(values)

            if n_valleys >= 2 and HAS_SKLEARN:
                policy = "gmm"
                distribution = "multimodal"
                boundaries = valley_pos
            elif n_valleys == 1:
                policy = "otsu"
                distribution = "bimodal"
                v_range = (values.max() - values.min()) + 1e-6
                norm = ((values - values.min()) / v_range * 255).astype(np.uint8)
                thresh, _ = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                otsu_val = float(values.min() + (thresh / 255.0) * v_range)
                boundaries = [otsu_val]
            else:
                policy = "percentile"
                distribution = "unimodal"
                boundaries = [percentiles["p25"], percentiles["p50"], percentiles["p75"]]

            analysis[feat] = {
                "policy": policy,
                "distribution": distribution,
                "n_valleys": n_valleys,
                "valley_positions": valley_pos,
                "boundaries": boundaries,
                "percentiles": percentiles,
            }

        save_json(analysis, self.output_dir / "distribution_analysis.json")
        policies = {f: analysis[f]["policy"] for f in analysis}
        logger.info(f"  Policies: {policies}")

    # ------------------------------------------------------------------
    # Step 5 — Morphology clustering (GMM/BIC on full feature matrix)
    # ------------------------------------------------------------------

    def step5_morphology_clustering(self) -> None:
        logger.info("[Step 5] Morphology clustering (GMM/BIC on full feature matrix)...")
        rows = _load_csv(self.output_dir / "morphology_features.csv")
        out_path = self.output_dir / "morphology_clusters.json"

        if not rows:
            save_json({"n_clusters": 0, "clusters": [], "cluster_assignments": {}}, out_path)
            return

        valid_rows, X_list = [], []
        for r in rows:
            try:
                vec = [float(r[f]) for f in MORPH_FEATURES]
                X_list.append(vec)
                valid_rows.append(r)
            except (KeyError, ValueError):
                pass

        if not X_list:
            save_json({"n_clusters": 0, "clusters": [], "cluster_assignments": {}}, out_path)
            return

        X = np.array(X_list)
        # Normalise to [0,1] per feature for GMM
        X_min = X.min(axis=0)
        X_range = X.max(axis=0) - X_min + 1e-6
        X_norm = (X - X_min) / X_range

        if HAS_SKLEARN and len(X_norm) >= 4:
            gmm, n_clusters = _fit_gmm_bic(X_norm, self.n_gmm_max)
            labels: List[int] = gmm.predict(X_norm).tolist()
            method = "gmm_bic"
        else:
            # Fallback: percentile bins on aspect_ratio; respect n_gmm_max as bin count cap
            n_clusters = min(self.n_gmm_max, 3)
            ar = X[:, MORPH_FEATURES.index("aspect_ratio")]
            # Build n_clusters-1 evenly spaced percentile boundaries
            pct_steps = [100 * i / n_clusters for i in range(1, n_clusters)]
            boundaries = [float(np.percentile(ar, p)) for p in pct_steps]
            labels = np.digitize(ar, boundaries).tolist()
            method = "percentile_fallback"

        # Build per-cluster summaries
        cluster_info = []
        for c in range(n_clusters):
            idxs = [i for i, lbl in enumerate(labels) if lbl == c]
            if not idxs:
                continue
            sub = X[idxs]
            centroid = {MORPH_FEATURES[i]: float(np.mean(sub[:, i])) for i in range(len(MORPH_FEATURES))}
            cluster_info.append({
                "cluster_id": c,
                "n_samples": len(idxs),
                "centroid": centroid,
                "label": _auto_label(centroid),
            })

        # Store assignments (image_id → cluster_id)
        for i, row in enumerate(valid_rows):
            self.cluster_assignments[row["image_id"]] = int(labels[i])

        save_json(
            {
                "n_clusters": n_clusters,
                "method": method,
                "clusters": cluster_info,
                "cluster_assignments": {k: int(v) for k, v in self.cluster_assignments.items()},
            },
            out_path,
        )
        sizes = {c["cluster_id"]: c["n_samples"] for c in cluster_info}
        logger.info(f"  {n_clusters} clusters ({method}), sizes: {sizes}")

    # ------------------------------------------------------------------
    # Step 6 — Compatibility learning P(context_cell | cluster)
    # ------------------------------------------------------------------

    def step6_compatibility_learning(self) -> None:
        logger.info("[Step 6] Compatibility learning P(context_cell | cluster)...")
        morph_rows = _load_csv(self.output_dir / "morphology_features.csv")
        clusters_data = load_json(self.output_dir / "morphology_clusters.json")
        context_rows = _load_csv(self.output_dir / "context_features.csv")
        out_path = self.output_dir / "compatibility_matrix.json"

        n_clusters = clusters_data.get("n_clusters", 0)
        if n_clusters == 0 or not morph_rows or not context_rows:
            save_json({}, out_path)
            return

        # Compute global bin edges from all context patches
        bin_edges = _compute_bin_edges(context_rows)
        self.context_bin_edges = bin_edges

        # Build image_id → mean context features from defect patches
        defect_patch_feats: Dict[str, Dict[str, List[float]]] = defaultdict(
            lambda: {f: [] for f in CONTEXT_FEATURES}
        )
        for r in context_rows:
            if r.get("image_type") == "defect":
                iid = r["image_id"]
                for feat in CONTEXT_FEATURES:
                    v = r.get(feat)
                    if v not in ("", None):
                        try:
                            defect_patch_feats[iid][feat].append(float(v))
                        except ValueError:
                            pass

        defect_mean_ctx: Dict[str, Dict[str, float]] = {}
        for iid, feat_lists in defect_patch_feats.items():
            if all(feat_lists[f] for f in CONTEXT_FEATURES):
                defect_mean_ctx[iid] = {f: float(np.mean(feat_lists[f])) for f in CONTEXT_FEATURES}

        # Global mean context as fallback for images missing patch data
        global_means: Dict[str, float] = {}
        for feat in CONTEXT_FEATURES:
            vals = _float_col(context_rows, feat)
            global_means[feat] = float(np.mean(vals)) if vals else 0.0

        # Count (cluster_id, context_cell) co-occurrences
        counts: Dict[int, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for row in morph_rows:
            iid = row["image_id"]
            cluster_id = self.cluster_assignments.get(iid)
            if cluster_id is None:
                continue
            ctx = defect_mean_ctx.get(iid, global_means)
            cell = _context_cell_key(ctx, bin_edges)
            counts[cluster_id][cell] += 1

        # P(cell | cluster)
        matrix: Dict[str, Dict[str, float]] = {}
        for c in range(n_clusters):
            total = sum(counts[c].values()) if c in counts else 0
            if total == 0:
                matrix[str(c)] = {}
            else:
                matrix[str(c)] = {cell: cnt / total for cell, cnt in counts[c].items()}

        save_json(
            {
                "n_clusters": n_clusters,
                "n_context_bins": N_CONTEXT_BINS,
                "context_features": CONTEXT_FEATURES,
                "bin_edges": bin_edges,
                "matrix": matrix,
            },
            out_path,
        )
        logger.info(f"  Compatibility matrix built for {n_clusters} clusters")

    # ------------------------------------------------------------------
    # Step 7 — Deficit analysis
    # ------------------------------------------------------------------

    def step7_deficit_analysis(self) -> None:
        logger.info("[Step 7] Deficit analysis...")
        compat_data = load_json(self.output_dir / "compatibility_matrix.json")
        context_rows = _load_csv(self.output_dir / "context_features.csv")
        clusters_data = load_json(self.output_dir / "morphology_clusters.json")
        out_path = self.output_dir / "deficit_analysis.json"

        if not compat_data or not context_rows:
            save_json({}, out_path)
            return

        n_clusters = compat_data.get("n_clusters", 0)
        bin_edges = compat_data.get("bin_edges", {})
        matrix = compat_data.get("matrix", {})
        if not self.context_bin_edges:
            self.context_bin_edges = bin_edges

        # Global context distribution over all patches
        global_counts: Dict[str, int] = defaultdict(int)
        for r in context_rows:
            try:
                cell = _context_cell_key(r, bin_edges)
                global_counts[cell] += 1
            except Exception:
                pass

        total_global = sum(global_counts.values()) + 1e-10
        global_dist = {cell: cnt / total_global for cell, cnt in global_counts.items()}
        all_cells = set(global_dist.keys())

        # Cluster prior P(cluster)
        cluster_sizes = {str(c): 0 for c in range(n_clusters)}
        for cl in clusters_data.get("clusters", []):
            cluster_sizes[str(cl["cluster_id"])] = cl["n_samples"]
        total_instances = sum(cluster_sizes.values()) + 1e-10
        cluster_prior = {c: s / total_instances for c, s in cluster_sizes.items()}

        # Deficit + target_synthetic per cluster
        deficit_output: Dict[str, dict] = {}
        for c in range(n_clusters):
            c_str = str(c)
            p_cell_given_c = matrix.get(c_str, {})
            deficit_cells: Dict[str, float] = {}
            for cell in all_cells:
                g = global_dist.get(cell, 0.0)
                p = p_cell_given_c.get(cell, 0.0)
                deficit_cells[cell] = max(0.0, g - p)

            deficit_sum = sum(deficit_cells.values())
            if deficit_sum < 1e-10:
                # Uniform fallback when cluster already covers all contexts
                n_cells = max(len(all_cells), 1)
                target_synthetic = {cell: 1.0 / n_cells for cell in all_cells}
            else:
                target_synthetic = {cell: d / deficit_sum for cell, d in deficit_cells.items()}

            deficit_output[c_str] = {
                "prior": float(cluster_prior.get(c_str, 0.0)),
                "deficit": {cell: float(d) for cell, d in deficit_cells.items()},
                "target_synthetic": {cell: float(v) for cell, v in target_synthetic.items()},
            }

        save_json(deficit_output, out_path)
        logger.info(
            f"  Deficit computed for {n_clusters} clusters over {len(all_cells)} context cells"
        )

    # ------------------------------------------------------------------
    # Step 8 — Threshold policies
    # ------------------------------------------------------------------

    def step8_threshold_policies(self) -> None:
        logger.info("[Step 8] Saving threshold policies...")
        analysis = load_json(self.output_dir / "distribution_analysis.json")
        clusters_data = load_json(self.output_dir / "morphology_clusters.json")

        policies: Dict[str, dict] = {}
        for feat, info in analysis.items():
            policies[feat] = {
                "feature": feat,
                "distribution": info.get("distribution", "unimodal"),
                "policy": info.get("policy", "percentile"),
                "boundaries": info.get("boundaries", []),
                "valley_positions": info.get("valley_positions", []),
                "percentiles": info.get("percentiles", {}),
                "n_valleys": info.get("n_valleys", 0),
            }

        save_json(
            {
                "morphology_features": policies,
                "clustering_method": clusters_data.get("method", "unknown"),
                "n_clusters": clusters_data.get("n_clusters", 0),
            },
            self.output_dir / "threshold_policies.json",
        )

    # ------------------------------------------------------------------
    # Step 9 — Reporting
    # ------------------------------------------------------------------

    def step9_reporting(self) -> None:
        logger.info("[Step 9] Generating recommended_config.yaml, report, and figures...")
        clusters_data = load_json(self.output_dir / "morphology_clusters.json")
        compat_data = load_json(self.output_dir / "compatibility_matrix.json")

        self._write_yaml(clusters_data, compat_data)
        self._write_report(clusters_data)
        self._plot_figures()

        logger.info("Output files:")
        for f in sorted(self.output_dir.glob("*")):
            if f.is_file():
                logger.info(f"  {f.name}")
        for f in sorted((self.output_dir / "figures").glob("*.png")):
            logger.info(f"  figures/{f.name}")

    def _write_yaml(self, clusters_data: dict, compat_data: dict) -> None:
        config = {
            "dataset_key": self.dataset_key,
            "domain": self.domain,
            "morphology": {
                "clustering_method": clusters_data.get("method", "unknown"),
                "n_clusters": clusters_data.get("n_clusters", 0),
                "features": MORPH_FEATURES,
                "clusters": [
                    {
                        "id": c["cluster_id"],
                        "label": c["label"],
                        "n_samples": c["n_samples"],
                        "centroid": c["centroid"],
                    }
                    for c in clusters_data.get("clusters", [])
                ],
            },
            "context": {
                "n_bins_per_feature": N_CONTEXT_BINS,
                "features": CONTEXT_FEATURES,
                "bin_edges": compat_data.get("bin_edges", {}),
            },
            "compatibility": {
                "source": "data_driven",
                "matrix_path": "compatibility_matrix.json",
            },
            "deficit": {
                "deficit_path": "deficit_analysis.json",
            },
        }
        with open(self.output_dir / "recommended_config.yaml", "w", encoding="utf-8") as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    def _write_report(self, clusters_data: dict) -> None:
        n_clusters = clusters_data.get("n_clusters", 0)
        clusters = clusters_data.get("clusters", [])
        analysis: dict = {}
        try:
            analysis = load_json(self.output_dir / "distribution_analysis.json")
        except Exception:
            pass

        morph_rows = _load_csv(self.output_dir / "morphology_features.csv")
        context_rows = _load_csv(self.output_dir / "context_features.csv")
        fallback_count = sum(
            1 for r in morph_rows if str(r.get("mask_source", "")).startswith("fallback")
        )

        lines = [
            "# AROMA Distribution Profiling Report",
            "",
            f"**Dataset key**: `{self.dataset_key}` | **Domain**: `{self.domain}`",
            f"**Defect instances**: {len(morph_rows)} | **Context patches**: {len(context_rows)}",
            f"**Fallback masks**: {fallback_count}",
            "",
            "## Morphology Clustering",
            "",
            f"Method: `{clusters_data.get('method', 'unknown')}` | Clusters: **{n_clusters}**",
            "",
            "| Cluster ID | Label | N samples | aspect_ratio | linearity | solidity |",
            "|-----------|-------|-----------|-------------|-----------|---------|",
        ]
        for c in clusters:
            ct = c.get("centroid", {})
            lines.append(
                f"| {c['cluster_id']} | {c['label']} | {c['n_samples']} "
                f"| {ct.get('aspect_ratio', 0):.2f} "
                f"| {ct.get('linearity', 0):.2f} "
                f"| {ct.get('solidity', 0):.2f} |"
            )

        lines += [
            "",
            "## Distribution Policies",
            "",
            "| Feature | Distribution | Policy | N valleys |",
            "|---------|-------------|--------|----------|",
        ]
        for feat, info in analysis.items():
            lines.append(
                f"| {feat} | {info.get('distribution','-')} "
                f"| {info.get('policy','-')} "
                f"| {info.get('n_valleys', 0)} |"
            )

        if self.domain == "isp" or fallback_count > 0:
            lines += [
                "",
                "> **Warning**: "
                + (
                    "ISP domain has no ground-truth masks. "
                    if self.domain == "isp"
                    else ""
                )
                + (
                    f"{fallback_count} instances used SAM/Otsu fallback masks. "
                    if fallback_count > 0
                    else ""
                )
                + "Morphology statistics for these instances may be less reliable.",
            ]

        lines += ["", "## Figures", "", "See `figures/` directory for histograms and heatmaps."]

        with open(self.output_dir / "analysis_report.md", "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

    def _plot_figures(self) -> None:
        figs_dir = self.output_dir / "figures"
        morph_rows = _load_csv(self.output_dir / "morphology_features.csv")
        analysis: dict = {}
        try:
            analysis = load_json(self.output_dir / "distribution_analysis.json")
        except Exception:
            pass

        # Morphology histograms with boundary lines
        if morph_rows:
            fig, axes = plt.subplots(2, 3, figsize=(15, 8))
            for idx, feat in enumerate(MORPH_FEATURES):
                ax = axes[idx // 3][idx % 3]
                values = _float_col(morph_rows, feat)
                if not values:
                    ax.set_visible(False)
                    continue
                ax.hist(values, bins=30, alpha=0.75, color="#4C72B0", edgecolor="white")
                info = analysis.get(feat, {})
                for b in info.get("boundaries", []):
                    ax.axvline(b, color="red", linestyle="--", linewidth=1.5, label=f"boundary={b:.2f}")
                ax.set_title(f"{feat}  [{info.get('policy', '')}]")
                ax.set_xlabel("Value")
                ax.set_ylabel("Count")
            plt.suptitle(f"Morphology Distributions — {self.dataset_key}", fontsize=13)
            plt.tight_layout()
            plt.savefig(figs_dir / "morphology_histograms.png", dpi=100, bbox_inches="tight")
            plt.close()

        # Compatibility heatmap
        compat_data: dict = {}
        try:
            compat_data = load_json(self.output_dir / "compatibility_matrix.json")
        except Exception:
            pass
        matrix_raw = compat_data.get("matrix", {})
        if matrix_raw:
            all_cells = sorted({cell for probs in matrix_raw.values() for cell in probs})
            cluster_ids = sorted(matrix_raw.keys(), key=lambda x: int(x))
            mat = np.zeros((len(cluster_ids), len(all_cells)))
            for ci, cid in enumerate(cluster_ids):
                for cj, cell in enumerate(all_cells):
                    mat[ci, cj] = matrix_raw[cid].get(cell, 0.0)

            if mat.size > 0:
                fig, ax = plt.subplots(
                    figsize=(max(6, len(all_cells) * 0.35 + 2), max(3, len(cluster_ids) + 1))
                )
                im = ax.imshow(mat, cmap="viridis", aspect="auto", vmin=0)
                plt.colorbar(im, ax=ax, label="P(cell | cluster)")
                ax.set_yticks(range(len(cluster_ids)))
                ax.set_yticklabels([f"Cluster {c}" for c in cluster_ids])
                ax.set_xlabel("Context Cell")
                ax.set_title(f"P(Context Cell | Morphology Cluster) — {self.dataset_key}")
                if len(all_cells) <= 30:
                    ax.set_xticks(range(len(all_cells)))
                    ax.set_xticklabels(all_cells, rotation=90, fontsize=6)
                plt.tight_layout()
                plt.savefig(figs_dir / "compatibility_heatmap.png", dpi=100, bbox_inches="tight")
                plt.close()

        # Deficit bar charts
        deficit_data: dict = {}
        try:
            deficit_data = load_json(self.output_dir / "deficit_analysis.json")
        except Exception:
            pass
        if deficit_data:
            n_clusters_d = len(deficit_data)
            fig, axes = plt.subplots(
                1, n_clusters_d, figsize=(max(6, n_clusters_d * 4), 4), squeeze=False
            )
            for ci, (cid, info) in enumerate(sorted(deficit_data.items())):
                ax = axes[0][ci]
                ts = info.get("target_synthetic", {})
                cells = sorted(ts.keys())
                vals = [ts.get(c, 0.0) for c in cells]
                ax.bar(range(len(cells)), vals, color="#DD8452")
                ax.set_title(f"Cluster {cid} — target_synthetic")
                ax.set_ylabel("Weight")
                ax.set_xlabel("Context Cell")
                if len(cells) <= 20:
                    ax.set_xticks(range(len(cells)))
                    ax.set_xticklabels(cells, rotation=45, ha="right", fontsize=7)
            plt.suptitle(f"Deficit-aware synthesis targets — {self.dataset_key}", fontsize=11)
            plt.tight_layout()
            plt.savefig(figs_dir / "deficit_bars.png", dpi=100, bbox_inches="tight")
            plt.close()

    # ------------------------------------------------------------------
    # Main runner
    # ------------------------------------------------------------------

    def run(self) -> None:
        logger.info(f"=== AROMA Distribution Profiling: {self.dataset_key} (domain={self.domain}) ===")
        self.step1_load_and_discover()
        self.step2_morphology_features()
        self.step3_context_features()
        self.step4_distribution_analysis()
        self.step5_morphology_clustering()
        self.step6_compatibility_learning()
        self.step7_deficit_analysis()
        self.step8_threshold_policies()
        self.step9_reporting()
        logger.info("=== Distribution Profiling complete ===")


# ---------------------------------------------------------------------------
# Auto-label helper (module level for clarity)
# ---------------------------------------------------------------------------


def _auto_label(centroid: dict) -> str:
    """Heuristic cluster label derived from centroid feature values."""
    ar  = centroid.get("aspect_ratio", 1.0)
    sol = centroid.get("solidity", 0.5)
    lin = centroid.get("linearity", 0.0)
    if lin > 0.7 and ar > 5.0:
        return "linear_scratch"
    if ar > 4.0:
        return "elongated"
    if ar < 2.0 and sol > 0.85:
        return "compact_blob"
    if sol < 0.65:
        return "irregular"
    return "general"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="AROMA Stage 0 — Distribution Profiling",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--dataset_config",
        default="/content/AROMA/dataset_config.json",
        help="Path to dataset_config.json (default: /content/AROMA/dataset_config.json)",
    )
    p.add_argument(
        "--dataset_key",
        required=True,
        help="Key in dataset_config.json, e.g. visa_cashew, mvtec_bottle, isp_LSM_1",
    )
    p.add_argument("--output_dir", required=True, help="Output directory for profiling results")
    p.add_argument(
        "--num_workers",
        type=int,
        default=-1,
        help="Worker count: 0=sequential, -1=auto(cpu_count-1), N=N (default: -1)",
    )
    p.add_argument(
        "--max_images",
        type=int,
        default=None,
        help="Limit images per split for debugging (default: all)",
    )
    p.add_argument(
        "--n_gmm_components",
        type=int,
        default=5,
        help="Max GMM components; BIC auto-selects k in [1, N] (default: 5)",
    )
    p.add_argument(
        "--sam_checkpoint",
        default=None,
        help="SAM checkpoint path for fallback mask extraction (optional)",
    )
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = _parse_args(argv)

    config_path = Path(args.dataset_config)
    if not config_path.exists():
        logger.error(f"dataset_config not found: {config_path}")
        sys.exit(1)

    config = load_json(config_path)
    if args.dataset_key not in config:
        logger.error(f"Key '{args.dataset_key}' not found in {config_path}")
        logger.info(f"Available keys: {sorted(config.keys())}")
        sys.exit(1)

    entry = config[args.dataset_key]
    output_dir = Path(args.output_dir)

    profiler = DistributionProfiler(
        entry=entry,
        dataset_key=args.dataset_key,
        output_dir=output_dir,
        num_workers=args.num_workers,
        max_images=args.max_images,
        n_gmm_max=args.n_gmm_components,
        checkpoint=args.sam_checkpoint,
    )
    profiler.run()


if __name__ == "__main__":
    main()
