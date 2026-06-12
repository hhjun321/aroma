#!/usr/bin/env python3
"""
AROMA Step 1 — Complexity Analysis

Reads Phase 0 (distribution_profiling.py) outputs and computes dataset-level
MCI / CCI scalar indices, then selects morphology and context modeling policies
via Empirical Policy Evaluation.

Usage (Colab):
    !python $AROMA_SCRIPTS/compute_complexity.py \
        --profiling_dir  $AROMA_OUT/profiling/isp_LSM_1 \
        --output_dir     $AROMA_OUT/complexity/isp_LSM_1 \
        --weight_mode    equal

Outputs (written to --output_dir):
    complexity_report.json   MCI, CCI, selected policies, evaluation trace
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import shutil
import sys
import tempfile
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Bootstrap: add AROMA_REF to sys.path so utils.io is importable
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("aroma.complexity")


def _bootstrap_aroma_ref() -> str:
    ref = os.environ.get("AROMA_REF") or r"D:\project\aroma"
    ref_path = Path(ref)
    if ref_path.is_dir():
        if str(ref_path) not in sys.path:
            sys.path.insert(0, str(ref_path))
        return "aroma_ref"
    logger.warning("AROMA_REF not found (%s) — using inline I/O fallbacks", ref)
    return "inline"


_REF_SOURCE = _bootstrap_aroma_ref()

try:
    from utils.io import load_json, save_json  # type: ignore[import]
    _IO_SOURCE = "aroma_ref"
except Exception:
    def load_json(p):  # type: ignore[misc]
        with open(p, encoding="utf-8") as f:
            return json.load(f)

    def save_json(data, p):  # type: ignore[misc]
        Path(p).parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    _IO_SOURCE = "inline"

try:
    from sklearn.mixture import GaussianMixture
    from sklearn.metrics import silhouette_score
    from sklearn.cluster import AgglomerativeClustering
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    GaussianMixture = None  # type: ignore[misc,assignment]
    silhouette_score = None  # type: ignore[assignment]
    AgglomerativeClustering = None  # type: ignore[assignment]
    warnings.warn("scikit-learn not found. Silhouette=0.0, GMM→Percentile fallback.")

try:
    import yaml as _yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False
    warnings.warn("PyYAML not found. Using built-in config defaults.")

# ---------------------------------------------------------------------------
# Constants (mirror Phase 0 for consistency)
# ---------------------------------------------------------------------------

MORPH_FEATURES = [
    "linearity", "solidity", "extent", "aspect_ratio", "eccentricity", "circularity",
]
CONTEXT_FEATURES = [
    "local_variance", "edge_density", "texture_entropy",
    "frequency_energy", "orientation_consistency",
]
MORPH_FEAT_IDX = {f: i for i, f in enumerate(MORPH_FEATURES)}
CTX_FEAT_IDX = {f: i for i, f in enumerate(CONTEXT_FEATURES)}

_DEFAULT_CONFIG: Dict[str, Any] = {
    "mci": {
        "normalization": "minmax",
        "weights": "equal",
        "expected_range": {
            "entropy":         [0.0, 4.0],
            "valley_count":    [0.0, 41.0],
            "class_diversity": [1.0, 9.6],
        },
    },
    "cci": {
        "normalization": "minmax",
        "weights": "equal",
        "expected_range": {
            "texture_entropy":   [0.0, 8.0],
            "cluster_count_ctx": [1.0, 8.0],
            "freq_complexity":   [0.0, 1.0],
            "orient_variance":   [0.0, 1.6],
        },
        "context_gmm": {
            "max_k": 5,
            "max_patches": 20000,
        },
    },
    "distribution": {
        "min_samples": 10,
        "valley_threshold": 1,
        "kurtosis_heavy_tail": 3.0,
    },
    "policy": {
        "stability_margin": 0.05,
        "high_complexity_mci": 0.6,
        "prune_low_mci": 0.3,
        "max_candidates": 3,
        "n_bootstrap": 5,
        "max_k": 8,
    },
}

_WEIGHT_PRESETS: Dict[str, Tuple[float, float, float, float]] = {
    "equal":            (0.25, 0.25, 0.25, 0.25),
    "entropy_heavy":    (0.40, 0.20, 0.20, 0.20),
    "diversity_heavy":  (0.20, 0.20, 0.40, 0.20),
}

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def _deep_merge(base: dict, override: dict) -> dict:
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def load_config(path: Optional[str] = None) -> dict:
    cfg = _deep_merge({}, _DEFAULT_CONFIG)
    if path is None:
        path = str(Path(__file__).parent / "config" / "aroma_step1.yaml")
    if HAS_YAML and Path(str(path)).is_file():
        with open(path, encoding="utf-8") as f:
            loaded = _yaml.safe_load(f) or {}
        cfg = _deep_merge(cfg, loaded)
    elif not HAS_YAML:
        logger.warning("PyYAML unavailable — using built-in defaults")
    else:
        logger.warning("Config not found: %s — using defaults", path)
    return cfg


# ---------------------------------------------------------------------------
# Local staging (Colab Drive latency mitigation)
# ---------------------------------------------------------------------------

_PHASE0_FILES = [
    "distribution_analysis.json",
    "morphology_clusters.json",
    "morphology_features.csv",
    "context_features.csv",
]


def _stage_to_local(profiling_dir: str, staging_root: Optional[str] = None) -> str:
    """Copy Phase 0 output files from profiling_dir (e.g. Google Drive) to a
    local temp directory to avoid repeated Drive I/O during processing.

    Returns the local staging directory path.
    """
    src = Path(profiling_dir)
    if staging_root:
        root = Path(staging_root)
    elif Path("/content").exists():  # Colab environment
        root = Path("/content/tmp/aroma_staging")
    else:
        root = Path(tempfile.gettempdir()) / "aroma_staging"
    dst = root / src.name
    dst.mkdir(parents=True, exist_ok=True)

    copied = 0
    for fname in _PHASE0_FILES:
        src_file = src / fname
        if src_file.exists():
            shutil.copy2(str(src_file), str(dst / fname))
            copied += 1
        else:
            logger.debug("Staging: %s not found, skipping", fname)

    logger.info("Staged %d/%d files: %s → %s", copied, len(_PHASE0_FILES), src, dst)
    return str(dst)


# ---------------------------------------------------------------------------
# Phase 0 output loaders
# ---------------------------------------------------------------------------


def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _rows_to_float_array(
    rows: List[Dict[str, str]], cols: List[str]
) -> Tuple[np.ndarray, List[str]]:
    """Extract numeric columns → (X, image_ids). Skips rows with any non-numeric value."""
    xs, ids = [], []
    for r in rows:
        try:
            vals = [float(r[c]) for c in cols]
            xs.append(vals)
            ids.append(str(r.get("image_id", "")))
        except (KeyError, ValueError, TypeError):
            continue
    if not xs:
        return np.empty((0, len(cols)), dtype=np.float64), []
    return np.array(xs, dtype=np.float64), ids


def load_phase0_outputs(profiling_dir: str) -> Dict[str, Any]:
    """
    Load Phase 0 output files from profiling_dir.

    Returns dict with keys:
        morph_X, morph_ids, morph_labels, context_X,
        dist_analysis, n_morph_clusters, status
    """
    d = Path(profiling_dir)
    status = "ok"

    da_path = d / "distribution_analysis.json"
    if not da_path.exists():
        return {"status": "missing_distribution_analysis"}
    dist_analysis = load_json(str(da_path))

    mc_path = d / "morphology_clusters.json"
    if not mc_path.exists():
        return {"status": "missing_morphology_clusters"}
    clusters_json = load_json(str(mc_path))
    n_morph_clusters = int(clusters_json.get("n_clusters", 1))
    raw_assignments: dict = clusters_json.get("cluster_assignments", {})
    cluster_assignments: Dict[str, int] = {
        str(k): int(v) for k, v in raw_assignments.items()
    }

    morph_rows = _read_csv_rows(d / "morphology_features.csv")
    morph_X, morph_ids = _rows_to_float_array(morph_rows, MORPH_FEATURES)

    if len(morph_X) == 0:
        status = "empty_morph"
        morph_labels = None
    else:
        raw_labels = np.array(
            [cluster_assignments.get(iid, -1) for iid in morph_ids], dtype=np.int32
        )
        missing = int((raw_labels == -1).sum())
        if missing > 0:
            logger.warning(
                "%d / %d morph rows had no cluster assignment (excluded from silhouette)",
                missing, len(morph_ids),
            )
        mask = raw_labels != -1
        morph_X = morph_X[mask]
        morph_ids = [iid for iid, m in zip(morph_ids, mask) if m]
        morph_labels = raw_labels[mask]

    ctx_rows = _read_csv_rows(d / "context_features.csv")
    good_rows = [r for r in ctx_rows if r.get("image_type", "").lower() in ("good", "normal")]
    if not good_rows:
        good_rows = ctx_rows
    context_X, _ = _rows_to_float_array(good_rows, CONTEXT_FEATURES)
    if len(context_X) == 0 and status == "ok":
        status = "empty_context"

    return {
        "morph_X":          morph_X,
        "morph_ids":        morph_ids,
        "morph_labels":     morph_labels,
        "morph_rows_raw":   morph_rows,
        "context_X":        context_X,
        "dist_analysis":    dist_analysis,
        "n_morph_clusters": n_morph_clusters,
        "status":           status,
    }


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------


def _clamp01(x: float) -> float:
    if not math.isfinite(x):
        return 0.0
    return max(0.0, min(1.0, float(x)))


def _normalize_scalar(value: float, lo: float, hi: float, mode: str = "minmax") -> float:
    v = float(value)
    if not math.isfinite(v):
        return 0.0
    if mode == "minmax":
        span = hi - lo
        if span < 1e-9:
            return 0.0
        return _clamp01((v - lo) / span)
    if mode == "zscore":
        # lo=mean, hi=std; sigmoid to [0,1]
        if hi < 1e-9:
            return 0.0  # constant value = zero complexity
        z = (v - lo) / hi
        return _clamp01(1.0 / (1.0 + math.exp(-z)))
    raise ValueError(f"unknown normalization mode: {mode!r}")


def _normalize_array(X: np.ndarray, mode: str = "zscore") -> np.ndarray:
    """Normalize each column of X (N×D). Constant columns → 0."""
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    Xn = np.empty_like(X, dtype=np.float64)
    for j in range(X.shape[1]):
        col = X[:, j].astype(np.float64)
        finite_mask = np.isfinite(col)
        finite = col[finite_mask]
        if len(finite) == 0:
            Xn[:, j] = 0.0
            continue
        if mode == "zscore":
            mu, sigma = float(finite.mean()), float(finite.std())
            if sigma < 1e-9:
                Xn[:, j] = 0.0
            else:
                Xn[:, j] = np.where(finite_mask, (col - mu) / sigma, 0.0)
        else:  # minmax
            lo, hi = float(finite.min()), float(finite.max())
            span = hi - lo
            if span < 1e-9:
                Xn[:, j] = 0.0
            else:
                Xn[:, j] = np.clip(np.where(finite_mask, (col - lo) / span, 0.0), 0.0, 1.0)
    return Xn


# ---------------------------------------------------------------------------
# MCI components
# ---------------------------------------------------------------------------


def _label_entropy(labels: np.ndarray) -> float:
    """Shannon entropy (bits) of cluster label distribution."""
    if len(labels) == 0:
        return 0.0
    _, counts = np.unique(labels, return_counts=True)
    probs = counts.astype(np.float64) / counts.sum()
    return float(-np.sum(probs * np.log2(probs + 1e-12)))


def _total_valley_count(dist_analysis: dict) -> int:
    """Sum n_valleys across all morphology features."""
    return sum(
        int(dist_analysis.get(feat, {}).get("n_valleys", 0))
        for feat in MORPH_FEATURES
    )


def _class_diversity_neff(morph_rows: List[Dict[str, str]]) -> float:
    """
    Effective number of defect categories estimated from the Shannon entropy
    of class distributions (Hill diversity index).

    Neff = exp(H)  where H = -sum(p_k * ln(p_k))

    For a uniform distribution over K classes: Neff = K.
    For a dominated distribution: Neff < K.
    Single class: Neff = 1.0.
    """
    if not morph_rows:
        return 1.0

    counts: Dict[str, int] = {}
    for r in morph_rows:
        dt = r.get("defect_type") or "unknown"
        counts[dt] = counts.get(dt, 0) + 1

    total = sum(counts.values())
    if total == 0:
        return 1.0

    H = -sum((c / total) * math.log(c / total) for c in counts.values() if c > 0)
    return math.exp(H)


def _compute_silhouette(
    X: np.ndarray,
    labels: np.ndarray,
    sample_size: int = 5000,
    seed: int = 0,
) -> float:
    """Silhouette score. Returns 0.0 on failure or when sklearn unavailable."""
    if not HAS_SKLEARN or silhouette_score is None:
        return 0.0
    if len(labels) < 2 or len(np.unique(labels)) < 2:
        return 0.0
    n = len(X)
    if n > sample_size:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, sample_size, replace=False)
        X, labels = X[idx], labels[idx]
        if len(np.unique(labels)) < 2:
            return 0.0
    try:
        return float(silhouette_score(X, labels))
    except Exception as e:
        logger.warning("silhouette_score failed: %s", e)
        return 0.0


def compute_mci(
    phase0: dict,
    cfg: dict,
    weight_mode: Optional[str] = None,
) -> Tuple[float, dict]:
    """
    Morphology Complexity Index.
    MCI = weighted mean of:
        norm(Entropy), norm(ValleyCount), norm(ClassDiversity), norm(1 - Silhouette)
    """
    wmode = weight_mode or cfg["mci"]["weights"]
    norm_mode = cfg["mci"]["normalization"]
    rng = cfg["mci"]["expected_range"]
    weights = list(_WEIGHT_PRESETS.get(wmode, _WEIGHT_PRESETS["equal"]))

    morph_X: np.ndarray = phase0.get("morph_X", np.empty((0, 6)))
    morph_labels: Optional[np.ndarray] = phase0.get("morph_labels")
    dist_analysis: dict = phase0.get("dist_analysis", {})

    Xn = _normalize_array(morph_X, "zscore") if len(morph_X) > 0 else morph_X

    entropy = (
        _label_entropy(morph_labels)
        if morph_labels is not None and len(morph_labels) > 0
        else 0.0
    )
    silhouette = (
        _compute_silhouette(Xn, morph_labels)
        if morph_labels is not None and len(Xn) > 0
        else 0.0
    )

    morph_rows_raw: List[Dict[str, str]] = phase0.get("morph_rows_raw", [])

    raw = {
        "entropy":          entropy,
        "valley_count":     float(_total_valley_count(dist_analysis)),
        "class_diversity":  _class_diversity_neff(morph_rows_raw),
        "inv_silhouette":   _clamp01(1.0 - silhouette),
    }
    normalized = {
        "entropy":          _normalize_scalar(raw["entropy"],         *rng["entropy"],         norm_mode),
        "valley_count":     _normalize_scalar(raw["valley_count"],    *rng["valley_count"],    norm_mode),
        "class_diversity":  _normalize_scalar(raw["class_diversity"], *rng["class_diversity"], norm_mode),
        "inv_silhouette":   _clamp01(1.0 - silhouette),
    }
    assert len(weights) == len(normalized), (
        f"MCI weight/component count mismatch: {len(weights)} != {len(normalized)}"
    )
    mci = float(np.dot(weights, list(normalized.values())))

    return mci, {
        "raw":             raw,
        "normalized":      normalized,
        "weights":         weights,
        "weight_mode":     wmode,
        "normalization":   norm_mode,
        "silhouette_score": silhouette,  # diagnostic: actual silhouette (not inverted)
    }


# ---------------------------------------------------------------------------
# CCI components + context clustering
# ---------------------------------------------------------------------------


def _fit_gmm_bic(X: np.ndarray, max_k: int, seed: int = 0):
    """GMM + BIC model selection. Returns (gmm, best_k).

    Two-phase strategy: BIC search with n_init=1 (fast) then refit best k
    with n_init=3 (quality). Reduces total GMM fits from max_k*3 to max_k+3.
    """
    if not HAS_SKLEARN or GaussianMixture is None:
        return None, 1
    best_bic, best_k = np.inf, 1
    for k in range(1, max_k + 1):
        try:
            gmm = GaussianMixture(n_components=k, random_state=seed, n_init=1)
            gmm.fit(X)
            bic = gmm.bic(X)
            if bic < best_bic:
                best_bic, best_k = bic, k
        except Exception:
            continue  # skip this k; higher k may still converge
    try:
        best_gmm = GaussianMixture(n_components=best_k, random_state=seed, n_init=3)
        best_gmm.fit(X)
        return best_gmm, best_k
    except Exception:
        return None, 1


def _cluster_context(
    context_X: np.ndarray, cfg: dict, seed: int = 0
) -> Tuple[int, np.ndarray, float]:
    """
    Cluster context features to derive CCI's ClusterCount_ctx.
    Phase 0 does not cluster context — Step 1 does it directly.
    Returns (n_clusters, labels, silhouette).
    NOTE: labels covers only the subsampled rows (≤ max_patches), NOT full context_X.
    Callers that discard labels (i.e. only use n_clusters) are safe.
    """
    ctx_cfg = cfg["cci"]["context_gmm"]
    max_k = int(ctx_cfg.get("max_k", 5))
    max_patches = int(ctx_cfg.get("max_patches", 20000))

    n = len(context_X)
    if n < 4:
        return 1, np.zeros(max(n, 0), dtype=np.int32), 0.0

    X = context_X
    if n > max_patches:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, max_patches, replace=False)
        X = X[idx]

    Xn = _normalize_array(X, "zscore")
    gmm, k = _fit_gmm_bic(Xn, max_k, seed=seed)
    if gmm is None:
        return 1, np.zeros(len(Xn), dtype=np.int32), 0.0

    labels = gmm.predict(Xn).astype(np.int32)
    sil = _compute_silhouette(Xn, labels, seed=seed)
    return k, labels, sil


def compute_cci(
    phase0: dict,
    cfg: dict,
    weight_mode: Optional[str] = None,
) -> Tuple[float, dict]:
    """
    Context Complexity Index.
    CCI = weighted mean of:
        norm(TextureEntropy), norm(ClusterCount_ctx),
        norm(FreqComplexity), norm(OrientVariance)
    """
    wmode = weight_mode or cfg["cci"]["weights"]
    norm_mode = cfg["cci"]["normalization"]
    rng = cfg["cci"]["expected_range"]
    weights = list(_WEIGHT_PRESETS.get(wmode, _WEIGHT_PRESETS["equal"]))

    context_X_full: np.ndarray = phase0.get("context_X", np.empty((0, 5)))
    max_patches = int(cfg["cci"]["context_gmm"].get("max_patches", 20000))
    n_ctx = len(context_X_full)
    if n_ctx == 0:
        context_X = context_X_full
    else:
        _sub_rng = np.random.default_rng(0)
        replace = n_ctx < max_patches  # bootstrap when fewer patches than target
        _sub_idx = _sub_rng.choice(n_ctx, max_patches, replace=replace)
        context_X = context_X_full[_sub_idx]
    n_ctx_clusters, _, _ = _cluster_context(context_X, cfg)

    def _col_mean(feat: str) -> float:
        idx = CTX_FEAT_IDX.get(feat, -1)
        if idx < 0 or len(context_X) == 0:
            return 0.0
        col = context_X[:, idx].astype(np.float64)
        finite = col[np.isfinite(col)]
        return float(np.mean(finite)) if len(finite) > 0 else 0.0

    def _col_var(feat: str) -> float:
        idx = CTX_FEAT_IDX.get(feat, -1)
        if idx < 0 or len(context_X) == 0:
            return 0.0
        col = context_X[:, idx].astype(np.float64)
        finite = col[np.isfinite(col)]
        return float(np.var(finite)) if len(finite) > 0 else 0.0

    raw = {
        "texture_entropy":   _col_mean("texture_entropy"),
        "cluster_count_ctx": float(n_ctx_clusters),
        "freq_complexity":   _col_var("frequency_energy"),
        "orient_variance":   _col_var("orientation_consistency"),
    }
    normalized = {
        "texture_entropy":   _normalize_scalar(raw["texture_entropy"],   *rng["texture_entropy"],   norm_mode),
        "cluster_count_ctx": _normalize_scalar(raw["cluster_count_ctx"], *rng["cluster_count_ctx"], norm_mode),
        "freq_complexity":   _normalize_scalar(raw["freq_complexity"],   *rng["freq_complexity"],   norm_mode),
        "orient_variance":   _normalize_scalar(raw["orient_variance"],   *rng["orient_variance"],   norm_mode),
    }
    assert len(weights) == len(normalized), (
        f"CCI weight/component count mismatch: {len(weights)} != {len(normalized)}"
    )
    cci = float(np.dot(weights, list(normalized.values())))

    return cci, {
        "raw":                       raw,
        "normalized":                normalized,
        "weights":                   weights,
        "weight_mode":               wmode,
        "normalization":             norm_mode,
        "n_context_patches_used":    len(context_X),
        "n_context_patches_total":   len(context_X_full),
    }


# ---------------------------------------------------------------------------
# Meta Policy Generator
# ---------------------------------------------------------------------------


def _get_candidate_policies(dist_analysis: dict, mci: float, cfg: dict) -> List[str]:
    """Derive candidate policy set from distribution diagnostics + MCI."""
    pol = cfg["policy"]
    dist_cfg = cfg["distribution"]
    valley_thr = int(dist_cfg.get("valley_threshold", 1))
    total_valleys = _total_valley_count(dist_analysis)

    cands: List[str] = []
    if total_valleys == 0:
        cands = ["percentile"]
    elif total_valleys <= valley_thr:
        cands = ["otsu", "percentile"]
    else:
        cands = ["gmm", "otsu"]

    if mci >= float(pol.get("high_complexity_mci", 0.6)) and "hierarchical" not in cands:
        cands.append("hierarchical")

    if mci < float(pol.get("prune_low_mci", 0.3)):
        cands = [p for p in cands if p not in ("hierarchical", "log_gmm")]

    if not cands:
        cands = ["percentile"]

    seen: set = set()
    unique: List[str] = []
    for p in cands:
        if p not in seen:
            seen.add(p)
            unique.append(p)
    max_c = int(pol.get("max_candidates", 3))
    return unique[:max_c]


def _apply_policy(policy: str, X: np.ndarray, max_k: int = 8, seed: int = 0) -> np.ndarray:
    """Apply policy to feature matrix X → cluster labels."""
    n = len(X)
    if n == 0:
        return np.array([], dtype=np.int32)

    if policy == "percentile":
        ar_idx = MORPH_FEAT_IDX.get("aspect_ratio", 3)
        ar = X[:, ar_idx] if X.ndim > 1 else X
        p33, p66 = np.percentile(ar, [33.33, 66.67])
        if p33 >= p66:  # uniform / near-uniform → single cluster
            return np.zeros(n, dtype=np.int32)
        return np.digitize(ar, bins=[p33, p66]).astype(np.int32)

    if policy == "otsu":
        ar_idx = MORPH_FEAT_IDX.get("aspect_ratio", 3)
        ar = X[:, ar_idx] if X.ndim > 1 else X
        thr = np.median(ar)
        return (ar >= thr).astype(np.int32)

    if policy in ("gmm", "log_gmm"):
        Xw = np.log1p(np.clip(X, 0, None)) if policy == "log_gmm" else X
        gmm, _ = _fit_gmm_bic(Xw, max_k, seed=seed)
        if gmm is None:
            return np.zeros(n, dtype=np.int32)
        return gmm.predict(Xw).astype(np.int32)

    if policy == "hierarchical":
        if not HAS_SKLEARN or AgglomerativeClustering is None:
            return np.zeros(n, dtype=np.int32)
        k = min(3, n)
        try:
            return AgglomerativeClustering(n_clusters=k).fit_predict(X).astype(np.int32)
        except Exception:
            return np.zeros(n, dtype=np.int32)

    logger.warning("Unknown policy %r — fallback to single cluster", policy)
    return np.zeros(n, dtype=np.int32)


def _eval_policy_silhouette(
    policy: str, X: np.ndarray, cfg: dict, seed: int = 0
) -> float:
    if len(X) < 4:
        return 0.0
    max_k = int(cfg["policy"].get("max_k", 8))
    try:
        labels = _apply_policy(policy, X, max_k=max_k, seed=seed)
        if len(np.unique(labels)) < 2:
            return 0.0
        return _compute_silhouette(X, labels, seed=seed)
    except Exception as e:
        logger.warning("Policy %r eval failed: %s", policy, e)
        return -1.0


def _stability_score(
    policy: str, X: np.ndarray, cfg: dict, n_bootstrap: int = 5
) -> float:
    """Mean silhouette over n_bootstrap random seeds (higher = more stable)."""
    if len(X) < 4 or not HAS_SKLEARN:
        return 0.0
    scores = [
        s for seed in range(n_bootstrap)
        if math.isfinite(s := _eval_policy_silhouette(policy, X, cfg, seed=seed))
    ]
    return float(np.mean(scores)) if scores else 0.0


def select_best_policy(
    candidates: List[str], X: np.ndarray, cfg: dict
) -> Tuple[str, float, List[dict]]:
    """
    Evaluate all candidates, select best by silhouette.
    |Δ(1st, 2nd)| < stability_margin → stability tie-break.
    Returns (best_policy, stability_margin, evaluation_results).
    """
    eps = float(cfg["policy"].get("stability_margin", 0.05))
    n_bs = int(cfg["policy"].get("n_bootstrap", 5))

    results = [
        {"policy": p, "silhouette": _eval_policy_silhouette(p, X, cfg)}
        for p in candidates
    ]
    ranked = sorted(results, key=lambda r: r["silhouette"], reverse=True)
    best = ranked[0]
    margin = 0.0

    if len(ranked) >= 2:
        margin = best["silhouette"] - ranked[1]["silhouette"]
        if abs(margin) < eps:
            top2 = ranked[:2]
            for r in top2:
                r["stability"] = _stability_score(r["policy"], X, cfg, n_bootstrap=n_bs)
            best = max(top2, key=lambda r: r.get("stability", 0.0))
            logger.info(
                "Boundary case |Δ|=%.4f < ε=%.3f → stability tie-break → %s",
                margin, eps, best["policy"],
            )

    return best["policy"], margin, results


def run_meta_policy_generator(phase0: dict, mci: float, cfg: dict) -> dict:
    """
    Full Meta Policy Generator:
    Distribution Diagnostics → Candidate Pruning → Empirical Eval → Best Policy
    """
    dist_analysis: dict = phase0.get("dist_analysis", {})
    morph_X: np.ndarray = phase0.get("morph_X", np.empty((0, 6)))
    context_X: np.ndarray = phase0.get("context_X", np.empty((0, 5)))

    morph_Xn = _normalize_array(morph_X, "zscore") if len(morph_X) > 0 else morph_X
    ctx_Xn = _normalize_array(context_X, "zscore") if len(context_X) > 0 else context_X

    m_cands = _get_candidate_policies(dist_analysis, mci, cfg)
    m_best, m_margin, m_results = select_best_policy(m_cands, morph_Xn, cfg)

    # Context candidates are independent of morphology valley counts.
    # Use fixed set: GMM (density-aware) vs percentile (simple baseline).
    max_c = int(cfg["policy"].get("max_candidates", 3))
    c_cands = ["gmm", "percentile"][:max_c]
    c_best, _, c_results = select_best_policy(c_cands, ctx_Xn, cfg)

    logger.info("Morphology policy: %s (candidates: %s)", m_best, m_cands)
    logger.info("Context policy:    %s (candidates: %s)", c_best, c_cands)

    return {
        "morphology_policy":  m_best,
        "context_policy":     c_best,
        "stability_margin":   m_margin,
        "candidate_policies": {"morphology": m_cands, "context": c_cands},
        "evaluation_results": (
            [{**r, "axis": "morphology"} for r in m_results]
            + [{**r, "axis": "context"} for r in c_results]
        ),
    }


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def run(
    profiling_dir: str,
    output_dir: str,
    cfg: dict,
    weight_mode: Optional[str] = None,
    local_staging: bool = False,
    staging_root: Optional[str] = None,
) -> dict:
    """
    Full Step 1 pipeline:
    load Phase 0 outputs → compute MCI/CCI → Meta Policy → save report.

    local_staging: if True, copy Phase 0 files to local temp dir first
                   (reduces Drive read latency in Colab).
    staging_root:  override staging base directory (default /content/tmp/aroma_staging).
    """
    if local_staging:
        profiling_dir = _stage_to_local(profiling_dir, staging_root=staging_root)
    logger.info("Loading Phase 0 outputs from: %s", profiling_dir)
    phase0 = load_phase0_outputs(profiling_dir)

    if phase0["status"] not in ("ok", "empty_context"):
        logger.error("Cannot proceed: Phase 0 status = %s", phase0["status"])
        report: dict = {
            "status":             phase0["status"],
            "profiling_dir":      str(profiling_dir),
            "mci":                None,
            "cci":                None,
            "morphology_policy":  None,
            "context_policy":     None,
            "stability_margin":   None,
            "weight_mode":        weight_mode,
            "mci_components":     None,
            "cci_components":     None,
            "candidate_policies": None,
            "evaluation_results": None,
        }
        return report

    if phase0["status"] == "empty_context":
        logger.warning("Context features empty — CCI will be 0.0")

    mci, mci_components = compute_mci(phase0, cfg, weight_mode=weight_mode)
    cci, cci_components = compute_cci(phase0, cfg, weight_mode=weight_mode)
    policy = run_meta_policy_generator(phase0, mci, cfg)

    logger.info(
        "MCI=%.4f  CCI=%.4f  morph_policy=%s  ctx_policy=%s",
        mci, cci, policy["morphology_policy"], policy["context_policy"],
    )

    report = {
        "mci":               mci,
        "cci":               cci,
        "morphology_policy": policy["morphology_policy"],
        "context_policy":    policy["context_policy"],
        "stability_margin":  policy["stability_margin"],
        "weight_mode":       weight_mode or cfg["mci"]["weights"],
        "mci_components":    mci_components,
        "cci_components":    cci_components,
        "candidate_policies": policy["candidate_policies"],
        "evaluation_results": policy["evaluation_results"],
        "profiling_dir":     str(profiling_dir),
        "status":            phase0["status"],
        "provenance": {
            "io_source":  _IO_SOURCE,
            "ref_source": _REF_SOURCE,
        },
    }

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    save_json(report, str(out_path / "complexity_report.json"))
    logger.info("Saved complexity_report.json → %s", out_path)

    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="AROMA Step 1 — Complexity Analysis (MCI/CCI + Meta Policy)"
    )
    p.add_argument(
        "--profiling_dir", required=True,
        help="Phase 0 output directory (contains distribution_analysis.json etc.)",
    )
    p.add_argument(
        "--output_dir", required=True,
        help="Directory to write complexity_report.json",
    )
    p.add_argument(
        "--config", default=None,
        help="Path to aroma_step1.yaml (default: scripts/aroma/config/aroma_step1.yaml)",
    )
    p.add_argument(
        "--weight_mode", choices=list(_WEIGHT_PRESETS), default=None,
        help="Weight preset for MCI/CCI ablation (overrides config)",
    )
    p.add_argument(
        "--local_staging", action="store_true", default=False,
        help="Copy input files to /content/tmp/aroma_staging/ before processing "
             "(reduces Google Drive read latency in Colab)",
    )
    p.add_argument(
        "--staging_root", default=None,
        help="Override local staging base directory (default: /content/tmp/aroma_staging)",
    )
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = _parse_args(argv)
    cfg = load_config(args.config)
    run(
        profiling_dir=args.profiling_dir,
        output_dir=args.output_dir,
        cfg=cfg,
        weight_mode=args.weight_mode,
        local_staging=args.local_staging,
        staging_root=args.staging_root,
    )


if __name__ == "__main__":
    main()
