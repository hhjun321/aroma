#!/usr/bin/env python3
"""AROMA self-contained ControlNet train.jsonl + hint/target builder (P0).

This is the AROMA-INTERNAL replacement for the CASDA-routed data path. Instead of
converting AROMA ROIs into a CASDA roi_metadata.csv and letting CASDA's
prepare_controlnet_data.py / ControlNetDatasetPackager regenerate hints+prompts,
this builder produces, ENTIRELY WITHIN AROMA, the exact artifacts that
scripts/train_controlnet.py consumes:

    {output_dir}/targets/  — bbox-cropped defect target PNGs
    {output_dir}/hints/    — 3-channel ControlNet conditioning hint PNGs
    {output_dir}/train.jsonl — one JSON object per real defect

per-real-defect loop: morphology row → crop(target) → hint → prompt → jsonl line.

Everything category-shaped is DATA-DRIVEN (no hardcoded category strings, no
magic thresholds):
  * pool             = morphology_features.csv rows (1 row = 1 real defect).
  * defect_subtype   = morph_label, joined from roi_candidates.json by
                       (image_id, defect_mask_path).
  * stability_score  = ctx_prior from roi_candidates (MAX across candidate cells).
  * background_type  = derived from that image's context_features.csv defect
                       patches, binned with bin_edges loaded from
                       recommended_config.yaml (context.bin_edges).

CAVEAT — background_type join granularity:
  context_features.csv is keyed by BARE image_id ('000'..) with NO defect_type
  column. In datasets where one image_id index is reused across several defect
  types (e.g. mvtec_cable: bent_wire/000.png, cable_swap/000.png, ... all share
  image_id '000'), a single context image_id bucket pools patches from every
  physical image sharing that index. `_load_context_means` therefore averages
  background context ACROSS those physical images, so background_type is
  per-image_id-BUCKET, not strictly per-defect's-own-physical-image. This cannot
  be disambiguated from context_features.csv alone (it carries no defect_type).
  build() emits a startup WARN quantifying the collision so the limitation is
  visible rather than silent, and logs the final background_type histogram so a
  degenerate all-one-category outcome (bins saturated by whole-image defect-patch
  means) is observable.

train.jsonl line schema (exactly what train_controlnet.py's dataset reads):
    {"target": <abs PNG>, "hint": <abs PNG>, "prompt": <str>,
     "negative_prompt": <str>, "source": <abs PNG == target>}
Only target/hint/prompt are read at train time; negative_prompt/source are legacy
schema fields (recorded, not used by the loader).

Reused (NOT reimplemented):
  * utils/hint_generator.py::HintImageGenerator  — 3-channel hint algorithm.
  * utils/prompt_generator.py::PromptGenerator   — per-defect prompt + negative.
  * scripts/aroma/aroma_to_casda_roi.py          — crop/morphology helpers
    (_make_crop_pair, _crop_xywh, _read_gray_or_color, _as_xywh, _remap_root)
    and its _bootstrap_aroma_ref / sys.path pattern.

Usage (CLI):
    python build_train_jsonl.py \
        --morphology_csv   /path/to/morphology_features.csv \
        --roi_candidates   /path/to/roi_candidates.json \
        --context_features /path/to/context_features.csv \
        --config           /path/to/recommended_config.yaml \
        --output_dir       /path/to/domain_out \
        [--image_root ...] [--style technical]

Usage (import):
    from build_train_jsonl import build
    n = build(morphology_csv=..., roi_candidates=..., context_features=...,
              config_yaml=..., output_dir=...)
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("aroma.build_train_jsonl")


# ---------------------------------------------------------------------------
# I/O bootstrap — same pattern as sibling scripts. Puts AROMA_REF (repo root,
# default D:\project\aroma) on sys.path so `from utils.* import ...` resolves,
# AND ensures this script's own dir (scripts/aroma) is importable so the sibling
# `from aroma_to_casda_roi import ...` resolves on Colab AND local.
# ---------------------------------------------------------------------------

def _bootstrap_aroma_ref() -> None:
    ref = os.environ.get("AROMA_REF") or r"D:\project\aroma"
    ref_path = Path(ref)
    if ref_path.is_dir() and str(ref_path) not in sys.path:
        sys.path.insert(0, str(ref_path))
    # aroma_to_casda_roi's own _bootstrap_aroma_ref does NOT add this dir; add it
    # ourselves so the sibling import below works regardless of cwd / launcher.
    here = Path(__file__).resolve().parent
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))


_bootstrap_aroma_ref()

# Reused crop / morphology helpers from the sibling adapter. Importing it also
# triggers its own _bootstrap_aroma_ref() at import time (idempotent).
from aroma_to_casda_roi import (  # noqa: E402
    _as_xywh,
    _crop_xywh,
    _make_crop_pair,
    _read_gray_or_color,
    _remap_root,
    HAS_CV2,
    HAS_NUMPY,
    HAS_PIL,
)

# Reused AROMA hint + prompt generators (algorithms live here; do NOT reimplement).
from utils.hint_generator import HintImageGenerator  # noqa: E402
from utils.prompt_generator import PromptGenerator  # noqa: E402


# Morphology metrics fed to hint (R channel) + prompt. Keys read by
# HintImageGenerator.generate_red_channel (linearity, solidity) and
# PromptGenerator.generate_technical_prompt (linearity, solidity, aspect_ratio).
_METRIC_KEYS: Tuple[str, ...] = (
    "linearity", "solidity", "extent", "aspect_ratio",
    "eccentricity", "circularity", "area",
)


# ---------------------------------------------------------------------------
# Config / bin_edges loading (data-driven thresholds — never literals)
# ---------------------------------------------------------------------------

def _load_bin_edges(config_yaml: str) -> Tuple[List[str], Dict[str, List[float]]]:
    """Load context feature order + per-feature bin_edges from recommended_config.yaml.

    Returns (feature_order, bin_edges) where feature_order is context.features
    (ordered) and bin_edges maps feature -> [edge0, edge1] (=> 3 ordinal bins).
    Raises FileNotFoundError / ValueError on a missing or malformed config.
    """
    cpath = Path(config_yaml)
    if not cpath.exists():
        raise FileNotFoundError(f"config not found: {config_yaml}")
    try:
        import yaml  # lazy; present on Colab
    except Exception as e:  # noqa: BLE001
        raise ValueError(
            "PyYAML is required to read recommended_config.yaml (pip install pyyaml)"
        ) from e
    with open(cpath, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    ctx = cfg.get("context") or {}
    features = list(ctx.get("features") or [])
    raw_edges = ctx.get("bin_edges") or {}
    if not features or not raw_edges:
        raise ValueError(
            f"config missing context.features / context.bin_edges: {config_yaml}"
        )
    bin_edges: Dict[str, List[float]] = {}
    for feat in features:
        edges = raw_edges.get(feat)
        if not edges:
            raise ValueError(f"config context.bin_edges missing feature '{feat}'")
        bin_edges[feat] = [float(e) for e in edges]
    return features, bin_edges


def _bin_value(value: float, edges: List[float]) -> int:
    """Ordinal bin (0/1/2 for 2 edges) for a value against ascending edges.

    bin = count of edges the value is >= (data-driven; edges come from config).
    """
    b = 0
    for e in edges:
        if value >= e:
            b += 1
        else:
            break
    return b


# ---------------------------------------------------------------------------
# context_features.csv aggregation (per image_id, defect patches only)
# ---------------------------------------------------------------------------

def _load_context_means(
    context_csv: str, features: List[str]
) -> Dict[str, Dict[str, float]]:
    """Mean of each context feature over an image_id bucket's DEFECT patches.

    Reads context_features.csv, keeps rows with image_type == 'defect', and
    averages each of `features` per image_id. Returns {image_id: {feature: mean}}.
    Returns {} (with WARN) if the file is missing. Non-numeric cells are skipped
    per-feature.

    NOTE: the key is the BARE image_id from context_features.csv, which has no
    defect_type column. If several physical images share an image_id index, this
    mean pools all of them (see module docstring CAVEAT). build() emits a startup
    WARN quantifying any such collision; this function cannot detect it alone.
    """
    means: Dict[str, Dict[str, float]] = {}
    cpath = Path(context_csv)
    if not cpath.exists():
        logger.warning("context_features not found: %s — background_type undetermined", context_csv)
        return means
    sums: Dict[str, Dict[str, float]] = defaultdict(lambda: {f: 0.0 for f in features})
    counts: Dict[str, Dict[str, int]] = defaultdict(lambda: {f: 0 for f in features})
    with open(cpath, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if (row.get("image_type") or "").strip() != "defect":
                continue
            img_id = (row.get("image_id") or "").strip()
            if not img_id:
                continue
            for feat in features:
                raw = row.get(feat)
                if raw is None or raw == "":
                    continue
                try:
                    val = float(raw)
                except (ValueError, TypeError):
                    continue
                sums[img_id][feat] += val
                counts[img_id][feat] += 1
    for img_id, feat_sums in sums.items():
        per_feat: Dict[str, float] = {}
        for feat in features:
            c = counts[img_id][feat]
            if c > 0:
                per_feat[feat] = feat_sums[feat] / c
        if per_feat:
            means[img_id] = per_feat
    logger.info("Loaded context defect-patch means for %d image_ids", len(means))
    return means


def _warn_context_collision(
    pool: List[Dict[str, str]],
    context_means: Dict[str, Dict[str, float]],
) -> None:
    """WARN if a context image_id bucket pools >1 physical image.

    context_features.csv keys background context by bare image_id with no
    defect_type. morphology_features.csv, in contrast, keys each physical image by
    (image_id, defect_type). When a single image_id maps to multiple defect_types
    in the morphology pool, the matching context bucket is an average across all
    those physical images, so background_type is per-image_id-bucket rather than
    per-defect's-own-image. This surfaces that data limitation loudly at startup
    instead of letting it stay silent. No effect on outputs (observability only).
    """
    types_per_id: Dict[str, set] = defaultdict(set)
    for row in pool:
        iid = (row.get("image_id") or "").strip()
        dtype = (row.get("defect_type") or "").strip()
        if iid and dtype:
            types_per_id[iid].add(dtype)
    colliding = {
        iid: sorted(ts)
        for iid, ts in types_per_id.items()
        if len(ts) > 1 and iid in context_means
    }
    if not colliding:
        return
    max_n = max(len(ts) for ts in colliding.values())
    logger.warning(
        "context join collision: %d/%d context image_id bucket(s) pool patches "
        "from >1 physical image (up to %d defect_types share one image_id). "
        "context_features.csv has no defect_type column, so background_type for "
        "these is a per-image_id-bucket average ACROSS defect types, NOT the "
        "defect's own physical image. Example image_id=%s <- %s",
        len(colliding), len(context_means), max_n,
        next(iter(colliding)), colliding[next(iter(colliding))],
    )


def _derive_background_type(
    image_id: str,
    context_means: Dict[str, Dict[str, float]],
    features: List[str],
    bin_edges: Dict[str, List[float]],
) -> Optional[str]:
    """Data-driven background_type for a defect's own image.

    Bins the image's mean defect-patch context features via config bin_edges,
    then applies the ordinal-bin policy (dev_note §_derive_background_type):
      orientation_consistency bin==2               -> 'directional'
      elif texture_entropy bin==2 OR local_variance bin==2 -> 'complex_pattern'
      elif edge_density bin==0 AND local_variance bin==0   -> 'smooth'
      else                                          -> 'complex_pattern'
    Thresholds come ONLY from bin_edges (data). Returns None if the image has no
    defect-patch context (caller skips + counts).
    """
    per_feat = context_means.get(image_id)
    if not per_feat:
        return None
    bins: Dict[str, int] = {}
    for feat in features:
        if feat in per_feat and feat in bin_edges:
            bins[feat] = _bin_value(per_feat[feat], bin_edges[feat])
    orient = bins.get("orientation_consistency")
    texture = bins.get("texture_entropy")
    local_var = bins.get("local_variance")
    edge = bins.get("edge_density")

    if orient == 2:
        return "directional"
    if texture == 2 or local_var == 2:
        return "complex_pattern"
    if edge == 0 and local_var == 0:
        return "smooth"
    return "complex_pattern"


# ---------------------------------------------------------------------------
# roi_candidates.json join (defect_subtype + stability_score sources)
# ---------------------------------------------------------------------------

def _load_candidate_join(
    roi_candidates: str,
) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """Index roi_candidates.json by (image_id, defect_mask_path).

    roi_candidates.json is a LIST; a defect typically appears in >1 entry (one per
    context cell). Aggregates per (image_id, defect_mask_path):
      * ctx_prior     = MAX across the defect's candidate cells (stability source)
      * morph_label   = first non-empty label seen (defect_subtype source)
      * cluster_id    = first seen (identity)
    Returns {(image_id, defect_mask_path): {ctx_prior, morph_label, cluster_id}}.
    Raises FileNotFoundError / ValueError on missing / malformed file.
    """
    rpath = Path(roi_candidates)
    if not rpath.exists():
        raise FileNotFoundError(f"roi_candidates not found: {roi_candidates}")
    with open(rpath, encoding="utf-8") as f:
        cands = json.load(f)
    if not isinstance(cands, list):
        raise ValueError(
            f"roi_candidates.json must be a list, got {type(cands).__name__}"
        )
    joined: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for c in cands:
        if not isinstance(c, dict):
            continue
        img_id = str(c.get("image_id", "")).strip()
        mask = str(c.get("defect_mask_path", "")).strip()
        if not img_id or not mask:
            continue
        key = (img_id, mask)
        try:
            ctx_prior = float(c.get("ctx_prior"))
        except (ValueError, TypeError):
            ctx_prior = None
        cur = joined.get(key)
        if cur is None:
            cur = {"ctx_prior": None, "morph_label": "", "cluster_id": None}
            joined[key] = cur
        if ctx_prior is not None:
            if cur["ctx_prior"] is None or ctx_prior > cur["ctx_prior"]:
                cur["ctx_prior"] = ctx_prior
        if not cur["morph_label"]:
            lbl = str(c.get("morph_label", "")).strip()
            if lbl:
                cur["morph_label"] = lbl
        if cur["cluster_id"] is None and c.get("cluster_id") is not None:
            cur["cluster_id"] = c.get("cluster_id")
    logger.info("Loaded candidate join for %d (image_id,mask) keys", len(joined))
    return joined


# ---------------------------------------------------------------------------
# morphology pool loading
# ---------------------------------------------------------------------------

def _load_morphology_pool(morphology_csv: str) -> List[Dict[str, str]]:
    """Load morphology_features.csv rows in file order (deterministic pool).

    Returns the list of row dicts. Raises FileNotFoundError if missing.
    """
    mpath = Path(morphology_csv)
    if not mpath.exists():
        raise FileNotFoundError(f"morphology_csv not found: {morphology_csv}")
    with open(mpath, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _metrics_from_row(row: Dict[str, str]) -> Dict[str, float]:
    """Extract numeric defect_metrics from a morphology row for hint + prompt."""
    metrics: Dict[str, float] = {}
    for key in _METRIC_KEYS:
        raw = row.get(key)
        if raw is None or raw == "":
            continue
        try:
            metrics[key] = float(raw)
        except (ValueError, TypeError):
            continue
    return metrics


# ---------------------------------------------------------------------------
# Core builder
# ---------------------------------------------------------------------------

def build(
    morphology_csv: str,
    roi_candidates: str,
    context_features: str,
    config_yaml: str,
    output_dir: str,
    image_root: Optional[str] = None,
    style: str = "technical",
) -> int:
    """Build one domain's ControlNet train.jsonl + target/hint PNGs.

    Pool = morphology_features.csv rows (1 row = 1 real defect). Per defect:
      1. join roi_candidates by (image_id, defect_mask_path)
         -> defect_subtype (morph_label), stability_score (ctx_prior MAX)
      2. derive background_type from the image's context defect patches + bin_edges
      3. crop target (image+mask) to defect_bbox via _make_crop_pair
      4. build 3-channel hint from the SAME crop via HintImageGenerator
      5. build prompt (technical, deterministic) + negative via PromptGenerator
      6. append a train.jsonl line with ABSOLUTE paths

    Returns the number of train.jsonl lines written.

    Raises:
        FileNotFoundError / ValueError: on missing or malformed required inputs.
        RuntimeError: when zero valid samples are produced (empty result guard).
    """
    if not HAS_NUMPY or not (HAS_CV2 or HAS_PIL):
        logger.warning(
            "numpy + (cv2 or PIL) required for crops/hints; environment lacks them. "
            "Every defect will be skipped (no crash). HAS_NUMPY=%s HAS_CV2=%s HAS_PIL=%s",
            HAS_NUMPY, HAS_CV2, HAS_PIL,
        )

    features, bin_edges = _load_bin_edges(config_yaml)
    candidate_join = _load_candidate_join(roi_candidates)
    context_means = _load_context_means(context_features, features)
    pool = _load_morphology_pool(morphology_csv)

    # Startup observability: surface the image_id-only context join limitation
    # (background_type is a per-image_id-bucket average when >1 physical image
    # shares an image_id index; see module docstring CAVEAT).
    _warn_context_collision(pool, context_means)

    out_root = Path(output_dir)
    targets_dir = out_root / "targets"
    hints_dir = out_root / "hints"
    targets_dir.mkdir(parents=True, exist_ok=True)
    hints_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_root / "train.jsonl"

    hint_gen = HintImageGenerator()
    prompt_gen = PromptGenerator(style=style)
    negative_prompt = prompt_gen.generate_negative_prompt()

    n_in = len(pool)
    n_written = 0
    n_no_mask = 0
    n_skipped = 0        # crop failure (missing / unreadable / degenerate)
    n_join_miss = 0      # roi_candidates join miss
    n_no_context = 0     # context / background_type undetermined
    region_counter: Dict[str, int] = {}
    bg_type_hist: Dict[str, int] = defaultdict(int)  # background_type distribution

    with open(jsonl_path, "w", encoding="utf-8") as jf:
        for row in pool:
            image_id = (row.get("image_id") or "").strip()
            image_path = _remap_root((row.get("image_path") or "").strip(), image_root)
            mask_path = _remap_root((row.get("defect_mask_path") or "").strip(), image_root)

            # Empty mask row (distribution_profiling save-failure) -> cannot crop.
            if not mask_path:
                n_no_mask += 1
                logger.warning("empty defect_mask_path for image_id=%s — skip", image_id or "?")
                continue

            bbox = _as_xywh(row.get("defect_bbox"))
            if bbox is None:
                n_skipped += 1
                logger.warning("missing/invalid defect_bbox for image_id=%s — skip", image_id or "?")
                continue

            # Candidate join -> defect_subtype (morph_label), stability (ctx_prior MAX).
            join = candidate_join.get((image_id, (row.get("defect_mask_path") or "").strip()))
            if join is None:
                n_join_miss += 1
                logger.warning("no roi_candidate join for (image_id=%s, mask=%s) — skip",
                               image_id or "?", mask_path)
                continue
            defect_subtype = join.get("morph_label") or ""
            if not defect_subtype:
                n_join_miss += 1
                logger.warning("empty morph_label for image_id=%s — skip", image_id or "?")
                continue
            ctx_prior = join.get("ctx_prior")
            stability_score = float(ctx_prior) if ctx_prior is not None else 0.5

            # Data-driven background_type from this image's context defect patches.
            background_type = _derive_background_type(image_id, context_means, features, bin_edges)
            if background_type is None:
                n_no_context += 1
                logger.warning("no defect-patch context for image_id=%s — skip", image_id or "?")
                continue

            metrics = _metrics_from_row(row)

            region_id = region_counter.get(image_id, 0)
            base = f"{image_id}_{defect_subtype}_region{region_id}"

            # ---- target crop (image + mask, same bbox) ---------------------
            pair = _make_crop_pair(image_path, mask_path, bbox, str(targets_dir), base)
            if pair is None:
                n_skipped += 1  # WARN already emitted by _make_crop_pair
                continue
            target_path, target_mask_path = pair

            # ---- hint from the SAME crop arrays (bbox-aligned) -------------
            roi_image = _read_gray_or_color(image_path, gray=False)
            roi_mask = _read_gray_or_color(mask_path, gray=True)
            if roi_image is None or roi_mask is None:
                n_skipped += 1
                logger.warning("failed to reload image/mask for hint (image_id=%s) — skip", image_id or "?")
                continue
            roi_image_crop = _crop_xywh(roi_image, bbox)
            roi_mask_crop = _crop_xywh(roi_mask, bbox)
            if roi_image_crop is None or roi_mask_crop is None:
                n_skipped += 1
                logger.warning("degenerate hint crop (image_id=%s) — skip", image_id or "?")
                continue

            try:
                hint = hint_gen.generate_hint_image(
                    roi_image_crop, roi_mask_crop, metrics, background_type, stability_score
                )
            except Exception as e:  # noqa: BLE001
                n_skipped += 1
                logger.warning("hint generation failed (image_id=%s): %s — skip", image_id or "?", e)
                continue

            hint_path = hints_dir / f"{base}_hint.png"
            try:
                hint_gen.save_hint_image(hint, hint_path)
            except Exception as e:  # noqa: BLE001
                n_skipped += 1
                logger.warning("hint save failed (image_id=%s): %s — skip", image_id or "?", e)
                continue

            # ---- prompt (technical: deterministic; no random.choice) -------
            prompt = prompt_gen.generate_prompt(
                defect_subtype=defect_subtype,
                background_type=background_type,
                stability_score=stability_score,
                defect_metrics=metrics,
                suitability_score=stability_score,
            )

            target_abs = str(Path(target_path).resolve())
            hint_abs = str(hint_path.resolve())
            line = {
                "target": target_abs,
                "hint": hint_abs,
                "prompt": prompt,
                "negative_prompt": negative_prompt,  # legacy: recorded, unused by loader
                "source": target_abs,                # legacy: duplicate of target
            }
            jf.write(json.dumps(line, ensure_ascii=False) + "\n")
            n_written += 1
            bg_type_hist[background_type] += 1
            region_counter[image_id] = region_id + 1

    # Ordered, deterministic histogram of emitted background_type values.
    bg_hist_sorted = dict(sorted(bg_type_hist.items(), key=lambda kv: (-kv[1], kv[0])))
    logger.info(
        "build_train_jsonl: n_in=%d, n_written=%d, n_no_mask=%d, n_join_miss=%d, "
        "n_no_context=%d, n_skipped=%d, background_type_hist=%s → %s",
        n_in, n_written, n_no_mask, n_join_miss, n_no_context, n_skipped,
        bg_hist_sorted, jsonl_path,
    )
    # Degenerate-distribution guard: background_type is meant to carry per-defect
    # signal, but whole-image defect-patch means can saturate the bins so every
    # sample collapses to one category (config bin_edges are tertiles over the
    # same defect-patch population). WARN rather than silently emit a constant.
    if n_written > 0 and len(bg_type_hist) <= 1:
        only = next(iter(bg_type_hist), None)
        logger.warning(
            "background_type collapsed to a single category (%s) for all %d "
            "written samples — it carries no per-defect signal. Likely the "
            "whole-image defect-patch aggregation saturates the config bin_edges "
            "(defects present => high local_variance/edge_density bins), leaving "
            "'smooth'/'directional' unreachable. Consider aggregating over "
            "GOOD/background or bbox-local patches, or revisiting the policy "
            "thresholds.",
            only, n_written,
        )

    if n_written == 0:
        raise RuntimeError(
            f"No valid training samples produced from {n_in} morphology rows "
            f"(n_no_mask={n_no_mask}, n_join_miss={n_join_miss}, "
            f"n_no_context={n_no_context}, n_skipped={n_skipped}). "
            f"train.jsonl at {jsonl_path} is empty — check inputs "
            f"(cv2/PIL+numpy availability, source image/mask paths, joins)."
        )
    return n_written


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build AROMA self-contained ControlNet train.jsonl + hint/target PNGs"
    )
    p.add_argument("--morphology_csv", required=True,
                   help="Path to morphology_features.csv (defect pool; 1 row = 1 defect)")
    p.add_argument("--roi_candidates", required=True,
                   help="Path to roi_candidates.json (morph_label + ctx_prior join)")
    p.add_argument("--context_features", required=True,
                   help="Path to context_features.csv (background_type derivation)")
    p.add_argument("--config", required=True, dest="config_yaml",
                   help="Path to recommended_config.yaml (context.bin_edges)")
    p.add_argument("--output_dir", required=True,
                   help="Output dir; writes targets/, hints/, train.jsonl")
    p.add_argument("--image_root", default=None,
                   help="Optional prefix to re-root image_path/defect_mask_path (default none)")
    p.add_argument("--style", default="technical",
                   help="PromptGenerator style: technical (default, deterministic) / simple / detailed")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = _parse_args(argv)
    try:
        n = build(
            morphology_csv=args.morphology_csv,
            roi_candidates=args.roi_candidates,
            context_features=args.context_features,
            config_yaml=args.config_yaml,
            output_dir=args.output_dir,
            image_root=args.image_root,
            style=args.style,
        )
        print(f"Wrote {n} train.jsonl lines → {Path(args.output_dir) / 'train.jsonl'}")
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        logger.error("%s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
