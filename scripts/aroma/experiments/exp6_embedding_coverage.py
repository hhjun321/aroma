#!/usr/bin/env python3
"""
AROMA Exp 6 — 임베딩 커버리지 평가 (CPU급, exp5 임베딩 캐시 공유)

두 모드 (low_compute_validation_plan.md 2·3순위):

--mode knn   : kNN test-coverage — held-out val 결함 crop 별로 학습 풀
               (real / real+random / real+aroma)까지의 최근접 cosine distance
               분포 비교. 검정 = 이미지 단위 clustered bootstrap
               (같은 이미지 crop 은 비독립 — 인스턴스 단위 검정 금지).
               diversity 항 병기 (중복 합성의 coverage 부풀림 차단).

--mode rare  : rare-mode hit rate — 후보 소스 crop 의 독립 임베딩(DINOv2)
               k-means 모드에서 rare 모드(빈도 p25 이하 AND val 등장)에 대한
               aroma 선택 hit rate 를 random 재선택 30-seed null 분포 대비로
               검정. k×clustering-seed sensitivity 전 그리드 보고
               (cherry-pick 금지). JSD-to-real 미사용 (rare 과대표집이 목적
               — real 빈도 근접을 성공으로 놓으면 내부 모순).

정보 분리: AROMA 선택은 val/test 를 보지 않는다 (선택 입력 = train 결함
프로파일링). val 기준 평가와 선택 알고리즘은 정보적으로 분리돼 있다.

Usage (Colab):
    !python $AROMA_SCRIPTS/experiments/exp6_embedding_coverage.py \\
        --mode knn \\
        --real_data_dir $AROMA_DATA \\
        --aroma_synthetic_dir  $AROMA_OUT/synthetic \\
        --random_synthetic_dir $AROMA_OUT/synthetic_random \\
        --dataset_keys severstal mvtec_leather aitex mtd \\
        --embed_cache_dir $AROMA_OUT/embed_cache \\
        --output_dir $AROMA_OUT/exp6

    !python $AROMA_SCRIPTS/experiments/exp6_embedding_coverage.py \\
        --mode rare \\
        --real_data_dir $AROMA_DATA \\
        --roi_dir_root $AROMA_OUT/roi \\
        --dataset_keys severstal mvtec_leather aitex mtd \\
        --kmeans_k 8 10 12 15 --cluster_seeds 0 1 2 3 4 \\
        --embed_cache_dir $AROMA_OUT/embed_cache \\
        --output_dir $AROMA_OUT/exp6
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("aroma.exp6")

_THIS_DIR = Path(__file__).resolve().parent
_AROMA_SCRIPTS = _THIS_DIR.parent


def _bootstrap() -> None:
    aroma_ref = os.environ.get("AROMA_REF") or r"D:\project\aroma"
    for p in (str(_THIS_DIR), str(_AROMA_SCRIPTS), str(Path(aroma_ref))):
        if p not in sys.path:
            sys.path.insert(0, p)


_bootstrap()

# exp5/exp3 재사용 — 둘 다 __main__ 가드 有, top-level 부작용은 로깅/sys.path 뿐.
from exp5_prdc import (  # noqa: E402
    _build_backbone,
    _cached_embed,
    _embed_crops,
    _equalize_n,
    _load_real_defect_crops,
    _load_synth_crops,
    _split_defects,
)
from exp3_generation_quality import _get_image_lists, _mask_to_bbox  # noqa: E402

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
# 공유 골격
# ---------------------------------------------------------------------------

def _cosine_dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """L2-normalize 후 1 − cosine similarity. shape (n_a, n_b).

    retrieval 관례 = cosine (exp5 PRDC 의 유클리드와 의도적으로 다름 — meta 명시).
    """
    an = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)
    bn = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-12)
    return 1.0 - an @ bn.T


def _img_id_of(crop_id: str) -> str:
    """crop_id (`{img_path}|{coords}`) → 이미지 경로 (clustered bootstrap 단위)."""
    return crop_id.rsplit("|", 1)[0]


def _min_dist_to_pool(val_feat: np.ndarray, pool_feat: np.ndarray,
                      chunk: int = 512) -> np.ndarray:
    """val crop 별 풀 최근접 cosine distance d1. 청크 처리로 메모리 방어."""
    d1 = np.full(val_feat.shape[0], np.inf)
    for s in range(0, pool_feat.shape[0], chunk):
        d = _cosine_dist(val_feat, pool_feat[s:s + chunk])
        d1 = np.minimum(d1, d.min(axis=1))
    return d1


# ---------------------------------------------------------------------------
# knn 모드
# ---------------------------------------------------------------------------

def _clustered_bootstrap(
    d1_random_pool: np.ndarray,
    d1_aroma_pool: np.ndarray,
    img_ids: List[str],
    reps: int,
    seed: int,
) -> Dict[str, Any]:
    """이미지 단위 재표집 bootstrap — Δ = mean d1(real+random) − mean d1(real+aroma).

    Δ > 0 = aroma 풀이 val 을 더 가깝게 커버. 방향 가설의 bootstrap p =
    (count(Δ_boot ≤ 0)+1)/(reps+1).
    """
    uniq = sorted(set(img_ids))
    by_img: Dict[str, List[int]] = {}
    for i, g in enumerate(img_ids):
        by_img.setdefault(g, []).append(i)
    obs = float(d1_random_pool.mean() - d1_aroma_pool.mean())
    rng = np.random.default_rng(seed)
    deltas = np.empty(reps)
    for r in range(reps):
        pick = rng.choice(len(uniq), size=len(uniq), replace=True)
        idx = np.fromiter(
            (i for p in pick for i in by_img[uniq[p]]), dtype=np.int64)
        deltas[r] = d1_random_pool[idx].mean() - d1_aroma_pool[idx].mean()
    p_boot = (int(np.sum(deltas <= 0.0)) + 1) / (reps + 1)
    return {
        "delta": round(obs, 6),
        "ci95": [round(float(np.percentile(deltas, 2.5)), 6),
                 round(float(np.percentile(deltas, 97.5)), 6)],
        "p_boot_one_sided": round(p_boot, 6),
        "n_images": len(uniq),
    }


def _coverage_auc(d1: np.ndarray, r_grid: np.ndarray) -> float:
    """coverage curve C(r) = P(d1 ≤ r) 의 grid 평균 (AUC 요약)."""
    return float(np.mean([(d1 <= r).mean() for r in r_grid]))


def _diversity_stats(feat: np.ndarray, n_distinct_sources: Optional[int]) -> Dict[str, Any]:
    """synth 풀 내 pairwise cosine distance 요약 — coverage 부풀림(중복 합성) 감시."""
    if feat.shape[0] < 2:
        return {"pairwise_mean": None, "pairwise_median": None,
                "n": int(feat.shape[0]), "n_distinct_sources": n_distinct_sources}
    d = _cosine_dist(feat, feat)
    iu = np.triu_indices(feat.shape[0], k=1)
    vals = d[iu]
    return {
        "pairwise_mean": round(float(vals.mean()), 6),
        "pairwise_median": round(float(np.median(vals)), 6),
        "n": int(feat.shape[0]),
        "n_distinct_sources": n_distinct_sources,
    }


def _distinct_sources(annotations_path: Path) -> Optional[int]:
    try:
        anns = load_json(str(annotations_path))
        srcs = {a.get("source_roi") for a in anns
                if not a.get("dry_run") and a.get("source_roi")}
        return len(srcs)
    except Exception:
        return None


def _run_knn_dataset(ds: str, args: argparse.Namespace) -> Optional[Dict[str, Any]]:
    lists = _get_image_lists(ds, args.real_data_dir)
    if lists is None:
        logger.warning("%s: dataset not found — skip", ds)
        return None
    train_paths, val_paths = _split_defects(
        lists["test_defect"], lists["mask_map"],
        val_frac=args.val_frac, seed=args.split_seed)
    if not train_paths or not val_paths:
        return {"skipped": True, "reason": "empty_split"}

    synth: Dict[str, Tuple[list, list, dict]] = {}
    missing = []
    for cond, root in (("aroma", args.aroma_synthetic_dir),
                       ("random", args.random_synthetic_dir)):
        ann = Path(root) / ds / "annotations.json"
        if not ann.exists():
            logger.warning("%s/%s: annotations 없음 — skip", ds, cond)
            missing.append(cond)
            continue
        synth[cond] = _load_synth_crops(str(ann))
    if missing:
        return {"skipped": True, "reason": f"missing_synth:{','.join(missing)}"}

    val_crops, val_ids = _load_real_defect_crops(val_paths, lists["mask_map"])
    trn_crops, trn_ids = _load_real_defect_crops(train_paths, lists["mask_map"])
    if not val_crops or not trn_crops:
        return {"skipped": True, "reason": "no_real_crops"}

    model, input_size, backbone_tag = _build_backbone(args.backbone, args.device)
    tag_v = f"real_val{args.val_frac}s{args.split_seed}"      # exp5 와 동일 tag → 캐시 공유
    tag_t = f"real_train{args.val_frac}s{args.split_seed}"
    val_feat = _cached_embed(args.embed_cache_dir, ds, tag_v, backbone_tag, val_ids,
                             lambda: _embed_crops(val_crops, model, input_size, args.device))
    trn_feat = _cached_embed(args.embed_cache_dir, ds, tag_t, backbone_tag, trn_ids,
                             lambda: _embed_crops(trn_crops, model, input_size, args.device))
    a_crops, a_ids, _ = synth["aroma"]
    r_crops, r_ids, _ = synth["random"]
    a_feat = _cached_embed(args.embed_cache_dir, ds, "synth_aroma", backbone_tag, a_ids,
                           lambda: _embed_crops(a_crops, model, input_size, args.device))
    r_feat = _cached_embed(args.embed_cache_dir, ds, "synth_random", backbone_tag, r_ids,
                           lambda: _embed_crops(r_crops, model, input_size, args.device))
    a_eq, r_eq, n = _equalize_n(a_feat, r_feat, args.seed)

    # d1 per pool
    d1_real = _min_dist_to_pool(val_feat, trn_feat)
    d1_rand = np.minimum(d1_real, _min_dist_to_pool(val_feat, r_eq))
    d1_arom = np.minimum(d1_real, _min_dist_to_pool(val_feat, a_eq))

    # leakage sanity: synth 가 val 과 사실상 동일하면 소스 분리 위반 신호
    for cond, feat in (("aroma", a_eq), ("random", r_eq)):
        n_zero = int((_min_dist_to_pool(val_feat, feat) < 1e-6).sum())
        if n_zero:
            logger.warning("%s/%s: %d val crops at ~0 distance to synth — "
                           "leakage 신호 (소스는 train split 이어야 함)", ds, cond, n_zero)

    img_ids = [_img_id_of(c) for c in val_ids]
    boot = _clustered_bootstrap(d1_rand, d1_arom, img_ids,
                                reps=args.bootstrap_reps, seed=args.seed)
    r_grid = np.percentile(d1_real, np.arange(10, 91, 10))
    unstable = len(set(img_ids)) < 10

    return {
        "d1_mean": {"real": round(float(d1_real.mean()), 6),
                    "real+random": round(float(d1_rand.mean()), 6),
                    "real+aroma": round(float(d1_arom.mean()), 6)},
        "d1_median": {"real": round(float(np.median(d1_real)), 6),
                      "real+random": round(float(np.median(d1_rand)), 6),
                      "real+aroma": round(float(np.median(d1_arom)), 6)},
        "delta_AR": boot,
        "coverage_auc": {"real": round(_coverage_auc(d1_real, r_grid), 6),
                         "real+random": round(_coverage_auc(d1_rand, r_grid), 6),
                         "real+aroma": round(_coverage_auc(d1_arom, r_grid), 6)},
        "diversity": {
            "aroma": _diversity_stats(a_eq, _distinct_sources(
                Path(args.aroma_synthetic_dir) / ds / "annotations.json")),
            "random": _diversity_stats(r_eq, _distinct_sources(
                Path(args.random_synthetic_dir) / ds / "annotations.json")),
        },
        "meta": {"n_synth": n, "n_val_crops": len(val_ids),
                 "n_val_images": len(set(img_ids)), "n_train_crops": len(trn_ids),
                 "backbone": backbone_tag, "metric": "cosine",
                 "val_frac": args.val_frac, "split_seed": args.split_seed,
                 "seed": args.seed, "unstable": unstable},
    }


def _build_knn_summary(results: Dict[str, Any]) -> str:
    lines = [
        "# Exp6-knn — kNN test-coverage 요약",
        "",
        "가설: Δ(1-NN dist) = mean d1(real+random) − mean d1(real+aroma) > 0",
        "(aroma 풀이 held-out val 결함을 더 가깝게 커버). 검정 = 이미지 단위 clustered bootstrap.",
        "정보 분리: AROMA 선택은 val 을 보지 않는다 (선택 입력 = train 프로파일링).",
        "",
        "| dataset | d1 real | +random | +aroma | Δ(A−R) [CI95] (p) | covAUC R/A | div(A/R pairwise-mean) | 판정 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for ds in sorted(results):
        node = results[ds]
        if node.get("skipped"):
            lines.append(f"| {ds} | — | — | — | — | — | — | skip ({node.get('reason')}) |")
            continue
        m, b = node["d1_mean"], node["delta_AR"]
        cov, dv = node["coverage_auc"], node["diversity"]
        ok = b["delta"] > 0 and b["p_boot_one_sided"] < 0.05
        verdict = "✅" if ok else "❌/보류"
        if node["meta"].get("unstable"):
            verdict += " ⚠few-imgs"
        da = dv["aroma"].get("pairwise_mean")
        dr = dv["random"].get("pairwise_mean")
        lines.append(
            f"| {ds} | {m['real']:.4f} | {m['real+random']:.4f} | {m['real+aroma']:.4f} "
            f"| {b['delta']:+.4f} [{b['ci95'][0]:+.4f},{b['ci95'][1]:+.4f}] ({b['p_boot_one_sided']:.4f}) "
            f"| {cov['real+random']:.3f}/{cov['real+aroma']:.3f} "
            f"| {da}/{dr} | {verdict} |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# rare 모드
# ---------------------------------------------------------------------------

def _cand_key(entry: Dict[str, Any]) -> Optional[str]:
    """candidate/selected 엔트리 → (image_path, defect_bbox) canonical key."""
    img = entry.get("image_path")
    bbox = entry.get("defect_bbox")
    if not img or not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
        return None
    x, y, w, h = (int(v) for v in bbox)
    return f"{img}|{x},{y},{w},{h}"


def _load_candidate_crops(
    candidates: List[Dict[str, Any]],
) -> Tuple[List[np.ndarray], List[str], int]:
    """unique (image, defect_bbox) 후보 crop 절단. bbox = [x,y,w,h] (roi_selection 규약).

    Returns (crops, keys, n_failed). mask(defect_mask_path) 우선, bbox 폴백.
    """
    from PIL import Image

    seen: Dict[str, bool] = {}
    order: List[Dict[str, Any]] = []
    for c in candidates:
        k = _cand_key(c)
        if k and k not in seen:
            seen[k] = True
            order.append(c)

    crops: List[np.ndarray] = []
    keys: List[str] = []
    n_failed = 0
    for c in order:
        key = _cand_key(c)
        try:
            img = np.array(Image.open(c["image_path"]).convert("RGB"))
            ih, iw = img.shape[:2]
            crop = None
            mp = c.get("defect_mask_path")
            if mp and Path(mp).exists():
                mask = np.array(Image.open(mp).convert("L"))
                bb = _mask_to_bbox(mask)
                if bb is not None:
                    x1, y1, x2, y2 = bb
                    mh, mw = mask.shape[:2]
                    if (mw, mh) != (iw, ih) and mw > 0 and mh > 0:
                        sx, sy = iw / mw, ih / mh
                        x1, y1 = int(x1 * sx), int(y1 * sy)
                        x2, y2 = max(x1 + 1, int(x2 * sx)), max(y1 + 1, int(y2 * sy))
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(iw, x2), min(ih, y2)
                    if x2 > x1 and y2 > y1:
                        crop = img[y1:y2, x1:x2]
            if crop is None:
                x, y, w, h = (int(v) for v in c["defect_bbox"])
                x1, y1 = max(0, x), max(0, y)
                x2, y2 = min(iw, x + max(1, w)), min(ih, y + max(1, h))
                if x2 > x1 and y2 > y1:
                    crop = img[y1:y2, x1:x2]
            if crop is None:
                n_failed += 1
                continue
            crops.append(crop)
            keys.append(key)
        except Exception as exc:
            logger.warning("candidate crop failed %s: %s", key, exc)
            n_failed += 1
    return crops, keys, n_failed


def _run_rare_dataset(ds: str, args: argparse.Namespace) -> Optional[Dict[str, Any]]:
    roi_dir = Path(args.roi_dir_root) / ds
    cand_p = roi_dir / "roi_candidates.json"
    sel_p = roi_dir / "roi_selected.json"
    if not cand_p.exists() or not sel_p.exists():
        logger.warning("%s: roi json 없음 (%s) — skip", ds, roi_dir)
        return {"skipped": True, "reason": "missing_roi_json"}
    candidates = load_json(str(cand_p))
    selected = load_json(str(sel_p))

    # unique 소스 crop 임베딩 (aroma/null 공유 좌표계)
    crops, keys, n_failed = _load_candidate_crops(candidates)
    # 최소 k 조차 불가능할 때만 데이터셋 skip — 큰 k 는 per-cell skip 으로 축퇴
    # (aitex 처럼 crop 수가 적으면 k=15 셀만 빠지고 k=8/10 은 평가됨).
    if len(crops) < min(args.kmeans_k):
        return {"skipped": True, "reason": f"too_few_crops({len(crops)})"}
    model, input_size, backbone_tag = _build_backbone(args.backbone, args.device)
    cand_feat = _cached_embed(args.embed_cache_dir, ds, "cand_src", backbone_tag, keys,
                              lambda: _embed_crops(crops, model, input_size, args.device))

    # val crop 임베딩 (test-presence 필터용) — knn/exp5 캐시 공유
    lists = _get_image_lists(ds, args.real_data_dir)
    if lists is None:
        return {"skipped": True, "reason": "dataset_not_found"}
    _, val_paths = _split_defects(lists["test_defect"], lists["mask_map"],
                                  val_frac=args.val_frac, seed=args.split_seed)
    val_crops, val_ids = _load_real_defect_crops(val_paths, lists["mask_map"])
    if not val_crops:
        return {"skipped": True, "reason": "no_val_crops"}
    val_feat = _cached_embed(
        args.embed_cache_dir, ds, f"real_val{args.val_frac}s{args.split_seed}",
        backbone_tag, val_ids,
        lambda: _embed_crops(val_crops, model, input_size, args.device))

    # 엔트리(중복 포함) → unique key 인덱스 매핑
    key_idx = {k: i for i, k in enumerate(keys)}
    entry_key_idx: List[int] = []
    for c in candidates:
        k = _cand_key(c)
        entry_key_idx.append(key_idx.get(k, -1) if k else -1)
    entry_key_idx = np.asarray(entry_key_idx)
    valid_entries = entry_key_idx >= 0

    # selected → entry 인덱스 (동일 key 의 첫 매칭 — hit 판정은 key 단위라 등가)
    sel_keys = [_cand_key(s) for s in selected]
    n_unmatched = sum(1 for k in sel_keys if not k or k not in key_idx)
    if n_unmatched:
        logger.warning("%s: selected %d/%d unmatched to candidates — 스키마 확인",
                       ds, n_unmatched, len(sel_keys))
    sel_key_idx = np.asarray([key_idx[k] for k in sel_keys if k and k in key_idx])
    if sel_key_idx.size == 0:
        return {"skipped": True, "reason": "no_matched_selection"}
    # 진단: 동일 (image,bbox) key 의 중복 선택 수 — entry 단위 budget 과
    # key 단위 hit 판정의 정합성 가시화 (중복이 크면 해석 시 명시).
    n_dup_keys = int(sel_key_idx.size - len(set(sel_key_idx.tolist())))
    if n_dup_keys:
        logger.info("%s: selected 중복 key %d/%d (동일 ROI 다중 선택 — entry 단위 집계)",
                    ds, n_dup_keys, sel_key_idx.size)

    from sklearn.cluster import KMeans

    grid: Dict[str, Any] = {}
    n_valid = 0
    n_sig = 0
    n_dir = 0
    for k in args.kmeans_k:
        if k >= cand_feat.shape[0]:
            grid[f"k{k}"] = {"skipped": True, "reason": "k_geq_crops"}
            continue
        for cs in args.cluster_seeds:
            km = KMeans(n_clusters=k, random_state=cs, n_init=10)
            key_mode = km.fit_predict(cand_feat)
            val_mode = km.predict(val_feat)
            cell = _rare_cell_keyed(
                key_mode=key_mode, val_mode=val_mode,
                sel_key_idx=sel_key_idx, entry_key_idx=entry_key_idx,
                valid_entries=valid_entries, k=k,
                rare_quantile=args.rare_quantile, null_seeds=args.null_seeds,
            )
            grid[f"k{k}_s{cs}"] = cell
            if not cell.get("skipped"):
                n_valid += 1
                if cell["observed"] > cell["null_mean"]:
                    n_dir += 1
                if cell["p_emp"] < 0.05:
                    n_sig += 1

    return {
        "grid": grid,
        "verdict": {"n_valid_cells": n_valid, "direction_consistent": n_dir,
                    "n_sig_p05": n_sig},
        "meta": {"n_unique_crops": len(keys), "n_entries": len(candidates),
                 "n_selected": int(sel_key_idx.size), "n_unmatched": n_unmatched,
                 "n_crop_failed": n_failed, "n_val_crops": len(val_ids),
                 "backbone": backbone_tag, "rare_quantile": args.rare_quantile,
                 "null_seeds": args.null_seeds,
                 "p_floor": round(1.0 / (args.null_seeds + 1), 4),
                 "val_frac": args.val_frac, "split_seed": args.split_seed},
    }


def _rare_cell_keyed(
    key_mode: np.ndarray,
    val_mode: np.ndarray,
    sel_key_idx: np.ndarray,
    entry_key_idx: np.ndarray,
    valid_entries: np.ndarray,
    k: int,
    rare_quantile: float,
    null_seeds: int,
) -> Dict[str, Any]:
    """(k, cseed) 셀: rare 모드 정의 → observed(선택셋) vs random-null hit rate.

    hit 판정·null 추출 모두 **candidate 엔트리 단위**(중복 포함) — 실제
    random 선택(generate_random.select_random: 엔트리 균일 무중복 추출)과
    budget·표본공간을 정확히 일치시킨다 (공정성 불변식).
    """
    counts = np.bincount(key_mode, minlength=k)
    nonzero = counts[counts > 0]
    thresh = np.percentile(nonzero, rare_quantile * 100)
    rare_by_freq = {m for m in range(k) if 0 < counts[m] <= thresh}
    val_present = set(np.unique(val_mode).tolist())
    rare = rare_by_freq & val_present
    if not rare:
        return {"skipped": True, "reason": "no_rare_modes",
                "n_rare_before_filter": len(rare_by_freq)}

    rare_list = np.asarray(sorted(rare))
    key_is_rare = np.isin(key_mode, rare_list)
    observed = float(key_is_rare[sel_key_idx].mean())

    entry_is_rare = np.where(valid_entries,
                             key_is_rare[np.maximum(entry_key_idx, 0)], False)
    n_entries = entry_key_idx.shape[0]
    n_sel = sel_key_idx.shape[0]
    null_rates = np.empty(null_seeds)
    for s in range(null_seeds):
        rng = np.random.default_rng(s)
        idx = rng.choice(n_entries, size=min(n_sel, n_entries), replace=False)
        null_rates[s] = float(entry_is_rare[idx].mean())
    p_emp = (int(np.sum(null_rates >= observed)) + 1) / (null_seeds + 1)
    return {
        "observed": round(observed, 6),
        "null_mean": round(float(null_rates.mean()), 6),
        "null_ci95": [round(float(np.percentile(null_rates, 2.5)), 6),
                      round(float(np.percentile(null_rates, 97.5)), 6)],
        "p_emp": round(p_emp, 6),
        "n_rare_modes": int(len(rare)),
        "n_rare_before_filter": int(len(rare_by_freq)),
    }


def _build_rare_summary(results: Dict[str, Any]) -> str:
    lines = [
        "# Exp6-rare — rare-mode hit rate 요약",
        "",
        "가설: rare 모드(빈도 p25 이하 AND val 등장) hit rate — aroma > random-null "
        "(empirical p, 전 그리드 방향 일치 + 다수 셀 p<0.05).",
        f"주의: null 30-seed 의 p 하한 ≈ 0.032 — 그보다 낮은 p 는 불가.",
        "",
    ]
    for ds in sorted(results):
        node = results[ds]
        if node.get("skipped"):
            lines.append(f"## {ds} — skip ({node.get('reason')})")
            continue
        v, meta = node["verdict"], node["meta"]
        lines.append(
            f"## {ds} — 방향일치 {v['direction_consistent']}/{v['n_valid_cells']}, "
            f"p<0.05 {v['n_sig_p05']}/{v['n_valid_cells']} "
            f"(unique crops={meta['n_unique_crops']}, selected={meta['n_selected']})")
        lines.append("")
        lines.append("| cell | observed | null_mean [CI95] | p_emp | n_rare |")
        lines.append("|---|---|---|---|---|")
        for cell_key in sorted(node["grid"]):
            c = node["grid"][cell_key]
            if c.get("skipped"):
                lines.append(f"| {cell_key} | — | — | — | skip ({c.get('reason')}) |")
                continue
            lines.append(
                f"| {cell_key} | {c['observed']:.4f} "
                f"| {c['null_mean']:.4f} [{c['null_ci95'][0]:.4f},{c['null_ci95'][1]:.4f}] "
                f"| {c['p_emp']:.4f} | {c['n_rare_modes']} |")
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Orchestration / CLI
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> Dict[str, Any]:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    res_path = out_dir / "exp6_results.json"
    all_results: Dict[str, Any] = {}
    if res_path.exists():
        try:
            all_results = load_json(str(res_path))
        except Exception:
            all_results = {}
    mode_node: Dict[str, Any] = {}
    for ds in args.dataset_keys:
        logger.info("=== Exp6 %s: %s ===", args.mode, ds)
        node = (_run_knn_dataset(ds, args) if args.mode == "knn"
                else _run_rare_dataset(ds, args))
        if node is not None:
            mode_node[ds] = node
            all_results[args.mode] = mode_node
            save_json(all_results, str(res_path))  # incremental
    summary = (_build_knn_summary(mode_node) if args.mode == "knn"
               else _build_rare_summary(mode_node))
    (out_dir / f"exp6_{args.mode}_summary.md").write_text(summary, encoding="utf-8")
    logger.info("Exp6 %s done: %s", args.mode, out_dir)
    return all_results


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="AROMA Exp 6 — 임베딩 커버리지 (knn test-coverage / rare-mode hit rate)")
    p.add_argument("--mode", required=True, choices=["knn", "rare"])
    p.add_argument("--real_data_dir", required=True,
                   help="시그니처 호환용 — 실제 경로는 dataset_config")
    p.add_argument("--dataset_keys", required=True, nargs="+")
    p.add_argument("--val_frac", type=float, default=0.3,
                   help="held-out val 비율 (exp4v2/exp5 규약 정렬)")
    p.add_argument("--split_seed", type=int, default=42)
    p.add_argument("--seed", type=int, default=42,
                   help="subsample/bootstrap seed")
    p.add_argument("--backbone", default="dinov2_vits14")
    p.add_argument("--embed_cache_dir", default=None,
                   help="exp5 와 공유하는 임베딩 캐시 (재합성 후 무효화 필요)")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--device", default="cuda")
    # knn 전용
    p.add_argument("--aroma_synthetic_dir", default=None)
    p.add_argument("--random_synthetic_dir", default=None)
    p.add_argument("--bootstrap_reps", type=int, default=2000)
    # rare 전용
    p.add_argument("--roi_dir_root", default=None,
                   help="roi_selected/candidates 루트 ({root}/{ds}/*.json)")
    p.add_argument("--kmeans_k", type=int, nargs="+", default=[8, 10, 12, 15])
    p.add_argument("--cluster_seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--null_seeds", type=int, default=30)
    p.add_argument("--rare_quantile", type=float, default=0.25)
    args = p.parse_args(argv)
    if args.mode == "knn" and not (args.aroma_synthetic_dir and args.random_synthetic_dir):
        p.error("--mode knn 은 --aroma_synthetic_dir / --random_synthetic_dir 필수")
    if args.mode == "rare" and not args.roi_dir_root:
        p.error("--mode rare 는 --roi_dir_root 필수")
    return args


def main(argv=None) -> None:
    args = _parse_args(argv)
    try:
        import torch
        if args.device.startswith("cuda") and not torch.cuda.is_available():
            logger.warning("CUDA unavailable — device=cpu 로 전환")
            args.device = "cpu"
    except ImportError:
        args.device = "cpu"
    run(args)


if __name__ == "__main__":
    main()
