#!/usr/bin/env python3
"""
AROMA Exp 5 — PRDC 커버리지 평가 (외부 임베딩 좌표계, 저비용 L2 증거)

사전 등록 가설 (반증 가능):
  aroma/random 은 동일 copy-paste 엔진을 쓰므로 —
    * Precision / Density : 두 조건 동등   (fidelity 는 엔진이 결정)
    * Recall / Coverage   : aroma > random (선택 전략만이 분포 커버를 바꿈)
  Recall 계열만 오르고 Precision 계열이 동등해야 "ROI 선택의 가치"가 입증된다.

설계 (dev_note aroma_exp5_prdc-coverage.md):
  * reference = held-out real 결함 패치 — exp4v2 와 동일 규약의 val split
    (_split_defects 복제, val_frac=0.3, seed=42). synth 소스는 train split
    ROI 에서만 나오므로 leakage 차단.
  * 조건 간 n 엄격 동일화 (min-n seeded subsample).
  * k ∈ {3,5,10} sensitivity 전체 보고 (주 보고 k=5).
  * 유의성 = permutation test (두 synth 풀 합쳐 재분할, 기본 1000회).
    관측치는 prdc 패키지 직접 호출, permutation 루프는 사전계산 거리 기반
    벡터화 재구현 (관측치와 numerical parity assert 로 검증).

Usage (Colab):
    !python $AROMA_SCRIPTS/experiments/exp5_prdc.py \\
        --real_data_dir $AROMA_DATA \\
        --aroma_synthetic_dir  $AROMA_OUT/synthetic \\
        --random_synthetic_dir $AROMA_OUT/synthetic_random \\
        --dataset_keys severstal mvtec_leather aitex mtd \\
        --nearest_k 3 5 10 --permutation_reps 1000 \\
        --embed_cache_dir $AROMA_OUT/embed_cache \\
        --output_dir $AROMA_OUT/exp5_prdc
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# exp5 의 basicConfig 를 exp3 import 이전에 실행 — exp3 top-level basicConfig 는
# 핸들러가 이미 있으면 no-op 이므로 로깅 설정이 exp5 기준으로 고정된다.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("aroma.exp5")

# ---------------------------------------------------------------------------
# Bootstrap (exp3 관례)
# ---------------------------------------------------------------------------

_THIS_DIR = Path(__file__).resolve().parent
_AROMA_SCRIPTS = _THIS_DIR.parent


def _bootstrap() -> None:
    aroma_ref = os.environ.get("AROMA_REF") or r"D:\project\aroma"
    for p in (str(_THIS_DIR), str(_AROMA_SCRIPTS), str(Path(aroma_ref))):
        if p not in sys.path:
            sys.path.insert(0, p)


_bootstrap()

# 재사용 import — exp3 는 __main__ 가드가 있어 import 부작용 없음(read-only).
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
# Reference split — exp4v2 _split_defects 복제 (verbatim)
# ---------------------------------------------------------------------------

def _split_defects(
    test_defect_paths: List[str],
    mask_map: Dict[str, str],
    val_frac: float = 0.3,
    seed: int = 42,
) -> Tuple[List[str], List[str]]:
    """exp4_v2_supervised_detection._split_defects 복제 (stdlib random — numpy 아님).

    exp4v2 와 byte-identical 한 val split 을 재현해 reference 를 downstream 과
    정렬한다. 유일한 차이는 기본 val_frac=0.3 (exp4v2 코드 기본은 0.5 이나
    실험 가이드가 항상 0.3 을 전달 — 그 운영값을 exp5 기본으로 채택).
    """
    eligible = sorted(p for p in test_defect_paths if mask_map.get(p))
    if not eligible:
        return [], []
    rng = random.Random(seed)
    rng.shuffle(eligible)
    n_val = max(1, int(round(len(eligible) * val_frac)))
    if len(eligible) >= 2:
        n_val = min(n_val, len(eligible) - 1)
    val = eligible[:n_val]
    train = eligible[n_val:]
    return train, val


# ---------------------------------------------------------------------------
# Crop loaders
# ---------------------------------------------------------------------------

def _load_real_defect_crops(
    defect_paths: List[str],
    mask_map: Dict[str, str],
    num_workers: int = 8,
) -> Tuple[List[np.ndarray], List[str]]:
    """real 결함 이미지 → GT mask bbox crop. Returns (crops, crop_ids).

    exp3 _load_real_defect_patches 는 full image 라 미사용 — reference 는
    결함 영역 crop 이어야 synth crop(결함 패치)과 좌표계가 맞는다.
    """
    from PIL import Image

    def _one(img_p: str) -> Optional[Tuple[np.ndarray, str]]:
        mask_p = mask_map.get(img_p)
        if not mask_p:
            return None
        try:
            mask = np.array(Image.open(mask_p).convert("L"))
            bbox = _mask_to_bbox(mask)
            if bbox is None:
                return None
            img = np.array(Image.open(img_p).convert("RGB"))
            # mask 해상도가 이미지와 다르면 bbox 를 이미지 좌표로 스케일
            ih, iw = img.shape[:2]
            mh, mw = mask.shape[:2]
            x1, y1, x2, y2 = bbox
            if (mw, mh) != (iw, ih) and mw > 0 and mh > 0:
                sx, sy = iw / mw, ih / mh
                x1, y1 = int(x1 * sx), int(y1 * sy)
                x2, y2 = max(x1 + 1, int(x2 * sx)), max(y1 + 1, int(y2 * sy))
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(iw, x2), min(ih, y2)
            if x2 <= x1 or y2 <= y1:
                return None
            return img[y1:y2, x1:x2], f"{img_p}|{x1},{y1},{x2},{y2}"
        except Exception as exc:
            logger.warning("real crop failed %s: %s", img_p, exc)
            return None

    with ThreadPoolExecutor(max_workers=num_workers) as ex:
        loaded = list(ex.map(_one, defect_paths))
    ok = [r for r in loaded if r is not None]
    n_skip = len(defect_paths) - len(ok)
    if n_skip:
        logger.warning("real crops: skipped %d/%d (no mask/bbox/load fail)",
                       n_skip, len(defect_paths))
    crops = [c for c, _ in ok]
    ids = [i for _, i in ok]
    return crops, ids


def _load_synth_crops(
    annotations_path: str,
    num_workers: int = 8,
) -> Tuple[List[np.ndarray], List[str], Dict[str, int]]:
    """synth annotations → 결함 crop. 폴백 체인: mask_path → bbox → skip.

    full-image crop 폴백은 금지 — 배경이 임베딩을 지배해 커버리지 신호를
    희석한다(dev_note 엣지 규칙). Returns (crops, crop_ids, skip_counts).
    """
    from PIL import Image

    anns = load_json(annotations_path)
    counts = {"n_total": len(anns), "n_mask": 0, "n_bbox": 0, "n_skipped": 0}

    ann_dir = Path(annotations_path).parent

    def _resolve(p: Optional[str], subdir: str) -> Optional[str]:
        # 합성 annotations 는 생성 시점의 절대경로(Colab)를 baking 하므로 다른 run·
        # 경로에서 어긋날 수 있다(예: 이미지는 synth_random/ 인데 image_path 는
        # 구 synth_random_F/ 를 가리킴). exp4v2 _load_synth_annotations._resolve_path
        # 와 동일하게, 절대경로가 없으면 annotations 디렉토리 기준으로 폴백한다
        # ({ann_dir}/{subdir}/basename → {ann_dir}/basename). 미해결 시 None.
        if not p:
            return None
        pp = Path(p)
        if pp.exists():
            return str(pp)
        cand = ann_dir / subdir / pp.name
        if cand.exists():
            return str(cand)
        cand2 = ann_dir / pp.name
        return str(cand2) if cand2.exists() else None

    def _one(ann: Dict[str, Any]) -> Optional[Tuple[np.ndarray, str, str]]:
        # dry_run 은 명시적으로 제외 (exp4v2 관례) — 파일 부재에 의존하는
        # 암묵 필터가 아니라 플래그로 차단(placeholder 파일 존재 시에도 안전).
        if ann.get("dry_run") is True:
            return None
        img_p = _resolve(ann.get("image_path"), "images")
        if not img_p:
            return None
        try:
            img = np.array(Image.open(img_p).convert("RGB"))
            ih, iw = img.shape[:2]
            mask_p = _resolve(ann.get("mask_path"), "masks")
            if mask_p:
                mask = np.array(Image.open(mask_p).convert("L"))
                bbox = _mask_to_bbox(mask)
                if bbox is not None:
                    x1, y1, x2, y2 = bbox
                    mh, mw = mask.shape[:2]
                    if (mw, mh) != (iw, ih) and mw > 0 and mh > 0:
                        sx, sy = iw / mw, ih / mh
                        x1, y1 = int(x1 * sx), int(y1 * sy)
                        x2, y2 = max(x1 + 1, int(x2 * sx)), max(y1 + 1, int(y2 * sy))
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(iw, x2), min(ih, y2)
                    if x2 > x1 and y2 > y1:
                        return img[y1:y2, x1:x2], f"{img_p}|m", "mask"
            raw = ann.get("bbox")
            if isinstance(raw, (list, tuple)) and len(raw) == 4:
                # generate_defects 는 [x, y, w, h] 로 기록 (x2y2 아님)
                x, y, w, h = (int(v) for v in raw)
                x1, y1 = max(0, x), max(0, y)
                x2, y2 = min(iw, x + max(1, w)), min(ih, y + max(1, h))
                if x2 > x1 and y2 > y1:
                    return img[y1:y2, x1:x2], f"{img_p}|b", "bbox"
            return None
        except Exception as exc:
            logger.warning("synth crop failed %s: %s", img_p, exc)
            return None

    with ThreadPoolExecutor(max_workers=num_workers) as ex:
        loaded = list(ex.map(_one, anns))

    crops: List[np.ndarray] = []
    ids: List[str] = []
    for r in loaded:
        if r is None:
            counts["n_skipped"] += 1
            continue
        crop, cid, how = r
        crops.append(crop)
        ids.append(cid)
        counts["n_mask" if how == "mask" else "n_bbox"] += 1
    logger.info("synth crops %s: mask=%d bbox=%d skipped=%d / %d",
                Path(annotations_path).parent.name, counts["n_mask"],
                counts["n_bbox"], counts["n_skipped"], counts["n_total"])
    return crops, ids, counts


# ---------------------------------------------------------------------------
# Embedding backbone + cache
# ---------------------------------------------------------------------------

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _build_backbone(name: str, device: str):
    """Returns (model, input_size, backbone_tag). DINOv2 실패 시 InceptionV3 폴백."""
    import torch

    if name.startswith("dinov2"):
        try:
            model = torch.hub.load("facebookresearch/dinov2", name)
            model = model.to(device).eval()
            return model, 224, name
        except Exception as exc:
            logger.warning("DINOv2 load failed (%s) — falling back to inception_v3", exc)
    from torchvision.models import inception_v3, Inception_V3_Weights
    model = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
    model.fc = torch.nn.Identity()
    model = model.to(device).eval()
    return model, 299, "inception_v3"


def _embed_crops(
    crops: List[np.ndarray],
    model,
    input_size: int,
    device: str,
    batch_size: int = 64,
) -> np.ndarray:
    import torch
    from PIL import Image

    feats: List[np.ndarray] = []
    with torch.no_grad():
        for s in range(0, len(crops), batch_size):
            batch = []
            for crop in crops[s:s + batch_size]:
                img = Image.fromarray(crop).resize((input_size, input_size))
                arr = np.asarray(img, dtype=np.float32) / 255.0
                arr = (arr - _IMAGENET_MEAN) / _IMAGENET_STD
                batch.append(arr.transpose(2, 0, 1))
            t = torch.from_numpy(np.stack(batch)).to(device)
            out = model(t)
            if isinstance(out, (tuple, list)):
                out = out[0]
            feats.append(out.detach().cpu().numpy().astype(np.float64))
    return np.concatenate(feats, axis=0) if feats else np.zeros((0, 1))


def _cached_embed(
    cache_dir: Optional[str],
    ds: str,
    tag: str,
    backbone_tag: str,
    crop_ids: List[str],
    embed_fn,
) -> np.ndarray:
    """Drive .npy 캐시 (manifest hash 로 무효화). 2·3순위 실험이 재사용."""
    if not cache_dir:
        return embed_fn()
    digest = hashlib.sha256("\n".join(crop_ids).encode("utf-8")).hexdigest()[:16]
    base = Path(cache_dir) / ds
    base.mkdir(parents=True, exist_ok=True)
    npy = base / f"{tag}_{backbone_tag}.npy"
    man = base / f"{tag}_{backbone_tag}.manifest.json"
    if npy.exists() and man.exists():
        try:
            meta = json.loads(man.read_text(encoding="utf-8"))
            if meta.get("digest") == digest and meta.get("n") == len(crop_ids):
                logger.info("  [EmbedCache] hit %s/%s (n=%d)", ds, tag, len(crop_ids))
                return np.load(str(npy))
        except Exception:
            pass
    feats = embed_fn()
    np.save(str(npy), feats)
    man.write_text(json.dumps({"digest": digest, "n": len(crop_ids),
                               "backbone": backbone_tag}), encoding="utf-8")
    logger.info("  [EmbedCache] built %s/%s (n=%d)", ds, tag, len(crop_ids))
    return feats


# ---------------------------------------------------------------------------
# PRDC — observed (prdc 패키지) + permutation (사전계산 거리 벡터화)
# ---------------------------------------------------------------------------

def _pairwise_dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Euclidean distance matrix (n_a, n_b) — prdc 와 동일한 비정규 유클리드."""
    aa = np.sum(a * a, axis=1)[:, None]
    bb = np.sum(b * b, axis=1)[None, :]
    d2 = np.maximum(aa + bb - 2.0 * (a @ b.T), 0.0)
    return np.sqrt(d2)


def _knn_radii(d_self: np.ndarray, k: int) -> np.ndarray:
    """자기 집합 kNN 반경 — prdc 관례(자기거리 0 포함, k+1 번째 최솟값 = k번째 이웃)."""
    return np.partition(d_self, k, axis=1)[:, k]


def _prdc_from_distances(
    d_rf: np.ndarray,       # (n_real, n_fake)
    radii_r: np.ndarray,    # (n_real,)
    radii_f: np.ndarray,    # (n_fake,)
    k: int,
) -> Dict[str, float]:
    """prdc.compute_prdc 공식을 사전계산 거리 위에서 재현 (strict <, 동일 방향)."""
    lt_r = d_rf < radii_r[:, None]
    return {
        "precision": float(lt_r.any(axis=0).mean()),
        "recall":    float((d_rf < radii_f[None, :]).any(axis=1).mean()),
        "density":   float(lt_r.sum(axis=0).mean() / float(k)),
        "coverage":  float((d_rf.min(axis=1) < radii_r).mean()),
    }


_METRICS = ("precision", "recall", "density", "coverage")


def _observed_prdc(real_feat: np.ndarray, fake_feat: np.ndarray, k: int) -> Dict[str, float]:
    from prdc import compute_prdc
    out = compute_prdc(real_features=real_feat, fake_features=fake_feat, nearest_k=k)
    return {m: float(out[m]) for m in _METRICS}


def _permutation_test(
    real_feat: np.ndarray,
    aroma_feat: np.ndarray,
    random_feat: np.ndarray,
    k: int,
    reps: int,
    seed: int,
) -> Dict[str, Any]:
    """두 synth 풀을 합쳐 무작위 재분할 — Δ(aroma−random) 의 null 분포.

    real 반경·real×combined 거리·combined×combined 거리를 1회 사전계산하고,
    permutation 마다 인덱스 선택 + fake 반경 재계산(np.partition)만 수행.
    """
    n = aroma_feat.shape[0]
    assert random_feat.shape[0] == n, "n must be equalized before permutation"
    combined = np.concatenate([aroma_feat, random_feat], axis=0)   # (2n, dim)

    d_rr = _pairwise_dist(real_feat, real_feat)
    radii_r = _knn_radii(d_rr, k)
    d_rc = _pairwise_dist(real_feat, combined)                     # (n_real, 2n)
    d_cc = _pairwise_dist(combined, combined)                      # (2n, 2n)

    def _metrics_for(idx: np.ndarray) -> Dict[str, float]:
        d_rf = d_rc[:, idx]
        radii_f = _knn_radii(d_cc[np.ix_(idx, idx)], k)
        return _prdc_from_distances(d_rf, radii_r, radii_f, k)

    obs_a = _metrics_for(np.arange(n))
    obs_r = _metrics_for(np.arange(n, 2 * n))
    obs_delta = {m: obs_a[m] - obs_r[m] for m in _METRICS}

    rng = np.random.default_rng(seed)
    null_delta = {m: np.empty(reps) for m in _METRICS}
    for r in range(reps):
        perm = rng.permutation(2 * n)
        pa = _metrics_for(perm[:n])
        pr = _metrics_for(perm[n:])
        for m in _METRICS:
            null_delta[m][r] = pa[m] - pr[m]

    out: Dict[str, Any] = {"delta": {}, "p_one_sided": {}, "null_ci95": {}}
    for m in _METRICS:
        nd = null_delta[m]
        # 방향 가설(aroma>random): one-sided empirical p = (r+1)/(K+1)
        p1 = (int(np.sum(nd >= obs_delta[m])) + 1) / (reps + 1)
        out["delta"][m] = round(obs_delta[m], 6)
        out["p_one_sided"][m] = round(p1, 6)
        out["null_ci95"][m] = [round(float(np.percentile(nd, 2.5)), 6),
                               round(float(np.percentile(nd, 97.5)), 6)]
    # vectorized 경로 검증용 관측치도 반환 (prdc 패키지 값과 parity check)
    out["_vec_observed"] = {"aroma": obs_a, "random": obs_r}
    return out


def _equalize_n(
    a_feat: np.ndarray, r_feat: np.ndarray, seed: int,
) -> Tuple[np.ndarray, np.ndarray, int]:
    n = min(a_feat.shape[0], r_feat.shape[0])
    rng = np.random.default_rng(seed)
    ai = rng.choice(a_feat.shape[0], size=n, replace=False)
    ri = rng.choice(r_feat.shape[0], size=n, replace=False)
    return a_feat[np.sort(ai)], r_feat[np.sort(ri)], n


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def _run_one_dataset(
    ds: str,
    real_data_dir: str,
    aroma_dir: str,
    random_dir: str,
    k_list: List[int],
    reps: int,
    val_frac: float,
    split_seed: int,
    seed: int,
    backbone_name: str,
    embed_cache_dir: Optional[str],
    device: str,
) -> Optional[Dict[str, Any]]:
    lists = _get_image_lists(ds, real_data_dir)
    if lists is None:
        logger.warning("%s: dataset not found — skip", ds)
        return None

    _, val_paths = _split_defects(lists["test_defect"], lists["mask_map"],
                                  val_frac=val_frac, seed=split_seed)
    if not val_paths:
        logger.warning("%s: empty val split — skip", ds)
        return None
    real_crops, real_ids = _load_real_defect_crops(val_paths, lists["mask_map"])

    cond_crops: Dict[str, Tuple[List[np.ndarray], List[str], Dict[str, int]]] = {}
    skipped_conds: List[str] = []
    for cond, root in (("aroma", aroma_dir), ("random", random_dir)):
        ann = Path(root) / ds / "annotations.json"
        if not ann.exists():
            logger.warning("%s/%s: annotations.json 없음 (%s) — 조건 skip", ds, cond, ann)
            skipped_conds.append(cond)
            continue
        cond_crops[cond] = _load_synth_crops(str(ann))

    if skipped_conds or not real_crops:
        return {
            "skipped": True,
            "reason": ("no_real_crops" if not real_crops
                       else f"missing_synth:{','.join(skipped_conds)}"),
            "n_ref": len(real_crops),
        }

    model, input_size, backbone_tag = _build_backbone(backbone_name, device)
    real_feat = _cached_embed(
        embed_cache_dir, ds, f"real_val{val_frac}s{split_seed}", backbone_tag,
        real_ids, lambda: _embed_crops(real_crops, model, input_size, device))
    a_crops, a_ids, a_counts = cond_crops["aroma"]
    r_crops, r_ids, r_counts = cond_crops["random"]
    a_feat = _cached_embed(embed_cache_dir, ds, "synth_aroma", backbone_tag, a_ids,
                           lambda: _embed_crops(a_crops, model, input_size, device))
    r_feat = _cached_embed(embed_cache_dir, ds, "synth_random", backbone_tag, r_ids,
                           lambda: _embed_crops(r_crops, model, input_size, device))

    a_eq, r_eq, n = _equalize_n(a_feat, r_feat, seed)
    n_ref = real_feat.shape[0]
    unstable = n_ref < 30
    if unstable:
        logger.warning("%s: n_ref=%d < 30 — PRDC unstable flag", ds, n_ref)

    result: Dict[str, Any] = {
        "meta": {
            "n": n, "n_ref": n_ref, "backbone": backbone_tag,
            "split_seed": split_seed, "val_frac": val_frac, "seed": seed,
            "unstable": unstable,
            "n_skipped_synth": {"aroma": a_counts["n_skipped"],
                                "random": r_counts["n_skipped"]},
        },
    }
    for k in k_list:
        if n <= k or n_ref <= k:
            logger.warning("%s k=%d: n(%d)/n_ref(%d) <= k — skip", ds, k, n, n_ref)
            result[f"k{k}"] = {"skipped": True, "reason": "n_leq_k"}
            continue
        obs_a = _observed_prdc(real_feat, a_eq, k)
        obs_r = _observed_prdc(real_feat, r_eq, k)
        perm = _permutation_test(real_feat, a_eq, r_eq, k, reps, seed)
        # parity check: prdc 패키지 vs 벡터화 경로 (관측치)
        for cond, obs, vec in (("aroma", obs_a, perm["_vec_observed"]["aroma"]),
                               ("random", obs_r, perm["_vec_observed"]["random"])):
            for m in _METRICS:
                if not np.isclose(obs[m], vec[m], atol=1e-6):
                    logger.warning(
                        "%s k=%d %s/%s parity mismatch: prdc=%.6f vec=%.6f",
                        ds, k, cond, m, obs[m], vec[m])
        perm.pop("_vec_observed", None)
        result[f"k{k}"] = {
            "aroma": {m: round(obs_a[m], 6) for m in _METRICS},
            "random": {m: round(obs_r[m], 6) for m in _METRICS},
            **perm,
        }
        logger.info(
            "%s k=%d: ΔRecall=%+.4f (p=%.4f)  ΔCoverage=%+.4f (p=%.4f)  "
            "ΔPrecision=%+.4f  ΔDensity=%+.4f",
            ds, k, perm["delta"]["recall"], perm["p_one_sided"]["recall"],
            perm["delta"]["coverage"], perm["p_one_sided"]["coverage"],
            perm["delta"]["precision"], perm["delta"]["density"],
        )
    return result


def _build_summary(results: Dict[str, Any], main_k: int = 5) -> str:
    lines = [
        "# Exp5 — PRDC 커버리지 요약",
        "",
        "사전 등록 가설: Precision/Density **동등**, Recall/Coverage **aroma > random** "
        "(one-sided permutation p).",
        "",
        f"## 주표 (k={main_k})",
        "",
        "| dataset | ΔPrecision | ΔDensity | ΔRecall (p) | ΔCoverage (p) | n / n_ref | 판정 |",
        "|---|---|---|---|---|---|---|",
    ]
    for ds in sorted(results):
        node = results[ds]
        if node.get("skipped"):
            lines.append(f"| {ds} | — | — | — | — | — | skip ({node.get('reason')}) |")
            continue
        kk = node.get(f"k{main_k}", {})
        if kk.get("skipped"):
            lines.append(f"| {ds} | — | — | — | — | — | k skip |")
            continue
        d, p = kk["delta"], kk["p_one_sided"]
        meta = node["meta"]
        ok = (p["recall"] < 0.05 and p["coverage"] < 0.05
              and d["recall"] > 0 and d["coverage"] > 0)
        verdict = "✅ 가설 방향" if ok else "❌/보류"
        if meta.get("unstable"):
            verdict += " ⚠unstable"
        lines.append(
            f"| {ds} | {d['precision']:+.4f} | {d['density']:+.4f} "
            f"| {d['recall']:+.4f} ({p['recall']:.4f}) "
            f"| {d['coverage']:+.4f} ({p['coverage']:.4f}) "
            f"| {meta['n']} / {meta['n_ref']} | {verdict} |")
    lines += ["", "## k-sensitivity (Δ, one-sided p)", ""]
    for ds in sorted(results):
        node = results[ds]
        if node.get("skipped"):
            continue
        lines.append(f"### {ds}")
        lines.append("| k | ΔPrecision | ΔDensity | ΔRecall (p) | ΔCoverage (p) |")
        lines.append("|---|---|---|---|---|")
        for key in sorted(k for k in node if k.startswith("k")):
            kk = node[key]
            if kk.get("skipped"):
                continue
            d, p = kk["delta"], kk["p_one_sided"]
            lines.append(
                f"| {key[1:]} | {d['precision']:+.4f} | {d['density']:+.4f} "
                f"| {d['recall']:+.4f} ({p['recall']:.4f}) "
                f"| {d['coverage']:+.4f} ({p['coverage']:.4f}) |")
        lines.append("")
    return "\n".join(lines)


def run(args: argparse.Namespace) -> Dict[str, Any]:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results: Dict[str, Any] = {}
    for ds in args.dataset_keys:
        logger.info("=== Exp5 PRDC: %s ===", ds)
        node = _run_one_dataset(
            ds=ds,
            real_data_dir=args.real_data_dir,
            aroma_dir=args.aroma_synthetic_dir,
            random_dir=args.random_synthetic_dir,
            k_list=args.nearest_k,
            reps=args.permutation_reps,
            val_frac=args.val_frac,
            split_seed=args.split_seed,
            seed=args.seed,
            backbone_name=args.backbone,
            embed_cache_dir=args.embed_cache_dir,
            device=args.device,
        )
        if node is not None:
            results[ds] = node
            save_json(results, str(out_dir / "exp5_prdc_results.json"))  # incremental
    (out_dir / "exp5_prdc_summary.md").write_text(
        _build_summary(results), encoding="utf-8")
    logger.info("Exp5 done: %s", out_dir)
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="AROMA Exp 5 — PRDC 커버리지 평가 (aroma vs random synth)")
    p.add_argument("--real_data_dir", required=True,
                   help="실제 데이터셋 루트 (시그니처 호환용 — 경로는 dataset_config)")
    p.add_argument("--aroma_synthetic_dir", required=True)
    p.add_argument("--random_synthetic_dir", required=True)
    p.add_argument("--dataset_keys", required=True, nargs="+")
    p.add_argument("--nearest_k", type=int, nargs="+", default=[3, 5, 10],
                   help="PRDC k sensitivity 목록 (default: 3 5 10; 주 보고 k=5)")
    p.add_argument("--permutation_reps", type=int, default=1000)
    p.add_argument("--val_frac", type=float, default=0.3,
                   help="reference(held-out val) 비율 — exp4v2 운영값 0.3 정렬")
    p.add_argument("--split_seed", type=int, default=42,
                   help="reference split seed — exp4v2 와 동일해야 정렬")
    p.add_argument("--seed", type=int, default=42,
                   help="subsample/permutation seed")
    p.add_argument("--backbone", default="dinov2_vits14",
                   help="dinov2_vits14 (기본) — 로드 실패 시 inception_v3 폴백")
    p.add_argument("--embed_cache_dir", default=None,
                   help="임베딩 .npy Drive 캐시 (2·3순위 실험 재사용)")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--device", default="cuda")
    return p.parse_args(argv)


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
