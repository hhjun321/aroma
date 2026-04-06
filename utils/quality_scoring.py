"""utils/quality_scoring.py

Stage 5 합성 품질 점수 계산 라이브러리.

2-metric: artifact_score (higher = fewer artifacts) + blur_score (higher = sharper).
CASDA score_casda_quality.py 에서 domain-agnostic 2개만 이식.
color_score 는 금속 표면 전용 캘리브레이션이므로 제거.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Internal metric functions
# ---------------------------------------------------------------------------

def _score_artifacts(gray: np.ndarray) -> float:
    """Gradient 기반 합성 아티팩트 감지.

    Sobel gradient magnitude의 3σ 이상치 비율(edge_score) +
    Laplacian energy / gradient mean 비율(hf_score).
    가중치: 0.6 * edge_score + 0.4 * hf_score.
    Returns: 1.0 = 아티팩트 없음 (higher = better).
    """
    gray_f = gray.astype(np.float32)

    # Sobel gradient magnitude
    gx = cv2.Sobel(gray_f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_f, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx ** 2 + gy ** 2)

    mean_mag = float(np.mean(mag))
    std_mag = float(np.std(mag))
    if mean_mag < 1e-6:
        return 1.0  # 균일 이미지 — 아티팩트 없음

    # edge_score: 3σ 이상치 픽셀 비율 (낮을수록 좋음 → 1 - ratio)
    threshold = mean_mag + 3.0 * std_mag
    outlier_ratio = float(np.mean(mag > threshold))
    edge_score = 1.0 - min(outlier_ratio * 10.0, 1.0)  # 10% 이상이면 최악

    # hf_score: Laplacian energy / gradient mean (높은 비율은 고주파 아티팩트)
    lap = cv2.Laplacian(gray_f, cv2.CV_32F)
    lap_energy = float(np.mean(np.abs(lap)))
    hf_ratio = lap_energy / (mean_mag + 1e-6)
    hf_score = 1.0 - min(hf_ratio / 5.0, 1.0)  # 5.0 이상이면 최악

    return float(np.clip(0.6 * edge_score + 0.4 * hf_score, 0.0, 1.0))


def _score_sharpness(gray: np.ndarray) -> float:
    """Laplacian variance + gradient contrast 기반 선명도.

    Laplacian variance(lap_score) + P90/P50 gradient contrast ratio(edge_sharpness).
    가중치: 0.5 * lap_score + 0.5 * edge_sharpness.
    Returns: 1.0 = 완전 선명 (higher = better).

    NOTE: lap_var 정규화는 해상도 비례 스케일링을 적용한다.
    기준 해상도 256×256 (ISP-AD ASM) 에서 경험적 상수 1000.0 을 사용하며,
    고해상도 이미지는 픽셀 수에 비례하여 기준값을 자동 상향한다.
    이로써 pruning_threshold 가 해상도에 무관하게 동일한 의미를 갖는다.
    """
    gray_f = gray.astype(np.float32)

    # Laplacian variance — 선명한 이미지일수록 높음
    lap = cv2.Laplacian(gray_f, cv2.CV_32F)
    lap_var = float(np.var(lap))
    # 해상도 비례 정규화: 기준 해상도(256×256) 대비 픽셀 비율로 스케일링
    _REF_PIXELS = 256 * 256
    num_pixels = gray.shape[0] * gray.shape[1]
    scale = num_pixels / _REF_PIXELS
    lap_score = float(np.clip(lap_var / (1000.0 * scale), 0.0, 1.0))

    # Gradient contrast: P90 / P50 비율
    gx = cv2.Sobel(gray_f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_f, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx ** 2 + gy ** 2).flatten()
    p50 = float(np.percentile(mag, 50))
    p90 = float(np.percentile(mag, 90))
    if p50 < 1e-6:
        edge_sharpness = 0.0
    else:
        ratio = p90 / (p50 + 1e-6)
        edge_sharpness = float(np.clip((ratio - 1.0) / 9.0, 0.0, 1.0))  # ratio 1~10 → 0~1

    return float(np.clip(0.5 * lap_score + 0.5 * edge_sharpness, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Worker (pickle-safe for ProcessPoolExecutor)
# ---------------------------------------------------------------------------

def _score_single_worker(args_tuple: tuple) -> dict | None:
    """ProcessPoolExecutor pickle-safe 워커.

    args: (img_path_str,)
    Returns: {"image_id": str, "artifact": float, "blur": float} | None
    """
    (img_path_str,) = args_tuple
    img = cv2.imread(img_path_str)
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    image_id = Path(img_path_str).stem
    return {
        "image_id": image_id,
        "artifact": _score_artifacts(gray),
        "blur": _score_sharpness(gray),
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def score_image(
    img_bgr: np.ndarray,
    w_artifact: float = 0.5,
    w_blur: float = 0.5,
) -> dict:
    """단일 이미지 numpy array를 받아 품질 점수를 반환.

    Args:
        img_bgr: BGR 이미지 numpy array.
        w_artifact: artifact_score 가중치 (기본 0.5).
        w_blur: blur_score 가중치 (기본 0.5).

    Returns:
        {"artifact_score": float, "blur_score": float, "quality_score": float}

    Raises:
        ValueError: w_artifact + w_blur != 1.0
    """
    if abs(w_artifact + w_blur - 1.0) > 1e-6:
        raise ValueError(
            f"w_artifact + w_blur must equal 1.0, got {w_artifact + w_blur}"
        )
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    artifact = _score_artifacts(gray)
    blur = _score_sharpness(gray)
    quality = w_artifact * artifact + w_blur * blur
    return {
        "artifact_score": round(float(artifact), 4),
        "blur_score": round(float(blur), 4),
        "quality_score": round(float(quality), 4),
    }


def score_defect_images(
    stage4_seed_dir: str,
    w_artifact: float = 0.5,
    w_blur: float = 0.5,
    workers: int = 0,
) -> dict:
    """stage4_seed_dir/defect/*.png 전체를 평가하고 quality_scores.json 저장.

    Args:
        stage4_seed_dir: Stage 4 seed 출력 디렉터리 경로.
        w_artifact: artifact 가중치 (기본 0.5).
        w_blur: blur 가중치 (기본 0.5).
        workers: 병렬 워커 수 (0=순차, -1=자동, N>=2=N 프로세스).

    Returns:
        {"weights": {...}, "scores": [...], "stats": {...}}
        count == 0 이면 파일 미생성, 빈 stats 반환.
        quality_scores.json 이 이미 존재하면 재계산 없이 로드하여 반환 (skip).

    Raises:
        ValueError: w_artifact + w_blur != 1.0
    """
    if abs(w_artifact + w_blur - 1.0) > 1e-6:
        raise ValueError(
            f"w_artifact + w_blur must equal 1.0, got {w_artifact + w_blur}"
        )

    seed_dir = Path(stage4_seed_dir)
    cache_path = seed_dir / "quality_scores.json"

    # Skip: 이미 존재하면 로드하여 반환
    if cache_path.exists():
        return json.loads(cache_path.read_text())

    defect_dir = seed_dir / "defect"
    img_paths = sorted(defect_dir.glob("*.png")) if defect_dir.exists() else []

    # count == 0: 빈 stats 반환, 파일 미생성
    if not img_paths:
        return {
            "weights": {"artifact": w_artifact, "blur": w_blur},
            "scores": [],
            "stats": {},
        }

    from utils.parallel import resolve_workers, run_parallel

    num_workers = resolve_workers(workers)
    tasks = [(str(p),) for p in img_paths]
    raw_results = run_parallel(_score_single_worker, tasks, num_workers,
                               desc="Stage5 quality scoring")

    # 병렬 결과를 image_id 키 dict로 수집
    result_map = {r["image_id"]: r for r in raw_results if r is not None}

    # sorted 순서로 재조립
    scores = []
    for p in img_paths:
        image_id = p.stem
        r = result_map.get(image_id)
        if r is None:
            continue
        artifact = r["artifact"]
        blur = r["blur"]
        quality = w_artifact * artifact + w_blur * blur
        scores.append({
            "image_id": image_id,
            "artifact_score": round(float(artifact), 4),
            "blur_score": round(float(blur), 4),
            "quality_score": round(float(quality), 4),
        })

    # stats 계산
    q_arr = np.array([s["quality_score"] for s in scores])
    stats = {
        "count": len(q_arr),
        "mean": round(float(np.mean(q_arr)), 4),
        "std": round(float(np.std(q_arr)), 4),
        "min": round(float(np.min(q_arr)), 4),
        "max": round(float(np.max(q_arr)), 4),
        "p25": round(float(np.percentile(q_arr, 25)), 4),
        "p50": round(float(np.percentile(q_arr, 50)), 4),
        "p75": round(float(np.percentile(q_arr, 75)), 4),
        "p90": round(float(np.percentile(q_arr, 90)), 4),
    }

    result = {
        "weights": {"artifact": w_artifact, "blur": w_blur},
        "scores": scores,
        "stats": stats,
    }
    cache_path.write_text(json.dumps(result, indent=2))
    return result
