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

def _score_artifacts(gray: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Gradient 기반 합성 아티팩트 감지.

    Sobel gradient magnitude의 3σ 이상치 비율(edge_score) +
    Laplacian energy / (gradient mean + std) 비율(hf_score).
    가중치: 0.6 * edge_score + 0.4 * hf_score.
    Returns: 1.0 = 아티팩트 없음 (higher = better).

    Args:
        gray: 그레이스케일 이미지.
        mask: Stage 4에서 저장한 이진 ROI 마스크 (uint8, 255=결함 영역).
              제공 시 dilate 후 해당 픽셀에서만 통계 산출 (배경 편향 제거).
              None=전체 이미지 사용 (기존 동작).

    Note (Option A): hf_ratio 분모를 mean_mag → mean_mag + std_mag 로 변경.
    매끄러운 배경(candle 등) 에서 mean_mag 가 작아 hf_ratio 가 과대평가되던
    문제를 텍스처 분산(std_mag)을 분모에 포함해 완화.
    """
    gray_f = gray.astype(np.float32)

    # Sobel/Laplacian은 전체 이미지에서 계산 (공간 연산이므로 컨텍스트 필요)
    gx = cv2.Sobel(gray_f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_f, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    lap = cv2.Laplacian(gray_f, cv2.CV_32F)

    # Option B: ROI 마스크가 있으면 결함 영역만으로 통계 산출
    if mask is not None:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
        roi = cv2.dilate(mask, kernel) > 127
        if np.any(roi):
            mag_vals = mag[roi]
            lap_vals = np.abs(lap[roi])
        else:
            mag_vals = mag.flatten()
            lap_vals = np.abs(lap).flatten()
    else:
        mag_vals = mag.flatten()
        lap_vals = np.abs(lap).flatten()

    mean_mag = float(np.mean(mag_vals))
    std_mag  = float(np.std(mag_vals))
    if mean_mag < 1e-6:
        return 1.0  # 균일 이미지 — 아티팩트 없음

    # edge_score: 3σ 이상치 픽셀 비율 (낮을수록 좋음 → 1 - ratio)
    threshold = mean_mag + 3.0 * std_mag
    outlier_ratio = float(np.mean(mag_vals > threshold))
    edge_score = 1.0 - min(outlier_ratio * 10.0, 1.0)  # 10% 이상이면 최악

    # hf_score: Laplacian energy / (gradient mean + std)
    # Option A: 분모에 std_mag 추가 → 매끄러운 배경에서 hf_ratio 과대평가 완화
    lap_energy = float(np.mean(lap_vals))
    hf_ratio = lap_energy / (mean_mag + std_mag + 1e-6)
    hf_score = 1.0 - min(hf_ratio / 5.0, 1.0)  # 5.0 이상이면 최악

    return float(np.clip(0.6 * edge_score + 0.4 * hf_score, 0.0, 1.0))


def _score_sharpness(gray: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Laplacian variance + gradient contrast 기반 선명도.

    Laplacian variance(lap_score) + P90/P50 gradient contrast ratio(edge_sharpness).
    가중치: 0.5 * lap_score + 0.5 * edge_sharpness.
    Returns: 1.0 = 완전 선명 (higher = better).

    Args:
        gray: 그레이스케일 이미지.
        mask: Stage 4에서 저장한 이진 ROI 마스크 (uint8, 255=결함 영역).
              제공 시 dilate 후 해당 픽셀에서만 통계 산출.
              None=전체 이미지 사용 (기존 동작).

    NOTE: lap_var 정규화는 해상도 비례 스케일링을 적용한다.
    기준 해상도 256×256 (ISP-AD ASM) 에서 경험적 상수 1000.0 을 사용하며,
    ROI 마스크 사용 시에는 ROI 픽셀 수를 기준으로 스케일링한다.
    """
    gray_f = gray.astype(np.float32)

    # Laplacian/Sobel은 전체 이미지에서 계산
    lap = cv2.Laplacian(gray_f, cv2.CV_32F)
    gx = cv2.Sobel(gray_f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_f, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx ** 2 + gy ** 2)

    # Option B: ROI 마스크가 있으면 결함 영역만으로 통계 산출
    if mask is not None:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
        roi = cv2.dilate(mask, kernel) > 127
        if np.any(roi):
            lap_vals = lap[roi]
            mag_vals = mag[roi].flatten()
            num_pixels = int(np.sum(roi))
        else:
            lap_vals = lap
            mag_vals = mag.flatten()
            num_pixels = gray.shape[0] * gray.shape[1]
    else:
        lap_vals = lap
        mag_vals = mag.flatten()
        num_pixels = gray.shape[0] * gray.shape[1]

    # Laplacian variance — 선명한 이미지일수록 높음
    lap_var = float(np.var(lap_vals))
    # 해상도 비례 정규화: 기준 해상도(256×256) 대비 픽셀 비율로 스케일링
    _REF_PIXELS = 256 * 256
    scale = num_pixels / _REF_PIXELS
    lap_score = float(np.clip(lap_var / (1000.0 * scale), 0.0, 1.0))

    # Gradient contrast: P90 / P50 비율
    p50 = float(np.percentile(mag_vals, 50))
    p90 = float(np.percentile(mag_vals, 90))
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

    Stage 4가 저장한 {stem}_mask.png 가 같은 디렉터리에 있으면 로드해
    ROI 기반 스코어링을 수행한다 (Option B). 없으면 전체 이미지 사용.
    """
    (img_path_str,) = args_tuple
    img = cv2.imread(img_path_str)
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # ROI 마스크 로드 (Stage 4 에서 저장한 {image_id}_mask.png)
    img_p = Path(img_path_str)
    mask_path = img_p.with_name(img_p.stem + "_mask.png")
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) if mask_path.exists() else None

    image_id = img_p.stem
    return {
        "image_id": image_id,
        "artifact": _score_artifacts(gray, mask),
        "blur": _score_sharpness(gray, mask),
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
    # *_mask.png 는 Stage 4 ROI 마스크 파일 — 스코어링 대상에서 제외
    img_paths = (
        sorted(p for p in defect_dir.glob("*.png") if not p.stem.endswith("_mask"))
        if defect_dir.exists() else []
    )

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
