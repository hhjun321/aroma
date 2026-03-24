"""utils/ad_metrics.py

Stage 7 이상 탐지 메트릭 추출 유틸리티.

sklearn.metrics 기반 image_auroc / image_f1 / pixel_auroc 계산.
"""
from __future__ import annotations


def extract_metrics(
    y_true: list[int],
    y_score: list[float],
    pixel_true: list | None = None,
    pixel_score: list | None = None,
) -> dict:
    """이미지·픽셀 레벨 이상 탐지 메트릭을 계산한다.

    Args:
        y_true: 이미지 레벨 레이블 (0=정상, 1=결함).
        y_score: 이미지 레벨 anomaly score (클수록 결함에 가까움).
        pixel_true: 픽셀 레벨 레이블 플랫 리스트 (옵션).
        pixel_score: 픽셀 레벨 anomaly score 플랫 리스트 (옵션).

    Returns:
        {
          "image_auroc": float,
          "image_f1":    float,
          "pixel_auroc": float | None   # pixel 데이터 없으면 None
        }
    """
    from sklearn.metrics import f1_score, roc_auc_score

    image_auroc = float(roc_auc_score(y_true, y_score))

    # F1: threshold = 0.5 (score > 0.5 → 결함 예측)
    y_pred = [1 if s > 0.5 else 0 for s in y_score]
    image_f1 = float(f1_score(y_true, y_pred, zero_division=0))

    pixel_auroc = None
    if pixel_true is not None and pixel_score is not None:
        pixel_auroc = float(roc_auc_score(pixel_true, pixel_score))

    return {
        "image_auroc": image_auroc,
        "image_f1": image_f1,
        "pixel_auroc": pixel_auroc,
    }
