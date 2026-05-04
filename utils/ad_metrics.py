"""utils/ad_metrics.py

Stage 7 이상 탐지 메트릭 추출 유틸리티.

sklearn.metrics 기반 image_auroc / image_f1 / image_best_f1 / pixel_auroc 계산.

image_f1      : threshold=0.5 고정 (분류기 그룹 전용 — softmax score 기준)
image_best_f1 : precision-recall curve에서 최적 threshold의 F1
                → baseline(cosine distance)과 분류기 그룹 간 공정 비교에 사용
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
          "image_auroc":    float,
          "image_f1":       float,   # threshold=0.5 고정
          "image_best_f1":  float,   # 최적 threshold F1 (baseline 비교 가능)
          "pixel_auroc":    float | None
        }
    """
    import numpy as np
    from sklearn.metrics import f1_score, precision_recall_curve, roc_auc_score

    image_auroc = float(roc_auc_score(y_true, y_score))

    # F1 (threshold=0.5 고정): 분류기 그룹 평가
    y_pred = [1 if s > 0.5 else 0 for s in y_score]
    image_f1 = float(f1_score(y_true, y_pred, zero_division=0))

    # best F1 (최적 threshold): baseline ↔ 분류기 그룹 공정 비교
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    f1_candidates = 2 * precision * recall / (precision + recall + 1e-8)
    image_best_f1 = float(np.max(f1_candidates)) if len(f1_candidates) > 0 else 0.0

    pixel_auroc = None
    if pixel_true is not None and pixel_score is not None:
        pixel_auroc = float(roc_auc_score(pixel_true, pixel_score))

    return {
        "image_auroc":   image_auroc,
        "image_f1":      image_f1,
        "image_best_f1": image_best_f1,
        "pixel_auroc":   pixel_auroc,
    }
