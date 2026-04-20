"""Hint Image Generation — CASDA src/preprocessing/hint_generator.py 포팅.

ControlNet 학습용 3채널 hint 이미지 생성.
  R: 결함 마스크 형태 (linearity/solidity 기반 강화)
  G: 배경 구조선 (Sobel 방향 엣지)
  B: 배경 세밀 텍스처 (국소 분산)

채널을 가중합(R×0.5 + G×0.3 + B×0.2)으로 그레이스케일 변환 후 3채널에 복제.
→ ControlNet의 색상 단축 학습 방지 (CASDA docs/20260219.docs 섹션 6-4 참조).
"""
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Optional


class HintImageGenerator:
    def __init__(self, enhance_linearity: bool = True, enhance_background: bool = True):
        self.enhance_linearity = enhance_linearity
        self.enhance_background = enhance_background

    def generate_red_channel(self, mask: np.ndarray, defect_metrics: Dict) -> np.ndarray:
        """R: 결함 마스크 형태 (linearity/solidity 기반 강화)."""
        h, w = mask.shape
        red = np.zeros((h, w), dtype=np.uint8)
        if mask.sum() == 0:
            return red

        linearity = defect_metrics.get('linearity', 0.5)
        solidity = defect_metrics.get('solidity', 0.5)

        if self.enhance_linearity and linearity > 0.7:
            from skimage.morphology import skeletonize
            skeleton = skeletonize(mask > 0)
            kernel = np.ones((3, 3), np.uint8)
            skel_u8 = (skeleton * 255).astype(np.uint8)
            red = np.where(cv2.dilate(skel_u8, kernel, iterations=1) > 0, 255, 0).astype(np.uint8)
        elif solidity > 0.8:
            red = np.where(mask > 0, 200, 0).astype(np.uint8)
        else:
            edges = cv2.Canny(mask.astype(np.uint8) * 255, 100, 200)
            kernel = np.ones((3, 3), np.uint8)
            dilated = cv2.dilate(edges, kernel, iterations=1)
            red = np.where(mask > 0, 100, 0).astype(np.uint8)
            red = np.where(dilated > 0, 255, red).astype(np.uint8)

        return red

    def generate_green_channel(
        self, image: np.ndarray, background_type: str, stability_score: float
    ) -> np.ndarray:
        """G: 배경 구조선 (방향성 Sobel 엣지)."""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image.copy()
        h, w = gray.shape
        green = np.zeros((h, w), dtype=np.uint8)

        if not self.enhance_background:
            return green

        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

        if background_type in ('vertical_stripe', 'directional'):
            edge_map = np.abs(sobel_x)
        elif background_type == 'horizontal_stripe':
            edge_map = np.abs(sobel_y)
        elif background_type in ('complex_pattern', 'periodic'):
            edge_map = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
        else:
            edge_map = np.sqrt(sobel_x ** 2 + sobel_y ** 2) * 0.3

        edge_norm = cv2.normalize(edge_map, None, 0, 255, cv2.NORM_MINMAX)
        factor = 0.5 + 0.5 * stability_score
        green = (edge_norm * factor).astype(np.uint8)
        return green

    def generate_blue_channel(self, image: np.ndarray, background_type: str) -> np.ndarray:
        """B: 배경 세밀 텍스처 (국소 분산)."""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image.copy()
        h, w = gray.shape

        if background_type == 'smooth':
            return np.full((h, w), 20, dtype=np.uint8)

        ksize = 7
        kernel = np.ones((ksize, ksize), np.float32) / (ksize * ksize)
        local_mean = cv2.filter2D(gray.astype(np.float32), -1, kernel)
        local_sq_mean = cv2.filter2D(gray.astype(np.float32) ** 2, -1, kernel)
        local_var = np.maximum(local_sq_mean - local_mean ** 2, 0)
        blue = cv2.normalize(local_var, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        if background_type in ('textured', 'complex_pattern', 'periodic'):
            blue = np.clip(blue * 1.2, 0, 255).astype(np.uint8)

        return blue

    def generate_hint_image(
        self,
        roi_image: np.ndarray,
        roi_mask: np.ndarray,
        defect_metrics: Dict,
        background_type: str,
        stability_score: float,
    ) -> np.ndarray:
        """3채널 hint 이미지 생성 → 가중합 그레이스케일로 변환 후 3채널 복제.

        RGB 색상 신호를 제거하여 ControlNet의 색상 단축 학습을 방지한다.
        """
        red = self.generate_red_channel(roi_mask, defect_metrics)
        green = self.generate_green_channel(roi_image, background_type, stability_score)
        blue = self.generate_blue_channel(roi_image, background_type)

        gray = np.clip(
            0.5 * red.astype(np.float32)
            + 0.3 * green.astype(np.float32)
            + 0.2 * blue.astype(np.float32),
            0, 255,
        ).astype(np.uint8)

        return np.stack([gray, gray, gray], axis=2)

    def save_hint_image(self, hint_image: np.ndarray, output_path: Path) -> None:
        cv2.imwrite(str(output_path), cv2.cvtColor(hint_image, cv2.COLOR_RGB2BGR))
