"""
Background Characterization Module

Grid-based background texture analysis.
Classifies image regions into 5 background types:
  smooth, textured, vertical_stripe, horizontal_stripe, complex_pattern

Ported from CASDA and made domain-agnostic:
- variance_threshold and grid_size are constructor parameters
- Lazy evaluation: variance → Sobel → FFT (cheapest first)
"""
import numpy as np
import cv2
from typing import Dict, Optional, Tuple
from enum import Enum


class BackgroundType(Enum):
    SMOOTH = "smooth"
    TEXTURED = "textured"
    VERTICAL_STRIPE = "vertical_stripe"
    HORIZONTAL_STRIPE = "horizontal_stripe"
    COMPLEX_PATTERN = "complex_pattern"


class BackgroundAnalyzer:
    """Grid-based background texture analyzer."""

    def __init__(self, grid_size: int = 64, variance_threshold: float = 100.0,
                 edge_threshold: float = 0.3):
        self.grid_size = grid_size
        self.variance_threshold = variance_threshold
        self.edge_threshold = edge_threshold

    def _compute_edge_directions(self, patch: np.ndarray) -> Dict[str, float]:
        sobel_x = cv2.Sobel(patch, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(patch, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
        return {
            "vertical": float(np.mean(np.abs(sobel_x))),
            "horizontal": float(np.mean(np.abs(sobel_y))),
            "total": float(np.mean(magnitude)),
        }

    def _compute_high_freq_ratio(self, patch: np.ndarray) -> float:
        f_shift = np.fft.fftshift(np.fft.fft2(patch))
        magnitude = np.abs(f_shift)
        h, w = patch.shape
        cy, cx = h // 2, w // 2
        radius = min(h, w) // 4
        y, x = np.ogrid[:h, :w]
        center_mask = (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2
        center_energy = np.sum(magnitude[center_mask])
        total_energy = np.sum(magnitude)
        return float(1.0 - center_energy / (total_energy + 1e-6))

    def classify_patch(self, patch: np.ndarray) -> Tuple[BackgroundType, float]:
        """Classify a grayscale patch. Returns (BackgroundType, stability_score)."""
        variance = float(np.var(patch))

        # Fast path: low variance → smooth
        if variance < self.variance_threshold:
            stability = float(np.clip(1.0 - variance / self.variance_threshold, 0.0, 1.0))
            return BackgroundType.SMOOTH, stability

        edge = self._compute_edge_directions(patch)
        total = edge["total"]

        if total < 1.0:
            return BackgroundType.TEXTURED, 0.5

        v_ratio = edge["vertical"] / (total + 1e-6)
        h_ratio = edge["horizontal"] / (total + 1e-6)

        if v_ratio > self.edge_threshold and v_ratio > h_ratio * 1.5:
            return BackgroundType.VERTICAL_STRIPE, float(np.clip(v_ratio, 0.0, 1.0))
        if h_ratio > self.edge_threshold and h_ratio > v_ratio * 1.5:
            return BackgroundType.HORIZONTAL_STRIPE, float(np.clip(h_ratio, 0.0, 1.0))

        # Only compute FFT when needed
        hf = self._compute_high_freq_ratio(patch)
        if hf > 0.3:
            return BackgroundType.COMPLEX_PATTERN, float(np.clip(1.0 - hf, 0.0, 1.0))

        return BackgroundType.TEXTURED, 0.5

    def analyze_image(self, image: np.ndarray) -> Dict:
        """Analyze full image using grid-based approach."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        h, w = gray.shape
        gs = self.grid_size
        grid_h = h // gs
        grid_w = w // gs

        background_map = np.empty((grid_h, grid_w), dtype=object)
        stability_map = np.zeros((grid_h, grid_w), dtype=np.float32)
        grid_info = []

        for i in range(grid_h):
            y1 = i * gs
            for j in range(grid_w):
                x1 = j * gs
                patch = gray[y1:y1 + gs, x1:x1 + gs]
                bg_type, stability = self.classify_patch(patch)
                background_map[i, j] = bg_type.value
                stability_map[i, j] = stability
                grid_info.append({
                    "grid_id": (i, j),
                    "bbox": (x1, y1, x1 + gs, y1 + gs),
                    "background_type": bg_type.value,
                    "stability_score": float(stability),
                })

        return {
            "background_map": background_map,
            "stability_map": stability_map,
            "grid_info": grid_info,
            "grid_size": gs,
            "grid_shape": (grid_h, grid_w),
        }

    def get_background_at_location(self, analysis_result: Dict, x: int, y: int) -> Optional[Dict]:
        gs = analysis_result["grid_size"]
        grid_h, grid_w = analysis_result["grid_shape"]
        gi, gj = y // gs, x // gs
        if not (0 <= gi < grid_h and 0 <= gj < grid_w):
            return None
        return {
            "background_type": analysis_result["background_map"][gi, gj],
            "stability_score": float(analysis_result["stability_map"][gi, gj]),
            "grid_id": (gi, gj),
        }

    def check_continuity(self, analysis_result: Dict,
                         bbox: Tuple[int, int, int, int]) -> float:
        x1, y1, x2, y2 = bbox
        gs = analysis_result["grid_size"]
        grid_h, grid_w = analysis_result["grid_shape"]
        gi1, gj1 = y1 // gs, x1 // gs
        gi2 = min(y2 // gs, grid_h - 1)
        gj2 = min(x2 // gs, grid_w - 1)

        bg_map = analysis_result["background_map"]
        stab_map = analysis_result["stability_map"]
        region_bg = bg_map[gi1:gi2 + 1, gj1:gj2 + 1]
        region_stab = stab_map[gi1:gi2 + 1, gj1:gj2 + 1]

        if region_bg.size == 0:
            return 0.0

        _, counts = np.unique(region_bg, return_counts=True)
        uniformity = float(np.max(counts) / region_bg.size)
        avg_stability = float(np.mean(region_stab))
        return float(0.6 * uniformity + 0.4 * avg_stability)
