"""
Background Characterization Module

Universal structure-based background taxonomy (domain-agnostic):
  smooth | directional | periodic | organic | complex

Detection pipeline (lazy evaluation, cheapest first):
  1. Variance         → smooth
  2. Gradient entropy → directional  (also computes dominant_angle)
  3. Autocorrelation  → periodic
  4. LBP entropy      → organic
  5. Fallback         → complex
"""
import numpy as np
import cv2
from typing import Dict, Optional, Tuple
from enum import Enum


class BackgroundType(Enum):
    SMOOTH      = "smooth"
    DIRECTIONAL = "directional"
    PERIODIC    = "periodic"
    ORGANIC     = "organic"
    COMPLEX     = "complex"


class BackgroundAnalyzer:
    """Grid-based background texture analyzer using universal structure-based taxonomy."""

    def __init__(
        self,
        grid_size: int = 64,
        variance_threshold: float = 100.0,
        periodic_threshold: float = 0.15,
        direction_entropy_threshold: float = 1.0,
        organic_entropy_threshold: float = 2.5,
    ):
        self.grid_size = grid_size
        self.variance_threshold = variance_threshold
        self.periodic_threshold = periodic_threshold
        self.direction_entropy_threshold = direction_entropy_threshold
        self.organic_entropy_threshold = organic_entropy_threshold

    def _compute_autocorrelation_peak(self, patch: np.ndarray) -> float:
        """Returns normalized off-origin autocorrelation peak (0-1). High = periodic."""
        f = np.fft.fft2(patch.astype(np.float32))
        ac = np.real(np.fft.ifft2(np.abs(f) ** 2))
        ac = ac / (ac[0, 0] + 1e-6)
        ac[0, 0] = 0.0
        return float(np.max(ac))

    def _compute_gradient_direction_entropy(
        self, patch: np.ndarray
    ) -> Tuple[float, Optional[float]]:
        """
        Returns (entropy_bits, dominant_angle_degrees).
        Low entropy → directional pattern. dominant_angle is None when entropy is high.
        Angles are in [0, 180) — orientation, not direction.
        """
        sobel_x = cv2.Sobel(patch, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(patch, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
        angles = np.degrees(np.arctan2(sobel_y, sobel_x)) % 180.0  # orientation

        bins = np.linspace(0.0, 180.0, 9)  # 8 bins × 22.5°
        counts, _ = np.histogram(angles.flatten(), bins=bins,
                                 weights=magnitude.flatten())
        total = np.sum(counts) + 1e-6
        probs = counts / total
        entropy = float(-np.sum(probs * np.log2(probs + 1e-12)))

        dominant_angle = None
        if entropy < self.direction_entropy_threshold:
            bin_idx = int(np.argmax(counts))
            dominant_angle = float(bins[bin_idx] + 11.25)  # bin centre

        return entropy, dominant_angle

    def _compute_lbp_entropy(self, patch: np.ndarray) -> float:
        """Returns LBP histogram entropy. High = organic/random texture."""
        from skimage.feature import local_binary_pattern
        lbp = local_binary_pattern(patch, P=8, R=1, method="uniform")
        counts, _ = np.histogram(lbp.ravel(), bins=10, density=True)
        counts = counts + 1e-12
        return float(-np.sum(counts * np.log2(counts)))

    def classify_patch(
        self, patch: np.ndarray
    ) -> Tuple[BackgroundType, float, Optional[float]]:
        """
        Classify a grayscale patch.

        Returns:
            (BackgroundType, stability_score 0-1, dominant_angle degrees or None)
            dominant_angle is only set when BackgroundType is DIRECTIONAL.
        """
        variance = float(np.var(patch))

        # Step 1: smooth
        if variance < self.variance_threshold:
            stability = float(np.clip(1.0 - variance / self.variance_threshold, 0.0, 1.0))
            return BackgroundType.SMOOTH, stability, None

        # Step 2: directional (checked before periodic — stripe patterns are both
        # periodic and directional, but low gradient entropy is the stronger signal)
        entropy, dominant_angle = self._compute_gradient_direction_entropy(patch)
        if entropy < self.direction_entropy_threshold:
            stability = float(np.clip(1.0 - entropy / self.direction_entropy_threshold, 0.0, 1.0))
            return BackgroundType.DIRECTIONAL, stability, dominant_angle

        # Step 3: periodic
        ac_peak = self._compute_autocorrelation_peak(patch)
        if ac_peak > self.periodic_threshold:
            return BackgroundType.PERIODIC, float(np.clip(ac_peak, 0.0, 1.0)), None

        # Step 4: organic
        lbp_entropy = self._compute_lbp_entropy(patch)
        if lbp_entropy > self.organic_entropy_threshold:
            return BackgroundType.ORGANIC, 0.5, None

        # Step 5: complex
        return BackgroundType.COMPLEX, 0.3, None

    def analyze_image(self, image: np.ndarray) -> Dict:
        """Analyze full image using grid-based approach."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        h, w = gray.shape
        gs = self.grid_size
        grid_h, grid_w = h // gs, w // gs

        background_map = np.empty((grid_h, grid_w), dtype=object)
        stability_map = np.zeros((grid_h, grid_w), dtype=np.float32)
        angle_map = np.full((grid_h, grid_w), None, dtype=object)
        grid_info = []

        for i in range(grid_h):
            for j in range(grid_w):
                y1, x1 = i * gs, j * gs
                patch = gray[y1:y1 + gs, x1:x1 + gs]
                bg_type, stability, dominant_angle = self.classify_patch(patch)
                background_map[i, j] = bg_type.value
                stability_map[i, j] = stability
                angle_map[i, j] = dominant_angle
                grid_info.append({
                    "grid_id": (i, j),
                    "bbox": (x1, y1, x1 + gs, y1 + gs),
                    "background_type": bg_type.value,
                    "stability_score": float(stability),
                    "dominant_angle": dominant_angle,
                })

        return {
            "background_map": background_map,
            "stability_map": stability_map,
            "angle_map": angle_map,
            "grid_info": grid_info,
            "grid_size": gs,
            "grid_shape": (grid_h, grid_w),
        }

    def get_background_at_location(
        self, analysis_result: Dict, x: int, y: int
    ) -> Optional[Dict]:
        gs = analysis_result["grid_size"]
        grid_h, grid_w = analysis_result["grid_shape"]
        gi, gj = y // gs, x // gs
        if not (0 <= gi < grid_h and 0 <= gj < grid_w):
            return None
        return {
            "background_type": analysis_result["background_map"][gi, gj],
            "stability_score": float(analysis_result["stability_map"][gi, gj]),
            "dominant_angle": analysis_result["angle_map"][gi, gj],
            "grid_id": (gi, gj),
        }

    def check_continuity(
        self, analysis_result: Dict, bbox: Tuple[int, int, int, int]
    ) -> float:
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
