# utils/defect_characterization.py
"""Defect characterization utility — geometric indicator analysis.

Ports the DefectCharacterizer from CASDA, computing four geometric indicators
for defect regions and classifying them into subtypes.
"""

from __future__ import annotations

import numpy as np
from skimage.measure import label, regionprops


class DefectCharacterizer:
    """Analyze geometric properties of defect regions and classify their subtype."""

    # ---------------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------------

    def analyze_defect_region(self, mask: np.ndarray) -> dict | None:
        """Return geometric metrics for the largest connected region in *mask*.

        Parameters
        ----------
        mask:
            2-D uint8 array with pixel values 0/1 or 0/255.

        Returns
        -------
        dict with keys ``linearity``, ``solidity``, ``extent``,
        ``aspect_ratio``, ``region_id``, ``area``, ``bbox``, ``centroid``,
        or ``None`` if the mask contains no foreground pixels.
        """
        # Normalise to binary 0/1
        binary = (mask > 0).astype(np.uint8)

        labeled = label(binary, connectivity=2)
        props = regionprops(labeled)

        if not props:
            return None

        # Pick the largest region by area
        region = max(props, key=lambda r: r.area)

        linearity = self._compute_linearity(region)
        aspect_ratio = self._compute_aspect_ratio(region)

        return {
            "linearity": linearity,
            "solidity": float(region.solidity),
            "extent": float(region.extent),
            "aspect_ratio": aspect_ratio,
            "region_id": int(region.label),
            "area": int(region.area),
            "bbox": region.bbox,          # (min_row, min_col, max_row, max_col)
            "centroid": region.centroid,  # (row, col)
        }

    def classify_defect_subtype(self, metrics: dict) -> str:
        """Classify a defect into a subtype string from its metrics dict.

        Classification rules (evaluated in priority order):
        1. linear_scratch  — linearity > 0.85 AND aspect_ratio > 5.0
        2. elongated       — aspect_ratio > 5.0 AND linearity > 0.6
        3. compact_blob    — aspect_ratio < 2.0 AND solidity > 0.9
        4. irregular       — solidity < 0.7
        5. general         — otherwise
        """
        linearity = metrics["linearity"]
        aspect_ratio = metrics["aspect_ratio"]
        solidity = metrics["solidity"]

        if linearity > 0.85 and aspect_ratio > 5.0:
            return "linear_scratch"
        if aspect_ratio > 5.0 and linearity > 0.6:
            return "elongated"
        if aspect_ratio < 2.0 and solidity > 0.9:
            return "compact_blob"
        if solidity < 0.7:
            return "irregular"
        return "general"

    # ---------------------------------------------------------------------------
    # Private helpers
    # ---------------------------------------------------------------------------

    @staticmethod
    def _compute_linearity(region) -> float:
        """Eigenvalue ratio of the covariance matrix of pixel coordinates.

        linearity = 1 - (lambda_min / lambda_max)

        A perfect line has lambda_min ≈ 0, so linearity ≈ 1.0.
        A circular blob has lambda_min ≈ lambda_max, so linearity ≈ 0.
        """
        coords = region.coords.astype(float)  # shape (N, 2) — (row, col)
        if coords.shape[0] < 2:
            return 0.0

        cov = np.cov(coords, rowvar=False)  # 2×2 covariance matrix
        eigenvalues = np.linalg.eigvalsh(cov)  # sorted ascending
        lambda_min, lambda_max = float(eigenvalues[0]), float(eigenvalues[1])

        if lambda_max < 1e-10:
            return 0.0

        return 1.0 - (lambda_min / lambda_max)

    @staticmethod
    def _compute_aspect_ratio(region) -> float:
        """Major axis length / minor axis length (protected against zero division)."""
        # Use new attribute names (skimage >= 0.26); fall back for older versions.
        minor = getattr(region, "axis_minor_length", None) or getattr(region, "minor_axis_length", 0.0)
        major = getattr(region, "axis_major_length", None) or getattr(region, "major_axis_length", 0.0)
        if minor < 1e-10:
            return 99.0
        return float(major / minor)
