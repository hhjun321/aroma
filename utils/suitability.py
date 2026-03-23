"""
Suitability Scoring — unified domain-agnostic matching rules.

A single table covers ISP-AD, MVTec AD, and VisA.
Structural relationships between defect geometry and background structure
are domain-invariant, so no per-domain branching is needed.
"""


# Unified matching rules — background types: smooth | directional | periodic | organic | complex
MATCHING_RULES = {
    "linear_scratch": {
        "smooth": 0.5, "directional": 1.0, "periodic": 0.7, "organic": 0.3, "complex": 0.3,
    },
    "elongated": {
        "smooth": 0.6, "directional": 0.9, "periodic": 0.7, "organic": 0.4, "complex": 0.4,
    },
    "compact_blob": {
        "smooth": 0.9, "directional": 0.4, "periodic": 0.7, "organic": 0.6, "complex": 0.5,
    },
    "irregular": {
        "smooth": 0.5, "directional": 0.4, "periodic": 0.5, "organic": 0.8, "complex": 0.9,
    },
    "general": {
        "smooth": 0.7, "directional": 0.7, "periodic": 0.7, "organic": 0.7, "complex": 0.7,
    },
}


class SuitabilityEvaluator:
    """Compute hybrid suitability score for defect-ROI placement."""

    W_MATCHING   = 0.4
    W_CONTINUITY = 0.3
    W_STABILITY  = 0.2
    W_GRAM       = 0.1

    def matching_score(self, defect_subtype: str, background_type: str) -> float:
        row = MATCHING_RULES.get(defect_subtype, MATCHING_RULES["general"])
        return float(row.get(background_type, 0.5))

    def compute_suitability(
        self,
        defect_subtype: str,
        background_type: str,
        continuity_score: float,
        stability_score: float,
        gram_similarity: float = 0.0,
    ) -> float:
        m = self.matching_score(defect_subtype, background_type)
        score = (
            self.W_MATCHING   * m
            + self.W_CONTINUITY * continuity_score
            + self.W_STABILITY  * stability_score
            + self.W_GRAM       * gram_similarity
        )
        return float(min(max(score, 0.0), 1.0))


class GPUSuitabilityEvaluator:
    """GPU-accelerated batch suitability scoring (PyTorch). Falls back to CPU gracefully."""

    BG_TYPES = ["smooth", "directional", "periodic", "organic", "complex"]
    SUBTYPES = ["linear_scratch", "elongated", "compact_blob", "irregular", "general"]

    def __init__(self, device: str = "cuda") -> None:
        import torch
        self._torch = torch
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.bg_map = {n: i for i, n in enumerate(self.BG_TYPES)}
        self.st_map = {n: i for i, n in enumerate(self.SUBTYPES)}

        W = [
            SuitabilityEvaluator.W_MATCHING,
            SuitabilityEvaluator.W_CONTINUITY,
            SuitabilityEvaluator.W_STABILITY,
            SuitabilityEvaluator.W_GRAM,
        ]
        self.W = torch.tensor(W, dtype=torch.float32, device=self.device)

        rules = torch.tensor(
            [[MATCHING_RULES.get(st, MATCHING_RULES["general"]).get(bg, 0.5)
              for bg in self.BG_TYPES]
             for st in self.SUBTYPES],
            dtype=torch.float32,
            device=self.device,
        )
        self.RULES_MATRIX = rules

    def compute_batch(self, defect_subtype: str, roi_boxes: list) -> "np.ndarray":
        """Return suitability scores for all roi_boxes as numpy array."""
        import numpy as np
        if not roi_boxes:
            return np.array([], dtype=np.float32)
        torch = self._torch
        st_idx     = self.st_map.get(defect_subtype, self.st_map["general"])
        bg_indices = [self.bg_map.get(b.get("background_type", "smooth"), 0) for b in roi_boxes]
        m_scores   = self.RULES_MATRIX[st_idx, bg_indices]
        c_scores   = torch.tensor([float(b.get("continuity_score") or 0.5) for b in roi_boxes],
                                   dtype=torch.float32, device=self.device)
        s_scores   = torch.tensor([float(b.get("stability_score") or 0.5) for b in roi_boxes],
                                   dtype=torch.float32, device=self.device)
        g_scores   = torch.zeros(len(roi_boxes), dtype=torch.float32, device=self.device)
        features   = torch.stack([m_scores, c_scores, s_scores, g_scores], dim=1)
        return torch.clamp(torch.matmul(features, self.W), 0.0, 1.0).cpu().numpy()
