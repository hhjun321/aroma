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
