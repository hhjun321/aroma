from typing import Optional

MATCHING_RULES = {
    "isp": {
        "linear_scratch":  {"vertical_stripe": 1.0, "horizontal_stripe": 1.0, "smooth": 0.5, "textured": 0.4, "complex_pattern": 0.3},
        "elongated":       {"vertical_stripe": 0.9, "horizontal_stripe": 0.9, "smooth": 0.6, "textured": 0.5, "complex_pattern": 0.4},
        "compact_blob":    {"vertical_stripe": 0.5, "horizontal_stripe": 0.5, "smooth": 1.0, "textured": 0.6, "complex_pattern": 0.4},
        "irregular":       {"vertical_stripe": 0.4, "horizontal_stripe": 0.4, "smooth": 0.5, "textured": 0.8, "complex_pattern": 1.0},
        "general":         {"vertical_stripe": 0.7, "horizontal_stripe": 0.7, "smooth": 0.7, "textured": 0.7, "complex_pattern": 0.7},
    },
    "mvtec": {
        "linear_scratch":  {"vertical_stripe": 0.8, "horizontal_stripe": 0.8, "smooth": 0.6, "textured": 0.5, "complex_pattern": 0.3},
        "elongated":       {"vertical_stripe": 0.7, "horizontal_stripe": 0.7, "smooth": 0.6, "textured": 0.6, "complex_pattern": 0.4},
        "compact_blob":    {"vertical_stripe": 0.4, "horizontal_stripe": 0.4, "smooth": 0.9, "textured": 0.7, "complex_pattern": 0.5},
        "irregular":       {"vertical_stripe": 0.4, "horizontal_stripe": 0.4, "smooth": 0.5, "textured": 0.8, "complex_pattern": 0.9},
        "general":         {"vertical_stripe": 0.7, "horizontal_stripe": 0.7, "smooth": 0.7, "textured": 0.7, "complex_pattern": 0.7},
    },
}
# Fallback: use "general" row from "isp" for unknown domains
_FALLBACK_RULES = MATCHING_RULES["isp"]["general"]


class SuitabilityEvaluator:
    # Hybrid score weights
    W_MATCHING    = 0.4
    W_CONTINUITY  = 0.3
    W_STABILITY   = 0.2
    W_GRAM        = 0.1

    def __init__(self, domain: str = "isp"):
        self._rules = MATCHING_RULES.get(domain, MATCHING_RULES["isp"])

    def matching_score(self, defect_subtype: str, background_type: str) -> float:
        row = self._rules.get(defect_subtype, self._rules.get("general", _FALLBACK_RULES))
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
