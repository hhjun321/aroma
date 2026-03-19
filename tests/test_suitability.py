import pytest
from utils.suitability import SuitabilityEvaluator


def test_linear_scratch_on_directional_is_max():
    ev = SuitabilityEvaluator()
    score = ev.matching_score("linear_scratch", "directional")
    assert score == 1.0


def test_compact_blob_on_smooth_is_high():
    ev = SuitabilityEvaluator()
    score = ev.matching_score("compact_blob", "smooth")
    assert score == 0.9


def test_linear_scratch_on_organic_is_low():
    ev = SuitabilityEvaluator()
    score = ev.matching_score("linear_scratch", "organic")
    assert score < 0.5


def test_general_is_uniform_across_all_types():
    ev = SuitabilityEvaluator()
    bg_types = ["smooth", "directional", "periodic", "organic", "complex"]
    scores = [ev.matching_score("general", bg) for bg in bg_types]
    assert all(abs(s - 0.7) < 0.01 for s in scores)


def test_irregular_on_complex_is_highest_for_that_subtype():
    ev = SuitabilityEvaluator()
    score_complex = ev.matching_score("irregular", "complex")
    score_smooth = ev.matching_score("irregular", "smooth")
    assert score_complex > score_smooth


def test_compute_suitability_returns_float_in_range():
    ev = SuitabilityEvaluator()
    score = ev.compute_suitability(
        defect_subtype="linear_scratch",
        background_type="directional",
        continuity_score=0.8,
        stability_score=0.75,
        gram_similarity=0.6,
    )
    assert 0.0 <= score <= 1.0


def test_unknown_subtype_falls_back_to_general():
    ev = SuitabilityEvaluator()
    score = ev.matching_score("nonexistent_type", "smooth")
    assert abs(score - 0.7) < 0.01


def test_no_domain_argument_needed():
    """SuitabilityEvaluator takes no domain argument."""
    ev = SuitabilityEvaluator()
    assert ev is not None
