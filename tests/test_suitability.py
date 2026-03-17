# tests/test_suitability.py
import pytest
from utils.suitability import SuitabilityEvaluator

def test_isp_linear_scratch_on_stripe_is_high():
    ev = SuitabilityEvaluator(domain="isp")
    score = ev.matching_score("linear_scratch", "horizontal_stripe")
    assert score == 1.0

def test_isp_compact_blob_on_smooth_is_high():
    ev = SuitabilityEvaluator(domain="isp")
    score = ev.matching_score("compact_blob", "smooth")
    assert score == 1.0

def test_isp_linear_scratch_on_complex_is_low():
    ev = SuitabilityEvaluator(domain="isp")
    score = ev.matching_score("linear_scratch", "complex_pattern")
    assert score < 0.5

def test_mvtec_general_returns_neutral():
    ev = SuitabilityEvaluator(domain="mvtec")
    for bg in ["smooth", "textured", "vertical_stripe", "horizontal_stripe", "complex_pattern"]:
        score = ev.matching_score("general", bg)
        assert abs(score - 0.7) < 0.01

def test_compute_suitability_returns_float_in_range():
    ev = SuitabilityEvaluator(domain="isp")
    score = ev.compute_suitability(
        defect_subtype="linear_scratch",
        background_type="horizontal_stripe",
        continuity_score=0.8,
        stability_score=0.75,
        gram_similarity=0.6
    )
    assert 0.0 <= score <= 1.0

def test_unknown_domain_falls_back_to_general_rules():
    ev = SuitabilityEvaluator(domain="unknown_domain")
    score = ev.matching_score("linear_scratch", "smooth")
    assert 0.0 <= score <= 1.0
