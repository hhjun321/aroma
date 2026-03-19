# tests/test_defect_characterization.py
import numpy as np
import pytest
from utils.defect_characterization import DefectCharacterizer

@pytest.fixture
def linear_mask():
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[30:32, 5:60] = 1  # thin horizontal line
    return mask

@pytest.fixture
def blob_mask():
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[20:40, 20:40] = 1  # square blob
    return mask

def test_linear_mask_high_linearity(linear_mask):
    dc = DefectCharacterizer()
    metrics = dc.analyze_defect_region(linear_mask)
    assert metrics is not None
    assert metrics["linearity"] > 0.7

def test_blob_mask_low_aspect_ratio(blob_mask):
    dc = DefectCharacterizer()
    metrics = dc.analyze_defect_region(blob_mask)
    assert metrics is not None
    assert metrics["aspect_ratio"] < 3.0

def test_linear_mask_classified_as_linear_scratch(linear_mask):
    dc = DefectCharacterizer()
    metrics = dc.analyze_defect_region(linear_mask)
    subtype = dc.classify_defect_subtype(metrics)
    assert subtype == "linear_scratch"

def test_blob_mask_classified_as_compact_blob(blob_mask):
    dc = DefectCharacterizer()
    metrics = dc.analyze_defect_region(blob_mask)
    subtype = dc.classify_defect_subtype(metrics)
    assert subtype == "compact_blob"

def test_empty_mask_returns_none():
    dc = DefectCharacterizer()
    mask = np.zeros((64, 64), dtype=np.uint8)
    result = dc.analyze_defect_region(mask)
    assert result is None
