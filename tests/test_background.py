import numpy as np
import pytest
from utils.background_characterization import BackgroundAnalyzer, BackgroundType


@pytest.fixture
def smooth_image():
    return np.full((128, 128, 3), 180, dtype=np.uint8)


@pytest.fixture
def stripe_image():
    img = np.zeros((128, 128, 3), dtype=np.uint8)
    for i in range(0, 128, 8):
        img[:, i:i+4] = 200  # vertical stripes
    return img


def test_smooth_patch_classified_as_smooth(smooth_image):
    analyzer = BackgroundAnalyzer(grid_size=64)
    result = analyzer.analyze_image(smooth_image)
    bg = result["background_map"][0, 0]
    assert bg == BackgroundType.SMOOTH.value


def test_stripe_patch_classified_as_vertical_stripe(stripe_image):
    analyzer = BackgroundAnalyzer(grid_size=64)
    result = analyzer.analyze_image(stripe_image)
    bg = result["background_map"][0, 0]
    assert bg == BackgroundType.VERTICAL_STRIPE.value


def test_continuity_uniform_region_is_high():
    analyzer = BackgroundAnalyzer(grid_size=32)
    img = np.full((128, 128, 3), 180, dtype=np.uint8)
    result = analyzer.analyze_image(img)
    score = analyzer.check_continuity(result, (0, 0, 128, 128))
    assert score > 0.7


def test_background_at_location_returns_dict(smooth_image):
    analyzer = BackgroundAnalyzer(grid_size=64)
    result = analyzer.analyze_image(smooth_image)
    info = analyzer.get_background_at_location(result, 32, 32)
    assert info is not None
    assert "background_type" in info
    assert "stability_score" in info
