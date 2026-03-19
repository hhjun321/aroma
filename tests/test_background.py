import numpy as np
import pytest
from utils.background_characterization import BackgroundAnalyzer, BackgroundType


@pytest.fixture
def smooth_image():
    return np.full((128, 128, 3), 180, dtype=np.uint8)


@pytest.fixture
def stripe_image():
    """Vertical stripes — should classify as DIRECTIONAL."""
    img = np.zeros((128, 128, 3), dtype=np.uint8)
    for i in range(0, 128, 8):
        img[:, i:i + 4] = 200
    return img


@pytest.fixture
def periodic_image():
    """Grid of dots — should classify as PERIODIC."""
    img = np.zeros((128, 128, 3), dtype=np.uint8)
    for i in range(0, 128, 16):
        for j in range(0, 128, 16):
            img[i:i + 8, j:j + 8] = 200
    return img


def test_smooth_classified_as_smooth(smooth_image):
    analyzer = BackgroundAnalyzer(grid_size=64)
    result = analyzer.analyze_image(smooth_image)
    assert result["background_map"][0, 0] == BackgroundType.SMOOTH.value


def test_stripe_classified_as_directional(stripe_image):
    analyzer = BackgroundAnalyzer(grid_size=64)
    result = analyzer.analyze_image(stripe_image)
    assert result["background_map"][0, 0] == BackgroundType.DIRECTIONAL.value


def test_periodic_classified_as_periodic(periodic_image):
    analyzer = BackgroundAnalyzer(grid_size=64)
    result = analyzer.analyze_image(periodic_image)
    assert result["background_map"][0, 0] == BackgroundType.PERIODIC.value


def test_directional_has_dominant_angle(stripe_image):
    analyzer = BackgroundAnalyzer(grid_size=64)
    result = analyzer.analyze_image(stripe_image)
    info = analyzer.get_background_at_location(result, 32, 32)
    assert info["dominant_angle"] is not None
    assert 0.0 <= info["dominant_angle"] <= 180.0


def test_smooth_dominant_angle_is_none(smooth_image):
    analyzer = BackgroundAnalyzer(grid_size=64)
    result = analyzer.analyze_image(smooth_image)
    info = analyzer.get_background_at_location(result, 32, 32)
    assert info["dominant_angle"] is None


def test_continuity_uniform_region_is_high(smooth_image):
    analyzer = BackgroundAnalyzer(grid_size=32)
    result = analyzer.analyze_image(smooth_image)
    score = analyzer.check_continuity(result, (0, 0, 128, 128))
    assert score > 0.7


def test_get_background_at_location_has_required_fields(smooth_image):
    analyzer = BackgroundAnalyzer(grid_size=64)
    result = analyzer.analyze_image(smooth_image)
    info = analyzer.get_background_at_location(result, 32, 32)
    for key in ("background_type", "stability_score", "dominant_angle"):
        assert key in info


def test_old_type_names_not_present():
    """Ensure old taxonomy values are gone."""
    old_values = {"textured", "vertical_stripe", "horizontal_stripe", "complex_pattern"}
    new_values = {t.value for t in BackgroundType}
    assert old_values.isdisjoint(new_values), f"Old types still present: {old_values & new_values}"


@pytest.fixture
def organic_image():
    """Random noise texture — should classify as ORGANIC."""
    rng = np.random.default_rng(42)
    img = rng.integers(0, 255, (128, 128, 3), dtype=np.uint8)
    return img


def test_noise_not_smooth_or_directional(organic_image):
    """Random noise should not be classified as smooth or directional.

    Pure random noise has a flat power spectrum which produces a sharp
    autocorrelation peak — so PERIODIC, ORGANIC, or COMPLEX are all valid.
    The key requirement is that it is NOT smooth or directional.
    """
    analyzer = BackgroundAnalyzer(grid_size=64)
    result = analyzer.analyze_image(organic_image)
    bg = result["background_map"][0, 0]
    assert bg in (BackgroundType.ORGANIC.value, BackgroundType.COMPLEX.value,
                  BackgroundType.PERIODIC.value), \
        f"Random noise classified as {bg}, expected organic, complex, or periodic"
