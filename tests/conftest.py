import numpy as np
import pytest


@pytest.fixture
def synthetic_image():
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    img[10:50, 10:50] = [120, 80, 60]
    return img


@pytest.fixture
def synthetic_defect():
    patch = np.ones((16, 16, 3), dtype=np.uint8) * 200
    patch[4:12, 4:12] = [30, 30, 30]
    return patch


@pytest.fixture
def temp_image_dir(tmp_path, synthetic_image):
    import cv2
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    cv2.imwrite(str(img_dir / "img_001.png"), synthetic_image)
    return img_dir
