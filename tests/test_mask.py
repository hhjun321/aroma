import numpy as np
from utils.mask import save_mask, load_mask


def test_save_and_load_mask(tmp_path):
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[10:30, 10:30] = 255
    path = tmp_path / "mask.png"
    save_mask(mask, path)
    loaded = load_mask(path)
    assert loaded.shape == (64, 64)
    assert loaded[20, 20] == 255
    assert loaded[0, 0] == 0
