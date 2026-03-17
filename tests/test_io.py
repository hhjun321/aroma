import json, pytest
from pathlib import Path
from utils.io import save_json, load_json, validate_dir


def test_save_and_load_json(tmp_path):
    data = {"image_id": "img_001", "roi_boxes": []}
    path = tmp_path / "meta.json"
    save_json(data, path)
    loaded = load_json(path)
    assert loaded == data


def test_validate_dir_raises_if_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        validate_dir(tmp_path / "nonexistent")


def test_validate_dir_passes_if_exists(tmp_path):
    validate_dir(tmp_path)
