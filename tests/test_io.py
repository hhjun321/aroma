import json, pytest
from pathlib import Path
from utils.io import save_json, load_json, validate_dir, validate_file


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


def test_validate_dir_raises_if_file(tmp_path):
    """validate_dir should reject a file path."""
    f = tmp_path / "afile.txt"
    f.write_text("hello")
    with pytest.raises(FileNotFoundError, match="not a directory"):
        validate_dir(f)


def test_validate_file_raises_if_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        validate_file(tmp_path / "nonexistent.json")


def test_validate_file_passes_if_exists(tmp_path):
    f = tmp_path / "data.json"
    f.write_text("{}")
    validate_file(f)


def test_validate_file_raises_if_directory(tmp_path):
    """validate_file should reject a directory path."""
    with pytest.raises(FileNotFoundError, match="not a file"):
        validate_file(tmp_path)
