# tests/test_prepare_visa.py
import csv
import os
from pathlib import Path
import pytest
from prepare_visa import reorganize_category, reorganize_all


@pytest.fixture
def fake_visa(tmp_path):
    """Create a minimal VisA-like directory structure with 4 images."""
    cat = "candle"
    img_dir = tmp_path / cat / "Data" / "Images"
    mask_dir = tmp_path / cat / "Data" / "Masks"
    (img_dir / "Normal").mkdir(parents=True)
    (img_dir / "Anomaly").mkdir(parents=True)
    (mask_dir / "Anomaly").mkdir(parents=True)

    # Create dummy image files
    for name in ("n001.png", "n002.png"):
        (img_dir / "Normal" / name).write_bytes(b"img")
    for name in ("a001.png",):
        (img_dir / "Anomaly" / name).write_bytes(b"img")
        (mask_dir / "Anomaly" / name).write_bytes(b"mask")

    # Create split CSV
    csv_dir = tmp_path / "split_csv"
    csv_dir.mkdir()
    rows = [
        {"object": cat, "split": "train", "label": "normal",
         "image_path": f"{cat}/Data/Images/Normal/n001.png", "mask_path": ""},
        {"object": cat, "split": "test", "label": "normal",
         "image_path": f"{cat}/Data/Images/Normal/n002.png", "mask_path": ""},
        {"object": cat, "split": "test", "label": "anomaly",
         "image_path": f"{cat}/Data/Images/Anomaly/a001.png",
         "mask_path": f"{cat}/Data/Masks/Anomaly/a001.png"},
    ]
    with open(csv_dir / f"{cat}.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["object","split","label","image_path","mask_path"])
        writer.writeheader()
        writer.writerows(rows)

    return tmp_path


def test_train_good_symlinks_created(fake_visa):
    reorganize_category(fake_visa, "candle")
    target = fake_visa / "candle" / "train" / "good" / "n001.png"
    assert target.is_symlink() or target.exists()


def test_test_good_symlinks_created(fake_visa):
    reorganize_category(fake_visa, "candle")
    target = fake_visa / "candle" / "test" / "good" / "n002.png"
    assert target.is_symlink() or target.exists()


def test_test_anomaly_symlinks_created(fake_visa):
    reorganize_category(fake_visa, "candle")
    target = fake_visa / "candle" / "test" / "anomaly" / "a001.png"
    assert target.is_symlink() or target.exists()


def test_ground_truth_symlinks_created(fake_visa):
    reorganize_category(fake_visa, "candle")
    target = fake_visa / "candle" / "ground_truth" / "anomaly" / "a001.png"
    assert target.is_symlink() or target.exists()


def test_idempotent_second_run_does_not_raise(fake_visa):
    reorganize_category(fake_visa, "candle")
    reorganize_category(fake_visa, "candle")  # second run — must not raise


def test_reorganize_all_processes_all_csv_categories(fake_visa):
    reorganize_all(fake_visa)
    assert (fake_visa / "candle" / "train" / "good" / "n001.png").exists()


def test_jpg_images_converted_to_png_when_flag_set(fake_visa):
    """If source images are JPG, convert_png=True creates PNG files in destination."""
    # Rename the normal image to .jpg in the fixture
    cat_dir = fake_visa / "candle" / "Data" / "Images" / "Normal"
    old = cat_dir / "n001.png"
    new_jpg = cat_dir / "n001.jpg"
    # Write a minimal valid JPEG (1x1 white pixel)
    import cv2
    import numpy as np
    img = np.ones((4, 4, 3), dtype=np.uint8) * 255
    cv2.imwrite(str(new_jpg), img)
    old.unlink()

    # Update split CSV to reference the jpg
    csv_path = fake_visa / "split_csv" / "candle.csv"
    text = csv_path.read_text()
    csv_path.write_text(text.replace("n001.png", "n001.jpg"))

    reorganize_category(fake_visa, "candle", convert_png=True)

    # The destination must be a PNG file (not a symlink to jpg)
    converted = fake_visa / "candle" / "train" / "good" / "n001.png"
    assert converted.exists(), f"Expected PNG at {converted}"
    assert not converted.name.endswith(".jpg")
