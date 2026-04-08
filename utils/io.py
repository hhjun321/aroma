from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def save_json(data, path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_json(path) -> dict:
    with open(path) as f:
        return json.load(f)


def validate_dir(path, *, name: str = "Directory") -> None:
    """Verify *path* is an existing directory.

    Raises:
        FileNotFoundError: if path does not exist or is not a directory.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"{name} not found: {path}")
    if not p.is_dir():
        raise FileNotFoundError(f"{name} is not a directory: {path}")


def validate_file(path, *, name: str = "File") -> None:
    """Verify *path* is an existing file.

    Raises:
        FileNotFoundError: if path does not exist or is not a file.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"{name} not found: {path}")
    if not p.is_file():
        raise FileNotFoundError(f"{name} is not a file: {path}")


def safe_imread(path: str | Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
    """Load image with validation and error handling.
    
    Args:
        path: Path to image file
        flags: OpenCV imread flags (default: cv2.IMREAD_COLOR)
    
    Returns:
        Loaded image as numpy array
    
    Raises:
        FileNotFoundError: If image file does not exist
        ValueError: If image cannot be decoded by OpenCV
    
    Example:
        >>> img = safe_imread("image.png")
        >>> gray = safe_imread("image.png", cv2.IMREAD_GRAYSCALE)
    """
    path = Path(path)
    
    # Check file exists before imread
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {path}")
    
    # Attempt to load image
    try:
        img = cv2.imread(str(path), flags)
    except Exception as e:
        raise ValueError(f"OpenCV error reading {path}: {e}") from e
    
    # Check if imread succeeded
    if img is None:
        raise ValueError(f"Cannot decode image (corrupted or unsupported format): {path}")
    
    return img


def validate_config_entry(entry: dict[str, Any], key: str, *, required_fields: list[str] | None = None) -> None:
    """Validate dataset config entry schema.
    
    Args:
        entry: Config dictionary entry
        key: Entry key (for error messages)
        required_fields: List of required field names (default: ["domain", "image_dir", "seed_dir"])
    
    Raises:
        KeyError: If required field is missing
        TypeError: If entry is not a dictionary
    
    Example:
        >>> entry = {"domain": "mvtec", "image_dir": "/path", "seed_dir": "/path2"}
        >>> validate_config_entry(entry, "mvtec_bottle")
    """
    if not isinstance(entry, dict):
        raise TypeError(f"Config entry '{key}' must be a dictionary, got {type(entry).__name__}")
    
    if required_fields is None:
        required_fields = ["domain", "image_dir", "seed_dir"]
    
    for field in required_fields:
        if field not in entry:
            raise KeyError(
                f"Config entry '{key}' missing required field: '{field}'\n"
                f"Available fields: {list(entry.keys())}"
            )
    
    # Validate domain value
    if "domain" in required_fields:
        valid_domains = {"isp", "mvtec", "visa"}
        domain = entry.get("domain", "")
        if domain not in valid_domains:
            logger.warning(
                f"Config entry '{key}' has unknown domain '{domain}'. "
                f"Valid domains: {valid_domains}"
            )


def validate_image(img: np.ndarray, *, name: str = "image") -> None:
    """Validate image array properties.
    
    Args:
        img: Image array to validate
        name: Name for error messages
    
    Raises:
        TypeError: If img is not a numpy array
        ValueError: If image has invalid shape or is empty
    
    Example:
        >>> img = cv2.imread("test.png")
        >>> validate_image(img, name="test.png")
    """
    if not isinstance(img, np.ndarray):
        raise TypeError(f"{name} must be numpy.ndarray, got {type(img).__name__}")
    
    if img.size == 0:
        raise ValueError(f"{name} is empty (size=0)")
    
    if len(img.shape) not in (2, 3):
        raise ValueError(
            f"{name} has invalid shape: {img.shape}. "
            f"Expected 2D (grayscale) or 3D (color) array."
        )
    
    # Validate 3-channel color image
    if len(img.shape) == 3 and img.shape[2] not in (1, 3, 4):
        raise ValueError(
            f"{name} has invalid number of channels: {img.shape[2]}. "
            f"Expected 1 (grayscale), 3 (BGR), or 4 (BGRA)."
        )
