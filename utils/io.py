import json
from pathlib import Path


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
