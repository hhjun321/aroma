import json
from pathlib import Path


def save_json(data, path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_json(path) -> dict:
    with open(path) as f:
        return json.load(f)


def validate_dir(path) -> None:
    if not Path(path).exists():
        raise FileNotFoundError(f"Directory not found: {path}")
