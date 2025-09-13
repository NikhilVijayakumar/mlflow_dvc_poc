#src.utils.io.py
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Union


def save_json(data: Dict[str, Any], file_path: Union[str, Path]):
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
    print(f"   - ✅ Successfully saved JSON to: {path}")


def load_json(file_path: Union[str, Path]) -> Dict[str, Any]:
    path = Path(file_path)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def file_sha256(file_path: Union[str, Path]) -> str:
    path = Path(file_path)
    h = hashlib.sha256()

    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)

    return h.hexdigest()

