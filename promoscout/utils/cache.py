"""Content-addressed, atomic disk cache for semantic analysis responses."""
import hashlib
import json
import os
from pathlib import Path
from typing import Any


def content_hash(value: Any) -> str:
    data = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


class JsonCache:
    def __init__(self, directory: str | Path):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def path_for(self, key: str) -> Path:
        return self.directory / f"{key}.json"

    def get(self, key: str):
        path = self.path_for(key)
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def put(self, key: str, value: Any) -> None:
        path = self.path_for(key)
        temp = path.with_suffix(f".{os.getpid()}.tmp")
        temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
        temp.replace(path)

