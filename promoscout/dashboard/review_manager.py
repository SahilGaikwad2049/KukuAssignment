"""Atomic local persistence for human candidate review decisions."""
from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from promoscout.dashboard.data_loader import DEFAULT_PATHS

SCHEMA_VERSION = "promo-review-decisions-v1"


class ReviewStoreError(RuntimeError):
    pass


def review_key(source: str, candidate_id: str) -> str:
    return f"{source}:{candidate_id}"


def load_review_decisions(path: str | Path | None = None) -> dict:
    file_path = Path(path or DEFAULT_PATHS["review_decisions"])
    if not file_path.exists():
        return {"schema_version": SCHEMA_VERSION, "decisions": {}}
    try:
        payload = json.loads(file_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReviewStoreError(f"Could not read review decisions at {file_path}: {exc}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("decisions"), dict):
        raise ReviewStoreError(f"Review decisions file has an invalid structure: {file_path}")
    return {"schema_version": payload.get("schema_version", SCHEMA_VERSION), "decisions": payload["decisions"]}


def save_review_decision(source: str, candidate_id: str, decision: str, note: str = "",
                         path: str | Path | None = None) -> dict:
    if decision not in {"approved", "rejected", "later"}:
        raise ValueError("decision must be approved, rejected, or later")
    file_path = Path(path or DEFAULT_PATHS["review_decisions"])
    current = load_review_decisions(file_path)
    current["decisions"][review_key(source, candidate_id)] = {
        "source": source,
        "candidate_id": candidate_id,
        "decision": decision,
        "note": note.strip(),
        "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    file_path.parent.mkdir(parents=True, exist_ok=True)
    temp_name = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=file_path.parent,
                                         prefix=f".{file_path.name}.", suffix=".tmp", delete=False) as temp:
            temp_name = temp.name
            json.dump(current, temp, ensure_ascii=False, indent=2)
            temp.flush()
            os.fsync(temp.fileno())
        os.replace(temp_name, file_path)
    except OSError as exc:
        if temp_name and os.path.exists(temp_name):
            os.unlink(temp_name)
        raise ReviewStoreError(f"Could not save review decision to {file_path}: {exc}") from exc
    return current

