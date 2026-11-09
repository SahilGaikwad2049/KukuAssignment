"""Pure local serialization helpers used by dashboard downloads."""
import json

import pandas as pd


def json_export_bytes(value) -> bytes:
    return json.dumps(value, ensure_ascii=False, indent=2).encode("utf-8")


def candidate_csv_bytes(rows: list[dict]) -> bytes:
    if not rows:
        return b""
    frame = pd.DataFrame(rows)
    for column in frame.columns:
        if frame[column].map(lambda value: isinstance(value, (dict, list))).any():
            frame[column] = frame[column].map(
                lambda value: json.dumps(value, ensure_ascii=False)
                if isinstance(value, (dict, list)) else value
            )
    return frame.to_csv(index=False).encode("utf-8")

