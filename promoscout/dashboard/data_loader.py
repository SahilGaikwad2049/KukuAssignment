"""Cached, schema-aware loaders for the local PromoScout dashboard artifacts."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import streamlit as st
from pydantic import ValidationError

from promoscout.schemas import ParsedEpisode

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PATHS = {
    "episodes": PROJECT_ROOT / "data/processed/parsed_all_episodes.json",
    "analysis": PROJECT_ROOT / "data/processed/episode_analysis.json",
    "discoveries": PROJECT_ROOT / "data/processed/event_discoveries.json",
    "semantic_candidates": PROJECT_ROOT / "data/outputs/semantic_candidates.json",
    "heuristic_candidates": PROJECT_ROOT / "data/outputs/continuous_candidates.json",
    "validation": PROJECT_ROOT / "data/outputs/pilot_validation.json",
    "validation_markdown": PROJECT_ROOT / "data/outputs/pilot_validation.md",
    "review_decisions": PROJECT_ROOT / "data/outputs/review_decisions.json",
    "blind_viewer": PROJECT_ROOT / "data/outputs/blind_viewer_evaluations.json",
    "blind_viewer_pilot_report": PROJECT_ROOT / "data/outputs/blind_viewer_pilot_report.json",
    "blind_viewer_pilot_report_markdown": PROJECT_ROOT / "data/outputs/blind_viewer_pilot_report.md",
    "blind_viewer_pilot_selection": PROJECT_ROOT / "data/outputs/blind_viewer_pilot_selection.json",
    "discovery_comparison": PROJECT_ROOT / "data/outputs/discovery_comparison.json",
    "discovery_comparison_markdown": PROJECT_ROOT / "data/outputs/discovery_comparison.md",
    "series_analysis": PROJECT_ROOT / "data/processed/series_analysis.json",
    "series_analysis_markdown": PROJECT_ROOT / "data/processed/series_analysis.md",
    "composite_candidates": PROJECT_ROOT / "data/outputs/composite_candidates.json",
    "final_rankings": PROJECT_ROOT / "data/outputs/final_rankings.json",
    "final_rankings_csv": PROJECT_ROOT / "data/outputs/final_rankings.csv",
    "final_ranking_report": PROJECT_ROOT / "data/outputs/final_ranking_report.md",
}
DEMO_DIR = PROJECT_ROOT / "data/demo"
DEMO_PATHS = {
    "episodes": DEMO_DIR / "parsed_all_episodes.json",
    "analysis": DEMO_DIR / "episode_analysis.json",
    "discoveries": DEMO_DIR / "event_discoveries.json",
    "semantic_candidates": DEMO_DIR / "semantic_candidates.json",
    "heuristic_candidates": DEMO_DIR / "continuous_candidates.json",
    "validation": DEMO_DIR / "pilot_validation.json",
    "validation_markdown": DEMO_DIR / "pilot_validation.md",
    "review_decisions": DEMO_DIR / "review_decisions.json",
    "blind_viewer": DEMO_DIR / "blind_viewer_evaluations.json",
    "blind_viewer_pilot_report": DEMO_DIR / "blind_viewer_pilot_report.json",
    "blind_viewer_pilot_report_markdown": DEMO_DIR / "blind_viewer_pilot_report.md",
    "blind_viewer_pilot_selection": DEMO_DIR / "blind_viewer_pilot_selection.json",
    "discovery_comparison": DEMO_DIR / "discovery_comparison.json",
    "discovery_comparison_markdown": DEMO_DIR / "discovery_comparison.md",
    "series_analysis": DEMO_DIR / "series_analysis.json",
    "series_analysis_markdown": DEMO_DIR / "series_analysis.md",
    "composite_candidates": DEMO_DIR / "composite_candidates.json",
    "final_rankings": DEMO_DIR / "final_rankings.json",
    "final_rankings_csv": DEMO_DIR / "final_rankings.csv",
    "final_ranking_report": DEMO_DIR / "final_ranking_report.md",
    "demo_candidates": DEMO_DIR / "demo_candidates.json",
}


@st.cache_data(show_spinner=False, ttl=8)
def _read_json_cached(path: str, modified_ns: int, size: int) -> Any:
    del modified_ns, size
    return json.loads(Path(path).read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False, ttl=8)
def _read_text_cached(path: str, modified_ns: int, size: int) -> str:
    del modified_ns, size
    return Path(path).read_text(encoding="utf-8")


def read_json_file(path: str | Path) -> dict:
    """Return a non-throwing file result; malformed/missing files become empty states."""
    file_path = Path(path)
    if not file_path.exists():
        return {"data": None, "exists": False, "error": None, "path": str(file_path)}
    if file_path.stat().st_size == 0:
        return {"data": None, "exists": True, "error": None, "path": str(file_path)}
    try:
        stat = file_path.stat()
        data = _read_json_cached(str(file_path.resolve()), stat.st_mtime_ns, stat.st_size)
        return {"data": data, "exists": True, "error": None, "path": str(file_path)}
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        return {"data": None, "exists": True, "error": str(exc), "path": str(file_path)}


def read_text_file(path: str | Path) -> dict:
    file_path = Path(path)
    if not file_path.exists():
        return {"data": None, "exists": False, "error": None, "path": str(file_path)}
    try:
        stat = file_path.stat()
        data = _read_text_cached(str(file_path.resolve()), stat.st_mtime_ns, stat.st_size)
        return {"data": data, "exists": True, "error": None, "path": str(file_path)}
    except (OSError, UnicodeDecodeError) as exc:
        return {"data": None, "exists": True, "error": str(exc), "path": str(file_path)}


def normalize_candidates(payload: Any, source: str) -> tuple[list[dict], list[str]]:
    """Normalize supported candidate output envelopes without losing source fields."""
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict):
        rows = payload.get("candidates", [])
        if not isinstance(rows, list):
            return [], ["Candidate artifact has a non-list 'candidates' field."]
    else:
        return [], ["Candidate artifact must be a list or an object containing 'candidates'."]
    normalized, errors = [], []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            errors.append(f"Candidate row {index + 1} is not an object.")
            continue
        if not row.get("candidate_id") or not isinstance(row.get("episode_number"), int):
            errors.append(f"Candidate row {index + 1} is missing candidate_id or episode_number.")
            continue
        item = dict(row)
        item["source"] = source
        normalized.append(item)
    return normalized, errors


def load_dashboard_data(paths: dict[str, str | Path] | None = None) -> dict:
    """Load local project data, falling back to labelled synthetic demo data."""
    resolved = {key: Path(value) for key, value in DEFAULT_PATHS.items()}
    if paths:
        resolved.update({key: Path(value) for key, value in paths.items()})

    def read_files(file_paths):
        return {
        key: (read_text_file(path) if key in {"validation_markdown", "blind_viewer_pilot_report_markdown", "discovery_comparison_markdown", "series_analysis_markdown", "final_rankings_csv", "final_ranking_report"} else read_json_file(path))
            for key, path in file_paths.items()
        }

    files = read_files(resolved)
    episodes, episode_errors = [], []
    raw_episodes = files["episodes"]["data"]
    if raw_episodes is not None:
        if not isinstance(raw_episodes, list):
            episode_errors.append("Parsed screenplay JSON must contain a top-level list.")
        else:
            for index, row in enumerate(raw_episodes):
                try:
                    episodes.append(ParsedEpisode.model_validate(row))
                except (ValidationError, TypeError) as exc:
                    episode_errors.append(f"Episode row {index + 1} is invalid: {exc}")

    demo_mode = raw_episodes is None or raw_episodes == []
    if demo_mode:
        # Keep real mode all-or-nothing: never mix a synthetic dataset with
        # partial local output artifacts when the source screenplay is absent.
        resolved = {key: Path(value) for key, value in DEMO_PATHS.items()}
        files = read_files(resolved)
        episodes, episode_errors = [], []
        raw_episodes = files["episodes"]["data"]
        if isinstance(raw_episodes, list):
            for index, row in enumerate(raw_episodes):
                try:
                    episodes.append(ParsedEpisode.model_validate(row))
                except (ValidationError, TypeError) as exc:
                    episode_errors.append(f"Demo episode row {index + 1} is invalid: {exc}")

    analysis_payload = files["analysis"]["data"]
    analysis = analysis_payload if isinstance(analysis_payload, dict) else {}
    discoveries_payload = files["discoveries"]["data"]
    discoveries = discoveries_payload if isinstance(discoveries_payload, dict) else {}
    semantic, semantic_errors = normalize_candidates(files["semantic_candidates"]["data"], "semantic") if files["semantic_candidates"]["data"] is not None else ([], [])
    heuristic, heuristic_errors = normalize_candidates(files["heuristic_candidates"]["data"], "heuristic") if files["heuristic_candidates"]["data"] is not None else ([], [])
    composite_payload = files["composite_candidates"]["data"]
    composite = composite_payload.get("candidates", []) if isinstance(composite_payload, dict) else composite_payload
    if not isinstance(composite, list):
        composite = []
    composite = [dict(row) for row in composite if isinstance(row, dict)]
    demo_payload = files.get("demo_candidates", {}).get("data")
    demo_candidates = demo_payload.get("candidates", []) if isinstance(demo_payload, dict) else []
    if not isinstance(demo_candidates, list):
        demo_candidates = []
    demo_candidates = [dict(row, source="demo", synthetic=True) for row in demo_candidates if isinstance(row, dict)]
    blind_payload = files["blind_viewer"]["data"]
    blind_viewer = blind_payload.get("evaluations", {}) if isinstance(blind_payload, dict) else {}
    if not isinstance(blind_viewer, dict):
        blind_viewer = {}
    return {
        "paths": resolved,
        "files": files,
        "data_mode": "demo" if demo_mode else "local",
        "demo_mode": demo_mode,
        "demo_candidates": demo_candidates,
        "episodes": episodes,
        "episode_errors": episode_errors,
        "analysis": analysis,
        "discoveries": discoveries,
        "semantic_candidates": semantic,
        "heuristic_candidates": heuristic,
        "composite_candidates": composite,
        "final_rankings": files["final_rankings"]["data"] if isinstance(files["final_rankings"]["data"], dict) else None,
        "final_rankings_csv": files["final_rankings_csv"]["data"] if isinstance(files["final_rankings_csv"]["data"], str) else None,
        "final_ranking_report": files["final_ranking_report"]["data"] if isinstance(files["final_ranking_report"]["data"], str) else None,
        "series_analysis": files["series_analysis"]["data"] if isinstance(files["series_analysis"]["data"], dict) else None,
        "series_analysis_markdown": files["series_analysis_markdown"]["data"] if isinstance(files["series_analysis_markdown"]["data"], str) else None,
        "candidate_errors": semantic_errors + heuristic_errors,
        "validation": files["validation"]["data"] if isinstance(files["validation"]["data"], dict) else None,
        "validation_markdown": files["validation_markdown"]["data"] if isinstance(files["validation_markdown"]["data"], str) else None,
        "blind_viewer_evaluations": blind_viewer,
        "blind_viewer_pilot_report": files["blind_viewer_pilot_report"]["data"] if isinstance(files["blind_viewer_pilot_report"]["data"], dict) else None,
        "blind_viewer_pilot_report_markdown": files["blind_viewer_pilot_report_markdown"]["data"] if isinstance(files["blind_viewer_pilot_report_markdown"]["data"], str) else None,
        "blind_viewer_pilot_selection": files["blind_viewer_pilot_selection"]["data"] if isinstance(files["blind_viewer_pilot_selection"]["data"], dict) else None,
        "discovery_comparison": files["discovery_comparison"]["data"] if isinstance(files["discovery_comparison"]["data"], dict) else None,
        "discovery_comparison_markdown": files["discovery_comparison_markdown"]["data"] if isinstance(files["discovery_comparison_markdown"]["data"], str) else None,
    }

