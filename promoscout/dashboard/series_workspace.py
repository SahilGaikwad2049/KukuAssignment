"""Local ZIP intake and per-series semantic analysis workspace for Streamlit."""
from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from zipfile import BadZipFile, ZipFile

import streamlit as st
from pydantic import ValidationError

from promoscout.analysis.episode_analyzer import analyze_episodes
from promoscout.config import get_settings
from promoscout.generation.continuous import generate_candidates_from_discoveries
from promoscout.parser.screenplay_parser import parse_zip_archive
from promoscout.schemas import ParsedEpisode
from promoscout.utils.llm_client import GeminiClient

WORKSPACES_DIR = Path("data/workspaces")
MAX_UPLOAD_BYTES = 100 * 1024 * 1024
MAX_EXPANDED_BYTES = 400 * 1024 * 1024
MAX_TEXT_FILES = 200


def _atomic_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as handle:
            temporary = handle.name
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except OSError:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)
        raise


def _validate_zip_bytes(content: bytes) -> None:
    if not content:
        raise ValueError("The uploaded ZIP file is empty.")
    if len(content) > MAX_UPLOAD_BYTES:
        raise ValueError("ZIP upload exceeds the 100 MB limit.")
    try:
        with ZipFile(BytesIO(content)) as archive:
            scripts = [
                item for item in archive.infolist()
                if not item.is_dir()
                and item.filename.lower().endswith(".txt")
                and "__MACOSX/" not in item.filename
                and not Path(item.filename).name.startswith("._")
            ]
            if not scripts:
                raise ValueError("The ZIP must contain screenplay .txt files.")
            if len(scripts) > MAX_TEXT_FILES:
                raise ValueError(f"The ZIP contains too many .txt files (maximum {MAX_TEXT_FILES}).")
            expanded_bytes = sum(item.file_size for item in archive.infolist() if not item.is_dir())
            if expanded_bytes > MAX_EXPANDED_BYTES:
                raise ValueError("The ZIP contents exceed the 400 MB uncompressed processing limit.")
            bad_member = archive.testzip()
            if bad_member:
                raise ValueError(f"The ZIP archive contains a corrupt member: {Path(bad_member).name}")
    except BadZipFile as exc:
        raise ValueError("The uploaded file is not a valid ZIP archive.") from exc


def parse_uploaded_zip(content: bytes) -> list[ParsedEpisode]:
    """Parse a ZIP locally using the existing parser, without extracting files."""
    _validate_zip_bytes(content)
    with tempfile.TemporaryDirectory(prefix="promoscout-upload-") as temporary_dir:
        archive_path = Path(temporary_dir) / "upload.zip"
        archive_path.write_bytes(content)
        episodes = parse_zip_archive(archive_path)
    episode_numbers = [episode.episode_number for episode in episodes]
    if len(episode_numbers) != len(set(episode_numbers)):
        raise ValueError("The ZIP contains duplicate episode numbers; episode numbers must be unique.")
    if not episodes:
        raise ValueError("No screenplay episodes could be parsed from the ZIP.")
    return episodes


def _slug(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "-", value.strip()).strip("-_ ").lower()
    return cleaned[:48] or "series"


def create_series_workspace(
    series_name: str,
    upload_name: str,
    content: bytes,
    *,
    root: str | Path = WORKSPACES_DIR,
) -> tuple[Path, list[ParsedEpisode]]:
    """Parse and persist a new show under a content-addressed local workspace."""
    if not series_name.strip():
        raise ValueError("Enter a series name before parsing the ZIP.")
    episodes = parse_uploaded_zip(content)
    digest = hashlib.sha256(content).hexdigest()
    workspace = Path(root) / f"{_slug(series_name)}-{digest[:10]}"
    workspace.mkdir(parents=True, exist_ok=True)
    _atomic_json(workspace / "parsed_episodes.json", [item.model_dump(mode="json") for item in episodes])
    manifest_path = workspace / "workspace.json"
    if not manifest_path.exists():
        _atomic_json(manifest_path, {
            "series_name": series_name.strip(),
            "uploaded_filename": Path(upload_name).name,
            "content_sha256": digest,
            "episode_count": len(episodes),
            "event_count": sum(len(item.events) for item in episodes),
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        })
    return workspace, episodes


def load_workspace(workspace: str | Path) -> tuple[dict, list[ParsedEpisode]]:
    path = Path(workspace)
    manifest = json.loads((path / "workspace.json").read_text(encoding="utf-8"))
    raw = json.loads((path / "parsed_episodes.json").read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("Workspace parsed episode data must be a JSON list.")
    episodes = [ParsedEpisode.model_validate(item) for item in raw]
    return manifest, episodes


def list_workspaces(root: str | Path = WORKSPACES_DIR) -> list[tuple[str, Path, dict]]:
    found = []
    base = Path(root)
    if not base.exists():
        return found
    for path in base.iterdir():
        manifest_path = path / "workspace.json"
        if not path.is_dir() or not manifest_path.is_file():
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            found.append((str(path.resolve()), path, manifest))
        except (OSError, json.JSONDecodeError):
            continue
    return sorted(found, key=lambda row: row[2].get("created_at", ""), reverse=True)


def process_series_workspace(
    workspace: str | Path,
    episodes: list[ParsedEpisode],
    *,
    model_id: str,
    llm=None,
) -> dict:
    """Run semantic analysis and deterministic candidate generation for this show."""
    path = Path(workspace)
    analysis_path = path / "episode_analysis.json"
    discoveries_path = path / "event_discoveries.json"
    candidate_path = path / "semantic_candidates.json"
    client = llm if llm is not None else GeminiClient(model=model_id)
    analyses, discoveries = analyze_episodes(
        episodes,
        llm=client,
        analysis_path=analysis_path,
        discoveries_path=discoveries_path,
    )
    candidates = generate_candidates_from_discoveries(episodes, discoveries)
    _atomic_json(candidate_path, {
        "method": "semantic_event_ids",
        "series_name": json.loads((path / "workspace.json").read_text(encoding="utf-8"))["series_name"],
        "candidates": [candidate.model_dump(mode="json") for candidate in candidates],
    })
    return {
        "analyses": analyses,
        "discoveries": discoveries,
        "candidates": candidates,
        "paths": {
            "analysis": analysis_path,
            "discoveries": discoveries_path,
            "candidates": candidate_path,
        },
    }


def _show_workspace_outputs(workspace: Path, episodes: list[ParsedEpisode]) -> None:
    manifest = json.loads((workspace / "workspace.json").read_text(encoding="utf-8"))
    analysis_path, discoveries_path, candidates_path = (
        workspace / "episode_analysis.json",
        workspace / "event_discoveries.json",
        workspace / "semantic_candidates.json",
    )
    analyses = json.loads(analysis_path.read_text(encoding="utf-8")) if analysis_path.exists() else {}
    discoveries = json.loads(discoveries_path.read_text(encoding="utf-8")) if discoveries_path.exists() else {}
    candidate_payload = json.loads(candidates_path.read_text(encoding="utf-8")) if candidates_path.exists() else {"candidates": []}
    candidates = candidate_payload.get("candidates", []) if isinstance(candidate_payload, dict) else []
    event_count = sum(len(episode.events) for episode in episodes)
    cols = st.columns(3)
    cols[0].metric("Episodes", len(episodes))
    cols[1].metric("Screenplay events", event_count)
    cols[2].metric("Semantic promo windows", len(candidates))

    episode_numbers = [episode.episode_number for episode in episodes]
    selected_number = st.selectbox("Explore episode output", episode_numbers, key=f"workspace-episode-{workspace.name}")
    episode = next(item for item in episodes if item.episode_number == selected_number)
    analysis = analyses.get(str(selected_number), {})
    if isinstance(analysis, dict) and analysis:
        st.subheader(f"Episode {selected_number} narrative analysis")
        st.write(analysis.get("episode_summary", ""))
        st.markdown(f"**Main conflict:** {analysis.get('main_conflict', 'Not available')}")
        with st.expander("Characters and story developments"):
            st.write("**Important characters**")
            st.write(analysis.get("important_characters", []))
            st.write("**Relationships**")
            st.write(analysis.get("character_relationships", []))
            st.write("**Developments**")
            st.write(analysis.get("important_story_developments", []))
            st.write("**Revelations and turning points**")
            st.write(analysis.get("major_revelations", []) + analysis.get("emotional_turning_points", []))
            st.write("**Unresolved questions**")
            st.write(analysis.get("unresolved_questions", []))
    else:
        st.info("Semantic episode analysis has not been run for this series yet.")

    discovered = discoveries.get(str(selected_number), []) if isinstance(discoveries, dict) else []
    event_by_id = {event.event_id: event for event in episode.events}
    if discovered:
        st.subheader("Narrative discoveries")
        for item in discovered:
            event = event_by_id.get(item.get("event_id"))
            title = f"{item.get('role', 'discovery').replace('_', ' ').title()} · {item.get('category', 'uncategorized')}"
            with st.expander(title):
                st.write(event.content if event else "Source event is unavailable.")
                st.caption(f"Source event: {item.get('event_id', 'unknown')} · confidence {item.get('confidence', '—')}")
                st.write(item.get("reason", ""))
                st.caption(item.get("narrative_context", ""))
    elif analysis:
        st.info("No potential promotional events were returned for this episode.")

    episode_candidates = [item for item in candidates if item.get("episode_number") == selected_number]
    st.subheader(f"Promo candidates · {len(episode_candidates)}")
    if not episode_candidates:
        st.caption("Candidate windows are generated from discovered event IDs while Python preserves source timestamps and duration rules.")
    for item in episode_candidates:
        with st.container(border=True):
            st.markdown(f"**{item.get('candidate_id')}** · {item.get('duration_seconds')} sec · family {item.get('candidate_family_id')}")
            st.caption(f"{item.get('start_timestamp')} → {item.get('end_timestamp')} · {item.get('hook_type', 'unknown')} hook")
            st.markdown(f"**Opening:** {item.get('hook_text', '')}")
            st.markdown(f"**Potential ending:** {item.get('ending_text', '')}")
    downloads = st.columns(3)
    for column, path, label, filename in [
        (downloads[0], analysis_path, "Episode analysis", "episode_analysis.json"),
        (downloads[1], discoveries_path, "Event discoveries", "event_discoveries.json"),
        (downloads[2], candidates_path, "Semantic candidates", "semantic_candidates.json"),
    ]:
        if path.exists():
            column.download_button(label, path.read_bytes(), file_name=filename, mime="application/json", key=f"download-{workspace.name}-{filename}", use_container_width=True)
    st.caption(f"This workspace is stored locally at `{workspace}`. Uploaded ZIP bytes are not retained after parsing.")


def series_workspace_page() -> None:
    st.title("Series Workspace")
    st.caption("Upload a screenplay ZIP, parse it locally, then choose whether to run semantic discovery for that series.")
    st.info("Uploading and parsing stays local. Gemini receives screenplay text only after you explicitly confirm and start semantic analysis.")

    with st.form("series-upload-form", clear_on_submit=False):
        series_name = st.text_input("Series name", placeholder="Enter the show name")
        upload = st.file_uploader("Screenplay ZIP (.txt scripts)", type=["zip"], accept_multiple_files=False)
        parse_clicked = st.form_submit_button("Upload and parse locally", use_container_width=True)
    if parse_clicked:
        if upload is None:
            st.error("Choose a ZIP file first.")
        else:
            try:
                workspace, episodes = create_series_workspace(series_name, upload.name, upload.getvalue())
                st.session_state["active-series-workspace"] = str(workspace.resolve())
                st.success(f"Parsed {len(episodes)} episode(s) and {sum(len(ep.events) for ep in episodes):,} screenplay event(s). No LLM request was made.")
            except (OSError, ValueError, BadZipFile, ValidationError) as exc:
                st.error(str(exc))

    workspaces = list_workspaces()
    if not workspaces:
        st.markdown("### Start a series workspace")
        st.caption("Upload a ZIP containing episode `.txt` screenplay files. The parser reads the existing `EPISODE NUMBER`, logline, continuity, scene, event, and timestamp structure.")
        return

    workspace_choices = {path_string: (path, manifest) for path_string, path, manifest in workspaces}
    preferred = st.session_state.get("active-series-workspace")
    options = list(workspace_choices)
    default_index = options.index(preferred) if preferred in workspace_choices else 0
    selected = st.selectbox(
        "Series workspace",
        options,
        index=default_index,
        format_func=lambda key: f"{workspace_choices[key][1].get('series_name', 'Series')} · {workspace_choices[key][1].get('episode_count', '?')} episodes",
    )
    workspace, manifest = workspace_choices[selected]
    st.session_state["active-series-workspace"] = selected
    try:
        loaded_manifest, episodes = load_workspace(workspace)
    except (OSError, json.JSONDecodeError, ValidationError, ValueError) as exc:
        st.error(f"Could not load this series workspace: {exc}")
        return

    st.markdown(f"### {loaded_manifest.get('series_name', manifest.get('series_name', 'Series'))}")
    st.caption(f"Source archive: {loaded_manifest.get('uploaded_filename', 'ZIP')} · Workspace ID: {workspace.name}")
    settings = get_settings()
    model_id = st.text_input(
        "Gemini model ID",
        value=settings.gemini_model,
        help="Uses the configured Gemini API key. Enter another supported Gemini model ID to test that model; this does not switch providers.",
        key=f"series-model-{workspace.name}",
    ).strip()
    api_key_configured = bool(settings.gemini_api_key) and settings.gemini_api_key.lower() not in {
        "your_api_key_here", "your_gemini_api_key", "changeme", "replace_me",
    }
    st.caption(f"API key configured: {'yes' if api_key_configured else 'no'} · Model: {model_id or '(not set)'} · Processing is sequential and cached by content and model.")
    consent = st.checkbox(
        "I explicitly authorize sending this series' screenplay text to the configured Gemini API for analysis.",
        key=f"series-consent-{workspace.name}",
    )
    can_run = consent and api_key_configured and bool(model_id) and settings.llm_provider == "gemini"
    if settings.llm_provider != "gemini":
        st.error("This workspace currently supports the Gemini provider only.")
    if not api_key_configured:
        st.warning("Configure GEMINI_API_KEY in .env/the local environment or in the deployed Streamlit app's Secrets settings before starting analysis.")
    if not model_id:
        st.warning("Enter a Gemini model ID before starting analysis.")
    if st.button("Analyze series and discover promo candidates", type="primary", disabled=not can_run,
                 key=f"series-run-{workspace.name}", use_container_width=True):
        try:
            with st.spinner(f"Analyzing {len(episodes)} episode(s) with {model_id}. Cached episodes will be reused."):
                result = process_series_workspace(workspace, episodes, model_id=model_id)
            st.success(
                f"Analysis saved for {len(result['analyses'])} episode(s); "
                f"{sum(len(items) for items in result['discoveries'].values())} discoveries and "
                f"{len(result['candidates'])} semantic promo candidate(s) generated."
            )
        except Exception as exc:
            safe_error = str(exc).replace(settings.gemini_api_key, "[redacted]") if settings.gemini_api_key else str(exc)
            st.error(f"Series processing stopped. Completed episode outputs remain saved for resume. {safe_error}")

    st.divider()
    _show_workspace_outputs(workspace, episodes)
