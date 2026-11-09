"""Hierarchical series narrative analysis with source event validation and resume cache."""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from promoscout.analysis.episode_analyzer import (
    _build_chunks, _cache_key as episode_cache_key, _scene_payload,
    analyze_episodes, load_episodes, select_episodes,
)
from promoscout.analysis.models import AnalysisEnvelope
from promoscout.analysis.series_models import SeriesAnalysis, SourceReference
from promoscout.config import get_settings
from promoscout.schemas import ParsedEpisode
from promoscout.utils.cache import JsonCache, content_hash
from promoscout.utils.llm_client import GeminiClient

PROMPT_PATH = Path(__file__).resolve().parents[2] / "prompts" / "series_analysis.txt"
PROMPT_VERSION = "series-analysis-v1"
SCHEMA_VERSION = "series-analysis-schema-v1"
DEFAULT_PARSED = Path("data/processed/parsed_all_episodes.json")
DEFAULT_ANALYSIS = Path("data/processed/episode_analysis.json")
DEFAULT_DISCOVERIES = Path("data/processed/event_discoveries.json")
DEFAULT_OUTPUT = Path("data/processed/series_analysis.json")
DEFAULT_MARKDOWN = Path("data/processed/series_analysis.md")


class InvalidSeriesEventReference(ValueError):
    """Series-level analysis refers to an event absent from its source episode."""


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                         prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
            temporary = handle.name
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except OSError:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)
        raise


def _atomic_json(path: Path, value: dict) -> None:
    _atomic_text(path, json.dumps(value, indent=2, ensure_ascii=False))


def _read_json(path: str | Path, default: Any):
    file = Path(path)
    if not file.exists() or not file.stat().st_size:
        return default
    try:
        return json.loads(file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {file}: {exc}") from exc


def validate_series_references(result: SeriesAnalysis, episodes: list[ParsedEpisode],
                               allowed_event_ids: dict[int, set[str]] | None = None) -> SeriesAnalysis:
    valid = {episode.episode_number: {event.event_id for event in episode.events} for episode in episodes}
    refs: list[SourceReference] = []
    for character in result.character_registry:
        refs.extend(character.source_references)
        for relation in character.relationships:
            refs.extend(relation.source_references)
            for change in relation.changes:
                refs.extend(change.source_references)
    for arc in result.story_arcs:
        for development in arc.developments + arc.spoiler_sensitive_developments:
            refs.extend(development.source_references)
        for uncertainty in arc.continuity_uncertainties:
            refs.extend(uncertainty.source_references)
    for development in result.episode_developments + result.chronological_narrative_structure + result.spoiler_sensitive_developments:
        refs.extend(development.source_references)
    for uncertainty in result.continuity_uncertainties:
        refs.extend(uncertainty.source_references)
    for opportunity in result.composite_promo_opportunities:
        if opportunity.hook_beat_index >= len(opportunity.beats) or opportunity.ending_beat_index >= len(opportunity.beats):
            raise ValueError(f"Composite opportunity {opportunity.story_arc_id} has an out-of-range hook/ending beat index")
        for beat in opportunity.beats:
            refs.extend(SourceReference(episode_number=beat.episode_number, event_id=event_id)
                        for event_id in beat.anchor_event_ids)
            if beat.episode_number not in valid:
                raise InvalidSeriesEventReference(f"Unknown episode {beat.episode_number} in composite opportunity {opportunity.story_arc_id}")
    invalid = sorted({(ref.episode_number, ref.event_id) for ref in refs
                      if ref.episode_number not in valid or ref.event_id not in valid.get(ref.episode_number, set())})
    if invalid:
        formatted = ", ".join(f"episode {episode}: {event_id}" for episode, event_id in invalid)
        raise InvalidSeriesEventReference(f"Series analysis contains invalid source event reference(s): {formatted}")
    if allowed_event_ids is not None:
        out_of_chunk = sorted({(ref.episode_number, ref.event_id) for ref in refs
                               if ref.event_id not in allowed_event_ids.get(ref.episode_number, set())})
        if out_of_chunk:
            formatted = ", ".join(f"episode {episode}: {event_id}" for episode, event_id in out_of_chunk)
            raise InvalidSeriesEventReference(f"Partial series analysis cited events outside its supplied chunk: {formatted}")
    selected = set(result.analyzed_episode_numbers)
    unknown_episodes = sorted(selected - set(valid))
    if unknown_episodes:
        raise InvalidSeriesEventReference(f"Series analysis contains unavailable episode number(s): {unknown_episodes}")
    return result


def _episode_payload(episode: ParsedEpisode, analysis: dict, discoveries: list[dict]) -> dict:
    # Include source event text so references can be grounded; no timestamps are
    # requested or generated by the model.
    return {
        "episode_number": episode.episode_number,
        "analysis": analysis,
        "discoveries": discoveries,
        "source_events": [{
            "event_id": event.event_id,
            "event_type": event.event_type,
            "speaker": event.speaker,
            "content": event.content,
        } for event in episode.events],
    }


def _episode_chunks(episodes: list[ParsedEpisode], analysis_map: dict,
                    discovery_map: dict, limit: int) -> list[dict]:
    records = []
    for episode in episodes:
        raw_analysis = analysis_map.get(str(episode.episode_number))
        if isinstance(raw_analysis, dict) and isinstance(raw_analysis.get("episode_analysis"), dict):
            raw_analysis = raw_analysis["episode_analysis"]
        if not isinstance(raw_analysis, dict):
            raise ValueError(f"Missing episode analysis for episode {episode.episode_number}; analyze the selected episodes first.")
        try:
            analysis = AnalysisEnvelope.model_validate({"episode_analysis": raw_analysis}).episode_analysis.model_dump(mode="json")
        except ValidationError as exc:
            raise ValueError(f"Invalid episode analysis for episode {episode.episode_number}: {exc}") from exc
        discoveries = discovery_map.get(str(episode.episode_number), [])
        records.append(_episode_payload(episode, analysis, discoveries if isinstance(discoveries, list) else []))

    chunks: list[dict] = []
    current: list[dict] = []
    for record in records:
        trial = {"episodes": current + [record]}
        if current and len(json.dumps(trial, ensure_ascii=False)) > limit:
            chunks.append({"episodes": current})
            current = []
        if len(json.dumps({"episodes": [record]}, ensure_ascii=False)) <= limit:
            current.append(record)
            continue
        if current:
            chunks.append({"episodes": current})
            current = []
        # A long episode is split by source event, with its episode analysis and
        # discoveries repeated. No screenplay event is truncated or discarded.
        fixed = {k: v for k, v in record.items() if k != "source_events"}
        event_group = []
        for event in record["source_events"]:
            candidate = {"episodes": [{**fixed, "source_events": event_group + [event]}]}
            if event_group and len(json.dumps(candidate, ensure_ascii=False)) > limit:
                chunks.append({"episodes": [{**fixed, "source_events": event_group}]})
                event_group = [event]
            else:
                event_group.append(event)
        if event_group:
            chunks.append({"episodes": [{**fixed, "source_events": event_group}]})
    if current:
        chunks.append({"episodes": current})
    return chunks


def _cache_key(episodes: list[ParsedEpisode], analysis_map: dict, discovery_map: dict, model: str) -> str:
    return content_hash({
        "episodes": [_episode_payload(ep, analysis_map[str(ep.episode_number)], discovery_map.get(str(ep.episode_number), []))
                     for ep in episodes],
        "model": model,
        "prompt_version": PROMPT_VERSION,
        "prompt_hash": content_hash(PROMPT_PATH.read_text(encoding="utf-8")),
        "schema_version": SCHEMA_VERSION,
    })


def _cache_file_exists(directory: Path, key: str) -> bool:
    """Read-only cache inspection for dry-run; never creates the directory."""
    return (directory / f"{key}.json").is_file()


def _generate(llm, prompt: str) -> SeriesAnalysis:
    method = getattr(llm, "generate_json_schema", None)
    result = method(prompt, SeriesAnalysis) if callable(method) else llm.generate(prompt, SeriesAnalysis)
    return result if isinstance(result, SeriesAnalysis) else SeriesAnalysis.model_validate(result)


def analyze_series(episodes: list[ParsedEpisode], analysis_map: dict, discovery_map: dict,
                   *, llm=None, cache: JsonCache | None = None,
                   max_chunk_chars: int | None = None) -> SeriesAnalysis:
    if not episodes:
        raise ValueError("Select at least one parsed episode for series analysis.")
    settings = get_settings()
    llm = llm or GeminiClient()
    model_id = getattr(llm, "model", settings.gemini_model)
    cache = cache or JsonCache(settings.cache_dir / "series_analysis")
    key = _cache_key(episodes, analysis_map, discovery_map, model_id)
    cached = cache.get(key)
    if cached is not None:
        return validate_series_references(SeriesAnalysis.model_validate(cached), episodes)
    chunks = _episode_chunks(episodes, analysis_map, discovery_map,
                             max_chunk_chars or settings.max_chunk_chars)
    prompt_header = PROMPT_PATH.read_text(encoding="utf-8")
    partials = []
    for index, chunk in enumerate(chunks):
        chunk_key = content_hash({"parent": key, "chunk": chunk, "index": index})
        partial = cache.get(chunk_key)
        if partial is None:
            partial = _generate(llm, prompt_header + "\n\nSERIES EPISODE MATERIAL:\n" + json.dumps(chunk, ensure_ascii=False))
            chunk_episode_numbers = {record["episode_number"] for record in chunk["episodes"]}
            if not set(partial.analyzed_episode_numbers).issubset(chunk_episode_numbers):
                raise ValueError("Partial series analysis claims episodes that were not present in its chunk")
            allowed = {}
            for record in chunk["episodes"]:
                number = record["episode_number"]
                allowed.setdefault(number, set()).update(row["event_id"] for row in record["source_events"])
                allowed[number].update(row.get("event_id") for row in record.get("discoveries", [])
                                       if isinstance(row, dict) and row.get("event_id"))
            partial = validate_series_references(partial, episodes, allowed)
            partial = partial.model_dump(mode="json")
            cache.put(chunk_key, partial)
        partials.append(SeriesAnalysis.model_validate(partial))
    aggregate_input = {
        "analyzed_episode_numbers": [episode.episode_number for episode in episodes],
        "partial_series_analyses": [partial.model_dump(mode="json") for partial in partials],
        "valid_episode_event_ids": {
            str(episode.episode_number): [event.event_id for event in episode.events]
            for episode in episodes
        },
    }
    final = _generate(llm, prompt_header + "\n\nSYNTHESIZE THE PARTIAL SERIES ANALYSES. Resolve aliases and cross-episode arcs conservatively. Preserve only supplied source IDs.\n" + json.dumps(aggregate_input, ensure_ascii=False))
    final.analyzed_episode_numbers = [episode.episode_number for episode in episodes]
    final = validate_series_references(final, episodes)
    cache.put(key, final.model_dump(mode="json"))
    return final


def render_markdown(result: SeriesAnalysis) -> str:
    lines = ["# Series Narrative Analysis", "", f"Episodes analyzed: {', '.join(map(str, result.analyzed_episode_numbers))}", "", "## Synopsis", "", result.series_synopsis, "", "## Character registry", ""]
    for character in result.character_registry:
        aliases = f" (aliases: {', '.join(character.aliases)})" if character.aliases else ""
        lines.append(f"- **{character.canonical_name}**{aliases}: {character.role}")
        for relation in character.relationships:
            lines.append(f"  - {character.canonical_name} / {relation.other_character}: {relation.relationship}")
            for change in relation.changes:
                refs = ", ".join(f"E{r.episode_number}:{r.event_id}" for r in change.source_references)
                lines.append(f"    - Episode {change.episode_number}: {change.description} [{refs}]")
    lines.extend(["", "## Story arcs", ""])
    for arc in result.story_arcs:
        lines.extend([f"### {arc.title} (`{arc.story_arc_id}`)", "", arc.overview, ""])
        for item in arc.developments:
            refs = ", ".join(f"E{r.episode_number}:{r.event_id}" for r in item.source_references)
            lines.append(f"- Episode {item.episode_number}: {item.development} ({item.development_type}; spoiler {item.spoiler_level}) — {refs}")
        if arc.unresolved_questions:
            lines.append("- Unresolved: " + "; ".join(arc.unresolved_questions))
    lines.extend(["", "## Composite promo opportunities", ""])
    for opportunity in result.composite_promo_opportunities:
        lines.extend([f"### {opportunity.story_arc_id}: {opportunity.narrative_theme}", "", opportunity.narrative_rationale])
        for beat in opportunity.beats:
            lines.append(f"- Episode {beat.episode_number} · {beat.beat_role}: {', '.join(beat.anchor_event_ids)} — {beat.reason} (spoiler {beat.spoiler_level})")
        lines.append("")
    lines.extend(["", "## Continuity uncertainties", ""])
    lines.extend(f"- {item.description} — " + ", ".join(f"E{r.episode_number}:{r.event_id}" for r in item.source_references)
                 for item in result.continuity_uncertainties)
    lines.append("")
    return "\n".join(lines)


def dry_run_report(all_episodes: list[ParsedEpisode], selected: list[ParsedEpisode],
                   analysis_map: dict, discovery_map: dict, *, series_output: Any = None) -> dict:
    settings = get_settings()
    episode_requests = 0
    missing = []
    for episode in selected:
        existing = analysis_map.get(str(episode.episode_number))
        if isinstance(existing, dict) and existing:
            continue
        missing.append(episode.episode_number)
        payload = _scene_payload(episode)
        ekey = episode_cache_key(payload, settings.gemini_model)
        if _cache_file_exists(settings.cache_dir, ekey):
            continue
        chunks = _build_chunks(episode, settings.max_chunk_chars)
        if chunks is None:
            episode_requests += 1
        else:
            for index, chunk in enumerate(chunks):
                ckey = content_hash({"parent": ekey, "chunk": chunk, "chunk_index": index})
                if not _cache_file_exists(settings.cache_dir, ckey):
                    episode_requests += 1
            episode_requests += 1  # aggregation request
    has_inputs = all(str(ep.episode_number) in analysis_map for ep in selected)
    series_cached = False
    series_requests = 0
    if has_inputs:
        try:
            skey = _cache_key(selected, analysis_map, discovery_map, settings.gemini_model)
            series_cached = _cache_file_exists(settings.cache_dir / "series_analysis", skey)
        except (KeyError, TypeError):
            pass
    if not series_cached:
        # Estimate aggregate chunk count even before missing episode analyses
        # exist, using schema-shaped placeholders rather than truncating scripts.
        estimation_map = dict(analysis_map)
        for episode in selected:
            estimation_map.setdefault(str(episode.episode_number), {
                "episode_number": episode.episode_number, "episode_summary": "",
                "important_characters": [], "character_relationships": [],
                "main_conflict": "", "important_story_developments": [],
                "major_revelations": [], "emotional_turning_points": [],
                "unresolved_questions": [], "narrative_progression": [],
            })
        try:
            series_requests = len(_episode_chunks(selected, estimation_map, discovery_map,
                                                   settings.max_chunk_chars)) + 1
        except ValueError:
            series_requests = max(1, len(selected)) + 1
    outputs = series_output if isinstance(series_output, dict) else {}
    opportunities = outputs.get("composite_promo_opportunities", []) if outputs else []
    report = {
        "episodes_available": len(all_episodes),
        "episodes_selected": [episode.episode_number for episode in selected],
        "existing_episode_analyses": sum(bool(analysis_map.get(str(ep.episode_number))) for ep in selected),
        "missing_episode_analyses": missing,
        "series_analysis_cache": "hit" if series_cached else "miss",
        "potential_composite_candidates_available": len(opportunities),
        "configured_model": settings.gemini_model or "(not set)",
        "estimated_requests": episode_requests + (0 if series_cached else series_requests),
        "episode_analysis_requests": episode_requests,
        "series_analysis_requests": 0 if series_cached else series_requests,
    }
    print(f"Episodes available: {report['episodes_available']}")
    print(f"Selected episodes: {', '.join(map(str, report['episodes_selected'])) or '(none)'}")
    print(f"Existing episode analyses: {report['existing_episode_analyses']}; missing: {len(missing)}")
    print(f"Series analysis cache: {report['series_analysis_cache']}")
    print(f"Potential composite candidates available: {report['potential_composite_candidates_available']}")
    print(f"Configured model: {report['configured_model']}")
    print(f"Estimated requests: {report['estimated_requests']}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Build source-grounded narrative understanding across selected episodes")
    parser.add_argument("--parsed", default=str(DEFAULT_PARSED))
    parser.add_argument("--analysis", default=str(DEFAULT_ANALYSIS))
    parser.add_argument("--discoveries", default=str(DEFAULT_DISCOVERIES))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--markdown", default=str(DEFAULT_MARKDOWN))
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--episode-number", type=int)
    selection.add_argument("--limit", type=int)
    selection.add_argument("--all-episodes", action="store_true", help="Explicitly authorize processing every parsed episode")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        all_episodes = load_episodes(args.parsed)
        if args.all_episodes:
            selected = all_episodes
        elif args.episode_number is not None or args.limit is not None:
            selected = select_episodes(all_episodes, episode_number=args.episode_number, limit=args.limit)
        elif args.dry_run:
            selected = all_episodes
        else:
            raise ValueError("Select scope explicitly with --episode-number, --limit, or --all-episodes.")
        analysis_map = _read_json(args.analysis, {})
        discovery_map = _read_json(args.discoveries, {})
        existing_series = _read_json(args.output, {})
        if args.dry_run:
            dry_run_report(all_episodes, selected, analysis_map, discovery_map,
                           series_output=existing_series)
            return
        missing = [ep for ep in selected if not analysis_map.get(str(ep.episode_number))]
        if missing:
            analysis_map, discovery_map = analyze_episodes(
                missing, analysis_path=args.analysis, discoveries_path=args.discoveries,
            )
        settings = get_settings()
        result = analyze_series(selected, analysis_map, discovery_map,
                                max_chunk_chars=settings.max_chunk_chars)
        _atomic_json(Path(args.output), result.model_dump(mode="json"))
        _atomic_text(Path(args.markdown), render_markdown(result))
        print(f"Series analysis saved for episodes {result.analyzed_episode_numbers} to {args.output}")
        print(f"Markdown report: {args.markdown}")
    except (OSError, ValueError, ValidationError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
