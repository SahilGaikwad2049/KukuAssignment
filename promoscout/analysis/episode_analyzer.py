"""Narrative analysis and semantic event discovery for parsed episodes."""
import argparse
import json
import os
from pathlib import Path

from promoscout.analysis.event_tagger import PROMPT_VERSION, validate_discoveries
from promoscout.analysis.models import AnalysisEnvelope
from promoscout.config import get_settings
from promoscout.schemas import ParsedEpisode
from promoscout.utils.cache import JsonCache, content_hash
from promoscout.utils.llm_client import GeminiClient

SCHEMA_VERSION = "episode-analysis-schema-v1"
ANALYSIS_PROMPT_VERSION = "episode-analysis-v1"
_PROMPTS = Path(__file__).resolve().parents[2] / "prompts"


def _write_json_atomic(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def _prompt(name: str) -> str:
    return (_PROMPTS / name).read_text(encoding="utf-8")


def _generate_analysis(llm, prompt: str) -> AnalysisEnvelope:
    """Prefer Gemini's JSON Schema mode, with compatibility for existing clients."""
    generate_json_schema = getattr(llm, "generate_json_schema", None)
    if callable(generate_json_schema):
        return generate_json_schema(prompt, AnalysisEnvelope)
    return llm.generate(prompt, AnalysisEnvelope)


def _scene_payload(episode: ParsedEpisode, scene_indexes: list[int] | None = None) -> dict:
    scenes = episode.scenes if scene_indexes is None else [episode.scenes[i] for i in scene_indexes]
    return {
        "episode_number": episode.episode_number,
        "logline": episode.logline,
        "continuity_notes": episode.continuity_notes,
        "characters_in_episode": sorted({e.speaker for e in episode.events if e.speaker}),
        "scenes": [{
            "scene_number": s.scene_number,
            "heading": s.heading,
            "events": [{
                "event_id": e.event_id, "event_type": e.event_type, "label": e.label,
                "speaker": e.speaker, "delivery": e.delivery,
                "start_timestamp": e.start_timestamp, "end_timestamp": e.end_timestamp,
                "content": e.content,
            } for e in s.events],
        } for s in scenes],
    }


def _cache_key(payload: dict, model: str) -> str:
    return content_hash({
        "episode_content": payload, "model": model,
        "prompt_versions": [ANALYSIS_PROMPT_VERSION, PROMPT_VERSION, SCHEMA_VERSION],
    })


def _build_chunks(episode: ParsedEpisode, limit: int) -> list[dict] | None:
    """Return scene/event chunks if the full payload exceeds limit, else None."""
    full = _scene_payload(episode)
    if len(json.dumps(full, ensure_ascii=False)) <= limit:
        return None
    chunks: list[dict] = []
    current_indexes: list[int] = []
    for index, scene in enumerate(episode.scenes):
        candidate = _scene_payload(episode, current_indexes + [index])
        if current_indexes and len(json.dumps(candidate, ensure_ascii=False)) > limit:
            chunks.append(_scene_payload(episode, current_indexes))
            current_indexes = []
        scene_payload = _scene_payload(episode, [index])
        if len(json.dumps(scene_payload, ensure_ascii=False)) <= limit:
            current_indexes.append(index)
            continue
        if current_indexes:
            chunks.append(_scene_payload(episode, current_indexes))
            current_indexes = []
        group = []
        for event in scene.events:
            trial = group + [event]
            fragment = {
                "episode_number": episode.episode_number,
                "logline": episode.logline,
                "continuity_notes": episode.continuity_notes,
                "characters_in_episode": sorted({e.speaker for e in episode.events if e.speaker}),
                "scenes": [{"scene_number": scene.scene_number, "heading": scene.heading,
                            "events": [e.model_dump(mode="json") for e in trial]}],
            }
            if group and len(json.dumps(fragment, ensure_ascii=False)) > limit:
                chunks.append({**fragment, "scenes": [{**fragment["scenes"][0], "events": [e.model_dump(mode="json") for e in group]}]})
                group = [event]
            else:
                group = trial
        if group:
            chunks.append({
                "episode_number": episode.episode_number,
                "logline": episode.logline,
                "continuity_notes": episode.continuity_notes,
                "characters_in_episode": sorted({e.speaker for e in episode.events if e.speaker}),
                "scenes": [{"scene_number": scene.scene_number, "heading": scene.heading,
                            "events": [e.model_dump(mode="json") for e in group]}],
            })
    if current_indexes:
        chunks.append(_scene_payload(episode, current_indexes))
    return chunks


def select_episodes(episodes: list[ParsedEpisode], *, episode_number: int | None = None,
                    limit: int | None = None) -> list[ParsedEpisode]:
    """Select episodes before any model client or request is created."""
    if episode_number is not None and limit is not None:
        raise ValueError("--episode-number and --limit cannot be used together")
    if limit is not None:
        if limit < 1:
            raise ValueError("--limit must be at least 1")
        return episodes[:limit]
    if episode_number is not None:
        selected = [episode for episode in episodes if episode.episode_number == episode_number]
        if not selected:
            available = ", ".join(str(episode.episode_number) for episode in episodes)
            raise ValueError(f"Episode {episode_number} was not found. Available episode numbers: {available or '(none)'}")
        return selected
    return episodes


def _load_existing_json_map(path: Path) -> dict:
    if not path.exists() or not path.stat().st_size:
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Cannot merge existing output {path}: invalid JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Cannot merge existing output {path}: expected an episode-number object")
    return value


def analyze_episode(episode: ParsedEpisode, *, llm=None, cache: JsonCache | None = None,
                    max_chunk_chars: int | None = None) -> AnalysisEnvelope:
    """Analyze an episode, combining scene chunks only when required."""
    settings = get_settings()
    llm = llm or GeminiClient()
    cache = cache or JsonCache(settings.cache_dir)
    model_id = getattr(llm, "model", settings.gemini_model)
    full_payload = _scene_payload(episode)
    key = _cache_key(full_payload, model_id)
    cached = cache.get(key)
    valid_ids = {event.event_id for event in episode.events}
    if len(valid_ids) != len(episode.events):
        raise ValueError("Original parsed episode contains duplicate event IDs")
    if cached is not None:
        cached["potential_events"] = [d.model_dump() for d in validate_discoveries(cached.get("potential_events", []), valid_ids)]
        return AnalysisEnvelope.model_validate(cached)

    limit = max_chunk_chars or settings.max_chunk_chars
    chunks = _build_chunks(episode, limit)
    prompt_header = _prompt("episode_analysis.txt") + "\n\n" + _prompt("event_tagging.txt")
    if chunks is None:
        response = _generate_analysis(llm, prompt_header + "\n\nEPISODE DATA:\n" + json.dumps(full_payload, ensure_ascii=False))
    else:
        partials = []
        for chunk_payload in chunks:
            chunk_key = content_hash({"parent": key, "chunk": chunk_payload, "chunk_index": len(partials)})
            chunk_data = cache.get(chunk_key)
            if chunk_data is None:
                chunk_response = _generate_analysis(llm, prompt_header + "\n\nEPISODE CHUNK (partial; analyze only supplied events):\n" + json.dumps(chunk_payload, ensure_ascii=False))
                chunk_data = chunk_response.model_dump(mode="json")
                cache.put(chunk_key, chunk_data)
            chunk_response = AnalysisEnvelope.model_validate(chunk_data)
            chunk_valid_ids = {
                event["event_id"]
                for scene in chunk_payload["scenes"]
                for event in scene["events"]
            }
            chunk_response.potential_events = validate_discoveries(
                chunk_response.potential_events, chunk_valid_ids
            )
            partials.append(chunk_response)

        aggregate_input = {
            "episode_metadata": {"episode_number": episode.episode_number, "logline": episode.logline, "continuity_notes": episode.continuity_notes},
            "chunk_analyses": [part.episode_analysis.model_dump() for part in partials],
            "chunk_discoveries": [d.model_dump() for part in partials for d in part.potential_events],
            "all_valid_event_ids": sorted(valid_ids),
        }
        response = _generate_analysis(
            llm,
            prompt_header + "\n\nAggregate these complete scene analyses into one episode-level response. Preserve discovery IDs exactly and only from all_valid_event_ids.\n" + json.dumps(aggregate_input, ensure_ascii=False),
        )

    validated = validate_discoveries(response.potential_events, valid_ids)
    response.potential_events = validated
    response.episode_analysis.episode_number = episode.episode_number
    result = response.model_dump(mode="json")
    cache.put(key, result)
    return response


def analyze_episodes(episodes: list[ParsedEpisode], *, llm=None,
                     analysis_path="data/processed/episode_analysis.json",
                     discoveries_path="data/processed/event_discoveries.json"):
    """Process episodes incrementally and atomically persist both deliverables."""
    analysis_file, discovery_file = Path(analysis_path), Path(discoveries_path)
    analysis_file.parent.mkdir(parents=True, exist_ok=True)
    discovery_file.parent.mkdir(parents=True, exist_ok=True)
    analyses = _load_existing_json_map(analysis_file)
    discoveries = _load_existing_json_map(discovery_file)
    llm = llm or GeminiClient()
    for episode in episodes:
        envelope = analyze_episode(episode, llm=llm)
        analyses[str(episode.episode_number)] = envelope.episode_analysis.model_dump(mode="json")
        discoveries[str(episode.episode_number)] = [d.model_dump(mode="json") for d in envelope.potential_events]
        _write_json_atomic(analysis_file, analyses)
        _write_json_atomic(discovery_file, discoveries)
    return analyses, discoveries


def load_episodes(path: str | Path) -> list[ParsedEpisode]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Parsed episodes JSON must contain a top-level list")
    return [ParsedEpisode.model_validate(item) for item in payload]


def dry_run_report(episodes: list[ParsedEpisode]) -> list[dict]:
    """Report requests/cache/config without constructing an API client or writing files."""
    settings = get_settings()
    key_configured = bool(settings.gemini_api_key and settings.gemini_api_key.lower() not in {
        "your_api_key_here", "your_gemini_api_key", "changeme", "replace_me",
    })
    model_configured = bool(settings.gemini_model) and settings.gemini_model.lower() not in {
        "your_selected_free_tier_model", "your_model_name", "model_name", "your_gemini_model",
    }
    provider_configured = settings.llm_provider == "gemini"
    print(f"Configured model: {settings.gemini_model or '(not set)'}")
    print(f"API key configured: {'yes' if key_configured else 'no'}")
    print(f"Provider configuration: {'ready' if provider_configured else 'unsupported: ' + settings.llm_provider}")
    print(f"Gemini configuration: {'ready' if key_configured and model_configured and provider_configured else 'incomplete'}")
    report = []
    for episode in episodes:
        payload = _scene_payload(episode)
        key = _cache_key(payload, settings.gemini_model)
        completed = (settings.cache_dir / f"{key}.json").is_file()
        chunks = _build_chunks(episode, settings.max_chunk_chars)
        if completed:
            cached_chunks = 0
            estimated = 0
        elif chunks is None:
            cached_chunks = 0
            estimated = 1
        else:
            cached_chunks = sum(
                (settings.cache_dir / f"{content_hash({'parent': key, 'chunk': chunk, 'chunk_index': index})}.json").is_file()
                for index, chunk in enumerate(chunks)
            )
            estimated = (len(chunks) - cached_chunks) + 1  # remaining chunks plus aggregation
        item = {
            "episode_number": episode.episode_number,
            "event_count": len(episode.events),
            "completed_cache": completed,
            "cached_chunks": cached_chunks,
            "chunk_count": len(chunks) if chunks is not None else 0,
            "estimated_requests": estimated,
        }
        report.append(item)
        print(
            f"Episode {episode.episode_number}: {len(episode.events)} events; "
            f"completed cache: {'yes' if completed else 'no'}; "
            f"chunks: {item['chunk_count']}; estimated requests: {estimated}"
        )
    print("Episodes that would be processed: " + ", ".join(str(e.episode_number) for e in episodes))
    print(f"Estimated total requests: {sum(item['estimated_requests'] for item in report)}")
    return report


def main():
    parser = argparse.ArgumentParser(description="Analyze episodes and discover semantic promo events")
    parser.add_argument("input")
    parser.add_argument("--analysis-output", default="data/processed/episode_analysis.json")
    parser.add_argument("--discoveries-output", default="data/processed/event_discoveries.json")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--episode-number", type=int, help="Process only this episode number")
    selection.add_argument("--limit", type=int, help="Process only the first N episodes in the input")
    parser.add_argument("--dry-run", action="store_true", help="Inspect selection, configuration, cache, and estimated requests without API calls or output writes")
    args = parser.parse_args()
    try:
        selected = select_episodes(
            load_episodes(args.input), episode_number=args.episode_number, limit=args.limit
        )
    except ValueError as exc:
        parser.error(str(exc))
    if args.dry_run:
        dry_run_report(selected)
        return
    analyze_episodes(selected, analysis_path=args.analysis_output, discoveries_path=args.discoveries_output)


if __name__ == "__main__":
    main()
