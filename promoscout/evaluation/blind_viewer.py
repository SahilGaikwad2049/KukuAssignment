"""Independent context-free evaluation of promotional clips for a first-time viewer."""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from promoscout.config import get_settings
from promoscout.generation.continuous import PromoCandidate
from promoscout.schemas import ParsedEpisode
from promoscout.evaluation.pilot import build_pilot_report, sample_candidate_records, save_pilot_report
from promoscout.utils.cache import JsonCache, content_hash
from promoscout.utils.llm_client import GeminiClient

PROMPT_VERSION = "blind-viewer-prompt-v1"
SCHEMA_VERSION = "blind-viewer-evaluation-schema-v1"
PROMPT_PATH = Path(__file__).resolve().parents[2] / "prompts" / "blind_viewer.txt"
DEFAULT_PARSED_PATH = Path("data/processed/parsed_all_episodes.json")
DEFAULT_SEMANTIC_CANDIDATES_PATH = Path("data/outputs/semantic_candidates.json")
DEFAULT_HEURISTIC_CANDIDATES_PATH = Path("data/outputs/continuous_candidates.json")
DEFAULT_OUTPUT_PATH = Path("data/outputs/blind_viewer_evaluations.json")
DEFAULT_SELECTION_PATH = Path("data/outputs/blind_viewer_pilot_selection.json")
DEFAULT_PILOT_REPORT_JSON = Path("data/outputs/blind_viewer_pilot_report.json")
DEFAULT_PILOT_REPORT_MARKDOWN = Path("data/outputs/blind_viewer_pilot_report.md")


class EvidenceReference(BaseModel):
    model_config = ConfigDict(extra="forbid")
    event_id: str
    excerpt: str = Field(min_length=1, max_length=240)
    why_it_matters: str = Field(min_length=1)


class DimensionScore(BaseModel):
    model_config = ConfigDict(extra="forbid")
    score: int = Field(ge=1, le=5)
    rationale: str
    evidence: list[EvidenceReference] = Field(min_length=1)


class PerceivedCharacter(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name_or_description: str
    perceived_role: str
    certainty: Literal["clear", "uncertain", "unknown"]
    evidence: list[EvidenceReference] = Field(default_factory=list)


class BlindViewerAssessment(BaseModel):
    """LLM response schema; contains no episode or candidate metadata."""

    model_config = ConfigDict(extra="forbid")
    standalone_story_reconstruction: str
    characters_the_viewer_believes_are_involved: list[PerceivedCharacter] = Field(default_factory=list)
    directly_shown_or_stated: list[str] = Field(default_factory=list)
    reasonable_inferences: list[str] = Field(default_factory=list)
    unclear_or_unestablished: list[str] = Field(default_factory=list)
    main_conflict_as_understood: str
    perceived_stakes: str
    main_unanswered_question: str | None = None
    confusing_elements: list[str] = Field(default_factory=list)
    missing_context_the_viewer_wishes_for: list[str] = Field(default_factory=list)
    standalone_clarity: DimensionScore
    hook_effectiveness: DimensionScore
    conflict_comprehension: DimensionScore
    stakes_clarity: DimensionScore
    cliffhanger_effectiveness: DimensionScore
    desire_to_continue: DimensionScore
    overall_viewer_verdict: Literal["strong", "mixed", "weak"]
    confidence: float = Field(ge=0, le=1)
    verdict_explanation: str


class BlindViewerEvaluation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    candidate_id: str
    candidate_source: Literal["semantic", "heuristic", "composite"]
    episode_number: int | None = None
    evaluated_at: str
    model_id: str
    clip_event_ids: list[str]
    blind_input_hash: str
    assessment: BlindViewerAssessment
    composite_source_event_mapping: dict[str, str] = Field(default_factory=dict)


class InvalidEvidenceReference(ValueError):
    """Model evidence cites an event that is not in the blind clip payload."""


def _candidate_model(value: PromoCandidate | dict) -> PromoCandidate:
    if isinstance(value, PromoCandidate):
        return value
    # Origin and any precomputed previews are pipeline metadata, not model input.
    clean = {key: item for key, item in value.items() if key not in {"source", "preview", "preview_events"}}
    return PromoCandidate.model_validate(clean)


def build_blind_view_input(episode: ParsedEpisode, candidate: PromoCandidate | dict) -> dict:
    """Build only the time-bounded clip content, with no episode-level metadata."""
    candidate = _candidate_model(candidate)
    if candidate.episode_number != episode.episode_number:
        raise ValueError("Candidate and parsed episode numbers do not match")
    if candidate.end_seconds < candidate.start_seconds:
        raise ValueError("Candidate clip end precedes its start")
    overlapping = [
        event for event in episode.events
        if event.start_seconds <= candidate.end_seconds
        and event.end_seconds >= candidate.start_seconds
    ]
    overlapping.sort(key=lambda event: (
        event.start_seconds, event.source_line_start, event.end_seconds, event.event_id,
    ))
    duration = candidate.end_seconds - candidate.start_seconds
    clip_events = []
    for event in overlapping:
        visible_start = max(event.start_seconds, candidate.start_seconds)
        visible_end = min(event.end_seconds, candidate.end_seconds)
        row = {
            "event_id": event.event_id,
            "event_type": event.event_type,
            "relative_start_seconds": round(visible_start - candidate.start_seconds, 2),
            "relative_end_seconds": round(visible_end - candidate.start_seconds, 2),
            "content": event.content,
        }
        if event.speaker:
            row["speaker"] = event.speaker
        if event.delivery:
            row["delivery"] = event.delivery
        clip_events.append(row)
    return {
        "clip_duration_seconds": round(duration, 2),
        "events": clip_events,
    }


def build_composite_blind_view_input(episodes: list[ParsedEpisode], candidate) -> tuple[dict, dict[str, str]]:
    """Rebuild stitched clip events from source and strip all episode provenance.

    Returned mapping is retained only in the local evaluation record; the LLM
    payload uses anonymous, sequential event IDs and relative promo time only.
    """
    from promoscout.generation.composite import CompositeCandidate
    composite = candidate if isinstance(candidate, CompositeCandidate) else CompositeCandidate.model_validate(candidate)
    episode_map = {episode.episode_number: episode for episode in episodes}
    blind_events, anonymous_to_source = [], {}
    for segment in composite.segments:
        episode = episode_map.get(segment.episode_number)
        if episode is None:
            raise ValueError(f"Composite references unavailable episode {segment.episode_number}")
        events = [event for event in episode.events
                  if event.start_seconds <= segment.source_end_seconds
                  and event.end_seconds >= segment.source_start_seconds]
        events.sort(key=lambda event: (event.start_seconds, event.source_line_start, event.end_seconds, event.event_id))
        expected_ids = [event.event_id for event in events]
        if expected_ids != segment.source_event_ids:
            raise ValueError(f"Composite source events for segment {segment.segment_index} do not match original screenplay window")
        for event in events:
            anonymous_id = f"CLIP_EVENT_{len(blind_events) + 1:04d}"
            anonymous_to_source[anonymous_id] = f"{segment.episode_number}:{event.event_id}"
            row = {
                "event_id": anonymous_id,
                "event_type": event.event_type,
                "relative_start_seconds": round(segment.relative_start_seconds + max(0.0, event.start_seconds - segment.source_start_seconds), 2),
                "relative_end_seconds": round(segment.relative_start_seconds + min(segment.duration_seconds, event.end_seconds - segment.source_start_seconds), 2),
                "content": event.content,
            }
            if event.speaker:
                row["speaker"] = event.speaker
            if event.delivery:
                row["delivery"] = event.delivery
            blind_events.append(row)
    blind_events.sort(key=lambda item: (item["relative_start_seconds"], item["relative_end_seconds"], item["event_id"]))
    return {"clip_duration_seconds": composite.total_duration_seconds, "events": blind_events}, anonymous_to_source


def evaluate_composite_candidate(episodes: list[ParsedEpisode], candidate, *, llm=None,
                                 cache: JsonCache | None = None) -> BlindViewerEvaluation:
    from promoscout.generation.composite import CompositeCandidate
    composite = candidate if isinstance(candidate, CompositeCandidate) else CompositeCandidate.model_validate(candidate)
    settings = get_settings()
    model_id = getattr(llm, "model", settings.gemini_model)
    payload, mapping = build_composite_blind_view_input(episodes, composite)
    key = content_hash({"candidate_id": composite.candidate_id, "candidate_source": "composite",
                        "candidate_content": payload, "model_id": model_id,
                        "prompt_version": PROMPT_VERSION, "prompt_content_hash": content_hash(_prompt_text()),
                        "schema_version": SCHEMA_VERSION})
    cache = cache or JsonCache(settings.cache_dir / "blind_viewer")
    source_by_id = {row["event_id"]: row["content"] for row in payload["events"]}
    cached = _cached_evaluation(cache, key, source_by_id)
    if cached is not None:
        return cached
    if llm is None:
        llm = GeminiClient()
        model_id = llm.model
        key = content_hash({"candidate_id": composite.candidate_id, "candidate_source": "composite",
                            "candidate_content": payload, "model_id": model_id,
                            "prompt_version": PROMPT_VERSION, "prompt_content_hash": content_hash(_prompt_text()),
                            "schema_version": SCHEMA_VERSION})
        cached = _cached_evaluation(cache, key, source_by_id)
        if cached is not None:
            return cached
    prompt = _prompt_text() + "\n\nBLIND CLIP INPUT (the only story information available):\n" + json.dumps(payload, ensure_ascii=False)
    assessment = llm.generate_json_schema(prompt, BlindViewerAssessment)
    if not isinstance(assessment, BlindViewerAssessment):
        assessment = BlindViewerAssessment.model_validate(assessment)
    _validate_evidence(assessment, source_by_id)
    result = BlindViewerEvaluation(
        candidate_id=composite.candidate_id, candidate_source="composite", episode_number=None,
        evaluated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"), model_id=model_id,
        clip_event_ids=list(mapping), blind_input_hash=content_hash(payload), assessment=assessment,
        composite_source_event_mapping=mapping,
    )
    cache.put(key, result.model_dump(mode="json"))
    return result


def _prompt_text() -> str:
    return PROMPT_PATH.read_text(encoding="utf-8")


def _validate_evidence(assessment: BlindViewerAssessment, source_by_id: dict[str, str]) -> None:
    evidence: list[EvidenceReference] = []
    for field_name in (
        "standalone_clarity", "hook_effectiveness", "conflict_comprehension",
        "stakes_clarity", "cliffhanger_effectiveness", "desire_to_continue",
    ):
        evidence.extend(getattr(assessment, field_name).evidence)
    for character in assessment.characters_the_viewer_believes_are_involved:
        evidence.extend(character.evidence)
    invalid = sorted({item.event_id for item in evidence if item.event_id not in source_by_id})
    if invalid:
        raise InvalidEvidenceReference(
            "Blind-viewer assessment cited event ID(s) outside the supplied clip: "
            + ", ".join(invalid)
        )
    invented_quotes = [
        item.event_id for item in evidence
        if item.excerpt not in source_by_id[item.event_id]
    ]
    if invented_quotes:
        raise InvalidEvidenceReference(
            "Blind-viewer assessment included quotation(s) not present in source events: "
            + ", ".join(sorted(set(invented_quotes)))
        )


def _cached_evaluation(cache: JsonCache, key: str, source_by_id: dict[str, str]):
    try:
        cached = cache.get(key)
        if cached is None:
            return None
        result = BlindViewerEvaluation.model_validate(cached)
        _validate_evidence(result.assessment, source_by_id)
        return result
    except (OSError, json.JSONDecodeError, ValidationError, InvalidEvidenceReference):
        return None


def _cache_key(candidate: PromoCandidate, source: str, model_id: str,
               blind_payload: dict) -> str:
    return content_hash({
        "candidate_id": candidate.candidate_id,
        "candidate_source": source,
        "candidate_content": blind_payload,
        "model_id": model_id,
        "prompt_version": PROMPT_VERSION,
        # Include file content too, so an edited prompt cannot accidentally
        # reuse old results when its version constant was not bumped.
        "prompt_content_hash": content_hash(_prompt_text()),
        "schema_version": SCHEMA_VERSION,
    })


def evaluate_candidate(
    episode: ParsedEpisode,
    candidate: PromoCandidate | dict,
    *,
    source: Literal["semantic", "heuristic"] = "semantic",
    llm=None,
    cache: JsonCache | None = None,
) -> BlindViewerEvaluation:
    """Evaluate one candidate independently; cache hits do not construct a client."""
    candidate = _candidate_model(candidate)
    settings = get_settings()
    model_id = getattr(llm, "model", settings.gemini_model)
    blind_payload = build_blind_view_input(episode, candidate)
    key = _cache_key(candidate, source, model_id, blind_payload)
    cache = cache or JsonCache(settings.cache_dir / "blind_viewer")
    source_by_id = {event["event_id"]: event["content"] for event in blind_payload["events"]}

    cached = _cached_evaluation(cache, key, source_by_id)
    if cached is not None:
        return cached

    if llm is None:
        llm = GeminiClient()
        model_id = llm.model
        key = _cache_key(candidate, source, model_id, blind_payload)
        cached = _cached_evaluation(cache, key, source_by_id)
        if cached is not None:
            return cached

    prompt = (
        _prompt_text()
        + "\n\nBLIND CLIP INPUT (the only story information available):\n"
        + json.dumps(blind_payload, ensure_ascii=False)
    )
    assessment = llm.generate_json_schema(prompt, BlindViewerAssessment)
    if not isinstance(assessment, BlindViewerAssessment):
        assessment = BlindViewerAssessment.model_validate(assessment)
    _validate_evidence(assessment, source_by_id)
    result = BlindViewerEvaluation(
        candidate_id=candidate.candidate_id,
        candidate_source=source,
        episode_number=episode.episode_number,
        evaluated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        model_id=model_id,
        clip_event_ids=[event["event_id"] for event in blind_payload["events"]],
        blind_input_hash=content_hash(blind_payload),
        assessment=assessment,
    )
    cache.put(key, result.model_dump(mode="json"))
    return result


def _read_json(path: Path, *, required: bool = False):
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Required input file does not exist: {path}")
        return None
    if not path.stat().st_size:
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc


def load_episodes(path: str | Path) -> list[ParsedEpisode]:
    payload = _read_json(Path(path), required=True)
    if not isinstance(payload, list):
        raise ValueError("Parsed screenplay JSON must contain a top-level list")
    return [ParsedEpisode.model_validate(row) for row in payload]


def load_candidate_records(payload: Any, source: Literal["semantic", "heuristic", "composite"]):
    if payload is None:
        return []
    rows = payload if isinstance(payload, list) else payload.get("candidates", []) if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        raise ValueError("Candidate JSON must be a list or an object containing a candidates list")
    result = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"Candidate row {index + 1} is not an object")
        if source == "composite":
            from promoscout.generation.composite import CompositeCandidate
            candidate = CompositeCandidate.model_validate(row)
        else:
            candidate = _candidate_model(row)
        result.append({"candidate": candidate, "source": source})
    return result


def _record_episode_numbers(record: dict) -> set[int]:
    candidate = record["candidate"]
    if record["source"] == "composite":
        return {segment.episode_number for segment in candidate.segments}
    return {candidate.episode_number}


def select_candidate_records(records: list[dict], *, episode_number: int | None = None,
                             candidate_id: str | None = None, limit: int | None = None) -> list[dict]:
    if limit is not None and limit < 1:
        raise ValueError("--limit must be at least 1")
    selected = records
    if episode_number is not None:
        selected = [row for row in selected if episode_number in _record_episode_numbers(row)]
    if candidate_id is not None:
        selected = [row for row in selected if row["candidate"].candidate_id == candidate_id]
        if not selected:
            raise ValueError(f"Candidate {candidate_id!r} was not found in the selected source and episode")
        if len(selected) > 1:
            raise ValueError(f"Candidate ID {candidate_id!r} is ambiguous; filter by source")
    if limit is not None:
        selected = selected[:limit]
    return selected


def evaluation_cache_key(record: dict, episode: ParsedEpisode, model_id: str) -> str:
    candidate = record["candidate"]
    return _cache_key(candidate, record["source"], model_id, build_blind_view_input(episode, candidate))


def dry_run_report(records: list[dict], episodes: list[ParsedEpisode], *, cache_dir: str | Path | None = None,
                   model_id: str | None = None) -> dict:
    """Inspect work/cache without creating JsonCache or GeminiClient instances."""
    settings = get_settings()
    model_id = model_id if model_id is not None else settings.gemini_model
    cache_path = Path(cache_dir) if cache_dir is not None else settings.cache_dir / "blind_viewer"
    episode_by_number = {episode.episode_number: episode for episode in episodes}
    cached = 0
    for record in records:
        if record["source"] == "composite":
            payload, mapping = build_composite_blind_view_input(episodes, record["candidate"])
            key = content_hash({"candidate_id": record["candidate"].candidate_id, "candidate_source": "composite",
                                "candidate_content": payload, "model_id": model_id,
                                "prompt_version": PROMPT_VERSION, "prompt_content_hash": content_hash(_prompt_text()),
                                "schema_version": SCHEMA_VERSION})
            source_by_id = {row["event_id"]: row["content"] for row in payload["events"]}
        else:
            episode = episode_by_number.get(record["candidate"].episode_number)
            if episode is None:
                continue
            key = evaluation_cache_key(record, episode, model_id)
            source_by_id = {event["event_id"]: event["content"] for event in build_blind_view_input(episode, record["candidate"])["events"]}
        cache_file = cache_path / f"{key}.json"
        if cache_file.exists():
            try:
                parsed = BlindViewerEvaluation.model_validate(json.loads(cache_file.read_text(encoding="utf-8")))
                _validate_evidence(parsed.assessment, source_by_id)
                cached += 1
            except (OSError, json.JSONDecodeError, ValidationError, InvalidEvidenceReference):
                pass
    return {
        "selected_candidates": len(records),
        "cached_candidates": cached,
        "requiring_evaluation": len(records) - cached,
        "model_id": model_id or "(not configured)",
        "estimated_requests": len(records) - cached,
    }


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                     prefix=f".{path.name}.", suffix=".tmp", delete=False) as temp:
        temp_name = temp.name
        json.dump(payload, temp, ensure_ascii=False, indent=2)
        temp.flush()
        os.fsync(temp.fileno())
    os.replace(temp_name, path)


def run_evaluations(
    records: list[dict], episodes: list[ParsedEpisode], *, llm=None,
    cache: JsonCache | None = None, cache_dir: str | Path | None = None,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
) -> tuple[dict, list[dict]]:
    """Evaluate sequentially and persist every successful record immediately."""
    settings = get_settings()
    cache = cache or JsonCache(cache_dir or (settings.cache_dir / "blind_viewer"))
    output_path = Path(output_path)
    existing = _read_json(output_path)
    if existing is None:
        output = {"schema_version": SCHEMA_VERSION, "evaluations": {}}
    elif isinstance(existing, dict) and isinstance(existing.get("evaluations"), dict):
        output = existing
    else:
        raise ValueError(f"Existing evaluation output has an invalid structure: {output_path}")
    episode_by_number = {episode.episode_number: episode for episode in episodes}
    failures = []
    if records and llm is None:
        # Client construction is delayed until after selection and successful dry-run.
        llm = GeminiClient()
    for record in records:
        candidate, source = record["candidate"], record["source"]
        key = f"{source}:{candidate.candidate_id}"
        try:
            if source == "composite":
                result = evaluate_composite_candidate(episodes, candidate, llm=llm, cache=cache)
            else:
                episode = episode_by_number.get(candidate.episode_number)
                if episode is None:
                    raise ValueError(f"Parsed episode {candidate.episode_number} was not found")
                result = evaluate_candidate(episode, candidate, source=source, llm=llm, cache=cache)
            output["evaluations"][key] = result.model_dump(mode="json")
            _atomic_write_json(output_path, output)
        except Exception as exc:
            failures.append({"candidate_id": candidate.candidate_id, "source": source, "error": str(exc)})
    return output, failures


def _report_dry_run(report: dict, *, records: list[dict] | None = None,
                    manifest_path: str | Path | None = None,
                    source_filter: str | None = None) -> None:
    print(f"Selected candidates: {report['selected_candidates']}")
    if source_filter:
        print(f"Candidate source filter: {source_filter}")
    if records is not None:
        episodes = sorted({number for row in records for number in _record_episode_numbers(row)})
        print(f"Selected episodes: {len(episodes)} ({', '.join(map(str, episodes)) or 'none'})")
        for row in records:
            episode_label = ",".join(f"EP{number:03d}" for number in sorted(_record_episode_numbers(row)))
            print(f"  {row['source']}: {episode_label} {row['candidate'].candidate_id}")
    print(f"Already evaluated in cache: {report['cached_candidates']}")
    print(f"Requiring evaluation: {report['requiring_evaluation']}")
    print(f"Configured model: {report['model_id']}")
    print(f"Estimated requests: {report['estimated_requests']}")
    if manifest_path:
        print(f"Selection manifest: {manifest_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Independently evaluate promo clips as a first-time viewer")
    parser.add_argument("--parsed", default=str(DEFAULT_PARSED_PATH), help="Parsed screenplay JSON")
    parser.add_argument("--candidates", default=str(DEFAULT_SEMANTIC_CANDIDATES_PATH), help="Semantic candidate JSON")
    parser.add_argument("--heuristic-candidates", default=str(DEFAULT_HEURISTIC_CANDIDATES_PATH), help="Heuristic candidate JSON")
    parser.add_argument("--composite-candidates", default="data/outputs/composite_candidates.json", help="Composite candidate JSON")
    parser.add_argument("--source", choices=("semantic", "heuristic", "composite", "both"), default="semantic", help="Candidate origin; semantic is the default")
    parser.add_argument("--episode-number", type=int)
    parser.add_argument("--candidate-id")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--sample-episodes", type=int, help="Select candidates from this many stratified episodes")
    parser.add_argument("--candidates-per-episode", type=int, default=1)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--selection-manifest", default=str(DEFAULT_SELECTION_PATH))
    parser.add_argument("--pilot-report-json", default=str(DEFAULT_PILOT_REPORT_JSON))
    parser.add_argument("--pilot-report-markdown", default=str(DEFAULT_PILOT_REPORT_MARKDOWN))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH))
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    if args.sample_episodes is not None and args.sample_episodes < 1:
        parser.error("--sample-episodes must be at least 1")
    if args.candidates_per_episode < 1:
        parser.error("--candidates-per-episode must be at least 1")
    if args.sample_episodes is None and args.candidates_per_episode != 1:
        parser.error("--candidates-per-episode requires --sample-episodes")
    if args.sample_episodes is not None and any((args.episode_number is not None, args.candidate_id is not None, args.limit is not None)):
        parser.error("--sample-episodes cannot be combined with --episode-number, --candidate-id, or --limit")
    if args.sample_episodes is not None and args.source == "composite":
        parser.error("--sample-episodes is for episode-level semantic or heuristic candidates; select composite candidates by ID or limit")
    try:
        episodes = load_episodes(args.parsed)
        sources = ("semantic", "heuristic") if args.source == "both" else (args.source,)
        source_paths = {"semantic": args.candidates, "heuristic": args.heuristic_candidates, "composite": args.composite_candidates}
        records = []
        for source in sources:
            records.extend(load_candidate_records(_read_json(Path(source_paths[source])), source))
        manifest_path = None
        if args.sample_episodes is not None:
            records, manifest = sample_candidate_records(
                records,
                sample_episodes=args.sample_episodes,
                candidates_per_episode=args.candidates_per_episode,
                seed=args.sample_seed,
            )
            manifest_path = Path(args.selection_manifest)
            prior_manifest = _read_json(manifest_path)
            if not args.dry_run and isinstance(prior_manifest, dict):
                same_request = all(prior_manifest.get(key) == manifest.get(key) for key in (
                    "candidate_sources", "sample_seed", "requested_episode_count",
                    "candidates_per_episode_requested",
                ))
                prior_ids = [
                    (row.get("candidate_source"), row.get("episode_number"), row.get("candidate_id"))
                    for row in prior_manifest.get("selected_candidates", [])
                ]
                current_ids = [
                    (row.get("candidate_source"), row.get("episode_number"), row.get("candidate_id"))
                    for row in manifest["selected_candidates"]
                ]
                if same_request and prior_ids != current_ids:
                    parser.error(
                        "The candidate pool changed since the saved sample was inspected. "
                        "Run --dry-run again and review the updated manifest before evaluation."
                    )
            _atomic_write_json(manifest_path, manifest)
            print(
                f"Sampled {manifest['selected_episode_count']} of "
                f"{manifest['requested_episode_count']} requested episode(s); "
                f"selected {manifest['selected_candidate_count']} candidate(s)."
            )
        else:
            records = select_candidate_records(records, episode_number=args.episode_number,
                                               candidate_id=args.candidate_id, limit=args.limit)
    except (OSError, ValueError, ValidationError) as exc:
        parser.error(str(exc))

    report = dry_run_report(records, episodes)
    if args.dry_run:
        _report_dry_run(report, records=records,
                        manifest_path=manifest_path, source_filter=args.source)
        return
    print(f"Evaluating {len(records)} candidate(s) sequentially with {report['model_id']}.")
    evaluation_output, failures = run_evaluations(records, episodes, output_path=args.output)
    review_payload = _read_json(Path("data/outputs/review_decisions.json")) or {}
    human_decisions = review_payload.get("decisions", {}) if isinstance(review_payload, dict) else {}
    if args.source == "composite":
        print("Composite blind-viewer evaluations saved; episode-sampling pilot reports apply to single-episode candidate sources.")
        return
    pilot_report = build_pilot_report(
        records,
        evaluation_output,
        failures,
        model_id=report["model_id"],
        cached_count=report["cached_candidates"],
        human_decisions=human_decisions,
    )
    save_pilot_report(pilot_report, args.pilot_report_json, args.pilot_report_markdown)
    if failures:
        print(f"Completed with {len(failures)} candidate failure(s):")
        for failure in failures:
            print(f"  {failure['source']}:{failure['candidate_id']}: {failure['error']}")
    else:
        print(f"Saved evaluations to {args.output}")
    print(f"Pilot report: {args.pilot_report_json}")
    print(f"Pilot report (Markdown): {args.pilot_report_markdown}")


if __name__ == "__main__":
    main()
