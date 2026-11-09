"""Source-grounded composite promo construction from series-level story plans."""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from promoscout.analysis.series_analyzer import validate_series_references
from promoscout.analysis.series_models import CompositePromoOpportunity, SeriesAnalysis
from promoscout.schemas import ParsedEpisode

DEFAULT_PARSED = "data/processed/parsed_all_episodes.json"
DEFAULT_SERIES = "data/processed/series_analysis.json"
DEFAULT_OUTPUT = "data/outputs/composite_candidates.json"
MIN_DURATION = 20.0
MAX_DURATION = 60.0


class CompositeEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")
    event_id: str
    event_type: str
    speaker: str | None = None
    delivery: str | None = None
    content: str
    original_start_timestamp: str
    original_end_timestamp: str
    original_start_seconds: float
    original_end_seconds: float
    relative_start_seconds: float
    relative_end_seconds: float


class CompositeSegment(BaseModel):
    model_config = ConfigDict(extra="forbid")
    segment_index: int
    episode_number: int
    beat_role: str
    anchor_event_ids: list[str]
    source_event_ids: list[str]
    source_start_timestamp: str
    source_end_timestamp: str
    source_start_seconds: float
    source_end_seconds: float
    duration_seconds: float
    relative_start_seconds: float
    relative_end_seconds: float
    narrative_reason: str
    spoiler_level: str
    events: list[CompositeEvent]


class CompositeCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    candidate_id: str
    candidate_family_id: str
    candidate_origin: Literal["semantic", "heuristic", "synthetic"]
    story_arc_id: str
    narrative_theme: str
    narrative_rationale: str
    segments: list[CompositeSegment] = Field(min_length=2)
    total_duration_seconds: float
    hook_segment_index: int
    hook_event_id: str
    potential_ending_segment_index: int
    potential_ending_event_id: str
    spoiler_warnings: list[str] = Field(default_factory=list)
    coherence_warnings: list[str] = Field(default_factory=list)
    source_event_ids: list[str]


def _timestamp(seconds: float) -> str:
    whole = int(seconds)
    hundredths = round((seconds - whole) * 100)
    if hundredths == 100:
        whole += 1
        hundredths = 0
    minutes, secs = divmod(whole, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{hundredths:02d}"


def _select_beats(opportunity: CompositePromoOpportunity, max_segments: int = 4):
    if opportunity.hook_beat_index >= len(opportunity.beats) or opportunity.ending_beat_index >= len(opportunity.beats):
        raise ValueError(f"{opportunity.story_arc_id}: hook or ending beat index is invalid")
    if opportunity.hook_beat_index == opportunity.ending_beat_index:
        raise ValueError(f"{opportunity.story_arc_id}: hook and ending must be distinct beats")
    required = {opportunity.hook_beat_index, opportunity.ending_beat_index}
    available = [index for index in range(len(opportunity.beats)) if index not in required]
    selected = sorted(required | set(available[:max(0, max_segments - len(required))]))
    beats = [opportunity.beats[index] for index in selected]
    if len({beat.episode_number for beat in beats}) < 2:
        raise ValueError(f"{opportunity.story_arc_id}: composite promo requires beats from at least two episodes")
    return selected, beats


def build_composite_candidates(
    episodes: list[ParsedEpisode],
    series_analysis: SeriesAnalysis | dict,
    *,
    total_duration_seconds: float = 30.0,
    candidate_origin: Literal["semantic", "heuristic", "synthetic"] = "semantic",
    max_segments: int = 4,
) -> list[CompositeCandidate]:
    """Turn LLM-selected event anchors into deterministic contiguous source segments."""
    if not MIN_DURATION <= total_duration_seconds <= MAX_DURATION:
        raise ValueError("Composite target duration must be between 20 and 60 seconds")
    analysis = series_analysis if isinstance(series_analysis, SeriesAnalysis) else SeriesAnalysis.model_validate(series_analysis)
    validate_series_references(analysis, episodes)
    episode_map = {episode.episode_number: episode for episode in episodes}
    result = []
    seen_arc_ids = set()
    seen_windows = set()
    series_arc_map = {arc.story_arc_id: arc for arc in analysis.story_arcs}
    for opportunity in analysis.composite_promo_opportunities:
        if opportunity.story_arc_id in seen_arc_ids:
            raise ValueError(f"Duplicate composite story arc ID: {opportunity.story_arc_id}")
        seen_arc_ids.add(opportunity.story_arc_id)
        source_indexes, beats = _select_beats(opportunity, max_segments=max_segments)
        segment_duration = total_duration_seconds / len(beats)
        if segment_duration < 4.0:
            raise ValueError(f"{opportunity.story_arc_id}: too many beats for a readable composite segment")
        segments = []
        cursor = 0.0
        all_source_ids = []
        source_windows = []
        for order, (source_index, beat) in enumerate(zip(source_indexes, beats)):
            episode = episode_map.get(beat.episode_number)
            if episode is None:
                raise ValueError(f"{opportunity.story_arc_id}: source episode {beat.episode_number} is unavailable")
            event_map = {event.event_id: event for event in episode.events}
            anchors = [event_map[event_id] for event_id in beat.anchor_event_ids if event_id in event_map]
            if len(anchors) != len(beat.anchor_event_ids):
                raise ValueError(f"{opportunity.story_arc_id}: invalid source event anchor in episode {beat.episode_number}")
            anchor_start = min(event.start_seconds for event in anchors)
            episode_end = max((event.end_seconds for event in episode.events), default=0)
            start = max(0.0, anchor_start - min(1.5, segment_duration * .15))
            end = start + segment_duration
            if end > episode_end:
                end = episode_end
                start = max(0.0, end - segment_duration)
            actual_duration = end - start
            if actual_duration < 3.0:
                raise ValueError(f"{opportunity.story_arc_id}: episode {beat.episode_number} has insufficient source duration")
            window_key = (beat.episode_number, round(start, 2), round(end, 2))
            if window_key in seen_windows:
                raise ValueError(f"{opportunity.story_arc_id}: duplicate source segment window")
            if any(ep_number == beat.episode_number and start < prior_end and end > prior_start
                   for ep_number, prior_start, prior_end in source_windows):
                raise ValueError(f"{opportunity.story_arc_id}: source windows overlap within episode {beat.episode_number}")
            seen_windows.add(window_key)
            overlapping = [event for event in episode.events
                           if event.start_seconds <= end and event.end_seconds >= start]
            overlapping.sort(key=lambda event: (event.start_seconds, event.source_line_start, event.end_seconds, event.event_id))
            ids = [event.event_id for event in overlapping]
            if not set(beat.anchor_event_ids).issubset(ids):
                raise ValueError(f"{opportunity.story_arc_id}: selected window does not include all anchor events")
            source_windows.append((beat.episode_number, start, end))
            all_source_ids.extend(ids)
            preview_events = [CompositeEvent(
                event_id=event.event_id, event_type=event.event_type, speaker=event.speaker,
                delivery=event.delivery, content=event.content,
                original_start_timestamp=event.start_timestamp, original_end_timestamp=event.end_timestamp,
                original_start_seconds=event.start_seconds, original_end_seconds=event.end_seconds,
                relative_start_seconds=round(cursor + max(0.0, event.start_seconds - start), 2),
                relative_end_seconds=round(cursor + min(actual_duration, event.end_seconds - start), 2),
            ) for event in overlapping]
            segments.append(CompositeSegment(
                segment_index=order, episode_number=beat.episode_number, beat_role=beat.beat_role,
                anchor_event_ids=beat.anchor_event_ids, source_event_ids=ids,
                source_start_timestamp=_timestamp(start), source_end_timestamp=_timestamp(end),
                source_start_seconds=round(start, 2), source_end_seconds=round(end, 2),
                duration_seconds=round(actual_duration, 2), relative_start_seconds=round(cursor, 2),
                relative_end_seconds=round(cursor + actual_duration, 2), narrative_reason=beat.reason,
                spoiler_level=beat.spoiler_level, events=preview_events,
            ))
            cursor += actual_duration
        if not (MIN_DURATION <= cursor <= MAX_DURATION):
            raise ValueError(f"{opportunity.story_arc_id}: constructed composite duration {cursor:.2f}s is outside 20–60 seconds")
        coherence_warnings = list(opportunity.uncertainty_warnings)
        arc = series_arc_map.get(opportunity.story_arc_id)
        if arc is None:
            coherence_warnings.append("Opportunity does not resolve to a declared series story arc; human coherence review is required.")
        else:
            arc_refs = {(ref.episode_number, ref.event_id) for development in arc.developments
                        for ref in development.source_references}
            for beat in opportunity.beats:
                if not any((beat.episode_number, event_id) in arc_refs for event_id in beat.anchor_event_ids):
                    coherence_warnings.append(
                        f"Episode {beat.episode_number} {beat.beat_role} anchor is not cross-referenced by the story arc developments."
                    )
        episode_order = [segment.episode_number for segment in segments]
        if episode_order != sorted(episode_order):
            coherence_warnings.append("Segment episode order is non-linear; verify chronology or intentional flashback structure.")
        if len(set(episode_order)) < 2:
            raise ValueError(f"{opportunity.story_arc_id}: composite candidate must use at least two episodes")
        hook_selected = source_indexes.index(opportunity.hook_beat_index)
        ending_selected = source_indexes.index(opportunity.ending_beat_index)
        hook_id = opportunity.beats[opportunity.hook_beat_index].anchor_event_ids[0]
        ending_id = opportunity.beats[opportunity.ending_beat_index].anchor_event_ids[-1]
        candidate_id = f"COMPOSITE_{opportunity.story_arc_id.upper()}_001"
        result.append(CompositeCandidate(
            candidate_id=candidate_id, candidate_family_id=f"COMPOSITE_FAMILY_{opportunity.story_arc_id.upper()}",
            candidate_origin=candidate_origin, story_arc_id=opportunity.story_arc_id,
            narrative_theme=opportunity.narrative_theme, narrative_rationale=opportunity.narrative_rationale,
            segments=segments, total_duration_seconds=round(cursor, 2), hook_segment_index=hook_selected,
            hook_event_id=hook_id, potential_ending_segment_index=ending_selected,
            potential_ending_event_id=ending_id,
            spoiler_warnings=opportunity.spoiler_warnings + [
                f"Episode {segment.episode_number}: {segment.spoiler_level} spoiler level."
                for segment in segments if segment.spoiler_level in {"moderate", "major", "unknown"}
            ], coherence_warnings=coherence_warnings,
            source_event_ids=all_source_ids,
        ))
    return result


def reconstruct_composite_preview(candidate: CompositeCandidate | dict) -> dict:
    """Return editor provenance plus one uninterrupted relative-time sequence."""
    candidate = candidate if isinstance(candidate, CompositeCandidate) else CompositeCandidate.model_validate(candidate)
    events = []
    for segment in candidate.segments:
        for event in segment.events:
            events.append({
                **event.model_dump(mode="json"),
                "segment_index": segment.segment_index,
                "episode_number": segment.episode_number,
                "segment_role": segment.beat_role,
            })
    events.sort(key=lambda row: (row["relative_start_seconds"], row["relative_end_seconds"], row["episode_number"], row["event_id"]))
    return {
        "candidate_id": candidate.candidate_id,
        "candidate_origin": candidate.candidate_origin,
        "total_duration_seconds": candidate.total_duration_seconds,
        "segments": [segment.model_dump(mode="json") for segment in candidate.segments],
        "events": events,
    }


def _atomic_json(path: str, value) -> None:
    target = __import__("pathlib").Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, target)


def main() -> None:
    parser = argparse.ArgumentParser(description="Construct multi-episode composite promo windows from series event anchors")
    parser.add_argument("--parsed", default=DEFAULT_PARSED)
    parser.add_argument("--series-analysis", default=DEFAULT_SERIES)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        parsed_data = json.loads(open(args.parsed, encoding="utf-8").read())
        episodes = [ParsedEpisode.model_validate(item) for item in parsed_data]
        if not episodes:
            raise ValueError("Parsed screenplay contains no episodes")
        series_path = __import__("pathlib").Path(args.series_analysis)
        analysis_data = json.loads(series_path.read_text(encoding="utf-8")) if series_path.exists() and series_path.stat().st_size else None
        opportunities = []
        if analysis_data is not None:
            analysis = SeriesAnalysis.model_validate(analysis_data)
            validate_series_references(analysis, episodes)
            opportunities = analysis.composite_promo_opportunities
        print(f"Episodes available: {len(episodes)}")
        print(f"Series-level composite opportunities: {len(opportunities)}")
        print(f"Configured candidates constructible: {len(opportunities)}")
        if args.dry_run:
            print("Dry-run: no Gemini client constructed; no output written.")
            return
        if analysis_data is None:
            raise ValueError("Series analysis is missing; run promoscout.analysis.series_analyzer first")
        candidates = build_composite_candidates(episodes, analysis, total_duration_seconds=args.duration)
        _atomic_json(args.output, {
            "schema_version": "composite-candidates-v1",
            "candidate_origin": "semantic",
            "synthetic": False,
            "candidates": [candidate.model_dump(mode="json") for candidate in candidates],
        })
        print(f"Generated {len(candidates)} composite candidate(s): {args.output}")
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
