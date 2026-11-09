"""Offline structural validation for one or more PromoScout pilot episodes.

This module checks artifact integrity and timing only. It deliberately does not
score promotional quality or decide whether a potential ending is a cliffhanger.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from promoscout.analysis.models import DiscoveredEvent, EpisodeAnalysis
from promoscout.generation.continuous import PromoCandidate, classify_hook_type
from promoscout.generation.preview import reconstruct_candidate_preview
from promoscout.schemas import ParsedEpisode

MIN_DURATION_SECONDS = 20.0
MAX_DURATION_SECONDS = 60.0
MAX_HOOK_LEAD_IN_SECONDS = 3.0
_EPSILON = 0.02


def _issue(severity: str, code: str, message: str, **details) -> dict:
    return {"severity": severity, "code": code, "message": message, **details}


def _as_episode_map(value: Any, value_field: str | None = None) -> dict[str, Any]:
    """Accept the pipeline's episode-number map and convenient single-episode data."""
    if isinstance(value, dict):
        if value_field and value_field in value:
            return {str(k): v for k, v in value[value_field].items()} if isinstance(value[value_field], dict) else {}
        if "episode_number" in value:
            return {str(value["episode_number"]): value}
        return {str(k): v for k, v in value.items()}
    if isinstance(value, list):
        result = {}
        for row in value:
            if isinstance(row, dict) and "episode_number" in row:
                result[str(row["episode_number"])] = row
        return result
    return {}


def _candidate_rows(value: Any) -> list[dict]:
    if isinstance(value, list):
        return [row for row in value if isinstance(row, dict)]
    if isinstance(value, dict):
        rows = value.get("candidates", [])
        return [row for row in rows if isinstance(row, dict)] if isinstance(rows, list) else []
    return []


def _preview_rows(raw: dict) -> list[dict] | None:
    preview = raw.get("preview_events", raw.get("preview"))
    if preview is None:
        return None
    if isinstance(preview, dict):
        preview = preview.get("events")
    return preview if isinstance(preview, list) else []


def _event_lookup(episodes: list[ParsedEpisode]):
    event_by_id: dict[str, tuple[ParsedEpisode, Any]] = {}
    duplicates: list[dict] = []
    for episode in episodes:
        for event in episode.events:
            if event.event_id in event_by_id:
                previous_episode, _ = event_by_id[event.event_id]
                duplicates.append({"event_id": event.event_id, "episodes": [previous_episode.episode_number, episode.episode_number]})
            else:
                event_by_id[event.event_id] = (episode, event)
    return event_by_id, duplicates


def _validate_preview(preview_rows: list[dict], candidate: PromoCandidate,
                      episode: ParsedEpisode, expected_ids: list[str]) -> list[dict]:
    issues = []
    expected_positions = {event_id: index for index, event_id in enumerate(expected_ids)}
    seen_ids = []
    lookup = {event.event_id: event for event in episode.events}
    for index, row in enumerate(preview_rows):
        if not isinstance(row, dict) or not isinstance(row.get("event_id"), str):
            issues.append(_issue("error", "invalid_preview_event", f"Preview row {index + 1} has no event_id."))
            continue
        event_id = row["event_id"]
        seen_ids.append(event_id)
        source = lookup.get(event_id)
        if source is None:
            issues.append(_issue("error", "invalid_preview_event_id", f"Preview references unknown event {event_id}.", event_id=event_id))
            continue
        if row.get("start_timestamp") != source.start_timestamp or row.get("end_timestamp") != source.end_timestamp:
            issues.append(_issue("error", "preview_timestamp_mismatch", f"Preview timestamps do not match source event {event_id}.", event_id=event_id))
        expected_relative_start = round(source.start_seconds - candidate.start_seconds, 2)
        expected_relative_end = round(source.end_seconds - candidate.start_seconds, 2)
        if "relative_start_seconds" not in row or "relative_end_seconds" not in row:
            issues.append(_issue("error", "missing_preview_relative_times", f"Preview is missing relative timestamps for event {event_id}.", event_id=event_id))
        for field, expected in (("relative_start_seconds", expected_relative_start), ("relative_end_seconds", expected_relative_end)):
            if field in row:
                try:
                    valid = abs(float(row[field]) - expected) <= _EPSILON
                except (TypeError, ValueError):
                    valid = False
                if not valid:
                    issues.append(_issue("error", "preview_relative_time_mismatch", f"{field} does not match source event {event_id}.", event_id=event_id))
    missing = [event_id for event_id in expected_ids if event_id not in seen_ids]
    if missing:
        issues.append(_issue("error", "missing_preview_events", "Preview omits source events overlapping this candidate.", event_ids=missing))
    positions = [expected_positions[event_id] for event_id in seen_ids if event_id in expected_positions]
    if positions != sorted(positions):
        issues.append(_issue("error", "preview_out_of_order", "Preview events are not in chronological source order."))
    duplicates = sorted({event_id for event_id in seen_ids if seen_ids.count(event_id) > 1})
    if duplicates:
        issues.append(_issue("error", "duplicate_preview_events", "Preview contains duplicate event IDs.", event_ids=duplicates))
    return issues


def validate_pilot_artifacts(
    parsed_episodes: list[ParsedEpisode] | list[dict],
    episode_analysis: Any,
    event_discoveries: Any,
    semantic_candidates: Any,
    *,
    episode_number: int | None = None,
    input_issues: list[dict] | None = None,
) -> dict:
    """Return a machine-readable technical validation report without API calls."""
    episodes = [e if isinstance(e, ParsedEpisode) else ParsedEpisode.model_validate(e) for e in parsed_episodes]
    all_episodes = episodes
    if episode_number is not None:
        episodes = [episode for episode in all_episodes if episode.episode_number == episode_number]
    report_issues = list(input_issues or [])
    event_by_id, duplicate_source_ids = _event_lookup(all_episodes)
    for duplicate in duplicate_source_ids:
        report_issues.append(_issue("error", "duplicate_source_event_id", "Original screenplay event ID is not unique.", **duplicate))
    analysis_map = _as_episode_map(episode_analysis, "episode_analysis")
    discovery_map = _as_episode_map(event_discoveries, "event_discoveries")
    candidate_rows = _candidate_rows(semantic_candidates)
    raw_candidate_collection = semantic_candidates.get("candidates", []) if isinstance(semantic_candidates, dict) else semantic_candidates
    if isinstance(raw_candidate_collection, list):
        for index, row in enumerate(raw_candidate_collection):
            if not isinstance(row, dict):
                report_issues.append(_issue("error", "invalid_candidate_row", f"Candidate row {index + 1} is not a JSON object."))
    elif semantic_candidates:
        report_issues.append(_issue("error", "invalid_candidate_collection", "Semantic candidates must be a list or an object containing a candidates list."))
    if episode_number is not None and not episodes:
        report_issues.append(_issue("error", "episode_not_found", f"Episode {episode_number} does not exist in parsed screenplay data.", episode_number=episode_number))
    selected_numbers = {episode.episode_number for episode in episodes}
    valid_discoveries_by_episode: dict[int, dict[tuple[str, str], DiscoveredEvent]] = defaultdict(dict)
    resolved_discoveries = []

    for episode in episodes:
        key = str(episode.episode_number)
        if key not in analysis_map:
            report_issues.append(_issue("error", "missing_episode_analysis", f"Episode {episode.episode_number} has no semantic analysis result.", episode_number=episode.episode_number))
        else:
            raw_analysis = analysis_map[key]
            try:
                analysis = EpisodeAnalysis.model_validate(raw_analysis)
                if analysis.episode_number != episode.episode_number:
                    report_issues.append(_issue("error", "analysis_episode_mismatch", f"Analysis payload says episode {analysis.episode_number} but is stored under episode {episode.episode_number}.", episode_number=episode.episode_number))
            except (ValidationError, TypeError) as exc:
                report_issues.append(_issue("error", "invalid_episode_analysis", f"Episode analysis is missing required fields or invalid: {exc}", episode_number=episode.episode_number))

        if key not in discovery_map:
            report_issues.append(_issue("error", "missing_event_discoveries", f"Episode {episode.episode_number} has no event discovery result.", episode_number=episode.episode_number))
            continue
        rows = discovery_map[key]
        if not isinstance(rows, list):
            report_issues.append(_issue("error", "invalid_discovery_collection", "Episode discoveries must be a list.", episode_number=episode.episode_number))
            continue
        seen = set()
        for index, raw in enumerate(rows):
            try:
                discovery = DiscoveredEvent.model_validate(raw)
            except (ValidationError, TypeError) as exc:
                report_issues.append(_issue("error", "invalid_discovery_schema", f"Discovery {index + 1} has missing or invalid fields: {exc}", episode_number=episode.episode_number))
                continue
            dupe_key = (discovery.event_id, discovery.role)
            if dupe_key in seen:
                report_issues.append(_issue("warning", "duplicate_discovery", f"Duplicate {discovery.role} reference to {discovery.event_id} was reported; pipeline deduplication keeps the first.", episode_number=episode.episode_number, event_id=discovery.event_id))
                continue
            seen.add(dupe_key)
            source = event_by_id.get(discovery.event_id)
            if source is None:
                report_issues.append(_issue("error", "invalid_event_reference", f"Discovery references unknown source event ID {discovery.event_id}.", episode_number=episode.episode_number, event_id=discovery.event_id))
                continue
            source_episode, source_event = source
            if source_episode.episode_number != episode.episode_number:
                report_issues.append(_issue("error", "wrong_episode_reference", f"Discovery event {discovery.event_id} belongs to episode {source_episode.episode_number}, not {episode.episode_number}.", episode_number=episode.episode_number, event_id=discovery.event_id))
                continue
            valid_discoveries_by_episode[episode.episode_number][dupe_key] = discovery
            resolved_discoveries.append({
                "episode_number": episode.episode_number,
                **discovery.model_dump(mode="json"),
                "resolved_event": {
                    "event_type": source_event.event_type,
                    "label": source_event.label,
                    "content": source_event.content,
                    "start_timestamp": source_event.start_timestamp,
                    "end_timestamp": source_event.end_timestamp,
                },
            })

    candidate_reports = []
    candidate_ids: dict[str, int] = {}
    windows: dict[tuple, str] = {}
    family_to_hooks: dict[tuple[int, str], set[str]] = defaultdict(set)
    hook_to_family: dict[tuple[int, str], set[str]] = defaultdict(set)
    candidate_episode_map: dict[int, list[dict]] = defaultdict(list)

    for raw in candidate_rows:
        raw_episode = raw.get("episode_number")
        if episode_number is not None and raw_episode != episode_number:
            continue
        if not isinstance(raw_episode, int) or raw_episode not in selected_numbers:
            report_issues.append(_issue("error", "candidate_episode_not_found", f"Candidate references unavailable episode {raw_episode}.", candidate_id=raw.get("candidate_id")))
            continue
        episode = next(e for e in episodes if e.episode_number == raw_episode)
        row_issues: list[dict] = []
        raw_preview = _preview_rows(raw)
        candidate_payload = {k: v for k, v in raw.items() if k not in {"preview", "preview_events"}}
        try:
            candidate = PromoCandidate.model_validate(candidate_payload)
        except ValidationError as exc:
            candidate_reports.append({
                "candidate_id": raw.get("candidate_id", "(missing)"),
                "episode_number": raw_episode,
                "structural_status": "fail",
                "errors": [_issue("error", "invalid_candidate_schema", str(exc))],
                "warnings": [],
                "human_review": _pending_review(),
            })
            continue

        cid = candidate.candidate_id
        candidate_ids[cid] = candidate_ids.get(cid, 0) + 1
        by_id = {event.event_id: event for event in episode.events}
        source_event_ids = set(by_id)
        for label, event_id in (("start", candidate.start_event_id), ("hook", candidate.hook_event_id),
                                ("ending", candidate.ending_event_id), ("end", candidate.end_event_id)):
            source = event_by_id.get(event_id)
            if source is None:
                row_issues.append(_issue("error", "invalid_event_reference", f"{label.title()} references unknown source event ID {event_id}.", event_id=event_id))
            elif source[0].episode_number != candidate.episode_number:
                row_issues.append(_issue("error", "wrong_episode_reference", f"{label.title()} event {event_id} belongs to episode {source[0].episode_number}.", event_id=event_id))
        for event_id in candidate.included_event_ids:
            source = event_by_id.get(event_id)
            if source is None:
                row_issues.append(_issue("error", "invalid_included_event_id", f"Included event ID {event_id} does not exist.", event_id=event_id))
            elif source[0].episode_number != candidate.episode_number:
                row_issues.append(_issue("error", "wrong_episode_reference", f"Included event {event_id} belongs to episode {source[0].episode_number}.", event_id=event_id))

        start_source = by_id.get(candidate.start_event_id)
        hook_source = by_id.get(candidate.hook_event_id)
        ending_source = by_id.get(candidate.ending_event_id)
        end_source = by_id.get(candidate.end_event_id)
        if start_source:
            if abs(candidate.start_seconds - start_source.start_seconds) > _EPSILON or candidate.start_timestamp != start_source.start_timestamp:
                row_issues.append(_issue("error", "start_metadata_mismatch", "Candidate start metadata does not match its original start event."))
        if hook_source:
            if (candidate.hook_start_timestamp != hook_source.start_timestamp or
                candidate.hook_end_timestamp != hook_source.end_timestamp or
                abs(candidate.hook_start_seconds - hook_source.start_seconds) > _EPSILON or
                abs(candidate.hook_end_seconds - hook_source.end_seconds) > _EPSILON):
                row_issues.append(_issue("error", "hook_timestamp_mismatch", "Hook timestamps do not match the original event."))
            if candidate.hook_text != hook_source.content:
                row_issues.append(_issue("error", "hook_text_mismatch", "Candidate hook text does not match the original event."))
            if candidate.hook_type != classify_hook_type(hook_source.event_type):
                row_issues.append(_issue("error", "hook_type_mismatch", "Candidate hook type does not match the source event type."))
            expected_hook_offset = candidate.hook_start_seconds - candidate.start_seconds
            if abs(candidate.hook_offset_seconds - expected_hook_offset) > _EPSILON:
                row_issues.append(_issue("error", "hook_offset_mismatch", "Hook offset does not match source timestamps."))
            if expected_hook_offset < -_EPSILON or expected_hook_offset > MAX_HOOK_LEAD_IN_SECONDS + _EPSILON:
                row_issues.append(_issue("error", "hook_timing_violation", f"Hook begins {expected_hook_offset:.2f}s after clip start; maximum is {MAX_HOOK_LEAD_IN_SECONDS:.0f}s."))
            if (candidate.hook_event_id, "hook") not in valid_discoveries_by_episode[candidate.episode_number]:
                row_issues.append(_issue("error", "hook_not_in_discoveries", "Candidate hook is not present as a validated hook discovery."))
        if ending_source:
            if candidate.ending_text != ending_source.content:
                row_issues.append(_issue("error", "ending_text_mismatch", "Potential ending text does not match the original event."))
            if candidate.end_timestamp != ending_source.end_timestamp or abs(candidate.end_seconds - ending_source.end_seconds) > _EPSILON:
                row_issues.append(_issue("error", "ending_timestamp_mismatch", "Candidate end metadata does not match its original ending event."))
            if (candidate.ending_event_id, "potential_ending") not in valid_discoveries_by_episode[candidate.episode_number]:
                row_issues.append(_issue("error", "ending_not_in_discoveries", "Candidate ending is not present as a validated potential-ending discovery."))
        if end_source and ending_source and candidate.end_event_id != candidate.ending_event_id:
            row_issues.append(_issue("error", "end_event_mismatch", "Candidate end_event_id and ending_event_id refer to different source events."))

        calculated_duration = candidate.end_seconds - candidate.start_seconds
        if abs(calculated_duration - candidate.duration_seconds) > _EPSILON:
            row_issues.append(_issue("error", "duration_metadata_mismatch", "Candidate duration does not equal end minus start."))
        if calculated_duration < MIN_DURATION_SECONDS - _EPSILON or calculated_duration > MAX_DURATION_SECONDS + _EPSILON:
            row_issues.append(_issue("error", "invalid_duration", f"Duration {calculated_duration:.2f}s is outside {MIN_DURATION_SECONDS:.0f}–{MAX_DURATION_SECONDS:.0f}s."))

        expected_events = [event for event in episode.events
                           if event.start_seconds <= candidate.end_seconds and event.end_seconds >= candidate.start_seconds]
        expected_events.sort(key=lambda event: (event.start_seconds, event.source_line_start, event.end_seconds, event.event_id))
        expected_ids = [event.event_id for event in expected_events]
        extra_included = [event_id for event_id in candidate.included_event_ids if event_id in source_event_ids and event_id not in expected_ids]
        if extra_included:
            row_issues.append(_issue("error", "non_overlapping_included_events", "Candidate includes events outside its clip window.", event_ids=extra_included))
        # Validate the candidate's own sequence using original screenplay timing.
        resolved_sequence = [by_id[event_id] for event_id in candidate.included_event_ids if event_id in by_id]
        sorted_sequence = sorted(resolved_sequence, key=lambda event: (event.start_seconds, event.source_line_start, event.end_seconds, event.event_id))
        if [event.event_id for event in resolved_sequence] != [event.event_id for event in sorted_sequence]:
            row_issues.append(_issue("error", "candidate_events_out_of_order", "Included event IDs are not in chronological source order."))

        if raw_preview is not None:
            row_issues.extend(_validate_preview(raw_preview, candidate, episode, expected_ids))
        else:
            # Reconstruct via the production utility and independently compare it to
            # every source event overlapping the clip. No original output is changed.
            try:
                reconstructed = reconstruct_candidate_preview(episode, candidate)
                row_issues.extend(_validate_preview(reconstructed["events"], candidate, episode, expected_ids))
            except ValueError as exc:
                row_issues.append(_issue("error", "preview_reconstruction_failed", str(exc)))

        window = (candidate.episode_number, round(candidate.start_seconds, 2), round(candidate.end_seconds, 2))
        if window in windows:
            row_issues.append(_issue("warning", "duplicate_window", f"Clip window duplicates candidate {windows[window]}.", duplicate_of=windows[window]))
        else:
            windows[window] = candidate.candidate_id

        family_to_hooks[(candidate.episode_number, candidate.candidate_family_id)].add(candidate.hook_event_id)
        hook_to_family[(candidate.episode_number, candidate.hook_event_id)].add(candidate.candidate_family_id)
        candidate_episode_map[candidate.episode_number].append(raw)
        error_count = sum(issue["severity"] == "error" for issue in row_issues)
        warning_count = sum(issue["severity"] == "warning" for issue in row_issues)
        candidate_reports.append({
            "candidate_id": candidate.candidate_id,
            "episode_number": candidate.episode_number,
            "start_timestamp": candidate.start_timestamp,
            "end_timestamp": candidate.end_timestamp,
            "duration_seconds": candidate.duration_seconds,
            "hook_type": candidate.hook_type,
            "hook_text": candidate.hook_text,
            "potential_ending_text": candidate.ending_text,
            "hook_event_id": candidate.hook_event_id,
            "candidate_family_id": candidate.candidate_family_id,
            "structural_status": "pass" if error_count == 0 else "fail",
            "errors": [i for i in row_issues if i["severity"] == "error"],
            "warnings": [i for i in row_issues if i["severity"] == "warning"],
            "human_review": _pending_review(),
        })

    for (ep_num, family), hook_ids in family_to_hooks.items():
        if len(hook_ids) > 1:
            report_issues.append(_issue("error", "family_contains_multiple_hooks", f"Candidate family {family} contains multiple hook event IDs.", episode_number=ep_num, hook_event_ids=sorted(hook_ids)))
            for row in candidate_reports:
                if row.get("episode_number") == ep_num and row.get("candidate_family_id") == family:
                    row["structural_status"] = "fail"
    for (ep_num, hook_id), families in hook_to_family.items():
        if len(families) > 1:
            report_issues.append(_issue("error", "hook_split_across_families", f"Hook {hook_id} is assigned to multiple candidate families.", episode_number=ep_num, family_ids=sorted(families)))
            for row in candidate_reports:
                if row.get("episode_number") == ep_num and row.get("hook_event_id") == hook_id:
                    row["structural_status"] = "fail"

    # Deterministic family numbering follows chronological first occurrence of hooks.
    for episode in episodes:
        hook_ids = {row.get("hook_event_id") for row in candidate_episode_map[episode.episode_number]}
        hook_events = [event for event in episode.events if event.event_id in hook_ids]
        hook_events.sort(key=lambda event: (event.start_seconds, event.source_line_start, event.event_id))
        for index, event in enumerate(hook_events, start=1):
            expected_family = f"EP{episode.episode_number:03d}_FAMILY_{index:03d}"
            actual = hook_to_family.get((episode.episode_number, event.event_id), set())
            if actual and actual != {expected_family}:
                report_issues.append(_issue("error", "non_deterministic_family_id", f"Hook {event.event_id} should use family ID {expected_family}.", episode_number=episode.episode_number, actual_family_ids=sorted(actual)))
                for row in candidate_reports:
                    if row.get("episode_number") == episode.episode_number and row.get("hook_event_id") == event.event_id:
                        row["structural_status"] = "fail"

    duplicate_candidate_ids = sorted(candidate_id for candidate_id, count in candidate_ids.items() if count > 1)
    for candidate_id in duplicate_candidate_ids:
        report_issues.append(_issue("error", "duplicate_candidate_id", f"Candidate ID {candidate_id} appears more than once.", candidate_id=candidate_id))
        for row in candidate_reports:
            if row.get("candidate_id") == candidate_id:
                row["structural_status"] = "fail"
    error_total = sum(issue["severity"] == "error" for issue in report_issues) + sum(len(row.get("errors", [])) for row in candidate_reports)
    warning_total = sum(issue["severity"] == "warning" for issue in report_issues) + sum(len(row.get("warnings", [])) for row in candidate_reports)
    discovered_hooks = sum(1 for ep in episodes for d in valid_discoveries_by_episode[ep.episode_number].values() if d.role == "hook")
    potential_endings = sum(1 for ep in episodes for d in valid_discoveries_by_episode[ep.episode_number].values() if d.role == "potential_ending")
    passing = sum(row.get("structural_status") == "pass" for row in candidate_reports)
    return {
        "report_version": "pilot-validation-v1",
        "episode_filter": episode_number,
        "processing_summary": {
            "episode_numbers": [episode.episode_number for episode in episodes],
            "episode_number": episode_number if episode_number is not None else None,
            "original_event_count": sum(len(episode.events) for episode in episodes),
            "episode_analysis_exists": all(str(episode.episode_number) in analysis_map for episode in episodes),
            "discovered_hook_count": discovered_hooks,
            "potential_ending_count": potential_endings,
            "generated_candidate_count": len(candidate_reports),
            "structurally_valid_candidate_count": passing,
        },
        "validation_summary": {
            "total_errors": error_total,
            "total_warnings": warning_total,
            "invalid_event_references": sum(issue["code"] in {"invalid_event_reference", "invalid_included_event_id", "invalid_preview_event_id", "wrong_episode_reference"} for issue in report_issues) + sum(issue["code"] in {"invalid_event_reference", "invalid_included_event_id", "invalid_preview_event_id", "wrong_episode_reference"} for row in candidate_reports for issue in row.get("errors", [])),
            "invalid_durations": sum(issue["code"] == "invalid_duration" for row in candidate_reports for issue in row.get("errors", [])),
            "hook_timing_violations": sum(issue["code"] in {"hook_timing_violation", "hook_offset_mismatch"} for row in candidate_reports for issue in row.get("errors", [])),
            "missing_preview_events": sum(issue["code"] == "missing_preview_events" for row in candidate_reports for issue in row.get("errors", [])),
            "duplicate_windows": sum(issue["code"] == "duplicate_window" for row in candidate_reports for issue in row.get("warnings", [])),
            "chronological_order_issues": sum(issue["code"] in {"preview_out_of_order", "candidate_events_out_of_order"} for row in candidate_reports for issue in row.get("errors", [])),
        },
        "issues": report_issues,
        "resolved_discoveries": resolved_discoveries,
        "candidates": candidate_reports,
        "quality_evaluation": {
            "status": "not_evaluated",
            "dimensions": [
                {"name": "opening_hook_attracts_attention", "status": "pending_human_review"},
                {"name": "standalone_clarity_for_new_viewer", "status": "pending_human_review"},
                {"name": "central_conflict_is_clear", "status": "pending_human_review"},
                {"name": "ending_is_unresolved_and_compelling", "status": "pending_human_or_semantic_evaluation"},
                {"name": "viewer_interest_in_full_episode", "status": "pending_human_review"},
            ],
            "note": "Structural validation does not establish narrative quality or confirm a cliffhanger.",
        },
    }


def _pending_review() -> dict:
    return {
        "does_opening_hook_attract_attention_immediately": "pending",
        "can_unfamiliar_viewer_understand_situation": "pending",
        "is_central_conflict_clear": "pending",
        "does_ending_create_unanswered_question_or_unresolved_situation": "pending",
        "would_viewer_want_to_watch_full_episode": "pending",
    }


def validate_composite_candidates(parsed_episodes: list[ParsedEpisode | dict], candidates: Any) -> dict:
    """Validate composite provenance and relative timing without mixing episode clocks."""
    from promoscout.generation.composite import CompositeCandidate

    episodes = [row if isinstance(row, ParsedEpisode) else ParsedEpisode.model_validate(row)
                for row in parsed_episodes]
    episode_map = {episode.episode_number: episode for episode in episodes}
    rows = _candidate_rows(candidates)
    reports, seen_ids = [], set()
    for index, raw in enumerate(rows):
        errors, warnings = [], []
        try:
            candidate = CompositeCandidate.model_validate(raw)
        except ValidationError as exc:
            reports.append({"candidate_id": raw.get("candidate_id", f"row-{index + 1}"),
                            "structural_status": "fail", "errors": [str(exc)], "warnings": []})
            continue
        if candidate.candidate_id in seen_ids:
            errors.append("Candidate ID is duplicated.")
        seen_ids.add(candidate.candidate_id)
        distinct_episodes = {segment.episode_number for segment in candidate.segments}
        if len(distinct_episodes) < 2:
            errors.append("Composite must contain segments from at least two distinct episodes.")
        if not MIN_DURATION_SECONDS <= candidate.total_duration_seconds <= MAX_DURATION_SECONDS:
            errors.append("Total composite duration is outside 20–60 seconds.")
        segment_sum = sum(segment.duration_seconds for segment in candidate.segments)
        if abs(segment_sum - candidate.total_duration_seconds) > _EPSILON:
            errors.append("Total duration does not equal the sum of segment durations.")
        cursor, flat_refs, seen_windows, windows_by_episode = 0.0, [], set(), defaultdict(list)
        for segment in candidate.segments:
            episode = episode_map.get(segment.episode_number)
            if episode is None:
                errors.append(f"Segment references unavailable episode {segment.episode_number}.")
                continue
            if segment.segment_index < 0 or segment.segment_index >= len(candidate.segments):
                errors.append("Segment index is outside the candidate segment list.")
            if abs(segment.relative_start_seconds - cursor) > _EPSILON:
                errors.append(f"Episode {segment.episode_number} segment relative start is inconsistent.")
            if abs(segment.relative_end_seconds - (cursor + segment.duration_seconds)) > _EPSILON:
                errors.append(f"Episode {segment.episode_number} segment relative end is inconsistent.")
            if abs((segment.source_end_seconds - segment.source_start_seconds) - segment.duration_seconds) > _EPSILON:
                errors.append(f"Episode {segment.episode_number} segment duration does not match its source window.")
            window_key = (segment.episode_number, round(segment.source_start_seconds, 2), round(segment.source_end_seconds, 2))
            if window_key in seen_windows:
                errors.append("Duplicate or overlapping source segment window was included more than once.")
            seen_windows.add(window_key)
            for prior_start, prior_end in windows_by_episode[segment.episode_number]:
                if segment.source_start_seconds < prior_end - _EPSILON and segment.source_end_seconds > prior_start + _EPSILON:
                    errors.append(f"Episode {segment.episode_number} source segment overlaps another segment without an explicit justification.")
                    break
            windows_by_episode[segment.episode_number].append((segment.source_start_seconds, segment.source_end_seconds))
            source_by_id = {event.event_id: event for event in episode.events}
            expected = [event for event in episode.events
                        if event.start_seconds <= segment.source_end_seconds
                        and event.end_seconds >= segment.source_start_seconds]
            expected.sort(key=lambda event: (event.start_seconds, event.source_line_start, event.end_seconds, event.event_id))
            expected_ids = [event.event_id for event in expected]
            if segment.source_event_ids != expected_ids:
                errors.append(f"Episode {segment.episode_number} segment events do not match all source events overlapping its window.")
            if not set(segment.anchor_event_ids).issubset(source_by_id):
                errors.append(f"Episode {segment.episode_number} has invalid anchor event IDs.")
            if not set(segment.anchor_event_ids).issubset(expected_ids):
                errors.append(f"Episode {segment.episode_number} source window omits an anchor event.")
            if segment.events:
                preview_ids = [event.event_id for event in segment.events]
                if preview_ids != expected_ids:
                    errors.append(f"Episode {segment.episode_number} preview events are missing or out of order.")
                for event_row in segment.events:
                    source_event = source_by_id.get(event_row.event_id)
                    if source_event is None:
                        errors.append(f"Unknown preview event {event_row.event_id} in episode {segment.episode_number}.")
                        continue
                    if (event_row.original_start_timestamp != source_event.start_timestamp
                            or event_row.original_end_timestamp != source_event.end_timestamp
                            or event_row.content != source_event.content):
                        errors.append(f"Preview provenance differs from original event {event_row.event_id}.")
                    expected_rel_start = cursor + max(0.0, source_event.start_seconds - segment.source_start_seconds)
                    expected_rel_end = cursor + min(segment.duration_seconds, source_event.end_seconds - segment.source_start_seconds)
                    if (abs(event_row.relative_start_seconds - expected_rel_start) > _EPSILON
                            or abs(event_row.relative_end_seconds - expected_rel_end) > _EPSILON):
                        errors.append(f"Relative preview timestamps are inconsistent for event {event_row.event_id}.")
            flat_refs.extend(segment.source_event_ids)
            cursor += segment.duration_seconds
        if candidate.source_event_ids != flat_refs:
            errors.append("Candidate source event list does not match its segment provenance.")
        if not (0 <= candidate.hook_segment_index < len(candidate.segments)):
            errors.append("Hook segment index is invalid.")
        elif candidate.hook_event_id not in candidate.segments[candidate.hook_segment_index].anchor_event_ids:
            errors.append("Hook event is not an anchor in the designated hook segment.")
        if not (0 <= candidate.potential_ending_segment_index < len(candidate.segments)):
            errors.append("Potential ending segment index is invalid.")
        elif candidate.potential_ending_event_id not in candidate.segments[candidate.potential_ending_segment_index].anchor_event_ids:
            errors.append("Potential ending event is not an anchor in the designated ending segment.")
        warnings.extend(candidate.coherence_warnings)
        warnings.extend(candidate.spoiler_warnings)
        reports.append({"candidate_id": candidate.candidate_id,
                        "episodes": sorted(distinct_episodes),
                        "duration_seconds": candidate.total_duration_seconds,
                        "structural_status": "pass" if not errors else "fail",
                        "errors": errors, "warnings": warnings})
    return {
        "status": "available",
        "candidate_count": len(rows),
        "passing_count": sum(item["structural_status"] == "pass" for item in reports),
        "failing_count": sum(item["structural_status"] != "pass" for item in reports),
        "candidates": reports,
    }


def render_markdown_report(report: dict) -> str:
    summary = report["processing_summary"]
    validation = report["validation_summary"]
    lines = [
        "# PromoScout Pilot Validation Report", "",
        "## A. Processing summary", "",
        f"- Episode(s): {', '.join(map(str, summary['episode_numbers'])) or 'none'}",
        f"- Original events: {summary['original_event_count']}",
        f"- Episode analysis exists: {'yes' if summary['episode_analysis_exists'] else 'no'}",
        f"- Discovered hooks: {summary['discovered_hook_count']}",
        f"- Potential endings: {summary['potential_ending_count']}",
        f"- Generated candidates: {summary['generated_candidate_count']}",
        f"- Candidates passing structural validation: {summary['structurally_valid_candidate_count']}", "",
        "## B. Validation results", "",
        f"- Errors: {validation['total_errors']}", f"- Warnings: {validation['total_warnings']}",
        f"- Invalid event references: {validation['invalid_event_references']}",
        f"- Invalid durations: {validation['invalid_durations']}",
        f"- Hook timing violations: {validation['hook_timing_violations']}",
        f"- Missing preview events: {validation['missing_preview_events']}",
        f"- Duplicate windows: {validation['duplicate_windows']}",
        f"- Chronological ordering issues: {validation['chronological_order_issues']}", "",
    ]
    if report["issues"]:
        lines.extend(["### Artifact-level issues", ""])
        lines.extend(f"- **{issue['severity'].upper()} `{issue['code']}`** — {issue['message']}" for issue in report["issues"])
        lines.append("")
    lines.extend(["## C. Candidate-by-candidate report", ""])
    if not report["candidates"]:
        lines.extend(["No semantic candidates were supplied for validation.", ""])
    for candidate in report["candidates"]:
        lines.extend([
            f"### {candidate['candidate_id']}", "",
            f"- Episode: {candidate.get('episode_number', 'unknown')}",
            f"- Window: {candidate.get('start_timestamp', 'unknown')}–{candidate.get('end_timestamp', 'unknown')}",
            f"- Duration: {candidate.get('duration_seconds', 'unknown')} seconds",
            f"- Hook type: {candidate.get('hook_type', 'unknown')}",
            f"- Hook: {_md(candidate.get('hook_text', ''))}",
            f"- Potential ending: {_md(candidate.get('potential_ending_text', ''))}",
            f"- Family: {candidate.get('candidate_family_id', 'unknown')}",
            f"- Structural status: **{candidate.get('structural_status', 'fail').upper()}**",
        ])
        for issue in candidate.get("errors", []):
            lines.append(f"- ERROR `{issue['code']}`: {issue['message']}")
        for issue in candidate.get("warnings", []):
            lines.append(f"- WARNING `{issue['code']}`: {issue['message']}")
        lines.extend(["", "#### Human review checklist", ""])
        lines.extend(f"- [ ] {question.replace('_', ' ')} — **{status}**" for question, status in candidate.get("human_review", {}).items())
        lines.append("")
    lines.extend([
        "## D. Human review and quality limits", "",
        "Every checklist item remains pending. Structural validity does not prove that a hook is compelling, that the clip is clear to a new viewer, or that the ending is a genuine cliffhanger. Those judgments require human review or a separate semantic evaluation.", "",
    ])
    composite = report.get("composite_validation")
    if isinstance(composite, dict):
        lines.extend(["", "## Composite candidate validation", ""])
        if composite.get("status") != "available":
            lines.append(composite.get("message", "Composite candidates are not available."))
        else:
            lines.append(f"{composite['passing_count']} of {composite['candidate_count']} composite candidate(s) pass structural validation.")
            for candidate in composite["candidates"]:
                lines.append(f"- **{candidate['candidate_id']}**: {candidate['structural_status']} · {candidate['duration_seconds']}s · episodes {candidate['episodes']}")
                lines.extend(f"  - ERROR: {error}" for error in candidate["errors"])
                lines.extend(f"  - WARNING: {warning}" for warning in candidate["warnings"])
    return "\n".join(lines)


def _md(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ").strip()


def _read_json(path: Path, name: str, *, required: bool = False):
    if not path.exists() or not path.stat().st_size:
        message = f"{name} file is missing or empty: {path}"
        if required:
            raise ValueError(message)
        return {}, _issue("error", f"missing_{name}_file", message, path=str(path))
    try:
        return json.loads(path.read_text(encoding="utf-8")), None
    except json.JSONDecodeError as exc:
        return {}, _issue("error", f"invalid_{name}_file_json", f"{name} file is not valid JSON: {path}", path=str(path), detail=str(exc))


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate PromoScout pilot artifacts against original screenplay events")
    parser.add_argument("--episode-number", type=int)
    parser.add_argument("--parsed", default="data/processed/parsed_all_episodes.json")
    parser.add_argument("--analysis", default="data/processed/episode_analysis.json")
    parser.add_argument("--discoveries", default="data/processed/event_discoveries.json")
    parser.add_argument("--candidates", default="data/outputs/semantic_candidates.json")
    parser.add_argument("--composite-candidates", default="data/outputs/composite_candidates.json")
    parser.add_argument("--json-output", default="data/outputs/pilot_validation.json")
    parser.add_argument("--markdown-output", default="data/outputs/pilot_validation.md")
    args = parser.parse_args()
    try:
        parsed, _ = _read_json(Path(args.parsed), "parsed", required=True)
        episodes = [ParsedEpisode.model_validate(row) for row in parsed]
    except (ValueError, ValidationError, TypeError) as exc:
        parser.error(f"Unable to load parsed screenplay JSON: {exc}")
    analysis, analysis_issue = _read_json(Path(args.analysis), "episode_analysis")
    discoveries, discovery_issue = _read_json(Path(args.discoveries), "event_discoveries")
    candidates, candidate_issue = _read_json(Path(args.candidates), "semantic_candidates")
    composite_data, composite_issue = _read_json(Path(args.composite_candidates), "composite_candidates")
    issues = [issue for issue in (analysis_issue, discovery_issue, candidate_issue) if issue]
    report = validate_pilot_artifacts(
        episodes, analysis, discoveries, candidates,
        episode_number=args.episode_number, input_issues=issues,
    )
    if composite_data:
        try:
            report["composite_validation"] = validate_composite_candidates(episodes, composite_data)
        except (ValidationError, ValueError, TypeError) as exc:
            report["composite_validation"] = {"status": "invalid", "message": str(exc), "candidate_count": 0,
                                               "passing_count": 0, "failing_count": 1, "candidates": []}
    else:
        report["composite_validation"] = {"status": "unavailable", "message": "No composite candidate artifact is available.",
                                           "candidate_count": 0, "passing_count": 0, "failing_count": 0, "candidates": []}
    json_path, markdown_path = Path(args.json_output), Path(args.markdown_output)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    markdown_path.write_text(render_markdown_report(report), encoding="utf-8")
    print(f"Pilot validation: {report['validation_summary']['total_errors']} error(s), "
          f"{report['validation_summary']['total_warnings']} warning(s)")
    print(f"JSON report: {json_path}")
    print(f"Markdown report: {markdown_path}")


if __name__ == "__main__":
    main()
