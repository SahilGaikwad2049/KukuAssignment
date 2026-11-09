"""Compare Episode 1 semantic and heuristic candidate discovery artifacts offline."""
from __future__ import annotations

import argparse
import json
import os
import statistics
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from promoscout.analysis.event_tagger import InvalidEventReference, validate_discoveries
from promoscout.analysis.models import DiscoveredEvent
from promoscout.evaluation.blind_viewer import (
    BlindViewerEvaluation,
    InvalidEvidenceReference,
    _validate_evidence,
    build_blind_view_input,
)
from promoscout.evaluation.pilot_validator import validate_pilot_artifacts
from promoscout.generation.continuous import PromoCandidate, classify_hook_type
from promoscout.generation.preview import reconstruct_candidate_preview
from promoscout.schemas import ParsedEpisode
from promoscout.utils.cache import content_hash

DEFAULT_PARSED = Path("data/processed/parsed_all_episodes.json")
DEFAULT_ANALYSIS = Path("data/processed/episode_analysis.json")
DEFAULT_DISCOVERIES = Path("data/processed/event_discoveries.json")
DEFAULT_SEMANTIC_CANDIDATES = Path("data/outputs/semantic_candidates.json")
DEFAULT_HEURISTIC_CANDIDATES = Path("data/outputs/continuous_candidates.json")
DEFAULT_EVALUATIONS = Path("data/outputs/blind_viewer_evaluations.json")
DEFAULT_REVIEW_DECISIONS = Path("data/outputs/review_decisions.json")
DEFAULT_JSON_OUTPUT = Path("data/outputs/discovery_comparison.json")
DEFAULT_MARKDOWN_OUTPUT = Path("data/outputs/discovery_comparison.md")

DIMENSIONS = (
    "standalone_clarity",
    "hook_effectiveness",
    "conflict_comprehension",
    "stakes_clarity",
    "cliffhanger_effectiveness",
    "desire_to_continue",
)


def _read_json(path: str | Path, *, required: bool = False):
    file_path = Path(path)
    if not file_path.exists() or not file_path.stat().st_size:
        if required:
            raise ValueError(f"Required input is missing or empty: {file_path}")
        return None
    try:
        return json.loads(file_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {file_path}: {exc}") from exc


def _candidate_rows(value: Any) -> list[dict]:
    if isinstance(value, list):
        return [row for row in value if isinstance(row, dict)]
    if isinstance(value, dict) and isinstance(value.get("candidates"), list):
        return [row for row in value["candidates"] if isinstance(row, dict)]
    return []


def _source_event_structure_errors(episode: ParsedEpisode, raw: dict) -> list[str]:
    """Apply common timestamp, duration, ID, overlap and ordering checks."""
    try:
        candidate = PromoCandidate.model_validate(raw)
    except ValidationError as exc:
        return [f"Invalid candidate schema: {exc}"]
    errors = []
    if candidate.episode_number != episode.episode_number:
        errors.append("Candidate episode number does not match the parsed episode.")
    by_id = {event.event_id: event for event in episode.events}
    if len(by_id) != len(episode.events):
        errors.append("Parsed episode contains duplicate source event IDs.")
    ids = {
        "start": candidate.start_event_id,
        "hook": candidate.hook_event_id,
        "ending": candidate.ending_event_id,
        "end": candidate.end_event_id,
    }
    for label, event_id in ids.items():
        if event_id not in by_id:
            errors.append(f"{label.title()} event ID {event_id} does not exist in the parsed episode.")
    start = by_id.get(candidate.start_event_id)
    hook = by_id.get(candidate.hook_event_id)
    ending = by_id.get(candidate.ending_event_id)
    end = by_id.get(candidate.end_event_id)
    if start and (candidate.start_timestamp != start.start_timestamp or abs(candidate.start_seconds - start.start_seconds) > .02):
        errors.append("Candidate start metadata does not match its source event.")
    if end and (candidate.end_timestamp != end.end_timestamp or abs(candidate.end_seconds - end.end_seconds) > .02):
        errors.append("Candidate end metadata does not match its source event.")
    if ending and candidate.end_event_id != candidate.ending_event_id:
        errors.append("Candidate end_event_id and ending_event_id differ.")
    if hook:
        if candidate.hook_text != hook.content:
            errors.append("Hook text does not match its source event.")
        if candidate.hook_type != classify_hook_type(hook.event_type):
            errors.append("Hook type does not match the source event type.")
        if (candidate.hook_start_timestamp != hook.start_timestamp
                or candidate.hook_end_timestamp != hook.end_timestamp
                or abs(candidate.hook_start_seconds - hook.start_seconds) > .02
                or abs(candidate.hook_end_seconds - hook.end_seconds) > .02):
            errors.append("Hook timestamps do not match the source event.")
        lead_in = candidate.hook_start_seconds - candidate.start_seconds
        if abs(candidate.hook_offset_seconds - lead_in) > .02:
            errors.append("Hook offset does not match the source timestamps.")
        if lead_in < -.02 or lead_in > 3.02:
            errors.append("Hook lead-in exceeds the 3-second limit.")
    if ending:
        if candidate.ending_text != ending.content:
            errors.append("Potential ending text does not match its source event.")
        if (candidate.end_event_id != candidate.ending_event_id
                or candidate.end_timestamp != ending.end_timestamp
                or abs(candidate.end_seconds - ending.end_seconds) > .02):
            errors.append("Candidate ending metadata does not match the potential-ending source event.")
    actual_duration = candidate.end_seconds - candidate.start_seconds
    if abs(candidate.duration_seconds - actual_duration) > .02:
        errors.append("Candidate duration does not match its source timestamps.")
    if actual_duration < 19.98 or actual_duration > 60.02:
        errors.append("Candidate duration is outside 20–60 seconds.")

    expected = [event for event in episode.events
                if event.start_seconds <= candidate.end_seconds and event.end_seconds >= candidate.start_seconds]
    expected.sort(key=lambda event: (event.start_seconds, event.source_line_start, event.end_seconds, event.event_id))
    expected_ids = [event.event_id for event in expected]
    unknown_included = [event_id for event_id in candidate.included_event_ids if event_id not in by_id]
    if unknown_included:
        errors.append("Included event IDs are invalid: " + ", ".join(unknown_included))
    if set(candidate.included_event_ids) != set(expected_ids):
        errors.append("Included events do not exactly match events overlapping the candidate window.")
    if candidate.included_event_ids != expected_ids:
        errors.append("Included events are not in chronological order.")
    if start and end:
        try:
            preview = reconstruct_candidate_preview(episode, candidate)
            if [row["event_id"] for row in preview["events"]] != expected_ids:
                errors.append("Reconstructed preview is incomplete or out of order.")
        except ValueError as exc:
            errors.append(f"Preview reconstruction failed: {exc}")
    return errors


def _validate_saved_evaluation(raw: dict, source: str, candidate: PromoCandidate,
                               episode: ParsedEpisode) -> tuple[dict | None, str]:
    try:
        evaluation = BlindViewerEvaluation.model_validate(raw)
        if evaluation.candidate_id != candidate.candidate_id:
            return None, "Candidate ID does not match evaluation key."
        if evaluation.candidate_source != source:
            return None, "Evaluation source does not match candidate source."
        if evaluation.episode_number != episode.episode_number:
            return None, "Evaluation episode does not match source episode."
        clip = build_blind_view_input(episode, candidate)
        clip_ids = [row["event_id"] for row in clip["events"]]
        if evaluation.clip_event_ids != clip_ids:
            return None, "Saved evaluation clip event IDs do not match reconstructed clip."
        if evaluation.blind_input_hash != content_hash(clip):
            return None, "Saved evaluation input hash does not match reconstructed clip."
        _validate_evidence(evaluation.assessment, {row["event_id"]: row["content"] for row in clip["events"]})
        return evaluation.model_dump(mode="json"), "validated"
    except (ValidationError, InvalidEvidenceReference, ValueError, TypeError) as exc:
        return None, f"Saved evaluation failed validation: {exc}"


def _duration_stats(candidates: list[PromoCandidate]) -> dict:
    values = [candidate.duration_seconds for candidate in candidates]
    return {
        "minimum": round(min(values), 2) if values else None,
        "median": round(statistics.median(values), 2) if values else None,
        "average": round(statistics.mean(values), 2) if values else None,
        "maximum": round(max(values), 2) if values else None,
        "bands_seconds": {
            "20_to_30": sum(20 <= value < 30 for value in values),
            "30_to_45": sum(30 <= value < 45 for value in values),
            "45_to_60": sum(45 <= value <= 60 for value in values),
        },
    }


def _evaluation_summary(candidate_rows: list[dict]) -> dict:
    valid = [row for row in candidate_rows if row.get("evaluation_status") == "validated"]
    score_averages = {}
    for dimension in DIMENSIONS:
        values = [row["scores"][dimension] for row in valid if row.get("scores", {}).get(dimension) is not None]
        score_averages[dimension] = round(statistics.mean(values), 2) if values else None
    verdicts = Counter(row.get("verdict") for row in valid)
    return {
        "successful_validated_evaluations": len(valid),
        "evaluations_not_available_or_invalid": len(candidate_rows) - len(valid),
        "dimension_average_scores": score_averages,
        "verdict_counts": {name: verdicts.get(name, 0) for name in ("strong", "mixed", "weak")},
    }


def _qualitative_findings(episode: ParsedEpisode, heuristic_candidates: list[PromoCandidate],
                          semantic_candidates: list[PromoCandidate], discoveries: list[DiscoveredEvent],
                          candidate_rows: dict[str, list[dict]]) -> dict:
    event_by_id = {event.event_id: event for event in episode.events}
    heuristic_endpoints = {
        event_id for candidate in heuristic_candidates
        for event_id in (candidate.hook_event_id, candidate.ending_event_id)
    }
    semantic_discovery_ids = {item.event_id for item in discoveries}
    semantic_only = []
    for item in discoveries:
        if item.event_id not in heuristic_endpoints:
            event = event_by_id.get(item.event_id)
            semantic_only.append({
                "event_id": item.event_id,
                "role": item.role,
                "category": item.category,
                "source_text": event.content if event else None,
                "semantic_reason": item.reason,
                "semantic_context": item.narrative_context,
                "note": "This source event was selected by semantic discovery but is not a heuristic candidate endpoint; a human should judge its promotional value.",
            })
    heuristic_only_ids = sorted(heuristic_endpoints - semantic_discovery_ids)
    heuristic_only = [{
        "event_id": event_id,
        "source_text": event_by_id[event_id].content if event_id in event_by_id else None,
        "note": "This event appears as a heuristic candidate endpoint but was not selected as a semantic discovery.",
    } for event_id in heuristic_only_ids]
    shared = sorted(heuristic_endpoints & semantic_discovery_ids)

    similar_pairs = []
    for semantic in semantic_candidates:
        for heuristic in heuristic_candidates:
            if semantic.hook_event_id == heuristic.hook_event_id or semantic.ending_event_id == heuristic.ending_event_id:
                similar_pairs.append({
                    "semantic_candidate_id": semantic.candidate_id,
                    "heuristic_candidate_id": heuristic.candidate_id,
                    "shared_hook_event": semantic.hook_event_id == heuristic.hook_event_id,
                    "shared_ending_event": semantic.ending_event_id == heuristic.ending_event_id,
                })

    clarity = {}
    for dimension, description in (
        ("standalone_clarity", "standalone narrative clarity"),
        ("cliffhanger_effectiveness", "unresolved ending effectiveness"),
    ):
        method_means = {}
        for source in ("semantic", "heuristic"):
            rows = [row for row in candidate_rows[source] if row.get("evaluation_status") == "validated"]
            values = [row["scores"][dimension] for row in rows if row.get("scores", {}).get(dimension) is not None]
            method_means[source] = {"average": round(statistics.mean(values), 2) if values else None, "n": len(values)}
        semantic_avg, heuristic_avg = method_means["semantic"]["average"], method_means["heuristic"]["average"]
        observed = "insufficient validated evaluations from both methods"
        if semantic_avg is not None and heuristic_avg is not None:
            if semantic_avg > heuristic_avg:
                observed = "semantic average is higher in this exploratory sample"
            elif heuristic_avg > semantic_avg:
                observed = "heuristic average is higher in this exploratory sample"
            else:
                observed = "the observed averages are equal"
        clarity[description] = {"scores": method_means, "descriptive_observation": observed}

    return {
        "semantic_discovery_events_not_used_as_heuristic_endpoints": semantic_only,
        "heuristic_endpoints_not_selected_by_semantic_discovery": heuristic_only,
        "shared_discovery_event_ids": shared,
        "candidate_pairs_sharing_hook_or_ending_event": similar_pairs,
        "standalone_clarity_and_ending_comparison": clarity,
        "interpretation": "Event overlap and score differences are descriptive evidence for review, not proof that either discovery method is better.",
    }


def build_discovery_comparison(
    *,
    episode: ParsedEpisode,
    analysis_data: dict,
    discoveries_data: dict,
    heuristic_rows: list[dict],
    semantic_rows: list[dict],
    evaluations_data: dict,
    review_decisions: dict | None = None,
) -> dict:
    """Build source-validated discovery statistics and evaluation comparisons."""
    if not semantic_rows:
        raise ValueError("No semantic candidates for the selected episode; run semantic analysis and candidate generation first.")
    if not heuristic_rows:
        raise ValueError("No heuristic candidates for the selected episode; comparison requires both methods.")
    heuristic_candidates = [PromoCandidate.model_validate(row) for row in heuristic_rows]
    semantic_candidates = [PromoCandidate.model_validate(row) for row in semantic_rows]
    if any(candidate.episode_number != episode.episode_number for candidate in heuristic_candidates + semantic_candidates):
        raise ValueError("Candidate rows must be filtered to the selected episode before comparison.")

    raw_disc = discoveries_data.get(str(episode.episode_number), [])
    if not isinstance(raw_disc, list):
        raise ValueError("Episode discovery output must contain a list for the selected episode.")
    discoveries = validate_discoveries(raw_disc, {event.event_id for event in episode.events})
    semantic_validation = validate_pilot_artifacts(
        [episode], analysis_data, discoveries_data,
        {"candidates": semantic_rows}, episode_number=episode.episode_number,
    )
    semantic_status = {row.get("candidate_id"): row for row in semantic_validation["candidates"]}
    evaluation_map = evaluations_data.get("evaluations", {}) if isinstance(evaluations_data, dict) else {}
    review_map = review_decisions.get("decisions", {}) if isinstance(review_decisions, dict) else {}
    grouped_candidate_rows = {"semantic": [], "heuristic": []}
    candidate_records = []
    evaluation_models = set()

    for source, candidates, raw_rows in (
        ("heuristic", heuristic_candidates, heuristic_rows),
        ("semantic", semantic_candidates, semantic_rows),
    ):
        for candidate, raw in zip(candidates, raw_rows):
            if source == "semantic":
                validation = semantic_status.get(candidate.candidate_id, {})
                structural_status = validation.get("structural_status", "fail")
                structural_errors = [item.get("message", item.get("code", "validation error")) for item in validation.get("errors", [])]
                structural_warnings = [item.get("message", item.get("code", "warning")) for item in validation.get("warnings", [])]
            else:
                structural_errors = _source_event_structure_errors(episode, raw)
                structural_status = "pass" if not structural_errors else "fail"
                structural_warnings = []

            key = f"{source}:{candidate.candidate_id}"
            evaluation_row = evaluation_map.get(key) if isinstance(evaluation_map, dict) else None
            validated_evaluation, evaluation_message = (None, "No saved blind-viewer evaluation.")
            if isinstance(evaluation_row, dict):
                validated_evaluation, evaluation_message = _validate_saved_evaluation(
                    evaluation_row, source, candidate, episode,
                )
            if structural_status != "pass":
                evaluation_status = "excluded_structurally_invalid" if validated_evaluation else "not_evaluated"
                validated_evaluation = None
            else:
                evaluation_status = "validated" if validated_evaluation else (
                    "invalid_saved_evaluation" if isinstance(evaluation_row, dict) else "not_evaluated"
                )
            assessment = validated_evaluation.get("assessment", {}) if validated_evaluation else {}
            scores = {name: assessment.get(name, {}).get("score") for name in DIMENSIONS}
            human = review_map.get(key, {}) if isinstance(review_map, dict) else {}
            if validated_evaluation:
                evaluation_models.add(validated_evaluation["model_id"])
            candidate_view = {
                "candidate_id": candidate.candidate_id,
                "candidate_source": source,
                "episode_number": candidate.episode_number,
                "candidate_family_id": candidate.candidate_family_id,
                "duration_seconds": candidate.duration_seconds,
                "hook_type": candidate.hook_type,
                "hook_event_id": candidate.hook_event_id,
                "hook_text": candidate.hook_text,
                "ending_event_id": candidate.ending_event_id,
                "potential_ending_text": candidate.ending_text,
                "start_timestamp": candidate.start_timestamp,
                "end_timestamp": candidate.end_timestamp,
                "unique_window_key": [candidate.episode_number, round(candidate.start_seconds, 2), round(candidate.end_seconds, 2)],
                "structural_status": structural_status,
                "structural_errors": structural_errors,
                "structural_warnings": structural_warnings,
                "evaluation_status": evaluation_status,
                "evaluation_validation_note": evaluation_message,
                "scores": scores,
                "verdict": assessment.get("overall_viewer_verdict") if validated_evaluation else None,
                "confidence": assessment.get("confidence") if validated_evaluation else None,
                "viewer_story_reconstruction": assessment.get("standalone_story_reconstruction") if validated_evaluation else None,
                "verdict_explanation": assessment.get("verdict_explanation") if validated_evaluation else None,
                "confusing_elements": assessment.get("confusing_elements", []) if validated_evaluation else [],
                "missing_context": assessment.get("missing_context_the_viewer_wishes_for", []) if validated_evaluation else [],
                "dimension_rationales": {
                    name: assessment.get(name, {}).get("rationale") for name in DIMENSIONS
                } if validated_evaluation else {},
                "human_decision": human.get("decision", "pending") if isinstance(human, dict) else "pending",
            }
            candidate_records.append(candidate_view)
            grouped_candidate_rows[source].append(candidate_view)

    # Global duplicate IDs/windows and deterministic family integrity checks.
    for source in ("semantic", "heuristic"):
        rows = grouped_candidate_rows[source]
        ids = Counter(row["candidate_id"] for row in rows)
        windows = Counter(tuple(row["unique_window_key"]) for row in rows)
        family_hooks: dict[str, set[str]] = {}
        hook_families: dict[str, set[str]] = {}
        for row in rows:
            family_hooks.setdefault(row["candidate_family_id"], set()).add(row["hook_event_id"])
            hook_families.setdefault(row["hook_event_id"], set()).add(row["candidate_family_id"])
        for row in rows:
            if ids[row["candidate_id"]] > 1:
                _invalidate_candidate_row(row, "Candidate ID is duplicated within its source.")
            if len(family_hooks[row["candidate_family_id"]]) > 1:
                _invalidate_candidate_row(row, "Candidate family contains more than one hook event.")
            if len(hook_families[row["hook_event_id"]]) > 1:
                _invalidate_candidate_row(row, "One hook event is assigned to multiple candidate families.")
            if windows[tuple(row["unique_window_key"])] > 1:
                row["structural_warnings"].append("Duplicate clip window exists within this source.")
        episode_hook_order = sorted(
            {row["hook_event_id"] for row in rows},
            key=lambda event_id: next(
                ((event.start_seconds, event.source_line_start, event.event_id)
                 for event in episode.events if event.event_id == event_id),
                (float("inf"), float("inf"), event_id),
            ),
        )
        expected_family_by_hook = {
            hook_id: f"EP{episode.episode_number:03d}_FAMILY_{index:03d}"
            for index, hook_id in enumerate(episode_hook_order, start=1)
        }
        for row in rows:
            if row["candidate_family_id"] != expected_family_by_hook.get(row["hook_event_id"]):
                _invalidate_candidate_row(row, "Candidate family ID does not follow deterministic chronological numbering.")

    source_stats = {}
    for source, candidates, candidate_views in (
        ("heuristic", heuristic_candidates, grouped_candidate_rows["heuristic"]),
        ("semantic", semantic_candidates, grouped_candidate_rows["semantic"]),
    ):
        windows = {(item.episode_number, round(item.start_seconds, 2), round(item.end_seconds, 2)) for item in candidates}
        source_stats[source] = {
            "candidate_count": len(candidates),
            "unique_window_count": len(windows),
            "hook_type_distribution": dict(sorted(Counter(item.hook_type for item in candidates).items())),
            "duration_distribution": _duration_stats(candidates),
            "candidate_families": sorted({item.candidate_family_id for item in candidates}),
            "candidate_family_count": len({item.candidate_family_id for item in candidates}),
            "structurally_valid_candidates": sum(row["structural_status"] == "pass" for row in candidate_views),
            "structurally_invalid_candidates": sum(row["structural_status"] != "pass" for row in candidate_views),
            "blind_viewer_comparison": _evaluation_summary(candidate_views),
        }

    return {
        "report_version": "discovery-comparison-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "episode_number": episode.episode_number,
        "original_event_count": len(episode.events),
        "semantic_discovery_statistics": {
            "validated_discovery_count": len(discoveries),
            "hook_count": sum(item.role == "hook" for item in discoveries),
            "potential_ending_count": sum(item.role == "potential_ending" for item in discoveries),
            "category_distribution": dict(sorted(Counter(item.category for item in discoveries).items())),
        },
        "method_statistics": source_stats,
        "candidate_level_comparison": candidate_records,
        "qualitative_comparison": _qualitative_findings(
            episode, heuristic_candidates, semantic_candidates, discoveries, grouped_candidate_rows,
        ),
        "semantic_structural_validation": semantic_validation["validation_summary"],
        "evaluation_model_ids": sorted(evaluation_models),
        "human_review_note": "Human decisions are displayed separately; AI verdicts never create or imply approval.",
        "limitations": [
            "This is an Episode 1 experiment only.",
            "A sample of 2–3 semantic candidates is exploratory.",
            "The blind-viewer model is itself an AI evaluator.",
            "AI verdicts are not human approval.",
            "This comparison does not establish statistical superiority.",
            "Technical structural validity is not promotional quality.",
        ],
    }


def _invalidate_candidate_row(row: dict, message: str) -> None:
    row["structural_status"] = "fail"
    row["structural_errors"].append(message)
    if row.get("evaluation_status") == "validated":
        row["evaluation_status"] = "excluded_structurally_invalid"
        row["scores"] = {name: None for name in DIMENSIONS}
        row["verdict"] = None
        row["confidence"] = None
        row["viewer_story_reconstruction"] = None
        row["verdict_explanation"] = None
        row["dimension_rationales"] = {}


def render_markdown(report: dict) -> str:
    lines = [
        "# Semantic vs Heuristic Discovery — Episode 1",
        "",
        f"- Source episode: **{report['episode_number']}** · Original events: **{report['original_event_count']}**",
        f"- Semantic discoveries: **{report['semantic_discovery_statistics']['validated_discovery_count']}** "
        f"({report['semantic_discovery_statistics']['hook_count']} hooks, {report['semantic_discovery_statistics']['potential_ending_count']} potential endings)",
        "",
        "## Discovery statistics",
        "",
        "| Method | Candidates | Unique windows | Structurally valid | Families | Duration median | Hook types |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for source in ("heuristic", "semantic"):
        row = report["method_statistics"][source]
        lines.append(
            f"| {source.title()} | {row['candidate_count']} | {row['unique_window_count']} | {row['structurally_valid_candidates']} | {row['candidate_family_count']} | {row['duration_distribution']['median'] if row['duration_distribution']['median'] is not None else '—'}s | {row['hook_type_distribution']} |"
        )
    lines.extend(["", "## Blind-viewer comparison", "", "| Method | n | Clarity | Hook | Conflict | Stakes | Ending | Continue | Strong / mixed / weak |", "|---|---:|---:|---:|---:|---:|---:|---:|---|"])
    for source in ("heuristic", "semantic"):
        summary = report["method_statistics"][source]["blind_viewer_comparison"]
        averages = summary["dimension_average_scores"]
        vals = [averages[name] if averages[name] is not None else "—" for name in DIMENSIONS]
        verdicts = summary["verdict_counts"]
        lines.append(f"| {source.title()} | {summary['successful_validated_evaluations']} | " + " | ".join(map(str, vals)) + f" | {verdicts['strong']} / {verdicts['mixed']} / {verdicts['weak']} |")
    lines.extend(["", "Only schema-valid assessments with matching candidate/source/episode, reconstructed clip IDs/hash, and source-grounded evidence are counted. Missing or failed evaluations are excluded, not scored as weak.", "", "## Candidate-level comparison", ""])
    for source in ("heuristic", "semantic"):
        lines.extend([f"### {source.title()} candidates", ""])
        rows = [row for row in report["candidate_level_comparison"] if row["candidate_source"] == source]
        if not rows:
            lines.append("No candidates.")
        for row in rows:
            lines.extend([
                f"#### {row['candidate_id']} · {row['structural_status'].upper()} · {row['evaluation_status']}",
                f"- Duration: {row['duration_seconds']}s · Hook type: {row['hook_type']} · Family: {row['candidate_family_id']}",
                f"- Opening: {_md(row['hook_text'])}",
                f"- Potential ending (not a confirmed cliffhanger): {_md(row['potential_ending_text'])}",
                f"- Scores (clarity / hook / conflict / stakes / ending / continue): "
                + " / ".join(str(row['scores'].get(name) if row['scores'].get(name) is not None else "—") for name in DIMENSIONS),
                f"- Verdict: {row['verdict'] or 'not available'} · Confidence: {row['confidence'] if row['confidence'] is not None else '—'} · Human decision: {row['human_decision']}",
            ])
            if row.get("viewer_story_reconstruction"):
                lines.append(f"- Blind-viewer reconstruction: {_md(row['viewer_story_reconstruction'])}")
            if row.get("verdict_explanation"):
                lines.append(f"- AI verdict reason: {_md(row['verdict_explanation'])}")
            if row.get("structural_errors"):
                lines.extend(f"- Structural error: {_md(error)}" for error in row["structural_errors"])
            lines.append("")
    qualitative = report["qualitative_comparison"]
    lines.extend(["## Qualitative comparison", "", "### Semantic discoveries not used as heuristic endpoints", ""])
    for item in qualitative["semantic_discovery_events_not_used_as_heuristic_endpoints"]:
        lines.append(f"- `{item['event_id']}` ({item['role']} / {item['category']}): {_md(item.get('source_text') or '')} — model reason: {_md(item['semantic_reason'])}")
    if not qualitative["semantic_discovery_events_not_used_as_heuristic_endpoints"]:
        lines.append("- None in the supplied artifacts.")
    lines.extend(["", "### Heuristic endpoints not selected by semantic discovery", ""])
    for item in qualitative["heuristic_endpoints_not_selected_by_semantic_discovery"]:
        lines.append(f"- `{item['event_id']}`: {_md(item.get('source_text') or '')}")
    if not qualitative["heuristic_endpoints_not_selected_by_semantic_discovery"]:
        lines.append("- None in the supplied artifacts.")
    lines.extend(["", f"### Shared source event references", "", ", ".join(f"`{event_id}`" for event_id in qualitative["shared_discovery_event_ids"]) or "None.", "", "### Narrative clarity and unresolved endings", ""])
    for label, comparison in qualitative["standalone_clarity_and_ending_comparison"].items():
        lines.append(f"- **{label.title()}:** {comparison['descriptive_observation']}; scores {comparison['scores']}.")
    lines.extend(["", qualitative["interpretation"], "", "## Limitations and human review", ""])
    lines.extend(f"- {item}" for item in report["limitations"])
    lines.append("- Do not approve candidates from these AI assessments; inspect the source previews and make human review decisions separately.")
    return "\n".join(lines) + "\n"


def _md(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ").strip()


def _atomic_write(path: Path, text: str) -> None:
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Episode 1 semantic and heuristic promotional discovery offline")
    parser.add_argument("--episode-number", type=int, default=1)
    parser.add_argument("--parsed", default=str(DEFAULT_PARSED))
    parser.add_argument("--analysis", default=str(DEFAULT_ANALYSIS))
    parser.add_argument("--discoveries", default=str(DEFAULT_DISCOVERIES))
    parser.add_argument("--semantic-candidates", default=str(DEFAULT_SEMANTIC_CANDIDATES))
    parser.add_argument("--heuristic-candidates", default=str(DEFAULT_HEURISTIC_CANDIDATES))
    parser.add_argument("--evaluations", default=str(DEFAULT_EVALUATIONS))
    parser.add_argument("--review-decisions", default=str(DEFAULT_REVIEW_DECISIONS))
    parser.add_argument("--json-output", default=str(DEFAULT_JSON_OUTPUT))
    parser.add_argument("--markdown-output", default=str(DEFAULT_MARKDOWN_OUTPUT))
    args = parser.parse_args()
    try:
        parsed_rows = _read_json(args.parsed, required=True)
        episodes = [ParsedEpisode.model_validate(row) for row in parsed_rows]
        episode = next((item for item in episodes if item.episode_number == args.episode_number), None)
        if episode is None:
            raise ValueError(f"Episode {args.episode_number} not found in parsed screenplay input.")
        analysis = _read_json(args.analysis) or {}
        discoveries = _read_json(args.discoveries) or {}
        semantic_all = _candidate_rows(_read_json(args.semantic_candidates))
        heuristic_all = _candidate_rows(_read_json(args.heuristic_candidates))
        semantic_rows = [row for row in semantic_all if row.get("episode_number") == args.episode_number]
        heuristic_rows = [row for row in heuristic_all if row.get("episode_number") == args.episode_number]
        evaluation_data = _read_json(args.evaluations) or {}
        review_data = _read_json(args.review_decisions) or {}
        report = build_discovery_comparison(
            episode=episode,
            analysis_data=analysis,
            discoveries_data=discoveries,
            heuristic_rows=heuristic_rows,
            semantic_rows=semantic_rows,
            evaluations_data=evaluation_data,
            review_decisions=review_data,
        )
    except (OSError, ValueError, ValidationError, TypeError, InvalidEventReference) as exc:
        parser.error(str(exc))
    json_path, markdown_path = Path(args.json_output), Path(args.markdown_output)
    _atomic_write(json_path, json.dumps(report, ensure_ascii=False, indent=2))
    _atomic_write(markdown_path, render_markdown(report))
    print(f"Episode {args.episode_number}: {report['method_statistics']['semantic']['candidate_count']} semantic and {report['method_statistics']['heuristic']['candidate_count']} heuristic candidate(s)")
    print(f"Validated saved evaluations: semantic {report['method_statistics']['semantic']['blind_viewer_comparison']['successful_validated_evaluations']}; heuristic {report['method_statistics']['heuristic']['blind_viewer_comparison']['successful_validated_evaluations']}")
    print(f"JSON report: {json_path}")
    print(f"Markdown report: {markdown_path}")


if __name__ == "__main__":
    main()
