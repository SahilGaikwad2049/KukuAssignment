"""Transparent provisional rankings from validated structure, AI assessment, and separate human decisions."""
from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from promoscout.analysis.episode_analyzer import load_episodes
from promoscout.evaluation.blind_viewer import (
    BlindViewerEvaluation, InvalidEvidenceReference, _validate_evidence,
    build_blind_view_input, build_composite_blind_view_input,
)
from promoscout.evaluation.discovery_comparison import _source_event_structure_errors
from promoscout.evaluation.pilot_validator import validate_composite_candidates, validate_pilot_artifacts
from promoscout.generation.composite import CompositeCandidate
from promoscout.generation.continuous import PromoCandidate
from promoscout.schemas import ParsedEpisode
from promoscout.utils.cache import content_hash

WEIGHTS = {
    "standalone_clarity": 0.20,
    "hook_effectiveness": 0.20,
    "conflict_comprehension": 0.10,
    "stakes_clarity": 0.10,
    "cliffhanger_effectiveness": 0.20,
    "desire_to_continue": 0.20,
}
DEFAULTS = {
    "parsed": "data/processed/parsed_all_episodes.json",
    "analysis": "data/processed/episode_analysis.json",
    "discoveries": "data/processed/event_discoveries.json",
    "continuous": "data/outputs/continuous_candidates.json",
    "semantic": "data/outputs/semantic_candidates.json",
    "composite": "data/outputs/composite_candidates.json",
    "evaluations": "data/outputs/blind_viewer_evaluations.json",
    "pilot_report": "data/outputs/blind_viewer_pilot_report.json",
    "validation": "data/outputs/pilot_validation.json",
    "reviews": "data/outputs/review_decisions.json",
    "json": "data/outputs/final_rankings.json",
    "csv": "data/outputs/final_rankings.csv",
    "markdown": "data/outputs/final_ranking_report.md",
}


def _read(path: str | Path, default=None):
    file = Path(path)
    if not file.is_file() or file.stat().st_size == 0:
        return default
    try:
        return json.loads(file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {file}: {exc}") from exc


def _rows(payload: Any) -> list[dict]:
    rows = payload.get("candidates", []) if isinstance(payload, dict) else payload
    return [row for row in rows if isinstance(row, dict)] if isinstance(rows, list) else []


def _valid_assessment(raw: dict, *, candidate_id: str, source: str,
                      episode: ParsedEpisode, candidate: PromoCandidate) -> tuple[dict | None, str]:
    try:
        evaluation = BlindViewerEvaluation.model_validate(raw)
        if evaluation.candidate_id != candidate_id:
            return None, "Saved evaluation candidate ID does not match."
        if evaluation.candidate_source != source:
            return None, "Saved evaluation source does not match."
        if evaluation.episode_number != episode.episode_number:
            return None, "Saved evaluation episode does not match."
        clip = build_blind_view_input(episode, candidate)
        ids = [item["event_id"] for item in clip["events"]]
        if evaluation.clip_event_ids != ids:
            return None, "Saved evaluation event IDs do not match the reconstructed candidate clip."
        if evaluation.blind_input_hash != content_hash(clip):
            return None, "Saved evaluation content hash does not match the reconstructed candidate clip."
        _validate_evidence(evaluation.assessment, {item["event_id"]: item["content"] for item in clip["events"]})
        scores = {name: getattr(evaluation.assessment, name).score for name in WEIGHTS}
        return {
            "scores": scores,
            "verdict": evaluation.assessment.overall_viewer_verdict,
            "confidence": evaluation.assessment.confidence,
            "reason": evaluation.assessment.verdict_explanation,
            "model_id": evaluation.model_id,
        }, "validated"
    except (ValidationError, InvalidEvidenceReference, ValueError, TypeError) as exc:
        return None, str(exc)


def _valid_composite_assessment(raw: dict, *, candidate: CompositeCandidate,
                                episodes: list[ParsedEpisode]) -> tuple[dict | None, str]:
    try:
        evaluation = BlindViewerEvaluation.model_validate(raw)
        if evaluation.candidate_id != candidate.candidate_id or evaluation.candidate_source != "composite":
            return None, "Saved composite evaluation identity does not match."
        if evaluation.episode_number is not None:
            return None, "Composite evaluation must not claim a single episode number."
        clip, mapping = build_composite_blind_view_input(episodes, candidate)
        ids = [row["event_id"] for row in clip["events"]]
        source_by_id = {row["event_id"]: row["content"] for row in clip["events"]}
        if evaluation.clip_event_ids != ids or evaluation.composite_source_event_mapping != mapping:
            return None, "Saved composite event mapping does not match reconstructed source events."
        if evaluation.blind_input_hash != content_hash(clip):
            return None, "Saved composite evaluation hash does not match the reconstructed blind clip."
        _validate_evidence(evaluation.assessment, source_by_id)
        return {
            "scores": {name: getattr(evaluation.assessment, name).score for name in WEIGHTS},
            "verdict": evaluation.assessment.overall_viewer_verdict,
            "confidence": evaluation.assessment.confidence,
            "reason": evaluation.assessment.verdict_explanation,
            "model_id": evaluation.model_id,
        }, "validated"
    except (ValidationError, InvalidEvidenceReference, ValueError, TypeError) as exc:
        return None, str(exc)


def _evaluation_failures(report: Any) -> dict[tuple[str, str], str]:
    output = {}
    if isinstance(report, dict):
        for row in report.get("failures", []):
            if isinstance(row, dict) and row.get("candidate_id"):
                output[(row.get("source", "heuristic"), row["candidate_id"])] = str(row.get("error", "Evaluation failed"))
    return output


def _composite_rows(rows: list[dict], episodes: list[ParsedEpisode]) -> tuple[list[dict], dict[str, dict]]:
    validation = validate_composite_candidates(episodes, {"candidates": rows})
    return rows, {row["candidate_id"]: row for row in validation["candidates"]}


def _weighted_score(scores: dict[str, int], weights: dict[str, float]) -> float:
    total_weight = sum(weights.values())
    if total_weight <= 0 or any(name not in scores for name in weights):
        raise ValueError("All six dimension weights must be present and have a positive total")
    return round(sum(scores[name] * weights[name] for name in weights) / total_weight / 5 * 100, 2)


def _time_overlap(left: dict, right: dict) -> float:
    if left["candidate_type"] == "composite" or right["candidate_type"] == "composite":
        return 0.0
    if left["episodes"] != right["episodes"]:
        return 0.0
    start = max(left["start_seconds"], right["start_seconds"])
    end = min(left["end_seconds"], right["end_seconds"])
    intersection = max(0.0, end - start)
    union = max(left["end_seconds"], right["end_seconds"]) - min(left["start_seconds"], right["start_seconds"])
    return intersection / union if union > 0 else 0.0


def _diversity_shortlist(ranked: list[dict], limit: int = 20) -> list[dict]:
    if not ranked:
        return []
    selected, remaining = [], list(ranked)
    categories = sorted({row["candidate_type"] for row in ranked})
    # Keep a representative of each available source category when capacity permits.
    if limit >= len(categories):
        for category in categories:
            row = next((item for item in remaining if item["candidate_type"] == category), None)
            if row is not None:
                selected.append(row)
                remaining.remove(row)
    while remaining and len(selected) < limit:
        def penalty(row):
            same_family = any(row["candidate_family_id"] == item["candidate_family_id"] for item in selected)
            same_episode = bool(set(row["episodes"]) & {ep for item in selected for ep in item["episodes"]})
            similar_window = any(_time_overlap(row, item) >= .8 for item in selected)
            return (same_family, similar_window, same_episode, -row["weighted_score"], row["candidate_id"])
        next_row = min(remaining, key=penalty)
        selected.append(next_row)
        remaining.remove(next_row)
    return sorted(selected, key=lambda row: (-row["weighted_score"], row["candidate_id"]))


def build_final_rankings(
    *, episodes: list[ParsedEpisode], continuous_rows: list[dict], semantic_rows: list[dict],
    composite_rows: list[dict], evaluations_data: dict, pilot_report: dict | None = None,
    validation_data: dict | None = None, review_data: dict | None = None,
    analysis_data: dict | None = None, discoveries_data: dict | None = None,
    weights: dict[str, float] | None = None,
) -> dict:
    weights = dict(weights or WEIGHTS)
    if set(weights) != set(WEIGHTS) or any(value < 0 for value in weights.values()) or sum(weights.values()) <= 0:
        raise ValueError("Weights must define the six evaluation dimensions with nonnegative values and positive total")
    evaluation_map = evaluations_data.get("evaluations", {}) if isinstance(evaluations_data, dict) else {}
    failed = _evaluation_failures(pilot_report)
    reviews = review_data.get("decisions", {}) if isinstance(review_data, dict) else {}
    episode_map = {episode.episode_number: episode for episode in episodes}
    all_rows = []
    validation_by_id = {}
    semantic_validation_map = {}
    if semantic_rows:
        semantic_report = validate_pilot_artifacts(
            episodes, analysis_data or {}, discoveries_data or {},
            {"candidates": semantic_rows},
        )
        semantic_validation_map = {row.get("candidate_id"): row for row in semantic_report["candidates"]}

    # Revalidate continuous candidates against the exact parsed source, including
    # timing, duration, hook, end, overlap coverage, and preview reconstruction.
    continuous_groups = (("heuristic", "continuous", continuous_rows), ("semantic", "continuous", semantic_rows))
    for source, candidate_type, rows in continuous_groups:
        for raw in rows:
            candidate_id = str(raw.get("candidate_id", "unknown"))
            errors = []
            try:
                candidate = PromoCandidate.model_validate(raw)
                episode = episode_map.get(candidate.episode_number)
                if episode is None:
                    errors.append(f"Source episode {candidate.episode_number} is unavailable.")
                elif source == "semantic":
                    semantic_status = semantic_validation_map.get(candidate_id, {})
                    if semantic_status.get("structural_status") != "pass":
                        errors.extend(item.get("message", item.get("code", "semantic validation failed"))
                                      for item in semantic_status.get("errors", []))
                        if not errors:
                            errors.append("Semantic source analysis or event discoveries did not pass structural validation.")
                    errors.extend(_source_event_structure_errors(episode, raw))
                else:
                    errors.extend(_source_event_structure_errors(episode, raw))
            except ValidationError as exc:
                candidate, episode = None, None
                errors.append(f"Candidate schema is invalid: {exc}")
            validation_by_id[(source, candidate_id)] = {"status": "passed" if not errors else "failed", "errors": errors}
            all_rows.append((source, candidate_type, raw, candidate, episode))

    composite_validation_by_id = {}
    if composite_rows:
        _, composite_validation_by_id = _composite_rows(composite_rows, episodes)
    for raw in composite_rows:
        try:
            candidate = CompositeCandidate.model_validate(raw)
            errors = composite_validation_by_id.get(candidate.candidate_id, {}).get("errors", ["Composite validation result missing."])
            status = "passed" if not errors else "failed"
        except ValidationError as exc:
            candidate = None
            errors = [f"Composite schema is invalid: {exc}"]
            status = "failed"
        validation_by_id[("composite", str(raw.get("candidate_id", "unknown")))] = {"status": status, "errors": errors}
        all_rows.append(("composite", "composite", raw, candidate, None))

    # Apply deterministic artifact-level duplicate and family integrity checks.
    for source in ("heuristic", "semantic"):
        grouped = [(s, t, raw, candidate, episode) for s, t, raw, candidate, episode in all_rows if s == source and candidate is not None]
        id_counts = Counter(candidate.candidate_id for _, _, _, candidate, _ in grouped)
        windows = Counter((candidate.episode_number, round(candidate.start_seconds, 2), round(candidate.end_seconds, 2)) for _, _, _, candidate, _ in grouped)
        family_hooks, hook_families = {}, {}
        for _, _, _, candidate, _ in grouped:
            family_hooks.setdefault(candidate.candidate_family_id, set()).add(candidate.hook_event_id)
            hook_families.setdefault(candidate.hook_event_id, set()).add(candidate.candidate_family_id)
        for _, _, _, candidate, _ in grouped:
            state = validation_by_id[(source, candidate.candidate_id)]
            if id_counts[candidate.candidate_id] > 1:
                state["errors"].append("Candidate ID is duplicated within source.")
            if windows[(candidate.episode_number, round(candidate.start_seconds, 2), round(candidate.end_seconds, 2))] > 1:
                state["errors"].append("Candidate window is duplicated within source.")
            if len(family_hooks[candidate.candidate_family_id]) > 1:
                state["errors"].append("Candidate family contains multiple hook event IDs.")
            if len(hook_families[candidate.hook_event_id]) > 1:
                state["errors"].append("One hook is assigned to multiple candidate families.")
            if state["errors"]:
                state["status"] = "failed"

    result_rows = []
    for source, candidate_type, raw, candidate, episode in all_rows:
        candidate_id = str(raw.get("candidate_id", "unknown"))
        key = f"{source}:{candidate_id}"
        structure = validation_by_id[(source, candidate_id)]
        evaluation = None
        evaluation_status, evaluation_reason = "pending", "No saved evaluation is available."
        if isinstance(evaluation_map, dict) and isinstance(evaluation_map.get(key), dict):
            if source == "composite":
                if candidate is not None:
                    evaluation, evaluation_reason = _valid_composite_assessment(evaluation_map[key], candidate=candidate, episodes=episodes)
                    evaluation_status = "evaluated" if evaluation is not None else "evaluation_failed"
                else:
                    evaluation_status, evaluation_reason = "evaluation_failed", "Composite candidate schema is invalid."
            elif episode is not None and candidate is not None:
                evaluation, evaluation_reason = _valid_assessment(evaluation_map[key], candidate_id=candidate_id,
                    source=source, episode=episode, candidate=candidate)
                evaluation_status = "evaluated" if evaluation is not None else "evaluation_failed"
            else:
                evaluation_status, evaluation_reason = "evaluation_failed", "Candidate could not be resolved to a parsed episode."
        elif (source, candidate_id) in failed:
            evaluation_status, evaluation_reason = "evaluation_failed", failed[(source, candidate_id)]
        review = reviews.get(key, {}) if isinstance(reviews, dict) else {}
        human_decision = review.get("decision", "pending") if isinstance(review, dict) else "pending"
        scores = evaluation["scores"] if evaluation else {name: None for name in WEIGHTS}
        score = _weighted_score(scores, weights) if evaluation and structure["status"] == "passed" else None
        is_synthetic = source == "composite" and (
            raw.get("candidate_origin") == "synthetic" or bool(raw.get("synthetic", False))
            or (candidate is not None and candidate.candidate_origin == "synthetic")
        )
        if is_synthetic:
            ranking_status = "synthetic_fixture_excluded"
            inclusion_reason = "Excluded: synthetic test fixture, not a real screenplay candidate."
            evaluation_status = "synthetic_fixture_excluded"
            evaluation = None
        elif structure["status"] != "passed":
            ranking_status = "structurally_ineligible"
            inclusion_reason = "Excluded because structural source, duration, timing, preview, or schema checks failed."
        elif evaluation_status == "evaluation_failed":
            ranking_status = "evaluation_failed"
            inclusion_reason = "Excluded because saved blind-viewer evaluation failed local validation."
        elif evaluation_status == "pending":
            ranking_status = "pending"
            inclusion_reason = "Not ranked: no successfully validated blind-viewer evaluation is available."
        else:
            ranking_status = "ranked"
            inclusion_reason = "Structurally valid; blind-viewer schema, clip identity, and evidence references validated; weighted provisional score calculated."
        if human_decision == "rejected":
            final_selection_reason = "Human rejected; do not present as an approved final selection."
        elif human_decision == "approved":
            final_selection_reason = "Human approved independently of AI score."
        else:
            final_selection_reason = "Human decision pending; AI ranking is not an approval."
        if candidate_type == "composite" and candidate is not None:
            candidate_episodes = sorted({segment.episode_number for segment in candidate.segments})
            duration = candidate.total_duration_seconds
            family = candidate.candidate_family_id
            start_seconds = end_seconds = None
        elif candidate is not None:
            candidate_episodes = [candidate.episode_number]
            duration = candidate.duration_seconds
            family = candidate.candidate_family_id
            start_seconds, end_seconds = candidate.start_seconds, candidate.end_seconds
        else:
            candidate_episodes = []
            duration = raw.get("duration_seconds")
            family = raw.get("candidate_family_id", "")
            start_seconds = end_seconds = None
        result_rows.append({
            "candidate_id": candidate_id,
            "source_type": source,
            "candidate_type": candidate_type,
            "synthetic_fixture": is_synthetic,
            "episodes": candidate_episodes,
            "duration_seconds": duration,
            "candidate_family_id": family,
            "start_seconds": start_seconds,
            "end_seconds": end_seconds,
            "dimension_scores": scores,
            "weighted_score": score,
            "blind_viewer_verdict": evaluation["verdict"] if evaluation else None,
            "evaluation_confidence": evaluation["confidence"] if evaluation else None,
            "evaluation_model": evaluation["model_id"] if evaluation else None,
            "evaluation_status": evaluation_status,
            "evaluation_status_reason": evaluation_reason,
            "structural_validation_status": structure["status"],
            "structural_errors": structure["errors"],
            "human_decision": human_decision,
            "reviewer_note": review.get("note", "") if isinstance(review, dict) else "",
            "ranking_status": ranking_status,
            "provisional_ai_eligible": ranking_status == "ranked",
            "final_selection_eligible": ranking_status == "ranked" and human_decision == "approved",
            "inclusion_or_exclusion_reason": inclusion_reason + " " + final_selection_reason,
        })

    ranked = [row for row in result_rows if row["ranking_status"] == "ranked"]
    ranked.sort(key=lambda row: (-row["weighted_score"], row["candidate_id"]))
    for index, row in enumerate(ranked, start=1):
        row["rank"] = index
    continuous = [row for row in ranked if row["candidate_type"] == "continuous"]
    composite = [row for row in ranked if row["candidate_type"] == "composite"]
    combined = ranked
    return {
        "schema_version": "final-rankings-v1",
        "generated_from_real_artifacts_only": True,
        "ranking_interpretation": "AI-assisted provisional ranking; not objective truth or a guaranteed measure of advertising performance.",
        "weights": weights,
        "score_scale": {"dimension": "1–5", "normalized_composite": "0–100"},
        "source_artifact_counts": {
            "continuous_candidates": len(continuous_rows), "semantic_candidates": len(semantic_rows),
            "composite_candidates_supplied": len(composite_rows),
            "real_composite_candidates": sum(row.get("candidate_origin") != "synthetic" and not row.get("synthetic", False) for row in composite_rows),
        },
        "leaderboards": {
            "continuous": _leaderboard(continuous),
            "composite": _leaderboard(composite),
            "combined": _leaderboard(combined),
        },
        "diversity_shortlist": _diversity_shortlist(combined),
        "pending_candidates": [row for row in result_rows if row["ranking_status"] == "pending"],
        "failed_evaluations": [row for row in result_rows if row["ranking_status"] == "evaluation_failed"],
        "structurally_ineligible_candidates": [row for row in result_rows if row["ranking_status"] == "structurally_ineligible"],
        "synthetic_composite_fixtures": [row for row in result_rows if row["ranking_status"] == "synthetic_fixture_excluded"],
        "all_candidates": result_rows,
        "human_review_is_independent": True,
        "validation_report_note": "Existing pilot_validation.json is retained as provenance; heuristic candidates are independently revalidated against source events because that report covers semantic pilot artifacts.",
    }


def _leaderboard(rows: list[dict]) -> list[dict]:
    output = []
    for index, row in enumerate(rows, start=1):
        output.append({**row, "rank": index})
    return output


def _csv_rows(report: dict) -> list[dict]:
    rows = []
    for row in report["all_candidates"]:
        value = {key: item for key, item in row.items() if key != "rank"}
        value["rank"] = next((item["rank"] for item in report["leaderboards"]["combined"] if item["candidate_id"] == row["candidate_id"] and item["source_type"] == row["source_type"]), None)
        for name in WEIGHTS:
            value[name] = row["dimension_scores"].get(name)
        del value["dimension_scores"]
        for key, item in list(value.items()):
            if isinstance(item, (dict, list)):
                value[key] = json.dumps(item, ensure_ascii=False)
        rows.append(value)
    return rows


def render_markdown(report: dict) -> str:
    lines = ["# PromoScout Final Provisional Rankings", "", report["ranking_interpretation"], "",
             "Scores include only candidates with passed structural checks and successfully validated blind-viewer evidence. Human decisions remain separate.", ""]
    for title, key in (("Continuous promo leaderboard", "continuous"), ("Composite promo leaderboard", "composite"), ("Combined leaderboard", "combined")):
        rows = report["leaderboards"][key]
        lines.extend([f"## {title}", "", "| Rank | Candidate | Source | Episode(s) | Seconds | Clarity | Hook | Conflict | Stakes | Ending | Continue | Weighted / 100 | Verdict | Structural | Human | Reviewer note | Inclusion / exclusion reason |", "|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|---|"])
        if not rows:
            lines.extend(["| — | No eligible evaluated candidates | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |", ""])
            continue
        for index, row in enumerate(rows, start=1):
            scores = row["dimension_scores"]
            values = [scores.get(name, "—") for name in WEIGHTS]
            note = str(row.get("reviewer_note", "")).replace("|", "\\|").replace("\n", " ")
            rationale = str(row["inclusion_or_exclusion_reason"]).replace("|", "\\|").replace("\n", " ")
            lines.append(f"| {index} | {row['candidate_id']} | {row['source_type']} | {','.join(map(str,row['episodes']))} | {row['duration_seconds']} | " + " | ".join(map(str, values)) + f" | {row['weighted_score']} | {row['blind_viewer_verdict']} | {row['structural_validation_status']} | {row['human_decision']} | {note} | {rationale} |")
        lines.append("")
    for title, key in (("Pending candidates", "pending_candidates"), ("Failed evaluations", "failed_evaluations"), ("Structurally ineligible candidates", "structurally_ineligible_candidates")):
        rows = report[key]
        lines.extend([f"## {title} ({len(rows)})", ""])
        if not rows:
            lines.extend(["None.", ""])
            continue
        lines.extend(["| Candidate | Source | Structural status | Human decision | Reason |", "|---|---|---|---|---|"])
        for row in rows:
            reason = row["evaluation_status_reason"] if row["ranking_status"] in {"pending", "evaluation_failed"} else "; ".join(row["structural_errors"])
            lines.append(f"| {row['candidate_id']} | {row['source_type']} | {row['structural_validation_status']} | {row['human_decision']} | {reason} |")
        lines.append("")
    lines.extend(["## Scoring and interpretation", "", "The weighted score is the weighted sum of six 1–5 model ratings, divided by five and multiplied by 100. It is provisional and is not an objective measure of ad performance. Candidates with human decision `rejected` may appear in the AI leaderboard for transparency, but `final_selection_eligible` is false. Pending or failed assessments receive no numeric score.", ""])
    return "\n".join(lines)


def _atomic_write(path: str | Path, data: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp_path = target.with_suffix(target.suffix + ".tmp")
    temp_path.write_text(data, encoding="utf-8")
    os.replace(temp_path, target)


def run_from_paths(paths: dict[str, str | Path] | None = None, *, weights: dict[str, float] | None = None) -> dict:
    file_paths = {key: Path(value) for key, value in DEFAULTS.items()}
    if paths:
        file_paths.update({key: Path(value) for key, value in paths.items()})
    episodes = load_episodes(file_paths["parsed"])
    continuous = _rows(_read(file_paths["continuous"], []))
    semantic = _rows(_read(file_paths["semantic"], []))
    composite = _rows(_read(file_paths["composite"], []))
    report = build_final_rankings(
        episodes=episodes, continuous_rows=continuous, semantic_rows=semantic,
        composite_rows=composite, evaluations_data=_read(file_paths["evaluations"], {}) or {},
        pilot_report=_read(file_paths["pilot_report"], {}) or {},
        validation_data=_read(file_paths["validation"], {}) or {},
        review_data=_read(file_paths["reviews"], {}) or {},
        analysis_data=_read(file_paths["analysis"], {}) or {},
        discoveries_data=_read(file_paths["discoveries"], {}) or {}, weights=weights,
    )
    _atomic_write(file_paths["json"], json.dumps(report, ensure_ascii=False, indent=2))
    csv_rows = _csv_rows(report)
    if csv_rows:
        from io import StringIO
        buffer = StringIO()
        writer = csv.DictWriter(buffer, fieldnames=list(csv_rows[0]), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(csv_rows)
        csv_content = buffer.getvalue()
    else:
        csv_content = "candidate_id,source_type,ranking_status\n"
    _atomic_write(file_paths["csv"], csv_content)
    _atomic_write(file_paths["markdown"], render_markdown(report))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Rank only structurally valid candidates with validated blind-viewer evaluations")
    for key in ("parsed", "analysis", "discoveries", "continuous", "semantic", "composite", "evaluations", "pilot_report", "validation", "reviews", "json", "csv", "markdown"):
        parser.add_argument(f"--{key.replace('_', '-')}", default=DEFAULTS[key])
    parser.add_argument("--weights-json", help="Optional JSON object overriding the six default dimension weights")
    args = parser.parse_args()
    try:
        args_map = vars(args).copy()
        weights_path = args_map.pop("weights_json", None)
        weights = _read(weights_path) if weights_path else None
        if weights is not None and not isinstance(weights, dict):
            raise ValueError("--weights-json must point to a JSON object of dimension weights")
        report = run_from_paths(args_map, weights=weights)
    except (OSError, ValueError, ValidationError) as exc:
        parser.error(str(exc))
    print(f"Ranked: {len(report['leaderboards']['combined'])}; pending: {len(report['pending_candidates'])}; failed: {len(report['failed_evaluations'])}; structurally ineligible: {len(report['structurally_ineligible_candidates'])}")
    print(f"JSON: {args.json}\nCSV: {args.csv}\nReport: {args.markdown}")


if __name__ == "__main__":
    main()
