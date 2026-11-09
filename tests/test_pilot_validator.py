import copy
import json
import sys

import pytest

from promoscout.analysis.models import AnalysisEnvelope, DiscoveredEvent, EpisodeAnalysis
from promoscout.evaluation.pilot_validator import main, render_markdown_report, validate_pilot_artifacts
from promoscout.generation.continuous import generate_episode_candidates_from_discoveries
from promoscout.generation.preview import reconstruct_candidate_preview
from promoscout.schemas import ParsedEpisode, Scene, ScreenplayEvent


def make_episode(number=1):
    prefix = f"EP{number:03d}"
    events = [
        event(f"{prefix}_OPEN", "camera", 3, 5, "The door opens.", 1),
        event(f"{prefix}_HOOK", "dialogue", 5, 7, "You knew the truth!", 2, speaker="MIRA"),
        event(f"{prefix}_ACTION", "action", 12, 13, "Mira opens an envelope.", 3),
        event(f"{prefix}_END", "camera", 30, 31, "A second photograph appears.", 4),
    ]
    # Source event extends one second beyond the scene boundary. This is valid
    # source timing and must not be rejected by candidate validation.
    scene = Scene(scene_number=1, heading="INT. ROOM - NIGHT", start_seconds=0,
                  end_seconds=30, events=events)
    return ParsedEpisode(episode_number=number, logline="A woman searches for evidence.", scenes=[scene])


def event(event_id, event_type, start, end, content, line, speaker=None):
    return ScreenplayEvent(
        event_id=event_id, event_type=event_type, label=event_type.upper(), speaker=speaker,
        delivery="shaken" if speaker else None,
        start_timestamp=f"00:{start:05.2f}", end_timestamp=f"00:{end:05.2f}",
        start_seconds=start, end_seconds=end, content=content, raw_block=content,
        source_line_start=line, source_line_end=line + 1,
    )


def artifacts(episode=None):
    episode = episode or make_episode()
    prefix = f"EP{episode.episode_number:03d}"
    discoveries = [
        DiscoveredEvent(event_id=f"{prefix}_HOOK", role="hook", category="confrontation",
                        reason="An accusation changes the exchange.", confidence=.9,
                        narrative_context="Mira confronts someone about a past event."),
        DiscoveredEvent(event_id=f"{prefix}_END", role="potential_ending", category="unexpected_discovery",
                        reason="New evidence appears.", confidence=.8,
                        narrative_context="The image raises a new question."),
    ]
    analysis = EpisodeAnalysis(
        episode_number=episode.episode_number, episode_summary="A mystery unfolds.",
        important_characters=["MIRA"], character_relationships=[], main_conflict="Mira searches for evidence.",
        important_story_developments=[], major_revelations=[], emotional_turning_points=[],
        unresolved_questions=[], narrative_progression=[],
    )
    candidate = generate_episode_candidates_from_discoveries(episode, discoveries)[0]
    return episode, {str(episode.episode_number): analysis.model_dump()}, {
        str(episode.episode_number): [item.model_dump() for item in discoveries]
    }, {"candidates": [candidate.model_dump(mode="json")]}, candidate


def issue_codes(report, *, candidate_index=0):
    return {item["code"] for item in report["candidates"][candidate_index].get("errors", [])}


def validate(episode, analysis, discoveries, candidates):
    return validate_pilot_artifacts([episode], analysis, discoveries, candidates)


def test_valid_candidate_passes_structural_validation_and_boundary_overlap():
    episode, analysis, discoveries, candidates, _ = artifacts()
    report = validate(episode, analysis, discoveries, candidates)
    assert report["candidates"][0]["structural_status"] == "pass"
    assert report["validation_summary"]["total_errors"] == 0
    assert report["processing_summary"]["structurally_valid_candidate_count"] == 1


def test_invalid_event_id_is_reported_as_error():
    episode, analysis, discoveries, candidates, _ = artifacts()
    discoveries["1"][0]["event_id"] = "NOT_IN_SCREENPLAY"
    report = validate(episode, analysis, discoveries, candidates)
    assert any(i["code"] == "invalid_event_reference" for i in report["issues"])
    assert "hook_not_in_discoveries" in issue_codes(report)


def test_wrong_episode_reference_is_detected():
    episode, analysis, discoveries, candidates, _ = artifacts()
    other = make_episode(2)
    discoveries["1"][0]["event_id"] = "EP002_HOOK"
    report = validate_pilot_artifacts([episode, other], analysis, discoveries, candidates)
    assert any(i["code"] == "wrong_episode_reference" for i in report["issues"])


@pytest.mark.parametrize("end_seconds", [18.0, 70.0])
def test_durations_outside_20_to_60_seconds_are_rejected(end_seconds):
    episode, analysis, discoveries, candidates, candidate = artifacts()
    row = copy.deepcopy(candidates["candidates"][0])
    row["end_seconds"] = end_seconds
    row["duration_seconds"] = end_seconds - row["start_seconds"]
    candidates["candidates"] = [row]
    report = validate(episode, analysis, discoveries, candidates)
    assert "invalid_duration" in issue_codes(report)


def test_hook_lead_in_over_three_seconds_is_rejected():
    episode, analysis, discoveries, candidates, _ = artifacts()
    row = candidates["candidates"][0]
    row["start_seconds"] = 0
    row["start_timestamp"] = "00:00.00"
    row["duration_seconds"] = row["end_seconds"]
    row["hook_offset_seconds"] = 5
    report = validate(episode, analysis, discoveries, candidates)
    assert "hook_timing_violation" in issue_codes(report)


def test_duplicate_clip_windows_are_reported():
    episode, analysis, discoveries, candidates, _ = artifacts()
    duplicate = copy.deepcopy(candidates["candidates"][0])
    duplicate["candidate_id"] = "EP001_PROMO_002"
    candidates["candidates"].append(duplicate)
    report = validate(episode, analysis, discoveries, candidates)
    assert report["validation_summary"]["duplicate_windows"] == 1
    assert any(i["code"] == "duplicate_window" for i in report["candidates"][1]["warnings"])


def test_missing_preview_events_are_detected():
    episode, analysis, discoveries, candidates, _ = artifacts()
    candidates["candidates"][0]["included_event_ids"].remove("EP001_ACTION")
    report = validate(episode, analysis, discoveries, candidates)
    assert "missing_preview_events" in issue_codes(report)


def test_out_of_order_preview_events_are_detected():
    episode, analysis, discoveries, candidates, candidate = artifacts()
    preview = reconstruct_candidate_preview(episode, candidate)["events"]
    candidates["candidates"][0]["preview_events"] = list(reversed(preview))
    report = validate(episode, analysis, discoveries, candidates)
    assert "preview_out_of_order" in issue_codes(report)


def test_candidate_family_consistency_is_checked():
    episode, analysis, discoveries, candidates, _ = artifacts()
    duplicate = copy.deepcopy(candidates["candidates"][0])
    duplicate["candidate_id"] = "EP001_PROMO_002"
    duplicate["ending_event_id"] = "EP001_END"
    duplicate["end_event_id"] = "EP001_END"
    duplicate["candidate_family_id"] = "WRONG_FAMILY"
    candidates["candidates"].append(duplicate)
    report = validate(episode, analysis, discoveries, candidates)
    assert any(issue["code"] in {"hook_split_across_families", "non_deterministic_family_id"}
               for issue in report["issues"])


def test_human_quality_review_dimensions_are_pending_by_default():
    episode, analysis, discoveries, candidates, _ = artifacts()
    report = validate(episode, analysis, discoveries, candidates)
    assert report["quality_evaluation"]["status"] == "not_evaluated"
    assert all(item["status"].startswith("pending") for item in report["quality_evaluation"]["dimensions"])
    assert set(report["candidates"][0]["human_review"].values()) == {"pending"}


def test_report_generation_has_candidate_and_human_review_sections():
    episode, analysis, discoveries, candidates, _ = artifacts()
    markdown = render_markdown_report(validate(episode, analysis, discoveries, candidates))
    assert "## C. Candidate-by-candidate report" in markdown
    assert "## D. Human review" in markdown
    assert "pending" in markdown
    assert "EP001_PROMO_001" in markdown


def test_episode_filter_limits_summary_and_candidates():
    one, analysis1, discoveries1, candidate1, _ = artifacts(make_episode(1))
    two, analysis2, discoveries2, candidate2, _ = artifacts(make_episode(2))
    analysis = {**analysis1, **analysis2}
    discoveries = {**discoveries1, **discoveries2}
    candidates = {"candidates": candidate1["candidates"] + candidate2["candidates"]}
    report = validate_pilot_artifacts([one, two], analysis, discoveries, candidates, episode_number=2)
    assert report["processing_summary"]["episode_numbers"] == [2]
    assert report["processing_summary"]["original_event_count"] == 4
    assert [candidate["episode_number"] for candidate in report["candidates"]] == [2]


def test_duplicate_discovery_role_and_invalid_category_are_reported():
    episode, analysis, discoveries, candidates, _ = artifacts()
    duplicate = copy.deepcopy(discoveries["1"][0])
    invalid_category = copy.deepcopy(discoveries["1"][1])
    invalid_category["category"] = "made_up"
    discoveries["1"].extend([duplicate, invalid_category])
    report = validate(episode, analysis, discoveries, candidates)
    assert any(item["code"] == "duplicate_discovery" for item in report["issues"])
    assert any(item["code"] == "invalid_discovery_schema" for item in report["issues"])


def test_missing_analysis_and_discoveries_are_reported_clearly():
    episode, _, _, candidates, _ = artifacts()
    report = validate(episode, {}, {}, candidates)
    assert {item["code"] for item in report["issues"]} >= {"missing_episode_analysis", "missing_event_discoveries"}


def test_cli_writes_json_and_markdown_reports_from_local_fixtures(tmp_path, monkeypatch, capsys):
    episode, analysis, discoveries, candidates, _ = artifacts()
    parsed_path = tmp_path / "parsed.json"
    analysis_path = tmp_path / "analysis.json"
    discoveries_path = tmp_path / "discoveries.json"
    candidates_path = tmp_path / "candidates.json"
    parsed_path.write_text(json.dumps([episode.model_dump(mode="json")]), encoding="utf-8")
    analysis_path.write_text(json.dumps(analysis), encoding="utf-8")
    discoveries_path.write_text(json.dumps(discoveries), encoding="utf-8")
    candidates_path.write_text(json.dumps(candidates), encoding="utf-8")
    json_output, markdown_output = tmp_path / "report.json", tmp_path / "report.md"
    monkeypatch.setattr(sys, "argv", [
        "pilot_validator", "--episode-number", "1", "--parsed", str(parsed_path),
        "--analysis", str(analysis_path), "--discoveries", str(discoveries_path),
        "--candidates", str(candidates_path), "--json-output", str(json_output),
        "--markdown-output", str(markdown_output),
    ])
    main()
    assert json.loads(json_output.read_text())["processing_summary"]["structurally_valid_candidate_count"] == 1
    assert "Pilot Validation Report" in markdown_output.read_text()
    assert "0 error(s)" in capsys.readouterr().out

