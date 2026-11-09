import json
import sys

from promoscout.evaluation.blind_viewer import _atomic_write_json, main
from promoscout.evaluation.pilot import build_pilot_report, sample_candidate_records, save_pilot_report
from promoscout.generation.continuous import PromoCandidate


def candidate(episode, index, *, hook_type="dialogue", start=None, duration=25):
    start = float(start if start is not None else index * 10)
    end = start + duration
    return PromoCandidate(
        candidate_id=f"EP{episode:03d}_PROMO_{index:03d}", candidate_family_id=f"FAMILY_{episode:03d}",
        episode_number=episode, start_event_id=f"S_{episode}_{index}", end_event_id=f"E_{episode}_{index}",
        start_timestamp="00:00.00", end_timestamp="00:25.00", start_seconds=start, end_seconds=end,
        duration_seconds=duration, hook_event_id=f"H_{episode}_{index}", hook_type=hook_type,
        hook_text="A question is asked.", hook_start_timestamp="00:00.00", hook_end_timestamp="00:01.00",
        hook_start_seconds=start, hook_end_seconds=start + 1, hook_offset_seconds=0,
        ending_event_id=f"E_{episode}_{index}", ending_text="A clue is revealed.",
        hook_score=.2, ending_score=.3, heuristic_score=.25,
    )


def records_for(episode_count=15, per_episode=2, source="heuristic"):
    rows = []
    for episode in range(1, episode_count + 1):
        for index in range(1, per_episode + 1):
            rows.append({"candidate": candidate(episode, index, hook_type="dialogue" if index == 1 else "visual"), "source": source})
    return rows


def test_sample_is_reproducible_and_spread_across_distinct_episodes():
    first, manifest = sample_candidate_records(records_for(), sample_episodes=10, seed=42)
    second, _ = sample_candidate_records(records_for(), sample_episodes=10, seed=42)
    assert [row["candidate"].candidate_id for row in first] == [row["candidate"].candidate_id for row in second]
    assert len(manifest["selected_episode_numbers"]) == 10
    assert len(set(manifest["selected_episode_numbers"])) == 10
    assert manifest["selected_episode_numbers"][0] <= 3
    assert manifest["selected_episode_numbers"][-1] >= 13


def test_fewer_available_episodes_is_reported_and_one_candidate_is_default():
    selected, manifest = sample_candidate_records(records_for(episode_count=4), sample_episodes=10, seed=42)
    assert manifest["requested_episode_count"] == 10
    assert manifest["selected_episode_count"] == 4
    assert len(selected) == 4
    assert len({row["candidate"].episode_number for row in selected}) == 4


def test_two_candidates_per_episode_and_metadata_variation():
    selected, manifest = sample_candidate_records(records_for(episode_count=8), sample_episodes=4,
                                                   candidates_per_episode=2, seed=42)
    assert len(selected) == 8
    by_episode = {}
    for row in selected:
        by_episode.setdefault(row["candidate"].episode_number, []).append(row["candidate"])
    assert len(by_episode) == 4
    assert all(len(items) == 2 for items in by_episode.values())
    assert all(len({item.hook_type for item in items}) == 2 for items in by_episode.values())
    assert manifest["candidates_per_episode_requested"] == 2


def test_duplicate_windows_are_excluded_and_sources_stay_separate():
    duplicate = {"candidate": candidate(1, 99, start=10), "source": "semantic"}
    input_rows = records_for(episode_count=4, per_episode=1, source="heuristic") + [duplicate]
    selected, manifest = sample_candidate_records(input_rows, sample_episodes=4, seed=4)
    assert len(selected) == 4
    assert {row["source"] for row in selected} == {"heuristic"}
    assert manifest["candidate_sources"] == ["heuristic", "semantic"]
    assert len({(row["candidate"].episode_number, row["candidate"].start_seconds, row["candidate"].end_seconds) for row in selected}) == 4


def test_two_sources_are_only_mixed_when_both_are_explicitly_loaded():
    heuristic = records_for(episode_count=4, per_episode=1, source="heuristic")
    semantic = [
        {"candidate": candidate(episode, 50, hook_type="action", start=100), "source": "semantic"}
        for episode in range(1, 5)
    ]
    selected, _ = sample_candidate_records(heuristic, sample_episodes=4, seed=42)
    assert {row["source"] for row in selected} == {"heuristic"}
    mixed, manifest = sample_candidate_records(heuristic + semantic, sample_episodes=4,
                                               candidates_per_episode=2, seed=42)
    assert manifest["selection_uses_evaluation_scores"] is False
    assert {row["source"] for row in mixed} == {"heuristic", "semantic"}


def test_dry_run_writes_manifest_without_gemini_or_evaluation_output(tmp_path, monkeypatch, capsys):
    import promoscout.evaluation.blind_viewer as blind

    parsed_path = tmp_path / "parsed.json"
    candidate_path = tmp_path / "candidates.json"
    episode_row = {
        "episode_number": 1, "logline": None, "continuity_notes": None, "source_file": "ep1.txt", "warnings": [],
        "scenes": [{"scene_number": 1, "heading": "Room", "start_timestamp": "00:00.00", "end_timestamp": "00:30.00",
                    "start_seconds": 0, "end_seconds": 30, "events": [{
                        "event_id": "E1", "event_type": "dialogue", "label": "DIALOGUE", "speaker": "A", "delivery": None,
                        "start_timestamp": "00:01.00", "end_timestamp": "00:02.00", "start_seconds": 1, "end_seconds": 2,
                        "content": "A question?", "raw_block": "A question?", "source_line_start": 1, "source_line_end": 2,
                    }]}],
    }
    parsed_path.write_text(json.dumps([episode_row]))
    candidate_row = candidate(1, 1, start=1).model_dump(mode="json")
    candidate_row.update({"start_event_id": "E1", "end_event_id": "E1", "hook_event_id": "E1", "ending_event_id": "E1",
                          "hook_start_seconds": 1, "hook_end_seconds": 2, "end_seconds": 26, "duration_seconds": 25})
    candidate_path.write_text(json.dumps([candidate_row]))
    output = tmp_path / "evaluations.json"
    output.write_text('{"keep": true}')
    output_before = output.read_bytes()
    manifest = tmp_path / "selection.json"

    class Settings:
        gemini_model = "mock-model"
        cache_dir = tmp_path / "cache"
    monkeypatch.setattr(blind, "get_settings", lambda: Settings())
    monkeypatch.setattr(blind, "GeminiClient", lambda: (_ for _ in ()).throw(AssertionError("dry run built client")))
    monkeypatch.setattr(sys, "argv", [
        "blind_viewer", "--source", "heuristic", "--parsed", str(parsed_path),
        "--heuristic-candidates", str(candidate_path), "--sample-episodes", "10",
        "--candidates-per-episode", "1", "--sample-seed", "42", "--dry-run",
        "--selection-manifest", str(manifest), "--output", str(output),
    ])
    main()
    printed = capsys.readouterr().out
    stored_manifest = json.loads(manifest.read_text())
    assert "Sampled 1 of 10 requested episode(s)" in printed
    assert "Selected candidates: 1" in printed
    assert "Estimated requests: 1" in printed
    assert stored_manifest["selected_candidates"][0]["candidate_id"] == "EP001_PROMO_001"
    assert stored_manifest["selected_candidates"][0]["candidate_source"] == "heuristic"
    assert output.read_bytes() == output_before
    assert not (tmp_path / "cache").exists()


def mock_eval(candidate_id, episode, source="heuristic", scores=None, verdict="mixed"):
    scores = scores or {name: 3 for name in (
        "standalone_clarity", "hook_effectiveness", "conflict_comprehension",
        "stakes_clarity", "cliffhanger_effectiveness", "desire_to_continue",
    )}
    dimensions = {name: {"score": value, "rationale": "Grounded reason.", "evidence": []} for name, value in scores.items()}
    return {"candidate_id": candidate_id, "candidate_source": source, "episode_number": episode,
            "assessment": {**dimensions, "overall_viewer_verdict": verdict, "confidence": .75,
                           "verdict_explanation": "The conflict is clear but the ending leaves some uncertainty.",
                           "missing_context_the_viewer_wishes_for": ["What happened before this scene?"],
                           "confusing_elements": []}}


def test_pilot_report_computes_distributions_and_keeps_human_decision_separate(tmp_path):
    selected = records_for(episode_count=2, per_episode=1)
    scores = {name: value for name, value in zip((
        "standalone_clarity", "hook_effectiveness", "conflict_comprehension",
        "stakes_clarity", "cliffhanger_effectiveness", "desire_to_continue",
    ), (1, 2, 3, 4, 5, 3))}
    payload = {"evaluations": {
        "heuristic:EP001_PROMO_001": mock_eval("EP001_PROMO_001", 1, scores=scores, verdict="strong"),
        "heuristic:EP002_PROMO_001": mock_eval("EP002_PROMO_001", 2, verdict="weak"),
    }}
    report = build_pilot_report(selected, payload, [], model_id="mock-model", cached_count=1,
                                human_decisions={"heuristic:EP001_PROMO_001": {"decision": "approved"}})
    assert report["overall_results"]["successfully_evaluated"] == 2
    assert report["overall_results"]["episodes_represented"] == 2
    assert report["score_distributions"]["standalone_clarity"] == {
        "average": 2, "median": 2, "minimum": 1, "maximum": 3,
        "distribution": {"1": 1, "2": 0, "3": 1, "4": 0, "5": 0},
    }
    assert report["verdict_distribution"] == {"strong": 1, "mixed": 0, "weak": 1}
    assert report["episode_level_results"][0]["human_decision"] == "approved"
    assert report["human_judgments_are_separate"] is True
    save_pilot_report(report, tmp_path / "report.json", tmp_path / "report.md")
    assert "Score distributions" in (tmp_path / "report.md").read_text()
    assert json.loads((tmp_path / "report.json").read_text())["candidate_source"] == "heuristic"
