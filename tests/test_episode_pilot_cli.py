import json
import sys

import pytest

from promoscout.analysis.episode_analyzer import analyze_episodes, select_episodes
from promoscout.analysis.models import AnalysisEnvelope, DiscoveredEvent, EpisodeAnalysis
from promoscout.generation.continuous import generate_candidates_from_discoveries
from promoscout.schemas import ParsedEpisode, Scene, ScreenplayEvent
from promoscout.utils.cache import JsonCache


def make_episode(number=1, prefix=None):
    prefix = prefix or f"EP{number}"
    events = [
        ScreenplayEvent(event_id=f"{prefix}_HOOK", event_type="dialogue", label="MIRA",
                        speaker="MIRA", delivery="shaken", start_timestamp="00:05.00",
                        end_timestamp="00:07.00", start_seconds=5, end_seconds=7,
                        content="You knew the truth!", raw_block="line", source_line_start=1, source_line_end=2),
        ScreenplayEvent(event_id=f"{prefix}_END", event_type="camera", label="CAMERA",
                        speaker=None, delivery=None, start_timestamp="00:30.00",
                        end_timestamp="00:31.00", start_seconds=30, end_seconds=31,
                        content="A hidden letter is revealed.", raw_block="shot", source_line_start=3, source_line_end=4),
    ]
    return ParsedEpisode(episode_number=number, logline="A mystery unfolds.",
                         continuity_notes="", scenes=[Scene(scene_number=1,
                         heading="INT. ROOM - NIGHT", start_seconds=0, end_seconds=40, events=events)])


def fake_response(number=1, prefix=None):
    prefix = prefix or f"EP{number}"
    return AnalysisEnvelope(
        episode_analysis=EpisodeAnalysis(
            episode_number=number, episode_summary="A mystery unfolds.", important_characters=["MIRA"],
            character_relationships=[], main_conflict="Find the truth.", important_story_developments=[],
            major_revelations=[], emotional_turning_points=[], unresolved_questions=[], narrative_progression=[],
        ),
        potential_events=[
            DiscoveredEvent(event_id=f"{prefix}_HOOK", role="hook", category="mystery", reason="Raises a question.", confidence=.9, narrative_context="Mira makes an accusation."),
            DiscoveredEvent(event_id=f"{prefix}_END", role="potential_ending", category="unexpected_discovery", reason="A clue appears.", confidence=.8, narrative_context="The letter may change the story."),
        ],
    )


class FakeLLM:
    model = "mock-model"
    def __init__(self, response):
        self.response = response
        self.calls = 0
    def generate(self, prompt, response_model):
        self.calls += 1
        return self.response


def configure_tmp(monkeypatch, tmp_path):
    import promoscout.analysis.episode_analyzer as analyzer
    class Settings:
        cache_dir = tmp_path / "cache"
        gemini_model = "mock-model"
        gemini_api_key = ""
        llm_provider = "gemini"
        max_chunk_chars = 24000
    monkeypatch.setattr(analyzer, "get_settings", lambda: Settings())
    return analyzer


def test_select_specific_episode_and_first_n():
    episodes = [make_episode(1), make_episode(2), make_episode(3)]
    assert [e.episode_number for e in select_episodes(episodes, episode_number=2)] == [2]
    assert [e.episode_number for e in select_episodes(episodes, limit=2)] == [1, 2]


def test_selection_options_reject_together_and_invalid_episode_is_clear():
    episodes = [make_episode(1), make_episode(2)]
    with pytest.raises(ValueError, match="cannot be used together"):
        select_episodes(episodes, episode_number=1, limit=1)
    with pytest.raises(ValueError, match=r"Episode 9 was not found.*1, 2"):
        select_episodes(episodes, episode_number=9)


def test_cli_parser_rejects_mutually_exclusive_options(monkeypatch, capsys):
    import promoscout.analysis.episode_analyzer as analyzer
    monkeypatch.setattr(sys, "argv", ["episode_analyzer", "input.json", "--episode-number", "1", "--limit", "1"])
    with pytest.raises(SystemExit) as exc:
        analyzer.main()
    assert exc.value.code == 2
    assert "not allowed with argument" in capsys.readouterr().err


def test_dry_run_never_constructs_gemini_or_changes_outputs(tmp_path, monkeypatch, capsys):
    analyzer = configure_tmp(monkeypatch, tmp_path)
    episode = make_episode()
    input_file = tmp_path / "parsed.json"
    input_file.write_text(json.dumps([episode.model_dump(mode="json")]))
    analysis_file = tmp_path / "analysis.json"
    discoveries_file = tmp_path / "discoveries.json"
    analysis_file.write_text('{"9": {"kept": true}}')
    discoveries_file.write_text('{"9": []}')
    before = (analysis_file.read_bytes(), discoveries_file.read_bytes())
    monkeypatch.setattr(analyzer, "GeminiClient", lambda: pytest.fail("Gemini client must not be constructed"))
    monkeypatch.setattr(sys, "argv", [
        "episode_analyzer", str(input_file), "--episode-number", "1", "--dry-run",
        "--analysis-output", str(analysis_file), "--discoveries-output", str(discoveries_file),
    ])
    analyzer.main()
    output = capsys.readouterr().out
    assert "Episode 1: 2 events" in output
    assert "Configured model:" in output
    assert "Estimated total requests:" in output
    assert (analysis_file.read_bytes(), discoveries_file.read_bytes()) == before
    assert not analyzer.get_settings().cache_dir.exists()


def test_selected_episode_updates_outputs_without_erasing_other_results(tmp_path, monkeypatch):
    analyzer = configure_tmp(monkeypatch, tmp_path)
    episode = make_episode(1)
    analysis_file, discoveries_file = tmp_path / "analysis.json", tmp_path / "discoveries.json"
    analysis_file.write_text(json.dumps({"2": {"episode_number": 2, "episode_summary": "Existing"}}))
    discoveries_file.write_text(json.dumps({"2": [{"event_id": "KEEP", "role": "hook"}]}))
    llm = FakeLLM(fake_response())
    analyze_episodes([episode], llm=llm, analysis_path=analysis_file, discoveries_path=discoveries_file)
    assert set(json.loads(analysis_file.read_text())) == {"1", "2"}
    assert json.loads(analysis_file.read_text())["2"]["episode_summary"] == "Existing"
    assert json.loads(discoveries_file.read_text())["2"][0]["event_id"] == "KEEP"
    analyze_episodes([episode], llm=llm, analysis_path=analysis_file, discoveries_path=discoveries_file)
    assert llm.calls == 1


def test_discoveries_for_selected_episode_generate_only_its_candidates():
    episodes = [make_episode(1), make_episode(2)]
    discoveries = {"1": [d.model_dump() for d in fake_response(1).potential_events]}
    selected = select_episodes(episodes, episode_number=1)
    candidates = generate_candidates_from_discoveries(selected, discoveries)
    assert candidates
    assert {candidate.episode_number for candidate in candidates} == {1}
    assert all(candidate.hook_event_id.startswith("EP1_") for candidate in candidates)
    assert all(20 <= candidate.duration_seconds <= 60 for candidate in candidates)

