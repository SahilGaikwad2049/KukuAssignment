import json

import pytest

from promoscout.analysis.series_analyzer import InvalidSeriesEventReference, analyze_series
from promoscout.analysis.series_models import (
    CharacterProfile, CharacterRelationship, CompositePromoOpportunity, PromoBeat,
    RelationshipChange, SeriesAnalysis, SourceReference, StoryArc, StoryDevelopment,
)
from promoscout.evaluation.blind_viewer import (
    BlindViewerAssessment, DimensionScore, EvidenceReference,
    build_composite_blind_view_input, evaluate_composite_candidate,
)
from promoscout.evaluation.pilot_validator import validate_composite_candidates
from promoscout.generation.composite import build_composite_candidates, reconstruct_composite_preview
from promoscout.schemas import ParsedEpisode, Scene, ScreenplayEvent
from promoscout.utils.cache import JsonCache


def make_episode(number):
    events = []
    for index, second in enumerate((10, 18, 27, 40), start=1):
        text = f"Synthetic event {index} from an unnamed story."
        events.append(ScreenplayEvent(
            event_id=f"EP{number:03d}_EV{index:03d}", event_type="dialogue" if index % 2 else "action",
            label="MIRA", speaker="MIRA", delivery="tense" if index == 1 else None,
            start_timestamp=f"00:{second:02d}.00", end_timestamp=f"00:{second + 1:02d}.00",
            start_seconds=second, end_seconds=second + 1, content=text, raw_block=text,
            source_line_start=index, source_line_end=index + 1,
        ))
    return ParsedEpisode(episode_number=number, logline="Synthetic-only fixture.",
                         scenes=[Scene(scene_number=1, heading="INT. SYNTHETIC ROOM", events=events)])


def series_fixture():
    episodes = [make_episode(number) for number in (1, 2, 3)]
    references = [SourceReference(episode_number=number, event_id=f"EP{number:03d}_EV001")
                  for number in (1, 2, 3)]
    opportunity = CompositePromoOpportunity(
        story_arc_id="synthetic_arc", narrative_theme="A warning becomes a discovery",
        narrative_rationale="Synthetic progression: warning, investigation, unresolved clue.",
        beats=[
            PromoBeat(episode_number=1, anchor_event_ids=["EP001_EV001"], beat_role="hook", reason="Synthetic warning.", spoiler_level="low"),
            PromoBeat(episode_number=2, anchor_event_ids=["EP002_EV001"], beat_role="escalation", reason="Synthetic investigation.", spoiler_level="moderate"),
            PromoBeat(episode_number=3, anchor_event_ids=["EP003_EV001"], beat_role="ending", reason="Synthetic unresolved clue.", spoiler_level="major"),
        ], hook_beat_index=0, ending_beat_index=2,
        spoiler_warnings=["Synthetic major revelation warning."],
        uncertainty_warnings=["Synthetic fixture; narrative connection needs review."],
    )
    analysis = SeriesAnalysis(
        analyzed_episode_numbers=[1, 2, 3], series_synopsis="Synthetic-only series fixture.",
        character_registry=[CharacterProfile(
            canonical_name="MIRA", aliases=["M."], role="Synthetic investigator.",
            source_references=references[:1], relationships=[CharacterRelationship(
                other_character="ALEX", relationship="Trust becomes uncertain.",
                source_references=references[:1], changes=[RelationshipChange(
                    episode_number=2, description="Synthetic relationship shift.", source_references=references[1:2])])])],
        story_arcs=[StoryArc(
            story_arc_id="synthetic_arc", title="Synthetic warning arc", overview="A fixture-only arc.",
            characters=["MIRA"], developments=[StoryDevelopment(
                episode_number=number, development="Synthetic fixture development.", development_type="turn",
                source_references=[references[number - 1]], spoiler_level="low") for number in (1, 2, 3)],
        )],
        composite_promo_opportunities=[opportunity],
    )
    return episodes, analysis


def test_synthetic_three_episode_composite_has_valid_duration_and_source_integrity():
    episodes, analysis = series_fixture()
    candidates = build_composite_candidates(episodes, analysis, candidate_origin="synthetic")
    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.candidate_origin == "synthetic"
    assert {segment.episode_number for segment in candidate.segments} == {1, 2, 3}
    assert 20 <= candidate.total_duration_seconds <= 60
    assert candidate.total_duration_seconds == sum(segment.duration_seconds for segment in candidate.segments)
    assert candidate.hook_event_id == "EP001_EV001"
    assert candidate.potential_ending_event_id == "EP003_EV001"
    assert candidate.segments[0].relative_start_seconds == 0
    assert candidate.segments[0].events[0].original_start_timestamp == "00:10.00"
    assert candidate.spoiler_warnings
    assert any("needs review" in warning for warning in candidate.coherence_warnings)
    assert analysis.character_registry[0].aliases == ["M."]
    assert analysis.character_registry[0].relationships[0].changes[0].episode_number == 2
    preview = reconstruct_composite_preview(candidate)
    assert [event["episode_number"] for event in preview["events"]] == [1, 1, 2, 2, 3, 3]
    report = validate_composite_candidates(episodes, {"candidates": [candidate.model_dump(mode="json")]})
    assert report["passing_count"] == 1
    assert report["candidates"][0]["structural_status"] == "pass"


def test_composite_rejects_invalid_event_ids_and_single_episode_plan():
    episodes, analysis = series_fixture()
    analysis.composite_promo_opportunities[0].beats[0].anchor_event_ids = ["HALLUCINATED"]
    with pytest.raises(InvalidSeriesEventReference):
        build_composite_candidates(episodes, analysis)
    episodes, analysis = series_fixture()
    for beat, event_id in zip(analysis.composite_promo_opportunities[0].beats,
                              ("EP001_EV001", "EP001_EV002", "EP001_EV003")):
        beat.episode_number = 1
        beat.anchor_event_ids = [event_id]
    analysis.story_arcs[0].developments = [analysis.story_arcs[0].developments[0]]
    with pytest.raises(ValueError, match="at least two episodes"):
        build_composite_candidates(episodes, analysis)


def test_composite_structural_validation_catches_wrong_episode_id_and_duration():
    episodes, analysis = series_fixture()
    row = build_composite_candidates(episodes, analysis)[0].model_dump(mode="json")
    row["segments"][0]["episode_number"] = 99
    row["total_duration_seconds"] = 61
    report = validate_composite_candidates(episodes, {"candidates": [row]})
    assert report["failing_count"] == 1
    errors = " ".join(report["candidates"][0]["errors"])
    assert "unavailable episode 99" in errors
    assert "outside 20–60 seconds" in errors


def test_composite_blind_input_is_anonymous_and_contains_only_clip_content(tmp_path):
    episodes, analysis = series_fixture()
    candidate = build_composite_candidates(episodes, analysis)[0]
    payload, mapping = build_composite_blind_view_input(episodes, candidate)
    serialized = json.dumps(payload)
    assert "episode_number" not in serialized and "EP001" not in serialized and "Episode 1" not in serialized
    assert "story_arc_id" not in serialized and "Mira" not in serialized
    assert [row["event_id"] for row in payload["events"]] == [f"CLIP_EVENT_{index:04d}" for index in range(1, 7)]
    assert mapping["CLIP_EVENT_0001"] == "1:EP001_EV001"

    class FakeLLM:
        model = "mock-composite-model"
        prompt = ""
        def generate_json_schema(self, prompt, response_model):
            self.prompt = prompt
            dimension = DimensionScore(score=4, rationale="Supported only by the supplied clip.", evidence=[
                EvidenceReference(event_id="CLIP_EVENT_0001", excerpt="Synthetic event 1 from an unnamed story.", why_it_matters="Opens with a warning.")])
            return BlindViewerAssessment(
                standalone_story_reconstruction="Someone receives a warning.",
                characters_the_viewer_believes_are_involved=[], directly_shown_or_stated=[],
                reasonable_inferences=[], unclear_or_unestablished=[], main_conflict_as_understood="An investigation begins.",
                perceived_stakes="Unclear.", confusing_elements=[], missing_context_the_viewer_wishes_for=[],
                standalone_clarity=dimension, hook_effectiveness=dimension, conflict_comprehension=dimension,
                stakes_clarity=dimension, cliffhanger_effectiveness=dimension, desire_to_continue=dimension,
                overall_viewer_verdict="strong", confidence=.7, verdict_explanation="Synthetic fixture response.",
            )
    llm = FakeLLM()
    result = evaluate_composite_candidate(episodes, candidate, llm=llm, cache=JsonCache(tmp_path / "cache"))
    assert result.candidate_source == "composite" and result.episode_number is None
    assert result.composite_source_event_mapping["CLIP_EVENT_0001"] == "1:EP001_EV001"
    assert "EP001" not in llm.prompt and "series_synopsis" not in llm.prompt


def test_series_analysis_cache_reuse_and_invalid_reference(tmp_path):
    episodes, analysis = series_fixture()
    analysis_map = {str(n): {"episode_number": n, "episode_summary": "Synthetic.",
                             "important_characters": [], "character_relationships": [], "main_conflict": "Synthetic.",
                             "important_story_developments": [], "major_revelations": [],
                             "emotional_turning_points": [], "unresolved_questions": [], "narrative_progression": []}
                    for n in (1, 2, 3)}
    discoveries = {str(n): [] for n in (1, 2, 3)}
    class FakeLLM:
        model = "mock-series-model"
        calls = 0
        def generate_json_schema(self, prompt, response_model):
            self.calls += 1
            # Three bounded episode chunks and then the final synthesis.
            if "SYNTHESIZE" in prompt:
                return analysis
            chunk = json.loads(prompt.split("SERIES EPISODE MATERIAL:\n", 1)[1])
            return SeriesAnalysis(analyzed_episode_numbers=[row["episode_number"] for row in chunk["episodes"]],
                                  series_synopsis="Partial synthetic.")
    llm = FakeLLM()
    cache = JsonCache(tmp_path / "cache")
    result = analyze_series(episodes, analysis_map, discoveries, llm=llm, cache=cache, max_chunk_chars=1000)
    assert result.series_synopsis == "Synthetic-only series fixture."
    assert llm.calls == 4  # three cached source chunks plus cross-chunk synthesis
    calls = llm.calls
    assert analyze_series(episodes, analysis_map, discoveries, llm=llm, cache=cache, max_chunk_chars=1000)
    assert llm.calls == calls
    broken = analysis.model_copy(deep=True)
    broken.story_arcs = []
    broken.composite_promo_opportunities[0].beats[0].anchor_event_ids = ["NOPE"]
    from promoscout.analysis.series_analyzer import validate_series_references
    with pytest.raises(InvalidSeriesEventReference):
        validate_series_references(broken, episodes)


def test_series_and_composite_dry_runs_are_read_only_and_do_not_construct_gemini(tmp_path, monkeypatch, capsys):
    import sys
    import promoscout.analysis.episode_analyzer as episode_analyzer
    import promoscout.analysis.series_analyzer as series_cli
    import promoscout.generation.composite as composite_cli
    episodes, _analysis = series_fixture()
    parsed_path = tmp_path / "parsed.json"
    parsed_path.write_text(json.dumps([episode.model_dump(mode="json") for episode in episodes]))
    empty_analysis, empty_discoveries = tmp_path / "analysis.json", tmp_path / "discoveries.json"
    empty_analysis.write_text("{}")
    empty_discoveries.write_text("{}")
    cache_path = tmp_path / "cache-do-not-create"
    class Settings:
        cache_dir = cache_path
        max_chunk_chars = 900
        gemini_model = "mock-series-model"
    monkeypatch.setattr(series_cli, "get_settings", lambda: Settings())
    monkeypatch.setattr(episode_analyzer, "GeminiClient", lambda **kwargs: pytest.fail("dry-run built Gemini client"))
    series_output, series_md = tmp_path / "series.json", tmp_path / "series.md"
    monkeypatch.setattr(sys, "argv", ["series_analyzer", "--parsed", str(parsed_path),
        "--analysis", str(empty_analysis), "--discoveries", str(empty_discoveries),
        "--output", str(series_output), "--markdown", str(series_md), "--limit", "2", "--dry-run"])
    series_cli.main()
    printed = capsys.readouterr().out
    assert "Episodes available: 3" in printed and "Selected episodes: 1, 2" in printed
    assert "Configured model: mock-series-model" in printed
    assert not cache_path.exists() and not series_output.exists() and not series_md.exists()

    composite_output = tmp_path / "composites.json"
    monkeypatch.setattr(sys, "argv", ["composite", "--parsed", str(parsed_path),
        "--series-analysis", str(tmp_path / "missing-series.json"),
        "--output", str(composite_output), "--dry-run"])
    composite_cli.main()
    printed = capsys.readouterr().out
    assert "Episodes available: 3" in printed
    assert "Series-level composite opportunities: 0" in printed
    assert "no output written" in printed
    assert not composite_output.exists()
