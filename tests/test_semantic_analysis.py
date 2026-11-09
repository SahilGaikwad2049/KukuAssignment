import json

import pytest

from promoscout.analysis.episode_analyzer import _build_chunks, _scene_payload, analyze_episode, analyze_episodes, select_episodes
from promoscout.analysis.event_tagger import InvalidEventReference, validate_discoveries
from promoscout.analysis.models import AnalysisEnvelope, DiscoveredEvent, EpisodeAnalysis
from promoscout.generation.continuous import generate_episode_candidates_from_discoveries
from promoscout.generation.preview import reconstruct_candidate_preview
from promoscout.schemas import ParsedEpisode, Scene, ScreenplayEvent
from promoscout.utils.cache import JsonCache
from promoscout.utils.llm_client import GeminiClient, LLMConfigurationError


def event(event_id, event_type, start, end, content, line=1, speaker=None, delivery=None):
    return ScreenplayEvent(
        event_id=event_id, event_type=event_type, label=event_type.upper(), speaker=speaker,
        delivery=delivery, start_timestamp=f"00:{start:05.2f}", end_timestamp=f"00:{end:05.2f}",
        start_seconds=start, end_seconds=end, content=content, raw_block=content,
        source_line_start=line, source_line_end=line + 1,
    )


def make_episode():
    events = [
        event("OPEN", "camera", 3, 5, "A figure watches from the doorway.", 1),
        event("HOOK", "dialogue", 5, 7, "You were there that night.", 2, "MIRA", "shaken"),
        event("ACTION", "action", 12, 13, "Mira tears open the envelope.", 3),
        event("END", "camera", 30, 31, "A second photograph slips out.", 4),
    ]
    return ParsedEpisode(episode_number=1, logline="A woman searches for the truth.",
                         continuity_notes="Mira trusts nobody.",
                         scenes=[Scene(scene_number=1, heading="INT. ROOM - NIGHT",
                                       start_seconds=0, end_seconds=40, events=events)])


def result(ids=("HOOK", "END")):
    return AnalysisEnvelope(
        episode_analysis=EpisodeAnalysis(
            episode_number=1, episode_summary="Mira finds evidence.",
            important_characters=["MIRA"], character_relationships=["Mira distrusts the watcher."],
            main_conflict="Mira searches for the truth.", important_story_developments=["A photograph appears."],
            major_revelations=[], emotional_turning_points=["Mira becomes alarmed."],
            unresolved_questions=["Who took the photograph?"], narrative_progression=["Suspicion grows."],
        ),
        potential_events=[
            DiscoveredEvent(event_id=ids[0], role="hook", category="mystery", reason="Raises suspicion.", confidence=.9, narrative_context="The line connects Mira to an unseen event."),
            DiscoveredEvent(event_id=ids[1], role="potential_ending", category="unexpected_discovery", reason="New evidence appears.", confidence=.8, narrative_context="The photograph creates a new question."),
        ],
    )


class FakeLLM:
    model = "mock-model"
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0
    def generate(self, prompt, response_model):
        self.calls += 1
        value = self.responses.pop(0)
        return value if isinstance(value, response_model) else response_model.model_validate(value)


def test_valid_combined_episode_analysis_and_discovery(tmp_path):
    llm = FakeLLM([result()])
    response = analyze_episode(make_episode(), llm=llm, cache=JsonCache(tmp_path))
    assert response.episode_analysis.important_characters == ["MIRA"]
    assert [d.event_id for d in response.potential_events] == ["HOOK", "END"]
    assert llm.calls == 1


def test_discovery_payload_uses_original_episode_events_and_preserves_metadata():
    episode = make_episode()
    payload = _scene_payload(episode)
    rows = payload["scenes"][0]["events"]
    assert [row["event_id"] for row in rows] == [event.event_id for event in episode.events]
    assert rows[1]["content"] == "You were there that night."
    assert rows[1]["speaker"] == "MIRA"
    assert rows[1]["delivery"] == "shaken"
    assert rows[0]["event_type"] == "camera"
    assert rows[0]["start_timestamp"] == "00:03.00"
    assert payload["scenes"][0]["heading"] == "INT. ROOM - NIGHT"
    # Candidate windows are not part of the analyzer's input format.
    assert "candidates" not in payload


def test_scene_chunking_retains_every_original_event_id():
    episode = make_episode()
    chunks = _build_chunks(episode, 850)
    assert chunks
    chunk_ids = [event["event_id"] for chunk in chunks for scene in chunk["scenes"] for event in scene["events"]]
    assert chunk_ids == [event.event_id for event in episode.events]
    assert select_episodes([episode, episode.model_copy(update={"episode_number": 2})], episode_number=1) == [episode]


def test_hallucinated_event_id_is_rejected(tmp_path):
    bad = result().model_dump()
    bad["potential_events"][0]["event_id"] = "NOT_IN_INPUT"
    with pytest.raises(InvalidEventReference):
        analyze_episode(make_episode(), llm=FakeLLM([bad]), cache=JsonCache(tmp_path))


def test_duplicate_event_discoveries_are_deduplicated():
    item = DiscoveredEvent(event_id="E1", role="hook", category="visual_intrigue",
                           reason="Visual question.", confidence=.8, narrative_context="A clue is shown.")
    assert validate_discoveries([item, item], {"E1"}) == [item]


@pytest.mark.parametrize("event_type,category", [
    ("dialogue", "confrontation"), ("camera", "visual_intrigue"),
    ("action", "unexpected_action"),
])
def test_multiple_hook_categories_and_event_types_supported(event_type, category):
    entry = DiscoveredEvent(event_id="E", role="hook", category=category, reason="Story turn.",
                            confidence=.7, narrative_context=event_type)
    assert validate_discoveries([entry], {"E"}) == [entry]


def test_missing_api_key_fails_cleanly():
    client = GeminiClient(api_key="", model="configured-model")
    with pytest.raises(LLMConfigurationError, match="GEMINI_API_KEY"):
        client.generate("prompt", AnalysisEnvelope)


def test_semantic_candidate_generation_constraints_and_family():
    episode = make_episode()
    discoveries = [
        {"event_id": "HOOK", "role": "hook", "category": "mystery", "reason": "Raises a question.", "confidence": .9, "narrative_context": "A troubling accusation."},
        {"event_id": "END", "role": "potential_ending", "category": "unexpected_discovery", "reason": "A clue appears.", "confidence": .8, "narrative_context": "A photograph is revealed."},
    ]
    candidates = generate_episode_candidates_from_discoveries(episode, discoveries)
    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.hook_event_id == "HOOK"
    assert candidate.hook_offset_seconds <= 3
    assert 20 <= candidate.duration_seconds <= 60
    assert candidate.start_event_id == "OPEN"
    assert candidate.included_event_ids == ["OPEN", "HOOK", "ACTION", "END"]


def test_semantic_candidate_enforces_hook_offset_and_groups_endings():
    episode = make_episode()
    episode.scenes[0].events.append(event("HOOK2", "action", 10, 11, "He turns away.", 5))
    episode.scenes[0].events.append(event("END2", "sound_fx", 33, 34, "A phone rings.", 6))
    discoveries = [
        {"event_id": "HOOK", "role": "hook", "category": "mystery", "reason": "One.", "confidence": .9, "narrative_context": "x"},
        {"event_id": "HOOK2", "role": "hook", "category": "power_reversal", "reason": "Two.", "confidence": .8, "narrative_context": "y"},
        {"event_id": "END", "role": "potential_ending", "category": "uncertain_outcome", "reason": "End.", "confidence": .8, "narrative_context": "z"},
        {"event_id": "END2", "role": "potential_ending", "category": "imminent_danger", "reason": "Another possible end.", "confidence": .7, "narrative_context": "w"},
    ]
    candidates = generate_episode_candidates_from_discoveries(episode, discoveries)
    assert {c.hook_event_id for c in candidates} == {"HOOK", "HOOK2"}
    assert all(c.hook_offset_seconds <= 3 for c in candidates)
    hook_variants = [c for c in candidates if c.hook_event_id == "HOOK"]
    assert len(hook_variants) == 2
    assert len({c.candidate_family_id for c in hook_variants}) == 1


def test_preview_contains_complete_chronological_events_and_relative_times():
    episode = make_episode()
    candidate = generate_episode_candidates_from_discoveries(episode, [
        {"event_id": "HOOK", "role": "hook", "category": "confrontation", "reason": "x", "confidence": .9, "narrative_context": "x"},
        {"event_id": "END", "role": "potential_ending", "category": "unexpected_discovery", "reason": "y", "confidence": .8, "narrative_context": "y"},
    ])[0]
    rows = reconstruct_candidate_preview(episode, candidate)["events"]
    assert [row["event_id"] for row in rows] == ["OPEN", "HOOK", "ACTION", "END"]
    assert rows[1]["speaker"] == "MIRA"
    assert rows[1]["delivery"] == "shaken"
    assert rows[0]["relative_start_seconds"] == 0


def test_cache_reuse_and_incremental_resume(tmp_path, monkeypatch):
    ep = make_episode()
    llm = FakeLLM([result()])
    cache = JsonCache(tmp_path / "cache")
    assert analyze_episode(ep, llm=llm, cache=cache)
    assert analyze_episode(ep, llm=llm, cache=cache)
    assert llm.calls == 1

    class FailsAfterOne(FakeLLM):
        def generate(self, prompt, response_model):
            if self.calls:
                raise RuntimeError("interrupted")
            return super().generate(prompt, response_model)

    # analyze_episodes uses its configured cache; point that setting into this test path.
    class Settings:
        cache_dir = tmp_path / "run-cache"
        gemini_model = "mock-model"
        max_chunk_chars = 24000
    import promoscout.analysis.episode_analyzer as analyzer_module
    monkeypatch.setattr(analyzer_module, "get_settings", lambda: Settings())
    analysis_file, discoveries_file = tmp_path / "a.json", tmp_path / "d.json"
    episodes = [ep, make_episode().model_copy(update={"episode_number": 2})]
    with pytest.raises(RuntimeError, match="interrupted"):
        analyze_episodes(episodes, llm=FailsAfterOne([result()]),
                         analysis_path=analysis_file, discoveries_path=discoveries_file)
    assert json.loads(analysis_file.read_text()).keys() == {"1"}
    resume_llm = FakeLLM([result()])
    analyze_episodes(episodes, llm=resume_llm, analysis_path=analysis_file,
                     discoveries_path=discoveries_file)
    assert resume_llm.calls == 1  # episode 1 was served from its per-episode cache
    assert json.loads(analysis_file.read_text()).keys() == {"1", "2"}

