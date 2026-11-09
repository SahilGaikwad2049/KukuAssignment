import json
import sys

import pytest
from pydantic import ValidationError

from promoscout.evaluation.blind_viewer import (
    BlindViewerAssessment,
    DimensionScore,
    EvidenceReference,
    InvalidEvidenceReference,
    PerceivedCharacter,
    build_blind_view_input,
    evaluate_candidate,
    load_candidate_records,
    main,
    run_evaluations,
    select_candidate_records,
)
from promoscout.generation.continuous import PromoCandidate
from promoscout.schemas import ParsedEpisode, Scene, ScreenplayEvent
from promoscout.utils.cache import JsonCache


def make_event(event_id, event_type, start, end, content, line, speaker=None):
    return ScreenplayEvent(
        event_id=event_id, event_type=event_type, label=event_type.upper(), speaker=speaker,
        delivery="shaken" if speaker else None,
        start_timestamp=f"00:{start:05.2f}", end_timestamp=f"00:{end:05.2f}",
        start_seconds=start, end_seconds=end, content=content, raw_block=content,
        source_line_start=line, source_line_end=line + 1,
    )


def make_episode(number=1):
    prefix = f"EP{number:03d}"
    events = [
        make_event(f"{prefix}_BEFORE", "action", 0, 2, "Before the clip.", 1),
        make_event(f"{prefix}_OPEN", "camera", 3, 5, "A shadow crosses the doorway.", 2),
        make_event(f"{prefix}_HOOK", "dialogue", 5, 7, "You knew the truth!", 3, "MIRA"),
        make_event(f"{prefix}_ACTION", "action", 12, 13, "Mira tears open an envelope.", 4),
        make_event(f"{prefix}_SOUND", "sound_fx", 18, 19, "A phone rings.", 5),
        make_event(f"{prefix}_END", "camera", 30, 31, "A second photograph appears.", 6),
        make_event(f"{prefix}_AFTER", "dialogue", 32, 34, "What happens next?", 7, "NOAH"),
    ]
    return ParsedEpisode(
        episode_number=number,
        logline="SECRET SERIES LOG LINE MUST NEVER BE SHARED",
        continuity_notes="SECRET RELATIONSHIP CONTEXT MUST NEVER BE SHARED",
        scenes=[Scene(scene_number=1, heading="SECRET SCENE NAME", start_seconds=0,
                      end_seconds=40, events=events)],
    )


def make_candidate(number=1):
    prefix = f"EP{number:03d}"
    return PromoCandidate(
        candidate_id=f"{prefix}_PROMO_001", candidate_family_id=f"{prefix}_FAMILY_001",
        episode_number=number, start_event_id=f"{prefix}_OPEN", end_event_id=f"{prefix}_END",
        start_timestamp="00:03.00", end_timestamp="00:31.00", start_seconds=3,
        end_seconds=31, duration_seconds=28, hook_event_id=f"{prefix}_HOOK",
        hook_type="dialogue", hook_text="You knew the truth!",
        hook_start_timestamp="00:05.00", hook_end_timestamp="00:07.00",
        hook_start_seconds=5, hook_end_seconds=7, hook_offset_seconds=2,
        ending_event_id=f"{prefix}_END", ending_text="A second photograph appears.",
        hook_score=.9, ending_score=.8, heuristic_score=.85,
        included_event_ids=[f"{prefix}_OPEN", f"{prefix}_HOOK", f"{prefix}_END"],
    )


def assessment(event_id="EP001_HOOK"):
    evidence = [EvidenceReference(event_id=event_id, excerpt="You knew the truth!", why_it_matters="The accusation implies hidden conflict.")]
    score = DimensionScore(score=3, rationale="Understandable, with some context missing.", evidence=evidence)
    return BlindViewerAssessment(
        standalone_story_reconstruction="Mira confronts someone about a past secret.",
        characters_the_viewer_believes_are_involved=[PerceivedCharacter(
            name_or_description="Mira", perceived_role="speaker making an accusation", certainty="clear", evidence=evidence,
        )],
        directly_shown_or_stated=["Mira accuses someone of knowing the truth."],
        reasonable_inferences=["There is hidden history between the speakers."],
        unclear_or_unestablished=["The clip does not establish what truth is at issue."],
        main_conflict_as_understood="Mira confronts someone about concealed information.",
        perceived_stakes="The relationship may be damaged, but the consequences are unclear.",
        main_unanswered_question="What truth was hidden?",
        confusing_elements=["The history behind the accusation is unknown."],
        missing_context_the_viewer_wishes_for=["What happened before this confrontation?"],
        standalone_clarity=score, hook_effectiveness=score, conflict_comprehension=score,
        stakes_clarity=score, cliffhanger_effectiveness=score, desire_to_continue=score,
        overall_viewer_verdict="mixed", confidence=.8,
        verdict_explanation="The accusation is intriguing, but the clip leaves key stakes unclear.",
    )


class FakeLLM:
    model = "mock-blind-model"
    def __init__(self, response=None):
        self.response = response or assessment()
        self.calls = 0
        self.prompts = []
    def generate(self, prompt, response_model):
        self.calls += 1
        self.prompts.append(prompt)
        return self.response
    def generate_json_schema(self, prompt, response_model):
        self.schema_calls = getattr(self, "schema_calls", 0) + 1
        return self.generate(prompt, response_model)


def test_blind_input_reconstructs_every_overlapping_event_in_chronological_order():
    episode, candidate = make_episode(), make_candidate()
    payload = build_blind_view_input(episode, candidate)
    ids = [event["event_id"] for event in payload["events"]]
    assert ids == ["EP001_OPEN", "EP001_HOOK", "EP001_ACTION", "EP001_SOUND", "EP001_END"]
    assert "EP001_ACTION" in ids and "EP001_SOUND" in ids  # absent from candidate's stored preview IDs
    assert "EP001_BEFORE" not in ids and "EP001_AFTER" not in ids


def test_blind_input_has_no_episode_synopsis_or_relationship_metadata():
    payload = build_blind_view_input(make_episode(), make_candidate())
    serialized = json.dumps(payload)
    assert "episode_number" not in payload
    assert "SECRET SERIES" not in serialized
    assert "SECRET RELATIONSHIP" not in serialized
    assert "SECRET SCENE" not in serialized
    assert "candidate_id" not in payload
    assert "hook_text" not in payload
    assert "ending_text" not in payload
    assert "start_timestamp" not in serialized and "end_timestamp" not in serialized


def test_relative_times_speaker_and_event_type_are_preserved():
    payload = build_blind_view_input(make_episode(), make_candidate())
    by_id = {event["event_id"]: event for event in payload["events"]}
    assert payload["clip_duration_seconds"] == 28
    assert by_id["EP001_OPEN"]["relative_start_seconds"] == 0
    assert by_id["EP001_HOOK"]["relative_start_seconds"] == 2
    assert by_id["EP001_HOOK"]["speaker"] == "MIRA"
    assert by_id["EP001_HOOK"]["event_type"] == "dialogue"
    assert by_id["EP001_ACTION"]["event_type"] == "action"
    assert by_id["EP001_SOUND"]["event_type"] == "sound_fx"


def test_evaluator_prompt_receives_only_sanitized_clip_payload(tmp_path, monkeypatch):
    import promoscout.evaluation.blind_viewer as blind
    class Settings:
        gemini_model = "mock-blind-model"
        cache_dir = tmp_path / "cache"
    monkeypatch.setattr(blind, "get_settings", lambda: Settings())
    llm = FakeLLM()
    evaluate_candidate(make_episode(), make_candidate(), llm=llm, cache=JsonCache(tmp_path / "cache"))
    prompt = llm.prompts[0]
    assert "SECRET SERIES" not in prompt
    assert "SECRET RELATIONSHIP" not in prompt
    assert "EP001_PROMO_001" not in prompt
    assert "EP001_HOOK" in prompt
    assert llm.schema_calls == 1


def test_invalid_evidence_event_id_is_rejected(tmp_path, monkeypatch):
    import promoscout.evaluation.blind_viewer as blind
    class Settings:
        gemini_model = "mock-blind-model"
        cache_dir = tmp_path / "cache"
    monkeypatch.setattr(blind, "get_settings", lambda: Settings())
    bad = assessment("EP999_NOT_IN_CLIP")
    with pytest.raises(InvalidEvidenceReference, match="outside the supplied clip"):
        evaluate_candidate(make_episode(), make_candidate(), llm=FakeLLM(bad), cache=JsonCache(tmp_path / "cache"))


def test_non_source_quotation_is_rejected(tmp_path, monkeypatch):
    import promoscout.evaluation.blind_viewer as blind
    class Settings:
        gemini_model = "mock-blind-model"
        cache_dir = tmp_path / "cache"
    monkeypatch.setattr(blind, "get_settings", lambda: Settings())
    bad = assessment()
    bad.standalone_clarity.evidence[0].excerpt = "A fabricated quote."
    with pytest.raises(InvalidEvidenceReference, match="quotation"):
        evaluate_candidate(make_episode(), make_candidate(), llm=FakeLLM(bad), cache=JsonCache(tmp_path / "cache"))


def test_six_dimensions_are_required_and_scores_must_be_between_one_and_five():
    value = assessment().model_dump()
    del value["stakes_clarity"]
    with pytest.raises(ValidationError):
        BlindViewerAssessment.model_validate(value)
    invalid = assessment().model_dump()
    invalid["hook_effectiveness"]["score"] = 6
    with pytest.raises(ValidationError):
        BlindViewerAssessment.model_validate(invalid)
    invalid["hook_effectiveness"]["score"] = 0
    with pytest.raises(ValidationError):
        BlindViewerAssessment.model_validate(invalid)


def test_schema_explicitly_supports_unknowns_and_missing_context():
    result = assessment()
    result.main_unanswered_question = None
    result.unclear_or_unestablished = ["The speaker's identity is unknown."]
    result.missing_context_the_viewer_wishes_for = ["Why the photograph matters."]
    assert result.main_unanswered_question is None
    assert result.unclear_or_unestablished
    assert result.missing_context_the_viewer_wishes_for


def test_cached_result_is_reused_without_second_llm_request(tmp_path, monkeypatch):
    import promoscout.evaluation.blind_viewer as blind
    class Settings:
        gemini_model = "mock-blind-model"
        cache_dir = tmp_path / "cache"
    monkeypatch.setattr(blind, "get_settings", lambda: Settings())
    cache, llm = JsonCache(tmp_path / "cache"), FakeLLM()
    first = evaluate_candidate(make_episode(), make_candidate(), llm=llm, cache=cache)
    second = evaluate_candidate(make_episode(), make_candidate(), llm=llm, cache=cache)
    assert first.candidate_id == second.candidate_id
    assert llm.calls == 1


def test_prompt_version_change_invalidates_cache(tmp_path, monkeypatch):
    import promoscout.evaluation.blind_viewer as blind
    class Settings:
        gemini_model = "mock-blind-model"
        cache_dir = tmp_path / "cache"
    monkeypatch.setattr(blind, "get_settings", lambda: Settings())
    cache, llm = JsonCache(tmp_path / "cache"), FakeLLM()
    evaluate_candidate(make_episode(), make_candidate(), llm=llm, cache=cache)
    monkeypatch.setattr(blind, "PROMPT_VERSION", "blind-viewer-prompt-v2")
    evaluate_candidate(make_episode(), make_candidate(), llm=llm, cache=cache)
    assert llm.calls == 2


def test_existing_output_is_preserved_and_successful_result_saved_incrementally(tmp_path, monkeypatch):
    import promoscout.evaluation.blind_viewer as blind
    class Settings:
        gemini_model = "mock-blind-model"
        cache_dir = tmp_path / "cache"
    monkeypatch.setattr(blind, "get_settings", lambda: Settings())
    output = tmp_path / "evaluations.json"
    output.write_text(json.dumps({"schema_version": blind.SCHEMA_VERSION,
                                 "evaluations": {"semantic:KEEP": {"kept": True}}}))
    record = {"candidate": make_candidate(), "source": "semantic"}
    result, failures = run_evaluations([record], [make_episode()], llm=FakeLLM(),
                                       cache=JsonCache(tmp_path / "cache"), output_path=output)
    stored = json.loads(output.read_text())
    assert not failures
    assert stored["evaluations"]["semantic:KEEP"] == {"kept": True}
    assert stored["evaluations"]["semantic:EP001_PROMO_001"]["candidate_id"] == "EP001_PROMO_001"
    assert result["evaluations"].keys() == stored["evaluations"].keys()


def test_episode_candidate_and_origin_filters_are_correct():
    records = load_candidate_records({"candidates": [make_candidate().model_dump(), make_candidate(2).model_dump()]}, "semantic")
    records += load_candidate_records([make_candidate().model_dump()], "heuristic")
    selected = select_candidate_records(records, episode_number=2, candidate_id="EP002_PROMO_001")
    assert len(selected) == 1
    assert selected[0]["source"] == "semantic"
    assert selected[0]["candidate"].episode_number == 2
    assert len(select_candidate_records(records, limit=1)) == 1
    assert {record["source"] for record in records} == {"semantic", "heuristic"}


def test_missing_candidate_data_is_handled_as_empty():
    assert load_candidate_records(None, "semantic") == []


def test_dry_run_does_not_construct_gemini_or_modify_output(tmp_path, monkeypatch, capsys):
    import promoscout.evaluation.blind_viewer as blind
    class Settings:
        gemini_model = "mock-blind-model"
        cache_dir = tmp_path / "cache"
        gemini_api_key = ""
    monkeypatch.setattr(blind, "get_settings", lambda: Settings())
    episode_path, candidate_path = tmp_path / "parsed.json", tmp_path / "candidates.json"
    episode_path.write_text(json.dumps([make_episode().model_dump(mode="json")]))
    candidate_path.write_text(json.dumps({"candidates": [make_candidate().model_dump(mode="json")]}))
    output = tmp_path / "output.json"
    output.write_text('{"keep": true}')
    before = output.read_bytes()
    monkeypatch.setattr(blind, "GeminiClient", lambda: pytest.fail("dry-run must not create GeminiClient"))
    monkeypatch.setattr(sys, "argv", [
        "blind_viewer", "--parsed", str(episode_path), "--candidates", str(candidate_path),
        "--episode-number", "1", "--candidate-id", "EP001_PROMO_001", "--dry-run", "--output", str(output),
    ])
    main()
    report = capsys.readouterr().out
    assert "Selected candidates: 1" in report
    assert "Estimated requests: 1" in report
    assert output.read_bytes() == before
    assert not (tmp_path / "cache").exists()


def test_each_candidate_gets_an_independent_llm_request(tmp_path, monkeypatch):
    import promoscout.evaluation.blind_viewer as blind
    class Settings:
        gemini_model = "mock-blind-model"
        cache_dir = tmp_path / "cache"
    monkeypatch.setattr(blind, "get_settings", lambda: Settings())
    second_candidate = make_candidate()
    second_candidate.candidate_id = "EP001_PROMO_002"
    records = [
        {"candidate": make_candidate(1), "source": "semantic"},
        {"candidate": second_candidate, "source": "semantic"},
    ]
    llm = FakeLLM()
    _, failures = run_evaluations(records, [make_episode(1)], llm=llm,
                                  cache=JsonCache(tmp_path / "cache"), output_path=tmp_path / "out.json")
    assert not failures
    assert llm.calls == 2
    assert all("EP001_PROMO_001" not in prompt for prompt in llm.prompts)


def test_later_candidate_failure_preserves_earlier_success(tmp_path, monkeypatch):
    import promoscout.evaluation.blind_viewer as blind

    class Settings:
        gemini_model = "mock-blind-model"
        cache_dir = tmp_path / "cache"

    monkeypatch.setattr(blind, "get_settings", lambda: Settings())
    second = make_candidate().model_copy(update={
        "candidate_id": "EP001_PROMO_002",
        "start_seconds": 4.0,
        "start_timestamp": "00:04.00",
        "hook_offset_seconds": 1.0,
    })

    class FailsOnSecond(FakeLLM):
        def generate_json_schema(self, prompt, response_model):
            if self.calls:
                self.calls += 1
                raise RuntimeError("mock transient failure")
            return super().generate_json_schema(prompt, response_model)

    output = tmp_path / "evaluations.json"
    output.write_text(json.dumps({"schema_version": blind.SCHEMA_VERSION,
                                  "evaluations": {"semantic:PREVIOUS": {"kept": True}}}))
    records = [
        {"candidate": make_candidate(), "source": "semantic"},
        {"candidate": second, "source": "semantic"},
    ]
    result, failures = run_evaluations(records, [make_episode()], llm=FailsOnSecond(),
                                       cache=JsonCache(tmp_path / "cache"), output_path=output)
    saved = json.loads(output.read_text())["evaluations"]
    assert len(failures) == 1
    assert "semantic:PREVIOUS" in saved
    assert "semantic:EP001_PROMO_001" in saved
    assert "semantic:EP001_PROMO_002" not in saved
    assert result["evaluations"].keys() == saved.keys()

