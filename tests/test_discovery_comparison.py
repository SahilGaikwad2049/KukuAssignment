import json

import pytest

from promoscout.analysis.models import AnalysisEnvelope, DiscoveredEvent, EpisodeAnalysis
from promoscout.evaluation.blind_viewer import (
    BlindViewerAssessment,
    BlindViewerEvaluation,
    DimensionScore,
    EvidenceReference,
    build_blind_view_input,
)
from promoscout.evaluation.discovery_comparison import (
    build_discovery_comparison,
    render_markdown,
)
from promoscout.generation.continuous import generate_episode_candidates_from_discoveries
from promoscout.schemas import ParsedEpisode, Scene, ScreenplayEvent
from promoscout.utils.cache import content_hash


def event(event_id, event_type, start, end, text, line, speaker=None):
    return ScreenplayEvent(
        event_id=event_id, event_type=event_type, label=event_type.upper(), speaker=speaker,
        delivery=None, start_timestamp=f"00:{start:05.2f}", end_timestamp=f"00:{end:05.2f}",
        start_seconds=start, end_seconds=end, content=text, raw_block=text,
        source_line_start=line, source_line_end=line + 1,
    )


def fixture_data():
    episode = ParsedEpisode(
        episode_number=1, logline="Not supplied to the blind evaluator.",
        continuity_notes="Not supplied to the blind evaluator.",
        scenes=[Scene(scene_number=1, heading="INT. ROOM - NIGHT", start_timestamp="00:00.00",
                      end_timestamp="00:40.00", start_seconds=0, end_seconds=40, events=[
            event("OPEN", "camera", 0, 1, "A shadow appears behind the door.", 1),
            event("SETUP", "action", 2, 3, "Mira reaches for the sealed letter.", 2),
            event("HOOK", "dialogue", 4, 5, "Do you know the truth?", 3, "MIRA"),
            event("MID", "action", 12, 13, "Mira opens the letter.", 4),
            event("END", "camera", 25, 26, "A second photograph falls out.", 5),
        ])],
    )
    discoveries = [
        DiscoveredEvent(event_id="OPEN", role="hook", category="visual_intrigue", reason="An unexplained figure appears.", confidence=.8, narrative_context="The shadow creates an immediate visual question."),
        DiscoveredEvent(event_id="HOOK", role="hook", category="mystery", reason="The question implies hidden information.", confidence=.9, narrative_context="The speaker asks about an unknown truth."),
        DiscoveredEvent(event_id="END", role="potential_ending", category="unexpected_discovery", reason="A photograph appears unexpectedly.", confidence=.8, narrative_context="The image may introduce a new question."),
    ]
    semantic = generate_episode_candidates_from_discoveries(episode, discoveries)
    semantic_rows = [candidate.model_dump(mode="json") for candidate in semantic]
    hook_candidate = next(item for item in semantic if item.hook_event_id == "HOOK")
    heuristic = [
        hook_candidate.model_copy(update={
            "candidate_id": "EP001_PROMO_001",
            "candidate_family_id": "EP001_FAMILY_001",
            "hook_score": .7,
            "ending_score": .6,
            "heuristic_score": .65,
        }).model_dump(mode="json")
    ]
    analysis = {"1": EpisodeAnalysis(
        episode_number=1, episode_summary="Mira finds an unexpected image.",
        important_characters=["MIRA"], character_relationships=[],
        main_conflict="Mira investigates the letter.", important_story_developments=[],
        major_revelations=[], emotional_turning_points=[], unresolved_questions=[], narrative_progression=[],
    ).model_dump(mode="json")}
    discovery_map = {"1": [item.model_dump(mode="json") for item in discoveries]}
    return episode, discoveries, semantic_rows, heuristic, analysis, discovery_map


def assessment(score=3, verdict="mixed"):
    evidence = [EvidenceReference(event_id="HOOK", excerpt="Do you know the truth?", why_it_matters="It raises a question." )]
    dimension = DimensionScore(score=score, rationale="The clip gives partial context.", evidence=evidence)
    return BlindViewerAssessment(
        standalone_story_reconstruction="Mira is looking for an unknown truth.",
        characters_the_viewer_believes_are_involved=[], directly_shown_or_stated=[],
        reasonable_inferences=[], unclear_or_unestablished=["The truth is not identified."],
        main_conflict_as_understood="Mira seeks an answer.", perceived_stakes="The stakes are not established.",
        main_unanswered_question="What truth is being discussed?", confusing_elements=[],
        missing_context_the_viewer_wishes_for=["What happened earlier?"],
        standalone_clarity=dimension, hook_effectiveness=dimension, conflict_comprehension=dimension,
        stakes_clarity=dimension, cliffhanger_effectiveness=dimension, desire_to_continue=dimension,
        overall_viewer_verdict=verdict, confidence=.75, verdict_explanation="The question is intriguing but context is missing.",
    )


def saved_evaluation(episode, candidate, source, *, score=3, verdict="mixed"):
    clip = build_blind_view_input(episode, candidate)
    return BlindViewerEvaluation(
        candidate_id=candidate["candidate_id"], candidate_source=source, episode_number=1,
        evaluated_at="2026-11-09T00:00:00+00:00", model_id="mock-model",
        clip_event_ids=[row["event_id"] for row in clip["events"]],
        blind_input_hash=content_hash(clip), assessment=assessment(score, verdict),
    ).model_dump(mode="json")


def compare_fixture(evaluations=None, reviews=None):
    episode, discoveries, semantic, heuristic, analysis, discovery_map = fixture_data()
    return build_discovery_comparison(
        episode=episode, analysis_data=analysis, discoveries_data=discovery_map,
        heuristic_rows=heuristic, semantic_rows=semantic,
        evaluations_data=evaluations or {"evaluations": {}},
        review_decisions=reviews or {"decisions": {}},
    )


def test_discovery_comparison_calculates_method_counts_and_shared_moments():
    episode, _discoveries, semantic, heuristic, *_ = fixture_data()
    scores = {"evaluations": {
        "heuristic:EP001_PROMO_001": saved_evaluation(episode, heuristic[0], "heuristic", score=4, verdict="strong"),
        f"semantic:{semantic[0]['candidate_id']}": saved_evaluation(episode, semantic[0], "semantic", score=2, verdict="weak"),
    }}
    report = compare_fixture(scores)
    assert report["method_statistics"]["heuristic"]["candidate_count"] == 1
    assert report["method_statistics"]["heuristic"]["structurally_valid_candidates"] == 1
    assert report["method_statistics"]["semantic"]["candidate_count"] == 2
    assert report["semantic_discovery_statistics"]["hook_count"] == 2
    assert report["method_statistics"]["heuristic"]["blind_viewer_comparison"]["dimension_average_scores"]["standalone_clarity"] == 4
    assert report["method_statistics"]["semantic"]["blind_viewer_comparison"]["verdict_counts"]["weak"] == 1
    assert report["qualitative_comparison"]["shared_discovery_event_ids"] == ["END", "HOOK"]
    assert report["qualitative_comparison"]["semantic_discovery_events_not_used_as_heuristic_endpoints"][0]["event_id"] == "OPEN"


def test_invalid_or_failed_evaluations_are_not_counted_as_weak():
    episode, _discoveries, semantic, heuristic, *_ = fixture_data()
    bad_eval = saved_evaluation(episode, heuristic[0], "heuristic", score=2, verdict="weak")
    bad_eval["assessment"]["standalone_clarity"]["evidence"][0]["event_id"] = "OUTSIDE"
    result = compare_fixture({"evaluations": {
        "heuristic:EP001_PROMO_001": bad_eval,
        # A failed semantic call has no saved assessment and must not become weak.
    }})
    stats = result["method_statistics"]
    assert stats["heuristic"]["blind_viewer_comparison"]["successful_validated_evaluations"] == 0
    assert stats["heuristic"]["blind_viewer_comparison"]["verdict_counts"] == {"strong": 0, "mixed": 0, "weak": 0}
    assert stats["semantic"]["blind_viewer_comparison"]["successful_validated_evaluations"] == 0
    semantic_row = next(row for row in result["candidate_level_comparison"] if row["candidate_source"] == "semantic")
    assert semantic_row["human_decision"] == "pending"


def test_ai_verdict_never_implies_human_approval_and_report_is_readable():
    episode, _discoveries, _semantic, heuristic, *_ = fixture_data()
    evaluation = saved_evaluation(episode, heuristic[0], "heuristic", score=5, verdict="strong")
    report = compare_fixture({"evaluations": {"heuristic:EP001_PROMO_001": evaluation}})
    row = next(row for row in report["candidate_level_comparison"] if row["candidate_source"] == "heuristic")
    assert row["verdict"] == "strong"
    assert row["human_decision"] == "pending"
    markdown = render_markdown(report)
    assert "does not establish statistical superiority" in markdown
    assert "Potential ending (not a confirmed cliffhanger)" in markdown


def test_missing_semantic_candidates_returns_informative_error():
    episode, _discoveries, _semantic, heuristic, analysis, discovery_map = fixture_data()
    with pytest.raises(ValueError, match="No semantic candidates"):
        build_discovery_comparison(
            episode=episode, analysis_data=analysis, discoveries_data=discovery_map,
            heuristic_rows=heuristic, semantic_rows=[], evaluations_data={"evaluations": {}},
        )
