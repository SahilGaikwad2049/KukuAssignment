from promoscout.evaluation.blind_viewer import (
    BlindViewerAssessment, BlindViewerEvaluation, DimensionScore,
    EvidenceReference, build_blind_view_input,
)
from promoscout.ranking.final_ranker import WEIGHTS, build_final_rankings
from promoscout.schemas import ParsedEpisode, Scene, ScreenplayEvent
from promoscout.utils.cache import content_hash


def make_episode(number):
    prefix = f"EP{number:03d}"
    specs = [(0, 1, "A warning arrives."), (10, 11, "Mira opens the letter."), (25, 26, "A photograph appears.")]
    events = [ScreenplayEvent(
        event_id=f"{prefix}_EV{index}", event_type="dialogue" if index == 1 else "action",
        label="MIRA", speaker="MIRA", delivery=None,
        start_timestamp=f"00:{start:02d}.00", end_timestamp=f"00:{end:02d}.00",
        start_seconds=start, end_seconds=end, content=text, raw_block=text,
        source_line_start=index, source_line_end=index + 1,
    ) for index, (start, end, text) in enumerate(specs, start=1)]
    return ParsedEpisode(episode_number=number, scenes=[Scene(scene_number=1, events=events)])


def candidate_row(episode_number, *, duration=26.0):
    prefix = f"EP{episode_number:03d}"
    return {
        "candidate_id": f"{prefix}_PROMO_001", "candidate_family_id": f"{prefix}_FAMILY_001",
        "episode_number": episode_number, "start_event_id": f"{prefix}_EV1", "end_event_id": f"{prefix}_EV3",
        "start_timestamp": "00:00.00", "end_timestamp": "00:26.00", "start_seconds": 0.0,
        "end_seconds": duration, "duration_seconds": duration, "hook_event_id": f"{prefix}_EV1",
        "hook_type": "dialogue", "hook_text": "A warning arrives.",
        "hook_start_timestamp": "00:00.00", "hook_end_timestamp": "00:01.00",
        "hook_start_seconds": 0.0, "hook_end_seconds": 1.0, "hook_offset_seconds": 0.0,
        "ending_event_id": f"{prefix}_EV3", "ending_text": "A photograph appears.",
        "heuristic_score": .7, "hook_score": .8, "ending_score": .6,
        "included_event_ids": [f"{prefix}_EV1", f"{prefix}_EV2", f"{prefix}_EV3"],
    }


def saved_evaluation(episode, candidate, *, invalid_evidence=False):
    clip = build_blind_view_input(episode, candidate)
    event_id = "OUTSIDE" if invalid_evidence else f"EP{episode.episode_number:03d}_EV1"
    excerpt = "A warning arrives."
    dimension = DimensionScore(score=5, rationale="Strong and grounded.", evidence=[
        EvidenceReference(event_id=event_id, excerpt=excerpt, why_it_matters="It opens with immediate tension.")])
    assessment = BlindViewerAssessment(
        standalone_story_reconstruction="A warning leads to a discovery.",
        characters_the_viewer_believes_are_involved=[], directly_shown_or_stated=[],
        reasonable_inferences=[], unclear_or_unestablished=[], main_conflict_as_understood="A clue is uncovered.",
        perceived_stakes="The consequence is not stated.", confusing_elements=[],
        missing_context_the_viewer_wishes_for=[], standalone_clarity=dimension,
        hook_effectiveness=dimension, conflict_comprehension=dimension, stakes_clarity=dimension,
        cliffhanger_effectiveness=dimension, desire_to_continue=dimension,
        overall_viewer_verdict="strong", confidence=.9, verdict_explanation="A clear opening and discovery." )
    return BlindViewerEvaluation(
        candidate_id=candidate["candidate_id"], candidate_source="heuristic",
        episode_number=episode.episode_number, evaluated_at="2026-01-01T00:00:00Z", model_id="mock",
        clip_event_ids=[row["event_id"] for row in clip["events"]],
        blind_input_hash=content_hash(clip), assessment=assessment,
    ).model_dump(mode="json")


def test_weighted_ranking_excludes_pending_and_failed_and_keeps_human_separate():
    episodes = [make_episode(number) for number in (1, 2, 3)]
    rows = [candidate_row(number) for number in (1, 2, 3)]
    report = build_final_rankings(
        episodes=episodes, continuous_rows=rows, semantic_rows=[], composite_rows=[],
        evaluations_data={"evaluations": {
            f"heuristic:{rows[0]['candidate_id']}": saved_evaluation(episodes[0], rows[0]),
            f"heuristic:{rows[2]['candidate_id']}": saved_evaluation(episodes[2], rows[2], invalid_evidence=True),
        }},
        pilot_report={"failures": [{"source": "heuristic", "candidate_id": rows[2]["candidate_id"], "error": "invalid cited event"}]},
        review_data={"decisions": {f"heuristic:{rows[0]['candidate_id']}": {"decision": "rejected", "note": "Needs a human recheck."}}},
    )
    assert len(report["leaderboards"]["combined"]) == 1
    ranked = report["leaderboards"]["continuous"][0]
    assert ranked["weighted_score"] == 100.0
    assert ranked["human_decision"] == "rejected"
    assert ranked["final_selection_eligible"] is False
    assert ranked["provisional_ai_eligible"] is True
    assert len(report["pending_candidates"]) == 1
    assert report["pending_candidates"][0]["weighted_score"] is None
    assert len(report["failed_evaluations"]) == 1
    assert report["failed_evaluations"][0]["ranking_status"] == "evaluation_failed"
    assert WEIGHTS["standalone_clarity"] == .2


def test_synthetic_fixture_never_enters_real_composite_or_combined_leaderboard():
    episode = make_episode(1)
    report = build_final_rankings(
        episodes=[episode], continuous_rows=[], semantic_rows=[],
        composite_rows=[{"candidate_id": "SYNTHETIC_ONLY", "candidate_origin": "synthetic", "synthetic": True}],
        evaluations_data={},
    )
    assert report["leaderboards"]["composite"] == []
    assert report["leaderboards"]["combined"] == []
    assert len(report["synthetic_composite_fixtures"]) == 1


def test_weight_configuration_requires_all_six_dimensions():
    episode = make_episode(1)
    try:
        build_final_rankings(episodes=[episode], continuous_rows=[], semantic_rows=[], composite_rows=[],
                             evaluations_data={}, weights={"standalone_clarity": 1})
    except ValueError as exc:
        assert "six evaluation dimensions" in str(exc)
    else:
        raise AssertionError("incomplete weight configuration was accepted")
