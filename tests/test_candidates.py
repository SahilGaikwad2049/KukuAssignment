
from promoscout.generation.continuous import (
    generate_episode_candidates,
    score_ending,
    score_hook,
    classify_hook_type,
)

from promoscout.schemas import ParsedEpisode, Scene, ScreenplayEvent


def make_event(
    event_id,
    event_type,
    start,
    end,
    content,
    speaker=None,
    delivery=None,
    line=1,
):
    return ScreenplayEvent(
        event_id=event_id,
        event_type=event_type,
        label=event_type.upper(),
        speaker=speaker,
        delivery=delivery,
        start_timestamp=f"00:{start:05.2f}",
        end_timestamp=f"00:{end:05.2f}",
        start_seconds=start,
        end_seconds=end,
        content=content,
        raw_block=content,
        source_line_start=line,
        source_line_end=line + 1,
    )


# ---------------------------------------------------------------------
# EXISTING TESTS
# ---------------------------------------------------------------------

def test_hook_scoring_detects_high_tension_dialogue():

    event = make_event(
        "HOOK",
        "dialogue",
        5,
        7,
        "You lied to me!",
        speaker="JULIANA",
        delivery="Furious",
        line=10,
    )

    score, signals = score_hook(event)

    assert score >= 1.2
    assert signals


def test_generator_returns_valid_20_to_60_second_window():

    hook = make_event(
        "HOOK",
        "dialogue",
        5,
        7,
        "You lied to me!",
        speaker="JULIANA",
        delivery="Furious",
        line=10,
    )

    context = make_event(
        "CTX",
        "camera",
        3,
        6,
        "Close-up on Juliana.",
        line=8,
    )

    middle = make_event(
        "MID",
        "dialogue",
        15,
        17,
        "Tell me the truth.",
        line=20,
    )

    ending = make_event(
        "END",
        "dialogue",
        29,
        31,
        "Who are you really?",
        speaker="GERALD",
        delivery="Shaken",
        line=40,
    )

    scene = Scene(
        scene_number=1,
        heading="INT. ROOM - DAY",
        start_timestamp="00:00.00",
        end_timestamp="00:40.00",
        start_seconds=0,
        end_seconds=40,
        events=[context, hook, middle, ending],
    )

    episode = ParsedEpisode(
        episode_number=1,
        logline="A betrayal is revealed.",
        scenes=[scene],
    )

    candidates = generate_episode_candidates(episode)

    assert candidates

    candidate = candidates[0]

    assert 20 <= candidate.duration_seconds <= 60
    assert candidate.hook_event_id == "HOOK"
    assert candidate.ending_event_id == "END"
    assert candidate.start_event_id == "CTX"

    assert candidate.included_event_ids == [
        "CTX",
        "HOOK",
        "MID",
        "END",
    ]

    assert candidate.start_seconds == 3
    assert candidate.end_seconds == 31


def test_scene_boundary_is_not_a_hard_clip_boundary():

    hook = make_event(
        "HOOK",
        "dialogue",
        5,
        7,
        "You lied to me!",
        speaker="JULIANA",
        delivery="Furious",
        line=10,
    )

    ending = make_event(
        "END",
        "camera",
        29,
        31,
        "Cut to black.",
        line=40,
    )

    scene1 = Scene(
        scene_number=1,
        heading="INT. ROOM - DAY",
        start_timestamp="00:00.00",
        end_timestamp="00:20.00",
        start_seconds=0,
        end_seconds=20,
        events=[hook],
    )

    scene2 = Scene(
        scene_number=2,
        heading="INT. HALL - CONTINUOUS",
        start_timestamp="00:20.00",
        end_timestamp="00:35.00",
        start_seconds=20,
        end_seconds=35,
        events=[ending],
    )

    episode = ParsedEpisode(
        episode_number=1,
        scenes=[scene1, scene2],
    )

    candidates = generate_episode_candidates(episode)

    assert candidates
    assert candidates[0].start_seconds == 5
    assert candidates[0].end_seconds == 31


def test_duration_constraints_are_enforced():

    hook = make_event(
        "HOOK",
        "dialogue",
        5,
        7,
        "You lied to me!",
        speaker="JULIANA",
        delivery="Furious",
        line=10,
    )

    ending = make_event(
        "END",
        "dialogue",
        70,
        72,
        "Who are you?",
        speaker="GERALD",
        delivery="Shaken",
        line=40,
    )

    scene = Scene(
        scene_number=1,
        start_timestamp="00:00.00",
        end_timestamp="02:00.00",
        start_seconds=0,
        end_seconds=120,
        events=[hook, ending],
    )

    episode = ParsedEpisode(
        episode_number=1,
        scenes=[scene],
    )

    assert generate_episode_candidates(episode) == []


# ---------------------------------------------------------------------
# NEW TESTS: HOOK METADATA
# ---------------------------------------------------------------------

def test_hook_timestamp_and_offset_are_preserved():

    hook = make_event(
        "HOOK",
        "dialogue",
        5,
        7,
        "You lied to me!",
        speaker="JULIANA",
        delivery="Furious",
        line=10,
    )

    context = make_event(
        "CTX",
        "camera",
        3,
        4,
        "Close-up on Juliana.",
        line=8,
    )

    ending = make_event(
        "END",
        "dialogue",
        29,
        31,
        "Who are you really?",
        line=40,
    )

    scene = Scene(
        scene_number=1,
        start_timestamp="00:00.00",
        end_timestamp="00:40.00",
        start_seconds=0,
        end_seconds=40,
        events=[context, hook, ending],
    )

    episode = ParsedEpisode(
        episode_number=1,
        scenes=[scene],
    )

    candidates = generate_episode_candidates(episode)

    assert candidates

    candidate = candidates[0]

    assert candidate.hook_start_timestamp == "00:05.00"
    assert candidate.hook_end_timestamp == "00:07.00"

    assert candidate.hook_start_seconds == 5
    assert candidate.hook_end_seconds == 7

    assert candidate.hook_offset_seconds == 2.0
    assert candidate.hook_offset_seconds <= 3.0


def test_hook_type_classification():

    assert classify_hook_type("dialogue") == "dialogue"
    assert classify_hook_type("action") == "action"
    assert classify_hook_type("camera") == "visual"
    assert classify_hook_type("sound_fx") == "other"


# ---------------------------------------------------------------------
# NEW TESTS: EXACT WINDOW DEDUPLICATION
# ---------------------------------------------------------------------

def test_exact_duplicate_windows_are_removed():

    context = make_event(
        "CTX",
        "camera",
        3,
        4,
        "Close-up on Juliana.",
        line=5,
    )

    hook1 = make_event(
        "HOOK1",
        "dialogue",
        5,
        6,
        "You lied to me!",
        speaker="JULIANA",
        delivery="Furious",
        line=10,
    )

    hook2 = make_event(
        "HOOK2",
        "dialogue",
        5,
        6,
        "How could you?",
        speaker="JULIANA",
        delivery="Angry",
        line=12,
    )

    ending = make_event(
        "END",
        "dialogue",
        29,
        31,
        "Who are you really?",
        speaker="GERALD",
        delivery="Shaken",
        line=40,
    )

    scene = Scene(
        scene_number=1,
        start_timestamp="00:00.00",
        end_timestamp="00:40.00",
        start_seconds=0,
        end_seconds=40,
        events=[context, hook1, hook2, ending],
    )

    episode = ParsedEpisode(
        episode_number=1,
        scenes=[scene],
    )

    candidates = generate_episode_candidates(episode)

    # Both hooks produce the same 3s-31s clip window.
    # Only the stronger candidate should survive.
    assert len(candidates) == 1

    assert candidates[0].start_seconds == 3
    assert candidates[0].end_seconds == 31

    assert candidates[0].hook_event_id == "HOOK1"


# ---------------------------------------------------------------------
# NEW TESTS: CANDIDATE FAMILIES
# ---------------------------------------------------------------------

def test_alternative_endings_share_candidate_family():

    context = make_event(
        "CTX",
        "camera",
        3,
        4,
        "Close-up on Juliana.",
        line=5,
    )

    hook = make_event(
        "HOOK",
        "dialogue",
        5,
        7,
        "You lied to me!",
        speaker="JULIANA",
        delivery="Furious",
        line=10,
    )

    ending1 = make_event(
        "END1",
        "dialogue",
        29,
        31,
        "Who are you really?",
        speaker="GERALD",
        line=40,
    )

    ending2 = make_event(
        "END2",
        "dialogue",
        34,
        36,
        "What happened?",
        speaker="GERALD",
        line=50,
    )

    scene = Scene(
        scene_number=1,
        start_timestamp="00:00.00",
        end_timestamp="00:40.00",
        start_seconds=0,
        end_seconds=40,
        events=[context, hook, ending1, ending2],
    )

    episode = ParsedEpisode(
        episode_number=1,
        scenes=[scene],
    )

    candidates = generate_episode_candidates(episode)

    hook_variants = [
        candidate
        for candidate in candidates
        if candidate.hook_event_id == "HOOK"
    ]

    assert len(hook_variants) >= 2

    family_ids = {
        candidate.candidate_family_id
        for candidate in hook_variants
    }

    # Different endings, same original hook.
    assert len(family_ids) == 1

    assert hook_variants[0].candidate_family_id.startswith(
        "EP001_FAMILY_"
    )


# ---------------------------------------------------------------------
# NEW TESTS: FAMILY ID DETERMINISM
# ---------------------------------------------------------------------

def test_candidate_family_ids_are_deterministic():

    context = make_event(
        "CTX",
        "camera",
        3,
        4,
        "Close-up on Juliana.",
        line=5,
    )

    hook = make_event(
        "HOOK",
        "dialogue",
        5,
        7,
        "You lied to me!",
        speaker="JULIANA",
        delivery="Furious",
        line=10,
    )

    ending1 = make_event(
        "END1",
        "dialogue",
        29,
        31,
        "Who are you really?",
        line=40,
    )

    ending2 = make_event(
        "END2",
        "dialogue",
        34,
        36,
        "What happened?",
        line=50,
    )

    scene = Scene(
        scene_number=1,
        start_timestamp="00:00.00",
        end_timestamp="00:40.00",
        start_seconds=0,
        end_seconds=40,
        events=[context, hook, ending1, ending2],
    )

    episode = ParsedEpisode(
        episode_number=1,
        scenes=[scene],
    )

    first_run = generate_episode_candidates(episode)
    second_run = generate_episode_candidates(episode)

    first_ids = [
        (c.candidate_id, c.candidate_family_id)
        for c in first_run
    ]

    second_ids = [
        (c.candidate_id, c.candidate_family_id)
        for c in second_run
    ]

    assert first_ids == second_ids
