from promoscout.parser.screenplay_parser import parse_episode_text, timestamp_to_seconds

SAMPLE_SCRIPT = '''
EPISODE NUMBER: Episode 1
EPISODE LOGLINE: A bride discovers a betrayal at the altar.
CONTINUITY NOTES: First episode.

--- SCRIPT START ---

SCENE 01
Scene heading: INT. CATHEDRAL – DAY
Scene start: 00:00.00
Scene end: 00:20.00

Actions and dialogue:

**CAMERA:**
Start: (00:00.00) | End: (00:04.80)
The church doors open.

**ACTION:**
Start: (00:01.00) | End: (00:04.80)
Juliana enters the cathedral.

**MATRIARCH (Cold, Commanding):**
Start: (00:11.30) | End: (00:13.10)
"You should turn around, Juliana."

--- SCRIPT END ---
'''


def test_timestamp_to_seconds():
    assert timestamp_to_seconds("00:00.00") == 0
    assert timestamp_to_seconds("01:12.40") == 72.4
    assert timestamp_to_seconds("02:18.00") == 138


def test_parse_episode_metadata_and_events():
    episode = parse_episode_text(SAMPLE_SCRIPT, source_file="sample.txt")

    assert episode.episode_number == 1
    assert episode.logline == "A bride discovers a betrayal at the altar."
    assert len(episode.scenes) == 1
    assert len(episode.events) == 3

    assert [event.event_type for event in episode.events] == [
        "camera", "action", "dialogue"
    ]
    assert episode.events[2].speaker == "MATRIARCH"
    assert episode.events[2].delivery == "Cold, Commanding"
    assert episode.events[2].content == "You should turn around, Juliana."


def test_overlapping_events_are_preserved():
    episode = parse_episode_text(SAMPLE_SCRIPT)
    camera, action, _ = episode.events

    assert camera.start_seconds == 0
    assert action.start_seconds == 1
    assert camera.end_seconds == action.end_seconds
    # Events remain in screenplay order; they are not forced into non-overlapping blocks.
    assert episode.events[0].start_seconds < episode.events[1].start_seconds
