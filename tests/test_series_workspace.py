import json
from io import BytesIO
from zipfile import ZipFile

import pytest

from promoscout.analysis.episode_analyzer import analyze_episodes
from promoscout.analysis.models import AnalysisEnvelope, DiscoveredEvent, EpisodeAnalysis
from promoscout.dashboard.series_workspace import (
    create_series_workspace,
    load_workspace,
    parse_uploaded_zip,
    process_series_workspace,
)
from promoscout.utils.cache import JsonCache


def script(episode_number=1):
    return f'''EPISODE NUMBER: Episode {episode_number}
EPISODE LOGLINE: A woman must decide whether to trust a warning.
CONTINUITY NOTES: Pilot episode.
--- SCRIPT START ---
SCENE 01
Scene heading: INT. ROOM – NIGHT
Scene start: 00:00.00
Scene end: 00:40.00
**MIRA (Shaken):**
Start: (00:01.00) | End: (00:02.00)
"Don't open it."
**CAMERA:**
Start: (00:25.00) | End: (00:26.00)
The envelope reveals a second photograph.
--- SCRIPT END ---
'''


def zip_bytes(scripts=None):
    scripts = scripts or {"episode-1.txt": script()}
    buffer = BytesIO()
    with ZipFile(buffer, "w") as archive:
        for name, content in scripts.items():
            archive.writestr(name, content)
    return buffer.getvalue()


class FakeLLM:
    model = "mock-gemini-model"

    def __init__(self):
        self.calls = 0

    def generate(self, prompt, response_model):
        self.calls += 1
        return AnalysisEnvelope(
            episode_analysis=EpisodeAnalysis(
                episode_number=1,
                episode_summary="Mira warns someone not to open an envelope, which reveals a second photograph.",
                important_characters=["MIRA"],
                character_relationships=[],
                main_conflict="Whether to trust the warning about the envelope.",
                important_story_developments=["A photograph is revealed."],
                major_revelations=["There is a second photograph."],
                emotional_turning_points=[],
                unresolved_questions=["Who placed the photograph there?"],
                narrative_progression=["A warning leads to an unexpected discovery."],
            ),
            potential_events=[
                DiscoveredEvent(event_id="EP001_SC001_EV0001", role="hook", category="mystery",
                                reason="The warning creates immediate curiosity.", confidence=.8,
                                narrative_context="A character urgently warns against opening something."),
                DiscoveredEvent(event_id="EP001_SC001_EV0002", role="potential_ending", category="unexpected_discovery",
                                reason="The photograph adds a new question.", confidence=.8,
                                narrative_context="The event describes a second photograph being revealed."),
            ],
        )


def test_uploaded_zip_parses_locally_and_creates_content_addressed_workspace(tmp_path):
    content = zip_bytes({"Episode 1.txt": script(), "Episode 2.txt": script(2)})
    workspace, episodes = create_series_workspace("My New Show", "scripts.zip", content, root=tmp_path)
    manifest, loaded = load_workspace(workspace)
    assert [ep.episode_number for ep in episodes] == [1, 2]
    assert len(loaded) == 2
    assert manifest["series_name"] == "My New Show"
    assert manifest["episode_count"] == 2
    assert manifest["event_count"] == 4
    assert (workspace / "parsed_episodes.json").exists()
    assert not list(workspace.glob("*.zip"))


def test_zip_validation_rejects_invalid_archive_and_duplicate_episode_numbers():
    with pytest.raises(ValueError, match="valid ZIP"):
        parse_uploaded_zip(b"not a zip")
    with pytest.raises(ValueError, match="duplicate episode numbers"):
        parse_uploaded_zip(zip_bytes({"first.txt": script(1), "copy.txt": script(1)}))


def test_series_processing_persists_analysis_discoveries_and_candidates(tmp_path, monkeypatch):
    workspace, episodes = create_series_workspace("Pilot Show", "pilot.zip", zip_bytes(), root=tmp_path)

    class Settings:
        cache_dir = tmp_path / "cache"
        gemini_model = "mock-gemini-model"
        max_chunk_chars = 24000

    monkeypatch.setattr("promoscout.analysis.episode_analyzer.get_settings", lambda: Settings())
    llm = FakeLLM()
    result = process_series_workspace(workspace, episodes, model_id=llm.model, llm=llm)

    assert llm.calls == 1
    assert result["analyses"]["1"]["important_characters"] == ["MIRA"]
    assert len(result["discoveries"]["1"]) == 2
    assert len(result["candidates"]) == 1
    candidate = result["candidates"][0]
    assert candidate.episode_number == 1
    assert 20 <= candidate.duration_seconds <= 60
    assert candidate.hook_event_id == "EP001_SC001_EV0001"
    assert candidate.ending_event_id == "EP001_SC001_EV0002"
    assert json.loads((workspace / "episode_analysis.json").read_text())["1"]
    assert json.loads((workspace / "event_discoveries.json").read_text())["1"]
    assert json.loads((workspace / "semantic_candidates.json").read_text())["candidates"][0]["candidate_id"] == candidate.candidate_id
