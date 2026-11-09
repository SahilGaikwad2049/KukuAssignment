import json
from pathlib import Path

from streamlit.testing.v1 import AppTest
from promoscout.dashboard.data_loader import load_dashboard_data, normalize_candidates, read_json_file
from promoscout.dashboard.export_utils import candidate_csv_bytes, json_export_bytes
from promoscout.dashboard.review_manager import load_review_decisions, review_key, save_review_decision
from promoscout.dashboard.styles import CSS
from promoscout.schemas import ParsedEpisode, Scene


def test_json_loader_handles_missing_and_malformed_files(tmp_path):
    missing = read_json_file(tmp_path / "missing.json")
    assert not missing["exists"] and missing["data"] is None and missing["error"] is None
    malformed_path = tmp_path / "broken.json"
    malformed_path.write_text("{not json", encoding="utf-8")
    malformed = read_json_file(malformed_path)
    assert malformed["exists"] and malformed["data"] is None
    assert malformed["error"]


def test_candidate_normalization_marks_source_and_reports_bad_rows():
    rows, errors = normalize_candidates({"metadata": {}, "candidates": [
        {"candidate_id": "SEM-1", "episode_number": 1, "hook_type": "visual"},
        "bad row",
    ]}, "semantic")
    assert rows[0]["source"] == "semantic"
    assert rows[0]["candidate_id"] == "SEM-1"
    assert errors == ["Candidate row 2 is not an object."]
    heuristic, _ = normalize_candidates([{"candidate_id": "H-1", "episode_number": 2}], "heuristic")
    assert heuristic[0]["source"] == "heuristic"


def test_dashboard_data_loads_valid_episode_and_empty_states(tmp_path):
    episode = ParsedEpisode(episode_number=1, scenes=[Scene(scene_number=1)])
    episodes_file = tmp_path / "episodes.json"
    episodes_file.write_text(json.dumps([episode.model_dump(mode="json")]), encoding="utf-8")
    keys = ["episodes", "analysis", "discoveries", "semantic_candidates", "heuristic_candidates", "validation", "validation_markdown", "review_decisions", "blind_viewer_pilot_report", "blind_viewer_pilot_report_markdown", "blind_viewer_pilot_selection", "discovery_comparison", "discovery_comparison_markdown", "series_analysis", "series_analysis_markdown", "composite_candidates", "final_rankings", "final_rankings_csv", "final_ranking_report"]
    paths = {key: tmp_path / f"{key}.json" for key in keys}
    paths["episodes"] = episodes_file
    data = load_dashboard_data(paths)
    assert data["data_mode"] == "local"
    assert data["demo_mode"] is False
    assert len(data["episodes"]) == 1
    assert data["semantic_candidates"] == []
    assert data["composite_candidates"] == []
    assert not data["files"]["semantic_candidates"]["exists"]


def test_local_screenplay_data_takes_precedence_over_demo(tmp_path):
    episode = ParsedEpisode(episode_number=77, source_file="local-test.txt", scenes=[Scene(scene_number=1)])
    source = tmp_path / "episodes.json"
    source.write_text(json.dumps([episode.model_dump(mode="json")]), encoding="utf-8")

    data = load_dashboard_data({"episodes": source})

    assert data["data_mode"] == "local"
    assert [item.episode_number for item in data["episodes"]] == [77]
    assert data["demo_candidates"] == []


def test_missing_local_data_automatically_loads_synthetic_demo(tmp_path):
    data = load_dashboard_data({"episodes": tmp_path / "missing-parsed.json"})

    assert data["data_mode"] == "demo"
    assert data["demo_mode"] is True
    assert len(data["episodes"]) == 3
    assert sum(len(episode.events) for episode in data["episodes"]) == 12
    assert all(event.event_id.startswith("DEMO_") for episode in data["episodes"] for event in episode.events)


def test_demo_banner_contains_required_warning():
    from promoscout.dashboard.components import render_demo_banner

    class Capture:
        value = ""

        def markdown(self, value, **kwargs):
            self.value = value

    capture = Capture()
    render_demo_banner(capture)
    assert "DEMO MODE — Illustrative synthetic data. Not actual Kuku screenplay results." in capture.value


def test_demo_artifacts_are_labeled_and_not_human_approved(tmp_path):
    data = load_dashboard_data({"episodes": tmp_path / "missing-parsed.json"})

    assert len(data["demo_candidates"]) == 3
    assert all(row["source"] == "demo" and row["synthetic"] for row in data["demo_candidates"])
    assert all(result["synthetic_demo"] for result in data["blind_viewer_evaluations"].values())
    assert len(data["composite_candidates"]) == 1
    assert data["composite_candidates"][0]["candidate_origin"] == "synthetic"
    assert data["files"]["review_decisions"]["data"]["decisions"] == {}
    assert all(row["human_decision"] == "pending (demo)" for row in data["final_rankings"]["leaderboards"]["combined"])
    assert data["semantic_candidates"] == []
    assert data["heuristic_candidates"] == []


def test_confidential_and_generated_data_paths_remain_gitignored():
    root = Path(__file__).resolve().parents[1]
    ignored = (root / ".gitignore").read_text(encoding="utf-8")
    for path in ("data/input/", "data/processed/", "data/outputs/", "data/cache/", "submission/raw_outputs/"):
        assert path in ignored


def test_demo_banner_and_all_pages_render_when_real_data_is_missing(tmp_path):
    import promoscout.dashboard.data_loader as loader

    original_paths = loader.DEFAULT_PATHS
    missing_paths = {key: str(tmp_path / f"missing-{key}.json") for key in [
        "episodes", "analysis", "discoveries", "semantic_candidates", "heuristic_candidates",
        "validation", "validation_markdown", "review_decisions", "blind_viewer",
        "blind_viewer_pilot_report", "blind_viewer_pilot_report_markdown",
        "blind_viewer_pilot_selection", "discovery_comparison", "discovery_comparison_markdown",
        "series_analysis", "series_analysis_markdown", "composite_candidates",
        "final_rankings", "final_rankings_csv", "final_ranking_report",
    ]}
    app_script = tmp_path / "demo_app.py"
    app_script.write_text(
        "from pathlib import Path\nimport promoscout.dashboard.data_loader as loader\n"
        f"loader.DEFAULT_PATHS = {missing_paths!r}\n"
        "import app\napp.main()\n",
        encoding="utf-8",
    )
    try:
        app = AppTest.from_file(str(app_script), default_timeout=30).run()
        assert len(app.exception) == 0
        assert any("DEMO MODE — Illustrative synthetic data. Not actual Kuku screenplay results." in item.value for item in app.markdown)
        for page_hash, _page in list(app._registered_pages.items()):
            app._page_hash = page_hash
            app.run()
            assert len(app.exception) == 0
    finally:
        loader.DEFAULT_PATHS = original_paths


def test_dashboard_loads_pilot_report_and_selection_when_present(tmp_path):
    report = {"overall_results": {"selected_candidates": 10, "episodes_represented": 10}}
    selection = {"selected_candidates": [{"candidate_id": "EP001_PROMO_001"}]}
    report_path = tmp_path / "pilot_report.json"
    selection_path = tmp_path / "pilot_selection.json"
    markdown_path = tmp_path / "pilot_report.md"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    selection_path.write_text(json.dumps(selection), encoding="utf-8")
    markdown_path.write_text("# Pilot report", encoding="utf-8")
    data = load_dashboard_data({
        "blind_viewer_pilot_report": report_path,
        "blind_viewer_pilot_selection": selection_path,
        "blind_viewer_pilot_report_markdown": markdown_path,
    })
    assert data["blind_viewer_pilot_report"] == report
    assert data["blind_viewer_pilot_selection"] == selection
    assert data["blind_viewer_pilot_report_markdown"] == "# Pilot report"


def test_dashboard_loads_discovery_comparison_for_exports(tmp_path):
    report = {"episode_number": 1, "method_statistics": {"semantic": {}, "heuristic": {}}}
    json_path, markdown_path = tmp_path / "comparison.json", tmp_path / "comparison.md"
    json_path.write_text(json.dumps(report), encoding="utf-8")
    markdown_path.write_text("# Comparison", encoding="utf-8")
    data = load_dashboard_data({"discovery_comparison": json_path,
                                "discovery_comparison_markdown": markdown_path})
    assert data["discovery_comparison"] == report
    assert data["discovery_comparison_markdown"] == "# Comparison"


def test_dashboard_loads_composite_candidates_without_assuming_episode_number(tmp_path):
    candidate = {"candidate_id": "COMPOSITE_ARC_001", "segments": [{"episode_number": 1}, {"episode_number": 2}]}
    path = tmp_path / "composites.json"
    path.write_text(json.dumps({"candidates": [candidate]}), encoding="utf-8")
    data = load_dashboard_data({"composite_candidates": path})
    assert data["composite_candidates"] == [candidate]


def test_dashboard_loads_final_ranking_artifacts_and_handles_missing_files(tmp_path):
    report = {"leaderboards": {"continuous": [], "composite": [], "combined": []}}
    json_path, csv_path, md_path = tmp_path / "ranking.json", tmp_path / "ranking.csv", tmp_path / "ranking.md"
    json_path.write_text(json.dumps(report), encoding="utf-8")
    csv_path.write_text("candidate_id,weighted_score\n", encoding="utf-8")
    md_path.write_text("# Final provisional rankings", encoding="utf-8")
    data = load_dashboard_data({"final_rankings": json_path, "final_rankings_csv": csv_path,
                                "final_ranking_report": md_path})
    assert data["final_rankings"] == report
    assert data["final_rankings_csv"].startswith("candidate_id")
    assert "provisional" in data["final_ranking_report"]
    missing = load_dashboard_data({"final_rankings": tmp_path / "absent.json",
                                   "final_rankings_csv": tmp_path / "absent.csv",
                                   "final_ranking_report": tmp_path / "absent.md"})
    assert missing["final_rankings"] is None
    assert missing["final_rankings_csv"] is None
    assert missing["final_ranking_report"] is None


def test_review_decision_persistence_is_atomic_and_source_scoped(tmp_path):
    path = tmp_path / "review_decisions.json"
    save_review_decision("semantic", "EP001_PROMO_001", "approved", "Strong opening", path)
    save_review_decision("heuristic", "EP001_PROMO_001", "later", "Compare later", path)
    saved = load_review_decisions(path)
    assert saved["decisions"][review_key("semantic", "EP001_PROMO_001")]["decision"] == "approved"
    assert saved["decisions"][review_key("heuristic", "EP001_PROMO_001")]["decision"] == "later"
    assert saved["decisions"][review_key("semantic", "EP001_PROMO_001")]["note"] == "Strong opening"
    assert not list(tmp_path.glob("*.tmp"))


def test_export_generation_for_json_and_csv():
    rows = [{"candidate_id": "C1", "episode_number": 1, "included_event_ids": ["E1", "E2"], "score": 0.8}]
    decoded = json.loads(json_export_bytes(rows))
    assert decoded == rows
    csv_text = candidate_csv_bytes(rows).decode("utf-8")
    assert "candidate_id,episode_number" in csv_text
    assert '""E1"", ""E2""' in csv_text
    assert candidate_csv_bytes([]) == b""


def test_streamlit_app_loads_and_registers_six_navigation_pages():
    root = Path(__file__).resolve().parents[1]
    app = AppTest.from_file(str(root / "app.py"), default_timeout=30).run()
    assert len(app.exception) == 0
    registered = list(app._registered_pages.items())
    page_names = {page["page_name"] for _, page in registered}
    assert page_names == {"Overview", "Episode Explorer", "Candidate Review", "Rankings", "Exports", "Series Workspace", "Composite Promos"}
    for page_hash, _page in registered:
        app._page_hash = page_hash
        app.run()
        assert len(app.exception) == 0


def test_rankings_page_shows_empty_state_when_final_report_is_missing(tmp_path):
    app_script = tmp_path / "rankings_empty.py"
    app_script.write_text('''
import promoscout.dashboard.pages as pages
pages.load_dashboard_data = lambda: {
    "episode_errors": [], "candidate_errors": [], "files": {},
    "final_rankings": None, "blind_viewer_pilot_report": None,
}
pages.rankings_page()
''', encoding="utf-8")
    app = AppTest.from_file(str(app_script)).run()
    assert len(app.exception) == 0
    assert any("Final rankings have not been generated" in item.value for item in app.markdown)


def test_page_eyebrow_has_legible_high_contrast_accent():
    assert ".ps-eyebrow" in CSS
    assert "color:#FF777B" in CSS
    assert ".ps-eyebrow::before" in CSS
    assert "padding-top: 4rem" in CSS
    assert "padding:3.5rem 1rem 3rem" in CSS
    assert "[data-testid=\"stHeader\"] { background: var(--ps-bg); }" in CSS

