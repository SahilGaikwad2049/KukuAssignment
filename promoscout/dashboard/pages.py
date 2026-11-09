"""Streamlit pages for overview, exploration, review, rankings, and exports."""
from __future__ import annotations

from html import escape

import streamlit as st

from promoscout.dashboard.components import (
    render_empty_state, render_event_item, render_header, render_metrics,
    render_status, section_divider,
)
from promoscout.dashboard.data_loader import DEFAULT_PATHS, load_dashboard_data
from promoscout.dashboard.export_utils import candidate_csv_bytes, json_export_bytes
from promoscout.dashboard.review_manager import (
    ReviewStoreError, load_review_decisions, review_key, save_review_decision,
)
from promoscout.generation.preview import reconstruct_candidate_preview


def _data():
    data = load_dashboard_data()
    for error in data["episode_errors"] + data["candidate_errors"]:
        st.warning(error)
    for key, file in data["files"].items():
        if file["error"]:
            st.error(f"Could not read {key} file: {file['error']}")
    return data


def _review_state():
    try:
        return load_review_decisions(), None
    except ReviewStoreError as exc:
        return {"schema_version": "promo-review-decisions-v1", "decisions": {}}, str(exc)


def _active_review_decisions(data: dict, review_data: dict) -> list[str]:
    decisions = review_data.get("decisions", {})
    candidates = data["semantic_candidates"] + data["heuristic_candidates"]
    return [
        decisions[review_key(row["source"], row["candidate_id"])].get("decision")
        for row in candidates
        if isinstance(decisions.get(review_key(row["source"], row["candidate_id"])), dict)
    ]


def _analysis_for(data: dict, episode_number: int):
    row = data["analysis"].get(str(episode_number))
    if isinstance(row, dict) and isinstance(row.get("episode_analysis"), dict):
        return row["episode_analysis"]
    return row if isinstance(row, dict) else None


def _discoveries_for(data: dict, episode_number: int):
    value = data["discoveries"].get(str(episode_number), [])
    return value if isinstance(value, list) else []


def _pipeline_status(data: dict, review_data: dict) -> list[tuple[str, str, str]]:
    episode_count = len(data["episodes"])
    analysis_count = sum(_analysis_for(data, e.episode_number) is not None for e in data["episodes"])
    discovery_count = sum(len(_discoveries_for(data, e.episode_number)) for e in data["episodes"])
    semantic_count = len(data["semantic_candidates"])
    validation = data["validation"] or {}
    decision_values = _active_review_decisions(data, review_data)
    reviewed = sum(value in {"approved", "rejected", "later"} for value in decision_values)
    return [
        ("Parsed", "Complete" if episode_count else "Waiting", f"{episode_count} episodes"),
        ("Semantic analysis", "Complete" if analysis_count == episode_count and episode_count else "In progress" if analysis_count else "Waiting", f"{analysis_count}/{episode_count} episodes"),
        ("Discovery", "Complete" if discovery_count else "Waiting", f"{discovery_count} events"),
        ("Candidates", "Complete" if semantic_count else "Waiting", f"{semantic_count} semantic windows"),
        ("Validation", "Complete" if validation and validation.get("validation_summary", {}).get("total_errors") == 0 else "Needs attention" if validation else "Waiting", "Pilot report" if validation else "No report yet"),
        ("Human review", "Complete" if semantic_count and reviewed >= semantic_count else "In progress" if reviewed else "Waiting", f"{reviewed}/{semantic_count} reviewed"),
    ]


def overview_page():
    data = _data()
    review, review_error = _review_state()
    if review_error:
        st.warning(review_error)
    render_header(st, "Story intelligence, made clear", "Explore source events, inspect promo candidates, and make the final call.", "PROMOSCOUT / WORKSPACE")
    episodes = data["episodes"]
    total_events = sum(len(episode.events) for episode in episodes)
    decisions = _active_review_decisions(data, review)
    render_metrics(st, [
        ("Episodes", len(episodes), "Parsed source scripts"),
        ("Source events", total_events, "Dialogue, action, camera and sound"),
        ("Promo candidates", len(data["semantic_candidates"]) + len(data["heuristic_candidates"]), f"{len(data['semantic_candidates'])} semantic · {len(data['heuristic_candidates'])} heuristic"),
        ("Human reviews", len(decisions), f"{decisions.count('approved')} approved · {decisions.count('rejected')} rejected"),
    ])
    st.subheader("Pipeline status")
    pipeline = _pipeline_status(data, review)
    # A compact status row keeps the landing page scannable; detailed artifacts
    # remain available on the relevant pages.
    markup = '<div class="ps-pipeline-compact">'
    for name, state, detail in (pipeline[0], pipeline[1], pipeline[3], pipeline[5]):
        css = "done" if state == "Complete" else "attention" if state == "Needs attention" else "active" if state == "In progress" else ""
        markup += f'<span class="ps-status {css}"><span class="ps-dot"></span><b>{escape(name)}</b> · {escape(state)} · {escape(detail)}</span>'
    markup += "</div>"
    st.markdown(markup, unsafe_allow_html=True)
    if not data["semantic_candidates"]:
        render_empty_state(st, "Semantic discovery is still pending", "The current candidate set is heuristic. Semantic results will appear here after an episode analysis run.")
    st.caption("Use Candidate Review for clip previews and decisions, or Rankings for validated AI-assisted comparisons. Scores are provisional; human review stays separate.")


def episode_explorer_page():
    data = _data()
    render_header(st, "Episode Explorer", "Move through the screenplay one scene and one source event at a time.", "PROMOSCOUT / SOURCE")
    if not data["episodes"]:
        render_empty_state(st, "Parsed screenplay unavailable", "Add a valid parsed_all_episodes.json file under data/processed.")
        return
    numbers = [episode.episode_number for episode in data["episodes"]]
    selected = st.selectbox("Episode", numbers, format_func=lambda number: f"Episode {number}")
    episode = next(item for item in data["episodes"] if item.episode_number == selected)
    events = sorted(episode.events, key=lambda event: (event.start_seconds, event.source_line_start, event.end_seconds))
    duration = max((event.end_seconds for event in events), default=0)
    total_minutes, total_seconds = divmod(int(duration), 60)
    m1, m2, m3 = st.columns(3)
    m1.metric("Events", len(events))
    m2.metric("Scenes", len(episode.scenes))
    m3.metric("Screenplay duration", f"{total_minutes}:{total_seconds:02d}")
    if episode.logline:
        st.info(episode.logline)
    selected_types = st.multiselect("Event types", sorted({event.event_type for event in events}), default=sorted({event.event_type for event in events}), key="explorer_event_types")
    filtered = [event for event in events if event.event_type in selected_types]
    st.caption(f"Showing {len(filtered)} of {len(events)} source events in chronological order.")
    events_by_id = {event.event_id: event for event in filtered}
    for scene in episode.scenes:
        scene_events = [event for event in scene.events if event.event_id in events_by_id]
        if not scene_events:
            continue
        heading = scene.heading or f"Scene {scene.scene_number}"
        st.markdown(f"#### Scene {scene.scene_number:02d} · {escape(heading)}")
        st.caption(f"{len(scene_events)} visible source events")
        for event in sorted(scene_events, key=lambda item: (item.start_seconds, item.source_line_start, item.end_seconds)):
            with st.expander(f"{event.start_timestamp} · {event.event_type.replace('_', ' ').title()} · {event.speaker or event.label}", expanded=False):
                render_event_item(st, event.model_dump(mode="json"))
    analysis = _analysis_for(data, selected)
    section_divider(st)
    st.subheader("Narrative analysis")
    if not analysis:
        render_empty_state(st, "No saved analysis for this episode", "Narrative analysis appears here after semantic analysis has completed.")
        return
    st.write(analysis.get("episode_summary", ""))
    summary_fields = [
        ("Important characters", "important_characters"),
        ("Character relationships", "character_relationships"),
        ("Main conflict", "main_conflict"),
        ("Story developments", "important_story_developments"),
        ("Revelations", "major_revelations"),
        ("Emotional turning points", "emotional_turning_points"),
        ("Unresolved questions", "unresolved_questions"),
        ("Narrative progression", "narrative_progression"),
    ]
    for title, key in summary_fields:
        value = analysis.get(key)
        if value:
            with st.expander(title, expanded=key in {"main_conflict", "important_characters"}):
                if isinstance(value, list):
                    for line in value:
                        st.markdown(f"- {line}")
                else:
                    st.write(value)


def _candidate_preview(candidate: dict, episode_map: dict[int, Any]) -> list[dict] | None:
    episode = episode_map.get(candidate.get("episode_number"))
    if not episode:
        return None
    try:
        payload = {key: value for key, value in candidate.items() if key != "source"}
        return reconstruct_candidate_preview(episode, payload)["events"]
    except Exception:
        return None


def _render_blind_viewer_result(st, evaluation: dict | None) -> None:
    if not isinstance(evaluation, dict):
        st.info("Blind-viewer evaluation is pending. No result has been saved for this candidate.")
        return
    assessment = evaluation.get("assessment")
    if not isinstance(assessment, dict):
        st.warning("Saved blind-viewer result has an invalid structure.")
        return
    score_fields = [
        ("Standalone clarity", "standalone_clarity"),
        ("Hook effectiveness", "hook_effectiveness"),
        ("Conflict comprehension", "conflict_comprehension"),
        ("Stakes clarity", "stakes_clarity"),
        ("Cliffhanger effectiveness", "cliffhanger_effectiveness"),
        ("Desire to continue", "desire_to_continue"),
    ]
    for start in range(0, len(score_fields), 3):
        columns = st.columns(3)
        for column, (label, key) in zip(columns, score_fields[start:start + 3]):
            dimension = assessment.get(key)
            if isinstance(dimension, dict) and dimension.get("score") is not None:
                column.metric(label, f"{dimension['score']} / 5")
            else:
                column.metric(label, "—")
    verdict = assessment.get("overall_viewer_verdict", "unknown")
    st.markdown(f"**Viewer verdict: {escape(str(verdict).title())}** · Confidence {assessment.get('confidence', '—')}")
    st.write(assessment.get("standalone_story_reconstruction", ""))
    for label, key in [
        ("Directly shown or stated", "directly_shown_or_stated"),
        ("Reasonable inferences", "reasonable_inferences"),
        ("Unclear or unestablished", "unclear_or_unestablished"),
        ("Characters as understood", "characters_the_viewer_believes_are_involved"),
        ("Main conflict", "main_conflict_as_understood"),
        ("Perceived stakes", "perceived_stakes"),
        ("Unanswered question", "main_unanswered_question"),
        ("Confusing elements", "confusing_elements"),
        ("Missing context wished for", "missing_context_the_viewer_wishes_for"),
        ("Verdict explanation", "verdict_explanation"),
    ]:
        value = assessment.get(key)
        if value:
            with st.expander(label):
                st.write(value)
    for label, key in score_fields:
        dimension = assessment.get(key)
        if isinstance(dimension, dict):
            with st.expander(f"{label} evidence and rationale"):
                st.write(dimension.get("rationale", ""))
                for evidence in dimension.get("evidence", []):
                    st.caption(f"{evidence.get('event_id', 'Unknown event')}: “{evidence.get('excerpt', '')}” — {evidence.get('why_it_matters', '')}")


def _render_blind_pilot_summary(data: dict) -> None:
    report = data.get("blind_viewer_pilot_report")
    if not isinstance(report, dict):
        st.caption("No multi-episode blind-viewer pilot report is saved yet.")
        return
    summary = report.get("overall_results", {})
    with st.expander("Multi-episode blind-viewer pilot", expanded=False):
        render_metrics(st, [
            ("Selected", summary.get("selected_candidates", 0), "Candidates in manifest"),
            ("Evaluated", summary.get("successfully_evaluated", 0), "Saved valid results"),
            ("Failed", summary.get("failed", 0), "Request or validation failures"),
            ("Episodes", summary.get("episodes_represented", 0), "Distinct episodes in sample"),
        ])
        st.caption(
            f"Source: {report.get('candidate_source', '—')} · Model: {report.get('model_id', '—')} · "
            f"Reused from cache: {summary.get('reused_from_cache', 0)} · "
            f"Episode coverage: {', '.join(map(str, summary.get('episode_numbers', []))) or 'none'}"
        )
        dimensions = report.get("score_distributions", {})
        if dimensions:
            rows = [{
                "Dimension": label.replace("_", " ").title(),
                "Average": values.get("average"),
                "Median": values.get("median"),
                "Range": f"{values.get('minimum', '—')}–{values.get('maximum', '—')}",
            } for label, values in dimensions.items()]
            st.dataframe(rows, hide_index=True, use_container_width=True)
        selection = data.get("blind_viewer_pilot_selection")
        if isinstance(selection, dict):
            with st.expander("Selected candidate manifest"):
                st.caption(f"Seed {selection.get('sample_seed')} · requested {selection.get('requested_episode_count')} episode(s), selected {selection.get('selected_episode_count')}")
                for row in selection.get("selected_candidates", []):
                    st.markdown(
                        f"- **EP{row.get('episode_number', 0):03d} · {row.get('candidate_id', 'unknown')}** "
                        f"· {row.get('candidate_source', 'unknown')} · {row.get('duration_seconds', '—')}s · "
                        f"{row.get('hook_type', 'unknown')} hook"
                    )
        if data.get("blind_viewer_pilot_report_markdown"):
            st.download_button(
                "Download pilot summary report",
                data["blind_viewer_pilot_report_markdown"],
                file_name="blind_viewer_pilot_report.md",
                mime="text/markdown",
                key="download-blind-viewer-pilot-report",
            )
        st.caption("AI verdicts and dimension scores are separate from human review decisions.")


def _render_candidate_card(candidate: dict, episode_map: dict[int, Any], decisions: dict,
                           blind_evaluations: dict,
                           review_error: str | None = None):
    source = candidate.get("source", "semantic")
    cid = str(candidate.get("candidate_id", "unknown"))
    key = review_key(source, cid)
    decision = decisions.get(key, {}) if isinstance(decisions.get(key, {}), dict) else {}
    review_status = decision.get("decision", "pending")
    with st.container(border=True):
        left, right = st.columns([3, 1])
        with left:
            st.markdown(f"**{cid}** · Episode {candidate.get('episode_number', '—')}")
            st.caption(f"{source.title()} candidate · family {candidate.get('candidate_family_id', '—')}")
        with right:
            render_status(st, "Review", review_status.title())
        meta = f"{candidate.get('start_timestamp', '—')} → {candidate.get('end_timestamp', '—')} · {candidate.get('duration_seconds', '—')}s · {candidate.get('hook_type', 'unknown')}"
        st.markdown(f"<div class='ps-meta'>{escape(meta)}</div>", unsafe_allow_html=True)
        score_key = "heuristic_score" if candidate.get("heuristic_score") is not None else "hook_score" if candidate.get("hook_score") is not None else None
        if score_key:
            st.caption(f"Existing {score_key.replace('_', ' ')}: {candidate[score_key]} · pipeline signal, not a quality judgment")
        st.markdown(f"<div class='ps-hook'><b>Hook</b><br>{escape(str(candidate.get('hook_text', '')))}<div class='ps-meta'>Event {escape(str(candidate.get('hook_event_id', '—')))}</div></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='ps-ending'><b>Potential ending</b> <span class='ps-meta'>· not a confirmed cliffhanger</span><br>{escape(str(candidate.get('ending_text', '')))}<div class='ps-meta'>Event {escape(str(candidate.get('ending_event_id', '—')))}</div></div>", unsafe_allow_html=True)
        with st.expander("Chronological screenplay preview", expanded=False):
            preview = _candidate_preview(candidate, episode_map)
            if preview is None:
                st.warning("Source events are unavailable or could not reconstruct this candidate preview.")
                st.caption("Included event IDs: " + ", ".join(candidate.get("included_event_ids", [])))
            elif not preview:
                st.info("No source events overlap the candidate window.")
            else:
                for event in preview:
                    render_event_item(st, event, relative_start=event.get("relative_start_seconds"))
        with st.expander("Blind-viewer evaluation", expanded=False):
            _render_blind_viewer_result(
                st,
                blind_evaluations.get(review_key(source, cid)),
            )
        if source in {"semantic", "heuristic"}:
            if review_error:
                st.warning(review_error)
            note_key = f"review-note-{source}-{cid}"
            form_key = f"review-form-{source}-{cid}"
            with st.form(form_key):
                note = st.text_area("Optional review note", value=decision.get("note", ""), key=note_key, max_chars=2000)
                choices = ["Pending", "Approve", "Reject", "Later"]
                current = {"approved": "Approve", "rejected": "Reject", "later": "Later"}.get(review_status, "Pending")
                selected = st.selectbox("Human decision", choices, index=choices.index(current), key=f"review-choice-{source}-{cid}")
                submitted = st.form_submit_button("Save review decision", use_container_width=True)
            if submitted and selected != "Pending":
                try:
                    decision_value = {"Approve": "approved", "Reject": "rejected", "Later": "later"}[selected]
                    save_review_decision(source, cid, decision_value, note)
                    st.success("Decision saved locally.")
                    st.rerun()
                except ReviewStoreError as exc:
                    st.error(str(exc))


def _candidate_controls(candidates: list[dict], prefix: str):
    if not candidates:
        return []
    filters1, filters2 = st.columns([1, 2])
    episodes = sorted({row["episode_number"] for row in candidates})
    with filters1:
        selected_episodes = st.multiselect("Episodes", episodes, default=episodes, key=f"{prefix}-episodes")
    types = sorted({str(row.get("hook_type", "unknown")) for row in candidates})
    with filters2:
        selected_types = st.multiselect("Hook types", types, default=types, key=f"{prefix}-types")
    filtered = [row for row in candidates if row["episode_number"] in selected_episodes and str(row.get("hook_type", "unknown")) in selected_types]
    score_choices = ["Chronological", "Heuristic score", "Hook score", "Duration"]
    selected_sort = st.selectbox("Sort", score_choices, key=f"{prefix}-sort")
    if selected_sort == "Heuristic score":
        scored = [row for row in filtered if row.get("heuristic_score") is not None]
        unscored = [row for row in filtered if row.get("heuristic_score") is None]
        scored.sort(key=lambda row: row["heuristic_score"], reverse=True)
        filtered = scored + unscored
    elif selected_sort == "Hook score":
        scored = [row for row in filtered if row.get("hook_score") is not None]
        unscored = [row for row in filtered if row.get("hook_score") is None]
        scored.sort(key=lambda row: row["hook_score"], reverse=True)
        filtered = scored + unscored
    elif selected_sort == "Duration":
        filtered.sort(key=lambda row: row.get("duration_seconds", 0), reverse=True)
    else:
        filtered.sort(key=lambda row: (row["episode_number"], row.get("start_seconds", 0)))
    return filtered


def candidate_review_page():
    data = _data()
    render_header(st, "Candidate Review", "Inspect source-grounded windows, rebuild their screenplay previews, and record human decisions.", "PROMOSCOUT / REVIEW DESK")
    _render_blind_pilot_summary(data)
    source_choice = st.radio("Candidate source", ["Both", "Semantic", "Heuristic"], horizontal=True)
    candidates = data["semantic_candidates"] + data["heuristic_candidates"]
    if source_choice != "Both":
        candidates = [row for row in candidates if row["source"] == source_choice.lower()]
    if not data["semantic_candidates"]:
        st.info("Semantic candidates are not available yet. Heuristic candidates are shown separately and are not semantic discoveries.")
    if not candidates:
        render_empty_state(st, "No candidates to review", "Generate semantic candidates or switch to the heuristic source to inspect existing candidates.")
        return
    filtered = _candidate_controls(candidates, "review")
    review, error = _review_state()
    if error:
        st.error(error)
    st.caption(f"{len(filtered)} candidate(s) match these filters. Existing scores are pipeline signals, not quality ratings.")
    page_size = 12
    total_pages = max(1, (len(filtered) + page_size - 1) // page_size)
    if st.session_state.get("review-page", 1) > total_pages:
        st.session_state["review-page"] = total_pages
    page = st.number_input("Page", min_value=1, max_value=total_pages, value=1, step=1, key="review-page")
    episodes = {episode.episode_number: episode for episode in data["episodes"]}
    for candidate in filtered[(page - 1) * page_size:page * page_size]:
        _render_candidate_card(
            candidate, episodes, review.get("decisions", {}),
            data["blind_viewer_evaluations"], error,
        )


def rankings_page():
    data = _data()
    render_header(st, "Rankings", "Provisional rankings only for candidates with validated source structure and blind-viewer evidence.", "PROMOSCOUT / SIGNALS")
    report = data.get("final_rankings")
    if not isinstance(report, dict):
        render_empty_state(st, "Final rankings have not been generated", "Run `python -m promoscout.ranking.final_ranker` to validate local candidates and build the separate continuous, composite, and combined leaderboards.")
        _render_blind_pilot_summary(data)
        return
    stats = [
        ("Ranked", len(report.get("leaderboards", {}).get("combined", [])), "Validated AI assessments"),
        ("Pending", len(report.get("pending_candidates", [])), "No validated assessment"),
        ("Failed", len(report.get("failed_evaluations", [])), "Excluded from ranking"),
        ("Structural failures", len(report.get("structurally_ineligible_candidates", [])), "Excluded from ranking"),
    ]
    render_metrics(st, stats)
    st.info(report.get("ranking_interpretation", "AI-assisted provisional ranking; human review remains separate."))
    tabs = st.tabs(["Continuous", "Composite", "Combined"])
    for tab, key in zip(tabs, ("continuous", "composite", "combined")):
        with tab:
            rows = report.get("leaderboards", {}).get(key, [])
            if not rows:
                render_empty_state(st, f"No eligible {key} candidates", "This leaderboard requires real candidates with passed structural validation and successfully validated blind-viewer assessments.")
                continue
            for row in rows:
                with st.container(border=True):
                    title_columns = st.columns([.5, 3, 1.2])
                    title_columns[0].markdown(f"**{row['rank']:02d}**")
                    title_columns[1].markdown(f"**{row['candidate_id']}** · {row['source_type'].title()} · Episodes {', '.join(map(str, row['episodes']))}")
                    title_columns[2].metric("Provisional / 100", row["weighted_score"])
                    st.caption(f"{row['duration_seconds']}s · {row['blind_viewer_verdict']} · Structural: {row['structural_validation_status']} · Human: {row['human_decision']} · {'Approved' if row['final_selection_eligible'] else 'Not approved for final selection'}")
                    score_items = list(row["dimension_scores"].items())
                    for offset in range(0, len(score_items), 3):
                        score_columns = st.columns(3)
                        for column, (label, score_value) in zip(score_columns, score_items[offset:offset + 3]):
                            column.metric(label.replace("_", " ").title(), f"{score_value}/5" if score_value is not None else "—")
                    if row.get("reviewer_note"):
                        st.caption(f"Reviewer note: {row['reviewer_note']}")
                    st.caption(row["inclusion_or_exclusion_reason"])
    pending = report.get("pending_candidates", [])
    failed = report.get("failed_evaluations", [])
    excluded = report.get("structurally_ineligible_candidates", [])
    with st.expander(f"Pending / failed / ineligible candidates ({len(pending) + len(failed) + len(excluded)})"):
        for title, rows in (("Pending", pending), ("Evaluation failed", failed), ("Structurally ineligible", excluded)):
            st.markdown(f"**{title}: {len(rows)}**")
            for row in rows[:30]:
                reason = row.get("evaluation_status_reason") or "; ".join(row.get("structural_errors", []))
                st.caption(f"{row['candidate_id']} · {row['source_type']} · {reason}")
            if len(rows) > 30:
                st.caption(f"Showing the first 30 of {len(rows)}. Full records are in final_rankings.json.")
    _render_blind_pilot_summary(data)


def _download_pair(st, label: str, rows: list[dict], key: str):
    first, second = st.columns(2)
    first.download_button(f"Download {label} JSON", json_export_bytes(rows), f"{key}.json", "application/json", key=f"download-{key}-json", use_container_width=True)
    second.download_button(f"Download {label} CSV", candidate_csv_bytes(rows), f"{key}.csv", "text/csv", key=f"download-{key}-csv", use_container_width=True)


def exports_page():
    data = _data()
    render_header(st, "Exports", "Download local pipeline artifacts and human-approved decisions for the next production step.", "PROMOSCOUT / OUTPUTS")
    st.subheader("Candidate exports")
    _download_pair(st, "semantic candidates", data["semantic_candidates"], "semantic_candidates")
    _download_pair(st, "heuristic candidates", data["heuristic_candidates"], "heuristic_candidates")
    _download_pair(st, "composite candidates", data.get("composite_candidates", []), "composite_candidates")
    review, review_error = _review_state()
    if review_error:
        st.error(review_error)
    approved = []
    reviewable = data["semantic_candidates"] + data["heuristic_candidates"] + [
        {**row, "source": "composite"} for row in data.get("composite_candidates", [])
    ]
    for row in reviewable:
        key = review_key(row["source"], row["candidate_id"])
        if review.get("decisions", {}).get(key, {}).get("decision") == "approved":
            approved.append(row)
    _download_pair(st, "approved candidates", approved, "approved_candidates")
    section_divider(st)
    st.subheader("Review and validation")
    decisions_payload = review if not review_error else {"schema_version": "promo-review-decisions-v1", "decisions": {}}
    st.download_button("Download review decisions JSON", json_export_bytes(decisions_payload), "review_decisions.json", "application/json", key="download-review-decisions", use_container_width=True)
    validation_file = data["files"]["validation"]
    validation_md_file = data["files"]["validation_markdown"]
    c1, c2 = st.columns(2)
    if validation_file["data"] is not None:
        c1.download_button("Download pilot validation JSON", json_export_bytes(validation_file["data"]), "pilot_validation.json", "application/json", key="download-validation-json", use_container_width=True)
    else:
        c1.button("Pilot validation JSON unavailable", disabled=True, use_container_width=True)
    if validation_md_file["data"] is not None:
        c2.download_button("Download pilot validation Markdown", validation_md_file["data"], "pilot_validation.md", "text/markdown", key="download-validation-md", use_container_width=True)
    else:
        c2.button("Pilot validation Markdown unavailable", disabled=True, use_container_width=True)
    pilot_report = data["files"].get("blind_viewer_pilot_report", {}).get("data")
    pilot_report_md = data["files"].get("blind_viewer_pilot_report_markdown", {}).get("data")
    pilot_selection = data["files"].get("blind_viewer_pilot_selection", {}).get("data")
    if pilot_report is not None or pilot_report_md is not None:
        st.subheader("Blind-viewer pilot")
        c3, c4 = st.columns(2)
        if pilot_report is not None:
            c3.download_button("Download blind-viewer pilot JSON", json_export_bytes(pilot_report), "blind_viewer_pilot_report.json", "application/json", key="download-blind-pilot-json", use_container_width=True)
        if pilot_report_md is not None:
            c4.download_button("Download blind-viewer pilot Markdown", pilot_report_md, "blind_viewer_pilot_report.md", "text/markdown", key="download-blind-pilot-md", use_container_width=True)
    if pilot_selection is not None:
        st.download_button("Download pilot selection manifest", json_export_bytes(pilot_selection), "blind_viewer_pilot_selection.json", "application/json", key="download-blind-pilot-selection", use_container_width=True)
    comparison = data["files"].get("discovery_comparison", {}).get("data")
    comparison_md = data["files"].get("discovery_comparison_markdown", {}).get("data")
    if comparison is not None or comparison_md is not None:
        st.subheader("Semantic vs heuristic comparison")
        c5, c6 = st.columns(2)
        if comparison is not None:
            c5.download_button("Download discovery comparison JSON", json_export_bytes(comparison), "discovery_comparison.json", "application/json", key="download-discovery-comparison-json", use_container_width=True)
        if comparison_md is not None:
            c6.download_button("Download discovery comparison Markdown", comparison_md, "discovery_comparison.md", "text/markdown", key="download-discovery-comparison-md", use_container_width=True)
    ranking_json = data["files"].get("final_rankings", {}).get("data")
    ranking_csv = data["files"].get("final_rankings_csv", {}).get("data")
    ranking_report = data["files"].get("final_ranking_report", {}).get("data")
    st.subheader("Final provisional rankings")
    r1, r2, r3 = st.columns(3)
    if ranking_json is not None:
        r1.download_button("Download final rankings JSON", json_export_bytes(ranking_json), "final_rankings.json", "application/json", key="download-final-rankings-json", use_container_width=True)
    else:
        r1.button("Final rankings JSON unavailable", disabled=True, use_container_width=True)
    if ranking_csv is not None:
        r2.download_button("Download final rankings CSV", ranking_csv, "final_rankings.csv", "text/csv", key="download-final-rankings-csv", use_container_width=True)
    else:
        r2.button("Final rankings CSV unavailable", disabled=True, use_container_width=True)
    if ranking_report is not None:
        r3.download_button("Download ranking report Markdown", ranking_report, "final_ranking_report.md", "text/markdown", key="download-final-ranking-report", use_container_width=True)
    else:
        r3.button("Ranking report unavailable", disabled=True, use_container_width=True)
    st.caption("Exports use only local files. Missing artifacts remain empty/unavailable; no data is generated for export.")



