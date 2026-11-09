"""Composite promo review page, isolated from single-episode screens."""
from __future__ import annotations

from html import escape

import streamlit as st

from promoscout.dashboard.components import render_empty_state, render_header
from promoscout.dashboard.data_loader import load_dashboard_data
from promoscout.dashboard.pages import _render_blind_viewer_result, _review_state
from promoscout.dashboard.review_manager import ReviewStoreError, review_key, save_review_decision
from promoscout.generation.composite import reconstruct_composite_preview


def composite_promos_page() -> None:
    data = load_dashboard_data()
    for error in data["episode_errors"] + data["candidate_errors"]:
        st.warning(error)
    for key, file in data["files"].items():
        if file["error"]:
            st.error(f"Could not read {key} file: {file['error']}")
    render_header(st, "Composite Promos", "Review source-grounded promos assembled from story moments across episodes.", "PROMOSCOUT / COMPOSITES")
    candidates = data.get("composite_candidates", [])
    if not candidates:
        render_empty_state(st, "No composite promos generated yet", "Run series-level analysis, then build composites from its source event anchors. The synthetic three-episode fixture is test-only and is not shown as a real candidate.")
        analysis = data.get("series_analysis")
        if analysis:
            st.caption(f"Series analysis is available for episodes: {', '.join(map(str, analysis.get('analyzed_episode_numbers', [])))}")
        else:
            st.caption("No saved series-level analysis is available yet.")
        return
    review, review_error = _review_state()
    if review_error:
        st.warning(review_error)
    validation = (data.get("validation") or {}).get("composite_validation", {})
    validation_by_id = {row.get("candidate_id"): row for row in validation.get("candidates", [])} if isinstance(validation, dict) else {}
    evaluations = data.get("blind_viewer_evaluations", {})
    for candidate in candidates:
        source = candidate.get("candidate_origin", "semantic")
        candidate_id = candidate.get("candidate_id", "unknown")
        key = review_key("composite", candidate_id)
        decision = review.get("decisions", {}).get(key, {})
        status = validation_by_id.get(candidate_id, {}).get("structural_status", "not validated")
        with st.container(border=True):
            st.markdown(f"### {escape(candidate_id)}")
            labels = ["SYNTHETIC FIXTURE"] if source == "synthetic" or candidate.get("synthetic") else []
            labels.append(f"{source.title()} origin")
            st.caption(" · ".join(labels) + f" · {candidate.get('total_duration_seconds', '—')}s · arc {candidate.get('story_arc_id', 'unknown')} · structural status: {status}")
            st.write(candidate.get("narrative_theme", ""))
            st.write(candidate.get("narrative_rationale", ""))
            episode_numbers = sorted({segment.get("episode_number") for segment in candidate.get("segments", [])})
            st.caption("Episodes represented: " + ", ".join(map(str, episode_numbers)))
            try:
                preview = reconstruct_composite_preview(candidate)
            except Exception as exc:
                preview = None
                st.warning(f"Could not reconstruct candidate preview: {exc}")
            if preview:
                for segment in preview["segments"]:
                    with st.expander(f"Segment {segment['segment_index'] + 1} · Episode {segment['episode_number']} · {segment['beat_role']} · {segment['duration_seconds']}s", expanded=False):
                        st.caption(f"Source {segment['source_start_timestamp']} → {segment['source_end_timestamp']} · Promo {segment['relative_start_seconds']:.2f}s → {segment['relative_end_seconds']:.2f}s · anchor IDs: {', '.join(segment['anchor_event_ids'])}")
                        for event in segment["events"]:
                            st.markdown(f"**{event['relative_start_seconds']:.2f}s** · `{event['event_id']}` · {event['event_type']} · {event.get('speaker') or '—'} · source {event['original_start_timestamp']}–{event['original_end_timestamp']}")
                            st.write(event["content"])
            hook_index = candidate.get("hook_segment_index")
            ending_index = candidate.get("potential_ending_segment_index")
            segments = candidate.get("segments", [])
            st.markdown(f"**Hook:** {candidate.get('hook_event_id', '—')}" + (f" · {segments[hook_index].get('beat_role')} segment" if isinstance(hook_index, int) and 0 <= hook_index < len(segments) else ""))
            st.markdown(f"**Potential ending (not a confirmed cliffhanger):** {candidate.get('potential_ending_event_id', '—')}" + (f" · {segments[ending_index].get('beat_role')} segment" if isinstance(ending_index, int) and 0 <= ending_index < len(segments) else ""))
            for warning in candidate.get("spoiler_warnings", []):
                st.warning(f"Spoiler warning: {warning}")
            for warning in candidate.get("coherence_warnings", []):
                st.warning(f"Human coherence check: {warning}")
            evaluation = evaluations.get(f"composite:{candidate_id}") if isinstance(evaluations, dict) else None
            _render_blind_viewer_result(st, evaluation)
            st.caption(f"Human decision: {decision.get('decision', 'pending').title()}")
            note = st.text_input("Review note (optional)", key=f"composite-note-{candidate_id}")
            columns = st.columns(3)
            for column, value, label in zip(columns, ("approved", "rejected", "later"), ("Approve", "Reject", "Review later")):
                if column.button(label, key=f"composite-{value}-{candidate_id}", use_container_width=True):
                    try:
                        save_review_decision("composite", candidate_id, value, note)
                        st.rerun()
                    except ReviewStoreError as exc:
                        st.error(str(exc))
