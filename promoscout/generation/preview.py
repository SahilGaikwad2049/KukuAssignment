"""Reconstruct complete candidate content for downstream evaluation."""
from promoscout.generation.continuous import PromoCandidate
from promoscout.schemas import ParsedEpisode


def reconstruct_candidate_preview(episode: ParsedEpisode, candidate: PromoCandidate | dict) -> dict:
    candidate = candidate if isinstance(candidate, PromoCandidate) else PromoCandidate.model_validate(candidate)
    if candidate.episode_number != episode.episode_number:
        raise ValueError("Candidate and episode numbers do not match")
    by_id = {event.event_id: event for event in episode.events}
    missing = [event_id for event_id in candidate.included_event_ids if event_id not in by_id]
    if missing:
        raise ValueError(f"Candidate references unknown event IDs: {', '.join(missing)}")
    selected = [by_id[event_id] for event_id in candidate.included_event_ids]
    selected.sort(key=lambda e: (e.start_seconds, e.source_line_start, e.end_seconds, e.event_id))
    rows = []
    for event in selected:
        if event.start_seconds > candidate.end_seconds or event.end_seconds < candidate.start_seconds:
            continue
        rows.append({
            "event_id": event.event_id,
            "event_type": event.event_type,
            "label": event.label,
            "speaker": event.speaker,
            "delivery": event.delivery,
            "content": event.content,
            "start_timestamp": event.start_timestamp,
            "end_timestamp": event.end_timestamp,
            "start_seconds": event.start_seconds,
            "end_seconds": event.end_seconds,
            "relative_start_seconds": round(event.start_seconds - candidate.start_seconds, 2),
            "relative_end_seconds": round(event.end_seconds - candidate.start_seconds, 2),
        })
    return {
        "candidate_id": candidate.candidate_id,
        "episode_number": candidate.episode_number,
        "start_timestamp": candidate.start_timestamp,
        "end_timestamp": candidate.end_timestamp,
        "duration_seconds": candidate.duration_seconds,
        "events": rows,
    }

