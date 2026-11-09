"""Semantic event discovery validation and helpers."""
from collections.abc import Iterable

from promoscout.analysis.models import DiscoveredEvent
from promoscout.schemas import ParsedEpisode

PROMPT_VERSION = "event-tagging-v1"


class InvalidEventReference(ValueError):
    """The model referenced an event that was not in the supplied episode."""


def validate_discoveries(
    discoveries: Iterable[DiscoveredEvent | dict], valid_event_ids: set[str]
) -> list[DiscoveredEvent]:
    """Validate IDs strictly and collapse repeated references by event and role."""
    result = []
    seen = set()
    for item in discoveries:
        discovery = item if isinstance(item, DiscoveredEvent) else DiscoveredEvent.model_validate(item)
        if discovery.event_id not in valid_event_ids:
            raise InvalidEventReference(f"LLM returned unknown event ID: {discovery.event_id}")
        key = (discovery.event_id, discovery.role)
        if key not in seen:
            result.append(discovery)
            seen.add(key)
    return result


def discover_events(episode: ParsedEpisode, *, llm=None) -> list[DiscoveredEvent]:
    """Tag one episode; ordinarily called through combined episode analysis."""
    from promoscout.analysis.episode_analyzer import analyze_episode
    result = analyze_episode(episode, llm=llm)
    return result.potential_events
