
"""Deterministic first-pass generation of continuous promo candidates.

This module discovers plausible hook-to-ending windows using transparent
heuristics. It does NOT decide whether a promo is genuinely good.

All clip boundaries and hook timestamps are copied from original screenplay
events. Candidates are provisional and require LLM/editor evaluation.
"""

from __future__ import annotations

import argparse
import json
import re

from collections import defaultdict
from pathlib import Path
from typing import Iterable

from pydantic import BaseModel, ConfigDict, Field

from promoscout.schemas import ParsedEpisode, ScreenplayEvent


# ---------------------------------------------------------------------
# 1. PROMO CANDIDATE SCHEMA
# ---------------------------------------------------------------------

class PromoCandidate(BaseModel):
    """A source-traceable, continuous promo window from one episode."""

    model_config = ConfigDict(extra="forbid")

    candidate_id: str
    candidate_family_id: str

    episode_number: int

    # Clip boundaries
    start_event_id: str
    end_event_id: str

    start_timestamp: str
    end_timestamp: str

    start_seconds: float
    end_seconds: float
    duration_seconds: float

    # Hook information
    hook_event_id: str
    hook_type: str

    hook_text: str
    hook_start_timestamp: str
    hook_end_timestamp: str

    hook_start_seconds: float
    hook_end_seconds: float
    hook_offset_seconds: float

    # Ending information
    ending_event_id: str
    ending_text: str

    # Heuristic scores
    hook_score: float
    ending_score: float
    heuristic_score: float

    hook_signals: list[str] = Field(default_factory=list)
    ending_signals: list[str] = Field(default_factory=list)

    # Original events included in this window
    included_event_ids: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------
# 2. HEURISTIC PATTERNS
# ---------------------------------------------------------------------

_HOOK_PHRASES = (
    "turn around",
    "who are you",
    "what did you",
    "what have you",
    "you lied",
    "you betrayed",
    "i know what",
    "i know the truth",
    "you have no idea",
    "stay away",
    "don't move",
    "do not move",
    "it's over",
    "we need to talk",
    "how could you",
    "you are not",
)

_HOOK_WORDS = re.compile(
    r"\b(secret|betray|betrayed|lie|lied|truth|murder|kill|dead|pregnant|"
    r"marry|marrying|divorce|affair|stole|stolen|never|revenge|arrested|"
    r"father|mother|sister|brother|husband|wife)\b",
    re.IGNORECASE,
)

_HOOK_DELIVERY = re.compile(
    r"\b(shocked|angry|furious|desperate|horrified|threatening|commanding|"
    r"gasping|outraged|accusing|panicked|terrified|urgent|gloating)\b",
    re.IGNORECASE,
)

_ACTION_HOOK = re.compile(
    r"\b(slaps?|punches?|hits?|kicks?|strikes?|grabs?|collapses?|falls?|"
    r"reveals?|discovers?|bursts? in|storms? in|throws?|rips?|tears?)\b",
    re.IGNORECASE,
)

_ENDING_PHRASES = (
    "you don't know",
    "you do not know",
    "there is something",
    "there's something",
    "before i tell",
    "before you",
    "who are you",
    "what if",
    "what do you mean",
    "how could you",
    "wait",
    "but...",
)

_UNRESOLVED_ACTION = re.compile(
    r"\b(about to|reaches? for|opens? the|pulls? out|turns? to|uncovers?|"
    r"reveals?|discovers?|stares? at|gasps?|freezes?|steps? forward)\b",
    re.IGNORECASE,
)

_EMOTIONAL_DELIVERY = re.compile(
    r"\b(shocked|horrified|trembling|shaking|whispering|crying|resolute|"
    r"determined|terrified|desperate|stunned|heartbroken)\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------
# 3. GENERAL HELPERS
# ---------------------------------------------------------------------

def _event_text(event: ScreenplayEvent) -> str:
    return (event.content or "").strip()


def classify_hook_type(event_type: str) -> str:
    """Map screenplay event types to promo hook categories.

    Camera events are classified as visual hooks so that the schema
    already supports visual-hook discovery in a later improvement.
    """

    mapping = {
        "dialogue": "dialogue",
        "action": "action",
        "camera": "visual",
    }

    return mapping.get(event_type.lower(), "other")


def _format_candidate_id(episode_number: int, index: int) -> str:
    return f"EP{episode_number:03d}_PROMO_{index:03d}"


def _format_family_id(episode_number: int, index: int) -> str:
    return f"EP{episode_number:03d}_FAMILY_{index:03d}"


def _candidate_quality_key(item: dict) -> tuple:
    """Deterministic tie-breaking when two candidates have identical windows."""

    return (
        item["heuristic_score"],
        item["hook_score"],
        item["ending_score"],
        -item["hook_offset"],
        item["hook"].event_id,
    )


# ---------------------------------------------------------------------
# 4. HOOK SCORING
# ---------------------------------------------------------------------

def score_hook(event: ScreenplayEvent) -> tuple[float, list[str]]:
    """Return a transparent heuristic hook score and matching signals."""

    text = _event_text(event)
    lower = text.lower()

    score = 0.0
    signals: list[str] = []

    if event.event_type == "dialogue":

        score += 0.25

        if "?" in text or "!" in text:
            score += 0.65
            signals.append("question_or_exclamation")

        if any(phrase in lower for phrase in _HOOK_PHRASES):
            score += 1.25
            signals.append("high_tension_phrase")

        if _HOOK_WORDS.search(text):
            score += 0.8
            signals.append("high_impact_word")

        delivery = event.delivery or ""

        if _HOOK_DELIVERY.search(delivery):
            score += 0.65
            signals.append("high_intensity_delivery")

        if len(text.split()) <= 5 and ("?" in text or "!" in text):
            score += 0.25
            signals.append("short_emphatic_line")

    elif event.event_type == "action":

        if _ACTION_HOOK.search(text):
            score += 1.6
            signals.append("physical_or_revelatory_action")

        if _EMOTIONAL_DELIVERY.search(event.delivery or ""):
            score += 0.4
            signals.append("emotional_delivery")

    return round(score, 2), signals


# ---------------------------------------------------------------------
# 5. ENDING SCORING
# ---------------------------------------------------------------------

def score_ending(
    event: ScreenplayEvent,
    scene_end_by_event: dict[str, float],
    episode_end: float,
) -> tuple[float, list[str]]:
    """Estimate whether an event could provide an intriguing promo endpoint.

    These are ending-candidate signals, NOT proof of a cliffhanger.
    The LLM will make the actual narrative judgment later.
    """

    text = _event_text(event)
    lower = text.lower()

    score = 0.0
    signals: list[str] = []

    if event.event_type == "dialogue":

        if "?" in text:
            score += 0.85
            signals.append("unanswered_question_candidate")

        if "..." in text or "…" in text:
            score += 0.9
            signals.append("interrupted_or_trailing_line")

        if any(phrase in lower for phrase in _ENDING_PHRASES):
            score += 0.85
            signals.append("suspense_phrase")

        if _EMOTIONAL_DELIVERY.search(event.delivery or ""):
            score += 0.35
            signals.append("emotional_delivery")

    elif event.event_type == "action":

        if _UNRESOLVED_ACTION.search(text):
            score += 0.75
            signals.append("unresolved_action_candidate")

        if _EMOTIONAL_DELIVERY.search(event.delivery or ""):
            score += 0.35
            signals.append("emotional_delivery")

    elif event.event_type == "camera":

        if re.search(r"\bcut to black\b", text, re.IGNORECASE):
            score += 1.8
            signals.append("cut_to_black")

        elif re.search(r"\b(cut|fade) to\b", text, re.IGNORECASE):
            score += 0.35
            signals.append("visual_transition")

    elif event.event_type == "sound_fx":

        if re.search(
            r"\b(slam|crash|gunshot|scream|ominous|drone|thud)\b",
            text,
            re.IGNORECASE,
        ):
            score += 0.35
            signals.append("dramatic_sound")

    scene_end = scene_end_by_event.get(event.event_id)

    if scene_end is not None and abs(scene_end - event.end_seconds) <= 3.0:
        score += 0.65
        signals.append("near_scene_boundary")

    if abs(episode_end - event.end_seconds) <= 3.0:
        score += 0.75
        signals.append("near_episode_end")

    return round(score, 2), signals


# ---------------------------------------------------------------------
# 6. CLIP START SELECTION
# ---------------------------------------------------------------------

def _choose_start_event(
    events: list[ScreenplayEvent],
    hook: ScreenplayEvent,
    max_lead_in: float,
) -> ScreenplayEvent:
    """Choose the earliest non-sound event starting before the hook.

    The hook must occur no more than max_lead_in seconds after the clip starts.
    """

    eligible = [
        event
        for event in events
        if event.event_type != "sound_fx"
        and 0 <= hook.start_seconds - event.start_seconds <= max_lead_in
    ]

    if not eligible:
        return hook

    return min(
        eligible,
        key=lambda event: (event.start_seconds, event.source_line_start),
    )


# ---------------------------------------------------------------------
# 7. CANDIDATE GENERATION
# ---------------------------------------------------------------------

def generate_episode_candidates(
    episode: ParsedEpisode,
    *,
    min_duration: float = 20.0,
    max_duration: float = 60.0,
    max_lead_in: float = 3.0,
    min_hook_score: float = 1.2,
    min_ending_score: float = 0.8,
    max_candidates: int = 25,
) -> list[PromoCandidate]:
    """Generate plausible hook-to-ending windows for one episode.

    Windows may cross scene boundaries.

    Exact duplicate clip windows are removed, regardless of which event
    was selected as the hook or endpoint.

    Different endpoints for the same hook are grouped into one
    candidate family.
    """

    if min_duration <= 0 or max_duration < min_duration:
        raise ValueError("Require 0 < min_duration <= max_duration")

    if not 0 <= max_lead_in <= 3.0:
        raise ValueError("max_lead_in must be between 0 and 3 seconds")

    if max_candidates < 1:
        raise ValueError("max_candidates must be at least 1")

    events = sorted(
        episode.events,
        key=lambda event: (
            event.start_seconds,
            event.source_line_start,
            event.end_seconds,
        ),
    )

    if not events:
        return []

    # Map each event to the end of its original scene.
    scene_end_by_event: dict[str, float] = {}
    scene_ends: list[float] = []

    for scene in episode.scenes:
        if scene.end_seconds is not None:
            scene_ends.append(scene.end_seconds)

            for event in scene.events:
                scene_end_by_event[event.event_id] = scene.end_seconds

    episode_end = max(
        scene_ends,
        default=max(event.end_seconds for event in events),
    )

    # Score potential hooks and endings.
    scored_hooks = []
    scored_endings = []

    for event in events:

        hook_score, hook_signals = score_hook(event)

        if hook_score >= min_hook_score:
            scored_hooks.append((event, hook_score, hook_signals))

        ending_score, ending_signals = score_ending(
            event,
            scene_end_by_event,
            episode_end,
        )

        if ending_score >= min_ending_score:
            scored_endings.append((event, ending_score, ending_signals))

    provisional: list[dict] = []

    for hook, hook_score, hook_signals in scored_hooks:

        start_event = _choose_start_event(events, hook, max_lead_in)

        clip_start = start_event.start_seconds
        hook_offset = round(hook.start_seconds - clip_start, 2)

        # Hard constraint: opening hook must occur within the first 3 seconds.
        if hook_offset < -1e-6 or hook_offset > max_lead_in + 1e-6:
            continue

        possible_ends = []

        for ending, ending_score, ending_signals in scored_endings:

            # The ending must begin after the hook has finished.
            if ending.start_seconds < hook.end_seconds - 1e-6:
                continue

            duration = ending.end_seconds - clip_start

            if duration < min_duration or duration > max_duration:
                continue

            possible_ends.append(
                (ending, ending_score, ending_signals, duration)
            )

        # Retain the strongest few endpoints for each hook.
        possible_ends.sort(
            key=lambda item: (
                item[1],
                -abs(item[3] - 35.0),
                -item[0].end_seconds,
            ),
            reverse=True,
        )

        for ending, ending_score, ending_signals, duration in possible_ends[:4]:

            included = [
                event
                for event in events
                if event.start_seconds <= ending.end_seconds
                and event.end_seconds >= clip_start
            ]

            heuristic_score = round(
                hook_score * 0.55 + ending_score * 0.45,
                2,
            )

            provisional.append({
                "episode_number": episode.episode_number,
                "start_event": start_event,
                "hook": hook,
                "ending": ending,
                "duration": round(duration, 2),
                "hook_offset": hook_offset,
                "hook_type": classify_hook_type(hook.event_type),
                "hook_score": hook_score,
                "ending_score": ending_score,
                "heuristic_score": heuristic_score,
                "hook_signals": hook_signals,
                "ending_signals": ending_signals,
                "included_event_ids": [
                    event.event_id for event in included
                ],
            })

    # -----------------------------------------------------------------
    # 8. DEDUPLICATE IDENTICAL CLIP WINDOWS
    # -----------------------------------------------------------------

    unique: dict[tuple, dict] = {}

    for item in provisional:

        # Deduplicate using the actual clip boundaries, not event IDs.
        key = (
            item["episode_number"],
            round(item["start_event"].start_seconds, 2),
            round(item["ending"].end_seconds, 2),
        )

        current = unique.get(key)

        if (
            current is None
            or _candidate_quality_key(item) > _candidate_quality_key(current)
        ):
            unique[key] = item

    # -----------------------------------------------------------------
    # 9. RANK AND LIMIT CANDIDATES
    # -----------------------------------------------------------------

    ordered = sorted(
        unique.values(),
        key=lambda item: (
            -item["heuristic_score"],
            -item["hook_score"],
            -item["ending_score"],
            item["start_event"].start_seconds,
            item["hook"].source_line_start,
        ),
    )[:max_candidates]

    # -----------------------------------------------------------------
    # 10. ASSIGN CANDIDATE FAMILIES
    # -----------------------------------------------------------------

    # All alternative endpoints belonging to the same hook share a family ID.
    hooks_by_id = {
        item["hook"].event_id: item["hook"]
        for item in ordered
    }

    chronological_hooks = sorted(
        hooks_by_id.values(),
        key=lambda event: (
            event.start_seconds,
            event.source_line_start,
            event.event_id,
        ),
    )

    family_ids = {
        event.event_id: _format_family_id(
            episode.episode_number,
            index,
        )
        for index, event in enumerate(chronological_hooks, start=1)
    }

    # -----------------------------------------------------------------
    # 11. BUILD FINAL PROMO CANDIDATES
    # -----------------------------------------------------------------

    candidates: list[PromoCandidate] = []

    for index, item in enumerate(ordered, start=1):

        start_event = item["start_event"]
        hook = item["hook"]
        ending = item["ending"]

        candidates.append(
            PromoCandidate(
                candidate_id=_format_candidate_id(
                    episode.episode_number,
                    index,
                ),

                candidate_family_id=family_ids[hook.event_id],
                episode_number=episode.episode_number,

                start_event_id=start_event.event_id,
                end_event_id=ending.event_id,

                start_timestamp=start_event.start_timestamp,
                end_timestamp=ending.end_timestamp,

                start_seconds=start_event.start_seconds,
                end_seconds=ending.end_seconds,
                duration_seconds=item["duration"],

                hook_event_id=hook.event_id,
                hook_type=item["hook_type"],

                hook_text=hook.content,
                hook_start_timestamp=hook.start_timestamp,
                hook_end_timestamp=hook.end_timestamp,

                hook_start_seconds=hook.start_seconds,
                hook_end_seconds=hook.end_seconds,
                hook_offset_seconds=item["hook_offset"],

                ending_event_id=ending.event_id,
                ending_text=ending.content,

                hook_score=item["hook_score"],
                ending_score=item["ending_score"],
                heuristic_score=item["heuristic_score"],

                hook_signals=item["hook_signals"],
                ending_signals=item["ending_signals"],

                included_event_ids=item["included_event_ids"],
            )
        )

    return candidates


# ---------------------------------------------------------------------
# 12. MULTI-EPISODE GENERATION
# ---------------------------------------------------------------------

def generate_candidates(
    episodes: Iterable[ParsedEpisode],
    **kwargs,
) -> list[PromoCandidate]:
    """Generate continuous candidates across multiple episodes."""

    all_candidates: list[PromoCandidate] = []

    for episode in episodes:
        all_candidates.extend(
            generate_episode_candidates(episode, **kwargs)
        )

    return all_candidates


def generate_episode_candidates_from_discoveries(
    episode: ParsedEpisode,
    discoveries,
    *,
    min_duration: float = 20.0,
    max_duration: float = 60.0,
    max_lead_in: float = 3.0,
    max_candidates: int = 25,
) -> list[PromoCandidate]:
    """Build deterministic windows from validated LLM event IDs.

    Heuristics are not consulted for eligibility; confidence is retained as a
    transparent provisional score. All timing and overlap decisions use source events.
    """
    from promoscout.analysis.event_tagger import validate_discoveries

    if min_duration <= 0 or max_duration < min_duration:
        raise ValueError("Require 0 < min_duration <= max_duration")
    if not 0 <= max_lead_in <= 3:
        raise ValueError("max_lead_in must be between 0 and 3 seconds")
    if max_candidates < 1:
        raise ValueError("max_candidates must be at least 1")
    events = sorted(episode.events, key=lambda e: (e.start_seconds, e.source_line_start, e.end_seconds))
    by_id = {event.event_id: event for event in events}
    if len(by_id) != len(events):
        raise ValueError("Original parsed episode contains duplicate event IDs")
    validated = validate_discoveries(discoveries, set(by_id))
    hooks = {d.event_id: d for d in validated if d.role == "hook"}
    endings = {d.event_id: d for d in validated if d.role == "potential_ending"}
    provisional = []
    for hook_id, hook_discovery in hooks.items():
        hook = by_id[hook_id]
        start_event = _choose_start_event(events, hook, max_lead_in)
        offset = round(hook.start_seconds - start_event.start_seconds, 2)
        if not -1e-6 <= offset <= max_lead_in + 1e-6:
            continue
        for ending_id, ending_discovery in endings.items():
            ending = by_id[ending_id]
            if ending.start_seconds < hook.end_seconds - 1e-6:
                continue
            duration = ending.end_seconds - start_event.start_seconds
            if not min_duration <= duration <= max_duration:
                continue
            included = [e.event_id for e in events if e.start_seconds <= ending.end_seconds and e.end_seconds >= start_event.start_seconds]
            hook_score, ending_score = hook_discovery.confidence, ending_discovery.confidence
            provisional.append({
                "episode_number": episode.episode_number, "start_event": start_event,
                "hook": hook, "ending": ending, "duration": round(duration, 2),
                "hook_offset": offset, "hook_type": classify_hook_type(hook.event_type),
                "hook_score": hook_score, "ending_score": ending_score,
                "heuristic_score": round((hook_score + ending_score) / 2, 2),
                "hook_signals": [hook_discovery.category, hook_discovery.reason],
                "ending_signals": [ending_discovery.category, ending_discovery.reason],
                "included_event_ids": included,
            })
    unique = {}
    for item in provisional:
        key = (episode.episode_number, round(item["start_event"].start_seconds, 2), round(item["ending"].end_seconds, 2))
        current = unique.get(key)
        if current is None or _candidate_quality_key(item) > _candidate_quality_key(current):
            unique[key] = item
    ordered = sorted(unique.values(), key=lambda i: (-i["heuristic_score"], -i["hook_score"], -i["ending_score"], i["start_event"].start_seconds, i["hook"].source_line_start))[:max_candidates]
    hook_events = {item["hook"].event_id: item["hook"] for item in ordered}
    family_ids = {
        event.event_id: _format_family_id(episode.episode_number, index)
        for index, event in enumerate(sorted(hook_events.values(), key=lambda e: (e.start_seconds, e.source_line_start, e.event_id)), 1)
    }
    result = []
    for index, item in enumerate(ordered, 1):
        start, hook, ending = item["start_event"], item["hook"], item["ending"]
        result.append(PromoCandidate(
            candidate_id=_format_candidate_id(episode.episode_number, index),
            candidate_family_id=family_ids[hook.event_id], episode_number=episode.episode_number,
            start_event_id=start.event_id, end_event_id=ending.event_id,
            start_timestamp=start.start_timestamp, end_timestamp=ending.end_timestamp,
            start_seconds=start.start_seconds, end_seconds=ending.end_seconds,
            duration_seconds=item["duration"], hook_event_id=hook.event_id,
            hook_type=item["hook_type"], hook_text=hook.content,
            hook_start_timestamp=hook.start_timestamp, hook_end_timestamp=hook.end_timestamp,
            hook_start_seconds=hook.start_seconds, hook_end_seconds=hook.end_seconds,
            hook_offset_seconds=item["hook_offset"], ending_event_id=ending.event_id,
            ending_text=ending.content, hook_score=item["hook_score"],
            ending_score=item["ending_score"], heuristic_score=item["heuristic_score"],
            hook_signals=item["hook_signals"], ending_signals=item["ending_signals"],
            included_event_ids=item["included_event_ids"],
        ))
    return result


def generate_candidates_from_discoveries(episodes, discoveries_by_episode, **kwargs):
    """Generate semantic candidates for episodes, keyed by episode number string/int."""
    result = []
    for episode in episodes:
        discoveries = discoveries_by_episode.get(str(episode.episode_number), discoveries_by_episode.get(episode.episode_number, []))
        result.extend(generate_episode_candidates_from_discoveries(episode, discoveries, **kwargs))
    return result


# ---------------------------------------------------------------------
# 13. LOAD PARSED EPISODES
# ---------------------------------------------------------------------

def load_parsed_episodes(path: str | Path) -> list[ParsedEpisode]:

    payload = json.loads(
        Path(path).read_text(encoding="utf-8")
    )

    if not isinstance(payload, list):
        raise ValueError(
            "Parsed episodes JSON must contain a top-level list"
        )

    return [
        ParsedEpisode.model_validate(item)
        for item in payload
    ]


# ---------------------------------------------------------------------
# 14. COMMAND LINE INTERFACE
# ---------------------------------------------------------------------

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Generate provisional continuous promo candidates "
            "from parsed screenplay JSON."
        )
    )

    parser.add_argument(
        "input",
        help="Path to parsed episodes JSON",
    )

    parser.add_argument(
        "--output",
        default="data/outputs/continuous_candidates.json",
    )

    parser.add_argument(
        "--min-duration",
        type=float,
        default=20.0,
    )

    parser.add_argument(
        "--max-duration",
        type=float,
        default=60.0,
    )

    parser.add_argument(
        "--max-per-episode",
        type=int,
        default=25,
    )

    args = parser.parse_args()

    episodes = load_parsed_episodes(args.input)

    candidates = generate_candidates(
        episodes,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
        max_candidates=args.max_per_episode,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "metadata": {
            "episode_count": len(episodes),
            "candidate_count": len(candidates),
            "min_duration_seconds": args.min_duration,
            "max_duration_seconds": args.max_duration,
            "method": (
                "deterministic_heuristic_discovery; "
                "exact_window_deduplication; candidates require review"
            ),
        },
        "candidates": [
            candidate.model_dump(mode="json")
            for candidate in candidates
        ],
    }

    output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    by_episode: dict[int, int] = defaultdict(int)

    for candidate in candidates:
        by_episode[candidate.episode_number] += 1

    print(
        f"Generated {len(candidates)} candidate(s) "
        f"from {len(episodes)} episode(s)."
    )

    print(f"Saved JSON to: {output}")

    print("Candidates per episode:")

    for episode_number in sorted(by_episode):
        print(
            f"  Episode {episode_number}: "
            f"{by_episode[episode_number]}"
        )

    empty_count = len(episodes) - len(by_episode)

    print(
        f"Episodes with no heuristic candidates: {empty_count}"
    )


if __name__ == "__main__":
    main()
