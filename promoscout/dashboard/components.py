"""Reusable visual components for the PromoScout dashboard."""
from __future__ import annotations

from html import escape
from typing import Any


def render_brand(st, *, sidebar: bool = False) -> None:
    markup = '<div class="ps-brand"><span class="ps-mark">PS</span><span class="ps-wordmark">PromoScout<small>Story intelligence</small></span></div>'
    st.markdown(markup, unsafe_allow_html=True)


def render_header(st, title: str, subtitle: str, eyebrow: str = "KUKU · PROMOSCOUT") -> None:
    st.markdown(
        f'<div class="ps-eyebrow">{escape(eyebrow)}</div>'
        f'<div class="ps-page-title">{escape(title)}</div>'
        f'<div class="ps-subtitle">{escape(subtitle)}</div>',
        unsafe_allow_html=True,
    )


def render_metrics(st, metrics: list[tuple[str, Any, str]]) -> None:
    cards = []
    for label, value, note in metrics:
        cards.append(
            '<div class="ps-metric">'
            f'<div class="ps-metric-label">{escape(str(label))}</div>'
            f'<div class="ps-metric-value">{escape(str(value))}</div>'
            f'<div class="ps-metric-note">{escape(str(note))}</div>'
            '</div>'
        )
    st.markdown('<div class="ps-metric-grid">' + "".join(cards) + "</div>", unsafe_allow_html=True)


def render_status(st, label: str, state: str, detail: str = "") -> None:
    state_class = "done" if state.lower() in {"complete", "ready", "approved"} else "attention" if state.lower() in {"needs attention", "partial", "in review"} else "active" if state.lower() in {"running", "in progress"} else ""
    st.markdown(
        f'<span class="ps-status {state_class}"><span class="ps-dot"></span>'
        f'<b>{escape(label)}</b> · {escape(state)}'
        f'{" · " + escape(detail) if detail else ""}</span>',
        unsafe_allow_html=True,
    )


def render_empty_state(st, title: str, message: str, action: str | None = None) -> None:
    extra = f"<p>{escape(action)}</p>" if action else ""
    st.markdown(
        f'<div class="ps-empty"><h3>{escape(title)}</h3><p>{escape(message)}</p>{extra}</div>',
        unsafe_allow_html=True,
    )


def render_event_item(st, event: dict, *, relative_start: float | None = None) -> None:
    event_type = escape(str(event.get("event_type", "other")).replace("_", " ").title())
    event_id = escape(str(event.get("event_id", "unknown")))
    start = escape(str(event.get("start_timestamp", "--:--")))
    end = escape(str(event.get("end_timestamp", "--:--")))
    relative = f'<span class="ps-tag">+{relative_start:.2f}s</span>' if relative_start is not None else ""
    speaker = event.get("speaker")
    delivery = event.get("delivery")
    who = " · ".join(str(item) for item in (speaker, delivery) if item)
    details = f'<div class="ps-meta">{escape(who)} · {event_id}</div>' if who else f'<div class="ps-meta">{event_id}</div>'
    st.markdown(
        '<div class="ps-event">'
        f'<div><div class="ps-event-time">{start} – {end}</div><span class="ps-tag">{event_type}</span>{relative}</div>'
        f'<div class="ps-event-content">{escape(str(event.get("content", "")))}{details}</div>'
        '</div>',
        unsafe_allow_html=True,
    )


def section_divider(st) -> None:
    st.markdown('<div class="ps-divider"></div>', unsafe_allow_html=True)

