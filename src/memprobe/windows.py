from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .schema import CanonicalEpisode, EventProposal, ScalarSignal, TimeSpan, stable_id


@dataclass(frozen=True)
class WindowProposal:
    window_id: str
    episode_uid: str
    span: TimeSpan
    sources: tuple[str, ...]
    confidence: float


def propose_signal_windows(
    episode: CanonicalEpisode,
    padding_s: float = 1.0,
    gripper_threshold: float = 0.5,
    motion_threshold: float = 0.2,
) -> list[WindowProposal]:
    if padding_s < 0:
        raise ValueError("padding_s must be non-negative")

    proposals: list[WindowProposal] = []
    for signal in episode.signals:
        name = signal.name.lower()
        if "gripper" in name:
            centers = _threshold_transitions(signal, gripper_threshold)
            for center in centers:
                proposals.append(
                    _window_around(episode, center, padding_s, f"signal:{signal.name}", confidence=0.8)
                )
        elif name in {"motion_score", "ee_speed", "end_effector_speed", "optical_flow"}:
            for span in _active_spans(signal, motion_threshold):
                padded = TimeSpan(
                    max(0.0, span.start_s - padding_s),
                    min(episode.duration_s, span.end_s + padding_s),
                )
                proposals.append(
                    WindowProposal(
                        window_id=stable_id("win", episode.episode_uid, signal.name, padded.start_s, padded.end_s),
                        episode_uid=episode.episode_uid,
                        span=padded,
                        sources=(f"signal:{signal.name}",),
                        confidence=0.7,
                    )
                )
    return proposals


def windows_from_events(events: Iterable[EventProposal], padding_s: float = 0.0) -> list[WindowProposal]:
    proposals = []
    for event in events:
        span = TimeSpan(max(0.0, event.span.start_s - padding_s), event.span.end_s + padding_s)
        proposals.append(
            WindowProposal(
                window_id=stable_id("win", event.episode_uid, event.event_id, span.start_s, span.end_s),
                episode_uid=event.episode_uid,
                span=span,
                sources=(f"event:{event.source}",),
                confidence=event.confidence,
            )
        )
    return proposals


def merge_windows(
    proposals: Iterable[WindowProposal],
    max_gap_s: float = 0.25,
) -> list[WindowProposal]:
    if max_gap_s < 0:
        raise ValueError("max_gap_s must be non-negative")
    ordered = sorted(proposals, key=lambda item: (item.episode_uid, item.span.start_s, item.span.end_s))
    if not ordered:
        return []

    merged: list[WindowProposal] = []
    current = ordered[0]
    for candidate in ordered[1:]:
        same_episode = candidate.episode_uid == current.episode_uid
        overlaps_or_close = candidate.span.start_s <= current.span.end_s + max_gap_s
        if same_episode and overlaps_or_close:
            span = TimeSpan(current.span.start_s, max(current.span.end_s, candidate.span.end_s))
            sources = tuple(sorted(set(current.sources + candidate.sources)))
            current = WindowProposal(
                window_id=stable_id("win", current.episode_uid, span.start_s, span.end_s, *sources),
                episode_uid=current.episode_uid,
                span=span,
                sources=sources,
                confidence=max(current.confidence, candidate.confidence),
            )
        else:
            merged.append(current)
            current = candidate
    merged.append(current)
    return merged


def _window_around(
    episode: CanonicalEpisode,
    center_s: float,
    padding_s: float,
    source: str,
    confidence: float,
) -> WindowProposal:
    span = TimeSpan(max(0.0, center_s - padding_s), min(episode.duration_s, center_s + padding_s))
    return WindowProposal(
        window_id=stable_id("win", episode.episode_uid, source, span.start_s, span.end_s),
        episode_uid=episode.episode_uid,
        span=span,
        sources=(source,),
        confidence=confidence,
    )


def _threshold_transitions(signal: ScalarSignal, threshold: float) -> list[float]:
    if len(signal.values) < 2:
        return []
    states = [value >= threshold for value in signal.values]
    return [
        signal.timestamps_s[index]
        for index in range(1, len(states))
        if states[index] != states[index - 1]
    ]


def _active_spans(signal: ScalarSignal, threshold: float) -> list[TimeSpan]:
    spans: list[TimeSpan] = []
    start_s = None
    for index, value in enumerate(signal.values):
        active = value >= threshold
        if active and start_s is None:
            start_s = signal.timestamps_s[index]
        if not active and start_s is not None:
            spans.append(TimeSpan(start_s, signal.timestamps_s[index]))
            start_s = None
    if start_s is not None and signal.timestamps_s:
        spans.append(TimeSpan(start_s, signal.timestamps_s[-1]))
    return spans
