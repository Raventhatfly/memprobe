from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

from .schema import CanonicalEpisode, EventProposal, ObjectTrack, ensure_unique_ids


@dataclass(frozen=True)
class EventStateGraph:
    episode: CanonicalEpisode
    events: tuple[EventProposal, ...]
    tracks: tuple[ObjectTrack, ...] = ()

    @classmethod
    def build(
        cls,
        episode: CanonicalEpisode,
        events: Iterable[EventProposal],
        tracks: Iterable[ObjectTrack] = (),
        min_confidence: float = 0.0,
        require_verified: bool = False,
    ) -> "EventStateGraph":
        selected = []
        for event in events:
            if event.episode_uid != episode.episode_uid:
                raise ValueError(f"event {event.event_id} belongs to another episode")
            if event.camera_id not in {stream.camera_id for stream in episode.streams}:
                raise ValueError(f"event {event.event_id} references unknown camera {event.camera_id}")
            if event.span.end_s > episode.duration_s + 1e-6:
                raise ValueError(f"event {event.event_id} exceeds episode duration")
            if event.confidence < min_confidence:
                continue
            if require_verified and not event.is_verified:
                continue
            selected.append(event)

        selected.sort(key=lambda item: (item.span.end_s, item.span.start_s, item.event_id))
        ensure_unique_ids(selected)
        return cls(episode=episode, events=tuple(selected), tracks=tuple(tracks))

    def completed_before(self, query_time_s: float) -> tuple[EventProposal, ...]:
        return tuple(event for event in self.events if event.span.end_s < query_time_s)

    def previous_event(self, query_time_s: float) -> Optional[EventProposal]:
        completed = self.completed_before(query_time_s)
        return completed[-1] if completed else None

    def event(self, event_id: str) -> EventProposal:
        for event in self.events:
            if event.event_id == event_id:
                return event
        raise KeyError(event_id)

    @property
    def capabilities(self) -> frozenset[str]:
        values = {"single_camera" if len(self.episode.streams) == 1 else "multi_camera"}
        if self.events:
            values.add("verified_events" if all(event.is_verified for event in self.events) else "proposed_events")
        if self.tracks:
            values.add("object_tracks")
        return frozenset(values)
