from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import Iterable, Protocol

from .graph import EventStateGraph
from .schema import MediaRef, ProbeChoice, ProbeItem, TimeSpan, stable_id


class ProbeGenerator(Protocol):
    probe_type: str

    def generate(self, graph: EventStateGraph, seed: int) -> Iterable[ProbeItem]:
        ...


@dataclass(frozen=True)
class PreviousEventGenerator:
    num_choices: int = 4
    probe_type: str = "PREVIOUS_EVENT"

    def generate(self, graph: EventStateGraph, seed: int) -> Iterable[ProbeItem]:
        stream = graph.episode.stream()
        for current in graph.events:
            query_time_s = current.span.start_s
            completed = list(graph.completed_before(query_time_s))
            if len(completed) < self.num_choices:
                continue
            answer_event = completed[-1]
            distractors = _select_temporal_distractors(completed[:-1], answer_event.event_type, self.num_choices - 1)
            selected = [answer_event, *distractors]
            probe_id = stable_id("probe", graph.episode.episode_uid, self.probe_type, current.event_id)
            rng = random.Random(_derived_seed(seed, probe_id))
            rng.shuffle(selected)
            choices = tuple(
                ProbeChoice(
                    choice_id=_choice_id(index),
                    media=_event_media(stream.uri, event.camera_id, event.span),
                    source_event_id=event.event_id,
                )
                for index, event in enumerate(selected)
            )
            answer = next(choice.choice_id for choice in choices if choice.source_event_id == answer_event.event_id)
            yield ProbeItem(
                probe_id=probe_id,
                episode_uid=graph.episode.episode_uid,
                probe_type=self.probe_type,
                response_type="choice_id",
                history=MediaRef(
                    uri=stream.uri,
                    camera_id=stream.camera_id,
                    span=TimeSpan(0.0, query_time_s),
                    kind="history_video",
                ),
                query_time_s=query_time_s,
                choices=choices,
                answer=answer,
                operator={
                    "name": "argmax_event_end_before_query",
                    "query_event_id": current.event_id,
                    "constraint": "event.end_s < query_time_s",
                },
                evidence=(_event_media(stream.uri, answer_event.camera_id, answer_event.span),),
                validation={},
                provenance={
                    "answer_event_id": answer_event.event_id,
                    "query_event_id": current.event_id,
                    "generator": self.__class__.__name__,
                },
            )


@dataclass(frozen=True)
class EventOrderGenerator:
    num_choices: int = 4
    stride: int = 2
    probe_type: str = "EVENT_ORDER"

    def generate(self, graph: EventStateGraph, seed: int) -> Iterable[ProbeItem]:
        if len(graph.events) < self.num_choices:
            return
        stream = graph.episode.stream()
        ordered_events = sorted(graph.events, key=lambda item: (item.span.start_s, item.span.end_s))
        last_start = len(ordered_events) - self.num_choices
        for start in range(0, last_start + 1, self.stride):
            chronological = ordered_events[start : start + self.num_choices]
            query_time_s = graph.episode.duration_s
            probe_id = stable_id(
                "probe",
                graph.episode.episode_uid,
                self.probe_type,
                *(event.event_id for event in chronological),
            )
            shuffled = list(chronological)
            rng = random.Random(_derived_seed(seed, probe_id))
            rng.shuffle(shuffled)
            choices = tuple(
                ProbeChoice(
                    choice_id=_choice_id(index),
                    media=_event_media(stream.uri, event.camera_id, event.span),
                    source_event_id=event.event_id,
                )
                for index, event in enumerate(shuffled)
            )
            choice_by_event = {choice.source_event_id: choice.choice_id for choice in choices}
            answer = [choice_by_event[event.event_id] for event in chronological]
            yield ProbeItem(
                probe_id=probe_id,
                episode_uid=graph.episode.episode_uid,
                probe_type=self.probe_type,
                response_type="ordered_choice_ids",
                history=MediaRef(
                    uri=stream.uri,
                    camera_id=stream.camera_id,
                    span=TimeSpan(0.0, query_time_s),
                    kind="history_video",
                ),
                query_time_s=query_time_s,
                choices=choices,
                answer=answer,
                operator={
                    "name": "sort_events_by_start_time",
                    "tie_breaker": "end_time_then_event_id",
                },
                evidence=tuple(_event_media(stream.uri, event.camera_id, event.span) for event in chronological),
                validation={},
                provenance={
                    "ordered_event_ids": [event.event_id for event in chronological],
                    "generator": self.__class__.__name__,
                },
            )


def build_generators(probe_types: Iterable[str], num_choices: int = 4) -> tuple[ProbeGenerator, ...]:
    registry = {
        "PREVIOUS_EVENT": PreviousEventGenerator(num_choices=num_choices),
        "EVENT_ORDER": EventOrderGenerator(num_choices=num_choices),
    }
    generators = []
    for probe_type in probe_types:
        normalized = probe_type.strip().upper()
        if not normalized:
            continue
        try:
            generators.append(registry[normalized])
        except KeyError as exc:
            raise ValueError(f"unsupported probe type {probe_type!r}; available: {', '.join(registry)}") from exc
    return tuple(generators)


def _select_temporal_distractors(events: list, answer_type: str, count: int) -> list:
    same_type = [event for event in reversed(events) if event.event_type == answer_type]
    other_type = [event for event in reversed(events) if event.event_type != answer_type]
    return (same_type + other_type)[:count]


def _event_media(uri: str, camera_id: str, span: TimeSpan) -> MediaRef:
    return MediaRef(uri=uri, camera_id=camera_id, span=span, kind="event_clip")


def _choice_id(index: int) -> str:
    if not 0 <= index < 26:
        raise ValueError("visual choice count cannot exceed 26")
    return chr(ord("A") + index)


def _derived_seed(seed: int, item_id: str) -> int:
    digest = hashlib.sha256(f"{seed}:{item_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")
