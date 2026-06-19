from __future__ import annotations

from dataclasses import replace
from typing import Any

from .graph import EventStateGraph
from .schema import ProbeItem


def validate_probe(item: ProbeItem, graph: EventStateGraph) -> ProbeItem:
    choice_ids = [choice.choice_id for choice in item.choices]
    media_keys = [
        (choice.media.uri, choice.media.camera_id, choice.media.span.start_s, choice.media.span.end_s)
        for choice in item.choices
    ]

    checks: dict[str, Any] = {
        "choice_ids_unique": len(choice_ids) == len(set(choice_ids)),
        "choice_media_unique": len(media_keys) == len(set(media_keys)),
        "history_ends_at_query": abs(item.history.span.end_s - item.query_time_s) < 1e-6,
        "query_inside_episode": 0.0 < item.query_time_s <= graph.episode.duration_s,
        "evidence_before_or_at_query": all(media.span.end_s <= item.query_time_s for media in item.evidence),
        "nonzero_memory_horizon": any(media.span.end_s < item.query_time_s for media in item.evidence),
        "source_events_verified": _source_events_verified(item, graph),
        "current_frame_insufficient": "pending_human_or_baseline",
        "oracle_evidence_answerable": "pending_model_or_human",
        "text_memory_adversary_fails": "pending_baseline",
        "leakage_scan_passes": "pending_release_audit",
    }
    checks["answer_references_choices"] = _answer_references_choices(item.answer, set(choice_ids))
    mandatory = (
        "choice_ids_unique",
        "choice_media_unique",
        "history_ends_at_query",
        "query_inside_episode",
        "evidence_before_or_at_query",
        "nonzero_memory_horizon",
        "answer_references_choices",
    )
    checks["automatic_pass"] = all(checks[name] is True for name in mandatory)
    checks["ready_for_human_review"] = checks["automatic_pass"] and checks["source_events_verified"] is True
    checks["release_ready"] = False
    checks["status"] = (
        "candidate_requires_review" if checks["ready_for_human_review"] else "proposal_requires_event_verification"
    )
    return replace(item, validation=checks)


def _answer_references_choices(answer: Any, choice_ids: set[str]) -> bool:
    if isinstance(answer, str):
        return answer in choice_ids
    if isinstance(answer, list):
        return len(answer) == len(choice_ids) and set(answer) == choice_ids
    return False


def _source_events_verified(item: ProbeItem, graph: EventStateGraph) -> bool:
    event_ids = [choice.source_event_id for choice in item.choices if choice.source_event_id]
    if not event_ids:
        return False
    return all(graph.event(event_id).is_verified for event_id in event_ids)
