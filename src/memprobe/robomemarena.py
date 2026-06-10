from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .hdf5_utils import DemoMetadata, iter_hdf5_files, read_demo_metadata


DEFAULT_ROBOMEMARENA_ROOT = Path("/n/netscratch/hankyang_lab/Lab/felix/dataset/robomemarena")


@dataclass(frozen=True)
class Subtask:
    path: Path
    suite: str
    task_name: str
    action: str
    object_name: str
    target: str | None
    order_index: int | None
    seed: str
    task_id: str
    metadata: DemoMetadata

    @property
    def episode_key(self) -> tuple[str, str, str, str]:
        return (self.suite, self.task_name, self.seed, self.task_id)


def scan_subtasks(root: Path = DEFAULT_ROBOMEMARENA_ROOT, limit_files: int | None = None) -> list[Subtask]:
    out: list[Subtask] = []
    for path in iter_hdf5_files(root):
        parsed = parse_subtask_path(path, root)
        if not parsed:
            continue
        metadata = read_demo_metadata(path)
        out.append(Subtask(path=path, metadata=metadata, **parsed))
        if limit_files and len(out) >= limit_files:
            break
    return out


def scan_episode_subtasks(root: Path = DEFAULT_ROBOMEMARENA_ROOT, max_episodes: int | None = None) -> list[Subtask]:
    """Scan complete RoboMemArena seed/task episodes instead of a file-prefix sample."""
    files_by_episode: dict[tuple[str, str, str, str], list[Path]] = {}
    for path in iter_hdf5_files(root):
        parsed = parse_subtask_path(path, root)
        if not parsed:
            continue
        key = (str(parsed["suite"]), str(parsed["task_name"]), str(parsed["seed"]), str(parsed["task_id"]))
        files_by_episode.setdefault(key, []).append(path)

    out: list[Subtask] = []
    for _, paths in sorted(files_by_episode.items())[:max_episodes]:
        for path in sorted(paths):
            parsed = parse_subtask_path(path, root)
            if not parsed:
                continue
            metadata = read_demo_metadata(path)
            out.append(Subtask(path=path, metadata=metadata, **parsed))
    return out


def parse_subtask_path(path: Path, root: Path) -> dict[str, object] | None:
    if path.parent.name != "subtask_data":
        return None
    try:
        rel = path.relative_to(root)
    except ValueError:
        return None
    parts = rel.parts
    if len(parts) < 4:
        return None
    suite = parts[0]
    task_name = parts[1]
    stem = path.stem
    match = re.match(r"(?P<payload>.+)_seed(?P<seed>\d+)_task(?P<task_id>\d+)$", stem)
    if not match:
        return None

    payload = match.group("payload")
    tokens = payload.split("_")
    if len(tokens) < 3:
        return None
    action = tokens[0]
    maybe_order = tokens[-1]
    order_index = int(maybe_order) if maybe_order.isdigit() else None
    core = tokens[1:-1] if order_index is not None else tokens[1:]
    object_name, target = _split_object_target(action, core)
    return {
        "suite": suite,
        "task_name": task_name,
        "action": action,
        "object_name": object_name,
        "target": target,
        "order_index": order_index,
        "seed": match.group("seed"),
        "task_id": match.group("task_id"),
    }


def generate_qa_items(subtasks: Iterable[Subtask], seed: int = 0, families: set[str] | None = None) -> list[dict]:
    if families is None:
        families = {"subtask_progress_memory"}
    grouped: dict[tuple[str, str, str, str], list[Subtask]] = {}
    for subtask in subtasks:
        grouped.setdefault(subtask.episode_key, []).append(subtask)

    rng = random.Random(seed)
    items: list[dict] = []
    for episode_key, episode_subtasks in sorted(grouped.items()):
        ordered = sorted(
            episode_subtasks,
            key=lambda x: (x.order_index is None, x.order_index if x.order_index is not None else 9999, x.action, x.object_name),
        )
        placed = [x for x in ordered if x.action == "place" and x.target]
        if len(placed) < 2:
            continue
        starts = _episode_starts(ordered)
        episode_query_frame = max(0, sum(_segment_len(x) for x in ordered) - 1)
        if "source_target_memory" in families:
            targets = sorted({x.target for x in placed})
            for event_idx, subtask in enumerate(placed):
                choices = _make_choices(subtask.target or "unknown", targets, rng)
                query_frame = episode_query_frame
                evidence_frame = starts[subtask.path] + _evidence_frame(subtask)
                event_label = f"placement event {event_idx + 1} of {len(placed)}"
                item_id = "memprobe_" + _stable_hash([str(subtask.path), subtask.object_name, subtask.target or ""])
                items.append(
                    {
                        "id": item_id,
                        "source_dataset": "robomemarena",
                        "episode_uid": "ep_" + _stable_hash(list(episode_key))[:12],
                        "query_frame": query_frame,
                        "question_family": "source_target_memory",
                        "question": (
                            f"At {event_label}, before the episode-end query frame, which target did the robot "
                            f"place the {subtask.object_name.replace('_', ' ')} into?"
                        ),
                        "choices": choices,
                        "answer": next(choice["id"] for choice in choices if choice["canonical"] == subtask.target),
                        "structured_query": {
                            "predicate": "placed_into",
                            "object": subtask.object_name,
                            "answer_type": "target_id",
                            "temporal_reference": {
                                "type": "subtask_boundary",
                                "query_frame": query_frame,
                                "anchor_event_index": event_idx + 1,
                                "anchor_event_count": len(placed),
                                "anchor_event_uid": _opaque_path_id(subtask.path),
                                "relation": "at_event_keyframe_before_episode_end",
                                "state_rule": "target_from_subtask_filename_at_hdf5_keyframe",
                            },
                            "spatial_reference_frame": "dataset_task_frame",
                        },
                        "answer_canonical": {
                            "predicate": "placed_into",
                            "object": subtask.object_name,
                            "target": subtask.target,
                        },
                        "evidence": [
                            {
                                "type": "subtask_keyframe",
                                "subtask_uid": _opaque_path_id(subtask.path),
                                "frame_range": [evidence_frame, evidence_frame],
                                "fact": f"place({subtask.object_name},{subtask.target})",
                                "language_instruction": subtask.metadata.language_instruction,
                            }
                        ],
                        "memory_horizon_frames": max(0, query_frame - evidence_frame),
                        "requires_memory": True,
                        "answerable_from_query_frame": False,
                        "verification_level": "auto_generated_candidate_requires_human_audit",
                        "provenance": {
                            "local_path": str(subtask.path),
                            "suite": subtask.suite,
                            "task_name": subtask.task_name,
                            "seed": subtask.seed,
                            "task_id": subtask.task_id,
                            "demo_id": subtask.metadata.demo_id,
                            "num_samples": subtask.metadata.num_samples,
                        },
                    }
                )
        if "subtask_progress_memory" in families:
            items.extend(_generate_previous_subtask_items(ordered, starts, rng, episode_key))
    return items


def write_jsonl(items: Iterable[dict], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def _split_object_target(action: str, core: list[str]) -> tuple[str, str | None]:
    if action == "pick":
        return "_".join(core), None
    if action == "place" and len(core) >= 2:
        target = core[-1]
        return "_".join(core[:-1]), target
    return "_".join(core), None


def _make_choices(answer: str, targets: list[str], rng: random.Random) -> list[dict]:
    distractors = [x for x in targets if x != answer]
    fallback = ["cabinet1", "cabinet2", "plate1", "plate2", "basket", "tray", "counter"]
    for candidate in fallback:
        if candidate != answer and candidate not in distractors:
            distractors.append(candidate)
    rng.shuffle(distractors)
    canonicals = [answer, *distractors[:3]]
    rng.shuffle(canonicals)
    return [{"id": chr(ord("A") + i), "text": _surface_target(c), "canonical": c} for i, c in enumerate(canonicals)]


def _generate_previous_subtask_items(
    ordered: list[Subtask],
    starts: dict[Path, int],
    rng: random.Random,
    episode_key: tuple[str, str, str, str],
) -> list[dict]:
    labels = [_subtask_label(x) for x in ordered]
    unique_labels = sorted(set(labels))
    items: list[dict] = []
    for subtask_index, current in enumerate(ordered[1:], start=1):
        previous = ordered[subtask_index - 1]
        answer = _subtask_label(previous)
        choices = _make_label_choices(answer, unique_labels, rng)
        query_frame = starts[current.path]
        evidence_frame = starts[previous.path] + _evidence_frame(previous)
        item_id = "memprobe_" + _stable_hash([str(current.path), "previous_completed_subtask", str(previous.path)])
        items.append(
            {
                "id": item_id,
                "source_dataset": "robomemarena",
                "episode_uid": "ep_" + _stable_hash(list(episode_key))[:12],
                "query_frame": query_frame,
                "question_family": "subtask_progress_memory",
                "question": "In the frame marked as the query frame, what action has the robot just completed?",
                "choices": choices,
                "answer": next(choice["id"] for choice in choices if choice["canonical"] == answer),
                "structured_query": {
                    "predicate": "previous_completed_subtask",
                    "answer_type": "subtask_label",
                    "temporal_reference": {
                        "type": "subtask_boundary",
                        "query_frame": query_frame,
                        "anchor_subtask_index": subtask_index + 1,
                        "anchor_subtask_count": len(ordered),
                        "anchor_subtask_uid": _opaque_path_id(current.path),
                        "relation": "immediately_before_subtask_start",
                        "state_rule": "previous_subtask_by_order_index",
                    },
                    "spatial_reference_frame": "not_applicable",
                },
                "answer_canonical": {
                    "predicate": "completed_subtask",
                    "subtask": answer,
                    "subtask_uid": _opaque_path_id(previous.path),
                },
                "evidence": [
                    {
                        "type": "subtask_keyframe",
                        "subtask_uid": _opaque_path_id(previous.path),
                        "frame_range": [evidence_frame, evidence_frame],
                        "fact": f"completed({answer})",
                        "language_instruction": previous.metadata.language_instruction,
                    }
                ],
                "memory_horizon_frames": max(0, query_frame - evidence_frame),
                "requires_memory": True,
                "answerable_from_query_frame": False,
                "verification_level": "auto_generated_candidate_requires_human_audit",
                "provenance": {
                    "local_path": str(current.path),
                    "previous_local_path": str(previous.path),
                    "suite": current.suite,
                    "task_name": current.task_name,
                    "seed": current.seed,
                    "task_id": current.task_id,
                    "demo_id": current.metadata.demo_id,
                    "num_samples": current.metadata.num_samples,
                },
            }
        )
    return items


def _make_label_choices(answer: str, labels: list[str], rng: random.Random) -> list[dict]:
    distractors = [x for x in labels if x != answer]
    rng.shuffle(distractors)
    canonicals = [answer, *distractors[:3]]
    rng.shuffle(canonicals)
    return [{"id": chr(ord("A") + i), "text": c, "canonical": c} for i, c in enumerate(canonicals)]


def _subtask_label(subtask: Subtask) -> str:
    if subtask.metadata.language_instruction:
        return subtask.metadata.language_instruction
    if subtask.action == "place" and subtask.target:
        return f"place the {subtask.object_name.replace('_', ' ')} into {subtask.target}"
    return f"{subtask.action} the {subtask.object_name.replace('_', ' ')}"


def _surface_target(value: str) -> str:
    return re.sub(r"([a-zA-Z]+)(\d+)$", r"\1 \2", value).replace("_", " ")


def _query_frame(subtask: Subtask) -> int:
    if subtask.metadata.num_samples is not None:
        return int(subtask.metadata.num_samples) - 1
    if subtask.metadata.keyframe_indices:
        return max(subtask.metadata.keyframe_indices)
    return 0


def _evidence_frame(subtask: Subtask) -> int:
    if subtask.metadata.keyframe_indices:
        return int(subtask.metadata.keyframe_indices[-1])
    return max(0, _query_frame(subtask) // 2)


def _episode_starts(subtasks: list[Subtask]) -> dict[Path, int]:
    starts: dict[Path, int] = {}
    cursor = 0
    for subtask in subtasks:
        starts[subtask.path] = cursor
        cursor += _segment_len(subtask)
    return starts


def _segment_len(subtask: Subtask) -> int:
    if subtask.metadata.num_samples is not None:
        return max(1, int(subtask.metadata.num_samples))
    return max(1, _query_frame(subtask) + 1)


def _opaque_path_id(path: Path) -> str:
    return "subtask_" + _stable_hash([str(path)])[:12]


def _stable_hash(values: list[str]) -> str:
    return hashlib.sha1("::".join(values).encode("utf-8")).hexdigest()
