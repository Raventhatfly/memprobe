from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

from .schema import EventProposal, ProbeItem


def load_event_proposals(path: Path, episode_uid: Optional[str] = None) -> list[EventProposal]:
    records = _load_records(path)
    proposals = [EventProposal.from_dict(record, episode_uid=episode_uid) for record in records]
    if episode_uid is not None:
        proposals = [item for item in proposals if item.episode_uid == episode_uid]
    return proposals


def write_probe_jsonl(items: Iterable[ProbeItem], path: Path, include_private: bool) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(item.to_dict(include_private=include_private), ensure_ascii=False) + "\n")
            count += 1
    return count


def write_json(data: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _load_records(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".jsonl":
        records = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSON on {path}:{line_number}: {exc}") from exc
        return records

    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and "events" in payload:
        return list(payload["events"])
    if isinstance(payload, dict):
        return [payload]
    raise ValueError(f"expected a JSON object or array in {path}")
