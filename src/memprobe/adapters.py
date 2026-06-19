from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Protocol

from .schema import CanonicalEpisode


class EpisodeAdapter(Protocol):
    """Dataset boundary for the generation pipeline."""

    def episode_ids(self) -> Iterable[str]:
        ...

    def load_episode(self, episode_id: str) -> CanonicalEpisode:
        ...


class ManifestAdapter:
    """Load canonical episodes from one JSON manifest or a JSONL collection."""

    def __init__(self, manifest_path: Path):
        self.manifest_path = manifest_path.resolve()
        self._episodes = _load_manifest_records(self.manifest_path)
        self._by_id = {episode.episode_uid: episode for episode in self._episodes}
        if len(self._by_id) != len(self._episodes):
            raise ValueError(f"duplicate episode_uid in {self.manifest_path}")

    def episode_ids(self) -> Iterable[str]:
        return tuple(sorted(self._by_id))

    def load_episode(self, episode_id: str) -> CanonicalEpisode:
        try:
            return self._by_id[episode_id]
        except KeyError as exc:
            raise KeyError(f"episode {episode_id!r} is not present in {self.manifest_path}") from exc


def _load_manifest_records(path: Path) -> list[CanonicalEpisode]:
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
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            records = payload
        elif isinstance(payload, dict) and "episodes" in payload:
            records = payload["episodes"]
        else:
            records = [payload]

    episodes = []
    for record in records:
        normalized = dict(record)
        normalized["streams"] = [
            _resolve_stream_uri(dict(stream), path.parent) for stream in normalized.get("streams", [])
        ]
        episodes.append(CanonicalEpisode.from_dict(normalized))
    return episodes


def _resolve_stream_uri(stream: dict, base_dir: Path) -> dict:
    uri = str(stream.get("uri", ""))
    if uri and "://" not in uri and not Path(uri).is_absolute():
        stream["uri"] = str((base_dir / uri).resolve())
    return stream
