from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Optional, Sequence


SCHEMA_VERSION = "memprobe.v1"


def stable_id(prefix: str, *parts: object, length: int = 16) -> str:
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return f"{prefix}_{hashlib.sha256(payload).hexdigest()[:length]}"


@dataclass(frozen=True)
class TimeSpan:
    start_s: float
    end_s: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.start_s) or not math.isfinite(self.end_s):
            raise ValueError("time span values must be finite")
        if self.start_s < 0:
            raise ValueError("time span start must be non-negative")
        if self.end_s < self.start_s:
            raise ValueError("time span end must not precede start")

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s

    def intersection_s(self, other: "TimeSpan") -> float:
        return max(0.0, min(self.end_s, other.end_s) - max(self.start_s, other.start_s))

    def temporal_iou(self, other: "TimeSpan") -> float:
        intersection = self.intersection_s(other)
        union = self.duration_s + other.duration_s - intersection
        return intersection / union if union > 0 else float(self == other)

    def clamp(self, duration_s: float) -> "TimeSpan":
        return TimeSpan(max(0.0, self.start_s), min(duration_s, self.end_s))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TimeSpan":
        return cls(start_s=float(data["start_s"]), end_s=float(data["end_s"]))


@dataclass(frozen=True)
class CameraStream:
    camera_id: str
    uri: str
    duration_s: float
    fps: Optional[float] = None
    width: Optional[int] = None
    height: Optional[int] = None

    def __post_init__(self) -> None:
        if not self.camera_id:
            raise ValueError("camera_id is required")
        if not self.uri:
            raise ValueError("camera stream uri is required")
        if self.duration_s <= 0:
            raise ValueError("camera duration must be positive")
        if self.fps is not None and self.fps <= 0:
            raise ValueError("camera fps must be positive")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CameraStream":
        return cls(
            camera_id=str(data["camera_id"]),
            uri=str(data["uri"]),
            duration_s=float(data["duration_s"]),
            fps=float(data["fps"]) if data.get("fps") is not None else None,
            width=int(data["width"]) if data.get("width") is not None else None,
            height=int(data["height"]) if data.get("height") is not None else None,
        )


@dataclass(frozen=True)
class ScalarSignal:
    name: str
    timestamps_s: tuple[float, ...]
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("signal name is required")
        if len(self.timestamps_s) != len(self.values):
            raise ValueError(f"signal {self.name!r} timestamps and values differ in length")
        if any(curr <= prev for prev, curr in zip(self.timestamps_s, self.timestamps_s[1:])):
            raise ValueError(f"signal {self.name!r} timestamps must be strictly increasing")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ScalarSignal":
        return cls(
            name=str(data["name"]),
            timestamps_s=tuple(float(value) for value in data.get("timestamps_s", [])),
            values=tuple(float(value) for value in data.get("values", [])),
        )


@dataclass(frozen=True)
class CanonicalEpisode:
    episode_uid: str
    duration_s: float
    streams: tuple[CameraStream, ...]
    signals: tuple[ScalarSignal, ...] = ()
    private_metadata: Mapping[str, Any] = field(default_factory=dict)
    source_provenance: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.episode_uid:
            raise ValueError("episode_uid is required")
        if self.duration_s <= 0:
            raise ValueError("episode duration must be positive")
        if not self.streams:
            raise ValueError("at least one camera stream is required")
        camera_ids = [stream.camera_id for stream in self.streams]
        if len(camera_ids) != len(set(camera_ids)):
            raise ValueError("camera IDs must be unique within an episode")
        if any(stream.duration_s + 1e-6 < self.duration_s for stream in self.streams):
            raise ValueError("camera stream is shorter than the canonical episode duration")

    def stream(self, camera_id: Optional[str] = None) -> CameraStream:
        if camera_id is None:
            return self.streams[0]
        for stream in self.streams:
            if stream.camera_id == camera_id:
                return stream
        raise KeyError(f"unknown camera_id {camera_id!r}")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CanonicalEpisode":
        return cls(
            episode_uid=str(data["episode_uid"]),
            duration_s=float(data["duration_s"]),
            streams=tuple(CameraStream.from_dict(item) for item in data["streams"]),
            signals=tuple(ScalarSignal.from_dict(item) for item in data.get("signals", [])),
            private_metadata=dict(data.get("private_metadata", {})),
            source_provenance=dict(data.get("source_provenance", {})),
            schema_version=str(data.get("schema_version", SCHEMA_VERSION)),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EventProposal:
    event_id: str
    episode_uid: str
    camera_id: str
    span: TimeSpan
    event_type: str
    confidence: float
    source: str
    peak_s: Optional[float] = None
    subject_refs: tuple[str, ...] = ()
    region_refs: tuple[str, ...] = ()
    evidence: tuple[TimeSpan, ...] = ()
    verification_status: str = "proposed"
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.event_id or not self.episode_uid or not self.camera_id:
            raise ValueError("event ID, episode ID, and camera ID are required")
        if not self.event_type:
            raise ValueError("event_type is required")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("event confidence must be in [0, 1]")
        if self.peak_s is not None and not self.span.start_s <= self.peak_s <= self.span.end_s:
            raise ValueError("event peak must lie inside its span")

    @property
    def is_verified(self) -> bool:
        return self.verification_status == "verified"

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], episode_uid: Optional[str] = None) -> "EventProposal":
        span_data = data.get("span") or {"start_s": data["start_s"], "end_s": data["end_s"]}
        uid = str(data.get("episode_uid") or episode_uid or "")
        event_type = str(data.get("event_type", "unknown_change"))
        source = str(data.get("source", "precomputed"))
        event_id = str(
            data.get("event_id")
            or stable_id("evt", uid, span_data["start_s"], span_data["end_s"], event_type, source)
        )
        return cls(
            event_id=event_id,
            episode_uid=uid,
            camera_id=str(data.get("camera_id", "cam_0")),
            span=TimeSpan.from_dict(span_data),
            event_type=event_type,
            confidence=float(data.get("confidence", 0.5)),
            source=source,
            peak_s=float(data["peak_s"]) if data.get("peak_s") is not None else None,
            subject_refs=tuple(str(value) for value in data.get("subject_refs", [])),
            region_refs=tuple(str(value) for value in data.get("region_refs", [])),
            evidence=tuple(TimeSpan.from_dict(item) for item in data.get("evidence", [])),
            verification_status=str(data.get("verification_status", "proposed")),
            attributes=dict(data.get("attributes", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["span"] = asdict(self.span)
        data["evidence"] = [asdict(item) for item in self.evidence]
        return data


@dataclass(frozen=True)
class TrackObservation:
    time_s: float
    point_xy_norm: tuple[float, float]
    visibility: str
    confidence: float
    box_xyxy_norm: Optional[tuple[float, float, float, float]] = None


@dataclass(frozen=True)
class ObjectTrack:
    track_id: str
    episode_uid: str
    camera_id: str
    observations: tuple[TrackObservation, ...]
    verification_status: str = "proposed"


@dataclass(frozen=True)
class MediaRef:
    uri: str
    camera_id: str
    span: TimeSpan
    kind: str = "video_clip"
    markers: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["span"] = asdict(self.span)
        return data


@dataclass(frozen=True)
class ProbeChoice:
    choice_id: str
    media: MediaRef
    source_event_id: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.choice_id,
            "media": self.media.to_dict(),
            "source_event_id": self.source_event_id,
        }


@dataclass(frozen=True)
class ProbeItem:
    probe_id: str
    episode_uid: str
    probe_type: str
    response_type: str
    history: MediaRef
    query_time_s: float
    choices: tuple[ProbeChoice, ...]
    answer: Any
    operator: Mapping[str, Any]
    evidence: tuple[MediaRef, ...]
    validation: Mapping[str, Any]
    provenance: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    def to_dict(self, include_private: bool = True) -> dict[str, Any]:
        data: dict[str, Any] = {
            "schema_version": self.schema_version,
            "id": self.probe_id,
            "episode_uid": self.episode_uid,
            "probe_type": self.probe_type,
            "response_type": self.response_type,
            "history": self.history.to_dict(),
            "query_time_s": self.query_time_s,
            "choices": [choice.to_dict() for choice in self.choices],
            "answer": self.answer,
            "operator": dict(self.operator),
            "validation": dict(self.validation),
        }
        if include_private:
            data["evidence"] = [item.to_dict() for item in self.evidence]
            data["provenance"] = dict(self.provenance)
        else:
            data.pop("query_time_s", None)
            data.pop("validation", None)
            data["operator"] = {"name": self.operator["name"]}
            data["history"] = {
                "uri": f"media/{self.probe_id}/history.mp4",
                "kind": "history_video",
            }
            for choice in data["choices"]:
                choice["media"] = {
                    "uri": f"media/{self.probe_id}/choice_{choice['id']}.mp4",
                    "kind": "event_clip",
                }
                choice.pop("source_event_id", None)
        return data


def ensure_unique_ids(items: Sequence[EventProposal]) -> None:
    ids = [item.event_id for item in items]
    if len(ids) != len(set(ids)):
        raise ValueError("event IDs must be unique")
