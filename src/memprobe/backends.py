from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol

from .schema import CanonicalEpisode, EventProposal
from .storage import load_event_proposals


EVENT_VOCABULARY = (
    "gripper_close",
    "gripper_open",
    "contact_change",
    "object_motion_start",
    "object_motion_stop",
    "container_state_change",
    "occlusion_change",
    "unknown_change",
)


class ProposalBackend(Protocol):
    backend_id: str

    def propose(self, episode: CanonicalEpisode) -> list[EventProposal]:
        ...


class PrecomputedProposalBackend:
    backend_id = "precomputed"

    def __init__(self, path: Path):
        self.path = path
        proposals = load_event_proposals(path)
        self._by_episode: dict[str, list[EventProposal]] = {}
        for proposal in proposals:
            self._by_episode.setdefault(proposal.episode_uid, []).append(proposal)

    def propose(self, episode: CanonicalEpisode) -> list[EventProposal]:
        return list(self._by_episode.get(episode.episode_uid, []))


class EmptyProposalBackend:
    backend_id = "none"

    def propose(self, episode: CanonicalEpisode) -> list[EventProposal]:
        return []


class Qwen3VLProposalBackend:
    """Lazy local Qwen3-VL backend for coarse event proposals.

    This backend intentionally emits unverified proposals. Dense local refinement,
    cross-model verification, and human review consume these records later.
    """

    backend_id = "qwen3_vl"

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-VL-8B-Instruct",
        overview_fps: float = 0.5,
        max_new_tokens: int = 4096,
        local_files_only: bool = True,
    ):
        if overview_fps <= 0:
            raise ValueError("overview_fps must be positive")
        self.model_name = model_name
        self.overview_fps = overview_fps
        self.max_new_tokens = max_new_tokens
        self.local_files_only = local_files_only
        self._processor = None
        self._model = None

    def propose(self, episode: CanonicalEpisode) -> list[EventProposal]:
        processor, model = self._load()
        stream = episode.stream()
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": _video_uri(stream.uri),
                        "fps": self.overview_fps,
                    },
                    {"type": "text", "text": _event_prompt(episode.duration_s, stream.camera_id)},
                ],
            }
        ]
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = inputs.to(model.device)
        generated = model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        trimmed = generated[:, inputs.input_ids.shape[1] :]
        text = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        records = _extract_event_records(text)

        proposals = []
        for record in records:
            normalized = dict(record)
            normalized["episode_uid"] = episode.episode_uid
            normalized["camera_id"] = stream.camera_id
            normalized["source"] = f"qwen3_vl:{self.model_name}"
            normalized["verification_status"] = "proposed"
            proposal = EventProposal.from_dict(normalized)
            if proposal.span.end_s <= episode.duration_s + 1e-6:
                proposals.append(proposal)
        return proposals

    def _load(self):
        if self._processor is not None and self._model is not None:
            return self._processor, self._model
        try:
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except ImportError as exc:
            raise RuntimeError("Qwen backend requires: pip install -e '.[qwen]'") from exc

        self._processor = AutoProcessor.from_pretrained(
            self.model_name,
            local_files_only=self.local_files_only,
        )
        self._model = AutoModelForImageTextToText.from_pretrained(
            self.model_name,
            dtype="auto",
            device_map="auto",
            local_files_only=self.local_files_only,
        )
        self._model.eval()
        return self._processor, self._model


def _video_uri(uri: str) -> str:
    if "://" in uri:
        return uri
    return Path(uri).resolve().as_uri()


def _event_prompt(duration_s: float, camera_id: str) -> str:
    vocabulary = ", ".join(EVENT_VOCABULARY)
    return f"""You propose visible robot-manipulation event intervals for later verification.
Return JSON only with schema:
{{"events": [{{"start_s": number, "peak_s": number|null, "end_s": number,
"event_type": string, "confidence": number, "subject_refs": [string],
"region_refs": [string], "evidence": [{{"start_s": number, "end_s": number}}]}}]}}

Video duration: {duration_s:.6f} seconds. Camera: {camera_id}.
Allowed event_type values: {vocabulary}.
Use only visible evidence. Use opaque local references such as object_1; do not
invent semantic object names. Intervals must be within the video. Include an
empty events list when no reliable event is visible."""


def _extract_event_records(text: str) -> list[dict[str, Any]]:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("Qwen output did not contain a JSON object")
    try:
        payload = json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        raise ValueError(f"Qwen output was not valid JSON: {exc}") from exc
    events = payload.get("events")
    if not isinstance(events, list):
        raise ValueError("Qwen JSON must contain an events array")
    return [dict(item) for item in events if isinstance(item, dict)]
