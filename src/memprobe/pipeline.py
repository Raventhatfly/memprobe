from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from .adapters import EpisodeAdapter
from .backends import ProposalBackend
from .graph import EventStateGraph
from .probes import build_generators
from .schema import ProbeItem
from .storage import write_json, write_probe_jsonl
from .validation import validate_probe
from .windows import merge_windows, propose_signal_windows, windows_from_events


@dataclass(frozen=True)
class GenerationConfig:
    probe_types: tuple[str, ...] = ("PREVIOUS_EVENT", "EVENT_ORDER")
    num_choices: int = 4
    seed: int = 0
    min_event_confidence: float = 0.0
    require_verified: bool = False
    signal_padding_s: float = 1.0
    window_merge_gap_s: float = 0.25


@dataclass(frozen=True)
class GenerationResult:
    episodes: int
    event_proposals: int
    candidate_windows: int
    probes: int
    release_ready_probes: int
    private_path: Path
    public_path: Path
    windows_path: Path


def generate_dataset(
    adapter: EpisodeAdapter,
    backend: ProposalBackend,
    output_dir: Path,
    config: GenerationConfig,
) -> GenerationResult:
    generators = build_generators(config.probe_types, num_choices=config.num_choices)
    probes: list[ProbeItem] = []
    window_records = []
    proposal_count = 0
    episode_count = 0

    for episode_id in adapter.episode_ids():
        episode_count += 1
        episode = adapter.load_episode(episode_id)
        events = backend.propose(episode)
        proposal_count += len(events)

        windows = merge_windows(
            [
                *propose_signal_windows(episode, padding_s=config.signal_padding_s),
                *windows_from_events(events),
            ],
            max_gap_s=config.window_merge_gap_s,
        )
        window_records.extend(
            {
                **asdict(window),
                "span": asdict(window.span),
            }
            for window in windows
        )

        graph = EventStateGraph.build(
            episode,
            events,
            min_confidence=config.min_event_confidence,
            require_verified=config.require_verified,
        )
        for generator in generators:
            for item in generator.generate(graph, seed=config.seed):
                probes.append(validate_probe(item, graph))

    output_dir.mkdir(parents=True, exist_ok=True)
    private_path = output_dir / "probes.private.jsonl"
    public_path = output_dir / "probes.public.jsonl"
    windows_path = output_dir / "candidate_windows.json"
    write_probe_jsonl(probes, private_path, include_private=True)
    write_probe_jsonl(probes, public_path, include_private=False)
    write_json({"schema_version": "memprobe.v1", "windows": window_records}, windows_path)

    return GenerationResult(
        episodes=episode_count,
        event_proposals=proposal_count,
        candidate_windows=len(window_records),
        probes=len(probes),
        release_ready_probes=sum(item.validation.get("release_ready") is True for item in probes),
        private_path=private_path,
        public_path=public_path,
        windows_path=windows_path,
    )
