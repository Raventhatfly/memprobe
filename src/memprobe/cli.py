from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

from .adapters import ManifestAdapter
from .backends import PrecomputedProposalBackend, Qwen3VLProposalBackend
from .materialize import materialize_probe_media
from .pipeline import GenerationConfig, generate_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="memprobe", description="Generate dataset-independent visual memory probes.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect", help="Validate and summarize a canonical episode manifest.")
    inspect_parser.add_argument("--manifest", type=Path, required=True)

    generate_parser = subparsers.add_parser("generate", help="Generate draft probes from canonical episodes.")
    generate_parser.add_argument("--manifest", type=Path, required=True)
    generate_parser.add_argument("--output-dir", type=Path, default=Path("outputs/memprobe_v1"))
    backend = generate_parser.add_mutually_exclusive_group(required=True)
    backend.add_argument("--proposals", type=Path, help="Precomputed event proposal JSON or JSONL.")
    backend.add_argument(
        "--qwen-model",
        nargs="?",
        const="Qwen/Qwen3-VL-8B-Instruct",
        help="Run a local Qwen3-VL model; optionally provide a model ID or local path.",
    )
    generate_parser.add_argument("--allow-model-download", action="store_true")
    generate_parser.add_argument("--overview-fps", type=float, default=0.5)
    generate_parser.add_argument("--probe-types", default="PREVIOUS_EVENT,EVENT_ORDER")
    generate_parser.add_argument("--num-choices", type=int, default=4)
    generate_parser.add_argument("--seed", type=int, default=0)
    generate_parser.add_argument("--min-event-confidence", type=float, default=0.0)
    generate_parser.add_argument("--verified-only", action="store_true")

    materialize_parser = subparsers.add_parser("materialize", help="Extract history and visual choices with ffmpeg.")
    materialize_parser.add_argument("--private-jsonl", type=Path, required=True)
    materialize_parser.add_argument("--output-dir", type=Path, required=True)
    materialize_parser.add_argument("--ffmpeg", default="ffmpeg")
    materialize_parser.add_argument("--overwrite", action="store_true")
    materialize_parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "inspect":
        _inspect(args.manifest)
        return
    if args.command == "generate":
        _generate(args)
        return
    if args.command == "materialize":
        completed, skipped = materialize_probe_media(
            args.private_jsonl,
            args.output_dir,
            ffmpeg_bin=args.ffmpeg,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
        )
        print(json.dumps({"materialized": completed, "skipped": skipped}, indent=2))
        return
    raise AssertionError(f"unhandled command: {args.command}")


def _inspect(manifest_path: Path) -> None:
    adapter = ManifestAdapter(manifest_path)
    records = []
    for episode_id in adapter.episode_ids():
        episode = adapter.load_episode(episode_id)
        records.append(
            {
                "episode_uid": episode.episode_uid,
                "duration_s": episode.duration_s,
                "cameras": [stream.camera_id for stream in episode.streams],
                "signals": [signal.name for signal in episode.signals],
            }
        )
    print(json.dumps({"episodes": records}, indent=2))


def _generate(args: argparse.Namespace) -> None:
    if args.num_choices < 2:
        raise SystemExit("--num-choices must be at least 2")
    if not 0.0 <= args.min_event_confidence <= 1.0:
        raise SystemExit("--min-event-confidence must be in [0, 1]")

    adapter = ManifestAdapter(args.manifest)
    if args.proposals is not None:
        backend = PrecomputedProposalBackend(args.proposals)
    else:
        backend = Qwen3VLProposalBackend(
            model_name=args.qwen_model,
            overview_fps=args.overview_fps,
            local_files_only=not args.allow_model_download,
        )

    config = GenerationConfig(
        probe_types=tuple(value.strip() for value in args.probe_types.split(",") if value.strip()),
        num_choices=args.num_choices,
        seed=args.seed,
        min_event_confidence=args.min_event_confidence,
        require_verified=args.verified_only,
    )
    result = generate_dataset(adapter, backend, args.output_dir, config)
    print(
        json.dumps(
            {
                "episodes": result.episodes,
                "event_proposals": result.event_proposals,
                "candidate_windows": result.candidate_windows,
                "probes": result.probes,
                "release_ready_probes": result.release_ready_probes,
                "private_output": str(result.private_path),
                "public_output": str(result.public_path),
                "candidate_windows_output": str(result.windows_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
