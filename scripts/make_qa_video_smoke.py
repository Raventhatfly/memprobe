#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from hdf5_to_video import convert_one
from memprobe.robomemarena import DEFAULT_ROBOMEMARENA_ROOT, generate_qa_items, scan_episode_subtasks, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a small RoboMemArena QA set and matching videos.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROBOMEMARENA_ROOT)
    parser.add_argument("--max-episodes", type=int, default=5)
    parser.add_argument("--max-qa", type=int, default=10)
    parser.add_argument("--families", default="subtask_progress_memory")
    parser.add_argument("--rewritten-qa", type=Path, default=None, help="Use this QA JSONL instead of generating new QA.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/qa_video_smoke"))
    parser.add_argument("--dataset", default="/data/demo_0/obs/agentview_rgb")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--episode-strategy",
        choices=["sorted", "round-robin-tasks"],
        default="sorted",
        help="How to choose episodes when generating QA.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    video_dir = args.output_dir / "videos"
    video_dir.mkdir(exist_ok=True)

    if args.rewritten_qa is not None:
        subtasks = []
        items = _load_jsonl(args.rewritten_qa)[: args.max_qa]
    else:
        subtasks = scan_episode_subtasks(
            args.root,
            max_episodes=args.max_episodes,
            episode_strategy=args.episode_strategy.replace("-", "_"),
        )
        families = set(args.families.split(",")) if args.families else None
        items = generate_qa_items(subtasks, families=families)[: args.max_qa]

    qa_path = args.output_dir / "qa.jsonl"
    write_jsonl(items, qa_path)

    path_to_video: dict[str, str] = {}
    for item in items:
        for key in ("local_path", "previous_local_path"):
            path_text = item.get("provenance", {}).get(key)
            if path_text:
                input_path = Path(path_text)
                output_path = video_dir / f"{input_path.stem}.mp4"
                if args.overwrite or not output_path.exists():
                    convert_one(input_path, output_path, args.dataset, fps=args.fps, max_frames=None)
                path_to_video[path_text] = str(output_path)

    manifest = []
    for item in items:
        provenance = item.get("provenance", {})
        manifest.append(
            {
                "id": item["id"],
                "question_family": item["question_family"],
                "question": item["question"],
                "answer": item["answer"],
                "choices": item["choices"],
                "query_frame": item["query_frame"],
                "evidence": item["evidence"],
                "video": path_to_video.get(provenance.get("local_path")),
                "previous_video": path_to_video.get(provenance.get("previous_local_path")),
                "local_path": provenance.get("local_path"),
                "previous_local_path": provenance.get("previous_local_path"),
            }
        )

    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"scanned_subtasks={len(subtasks)}")
    print(f"generated_qa={len(items)}")
    print(f"videos={len(path_to_video)}")
    print(f"qa={qa_path}")
    print(f"manifest={manifest_path}")
    print(f"video_dir={video_dir}")


def _load_jsonl(path: Path) -> list[dict]:
    items = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items


if __name__ == "__main__":
    main()
