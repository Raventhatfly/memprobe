#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from memprobe.robomemarena import (
    DEFAULT_ROBOMEMARENA_ROOT,
    generate_qa_items,
    scan_episode_subtasks,
    scan_subtasks,
    write_jsonl,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate prototype MemProbe QA from RoboMemArena HDF5 subtasks.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROBOMEMARENA_ROOT)
    parser.add_argument("--output", type=Path, default=Path("outputs/robomemarena_memprobe_prototype.jsonl"))
    parser.add_argument("--limit-files", type=int, default=None)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--families", default=None, help="Comma-separated question families to generate.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.max_episodes is not None:
        subtasks = scan_episode_subtasks(args.root, max_episodes=args.max_episodes)
    else:
        subtasks = scan_subtasks(args.root, limit_files=args.limit_files)
    families = set(args.families.split(",")) if args.families else None
    items = generate_qa_items(subtasks, seed=args.seed, families=families)
    write_jsonl(items, args.output)
    print(f"scanned_subtasks={len(subtasks)}")
    print(f"generated_items={len(items)}")
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
