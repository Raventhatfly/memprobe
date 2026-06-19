#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate MemProbe MCQ JSONL with an open Qwen-VL style model.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Print prompts without loading the model.")
    args = parser.parse_args()

    items = _load_items(args.input, None if args.dry_run else args.max_items)
    if args.dry_run:
        display_items = _select_dry_run_items(items, args.max_items or 5)
        for item in display_items:
            print(_format_prompt(item))
            print()
        print(f"dry_run_items={len(display_items)}")
        print(f"total_items={len(items)}")
        return

    raise SystemExit(
        "Model execution is intentionally left as an explicit step. Install the qwen extra and "
        "wire extracted evidence/current frames into this script before running GPU inference."
    )


def _load_items(path: Path, max_items: int | None) -> list[dict]:
    items: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
                if max_items is not None and len(items) >= max_items:
                    break
    return items


def _select_dry_run_items(items: list[dict], max_items: int) -> list[dict]:
    if max_items <= 0:
        return []

    grouped: dict[str, list[dict]] = {}
    for item in items:
        provenance = item.get("provenance", {})
        group = str(provenance.get("task_name") or item.get("episode_uid") or "items")
        grouped.setdefault(group, []).append(item)

    selected: list[dict] = []
    while len(selected) < max_items:
        added = False
        for group in sorted(grouped):
            if grouped[group]:
                selected.append(grouped[group].pop(0))
                added = True
                if len(selected) >= max_items:
                    break
        if not added:
            break
    return selected


def _format_prompt(item: dict) -> str:
    choices = "\n".join(f"{choice['id']}. {choice['text']}" for choice in item["choices"])
    return (
        "Answer the multiple-choice robot memory question using only the provided visual context.\n"
        f"Question: {item['question']}\n"
        f"Choices:\n{choices}\n"
        "Return only one letter."
    )


if __name__ == "__main__":
    main()
