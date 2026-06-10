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

    items = _load_items(args.input, args.max_items)
    prompts = [_format_prompt(item) for item in items]
    if args.dry_run:
        for prompt in prompts[:5]:
            print(prompt)
            print()
        print(f"dry_run_items={len(items)}")
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
