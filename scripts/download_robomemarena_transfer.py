#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import re
from collections import defaultdict
from pathlib import Path

REPO_ID = "RoboMemArenaBenchmark/RoboMemArena"
TRANSFER_PREFIX = "Multi-Transferring"
TASK_PREFIXES = {
    "18": "Multi-Transferring/18_chocolate_butter_cabinet_dataset/subtask_data",
    "19": "Multi-Transferring/19_tomato_milk_orange_cabinet_dataset/subtask_data",
    "25": "Multi-Transferring/25_butter_cream_dataset/subtask_data",
    "26": "Multi-Transferring/26_chocolate_pudding_cream_dataset/subtask_data",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Download RoboMemArena transfer data from Hugging Face.")
    parser.add_argument("--root", type=Path, default=Path("data/robomemarena"))
    parser.add_argument("--task", choices=sorted(TASK_PREFIXES), default="18")
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--full", action="store_true", help="Download the complete Multi-Transferring folder.")
    parser.add_argument("--repo-id", default=REPO_ID)
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    args = parser.parse_args()

    try:
        from huggingface_hub import HfApi, hf_hub_download, snapshot_download
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing huggingface_hub. In your fluxvla env, run: pip install -U huggingface_hub hf-xet"
        ) from exc

    token = args.token if args.token else True
    print(f"[download] repo={args.repo_id}")
    print(f"[download] root={args.root}")

    if args.full:
        print(f"[download] prefix={TRANSFER_PREFIX}/**")
        print("[download] mode=full transfer folder")
        try:
            snapshot_download(
                repo_id=args.repo_id,
                repo_type="dataset",
                allow_patterns=f"{TRANSFER_PREFIX}/**",
                local_dir=args.root,
                token=token,
            )
        except Exception as exc:
            raise SystemExit(_access_error(exc)) from exc
        print("[download] done")
        return

    prefix = TASK_PREFIXES[args.task]
    print(f"[download] prefix={prefix}")
    print("[download] mode=tiny smoke subset")

    api = HfApi(token=token)
    try:
        files = api.list_repo_files(args.repo_id, repo_type="dataset")
    except Exception as exc:
        raise SystemExit(_access_error(exc)) from exc

    hdf5_files = sorted(
        file for file in files
        if file.startswith(prefix + "/") and file.endswith((".hdf5", ".h5"))
    )
    if not hdf5_files:
        raise SystemExit(f"No HDF5 files found under {prefix}")

    grouped: dict[tuple[int, int], list[str]] = defaultdict(list)
    for file in hdf5_files:
        match = re.search(r"_seed(\d+)_task(\d+)\.h(?:df5|5)$", file)
        if match:
            grouped[(int(match.group(2)), int(match.group(1)))].append(file)

    if not grouped:
        raise SystemExit(f"Could not group HDF5 files by seed/task under {prefix}")

    selected_groups = sorted(grouped.items())[: args.episodes]
    selected_files = [file for _, group_files in selected_groups for file in sorted(group_files)]

    print(f"[download] episodes={len(selected_groups)} files={len(selected_files)}")
    args.root.mkdir(parents=True, exist_ok=True)
    for idx, filename in enumerate(selected_files, start=1):
        print(f"[download] {idx}/{len(selected_files)} {filename}")
        try:
            hf_hub_download(
                repo_id=args.repo_id,
                repo_type="dataset",
                filename=filename,
                local_dir=args.root,
                token=token,
            )
        except Exception as exc:
            raise SystemExit(_access_error(exc)) from exc

    print("[download] done")


def _access_error(exc: Exception) -> str:
    return (
        "Could not download RoboMemArena files. Make sure you accepted the gated dataset on Hugging Face "
        "and ran `huggingface-cli login`, or export HF_TOKEN. Original error: " + str(exc)
    )


if __name__ == "__main__":
    main()
