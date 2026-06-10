#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import imageio.v2 as imageio
import numpy as np


DEFAULT_DATASET = "/data/demo_0/obs/agentview_rgb"


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert RGB frames in robot HDF5 files to videos.")
    parser.add_argument("inputs", nargs="*", type=Path, help="HDF5 files to convert.")
    parser.add_argument("--input-root", type=Path, default=None, help="Root to scan for HDF5 files.")
    parser.add_argument("--limit", type=int, default=None, help="Convert only the first N files after sorting.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/videos"))
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help=f"HDF5 RGB dataset path. Default: {DEFAULT_DATASET}")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    inputs = list(args.inputs)
    if args.input_root is not None:
        inputs.extend(sorted(args.input_root.rglob("*.hdf5")))
        inputs.extend(sorted(args.input_root.rglob("*.h5")))
    inputs = [path for path in inputs if path.is_file()]
    if args.limit is not None:
        inputs = inputs[: args.limit]
    if not inputs:
        raise SystemExit("No input HDF5 files found.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for input_path in inputs:
        output_path = args.output_dir / f"{input_path.stem}.mp4"
        if output_path.exists() and not args.overwrite:
            print(f"skip existing: {output_path}")
            continue
        convert_one(input_path, output_path, args.dataset, fps=args.fps, max_frames=args.max_frames)
        print(f"wrote: {output_path}")


def convert_one(input_path: Path, output_path: Path, dataset: str, fps: int, max_frames: int | None) -> None:
    with h5py.File(input_path, "r") as f:
        if dataset not in f:
            raise KeyError(f"{dataset} not found in {input_path}")
        frames = f[dataset]
        n_frames = int(frames.shape[0])
        if max_frames is not None:
            n_frames = min(n_frames, max_frames)

        with imageio.get_writer(output_path, fps=fps, codec="libx264", quality=8, macro_block_size=1) as writer:
            for idx in range(n_frames):
                writer.append_data(_as_uint8_rgb(frames[idx]))


def _as_uint8_rgb(frame: np.ndarray) -> np.ndarray:
    arr = np.asarray(frame)
    if arr.ndim != 3:
        raise ValueError(f"expected HWC or CHW RGB frame, got shape {arr.shape}")
    if arr.shape[0] in {1, 3, 4} and arr.shape[-1] not in {1, 3, 4}:
        arr = np.moveaxis(arr, 0, -1)
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


if __name__ == "__main__":
    main()
