from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class DemoMetadata:
    demo_id: str
    num_samples: int | None
    language_instruction: str | None
    keyframe_indices: tuple[int, ...]
    datasets: dict[str, tuple[int, ...]]


def read_demo_metadata(path: Path, demo_id: str = "demo_0") -> DemoMetadata:
    """Read lightweight HDF5 metadata without loading image arrays."""
    try:
        return _read_with_h5py(path, demo_id)
    except ModuleNotFoundError:
        return _read_with_h5dump(path, demo_id)


def iter_hdf5_files(root: Path) -> Iterable[Path]:
    yield from sorted(root.rglob("*.hdf5"))
    yield from sorted(root.rglob("*.h5"))


def _read_with_h5py(path: Path, demo_id: str) -> DemoMetadata:
    import h5py

    datasets: dict[str, tuple[int, ...]] = {}
    with h5py.File(path, "r") as f:
        demo = f[f"data/{demo_id}"]
        attrs = dict(demo.attrs)
        for key, value in demo.items():
            if hasattr(value, "shape"):
                datasets[key] = tuple(int(x) for x in value.shape)
            elif key == "obs":
                for obs_key, obs_value in value.items():
                    datasets[f"obs/{obs_key}"] = tuple(int(x) for x in obs_value.shape)

        keyframes: tuple[int, ...] = ()
        if "keyframe_indices" in demo:
            keyframes = tuple(int(x) for x in demo["keyframe_indices"][()])

    return DemoMetadata(
        demo_id=demo_id,
        num_samples=_maybe_int(attrs.get("num_samples")),
        language_instruction=_maybe_text(attrs.get("language_instruction")),
        keyframe_indices=keyframes,
        datasets=datasets,
    )


def _read_with_h5dump(path: Path, demo_id: str) -> DemoMetadata:
    header = _run_h5dump("-H", path)
    datasets = _parse_dataset_shapes(header, f"/data/{demo_id}/")
    instruction = _parse_h5dump_scalar(
        _run_h5dump("-a", f"/data/{demo_id}/language_instruction", path),
    )
    num_samples_text = _parse_h5dump_scalar(
        _run_h5dump("-a", f"/data/{demo_id}/num_samples", path),
    )
    keyframes_text = _run_h5dump("-d", f"/data/{demo_id}/keyframe_indices", path)

    return DemoMetadata(
        demo_id=demo_id,
        num_samples=_maybe_int(num_samples_text),
        language_instruction=instruction,
        keyframe_indices=tuple(int(x) for x in re.findall(r"\(\d+\):\s*(-?\d+)", keyframes_text)),
        datasets=datasets,
    )


def _run_h5dump(*args: object) -> str:
    cmd = ["h5dump", *(str(arg) for arg in args)]
    completed = subprocess.run(cmd, check=False, text=True, capture_output=True)
    if completed.returncode != 0:
        return ""
    return completed.stdout


def _parse_dataset_shapes(header: str, prefix: str) -> dict[str, tuple[int, ...]]:
    datasets: dict[str, tuple[int, ...]] = {}
    stack: list[str] = []
    pending: str | None = None
    for raw in header.splitlines():
        line = raw.strip()
        group = re.match(r'GROUP "([^"]+)"', line)
        if group:
            name = group.group(1)
            if name not in {"/", "data"}:
                stack.append(name)
            continue
        if line == "}":
            if stack:
                stack.pop()
            continue
        dataset = re.match(r'DATASET "([^"]+)"', line)
        if dataset:
            pending = "/".join([*stack, dataset.group(1)])
            continue
        shape = re.search(r"SIMPLE \{ \( ([^)]+) \)", line)
        if pending and shape:
            full = "/" + pending
            if full.startswith(prefix):
                rel = full[len(prefix) :]
                dims = tuple(int(x.strip()) for x in shape.group(1).split(",") if x.strip())
                datasets[rel] = dims
            pending = None
    return datasets


def _parse_h5dump_scalar(text: str) -> str | None:
    match = re.search(r"\(0\):\s*(.+)", text)
    if not match:
        return None
    value = match.group(1).strip()
    if value.startswith('"') and value.endswith('"'):
        value = value[1:-1]
    return value


def _maybe_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _maybe_text(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)
