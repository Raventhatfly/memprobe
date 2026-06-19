from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Iterable


def materialize_probe_media(
    private_jsonl: Path,
    output_dir: Path,
    ffmpeg_bin: str = "ffmpeg",
    overwrite: bool = False,
    dry_run: bool = False,
) -> tuple[int, int]:
    resolved_ffmpeg = ffmpeg_bin if dry_run else _resolve_ffmpeg(ffmpeg_bin)

    commands = []
    for item in _iter_jsonl(private_jsonl):
        probe_id = str(item["id"])
        probe_dir = output_dir / "media" / probe_id
        history = item["history"]
        commands.append((_ffmpeg_command(resolved_ffmpeg, history, probe_dir / "history.mp4", overwrite), probe_dir))
        for choice in item["choices"]:
            commands.append(
                (
                    _ffmpeg_command(
                        resolved_ffmpeg,
                        choice["media"],
                        probe_dir / f"choice_{choice['id']}.mp4",
                        overwrite,
                    ),
                    probe_dir,
                )
            )

    completed = 0
    skipped = 0
    for command, parent in commands:
        destination = Path(command[-1])
        if destination.exists() and not overwrite:
            skipped += 1
            continue
        if dry_run:
            print(" ".join(_shell_quote(part) for part in command))
            completed += 1
            continue
        parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(command, check=True)
        completed += 1
    return completed, skipped


def _resolve_ffmpeg(ffmpeg_bin: str) -> str:
    executable = shutil.which(ffmpeg_bin)
    if executable is not None:
        return executable
    if ffmpeg_bin == "ffmpeg":
        try:
            import imageio_ffmpeg
        except ImportError as exc:
            raise RuntimeError(
                "ffmpeg was not found; install ffmpeg or the imageio-ffmpeg package"
            ) from exc
        return imageio_ffmpeg.get_ffmpeg_exe()
    raise RuntimeError(f"ffmpeg executable not found: {ffmpeg_bin}")


def _ffmpeg_command(ffmpeg_bin: str, media: dict[str, Any], output: Path, overwrite: bool) -> list[str]:
    span = media["span"]
    start_s = float(span["start_s"])
    end_s = float(span["end_s"])
    if start_s < 0 or end_s <= start_s:
        raise ValueError(f"invalid media span: {span}")
    command = [
        ffmpeg_bin,
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{start_s:.6f}",
        "-i",
        str(media["uri"]),
        "-t",
        f"{end_s - start_s:.6f}",
        "-map",
        "0:v:0",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "18",
        "-movflags",
        "+faststart",
        "-y" if overwrite else "-n",
        str(output),
    ]
    return command


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on {path}:{line_number}: {exc}") from exc


def _shell_quote(value: str) -> str:
    import shlex

    return shlex.quote(value)
