from pathlib import Path

from memprobe.materialize import _ffmpeg_command


def test_ffmpeg_command_uses_exact_span() -> None:
    command = _ffmpeg_command(
        "ffmpeg",
        {
            "uri": "/private/source.mp4",
            "span": {"start_s": 2.5, "end_s": 4.0},
        },
        Path("out.mp4"),
        overwrite=False,
    )
    assert command[command.index("-ss") + 1] == "2.500000"
    assert command[command.index("-t") + 1] == "1.500000"
    assert command[-2] == "-n"
    assert command[-1] == "out.mp4"
