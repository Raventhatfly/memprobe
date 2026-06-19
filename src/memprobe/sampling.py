from __future__ import annotations

import math
from typing import Optional

from .schema import TimeSpan


def uniform_sample_times(
    span: TimeSpan,
    fps: float,
    max_frames: Optional[int] = None,
) -> tuple[float, ...]:
    """Return stable timestamps including both ends when possible."""
    if fps <= 0:
        raise ValueError("fps must be positive")
    if max_frames is not None and max_frames <= 0:
        raise ValueError("max_frames must be positive")
    if span.duration_s == 0:
        return (span.start_s,)

    count = int(math.floor(span.duration_s * fps)) + 1
    count = max(2, count)
    if max_frames is not None:
        count = min(count, max_frames)
    if count == 1:
        return (span.start_s,)
    step = span.duration_s / (count - 1)
    return tuple(span.start_s + index * step for index in range(count))


def frame_indices_for_times(timestamps_s: tuple[float, ...], fps: float) -> tuple[int, ...]:
    if fps <= 0:
        raise ValueError("fps must be positive")
    return tuple(max(0, int(round(timestamp * fps))) for timestamp in timestamps_s)
