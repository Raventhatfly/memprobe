from memprobe.sampling import frame_indices_for_times, uniform_sample_times
from memprobe.schema import TimeSpan


def test_uniform_sample_includes_span_endpoints() -> None:
    times = uniform_sample_times(TimeSpan(2.0, 4.0), fps=1.0)
    assert times == (2.0, 3.0, 4.0)


def test_uniform_sample_respects_frame_cap() -> None:
    times = uniform_sample_times(TimeSpan(0.0, 10.0), fps=10.0, max_frames=3)
    assert times == (0.0, 5.0, 10.0)
    assert frame_indices_for_times(times, fps=2.0) == (0, 10, 20)
