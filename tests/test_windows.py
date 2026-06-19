from memprobe.schema import CanonicalEpisode, CameraStream, ScalarSignal, TimeSpan
from memprobe.windows import WindowProposal, merge_windows, propose_signal_windows


def test_signal_transitions_create_windows() -> None:
    episode = CanonicalEpisode(
        episode_uid="ep",
        duration_s=5.0,
        streams=(CameraStream(camera_id="cam", uri="video.mp4", duration_s=5.0),),
        signals=(
            ScalarSignal(
                name="gripper_state",
                timestamps_s=(0.0, 1.0, 2.0, 3.0),
                values=(0.0, 1.0, 1.0, 0.0),
            ),
        ),
    )
    windows = propose_signal_windows(episode, padding_s=0.5)
    assert [window.span for window in windows] == [TimeSpan(0.5, 1.5), TimeSpan(2.5, 3.5)]


def test_merge_windows_unions_sources() -> None:
    merged = merge_windows(
        [
            WindowProposal("a", "ep", TimeSpan(1.0, 2.0), ("signal",), 0.7),
            WindowProposal("b", "ep", TimeSpan(2.1, 3.0), ("vlm",), 0.9),
        ],
        max_gap_s=0.2,
    )
    assert len(merged) == 1
    assert merged[0].span == TimeSpan(1.0, 3.0)
    assert merged[0].sources == ("signal", "vlm")
    assert merged[0].confidence == 0.9
