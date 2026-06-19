import pytest

from memprobe.backends import _extract_event_records


def test_extract_event_records_accepts_fenced_surrounding_text() -> None:
    records = _extract_event_records(
        '```json\n{"events": [{"start_s": 1, "end_s": 2, "event_type": "contact_change"}]}\n```'
    )
    assert records == [{"start_s": 1, "end_s": 2, "event_type": "contact_change"}]


def test_extract_event_records_rejects_missing_events_array() -> None:
    with pytest.raises(ValueError, match="events array"):
        _extract_event_records('{"result": []}')
