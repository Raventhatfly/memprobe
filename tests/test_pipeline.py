import json
from pathlib import Path

from memprobe.adapters import ManifestAdapter
from memprobe.backends import PrecomputedProposalBackend
from memprobe.pipeline import GenerationConfig, generate_dataset


FIXTURE_ROOT = Path(__file__).parents[1] / "examples" / "mvp"


def test_pipeline_generates_visual_temporal_probes(tmp_path: Path) -> None:
    result = generate_dataset(
        ManifestAdapter(FIXTURE_ROOT / "episode.json"),
        PrecomputedProposalBackend(FIXTURE_ROOT / "events.json"),
        tmp_path,
        GenerationConfig(require_verified=True),
    )

    assert result.episodes == 1
    assert result.event_proposals == 6
    assert result.probes == 4
    assert result.release_ready_probes == 0

    private_items = _read_jsonl(result.private_path)
    public_items = _read_jsonl(result.public_path)
    assert {item["probe_type"] for item in private_items} == {"PREVIOUS_EVENT", "EVENT_ORDER"}
    assert all("evidence" in item and "provenance" in item for item in private_items)
    assert all("evidence" not in item and "provenance" not in item for item in public_items)
    assert all("source_event_id" not in choice for item in public_items for choice in item["choices"])
    assert all(not item["history"]["uri"].startswith("/") for item in public_items)
    assert all(not choice["media"]["uri"].startswith("/") for item in public_items for choice in item["choices"])
    assert all("span" not in item["history"] for item in public_items)
    assert all("span" not in choice["media"] for item in public_items for choice in item["choices"])
    assert all("query_time_s" not in item and "validation" not in item for item in public_items)
    assert "start_s" not in result.public_path.read_text()
    assert all(item["validation"]["automatic_pass"] for item in private_items)
    assert all(not item["validation"]["release_ready"] for item in private_items)


def test_pipeline_is_deterministic(tmp_path: Path) -> None:
    adapter = ManifestAdapter(FIXTURE_ROOT / "episode.json")
    backend = PrecomputedProposalBackend(FIXTURE_ROOT / "events.json")
    first = generate_dataset(adapter, backend, tmp_path / "first", GenerationConfig(seed=7))
    second = generate_dataset(adapter, backend, tmp_path / "second", GenerationConfig(seed=7))
    assert first.public_path.read_text() == second.public_path.read_text()


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
