# MemProbe

MemProbe generates dataset-independent visual episodic-memory probes from robot
videos. The primary interface is:

```text
visual history -> query after history -> formal probe + visual choices -> choice ID
```

Natural-language task names and file names are not ground truth. VLMs propose
event intervals; deterministic operators and human verification define answers.

## Current MVP

Implemented:

- canonical JSON/JSONL episode manifests;
- synchronized scalar signals in the canonical schema;
- gripper/motion candidate-window proposals;
- precomputed and local Qwen3-VL event-proposal backends;
- event-state graph construction;
- `PREVIOUS_EVENT` visual-choice probes;
- `EVENT_ORDER` visual-card ordering probes;
- private/public provenance separation;
- ffmpeg materialization of history and choice clips;
- deterministic generation and focused tests.

Not yet implemented:

- dense L1/L2 Qwen refinement calls;
- Molmo2 object tracking and spatial probe generators;
- oracle/current-frame/text-memory model gates;
- human review UI;
- DROID RLDS adapter.

Generated items remain review candidates. The code never marks an item release
ready while those gates are pending.

## Install

```bash
python -m pip install -e '.[dev]'
```

For local Qwen3-VL annotation:

```bash
python -m pip install -e '.[qwen]'
scripts/download_vlm_models.sh --profile core
```

## One-Command Smoke Test

```bash
scripts/smoke_v1.sh
```

This uses a synthetic canonical episode plus verified fixture events. It does
not load a VLM or require a real video.

Equivalent command:

```bash
PYTHONPATH=src python -m memprobe generate \
  --manifest examples/mvp/episode.json \
  --proposals examples/mvp/events.json \
  --verified-only \
  --output-dir outputs/smoke_v1
```

Expected outputs:

```text
outputs/smoke_v1/probes.private.jsonl
outputs/smoke_v1/probes.public.jsonl
outputs/smoke_v1/candidate_windows.json
```

## Canonical Episode Manifest

```json
{
  "schema_version": "memprobe.v1",
  "episode_uid": "opaque_episode_id",
  "duration_s": 90.0,
  "streams": [
    {
      "camera_id": "cam_0",
      "uri": "/private/path/video.mp4",
      "duration_s": 90.0,
      "fps": 30.0
    }
  ],
  "signals": [
    {
      "name": "gripper_state",
      "timestamps_s": [0.0, 0.1, 0.2],
      "values": [0.0, 0.0, 1.0]
    }
  ],
  "private_metadata": {},
  "source_provenance": {}
}
```

Validate it before generation:

```bash
PYTHONPATH=src python -m memprobe inspect --manifest episode.json
```

Relative video paths are resolved relative to the manifest file.

## Generate from Verified or Edited Events

Event proposal JSON:

```json
{
  "events": [
    {
      "episode_uid": "opaque_episode_id",
      "camera_id": "cam_0",
      "start_s": 12.1,
      "peak_s": 12.8,
      "end_s": 13.4,
      "event_type": "contact_change",
      "confidence": 0.95,
      "source": "human_review",
      "verification_status": "verified"
    }
  ]
}
```

Generate probes:

```bash
PYTHONPATH=src python -m memprobe generate \
  --manifest episode.json \
  --proposals events.json \
  --verified-only \
  --output-dir outputs/run_001
```

## Generate Qwen3-VL Proposals

The model is local-only by default so an accidental cache miss does not trigger
a large download during generation:

```bash
PYTHONPATH=src python -m memprobe generate \
  --manifest episode.json \
  --qwen-model Qwen/Qwen3-VL-8B-Instruct \
  --overview-fps 0.5 \
  --output-dir outputs/qwen_draft
```

Use `--allow-model-download` only when implicit Hugging Face downloads are
intended. Qwen-generated events have `verification_status=proposed` and all
derived probes require review.

## Materialize Visual Media

After reviewing private event/probe records:

```bash
PYTHONPATH=src python -m memprobe materialize \
  --private-jsonl outputs/run_001/probes.private.jsonl \
  --output-dir outputs/run_001
```

This writes:

```text
media/<probe_id>/history.mp4
media/<probe_id>/choice_A.mp4
media/<probe_id>/choice_B.mp4
...
```

The materializer uses a system `ffmpeg` when available and otherwise falls back
to the executable distributed by `imageio-ffmpeg`.

Public JSON uses only these opaque relative paths. Absolute source paths,
evidence spans, and source event IDs remain private.

## Tests

```bash
PYTHONPATH=src python -m pytest -q
```

## Design

- [Research direction](ideas/main.md)
- [Probe catalog](ideas/probe_question_catalog.md)
- [Open-source VLM generation pipeline](ideas/open_source_vlm_generation_pipeline.md)
