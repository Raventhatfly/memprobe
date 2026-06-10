# memprobe

Prototype code for building MemProbe-style robotic memory QA from HDF5 robot trajectories.

Current RoboMemArena outputs are candidate QA items for smoke testing and human audit. They are not final benchmark items until simulator-state facts, ambiguity filters, leakage stripping, and verification are added.

The current first pass targets RoboMemArena under:

```bash
/n/netscratch/hankyang_lab/Lab/felix/dataset/robomemarena
```

It follows the proposal's conservative rule: derive facts deterministically from trajectory provenance and HDF5 metadata, then render controlled multiple-choice questions with canonical answers and evidence records.

## Generate RoboMemArena QA

Use the existing `fluxvla` conda environment:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
/n/holylabs/LABS/hankyang_lab/Lab/felix/.conda/envs/fluxvla/bin/python \
  scripts/generate_robomemarena_qa.py \
  --root /n/netscratch/hankyang_lab/Lab/felix/dataset/robomemarena \
  --output outputs/robomemarena_memprobe_prototype.jsonl
```

For a quick smoke test:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
/n/holylabs/LABS/hankyang_lab/Lab/felix/.conda/envs/fluxvla/bin/python \
  scripts/generate_robomemarena_qa.py --max-episodes 5 \
  --output outputs/robomemarena_memprobe_smoke.jsonl
```

## Qwen-VL Baseline Skeleton

The evaluation script formats MemProbe multiple-choice prompts for an open Qwen-VL style model:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
/n/holylabs/LABS/hankyang_lab/Lab/felix/.conda/envs/fluxvla/bin/python \
  scripts/eval_qwen_vl_mcq.py \
  --input outputs/robomemarena_memprobe_prototype.jsonl \
  --dry-run
```

The GPU inference path is intentionally not auto-run yet; the next step is wiring extracted evidence/current frames into the Qwen-VL message format.

## Convert HDF5 to Videos

Smoke convert the first 10 RoboMemArena HDF5 files from one subtask directory:

```bash
PYTHONDONTWRITEBYTECODE=1 \
/n/holylabs/LABS/hankyang_lab/Lab/felix/.conda/envs/fluxvla/bin/python \
  scripts/hdf5_to_video.py \
  --input-root /n/netscratch/hankyang_lab/Lab/felix/dataset/robomemarena/Multi-Transferring/18_chocolate_butter_cabinet_dataset/subtask_data \
  --limit 10 \
  --output-dir outputs/videos_smoke \
  --fps 20 \
  --overwrite
```

By default the script reads `/data/demo_0/obs/agentview_rgb`. Use `--dataset /data/demo_0/obs/eye_in_hand_rgb` for wrist-camera videos.
