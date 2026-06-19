# MemProbe: Working Research Ideas

## 1. Research Goal

MemProbe should directly evaluate whether a robot memory system retains and retrieves useful information from past visual observations.

The primary research question is:

```text
Given the same visual history and query point, what information about the past can a robot memory representation recover?
```

The primary benchmark should not conflate memory quality with natural-language understanding, free-form language generation, policy execution, low-level control, or task success. Task success mixes memory with perception, planning, and control.

The original proposal uses language as a diagnostic interface. This remains useful as an optional interface, but the core benchmark should first measure visual memory with formal probes and visual answers. Language alignment should be a separate track.

## 2. Core Benchmark Interface

The preferred core interface is:

```text
visual history + query marker + formal probe type + visual choices -> choice ID
```

Examples:

```text
PROBE_TYPE: PREVIOUS_EVENT
INPUT: history video ending at QUERY
CHOICES: visual clips A, B, C, D
OUTPUT: one choice ID
```

```text
PROBE_TYPE: LAST_VISIBLE_REGION
INPUT: history video, target marker, query image
CHOICES: marked visual regions A, B, C, D
OUTPUT: one choice ID
```

Natural language should only describe the fixed operator or output format. It should not carry episode-specific ground truth.

This design lets the original video supply answer media directly, avoids free-form answer judging, removes the need to name every object or region, weakens text-event summaries, and measures visual retrieval and binding more directly.

### 2.1 Input Contract

The primary track does not assume that a model knows original frame indices.

- `QUERY` is public. Prefer truncating the visual history exactly at the query point, so the current observation is simply the final observation in the supplied history.
- A separate query image may be supplied after the history when needed. In that case, its message position or a visible `Q` marker defines it as QUERY.
- `EVIDENCE` is private in the primary track. Evidence indices and clips are used for generation, auditing, and oracle evaluation, but are not revealed to the evaluated memory model.
- `ANCHOR` is an optional public, non-semantic cue embedded once in the historical stream. It may identify a moment to remember, but the anchored frame is not replayed at answer time.

This creates three distinct protocols:

1. **Self-anchored primary track:** operators such as first, last, previous, initial, and last-visible locate their own evidence in the history.
2. **Cue-marked track:** a neutral `ANCHOR` appears during the history and the model must retain the visual state seen there.
3. **Oracle-evidence track:** the correct evidence media is explicitly supplied at answer time. This is an upper-bound diagnostic, not a memory score.

## 3. What Counts as Memory

A valid MemProbe item should satisfy all of the following:

1. The answer is supported by a localized span in the visual history.
2. The query frame alone is insufficient.
3. File names, task instructions, paths, and public metadata do not reveal the answer.
4. A written formal operator gives one deterministic answer.
5. A human or oracle-evidence model can answer from the evidence.
6. A realistic text-event-summary memory performs poorly.

The core benchmark has two categories:

- temporal or sequential episodic memory;
- spatial visual memory.

Metric time, such as duration in seconds, should be a separate hard track because many current video models do not reliably encode physical time.

## 4. Temporal Memory

Temporal memory should primarily mean event order and state progression, not vague elapsed time.

### 4.1 Previous Event Retrieval

Definition:

```text
previous_event(query) = completed event with the largest end index before QUERY
```

Probe:

```text
Select the visual clip showing the event completed immediately before QUERY.
```

Choices should be clips or thumbnails from the same episode, not text labels such as `pick mug` or `open drawer`.

Reject the item if event boundaries overlap, the previous event is incomplete, or candidates are visually indistinguishable.

### 4.2 Event Order

The model receives shuffled event clips from the observed history and returns their chronological order.

```text
Visual cards: A, B, C, D
Output: B -> D -> A -> C
```

Clips must not contain future observations after QUERY. Strong distractors contain similar actions involving different visual instances, so a generic event caption is insufficient.

### 4.3 First and Last Interaction

The model selects which visual object was manipulated first or last in a specified history interval.

Answers should be object crops or marked instances rather than object names. This tests order together with object-instance binding.

### 4.4 State Transition Memory

The model retrieves the visual state immediately before or after a precisely anchored event.

Candidate states include:

- open versus closed;
- inside versus outside;
- upright versus fallen;
- held versus released;
- object on one marked support region versus another.

The state must not remain directly visible at QUERY, or the item becomes a perception task.

### 4.5 Repetition and Counting

The model counts visually grounded repeated events before QUERY.

Counting should be used carefully because a text event log can solve simple counts. Stronger items require tracking repeated, visually similar interactions involving different instances or containers.

### 4.6 Visual Permutation Tracking

An object is placed under or inside one of several visually similar containers. The containers are moved or shuffled, and the model selects the final container that contains the object.

This is a strong anti-text-memory probe because a generic summary such as `the ball was placed under a bowl and the bowls were shuffled` does not preserve the visual permutation.

These probes may require controlled real-robot or simulation collection because ordinary policy datasets may not contain enough clean permutation events.

### 4.7 Metric-Time Hard Track

Questions such as `what happened three seconds before QUERY?` are scientifically interesting but should not be mixed into the core event-order benchmark.

They require explicit timestamps or disclosed fixed frame rate, no hidden frame dropping or speed changes, a visible QUERY time anchor, and separate reporting.

## 5. Spatial Memory

Spatial probes should recover past visual state without ambiguous names such as `cabinet1`, `cabinet2`, `over there`, or `nearby`.

### 5.1 Last Visible Location

Definition:

```text
last_visible(target, query) = latest observation before QUERY where the target is visibly localized
```

Probe:

```text
Select the marked region corresponding to the target's last visible location before QUERY.
```

Specify the target with a visual crop or marker. Answers should be image regions or crops.

Reject if the target is still visible at QUERY, tracking confidence is low, or the answer lies near a region boundary.

### 5.2 Object-Region Binding

The model remembers which visual object instance occupied a marked region at an anchored point in the history.

Choices should be visually plausible, preferably same-category instances. If only one mug exists, the probe may be solved from object priors rather than memory.

### 5.3 Container or Surface Membership

The model selects which marked container or surface held the target object at the evidence point.

Use visual answer regions instead of semantic labels. Public inputs must not expose paths or instructions such as `place_butter_cabinet2`.

### 5.4 Object-Object Relation

Fine-grained relations such as left, right, above, or below are allowed only with an explicit camera reference and deterministic coordinate rule.

For a fixed camera view:

```text
left_of(A, B)  := center_x(A) < center_x(B) - margin
right_of(A, B) := center_x(A) > center_x(B) + margin
above(A, B)    := center_y(A) < center_y(B) - margin
below(A, B)    := center_y(A) > center_y(B) + margin
```

Reject items near a threshold. Do not use `near`, `next to`, `front`, or `behind` unless a calibrated 3D rule exists.

### 5.5 Initial Layout and Spatial Change

The model binds a current visual object to its initial location, or selects the location from which it moved.

These probes are valid only when the scene changed after the evidence point and the current frame alone cannot reveal the answer.

### 5.6 Occlusion Memory

The model remembers the location or state of an object that later becomes hidden.

The last visible evidence must be clear, and the hidden state must not be trivially inferable from the final layout.

## 6. Removing Ambiguity

Natural-language descriptions are not the source of truth. Every probe first needs a formal operator and private evidence record.

### 6.1 Temporal Rules

Avoid vague terms such as:

```text
before
earlier
recently
just now
a while ago
```

unless they belong to a formally defined fixed template.

Prefer:

- the event with maximum `end_index < query_index`;
- the first completed event in a provided interval;
- the last visible observation before QUERY;
- the chronological order of supplied visual clips.

Models do not need to know original frame numbers. QUERY can be the end of the supplied history or a visible non-semantic marker.

### 6.2 Spatial Rules

Use one camera view per core probe. Front and wrist views should not be mixed in one 2D relation unless the task explicitly tests cross-view association.

Prefer marked image regions, object crops, segmentation masks, fixed-camera image coordinates, robust inside/on relations, and explicit margins and confidence thresholds.

Reject relations that depend on an unnamed viewpoint or unstable perspective.

### 6.3 Visual Anchors Instead of Semantic Names

Do not call two containers `cabinet1` and `cabinet2` and expect the model to infer their mapping.

Use region A/B/C/D, a target crop, a colored instance marker, or candidate image/clip. These markers define the interface without disclosing the answer.

## 7. Ground Truth from Existing Videos

If the output is visual, past video can provide much of the ground truth directly:

- the correct previous-event choice is the actual preceding clip;
- the correct event order is the original clip order;
- the correct past-state choice is a crop from the evidence frame;
- the correct last-visible location is the region containing the tracked object;
- distractors come from other times, objects, or regions in the same episode.

This reduces semantic annotation but does not remove all annotation. The pipeline still needs reliable event boundaries, query points, object tracks or visual correspondences, visibility decisions, ambiguity checks, and human verification.

MemProbe is therefore not a new source-video dataset. It is a reproducible probe layer derived from existing robot videos.

## 8. DROID as a Real-World Source

DROID provides diverse real scenes, camera viewpoints, objects, operators, and trajectories. It is better suited than a small set of simulation templates for visual diversity.

However, DROID is not primarily a dedicated long-horizon benchmark. Many demonstrations are relatively short manipulation episodes. It should be treated as:

- a source of real-world visual diversity;
- a source of action, gripper, camera, and language signals for pre-segmentation;
- a test bed for event-order and spatial probes within episodes;
- not the only source for very long-horizon memory.

Language annotations may help discover or stratify episodes, but must not be exposed to evaluated models when they leak answers.

## 9. Candidate Generation Pipeline

### Stage 1: Canonical Episode Adapter

Normalize each source into:

```json
{
  "episode_uid": "opaque_id",
  "camera_streams": [],
  "timestamps": [],
  "actions": [],
  "gripper_state": [],
  "robot_state": [],
  "private_language_metadata": {},
  "source_provenance": {}
}
```

### Stage 2: Candidate Event Segmentation

For DROID, use action and gripper signals, end-effector motion, and visual change to propose manipulation phases. A VLM may suggest boundaries, but must not define ground truth by itself.

### Stage 3: Visual Tracking and Keyframes

Build candidate object tracks, visibility spans, stable-state frames, and region partitions. Keep confidence scores and camera identity for every relation.

### Stage 4: Formal Probe Generation

Generate probes from deterministic operators over event boundaries, frame order, tracks, and marked regions. Select answer media from the source episode and construct hard visual distractors.

### Stage 5: VLM-Assisted Pre-Annotation

A long-video VLM may propose event boundaries, manipulated objects, evidence spans, and whether media is visually understandable. Its output is a proposal for review, not benchmark truth.

### Stage 6: Automatic Filtering

Reject candidates unless:

- the formal operator has one answer;
- evidence is localized;
- QUERY alone is insufficient;
- visual options are readable and non-duplicated;
- metadata does not leak the answer;
- spatial margins exceed thresholds;
- temporal boundaries are stable;
- the item survives text-memory adversarial evaluation.

### Stage 7: Human Verification

Annotators accept, edit, or reject a prebuilt probe. They verify correctness, evidence necessity and sufficiency, ambiguity, distractor quality, and whether memory is required.

Test and hidden-test splits should be fully human verified.

## 10. Anti-Language-Memory Design

MemProbe should include a realistic language-memory adversary:

```text
history video -> question-independent event/caption summary -> text-only solver -> choice ID
```

Favor items where this baseline is near chance while an oracle visual-evidence model succeeds.

Strong construction patterns include:

- visually similar objects with the same category name;
- episodes with the same caption-level sequence but different visual answers;
- permutation or shuffle events;
- instance-region binding absent from instructions;
- image or clip answers instead of nouns;
- low-salience details omitted by generic summaries.

An unlimited frame-by-frame transcription could encode the video. The intended adversary is a realistic fixed-budget, question-independent event-summary memory, not an unbounded lossless text codec.

## 11. Evaluation Tracks and Baselines

### Primary Visual Memory Probe

The model receives visual history under a fixed frame/token/memory budget, a formal probe type, and visual choices.

### No-Memory Baseline

The model receives only QUERY and choices.

### Recent-Frame Baseline

The model receives the latest `k` frames before QUERY.

### Random-Frame Baseline

The model receives `k` random history frames.

### Oracle-Evidence Baseline

The model receives ground-truth evidence frames or clip. This tests whether the probe is visually answerable.

### Text-Memory Adversary

The model receives a fixed-budget event or caption summary but no visual history.

### Optional Language-Alignment Track

The structured probe is rendered as a natural-language question with text answers. This measures memory-to-language alignment and must be reported separately from core memory quality.

## 12. Metrics

Report:

- overall choice accuracy;
- temporal versus spatial accuracy;
- accuracy versus evidence-query lag;
- accuracy under visual-memory budgets;
- improvement over no-memory;
- oracle-evidence gap;
- visual-memory minus text-memory accuracy gap;
- occlusion-conditioned performance;
- human agreement and rejection rate.

Desired pattern:

```text
oracle visual evidence: high
full visual history: model-dependent
current frame only: low
text event summary: near chance
human agreement: high
```

## 13. Public and Private Representations

Public items use opaque identifiers and contain no source path, task label, event description, or instruction that leaks the answer.

```json
{
  "id": "memprobe_droid_000001",
  "probe_type": "PREVIOUS_EVENT",
  "history_media": "history.mp4",
  "query_media": "query.png",
  "choices": [
    {"id": "A", "media": "clip_A.mp4"},
    {"id": "B", "media": "clip_B.mp4"},
    {"id": "C", "media": "clip_C.mp4"},
    {"id": "D", "media": "clip_D.mp4"}
  ],
  "answer": "B"
}
```

Private provenance retains:

```json
{
  "query_index": 1240,
  "evidence_range": [980, 1110],
  "camera_id": "exterior_1",
  "operator": "argmax(event.end_index < query_index)",
  "source_episode_path": "private",
  "human_verification": "accepted"
}
```

## 14. Initial MVP

The first DROID pilot should be small and measurable:

- sample 100-500 diverse successful episodes;
- use one exterior camera per probe;
- keep wrist video as optional evidence, not for 2D spatial relations;
- generate only four probe families;
- use visual choices wherever possible;
- run current-frame, random-frame, oracle-evidence, and text-memory baselines;
- fully human verify the pilot.

Initial probe families:

1. `PREVIOUS_EVENT`
2. `EVENT_ORDER`
3. `LAST_VISIBLE_REGION`
4. `OBJECT_REGION_BINDING`

Do not begin with metric-time questions, fine-grained 3D relations, or unrestricted free-form answers.

## 15. Open Research Questions

- How should visual choices be encoded for models accepting video, multi-image input, or latent memory tokens?
- How can DROID event boundaries be obtained without turning VLM predictions into ground truth?
- What fixed budget defines a fair text-memory adversary?
- How can visually similar, non-duplicate distractors be generated automatically?
- How much human verification is needed?
- Which controlled real-robot tasks should complement DROID?
- How can visual-memory quality be separated from the downstream decoder's ability to compare choices?

## 16. Current Position

MemProbe should primarily be a visual episodic memory probe layer over existing robot video datasets.

The original language-QA framing remains useful for interpretability and compatibility, but it should not define ground truth or dominate primary evaluation. The benchmark should first determine whether visual memory preserves the right event, object, location, and state. Language alignment is a separate capability.
