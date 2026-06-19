# MemProbe Visual Probe Question Catalog

This document lists candidate visual-memory probe templates. These are not free-form language QA items. Each prompt assumes a formal operator, visual history, a visible `QUERY` anchor, and visual choices labeled `A-D`.

## Notation

- `QUERY`: the public query point. Prefer ending the supplied history exactly at QUERY, so QUERY is the final observation and no frame-number understanding is required.
- `ANCHOR`: an optional public, non-semantic cue shown once inside the historical stream. The anchored frame is not replayed at answer time.
- `EVIDENCE`: private ground-truth frames or clips used for generation, auditing, and the oracle baseline. EVIDENCE is not shown in the primary memory track.
- `TARGET`: an object specified by a crop, mask, or visual marker rather than an object name.
- `A-D`: image regions, object crops, frames, or video clips.
- `history`: only observations available before or at `QUERY`; it never contains future observations.

The answer is normally one choice ID or an ordering of choice IDs. Episode-specific ground truth must come from private frame indices, event boundaries, tracks, and relation rules, not from the wording.

## Input Protocols

### Primary Self-Anchored Track

The history is truncated at QUERY. Operators such as `previous`, `first`, `last`, `initial`, and `last_visible` locate the relevant moment without exposing EVIDENCE.

### Cue-Marked Track

A neutral `ANCHOR` appears once during the history. The model sees it while encoding the video, but the anchored image is not separately supplied during answering.

### Oracle-Evidence Track

EVIDENCE is supplied explicitly at answer time. Questions in this track diagnose visual answerability and do not count as primary memory evaluation.

# Temporal Memory

## T1. Previous Event Retrieval

Formal operator:

```text
argmax(event.end_index) subject to event.end_index < query_index
```

1. Select the choice clip showing the completed manipulation event whose end is closest to, but earlier than, `QUERY`.
2. Which event card belongs immediately before `QUERY` on the episode timeline?
3. Among clips `A-D`, select the latest fully completed event in the history ending at `QUERY`.
4. Select the clip for which no other completed manipulation event occurs between its final frame and `QUERY`.
5. Which choice clip should fill the empty timeline slot directly preceding `QUERY`?
6. The choices come from one episode. Select the clip with the greatest end index smaller than the query index.
7. Which clip shows the robot's final completed interaction before the visual state marked `QUERY`?
8. Select the event clip adjacent to the left side of `QUERY` in chronological order.
9. Which choice ends at the last event boundary observed before `QUERY`?
10. Ignoring incomplete motion at `QUERY`, which clip shows the most recently completed event?

## T2. Event Order

Formal operator:

```text
sort(choice_clips, key=clip.start_index)
```

1. Return the chronological order of visual event cards `A-D`.
2. Which offered sequence places all four choice clips in increasing start-index order?
3. Arrange the three event clips from earliest to latest in the history.
4. Select the ordering that matches the original episode timeline before `QUERY`.
5. Which sequence correctly orders the clips by their first visible robot-object contact?
6. Arrange the choices by the time at which each manipulation event was completed.
7. Which candidate sequence preserves the observed order of the four visual state transitions?
8. Two clips are already placed as `A -> D`. Select the ordering of `B` and `C` that reconstructs the full timeline.
9. Select the event card that belongs between visual clips `B` and `D` in the original history.
10. The displayed cards were shuffled. Return the permutation that restores their original visual order.

## T3. First Interaction Memory

Formal operator:

```text
object(event with minimum start_index in the specified interval)
```

1. Which marked object was manipulated first in the history interval ending at `QUERY`?
2. Select the object crop corresponding to the robot's first completed interaction.
3. Which marked container was the first one touched by the robot in the supplied interval?
4. Select the object instance involved in the earliest grasp event.
5. Which choice object began moving with the robot before any other choice object?
6. Select the visual instance associated with the first open-or-close interaction.
7. Which marked object was released first in the episode segment?
8. Select the object whose first contact with the gripper has the smallest frame index.
9. Among objects `A-D`, which one participated in the earliest completed manipulation?
10. Which object card should be attached to the first event node in the visual timeline?

## T4. Last Interaction Memory

Formal operator:

```text
object(event with maximum end_index before query_index)
```

1. Which marked object was manipulated most recently before `QUERY`?
2. Select the object crop associated with the final completed interaction before `QUERY`.
3. Which marked container was the last one touched before the query point?
4. Select the object instance involved in the latest completed grasp event.
5. Which choice object stopped moving with the robot most recently before `QUERY`?
6. Select the visual instance associated with the last completed open-or-close interaction.
7. Which marked object was released last in the observed history?
8. Select the object whose final gripper contact has the greatest index below the query index.
9. Among objects `A-D`, which one participated in the latest completed manipulation?
10. Which object card belongs to the event node directly preceding `QUERY`?

## T5. State Transition Memory

Formal operators:

```text
state_before(event)
state_after(event)
delta(event) = state_after(event) - state_before(event)
```

1. Which choice image shows the scene state immediately before event clip `ANCHOR` begins?
2. Which choice image shows the stable scene state immediately after event clip `ANCHOR` ends?
3. Select the choice showing the target container's state before the robot interacted with it.
4. Select the choice showing the target container's state after the interaction was completed.
5. Which marked object changed from outside to inside a container during the anchored event?
6. Which choice shows the target object's orientation before the marked transition?
7. Which choice shows the target object's orientation after the marked transition?
8. Select the visual state that existed between event clips `B` and `C`.
9. Which choice shows the object-support relation created by the marked event?
10. Which visual difference correctly represents the transition from the pre-event state to the post-event state?

## T6. Repetition and Counting

Formal operator:

```text
count(events satisfying a fixed visual predicate before query_index)
```

1. How many completed grasp events occur before `QUERY`?
2. How many times does the robot release an object onto one of the marked support regions?
3. How many distinct marked objects are manipulated before `QUERY`?
4. How many completed open events occur in the supplied history?
5. How many times does `TARGET` visibly change support regions before `QUERY`?
6. How many choice containers receive an object during the history?
7. How many complete pick-and-place cycles occur before `QUERY`?
8. How many times does the gripper make visible contact with `TARGET`?
9. How many marked objects cross the specified region boundary before `QUERY`?
10. How many completed state changes of the displayed type occur in the observed interval?

## T7. Visual Permutation Tracking

Formal operator:

```text
track(container identity through the observed permutation)
```

1. A ball is placed under one marked bowl. After the bowls are shuffled, which final bowl contains the ball?
2. `TARGET` is placed inside one of three visually similar containers. Select that container after all containers move.
3. Which final cup corresponds to the cup that covered `TARGET` in the initial anchored state?
4. Three marked boxes exchange positions. Select the final box that started in region `A`.
5. Which final container inherits the hidden content of the container marked at the beginning?
6. Two identical-looking trays cross paths. Select the tray that originally held `TARGET`.
7. After the marked containers are rotated and reordered, which choice preserves the tracked container identity?
8. A hidden object remains with one moving container. Select its final visual instance.
9. Which final marked region contains the container that occupied the highlighted initial region?
10. Select the final object-container pairing consistent with the full visual shuffle history.

## T8. Metric-Time Hard Track

Required conditions:

- timestamps are explicit;
- frame sampling and playback speed are fixed;
- `QUERY` has a visible timestamp;
- results are reported separately from event-order memory.

1. Which choice clip contains the observation exactly three seconds before `QUERY`?
2. Which marked object moves during the interval from `QUERY - 5s` to `QUERY - 3s`?
3. Which choice image is closest to two seconds before the query timestamp?
4. Which event has the longest measured duration between its marked start and end?
5. How many seconds elapse between the first visible grasp and the corresponding release?
6. Which event begins within the one-second interval immediately preceding `QUERY`?
7. Select the object that remains stationary for the longest marked time interval.
8. Which pair of event clips has the shorter elapsed gap between them?
9. At timestamp `t`, which marked spatial state is visible in the specified camera?
10. Which event occupies the largest fraction of the displayed ten-second interval?

# Spatial Memory

## S1. Last Visible Location

Formal operator:

```text
location(target, max(frame_index < query_index where target is visible))
```

1. Select the marked region containing `TARGET` at its last visible observation before `QUERY`.
2. Which region `A-D` contains the final visible pixels of `TARGET` in the history?
3. Immediately before `TARGET` becomes occluded, which marked region contains it?
4. Select the region associated with the latest valid target track before `QUERY`.
5. Which region contains `TARGET` in its last frame above the visibility-confidence threshold?
6. Select the camera region where `TARGET` is most recently visible before leaving the view.
7. Which marked support region contains `TARGET` at its final visible stable state?
8. Select the region containing the final unoccluded observation of the target crop.
9. Which region should be linked to the last visible node of the target track?
10. Before the query point, where does `TARGET` appear for the final visually verifiable time?

## S2. Oracle Evidence Localization

This is an oracle/perception diagnostic, not a core memory probe, because EVIDENCE is supplied again at answer time.

Formal operator:

```text
location(target, evidence_index, camera_id)
```

1. In the frame marked `EVIDENCE`, which region contains `TARGET`?
2. At the supplied evidence image, select the support region under the target object.
3. In camera `CAM-1` at `EVIDENCE`, which marked region contains the target center?
4. Select the region overlapping the target mask in the evidence frame.
5. Which marked container contains `TARGET` at the exact evidence point?
6. At `EVIDENCE`, which region contains most of the target bounding box?
7. Select the workspace zone assigned to `TARGET` in the displayed evidence image.
8. Which choice crop contains `TARGET` at the marked evidence point?
9. At the evidence index, which marked surface directly supports `TARGET`?
10. Select the region connected to `TARGET` in the evidence-frame spatial graph.

## S3. Object-Region Binding

Formal operator:

```text
object occupying region R at the public ANCHOR in the history
```

1. Which object crop corresponds to the object occupying marked region `B` when `ANCHOR` appeared in the history?
2. Select the visual object instance that was inside the highlighted region at the anchor point.
3. Which marked object was supported by region `C` at `ANCHOR`?
4. Select the object track associated with region `A` at the specified historical state.
5. Which choice object occupied the highlighted container before `QUERY`?
6. Select the same-category instance bound to the marked workspace zone at `ANCHOR`.
7. Which object crop belongs to the region-object pair shown in the historical scene graph?
8. At the anchor point, which marked object overlaps the selected region mask?
9. Select the object that remained in region `D` during the displayed stable interval.
10. Which visual instance should fill the missing object node connected to the marked region?

## S4. Object-Object Relation at an Anchored Frame

Formal operators use a fixed camera and margin:

```text
left_of(A, B), right_of(A, B), above(A, B), below(A, B)
```

1. In `CAM-1` when `ANCHOR` appeared, was the yellow-marked object left, right, above, or below the blue-marked object?
2. Select the image-plane relation between object crops `A` and `B` at the anchor point.
3. Which relation holds between the target center and the reference center with the stated pixel margin?
4. At `ANCHOR`, which marked object lies strictly left of `TARGET` in the specified camera?
5. At the anchor point, which marked object lies strictly above `TARGET`?
6. Which choice image preserves the historical left-right relation between the two target objects?
7. Select the relation that holds after excluding positions inside the ambiguity margin.
8. In the fixed front view, which object is horizontally closer to the left image boundary?
9. At `ANCHOR`, which ordered object pair satisfies the displayed spatial predicate?
10. Which relation label matches the two segmentation-mask centers in the anchored frame?

## S5. Container or Surface Membership

Formal operators:

```text
inside(target, region)
on(target, support)
```

1. Which marked container contained `TARGET` when `ANCHOR` appeared in the history?
2. Select the marked surface supporting `TARGET` in the anchored historical state.
3. Which region contained the target-mask center at `ANCHOR`?
4. Select the container whose interior overlaps the target mask under the membership rule.
5. Which marked support region is directly below `TARGET` at the anchor point?
6. Select the visual container into which `TARGET` was placed before `QUERY`.
7. Which region held `TARGET` immediately after the marked placement event?
8. Select the support surface from which `TARGET` was removed during the anchored event.
9. Which choice image shows the correct target-container membership from the historical state?
10. Select the region connected to `TARGET` by the private `inside` or `on` predicate.

## S6. Initial Layout Memory

Formal operator:

```text
location or relation at the first valid stable-state snapshot
```

1. Which marked region contains `TARGET` in the episode's first valid stable state?
2. Select the initial support surface of the object shown in the query crop.
3. Which object crop occupied region `A` at the beginning of the observed episode?
4. Select the initial object-container pairing from the visual choices.
5. Before any manipulation begins, which marked object is inside the highlighted region?
6. Which choice image matches the initial layout of the marked objects?
7. Select the region containing `TARGET` before its first interaction event.
8. Which historical relation between the marked objects holds in the initial stable frame?
9. Select the initial location of the object currently shown at `QUERY`.
10. Which candidate scene crop reconstructs the target's initial workspace placement?

## S7. Spatial Change Memory

Formal operator:

```text
delta_location(target, anchor_state, query_state)
```

1. Which marked object changes regions between `ANCHOR` and `QUERY`?
2. Select the region from which `TARGET` moved before reaching its query-state region.
3. Which choice image shows `TARGET` before the spatial transition?
4. Select the object whose support relation differs between the two anchored states.
5. Which marked object leaves its original container before `QUERY`?
6. Select the correct source-to-target region pair for the tracked object.
7. Which candidate visual difference represents the observed location change?
8. Select the marked region that becomes empty after the target moves.
9. Which region gains the target object between the anchored state and query state?
10. Select the object-region edge that is removed from the spatial graph after the manipulation.

## S8. Occlusion Memory

Formal operator:

```text
recover the last verified state before a target becomes hidden
```

1. Which marked region contains `TARGET` immediately before it becomes occluded?
2. Select the final visible target crop before the object disappears from view.
3. Which container hides `TARGET` at the end of the marked interaction?
4. Before the target leaves the camera view, which support region contains it?
5. Select the last verified object-region binding before occlusion begins.
6. Which marked object becomes hidden inside the highlighted container?
7. Select the region associated with the target's final unoccluded track point.
8. Which choice image shows the target state immediately before visual disappearance?
9. After `TARGET` is covered, which marked container should retain its hidden-state association?
10. Select the historical visual patch that provides the last sufficient evidence for the hidden target's location.

# Validation Rules for All Categories

Every generated instance should be rejected unless:

1. Private provenance defines one deterministic answer.
2. A human can answer from the intended visual evidence.
3. The query input alone is insufficient.
4. File names, instructions, and metadata do not leak the answer.
5. Visual choices are readable, plausible, and non-duplicated.
6. The oracle-evidence baseline succeeds.
7. A fixed-budget, question-independent text-memory baseline is near chance.
8. Temporal boundaries and spatial thresholds are outside ambiguity margins.

