# Semantic Teacher Transfer 2026-06-24

## Purpose

The latest geometry Patch route produced better structural ownership candidates,
but direct SAM semantic voting on those Patch objects had poor semantic quality.
This document records the recovery test that uses the earlier validated
structure-prior + VLM/MASK result as a teacher field.

## Teacher Route

Teacher artifact:

- `server_parking_priority_s10/full_scene_objects_refined_v20/full_scene_objects_refined_v20_stride10.ply`
- `server_parking_priority_s10/full_scene_objects_refined_v20/full_scene_objects.jsonl`

Why this teacher is used:

- It belongs to the pre-Patch route based on priority masks, geometry refinement,
  and surface safeguards.
- It is earlier than the known-bad global VLM relabel variants.
- It preserves the useful labels from the structure-prior + VLM/MASK phase:
  `wall`, `floor`, `grass`, `car`, `railing`, and `unknown`.

## Target Route

Target artifact:

- `server_parking_priority_s10/geo_patch_las_opt_cpp_v2_voxel003_r4_4090d_20260623/objects_v9_grid6_samegeom_structural_guard/objects_v9_grid6_samegeom_structural_guard_stride10.ply`
- `server_parking_priority_s10/geo_patch_las_opt_cpp_v2_voxel003_r4_4090d_20260623/objects_v9_grid6_samegeom_structural_guard/objects_v9_grid6_samegeom_structural_guard.jsonl`

The target route has 197,583 objects, most of them very small. Its own labels are
geometry labels such as `horizontal`, `vertical`, `thin_linear`, and
`rough_mixed`, so it should not be treated as a semantic result by itself.

## Method

Script:

- `scripts/transfer_teacher_semantics_to_objects.py`

Process:

1. Load the target viewer PLY and teacher semantic PLY.
2. Build a KD-tree over teacher points.
3. For each target point, find the nearest teacher point within `0.12m`.
4. Aggregate teacher labels per target object.
5. Apply target object `geometry_type` as a veto:
   - horizontal objects accept surface/ground-like labels.
   - vertical objects accept wall/building and selected vertical fine labels.
   - thin-linear objects accept railing/pipe/equipment/tree.
   - rough/mixed objects accept fine and surface labels.
6. Rewrite target object JSONL and viewer PLY semantics without changing point
   ownership.

## Remote Run

Host:

- `scan-train` / RTX 4090D

Remote output:

- `/root/epfs/SCAN/work_MT20260616-175807/geo_patch_las_opt_cpp_v2_voxel003_r4_4090d_20260623/objects_v9_teacher_v20_semantic/`

Local mirrored output:

- `server_parking_priority_s10/geo_patch_las_opt_cpp_v2_voxel003_r4_4090d_20260623/objects_v9_teacher_v20_semantic/`

Command:

```bash
python3 /root/epfs/SCAN/new_route/scripts/transfer_teacher_semantics_to_objects.py \
  --source-ply /root/epfs/SCAN/work_MT20260616-175807/geo_patch_las_opt_cpp_v2_voxel003_r4_4090d_20260623/objects_v9_grid6_samegeom_structural_guard/objects_v9_grid6_samegeom_structural_guard_stride10.ply \
  --source-objects-jsonl /root/epfs/SCAN/work_MT20260616-175807/geo_patch_las_opt_cpp_v2_voxel003_r4_4090d_20260623/objects_v9_grid6_samegeom_structural_guard/objects_v9_grid6_samegeom_structural_guard.jsonl \
  --teacher-ply /root/epfs/SCAN/work_MT20260616-175807/full_scene_objects_refined_v20/full_scene_objects_refined_v20_stride10.ply \
  --output-dir /root/epfs/SCAN/work_MT20260616-175807/geo_patch_las_opt_cpp_v2_voxel003_r4_4090d_20260623/objects_v9_teacher_v20_semantic \
  --output-prefix objects_v9_teacher_v20_semantic \
  --max-distance 0.12 \
  --min-teacher-votes 3 \
  --min-winner-ratio 0.55 \
  --min-global-winner-ratio 0.35 \
  --min-allowed-ratio 0.35 \
  --allow-surface-teacher-on-unknown
```

## Result

Report:

- object count: `197583`
- teacher matched point ratio: `0.416858`
- changed object count: `1307`
- transferred object count: `1548`
- kept because insufficient teacher votes: `194796`
- kept because teacher conflicted with geometry veto: `1206`

Point label counts after transfer:

- wall: `650971`
- floor: `496641`
- grass: `205461`
- unknown: `73085`
- railing: `13599`
- car: `8486`

## Interpretation

This test confirms that the old structure-prior + VLM/MASK route is a better
semantic teacher than raw SAM semantic PNG votes. It also shows that teacher
transfer alone is not enough: the current Patch route is too fragmented, and the
teacher stride10 field only covers about 42% of target preview points within the
0.12m match radius.

The next semantic step should not be direct SAM voting on every small Patch.
Use the earlier surface-prior route as a teacher/evidence source, then improve
Patch coarsening or object formation before semantic classification.

## Teacher-Guided Object Coarsening

Script:

- `scripts/coarsen_objects_with_semantic_teacher.py`

Purpose:

- Use the v20 teacher-transfer labels as object merge evidence.
- Keep voxel ownership exclusive.
- Merge only adjacent objects when semantic label, geometry bucket, color, and
  contact support agree.

Runs:

- `objects_v10_teacher_v20_coarsened`
  - edge source: original region graph
  - candidate edges: `7745`
  - accepted merges: `43`
  - output objects: `197540`
  - conclusion: too conservative because the original sparse graph misses many
    true object-object contacts.

- `objects_v11_teacher_v20_coarsened_unknown_absorb`
  - edge source: original region graph
  - accepted merges: `156`
  - output objects: `197427`
  - change: allowed small `unknown` fragments to be absorbed by known semantic
    neighbors under stricter color/contact rules.
  - conclusion: improves recall slightly, but still graph-limited.

- `objects_v12_teacher_v20_grid6_unknown_absorb`
  - edge source: full `grid6` voxel adjacency
  - candidate edges: `48630`
  - accepted merges: `2415`
  - output objects: `195168`
  - accepted by label: wall `1557`, floor `281`, grass `539`, railing `37`,
    car `1`
  - conclusion: this is the first useful teacher-guided coarsening diagnostic,
    but it is rejected as a semantic display baseline after visual QA. It still
    shows obvious errors such as ground being labeled as wall and horizontal car
    surfaces being shown as floor.

- `objects_v14_teacher_v20_grid6_geometry_guard_wall_recall`
  - input: v12
  - stage: hard semantic geometry guard
  - changed objects: `67749`
  - output labels: unknown `165892`, railing `27262`, floor `282`,
    wall `1470`, grass `186`, car `76`
  - point labels: floor `929024`, unknown `239132`, grass `206076`,
    wall `52386`, railing `13518`, car `8107`
  - conclusion: rejected after visual QA. It allowed `wall -> floor` promotion
    for horizontal/up-normal wall conflicts, which made many wall-like regions
    appear as floor.

- `objects_v15_teacher_v20_grid6_geometry_guard_no_wall_to_floor`
  - input: v12
  - stage: hard semantic geometry guard with wall-to-floor promotion disabled
  - point labels: unknown `1079157`, grass `206076`, floor `88999`,
    wall `52386`, railing `13518`, car `8107`
  - conclusion: fixes the direct wall-as-floor failure, but remains too
    conservative for final semantics. It should be used as a diagnostic safety
    bound, not as the active semantic baseline.

- `objects_v16_teacher_v20_grid6_geometry_guard_surface_recall`
  - input: v12
  - stage: no wall-to-floor promotion, and no automatic demotion of small
    floor/wall fragments
  - changed objects: `3236`
  - object labels: unknown `101463`, wall `38507`, floor `27674`,
    railing `27262`, grass `186`, car `76`
  - point labels: unknown `1030136`, grass `206076`, floor `124081`,
    wall `66325`, railing `13518`, car `8107`
  - conclusion: repairs v15's over-demotion of surfaces, but the high point-level
    unknown ratio shows that v12 remains a weak semantic/coarsening base.

Viewer:

```text
http://127.0.0.1:8765/tools/semantic_ply_viewer.html?file=/server_parking_priority_s10/geo_patch_las_opt_cpp_v2_voxel003_r4_4090d_20260623/objects_v12_teacher_v20_grid6_unknown_absorb/objects_v12_teacher_v20_grid6_unknown_absorb_stride10.ply&objects=/server_parking_priority_s10/geo_patch_las_opt_cpp_v2_voxel003_r4_4090d_20260623/objects_v12_teacher_v20_grid6_unknown_absorb/objects_v12_teacher_v20_grid6_unknown_absorb.jsonl&mode=semantic&stride=1&pointSize=1.2
```

Guarded diagnostic viewer:

```text
http://127.0.0.1:8765/tools/semantic_ply_viewer.html?file=/server_parking_priority_s10/geo_patch_las_opt_cpp_v2_voxel003_r4_4090d_20260623/objects_v14_teacher_v20_grid6_geometry_guard_wall_recall/objects_v14_teacher_v20_grid6_geometry_guard_wall_recall.ply&objects=/server_parking_priority_s10/geo_patch_las_opt_cpp_v2_voxel003_r4_4090d_20260623/objects_v14_teacher_v20_grid6_geometry_guard_wall_recall/objects_v14_teacher_v20_grid6_geometry_guard_wall_recall.jsonl&mode=semantic&stride=1&pointSize=1.2
```

No wall-to-floor diagnostic viewer:

```text
http://127.0.0.1:8765/tools/semantic_ply_viewer.html?file=/server_parking_priority_s10/geo_patch_las_opt_cpp_v2_voxel003_r4_4090d_20260623/objects_v15_teacher_v20_grid6_geometry_guard_no_wall_to_floor/objects_v15_teacher_v20_grid6_geometry_guard_no_wall_to_floor.ply&objects=/server_parking_priority_s10/geo_patch_las_opt_cpp_v2_voxel003_r4_4090d_20260623/objects_v15_teacher_v20_grid6_geometry_guard_no_wall_to_floor/objects_v15_teacher_v20_grid6_geometry_guard_no_wall_to_floor.jsonl&mode=semantic&stride=1&pointSize=1.2
```

Surface-recall diagnostic viewer:

```text
http://127.0.0.1:8765/tools/semantic_ply_viewer.html?file=/server_parking_priority_s10/geo_patch_las_opt_cpp_v2_voxel003_r4_4090d_20260623/objects_v16_teacher_v20_grid6_geometry_guard_surface_recall/objects_v16_teacher_v20_grid6_geometry_guard_surface_recall.ply&objects=/server_parking_priority_s10/geo_patch_las_opt_cpp_v2_voxel003_r4_4090d_20260623/objects_v16_teacher_v20_grid6_geometry_guard_surface_recall/objects_v16_teacher_v20_grid6_geometry_guard_surface_recall.jsonl&mode=semantic&stride=1&pointSize=1.2
```

Remaining bottleneck:

- `shared_edges` is still the largest rejection reason, meaning many fragments
  are spatially close but do not have enough direct voxel-face contact under
  current `0.03m` graph adjacency.
- `label_mismatch` is also still high, so semantic teacher disagreement should
  be handled as review evidence, not forced into object merging.
- The teacher semantic label is not a hard truth. Future object coarsening must
  use teacher semantics only after a geometry guard has already accepted the
  candidate; otherwise local teacher errors are amplified into larger object
  errors.
- A geometry guard must demote unsafe labels; it must not promote a conflicting
  wall label into floor unless there is independent ground/drivability evidence.
