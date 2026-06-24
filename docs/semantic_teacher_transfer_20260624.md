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
