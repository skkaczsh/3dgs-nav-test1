# Spatial Partition Mainline

This route treats semantic point-cloud output as a spatial segmentation problem, not a frame-target merge problem.

## Invariants

- One voxel has exactly one semantic winner and one object owner.
- One object is a 6-neighbor connected component in voxel space.
- Different objects and different semantic labels must not overlap in the same output voxel.
- Small components are preserved by default as `small_component`; filtered previews must opt into `--small-component-policy drop`.

## Current Inputs

The first full-scale implementation uses existing V2/V8 semantic PLY outputs as teachers:

- V2: stable scene consistency and fewer indoor false cars.
- V8: better stair/height handling and finer local geometry splits.

Teacher votes are only evidence. The partition step owns the final voxel-to-object assignment.

## Current Output

Latest full stride10 run:

- directory: `server_parking_priority_s10/spatial_partition_mainline_v2_keepall_v2v8_full_s10`
- voxel size: `0.10m`
- assigned voxels: `465467`
- unassigned voxels: `0`
- mixed object voxels: `0`
- mixed semantic voxels: `0`

The object count is currently high because every isolated small connected component is preserved. The next stage should merge or absorb small components using geometry/color/visibility gates, while keeping the one-voxel-one-owner invariant.

## Next Engineering Step

Add a spatial absorption stage after partition:

1. Keep large connected components as anchors.
2. For each small component, search nearby compatible anchor components.
3. Merge only when geometry type, color, normal, and teacher-vote compatibility agree.
4. If no anchor is compatible, keep it as residual instead of overlapping or silently deleting it.

This keeps the problem mathematically well-formed: segmentation first, semantic refinement second.

## Absorption Stage

`absorb_spatial_partition_objects.py` is the second stage. It remaps small component object ids to nearby compatible anchor ids; it does not duplicate points or allow overlap.

Current full stride10 experiments:

- conservative: `server_parking_priority_s10/spatial_partition_mainline_v3_absorbed_conservative_v2v8_full_s10`
  - radius: `2` voxels
  - cross-label groups: disabled
  - final objects: `143539`
  - mixed object voxels: `0`
  - mixed semantic voxels: `0`
- grouped: `server_parking_priority_s10/spatial_partition_mainline_v4_absorbed_grouped_v2v8_full_s10`
  - radius: `3` voxels
  - cross-label groups: `floor/indoor_floor/roof`, `wall/building`, `grass/tree`
  - final objects: `131471`
  - mixed object voxels: `0`
  - mixed semantic voxels: `0`

Use the conservative version as the default review baseline. Use the grouped version only when checking whether reduced fragmentation is worth the risk of semantic absorption into large surfaces.
