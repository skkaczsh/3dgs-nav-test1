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

## Geometry Patch Demo

`build_geo_patch_demo.py` is a geometry-only boundary QA tool. It ignores existing
object and semantic ids, voxelizes the colored point cloud, computes local PCA
features, and uses gated BFS to create random-color geometry patches.

Full stride10 demo outputs:

- conservative geometry/color: `server_parking_priority_s10/geo_patch_demo_full_v1_geom_color`
  - voxel size: `0.10m`
  - patch count: `140233`
  - small patch count: `136945`
- relaxed radius-2 connectivity: `server_parking_priority_s10/geo_patch_demo_full_v2_relaxed_radius2`
  - voxel size: `0.10m`
  - patch count: `41208`
  - small patch count: `39468`

The relaxed version is the current review candidate. The high small-patch count
means pure local BFS is still too sensitive to sparse LiDAR sampling; the next
version should add plane-model absorption or supervoxel merging while preserving
hard boundaries from normal/color/depth discontinuities.

Voxel resolution note:

- `0.10m` is too coarse for stair/railing/wall-foot boundaries. It is useful as
  a smoke test only.
- `0.03m` preserves fine geometry better, but pure local BFS becomes fragmented
  because the scan has real sampling gaps. Full stride10 results:
  - `geo_patch_demo_full_v3_voxel003_radius2`: `828874` voxels, `373873` patches.
  - `geo_patch_demo_full_v4_voxel003_radius4`: `828874` voxels, `125230` patches.
- The next geometry mainline should keep `0.03m` voxels but replace pure BFS
  with model-aware merging: large plane extraction, stair-step grouping, and
  thin-structure handling.

Score-based BFS test:

- `build_geo_patch_demo.py --edge-mode score` replaces all-pass hard gates with
  a weighted edge score over normal, height, color, bucket compatibility,
  roughness, and planarity. Loose hard vetoes still block impossible bridges.
- Full stride10 results at `0.03m`:
  - hard gate v4: `125230` patches.
  - score `0.52`: `121052` patches.
  - score `0.46`: `116319` patches.
- The score threshold reduces fragmentation only moderately. The main bottleneck
  is still sparse LiDAR sampling plus local PCA instability, so the next step
  should be seed/model-aware merging rather than further global threshold tuning.

Region-model BFS test:

- `build_geo_patch_demo.py --edge-mode region-model` compares each candidate
  voxel against the growing patch state: patch mean normal, mean color,
  roughness, planarity, and a simple plane residual.
- Full stride10 result:
  - `geo_patch_region_model_full_v1_voxel003_score048`: `123823` patches,
    `117803` small patches.
- This first region-model variant does not significantly reduce fragmentation.
  It is still limited by sparse candidate reachability and a FIFO queue. A real
  production version should use prioritized region growing plus explicit
  seed/model classes, preferably in the `drivability_cpp` grid implementation.

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
