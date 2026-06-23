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

Object-model BFS test:

- `build_geo_patch_demo.py --edge-mode object-model` relaxes the assumption that
  one patch must have a stable normal. Candidate voxels are compared with the
  growing patch state using color/texture, roughness, linearity, planarity,
  local height range, height continuity, bucket compatibility, and weak
  normal/plane evidence.
- Full 4090D torch result at `0.03m`:
  - previous `region-model`: `123819` patches, `117800` small patches.
  - `object-model`: `113503` patches, `108409` small patches.
- This confirms that allowing normal jumps helps folded/same-texture objects,
  but the remaining fragmentation is still too high. The next production step
  should not keep tuning global BFS thresholds; it should use structure-specific
  seed models and priority growth: hard ground seeds, weak wall seeds,
  stair-step grouping, and thin-structure grouping.

## Array Region-Model Patch Growth

`build_geo_patch_graph.py` is fast and scalable, but its connected-component
step is pairwise: if `A-B` and `B-C` are valid edges, `A` and `C` enter the same
patch even when `C` no longer fits the whole patch.  This is the main failure
mode behind both over-fragmentation and bridge-based over-merging.

`build_geo_patch_region_model.py` keeps the graph route's voxel/PCA feature
assets and edge-admissibility pass, then replaces connected components with
model-aware region growing:

1. Build local candidate edges using the existing graph scorer.
2. Grow seeds in a stable-first order, so large horizontal/vertical surfaces
   lock their structure before rough objects can bridge through them.
3. For each frontier candidate, check whether the voxel fits the accumulated
   patch model, not just whether it matches the boundary voxel.
4. Use different membership gates for stable surfaces and object-like patches:
   stable surfaces emphasize plane residual and normal consistency; rough/thin
   objects emphasize color/texture, shape statistics, and height continuity with
   weak normal evidence.
5. Emit per-patch `rejected_reasons_top` so QA can see whether a boundary was
   cut by plane residual, height jump, color/texture jump, or low membership
   score.

Initial 4090D smoke results on
`colorized_visible_0000_6180_voxel010.ply`:

- `region_model_smoke_200k`: `200000` voxels, `17844` patches,
  `16479` small patches.
- `region_model_smoke_1m`: `1000000` voxels, `45667` patches,
  `41636` small patches.

Largest-patch inspection shows no obvious cross-bucket disaster: the largest
patches are pure `horizontal`, pure `vertical`, or mostly `rough_mixed` with
small `unknown/thin_linear` support.  The route is therefore safer than loose
graph connectivity, but still conservative.  Next tuning should focus on
object-like membership gates and adaptive radius for rough/vegetation patches,
not on globally loosening normal thresholds.

Object height gate correction:

- The initial region-model route used candidate-to-patch-centroid `dz` as a hard
  veto for object-like patches and produced many `object_height_jump` rejects.
- This is physically wrong for shrubs, railings, stairs, and wall-attached
  objects: local edge continuity should constrain height jumps, while the patch
  model should allow a tall object to grow away from its centroid.
- `object_height_jump` has therefore been removed as a hard veto.  Height remains
  weak membership evidence for object-like patches; hard height gates are kept
  only for stable horizontal surfaces.
- A 1M-voxel comparison changed patch count from `45667` to `45191`; largest
  rough patches grew without cross-bucket surface contamination.  The dominant
  remaining object-like rejection is now `membership_score_low`, so further
  improvement should tune the object membership score rather than add back a
  centroid-height veto.

Rough/object membership score update:

- `rough_mixed` patches now use a separate membership score that strongly
  weights color/texture and local shape statistics, while downweighting normal,
  plane residual, and centroid height.
- This matches shrubs and other rough objects: top and side faces can have very
  different normals while still being the same physical object if color,
  texture, roughness, and local shape remain compatible.
- 1M-voxel comparison:
  - previous no-height-veto: `45191` patches, `40092` small patches.
  - rough-score update: `43836` patches, `40092` small patches.
  - `membership_score_low` rejects dropped from `81371` to `37486`.
- Full `voxel010` result:
  - output: `full_region_model_voxel010_rough_score_v3`
  - `3729992` voxels, `141958` patches, `129824` small patches.
  - previous no-height-veto full run had `152549` patches and `139188` small
    patches.
  - largest rough patches grew substantially while largest horizontal/vertical
    patches remained pure buckets, so the update improves object-like merging
    without obvious stable-surface contamination.

Multimodal patch signature update:

- Mean-only patch state is insufficient for objects or regions with multiple
  valid local modes.  Examples: shrub top vs side, flat ground vs stair edge,
  and rough object parts with similar color but different normals.
- `build_geo_patch_region_model.py` now maintains up to 24 local prototypes per
  growing patch.  A candidate voxel can match any prototype, not only the patch
  mean.  The prototype signature contains normalized RGB, roughness, planarity,
  linearity, local color standard deviation, local height range, and `abs(nz)`.
- Horizontal stable surfaces also have a conservative multimodal bridge: if a
  nearby rough/unknown/thin voxel has compatible color/texture and local shape,
  it can enter the patch as a new mode without requiring normal/plane agreement.
  Vertical wall bridging remains blocked by default.
- 1M-voxel comparison:
  - rough-score v3: `43836` patches, `40092` small patches.
  - multimodal v4: `43373` patches, `39682` small patches.
- Full `voxel010` result:
  - output: `full_region_model_voxel010_multimodal_v4`
  - `3729992` voxels, `140152` patches, `128298` small patches.
  - previous rough-score full run had `141958` patches and `129824` small
    patches.
  - largest horizontal and vertical patches remain pure buckets; largest
    `rough_mixed` patch grew from `316850` to `377903` voxels.
- Performance note: the prototype implementation is Python-loop based and is
  significantly slower than v3.  Before making it the default production route,
  prototype matching should be vectorized or restricted to object-like buckets.

Local chart atlas update:

- The v4 prototype route still used a global centroid/mean-normal plane for
  stable surfaces.  That conflicts with the multimodal premise: a large wall,
  ground region, roof, or stair-connected floor may be one patch made from
  multiple local plane charts.
- `PatchModel` now stores `prototype_xyz` and `prototype_normals` alongside each
  feature prototype.  Stable-surface `plane_residual`, `normal`, and height gates
  are evaluated against the best local chart instead of the global patch mean.
- This keeps the single graph/region-growing data flow intact.  It does not
  split large surfaces into a separate algorithm; it changes the patch model
  from a single plane to a small local atlas.
- 1M-voxel comparison:
  - multimodal v4: `43373` patches, `39682` small patches.
  - full local-chart v5 smoke: `42470` patches, `38935` small patches.
  - rescue-only local-chart v5 smoke: `42531` patches, `39000` small patches.
  - `stable_plane_residual` rejects dropped from `235380` to `36735` in the
    rescue-only version.
  - largest vertical chart grew from `123347` to `148482` voxels while staying
    pure `vertical`.
- Implementation note: local chart lookup is now rescue-only.  The script uses
  the fast global surface model first and only queries the local chart atlas
  when global plane/normal/height gates would reject the candidate.  This keeps
  the single graph formulation while avoiding chart lookup for every stable
  surface edge.
- Graph formulation update: stable-surface membership now evaluates a candidate
  against both the patch's chart atlas and the accepted frontier voxel that
  reached it.  This makes a patch a connected subgraph with multiple local
  chart peaks, not one global plane.  `stable_plane_residual` therefore remains
  a useful reject reason only when the candidate fails both global/atlas
  consistency and graph-local continuity.
- 1M-voxel frontier-chart smoke:
  - patches: `42180`, small patches: `38719`.
  - `stable_plane_residual` rejects dropped from rescue-only `36735` to `21`.
  - largest vertical patch grew from `148482` to `160383` voxels while remaining
    pure `vertical`.
  - largest horizontal patch remained pure `horizontal` at `178505` voxels.
  - dominant remaining reject is now `membership_score_low`, which is a real
    model-fit boundary rather than a single-plane contradiction.
- Full run candidate:
  - output: `full_region_model_voxel010_frontier_chart_v6`
  - status: launched on 4090D.

C++ backend status:

- `tools/geo_patch_region_model_core.hpp` now holds the shared C++ region-model
  core.
- `tools/geo_patch_region_model_smoke.cpp` validates the core graph-region
  invariant in a small synthetic fixture.
- `tools/geo_patch_region_grower.cpp` is wired into
  `build_geo_patch_region_model.py` behind `--region-grow-backend cpp`.
- Scope: Python still owns PLY IO, voxelization, local PCA/features, edge
  construction, output PLY/JSONL, and QA reports.  C++ currently owns only the
  region-growing label assignment.
- Verified invariants:
  - frontier local chart can rescue stable-surface growth that a single global
    plane would reject.
  - pairwise graph chains cannot bridge a large horizontal height jump into the
    same patch.
- Build entry: `scripts/build_geo_patch_cpp_smoke.sh`.
- Verified locally with pytest and smoke binary.  Next benchmark should compare
  `--region-grow-backend python` vs `cpp` on a fixed 200k/1M voxel sample before
  changing production defaults.
- CLI smoke on a real parking-scene PLY also succeeds with
  `--region-grow-backend cpp`; output remains compatible with the existing
  `geo_patches_region_model_random_color.ply` / JSONL viewer format.

5070Ti 1M-voxel backend comparison:

- Input:
  `work_MT20260616-175807/outputs/colorized_full/colorized_visible_0000_6180_voxel010_ascii.ply`
- Parameters: `--voxel-size 0.10 --max-points 1000000 --feature-backend torch`.
- Initial full-pipeline timing:
  - Python backend: `210s`.
  - C++ backend before summary optimization: `209s`.
  - Diagnosis: the C++ core was fast, but Python's label-to-patch summary used
    one full boolean mask per patch, making the native backend spend most time
    in O(voxels * patches) post-processing.
- Region-grow core-only timing on the same serialized arrays/edges:
  - Python core: `190.775s`.
  - C++ core: `1.308s`.
  - core speedup: `145.868x`.
- After vectorizing the C++ label summary by sorted label groups:
  - C++ full pipeline: `18s`.
  - end-to-end speedup vs Python backend: about `11.7x`.
- Result difference:
  - Python: `999998` voxels, `42189` patches, `38729` small patches.
  - C++: `999998` voxels, `42186` patches, `38726` small patches.
  - same label id ratio: `0.9809`.
  - Python patch -> C++ majority overlap: `1.0000`.
  - C++ patch -> Python majority overlap: `0.999992`.
- Interpretation: C++ backend is now suitable as the default region-growing
  backend for large runs. Remaining pipeline bottlenecks are feature/edge
  preparation and output writing, not region growth.

Patch graph optimization v1/v2:

- Script: `scripts/optimize_geo_patch_merges.py`.
- Input: C++ region-grower `_cpp_region_grower_input.bin` and
  `_cpp_region_grower_labels.bin`; no PCA/edge rebuild.
- Method: conservative greedy small-patch absorption.  Only original small
  patches may merge into existing anchors; large-large merges are disabled.
- v1 parameters:
  - `small_patch_voxels=8`, `anchor_min_voxels=64`, `min_gain=0.66`.
  - merged small patches: `1050`.
  - final patches: `135904`.
- v2 parameters:
  - `small_patch_voxels=8`, `anchor_min_voxels=16`, `min_gain=0.58`.
  - merged small patches: `3820`.
  - final patches: `133134`.
- v2 viewer output:
  `server_parking_priority_s10/geo_patch_full_cpp_v1_4090d_optimized_small_absorb_v2/`.
- Overlap result on stride5 preview:
  - original half-voxel conflict extra ratio: about `0.0948%`.
  - v2 half-voxel conflict extra ratio: about `0.0938%`.
- Interpretation: patch graph absorption can reduce fragmentation modestly, but
  it does not materially solve spatial ownership conflicts.  The next optimizer
  must include fine-cell ownership directly in the objective, rather than only
  merging by patch adjacency.

Boundary transfer optimizer v1:

- Script: `scripts/optimize_geo_patch_boundaries.py`.
- Method: operate only on `0.05m` fine cells containing multiple patch ids.
  Each conflict cell is assigned to the candidate patch whose current model
  best explains that cell using RGB, geometry bucket, normal, occupancy support,
  and patch size prior.  This dynamically changes boundary ownership instead of
  only absorbing small patches.
- Output:
  `server_parking_priority_s10/geo_patch_full_cpp_v1_4090d_boundary_transfer_v1/`.
- Results:
  - input patches: `136954`.
  - output patches: `136726`.
  - conflict cells seen: `18711`.
  - resolved cells: `12493`.
  - transferred points: `12538`.
  - `0.05m` conflict extra ratio: `0.5024%` -> `0.1668%`.
- `0.10m` conflict extra ratio: `4.7814%` -> `4.7017%`.
- Interpretation: direct boundary ownership transfer is much more effective
  than small-patch absorption for fine-cell conflicts.  The remaining coarse
  `0.10m` conflicts are mostly caused by using a coarser analysis grid than the
  ownership grid; reducing them will require either ownership at `0.10m` or
  producing viewer/output points at canonical voxel centers.

Patch graph energy v3 on 5070Ti:

- Scripts:
  - `scripts/optimize_patch_graph_energy.py`
  - `scripts/run_rtx5070_geo_patch_energy.sh`
- Input:
  `frame_object_viewer_attachment_localgeom_pure_surface_visibility_full_0000_6180/frame_object_points_stride10.ply`.
- Parameters:
  - voxel size: `0.03m`.
  - backend: torch feature extraction + C++ region grower.
  - optimizer: split + boundary transfer + annealing merge, output stem
    `geo_patches_energy_v3`.
- Implementation fixes in v3:
  - Fine-cell normal is now the normalized mean normal of all normals in the
    cell, not the first normalized row.
  - Random-color preview no longer has bitwise precedence leakage, so patch
    display has stronger deterministic contrast.
  - Reports now include boundary and merge rejection reason counts.
- 2026-06-24 full-run result:
  - voxel count: `4,572,554`.
  - initial C++ region patches: `718,025`.
  - small patches: `694,648`.
  - energy output patches: `705,693`.
  - boundary moved points: `75,275`.
  - boundary rejected cells: `735,815`.
  - merge accepted/rejected: `75 / 24`.
  - top-1000 AABB overlap pairs: `2,843 / 499,500`.
  - `0.20m` mixed object voxel ratio on stride3 preview: `0.4553`.
- Interpretation:
  - v3 confirms the dominant failure is not viewer, VLM, or color display.  The
    region candidate layer is already too fragmented, and the current energy
    merge objective is too weak to recover object-scale patches after the fact.
  - The next change should move object-scale compatibility into candidate
    generation or make the energy objective optimize patch assignment over a
    coarser supernode graph.  Continuing to tune scalar merge thresholds on the
    current 700k-patch graph is unlikely to reach the target thousand-level
    patch budget without over-merging.

Coarse supernode diagnostic on 5070Ti:

- Scripts:
  - `scripts/coarsen_geo_patches_to_budget.py`
  - `scripts/run_rtx5070_geo_patch_coarsen.sh`
- Reason:
  - The 0.03m C++ region run produced `718,025` patches, with median size `1`
    voxel and p90 size `3` voxels.  A full Python coarsen on all 718k labels did
    not produce any stage output after several minutes because it was spending
    time building per-patch Python statistics for mostly tiny components.
- Implementation change:
  - Added vectorized pre-collapse for tiny labels before expensive patch
    statistics.
  - Added stage logging to the coarsen script so long runs show whether they
    are in read, pre-collapse, stats, merge, or write.
- Size distribution from `_cpp_region_grower_labels.bin`:
  - total voxels: `4,572,554`.
  - total patches: `718,025`.
  - patches `<=1` voxel: `516,451` (`516,451` voxels).
  - patches `<=3` voxels: `651,931` (`822,595` voxels).
  - patches `<=8` voxels: `698,189` (`1,062,108` voxels).
- `target=20,000`, pre-collapse `<=8` result:
  - active input patches after pre-collapse: `19,837`.
  - output patches: `19,533`.
  - main merge count: `0` because active count was already below target.
  - overlap suppression merge count: `304`.
  - `0.20m` mixed object voxel ratio on stride5 preview: `0.2268`.
- `target=5,000`, pre-collapse `<=8` result:
  - active input patches after pre-collapse: `19,837`.
  - output patches: `4,804`.
  - main merge count: `14,837`.
  - evaluated edges: `15,278`.
  - low-score rejects: `440`.
  - overlap suppression merge count: `196`.
  - `0.20m` mixed object voxel ratio on stride5 preview: `0.2096`.
  - top-1000 AABB overlap pairs: `3,639 / 499,500`.
- `target=5,000`, connected-grid residual result:
  - mode: `connected-grid`, grid size `0.50m`, tiny threshold `<=8`.
  - tiny source patches: `698,189`, `1,062,108` voxels.
  - residual components after local grouping: `55,628`.
  - active input patches after residual grouping: `75,464`.
  - output patches: `4,469`.
  - main merge count: `50,000`.
  - evaluated edges: `88,810`.
  - overlap suppression merge count: `531`.
  - `0.20m` mixed object voxel ratio on stride5 preview: `0.1545`.
  - top-1000 AABB overlap pairs: `4,136 / 499,500`.
- Interpretation:
  - Pre-collapse proves the coarsen stage can operate at useful scale once the
    pathological one-voxel patch population is removed.
  - Global residual collapse was only a diagnostic: collapsing all `<=8` voxel
    patches into one residual consumed `1,062,108` voxels, so it could not be a
    final object boundary model.
  - Connected-grid residual is the current preferred architecture: it keeps
    dense 0.03m voxels in local spatial components, preserves one-object-per-
    voxel ownership, and gives the coarsen stage a tractable graph without
    deleting or globally pooling the sparse original data.
  - The next production change should make the connected residual grouping
    less grid-shaped by using local adjacency + color/geometry compatibility
    before fallback grid bucketing.

Dense colorized source note:

- The full colorized reconstruction is
  `work_MT20260616-175807/outputs/colorized_full/colorized_visible_0000_6180_full.ply`.
  It is a binary PLY with `92984215` colored points and about `95%` color
  coverage.
- `build_geo_patch_demo.py` now supports binary little-endian XYZRGB PLY input,
  but the current Python dict/FIFO BFS implementation is not suitable for
  repeated dense full-scene production. A full `0.05m` rebuild from 93M raw
  colored points spends minutes in CPU voxel aggregation before reaching GPU
  PCA.
- For viewer QA, use the cached full-scene colorized voxel file:
  `colorized_visible_0000_6180_voxel010.ply` (`3729996` voxels). The helper
  `export_binary_ply_viewer_ascii.py` converts it to the ASCII schema accepted
  by `semantic_ply_viewer.html`. Use stride exports for browser stability.
- `build_geo_patch_demo.py` has two dense-path acceleration switches:
  - `--voxel-backend torch`: use torch/CUDA for binary PLY voxel aggregation.
  - `--binary-voxel-input`: treat an already-voxelized binary PLY as voxel rows
    and skip re-aggregation.
- Validated smoke tests:
  - cached voxel direct-read `200k` rows + torch PCA: completed in about `11s`.
  - full binary dense `1M` raw points + torch voxelization + torch PCA:
    completed in about `20s`.
- Full cached voxel010 patch run:
  - `3729979` voxels, `100106` patches, `91538` small patches.
  - output: `server_parking_priority_s10/full_dense_cached_voxel010_identity_object_model_torch`
  - viewer preview: `geo_patches_random_color_stride5.ply` (`745996` points).
- Graph segmentation test:
  - `build_geo_patch_graph.py` replaces Python FIFO region growing with a
    vectorized pairwise similarity graph and sparse connected components.
  - It is a scaling baseline: no dynamic patch state, but much faster on dense
    full-scene inputs.
  - Full cached voxel010 `radius=2, score=0.46` result:
    `3729979` voxels, `56801` patches, `53868` small patches.
  - output: `server_parking_priority_s10/full_graph_cached_voxel010_r2_s046_torch`
  - viewer preview: `geo_patches_graph_random_color_stride3.ply`
    (`1243327` points).
  - Failure mode: the loose graph produced a giant mixed component
    (`3259547` voxels) because connected components are transitive; rough and
    unknown voxels acted as bridges across unrelated structures.
  - `--bucket-guard same-bucket` fixes the mixed-bridge failure by only
    allowing graph edges within the same geometry bucket. Full result:
    `3729979` voxels, `182168` patches, `169215` small patches.
  - output: `server_parking_priority_s10/full_graph_cached_voxel010_r2_s046_samebucket_torch`
  - This is a safe-boundary graph baseline, not a final object model. It still
    contains large pure horizontal/vertical components that need a second split
    stage by plane model, spatial zone, stair rhythm, or scene structure.
- Production patching should cache voxelized dense clouds as NPZ/PLY once and
  move connectivity/growing out of Python dict BFS into C++ or tensor graph
  operations.

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
