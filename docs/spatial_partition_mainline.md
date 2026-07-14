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
- 5070Ti run on 2026-06-28:
  - input:
    `/home/zsh/Work/SCAN/work_MT20260616-175807/outputs/colorized_full/colorized_visible_0000_6180_voxel010_ascii.ply`.
  - 1M C++ smoke:
    `geo_patch_5070_region_model_cpp_smoke_1m_20260628`.
    Result: `999998` voxels, `42186` patches, `38726` small patches.  This is
    within single-digit patch-count difference from the Python frontier-chart
    smoke (`42180` / `38719`).
  - full C++ run:
    `geo_patch_5070_region_model_cpp_full_20260628`.
    Result: `3729976` voxels, `136954` patches, `125715` small patches.
  - Local viewer artifact:
    `server_parking_priority_s10/geo_patch_5070_region_model_cpp_full_20260628/geo_patches_region_model_random_color_stride5.ply`.
  - Direct `192.168.0.2` SSH was not reachable from the Mac at the time of
    transfer; artifact sync used `skkac.top:6010`.

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

- Dense source contract:
  - The authoritative dense geometry/RGB source for `MT20260616-175807` is the
    2.92GB LAS export:
    `datasets/MT20260616-175807/3dgs_20260616_3dgs_0095fd8ebe7845f2b24820cb98de4abb_task_c967d04831bf45f0b9d7c3fac31b4103_output_pointcloud_CommandProcessLXOutput_MANIFOLD_MT20260616-175807-Opt.las`.
  - It contains `97194579` points with XYZ, RGB, intensity, classification, and
    extra `Label` dimensions.
  - The canonical patch input derived from it on the 5070Ti host is:
    `/home/zsh/Work/SCAN/work_MT20260616-175807/dense_sources/dense_las_voxel003_20260624/dense_las_voxel003_binary.ply`.
  - Conversion command:
    `python scripts/las_to_voxel_ascii_ply.py --input-las <las> --output-ply <ply> --report-json <json> --voxel-size 0.03 --output-format binary_little_endian`.
  - Conversion result: `14482557` occupied `0.03m` voxels, binary XYZRGB PLY,
    `208MB`, validated by `build_geo_patch_demo.py` binary direct-read smoke.
  - `frame_object_points_stride10.ply` and other viewer stride PLY files are
    diagnostic/review artifacts only. They must not be used as authoritative
    geometry inputs for patch-quality conclusions.
- First full dense LAS patch run:
  - output: `server_parking_priority_s10/dense_las_voxel003_region_model_full_cpp_20260624`
    on the 5070Ti host.
  - command input: `dense_las_voxel003_binary.ply`, `--voxel-size 0.03`,
    `--binary-voxel-input`, torch local features, C++ region grower.
  - runtime: `3:44.59`, peak RSS `24.4GB`.
  - result: `14405828` voxels, `1017019` patches, `971957` small patches
    (`95.6%` of patches).
  - review preview: `geo_patches_region_model_random_color_stride10.ply`
    (`1440583` points).
  - interpretation: the authoritative dense source is now correct and usable,
    but the current region-grow patch model is still much too fragmented at
    `0.03m`.  The next architecture step is not another data-source change; it
    is a real patch coarsening/graph optimization stage with one-voxel-one-owner
    invariants and streaming/binary output.
- Dense coarsening tests on the above run:
  - `dense_las_voxel003_coarsen50k_conncompat_20260624`
    - tiny source patches `<=8`: `977000` patches / `1489358` voxels.
    - connected-compatible precollapse reduced active stats to `130990`
      patches.
    - budget coarsen output: `50000` patches in `0:26.15`, peak RSS `4.3GB`.
    - top-1000 AABB overlap: `3804 / 499500` pairs, including `1736`
      near-contained pairs (`ratio_min_volume >= 0.95`).
  - `dense_las_voxel003_coarsen50k_conncompat_overlap020_20260624`
    - same precollapse and budget, plus `0.20m` overlap suppression.
    - output: `44765` patches in `0:31.35`, peak RSS `4.3GB`.
    - overlap suppression merged `5235` pairs, but top-1000 AABB overlap only
      changed to `3718 / 499500` pairs and near-contained pairs increased to
      `1792`.
  - Interpretation: connected-compatible precollapse is useful and should stay.
    Post-hoc AABB overlap suppression is not the right boundary fix.  The next
    stage should do unified graph/energy optimization over patch boundaries and
    object ownership, using overlap as a term in the objective rather than a
    separate after-the-fact merge pass.
- Dense energy-graph test:
  - input labels: `dense_las_voxel003_coarsen50k_conncompat_labels_20260624/geo_patches_coarse_labels.bin`.
  - output: `dense_las_voxel003_energy_v3_from_coarsen50k_20260624`.
  - settings: `3` iterations, boundary transfer enabled, annealing enabled,
    no split stage.
  - result: `50000 -> 49313` patches, `364700` boundary points moved,
    `480` merge accepts, `235` merge rejects.
  - runtime: `1:10.29`, peak RSS `4.2GB`.
  - top-1000 AABB overlap: `2951 / 499500` pairs, including `1651`
    near-contained pairs.  This improves over plain coarsen (`3804` / `1736`)
    and post-hoc overlap suppression (`3718` / `1792`), so unified
    boundary/merge optimization is the better direction.
  - Engineering note: `optimize_patch_graph_energy.py` now emits stage logs;
    previous silent runs were hard to diagnose.
- Dense energy-graph v4 with overlap/containment candidates:
  - output: `dense_las_voxel003_energy_v4_overlap_candidates_20260624`.
  - code change: high AABB-overlap patch pairs are now inserted into the same
    merge-candidate list as graph-adjacent pairs, then judged by the existing
    energy objective. This is not a post-hoc merge pass.
  - result: `50000 -> 48750` patches, `396659` boundary points moved,
    `1024` merge accepts, `56` merge rejects.
  - merge source: `696` accepts from `adjacency+overlap`, `326` from pure
    `overlap`, and `2` from pure adjacency. This confirms the change targets
    the previously missed high-overlap candidates.
  - runtime: `2:00.66`, peak RSS `4.2GB`.
  - top-1000 AABB overlap: `2055 / 499500` pairs, including `1350`
    near-contained pairs. This is the current best overlap metric among the
    dense LAS runs.
  - Remaining issue: most residual near-contained pairs are small patches
    inside huge `mixed` AABBs. AABB containment alone is too weak to prove true
    point ownership conflict. The next metric must use fine voxel intersection
    or shared occupied-cell evidence before accepting more merges, otherwise
    the optimizer will over-merge buildings, trees, and ground again.
- Dense energy-graph labels and fine-cell overlap diagnostics:
  - `optimize_patch_graph_energy.py` now writes final one-voxel-one-owner labels
    as `*_labels.bin` with the same `GPRGlabels1` schema used by coarsening.
    This is required for precise diagnostics and later object construction.
  - output: `dense_las_voxel003_energy_v4_overlap_candidates_rerun_labels_20260624`.
    It reproduces the v4 result (`50000 -> 48750`) and adds final labels.
  - `analyze_geo_patch_bbox_overlap.py` now optionally reads region input and
    labels to report fine-cell co-occupancy, not just AABB overlap.
  - top-1000 v4 fine-cell diagnostic:
    - AABB overlap pairs: `2055`.
    - AABB near-contained pairs: `1350`.
    - at `0.08m` fine cells: `1165` bbox pairs share at least one occupied
      cell, `352` have fine-cell ratio `>=0.5`, and only `4` have ratio
      `>=0.95`.
    - at `0.05m` fine cells: `1093` bbox pairs share at least one occupied
      cell, `65` have fine-cell ratio `>=0.5`, and `0` have ratio `>=0.95`.
    - `418` of the AABB near-contained pairs have no `0.05m` fine-cell overlap
      at all.
  - Interpretation: residual AABB containment is mostly a bounding-box artifact,
    not proof that two patches occupy the same space. Continuing to merge based
    on AABB alone is mathematically wrong.
- Dense energy-graph v5 with fine-cell candidates:
  - output: `dense_las_voxel003_energy_v5_fine_overlap_candidates_20260624`.
  - settings: v4 plus `--enable-fine-overlap-merge-candidates` at `0.05m`.
  - result: same patch counts and overlap metrics as v4 (`50000 -> 48750`,
    top-1000 AABB overlap `2055`).
  - merge log: `7` accepted candidates carried `fine_overlap` evidence; the run
    remained dominated by AABB overlap candidates.
  - Interpretation: fine-cell evidence is useful as a diagnostic and future
    gate, but the current v5 still treats it as an additive candidate source.
    The next optimizer revision should make occupied-cell support a gate or
    primary support term for overlap-only merges, while AABB should be only a
    cheap recall prefilter.
- Dense energy-graph v6 with fine-cell gated overlap:
  - output: `dense_las_voxel003_energy_v6_fine_gated_overlap_20260624`.
  - settings: v5 plus `--overlap-only-require-fine-overlap`.
  - behavior: pure AABB-overlap candidates must have `0.05m` fine-cell support
    before they can be merged. AABB remains a recall prefilter, not a merge
    proof.
  - result: `50000 -> 48863` patches, `411277` boundary points moved,
    `911` merge accepts, `918` merge rejects.
  - reject reasons: `813` pure-overlap candidates rejected as
    `overlap_only_without_fine_overlap`; `105` rejected by normal
    gain/annealing.
  - merge source: accepted merges were `891` from `adjacency+overlap`, `13`
    from `adjacency+overlap+fine_overlap`, and `7` from plain adjacency. No
    pure AABB-only merge was accepted.
  - top-1000 diagnostic at `0.05m`: AABB overlap pairs increased from `2055`
    to `2168` because more patch boundaries were preserved, but fine-cell
    high-overlap pairs (`fine_ratio_min_cells >= 0.5`) dropped from `65` to
    `51`; `fine_ratio >= 0.95` remained `0`.
  - Interpretation: v6 is more faithful to the spatial-partition invariant than
    v4/v5. AABB metrics alone look slightly worse, but the stronger occupied
    cell metric improves. Use v6 as the current decision baseline unless visual
    QA shows unacceptable fragmentation.
- Patch-to-object candidate report on v6:
  - script: `propose_geo_patch_object_merges.py`.
  - purpose: report adjacent, visually/geometrically compatible patch pairs for
    the Object-building stage without changing patch ownership.
  - output: `dense_las_voxel003_energy_v6_fine_gated_overlap_20260624/object_merge_candidates_v1`.
  - input: v6 labels and the dense LAS region input.
  - result: `48863` patches, `22921` adjacent patch pairs, `273` high-score
    object-merge candidates.
  - main reject reasons: `21694` small patch pairs, `481` bucket mismatch,
    `240` color distance, `104` low shared edges, `70` score, `44` low contact
    ratio, `15` stable normal mismatch.
  - candidate geometry pairs: `mixed+mixed` (`101`), `mixed+rough_mixed`
    (`59`), `mixed+unknown` (`45`), `rough_mixed+vertical` (`26`),
    `unknown+vertical` (`15`), plus smaller stable pairs.
  - `181 / 273` candidates are big-mixed-attachment cases. These should not be
    auto-merged blindly; they are a review or stricter object-building class.
    The clean automatic object-building set should start with same-scale,
    same/stable-bucket, high-contact candidates and keep big mixed attachments
    separate until visual or structural evidence supports absorption.
- Object v1 from clean candidates:
  - script: `build_geo_patch_objects_from_candidates.py`.
  - output: `dense_las_voxel003_objects_v1_clean_candidates_20260624`.
  - input: v6 labels plus `object_merge_candidates_v1`.
  - invariant: voxel ownership remains exclusive; the stage only remaps patch
    ids to object ids through accepted candidate unions.
  - default gates: score `>=0.78`, contact ratio `>=0.08`, shared edges
    `>=32`, color distance `<=55`, bbox gap `<=0.08m`, and no automatic
    `big_mixed_attachment` merge.
  - result: `48863` input patches, `273` candidate rows, `16` accepted
    candidate rows, `48847` output objects, `1440583` preview points at
    stride `10`.
  - reject reasons: `181` big mixed attachments, `71` score, `2` stable normal,
    `2` contact ratio, `1` color distance.
  - Interpretation: the clean automatic object-building space is intentionally
    small. This protects the spatial-partition invariant but does not solve
    fragmentation. The next meaningful object stage should handle big-mixed
    attachments with local structural evidence and visual/depth evidence, not
    by globally lowering thresholds.
- High-recall object candidates on v6:
  - output:
    `dense_las_voxel003_energy_v6_fine_gated_overlap_20260624/object_merge_candidates_v2_high_recall`.
  - settings: `min_patch_voxels=80`, `min_shared_edges=4`,
    `min_contact_ratio=0.008`, `max_color_distance=110`,
    `min_bucket_score=0.45`, `min_score=0.55`.
  - result: same `48863` patches and `22921` adjacent patch pairs, but candidate
    count increased from `273` to `1905`.
  - reject reasons: `16645` small patch, `2646` bucket mismatch, `882` color
    distance, `642` low shared edges, `92` score, `67` stable normal mismatch,
    `42` low contact ratio.
  - `1046 / 1905` candidates are big-mixed attachments.
  - Interpretation: the original candidate stage had low recall. Loosening
    candidate generation is useful as long as acceptance remains conservative.
- Object v3 from high-recall candidates:
  - output: `dense_las_voxel003_objects_v3_high_recall_clean_20260624`.
  - input: v6 labels plus high-recall candidate set.
  - acceptance: same conservative Object v1 gates; `big_mixed_attachment` is
    still blocked.
  - result: `1905` candidate rows, `136` effective accepted unions, `48863 ->
    48727` objects, `1440583` preview points at stride `10`.
  - reject reasons: `1046` big mixed attachments, `644` score, `57` shared
    edges, `14` contact ratio, `4` color distance, `3` stable normal.
  - Interpretation: candidate recall matters, but the dominant unresolved class
    is now big mixed attachment. The next stage should model attachments to
    large mixed surfaces explicitly instead of globally relaxing object gates.
- Attachment-specific object test on 4090D r4 labels:
  - input: `geo_patch_las_opt_cpp_v2_voxel003_r4_4090d_20260623`, using
    `_cpp_region_grower_input.bin` and `_cpp_region_grower_labels.bin`.
  - tiny-fragment candidate output:
    `geo_patch_las_opt_cpp_v2_voxel003_r4_4090d_20260623/object_merge_candidates_v3_tiny_attach_recall`.
  - candidate settings: `min_patch_voxels=1`, `min_shared_edges=1`,
    `min_contact_ratio=0.001`, `max_color_distance=120`,
    `min_bucket_score=0.40`, `min_score=0.50`.
  - candidate result: `200535` patches, `10723` adjacent pairs, `4195`
    candidates, `95` big-mixed attachments. The previous high-recall run with
    `min_patch_voxels=80` only yielded `61` candidates and `7` big-mixed
    attachments, so the small-patch filter was suppressing the exact class this
    stage needs to handle.
  - clean baseline output:
    `geo_patch_las_opt_cpp_v2_voxel003_r4_4090d_20260623/objects_v4_tiny_recall_clean`.
    Result: `82` accepted rows, `200535 -> 200453` objects, and all `95`
    big-mixed attachments blocked.
  - strict attachment output:
    `geo_patch_las_opt_cpp_v2_voxel003_r4_4090d_20260623/objects_v5_attachment_model`.
    Result: same as clean (`82` accepted rows); strict defaults rejected `93`
    attachments by score and `2` by size ratio.
  - relaxed attachment output:
    `geo_patch_las_opt_cpp_v2_voxel003_r4_4090d_20260623/objects_v6_attachment_model_relaxed`.
    Settings: `attachment_min_score=0.76`, `attachment_min_contact_ratio=0.10`,
    `attachment_min_shared_edges=1`, `attachment_max_color_distance=65`,
    `attachment_min_normal_score=0.45`.
    Result: `103` accepted rows, including `21` `accepted_attachment`, and
    `200535 -> 200432` objects.
  - Interpretation: attachment-specific gating works and avoids global
    threshold relaxation, but the current r4 patch labels are still too
    fragmented (`200k` patches). The next improvement should move small-fragment
    attachment evidence earlier into patch candidate generation or seed growth,
    rather than relying on object-stage union alone.
- Patch-stage attachment merge:
  - script: `optimize_patch_graph_energy.py`.
  - output:
    `geo_patch_las_opt_cpp_v2_voxel003_r4_4090d_20260623/energy_attach_v1_patch_stage`.
  - change: `--enable-attachment-merge` lets merge step process small fragment
    candidates before the normal `min_anchor_voxels` filter, but only through
    the attachment-specific gate. General patch merges are unchanged.
  - settings: one iteration, no split/boundary, `min_anchor_voxels=900`,
    attachment defaults (`score>=0.76`, contact ratio `>=0.10`,
    shared edges `>=1`, color distance `<=65`, normal score `>=0.45`).
  - result: `200535 -> 200369` patches, `166` merge accepts, `10544` merge
    rejects, preview points `1448243`.
  - merge log: `160` accepts were `accepted_attachment`; `6` were ordinary
    adjacency merges. Main attachment rejects were color distance (`5393`),
    anchor too small (`4256`), score (`408`), normal (`290`), bucket (`156`),
    contact ratio (`31`), and size ratio (`6`).
  - Interpretation: moving attachment into patch-stage labels works. The effect
    is still modest because this r4 source is extremely fragmented and many
    fragments attach to anchors below `100000` voxels or differ strongly in
    color. This is now the correct place to iterate: improve attachment seed
    eligibility and local color/normal evidence, instead of doing object-stage
    post-hoc union.
- Patch-stage attachment merge with contact-local evidence:
  - output:
    `geo_patch_las_opt_cpp_v2_voxel003_r4_4090d_20260623/energy_attach_v4_contact_evidence`.
  - code change: adjacency candidates now carry contact-local color and normal
    evidence computed from the actual shared edge endpoints.  Attachment gating
    uses this local contact evidence instead of only comparing the fragment with
    the large anchor's global mean color/normal.
  - reason: a large patch can legitimately contain several local color/normal
    modes.  Rejecting a tiny fragment because it differs from the whole-anchor
    mean is mathematically wrong; the decision should ask whether the fragment
    matches the local boundary it touches.
  - settings: same v1 patch-stage command plus the default
    `--attachment-use-contact-evidence`.
  - result: `200535 -> 197630` patches, `2905` merge accepts, `7787` merge
    rejects, preview points `1448243`.
  - merge log: `2899` accepts were `accepted_attachment`; `6` were ordinary
    adjacency merges. Main rejects were anchor too small (`4238`), contact
    color distance (`2013`), score (`717`), contact normal (`622`), bucket
    (`156`), contact ratio (`31`), and size ratio (`6`).
  - Interpretation: contact-local evidence is the first patch-stage change that
    materially reduces tiny-fragment count without globally relaxing all graph
    edges.  Use this as the current review candidate, but visual QA is still
    required because the accept count is much higher than v1 and could expose
    local over-attachment in cluttered areas.
- Object-stage structural multimaterial merge:
  - scripts:
    - `propose_geo_patch_object_merges.py`
    - `build_geo_patch_objects_from_candidates.py`
  - code change: object merge candidates can now be generated from complete
    `grid6` voxel adjacency instead of only the region-grower edge graph.
    Candidate rows also carry contact-local color/normal evidence and a
    `structural_multimaterial` class for same-object joins where material or
    color legitimately changes across a boundary.
  - reason: examples such as car body parts or window/facade boundaries are not
    same-material merges.  Treating color similarity as mandatory makes the
    object layer over-fragment even when the geometry is spatially continuous.
  - region-edge baseline:
    `object_merge_candidates_v4_structural_multimaterial` found only `7793`
    edge pairs and `194` candidates; object output
    `objects_v7_structural_multimaterial_from_attach_v4` accepted `123` rows
    and produced `197507` objects.
  - grid6 result:
    `object_merge_candidates_v5_grid6_structural_multimaterial` found `48786`
    edge pairs and `4363` candidates; object output
    `objects_v8_grid6_structural_multimaterial_from_attach_v4` accepted `3547`
    effective union rows and produced `194083` objects from `197630` patch
    labels.
  - diagnostic: only `240 / 194083` objects contain more than one source patch
    and p99 `patch_count` is still `1`.  The candidate graph fix is real, but
    the dominant remaining issue is still the huge singleton/small-patch
    population.  The next stage should handle small singleton absorption or
    coarsened supernodes explicitly; simply lowering color/score thresholds is
    not the right lever.
- Geometry-first SAM semantic vote smoke:
  - script: `accumulate_semantic_png_votes_to_objects.py`.
  - purpose: push SKYMASK/SAM semantic PNGs back onto the geometry-first object
    layer without changing object ownership.  SAM evidence updates only
    `semantic_label`; patch/object boundaries remain fixed by the spatial
    segmentation route.
  - input object baseline:
    `objects_v9_grid6_samegeom_structural_guard`.
  - semantic source smoke:
    `/root/epfs/sam2_tensorrt/semantic_eval_rle50_default_downstream50_cam0`,
    combo `sam2_prompt_v3_sky_label_merge_completion`, cam0 only, 50 frames.
  - output:
    `geo_patch_las_opt_cpp_v2_voxel003_r4_4090d_20260623/objects_v9_sam_vote_cam0_50_smoke_v2`.
  - result: `197583` objects, `9060` changed by SAM votes, `50` frames used,
    `8269039` projected visible samples, `4991864` accepted geometry-guarded
    semantic votes.
  - point label counts on stride10 preview: floor `676046`, wall `399934`,
    unknown `219850`, equipment `141857`, railing `8553`, other `1894`, building
    `69`, pipe `40`.
  - guardrail: horizontal/vertical/thin/rough geometry each has an allowed label
    set, so SAM cannot directly relabel a horizontal surface as car/railing or a
    stable vertical surface as floor.  Vetoed labels are kept in
    `semantic_veto_votes` for QA instead of being silently applied.
  - Interpretation: this is the correct way to reintroduce SKYMASK/SAM: as
    object-level evidence after geometry ownership is fixed.  It is still only a
    cam0/50-frame smoke; full production needs all cameras and a cached/binary
    vote path to avoid repeated ASCII PLY projection cost.
- Official-Superpoint multiview evidence mainline:
  - ownership source: `run_official_superpoints_patch.py` on the authoritative
    same-order `dense_las_voxel003_binary` input.  A Superpoint id is permanent
    for one run and is the only owner of its points; image evidence cannot
    split, absorb, or relabel neighbouring Superpoints.
  - evidence source: `accumulate_semantic_png_votes_to_objects.py` projects
    each camera pose through the validated calibration chain, first-touch
    z-buffer, and SKYMASK/SAM semantic PNG.  It writes
    `<output-stem>_observations.jsonl`; one row is one
    `(patch_id, frame_id, cam_id)` observation with visible support, accepted
    semantic pixels, and geometry-vetoed pixels.
  - posterior rule: label pixels are normalized inside each camera observation.
    A view contributes at most one unit of evidence, after reaching
    `--min-observation-points`; independent viewpoints therefore outweigh a
    single close-up with many projected pixels.  The object result retains
    `semantic_candidate_label`, `semantic_candidate_ratio`,
    `semantic_observation_count`, and `semantic_evidence_weight` even when
    evidence is too sparse for a hard label.
  - operational consequence: use the observation ledger as the only input to
    later Object/Zone graph construction and VLM review.  Do not create a
    second label-first target route or allow VLM to rewrite Superpoint
    ownership.
  - canonical 2D producer: `semantic_eval/` now lives inside this repository.
    A VLM transport or parse failure writes `unknown`, never `other`; otherwise
    a failed API credential can silently become false semantic evidence.
  - source-frame contract: the official LAS-derived Superpoint PLY does not
    retain raw `.lx` frame provenance.  `build_superpoint_frame_provenance.py`
    restores a conservative sidecar by matching raw section points to the
    immutable reference cloud (`<=0.05m`) and recording each Superpoint's
    strongest source frames.  Evidence selection must use this sidecar before
    first-touch depth gating.  A global point that projects inside an unrelated
    frame but fails that frame's first-touch depth test is correctly occluded;
    increasing the depth tolerance would reintroduce through-wall evidence.
  - validation (2026-07-14): all `6,180` raw sections produced `28,749,815`
    conservative matches out of `97,855,095` raw points (29.38%), supporting
    `17,164` official Superpoints.  A stratified 20-object review then reached
    `20/20` objects with source-frame first-touch image evidence.  Local Qwen
    returned strict JSON for `20/20` reviews when
    `VLM_DISABLE_THINKING=1`; its free-form description and controlled label
    can still disagree (for example, a description of an indoor ceiling with a
    `floor` label).  Descriptions are evidence; geometry remains the canonical
    structural-label arbiter.
  - contact graph validation (2026-07-14):
    `export_official_superpoint_contact_graph.py` rebuilt 6-neighbor contacts
    from the same `14,482,557` 3cm reference voxels and emitted `19,357`
    cross-Superpoint pairs in 16.5 seconds.  The corrected grid builder rejects
    linear-key row-wrap edges.  The graph has `12,680` connected nodes but a
    largest component of `10,377`; true contact alone is therefore not a
    propagation permission.  The posterior must retain only edges with
    structural and appearance compatibility, and must never propagate fine
    object labels over the component.
  - anchor validation (2026-07-14): a source-supported, geometry-stratified
    100-node sample initially had only 23% reviewable image evidence because
    most tiny Superpoints cannot form a stable crop.  Restricting VLM review to
    nodes with at least 500 reference voxels raised source-frame + first-touch
    + SKYMASK evidence coverage to `90/100`; all 90 Qwen outputs parsed.  Of
    these, only 20 were marked as a structural surface fragment and became
    graph anchors (`15 building_part`, `4 floor`, `1 grass`).  The remaining
    70 descriptions stay local-only.  Global PCA geometry types are not hard
    label vetoes: car panels, glass doors, vegetation crowns, and railings can
    all be locally planar or linear.
- Planned graph posterior, after a valid multiview semantic set exists:
  - node: one official Superpoint.  Node ownership is immutable; neither label
    propagation nor a VLM may merge nodes or move points across a boundary.
  - edge: only a measured dense-voxel contact or an edge from the same original
    partition graph.  Centroid-neighbour edges are prohibited: they connect
    nearby but disconnected walls, railings, vehicles, and vegetation.
  - edge weight: the first production pass uses measured contact and boundary
    color only: `w_ij = min(shared_faces/100, 1) * exp(-0.5*(rgb_delta/40)^2)`.
    Edges below 10 shared faces have zero weight.  This is deliberately local:
    a label score is the best product of edge weights over paths of at most two
    hops from a structural anchor.  It is not an unbounded graph smoother.
    `propagate_superpoint_structural_anchors.py` records the winning source
    anchor and hop count for every posterior.
  - deferred upgrade: after a manually calibrated anchor set exists, compare
    the bounded result against the constrained harmonic objective
    `sum_i alpha_i ||p_i-y_i||^2 + lambda sum_(i,j) w_ij ||p_i-p_j||^2`.
    Until then, solving it over the 10k-node contact component would hide an
    unmeasured propagation radius behind a neat equation.
  - promotion: only specific structural labels (`floor/wall/grass/roof/ceiling/stair`)
    may propagate, and only when posterior confidence and margin both exceed
    review thresholds.  `building_part` is intentionally not specific enough:
    it enters a second structural-refinement review rather than propagating.
    Fine labels (`person/car/railing/pipe/equipment`) also stay as local
    candidates because propagating them over a contact graph creates exactly
    the large false-positive regions seen in earlier runs.
  - structural refinement: reuse `run_mimo_object_review.py --task structure`
    only on high-confidence `building_part` surface fragments.  This narrow
    review may return `wall/roof/ceiling/stair/floor/grass`; only then can the
    existing anchor/posterior pipeline consider the result for propagation.
    The broad first-pass review remains untouched and no completed VLM work is
    discarded.
  - observation materialization (2026-07-14):
    `materialize_superpoint_observation_ledger.py` is the canonical export for
    source-supported image evidence.  It does not recompute projection or
    create a second semantic route: it flattens the existing accepted
    `(Superpoint, frame, camera)` rows with geometry, first-touch/SKYMASK
    metrics, artifact paths, and any completed VLM review.  A partial run over
    381 review candidates produced 713 observations with a 100% source-frame
    confirmation ratio.  Its 36 structural anchors promoted 262 nodes under
    the bounded two-hop rule.  This is a calibration artifact, not a production
    semantic map: `building_part` dominates the provisional anchors and must
    remain a structural candidate until object-level QA validates its meaning.

## Scene Prior Baseline

## External Method Boundary

The mainline intentionally adopts the useful boundary from three related
methods, without importing their unnecessary training/runtime surface:

- Superpoint Graph partitions a large cloud into geometrically homogeneous
  elements and performs semantic reasoning over their graph.  Our immutable
  official Superpoints and measured voxel-contact graph are that representation;
  replacing them with a second heuristic patch generator is a regression.
- Superpoint Transformer adds learned multi-scale attention.  It is not the
  next production step because no task-specific 3D labels exist to calibrate
  such a model.  The bounded posterior is a transparent, conservative
  substitute until a reviewed anchor set is large enough for learning.
- OVI-MAP separates class-agnostic instance reconstruction from open-vocabulary
  semantic inference.  This is the decisive local rule: geometric ownership is
  fixed before VLM evidence arrives.  A VLM cannot move a point across a
  Superpoint boundary.
- HOV-SG builds hierarchy after segment-level mapping.  For this mixed parking
  scene the useful hierarchy is `scene -> spatial_region -> structure/object
  -> superpoint -> observation`, not its indoor-only `floor -> room -> object`
  naming.  Region nodes must be built only from QA-approved structural
  posteriors; local open-vocabulary candidates stay attached to their
  Superpoints until then.

This keeps the first production version small: one dense geometric ownership
map, one evidence ledger, and one contact graph.  No dense per-point language
feature store, graph neural network, or second segmentation pipeline is needed
to test the next decision.

The first usable route-level prior is a `30:1` cam0 sample of the parking
scan, generated on the local Qwen VL server from `207` frames. It identifies
entrance plaza, outdoor parking, landscape, indoor lobby, stairwell, and roof
segments. The authoritative run artifact is
`/root/epfs/work_MT20260616-175807/scene_prior_qwen30_v2_20260714/mimo_scene_prior.json`
on `scan-vlm`.

The server must be called with `VLM_DISABLE_THINKING=1`; otherwise Qwen may
consume the output budget in reasoning and truncate the required JSON. Both
`build_mimo_scene_prior.py` and `run_mimo_object_review.py` now pass the
OpenAI-compatible `chat_template_kwargs.enable_thinking=false` only when that
environment flag is set.

This prior is deliberately weak evidence. Its time ranges may overlap, for
example a grass landscape inside an outdoor-parking route segment. It may
select candidate labels or veto implausible ones, but cannot move a voxel,
merge Superpoints, or turn a local visual observation into a hard label.
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
