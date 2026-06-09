# Semantic Route Status - 2026-06-09

## Current Main Route

Authoritative semantic point-cloud route:

- `scripts/project_color.py`
- `scripts/project_semantic.py`
- `scripts/build_targets_from_masks.py`
- `scripts/fuse_targets_to_objects.py`
- `scripts/assign_residuals_to_surface_objects.py`
- `scripts/build_consolidated_object_ply.py`

Deprecated semantic correctness route:

- `MT20260511-165822/semantic_pointcloud_pipeline/project_frame_local_semantics.py`
- `transforms.json + project_world_points()` semantic projection branch

The deprecated route remains useful only for visual colorization comparison.

## 1000 Frame Dataset State

2D semantic source:

- combo: `sam2_prompt_v3_sky_label_merge_completion`
- images: `3000 / 3000`
- frames: `0-999`

Correct-route semantic projection:

- output: `/root/epfs/new_route_stage1_skymask/semantic_projection_0000_0999_completion_correct_route`
- frames: `1000 / 1000`
- average per-frame labeled ratio: `0.9434`
- merged labeled ratio: `0.9406`

Target/object fusion:

- output: `/root/epfs/new_route_stage1_skymask/target_object_fusion_0000_0999/objects_status_fixed`
- targets: `34,252`
- objects: `2,978`
- stable objects: `1,785`
- ambiguous objects: `207`
- single-target objects: `986`
- merge ratio: `0.9131`

Consolidated QA PLY:

- output: `/root/epfs/new_route_stage1_skymask/consolidated_object_qa_0000_0999/consolidated_object_qa_0000_0999.ply`
- validation: `ok`
- points: `8,374,961`
- target object points: `6,993,947`
- absorbed residual points: `1,381,014`

## Residual Absorption Findings

Default surface absorption:

- params: `bbox=0.35m, plane=0.12m, color=70`
- residual points: `2,990,515`
- absorbed points: `1,381,014`
- absorbed ratio: `0.4618`
- absorbed labels:
  - `floor`: `1,240,766`
  - `building`: `116,226`
  - `wall`: `24,022`

Parameter sweep:

| Params | Absorbed | Ratio | Main remaining residual |
|---|---:|---:|---|
| `bbox=0.35, plane=0.12, color=70` | `1,381,014` | `0.4618` | building, floor, equipment |
| `bbox=0.50, plane=0.12, color=90` | `1,462,531` | `0.4891` | building, floor, equipment |
| `bbox=0.50, plane=0.20, color=90` | `1,529,355` | `0.5114` | building, floor, equipment |
| `bbox=0.80, plane=0.20, color=110` | `1,595,225` | `0.5334` | building, floor, equipment |

Interpretation:

- Loosening surface thresholds mostly absorbs more `floor/building/wall`.
- `equipment` and `railing` are not absorbed by surface rules because label compatibility blocks them.
- A loose surface-absorption QA PLY is worth generating only after checking the current consolidated PLY in CloudCompare.

## Current Bottlenecks

1. Building/wall/floor ambiguity dominates large ambiguous objects.
2. Large single-target objects indicate missed cross-frame object merge opportunities.
3. Remaining equipment/railing residuals should not be solved by surface absorption; they need fine-object clustering/review.
4. 2D-to-point coverage is already high enough for this stage, so the next gains should come from object fusion and residual absorption rules.

## Fine Residual Clustering

Fine-object residual clustering was run on unassigned `equipment` and `railing`
points from `residual_surface_assigned_0000_0999.ply`.

Output:

- `/root/epfs/new_route_stage1_skymask/fine_residual_clusters_0000_0999/fine_residual_clusters_equipment_railing.ply`
- `/root/epfs/new_route_stage1_skymask/fine_residual_clusters_0000_0999/fine_residual_clusters_report.json`

Result:

- selected fine residual points: `286,677`
- clustered points: `264,828`
- clusters: `124`
- small-cluster residual: `21,849`
- `equipment`: `201,070 / 214,092` clustered
- `railing`: `63,758 / 72,585` clustered

Interpretation:

- Fine residuals are mostly structured enough for a second-stage object review path.
- The largest `equipment` cluster has `118,365` points and a very large bbox, so it is likely a mixed or misclassified region rather than one equipment object.
- Fine-object handling should therefore split/review large clusters before merging them into stable objects.

Fine-clustering parameter sweep:

| Params | Clusters | Clustered ratio | Small points | Largest cluster |
|---|---:|---:|---:|---:|
| `voxel=0.16, min=80` | `62` | `0.9531` | `13,443` | `150,072 equipment` |
| `voxel=0.12, min=50` | `124` | `0.9238` | `21,849` | `118,365 equipment` |
| `voxel=0.08, min=40` | `320` | `0.8395` | `46,004` | `35,462 equipment` |
| `voxel=0.06, min=30` | `543` | `0.7414` | `74,144` | `19,111 railing` |

Recommended QA setting:

- `voxel=0.08, min_cluster_points=40`

Reason:

- It breaks the largest `equipment` band from `118k` to `35k` points.
- It does not fragment the data as aggressively as `voxel=0.06`.
- The remaining large orange `equipment` regions in the XY preview still look like likely label/projection contamination, so they should be reviewed rather than blindly accepted.

Fine-cluster review set:

- output: `/root/epfs/new_route_stage1_skymask/fine_residual_review_0000_0999_v008`
- local copy: `/Users/skkac/Work/SCAN/server_fine_cluster_review_v008`
- review rows: `100`
- suspicious clusters: `12`
- likely fine objects: `8`
- suspicious PLY points: `134,687`

Top suspicious clusters:

| Cluster | Label | Points | Main reasons |
|---:|---|---:|---|
| `97` | `equipment` | `35,462` | large points, large XY span, surface-like geometry |
| `98` | `equipment` | `15,085` | large points, large XY span, surface-like geometry |
| `99` | `equipment` | `13,457` | large points, large Z span, surface-like geometry |
| `1` | `railing` | `22,324` | large points, large XY span |
| `2` | `railing` | `16,981` | large points, large XY span, linear railing-like geometry |

Interpretation:

- The suspicious preview is structured, not random noise.
- Large `equipment` clusters are likely mixed surface/edge projection contamination.
- Large `railing` clusters preserve useful line geometry but still need object-level review before being accepted as one global object.
- The next useful step is to review suspicious cluster masks/images, then either split them or demote contaminated parts back to surface/residual.

Fine-cluster mask trace:

- output: `/root/epfs/new_route_stage1_skymask/fine_residual_trace_0000_0999_v008`
- local copy: `/Users/skkac/Work/SCAN/server_fine_residual_trace_v008`
- traced suspicious clusters: `12`
- traced points: `134,687 / 134,687`
- raw duplicate point matches: `3,836`
- contact sheet: `/Users/skkac/Work/SCAN/server_fine_residual_trace_v008/fine_cluster_mask_trace_contact_sheet.png`

Trace interpretation:

- The suspicious clusters trace back to many consecutive frame/camera/mask observations, not one isolated bad frame.
- Top source masks usually explain only `0.5% - 3%` of a large cluster, which means the large 3D cluster is accumulated from repeated over-large 2D masks.
- Overlay review shows several masks include surface, railing edge, and equipment/background together.
- This shifts the immediate bottleneck from 3D object fusion to pre-fusion mask hygiene: large-mask splitting and stable-surface subtraction should run before accepting fine-object clusters.

Top ambiguous examples are listed in:

- `/root/epfs/new_route_stage1_skymask/consolidated_object_qa_0000_0999/object_pipeline_qa_summary.json`

## New Model Status

ConceptSeg-R1/SAM3 smoke:

- problem sample outputs: `40 / 40`
- QA mean non-black ratio: `0.6649`
- visual result: category responses exist but are not stable enough to replace SAM2 main route.

Current use:

- keep as second-stage fine-object candidate only.
- do not replace `sam2_prompt_v3_sky_label_merge_completion` in the main pipeline yet.

## Old Route Status

Old world-fused visual color smoke on server:

- sections: `8`
- source points: `64,437`
- fused points: `31,323`
- colored points: `27,613`
- colored ratio: `0.8816`

Use:

- valid as visual colorization comparison.
- not valid as semantic correctness route.

## Next Steps

1. Inspect current consolidated QA PLY in CloudCompare.
2. If surface regions look under-absorbed and fine objects are not being swallowed, generate a loose absorption QA variant with:
   - `bbox=0.80`
   - `plane=0.20`
   - `color=110`
3. Review top ambiguous large objects, especially floor/wall and building/railing conflicts.
4. Review the `fine_residual_review_0000_0999_v008` suspicious clusters against source masks/images.
5. Add a pre-fusion hygiene step for oversized masks:
   - subtract known stable surface projections first
   - split mask projections by 3D connected components
   - reject/demote surface-like fragments before fine-object clustering
6. Add a fine-object residual path for accepted equipment/railing clusters instead of merging them into surfaces.
7. Keep ConceptSeg-R1 as a small-sample second-stage experiment until it has stable binary masks.
