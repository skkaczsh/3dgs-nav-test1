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
- top source rows: `96`
- source mask area mean/median/max: `0.1143 / 0.0447 / 0.6854`
- source mask bbox area mean/median/max: `0.2101 / 0.0918 / 1.0000`
- source rows with mask area `>= 10%`: `21 / 96`
- source rows with bbox area `>= 30%`: `20 / 96`
- contact sheet: `/Users/skkac/Work/SCAN/server_fine_residual_trace_v008/fine_cluster_mask_trace_contact_sheet.png`

Trace interpretation:

- The suspicious clusters trace back to many consecutive frame/camera/mask observations, not one isolated bad frame.
- Top source masks usually explain only `0.5% - 3%` of a large cluster, which means the large 3D cluster is accumulated from repeated over-large 2D masks.
- Overlay review shows several masks include surface, railing edge, and equipment/background together.
- The largest traced masks cover `64% - 68%` of the image with full-image bounding boxes, so at least part of the fine residual pollution is created before 3D fusion.
- This shifts the immediate bottleneck from 3D object fusion to pre-fusion mask hygiene: large-mask splitting and stable-surface subtraction should run before accepting fine-object clusters.

Full-source oversized mask hygiene evaluation:

- output: `/root/epfs/new_route_stage1_skymask/fine_residual_trace_fullsources_0000_0999_v008`
- local copy: `/Users/skkac/Work/SCAN/server_fine_residual_trace_fullsources_v008`
- traced source rows: `2,135`
- source mask area mean/median/max: `0.0368 / 0.0084 / 0.8119`
- source mask bbox area mean/median/max: `0.0755 / 0.0184 / 1.0000`
- source rows with mask area `>= 10%`: `220 / 2,135`
- source rows with bbox area `>= 30%`: `136 / 2,135`
- action counts:
  - `pre_fusion_split_or_demote`: `2`
  - `manual_review`: `8`
  - `fine_object_candidate`: `2`

Cluster-level hygiene decisions:

| Cluster | Label | Points | Oversized source point share | Action |
|---:|---|---:|---:|---|
| `1` | `railing` | `22,324` | `0.848` | `pre_fusion_split_or_demote` |
| `2` | `railing` | `16,981` | `0.966` | `pre_fusion_split_or_demote` |
| `107` | `equipment` | `5,104` | `0.018` | `fine_object_candidate` |
| `122` | `equipment` | `1,233` | `0.000` | `fine_object_candidate` |

Interpretation:

- The two largest railing clusters are dominated by oversized masks and should be handled before object fusion.
- Most equipment suspicious clusters are not dominated by oversized masks; they are accumulated from many smaller source masks.
- For equipment, the next improvement is not broad mask deletion. It should be 3D connected splitting plus color/PCA consistency checks.
- For railing, the next improvement is pre-fusion stable-surface subtraction and mask/component splitting.

Oversized mask hygiene QA output:

- output: `/root/epfs/new_route_stage1_skymask/fine_residual_hygiene_0000_0999_v008`
- local copy: `/Users/skkac/Work/SCAN/server_fine_residual_hygiene_v008`
- status PLY: `fine_residual_clusters_hygiene_status_v008.ply`
- filtered PLY: `fine_residual_clusters_hygiene_filtered_v008.ply`
- total fine residual cluster points: `240,673`
- demoted points: `39,305`
- demoted ratio: `0.1633`
- kept points: `201,368`
- demoted clusters: `1`, `2`
- status counts:
  - `pre_fusion_split_or_demote`: `39,305`
  - `manual_review`: `89,045`
  - `fine_object_candidate`: `6,337`
  - `other`: `105,986`

QA interpretation:

- The status preview localizes the demoted points to two large railing-pollution regions.
- The filtered preview removes those regions while preserving compact equipment candidates.
- Remaining magenta/manual regions are still substantial, so the next step should split equipment by 3D geometry/color rather than accept all equipment residuals.

Manual equipment split QA:

- output: `/root/epfs/new_route_stage1_skymask/manual_equipment_split_0000_0999_v008`
- local copy: `/Users/skkac/Work/SCAN/server_manual_equipment_split_v008`
- input manual equipment points: `89,045`
- source clusters: `8`
- subclusters: `68`
- clustered points: `71,743`
- small residual points: `17,302`
- split params:
  - voxel: `0.06m`
  - max visual RGB distance: `45`
  - min subcluster points: `80`

Manual equipment subcluster review:

- `fine_candidate`: `61` subclusters, `42,467` points
- `linear_edge_review`: `6` subclusters, `22,226` points
- `large_mixed_review`: `1` subcluster, `7,050` points

Interpretation:

- The first manual-equipment split turns most subclusters into compact candidates by count.
- A large fraction of points still sits in linear/edge or large-mixed review buckets.
- The next QA baseline should combine:
  - existing hygiene `fine_object_candidate` clusters
  - manual-equipment `fine_candidate` subclusters
  - excluding `linear_edge_review` and `large_mixed_review`

Accepted fine-object QA:

- output: `/root/epfs/new_route_stage1_skymask/accepted_fine_object_qa_0000_0999_v008`
- local copy: `/Users/skkac/Work/SCAN/server_accepted_fine_object_qa_v008`
- candidate count: `63`
- accepted points: `48,804`
- candidate sources:
  - hygiene clusters: `2` candidates, `6,337` points
  - manual equipment subclusters: `61` candidates, `42,467` points

Accepted QA interpretation:

- The accepted preview is substantially cleaner than the raw fine residual and hygiene-filtered previews.
- The two large railing-pollution clusters are excluded.
- Most accepted candidates are compact manual-equipment subclusters.
- Two hygiene-derived candidates remain line-like (`100107`, `100122`), so the accepted set still needs a stricter geometry review before object fusion.

Strict accepted fine-object QA:

- output: `/root/epfs/new_route_stage1_skymask/accepted_fine_object_strict_qa_0000_0999_v008`
- local copy: `/Users/skkac/Work/SCAN/server_accepted_fine_object_strict_qa_v008`
- input candidates: `63`
- input points: `48,804`
- kept strict candidates: `61`
- kept strict points: `42,467`
- demoted candidates: `2`
- demoted points: `6,337`
- demoted candidate ids: `100107`, `100122`

Strict QA interpretation:

- The strict geometry review removes the two line-like hygiene-derived candidates.
- Both demoted candidates have long linear geometry:
  - `100107`: `5,104` points, span approximately `9.30m x 3.05m x 2.12m`, linearity `0.961`
  - `100122`: `1,233` points, span approximately `6.41m x 0.87m x 1.23m`, linearity `0.959`
- The strict filtered PLY is the current best fine-object QA baseline for object-fusion testing.

Fine-object fusion QA:

- default output: `/root/epfs/new_route_stage1_skymask/accepted_fine_object_fusion_0000_0999_v008`
- local copy: `/Users/skkac/Work/SCAN/server_accepted_fine_object_fusion_v008`
- input strict candidates: `61`
- input points: `42,467`
- default fine objects: `42`
- default merges: `19`
- status counts:
  - `stable_fine_object`: `13`
  - `single_fine_candidate`: `29`

Fine-object fusion parameter sweep:

| Variant | Centroid | Cross-source centroid | BBox | Color | Objects | Merges |
|---|---:|---:|---:|---:|---:|---:|
| `default` | `0.90` | `0.45` | `0.25` | `45` | `42` | `19` |
| `strict` | `0.60` | `0.30` | `0.10` | `35` | `44` | `17` |
| `strict2` | `0.45` | `0.25` | `0.05` | `30` | `47` | `14` |
| `strict3` | `0.30` | `0.20` | `0.02` | `25` | `50` | `11` |

Fine-object fusion interpretation:

- Object count is sensitive to merge thresholds.
- The default fusion is useful as a QA view but can merge candidates with centroid distances around `2-3m` when bboxes overlap.
- `strict2` is the current better baseline for conservative object-fusion testing.
- This remains a spatial fine-object fusion QA because strict accepted PLYs do not yet carry frame/time metadata.

Accepted fine-object metadata enrichment:

- output: `/root/epfs/new_route_stage1_skymask/accepted_fine_object_enriched_0000_0999_v008`
- local copy: `/Users/skkac/Work/SCAN/server_accepted_fine_object_enriched_v008`
- enriched PLY: `accepted_fine_object_enriched_v008.ply`
- points: `42,467`
- matched points: `42,467`
- unmatched points: `0`
- matched ratio: `1.0000`
- duplicate matches: `1,524`
- frame count: `766`
- matched candidates: `61 / 61`
- camera point counts:
  - `cam0`: `10,289`
  - `cam1`: `26,031`
  - `cam2`: `6,147`

Enrichment interpretation:

- The enriched PLY preserves the strict accepted fine-object geometry while adding `frame`, `camera`, `mask`, and `point_index`.
- All strict accepted points were traced back to residual diagnostic metadata.
- The next fine-object fusion can now become incremental / scan-order aware instead of spatial-only.

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
6. Build a combined accepted fine-object QA PLY from:
   - hygiene fine-object candidate clusters
   - manual equipment fine-candidate subclusters
7. Use the `strict2` fine-object fusion parameters as the next conservative object-fusion baseline.
8. Use `accepted_fine_object_enriched_0000_0999_v008` for the next incremental / scan-order fine-object fusion.
9. Keep `linear_edge_review`, `large_mixed_review`, and strict-demoted line-like candidates out of accepted object fusion until further split/relabel.
10. Keep ConceptSeg-R1 as a small-sample second-stage experiment until it has stable binary masks.

## Incremental Fine-Object Fusion

Scan-order fusion from enriched accepted fine points:

- output: `/root/epfs/new_route_stage1_skymask/accepted_fine_object_incremental_fusion_0000_0999_v008`
- local copy: `/Users/skkac/Work/SCAN/server_accepted_fine_object_incremental_fusion_v008`
- source: `accepted_fine_object_enriched_v008.ply`
- params:
  - `centroid_distance=0.45`
  - `cross_source_centroid_distance=0.25`
  - `bbox_distance=0.05`
  - `color_distance=30`
  - `active_frame_window=120`
  - `zone_size=100`
- candidates: `61`
- incremental fine objects: `47`
- points: `42,467`
- merges: `14`
- zones: `7`
- status:
  - `stable_incremental_fine_object`: `12`
  - `single_incremental_fine_candidate`: `35`

Interpretation:

- The result matches the previous `strict2` spatial-fusion baseline: `47` objects and `14` merges.
- The scan-order constraint does not materially change the result because the enriched input candidates were already formed by global residual clustering before frame metadata was attached.
- Several candidates span hundreds of frames by themselves, for example a top object spans `0-999`; therefore this is not yet a true online target/object pipeline.
- To validate the original incremental object-building idea, the next implementation should build per-frame or short-window `Target` records first, then fuse those targets into objects. Do not reuse globally clustered fine candidates as the only unit for scan-order fusion.

## Frame Fine-Target Fusion

Per-frame fine-target reconstruction from enriched accepted fine points:

- source: `/root/epfs/new_route_stage1_skymask/accepted_fine_object_enriched_0000_0999_v008/accepted_fine_object_enriched_v008.ply`
- selected baseline target output: `/root/epfs/new_route_stage1_skymask/frame_fine_targets_0000_0999_v008_sweep/v0.16_m3`
- local copy: `/Users/skkac/Work/SCAN/server_frame_fine_target_object_v008`
- params:
  - `voxel_size=0.16`
  - `min_target_points=3`
- source points: `42,467`
- groups `(frame,camera,mask,semantic)`: `2,571`
- targets: `3,164`
- target points: `36,970`
- small residual points: `5,497`
- target point stats:
  - min: `3`
  - max: `157`
  - mean: `11.68`

Parameter sweep:

- `v0.08_m3`: `3,880` targets, `24,786` target points, `17,681` residual points
- `v0.08_m5`: `2,033` targets, `18,490` target points, `23,977` residual points
- `v0.12_m3`: `3,559` targets, `33,372` target points, `9,095` residual points
- `v0.12_m5`: `2,285` targets, `29,059` target points, `13,408` residual points
- `v0.16_m3`: `3,164` targets, `36,970` target points, `5,497` residual points
- `v0.16_m5`: `2,252` targets, `33,866` target points, `8,601` residual points

Frame target fragmentation:

- The `61` strict accepted candidates expand into `3,164` frame-level targets.
- Average frame targets per accepted candidate: `51.87`.
- Worst cases:
  - candidate `200003`: `337` targets, `4,562` points, `117` frames, frames `354-798`
  - candidate `200033`: `307` targets, `4,369` points, `64` frames, frames `872-999`
  - candidate `200005`: `253` targets, `3,170` points, `156` frames, frames `256-825`

Object fusion from frame targets:

- online-like baseline output: `/root/epfs/new_route_stage1_skymask/frame_fine_object_fusion_0000_0999_v008_v016_m3`
  - params: `centroid=0.45`, `bbox=0.12`, `color=45`, `normal=180`, `active_zone_window=2`
  - targets: `3,164`
  - objects: `302`
  - merge ratio: `0.9046`
  - stable objects: `260`
  - single-target objects: `42`
- global no-time-window control: `/root/epfs/new_route_stage1_skymask/frame_fine_object_fusion_0000_0999_v008_v016_m3_global`
  - objects: `254`
  - merge ratio: `0.9197`
- loose global control: `/root/epfs/new_route_stage1_skymask/frame_fine_object_fusion_0000_0999_v008_v016_m3_loose_global`
  - params: `centroid=0.9`, `bbox=0.25`, `color=70`, `normal=180`, no time window
  - objects: `127`
  - merge ratio: `0.9599`

Interpretation:

- `0.16m / min 3` is a better Mid360 frame-target baseline than `0.08m / min 5`; the latter loses too many single-frame sparse points.
- Even without a temporal window, frame targets do not collapse back to the `47` strict2 global fine objects. The bottleneck is target fragmentation and missing cross-time re-identification / tracklet association.
- The next route should add an intermediate `Tracklet` layer:
  - frame target -> short-window tracklet by spatial/color continuity
  - tracklet -> object by longer-range re-identification
  - keep the global accepted-candidate result as QA reference only, not as online input.

## Short-Window Tracklet Prototype

Tracklet prototype from `v0.16_m3` frame targets:

- local copy: `/Users/skkac/Work/SCAN/server_frame_fine_tracklet_v008`
- server outputs:
  - `/root/epfs/new_route_stage1_skymask/frame_fine_tracklets_0000_0999_v008_v016_m3_gap10`
  - `/root/epfs/new_route_stage1_skymask/frame_fine_tracklets_0000_0999_v008_v016_m3_gap30`
  - `/root/epfs/new_route_stage1_skymask/frame_fine_tracklets_0000_0999_v008_v016_m3_gap60`
- tracklet params:
  - `centroid_distance=0.45`
  - `bbox_distance=0.12`
  - `color_distance=45`
  - `normal_angle=180`

Results:

- gap `10`:
  - `3,164` frame targets -> `519` tracklets
  - tracklet merge ratio: `0.8360`
  - tracklets -> objects: `160`
- gap `30`:
  - `3,164` frame targets -> `379` tracklets
  - tracklet merge ratio: `0.8802`
  - tracklets -> objects: `141`
- gap `60`:
  - `3,164` frame targets -> `328` tracklets
  - tracklet merge ratio: `0.8963`
  - tracklets -> objects: `135`

Interpretation:

- Tracklets materially reduce short-term target fragmentation: `3,164` frame targets become `328-519` tracklets depending on frame-gap tolerance.
- Tracklet-to-object fusion improves over direct strict frame-target object fusion (`302` objects) and global no-window frame-target fusion (`254` objects).
- Tracklets still do not recover the global strict2 fine-object count (`47` objects). The remaining gap is long-range re-identification, not short-window continuity.
- Next practical step is a second-stage long-range association over tracklets using stronger descriptors:
  - spatial bbox/centroid
  - visual RGB statistics
  - frame span and revisit pattern
  - original mask/camera evidence
  - optional VLM/ConceptSeg review only for conflicting high-value tracklet pairs

## Long-Range Tracklet Association

Long-range association over `gap60` tracklets:

- local copy: `/Users/skkac/Work/SCAN/server_frame_fine_long_assoc_v008`
- baseline output: `/root/epfs/new_route_stage1_skymask/frame_fine_tracklet_long_assoc_0000_0999_v008_gap60_v2`
- source tracklets: `/root/epfs/new_route_stage1_skymask/frame_fine_tracklets_0000_0999_v008_v016_m3_gap60_v2`

Baseline params:

- same accepted-candidate:
  - `centroid_distance=1.5`
  - `bbox_distance=0.5`
  - `color_distance=90`
- same source-cluster:
  - `frame_gap=240`
  - `centroid_distance=0.8`
  - `bbox_distance=0.25`
  - `color_distance=60`
- strict cross-source:
  - `frame_gap=80`
  - `centroid_distance=0.35`
  - `bbox_distance=0.08`
  - `color_distance=35`

Baseline result:

- tracklets: `328`
- objects: `95`
- merge ratio: `0.7104`
- stable long objects: `69`
- single-tracklet objects: `26`
- merge reasons:
  - `same_accepted_candidate`: `229`
  - `same_source_cluster`: `4`
  - `new_object`: `95`

Controls:

- same-candidate loose:
  - params: `same_candidate centroid=3.0`, `bbox=1.2`, `color=140`; source/cross disabled
  - objects: `66`
  - merge ratio: `0.7988`
- same-candidate upper bound:
  - params: same accepted-candidate only, effectively no spatial/color gate
  - objects: `60`
  - merge ratio: `0.8171`

Interpretation:

- Long-range association materially improves over short-window tracklets alone: `135` objects -> `95` conservative objects.
- If the original global accepted-candidate ID is treated as authoritative source evidence, the lower bound is around `60` objects.
- The remaining difference from strict2 spatial fine-object fusion (`47` objects) is mostly cross accepted-candidate merging, not failure of the tracklet layer.
- Conservative next baseline should be `66` objects from same-candidate loose, with manual/VLM review only for cross-candidate merge proposals.
- Do not force convergence to `47` automatically; that risks merging distinct thin structures that were separated by source masks.
