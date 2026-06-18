# Parking Dataset Priority Surface Route - 2026-06-17

## Dataset

- Local raw dataset: `/Users/skkac/Work/SCAN/MT20260616-175807`
- Server raw dataset: `/root/epfs/datasets/MT20260616-175807`
- Generated work directory: `/root/epfs/work_MT20260616-175807`
- Raw dataset structure is treated as read-only. All generated files are outside the dataset directory.

## Implemented Route

1. Extract synchronized undistorted frames:
   - script: `scripts/extract_undistorted_frames_jpeg.py`
   - output: `/root/epfs/work_MT20260616-175807/frames_jpeg`
   - result: `18,543/18,543` JPGs, 3 cameras x 6,181 frames.
2. Segment priority classes before free object clustering:
   - script: `scripts/segment_priority_classes.py`
   - model used for this run: `facebook/mask2former-swin-large-mapillary-vistas-semantic`
   - output: `/root/epfs/work_MT20260616-175807/priority_surface_mapillary_s10`
   - classes: `ground`, `wall`, `grass`, `car`, `railing`, `sky`
   - result: `1,857/1,857` masks for `stride=10`.
3. Project priority masks to `.lx` section points with per-camera z-buffer:
   - script: `scripts/project_priority_masks_to_lx.py`
   - output: `/root/epfs/work_MT20260616-175807/priority_projection_s10`
   - sky pixels are hard-filtered and never exported.
4. Cluster residual points:
   - script: `scripts/cluster_residual_points.py`
   - output: `/root/epfs/work_MT20260616-175807/residual_clusters_s10_pca`
   - clustering: `0.15m` voxel connectivity + RGB distance threshold `60`.
   - PCA flags large planar residuals as `hold_as_surface_residual`.
5. Apply `drivability_cpp` geometry prior to residual objects:
   - script: `scripts/apply_drivability_prior_to_residual.py`
   - geometry prior: `/Users/skkac/Work/SCAN/drivability_cpp/output/MT20260616-175807_drivable_points_collision_arm64_wallbfs.pcd`
   - output: `server_parking_priority_s10/residual_clusters_s10_pca_drivability_prior_v3`
   - red/white/blue prior votes are mapped to `ground/wall/other`.
   - large horizontal residuals with dominant ground votes are absorbed even when edge clutter makes PCA thickness high.
   - clean horizontal surfaces missing from the prior are absorbed by geometry-only fallback.
6. Build unified full-scene viewer input:
   - script: `scripts/make_full_scene_object_view.py`
   - output: `server_parking_priority_s10/full_scene_objects_v3`
   - combines priority-layer classes and residual objects into one PLY/JSONL pair.
   - this is the default review view when judging whether important targets were removed too aggressively.
7. Cluster priority-layer classes into objects:
   - script: `scripts/cluster_priority_points.py`
   - local review output: `server_parking_priority_s10/priority_objects_s10_v1`
   - server full output: `/root/epfs/work_MT20260616-175807/priority_objects_s10_full_v1`
   - object ids are assigned per 3D connected component inside each priority class, so `car` and `railing` are visible as independent objects instead of a single class-level pseudo object.
8. Enrich object metadata with scene context:
   - script: `scripts/enrich_scene_object_context.py`
   - adds height-layer context, parking-scene descriptions, geometry quality flags, downstream routing, and DINO prompt groups.
   - server full output: `/root/epfs/work_MT20260616-175807/full_scene_objects_s10_full_v1/full_scene_objects_enriched.jsonl`
   - DINO/fine-object input: `/root/epfs/work_MT20260616-175807/full_scene_objects_s10_full_v1/dino_review_candidates.jsonl`

## Current Metrics

- projected frames: `619` (`0..6180`, `stride=10`)
- raw points processed: `9,905,881`
- visible non-sky points: `9,341,162` (`94.30%`)
- priority points: `8,758,817` (`93.77%` of visible non-sky)
- residual points: `582,345` (`6.23%` of visible non-sky)
- priority counts:
  - ground: `1,583,954`
  - wall: `5,638,059`
  - grass: `1,095,672`
  - car: `249,403`
  - railing: `191,729`
  - residual: `582,345`

Residual clustering:

- residual points: `582,345`
- objects: `201`
- assigned points: `566,778`
- noise points: `15,567`
- PCA surface residuals: `11` objects, `147,372` points
- semantic review candidates: `190` objects, `419,406` points

After `drivability_cpp` prior:

- residual points inspected: `582,345`
- objects inspected: `201`
- point prior votes:
  - ground: `361,056`
  - wall: `50,727`
  - other: `110,434`
  - unknown: `60,128`
- absorbed as ground surface: `61` objects
- absorbed as wall surface: `15` objects
- semantic review candidates: `125` objects, `96,290` points
- previous indoor-ground leakage was dominated by two large horizontal residual objects:
  - status: `absorbed_by_drivability_ground_contaminated`
  - points absorbed: `171,373`
- extra clean horizontal surfaces not covered by the prior:
  - status: `absorbed_by_geometry_ground_unmatched`
  - points absorbed: `23,381`

Unified full-scene viewer:

- total points: `1,442,660`
- priority-layer points: `875,882`
- residual object points: `566,778`
- label counts:
  - floor: `596,418`
  - wall: `596,288`
  - grass: `109,552`
  - unknown/residual review: `96,290`
  - car: `24,927`
  - railing: `19,185`

Clustered priority-object review view:

- local review output: `server_parking_priority_s10/full_scene_objects_v4_clustered_priority`
- total points: `1,414,680`
- priority object mode: `clustered`
- priority objects: `121`
- priority object counts:
  - floor: `11`
  - wall: `9`
  - grass: `18`
  - car: `47`
  - railing: `36`

Server full reusable object dataset:

- output: `/root/epfs/work_MT20260616-175807/full_scene_objects_s10_full_v1`
- total points: `9,236,274`
- priority-layer object points: `8,669,496`
- residual object points: `566,778`
- priority object mode: `clustered`
- priority objects: `102`
- priority object counts:
  - floor: `8`
  - wall: `4`
  - grass: `22`
  - car: `32`
  - railing: `36`

Scene-context enrichment:

- server enriched objects: `303`
- height layers:
  - `ground_level`: median z `-0.405`, `50` floor objects
  - `upper_level_1`: median z `11.105`, `19` floor objects
- scene contexts:
  - outdoor parking ground / pavement: `52`
  - upper-level floor / deck: `17`
  - building / indoor wall: `19`
  - parking-lot vegetation: `22`
  - parked vehicle candidates: `32`
  - guardrail / fence candidates: `36`
  - residual objects after surface removal: `125`
- downstream stages:
  - stable surfaces: `84`
  - stable context objects: `13`
  - DINO fine-object review: `68`
  - fine semantic review: `125`
  - geometry review: `13`

## Review Assets

Local previews:

- `/Users/skkac/Work/SCAN/work_MT20260616-175807/review/contact_cam0_priority_smoke.jpg`
- `/Users/skkac/Work/SCAN/work_MT20260616-175807/review/priority_points_s10_xy.png`
- `/Users/skkac/Work/SCAN/work_MT20260616-175807/review/residual_points_rgb_s10_xy.png`
- `/Users/skkac/Work/SCAN/work_MT20260616-175807/review/residual_objects_s10_xy.png`

Server PLY/JSON outputs:

- `/root/epfs/work_MT20260616-175807/priority_projection_s10/priority_points.ply`
- `/root/epfs/work_MT20260616-175807/priority_projection_s10/residual_points_rgb.ply`
- `/root/epfs/work_MT20260616-175807/priority_projection_s10/priority_projection_report.json`
- `/root/epfs/work_MT20260616-175807/residual_clusters_s10_pca/residual_objects.ply`
- `/root/epfs/work_MT20260616-175807/residual_clusters_s10_pca/residual_objects.jsonl`
- `/root/epfs/work_MT20260616-175807/residual_clusters_s10_pca/residual_cluster_report.json`

Local review outputs:

- `/Users/skkac/Work/SCAN/new_route/server_parking_priority_s10/residual_clusters_s10_pca_drivability_prior_v3/all_status_ascii.ply`
- `/Users/skkac/Work/SCAN/new_route/server_parking_priority_s10/residual_clusters_s10_pca_drivability_prior_v3/residual_objects_drivability_prior_view.jsonl`
- `/Users/skkac/Work/SCAN/new_route/server_parking_priority_s10/residual_clusters_s10_pca_drivability_prior_v3/semantic_review_candidates_ascii.ply`
- `/Users/skkac/Work/SCAN/new_route/server_parking_priority_s10/residual_clusters_s10_pca_drivability_prior_v3/semantic_review_candidates.jsonl`
- `/Users/skkac/Work/SCAN/new_route/server_parking_priority_s10/full_scene_objects_v3/full_scene_objects_ascii.ply`
- `/Users/skkac/Work/SCAN/new_route/server_parking_priority_s10/full_scene_objects_v3/full_scene_objects.jsonl`
- `/Users/skkac/Work/SCAN/new_route/server_parking_priority_s10/priority_objects_s10_v1/priority_objects_ascii.ply`
- `/Users/skkac/Work/SCAN/new_route/server_parking_priority_s10/priority_objects_s10_v1/priority_objects.jsonl`
- `/Users/skkac/Work/SCAN/new_route/server_parking_priority_s10/full_scene_objects_v4_clustered_priority/full_scene_objects_ascii.ply`
- `/Users/skkac/Work/SCAN/new_route/server_parking_priority_s10/full_scene_objects_v4_clustered_priority/full_scene_objects.jsonl`
- `/Users/skkac/Work/SCAN/new_route/server_parking_priority_s10/full_scene_objects_v4_clustered_priority/full_scene_objects_enriched.jsonl`
- `/Users/skkac/Work/SCAN/new_route/server_parking_priority_s10/full_scene_objects_v4_clustered_priority/all_review_candidates.jsonl`
- `/Users/skkac/Work/SCAN/new_route/server_parking_priority_s10/full_scene_objects_v4_clustered_priority/dino_review_candidates.jsonl`
- `/Users/skkac/Work/SCAN/new_route/server_parking_priority_s10/full_scene_objects_s10_full_v1/full_scene_objects_report.json`
- `/Users/skkac/Work/SCAN/new_route/server_parking_priority_s10/full_scene_objects_s10_full_v1/full_scene_objects.jsonl`
- `/Users/skkac/Work/SCAN/new_route/server_parking_priority_s10/full_scene_objects_s10_full_v1/full_scene_objects_enriched.jsonl`
- `/Users/skkac/Work/SCAN/new_route/server_parking_priority_s10/full_scene_objects_s10_full_v1/all_review_candidates.jsonl`
- `/Users/skkac/Work/SCAN/new_route/server_parking_priority_s10/full_scene_objects_s10_full_v1/dino_review_candidates.jsonl`
- `/Users/skkac/Work/SCAN/new_route/server_parking_priority_s10/full_scene_objects_s10_full_v1/scene_context_report.json`

Viewer URL:

Default parking full-scene object entry:

`http://127.0.0.1:8765/tools/parking_full_scene_viewer.html`

Class-level review:

`http://127.0.0.1:8765/tools/semantic_ply_viewer.html?file=/server_parking_priority_s10/full_scene_objects_v3/full_scene_objects_ascii.ply&objects=/server_parking_priority_s10/full_scene_objects_v3/full_scene_objects.jsonl&mode=semantic&stride=1&pointSize=1.5`

Object-level review:

`http://127.0.0.1:8765/tools/semantic_ply_viewer.html?file=/server_parking_priority_s10/full_scene_objects_v4_clustered_priority/full_scene_objects_ascii.ply&objects=/server_parking_priority_s10/full_scene_objects_v4_clustered_priority/full_scene_objects.jsonl&mode=object&stride=1&pointSize=1.5`

Object-level scene-context review:

`http://127.0.0.1:8765/tools/semantic_ply_viewer.html?file=/server_parking_priority_s10/full_scene_objects_v4_clustered_priority/full_scene_objects_ascii.ply&objects=/server_parking_priority_s10/full_scene_objects_v4_clustered_priority/full_scene_objects_enriched.jsonl&mode=object&stride=1&pointSize=1.5`

## Findings

- This route is materially better structured than free clustering over the whole cloud: most stable surfaces and known large classes are removed before residual object clustering.
- The priority model is aggressive. It overuses `wall` on building/interior-looking views, so some valid objects may be swallowed into the priority layer.
- Residual still contains large planar fragments. PCA now flags these instead of sending them directly to semantic review.
- User review showed outdoor ground was removed well, while indoor ground remained in the residual view. The cause was not projection failure: two residual objects had strong drivability ground votes (`70.8%` and `75.7%`) and horizontal normals, but were kept because edge clutter made thickness too high for the original strict planar threshold.
- The current fix is object-level geometry absorption, not another VLM pass. The remaining review candidates are mostly `other`, `unknown`, or mixed object geometry; high-confidence ground-like fragments are now small (`288` points in the v3 report).
- Reviewing only `semantic_review_candidates_ascii.ply` is misleading because it intentionally hides priority-layer objects such as cars and railings. Use `tools/parking_full_scene_viewer.html` or the unified full-scene view for user QA, and use the candidate-only view only for debugging the next semantic clustering stage.
- Viewer policy: default QA must show complete scene objects first. Residual/candidate-only views are secondary diagnostics and should be named as such in status reports.
- Reviewing only class-level priority objects is also insufficient for downstream target reasoning. `cluster_priority_points.py` now gives priority-layer classes object ids, so the next stage can reason over individual car/railing/grass components.
- Scene-context enrichment gives stable surfaces descriptive roles before any VLM/DINO work: parking-lot ground, upper-level floor/deck, building/indoor wall, vegetation, parked vehicle candidates, guardrail/fence candidates, and residual fine-object candidates.
- The next DINO-style stage should consume `dino_review_candidates.jsonl` first. It contains only car/railing fine-object candidates, while `all_review_candidates.jsonl` also includes geometry-review surfaces and residual semantic candidates.
- The next useful correction is not another free VLM label pass. It is a geometry guard for priority classes:
  - ground should be low horizontal surfaces,
  - wall/building should be near-vertical planar surfaces,
  - car should be compact object-shaped clusters,
  - railing should be thin/linear and not broad planes.

## Next Step

Build `refine_priority_by_geometry.py`:

- consume `priority_points.ply` and `residual_objects.jsonl`
- split priority points by class, voxel cluster them, and apply PCA/height/extent guards
- demote invalid broad `railing`, invalid floating `ground`, and non-planar `wall` back into residual
- merge PCA-held residual surface candidates into stable surface layers where geometry agrees
- re-run residual object clustering on the refined residual set
