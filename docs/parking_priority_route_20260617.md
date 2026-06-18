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
9. Build image evidence for DINO/fine-object review:
   - script: `scripts/build_object_image_evidence.py`
   - projects each candidate object back into undistorted frames and writes top-k crop/overlay evidence.
   - server output: `/root/epfs/work_MT20260616-175807/object_image_evidence_dino_v1`
   - local summary copy: `server_parking_priority_s10/object_image_evidence_dino_v1`
   - result: `68/68` DINO candidates have evidence, `204` evidence rows, rank-1 labels `car=32`, `railing=36`.
10. Apply deterministic geometry/evidence guard to priority fine-object candidates:
   - scripts: `scripts/refine_priority_candidates_by_guard.py`, `scripts/apply_priority_guard_to_full_scene.py`
   - server full guard output: `/root/epfs/work_MT20260616-175807/full_scene_objects_s10_full_v2_priority_guarded`
   - local guarded review output: `server_parking_priority_s10/full_scene_objects_v5_priority_guarded_local`
   - full result: `68` candidates -> `29` geometry-plausible, `23` visual-review, `16` geometry-rejected.
   - full guarded PLY keeps all `9,236,274` points; `144,092` points from rejected priority objects are demoted from `car/railing` to `unknown`.
11. Run GroundingDINO crop-level visual review on guarded ambiguous candidates:
   - script: `scripts/run_groundingdino_evidence_review.py`
   - model/config: `/root/epfs/vlm_seg_project/weights/groundingdino_swint_ogc.pth`, `GroundingDINO_SwinT_OGC.py`
   - runtime: `/root/epfs/conda_envs/vlm_seg`, GPU1
   - output: `/root/epfs/work_MT20260616-175807/groundingdino_review_v1`
   - result: `23` candidates, `69` crop evidence rows, `22` visual-confirmed, `1` weak.
   - merge script: `scripts/merge_visual_review_into_objects.py`
   - merged full object metadata: `/root/epfs/work_MT20260616-175807/full_scene_objects_s10_full_v2_priority_guarded/full_scene_objects_guarded_visual.jsonl`
   - compatibility note: current `transformers` removed older BERT helpers used by GroundingDINO 0.4, so the runner applies local process-only compatibility patches for `get_head_mask` and `get_extended_attention_mask`.
12. Run priority-object geometry conflict QA:
   - script: `scripts/qa_priority_geometry_conflicts.py`
   - local report: `server_parking_priority_s10/full_scene_objects_s10_full_v2_priority_guarded/priority_geometry_conflict_report.json`
   - local findings: `server_parking_priority_s10/full_scene_objects_s10_full_v2_priority_guarded/priority_geometry_conflicts.jsonl`
   - result: `65/303` objects flagged, including `8` high-severity objects.
   - high-severity conflicts account for `7,175,595` points, dominated by overmerged/misclassified priority surfaces rather than residual-object clustering.
13. Test finer priority connectivity and conservative geometry relabel preview:
   - scripts: `scripts/cluster_priority_points.py`, `scripts/apply_geometry_conflict_relabels.py`
   - server priority output: `/root/epfs/work_MT20260616-175807/priority_objects_s10_full_v2_voxel012`
   - server preview output: `/root/epfs/work_MT20260616-175807/full_scene_objects_s10_full_v4_voxel012_geometry_relabel`
   - local preview output: `server_parking_priority_s10/full_scene_objects_s10_full_v4_voxel012_geometry_relabel`
   - result: priority object count increases from `102` to `585`; high-severity conflict points drop from `7,175,595` to `1,748,088`, but medium conflicts remain high.
   - conservative relabel preview changes `112` objects / `5,880,678` points, mostly demoting mixed geometry conflicts to `unknown`.
14. Split priority objects by local 3D geometry:
   - scripts: `scripts/qa_priority_geometry_conflicts.py`, `scripts/split_priority_objects_by_local_geometry.py`
   - server priority output: `/root/epfs/work_MT20260616-175807/priority_objects_s10_full_v4_local_geometry_v2`
   - server full-scene output: `/root/epfs/work_MT20260616-175807/full_scene_objects_s10_full_v6_local_geometry_split_v2`
   - local full-scene preview: `server_parking_priority_s10/full_scene_objects_s10_full_v6_local_geometry_split_v2`
   - method: recompute conflicts with `railing_clean_horizontal_surface`, split selected floor/wall/grass/railing objects by local `0.80m` voxel PCA and 6-neighbor connected components.
   - result: priority objects `585 -> 1128`; priority railing points `124,493 -> 111,642`; remaining railing geometry conflicts are small fragments totaling `2,709` points.
15. Apply railing-only geometry conflict demotion:
   - script: `scripts/apply_geometry_conflict_relabels.py --only-label railing`
   - local full-scene preview: `server_parking_priority_s10/full_scene_objects_s10_full_v7_local_geometry_railing_demote`
   - method: only apply relabels for residual `railing` conflicts after local geometry split; do not demote wall/car/grass conflicts in this pass.
   - result: `17` residual railing objects relabeled; stride10 preview changes `271` points, with `15` clean horizontal railing fragments converted to `floor` and `2` surface-like railing fragments converted to `unknown`.
16. Apply clean horizontal wall surface refinement and scene-context enrichment:
   - scripts: `scripts/apply_geometry_conflict_relabels.py`, `scripts/enrich_scene_object_context.py`
   - local full-scene preview: `server_parking_priority_s10/full_scene_objects_s10_full_v8_surface_geometry_refine`
   - method: only apply generated `wall_clean_horizontal_surface_to_*` relabels; low horizontal wall fragments (`z < 6m`) become `floor`, high horizontal wall fragments become `ceiling`; no mixed/low-planarity wall is demoted in this pass.
   - result: `19` wall objects relabeled; stride10 preview changes `3,219` points; high-severity geometry conflicts drop from `29` to `10`.
   - scene context now assigns `ground_zone`, `transition_zone`, and `upper_zone` metadata for floor/wall/ceiling objects so downstream DINO/VLM review can operate on residual/fine targets instead of stable surfaces.
17. Mask unconfirmed fine-object candidates for user QA:
   - script: `scripts/mask_unconfirmed_fine_candidates.py`
   - local full-scene preview: `server_parking_priority_s10/full_scene_objects_s10_full_v9_candidate_safe`
   - reason: v8 displayed `car` / `railing` candidate labels as if they were confirmed semantic labels, causing wall/surface false positives to appear as cars or railings.
   - method: objects routed to `dino_fine_object_review` with labels `car` or `railing` are displayed as `fine_candidate`; original labels are preserved as `candidate_label`.
   - result: `323` unconfirmed fine-object candidates masked (`car=141`, `railing=182`); semantic label counts no longer include final `car` or `railing` until visual confirmation promotes them.
18. Apply full-scene `drivability_cpp` ground/wall prior guard:
   - script: `scripts/apply_drivability_prior_to_full_scene.py`
   - local/remote output: `server_parking_priority_s10/full_scene_objects_s10_full_v13_drivability_full_scene_guard`
   - remote reusable output: `/root/epfs/work_MT20260616-175807/full_scene_objects_s10_full_v13_drivability_full_scene_guard`
   - geometry prior: `MT20260616-175807_drivable_points_collision_arm64_wallbfs.pcd`
   - method: voxelize drivability red/white/blue PCD, vote it onto every full-scene object, and relabel only when drivability prior and object PCA geometry agree.
   - result: `51` objects / `9,272` preview points changed.
   - important corrections: `car->wall=3`, `railing->wall=6`, `fine_candidate->floor/wall=15`, `unknown->floor/wall=16`; this directly targets the observed wall/ground-as-railing/car failure.
   - default viewer: `tools/parking_full_scene_viewer.html` now opens this v13 output.

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

Priority candidate guard:

- full server candidates: `68`
- geometry plausible: `29`
  - car: `10`
  - railing: `19`
- needs visual review: `23`
  - car: `11`
  - railing: `12`
- geometry rejected: `16`
  - car: `11`
  - railing: `5`
- guarded full-scene object labels:
  - floor: `69`
  - wall: `19`
  - grass: `22`
  - car: `21`
  - railing: `31`
  - unknown: `141`

GroundingDINO visual review:

- reviewed candidates: `23`
- crop evidence rows: `69`
- visual confirmed: `22`
  - car: `11`
  - railing: `11`
- visual weak: `1`
  - railing: `1`
- full guarded visual metadata merge:
  - merged object rows: `23 / 303`
  - not visual reviewed: `280`

Priority geometry conflict QA:

- objects inspected: `303`
- findings: `65`
- severity counts:
  - high: `8`
  - medium: `57`
  - ok: `238`
- high-severity point count: `7,175,595`
- top high-impact conflict:
  - object `1200009`, label `wall`, points `5,568,831`
  - reasons: `wall_has_horizontal_normal`, `wall_high_thickness`
  - interpretation: the priority segmenter produced a huge mixed/horizontal component under `wall`; this must be split before trusting wall/floor semantics.

Voxel `0.12m` priority recluster test:

- priority objects: `585`
  - floor: `23`
  - wall: `69`
  - grass: `153`
  - car: `141`
  - railing: `199`
- high-severity conflict points: `1,748,088`
- medium conflict points: `5,343,558`
- conclusion: finer connectivity reduces the worst overmerge, but it does not solve mixed surfaces. A local plane/normal split is still required.

Conservative geometry relabel preview:

- full-scene objects: `786`
- relabeled objects: `112`
- changed points: `5,880,678`
- object labels after relabel:
  - floor: `94`
  - wall: `51`
  - grass: `129`
  - car: `95`
  - railing: `192`
  - unknown: `225`
- interpretation: this preview is more honest than the confident mislabel view, but it intentionally increases `unknown`. It is a QA/debug preview, not the final semantic product.

Local geometry split v2:

- split source objects: `20`
- priority objects after split: `1128`
- priority point labels after split:
  - floor: `2,626,523`
  - wall: `3,342,814`
  - grass: `823,978`
  - car: `237,670`
  - railing: `111,642`
  - unknown: `1,444,321`
- user-reported issue addressed: several floor/wall regions were being shown as `railing`; the updated QA flags `railing_clean_horizontal_surface` and the split stage converts local horizontal surface voxels to `floor` or `unknown`.
- remaining `railing` geometry conflicts: `17` objects, `2,709` points total. These are small fragments and should be handled by a later small-object merge/demotion pass, not by broad class-level relabel.

Railing-only demotion v7:

- source preview: `server_parking_priority_s10/full_scene_objects_s10_full_v6_local_geometry_split_v2`
- relabel count: `17` objects
- changed stride10 preview points: `271`
- relabel reasons:
  - `railing_clean_horizontal_surface_to_floor`: `15`
  - `railing_surface_like_to_unknown`: `2`
- object labels after relabel:
  - floor: `232`
  - wall: `201`
  - grass: `177`
  - car: `141`
  - railing: `182`
  - unknown: `396`
- interpretation: this is the current default review preview for the user-reported "ground/wall labeled as railing" issue. It deliberately touches only railing conflicts to avoid the overly aggressive behavior of the older all-conflict relabel preview.

Clean horizontal wall refinement v8:

- source preview: `server_parking_priority_s10/full_scene_objects_s10_full_v7_local_geometry_railing_demote`
- relabel count: `19` objects
- changed stride10 preview points: `3,219`
- relabel reasons:
  - `wall_clean_horizontal_surface_to_floor`: `9`
  - `wall_clean_horizontal_surface_to_ceiling`: `10`
- object labels after relabel:
  - floor: `241`
  - wall: `182`
  - ceiling: `10`
  - grass: `177`
  - car: `141`
  - railing: `182`
  - unknown: `396`
- QA after relabel:
  - ok objects: `879`
  - medium findings: `440`
  - high findings: `10`
  - high-severity points: `338,629`
- scene-context counts:
  - outdoor parking ground / pavement: `99`
  - parking ramp / transition floor: `32`
  - upper parking deck floor: `110`
  - ground-zone wall: `56`
  - transition-zone wall: `46`
  - upper-zone wall: `80`
  - ceiling / overhead deck surface: `10`
  - parked vehicle candidates: `141`
  - guardrail / fence candidates: `182`
  - residual objects after surface removal: `396`
- interpretation: this is the current default preview. It moves the route closer to the intended structure-first split: stable floor/wall/ceiling/vegetation surfaces carry scene metadata, while car/railing/unknown remain routed to DINO/fine-object review.

Candidate-safe v9:

- source preview: `server_parking_priority_s10/full_scene_objects_s10_full_v8_surface_geometry_refine`
- masked objects: `323`
- masked candidate labels:
  - car: `141`
  - railing: `182`
- object labels after masking:
  - floor: `241`
  - wall: `182`
  - ceiling: `10`
  - grass: `177`
  - fine_candidate: `323`
  - unknown: `396`
- changed stride10 preview points: `34,670`
- interpretation: this is the current default user-review preview. It prevents unconfirmed `car` / `railing` candidate labels from being read as final semantics. Real cars and railings should be promoted back only after DINO/GroundingDINO or manual visual confirmation.

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
- `/Users/skkac/Work/SCAN/new_route/server_parking_priority_s10/object_image_evidence_dino_v1/object_image_evidence_report.json`
- `/Users/skkac/Work/SCAN/new_route/server_parking_priority_s10/object_image_evidence_dino_v1/object_image_evidence.jsonl`
- `/Users/skkac/Work/SCAN/new_route/server_parking_priority_s10/object_image_evidence_dino_v1/object_image_evidence_contact.jpg`
- `/Users/skkac/Work/SCAN/new_route/server_parking_priority_s10/priority_candidate_guard_v1/priority_candidate_guard_report.json`
- `/Users/skkac/Work/SCAN/new_route/server_parking_priority_s10/priority_candidate_guard_v1/priority_candidate_guard_all.jsonl`
- `/Users/skkac/Work/SCAN/new_route/server_parking_priority_s10/full_scene_objects_s10_full_v2_priority_guarded/full_scene_guard_report.json`
- `/Users/skkac/Work/SCAN/new_route/server_parking_priority_s10/full_scene_objects_s10_full_v2_priority_guarded/full_scene_objects_guarded.jsonl`
- `/Users/skkac/Work/SCAN/new_route/server_parking_priority_s10/full_scene_objects_s10_full_v2_priority_guarded/full_scene_objects_guarded_stride10.ply`
- `/Users/skkac/Work/SCAN/new_route/server_parking_priority_s10/full_scene_objects_s10_full_v2_priority_guarded/full_scene_objects_guarded_visual.jsonl`
- `/Users/skkac/Work/SCAN/new_route/server_parking_priority_s10/full_scene_objects_s10_full_v2_priority_guarded/visual_merge_report.json`
- `/Users/skkac/Work/SCAN/new_route/server_parking_priority_s10/full_scene_objects_s10_full_v4_voxel012_geometry_relabel/full_scene_objects_geometry_relabel_stride10.ply`
- `/Users/skkac/Work/SCAN/new_route/server_parking_priority_s10/full_scene_objects_s10_full_v4_voxel012_geometry_relabel/full_scene_objects_geometry_relabel.jsonl`
- `/Users/skkac/Work/SCAN/new_route/server_parking_priority_s10/full_scene_objects_s10_full_v4_voxel012_geometry_relabel/full_scene_objects_geometry_relabel_report.json`
- `/Users/skkac/Work/SCAN/new_route/server_parking_priority_s10/full_scene_objects_s10_full_v4_voxel012_geometry_relabel/full_scene_objects_geometry_relabel_relabels.jsonl`
- `/Users/skkac/Work/SCAN/new_route/server_parking_priority_s10/full_scene_objects_s10_full_v6_local_geometry_split_v2/full_scene_objects_stride10.ply`
- `/Users/skkac/Work/SCAN/new_route/server_parking_priority_s10/full_scene_objects_s10_full_v6_local_geometry_split_v2/full_scene_objects.jsonl`
- `/Users/skkac/Work/SCAN/new_route/server_parking_priority_s10/full_scene_objects_s10_full_v6_local_geometry_split_v2/full_scene_objects_report.json`
- `/Users/skkac/Work/SCAN/new_route/server_parking_priority_s10/full_scene_objects_s10_full_v7_local_geometry_railing_demote/full_scene_objects_railing_demote.ply`
- `/Users/skkac/Work/SCAN/new_route/server_parking_priority_s10/full_scene_objects_s10_full_v7_local_geometry_railing_demote/full_scene_objects_railing_demote.jsonl`
- `/Users/skkac/Work/SCAN/new_route/server_parking_priority_s10/full_scene_objects_s10_full_v7_local_geometry_railing_demote/full_scene_objects_railing_demote_report.json`
- `/Users/skkac/Work/SCAN/new_route/server_parking_priority_s10/full_scene_objects_s10_full_v7_local_geometry_railing_demote/full_scene_objects_railing_demote_relabels.jsonl`
- `/Users/skkac/Work/SCAN/new_route/server_parking_priority_s10/full_scene_objects_s10_full_v8_surface_geometry_refine/full_scene_objects_surface_refine.ply`
- `/Users/skkac/Work/SCAN/new_route/server_parking_priority_s10/full_scene_objects_s10_full_v8_surface_geometry_refine/full_scene_objects_surface_refine.jsonl`
- `/Users/skkac/Work/SCAN/new_route/server_parking_priority_s10/full_scene_objects_s10_full_v8_surface_geometry_refine/full_scene_objects_surface_refine_enriched.jsonl`
- `/Users/skkac/Work/SCAN/new_route/server_parking_priority_s10/full_scene_objects_s10_full_v8_surface_geometry_refine/full_scene_objects_surface_refine_report.json`
- `/Users/skkac/Work/SCAN/new_route/server_parking_priority_s10/full_scene_objects_s10_full_v8_surface_geometry_refine/full_scene_objects_surface_refine_relabels.jsonl`
- `/Users/skkac/Work/SCAN/new_route/server_parking_priority_s10/full_scene_objects_s10_full_v8_surface_geometry_refine/full_scene_objects_v8_geometry_conflict_report.json`
- `/Users/skkac/Work/SCAN/new_route/server_parking_priority_s10/full_scene_objects_s10_full_v8_surface_geometry_refine/scene_context_report.json`
- `/Users/skkac/Work/SCAN/new_route/server_parking_priority_s10/full_scene_objects_s10_full_v8_surface_geometry_refine/all_review_candidates.jsonl`
- `/Users/skkac/Work/SCAN/new_route/server_parking_priority_s10/full_scene_objects_s10_full_v8_surface_geometry_refine/dino_review_candidates.jsonl`
- `/Users/skkac/Work/SCAN/new_route/server_parking_priority_s10/full_scene_objects_s10_full_v9_candidate_safe/full_scene_objects_candidate_safe.ply`
- `/Users/skkac/Work/SCAN/new_route/server_parking_priority_s10/full_scene_objects_s10_full_v9_candidate_safe/full_scene_objects_candidate_safe.jsonl`
- `/Users/skkac/Work/SCAN/new_route/server_parking_priority_s10/full_scene_objects_s10_full_v9_candidate_safe/full_scene_objects_candidate_safe_report.json`
- `/Users/skkac/Work/SCAN/new_route/server_parking_priority_s10/full_scene_objects_s10_full_v9_candidate_safe/full_scene_objects_candidate_safe_masked.jsonl`
- `/Users/skkac/Work/SCAN/new_route/server_parking_priority_s10/priority_objects_s10_full_v4_local_geometry_v2/priority_objects_local_geometry_report.json`
- `/Users/skkac/Work/SCAN/new_route/server_parking_priority_s10/priority_objects_s10_full_v4_local_geometry_v2/priority_geometry_conflict_report.json`
- `/Users/skkac/Work/SCAN/new_route/server_parking_priority_s10/full_scene_objects_v5_priority_guarded_local/full_scene_objects_guarded_ascii.ply`
- `/Users/skkac/Work/SCAN/new_route/server_parking_priority_s10/full_scene_objects_v5_priority_guarded_local/full_scene_objects_guarded.jsonl`
- `/Users/skkac/Work/SCAN/new_route/server_parking_priority_s10/groundingdino_review_v1/groundingdino_review_report.json`
- `/Users/skkac/Work/SCAN/new_route/server_parking_priority_s10/groundingdino_review_v1/groundingdino_object_review.jsonl`
- `/Users/skkac/Work/SCAN/new_route/server_parking_priority_s10/groundingdino_review_v1/groundingdino_review_contact.jpg`

Viewer URL:

Default parking full-scene object entry:

`http://127.0.0.1:8765/tools/parking_full_scene_viewer.html`

Guarded server-full stride review:

`http://127.0.0.1:8765/tools/semantic_ply_viewer.html?file=/server_parking_priority_s10/full_scene_objects_s10_full_v2_priority_guarded/full_scene_objects_guarded_stride10.ply&objects=/server_parking_priority_s10/full_scene_objects_s10_full_v2_priority_guarded/full_scene_objects_guarded_visual.jsonl&mode=semantic&stride=1&pointSize=1.5`

Conservative geometry relabel preview:

`http://127.0.0.1:8765/tools/semantic_ply_viewer.html?file=/server_parking_priority_s10/full_scene_objects_s10_full_v4_voxel012_geometry_relabel/full_scene_objects_geometry_relabel_stride10.ply&objects=/server_parking_priority_s10/full_scene_objects_s10_full_v4_voxel012_geometry_relabel/full_scene_objects_geometry_relabel.jsonl&mode=semantic&stride=1&pointSize=1.5`

Local geometry split v2 preview:

`http://127.0.0.1:8765/tools/semantic_ply_viewer.html?file=/server_parking_priority_s10/full_scene_objects_s10_full_v6_local_geometry_split_v2/full_scene_objects_stride10.ply&objects=/server_parking_priority_s10/full_scene_objects_s10_full_v6_local_geometry_split_v2/full_scene_objects.jsonl&mode=semantic&stride=1&pointSize=1.5`

Railing-only demotion v7 preview:

`http://127.0.0.1:8765/tools/semantic_ply_viewer.html?file=/server_parking_priority_s10/full_scene_objects_s10_full_v7_local_geometry_railing_demote/full_scene_objects_railing_demote.ply&objects=/server_parking_priority_s10/full_scene_objects_s10_full_v7_local_geometry_railing_demote/full_scene_objects_railing_demote.jsonl&mode=semantic&stride=1&pointSize=1.5`

Clean horizontal wall refinement v8 preview:

`http://127.0.0.1:8765/tools/semantic_ply_viewer.html?file=/server_parking_priority_s10/full_scene_objects_s10_full_v8_surface_geometry_refine/full_scene_objects_surface_refine.ply&objects=/server_parking_priority_s10/full_scene_objects_s10_full_v8_surface_geometry_refine/full_scene_objects_surface_refine_enriched.jsonl&mode=semantic&stride=1&pointSize=1.5`

Candidate-safe v9 preview:

`http://127.0.0.1:8765/tools/semantic_ply_viewer.html?file=/server_parking_priority_s10/full_scene_objects_s10_full_v9_candidate_safe/full_scene_objects_candidate_safe.ply&objects=/server_parking_priority_s10/full_scene_objects_s10_full_v9_candidate_safe/full_scene_objects_candidate_safe.jsonl&mode=semantic&stride=1&pointSize=1.5`

Visual-promoted v10 preview:

`http://127.0.0.1:8765/tools/semantic_ply_viewer.html?file=/server_parking_priority_s10/full_scene_objects_s10_full_v10_visual_promoted/full_scene_objects_visual_promoted.ply&objects=/server_parking_priority_s10/full_scene_objects_s10_full_v10_visual_promoted/full_scene_objects_visual_promoted.jsonl&mode=semantic&stride=1&pointSize=1.5`

Visual + geometry guarded v11 preview:

`http://127.0.0.1:8765/tools/semantic_ply_viewer.html?file=/server_parking_priority_s10/full_scene_objects_s10_full_v11_visual_geometry_guard/full_scene_objects_visual_geometry_guard.ply&objects=/server_parking_priority_s10/full_scene_objects_s10_full_v11_visual_geometry_guard/full_scene_objects_visual_geometry_guard.jsonl&mode=semantic&stride=1&pointSize=1.5`

Tight-evidence visual + geometry guarded v12 preview:

`http://127.0.0.1:8765/tools/semantic_ply_viewer.html?file=/server_parking_priority_s10/full_scene_objects_s10_full_v12_tight_visual_geometry_guard/full_scene_objects_tight_visual_geometry_guard.ply&objects=/server_parking_priority_s10/full_scene_objects_s10_full_v12_tight_visual_geometry_guard/full_scene_objects_tight_visual_geometry_guard.jsonl&mode=semantic&stride=1&pointSize=1.5`

Guarded local light review:

`http://127.0.0.1:8765/tools/semantic_ply_viewer.html?file=/server_parking_priority_s10/full_scene_objects_v5_priority_guarded_local/full_scene_objects_guarded_ascii.ply&objects=/server_parking_priority_s10/full_scene_objects_v5_priority_guarded_local/full_scene_objects_guarded.jsonl&mode=semantic&stride=1&pointSize=1.5`

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
- Image evidence shows the current priority layer still has false fine-object positives: some wall seams, ceiling panels, indoor boards, and clutter are labeled as `car` or `railing`. Treat `car/railing` priority objects as candidates until a crop-level detector/reviewer confirms them.
- Geometry/evidence guard now blocks the clearest false positives from becoming stable car/railing labels. This is deliberately conservative: plausible objects and ambiguous objects are kept for DINO/GroundingDINO/visual review rather than removed.
- GroundingDINO is useful as a crop-level reviewer, but it should not directly override geometry guard. Contact-sheet inspection still shows some line/structure crops where a weak detector response is plausible but not decisive. Current policy: geometry-rejected stays demoted; visual review metadata is attached for manual/model audit and later threshold tuning.
- Geometry conflict QA shows the largest remaining error is upstream priority overmerge/misclassification, especially a huge `wall` object with horizontal PCA normal. This cannot be fixed by a crop-level detector alone; the next correction must split/relabel priority surfaces by 3D geometry before object-level semantic review.
- Finer priority voxel connectivity helps but is not sufficient. It splits some large components and reduces high-severity conflict points, but many medium conflicts remain because connected mixed surfaces still need plane/local-normal splitting.
- Conservative geometry relabel preview is useful for QA because it removes confident false labels, but it should not be mistaken for final semantics: the large `unknown` region is a signal that geometry splitting is still missing.
- Local geometry split v2 is the better default preview than conservative relabel: it keeps more geometry-specific labels while correcting large mixed priority objects. It still leaves small railing/floor/wall fragments for a later merge/demotion pass.
- Railing-only demotion v7 is the current default user-review preview for the reported "ground/wall labeled as railing" problem. It avoids the older all-conflict relabel pass because that pass demotes unrelated car/wall/grass conflicts too aggressively.
- Clean horizontal wall refinement v8 is now the default user-review preview. It addresses the next structural error class after railing cleanup: clean horizontal wall fragments are not kept as wall; low fragments become floor and high fragments become ceiling / overhead deck.
- Scene-context enrichment now uses coarse height zones in addition to floor-layer clustering because parking lots, ramps, and upper decks can form a continuous z distribution. This avoids incorrectly treating all floor objects as one undifferentiated ground layer.
- Candidate-safe v9 corrects a display/semantic contract problem: priority `car` and `railing` outputs are not confirmed labels. They should be shown as fine-object candidates until crop-level visual review promotes them. This directly addresses the user-observed v8 issue where wall fragments appeared as cars or railings.
- Visual-promoted v10 is the current user-review default. It runs object-image evidence plus GroundingDINO over v8 fine-object candidates, then promotes only visually confirmed candidates from v9 `fine_candidate` back to `car` / `railing`.
  - evidence candidates: `323`
  - objects with crop evidence: `93`
  - promoted objects: `93` (`car=61`, `railing=32`)
  - unconfirmed candidates kept as `fine_candidate`: `230`
  - changed points from v9: `33,229`
  - final point counts: `floor=307,238`, `wall=334,648`, `ceiling=2,616`, `grass=82,414`, `car=23,144`, `railing=10,085`, `fine_candidate=1,441`, `unknown=153,787`
- Visual + geometry guarded v11 is the current default user-review preview. It keeps v10's visual confirmation, then demotes visually confirmed fine objects whose 3D shape is inconsistent with the class.
  - checked visual promotions: `93` (`car=61`, `railing=32`)
  - demoted after geometry guard: `16` (`car=4`, `railing=12`)
  - main demotion reasons: horizontal surface fragments, dense/thick railing clusters, broad railing volume, wall-like railing plane
  - changed points from v10: `5,760`
  - final point counts: `floor=307,238`, `wall=334,648`, `ceiling=2,616`, `grass=82,414`, `car=22,863`, `railing=4,606`, `fine_candidate=7,201`, `unknown=153,787`
- Fine-object QA v11:
  - script: `scripts/build_fine_object_qa_pack.py`
  - local QA pack: `server_parking_priority_s10/fine_object_qa_v11`
  - checked objects: `323`
  - final labels in QA set: `car=57`, `railing=20`, `fine_candidate=246`
  - contact sheets: `fine_object_qa_top_crops.jpg`, `fine_object_qa_top_overlays.jpg`
  - finding: the highest-risk cases are often loose image-evidence boxes over walls, stairs, building panels, or nearby railings. GroundingDINO can confirm something present in the crop without proving that the projected 3D object is that thing.
  - implication: the next correction should tighten projection evidence with point-overlay coverage, depth/edge continuity, and smaller crop boxes before detector review; it should not be another text-threshold-only detector sweep.
- Tight-evidence v12:
  - script change: `scripts/build_object_image_evidence.py` now supports percentile bboxes, bbox area-ratio filtering, bbox inlier ratio, and tight scoring while preserving legacy defaults.
  - run params: `bbox_percentile=4`, `max_bbox_area_ratio=0.65`, `score_mode=tight`
  - evidence recall unchanged: `93` objects, `279` evidence rows.
  - GroundingDINO result: `92` confirmed, `230` not detected, `1` weak (`car=61`, `railing=31` confirmed).
  - after geometry guard: `car=57`, `railing=19`, `fine_candidate=247`.
  - point counts: `car=22,863`, `railing=4,583`, `fine_candidate=7,224`.
  - interpretation: tightening image evidence removes one weak railing but does not materially change the result. The remaining error mode is likely over-merged/misaligned 3D object candidates, so the next useful step is object-internal 3D splitting / depth-continuity filtering, not more detector prompt/threshold tuning.
- DINO-style dense feature pilot:
  - script: `scripts/run_dino_feature_evidence_qa.py`
  - DINOv3 status: `facebook/dinov3-vits16-pretrain-lvd1689m` is gated on Hugging Face; scan-train receives `403 Forbidden` without approved access.
  - fallback pilot: `facebook/dinov2-small` ran on the v12 tight evidence crops.
  - output: `server_parking_priority_s10/dino_feature_qa_v12_dinov2_small`
  - result: `93` evidence objects evaluated; `91` rows were flagged risky by naive bbox-level ROI/context feature separation.
  - interpretation: crop-rectangle feature averaging is too coarse. The useful DINOv3/DINOv2 role should be patch/point-level binding or object-internal splitting using projected points and depth continuity, not a simple bbox-level post-filter.
- DINOv3 ONNX setup:
  - official `facebook/dinov3-vits16-pretrain-lvd1689m` remains gated; the provided HF token can reach Hugging Face but the model request is still awaiting repo-author review.
  - public fallback model synced locally and to scan-train: `onnx-community/dinov3-vits16-pretrain-lvd1689m-ONNX`.
  - remote path: `/root/epfs/models/dinov3-vits16-pretrain-lvd1689m-onnx`.
  - `onnxruntime-gpu==1.23.2` is installed in `/root/epfs/conda_envs/vlm_seg`; available providers are `TensorrtExecutionProvider`, `CUDAExecutionProvider`, and `CPUExecutionProvider`.
  - script update: `scripts/run_dino_feature_evidence_qa.py` now supports ONNXRuntime local model directories and selects CUDA provider when `--device cuda`.
  - output: `server_parking_priority_s10/dino_feature_qa_v12_dinov3_onnx_cuda`.
  - result: `93` evidence objects evaluated; all `93` rows were flagged risky by the same bbox ROI/context metric.
  - timing: the 93-object CUDA run took about `18s`; the CPU run was about `20s`. ORT reports many memcpy nodes, so single-crop ONNX GPU inference is functionally correct but not a meaningful speedup.
  - interpretation: DINOv3 is now runnable, but the current metric is the bottleneck. Next use should be batched patch/point-level feature binding or TensorRT engine caching, not per-crop bbox averaging.
- DINOv3 batch + projected-point evidence v13:
  - script change: `scripts/run_dino_feature_evidence_qa.py` now supports `--batch-size`, `--roi-source auto|bbox|points`, and projected-point patch masks.
  - script change: `scripts/build_object_image_evidence.py` now supports `--save-projected-samples`; each evidence row can store projected `uv/depth` samples for patch-level feature binding.
  - batch timing: v12 ONNX CUDA `batch_size=16` reduced the 93-object run from about `18s` to about `8s`.
  - point-evidence output: `server_parking_priority_s10/object_image_evidence_dino_v13_points`.
  - point-feature output: `server_parking_priority_s10/dino_feature_qa_v13_points`.
  - v13 evidence count: `323` candidate objects, `80` objects with evidence, `232` evidence rows. This is a new evidence snapshot and should not overwrite v12 because v12 had `93` objects with evidence.
  - v13 point ROI check: all `80` QA rows used projected-point ROI (`roi_source=points`), with about `36-158` DINO patches per object.
  - v13 result: all `80` rows were still flagged risky by ROI/context feature separation.
  - interpretation: even projected-point ROI does not make the current ROI-vs-context metric reliable. DINO features should be used as local patch descriptors for object-internal split/merge, not as a standalone accept/reject classifier.
- Full-scene drivability prior guard v13:
  - script: `scripts/apply_drivability_prior_to_full_scene.py`
  - output: `server_parking_priority_s10/full_scene_objects_s10_full_v13_drivability_full_scene_guard`
  - input: v12 tight visual geometry guard full-scene bundle.
  - point prior votes on preview PLY: `ground=320,751`, `wall=194,138`, `other=258,339`, `unknown=142,145`.
  - changed objects: `51`; changed preview points: `9,272`.
  - after relabel preview counts: `floor=308,796`, `wall=341,665`, `unknown=152,726`, `ceiling=2,616`, `grass=82,088`, `car=18,382`, `fine_candidate=6,857`, `railing=2,243`.
  - interpretation: this is the first full-scene use of the successful `drivability_cpp` wall/floor prior, not just residual cleanup. It reduces visually promoted fine-object false positives when the 3D object is actually a ground/wall surface.
- DINOv3 v14 fine-candidate enrichment:
  - script update: `scripts/enrich_scene_object_context.py` routes `fine_candidate` objects with preserved `candidate_label=car/railing` into `dino_fine_object_review` instead of treating them as generic residuals.
  - enriched object JSON: `server_parking_priority_s10/full_scene_objects_s10_full_v13_drivability_full_scene_guard/full_scene_objects_drivability_full_scene_guard_enriched.jsonl`
  - DINO/fine candidates: `299` (`car=54`, `railing=10`, `fine_candidate=235` with preserved prompt groups).
  - evidence output: `server_parking_priority_s10/object_image_evidence_dino_v14_v13_fine_points`
  - evidence result: `299` candidate objects, `66` objects with image evidence, `190` evidence rows. The lower recall is expected because this run uses the preview/stride object PLY; full point-level evidence is still needed before any full production decision.
  - DINOv3 feature output: `server_parking_priority_s10/dino_feature_qa_v14_v13_fine_points`
  - feature result: `66` objects, feature labels `car=47`, `railing=19`, all `66` rows flagged risky by ROI/context separation.
  - prototype check: car/railing prototype cosine is `0.940977`, too close for direct class separation.
  - interpretation: DINOv3 is useful as a local patch descriptor asset, but current ROI pooling cannot classify or confirm fine objects reliably.
- DINOv3 seed-similarity maps v1:
  - script: `scripts/run_dino_seed_similarity_maps.py`
  - purpose: use projected 3D point samples as seed patches and visualize DINO patch-feature expansion inside the crop.
  - car output: `server_parking_priority_s10/dino_seed_similarity_v1`
  - railing output: `server_parking_priority_s10/dino_seed_similarity_v1_railing`
  - runtime: scan-train, DINOv3 ONNX, `CUDAExecutionProvider`, `batch_size=4`.
  - model limitation: the public ONNX fallback is fixed at `224x224` with `patch_size=16`, giving only a `14x14` feature grid.
  - result: `12/12` car samples and `12/12` railing samples were flagged as bleed-risk cases.
  - railing stats: seed patch count median `77.5`, foreground patch count median `36`, seed similarity mean median `0.969492`, context similarity p95 median `0.973396`.
  - interpretation: the projected-point seed and surrounding wall/floor context are not separable enough at this DINOv3 ViT-S/16 ONNX resolution. DINOv3 should not be used as a direct railing segmentation stage in this form.
  - useful role remains: after tighter 3D object splitting and better crop/point evidence, DINO features can support local same-object binding or object-internal split checks.
- v14 evidence recall diagnosis:
  - script update: `scripts/build_object_image_evidence.py` now reports failure reasons for objects without evidence.
  - diagnostic output: `server_parking_priority_s10/object_image_evidence_dino_v14_v13_fine_points_diag/object_image_evidence_report.json`
  - same parameters as v14: `299` candidates, `66` objects with evidence, `190` evidence rows.
  - objects without evidence: `233` (`car=83`, `railing=150` by DINO prompt group).
  - failure counts across no-evidence object frame/camera attempts: `low_projected_before_image_filter=54,395`, `low_projected_in_image=1,069`, `bbox_too_small=456`.
  - interpretation: the recall bottleneck is not loose crop filtering. Most fine candidates do not project enough sampled points into the candidate frame pool. This is consistent with over-fragmented/stride-thinned fine objects and means DINO review should be fed by denser object points or by local frame-level target clusters, not only the current preview full-scene PLY.
- v15 full-point evidence check:
  - input object JSON: v13 drivability-guarded `dino_review_candidates.jsonl`.
  - input point source changed from the 46MB preview PLY to the 470MB full-scene PLY: `full_scene_objects_s10_full_v6_local_geometry_split_v2/full_scene_objects_ascii.ply`.
  - diagnostic output: `server_parking_priority_s10/object_image_evidence_dino_v15_v13_full_points_diag/object_image_evidence_report.json`
  - evidence recall improved from `66/299` to `157/299`, with `433` evidence rows.
  - objects without evidence dropped from `233` to `142` (`car=51`, `railing=91`).
  - remaining no-evidence failure counts: `low_projected_before_image_filter=17,302`, `low_projected_in_image=11,137`, `bbox_too_small=5,641`.
  - DINO seed-similarity rerun on v15 railing evidence: `server_parking_priority_s10/dino_seed_similarity_v2_fullpoints_railing`; `12/12` railing samples still flagged bleed-risk.
  - interpretation: dense/full point source materially improves evidence recall, but does not solve thin-object visual separation. The next step is to preserve/generate full-density object point bundles for review while splitting candidates by 3D local geometry before DINO feature use.
- The next useful correction is not another free VLM label pass. It is a geometry guard for priority classes:
  - ground should be low horizontal surfaces,
  - wall/building should be near-vertical planar surfaces,
  - car should be compact object-shaped clusters,
  - railing should be thin/linear and not broad planes.

## Next Step

Build a point-evidence-first fine-object refinement pass:

- consume the v13 drivability-guarded full-scene bundle and its enriched object JSONL
- rebuild fine-object image evidence from denser/full object points, not only preview stride points
- split candidate objects by 3D connectedness, local plane/line PCA, and depth-continuity before visual review
- use DINOv3 patch features only as a local binding/split signal after the geometry split
- keep `car/railing` as candidate labels until geometry plus visual evidence agree
