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

## Findings

- This route is materially better structured than free clustering over the whole cloud: most stable surfaces and known large classes are removed before residual object clustering.
- The priority model is aggressive. It overuses `wall` on building/interior-looking views, so some valid objects may be swallowed into the priority layer.
- Residual still contains large planar fragments. PCA now flags these instead of sending them directly to semantic review.
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

