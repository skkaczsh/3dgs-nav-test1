# Workspace Cleanup - 2026-06-17

Purpose:

- prepare the workspace for a new dataset
- preserve reusable technical conclusions
- remove failed or reproducible local artifacts that should not drive the next
  route

## Keep

- source code, scripts, tests, manifests, and README files
- route decision documents under `new_route/docs`
- compact JSON/Markdown reports copied from local server result directories
- representative reports already committed in experiment folders
- raw scan data and calibration assets unless explicitly superseded

## Delete

- local `server_*` synchronized result directories when they only contain
  regenerated experiment outputs
- old SAM2/Mimo projection products from failed routes
- large PLY/PNG/NPY visual artifacts under experiment `outputs*`,
  `analysis*`, `samples`, and `staged_samples`
- Python `__pycache__` folders
- ad-hoc debug PLY/PNG files from colorization and route experiments

## Preserved conclusions

- full-image SAM2/SAM3 auto-mask plus VLM label is not a reliable dense
  semantic main route
- generic Mask2Former/OneFormer ADE20K/Cityscapes/Mapillary models are smoother
  than noisy SAM2 masks but lack rooftop-domain priors
- GroundingDINO/Florence-style detector plus SAM2 is the most useful fine-object
  branch so far, especially for `railing` and `pipe`
- ConceptSeg-R1 and broad VLM-driven box proposals are useful as side evidence,
  not as a replacement for geometry-aware projection
- future work should prioritize geometry-aware surface extraction first, then
  run fine-object detection on non-stable residuals

## Local report archive

Compact reports from deleted local result folders are archived under:

- `/Users/skkac/Work/SCAN/new_route/docs/preserved_run_reports_20260617`

This archive is intended for text-level traceability only. Heavy point clouds,
rendered panels, intermediate masks, and rerunnable predictions are deliberately
not kept.

## Cleanup Update - 2026-07-08

Deleted locally after the patch/Sonata branch moved to
`geo_patch_las_opt_cpp_v2_voxel003_r4_4090d_20260623`:

- old failed semantic relabel outputs:
  `full_scene_objects_refined_v20..v23`,
  `full_scene_objects_s10_full_v1..v13`, and the matching Mimo review cache
- rejected or superseded patch experiments:
  `full_graph_cached_voxel010_r2_s046_samebucket_torch`,
  `geo_patch_coarse_budget1000_voxel003_r4_v6/v8/v9`,
  old `geo_patch_full_cpp_v1_4090d*`,
  old `geo_patch_objects_window_3000_3600_*`,
  old `full_region_model_voxel010_*`
- rerunnable depth-review panels:
  `fullcloud_reverse_triptych_*`
- local-only failed MPS fallback smoke caches
- Python cache folders
- large derived `drivability_cpp/output/*.pcd` full-point visualizations and
  intermediate point exports

Preserved:

- raw scan data and the 2.7GB optimized LAS source
- current mainline patch result directory
  `server_parking_priority_s10/geo_patch_las_opt_cpp_v2_voxel003_r4_4090d_20260623`
- compact drivability reports and small `drivable_voxels*.pcd` structure outputs
- Sonata smoke outputs and route documentation

Freed approximately 13.7GB locally. Generated result directories remain
untracked; do not commit or rely on deleted historical artifacts as baselines.
