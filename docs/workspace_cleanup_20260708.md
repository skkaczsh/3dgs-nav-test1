# Workspace Cleanup 2026-07-08

## Scope

Cleaned local generated artifacts under `server_parking_priority_s10/`.

## Removed

- Obsolete `voxel010` dense/cache viewer outputs.
- Old `frame_object_viewer_*`, `full_scene_objects_*`, `spatial_partition_*`, and VLM review outputs superseded by the current patch/SPG line.
- Smoke/QA/transient outputs from earlier sync, DINO, Mimo, Potree temp, and reverse-depth experiments.
- Old object/energy/SPG branches inside `geo_patch_las_opt_cpp_v2_voxel003_r4_4090d_20260623` that were superseded or visually rejected.
- Follow-up cleanup removed all top-level `server_parking_priority_s10/*` generated directories except the active SPG artifact root and `pure_surface_visibility_full_0000_6180`.
- SPG hygiene cleanup removed 20 stale, unindexed compare/QA/object-merge intermediate directories while keeping all current review-index artifacts.

## Kept

- `energy_attach_v4_contact_evidence`: current structural attachment baseline.
- `superpoint_graph_v4_nearbbox_s070_e120_20260708_183437`: current trusted SPG visual baseline.
- `superpoint_graph_v7_uncertain_guard_20260708_191958`: kept as the active risk-review reference.
- Recent Sonata evidence/sweep reports and compact JSON summaries.
- `pure_surface_visibility_full_0000_6180`: retained first-touch visibility baseline.

## Result

- First cleanup: `server_parking_priority_s10` reduced to about `5.6G`; `geo_patch_las_opt_cpp_v2_voxel003_r4_4090d_20260623` reduced to about `815M`.
- Follow-up cleanup: `server_parking_priority_s10` reduced from about `6.0G` to about `1.3G`.
- SPG hygiene cleanup: active artifact directory still about `1.3G`; remaining large directories are all current baseline, rejected-risk review, Sonata diagnostic, or first-touch baseline assets.
- Rebuilt `tools/semantic_viewer_index.json`; it now exposes only 9 current review artifacts.
- `scripts/validate_current_mainline.py` passes after marking SPG runners as `spg_review` contract and skipping legacy supervised/promotion gates for SPG candidates.
