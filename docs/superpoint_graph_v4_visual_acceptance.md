# Superpoint Graph v4 Visual Acceptance

Status: `pending`

Candidate: `superpoint_graph_v4_nearbbox_s070_e120_20260708_183437`

Viewer:
http://127.0.0.1:8765/tools/semantic_ply_viewer.html?file=/server_parking_priority_s10/geo_patch_las_opt_cpp_v2_voxel003_r4_4090d_20260623/superpoint_graph_v4_nearbbox_s070_e120_20260708_183437/superpoint_graph_v1_stride10.ply&objects=/server_parking_priority_s10/geo_patch_las_opt_cpp_v2_voxel003_r4_4090d_20260623/superpoint_graph_v4_nearbbox_s070_e120_20260708_183437/superpoint_graph_v1.jsonl&mode=object&stride=1&pointSize=1.5

Review doc: `docs/superpoint_graph_v4_visual_qa.md`

## Required Checks

- `large_surfaces_not_cross_merged` [required] `pending`: Large horizontal surfaces remain separated from vertical building/wall surfaces.
- `vertical_surfaces_do_not_swallow_vegetation` [required] `pending`: Large vertical surfaces do not swallow trees or rough vegetation.
- `rough_mixed_not_merged_into_facade` [required] `pending`: `rough_mixed` patches around trees/shrubs are not merged into building facades.
- `cars_separated_from_surfaces` [required] `pending`: Cars remain visually separated from ground and building surfaces where visible.
- `fragmentation_better_without_giant_mixed_patch` [required] `pending`: Patch fragmentation is visibly better than v3/v1 without creating giant mixed patches.

Decision rule: do not promote v4 unless every required check is accepted after visual QA.
