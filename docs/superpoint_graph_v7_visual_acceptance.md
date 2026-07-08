# Superpoint Graph v7 Visual Acceptance

Status: `failed`

Candidate: `superpoint_graph_v7_uncertain_guard_20260708_191958`

Viewer:
http://127.0.0.1:8765/tools/semantic_ply_viewer.html?file=/server_parking_priority_s10/geo_patch_las_opt_cpp_v2_voxel003_r4_4090d_20260623/superpoint_graph_v7_uncertain_guard_20260708_191958/superpoint_graph_v1_stride10.ply&objects=/server_parking_priority_s10/geo_patch_las_opt_cpp_v2_voxel003_r4_4090d_20260623/superpoint_graph_v7_uncertain_guard_20260708_191958/superpoint_graph_v1.jsonl&mode=object&stride=1&pointSize=1.5

Review doc: `docs/superpoint_graph_v7_uncertain_guard_20260708.md`

## Required Checks

- `fragment_noise_reduced_on_stable_surfaces` [required] `failed`: Structural over-merge makes the fragment cleanup unsafe.
- `no_ground_building_tree_car_overmerge` [required] `failed`: User QA found ground, wall, and grass grouped into one object, with part of shrub also merged into that object.
- `new_overlap_pair_70503_9366_safe` [required] `failed`: The local risk area is not visually harmless.
- `large_surfaces_remain_structurally_separated` [required] `failed`: Large surface and vegetation ownership is mixed.
- `spg_direction_still_valid` [required] `accepted`: The SPG direction remains valid, but v7's uncertain-fragment attachment rule is rejected.

Known risk: v7 adds one top1000 fine-overlap pair versus v4, patch `70503` (`rough_mixed`) with patch `9366` (`horizontal`), fine ratio `0.500`.

Local risk QA viewer:
http://127.0.0.1:8765/tools/semantic_ply_viewer.html?file=/server_parking_priority_s10/geo_patch_las_opt_cpp_v2_voxel003_r4_4090d_20260623/superpoint_graph_v7_uncertain_guard_20260708_191958/risk_70503_9366_local_qa/risk_70503_9366_context_rgb.ply&objects=/server_parking_priority_s10/geo_patch_las_opt_cpp_v2_voxel003_r4_4090d_20260623/superpoint_graph_v7_uncertain_guard_20260708_191958/risk_70503_9366_local_qa/risk_70503_9366_objects.jsonl&mode=rgb&stride=1&pointSize=2

Color code: patch `70503` is red, patch `9366` is cyan, local context is gray.
The local pack contains `37,547` stride10 points: `15,983` from patch `70503`,
`54` from patch `9366`, and `21,510` context points.

Decision rule: do not promote v7 unless every required check is accepted after visual QA. If any required check fails, keep v4 as the metric baseline and treat v7 only as a diagnostic edge-recall experiment.

## User QA Result

Decision: `rejected_not_promoted`

Observed failure: ground, wall, and grass are assigned to the same object. Shrub
is mostly a separate object, but part of shrub is also merged into the
ground/surface object.

Conclusion: v7 improves some metrics, but its guarded uncertain-fragment
attachment creates visible structural ownership errors. Keep v4 as the metric
baseline and do not continue tuning v7 by adding more local exceptions.
