# Superpoint Graph v7 Visual Acceptance

Status: `pending`

Candidate: `superpoint_graph_v7_uncertain_guard_20260708_191958`

Viewer:
http://127.0.0.1:8765/tools/semantic_ply_viewer.html?file=/server_parking_priority_s10/geo_patch_las_opt_cpp_v2_voxel003_r4_4090d_20260623/superpoint_graph_v7_uncertain_guard_20260708_191958/superpoint_graph_v1_stride10.ply&objects=/server_parking_priority_s10/geo_patch_las_opt_cpp_v2_voxel003_r4_4090d_20260623/superpoint_graph_v7_uncertain_guard_20260708_191958/superpoint_graph_v1.jsonl&mode=object&stride=1&pointSize=1.5

Review doc: `docs/superpoint_graph_v7_uncertain_guard_20260708.md`

## Required Checks

- `fragment_noise_reduced_on_stable_surfaces` [required] `pending`: Small isolated fragments around stable ground, wall, roof, and facade surfaces are visibly reduced versus v4.
- `no_ground_building_tree_car_overmerge` [required] `pending`: The extra uncertain-fragment edges do not visibly over-merge ground, building, tree, shrub, or car regions.
- `new_overlap_pair_70503_9366_safe` [required] `pending`: The new fine-overlap pair `70503/9366` is acceptable or visually irrelevant.
- `large_surfaces_remain_structurally_separated` [required] `pending`: Large horizontal surfaces remain separated from vertical walls/facades and rough vegetation.
- `spg_direction_still_valid` [required] `pending`: The result supports the Superpoint Graph direction: merge only adjacent regions with geometry/contact/visual evidence, not post-hoc object relabeling.

Known risk: v7 adds one top1000 fine-overlap pair versus v4, patch `70503` (`rough_mixed`) with patch `9366` (`horizontal`), fine ratio `0.500`.

Local risk QA viewer:
http://127.0.0.1:8765/tools/semantic_ply_viewer.html?file=/server_parking_priority_s10/geo_patch_las_opt_cpp_v2_voxel003_r4_4090d_20260623/superpoint_graph_v7_uncertain_guard_20260708_191958/risk_70503_9366_local_qa/risk_70503_9366_context_rgb.ply&objects=/server_parking_priority_s10/geo_patch_las_opt_cpp_v2_voxel003_r4_4090d_20260623/superpoint_graph_v7_uncertain_guard_20260708_191958/risk_70503_9366_local_qa/risk_70503_9366_objects.jsonl&mode=rgb&stride=1&pointSize=2

Color code: patch `70503` is red, patch `9366` is cyan, local context is gray.
The local pack contains `37,547` stride10 points: `15,983` from patch `70503`,
`54` from patch `9366`, and `21,510` context points.

Decision rule: do not promote v7 unless every required check is accepted after visual QA. If any required check fails, keep v4 as the metric baseline and treat v7 only as a diagnostic edge-recall experiment.
