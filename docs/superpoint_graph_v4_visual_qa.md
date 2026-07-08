# Superpoint Graph v4 Visual QA

Candidate: `superpoint_graph_v4_nearbbox_s070_e120_20260708_183437`

Viewer:
http://127.0.0.1:8765/tools/semantic_ply_viewer.html?file=/server_parking_priority_s10/geo_patch_las_opt_cpp_v2_voxel003_r4_4090d_20260623/superpoint_graph_v4_nearbbox_s070_e120_20260708_183437/superpoint_graph_v1_stride10.ply&objects=/server_parking_priority_s10/geo_patch_las_opt_cpp_v2_voxel003_r4_4090d_20260623/superpoint_graph_v4_nearbbox_s070_e120_20260708_183437/superpoint_graph_v1.jsonl&mode=object&stride=1&pointSize=1.5

## Quantitative Gate

Compared with `spg_v3_bridge`:

| metric | v3 | v4 |
| --- | ---: | ---: |
| accepted edges | 224 | 422 |
| high entropy patches | 6415 | 6410 |
| large high entropy patches | 2 | 1 |
| large low purity patches | 17 | 12 |
| fine overlap pairs >= 50%, top1000 | 4 | 3 |
| fine overlap pairs >= 95%, top1000 | 0 | 0 |

Interpretation: v4 improves merge recall and does not show a fine-cell overlap regression. AABB overlap remains noisy and overstates risk because large surfaces have large bounding boxes.

## Required Visual Checks

1. Large horizontal surfaces remain separated from vertical building/wall surfaces.
2. Large vertical surfaces do not swallow trees or rough vegetation.
3. `rough_mixed` patches around trees/shrubs are not merged into building facades.
4. Cars remain visually separated from ground and building surfaces where visible.
5. Patch fragmentation is visibly better than v3/v1 without creating giant mixed patches.

## Priority Patch Links

Large patch spot checks:

- patch `2` vertical, 2,666,723 voxels:
  http://127.0.0.1:8765/tools/semantic_ply_viewer.html?file=/server_parking_priority_s10/geo_patch_las_opt_cpp_v2_voxel003_r4_4090d_20260623/superpoint_graph_v4_nearbbox_s070_e120_20260708_183437/superpoint_graph_v1_stride10.ply&objects=/server_parking_priority_s10/geo_patch_las_opt_cpp_v2_voxel003_r4_4090d_20260623/superpoint_graph_v4_nearbbox_s070_e120_20260708_183437/superpoint_graph_v1.jsonl&mode=object&stride=1&pointSize=1.5&object=2
- patch `4` horizontal, 2,850,203 voxels:
  http://127.0.0.1:8765/tools/semantic_ply_viewer.html?file=/server_parking_priority_s10/geo_patch_las_opt_cpp_v2_voxel003_r4_4090d_20260623/superpoint_graph_v4_nearbbox_s070_e120_20260708_183437/superpoint_graph_v1_stride10.ply&objects=/server_parking_priority_s10/geo_patch_las_opt_cpp_v2_voxel003_r4_4090d_20260623/superpoint_graph_v4_nearbbox_s070_e120_20260708_183437/superpoint_graph_v1.jsonl&mode=object&stride=1&pointSize=1.5&object=4
- patch `70448` rough_mixed, 1,037,993 voxels:
  http://127.0.0.1:8765/tools/semantic_ply_viewer.html?file=/server_parking_priority_s10/geo_patch_las_opt_cpp_v2_voxel003_r4_4090d_20260623/superpoint_graph_v4_nearbbox_s070_e120_20260708_183437/superpoint_graph_v1_stride10.ply&objects=/server_parking_priority_s10/geo_patch_las_opt_cpp_v2_voxel003_r4_4090d_20260623/superpoint_graph_v4_nearbbox_s070_e120_20260708_183437/superpoint_graph_v1.jsonl&mode=object&stride=1&pointSize=1.5&object=70448
- patch `70415` rough_mixed, 663,970 voxels:
  http://127.0.0.1:8765/tools/semantic_ply_viewer.html?file=/server_parking_priority_s10/geo_patch_las_opt_cpp_v2_voxel003_r4_4090d_20260623/superpoint_graph_v4_nearbbox_s070_e120_20260708_183437/superpoint_graph_v1_stride10.ply&objects=/server_parking_priority_s10/geo_patch_las_opt_cpp_v2_voxel003_r4_4090d_20260623/superpoint_graph_v4_nearbbox_s070_e120_20260708_183437/superpoint_graph_v1.jsonl&mode=object&stride=1&pointSize=1.5&object=70415

High-risk pair spot checks:

- pair `70448` / `337`: fine overlap ratio `0.259`, rough_mixed + vertical.
- pair `2` / `70398`: fine overlap ratio `0.184`, vertical + mixed.
- pair `31` / `70399`: fine overlap ratio `0.157`, vertical + mixed.
- pair `31` / `70433`: fine overlap ratio `0.140`, vertical + unknown.

Decision state: not promoted. Accept only if visual QA confirms no obvious over-merge in the required checks above.
