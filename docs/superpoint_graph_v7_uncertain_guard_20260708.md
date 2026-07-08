# Superpoint Graph v7 Uncertain Guard 2026-07-08

Candidate: `superpoint_graph_v7_uncertain_guard_20260708_191958`

Viewer:
http://127.0.0.1:8765/tools/semantic_ply_viewer.html?file=/server_parking_priority_s10/geo_patch_las_opt_cpp_v2_voxel003_r4_4090d_20260623/superpoint_graph_v7_uncertain_guard_20260708_191958/superpoint_graph_v1_stride10.ply&objects=/server_parking_priority_s10/geo_patch_las_opt_cpp_v2_voxel003_r4_4090d_20260623/superpoint_graph_v7_uncertain_guard_20260708_191958/superpoint_graph_v1.jsonl&mode=object&stride=1&pointSize=1.5

## Metrics

| metric | v4 | v7 uncertain |
| --- | ---: | ---: |
| output patches | 197,208 | 197,014 |
| accepted edges | 422 | 616 |
| uncertain fragment accepted edges | 0 | 300 |
| high entropy patches | 6,410 | 6,361 |
| large high entropy patches | 1 | 1 |
| large low purity patches | 12 | 12 |
| isolated patches | 189,029 | 188,780 |
| isolated voxels | 2,716,607 | 2,590,102 |
| isolated 10000+ patches | 13 | 9 |
| fine overlap pairs >= 50%, top1000 | 3 | 4 |
| fine overlap pairs >= 95%, top1000 | 0 | 0 |

## Interpretation

The guarded uncertain-fragment attachment is the first SPG variant that improves
high-entropy count and large isolated surface count against v4. It also slightly
increases fine-cell overlap risk, so it is a visual QA candidate, not a promoted
baseline.

Visual QA must check whether the added uncertain fragments are legitimate
surface cleanup or whether they create visible over-merge around building,
ground, tree, and car boundaries.
