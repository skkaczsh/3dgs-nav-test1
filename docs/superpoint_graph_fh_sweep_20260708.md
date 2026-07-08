# Superpoint Graph FH Sweep 2026-07-08

Purpose: test whether FH-style adaptive thresholding can reduce weak
large-component merges without reintroducing post-pass patch logic.

| candidate | FH_K | accepted edges | output patches | FH rejects | fine50 | fine95 | status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `superpoint_graph_v5_fh_k120_20260708_185458` | 120 | 3 | 197,627 | 364 | not run | not run | rejected: too strict |
| `superpoint_graph_fh_k60000_20260708_190210` | 60,000 | 259 | 197,371 | 74 | 3 | 0 | metric-only |
| `superpoint_graph_v6_fh_k120000_20260708_185559` | 120,000 | 281 | 197,349 | 45 | 3 | 0 | superseded |
| `superpoint_graph_fh_k240000_20260708_190054` | 240,000 | 304 | 197,326 | 19 | 3 | 0 | visual QA candidate |

Selected viewer:
http://127.0.0.1:8765/tools/semantic_ply_viewer.html?file=/server_parking_priority_s10/geo_patch_las_opt_cpp_v2_voxel003_r4_4090d_20260623/superpoint_graph_fh_k240000_20260708_190054/superpoint_graph_v1_stride10.ply&objects=/server_parking_priority_s10/geo_patch_las_opt_cpp_v2_voxel003_r4_4090d_20260623/superpoint_graph_fh_k240000_20260708_190054/superpoint_graph_v1.jsonl&mode=object&stride=1&pointSize=1.5

Conclusion: `FH_K=240000` is the best current FH candidate. It keeps the same
fine-cell overlap risk as stricter FH runs while recovering more merge recall.
It still needs visual QA against v4 before promotion.
