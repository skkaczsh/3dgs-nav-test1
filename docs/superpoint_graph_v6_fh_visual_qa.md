# Superpoint Graph v6 FH Visual QA

Candidate: `superpoint_graph_v6_fh_k120000_20260708_185559`

Viewer:
http://127.0.0.1:8765/tools/semantic_ply_viewer.html?file=/server_parking_priority_s10/geo_patch_las_opt_cpp_v2_voxel003_r4_4090d_20260623/superpoint_graph_v6_fh_k120000_20260708_185559/superpoint_graph_v1_stride10.ply&objects=/server_parking_priority_s10/geo_patch_las_opt_cpp_v2_voxel003_r4_4090d_20260623/superpoint_graph_v6_fh_k120000_20260708_185559/superpoint_graph_v1.jsonl&mode=object&stride=1&pointSize=1.5

## Quantitative Gate

Compared with `superpoint_graph_v4_nearbbox_s070_e120_20260708_183437`:

| metric | v4 | v6 FH |
| --- | ---: | ---: |
| accepted edges | 422 | 281 |
| output patches | 197,208 | 197,349 |
| FH threshold rejects | 0 | 45 |
| fine overlap pairs >= 50%, top1000 | 3 | 3 |
| fine overlap pairs >= 95%, top1000 | 0 | 0 |

Interpretation: v6 keeps the same fine-cell overlap risk level as v4, but uses
the FH-style adaptive threshold to reject weak large-component merges. This is
a precision candidate, not a fragmentation-improvement candidate.

## Required Visual Checks

1. Large building/ground/tree over-merge is lower than or equal to v4.
2. Cars remain separated from ground/building surfaces.
3. Large surfaces are not more fragmented than v4 in visually important areas.
4. Tree/shrub rough patches are not swallowed by facade patches.

Decision state: not promoted. Accept only if visual QA shows v6 removes v4
over-merge risk without making fragmentation visibly worse.
