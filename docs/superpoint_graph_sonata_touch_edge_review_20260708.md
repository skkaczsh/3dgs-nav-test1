# SPG Sonata Touch-Edge Review 2026-07-08

## Current Decision

- Keep `superpoint_graph_v4_nearbbox_s070_e120_20260708_183437` as the trusted visual baseline.
- Review Sonata touch-edge weight `0.15` visually before promotion.
- Do not prioritize Sonata touch-edge weight `0.30`; it fails the SPG
  over-merge risk gate by accepted-edge growth.
- Do not promote `superpoint_graph_v7_uncertain_guard_20260708_191958`; user QA found ground/wall/grass mixed into one object and shrub partially merged.

## Metrics

| candidate | accepted edges | output patches | fine50/fine95 | external evidence | note |
| --- | ---: | ---: | ---: | ---: | --- |
| v4 baseline | 422 | 197208 | 3 / 0 | 0 | trusted current baseline |
| Sonata weight 0.15 | 501 | 197129 | 3 / 0 | 7793 | passes risk gate; ready for visual QA |
| Sonata weight 0.30 | 728 | 196902 | 3 / 0 | 7793 | fails `accepted_edges_growth=728>633` |
| v7 uncertain guard | 616 | 197014 | 4 / 0 | 0 | rejected; `uncertain_fragment_bridge=300` |

## Viewer Links

- Index: <http://127.0.0.1:8765/tools/semantic_viewer_index.html>
- v4 full: <http://127.0.0.1:8765/tools/semantic_ply_viewer.html?file=/server_parking_priority_s10/geo_patch_las_opt_cpp_v2_voxel003_r4_4090d_20260623/superpoint_graph_v4_nearbbox_s070_e120_20260708_183437/superpoint_graph_v1_stride10.ply&objects=/server_parking_priority_s10/geo_patch_las_opt_cpp_v2_voxel003_r4_4090d_20260623/superpoint_graph_v4_nearbbox_s070_e120_20260708_183437/superpoint_graph_v1.jsonl&mode=object&stride=1&pointSize=1.5>
- Sonata 0.15 full: <http://127.0.0.1:8765/tools/semantic_ply_viewer.html?file=/server_parking_priority_s10/geo_patch_las_opt_cpp_v2_voxel003_r4_4090d_20260623/superpoint_graph_sonata_touch_edge_sample_v1_20260708/superpoint_graph_v1_stride10.ply&objects=/server_parking_priority_s10/geo_patch_las_opt_cpp_v2_voxel003_r4_4090d_20260623/superpoint_graph_sonata_touch_edge_sample_v1_20260708/superpoint_graph_v1.jsonl&mode=object&stride=1&pointSize=1.5>
- Sonata 0.30 full: <http://127.0.0.1:8765/tools/semantic_ply_viewer.html?file=/server_parking_priority_s10/geo_patch_las_opt_cpp_v2_voxel003_r4_4090d_20260623/superpoint_graph_sonata_touch_edge_w030_20260708/superpoint_graph_v1_stride10.ply&objects=/server_parking_priority_s10/geo_patch_las_opt_cpp_v2_voxel003_r4_4090d_20260623/superpoint_graph_sonata_touch_edge_w030_20260708/superpoint_graph_v1.jsonl&mode=object&stride=1&pointSize=1.5>

## `70503/9366` Risk Context

Color code: `70503` red, `9366` cyan, local context gray.

- v4 risk: <http://127.0.0.1:8765/tools/semantic_ply_viewer.html?file=/server_parking_priority_s10/geo_patch_las_opt_cpp_v2_voxel003_r4_4090d_20260623/superpoint_graph_v4_nearbbox_s070_e120_20260708_183437/risk_70503_9366_local_qa/risk_70503_9366_context.ply&objects=/server_parking_priority_s10/geo_patch_las_opt_cpp_v2_voxel003_r4_4090d_20260623/superpoint_graph_v4_nearbbox_s070_e120_20260708_183437/risk_70503_9366_local_qa/risk_70503_9366_context.jsonl&mode=rgb&stride=1&pointSize=2>
- Sonata 0.15 risk: <http://127.0.0.1:8765/tools/semantic_ply_viewer.html?file=/server_parking_priority_s10/geo_patch_las_opt_cpp_v2_voxel003_r4_4090d_20260623/superpoint_graph_sonata_touch_edge_sample_v1_20260708/risk_70503_9366_local_qa/risk_70503_9366_context.ply&objects=/server_parking_priority_s10/geo_patch_las_opt_cpp_v2_voxel003_r4_4090d_20260623/superpoint_graph_sonata_touch_edge_sample_v1_20260708/risk_70503_9366_local_qa/risk_70503_9366_context.jsonl&mode=rgb&stride=1&pointSize=2>
- Sonata 0.30 risk: <http://127.0.0.1:8765/tools/semantic_ply_viewer.html?file=/server_parking_priority_s10/geo_patch_las_opt_cpp_v2_voxel003_r4_4090d_20260623/superpoint_graph_sonata_touch_edge_w030_20260708/risk_70503_9366_local_qa/risk_70503_9366_context.ply&objects=/server_parking_priority_s10/geo_patch_las_opt_cpp_v2_voxel003_r4_4090d_20260623/superpoint_graph_sonata_touch_edge_w030_20260708/risk_70503_9366_local_qa/risk_70503_9366_context.jsonl&mode=rgb&stride=1&pointSize=2>

## Interpretation

The `70503/9366` local counts are effectively unchanged across v4 and Sonata `0.15/0.30`, so the rejected v7 behavior is not caused by Sonata touch-edge evidence. The failure boundary is the separate `uncertain_fragment_bridge` path, which should stay disabled unless it gets a stronger mixed-structure veto.
