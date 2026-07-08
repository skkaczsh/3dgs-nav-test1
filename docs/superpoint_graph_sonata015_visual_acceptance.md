# SPG Sonata 0.15 Visual Acceptance

Status: `failed`
Candidate: `superpoint_graph_sonata_touch_edge_sample_v1_20260708`
Review doc: `docs/superpoint_graph_sonata_touch_edge_review_20260708.md`
Viewer: http://127.0.0.1:8765/tools/semantic_ply_viewer.html?file=/server_parking_priority_s10/geo_patch_las_opt_cpp_v2_voxel003_r4_4090d_20260623/superpoint_graph_sonata_touch_edge_sample_v1_20260708/superpoint_graph_v1_stride10.ply&objects=/server_parking_priority_s10/geo_patch_las_opt_cpp_v2_voxel003_r4_4090d_20260623/superpoint_graph_sonata_touch_edge_sample_v1_20260708/superpoint_graph_v1.jsonl&mode=object&stride=1&pointSize=1.5
Reviewer: `user`
Reviewed at: `2026-07-08T14:35:00+00:00`

## Required Checks

- `large_building_fragmentation_improved` [optional] `accepted`: A building body can be represented as one object more often than in the previous baseline. Notes: User QA: Sonata 0.15 improves at least one building-scale merge.
- `large_vertical_object_does_not_swallow_shrub` [required] `failed`: Large vertical building objects do not swallow nearby shrubs or rough vegetation faces. Notes: User QA: the shrub face perpendicular to the ground was merged into the same object as the building.
- `large_vertical_object_does_not_swallow_neighbor_building` [required] `failed`: Large building objects do not merge adjacent small buildings or independent structures. Notes: User QA: a nearby small building was also merged into the same large object.
- `delta_queue_object_70448_safe` [required] `pending`: Delta object 70448 does not over-merge rough vegetation/object fragments with facade or building structure.
- `delta_queue_object_70415_safe` [required] `pending`: Delta object 70415 does not over-merge rough vegetation/object fragments with facade or building structure.

Run `python3 scripts/validate_current_mainline.py` after updating checks.
