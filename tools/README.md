# Tools

## Semantic PLY/Object Viewer

`semantic_ply_viewer.html` is a static browser viewer for semantic point-cloud QA.

Run:

```bash
cd /Users/skkac/Work/SCAN/new_route
bash scripts/start_semantic_ply_viewer.sh
```

Open:

```text
http://127.0.0.1:8765/tools/semantic_ply_viewer.html
```

Parking dataset full-scene object view:

```text
http://127.0.0.1:8765/tools/parking_full_scene_viewer.html
```

Use this parking entry when judging whether large objects were removed too aggressively. It loads the unified full-scene object PLY/JSONL, including priority objects such as cars, railings, grass, floor, wall, plus residual objects. Candidate-only files such as `semantic_review_candidates_ascii.ply` are intentionally filtered debug views.

The viewer displays common semantic labels, statuses, and scene contexts in Chinese. The underlying PLY/JSONL values remain unchanged English machine labels for script compatibility.

Supported drag-and-drop inputs:

- ASCII PLY with vertex properties such as `x y z red green blue object semantic frame`.
- Target/object fusion `objects.jsonl`; the viewer displays object centroids, labels, status, vote summary, target count, point count, and optional identity fields such as `description`, `identity_hint`, `dominant_attributes`, and `description_votes`.
- ASCII PLY plus `objects.jsonl` together; the viewer keeps PLY point geometry and enriches selected points with object-level description, identity, attributes, and vote metadata by object id.

Useful local files:

```text
/Users/skkac/Work/SCAN/server_target_object_fusion_1000_1999_surface024_fine012/objects/object_points_identity_relabel_stride10.ply
/Users/skkac/Work/SCAN/server_target_object_fusion_1000_1999_surface024_fine012/objects/objects_identity_relabel.jsonl
/Users/skkac/Work/SCAN/server_resume_target_object_fusion_0000_0999/objects/objects.jsonl
/Users/skkac/Work/SCAN/server_target_object_existing_completion_0000_0999/object_points.ply
```

The viewer is intentionally local-only and does not upload data.
