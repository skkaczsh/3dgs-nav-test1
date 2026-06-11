# Tools

## Semantic PLY/Object Viewer

`semantic_ply_viewer.html` is a static browser viewer for semantic point-cloud QA.

Run:

```bash
cd /Users/skkac/Work/SCAN/new_route
python3 -m http.server 8765
```

Open:

```text
http://127.0.0.1:8765/tools/semantic_ply_viewer.html
```

Supported drag-and-drop inputs:

- ASCII PLY with vertex properties such as `x y z red green blue object semantic frame`.
- Target/object fusion `objects.jsonl`; the viewer displays object centroids, labels, status, vote summary, target count, point count, and optional identity fields such as `description`, `identity_hint`, `dominant_attributes`, and `description_votes`.

Useful local files:

```text
/Users/skkac/Work/SCAN/server_resume_target_object_fusion_0000_0999/objects/objects.jsonl
/Users/skkac/Work/SCAN/server_target_object_existing_completion_0000_0999/object_points.ply
```

The viewer is intentionally local-only and does not upload data.
