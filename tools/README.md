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

Version index:

```text
http://127.0.0.1:8765/tools/semantic_viewer_index.html
```

`semantic_viewer_index.html` lists generated viewer artifacts by file update time. It reads
`tools/semantic_viewer_index.json`, which is generated from lightweight metadata and QA reports
without reading large PLY payloads.

Refresh the index manually:

```bash
cd /Users/skkac/Work/SCAN/new_route
python3 scripts/build_semantic_viewer_index.py --pretty
```

Run a persistent remote viewer on `scan-rtx5070`:

```bash
ssh scan-rtx5070
cd /home/zsh/Work/SCAN/new_route
ln -sfn /home/zsh/Work/SCAN/work_MT20260616-175807 work_MT20260616-175807
HOST=0.0.0.0 \
PORT=8765 \
REPO_ROOT=/home/zsh/Work/SCAN/new_route \
ARTIFACT_ROOT=work_MT20260616-175807 \
LOG_DIR=/home/zsh/Work/SCAN/.local/logs \
PID_FILE=/home/zsh/Work/SCAN/.local/semantic_ply_viewer_8765.pid \
INDEX_PID_FILE=/home/zsh/Work/SCAN/.local/semantic_viewer_index_refresh_8765.pid \
INDEX_REFRESH_INTERVAL=60 \
bash scripts/start_semantic_ply_viewer.sh
```

The remote index is then available at:

```text
http://scan-rtx5070:8765/tools/semantic_viewer_index.html
```

Use `ARTIFACT_ROOT` to point the indexer at the artifact tree exposed under the HTTP root. For
remote work directories outside the repo, expose them with a symlink instead of copying PLY files.

Parking dataset full-scene object view:

```text
http://127.0.0.1:8765/tools/parking_full_scene_viewer.html
```

Use this parking entry when judging whether large objects were removed too aggressively. It loads the guarded unified full-scene object PLY/JSONL, including priority objects such as cars, railings, grass, floor, wall, plus residual objects. Candidate-only files such as `semantic_review_candidates_ascii.ply` are intentionally filtered debug views.

The guarded parking view keeps all points but demotes priority fine-object candidates that fail geometry/height checks from `car` or `railing` to `unknown`. Their object metadata keeps `priority_guard_status`, `priority_guard_reasons`, and the best image evidence pointer.

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
