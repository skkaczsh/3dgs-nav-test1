# Parking Route Asset Recovery 2026-06-18

Purpose: preserve the current working route before switching LAN environments.

## Current Baseline

- Baseline route: frame-local `.lx` section + same-frame priority mask -> geometry-refined Target -> strict Object fusion -> viewer PLY/JSONL.
- Current preferred result: `v5_geometry_ceiling`.
- Do not use v6/v8 as baseline. They are retained as height-split experiments only.

## Local Review Entrypoints

- Viewer service:
  - `http://127.0.0.1:8765/tools/semantic_ply_viewer.html`
- Current v5 viewer URL:
  - `http://127.0.0.1:8765/tools/semantic_ply_viewer.html?file=/server_parking_priority_s10/frame_object_viewer_priority_full_s10_v5_geometry_ceiling/frame_object_points_stride10.ply&objects=/server_parking_priority_s10/frame_object_viewer_priority_full_s10_v5_geometry_ceiling/frame_objects_viewer.jsonl&mode=semantic&stride=1&pointSize=1.5`
- Source-mask QA overlay:
  - `server_parking_priority_s10/frame_local_object_qa_full_s10_v5_geometry_ceiling_overlay/frame_local_object_qa_contact.jpg`

## Local Recovered Outputs

- `server_parking_priority_s10/frame_object_viewer_priority_full_s10_v5_geometry_ceiling/`
  - `frame_object_points_stride10.ply`
  - `frame_objects_viewer.jsonl`
  - `frame_object_viewer_export_report.json`
- `server_parking_priority_s10/frame_local_object_qa_full_s10_v5_geometry_ceiling_overlay/`
  - `frame_local_object_qa_contact.jpg`
  - `frame_local_object_qa_report.json`
  - `frame_local_object_qa_candidates.jsonl`
  - `frame_local_object_qa_evidence.jsonl`
  - `crops/`
- Also available locally for comparison:
  - `frame_object_viewer_priority_full_s10_v3_strict_surface_ground/`
  - `frame_object_viewer_priority_full_s10_v4_geometry_refined/`
  - `frame_object_viewer_priority_full_s10_v8_surface_height08/`

## Remote Reusable Outputs

Remote workdir: `/root/epfs/work_MT20260616-175807`

- Base targets:
  - `frame_targets_priority_full_s10_v1`
- Current preferred targets/objects/viewer:
  - `frame_targets_priority_full_s10_v5_geometry_ceiling`
  - `frame_objects_priority_full_s10_v5_geometry_ceiling`
  - `frame_object_viewer_priority_full_s10_v5_geometry_ceiling`
  - `frame_local_object_qa_full_s10_v5_geometry_ceiling_overlay`
- Experiments:
  - `frame_targets_priority_full_s10_v6_surface_height_split`
  - `frame_targets_priority_full_s10_v8_surface_height08`
  - `frame_object_viewer_priority_full_s10_v8_surface_height08`

## Code Assets

- `scripts/build_frame_targets_from_priority.py`
- `scripts/refine_frame_targets_by_geometry.py`
- `scripts/fuse_targets_to_objects.py`
- `scripts/export_frame_target_objects_for_viewer.py`
- `scripts/build_frame_local_object_qa_pack.py`
- `tools/semantic_ply_viewer.html`

## Verified Tests

- `pytest -q tests/test_build_frame_local_object_qa_pack.py tests/test_refine_frame_targets_by_geometry.py tests/test_target_object_fusion.py tests/test_export_frame_target_objects_for_viewer.py`
- Latest local result before this note: `62 passed`.

## Current Technical Conclusion

- Calibration/projection is not the leading failure in the frame-local route.
- Fusion-level cross-surface ambiguity is fixed.
- Main bottleneck is source priority mask quality: broad masks still swallow adjacent wall/ground/railing/car regions.
- QA overlay now displays the source priority mask; refined 3D targets are point subsets and do not have exact standalone 2D masks.

## Next Work

- Keep v5 as review baseline.
- Improve source priority mask generation or post-mask target splitting before object fusion.
- Do not spend more full-scene VLM relabel passes until target/mask quality improves.
