# Potree Viewer

## Purpose

`tools/semantic_ply_viewer.html` is useful for small QA files, but it uploads one
large `THREE.Points` buffer to the GPU. Dense patch clouds can overload the
browser GPU stack.

Potree is the preferred dense viewer path. It streams an octree LOD point cloud
and enforces a point budget, so the browser does not need to keep all points
resident in one WebGL buffer.

## One-time local setup

```bash
scripts/setup_potree_viewer_assets.sh
```

This downloads and builds Potree into `third_party/potree/`. The directory is
ignored by git.

## Build Potree data

PotreeConverter 2.x consumes LAS/LAZ, while our viewer files are ASCII PLY. Use:

```bash
scripts/build_potree_from_ply.sh INPUT_ASCII_PLY OUTPUT_DIR /path/to/PotreeConverter
```

The script runs:

1. `scripts/convert_ascii_ply_to_las.py`
2. `PotreeConverter input.las -o OUTPUT_DIR/data -m random`

For object/semantic review, use a PLY whose RGB is already baked for the desired
mode, for example random object colors or semantic colors. The current bridge
preserves XYZ/RGB only.

## Open

Serve the repository root:

```bash
python3 -m http.server 8765 --bind 127.0.0.1
```

Open:

```text
http://127.0.0.1:8765/tools/potree_viewer.html?cloud=/path/to/potree/data/metadata.json&budget=1500000&pointSize=1.0
```

Example generated in this workspace:

```text
http://127.0.0.1:8765/tools/potree_viewer.html?cloud=/server_parking_priority_s10/full_graph_cached_voxel010_r2_s046_samebucket_torch/potree/data/metadata.json&name=samebucket_patch&budget=1500000&pointSize=1.0
```

## Notes

- Potree output should remain under run artifact directories and should not be
  committed.
- `third_party/potree/` and PotreeConverter source/build directories are ignored.
- If building PotreeConverter on macOS fails, build it on Linux. AppleClang lacks
  the parallel STL support used by PotreeConverter 2.1.2.
