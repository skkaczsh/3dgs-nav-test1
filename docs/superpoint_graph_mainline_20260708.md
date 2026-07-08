# Superpoint Graph Mainline Decision

## Decision

Move the dense patch mainline from chained post-passes to a single superpoint-graph clustering objective.

## Why

- Current bucket split / boundary transfer / attachment passes can reduce patch count, but they repeatedly violate structural ownership.
- 20260708 diagnostics show the failure mode directly:
  - `attach_v4`: 197,630 patches, 6,424 high-entropy patches, 1 large high-entropy patch.
  - `bucket_split`: 188,481 patches, 8,406 high-entropy patches, 20 large high-entropy patches.
  - `bucket_structural_veto`: 190,617 patches, 8,687 high-entropy patches, 29 large high-entropy patches.
  - `bucket_structural_upstream`: 190,617 patches, 8,688 high-entropy patches, 29 large high-entropy patches.
- The repeated regression means local guards are too late or too narrow. The ownership decision must be one graph problem, not split/transfer/merge patches that fight each other.

## Mainstream Reference Model

- Superpoint Graph: over-segment into geometrically pure superpoints, then reason on a graph.
- KPConv / RandLA-Net: learn local geometric features for large point clouds, but still need clean ownership if we want inspectable objects.
- Practical local route: keep our deterministic dense 0.03m voxel features, but restructure optimization like Superpoint Graph.

## Next Implementation Shape

1. Build over-segmented superpoints from dense raw Opt-LAS voxel cloud.
2. Build an edge table only once:
   - contact count / contact ratio
   - contact color distance and p90
   - normal distribution difference
   - roughness / planarity / linearity difference
   - structural bucket compatibility
3. Cluster graph edges by a single monotonic rule:
   - hard veto only for stable incompatible surfaces.
   - otherwise accept edges by one score.
   - optional FH-style adaptive threshold: accept an edge only when its
     dissimilarity is below `internal_diff + k / component_size`, so small
     fragments can merge while large patches stop swallowing weakly related
     neighbors.
   - no later boundary transfer that can override ownership.
4. Export exclusive voxel labels and viewer PLY.
5. Only after this, add SAM/skymask/VLM evidence as semantic evidence on nodes/objects.

## Stop Doing

- Do not add another bucket-split post-pass.
- Do not let semantic labels create patch boundaries.
- Do not use stride viewer PLY as production geometry input.

## Current Implementation

- `scripts/cluster_superpoint_graph.py` is the active minimal SPG entrypoint.
- Default behavior preserves the last v4 run.
- Set `FH_K>0` in `scripts/run_scan_train_superpoint_graph.sh` to test the
  adaptive threshold without creating another post-pass pipeline.
