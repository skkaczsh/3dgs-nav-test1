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

## 2026-07-08 FH Threshold Run

- `FH_K=120` was too strict: only 3 accepted edges, so it is rejected as a
  useful clustering candidate.
- `FH_K=120000` produced `superpoint_graph_v6_fh_k120000_20260708_185559`:
  281 accepted edges, 45 FH-threshold rejects, and top1000 fine overlap stayed
  at 3 pairs >= 50% and 0 pairs >= 95%.
- `FH_K=240000` produced `superpoint_graph_fh_k240000_20260708_190054`:
  304 accepted edges, 19 FH-threshold rejects, and the same fine overlap
  result. It supersedes the 120000 run as the FH visual QA candidate, but not
  as the main metric baseline: v4 still has fewer high-entropy patches.

## 2026-07-08 Edge Sparsity Diagnosis

- v4 has 197,208 patches but only 7,569 graph edge pairs; 189,029 patches are
  isolated.
- FH240 has the same failure mode and additionally isolates a 470,534-voxel
  horizontal patch.
- Therefore the next useful fix is candidate-edge generation, not FH threshold
  tuning and not `precluster_small_patches.py`.
