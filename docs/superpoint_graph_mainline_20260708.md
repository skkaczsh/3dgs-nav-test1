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
- Superpoint Transformer: keep the same region-token premise, but replace fixed
  graph message passing with stronger region-level attention. This is a future
  model baseline, not a reason to abandon deterministic patch ownership now.
- KPConv / RandLA-Net: learn local geometric features for large point clouds, but still need clean ownership if we want inspectable objects.
- Pointcept / PTv3 / Sonata: useful supervised point-level baselines for domain
  gap measurement. Their logits can become semantic evidence on patches, but
  they do not replace the one-voxel-one-owner invariant.
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
- Fine-cell neighbor diagnostics show the missing candidates are mostly
  stable-surface to `unknown` / `mixed` / `rough_mixed` fragments, not
  same-geometry patch pairs. The next experiment should be guarded uncertain
  fragment attachment, not broad same-geometry bridging.

## 2026-07-08 Guarded Uncertain Attachment

- `superpoint_graph_v7_uncertain_guard_20260708_191958` accepted 300 guarded
  uncertain-fragment edges.
- It reduced high-entropy patches from 6,410 to 6,361 and isolated `10000+`
  patches from 13 to 9.
- Fine overlap top1000 `>=50%` increased from 3 to 4.
- User visual QA rejected v7: ground, wall, and grass were grouped into one
  object, and part of shrub was merged into that object.
- Keep v4 as the metric baseline. The next edge-recall experiment must improve
  graph candidate coverage without using uncertain fragments as bridges across
  stable surface / vegetation ownership.

## 2026-07-08 Sonata Evidence Contract

- Sonata smoke on the `70503/9366` risk crop is now available, but only as
  representation evidence.
- The edge diagnosis gives a weak-to-moderate separation signal for `70503/9366`
  (`distance_over_pooled_std=1.746`) with low support on patch `9366`
  (`29` matched points).
- Therefore Sonata must not hard-veto or hard-accept ownership by itself.
- Allowed use: add a weighted `sonata_feature_distance` term to
  `cluster_superpoint_graph.py` edge features after generating patch-level
  Sonata descriptors for the same dense voxel source.
- Disallowed use: hard-code the `70503/9366` pair, use Sonata PCA viewer colors
  as semantic labels, or let Sonata override structural veto / exclusive voxel
  ownership.
- Five fixed smoke crops now have Sonata PCA reports:
  horizontal, vertical, rough-mixed, thin-linear, and the `70503/9366`
  mixed-risk crop. All show local smoothness, but no crop gives a dominant
  standalone object cluster.
- Minimum gate before production use is now narrower: generate patch-level
  Sonata descriptors for the full dense voxel source, add them as one weighted
  edge term, and compare against the current v4 baseline. A single crop, or
  PCA viewer color alone, remains diagnostic only.

Implementation hook:

- `scripts/run_sonata_crop_smoke.py --save-feature-npz` can write point-level
  descriptors as `features[N,D]`.
- `scripts/pool_point_features_to_patch_features.py` pools point descriptors
  into `patch_ids/features/counts` NPZ using the current labels file.
- `scripts/build_patch_feature_edge_evidence.py` converts patch descriptors
  into touch-edge evidence CSV.
- `scripts/cluster_superpoint_graph.py` accepts optional external edge
  evidence via `--external-edge-evidence`, with rows containing
  `patch_a,patch_b,similarity` or `patch_a,patch_b,distance`.
- `scripts/run_scan_train_superpoint_graph.sh` forwards this through
  `EXTERNAL_EDGE_EVIDENCE` and `EXTERNAL_EDGE_WEIGHT`.
- Default behavior is unchanged when no external evidence file is provided.
- The fixed five smoke crops have passed the patch-feature edge wiring smoke.
  This proves the interface, not the value of Sonata as a full-scene baseline.
- A touch-edge endpoint sample now covers all `7793` current SPG touch edges
  with sampled Sonata descriptors. This is the first useful cheap proxy for
  full-source descriptors; next compare visual QA and a small weight sweep
  against v4 before any baseline promotion.
- Weight sweep shows monotonic edge acceptance from `0.05` to `0.50`; do not
  use `0.50` as a default before visual QA because it sharply increases merges.

## 2026-07-08 Over-Merge Risk Gate

- `scripts/compare_spg_risk.py` now compares any SPG candidate against the
  trusted v4 baseline before promotion review.
- It fails candidates that add uncertain-fragment bridge edges by default,
  increase fine occupied-cell overlap, or grow accepted edge count too fast.
- Current check:
  - v4 vs v4 passes.
  - v7 fails with `uncertain_fragment_bridge_exceeded=300>0` and
    `fine_high_pairs_50_regression=4>3`, matching user visual QA where
    ground/wall/grass and shrub ownership were over-merged.
  - Sonata 0.15 passes after fine-overlap QA:
    `accepted_edges=501`, `fine_high_pairs_50=3`, `fine_high_pairs_95=0`.
  - Sonata 0.30 keeps fine-overlap stable but fails with
    `accepted_edges_growth=728>633`; it should not be promoted without a
    stricter visual justification.
- This gate is deliberately small: it does not replace visual QA or improve
  clustering by itself; it prevents known-bad structural over-merge candidates
  from being treated as serious promotion candidates.
