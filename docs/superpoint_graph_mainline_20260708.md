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

## Semantic Graph Inference Boundary

Official superpoints are immutable spatial tokens. Multi-view VLM and 2D masks
contribute unary evidence only; they never change voxel ownership. The next
semantic pass minimizes a graph energy over superpoint labels:

`E(y) = sum_i unary(i, y_i) + lambda * sum_(i,j) contact(i,j) * [y_i != y_j]`.

The unary term combines first-touch-visible observations, sky rejection, VLM
confidence, and structural-region compatibility. The pairwise term uses only
real face contact plus color/geometry compatibility, so disconnected objects
cannot be smoothed together. Hard structural contradictions remain vetoes.
After label inference, each label-induced connected component is an object;
there is no label-first patch merge pass.

This follows the superpoint-token/context separation in SPG and Superpoint
Transformer while avoiding a scene-trained semantic network. PointGroup's
two-coordinate proposal is useful later for fine-object candidates, not for
changing exclusive voxel ownership.

VLM evidence carries both an intrinsic `controlled_label` and a
`surface_attachment`. The former describes the superpoint itself; the latter
describes a broad floor/wall/ceiling-like parent when applicable. This prevents
a thin light strip or railing from becoming a ceiling/wall token merely because
it is attached to that surface.

Image orientation is explicit evidence, not a visual convention: every overlay
contains a projected world-`+Z` arrow and each review row stores its pixel
unit vector. Each row also carries calibrated camera center, camera optical
axis, image-up axis, object view direction, object-relative height, and view
elevation from the exact world-to-LiDAR-to-camera projection chain. These are
hard conditioning facts for review, not visual cues that a VLM must infer.
`world_normal_abs_z` / `gravity_orientation_hint` remain authoritative for
surface orientation; the local PCA `verticality` feature is not a gravity
direction. Free-form descriptions are retained, but a VLM value outside the
controlled-label contract is normalized to `unknown` and can never become a
graph anchor.

The same contradiction function is applied both when an anchor is created and
when it traverses a contact edge. A color-compatible edge may not propagate a
floor/ceiling/roof/grass/stair label into a vertical token, nor a wall label
into a horizontal token. This makes the geometry veto a graph invariant rather
than a seed-only filter.

Propagation receives the full official-superpoint geometry catalogue, not just
the VLM-reviewed subset. Nodes absent from that catalogue are skipped. This
keeps a small review batch as sparse unary evidence rather than accidentally
treating unreviewed graph nodes as geometry-free recipients.

The production CLI additionally verifies that the geometry catalogue covers
every contact-graph endpoint before it writes a posterior.

`annotate_superpoints_structural_regions.py` adds dense drivability votes to
the full geometry catalogue with an exact world-voxel lookup. These votes are
explicitly non-semantic: they are usable for candidate sampling and graph
compatibility, but never translate `ground_like_region` into `floor` or
`vertical_surface_region` into `wall`.

For bounded structural propagation, a high-confidence contradictory region
multiplies the edge score by a calibrated penalty (default `0.25`); it does not
create a hard relabel. This is an evidence term in the posterior, alongside
contact/color weight and geometry veto.

## 2026-07-14 Evidence And Seed Coverage

- The full 3cm reference cloud has `14,482,557` points and `27,019` official
  Superpoints. Its `12,680` contact-graph nodes are fully covered by the
  geometry catalogue; graph propagation fails fast on an incomplete catalogue.
- The drivability field has an exact world-voxel hit ratio of `99.992%` on that
  reference cloud. It contributes non-semantic ground-like / vertical-like
  compatibility only.
- A 69-object Qwen structural retry with explicit world-up evidence reduced
  geometry-label conflicts from `24` to `15`, but reduced safe anchors from
  `43` to `31`. It fixed no previously unsafe anchor. The retry is therefore
  rejected as an automatic replacement; retain the old geometry-safe anchors.
- Graph-coverage sampling found only eight new horizontal/vertical candidates.
  The first preflight used a globally reservoir-sampled candidate PLY, which
  was not the same spatial support used by provenance. It has been retested by
  matching raw world-coordinate `.lx` points from each candidate's proven
  source frames with the same `0.05m` KD-tree radius as provenance.
  - Full source support materialized `211` actual source points across all 8.
  - Restricting to frames with an existing priority/skymask materialized `100`
    points across 5 candidates; the other 3 have no reviewable priority frame.
  - The five reviewable candidates still produced zero first-touch-valid image
    observations under the normal depth/sky/bbox gates. They must not enter VLM
    review or anchor propagation.
  Therefore source-frame provenance is necessary but not sufficient: it proves
  raw scan contribution, not camera-FOV visibility. Candidate selection for
  VLM must use the same `priority_top32` support set as the image pipeline.

`sample_official_superpoints.py --source-aware --source-support ... --lx ...`
is the sole materializer for this check. It reuses the provenance KD-tree
match and writes only source-supported raw points; the default reference-uniform
mode remains for geometry-only viewer samples.

## 2026-07-14 Source-Aware Pose Evidence Pilot

- Rebuilding the full 418-object review set with `priority_top32` source-aware
  raw-LX samples retained `262` objects / `492` first-touch-valid observations.
  The older global-reference sampling reported `381` objects; the difference is
  deliberately rejected pseudo-visibility from distant parts of a global
  Superpoint, not missing geometry.
- All 492 observations carry calibrated camera pose facts and `314` also have
  a usable projected world-up arrow. The absence of an image-space arrow is
  tolerated because the numeric pose facts remain available.
- A 21-object Qwen smoke had `21/21` parse success and changed `10` controlled
  labels versus the old evidence. Plausible corrections include `floor ->
  stair` and `building_part -> railing`; plausible but non-final descriptions
  such as a drain cover as `equipment` demonstrate why VLM remains a unary
  term behind geometry/structure vetoes, never an automatic graph anchor.
- This source-aware materialization is retained only as a frame-exact
  diagnostic. It is not valid input for global first-touch evidence, because
  it may retain only a handful of raw-frame points from an otherwise large
  global Superpoint.

## 2026-07-14 Stable-Geometry Propagation Split

- A strict counterfactual keeps `rough_mixed` Superpoints as local VLM evidence
  but permits only `horizontal` / `vertical` geometry to seed graph propagation.
  It reduced source-aware propagation from `39` to `22` superpoints and wall
  assignments from `15` to `4`; the removed labels were primarily rough-mixed
  wall/stair candidates.
- This is the intended two-tier contract: stable geometry can be automatic;
  rough-mixed observations remain useful descriptions and QA targets but must
  not spread a surface label through contact edges.
- `STABLE_GEOMETRY_ONLY=1` on `run_superpoint_structure_refinement.sh` selects
  this high-precision mode. The default remains the wider experiment baseline
  until region QA establishes that its additional rough-mixed propagation is
  reliable.

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

## 2026-07-17 Global First-Touch Evidence

- `source_frame` is provenance, not a camera-pose constraint. The MANIFOLD
  scan is keyframe/incremental while video poses are independently sampled;
  requiring a point and an image to share a frame number causes valid global
  geometry to have an empty review-frame pool.
- The production visibility invariant is now: project a complete world-space
  Superpoint through candidate `img_pos` poses, then retain only pixels that
  agree with a pre-rendered full-cloud first-touch depth map and are not sky.
  This rejects occluded/see-through evidence without assuming a one-to-one
  video/LiDAR timestamp correspondence.
- The old frame-exact pass remains a high-precision diagnostic baseline, not
  the coverage baseline. Its strict route reduced 418 candidates to 242 with
  image evidence and eventually to 34 propagated Superpoints.
- `build_object_image_evidence.py --global-visibility --global-depth-map-dir`
  makes the new contract explicit and refuses accidental mixing with
  `--source-frame-support` or frame-local `--lx` depth.
- Controlled 20-object comparison on the same 0.03m dense source:
  - source-frame + frame-local depth: `8/20` objects with evidence, with all
    `12` failures reported as `empty_frame_pool`;
  - global pose + full-cloud first-touch: `14/20` objects with evidence, no
    empty-pool failure. The remaining failures are either too few sampled
    object points or genuinely fail projection/depth visibility.
  This is the required coverage gain before spending VLM budget.
- CUDA projector validation on 4090D:
  - the Torch z-buffer keeps the existing NumPy first-touch/edge postprocess;
  - frame 0 across three cameras differs by only `4-18` valid pixels, with
    99.9-percentile shared-pixel depth error at numerical precision and one
    `0.031m` near-point tie, below the `0.12m` first-touch tolerance;
  - 10 poses / 30 maps: NumPy `58.8s`, Torch CUDA `23.2s` (`2.54x`).
  CUDA is therefore the production projector for the global cache; NumPy
  remains the default compatibility backend and numerical reference.
- This follows the useful part of modern 2D/3D graph approaches: geometry
  primitives own disjoint 3D support, multi-view masks/features are evidence
  on graph nodes/edges, and semantic decisions are posterior labels rather
  than 2D masks redefining object boundaries. See the official
  [Superpoint Graph implementation](https://github.com/loicland/superpoint_graph),
  [SAI3D paper](https://arxiv.org/abs/2312.11557), and
  [SAM-Graph implementation](https://github.com/zju3dv/SAM_Graph).

## 2026-07-17 Dense Superpoint Sampling Contract

- A global visibility pass and a source-aware sample have incompatible support
  definitions. `source_aware` preserves only raw `.lx` points that contributed
  to selected provenance frames; `global-visibility` projects the complete
  world-space Superpoint through arbitrary valid camera poses.
- The first full global run accidentally reused source-aware samples. It
  produced evidence for `250/418` candidates (59.8%), and `72` objects had
  fewer than the required 12 sample points. This was not a camera, skymask, or
  first-touch failure. For example, Superpoint `52` has 601 dense voxels but
  only one retained source-aware sample.
- Global evidence must therefore materialize samples directly from the dense
  reference PLY and the immutable official-superpoint label array, with a
  deterministic per-object cap. The reference-uniform materialization at
  `2500` points per candidate produced `614,196` samples and covered all 418
  candidates before image projection.
- With identical poses, full-cloud depth maps, sky/priority masks, and image
  gates, the dense-sample rerun produced evidence for `391/418` candidates
  (93.5%) and `1105` accepted observations. The `141` additional observable
  candidates prove that sampling support, rather than relaxed geometry gates,
  was the dominant coverage bottleneck.
- The remaining 27 candidates are genuine current no-evidence cases: they are
  outside the selected camera FOV, too small after perspective projection, or
  occluded by the full-cloud first-touch surface. They remain `unobserved`, not
  semantic `unknown`.
- The global first-touch frame plan is a function of the object samples. When
  the support set changes, regenerate the plan before rendering maps. The
  dense-plan rerun added 21 poses / 63 maps; after the delta render, the final
  result remained `391/418` objects but removed all `missing_global_depth_map`
  failures. The remaining 27 are therefore not a cache artifact.

Operational rule: use `sample_official_superpoints.py` without
`--source-aware` whenever `build_object_image_evidence.py` uses
`--global-visibility`; use `--source-aware` only together with the old
frame-exact/source-provenance diagnostic path. This is a support-set invariant,
not a tuning preference.
