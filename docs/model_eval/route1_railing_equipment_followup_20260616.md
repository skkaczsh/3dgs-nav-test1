# Route1 Railing/Equipment Follow-up - 2026-06-16

This note records one additional `railing` replay, one support-preserving
`railing` box-growth branch, and one stricter `equipment/HVAC` evidence pack
after the initial Route1 small-eval summary.

It does not change the large-surface conclusion. It refines the fine-object
conclusion.

## Scope

- `railing` thin-mask replay:
  `/Users/skkac/Work/SCAN/new_route/experiments/fine_object_grounded_small_eval/railing_rich_2000_2999_strict_v3_hybrid`
- `equipment/HVAC` strict-precision evidence:
  `/Users/skkac/Work/SCAN/new_route/experiments/fine_object_grounded_small_eval/equipment_rich_2000_2999_ext80_strict_precision`

## 1. Railing Thin-Mask Replay

The `strict_v2` railing branch already showed that prompt tightening and
geometry filtering can keep the semantics mostly on target. The next question
was whether a better thin-mask post-pass could materially improve 3D support.

Three post-pass settings now exist on the same 10-image rich sample:

- `v1`
  - `mean_area_ratio = 0.1172`
  - fused `point_count = 25`
- `v2_relaxed`
  - `mean_area_ratio = 0.2187`
  - fused `point_count = 58`
- `v3_hybrid`
  - `mean_area_ratio = 0.1579`
  - fused `point_count = 34`

`v3_hybrid` parameters:

- `dilate_px = 3`
- `min_component_px = 12`
- `min_component_aspect = 2.2`
- `fallback_keep_topk = 3`

### `v3_hybrid` 3D outcome

Projected and guarded result:

- projected candidates with points: `7`
- kept candidates after guard: `4`
- input projected points: `127`
- kept points: `34`
- actions:
  - `keep_linear_railing = 4`
  - `review_ambiguous_railing = 2`
  - `demote_surface_like_railing = 1`

Fused object result:

- `candidate_count = 4`
- `fine_object_count = 4`
- `point_count = 34`
- `merge_count = 0`
- all survivors are still `single_fine_candidate`

### Interpretation

This replay is useful because it narrows the failure mode.

What improved:

- `v3_hybrid` is less bloated than `v2_relaxed`
- the kept subsets remain highly linear after projection

What did **not** improve enough:

- 3D support is still extremely sparse
- cross-view consolidation does not happen
- all survivors remain tiny singleton objects

Therefore the current `railing` bottleneck is not primarily phrase selection
any more. It is the combination of:

- thin-structure 2D mask incompleteness
- low projected point support on narrow structures
- view-to-view fragmentation before object fusion

Practical conclusion:

- `railing` remains a guarded proposal route
- a better thin-mask morphology setting alone is not enough
- the next step should shift from pure mask dilation tuning to one of:
  - detector-box constrained point extraction
  - line/rail geometry growth in 3D after a small seed is found
  - multi-view support accumulation before final accept/reject

## 1.5. Railing Box-Growth Branch

After the thin-mask replay, a support-preserving branch was tested on the same
`strict_v2` railing sample:

- start from accepted `GroundingDINO + SAM2` detections
- project all visible points inside the detector box
- keep points near the 2D mask and inside a seed-derived depth range
- grow support by 3D connected components
- then re-apply the existing geometry guard

Local evidence:

- `/Users/skkac/Work/SCAN/new_route/experiments/fine_object_grounded_small_eval/project_detector_box_growth.py`
- `/Users/skkac/Work/SCAN/new_route/experiments/fine_object_grounded_small_eval/railing_rich_2000_2999_box_growth_v1`
- `/Users/skkac/Work/SCAN/new_route/experiments/fine_object_grounded_small_eval/railing_rich_2000_2999_box_growth_v2_tight`

### Quantitative comparison

Same 10-image rich sample:

- thin-mask `v1`
  - fused `point_count = 25`
- thin-mask `v2_relaxed`
  - fused `point_count = 58`
- thin-mask `v3_hybrid`
  - fused `point_count = 34`
- box-growth `v1`
  - projected accepted points before guard: `1889`
  - kept points after guard: `942`
  - kept candidates after guard: `6`
  - fused `point_count = 942`
- box-growth `v2_tight`
  - projected accepted points before guard: `1545`
  - kept points after guard: `814`
  - kept candidates after guard: `7`
  - fused `point_count = 814`

### Interpretation

This is the first `railing` branch that materially improves 3D support instead
of only changing 2D mask appearance.

What improved:

- support grew from tens of points to hundreds of points
- geometry guard still removed the obvious surface-like overgrowth
- the kept subsets remain mostly linear:
  - one major survivor at `428` points has `linearity = 0.9877`
  - several others remain above `0.84`

What still did not improve enough:

- object fusion is still fragmented
- all survivors remain `single_fine_candidate`
- even the tighter variant still leaves one large expanded candidate to demote

Practical conclusion:

- this branch is strictly better than pure thin-mask tuning
- the tighter setting is the better default:
  - it drops kept points from `942` to `814`
  - but reduces demoted surface-like growth from `2` candidates to `1`
  - and increases kept candidates from `6` to `7`
- the correct next step for `railing` is now:
  - keep detector-box constrained growth
  - tighten growth acceptance
  - add cross-view accumulation / tracklet merge after support has been grown

This changes the earlier conclusion in one narrow way:

- `railing` is still not production-ready
- but the bottleneck has moved from “mask post-process cannot help” to
  “support-preserving growth works, yet cross-view consolidation is still weak”

## 2. Equipment/HVAC Strict Precision

The broader equipment branch remained semantically risky. A stricter
phrase/shape filter was evaluated to measure how much of the branch survives
when surface pollution is treated as the primary failure mode.

Strict-precision filter summary:

- `accepted_count = 34`
- `rejected_count = 278`
- accepted focus:
  - `equipment = 34`
- main rejection reasons:
  - `low_score = 146`
  - `phrase_too_weak = 62`
  - `phrase_too_broad = 49`
  - `oversized_mask = 14`
  - `tiny_mask = 6`

Fused object result:

- `candidate_count = 25`
- `fine_object_count = 19`
- `point_count = 3051`
- `merge_count = 6`
- status:
  - `single_fine_candidate = 16`
  - `stable_fine_object = 3`

Global vote result:

- `voxel_count = 857`
- `object_count = 35`
- status:
  - `stable = 16`
  - `single_voxel = 19`
- all surviving voxels remain labeled `equipment`

### Interpretation

This is materially better than the current `railing` branch in one important
sense: once the phrase gate is made strict, the branch can still survive into
nontrivial 3D support.

But the same evidence also shows the remaining limit:

- the branch is viable only under aggressive rejection
- semantics are still too coarse if the target identity matters beyond the
  umbrella `equipment/HVAC` family

Practical conclusion:

- keep `equipment/HVAC` as a viable fine-object branch
- keep strict phrase gating as mandatory, not optional
- if identity granularity matters later, add object-level text description
  after geometry consolidation instead of relaxing the detector phrases now

## 3. Fine-Object Update

The Route1 fine-object ranking remains:

1. `pipe`
2. `equipment/HVAC` under strict precision
3. `railing` with box-growth as the preferred guarded proposal branch

This update makes the `railing` limitation clearer:

- current issue is not mainly label quality
- current issue is now mostly cross-view support continuity and consolidation

So the next productive iteration should not be another broad phrase sweep.
It should be a support-preserving branch focused on:

- candidate-region projection
- 3D geometry growth from thin seeds
- cross-view accumulation before fusion
