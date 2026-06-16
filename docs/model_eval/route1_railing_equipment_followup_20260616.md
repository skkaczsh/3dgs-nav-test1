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
- `/Users/skkac/Work/SCAN/new_route/experiments/fine_object_grounded_small_eval/build_raw_frame_window_manifest.py`
- `/Users/skkac/Work/SCAN/new_route/experiments/fine_object_grounded_small_eval/run_server_box_growth_focus_eval.py`
- `/Users/skkac/Work/SCAN/new_route/experiments/fine_object_grounded_small_eval/railing_rich_2000_2999_box_growth_v1`
- `/Users/skkac/Work/SCAN/new_route/experiments/fine_object_grounded_small_eval/railing_rich_2000_2999_box_growth_v2_tight`
- `/Users/skkac/Work/SCAN/new_route/scripts/run_box_growth_tracklet_pipeline.py`
- `/Users/skkac/Work/SCAN/new_route/experiments/fine_object_grounded_small_eval/railing_rich_2000_2999_box_growth_v2_tight_tracklet_samecand_loose`
- `/Users/skkac/Work/SCAN/new_route/experiments/fine_object_grounded_small_eval/raw_adjacent_railing_2760_2860`

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

## 1.6. Tracklet Consolidation on Box-Growth

The box-growth branch is now wrapped in a reproducible runner:

- `/Users/skkac/Work/SCAN/new_route/scripts/run_box_growth_tracklet_pipeline.py`

This script runs:

1. enriched PLY -> frame fine targets
2. frame targets -> short tracklets
3. tracklets -> long objects

### Default consolidation

On `box_growth_v2_tight`:

- `814` guarded points
- `23` frame targets
- `19` short tracklets
- `14` long objects

Default merge behavior:

- frame-target to tracklet merge ratio: `0.174`
- tracklet to long-object merge ratio: `0.263`
- long-association reasons:
  - `same_accepted_candidate = 5`

Interpretation:

- the new branch no longer dies in the projection stage
- fragmentation is now pushed downstream into 3D clustering and same-candidate
  reconsolidation

### Same-candidate loose association

Only the `same_accepted_candidate` thresholds were loosened:

- same-candidate centroid distance: `3.5`
- same-candidate bbox distance: `1.5`
- same-candidate color distance: `120`

Result:

- `23` frame targets
- `19` short tracklets
- `13` long objects
- long-association merge ratio: `0.316`
- `same_accepted_candidate = 6`

Interpretation:

- loosening only same-candidate association does help
- the gain is real but still modest
- the rich sample remains dominated by same-frame fragmentation, not true
  multi-frame repeat observations

This matters because it tells us the next bottleneck more precisely:

- not surface-vs-railing semantics
- not only 2D mask recall
- now mainly:
  - same-candidate over-splitting inside one observation
  - insufficient repeated support across neighboring frames

Practical conclusion:

- keep `box_growth_v2_tight` as the preferred railing base
- keep same-candidate loose association as the current better default for this
  branch
- the next improvement should target:
  - in-frame target over-splitting reduction
  - then true neighboring-frame support accumulation on a denser sample than
    this 10-image rich set

## 1.7. Raw Adjacent-Frame Window

The earlier rich sample is still a useful microscope, but it is not a good
test of neighboring-frame accumulation because it is sparse and cherry-picked.
To test the real question, a raw frame-window run was added:

- raw image window:
  `2760-2860`
- stride:
  `10`
- cameras:
  `0, 1, 2`
- total images:
  `33`

New helper scripts:

- `/Users/skkac/Work/SCAN/new_route/experiments/fine_object_grounded_small_eval/build_raw_frame_window_manifest.py`
- `/Users/skkac/Work/SCAN/new_route/experiments/fine_object_grounded_small_eval/run_server_box_growth_focus_eval.py`

Local evidence:

- `/Users/skkac/Work/SCAN/new_route/experiments/fine_object_grounded_small_eval/raw_adjacent_railing_2760_2860`

### Detection and 3D support

Detector + SAM2 + railing filter:

- accepted 2D detections: `26`
- rejected 2D detections: `47`
- main rejection reasons:
  - `oversized_mask = 24`
  - `not_elongated = 20`
  - `too_filled_for_railing = 3`

Box-growth + geometry guard:

- projected candidates: `24`
- kept candidates: `22`
- accepted 3D points before guard decision:
  `3800`
- kept 3D points:
  `3679`

This is already a meaningful shift from the 10-image sample:

- the branch now produces sustained support across a real neighboring-frame
  segment
- the geometry guard no longer spends its time stripping away mostly bad
  overgrowth; it keeps the overwhelming majority of projected points

### Tracklet and long-object result

Frame-target stage:

- frame targets: `47`
- frame-target kept points: `3679`
- small residual points: `0`

Short tracklets:

- `47 -> 41`
- tracklet merge ratio: `0.128`
- stable tracklets: `6`

Long association:

- `41 -> 22`
- long-object merge ratio: `0.463`
- status:
  - `stable_long_object = 12`
  - `single_tracklet_object = 10`
- merge reasons:
  - `same_accepted_candidate = 18`
  - `strict_cross_source = 1`

### Interpretation

This is the first result that demonstrates real neighboring-frame benefit for
the `railing` branch.

What it proves:

- the `box_growth_v2_tight` route is not limited to the tiny rich sample
- once a real adjacent-frame window is used, consolidation becomes materially
  stronger
- most of the useful merge signal still comes from same-candidate consistency,
  but one merge already crosses sources under the stricter cross-source gate

What it still does not solve:

- frame-target over-splitting is still heavy:
  `22` accepted candidates become `47` targets
- short-tracklet merging remains weak relative to long-association merging
- many large long objects are still single-tracklet observations

Practical conclusion:

- the previous conclusion holds, but is now stronger:
  - `railing` is no longer blocked by 2D support alone
  - the next bottleneck is now clearly target splitting and cross-frame
    persistence
- this branch is worth scaling further on adjacent windows before returning to
  any new model family

## 1.8. In-Frame Same-Candidate Merge

The raw adjacent-window result identified one concrete structural bottleneck:

- `22` accepted 3D candidates
- but `47` frame targets after voxel connected-components

That means the next useful change is not another detector or VLM swap. It is a
small in-frame recombination layer before short-tracklet building.

Implementation update:

- `/Users/skkac/Work/SCAN/new_route/scripts/build_frame_fine_targets_from_enriched.py`
  now supports a same-frame merge pass for fragments that share the same
  dominant `accepted_candidate`
- `/Users/skkac/Work/SCAN/new_route/scripts/run_box_growth_tracklet_pipeline.py`
  now exposes the in-frame merge thresholds explicitly so the replay is
  reproducible

Local replay evidence:

- baseline:
  `/Users/skkac/Work/SCAN/new_route/experiments/fine_object_grounded_small_eval/raw_adjacent_railing_2760_2860`
- conservative merge:
  `/Users/skkac/Work/SCAN/new_route/experiments/fine_object_grounded_small_eval/raw_adjacent_railing_2760_2860/inframe_merge_v1`
- looser merge:
  `/Users/skkac/Work/SCAN/new_route/experiments/fine_object_grounded_small_eval/raw_adjacent_railing_2760_2860/inframe_merge_v2`

### `v1` conservative merge

Parameters:

- `in_frame_candidate_ratio = 0.8`
- `in_frame_centroid_distance = 0.5`
- `in_frame_bbox_distance = 0.5`
- `in_frame_color_distance = 80`

Result relative to baseline:

- frame targets:
  `47 -> 27`
- short tracklets:
  `41 -> 22`
- long objects:
  `22 -> 19`
- mean points per tracklet:
  `89.7 -> 167.2`

Interpretation:

- this removes a large amount of same-frame fragmentation
- larger support moves earlier into frame-target and short-tracklet stages
- the later `same_accepted_candidate` long-association count drops sharply
  because much of that work is no longer deferred downstream

### `v2` looser merge

Parameters:

- `in_frame_candidate_ratio = 0.8`
- `in_frame_centroid_distance = 0.8`
- `in_frame_bbox_distance = 0.8`
- `in_frame_color_distance = 100`

Result relative to baseline:

- frame targets:
  `47 -> 23`
- short tracklets:
  `41 -> 19`
- long objects:
  `22 -> 18`
- mean points per tracklet:
  `89.7 -> 193.6`

Interpretation:

- this pushes target reduction closer to the underlying `22` accepted
  candidates
- but it also erases too much downstream consolidation structure
- the long-object stage is left with only `1` `stable_long_object`

### Practical conclusion

The in-frame merge idea is correct, but the threshold cannot be loose.

What this proves:

- frame-target over-splitting is a real, fixable bottleneck
- conservative same-candidate merge materially reduces fragmentation
- overly loose in-frame merge starts to over-collapse the evidence graph

Current recommendation:

- keep the adjacent-window railing branch
- keep in-frame same-candidate merge enabled
- use the conservative `v1` setting as the current default
- do not broaden to the `v2` setting
- the next improvement should target:
  - candidate-level depth continuity / occlusion guards
  - then cross-frame persistence on denser windows

## 1.9. Seed-Depth Guard Prototype

The next projector-side refinement was not another detector or mask change. It
was a candidate-level depth guard that only trusts local depth support derived
from seed-mask points.

Implementation update:

- `/Users/skkac/Work/SCAN/new_route/experiments/fine_object_grounded_small_eval/project_detector_box_growth.py`
  now supports:
  - local depth-edge guard
  - seed-depth local-min guard
- `/Users/skkac/Work/SCAN/new_route/experiments/fine_object_grounded_small_eval/run_server_box_growth_focus_eval.py`
  now exposes these parameters for remote replays

Tested replay:

- source window:
  `/Users/skkac/Work/SCAN/new_route/experiments/fine_object_grounded_small_eval/raw_adjacent_railing_2760_2860`
- seed-depth replay:
  `/Users/skkac/Work/SCAN/new_route/experiments/fine_object_grounded_small_eval/raw_adjacent_railing_2760_2860/seeddepth_v1`

Parameters:

- `seed_depth_window_px = 1`
- `seed_depth_threshold = 0.12`

### Quantitative effect

Relative to the original adjacent-window replay:

- projected 3D candidate points before review:
  `3800 -> 2347`
- kept 3D points after geometry guard:
  `3679 -> 2247`
- kept candidates after guard:
  `22 -> 21`
- frame targets:
  `27 -> 27`
- short tracklets:
  `22 -> 23`
- long objects:
  `19 -> 18`
- `stable_long_object`:
  `3 -> 5`

Additional projector statistic:

- `seed_depth_filtered_points = 1458`
- all `24` successful projected candidates had some points removed by this gate

### Interpretation

This confirms the intended mechanical behavior:

- the seed-depth guard is not a no-op
- it removes a large number of candidate points before geometry review
- it can suppress one accepted candidate entirely on this window

But it is not yet proven to be a net win.

Why:

- support drops sharply:
  `3679 -> 2247`
- short-tracklet count does not improve
- frame-target count also does not improve beyond the current in-frame merge

The one encouraging sign is that the long-object stage becomes slightly more
compact while keeping more multi-tracklet objects alive:

- `objects: 19 -> 18`
- `stable_long_object: 3 -> 5`

### Practical conclusion

The seed-depth guard is worth keeping as an experimental projector constraint,
but not yet as the default.

Current recommendation:

- keep the conservative in-frame merge as the default
- keep seed-depth guard as an opt-in experimental branch
- next refine this guard by making it selective rather than global:
  - enable only for `railing`
  - gate by mask elongation / candidate linearity
  - avoid stripping support from already-clean candidates

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

## 4. Global Color-Assisted Matching Prototype

A prototype matcher was added to test the user's proposal:

- first keep the validated per-frame projection route
- then use global colored semantic voxels/objects as an additional association
  prior for frame-level targets

Prototype script:

- `/Users/skkac/Work/SCAN/new_route/scripts/match_targets_to_global_color_objects.py`

Scoring inputs:

- exact voxel overlap
- relaxed voxel-neighborhood support
- centroid distance
- bbox distance
- mean RGB distance
- same-label bonus

### Results

1. Same-source sanity check

Using:

- targets:
  `.../railing_rich_grounded_eval_2000_2999_strict_v2/promoted_guard_focus_v1/frame_targets/targets_all.jsonl`
- global votes:
  `.../railing_rich_grounded_eval_2000_2999_strict_v2/promoted_guard_focus_v1/global_votes/*`

Result:

- `match_count = 14 / 14`
- many targets achieved strong exact overlap or near-perfect centroid agreement

Interpretation:

- the matcher itself is valid
- when target generation and global aggregation come from the same source
  branch, it can reliably re-attach frame targets to global objects

2. Cross-run transfer check

Using:

- targets:
  `.../raw_adjacent_railing_2760_2860_run/tracklet_pipeline/frame_targets/targets_all.jsonl`
- global votes:
  `.../railing_rich_grounded_eval_2000_2999_strict_v2/promoted_guard_focus_v1/global_votes/*`

Exact-overlap only:

- `match_count = 0 / 47`

With relaxed support (`neighbor_radius_voxels = 5`):

- `match_count = 3 / 47`
- `best_match.support_overlap_ratio > 0` for `8 / 47`

Interpretation:

- there is some geometric/color association signal
- but hard voxel overlap does not transfer across different sparse sampling
  runs, even when frame ranges overlap
- relaxed support helps a little, but does not turn a different target branch
  into a drop-in compatible global reference

### Practical Conclusion

Global color-assisted matching is useful, but only in a narrow role:

- as an extra association prior inside the same validated production branch
- to stabilize `Target -> Object` attachment
- to provide another cue beyond bbox continuity

It is not a fix for:

- wrong mask coverage
- wrong target geometry
- semantically polluted global objects
- cross-branch inconsistencies between separately generated sparse target sets

So if this route is promoted, the next correct version is:

1. build global colored support from the same target branch
2. use local target geometry + global color/object evidence jointly
3. do not use global semantic voxels from a different sparse branch as a hard
   reference
