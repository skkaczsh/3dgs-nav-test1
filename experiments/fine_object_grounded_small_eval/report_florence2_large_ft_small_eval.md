# Florence-2 Large FT Small Eval - 2026-06-16

## Goal

Check whether `microsoft/Florence-2-large-ft` can replace or complement the
current `GroundingDINO + SAM2` fine-object route on the same small review set.

The comparison target is not raw 2D recall alone. The real question is whether
the accepted detections survive the validated 3D route and remain clusterable in
global votes.

## Runtime

- Server: `scan-train`
- Model: `microsoft/Florence-2-large-ft`
- SAM backend: `SAM2` on `cuda:0`
- Cache root:
  `/root/epfs/model_side_tracks/florence2/cache`
- Output root:
  `/root/epfs/vlm_seg_project/tmp_fine_object_florence_small_eval`

Key scripts:

- local eval runner:
  `/Users/skkac/Work/SCAN/new_route/experiments/fine_object_grounded_small_eval/remote_batch_eval_florence.py`
- projection / filtering / fusion:
  `/root/epfs/new_route_scripts/filter_grouped_detections.py`
  `/root/epfs/new_route_scripts/project_filtered_grouped_masks.py`
  `/root/epfs/new_route_scripts/fuse_accepted_fine_objects.py`

## 2D output summary

Primary summary:

- `/root/epfs/vlm_seg_project/tmp_fine_object_florence_small_eval/outputs_florence_large_ft/summary.json`

Observed detections:

- total samples: `14`
- focus-box counts:
  - `hvac = 2`
  - `railing = 16`
  - `equipment = 22`
  - `pipe = 3`

Median / max mask area:

- `hvac`: median `35,928`, max `66,211`
- `railing`: median `404,054`, max `1,962,039`
- `equipment`: median `37,992`, max `1,468,381`
- `pipe`: median `104,376`, max `105,410`

Interpretation:

- Florence clearly sees the categories.
- The problem is not missing prompts. The problem is that `railing` and
  `equipment` often come back as very broad regions before filtering.

Representative failure:

- `review_002__a0`
  - `railing` returns a very wide top-band candidate
  - `equipment` returns almost the entire image

## Filtered 2D result

Filter summary:

- `/root/epfs/vlm_seg_project/tmp_fine_object_florence_small_eval/filtered_large_ft/filter_summary.json`

Numbers:

- accepted detections: `14`
- rejected detections: `23`
- accepted by focus:
  - `hvac = 1`
  - `railing = 1`
  - `equipment = 10`
  - `pipe = 2`
- rejected by reason:
  - `oversized_mask = 18`
  - `not_elongated = 3`
  - `too_elongated = 1`
  - `oversized_box = 1`

Interpretation:

- The filter is doing almost all the work.
- `railing` is especially weak: most proposals are eliminated as oversized or
  structurally wrong.

## 3D projection result

Accepted projection summary:

- `/root/epfs/vlm_seg_project/tmp_fine_object_florence_small_eval/projected_large_ft/accepted_report.json`

Numbers:

- accepted 3D candidates: `13`
- accepted 3D points: `2,317`

Notable survivors:

- `railing`: frame `522`, cam `1`, `492` points
- `hvac`: frame `399`, cam `1`, `248` points
- `pipe`: frames `181` and `189`, `244` and `240` points

Interpretation:

- Florence can produce valid projected subsets.
- But the accepted set is strongly skewed toward `equipment`, and the single
  surviving `railing` candidate is still a broad region rather than a clean
  thin-structure detector.

## Fine-object fusion

Fine-object report:

- `/root/epfs/vlm_seg_project/tmp_fine_object_florence_small_eval/fused_large_ft/fine_object_report.json`

Numbers:

- candidate count: `13`
- fine-object count: `12`
- point count: `2,317`
- merge count: `1`
- status:
  - `single_fine_candidate = 11`
  - `stable_fine_object = 1`

Interpretation:

- Florence does not naturally create multi-view reusable fine-object identities
  on this sample.
- Almost every survivor remains a one-off candidate.

## Promoted global votes

The Florence accepted points were pushed through the same validated
`frame targets -> global votes` chain used by the grounded detector route.

Frame-target stage:

- source points: `2,317`
- groups: `12`
- targets: `46`
- kept target points: `2,028`
- small residual points: `289`

Global-vote stage:

- `/root/epfs/vlm_seg_project/tmp_fine_object_florence_small_eval/global_votes_large_ft/global_semantic_vote_report.json`
- voxel count: `1,443`
- object count: `35`
- status:
  - `stable = 15`
  - `single_voxel = 20`
- label counts:
  - `equipment = 881`
  - `railing = 327`
  - `pipe = 235`

Interpretation:

- Florence can be projected and clustered end-to-end.
- But the output is still heavily fragmented and dominated by `equipment`.
- This is not a cleaner fine-object branch than the current grounded-detector
  route.

## Comparison against grounded detector route

Grounded detector references:

- `/Users/skkac/Work/SCAN/new_route/experiments/fine_object_grounded_small_eval/report_20260616.md`
- `/Users/skkac/Work/SCAN/new_route/experiments/fine_object_grounded_small_eval/report_focus_rich_2000_2999.md`
- `/Users/skkac/Work/SCAN/new_route/experiments/fine_object_grounded_small_eval/report_railing_rich_2000_2999.md`

Current 3D-ready ranking remains:

1. `pipe` via `GroundingDINO + SAM2`
2. `railing` via strict grounded prompts plus geometry guards
3. `equipment/HVAC` via grounded prompts, still unstable
4. `Florence-2-large-ft` as a broad proposal side-branch, not a replacement

Why Florence loses right now:

- it over-expands `railing` and `equipment` more often than the strict
  grounded-detector branch
- its accepted set needs heavy geometric rejection
- after 3D promotion it still yields many singleton / fragmented objects

### Fair replay with the same projected geometry guard

After the new projected-candidate geometry guard was added to the validated
grounded route, the Florence projected outputs were replayed through the same
guard so the comparison would stay fair.

Server replay outputs:

- guarded projection:
  `/root/epfs/vlm_seg_project/tmp_fine_object_florence_small_eval/projected_large_ft_guard_focus_v1`
- guarded fusion:
  `/root/epfs/vlm_seg_project/tmp_fine_object_florence_small_eval/fused_large_ft_guard_focus_v1`

Replay result:

- candidates: `13 -> 8 keep`
- points: `2317 -> 1449`
- actions:
  - `keep_linear_railing = 1`
  - `keep_linear_pipe = 2`
  - `keep_compact_equipment = 5`
  - `review_large_equipment = 3`
  - `demote_surface_like_equipment = 1`
  - `demote_line_like = 1`
- fused fine objects: `12 -> 8`
- merge count: `1 -> 0`

Interpretation:

- the geometry guard removes some obvious Florence over-expansions
- but the route still collapses into mostly singleton survivors
- unlike the grounded `pipe` branch, the Florence guarded survivors do not
  become cleaner reusable identities after projection
- therefore the current ranking does **not** change after equalizing the
  post-projection guard

## New promotion results for grounded detector categories

To make the comparison fair, the grounded detector categories were also pushed
to the same global-vote stage:

- `pipe`
  - voxel count: `133`
  - object count: `4`
  - status: all `stable`
- `equipment`
  - voxel count: `350`
  - object count: `5`
  - status: all `stable`
- `railing`
  - voxel count: `813`
  - object count: `70`
  - status:
    - `stable = 10`
    - `single_voxel = 60`

Interpretation:

- `pipe` is still the cleanest transferable fine-object branch.
- `equipment` is clusterable but semantically risky.
- `railing` remains the hardest case because the 2D candidates still break into
  many tiny 3D pieces even after stricter prompt and geometry guards.

## Decision

Florence-2 should not replace the current grounded-detector route.

The practical next use for Florence is narrower:

1. use it as a secondary proposal source on hard `equipment/HVAC` images
2. compare it only against grounded-detector misses
3. do not feed its broad raw output directly into the main global semantic
   route
