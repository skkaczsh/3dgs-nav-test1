# Surface plane-split eval on `2000-2999 surface024_fine012`

## Scope

This is a bounded geometry-only follow-up on the existing authoritative
target/object bundle:

- input bundle:
  `/root/epfs/new_route_stage1_skymask/target_object_fusion_2000_2999_surface024_fine012`
- orchestrator:
  `/Users/skkac/Work/SCAN/new_route/scripts/run_surface_plane_split_eval.py`
- remote output:
  `/root/epfs/new_route_stage1_skymask/surface_plane_split_eval_2000_2999_surface024_fine012_v1`

Pipeline:

1. analyze current target/object bottleneck
2. split large surface targets by plane consistency
3. rerun object fusion with `--strict-surface-labels`
4. run same-label surface consolidation on `floor / wall / building`

## Goal

Test whether the current `floor / wall` collapse is primarily a geometry/fusion
problem rather than a VLM-label problem.

## Artifacts

Local copies:

- summary:
  `/Users/skkac/Work/SCAN/new_route/experiments/surface_plane_split_eval/summary_2000_2999_surface024_fine012_v1.json`
- split report:
  `/Users/skkac/Work/SCAN/new_route/experiments/surface_plane_split_eval/split_surface_targets_report_2000_2999_surface024_fine012_v1.json`
- same-label consolidation report:
  `/Users/skkac/Work/SCAN/new_route/experiments/surface_plane_split_eval/same_label_surface_consolidation_report_2000_2999_surface024_fine012_v1.json`

Remote outputs:

- split targets:
  `/root/epfs/new_route_stage1_skymask/surface_plane_split_eval_2000_2999_surface024_fine012_v1/split_targets`
- strict fusion:
  `/root/epfs/new_route_stage1_skymask/surface_plane_split_eval_2000_2999_surface024_fine012_v1/fused_strict_surface`
- reports:
  `/root/epfs/new_route_stage1_skymask/surface_plane_split_eval_2000_2999_surface024_fine012_v1/reports`

## Headline result

The result is strong enough to change what we blame:

- baseline wall objects: `6`
- after plane split + strict surface fusion: `654`
- after same-label consolidation: `574`

At the same time:

- baseline floor objects: `588`
- strict surface fusion floor objects: `905`
- consolidated floor objects: `789`

And ceiling remains unresolved:

- baseline ceiling objects: `0`
- strict surface fusion ceiling objects: `0`
- consolidated ceiling objects: `0`

## Surface target split signal

From `5957` original targets:

- `2486` were split (`split_ratio = 0.4173`)
- `20421` child targets were created
- major relabel flows:
  - `floor -> wall = 10264`
  - `building -> wall = 452`
  - `building -> floor = 141`
  - `wall -> floor = 315`

Interpretation:

- The existing bundle contains a large number of mixed or over-merged surface
  targets.
- The wall signal was mostly being swallowed before object fusion, not merely
  misnamed after fusion.

## Strict surface fusion result

After re-fusing the split targets with `--strict-surface-labels`:

- targets: `23892`
- objects: `2051`
- merge ratio: `0.9142`
- ambiguous objects: `7`

Object counts by label:

- `wall`: `654` objects, `7,105,787` points
- `floor`: `905` objects, `4,136,104` points
- `building`: `109` objects, `41,681` points

Compared with baseline:

- wall points moved from effectively absent (`4,067`) to a dominant,
  geometrically plausible surface layer (`7.1M`)
- ambiguous points collapsed sharply (`1,318,266 -> 3,140`)

## Same-label surface consolidation result

Same-label-only consolidation reduced fragmentation further:

- input objects: `2051`
- output objects: `1852`
- merged reduction: `199`

By label:

- `floor`: `905 -> 789`
- `wall`: `654 -> 574`
- `building`: `109 -> 106`

Interpretation:

- The strict-surface rerun recovers the missing wall layer.
- Same-label consolidation helps, but only moderately. The remaining wall/floor
  fragmentation is no longer the catastrophic failure mode.

## What this means

This experiment changes the diagnosis:

1. `wall -> floor` collapse in the current `2000-2999` bundle is mainly caused
   by **mixed-plane surface targets + permissive cross-label surface fusion**.
2. A geometry-first surface repair pass is more important than another VLM
   prompt iteration for this failure mode.
3. `ceiling` is still unsolved because this chain does not yet have a reliable
   ceiling relabel heuristic; it only recovers `floor / wall / building`.

## Recommended next step

Do not spend the next cycle on another generic VLM relabel sweep.

Instead:

1. keep this `plane split -> strict surface fusion -> same-label consolidate`
   chain as the new surface repair baseline
2. add a bounded `ceiling` heuristic to the plane-split stage
3. only then re-evaluate the global semantic vote output in the viewer

The next improvement target is therefore **ceiling recovery**, not general wall
recovery.
