# Ceiling sweep on `2000-2999 surface024_fine012`

## Why this sweep exists

The first plane-split strict-surface rerun already proved that the main
`wall -> floor` collapse was primarily a geometry/fusion problem:

- baseline wall objects: `6`
- strict surface rerun wall objects: `654`
- same-label consolidated wall objects: `574`

However, that first rerun still produced:

- ceiling objects: `0`

So the remaining question was narrower:

> can a conservative geometry-only ceiling heuristic recover some ceiling
> surfaces without breaking the wall recovery?

## Variants

### `v1`: no ceiling heuristic

Artifact:

- summary:
  `/Users/skkac/Work/SCAN/new_route/experiments/surface_plane_split_eval/summary_2000_2999_surface024_fine012_v1.json`

Headline:

- strict wall objects: `654`
- strict floor objects: `905`
- strict ceiling objects: `0`

### `v2`: conservative ceiling heuristic

Added in `split_surface_targets_by_plane.py`:

- source labels: `floor`, `building`
- horizontal only
- `centroid_z >= 2.0`
- `xy_area <= 8.0`
- `z_extent <= 0.35`
- `minor_extent >= 0.30`
- `aspect_ratio <= 4.0`

Artifact:

- summary:
  `/Users/skkac/Work/SCAN/new_route/experiments/surface_plane_split_eval/summary_2000_2999_surface024_fine012_v2_ceiling.json`

Observed split signal:

- `child_label:ceiling = 237`
- `floor -> ceiling = 225`
- `building -> ceiling = 12`

Fusion result:

- strict ceiling objects: `5`
- strict ceiling points: `26,775`
- strict ceiling targets: `133`
- strict wall objects: `652`
- strict floor objects: `899`

Interpretation:

- the heuristic is directionally correct
- ceiling survives fusion as a real label instead of disappearing immediately
- but recovery is still weak relative to the amount of `floor -> ceiling`
  candidate relabeling at split time

### `v3`: relaxed ceiling area cap

Only one parameter changed from `v2`:

- `ceiling_max_xy_area: 8.0 -> 12.0`

Artifact:

- summary:
  `/Users/skkac/Work/SCAN/new_route/experiments/surface_plane_split_eval/summary_2000_2999_surface024_fine012_v3_ceiling_xy12.json`

Observed split signal:

- `child_label:ceiling = 267`
- `floor -> ceiling = 254`
- `building -> ceiling = 13`

Fusion result:

- strict ceiling objects: `6`
- strict ceiling points: `50,253`
- strict ceiling targets: `154`
- strict wall objects: `651`
- strict floor objects: `896`

## Comparison

| variant | wall objects | floor objects | ceiling objects | ceiling points |
| --- | ---: | ---: | ---: | ---: |
| `v1` | `654` | `905` | `0` | `0` |
| `v2` | `652` | `899` | `5` | `26,775` |
| `v3` | `651` | `896` | `6` | `50,253` |

## What this proves

1. The big wall/floor failure is already fixed at the right level:
   `surface target split + strict surface fusion`.
2. A conservative ceiling heuristic can recover a non-zero ceiling layer
   without collapsing wall recovery.
3. Simply relaxing the area cap gives only marginal extra gain:
   `5 -> 6` ceiling objects.

So the current ceiling bottleneck is **not** mainly about a slightly too-tight
area threshold.

## Current diagnosis

The remaining missing signal is structural, not lexical:

- wall/floor needed plane-aware target repair
- ceiling now needs a better notion of *underside / enclosed roof-like
  horizontal structure* rather than just *high horizontal patch*

In other words, the next useful improvement is not another broad threshold
sweep. It should incorporate at least one stronger cue such as:

1. local top-band consistency per frame instead of only absolute `z`
2. relation to nearby vertical wall/building supports
3. exclusion of broad rooftop cap surfaces versus compact enclosed overhead
   slabs

## Recommendation

Keep `v3` as the current best bounded ceiling-aware surface baseline because it
improves ceiling from `0` to `6` while keeping the wall recovery intact.

But do not treat threshold tuning as the main next path. The next step should
be a new structural ceiling cue, not `xy_area` tuning alone.
