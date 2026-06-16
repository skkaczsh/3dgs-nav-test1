# Route1 Small-Eval Recommendation - 2026-06-16

This note consolidates the current small-sample evidence for the proposed
Route1 stack:

1. large-surface semantic baseline
2. grounded fine-object detector route
3. ConceptSeg-R1 side-track

The goal is not to restate every intermediate experiment. The goal is to make
the next engineering decision explicit.

## Decision Summary

Recommended Route1 composition right now:

- large surfaces:
  keep the current mainline image semantic source and add geometry-aware
  surface repair; do **not** replace it with pure Mask2Former/OneFormer
- fine objects:
  use `GroundingDINO/Florence-style prompt -> SAM2 -> 2D geometry guard ->
  correct-route 3D projection`, but prefer the existing grouped
  `GroundingDINO + SAM2` branch over Florence-2
- ConceptSeg-R1:
  keep as reviewer / candidate proposer only; do not move it into the mainline

In short:

- surface branch winner: **none of the pure semantic baselines**
- fine-object branch winner: **grouped GroundingDINO + SAM2**
- side-track reviewer: **ConceptSeg-R1**

## 1. Large-Surface Baseline

Primary evidence:

- `/Users/skkac/Work/SCAN/new_route/experiments/surface_baseline_small_eval/README.md`
- `/Users/skkac/Work/SCAN/new_route/experiments/surface_baseline_small_eval/outputs_compare_gpu0/report.md`
- `/Users/skkac/Work/SCAN/new_route/experiments/surface_baseline_small_eval/outputs_compare_city_map/report.md`
- `/Users/skkac/Work/SCAN/new_route/experiments/surface_baseline_small_eval/outputs_compare_oneformer_ade20k_large/report.md`

Models already tested on the same rooftop-oriented 12-sample set:

- `shi-labs/oneformer_ade20k_swin_tiny`
- `shi-labs/oneformer_ade20k_swin_large`
- `shi-labs/oneformer_cityscapes_swin_large`
- `facebook/mask2former-swin-tiny-ade-semantic`
- `facebook/mask2former-swin-large-cityscapes-semantic`
- `facebook/mask2former-swin-large-mapillary-vistas-semantic`

Evaluated labels:

- `floor/ground`
- `wall`
- `ceiling`
- `building`
- `sky`

Observed pattern:

- `ADE20K` priors over-predict `wall` and `ceiling` on rooftop planes
- `Cityscapes` / `Mapillary` priors shift many rooftop planes toward
  `building`
- `OneFormer` is occasionally less noisy on some wall-vs-ceiling cases
- `Mask2Former` is sometimes smoother overall
- but **none** of these models is stable enough on the rooftop domain to
  replace the current route directly

Concrete verdict:

- pure `Mask2Former/OneFormer` is **not better than the current route** in the
  sense that matters here
- it may produce visually smoother 2D maps, but it still assigns the wrong
  coarse surface class systematically
- therefore the bottleneck is not solved by swapping in a generic universal
  segmentation model

Implication:

- keep large-surface work focused on:
  - geometry-aware repair
  - surface-first relabel
  - domain-specific post rules
- do not spend more immediate budget on more `Mask2Former/OneFormer` checkpoint
  sweeps unless the dataset prior changes materially

## 2. Fine-Object Pipeline

Primary evidence:

- `/Users/skkac/Work/SCAN/new_route/experiments/fine_object_grounded_small_eval/report_20260616.md`
- `/Users/skkac/Work/SCAN/new_route/experiments/fine_object_grounded_small_eval/report_focus_rich_2000_2999.md`
- `/Users/skkac/Work/SCAN/new_route/experiments/fine_object_grounded_small_eval/report_railing_rich_2000_2999.md`
- `/Users/skkac/Work/SCAN/new_route/experiments/fine_object_grounded_small_eval/report_florence2_large_ft_small_eval.md`

### Current best branch

Best current fine-object route:

- grouped `GroundingDINO` prompts
- `SAM2` mask refinement
- 2D geometry filtering
- correct-route 3D projection
- accepted fine-object fusion / global vote promotion

This branch is already operational on `scan-train` with:

- `GroundingDINO = cpu`
- `SAM2 = cuda`

### Category-by-category verdict

`pipe`

- strongest class so far
- best transfer through 2D detection, projection, and downstream 3D stages
- larger tail extension still shows fragmentation, but the branch remains
  clearly usable

`railing`

- workable only after prompt tightening and thin-structure guards
- default broad phrases like `metal fence` are too noisy
- the main bottleneck is now **2D mask shape**, not only prompt recall
- strict prompt + geometry guard is usable
- thin-mask replay shows fragmentation can drop sharply, but current relaxed
  replay still trades away a lot of recall

`equipment/HVAC`

- usable, but semantically risky
- broader phrases frequently expand into surface-like regions
- after widening to real tail imagery the branch survives 3D projection better
  than early smokes suggested
- the next improvement should be stricter phrase gating and surface-aware
  rejection, not more blind scaling

### Florence-2 result

`Florence-2-large-ft` does not beat the grounded-detector branch.

Observed failure:

- category presence is often correct
- raw detections for `railing` and `equipment` are too broad
- filtering does most of the work
- after 3D promotion the survivors remain heavily singleton / fragmented

Concrete verdict:

- Florence-2 is **not** the preferred fine-object branch
- if kept at all, it should stay a secondary proposal source for hard
  `equipment/HVAC` cases only

## 3. ConceptSeg-R1

Primary evidence:

- `/Users/skkac/Work/SCAN/new_route/experiments/conceptseg_r1_small_eval/report.md`
- `/Users/skkac/Work/SCAN/new_route/docs/model_eval/conceptseg_r1_side_track.md`

Stable conclusion across the reviewed runs:

- ConceptSeg-R1 can act as a concept recognizer / constrained candidate source
- it does **not** reliably provide topology-stable thin-structure masks
- it does **not** discriminate targets cleanly enough to become a direct
  semantic source

Most useful role:

- reviewer
- proposal source on ambiguous fine-object crops
- optional second-stage concept hint after existing instance / geometry gates

Not recommended:

- direct dense semantic replacement
- direct target-level semantic overwrite
- broad rollout on floor/wall/building surfaces

## Recommended Next Engineering Steps

Route1 should now branch like this:

1. large surfaces:
   keep the current mainline source, then repair with geometry-aware
   surface-first logic
2. fine objects:
   continue on grouped `GroundingDINO + SAM2`
3. `pipe`:
   continue scaling and improve cross-frame consolidation
4. `railing`:
   prioritize mask-shape control before projection
5. `equipment/HVAC`:
   tighten phrase gate and reject surface-like masks earlier
6. ConceptSeg-R1:
   keep as review-only side-track

## Explicit Recommendation

If one Route1 stack must be chosen today, it should be:

- surfaces:
  current semantic route + geometry-aware surface repair
- fine objects:
  grouped `GroundingDINO + SAM2`
- reviewer:
  `ConceptSeg-R1`

This is the highest-signal, lowest-drift configuration supported by the
current evidence.
