# Railing-Rich Grounded Eval - 2000-2999

## Goal

Verify whether the current `GroundingDINO + SAM2` fine-object detector route
can isolate railing-like structures on the problematic tail segment
`2000-2999`, instead of swallowing broad wall / fence / mesh regions.

This is a targeted 2D diagnostic only. It does not replace the current 3D
fusion route.

## Input

- semantic source:
  `/root/epfs/manifold_3dgs_project/processed/semantic_eval_mimo25_identity_tail_2000_2999_s10`
- combo:
  `sam2_prompt_v3_sky_label_merge_completion`
- manifest builder:
  `/Users/skkac/Work/SCAN/new_route/experiments/fine_object_grounded_small_eval/build_railing_rich_manifest.py`
- remote eval runtime:
  `/root/epfs/conda_envs/conceptseg-r1/bin/python`
- remote inference script:
  `/Users/skkac/Work/SCAN/new_route/experiments/fine_object_grounded_small_eval/remote_batch_eval.py`

## Sample selection

- selected `12` images whose existing labels contain the strongest
  `railing / guardrail / handrail / fence` signal
- all prompts were limited to the railing group:
  `railing / guardrail / handrail / metal fence`

## Result

- total detections: `36`
- median mask area ratio: `0.0541`
- max mask area ratio: `0.5974`
- detections violating a simple guard
  (`mask_area_ratio > 0.18` or `box_aspect_ratio < 2.2`): `26 / 36`

Representative failures:

- `cam0_002640`
  - phrase: `metal fence`
  - score: `0.315`
  - mask area ratio: `0.5974`
  - aspect ratio: `1.23`
- `cam0_002780`
  - phrase: `metal fence`
  - score: `0.272`
  - mask area ratio: `0.3802`
  - aspect ratio: `1.23`
- `cam0_002920`
  - phrase: `##rail metal fence`
  - score: `0.289`
  - mask area ratio: `0.3682`
  - aspect ratio: `1.24`

## Interpretation

This run confirms that the current detector branch has moved the problem
earlier in the pipeline:

- it is no longer missing railing-like regions entirely
- but it still often detects a **broad fence / mesh surface patch**
  instead of the thin railing structure itself

So the main railing/wall confusion is currently rooted in **2D candidate
quality**, not only in 3D fusion.

More concretely:

1. `metal fence` is the noisiest phrase in this scene family
2. many accepted-looking detections are not elongated enough to represent a
   real thin railing target
3. SAM2 then faithfully fills the broad candidate region, which later pollutes
   wall-like spatial zones in 3D

## Immediate next step

Do not broaden this branch yet. Tighten the railing candidate stage first:

1. split `railing / guardrail / handrail` from `metal fence`
2. reject low-elongation fence-like boxes earlier
3. keep `metal fence` only as a fallback phrase, not a first-class positive
   target
4. for accepted masks, add a thin-structure guard using mask skeleton length or
   width profile before point projection

## Strict prompt A/B

After the first run, the same `12` samples were re-run with a stricter prompt:

- strict:
  `railing / guardrail / handrail`
- removed:
  `metal fence`

Comparison:

| variant | detections | median mask area ratio | max mask area ratio | problematic |
| --- | ---: | ---: | ---: | ---: |
| with `metal fence` | `36` | `0.0541` | `0.5974` | `26` |
| strict no-fence | `30` | `0.0507` | `0.5976` | `20` |

Interpretation:

- removing `metal fence` reduces the total number of noisy detections
- the most obvious broad surface-swallowing cases become less frequent
- but one severe large-surface failure still remains, so phrase cleanup alone
  is not sufficient

Therefore the next tightening step should combine:

1. strict positive prompt by default
2. fallback `metal fence` only when strict prompt returns no useful candidate
3. geometric rejection on low aspect-ratio / high area-ratio masks before 3D
   projection

## Strict prompt + geometry filter v2

The strict prompt was then combined with stronger 2D geometry filtering:

- reject `oversized_mask`
- reject `not_elongated`
- reject `too_filled_for_railing`
- reject `too_fragmented`
- for multiple accepted-looking railing detections in one image, rank by:
  - oriented elongation
  - largest-component ratio
  - lower fill ratio
  - smaller area ratio
  - then `grounding_score`

Filtered result:

- accepted 2D detections: `10`
- rejected 2D detections: `14`
- rejection reasons:
  - `not_elongated = 9`
  - `oversized_mask = 5`

3D projection result:

- projected successfully: `9`
- `mask_no_points`: `1`

Projected railing subsets stayed compact:

- `cam0_002640`: `62` points, ratio `0.0041`
- `cam2_002670`: `714` points, ratio `0.0288`
- `cam2_002840`: `334` points, ratio `0.0231`
- `cam0_002780`: `53` points, ratio `0.0019`
- `cam0_002920`: `39` points, ratio `0.0015`

Interpretation:

- this stage no longer returns the large broad fence/wall blobs seen in the
  earlier `metal fence` run
- the accepted results are now much closer to "thin-object candidate subsets"
  than to coarse wall-like surface regions
- the remaining issue is recall/selectivity balance, not catastrophic
  surface swallowing

## Accepted format integration

The grounded `railing strict v2` path was then upgraded to emit the same core
fields expected by the existing fine-object fusion chain:

- projected accepted PLY now carries:
  - `accepted_candidate`
  - `visual_red/green/blue`
  - `frame/camera/mask/point_index`
  - source metadata compatible with the accepted fine-object path
- projection stage now also writes `accepted_report.json` with:
  - 3D bbox
  - centroid
  - mean visual color
  - PCA linearity / planarity

Server rerun result on:

- `/root/epfs/new_route_stage1_skymask/railing_rich_grounded_eval_2000_2999_strict_v2/projected_accepted_v1`
- `/root/epfs/new_route_stage1_skymask/railing_rich_grounded_eval_2000_2999_strict_v2/fused_accepted_v1`

Summary:

- accepted candidates: `9`
- accepted points: `1295`
- fused fine objects: `9`
- merge count: `0`

Interpretation:

- the new grounded output is now compatible with the object-fusion toolchain
- for this small railing-rich set, the surviving 3D candidates remain spatially
  separate enough that they do not merge under the current conservative
  thresholds
- this means the next useful expansion is not to relax merge rules on this
  sample, but to feed the same accepted-format path with `pipe` and
  `equipment/HVAC` candidates and compare cross-focus object statistics
## Thin-structure refinement replay

To isolate whether the remaining `railing` fragmentation is primarily a
**2D mask-shape** problem or a deeper 3D visibility problem, the already
accepted strict-v2 masks were replayed without rerunning GroundingDINO or SAM2.

New local script:

- `/Users/skkac/Work/SCAN/new_route/experiments/fine_object_grounded_small_eval/refine_thin_structure_masks.py`

Method:

1. load the existing accepted `railing` masks
2. build a morphological skeleton
3. dilate the skeleton into a narrow band
4. keep elongated band components only
5. replay the same validated chain:
   - projection
   - projected geometry guard
   - fine-object fusion
   - promoted global votes

### Baseline vs guard-only

Reference paths:

- baseline projection:
  `/root/epfs/new_route_stage1_skymask/railing_rich_grounded_eval_2000_2999_strict_v2/projected_accepted_v1`
- guard-only projection:
  `/root/epfs/new_route_stage1_skymask/railing_rich_grounded_eval_2000_2999_strict_v2/projected_guard_focus_v1`
- guard-only promoted votes:
  `/root/epfs/new_route_stage1_skymask/railing_rich_grounded_eval_2000_2999_strict_v2/promoted_guard_focus_v1`

Reference numbers:

- baseline:
  - projected candidates: `9`
  - projected points: `1295`
  - vote objects: `70`
  - vote status:
    - `stable = 10`
    - `single_voxel = 60`
- guard-only:
  - fused objects: `6`
  - fused points: `237`
  - vote objects: `44`
  - vote status:
    - `stable = 5`
    - `single_voxel = 39`

Interpretation:

- the projected geometry guard already removes a real amount of wall-like
  contamination
- but the branch remains highly fragmented after promotion

### Thin refine v1

Paths:

- refined masks:
  `/root/epfs/new_route_stage1_skymask/railing_rich_grounded_eval_2000_2999_strict_v2/refined_thin_structure_v1`
- replayed projection:
  `/root/epfs/new_route_stage1_skymask/railing_rich_grounded_eval_2000_2999_strict_v2/projected_refined_thin_v1`
- replayed promoted votes:
  `/root/epfs/new_route_stage1_skymask/railing_rich_grounded_eval_2000_2999_strict_v2/promoted_refined_thin_guard_v1`

Numbers:

- mean refined/original mask area ratio: `0.117`
- projected candidates: `7`
- projected points: `101`
- guard-kept candidates: `4`
- guard-kept points: `25`
- vote objects: `4`
- vote status:
  - `single_voxel = 4`

Interpretation:

- this proves the replay direction is real: mask-shape tightening can collapse
  the broad fragmentation
- but this first parameter set is too aggressive and destroys recall

### Thin refine v2 (relaxed)

Paths:

- refined masks:
  `/root/epfs/new_route_stage1_skymask/railing_rich_grounded_eval_2000_2999_strict_v2/refined_thin_structure_v2_relaxed`
- replayed projection:
  `/root/epfs/new_route_stage1_skymask/railing_rich_grounded_eval_2000_2999_strict_v2/projected_refined_thin_v2_relaxed`
- replayed promoted votes:
  `/root/epfs/new_route_stage1_skymask/railing_rich_grounded_eval_2000_2999_strict_v2/promoted_refined_thin_guard_v2_relaxed`

Numbers:

- mean refined/original mask area ratio: `0.219`
- projected candidates: `8`
- projected points: `192`
- guard-kept candidates: `5`
- guard-kept points: `58`
- vote objects: `2`
- vote status:
  - `stable = 2`

Interpretation:

- this is the first strong evidence that the current `railing` bottleneck is
  primarily upstream in **2D mask shape**, not only prompt phrasing or global
  vote settings
- a relaxed thin-structure replay can reduce the final fragmentation from
  `70` vote objects (`60` singleton voxels) to only `2` stable objects
- however, that comes at a large recall cost because only `58` guard-kept
  points remain

## Current conclusion

`railing` is now much better localized:

1. prompt cleanup helped
2. projected geometry guard helped
3. but the decisive lever is still mask-shape control before projection

The next iteration should therefore not spend its first budget on more prompt
variants. It should refine the **thin-mask replay** itself:

1. hybrid fallback between original and refined mask per component
2. width-aware trimming that preserves long main rails but does not keep broad
   mesh interiors
3. visibility / occlusion checks only after this shape-control step is tuned
