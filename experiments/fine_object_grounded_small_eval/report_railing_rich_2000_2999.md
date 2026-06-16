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

