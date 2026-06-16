# Focus-Rich Grounded Eval on Tail 2000-2999

This report extends the strict `railing` small-sample path onto real
`2000-2999` semantic-eval imagery using the same accepted-format pipeline:

1. grouped detector prompts
2. SAM2 masks
3. 2D geometry filtering
4. correct-route 3D projection
5. accepted fine-object fusion

## Runtime finding

On the current `scan-train` server, the practical stable configuration is:

- `GroundingDINO = cpu`
- `SAM2 = cuda:0`

Reason:

- `GroundingDINO` on GPU enters the CUDA path in `ms_deform_attn.py`, but the
  custom `_C` op is not loaded in the current environment.
- `vlm_seg` is also not a reliable fallback because its `transformers` package
  is too new for the local GroundingDINO integration.

Therefore the reusable server wrapper now defaults to CPU detector mode and
sets `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1` to avoid unnecessary remote
tokenizer probing.

## Pipe-rich sample

Source:

- `/root/epfs/new_route_stage1_skymask/pipe_rich_grounded_eval_2000_2999`

Summary:

- manifest samples: `12`
- accepted 2D detections: `6`
- projected 3D candidates: `6 / 6`
- accepted 3D points: `247`
- fused fine objects: `6`
- merge count: `0`

Rejected by reason:

- `not_elongated = 18`
- `oversized_mask = 3`
- `tiny_mask = 2`

Interpretation:

- `pipe` is the cleanest transferable fine-object category so far.
- The geometry filters are doing real work rather than just shrinking masks
  after projection.
- The remaining limitation is sparse 3D support, not catastrophic 2D drift.

## Equipment-rich sample

Source:

- `/root/epfs/new_route_stage1_skymask/equipment_rich_grounded_eval_2000_2999`

Summary:

- manifest samples: `12`
- accepted 2D detections: `9`
- projected 3D candidates with points: `5`
- `mask_no_points = 4`
- accepted 3D points: `537`
- fused fine objects: `5`
- merge count: `0`

Rejected by reason:

- `low_score = 28`
- `oversized_mask = 6`
- `phrase_too_weak = 2`
- `oversized_box = 1`

Interpretation:

- `equipment/HVAC` remains usable as a detector path, but it is much less
  geometrically stable than `pipe`.
- The dominant issue is no longer only broad 2D masks. It is also 3D
  observability: almost half of the 2D-accepted detections do not survive
  projection with points.

## Comparison

Current ranking for grounded fine-object transferability on real tail samples:

1. `pipe`: best precision / cleanest projection
2. `railing`: better than before, but still recall-limited and fragmented
3. `equipment/HVAC`: detector can fire, but 3D survival is weak and broad-mask
   risk remains

## Immediate implication

For the next server-side extension of the main route:

- `pipe` should be the first grounded fine-object class promoted from small
  sample into a larger tail batch.
- `equipment/HVAC` still needs stricter 2D guards and likely depth/surface
  rejection before it is worth widening.
