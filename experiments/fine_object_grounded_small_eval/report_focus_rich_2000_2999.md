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

### Pipe -> frame targets -> global votes

The `pipe` accepted points were also pushed one step deeper into the validated
target/object route:

- enriched accepted points: `247`
- frame-level fine targets: `9`
- frame target kept points: `221`
- small residual points: `26`
- global semantic vote voxels: `133`
- global vote objects: `4`
- global vote status: all `stable`

Implication:

- `pipe` is currently the only grounded fine-object class that has already
  demonstrated a clean path through accepted 2D detections, correct-route 3D
  projection, frame target generation, and global vote object consolidation.

### Pipe extension on real tail frames

A larger server-side extension was then run on the same `2000-2999` tail
segment using a freshly rebuilt `pipe-rich` manifest instead of reusing the
earlier 12-sample set.

Source:

- manifest:
  `/root/epfs/new_route_stage1_skymask/pipe_rich_grounded_eval_2000_2999_ext80/manifest.json`
- run root:
  `/root/epfs/new_route_stage1_skymask/pipe_rich_grounded_eval_2000_2999_ext17_run`
- promoted votes:
  `/root/epfs/new_route_stage1_skymask/pipe_rich_grounded_eval_2000_2999_ext17_votes`

Summary:

- manifest samples: `17`
- accepted 2D detections: `12`
- rejected 2D detections: `32`
- main rejection reasons:
  - `not_elongated = 22`
  - `oversized_mask = 9`
  - `tiny_mask = 1`
- projected 3D candidates: `12`
- accepted 3D points: `448`
- fused fine objects: `12`
- merge count: `0`

Promoted vote result:

- frame-level fine targets: `19`
- frame target kept points: `374`
- small residual points: `74`
- global semantic vote voxels: `221`
- global vote objects: `47`
- global vote status:
  - `stable = 9`
  - `single_voxel = 38`

Interpretation:

- the `pipe` branch still scales better than the other grounded fine-object
  classes in raw acceptance and projection survival
- but the larger tail sample breaks the earlier illusion that `pipe` naturally
  consolidates into only a handful of stable objects
- once the sample expands from `12` to `17` real tail frames, the branch still
  produces valid thin-object subsets, but many of them remain isolated in the
  current vote/object settings
- therefore `pipe` remains the strongest fine-object branch, but its next
  problem is no longer 2D detection quality; it is cross-frame consolidation
  and object persistence

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

### Equipment extension on real tail frames

A wider server-side extension was then run on the same `2000-2999` tail
segment using a rebuilt `equipment-rich` manifest instead of the earlier
12-sample set.

Source:

- manifest:
  `/root/epfs/new_route_stage1_skymask/equipment_rich_grounded_eval_2000_2999_ext80/manifest.json`
- run root:
  `/root/epfs/new_route_stage1_skymask/equipment_rich_grounded_eval_2000_2999_ext80_run`
- promoted votes:
  `/root/epfs/new_route_stage1_skymask/equipment_rich_grounded_eval_2000_2999_ext80_votes`

Summary:

- manifest samples: `80`
- accepted 2D detections: `37`
- rejected 2D detections: `273`
- projected 3D candidates: `28`
- accepted 3D points: `3480`
- fused fine objects: `21`
- merge count: `7`

Promoted vote result:

- frame-level fine targets: `49`
- frame target kept points: `3252`
- small residual points: `228`
- global semantic vote voxels: `1124`
- global vote objects: `38`
- global vote status:
  - `stable = 19`
  - `single_voxel = 19`

Interpretation:

- `equipment/HVAC` is no longer the weakest grounded branch once widened to
  real tail imagery; it survives projection much better than the earlier
  12-sample result suggested.
- However, that gain comes with a familiar failure mode: broad prompt drift.
  Several accepted image-level detections still correspond to very large
  masks driven by weak phrases such as `outdoor unit` or `unit`.
- In other words, the branch can now produce enough 3D evidence to matter,
  but its main bottleneck is semantic precision, not raw 3D visibility.
- Therefore the next improvement for `equipment/HVAC` should be stricter
  phrase gating and surface-aware rejection, not simply scaling sample count
  further.

## Comparison

Current ranking for grounded fine-object transferability on real tail samples:

1. `pipe`: best precision / cleanest projection, but still weak on cross-frame
   consolidation once widened
2. `equipment/HVAC`: now has materially better 3D survival on the extended
   tail batch, but semantic drift from broad phrases remains the main risk
3. `railing`: still the hardest branch because recall and geometric continuity
   both remain fragile

## Immediate implication

For the next server-side extension of the main route:

- `pipe` remains the cleanest candidate for precision-first thin-object
  promotion, but it needs better object persistence before it scales.
- `equipment/HVAC` is now worth keeping in the promoted branch, but only with
  tighter phrase guards and large-surface rejection.
- `railing` still should not be widened again until we improve either the
  2D detector proposal quality or the post-projection continuity logic.
