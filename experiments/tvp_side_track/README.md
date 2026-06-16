# TVP Side-Track

This folder holds the minimal server-side validation assets for
`Thinking-with-Visual-Primitives-pytorch`.

## Current scope

- `tvp_candidate_manifest_10.json`
  - 10-sample manifest built from accepted fine-object candidates.
- `run_tvp_manifest_inference.py`
  - raw manifest runner that parses `<|ref|>...<|box|>` and `<|point|>...`
    outputs.
- `../scripts/run_tvp_text_label_eval.py`
  - closed-vocabulary crop/text benchmark runner for the same TVP candidates.

## 2026-06-16 smoke result

Server: `scan-train`

Validated models:

- `yunfengwang/TVP-OPD-Qwen2VL-2B`
- `yunfengwang/TVP-SFTBox-Qwen2VL-2B`

Single-sample prompt:

- `Locate the railing in the image.`

Observed behavior on the current railing sample:

- both models loaded successfully on the server
- both models returned plain text
- both models returned `0` boxes and `0` points
- both responses said: `There is no railing visible in the image.`

## 2026-06-16 explicit multi-class smoke

The same server was then tested again with a broader 6-sample manifest and
explicit locate prompts:

- `Locate the pipe in the image.`
- `Locate the HVAC outdoor unit in the image.`
- `Locate the rooftop equipment box in the image.`
- `Locate the railing in the image.`
- `Locate the guardrail in the image.`

Outputs:

- `/root/epfs/model_side_tracks/tvp/tvp_opd_explicit_6.jsonl`
- `/root/epfs/model_side_tracks/tvp/tvp_sftbox_explicit_6.jsonl`

Observed behavior:

- both models still returned `0` boxes and `0` points` on all 6 samples
- some responses gave coarse natural-language location hints such as
  `towards the bottom right corner`
- but the primitive output channel remained empty

Interpretation:

- TVP is not merely failing on one difficult railing sample
- on the current rooftop data, it is not emitting usable primitives for
  `pipe`, `HVAC/equipment`, or `railing`
- therefore it is not currently actionable as an automatic proposal source
  either, unless we add a separate NLP-to-region post-processor, which would be
  a different project

## 2026-06-16 bbox-crop sanity run

One remaining confounder was then tested directly: perhaps the previous TVP
failures were caused only by tiny targets inside full-resolution frames.

To test that, the existing accepted-candidate manifest was converted into a
6-sample bbox-crop manifest using:

- `build_tvp_bboxcrop_manifest.py`

Artifacts:

- crop manifest:
  `/root/epfs/model_side_tracks/tvp/tvp_bboxcrop_manifest_6.json`
- cropped images:
  `/root/epfs/model_side_tracks/tvp/bboxcrop_6`
- outputs:
  - `/root/epfs/model_side_tracks/tvp/tvp_opd_bboxcrop_6.jsonl`
  - `/root/epfs/model_side_tracks/tvp/tvp_sftbox_bboxcrop_6.jsonl`

Observed behavior:

- both models again returned `0` boxes and `0` points on all 6 cropped samples
- this time the models often described the visible object in plain text
  correctly, for example `thin metal guardrail` or `rooftop equipment box or
  HVAC unit`
- however, the structured primitive channel still remained empty

Interpretation:

- the failure is not only caused by tiny-object scale in a full image
- even after cropping around the target, TVP still does not produce usable
  primitive outputs on this rooftop dataset
- therefore TVP is not just a bad dense-semantic fit; it is also not currently
  a reliable fine-object proposal source for this project

## 2026-06-16 independent crop-level closed-vocabulary benchmark

The primitive-output runs above answer one question only: does TVP emit usable
boxes or points on our rooftop data? The answer is still no.

To separate that from pure object recognition, the same candidate set was then
tested as an independent crop-level closed-vocabulary benchmark using
`run_tvp_text_label_eval.py`.

Server artifacts:

- manifest:
  `/root/epfs/model_side_tracks/tvp/tvp_candidate_manifest_10.json`
- cropped-label eval dir:
  `/root/epfs/model_side_tracks/tvp/tvp_text_eval_v001`
- outputs:
  - `/root/epfs/model_side_tracks/tvp/tvp_text_eval_v001/predictions.jsonl`
  - `/root/epfs/model_side_tracks/tvp/tvp_text_eval_v001/report.json`
  - `/root/epfs/model_side_tracks/tvp/tvp_text_eval_v001/crops`

Benchmark prompt:

- `Choose one label for the cropped rooftop object: railing, pipe, equipment, none. Return one label only.`

Important audit note:

- the current server `report.json` records `accuracy = 0.3`
- that file was generated before the truth-label precedence was fixed
- the old runner resolved truth from `source_label` first
- in this manifest, all 10 `source_label` values are the coarse upstream label
  `equipment`
- therefore the raw `report.json` measures only `predicted label vs coarse
  source bucket`, not the intended per-crop fine label

Corrected interpretation against the intended benchmark target
(`answer_class` / `concept_class`):

- sample count: `10`
- class mix: `3 railing`, `5 pipe`, `2 equipment`
- exact accuracy: `0.2` (`2 / 10`)
- predictions used only two labels: `railing` and `equipment`
- confusion:
  - `railing`: `1 railing`, `2 equipment`
  - `pipe`: `5 railing`, `0 pipe`
  - `equipment`: `1 equipment`, `1 railing`

Interpretation:

- the crop-level prompt does make TVP emit a class word consistently
- but the discrimination is too weak to be operationally useful
- the most obvious failure mode is systematic `pipe -> railing`
- so even under the friendliest setting for TVP in this repo
  (`bbox crop + closed vocab + no primitive parsing`), it still does not clear
  the bar for a dependable rooftop fine-object recognizer

## Interpretation

At the moment TVP is not a drop-in replacement for the dense semantic route.

- It is a visual-primitive model for box/point reasoning, not dense mask
  generation.
- On our rooftop candidate set, it does not yet produce usable primitives.
- Even after removing the primitive requirement and forcing a closed vocabulary
  on cropped targets, it reaches only `20%` fine-label accuracy.
- The most reasonable role for TVP remains a proposal / spatial-anchor
  side-track, not the main 2D dense segmentation path.

## Integration verdict for current project

The user's suggested idea was: merge the full PLY first, then use
`img_pos/cam_in_ex` plus TVP semantics to project back into space.

For the current repository, that is not the direct fit:

- TVP does not emit dense semantic maps; it emits sparse visual primitives such
  as boxes and points.
- Therefore TVP can help with `where to look`, but not with dense surface or
  thin-structure coverage by itself.
- If we use it at all, the correct integration point is:
  `TVP primitive proposal -> SAM2 local mask -> validated 3D projection`,
  not `TVP -> direct whole-scene semantic projection`.

## Stop condition

The bbox-crop sanity run closes the only serious remaining confounder for this
side-track.

Current stop condition is now satisfied:

- `tvp_opd_bboxcrop_6.jsonl`: `sum_boxes = 0`, `sum_points = 0`
- `tvp_sftbox_bboxcrop_6.jsonl`: `sum_boxes = 0`, `sum_points = 0`
- `tvp_text_eval_v001`: corrected crop-level closed-vocab accuracy = `0.2`

Therefore:

1. TVP is not compatible as a direct dense semantic source for the current
   `semantic.png -> project_semantic.py` route.
2. TVP is not worth further pursuit as an automatic proposal side-track on the
   current rooftop dataset unless the task itself changes materially
   (for example: new model family, targeted fine-tune, or a separate
   NLP-to-region / box-to-mask subsystem).
3. This side-track should be treated as closed for the current project goal.

## Engineering note

The server runner now supports a persistent snapshot directory via
`--download-dir` / `TVP_DOWNLOAD_DIR` so model weights no longer spill into
`/tmp`.

The text-label runner now resolves ground truth in the right order for this
side-track:

- default truth precedence:
  `answer_class -> concept_class -> source_label`
- emitted rows now also record:
  `truth_field`, `answer_class`, and `concept_class`

The manifest builder now also supports prompt variants:

- `--prompt-mode concept`
- `--prompt-mode locate --locate-field answer_class`
- `build_tvp_bboxcrop_manifest.py` for the final bbox-crop sanity check

This makes prompt-shape comparisons reproducible instead of one-off shell edits.
