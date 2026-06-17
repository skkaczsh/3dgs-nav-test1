# Surface Fine Combo Small Eval

Purpose:

- reuse the existing fine-object small sample set
- run a generic large-surface semantic baseline on the same images
- measure whether `surface-first + fine-object` is structurally promising

Inputs:

- fine sample manifest:
  `/Users/skkac/Work/SCAN/new_route/experiments/fine_object_grounded_small_eval/sample_manifest.json`
- fine grouped detector outputs:
  `/Users/skkac/Work/SCAN/new_route/experiments/fine_object_grounded_small_eval/outputs_groups`

Scripts:

- build sample pack for surface baseline:
  `/Users/skkac/Work/SCAN/new_route/experiments/surface_fine_combo_small_eval/build_combo_surface_samples.py`
- run server-side surface model:
  `/Users/skkac/Work/SCAN/new_route/experiments/surface_fine_combo_small_eval/run_server_surface_combo_eval.py`
- analyze overlap between fine detections and surface classes:
  `/Users/skkac/Work/SCAN/new_route/experiments/surface_fine_combo_small_eval/analyze_surface_fine_overlap.py`

Current run:

- surface model:
  `facebook/mask2former-swin-tiny-ade-semantic`
- local synced output:
  `/Users/skkac/Work/SCAN/new_route/experiments/surface_fine_combo_small_eval/outputs_mask2former_ade20k`
- overlap analysis:
  `/Users/skkac/Work/SCAN/new_route/experiments/surface_fine_combo_small_eval/analysis_mask2former_ade20k`

Current conclusion:

- generic ADE20K surface semantics do not cleanly separate rooftop fine targets
- `railing` detections are dominated by `background/building/floor_ground`
- `pipe` detections mostly land on `floor_ground`
- `equipment` detections split across `building/background/floor_ground`

Implication:

- plain `surface-first + generic semantic model + fine detector` is not enough
- the next viable version must add geometry-aware surface rules before using
  surface classes as hard priors for fine-object confirmation
