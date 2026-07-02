# Patch Experiment Visual Acceptance

Status: `pending`
Selected candidate: `v2_bucket_attach`
Candidate policy: `geometry_input_only`
Review index: http://127.0.0.1:8765/docs/patch_experiment_review_index.html

## Required Checks

- `metric_comparison_reviewed` [required] `pending`: The v2/v5 metric comparison has been reviewed and the selected candidate is intentional.
- `no_major_structure_overmerge` [required] `pending`: The selected candidate does not visibly merge unrelated large structures such as ground/building/tree.
- `small_fragment_tradeoff_accepted` [required] `pending`: Residual small-fragment behavior is acceptable for the next object/semantic layer.
- `semantic_layer_input_decision` [required] `pending`: The selected patch run is explicitly approved as geometry input only, not as semantic truth.

## Promotion

This experiment remains blocked from semantic/object promotion until every required check is set to `accepted` and `gate_patch_experiment_promotion.py` passes.
