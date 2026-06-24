# Current Dense Visual Acceptance

Status: `pending`
Candidate: `v8_object_refinement`
Review index: http://127.0.0.1:8765/docs/current_dense_review_index.html

## QA Summary

- `accepted_delta`: `1139`
- `output_object_delta`: `-1139`
- `overlap_delta`: `-0.0004186838888514677`
- `surface_guard_label_delta`: `{'car': 0, 'floor': 0, 'grass': 0, 'railing': 0, 'unknown': 0, 'wall': 0}`

## Required Checks

- `v8_fragmentation_improves` [required] `pending`: v8 visibly reduces object fragmentation compared with v7 in the same areas.
- `v8_no_obvious_overmerge` [required] `pending`: v8 does not visibly merge unrelated large structures such as ground/building/tree into one object.
- `surface_guard_no_unknown_regression` [required] `pending`: v17 keeps floor/wall visible and does not reproduce the v15/v16 unknown spike.
- `semantic_not_promoted_from_object_view` [required] `pending`: Object refinement is only promoted as geometry ownership; semantic labels remain evidence/QA references.

## Promotion

Promotion remains blocked until every required check is set to `accepted` in `docs/current_dense_visual_acceptance.json` and `gate_current_dense_mainline_promotion.py` passes.

## Update Commands

After visual inspection, update each required check with:

```bash
python3 scripts/update_current_dense_visual_acceptance.py \
  --check-id v8_fragmentation_improves \
  --status accepted \
  --reviewer "<name>" \
  --notes "<brief evidence>"
```

Valid statuses are `pending`, `accepted`, `rejected`, and `blocked`. Promotion only passes after all required checks are `accepted` and:

```bash
python3 scripts/gate_current_dense_mainline_promotion.py \
  --qa-json docs/current_dense_mainline_qa.json \
  --visual-acceptance docs/current_dense_visual_acceptance.json \
  --output docs/current_dense_promotion_gate.json
```
