# Object Semantic Evidence Fusion

## Goal

The semantic stage should classify existing geometry-owned objects.  It must not
create object ownership, and it must not promote geometry buckets into semantic
labels without visual or teacher evidence.

## Inputs

Each object row may contain:

- geometry ownership fields: `object_id`, `geometry_type`, `bbox_3d`,
  `voxel_count`, `mean_rgb`, `mean_normal`
- geometry-only contract fields: `semantic_status=geometry_only_unlabeled`,
  `label_policy=geometry_is_not_semantic`
- visual evidence: `semantic_votes`, `semantic_veto_votes`
- teacher evidence: `teacher_allowed_votes`, `teacher_vetoed_votes`
- scene evidence: `scene_prior.scene_expected_label_weights`

## Fusion Rules

- Geometry-only rows start from `unknown`.
- Scene prior alone does not promote a label by default.
- SAM/semantic PNG votes and teacher votes can promote a label only when the
  winner ratio and total evidence weight pass thresholds.
- Geometry vetoes are applied before winner selection.  For example, `car` votes
  on a horizontal surface are recorded as vetoed evidence, not accepted labels.
- The output keeps `semantic_evidence_scores`, `semantic_vetoed_scores`,
  `semantic_fusion_status`, and `semantic_fusion_confidence` so every label is
  auditable.

## CLI

Safe launcher, default dry-run:

```bash
PYTHONPATH=. python scripts/run_object_semantic_evidence_fusion.py \
  --objects-jsonl input_objects.jsonl \
  --output-jsonl fused_objects.jsonl \
  --report fused_objects_report.json
```

The launcher reads `docs/patch_experiment_promotion_gate.json`.  If the patch
experiment has not passed visual promotion, the plan is written as `blocked` and
`--run` exits without executing fusion.  For explicitly marked experiments, use
`--allow-unpromoted-patch-experiment` and keep the output out of promoted
mainline artifacts.

When the plan is ready, the launcher executes two commands in order:

1. `scripts/fuse_object_semantic_evidence.py`
2. `scripts/validate_object_semantic_evidence_fusion.py`

The validator checks that object ids and ownership fields did not change, that
fusion status/evidence fields exist, and that scene-only promotion has not
slipped into a default run.

Direct fusion entry point:

```bash
PYTHONPATH=. python scripts/fuse_object_semantic_evidence.py \
  --objects-jsonl input_objects.jsonl \
  --output-jsonl fused_objects.jsonl \
  --report fused_objects_report.json
```

This stage only rewrites object metadata.  Viewer PLY recoloring should be a
separate export step after the fused object JSONL passes QA.
