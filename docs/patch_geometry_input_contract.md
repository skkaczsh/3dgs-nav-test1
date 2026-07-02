# Patch Geometry Input Contract

## Purpose

Patch/object artifacts produced by the geometry optimizer are ownership
partitions, not semantic annotations.  They may be used as the 3D support for
later VLM, mask, scene-prior, or structure-prior evidence, but they must not
pretend that geometry buckets are semantic labels.

## Contract

- `geometry_type` describes the point-cloud structure: `horizontal`,
  `vertical`, `thin_linear`, `rough_mixed`, `mixed`, or `unknown`.
- `geometry_label` mirrors `geometry_type` for viewer/debug consumers.
- `semantic_label` must remain `unknown` until a semantic fusion stage assigns a
  label from visual and geometric evidence.
- `semantic_status` must be `geometry_only_unlabeled` for geometry-only rows.
- `label_policy` must be `geometry_is_not_semantic`.
- Teacher-transfer stages must treat rows with this policy as unlabeled
  geometry.  They may assign a semantic label from teacher votes, but they must
  not fall back from `horizontal -> floor` or `vertical -> wall` without direct
  semantic evidence.

## Reason

Previous failed branches repeatedly polluted semantics by treating structural
evidence as a hard class.  Examples include wall/floor/ceiling confusion,
indoor cars, and surface guards demoting large areas to unknown.  The corrected
flow is:

1. Build exclusive 3D ownership from dense geometry.
2. Preserve structural fields as evidence only.
3. Add visual/skymask/first-touch/VLM observations.
4. Assign semantic labels only when evidence converges.

This keeps drivability-style structure priors useful without allowing them to
override semantic evidence.

## Guardrail

`scripts/validate_geometry_input_contract_usage.py` is part of the current
mainline health check.  It protects the semantic PNG voting and teacher-transfer
entry points that normalize original object labels, because those are the places
where geometry-only rows can otherwise be silently promoted into surface
semantics before any visual evidence is applied.
