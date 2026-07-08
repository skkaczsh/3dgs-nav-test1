# External Dense Scene Segmentation Advice Alignment

Source: user-provided external note, 2026-07-08.

## Decision

Adopt the high-level architecture, but do not replace the current dense patch
mainline with a new model stack yet.

The useful correction is conceptual:

```text
dense scene cloud
  -> clean geometric regions / superpoints
  -> region evidence and labels
  -> dense point label backfill
```

This matches the current Superpoint Graph direction. It also confirms that
SAM/VLM should not own 3D boundaries.

## Adopt Now

- Keep `0.03m` dense Opt-LAS voxel cloud as the production geometry source.
- Keep SPG-style region tokens as the active patch architecture.
- Treat VLM as a region-level teacher, auditor, and open-vocabulary proposer.
- Add a supervised point-cloud baseline matrix as a separate diagnostic:
  - Pointcept + PTv3/Sonata if the environment cost is acceptable.
  - Inputs: `XYZ`, `XYZ+RGB`, `XYZ+normal`, `XYZ+RGB+normal+height`.
  - Output is a benchmark/teacher, not an ownership source.
- Preserve a strict distinction between:
  - geometry ownership: one voxel, one patch/object owner;
  - semantic posterior: labels/descriptions from masks, VLM, scene prior, or supervised models.

## Defer

- Full ScanNet++/ScanNet/S3DIS fine-tuning.
  - Useful, but it requires label taxonomy mapping and training infrastructure.
  - First run a pretrained/domain-gap smoke on the current dense cloud.
- Superpoint Transformer integration.
  - Architecturally aligned, but current deterministic SPG still needs visual acceptance and edge coverage fixes.
  - SPT should be evaluated after patch ownership is stable enough to supply useful tokens.
- 3DGS semantic distillation.
  - Useful for open vocabulary, but old 3DGS training quality was a blocker.
  - Use the validated first-touch dense projection route before reopening 3DGS.

## Reject As Mainline

- VLM direct dense segmentation.
  - Prior runs already showed unstable labels and mask/point contamination.
- 2D mask labels as 3D object boundaries.
  - Prior SAM/Mimo paths mixed ground, wall, car, railing, and ceiling.
- SemanticKITTI/nuScenes-first training.
  - Their scan pattern and domain are not the closest fit for this complete dense colored scene task.

## Immediate Test Matrix

Minimum next tests that add real information:

| test | purpose | pass condition |
| --- | --- | --- |
| v7 SPG visual QA | decide whether guarded uncertain edges improve boundary quality | no visible over-merge on ground/building/tree/car and `70503/9366` risk is safe |
| Pointcept pretrained smoke | estimate domain gap of supervised point-cloud models | large surfaces are at least competitive with V20/V17 teacher labels |
| feature ablation on a fixed crop | identify whether geometry, RGB, normals, or height drive errors | ablation result explains at least one current failure mode |
| region-level VLM crop QA | validate VLM as label teacher, not segmenter | labels/descriptions improve on existing object candidates without changing ownership |

## Architecture Implication

The project should keep two parallel evidence tracks:

1. `patch ownership track`: deterministic dense geometry/SPG, exclusive voxel owners.
2. `semantic evidence track`: supervised point-cloud logits, SAM/mask evidence, first-touch RGB/depth, VLM descriptions.

Only the ownership track can split or merge voxels. The evidence track can
change labels, confidence, descriptions, and ambiguity flags.
