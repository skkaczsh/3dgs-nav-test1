# Pointcloud Supervised Baseline Smoke

Status: `planned`

Purpose: run Pointcept/PTv3 or Sonata as a supervised semantic teacher and
domain-gap diagnostic, not as a geometry ownership stage.

## Contract

- Input must be the canonical `0.03m` dense Opt-LAS voxel source:
  `dense_las_voxel003_binary`.
- Output may be semantic logits, labels, per-patch votes, and QA previews.
- Output must not split patches, merge patches, or replace existing voxel
  ownership.
- Any usable result must be fused as evidence after the SPG/patch ownership
  stage.

## Required Ablations

- `xyz`
- `xyz_rgb`
- `xyz_normal`
- `xyz_rgb_normal_height`

These four runs are the minimum needed to tell whether current errors come from
geometry, color, normal estimation, or height priors.

## Acceptance

A supervised smoke is useful only if it explains at least one current failure
mode without violating exclusive voxel ownership. It can become teacher
evidence after visual QA; it cannot become the patch mainline by itself.
