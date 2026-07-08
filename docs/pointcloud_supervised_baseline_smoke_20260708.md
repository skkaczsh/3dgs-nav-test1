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

## Fixed Smoke Crops

Use `docs/pointcloud_supervised_baseline_smoke_manifest_20260708.json`.

It contains five local crops from the canonical dense `0.03m` Opt-LAS voxel
source: largest horizontal, vertical, rough, thin-linear patches, plus the
known v7 `70503/9366` risk area. These crops are small enough for a smoke test
and large enough to expose the current surface/object failure modes.

Exported crop PLYs:
`server_parking_priority_s10/pointcloud_supervised_baseline_smoke_crops_20260708/`.

Export report:
`server_parking_priority_s10/pointcloud_supervised_baseline_smoke_crops_20260708/crop_export_report.json`.

The export is reproducible with:

```bash
python3 scripts/export_pointcloud_supervised_smoke_crops.py
python3 scripts/validate_pointcloud_supervised_smoke_crop_export.py
```

The export report records a `sha256` for every crop PLY. Remote Pointcept/PTv3
smoke runs must verify these hashes before inference so results stay comparable
across 4090D, 5070Ti, and local runs.

Remote sync and hash verification:

```bash
DRY_RUN=1 bash scripts/sync_supervised_smoke_crops_to_remote.sh
bash scripts/sync_supervised_smoke_crops_to_remote.sh
```

Defaults are `SSH_HOST=scan-train` and
`REMOTE_DIR=/root/epfs/SCAN/work_MT20260616-175807/pointcloud_supervised_baseline_smoke_crops_20260708`.

Remote sync evidence on 2026-07-08:

- host: `scan-train`
- remote dir:
  `/root/epfs/SCAN/work_MT20260616-175807/pointcloud_supervised_baseline_smoke_crops_20260708`
- synced crop count: `5`
- remote sha256 verification: `passed`

## Acceptance

A supervised smoke is useful only if it explains at least one current failure
mode without violating exclusive voxel ownership. It can become teacher
evidence after visual QA; it cannot become the patch mainline by itself.
