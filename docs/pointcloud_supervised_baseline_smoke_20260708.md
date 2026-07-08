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

Remote feature probe on 2026-07-08:

- runner: `scripts/run_scan_train_supervised_smoke_probe.sh`
- python: `/opt/conda/envs/depth-anything-3/bin/python`
- report:
  `/root/epfs/SCAN/work_MT20260616-175807/pointcloud_supervised_baseline_smoke_probe_v2_20260708/feature_probe_report.json`
- local copy:
  `server_parking_priority_s10/pointcloud_supervised_baseline_smoke_probe_v2_20260708/feature_probe_report.json`
- result: all `5` crop inputs loaded and probed successfully.
- diagnostic: on `risk_70503_9366_local`, `xyz_rgb_normal_height` still has
  high entropy (`2.935`) and no clear dominant cluster (`0.198` largest
  cluster ratio). This supports the current diagnosis: surface/grass/shrub
  ownership failure is not solved by simple local feature clustering; the next
  useful test is a real supervised point model as patch/object evidence, not
  another local exception in v7.

Real supervised model gate:

```bash
bash scripts/check_scan_train_supervised_point_runtime.sh
RUN=1 bash scripts/sync_scan_train_supervised_point_repos.sh
bash scripts/check_scan_train_supervised_point_runtime.sh
```

This only checks/clones the model repositories. It does not install a new
environment and does not treat the KMeans feature probe as a supervised model.
The next valid supervised smoke should run Sonata first, because its upstream
repo is a smaller inference-oriented entrypoint; Pointcept remains the full
framework path for PTv3/Sonata training-style experiments.

Environment setup policy:

```bash
RUN=1 bash scripts/setup_scan_train_sonata_env.sh
```

Use a dedicated prefix, defaulting to `/root/epfs/conda_envs/sonata`. Do not
install Sonata dependencies into `/opt/conda/envs/depth-anything-3`: the current
available environment is PyTorch `2.7.1+cu118`, while upstream Sonata's
standalone environment pins PyTorch `2.5.0` with CUDA `12.4`. Mixing those
stacks would make smoke failures ambiguous.

Remote repo/runtime status on 2026-07-08:

- cloned Sonata to `/root/epfs/model_side_tracks/sonata`, commit `18c09ff`
- cloned Pointcept to `/root/epfs/model_side_tracks/pointcept`, commit `2b97e6e`
- existing smoke Python `/opt/conda/envs/depth-anything-3/bin/python` has
  `torch`, `numpy`, `sklearn`, `open3d`, `huggingface_hub`, and CUDA available.
- missing for real Sonata inference in that environment:
  `fast_pytorch_kmeans`, `spconv`, `torch_scatter`, `timm`.
- started dedicated Sonata env setup in tmux session `scan_sonata_env_setup`.
  run dir: `/root/epfs/conda_envs/sonata_setup_20260708_123238`.
  Current observed state: conda metadata collection still running; no smoke
  inference result yet.

Prepared Sonata crop smoke:

```bash
RUN=1 bash scripts/run_scan_train_sonata_crop_smoke.sh
```

Default input is the known mixed-risk crop:
`/root/epfs/SCAN/work_MT20260616-175807/pointcloud_supervised_baseline_smoke_crops_20260708/risk_70503_9366_local.ply`.
The script writes a Sonata encoder PCA-colored PLY and a small JSON report.
Run it only after `/root/epfs/conda_envs/sonata/bin/python` exists and imports
`torch`, `sonata`, `spconv`, `torch_scatter`, `timm`, `open3d`, and
`fast_pytorch_kmeans`.

## Acceptance

A supervised smoke is useful only if it explains at least one current failure
mode without violating exclusive voxel ownership. It can become teacher
evidence after visual QA; it cannot become the patch mainline by itself.
